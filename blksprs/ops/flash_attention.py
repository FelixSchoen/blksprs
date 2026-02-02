"""Block-sparse Flash Attention implementation for blksprs.

This module implements Flash Attention 2 algorithm with block-sparse support,
including cross-attention (seq_q != seq_k) and custom attention masks.
"""

import math
from typing import Tuple

import torch
import triton
from torch import Tensor
from triton import language as tl


# === Forward Kernel ===

@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    mask_ptr,
    attn_lut_ptr, attn_offsets_ptr,
    m_ptr, l_ptr,
    stride_q_batch, stride_q_seq, stride_q_dim,
    stride_kv_batch, stride_kv_seq, stride_kv_dim,
    stride_mask_batch, stride_mask_row, stride_mask_col,
    n_batches: tl.constexpr,
    seq_q: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    sparsity_block_size: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    scale,
    has_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention forward kernel with block-sparse mask support."""
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    n_m_tiles: tl.constexpr = sparsity_block_size // BLOCK_M
    n_n_tiles: tl.constexpr = sparsity_block_size // BLOCK_N
    
    q_seq_block = pid_m // n_m_tiles
    m_tile_idx = pid_m % n_m_tiles
    
    q_row_start = q_seq_block * sparsity_block_size + m_tile_idx * BLOCK_M
    offs_m = q_row_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # Load Q
    q_ptrs = q_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    q_mask = offs_m[:, None] < seq_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    qk_scale = scale * 1.44269504  # log2(e)
    
    # Get attention LUT
    attn_offset_idx = pid_batch * n_seq_blocks_q + q_seq_block
    attn_start = tl.load(attn_offsets_ptr + attn_offset_idx)
    attn_end = tl.load(attn_offsets_ptr + attn_offset_idx + 1)
    n_kv_blocks = attn_end - attn_start
    
    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attn_lut_ptr + attn_start + kv_idx)
            
            # With BLOCK_N = sparsity_block_size, process entire K block at once
            k_row_start = k_seq_block * sparsity_block_size
            offs_n = k_row_start + tl.arange(0, BLOCK_N)
            
            # Load K
            k_ptrs = k_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
            k_mask = offs_n[:, None] < seq_k
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            
            # Compute QK
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            
            # Apply attention mask if provided
            if has_mask:
                mask_ptrs = mask_ptr + pid_batch * stride_mask_batch + offs_m[:, None] * stride_mask_row + offs_n[None, :] * stride_mask_col
                mask_vals = tl.load(mask_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k), other=0.0)
                qk = qk + mask_vals
            
            # Online softmax
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            # Load V and accumulate
            v_ptrs = v_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)
            acc = tl.dot(p.to(v.dtype), v, acc)
            
            m_i = m_ij
    
    # Normalize
    has_attention = l_i > 0
    l_safe = tl.where(has_attention, l_i, 1.0)
    acc = acc / l_safe[:, None]
    acc = tl.where(has_attention[:, None], acc, 0.0)
    
    # Store output
    o_ptrs = o_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < seq_q)
    
    # Store m and l for backward
    lse = tl.where(has_attention, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(m_ptr + pid_batch * seq_q + offs_m, lse, mask=offs_m < seq_q)
    tl.store(l_ptr + pid_batch * seq_q + offs_m, l_i, mask=offs_m < seq_q)


# === Backward Kernels ===

@triton.jit
def flash_attention_bwd_preprocess_kernel(
    o_ptr, do_ptr, delta_ptr,
    stride_batch, stride_seq, stride_dim,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute delta = (O * dO).sum(dim=-1)."""
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    o_ptrs = o_ptr + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_d[None, :]
    do_ptrs = do_ptr + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_d[None, :]
    mask = offs_m[:, None] < seq_len
    
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    
    tl.store(delta_ptr + pid_batch * seq_len + offs_m, delta, mask=offs_m < seq_len)


@triton.jit
def flash_attention_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    dk_ptr, dv_ptr,
    lse_ptr, delta_ptr,
    mask_ptr,
    rev_attn_lut_ptr, rev_attn_offsets_ptr,
    stride_q_batch, stride_q_seq,
    stride_kv_batch, stride_kv_seq,
    stride_dim,
    stride_mask_batch, stride_mask_row, stride_mask_col,
    n_batches: tl.constexpr,
    seq_q: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    sparsity_block_size: tl.constexpr,
    n_seq_blocks_k: tl.constexpr,
    max_q_per_k: tl.constexpr,
    scale,
    has_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dK and dV gradients."""
    pid_n = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    n_n_tiles = sparsity_block_size // BLOCK_N
    n_m_tiles = sparsity_block_size // BLOCK_M
    
    k_seq_block = pid_n // n_n_tiles
    n_tile_idx = pid_n % n_n_tiles
    
    k_row_start = k_seq_block * sparsity_block_size + n_tile_idx * BLOCK_N
    offs_n = k_row_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)
    
    qk_scale = scale * 1.44269504
    
    # Load K and V
    k_ptrs = k_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
    v_ptrs = v_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
    k_mask = offs_n[:, None] < seq_k
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=k_mask, other=0.0)
    
    dk = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
    
    # Get reverse LUT
    rev_offset_idx = pid_batch * n_seq_blocks_k + k_seq_block
    rev_start = tl.load(rev_attn_offsets_ptr + rev_offset_idx)
    rev_end = tl.load(rev_attn_offsets_ptr + rev_offset_idx + 1)
    n_q_blocks = rev_end - rev_start
    
    for q_idx in range(max_q_per_k):
        if q_idx < n_q_blocks:
            q_seq_block = tl.load(rev_attn_lut_ptr + rev_start + q_idx)
            
            for m_tile_idx in range(n_m_tiles):
                q_row_start = q_seq_block * sparsity_block_size + m_tile_idx * BLOCK_M
                offs_m = q_row_start + tl.arange(0, BLOCK_M)
                
                # Load Q and dO
                q_ptrs = q_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
                do_ptrs = do_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
                q_mask = offs_m[:, None] < seq_q
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)
                
                # Load LSE and delta
                m = tl.load(lse_ptr + pid_batch * seq_q + offs_m, mask=offs_m < seq_q, other=0.0)
                Di = tl.load(delta_ptr + pid_batch * seq_q + offs_m, mask=offs_m < seq_q, other=0.0)
                
                # Recompute P
                qk = tl.dot(q, tl.trans(k)) * qk_scale
                
                if has_mask:
                    mask_ptrs = mask_ptr + pid_batch * stride_mask_batch + offs_m[:, None] * stride_mask_row + offs_n[None, :] * stride_mask_col
                    mask_vals = tl.load(mask_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k), other=0.0)
                    qk = qk + mask_vals
                
                valid_lse = m > float("-inf")
                safe_m = tl.where(valid_lse, m, 0.0)
                p = tl.math.exp2(qk - safe_m[:, None])
                p = tl.where(valid_lse[:, None], p, 0.0)
                
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    
    dk = dk * scale
    tl.store(dk_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :], dk.to(dk_ptr.dtype.element_ty), mask=k_mask)
    tl.store(dv_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :], dv.to(dv_ptr.dtype.element_ty), mask=k_mask)


@triton.jit
def flash_attention_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    dq_ptr,
    lse_ptr, delta_ptr,
    mask_ptr,
    attn_lut_ptr, attn_offsets_ptr,
    stride_q_batch, stride_q_seq,
    stride_kv_batch, stride_kv_seq,
    stride_dim,
    stride_mask_batch, stride_mask_row, stride_mask_col,
    n_batches: tl.constexpr,
    seq_q: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    sparsity_block_size: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    scale,
    has_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dQ gradients."""
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    n_m_tiles = sparsity_block_size // BLOCK_M
    n_n_tiles = sparsity_block_size // BLOCK_N
    
    q_seq_block = pid_m // n_m_tiles
    m_tile_idx = pid_m % n_m_tiles
    
    q_row_start = q_seq_block * sparsity_block_size + m_tile_idx * BLOCK_M
    offs_m = q_row_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    qk_scale = scale * 1.44269504
    
    # Load Q and dO
    q_ptrs = q_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    do_ptrs = do_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    q_mask = offs_m[:, None] < seq_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    do = tl.load(do_ptrs, mask=q_mask, other=0.0)
    
    m = tl.load(lse_ptr + pid_batch * seq_q + offs_m, mask=offs_m < seq_q, other=0.0)
    Di = tl.load(delta_ptr + pid_batch * seq_q + offs_m, mask=offs_m < seq_q, other=0.0)
    
    dq = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    attn_offset_idx = pid_batch * n_seq_blocks_q + q_seq_block
    attn_start = tl.load(attn_offsets_ptr + attn_offset_idx)
    attn_end = tl.load(attn_offsets_ptr + attn_offset_idx + 1)
    n_kv_blocks = attn_end - attn_start
    
    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attn_lut_ptr + attn_start + kv_idx)
            
            for n_tile_idx in range(n_n_tiles):
                k_row_start = k_seq_block * sparsity_block_size + n_tile_idx * BLOCK_N
                offs_n = k_row_start + tl.arange(0, BLOCK_N)
                
                k_ptrs = k_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
                v_ptrs = v_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
                k_mask = offs_n[:, None] < seq_k
                k = tl.load(k_ptrs, mask=k_mask, other=0.0)
                v = tl.load(v_ptrs, mask=k_mask, other=0.0)
                
                qk = tl.dot(q, tl.trans(k)) * qk_scale
                
                if has_mask:
                    mask_ptrs = mask_ptr + pid_batch * stride_mask_batch + offs_m[:, None] * stride_mask_row + offs_n[None, :] * stride_mask_col
                    mask_vals = tl.load(mask_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k), other=0.0)
                    qk = qk + mask_vals
                
                valid_lse = m > float("-inf")
                safe_m = tl.where(valid_lse, m, 0.0)
                p = tl.math.exp2(qk - safe_m[:, None])
                p = tl.where(valid_lse[:, None], p, 0.0)
                
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                dq += tl.dot(ds.to(k.dtype), k)
    
    dq = dq * scale
    tl.store(dq_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :], dq.to(dq_ptr.dtype.element_ty), mask=q_mask)


# === Autograd Function ===

class BlockSparseFlashAttention(torch.autograd.Function):
    """Block-sparse Flash Attention with autograd support."""
    
    @staticmethod
    def forward(ctx, q, k, v, attention_mask, attn_lut, attn_offsets, rev_attn_lut, rev_attn_offsets,
                sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k, max_kv_blocks, max_q_per_k, scale, has_mask):
        batch, seq_q, n_heads, head_dim = q.shape
        _, seq_k, _, _ = k.shape
        n_batches = batch * n_heads
        
        # Reshape to [batch*heads, seq, head_dim]
        q_flat = q.permute(0, 2, 1, 3).reshape(n_batches, seq_q, head_dim).contiguous()
        k_flat = k.permute(0, 2, 1, 3).reshape(n_batches, seq_k, head_dim).contiguous()
        v_flat = v.permute(0, 2, 1, 3).reshape(n_batches, seq_k, head_dim).contiguous()
        
        o_flat = torch.empty_like(q_flat)
        lse = torch.empty(n_batches, seq_q, device=q.device, dtype=torch.float32)
        l = torch.empty(n_batches, seq_q, device=q.device, dtype=torch.float32)
        
        # Block sizes - use sparsity_block_size for N to eliminate inner loop
        # For M, use larger blocks when possible for better parallelism
        if head_dim <= 64:
            BLOCK_M = min(128, sparsity_block_size)
        elif head_dim <= 128:
            BLOCK_M = min(64, sparsity_block_size)
        else:
            BLOCK_M = min(32, sparsity_block_size)
        # Use sparsity_block_size for N to process entire K block in one iteration
        BLOCK_N = sparsity_block_size
        
        n_m_tiles = seq_q // BLOCK_M
        grid = (n_m_tiles, n_batches)
        
        # Mask strides
        if has_mask:
            mask_stride_batch = attention_mask.stride(0)
            mask_stride_row = attention_mask.stride(1)
            mask_stride_col = attention_mask.stride(2)
        else:
            mask_stride_batch = 0
            mask_stride_row = 0
            mask_stride_col = 0
        
        flash_attention_fwd_kernel[grid](
            q_flat, k_flat, v_flat, o_flat,
            attention_mask if has_mask else q_flat,
            attn_lut, attn_offsets,
            lse, l,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            mask_stride_batch, mask_stride_row, mask_stride_col,
            n_batches, seq_q, seq_k, head_dim, sparsity_block_size, n_seq_blocks_q, max_kv_blocks,
            scale,
            has_mask,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_stages=4, num_warps=4,
        )
        
        o = o_flat.reshape(batch, n_heads, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()
        
        # Save for backward
        ctx.save_for_backward(q_flat, k_flat, v_flat, o_flat, lse,
                               attn_lut, attn_offsets, rev_attn_lut, rev_attn_offsets,
                               attention_mask if has_mask else torch.empty(0, device=q.device))
        ctx.sparsity_block_size = sparsity_block_size
        ctx.n_seq_blocks_q = n_seq_blocks_q
        ctx.n_seq_blocks_k = n_seq_blocks_k
        ctx.max_kv_blocks = max_kv_blocks
        ctx.max_q_per_k = max_q_per_k
        ctx.scale = scale
        ctx.has_mask = has_mask
        ctx.batch = batch
        ctx.n_heads = n_heads
        ctx.seq_q = seq_q
        ctx.seq_k = seq_k
        ctx.head_dim = head_dim
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        (q_flat, k_flat, v_flat, o_flat, lse,
         attn_lut, attn_offsets, rev_attn_lut, rev_attn_offsets, attention_mask) = ctx.saved_tensors
        
        batch = ctx.batch
        n_heads = ctx.n_heads
        seq_q = ctx.seq_q
        seq_k = ctx.seq_k
        head_dim = ctx.head_dim
        n_batches = batch * n_heads
        sparsity_block_size = ctx.sparsity_block_size
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        has_mask = ctx.has_mask
        
        do_flat = grad_output.permute(0, 2, 1, 3).reshape(n_batches, seq_q, head_dim).contiguous()
        
        dq_flat = torch.zeros_like(q_flat)
        dk_flat = torch.zeros_like(k_flat)
        dv_flat = torch.zeros_like(v_flat)
        delta = torch.empty(n_batches, seq_q, device=q_flat.device, dtype=torch.float32)
        
        # Mask strides
        if has_mask:
            mask_stride_batch = attention_mask.stride(0)
            mask_stride_row = attention_mask.stride(1)
            mask_stride_col = attention_mask.stride(2)
        else:
            mask_stride_batch = 0
            mask_stride_row = 0
            mask_stride_col = 0
        
        # 1. Preprocess
        n_m_tiles_q = seq_q // BLOCK_M
        flash_attention_bwd_preprocess_kernel[(n_m_tiles_q, n_batches)](
            o_flat, do_flat, delta,
            o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
            seq_q, head_dim,
            BLOCK_M=BLOCK_M,
        )
        
        # 2. dK and dV
        n_n_tiles_k = seq_k // BLOCK_N
        flash_attention_bwd_dkdv_kernel[(n_n_tiles_k, n_batches)](
            q_flat, k_flat, v_flat, do_flat,
            dk_flat, dv_flat,
            lse, delta,
            attention_mask if has_mask else q_flat,
            rev_attn_lut, rev_attn_offsets,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            q_flat.stride(2),
            mask_stride_batch, mask_stride_row, mask_stride_col,
            n_batches, seq_q, seq_k, head_dim, sparsity_block_size, ctx.n_seq_blocks_k, ctx.max_q_per_k,
            ctx.scale,
            has_mask,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        
        # 3. dQ
        flash_attention_bwd_dq_kernel[(n_m_tiles_q, n_batches)](
            q_flat, k_flat, v_flat, do_flat,
            dq_flat,
            lse, delta,
            attention_mask if has_mask else q_flat,
            attn_lut, attn_offsets,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            q_flat.stride(2),
            mask_stride_batch, mask_stride_row, mask_stride_col,
            n_batches, seq_q, seq_k, head_dim, sparsity_block_size, ctx.n_seq_blocks_q, ctx.max_kv_blocks,
            ctx.scale,
            has_mask,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        
        dq = dq_flat.reshape(batch, n_heads, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()
        dk = dk_flat.reshape(batch, n_heads, seq_k, head_dim).permute(0, 2, 1, 3).contiguous()
        dv = dv_flat.reshape(batch, n_heads, seq_k, head_dim).permute(0, 2, 1, 3).contiguous()
        
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


# === Public API ===

@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_layout: Tensor,
    sparsity_block_size: int,
    scale: float = None,
    attention_mask: Tensor = None,
    lut: dict = None,
) -> Tensor:
    """Block-sparse flash attention with optional attention mask.
    
    Args:
        q: Query tensor [batch, seq_q, n_heads, head_dim]
        k: Key tensor [batch, seq_k, n_heads, head_dim]
        v: Value tensor [batch, seq_k, n_heads, head_dim]
        attention_layout: Block attention pattern [batch*heads, n_seq_blocks_q, n_seq_blocks_k]
        sparsity_block_size: Block size for sparsity pattern
        scale: Attention scale (default: 1/sqrt(head_dim))
        attention_mask: Boolean mask [batch*heads, seq_q, seq_k] where True=masked (default None)
        lut: Optional pre-computed LUT dictionary
        
    Returns:
        Output tensor [batch, seq_q, n_heads, head_dim]
    """
    batch, seq_q, n_heads, head_dim = q.shape
    _, seq_k, _, _ = k.shape
    
    # Validation
    if k.shape[0] != batch or k.shape[2] != n_heads or k.shape[3] != head_dim:
        raise ValueError("K must have compatible shape with Q")
    if v.shape != k.shape:
        raise ValueError("V must have same shape as K")
    if not (sparsity_block_size >= 16 and (sparsity_block_size & (sparsity_block_size - 1)) == 0):
        raise ValueError(f"sparsity_block_size must be power of 2 >= 16, got {sparsity_block_size}")
    if seq_q % sparsity_block_size != 0:
        raise ValueError(f"seq_q ({seq_q}) must be divisible by sparsity_block_size")
    if seq_k % sparsity_block_size != 0:
        raise ValueError(f"seq_k ({seq_k}) must be divisible by sparsity_block_size")
    
    n_batches = batch * n_heads
    n_seq_blocks_q = seq_q // sparsity_block_size
    n_seq_blocks_k = seq_k // sparsity_block_size
    
    expected_layout_shape = (n_batches, n_seq_blocks_q, n_seq_blocks_k)
    if attention_layout.shape != expected_layout_shape:
        raise ValueError(f"attention_layout shape {tuple(attention_layout.shape)} doesn't match expected {expected_layout_shape}")
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Build LUTs
    if lut is None:
        lut = flash_attention_build_lut(attention_layout, n_seq_blocks_q, n_seq_blocks_k)
    
    # Convert boolean mask to additive mask (-inf for True, 0 for False)
    has_mask = attention_mask is not None
    if has_mask:
        if attention_mask.shape != (n_batches, seq_q, seq_k):
            raise ValueError(f"attention_mask shape {tuple(attention_mask.shape)} doesn't match expected ({n_batches}, {seq_q}, {seq_k})")
        attention_mask_additive = torch.where(
            attention_mask, 
            torch.tensor(float("-inf"), device=attention_mask.device, dtype=q.dtype),
            torch.tensor(0.0, device=attention_mask.device, dtype=q.dtype)
        )
    else:
        attention_mask_additive = torch.empty(0, device=q.device, dtype=q.dtype)
    
    return BlockSparseFlashAttention.apply(
        q, k, v,
        attention_mask_additive,
        lut["attn_lut"], lut["attn_offsets"],
        lut["rev_attn_lut"], lut["rev_attn_offsets"],
        sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k,
        lut["max_kv_blocks"], lut["max_q_per_k"],
        scale, has_mask,
    )


def flash_attention_build_lut(
    attention_layout: Tensor,
    n_seq_blocks_q: int = None,
    n_seq_blocks_k: int = None,
) -> dict:
    """Build attention LUTs for reuse across multiple calls."""
    n_batches = attention_layout.shape[0]
    if n_seq_blocks_q is None:
        n_seq_blocks_q = attention_layout.shape[1]
    if n_seq_blocks_k is None:
        n_seq_blocks_k = attention_layout.shape[2]
    
    attn_lut, attn_offsets, max_kv_blocks = _build_attention_lut_fast(
        attention_layout, n_batches, n_seq_blocks_q, n_seq_blocks_k
    )
    
    attention_layout_t = attention_layout.transpose(1, 2).contiguous()
    rev_attn_lut, rev_attn_offsets, max_q_per_k = _build_attention_lut_fast(
        attention_layout_t, n_batches, n_seq_blocks_k, n_seq_blocks_q
    )
    
    return {
        "attn_lut": attn_lut,
        "attn_offsets": attn_offsets,
        "max_kv_blocks": max_kv_blocks,
        "rev_attn_lut": rev_attn_lut,
        "rev_attn_offsets": rev_attn_offsets,
        "max_q_per_k": max_q_per_k,
    }


def _build_attention_lut_fast(
    attention_layout: Tensor,
    n_batches: int,
    n_blocks_row: int,
    n_blocks_col: int,
) -> Tuple[Tensor, Tensor, int]:
    """Build attention LUT efficiently."""
    device = attention_layout.device
    
    counts = attention_layout.sum(dim=2).flatten()
    max_blocks_per_row = int(counts.max().item())
    
    if max_blocks_per_row == 0:
        offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=torch.int32, device=device)
        lut = torch.empty(0, dtype=torch.int32, device=device)
        return lut, offsets, 1
    
    offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    
    indices = attention_layout.reshape(n_batches * n_blocks_row, n_blocks_col).nonzero(as_tuple=False)
    lut = indices[:, 1].to(torch.int32)
    
    return lut, offsets, max_blocks_per_row

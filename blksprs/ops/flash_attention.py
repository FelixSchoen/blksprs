"""Block-sparse Flash Attention implementation for blksprs.

This module implements Flash Attention 2 algorithm with block-sparse support,
including cross-attention (seq_q != seq_k), custom attention masks, and
differentiable attention bias (e.g., for relative positional encodings).

Note: This implementation was developed with AI assistance.
"""

import math
from typing import Tuple

import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.validation import validate_contiguous, validate_device, validate_dtype_float, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_layout: Tensor,
    sparsity_block_size: int,
    scale: float = None,
    attention_mask: Tensor = None,
    attention_bias: Tensor = None,
    lut: dict = None,
) -> Tensor:
    """Block-sparse flash attention with optional attention mask and differentiable bias.
    
    Args:
        q: Query tensor [batch, seq_q, n_heads, head_dim]
        k: Key tensor [batch, seq_k, n_heads, head_dim]
        v: Value tensor [batch, seq_k, n_heads, head_dim]
        attention_layout: Block attention pattern [batch*heads, n_seq_blocks_q, n_seq_blocks_k]
        sparsity_block_size: Block size for sparsity pattern
        scale: Attention scale (default: 1/sqrt(head_dim))
        attention_mask: Boolean mask [batch*heads, seq_q, seq_k] where True=masked (default None).
            Does not participate in gradient computation.
        attention_bias: Additive bias [batch*heads, seq_q, seq_k] added to attention scores before
            softmax (default None). The bias is simply added to the scores in natural scale, i.e.
            softmax(Q@K^T * scale + bias), exactly like relative positional encodings. Supports
            gradient computation. The bias may have a different block-sparsity layout than the
            attention layout -- values in positions not covered by the attention layout will be
            ignored since those blocks are not computed.
        lut: Optional pre-computed LUT dictionary
        
    Returns:
        Output tensor [batch, seq_q, n_heads, head_dim]
    """
    q, k, v = ensure_contiguous(q, k, v)
    
    validate_contiguous(q, k, v)
    validate_dtype_float(q, k, v)
    validate_device(q, k, v)
    
    batch, seq_q, n_heads, head_dim = q.shape
    _, seq_k, _, _ = k.shape
    
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
    
    if lut is None:
        lut = flash_attention_build_lut(attention_layout, n_seq_blocks_q, n_seq_blocks_k)
    
    # Convert boolean attention mask to additive form (-inf for masked positions)
    has_mask = attention_mask is not None
    if has_mask:
        if attention_mask.shape != (n_batches, seq_q, seq_k):
            raise ValueError(f"attention_mask shape {tuple(attention_mask.shape)} doesn't match expected ({n_batches}, {seq_q}, {seq_k})")
        attention_mask_additive = torch.where(
            attention_mask,
            torch.tensor(float("-inf"), device=attention_mask.device, dtype=q.dtype),
            torch.tensor(0.0, device=attention_mask.device, dtype=q.dtype)
        ).contiguous()
    else:
        attention_mask_additive = torch.empty(0, device=q.device, dtype=q.dtype)
    
    # Validate attention bias
    has_bias = attention_bias is not None
    if has_bias:
        if attention_bias.shape != (n_batches, seq_q, seq_k):
            raise ValueError(f"attention_bias shape {tuple(attention_bias.shape)} doesn't match expected ({n_batches}, {seq_q}, {seq_k})")
        attention_bias = attention_bias.contiguous()
    else:
        attention_bias = torch.empty(0, device=q.device, dtype=q.dtype)
    
    return BlockSparseFlashAttention.apply(
        q, k, v,
        attention_mask_additive, attention_bias,
        lut["attn_lut"], lut["attn_offsets"],
        lut["rev_attn_lut"], lut["rev_attn_offsets"],
        sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k,
        lut["max_kv_blocks"], lut["max_q_per_k"],
        scale, has_mask, has_bias,
    )


class BlockSparseFlashAttention(torch.autograd.Function):
    """Block-sparse Flash Attention with autograd support."""
    
    @staticmethod
    def forward(ctx, q, k, v, attention_mask, attention_bias,
                attn_lut, attn_offsets, rev_attn_lut, rev_attn_offsets,
                sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k, max_kv_blocks, max_q_per_k,
                scale, has_mask, has_bias):
        batch, seq_q, n_heads, head_dim = q.shape
        _, seq_k, _, _ = k.shape
        n_batches = batch * n_heads
        
        q_flat = q.permute(0, 2, 1, 3).reshape(n_batches, seq_q, head_dim).contiguous()
        k_flat = k.permute(0, 2, 1, 3).reshape(n_batches, seq_k, head_dim).contiguous()
        v_flat = v.permute(0, 2, 1, 3).reshape(n_batches, seq_k, head_dim).contiguous()
        
        o_flat = torch.empty_like(q_flat)
        lse = torch.empty(n_batches, seq_q, device=q.device, dtype=torch.float32)
        
        if head_dim <= 64:
            BLOCK_M = min(128, sparsity_block_size)
        elif head_dim <= 128:
            BLOCK_M = min(64, sparsity_block_size)
        else:
            BLOCK_M = min(32, sparsity_block_size)
        BLOCK_N = sparsity_block_size
        
        n_m_tiles = seq_q // BLOCK_M
        grid = (n_m_tiles, n_batches)
        
        has_additive = has_mask or has_bias
        
        # Build combined additive tensor (mask + bias) for the forward kernel.
        #
        # The user-facing semantics are: softmax(Q@K^T * scale + mask + bias).
        # The kernel uses exp2 instead of exp for performance, and handles the log2(e)
        # conversion internally. The mask (0 / -inf) is invariant under this scaling.
        # The bias requires log2(e) scaling inside the kernel for numerical correctness.
        if has_additive:
            if has_mask and has_bias:
                additive = (attention_mask + attention_bias).contiguous()
            elif has_mask:
                additive = attention_mask.contiguous()
            else:
                additive = attention_bias.contiguous()
            additive_stride_batch = additive.stride(0)
            additive_stride_row = additive.stride(1)
            additive_stride_col = additive.stride(2)
        else:
            additive = q_flat  # dummy pointer
            additive_stride_batch = 0
            additive_stride_row = 0
            additive_stride_col = 0
        
        flash_attention_fwd_kernel[grid](
            q_flat, k_flat, v_flat, o_flat,
            additive,
            attn_lut, attn_offsets,
            lse,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            additive_stride_batch, additive_stride_row, additive_stride_col,
            n_batches, seq_q, seq_k, head_dim, sparsity_block_size, n_seq_blocks_q, max_kv_blocks,
            scale,
            has_additive,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_stages=4, num_warps=4,
        )
        
        o = o_flat.reshape(batch, n_heads, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()
        
        # Save combined additive for backward (needed to recompute attention weights)
        ctx.save_for_backward(q_flat, k_flat, v_flat, o_flat, lse,
                               attn_lut, attn_offsets, rev_attn_lut, rev_attn_offsets,
                               additive if has_additive else torch.empty(0, device=q.device))
        ctx.sparsity_block_size = sparsity_block_size
        ctx.n_seq_blocks_q = n_seq_blocks_q
        ctx.n_seq_blocks_k = n_seq_blocks_k
        ctx.max_kv_blocks = max_kv_blocks
        ctx.max_q_per_k = max_q_per_k
        ctx.scale = scale
        ctx.has_additive = has_additive
        ctx.has_bias = has_bias
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
         attn_lut, attn_offsets, rev_attn_lut, rev_attn_offsets, additive) = ctx.saved_tensors
        
        batch = ctx.batch
        n_heads = ctx.n_heads
        seq_q = ctx.seq_q
        seq_k = ctx.seq_k
        head_dim = ctx.head_dim
        n_batches = batch * n_heads
        sparsity_block_size = ctx.sparsity_block_size
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        has_additive = ctx.has_additive
        has_bias = ctx.has_bias
        
        do_flat = grad_output.permute(0, 2, 1, 3).reshape(n_batches, seq_q, head_dim).contiguous()
        
        dq_flat = torch.zeros_like(q_flat)
        dk_flat = torch.zeros_like(k_flat)
        dv_flat = torch.zeros_like(v_flat)
        delta = torch.empty(n_batches, seq_q, device=q_flat.device, dtype=torch.float32)
        
        # Allocate dbias if needed
        if has_bias:
            dbias = torch.zeros(n_batches, seq_q, seq_k, device=q_flat.device, dtype=q_flat.dtype)
            dbias_stride_batch = dbias.stride(0)
            dbias_stride_row = dbias.stride(1)
            dbias_stride_col = dbias.stride(2)
        else:
            dbias = q_flat  # dummy pointer
            dbias_stride_batch = 0
            dbias_stride_row = 0
            dbias_stride_col = 0
        
        if has_additive:
            additive_stride_batch = additive.stride(0)
            additive_stride_row = additive.stride(1)
            additive_stride_col = additive.stride(2)
        else:
            additive_stride_batch = 0
            additive_stride_row = 0
            additive_stride_col = 0
        
        n_m_tiles_q = seq_q // BLOCK_M
        flash_attention_bwd_preprocess_kernel[(n_m_tiles_q, n_batches)](
            o_flat, do_flat, delta,
            o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
            seq_q, head_dim,
            BLOCK_M=BLOCK_M,
        )
        
        n_n_tiles_k = seq_k // BLOCK_N
        flash_attention_bwd_dkdv_kernel[(n_n_tiles_k, n_batches)](
            q_flat, k_flat, v_flat, do_flat,
            dk_flat, dv_flat, dbias,
            lse, delta,
            additive if has_additive else q_flat,
            rev_attn_lut, rev_attn_offsets,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            q_flat.stride(2),
            additive_stride_batch, additive_stride_row, additive_stride_col,
            dbias_stride_batch, dbias_stride_row, dbias_stride_col,
            n_batches, seq_q, seq_k, head_dim, sparsity_block_size, ctx.n_seq_blocks_k, ctx.max_q_per_k,
            ctx.scale,
            has_additive, has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        
        flash_attention_bwd_dq_kernel[(n_m_tiles_q, n_batches)](
            q_flat, k_flat, v_flat, do_flat,
            dq_flat,
            lse, delta,
            additive if has_additive else q_flat,
            attn_lut, attn_offsets,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            q_flat.stride(2),
            additive_stride_batch, additive_stride_row, additive_stride_col,
            n_batches, seq_q, seq_k, head_dim, sparsity_block_size, ctx.n_seq_blocks_q, ctx.max_kv_blocks,
            ctx.scale,
            has_additive,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        
        dq = dq_flat.reshape(batch, n_heads, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()
        dk = dk_flat.reshape(batch, n_heads, seq_k, head_dim).permute(0, 2, 1, 3).contiguous()
        dv = dv_flat.reshape(batch, n_heads, seq_k, head_dim).permute(0, 2, 1, 3).contiguous()
        
        dbias_out = dbias if has_bias else None
        
        # Return grads for: q, k, v, attention_mask, attention_bias, attn_lut, attn_offsets,
        #                    rev_attn_lut, rev_attn_offsets, sparsity_block_size, n_seq_blocks_q,
        #                    n_seq_blocks_k, max_kv_blocks, max_q_per_k, scale, has_mask, has_bias
        return dq, dk, dv, None, dbias_out, None, None, None, None, None, None, None, None, None, None, None, None


@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    additive_ptr,
    attn_lut_ptr, attn_offsets_ptr,
    m_ptr,
    stride_q_batch, stride_q_seq, stride_q_dim,
    stride_kv_batch, stride_kv_seq, stride_kv_dim,
    stride_additive_batch, stride_additive_row, stride_additive_col,
    n_batches: tl.constexpr,
    seq_q: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    sparsity_block_size: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    scale,
    has_additive: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention forward kernel with block-sparse support."""
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    n_m_tiles: tl.constexpr = sparsity_block_size // BLOCK_M
    n_n_tiles: tl.constexpr = sparsity_block_size // BLOCK_N
    
    q_seq_block = pid_m // n_m_tiles
    m_tile_idx = pid_m % n_m_tiles
    
    q_row_start = q_seq_block * sparsity_block_size + m_tile_idx * BLOCK_M
    offs_m = q_row_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    q_ptrs = q_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    q_mask = offs_m[:, None] < seq_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    qk_scale = scale * 1.44269504
    
    attn_offset_idx = pid_batch * n_seq_blocks_q + q_seq_block
    attn_start = tl.load(attn_offsets_ptr + attn_offset_idx)
    attn_end = tl.load(attn_offsets_ptr + attn_offset_idx + 1)
    n_kv_blocks = attn_end - attn_start
    
    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attn_lut_ptr + attn_start + kv_idx)
            
            k_row_start = k_seq_block * sparsity_block_size
            offs_n = k_row_start + tl.arange(0, BLOCK_N)
            
            k_ptrs = k_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
            k_mask = offs_n[:, None] < seq_k
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            
            # The additive tensor is in natural scale (0/-inf for mask, arbitrary for bias).
            # Since the kernel works in log2 space (exp2 instead of exp), we must convert
            # the additive values: exp(x) = exp2(x * log2(e)).
            if has_additive:
                add_ptrs = additive_ptr + pid_batch * stride_additive_batch + offs_m[:, None] * stride_additive_row + offs_n[None, :] * stride_additive_col
                add_vals = tl.load(add_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k), other=0.0)
                qk = qk + add_vals * 1.44269504
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            # Handle case where m_i and m_ij are both -inf (all positions masked so far)
            # In this case, alpha should be 1.0 (no scaling needed) rather than NaN from exp2(-inf - (-inf))
            both_neg_inf = (m_i == float("-inf")) & (m_ij == float("-inf"))
            alpha = tl.where(both_neg_inf, 1.0, tl.math.exp2(m_i - m_ij))
            # For p, when qk is -inf and m_ij is -inf, exp2(-inf - (-inf)) = NaN
            # We need to set p = 0 for these cases to avoid NaN contamination
            p_raw = tl.math.exp2(qk - m_ij[:, None])
            p = tl.where((qk == float("-inf")) & (m_ij[:, None] == float("-inf")), 0.0, p_raw)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            v_ptrs = v_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)
            acc = tl.dot(p.to(v.dtype), v, acc)
            
            m_i = m_ij
    
    has_attention = l_i > 0
    l_safe = tl.where(has_attention, l_i, 1.0)
    acc = acc / l_safe[:, None]
    acc = tl.where(has_attention[:, None], acc, 0.0)
    
    o_ptrs = o_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < seq_q)
    
    lse = tl.where(has_attention, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(m_ptr + pid_batch * seq_q + offs_m, lse, mask=offs_m < seq_q)


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
    dk_ptr, dv_ptr, dbias_ptr,
    lse_ptr, delta_ptr,
    additive_ptr,
    rev_attn_lut_ptr, rev_attn_offsets_ptr,
    stride_q_batch, stride_q_seq,
    stride_kv_batch, stride_kv_seq,
    stride_dim,
    stride_additive_batch, stride_additive_row, stride_additive_col,
    stride_dbias_batch, stride_dbias_row, stride_dbias_col,
    n_batches: tl.constexpr,
    seq_q: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    sparsity_block_size: tl.constexpr,
    n_seq_blocks_k: tl.constexpr,
    max_q_per_k: tl.constexpr,
    scale,
    has_additive: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dK, dV, and optionally dBias gradients."""
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
    
    k_ptrs = k_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
    v_ptrs = v_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
    k_mask = offs_n[:, None] < seq_k
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=k_mask, other=0.0)
    
    dk = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
    
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
                
                q_ptrs = q_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
                do_ptrs = do_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :]
                q_mask = offs_m[:, None] < seq_q
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)
                
                m = tl.load(lse_ptr + pid_batch * seq_q + offs_m, mask=offs_m < seq_q, other=0.0)
                Di = tl.load(delta_ptr + pid_batch * seq_q + offs_m, mask=offs_m < seq_q, other=0.0)
                
                qk = tl.dot(q, tl.trans(k)) * qk_scale
                
                if has_additive:
                    add_ptrs = additive_ptr + pid_batch * stride_additive_batch + offs_m[:, None] * stride_additive_row + offs_n[None, :] * stride_additive_col
                    add_vals = tl.load(add_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k), other=0.0)
                    qk = qk + add_vals * 1.44269504
                
                valid_lse = m > float("-inf")
                safe_m = tl.where(valid_lse, m, 0.0)
                p = tl.math.exp2(qk - safe_m[:, None])
                p = tl.where(valid_lse[:, None], p, 0.0)
                
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
                
                # Bias gradient: the forward computes qk + bias * log2(e) internally.
                # By chain rule dL/d(bias) = dL/d(qk_scaled) * d(qk_scaled)/d(bias)
                # = ds * log2(e), since qk_scaled = ... + bias * log2(e).
                if has_bias:
                    ds_bias = ds * 1.44269504
                    dbias_ptrs = dbias_ptr + pid_batch * stride_dbias_batch + offs_m[:, None] * stride_dbias_row + offs_n[None, :] * stride_dbias_col
                    tl.store(dbias_ptrs, ds_bias.to(dbias_ptr.dtype.element_ty),
                             mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k))
    
    dk = dk * scale
    tl.store(dk_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :], dk.to(dk_ptr.dtype.element_ty), mask=k_mask)
    tl.store(dv_ptr + pid_batch * stride_kv_batch + offs_n[:, None] * stride_kv_seq + offs_d[None, :], dv.to(dv_ptr.dtype.element_ty), mask=k_mask)


@triton.jit
def flash_attention_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    dq_ptr,
    lse_ptr, delta_ptr,
    additive_ptr,
    attn_lut_ptr, attn_offsets_ptr,
    stride_q_batch, stride_q_seq,
    stride_kv_batch, stride_kv_seq,
    stride_dim,
    stride_additive_batch, stride_additive_row, stride_additive_col,
    n_batches: tl.constexpr,
    seq_q: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    sparsity_block_size: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    scale,
    has_additive: tl.constexpr,
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
                
                if has_additive:
                    add_ptrs = additive_ptr + pid_batch * stride_additive_batch + offs_m[:, None] * stride_additive_row + offs_n[None, :] * stride_additive_col
                    add_vals = tl.load(add_ptrs, mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k), other=0.0)
                    qk = qk + add_vals * 1.44269504
                
                valid_lse = m > float("-inf")
                safe_m = tl.where(valid_lse, m, 0.0)
                p = tl.math.exp2(qk - safe_m[:, None])
                p = tl.where(valid_lse[:, None], p, 0.0)
                
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                dq += tl.dot(ds.to(k.dtype), k)
    
    dq = dq * scale
    tl.store(dq_ptr + pid_batch * stride_q_batch + offs_m[:, None] * stride_q_seq + offs_d[None, :], dq.to(dq_ptr.dtype.element_ty), mask=q_mask)


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

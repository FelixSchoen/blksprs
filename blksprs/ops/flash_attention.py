"""Block-sparse Flash Attention implementation for blksprs.

This module implements Flash Attention 2 algorithm with block-sparse support.
All input tensors (Q, K, V, attention mask, attention bias) are expected in
the compressed block-sparse format used throughout the blksprs library.

Q, K, V sparsity layouts have shape (n_batches, seq // bs, head_dim // bs),
and the attention_layout has shape (n_batches, seq_q // bs, seq_k // bs).

Note: This implementation was developed with AI assistance.
"""

import math
from typing import Tuple

import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs_exact
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride
from blksprs.utils.validation import (
    validate_contiguous, validate_device, validate_dtype_float,
    validate_dimensions, validate_sparsity, validate_sparsity_block_size,
    ensure_contiguous,
)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def flash_attention(
    q: BlksprsTensor,
    sparsity_layout_q: Tensor,
    k: BlksprsTensor,
    sparsity_layout_k: Tensor,
    v: BlksprsTensor,
    sparsity_layout_v: Tensor,
    attention_layout: Tensor,
    sparsity_block_size: int,
    scale: float = None,
    attention_mask: BlksprsTensor = None,
    sparsity_layout_mask: Tensor = None,
    attention_bias: BlksprsTensor = None,
    sparsity_layout_bias: Tensor = None,
    lut: dict = None,
    sparsity_layout_o: Tensor = None,
) -> BlksprsTensor:
    """Block-sparse flash attention operating on compressed block-sparse tensors.

    All inputs use the standard blksprs compressed format: tensors have shape
    ``(n_sparse_blocks, sparsity_block_size, sparsity_block_size)`` with an
    accompanying sparsity layout.

    Args:
        q (BlksprsTensor): Query tensor in compressed form.
            Sparsity layout shape: ``(n_batches, seq_q // bs, head_dim // bs)``.
        sparsity_layout_q (Tensor): Sparsity layout for Q.
        k (BlksprsTensor): Key tensor in compressed form.
            Sparsity layout shape: ``(n_batches, seq_k // bs, head_dim // bs)``.
        sparsity_layout_k (Tensor): Sparsity layout for K.
        v (BlksprsTensor): Value tensor in compressed form (same layout as K).
        sparsity_layout_v (Tensor): Sparsity layout for V.
        attention_layout (Tensor): Block attention pattern
            ``(n_batches, seq_q // bs, seq_k // bs)`` indicating which Q-K block
            pairs participate in attention.
        sparsity_block_size (int): Block size for the sparsity pattern.
        scale (float, optional): Attention scale (default: ``1/sqrt(head_dim)``).
        attention_mask (BlksprsTensor, optional): Boolean mask in compressed form
            where ``True`` means *masked* (position ignored). Does not participate
            in gradient computation.
        sparsity_layout_mask (Tensor, optional): Sparsity layout for the mask.
            Shape: ``(n_batches, seq_q // bs, seq_k // bs)``.
        attention_bias (BlksprsTensor, optional): Additive bias in compressed form,
            added to attention scores before softmax.  Supports gradient computation.
        sparsity_layout_bias (Tensor, optional): Sparsity layout for the bias.
            Shape: ``(n_batches, seq_q // bs, seq_k // bs)``.
        lut (dict, optional): Pre-computed LUT dictionary.
        sparsity_layout_o (Tensor, optional): Output sparsity layout.
            Shape: ``(n_batches, seq_q // bs, d_v // bs)``.
            Defaults to ``sparsity_layout_q`` if ``d_v == d_att``.

    Returns:
        BlksprsTensor: Output tensor in compressed form with layout
        ``sparsity_layout_o``.
    """
    q, k, v = ensure_contiguous(q, k, v)

    validate_dimensions(q, k, v)
    validate_contiguous(q, k, v)
    validate_dtype_float(q, k, v)
    validate_device(q, k, v)
    validate_sparsity(sparsity_block_size, (q, sparsity_layout_q), (k, sparsity_layout_k), (v, sparsity_layout_v))
    validate_sparsity_block_size(sparsity_block_size, q, k, v)

    n_batches = sparsity_layout_q.size(0)
    n_seq_blocks_q = sparsity_layout_q.size(1)
    n_head_blocks_qk = sparsity_layout_q.size(2)
    n_seq_blocks_k = sparsity_layout_k.size(1)
    n_head_blocks_v = sparsity_layout_v.size(2)
    head_dim = n_head_blocks_qk * sparsity_block_size

    if sparsity_layout_k.size(0) != n_batches or sparsity_layout_k.size(2) != n_head_blocks_qk:
        raise ValueError("K sparsity layout must be compatible with Q")
    if (sparsity_layout_v.size(0) != n_batches or
            sparsity_layout_v.size(1) != n_seq_blocks_k):
        raise ValueError("V sparsity layout must be compatible with K")

    if sparsity_layout_o is None:
        if n_head_blocks_v != n_head_blocks_qk:
            raise ValueError(
                "sparsity_layout_o is required when V head dimension differs from Q/K"
            )
        sparsity_layout_o = sparsity_layout_q
    else:
        sparsity_layout_o = ensure_contiguous(sparsity_layout_o)
        validate_dimensions(sparsity_layout_o)
        validate_contiguous(sparsity_layout_o)
        validate_device(sparsity_layout_o, q)
        if not torch.all(torch.logical_or(sparsity_layout_o == 0, sparsity_layout_o == 1)):
            raise ValueError("Output sparsity layout values must be either 0 or 1")

    expected_output_shape = (n_batches, n_seq_blocks_q, n_head_blocks_v)
    if sparsity_layout_o.shape != torch.Size(expected_output_shape):
        raise ValueError(
            f"sparsity_layout_o shape {tuple(sparsity_layout_o.shape)} doesn't match "
            f"expected {expected_output_shape}"
        )

    expected_attn_shape = (n_batches, n_seq_blocks_q, n_seq_blocks_k)
    if attention_layout.shape != torch.Size(expected_attn_shape):
        raise ValueError(
            f"attention_layout shape {tuple(attention_layout.shape)} doesn't match "
            f"expected {expected_attn_shape}"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Build or reuse LUTs
    if lut is None:
        lut = flash_attention_build_lut(
            attention_layout,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v,
            n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk,
            sparsity_layout_o=sparsity_layout_o,
            n_head_blocks_v=n_head_blocks_v,
        )
    if "reverse_lut_o" not in lut:
        lut["reverse_lut_o"] = _build_reverse_lut(sparsity_layout_o)
    if "n_sparse_blocks_o" not in lut:
        lut["n_sparse_blocks_o"] = int(sparsity_layout_o.sum().item())

    # Handle attention mask
    has_mask = attention_mask is not None
    if has_mask:
        attention_mask = ensure_contiguous(attention_mask)
        validate_dimensions(attention_mask)
        validate_contiguous(attention_mask)
        validate_sparsity(sparsity_block_size, (attention_mask, sparsity_layout_mask))
        reverse_lut_mask = lut.get("reverse_lut_mask")
        if reverse_lut_mask is None:
            reverse_lut_mask = _build_reverse_lut(sparsity_layout_mask)
            lut["reverse_lut_mask"] = reverse_lut_mask
    else:
        attention_mask = torch.empty(0, device=q.device, dtype=q.dtype)
        reverse_lut_mask = None

    # Handle attention bias
    has_bias = attention_bias is not None
    if has_bias:
        attention_bias = ensure_contiguous(attention_bias)
        validate_dimensions(attention_bias)
        validate_contiguous(attention_bias)
        validate_sparsity(sparsity_block_size, (attention_bias, sparsity_layout_bias))
        reverse_lut_bias = lut.get("reverse_lut_bias")
        if reverse_lut_bias is None:
            reverse_lut_bias = _build_reverse_lut(sparsity_layout_bias)
            lut["reverse_lut_bias"] = reverse_lut_bias
    else:
        attention_bias = torch.empty(0, device=q.device, dtype=q.dtype)
        reverse_lut_bias = None

    return BlksprsTensor.wrap(
        BlockSparseFlashAttention.apply(
            q, k, v,
            attention_mask, attention_bias,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
            lut["reverse_lut_q"], lut["reverse_lut_k"], lut["reverse_lut_v"], lut["reverse_lut_o"],
            lut["attn_lut"], lut["attn_offsets"],
            lut["rev_attn_lut"], lut["rev_attn_offsets"],
            reverse_lut_mask,
            reverse_lut_bias,
            sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v,
            lut["max_kv_blocks"], lut["max_q_per_k"],
            lut["n_sparse_blocks_q"], lut["n_sparse_blocks_o"],
            scale, has_mask, has_bias,
            n_batches,
        )
    )


class BlockSparseFlashAttention(torch.autograd.Function):
    """Block-sparse Flash Attention with autograd support."""

    @staticmethod
    def forward(
        ctx, q, k, v, attention_mask, attention_bias,
        sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
        reverse_lut_q, reverse_lut_k, reverse_lut_v, reverse_lut_o,
        attn_lut, attn_offsets,
        rev_attn_lut, rev_attn_offsets,
        reverse_lut_mask, reverse_lut_bias,
        sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v,
        max_kv_blocks, max_q_per_k,
        n_sparse_blocks_q, n_sparse_blocks_o,
        scale, has_mask, has_bias,
        n_batches,
    ):
        SBS = sparsity_block_size

        # Strides for compressed block-sparse tensors
        q_b_s, q_r_s, q_c_s = stride(q)
        k_b_s, k_r_s, k_c_s = stride(k)
        v_b_s, v_r_s, v_c_s = stride(v)

        # Output shape is defined by sparsity_layout_o.
        o = torch.zeros(
            (n_sparse_blocks_o, SBS, SBS),
            device=q.device,
            dtype=q.dtype,
        )
        o_b_s, o_r_s, o_c_s = stride(o)

        # LSE: one value per row of each Q sequence block, for each batch
        # Shape: (n_batches, n_seq_blocks_q, SBS)
        lse = torch.full(
            (n_batches, n_seq_blocks_q, SBS),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )

        # Sparsity layout strides
        s_l_q_b_s, s_l_q_r_s, s_l_q_c_s = stride(sparsity_layout_q)
        s_l_k_b_s, s_l_k_r_s, s_l_k_c_s = stride(sparsity_layout_k)
        s_l_v_b_s, s_l_v_r_s, s_l_v_c_s = stride(sparsity_layout_v)
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)

        # Mask/bias strides
        if has_mask:
            mask_b_s, mask_r_s, mask_c_s = stride(attention_mask)
        else:
            mask_b_s = mask_r_s = mask_c_s = 0

        if has_bias:
            bias_b_s, bias_r_s, bias_c_s = stride(attention_bias)
        else:
            bias_b_s = bias_r_s = bias_c_s = 0

        # Dummy tensors for mask/bias reverse LUTs
        dummy_lut = torch.empty(0, device=q.device, dtype=torch.long)

        # Grid: one program per (batch, q_seq_block)
        grid = (n_batches, n_seq_blocks_q)

        flash_attention_fwd_kernel[grid](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            o, o_b_s, o_r_s, o_c_s,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            reverse_lut_q, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            reverse_lut_k, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            reverse_lut_v, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            reverse_lut_o, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            reverse_lut_mask if has_mask else dummy_lut,
            reverse_lut_bias if has_bias else dummy_lut,
            attn_lut, attn_offsets,
            lse,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v, max_kv_blocks,
            q.size(0),  # total Q blocks for bounds checking
            k.size(0),  # total K blocks
            v.size(0),  # total V blocks
            o.size(0),  # total O blocks
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            sparsity_block_size,
            scale,
            has_mask, has_bias,
        )

        ctx.save_for_backward(
            q, k, v, o, lse,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
            reverse_lut_q, reverse_lut_k, reverse_lut_v, reverse_lut_o,
            attn_lut, attn_offsets,
            rev_attn_lut, rev_attn_offsets,
            attention_mask if has_mask else torch.empty(0, device=q.device),
            attention_bias if has_bias else torch.empty(0, device=q.device),
            reverse_lut_mask if has_mask else torch.empty(0, device=q.device, dtype=torch.long),
            reverse_lut_bias if has_bias else torch.empty(0, device=q.device, dtype=torch.long),
        )
        ctx.sparsity_block_size = sparsity_block_size
        ctx.n_seq_blocks_q = n_seq_blocks_q
        ctx.n_seq_blocks_k = n_seq_blocks_k
        ctx.n_head_blocks_qk = n_head_blocks_qk
        ctx.n_head_blocks_v = n_head_blocks_v
        ctx.max_kv_blocks = max_kv_blocks
        ctx.max_q_per_k = max_q_per_k
        ctx.n_sparse_blocks_q = n_sparse_blocks_q
        ctx.n_sparse_blocks_o = n_sparse_blocks_o
        ctx.scale = scale
        ctx.has_mask = has_mask
        ctx.has_bias = has_bias
        ctx.n_batches = n_batches

        return o

    @staticmethod
    def backward(ctx, grad_output):
        (q, k, v, o, lse,
         sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
         reverse_lut_q, reverse_lut_k, reverse_lut_v, reverse_lut_o,
         attn_lut, attn_offsets,
         rev_attn_lut, rev_attn_offsets,
         attention_mask, attention_bias,
         reverse_lut_mask, reverse_lut_bias,
         ) = ctx.saved_tensors

        SBS = ctx.sparsity_block_size
        n_batches = ctx.n_batches
        n_seq_blocks_q = ctx.n_seq_blocks_q
        n_seq_blocks_k = ctx.n_seq_blocks_k
        n_head_blocks_qk = ctx.n_head_blocks_qk
        n_head_blocks_v = ctx.n_head_blocks_v
        has_mask = ctx.has_mask
        has_bias = ctx.has_bias

        q_b_s, q_r_s, q_c_s = stride(q)
        k_b_s, k_r_s, k_c_s = stride(k)
        v_b_s, v_r_s, v_c_s = stride(v)
        do_b_s, do_r_s, do_c_s = stride(grad_output)

        s_l_q_b_s, s_l_q_r_s, s_l_q_c_s = stride(sparsity_layout_q)
        s_l_k_b_s, s_l_k_r_s, s_l_k_c_s = stride(sparsity_layout_k)
        s_l_v_b_s, s_l_v_r_s, s_l_v_c_s = stride(sparsity_layout_v)
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)

        if has_mask:
            mask_b_s, mask_r_s, mask_c_s = stride(attention_mask)
        else:
            mask_b_s = mask_r_s = mask_c_s = 0

        if has_bias:
            bias_b_s, bias_r_s, bias_c_s = stride(attention_bias)
        else:
            bias_b_s = bias_r_s = bias_c_s = 0

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Precompute delta = rowsum(O * dO) per row of each Q block
        # Shape: (n_batches, n_seq_blocks_q, SBS)
        delta = torch.zeros(n_batches, n_seq_blocks_q, SBS, device=q.device, dtype=torch.float32)

        flash_attention_bwd_preprocess_kernel[(n_batches, n_seq_blocks_q)](
            o, grad_output, delta,
            reverse_lut_o,
            o.size(0),
            do_b_s, do_r_s, do_c_s,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            n_batches, n_seq_blocks_q, n_head_blocks_v,
            SBS,
        )

        # Allocate dbias if needed
        if has_bias:
            dbias = torch.zeros_like(attention_bias)
            dbias_b_s, dbias_r_s, dbias_c_s = stride(dbias)
        else:
            dbias = torch.empty(0, device=q.device, dtype=q.dtype)
            dbias_b_s = dbias_r_s = dbias_c_s = 0

        dummy_lut = torch.empty(0, device=q.device, dtype=torch.long)

        # dK, dV kernel
        flash_attention_bwd_dkdv_kernel[(n_batches, n_seq_blocks_k)](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            grad_output, do_b_s, do_r_s, do_c_s,
            dk, dv,
            dbias, dbias_b_s, dbias_r_s, dbias_c_s,
            lse, delta,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            reverse_lut_q, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            reverse_lut_k, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            reverse_lut_v, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            reverse_lut_o, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            reverse_lut_mask if has_mask else dummy_lut,
            reverse_lut_bias if has_bias else dummy_lut,
            rev_attn_lut, rev_attn_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v, ctx.max_q_per_k,
            q.size(0), k.size(0), v.size(0), grad_output.size(0),
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            SBS,
            ctx.scale,
            has_mask, has_bias,
        )

        # dQ kernel
        flash_attention_bwd_dq_kernel[(n_batches, n_seq_blocks_q)](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            grad_output, do_b_s, do_r_s, do_c_s,
            dq,
            lse, delta,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            reverse_lut_q, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            reverse_lut_k, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            reverse_lut_v, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            reverse_lut_o, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            reverse_lut_mask if has_mask else dummy_lut,
            reverse_lut_bias if has_bias else dummy_lut,
            attn_lut, attn_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v, ctx.max_kv_blocks,
            q.size(0), k.size(0), v.size(0), grad_output.size(0),
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            SBS,
            ctx.scale,
            has_mask, has_bias,
        )

        dbias_out = dbias if has_bias else None

        # Return grads for all forward arguments
        return (
            dq, dk, dv,
            None, dbias_out,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None,
            None, None, None, None, None,
            None, None, None,
            None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["o_ptr"],
)
@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, q_b_s, q_r_s, q_c_s,
    k_ptr, k_b_s, k_r_s, k_c_s,
    v_ptr, v_b_s, v_r_s, v_c_s,
    o_ptr, o_b_s, o_r_s, o_c_s,
    mask_ptr, mask_b_s, mask_r_s, mask_c_s,
    bias_ptr, bias_b_s, bias_r_s, bias_c_s,
    r_lut_q, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
    r_lut_k, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
    r_lut_v, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
    r_lut_o, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
    r_lut_mask, r_lut_bias,
    attn_lut_ptr, attn_offsets_ptr,
    lse_ptr,
    n_batches: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    n_seq_blocks_k: tl.constexpr,
    n_head_blocks_qk: tl.constexpr,
    n_head_blocks_v: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    total_q_blocks,
    total_k_blocks,
    total_v_blocks,
    total_o_blocks,
    total_mask_blocks,
    total_bias_blocks,
    sparsity_block_size,
    scale,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Flash attention forward kernel operating on compressed block-sparse tensors.

    Grid: (n_batches, n_seq_blocks_q)

    For each (batch, q_seq_block), iterates over all K seq blocks in the
    attention layout, reconstructs full Q and K vectors across head_dim blocks,
    computes the dot product, applies online softmax, and accumulates output.
    """
    pid_batch = tl.program_id(0)
    pid_q_seq = tl.program_id(1)

    offs_m = tl.arange(0, TRITON_BLOCK_SIZE)  # rows within the seq block
    offs_d = tl.arange(0, TRITON_BLOCK_SIZE)  # cols within a head block

    # Online softmax accumulators
    m_i = tl.full([TRITON_BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([TRITON_BLOCK_SIZE], dtype=tl.float32)

    # Get attention LUT for this (batch, q_seq) pair
    attn_offset_idx = pid_batch * n_seq_blocks_q + pid_q_seq
    attn_start = tl.load(attn_offsets_ptr + attn_offset_idx)
    attn_end = tl.load(attn_offsets_ptr + attn_offset_idx + 1)
    n_kv_blocks = attn_end - attn_start

    # Iterate over K sequence blocks
    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attn_lut_ptr + attn_start + kv_idx)

            # Compute S = Q @ K^T by accumulating across head_dim blocks
            S = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)

            for h in range(n_head_blocks_qk):
                rev_idx_q = tl.load(r_lut_q + (pid_batch * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)).to(tl.int32)
                rev_idx_k = tl.load(r_lut_k + (pid_batch * s_l_k_b_s + k_seq_block * s_l_k_r_s + h * s_l_k_c_s)).to(tl.int32)

                if rev_idx_q >= 0 and rev_idx_k >= 0:
                    q_blk_idx = (rev_idx_q * q_b_s +
                                 offs_m[:, None] * q_r_s +
                                 offs_d[None, :] * q_c_s)
                    q_blk = tl.load(q_ptr + q_blk_idx)

                    k_blk_idx = (rev_idx_k * k_b_s +
                                 offs_m[:, None] * k_r_s +
                                 offs_d[None, :] * k_c_s)
                    k_blk = tl.load(k_ptr + k_blk_idx)

                    S += tl.dot(q_blk, tl.trans(k_blk))

            # Scale scores
            qk_scale = scale * 1.44269504
            S = S * qk_scale

            # Apply mask if present
            if has_mask:
                rev_idx_mask = tl.load(r_lut_mask + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + k_seq_block)).to(tl.int32)
                if rev_idx_mask >= 0:
                    mask_blk_idx = (rev_idx_mask * mask_b_s +
                                    offs_m[:, None] * mask_r_s +
                                    offs_d[None, :] * mask_c_s)
                    mask_blk = tl.load(mask_ptr + mask_blk_idx)
                    S = tl.where(mask_blk != 0, float("-inf") * 1.44269504, S)

            # Apply bias if present
            if has_bias:
                rev_idx_bias = tl.load(r_lut_bias + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + k_seq_block)).to(tl.int32)
                if rev_idx_bias >= 0:
                    bias_blk_idx = (rev_idx_bias * bias_b_s +
                                    offs_m[:, None] * bias_r_s +
                                    offs_d[None, :] * bias_c_s)
                    bias_blk = tl.load(bias_ptr + bias_blk_idx)
                    S = S + bias_blk * 1.44269504

            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(S, axis=1))
            both_neg_inf = (m_i == float("-inf")) & (m_ij == float("-inf"))
            alpha = tl.where(both_neg_inf, 1.0, tl.math.exp2(m_i - m_ij))
            p_raw = tl.math.exp2(S - m_ij[:, None])
            p = tl.where((S == float("-inf")) & (m_ij[:, None] == float("-inf")), 0.0, p_raw)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            # For each head block, update the output accumulator
            for h in range(n_head_blocks_v):
                rev_idx_o = tl.load(r_lut_o + (pid_batch * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)).to(tl.int32)
                rev_idx_v = tl.load(r_lut_v + (pid_batch * s_l_v_b_s + k_seq_block * s_l_v_r_s + h * s_l_v_c_s)).to(tl.int32)

                if rev_idx_o >= 0:
                    o_blk_idx = (rev_idx_o * o_b_s +
                                 offs_m[:, None] * o_r_s +
                                 offs_d[None, :] * o_c_s)
                    o_blk = tl.load(o_ptr + o_blk_idx).to(tl.float32)
                    o_blk = o_blk * alpha[:, None]

                    if rev_idx_v >= 0:
                        v_blk_idx = (rev_idx_v * v_b_s +
                                     offs_m[:, None] * v_r_s +
                                     offs_d[None, :] * v_c_s)
                        v_blk = tl.load(v_ptr + v_blk_idx)
                        o_blk = o_blk + tl.dot(p.to(v_blk.dtype), v_blk).to(tl.float32)

                    tl.store(o_ptr + o_blk_idx, o_blk.to(o_ptr.dtype.element_ty))

            m_i = m_ij

    # Final normalization: O = O / l
    has_attention = l_i > 0
    l_safe = tl.where(has_attention, l_i, 1.0)

    for h in range(n_head_blocks_v):
        rev_idx_o = tl.load(r_lut_o + (pid_batch * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)).to(tl.int32)
        if rev_idx_o >= 0:
            o_blk_idx = (rev_idx_o * o_b_s +
                         offs_m[:, None] * o_r_s +
                         offs_d[None, :] * o_c_s)
            o_blk = tl.load(o_ptr + o_blk_idx).to(tl.float32)
            o_blk = o_blk / l_safe[:, None]
            o_blk = tl.where(has_attention[:, None], o_blk, 0.0)
            tl.store(o_ptr + o_blk_idx, o_blk.to(o_ptr.dtype.element_ty))

    # Store LSE
    lse_val = tl.where(has_attention, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(
        lse_ptr + pid_batch * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m,
        lse_val,
        mask=offs_m < TRITON_BLOCK_SIZE
    )


# ---------------------------------------------------------------------------
# Backward preprocess kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["delta_ptr"],
)
@triton.jit
def flash_attention_bwd_preprocess_kernel(
    o_ptr, do_ptr, delta_ptr,
    r_lut_o,
    total_o_blocks,
    o_b_s, o_r_s, o_c_s,
    s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
    n_batches: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    n_head_blocks: tl.constexpr,
    sparsity_block_size,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Compute delta = sum_h (O_h * dO_h).sum(dim=-1) for each (batch, q_seq, row)."""
    pid_batch = tl.program_id(0)
    pid_q_seq = tl.program_id(1)

    offs_m = tl.arange(0, TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, TRITON_BLOCK_SIZE)

    delta_acc = tl.zeros([TRITON_BLOCK_SIZE], dtype=tl.float32)

    for h in range(n_head_blocks):
        rev_idx_o = tl.load(r_lut_o + (pid_batch * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)).to(tl.int32)
        if rev_idx_o >= 0:
            blk_idx = (rev_idx_o * o_b_s +
                       offs_m[:, None] * o_r_s +
                       offs_d[None, :] * o_c_s)
            o_blk = tl.load(o_ptr + blk_idx).to(tl.float32)
            do_blk = tl.load(do_ptr + blk_idx).to(tl.float32)
            delta_acc += tl.sum(o_blk * do_blk, axis=1)

    tl.store(
        delta_ptr + pid_batch * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m,
        delta_acc
    )


# ---------------------------------------------------------------------------
# Backward dK, dV kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["dk_ptr", "dv_ptr", "dbias_ptr"],
)
@triton.jit
def flash_attention_bwd_dkdv_kernel(
    q_ptr, q_b_s, q_r_s, q_c_s,
    k_ptr, k_b_s, k_r_s, k_c_s,
    v_ptr, v_b_s, v_r_s, v_c_s,
    do_ptr, do_b_s, do_r_s, do_c_s,
    dk_ptr, dv_ptr,
    dbias_ptr, dbias_b_s, dbias_r_s, dbias_c_s,
    lse_ptr, delta_ptr,
    mask_ptr, mask_b_s, mask_r_s, mask_c_s,
    bias_ptr, bias_b_s, bias_r_s, bias_c_s,
    r_lut_q, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
    r_lut_k, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
    r_lut_v, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
    r_lut_o, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
    r_lut_mask, r_lut_bias,
    rev_attn_lut_ptr, rev_attn_offsets_ptr,
    n_batches: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    n_seq_blocks_k: tl.constexpr,
    n_head_blocks_qk: tl.constexpr,
    n_head_blocks_v: tl.constexpr,
    max_q_per_k: tl.constexpr,
    total_q_blocks,
    total_k_blocks,
    total_v_blocks,
    total_o_blocks,
    total_mask_blocks,
    total_bias_blocks,
    sparsity_block_size,
    scale,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Compute dK, dV, and optionally dBias gradients."""
    pid_batch = tl.program_id(0)
    pid_k_seq = tl.program_id(1)

    offs_m = tl.arange(0, TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, TRITON_BLOCK_SIZE)
    qk_scale = scale * 1.44269504

    # Get reverse attention LUT: which Q blocks attend to this K block
    rev_offset_idx = pid_batch * n_seq_blocks_k + pid_k_seq
    rev_start = tl.load(rev_attn_offsets_ptr + rev_offset_idx)
    rev_end = tl.load(rev_attn_offsets_ptr + rev_offset_idx + 1)
    n_q_blocks = rev_end - rev_start

    for q_idx in range(max_q_per_k):
        if q_idx < n_q_blocks:
            q_seq_block = tl.load(rev_attn_lut_ptr + rev_start + q_idx)

            # Recompute S = Q @ K^T (across head blocks)
            S = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_qk):
                rev_idx_q = tl.load(r_lut_q + (pid_batch * s_l_q_b_s + q_seq_block * s_l_q_r_s + h * s_l_q_c_s)).to(tl.int32)
                rev_idx_k = tl.load(r_lut_k + (pid_batch * s_l_k_b_s + pid_k_seq * s_l_k_r_s + h * s_l_k_c_s)).to(tl.int32)
                if rev_idx_q >= 0 and rev_idx_k >= 0:
                    q_blk_idx = (rev_idx_q * q_b_s + offs_m[:, None] * q_r_s + offs_d[None, :] * q_c_s)
                    q_blk = tl.load(q_ptr + q_blk_idx)
                    k_blk_idx = (rev_idx_k * k_b_s + offs_m[:, None] * k_r_s + offs_d[None, :] * k_c_s)
                    k_blk = tl.load(k_ptr + k_blk_idx)
                    S += tl.dot(q_blk, tl.trans(k_blk))

            S = S * qk_scale

            # Apply mask
            if has_mask:
                rev_idx_mask = tl.load(r_lut_mask + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + q_seq_block * n_seq_blocks_k + pid_k_seq)).to(tl.int32)
                if rev_idx_mask >= 0:
                    mask_blk_idx = (rev_idx_mask * mask_b_s + offs_m[:, None] * mask_r_s + offs_d[None, :] * mask_c_s)
                    mask_blk = tl.load(mask_ptr + mask_blk_idx)
                    S = tl.where(mask_blk != 0, float("-inf") * 1.44269504, S)

            # Apply bias
            if has_bias:
                rev_idx_bias = tl.load(r_lut_bias + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + q_seq_block * n_seq_blocks_k + pid_k_seq)).to(tl.int32)
                if rev_idx_bias >= 0:
                    bias_blk_idx = (rev_idx_bias * bias_b_s + offs_m[:, None] * bias_r_s + offs_d[None, :] * bias_c_s)
                    bias_blk = tl.load(bias_ptr + bias_blk_idx)
                    S = S + bias_blk * 1.44269504

            # Recompute P from S and saved LSE
            m = tl.load(
                lse_ptr + pid_batch * n_seq_blocks_q * TRITON_BLOCK_SIZE + q_seq_block * TRITON_BLOCK_SIZE + offs_m
            )
            Di = tl.load(
                delta_ptr + pid_batch * n_seq_blocks_q * TRITON_BLOCK_SIZE + q_seq_block * TRITON_BLOCK_SIZE + offs_m
            )

            valid_lse = m > float("-inf")
            safe_m = tl.where(valid_lse, m, 0.0)
            p = tl.math.exp2(S - safe_m[:, None])
            p = tl.where(valid_lse[:, None], p, 0.0)

            # Compute dp = sum_h dO_h @ V_h^T
            dp = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_v):
                rev_idx_o_h = tl.load(r_lut_o + (pid_batch * s_l_o_b_s + q_seq_block * s_l_o_r_s + h * s_l_o_c_s)).to(tl.int32)
                rev_idx_v_h = tl.load(r_lut_v + (pid_batch * s_l_v_b_s + pid_k_seq * s_l_v_r_s + h * s_l_v_c_s)).to(tl.int32)
                if rev_idx_o_h >= 0 and rev_idx_v_h >= 0:
                    do_blk_idx = (rev_idx_o_h * do_b_s + offs_m[:, None] * do_r_s + offs_d[None, :] * do_c_s)
                    do_blk = tl.load(do_ptr + do_blk_idx)
                    v_blk_idx = (rev_idx_v_h * v_b_s + offs_m[:, None] * v_r_s + offs_d[None, :] * v_c_s)
                    v_blk = tl.load(v_ptr + v_blk_idx)
                    dp += tl.dot(do_blk, tl.trans(v_blk)).to(tl.float32)

            # ds = P * (dp - Di)
            ds = p * (dp - Di[:, None])

            # Accumulate dK across Q/K head blocks
            for h in range(n_head_blocks_qk):
                rev_idx_q_h = tl.load(r_lut_q + (pid_batch * s_l_q_b_s + q_seq_block * s_l_q_r_s + h * s_l_q_c_s)).to(tl.int32)
                rev_idx_k_h = tl.load(r_lut_k + (pid_batch * s_l_k_b_s + pid_k_seq * s_l_k_r_s + h * s_l_k_c_s)).to(tl.int32)

                if rev_idx_q_h >= 0:
                    # dK += ds^T @ Q * scale
                    if rev_idx_k_h >= 0:
                        q_blk_idx = (rev_idx_q_h * q_b_s + offs_m[:, None] * q_r_s + offs_d[None, :] * q_c_s)
                        q_blk = tl.load(q_ptr + q_blk_idx)
                        dk_blk_idx = (rev_idx_k_h * k_b_s + offs_m[:, None] * k_r_s + offs_d[None, :] * k_c_s)
                        dk_blk = tl.load(dk_ptr + dk_blk_idx).to(tl.float32)
                        dk_blk += tl.dot(tl.trans(ds.to(q_blk.dtype)), q_blk).to(tl.float32) * scale
                        tl.store(dk_ptr + dk_blk_idx, dk_blk.to(dk_ptr.dtype.element_ty))

            # Accumulate dV across V/output head blocks
            for h in range(n_head_blocks_v):
                rev_idx_o_h = tl.load(r_lut_o + (pid_batch * s_l_o_b_s + q_seq_block * s_l_o_r_s + h * s_l_o_c_s)).to(tl.int32)
                rev_idx_v_h = tl.load(r_lut_v + (pid_batch * s_l_v_b_s + pid_k_seq * s_l_v_r_s + h * s_l_v_c_s)).to(tl.int32)
                if rev_idx_o_h >= 0 and rev_idx_v_h >= 0:
                    do_blk_idx = (rev_idx_o_h * do_b_s + offs_m[:, None] * do_r_s + offs_d[None, :] * do_c_s)
                    do_blk = tl.load(do_ptr + do_blk_idx)
                    dv_blk_idx = (rev_idx_v_h * v_b_s + offs_m[:, None] * v_r_s + offs_d[None, :] * v_c_s)
                    dv_blk = tl.load(dv_ptr + dv_blk_idx).to(tl.float32)
                    dv_blk += tl.dot(tl.trans(p.to(do_blk.dtype)), do_blk).to(tl.float32)
                    tl.store(dv_ptr + dv_blk_idx, dv_blk.to(dv_ptr.dtype.element_ty))

            # dBias
            if has_bias:
                rev_idx_bias = tl.load(r_lut_bias + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + q_seq_block * n_seq_blocks_k + pid_k_seq)).to(tl.int32)
                if rev_idx_bias >= 0:
                    ds_bias = ds * 1.44269504
                    dbias_blk_idx = (rev_idx_bias * dbias_b_s + offs_m[:, None] * dbias_r_s + offs_d[None, :] * dbias_c_s)
                    dbias_blk = tl.load(dbias_ptr + dbias_blk_idx).to(tl.float32)
                    dbias_blk += ds_bias
                    tl.store(dbias_ptr + dbias_blk_idx, dbias_blk.to(dbias_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Backward dQ kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["dq_ptr"],
)
@triton.jit
def flash_attention_bwd_dq_kernel(
    q_ptr, q_b_s, q_r_s, q_c_s,
    k_ptr, k_b_s, k_r_s, k_c_s,
    v_ptr, v_b_s, v_r_s, v_c_s,
    do_ptr, do_b_s, do_r_s, do_c_s,
    dq_ptr,
    lse_ptr, delta_ptr,
    mask_ptr, mask_b_s, mask_r_s, mask_c_s,
    bias_ptr, bias_b_s, bias_r_s, bias_c_s,
    r_lut_q, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
    r_lut_k, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
    r_lut_v, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
    r_lut_o, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
    r_lut_mask, r_lut_bias,
    attn_lut_ptr, attn_offsets_ptr,
    n_batches: tl.constexpr,
    n_seq_blocks_q: tl.constexpr,
    n_seq_blocks_k: tl.constexpr,
    n_head_blocks_qk: tl.constexpr,
    n_head_blocks_v: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    total_q_blocks,
    total_k_blocks,
    total_v_blocks,
    total_o_blocks,
    total_mask_blocks,
    total_bias_blocks,
    sparsity_block_size,
    scale,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Compute dQ gradients."""
    pid_batch = tl.program_id(0)
    pid_q_seq = tl.program_id(1)

    offs_m = tl.arange(0, TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, TRITON_BLOCK_SIZE)
    qk_scale = scale * 1.44269504

    m = tl.load(lse_ptr + pid_batch * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m)
    Di = tl.load(delta_ptr + pid_batch * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m)

    attn_offset_idx = pid_batch * n_seq_blocks_q + pid_q_seq
    attn_start = tl.load(attn_offsets_ptr + attn_offset_idx)
    attn_end = tl.load(attn_offsets_ptr + attn_offset_idx + 1)
    n_kv_blocks = attn_end - attn_start

    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attn_lut_ptr + attn_start + kv_idx)

            # Recompute S = Q @ K^T across head blocks
            S = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_qk):
                rev_idx_q = tl.load(r_lut_q + (pid_batch * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)).to(tl.int32)
                rev_idx_k = tl.load(r_lut_k + (pid_batch * s_l_k_b_s + k_seq_block * s_l_k_r_s + h * s_l_k_c_s)).to(tl.int32)
                if rev_idx_q >= 0 and rev_idx_k >= 0:
                    q_blk_idx = (rev_idx_q * q_b_s + offs_m[:, None] * q_r_s + offs_d[None, :] * q_c_s)
                    q_blk = tl.load(q_ptr + q_blk_idx)
                    k_blk_idx = (rev_idx_k * k_b_s + offs_m[:, None] * k_r_s + offs_d[None, :] * k_c_s)
                    k_blk = tl.load(k_ptr + k_blk_idx)
                    S += tl.dot(q_blk, tl.trans(k_blk))

            S = S * qk_scale

            if has_mask:
                rev_idx_mask = tl.load(r_lut_mask + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + k_seq_block)).to(tl.int32)
                if rev_idx_mask >= 0:
                    mask_blk_idx = (rev_idx_mask * mask_b_s + offs_m[:, None] * mask_r_s + offs_d[None, :] * mask_c_s)
                    mask_blk = tl.load(mask_ptr + mask_blk_idx)
                    S = tl.where(mask_blk != 0, float("-inf") * 1.44269504, S)

            if has_bias:
                rev_idx_bias = tl.load(r_lut_bias + (pid_batch * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + k_seq_block)).to(tl.int32)
                if rev_idx_bias >= 0:
                    bias_blk_idx = (rev_idx_bias * bias_b_s + offs_m[:, None] * bias_r_s + offs_d[None, :] * bias_c_s)
                    bias_blk = tl.load(bias_ptr + bias_blk_idx)
                    S = S + bias_blk * 1.44269504

            valid_lse = m > float("-inf")
            safe_m = tl.where(valid_lse, m, 0.0)
            p = tl.math.exp2(S - safe_m[:, None])
            p = tl.where(valid_lse[:, None], p, 0.0)

            # dp = sum_h dO_h @ V_h^T
            dp = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_v):
                rev_idx_o_h = tl.load(r_lut_o + (pid_batch * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)).to(tl.int32)
                rev_idx_v_h = tl.load(r_lut_v + (pid_batch * s_l_v_b_s + k_seq_block * s_l_v_r_s + h * s_l_v_c_s)).to(tl.int32)
                if rev_idx_o_h >= 0 and rev_idx_v_h >= 0:
                    do_blk_idx = (rev_idx_o_h * do_b_s + offs_m[:, None] * do_r_s + offs_d[None, :] * do_c_s)
                    do_blk = tl.load(do_ptr + do_blk_idx)
                    v_blk_idx = (rev_idx_v_h * v_b_s + offs_m[:, None] * v_r_s + offs_d[None, :] * v_c_s)
                    v_blk = tl.load(v_ptr + v_blk_idx)
                    dp += tl.dot(do_blk, tl.trans(v_blk)).to(tl.float32)

            # ds = P * (dp - Di)
            ds = p * (dp - Di[:, None])

            # dQ += ds @ K * scale (for each head block)
            for h in range(n_head_blocks_qk):
                rev_idx_q_h = tl.load(r_lut_q + (pid_batch * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)).to(tl.int32)
                rev_idx_k_h = tl.load(r_lut_k + (pid_batch * s_l_k_b_s + k_seq_block * s_l_k_r_s + h * s_l_k_c_s)).to(tl.int32)
                if rev_idx_q_h >= 0 and rev_idx_k_h >= 0:
                    k_blk_idx = (rev_idx_k_h * k_b_s + offs_m[:, None] * k_r_s + offs_d[None, :] * k_c_s)
                    k_blk = tl.load(k_ptr + k_blk_idx)
                    dq_blk_idx = (rev_idx_q_h * q_b_s + offs_m[:, None] * q_r_s + offs_d[None, :] * q_c_s)
                    dq_blk = tl.load(dq_ptr + dq_blk_idx).to(tl.float32)
                    dq_blk += tl.dot(ds.to(k_blk.dtype), k_blk).to(tl.float32) * scale
                    tl.store(dq_ptr + dq_blk_idx, dq_blk.to(dq_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# LUT building
# ---------------------------------------------------------------------------

def flash_attention_build_lut(
    attention_layout: Tensor,
    sparsity_layout_q: Tensor = None,
    sparsity_layout_k: Tensor = None,
    sparsity_layout_v: Tensor = None,
    n_seq_blocks_q: int = None,
    n_seq_blocks_k: int = None,
    n_head_blocks: int = None,
    sparsity_layout_o: Tensor = None,
    n_head_blocks_v: int = None,
) -> dict:
    """Build lookup tables for block-sparse flash attention.

    Args:
        attention_layout: ``(n_batches, n_seq_blocks_q, n_seq_blocks_k)``
        sparsity_layout_q: ``(n_batches, n_seq_blocks_q, n_head_blocks)``
        sparsity_layout_k: ``(n_batches, n_seq_blocks_k, n_head_blocks)``
        sparsity_layout_v: ``(n_batches, n_seq_blocks_k, n_head_blocks)``
        n_seq_blocks_q: Number of Q sequence blocks.
        n_seq_blocks_k: Number of K sequence blocks.
        n_head_blocks: Number of Q/K head dim blocks.
        sparsity_layout_o: Optional output layout
            ``(n_batches, n_seq_blocks_q, n_head_blocks_v)``.
        n_head_blocks_v: Number of V/output head dim blocks.

    Returns:
        Dictionary of lookup tables.
    """
    n_batches = attention_layout.shape[0]
    if n_seq_blocks_q is None:
        n_seq_blocks_q = attention_layout.shape[1]
    if n_seq_blocks_k is None:
        n_seq_blocks_k = attention_layout.shape[2]

    # Forward attention LUT: for each (batch, q_seq), list of k_seq blocks
    attn_lut, attn_offsets, max_kv_blocks = _build_attention_lut_fast(
        attention_layout, n_batches, n_seq_blocks_q, n_seq_blocks_k
    )

    # Reverse attention LUT: for each (batch, k_seq), list of q_seq blocks
    attention_layout_t = attention_layout.transpose(1, 2).contiguous()
    rev_attn_lut, rev_attn_offsets, max_q_per_k = _build_attention_lut_fast(
        attention_layout_t, n_batches, n_seq_blocks_k, n_seq_blocks_q
    )

    result = {
        "attn_lut": attn_lut,
        "attn_offsets": attn_offsets,
        "max_kv_blocks": max_kv_blocks,
        "rev_attn_lut": rev_attn_lut,
        "rev_attn_offsets": rev_attn_offsets,
        "max_q_per_k": max_q_per_k,
    }

    if sparsity_layout_q is not None:
        result["reverse_lut_q"] = _build_reverse_lut(sparsity_layout_q)
        result["n_sparse_blocks_q"] = int(sparsity_layout_q.sum().item())

    if sparsity_layout_k is not None:
        result["reverse_lut_k"] = _build_reverse_lut(sparsity_layout_k)

    if sparsity_layout_v is not None:
        result["reverse_lut_v"] = _build_reverse_lut(sparsity_layout_v)

    if sparsity_layout_o is not None:
        result["reverse_lut_o"] = _build_reverse_lut(sparsity_layout_o)
        result["n_sparse_blocks_o"] = int(sparsity_layout_o.sum().item())
    elif sparsity_layout_q is not None and (n_head_blocks_v is None or n_head_blocks_v == sparsity_layout_q.size(2)):
        # Backward-compatible path: output layout equals Q layout.
        result["reverse_lut_o"] = result["reverse_lut_q"]
        result["n_sparse_blocks_o"] = result["n_sparse_blocks_q"]

    return result


def _build_reverse_lut(sparsity_layout: Tensor) -> Tensor:
    """Build reverse sparsity LUT: maps flat index -> compressed block index or -1."""
    flat = sparsity_layout.reshape(-1)
    reverse_lut = (
        (torch.cumsum(flat, dim=-1) - 1) * (flat == 1)
        - (1 * (flat == 0))
    )
    return reverse_lut.contiguous()


def _build_attention_lut_fast(
    attention_layout: Tensor,
    n_batches: int,
    n_blocks_row: int,
    n_blocks_col: int,
) -> Tuple[Tensor, Tensor, int]:
    """Build attention LUT efficiently (CSR-like format)."""
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

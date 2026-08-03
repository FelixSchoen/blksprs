import math
from numbers import Real

import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs_exact
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import as_base_tensor, stride, build_packed_indices, can_use_int32_indexing, \
    cast_for_autocast, prepare_layout_cache, finalize_layout_cache, INT32_INDEX_MAX
from blksprs.utils.validation import validate_contiguous, validate_device, validate_dtype_float, \
    validate_dimensions, validate_sparsity, validate_sparsity_block_size, validate_shape, ensure_contiguous, \
    validate_sparsity_layout, validate_binary


@torch.amp.custom_fwd(device_type="cuda")
def flash_attention(q: BlksprsTensor, sparsity_layout_q: Tensor,
                    k: BlksprsTensor, sparsity_layout_k: Tensor,
                    v: BlksprsTensor, sparsity_layout_v: Tensor,
                    attention_layout: Tensor,
                    sparsity_block_size: int,
                    scale: float | None = None,
                    attention_mask: BlksprsTensor | None = None, sparsity_layout_mask: Tensor | None = None,
                    attention_bias: BlksprsTensor | None = None, sparsity_layout_bias: Tensor | None = None,
                    layout_cache: dict | None = None,
                    sparsity_layout_o: Tensor | None = None) -> BlksprsTensor:
    """Computes block-sparse Flash Attention on compressed tensors.

    All inputs use the standard BLK-SPRS compressed format: tensors have shape
    ``(n_sparse_blocks, sparsity_block_size, sparsity_block_size)`` with an
    accompanying sparsity layout.

    Note:
        The ``attention_mask`` convention used here (``True`` = **masked/ignored**)
        is the same as :meth:`torch.Tensor.masked_fill` but **opposite** to
        :func:`torch.nn.functional.scaled_dot_product_attention`, where a boolean
        ``attn_mask`` of ``True`` means the position **participates** in attention.

        Supported sparsity block sizes are 16, 32, and 64. ``attention_mask`` and
        ``sparsity_layout_mask`` must be provided together, as must
        ``attention_bias`` and ``sparsity_layout_bias``.

    Args:
        q (BlksprsTensor): The query tensor in compressed form.
            Sparsity layout shape: ``(n_batches, seq_q // bs, head_dim // bs)``.
        sparsity_layout_q (Tensor): The sparsity layout of ``q``.
        k (BlksprsTensor): The key tensor in compressed form.
            Sparsity layout shape: ``(n_batches, seq_k // bs, head_dim // bs)``.
        sparsity_layout_k (Tensor): The sparsity layout of ``k``.
        v (BlksprsTensor): The value tensor in compressed form.
        sparsity_layout_v (Tensor): The sparsity layout of ``v``.
        attention_layout (Tensor): The block attention pattern
            ``(n_batches, seq_q // bs, seq_k // bs)`` indicating which Q-K block
            pairs participate in attention.
        sparsity_block_size (int): The size of the sparsity blocks.
        scale (float, optional): The real-valued attention scale (default ``1/sqrt(head_dim)``).
        attention_mask (BlksprsTensor, optional): A binary mask in compressed form where a non-zero value marks an
            ignored position. The mask does not participate in gradient computation (default ``None``).
        sparsity_layout_mask (Tensor, optional): The sparsity layout for the mask (default ``None``).
        attention_bias (BlksprsTensor, optional): An additive bias in compressed form, applied before softmax. The bias
            supports gradient computation (default ``None``).
        sparsity_layout_bias (Tensor, optional): The sparsity layout for the bias (default ``None``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).
        sparsity_layout_o (Tensor, optional): The output sparsity layout. When omitted, it is derived from
            the structural product of ``attention_layout`` and ``sparsity_layout_v`` (default ``None``).

    Returns:
        BlksprsTensor: The attention output in compressed form.

    """
    q, k, v, attention_layout = ensure_contiguous(q, k, v, attention_layout)
    q, k, v = cast_for_autocast(q, k, v)

    validate_dimensions(q, k, v)
    validate_contiguous(q, k, v)
    validate_dtype_float(q, k, v)
    validate_device(q, k, v)
    validate_sparsity(sparsity_block_size, (q, sparsity_layout_q), (k, sparsity_layout_k), (v, sparsity_layout_v))
    validate_sparsity_block_size(sparsity_block_size, q, k, v)
    if sparsity_block_size > 64:
        raise ValueError("Flash Attention sparsity block size must be at most 64")

    validate_dimensions(attention_layout)
    validate_contiguous(attention_layout)
    validate_device(attention_layout, q)
    validate_sparsity_layout(attention_layout)

    n_batches = sparsity_layout_q.size(0)
    n_seq_blocks_q = sparsity_layout_q.size(1)
    n_head_blocks_qk = sparsity_layout_q.size(2)
    n_seq_blocks_k = sparsity_layout_k.size(1)
    n_head_blocks_v = sparsity_layout_v.size(2)

    if sparsity_layout_k.size(0) != n_batches or sparsity_layout_k.size(2) != n_head_blocks_qk:
        raise ValueError("K sparsity layout must be compatible with Q")
    if sparsity_layout_v.size(0) != n_batches or sparsity_layout_v.size(1) != n_seq_blocks_k:
        raise ValueError("V sparsity layout must be compatible with K")

    expected_attn_shape = (n_batches, n_seq_blocks_q, n_seq_blocks_k)
    validate_shape(attention_layout, expected_attn_shape, "attention_layout")

    if sparsity_layout_o is not None:
        sparsity_layout_o = ensure_contiguous(sparsity_layout_o)
        validate_dimensions(sparsity_layout_o)
        validate_contiguous(sparsity_layout_o)
        validate_device(sparsity_layout_o, q)

    expected_output_shape = (n_batches, n_seq_blocks_q, n_head_blocks_v)
    if sparsity_layout_o is not None:
        validate_sparsity_layout(sparsity_layout_o)
        validate_shape(sparsity_layout_o, expected_output_shape, "sparsity_layout_o")

    if scale is None:
        head_dimension = n_head_blocks_qk * sparsity_block_size
        scale = 1.0 if head_dimension == 0 else 1.0 / math.sqrt(head_dimension)
    elif not isinstance(scale, Real):
        raise TypeError("Attention scale must be a real number or None")
    else:
        scale = float(scale)

    # Resolve optional mask
    if (attention_mask is None) != (sparsity_layout_mask is None):
        raise ValueError("attention_mask and sparsity_layout_mask must be provided together")

    attention_mask_value: Tensor
    has_mask = attention_mask is not None
    if attention_mask is not None and sparsity_layout_mask is not None:
        attention_mask_value = ensure_contiguous(attention_mask)
        sparsity_layout_mask = ensure_contiguous(sparsity_layout_mask)
        validate_dimensions(attention_mask_value, sparsity_layout_mask)
        validate_contiguous(attention_mask_value, sparsity_layout_mask)
        validate_device(attention_mask_value, sparsity_layout_mask, q)
        if attention_mask_value.dtype != torch.bool:
            validate_dtype_float(attention_mask_value)
        validate_binary(attention_mask_value)
        validate_sparsity(sparsity_block_size, (attention_mask_value, sparsity_layout_mask))
        validate_shape(sparsity_layout_mask, expected_attn_shape, "sparsity_layout_mask")
    else:
        attention_mask_value = torch.empty(0, device=q.device, dtype=q.dtype)

    # Resolve optional bias
    if (attention_bias is None) != (sparsity_layout_bias is None):
        raise ValueError("attention_bias and sparsity_layout_bias must be provided together")

    attention_bias_value: Tensor
    has_bias = attention_bias is not None
    if attention_bias is not None and sparsity_layout_bias is not None:
        attention_bias_value = ensure_contiguous(attention_bias)
        sparsity_layout_bias = ensure_contiguous(sparsity_layout_bias)
        validate_dimensions(attention_bias_value, sparsity_layout_bias)
        validate_contiguous(attention_bias_value, sparsity_layout_bias)
        validate_device(attention_bias_value, sparsity_layout_bias, q)
        validate_dtype_float(attention_bias_value)
        validate_sparsity(sparsity_block_size, (attention_bias_value, sparsity_layout_bias))
        validate_shape(sparsity_layout_bias, expected_attn_shape, "sparsity_layout_bias")
    else:
        attention_bias_value = torch.empty(0, device=q.device, dtype=q.dtype)

    layout_cache = flash_attention_build_layout_cache(
        attention_layout,
        sparsity_layout_q,
        sparsity_layout_k,
        sparsity_layout_v,
        n_seq_blocks_q,
        n_seq_blocks_k,
        n_head_blocks_qk,
        sparsity_layout_o=sparsity_layout_o,
        n_head_blocks_v=n_head_blocks_v,
        sparsity_layout_mask=sparsity_layout_mask,
        sparsity_layout_bias=sparsity_layout_bias,
        layout_cache=layout_cache,
    )
    sparsity_layout_o = layout_cache["sparsity_layout_o"]

    dummy_packed_indices = torch.empty(0, device=q.device, dtype=torch.long)
    packed_indices_mask = layout_cache["packed_indices_mask"] if has_mask else dummy_packed_indices
    packed_indices_bias = layout_cache["packed_indices_bias"] if has_bias else dummy_packed_indices

    return BlksprsTensor.wrap(
        _FlashAttentionAutograd.apply(
            q, k, v,
            attention_mask_value, attention_bias_value,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
            layout_cache["packed_indices_q"], layout_cache["packed_indices_k"],
            layout_cache["packed_indices_v"], layout_cache["packed_indices_o"],
            layout_cache["key_indices"], layout_cache["key_offsets"],
            layout_cache["query_indices"], layout_cache["query_offsets"],
            packed_indices_mask, packed_indices_bias,
            sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v,
            layout_cache["max_keys_per_query"], layout_cache["max_queries_per_key"],
            layout_cache["n_sparse_blocks_o"],
            scale, has_mask, has_bias,
            n_batches))


class _FlashAttentionAutograd(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, q, k, v, attention_mask, attention_bias,
        sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
        packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
        key_indices, key_offsets,
        query_indices, query_offsets,
        packed_indices_mask, packed_indices_bias,
        sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k,
        n_head_blocks_qk, n_head_blocks_v,
        max_keys_per_query, max_queries_per_key,
        n_sparse_blocks_o,
        scale, has_mask, has_bias,
        n_batches,
    ):
        # The online softmax spans multiple key blocks. Keep the intermediate
        # numerator in float32 until the final normalisation; storing it in the
        # input dtype after every key block can accumulate large rounding errors
        # or overflow for long sequences.
        output_accumulator = torch.zeros(
            size=(n_sparse_blocks_o, sparsity_block_size, sparsity_block_size),
            dtype=torch.float32,
            device=q.device,
        )

        lse = torch.full(size=(n_batches, n_seq_blocks_q, sparsity_block_size),
                         fill_value=float("-inf"), dtype=torch.float32, device=q.device)

        q_b, q_r, q_c = q.size()
        q_b_s, q_r_s, q_c_s = stride(q)
        k_b, k_r, k_c = k.size()
        k_b_s, k_r_s, k_c_s = stride(k)
        v_b, v_r, v_c = v.size()
        v_b_s, v_r_s, v_c_s = stride(v)
        o_b, o_r, o_c = output_accumulator.size()
        o_b_s, o_r_s, o_c_s = stride(output_accumulator)
        s_l_q_b, s_l_q_r, s_l_q_c = sparsity_layout_q.size()
        s_l_q_b_s, s_l_q_r_s, s_l_q_c_s = stride(sparsity_layout_q)
        s_l_k_b, s_l_k_r, s_l_k_c = sparsity_layout_k.size()
        s_l_k_b_s, s_l_k_r_s, s_l_k_c_s = stride(sparsity_layout_k)
        s_l_v_b, s_l_v_r, s_l_v_c = sparsity_layout_v.size()
        s_l_v_b_s, s_l_v_r_s, s_l_v_c_s = stride(sparsity_layout_v)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)

        if has_mask:
            mask_b_s, mask_r_s, mask_c_s = stride(attention_mask)
        else:
            mask_b_s = mask_r_s = mask_c_s = 0

        if has_bias:
            bias_b_s, bias_r_s, bias_c_s = stride(attention_bias)
        else:
            bias_b_s = bias_r_s = bias_c_s = 0

        dummy_packed_indices = torch.empty(0, device=q.device, dtype=torch.long)

        triton_grid = lambda meta: [n_batches,
                                    n_seq_blocks_q]

        use_int64 = not can_use_int32_indexing(
            q,
            k,
            v,
            output_accumulator,
            attention_mask if has_mask else None,
            attention_bias if has_bias else None,
            packed_indices_q,
            packed_indices_k,
            packed_indices_v,
            packed_indices_o,
            packed_indices_mask if has_mask else None,
            packed_indices_bias if has_bias else None,
            key_indices,
            key_offsets,
            lse,
        )

        flash_attention_kernel[triton_grid](
            q,
            q_b, q_b_s, q_r_s, q_c_s,
            k,
            k_b, k_b_s, k_r_s, k_c_s,
            v,
            v_b, v_b_s, v_r_s, v_c_s,
            output_accumulator,
            o_b, o_b_s, o_r_s, o_c_s,
            attention_mask,
            mask_b_s, mask_r_s, mask_c_s,
            attention_bias,
            bias_b_s, bias_r_s, bias_c_s,
            packed_indices_q,
            s_l_q_b, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            packed_indices_k,
            s_l_k_b, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            packed_indices_v,
            s_l_v_b, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            packed_indices_o,
            s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            packed_indices_mask if has_mask else dummy_packed_indices,
            packed_indices_bias if has_bias else dummy_packed_indices,
            key_indices, key_offsets,
            lse,
            n_batches, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v, max_keys_per_query,
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            scale,
            has_mask, has_bias,
            sparsity_block_size,
            USE_INT64=use_int64)

        output = output_accumulator.to(q.dtype)

        ctx.save_for_backward(
            q, k, v, output, lse,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
            packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
            key_indices, key_offsets,
            query_indices, query_offsets,
            attention_mask if has_mask else torch.empty(0, device=q.device),
            attention_bias if has_bias else torch.empty(0, device=q.device),
            packed_indices_mask if has_mask else torch.empty(0, device=q.device, dtype=torch.long),
            packed_indices_bias if has_bias else torch.empty(0, device=q.device, dtype=torch.long),
        )
        ctx.sparsity_block_size = sparsity_block_size
        ctx.n_seq_blocks_q = n_seq_blocks_q
        ctx.n_seq_blocks_k = n_seq_blocks_k
        ctx.n_head_blocks_qk = n_head_blocks_qk
        ctx.n_head_blocks_v = n_head_blocks_v
        ctx.max_keys_per_query = max_keys_per_query
        ctx.max_queries_per_key = max_queries_per_key
        ctx.scale = scale
        ctx.has_mask = has_mask
        ctx.has_bias = has_bias
        ctx.n_batches = n_batches

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        (q, k, v, o, lse,
         sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
         packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
         key_indices, key_offsets,
         query_indices, query_offsets,
         attention_mask, attention_bias,
         packed_indices_mask, packed_indices_bias,
         ) = ctx.saved_tensors

        sparsity_block_size = ctx.sparsity_block_size
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

        # dQ, dK, and dV are accumulated across attended blocks. Accumulate in
        # float32 and cast once at the end for the same reason as the forward
        # online-softmax numerator.
        dq_accumulator = torch.zeros_like(q, dtype=torch.float32)
        dk_accumulator = torch.zeros_like(k, dtype=torch.float32)
        dv_accumulator = torch.zeros_like(v, dtype=torch.float32)

        # Precompute delta = rowsum(O * dO)
        delta = torch.zeros(n_batches, n_seq_blocks_q, sparsity_block_size,
                            device=q.device, dtype=torch.float32)

        use_int64_preprocess = not can_use_int32_indexing(
            o,
            grad_output,
            delta,
            packed_indices_o,
        )

        flash_attention_kernel_bwd_preprocess[(n_batches, n_seq_blocks_q)](
            o, grad_output, delta,
            packed_indices_o,
            o.size(0),
            do_b_s, do_r_s, do_c_s,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            n_batches, n_seq_blocks_q, n_head_blocks_v,
            sparsity_block_size,
            USE_INT64=use_int64_preprocess)

        # Allocate dbias if needed
        if has_bias:
            dbias_accumulator = torch.zeros_like(attention_bias, dtype=torch.float32)
            dbias_b_s, dbias_r_s, dbias_c_s = stride(dbias_accumulator)
        else:
            dbias_accumulator = torch.empty(0, device=q.device, dtype=torch.float32)
            dbias_b_s = dbias_r_s = dbias_c_s = 0

        dummy_packed_indices = torch.empty(0, device=q.device, dtype=torch.long)

        # dK, dV kernel
        use_int64_dkdv = not can_use_int32_indexing(
            q,
            k,
            v,
            grad_output,
            dk_accumulator,
            dv_accumulator,
            dbias_accumulator if has_bias else None,
            lse,
            delta,
            attention_mask if has_mask else None,
            attention_bias if has_bias else None,
            packed_indices_q,
            packed_indices_k,
            packed_indices_v,
            packed_indices_o,
            packed_indices_mask if has_mask else None,
            packed_indices_bias if has_bias else None,
            query_indices,
            query_offsets,
        )

        flash_attention_kernel_bwd_dkdv[(n_batches, n_seq_blocks_k)](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            grad_output, do_b_s, do_r_s, do_c_s,
            dk_accumulator, dv_accumulator,
            dbias_accumulator, dbias_b_s, dbias_r_s, dbias_c_s,
            lse, delta,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            packed_indices_q,
            s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            packed_indices_k,
            s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            packed_indices_v,
            s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            packed_indices_o,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            packed_indices_mask if has_mask else dummy_packed_indices,
            packed_indices_bias if has_bias else dummy_packed_indices,
            query_indices, query_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v, ctx.max_queries_per_key,
            q.size(0), k.size(0), v.size(0), grad_output.size(0),
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            ctx.scale,
            has_mask, has_bias,
            sparsity_block_size,
            USE_INT64=use_int64_dkdv)

        # dQ kernel
        use_int64_dq = not can_use_int32_indexing(
            q,
            k,
            v,
            grad_output,
            dq_accumulator,
            lse,
            delta,
            attention_mask if has_mask else None,
            attention_bias if has_bias else None,
            packed_indices_q,
            packed_indices_k,
            packed_indices_v,
            packed_indices_o,
            packed_indices_mask if has_mask else None,
            packed_indices_bias if has_bias else None,
            key_indices,
            key_offsets,
        )

        flash_attention_kernel_bwd_dq[(n_batches, n_seq_blocks_q)](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            grad_output, do_b_s, do_r_s, do_c_s,
            dq_accumulator,
            lse, delta,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            packed_indices_q,
            s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            packed_indices_k,
            s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            packed_indices_v,
            s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            packed_indices_o,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            packed_indices_mask if has_mask else dummy_packed_indices,
            packed_indices_bias if has_bias else dummy_packed_indices,
            key_indices, key_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v, ctx.max_keys_per_query,
            q.size(0), k.size(0), v.size(0), grad_output.size(0),
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            ctx.scale,
            has_mask, has_bias,
            sparsity_block_size,
            USE_INT64=use_int64_dq)

        dq = dq_accumulator.to(q.dtype)
        dk = dk_accumulator.to(k.dtype)
        dv = dv_accumulator.to(v.dtype)
        dbias_out = dbias_accumulator.to(attention_bias.dtype) if has_bias else None

        return (
            dq, dk, dv,
            None, dbias_out,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None,
            None, None, None, None, None,
            None, None, None,
            None, None, None,
            None)


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["o"],
)
@triton.jit
def flash_attention_kernel(q,
                           q_b, q_b_s, q_r_s, q_c_s,
                           k,
                           k_b, k_b_s, k_r_s, k_c_s,
                           v,
                           v_b, v_b_s, v_r_s, v_c_s,
                           o,
                           o_b, o_b_s, o_r_s, o_c_s,
                           attention_mask,
                           mask_b_s, mask_r_s, mask_c_s,
                           attention_bias,
                           bias_b_s, bias_r_s, bias_c_s,
                           pidx_q,
                           s_l_q_b, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
                           pidx_k,
                           s_l_k_b, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
                           pidx_v,
                           s_l_v_b, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
                           pidx_o,
                           s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                           pidx_mask, pidx_bias,
                           key_indices, key_offsets,
                           lse,
                           n_batches, n_seq_blocks_q, n_seq_blocks_k,
                           n_head_blocks_qk, n_head_blocks_v, max_keys_per_query,
                           total_mask_blocks, total_bias_blocks,
                           scale,
                           has_mask, has_bias,
                           sparsity_block_size,
                           USE_INT64: tl.constexpr,
                           TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_q_seq = tl.cast(tl.program_id(axis=1), index_dtype)

    idx_row = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    idx_col = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    # Online softmax accumulators
    m_i = tl.full([TRITON_BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([TRITON_BLOCK_SIZE], dtype=tl.float32)

    # Get attention layout cache for this (batch, q_seq) pair
    key_offset_idx = pid_bat * n_seq_blocks_q + pid_q_seq
    key_start = tl.load(key_offsets + key_offset_idx)
    key_end = tl.load(key_offsets + key_offset_idx + 1)
    n_key_blocks = key_end - key_start

    # Iterate over K sequence blocks
    for key_idx in range(max_keys_per_query):
        if key_idx < n_key_blocks:
            k_seq_block = tl.load(key_indices + key_start + key_idx)

            # Compute S = Q @ K^T by accumulating across head_dim blocks
            buf_s = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)

            for h in range(n_head_blocks_qk):
                packed_idx_q_idx = (pid_bat * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)
                packed_idx_q_msk = ((packed_idx_q_idx >= 0) &
                                     (packed_idx_q_idx < tl.cast(s_l_q_b, index_dtype) * s_l_q_b_s))
                packed_idx_q = tl.cast(tl.load(pidx_q + packed_idx_q_idx, mask=packed_idx_q_msk, other=-1), index_dtype)

                packed_idx_k_idx = (pid_bat * s_l_k_b_s + tl.cast(k_seq_block, index_dtype) * s_l_k_r_s + h * s_l_k_c_s)
                packed_idx_k_msk = ((packed_idx_k_idx >= 0) &
                                     (packed_idx_k_idx < tl.cast(s_l_k_b, index_dtype) * s_l_k_b_s))
                packed_idx_k = tl.cast(tl.load(pidx_k + packed_idx_k_idx, mask=packed_idx_k_msk, other=-1), index_dtype)

                if packed_idx_q >= 0 and packed_idx_k >= 0:
                    blk_q_idx = ((tl.cast(packed_idx_q, index_dtype) * q_b_s) +
                                 (idx_row[:, None] * q_r_s) +
                                 (idx_col[None, :] * q_c_s))
                    blk_q_msk = ((blk_q_idx >= 0) &
                                 (blk_q_idx < tl.cast(q_b, index_dtype) * q_b_s))
                    blk_q = tl.load(q + blk_q_idx, mask=blk_q_msk, other=0)

                    blk_k_idx = ((tl.cast(packed_idx_k, index_dtype) * k_b_s) +
                                 (idx_row[:, None] * k_r_s) +
                                 (idx_col[None, :] * k_c_s))
                    blk_k_msk = ((blk_k_idx >= 0) &
                                 (blk_k_idx < tl.cast(k_b, index_dtype) * k_b_s))
                    blk_k = tl.load(k + blk_k_idx, mask=blk_k_msk, other=0)

                    buf_s += tl.dot(blk_q, tl.trans(blk_k))

            # Scale scores
            qk_scale = scale * 1.4426950408889634
            buf_s = buf_s * qk_scale

            # Apply bias if present
            if has_bias:
                packed_idx_bias_idx = (pid_bat * n_seq_blocks_q * n_seq_blocks_k +
                                        pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))
                packed_idx_bias = tl.cast(tl.load(pidx_bias + packed_idx_bias_idx), index_dtype)
                if packed_idx_bias >= 0:
                    blk_bias_idx = ((tl.cast(packed_idx_bias, index_dtype) * bias_b_s) +
                                    (idx_row[:, None] * bias_r_s) +
                                    (idx_col[None, :] * bias_c_s))
                    blk_bias = tl.load(attention_bias + blk_bias_idx)
                    buf_s = buf_s + blk_bias * 1.4426950408889634

            # Apply the mask after the bias so that masked positions remain
            # ignored even when their bias is non-finite.
            if has_mask:
                packed_idx_mask_idx = (pid_bat * n_seq_blocks_q * n_seq_blocks_k +
                                        pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))
                packed_idx_mask = tl.cast(tl.load(pidx_mask + packed_idx_mask_idx), index_dtype)
                if packed_idx_mask >= 0:
                    blk_mask_idx = ((tl.cast(packed_idx_mask, index_dtype) * mask_b_s) +
                                    (idx_row[:, None] * mask_r_s) +
                                    (idx_col[None, :] * mask_c_s))
                    blk_mask = tl.load(attention_mask + blk_mask_idx)
                    buf_s = tl.where(blk_mask != 0, float("-inf") * 1.4426950408889634, buf_s)

            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(buf_s, axis=1))
            both_neg_inf = (m_i == float("-inf")) & (m_ij == float("-inf"))
            alpha = tl.where(both_neg_inf, 1.0, tl.math.exp2(m_i - m_ij))
            p_raw = tl.math.exp2(buf_s - m_ij[:, None])
            p = tl.where((buf_s == float("-inf")) & (m_ij[:, None] == float("-inf")), 0.0, p_raw)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            # For each V head block, update the output accumulator
            for h in range(n_head_blocks_v):
                packed_idx_o_idx = (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)
                packed_idx_o_msk = ((packed_idx_o_idx >= 0) &
                                     (packed_idx_o_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
                packed_idx_o = tl.cast(tl.load(pidx_o + packed_idx_o_idx, mask=packed_idx_o_msk, other=-1), index_dtype)

                packed_idx_v_idx = (pid_bat * s_l_v_b_s + tl.cast(k_seq_block, index_dtype) * s_l_v_r_s + h * s_l_v_c_s)
                packed_idx_v_msk = ((packed_idx_v_idx >= 0) &
                                     (packed_idx_v_idx < tl.cast(s_l_v_b, index_dtype) * s_l_v_b_s))
                packed_idx_v = tl.cast(tl.load(pidx_v + packed_idx_v_idx, mask=packed_idx_v_msk, other=-1), index_dtype)

                if packed_idx_o >= 0:
                    blk_o_idx = ((tl.cast(packed_idx_o, index_dtype) * o_b_s) +
                                 (idx_row[:, None] * o_r_s) +
                                 (idx_col[None, :] * o_c_s))
                    blk_o_msk = ((blk_o_idx >= 0) &
                                 (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
                    blk_o = tl.cast(tl.load(o + blk_o_idx, mask=blk_o_msk, other=0), tl.float32)
                    blk_o = blk_o * alpha[:, None]

                    if packed_idx_v >= 0:
                        blk_v_idx = ((tl.cast(packed_idx_v, index_dtype) * v_b_s) +
                                     (idx_row[:, None] * v_r_s) +
                                     (idx_col[None, :] * v_c_s))
                        blk_v_msk = ((blk_v_idx >= 0) &
                                     (blk_v_idx < tl.cast(v_b, index_dtype) * v_b_s))
                        blk_v = tl.load(v + blk_v_idx, mask=blk_v_msk, other=0)
                        blk_o = blk_o + tl.cast(tl.dot(tl.cast(p, blk_v.dtype), blk_v), tl.float32)

                    tl.store(o + blk_o_idx, tl.cast(blk_o, o.dtype.element_ty), mask=blk_o_msk)

            m_i = m_ij

    # Final normalisation
    has_attention = l_i != 0
    l_safe = tl.where(has_attention, l_i, 1.0)

    for h in range(n_head_blocks_v):
        packed_idx_o_idx = (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)
        packed_idx_o_msk = ((packed_idx_o_idx >= 0) &
                             (packed_idx_o_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
        packed_idx_o = tl.cast(tl.load(pidx_o + packed_idx_o_idx, mask=packed_idx_o_msk, other=-1), index_dtype)
        if packed_idx_o >= 0:
            blk_o_idx = ((tl.cast(packed_idx_o, index_dtype) * o_b_s) +
                         (idx_row[:, None] * o_r_s) +
                         (idx_col[None, :] * o_c_s))
            blk_o_msk = ((blk_o_idx >= 0) &
                         (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
            blk_o = tl.cast(tl.load(o + blk_o_idx, mask=blk_o_msk, other=0), tl.float32)
            blk_o = blk_o / l_safe[:, None]
            blk_o = tl.where(has_attention[:, None], blk_o, 0.0)
            tl.store(o + blk_o_idx, tl.cast(blk_o, o.dtype.element_ty), mask=blk_o_msk)

    # Store LSE
    lse_val = tl.where(has_attention, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(
        lse + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + idx_row,
        lse_val,
        mask=idx_row < TRITON_BLOCK_SIZE)


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["delta"],
)
@triton.jit
def flash_attention_kernel_bwd_preprocess(o, do, delta,
                                          pidx_o,
                                          total_o_blocks,
                                          o_b_s, o_r_s, o_c_s,
                                          s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                          n_batches, n_seq_blocks_q, n_head_blocks,
                                          sparsity_block_size,
                                          USE_INT64: tl.constexpr,
                                          TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_q_seq = tl.cast(tl.program_id(axis=1), index_dtype)

    idx_row = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    idx_col = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    delta_acc = tl.zeros([TRITON_BLOCK_SIZE], dtype=tl.float32)

    for h in range(n_head_blocks):
        packed_idx_o_idx = (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)
        packed_idx_o = tl.cast(tl.load(pidx_o + packed_idx_o_idx), index_dtype)
        if packed_idx_o >= 0:
            blk_idx = ((tl.cast(packed_idx_o, index_dtype) * o_b_s) +
                       (idx_row[:, None] * o_r_s) +
                       (idx_col[None, :] * o_c_s))
            blk_msk = ((blk_idx >= 0) & (blk_idx < tl.cast(total_o_blocks, index_dtype) * o_b_s))
            blk_o = tl.cast(tl.load(o + blk_idx, mask=blk_msk, other=0), tl.float32)
            blk_do = tl.cast(tl.load(do + blk_idx, mask=blk_msk, other=0), tl.float32)
            delta_acc += tl.sum(blk_o * blk_do, axis=1)

    tl.store(
        delta + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + idx_row,
        delta_acc)


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["dk", "dv", "dbias"],
)
@triton.jit
def flash_attention_kernel_bwd_dkdv(q, q_b_s, q_r_s, q_c_s,
                                    k, k_b_s, k_r_s, k_c_s,
                                    v, v_b_s, v_r_s, v_c_s,
                                    do, do_b_s, do_r_s, do_c_s,
                                    dk, dv,
                                    dbias, dbias_b_s, dbias_r_s, dbias_c_s,
                                    lse, delta,
                                    attention_mask, mask_b_s, mask_r_s, mask_c_s,
                                    attention_bias, bias_b_s, bias_r_s, bias_c_s,
                                    pidx_q,
                                    s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
                                    pidx_k,
                                    s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
                                    pidx_v,
                                    s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
                                    pidx_o,
                                    s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                    pidx_mask, pidx_bias,
                                    query_indices, query_offsets,
                                    n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                    n_head_blocks_qk, n_head_blocks_v, max_queries_per_key,
                                    total_q_blocks, total_k_blocks,
                                    total_v_blocks, total_o_blocks,
                                    total_mask_blocks, total_bias_blocks,
                                    scale,
                                    has_mask, has_bias,
                                    sparsity_block_size,
                                    USE_INT64: tl.constexpr,
                                    TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_k_seq = tl.cast(tl.program_id(axis=1), index_dtype)

    idx_row = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    idx_col = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    qk_scale = scale * 1.4426950408889634

    # Get query adjacency: which Q blocks attend to this K block
    query_offset_idx = pid_bat * n_seq_blocks_k + pid_k_seq
    query_start = tl.load(query_offsets + query_offset_idx)
    query_end = tl.load(query_offsets + query_offset_idx + 1)
    n_q_blocks = query_end - query_start

    for q_idx in range(max_queries_per_key):
        if q_idx < n_q_blocks:
            q_seq_block = tl.load(query_indices + query_start + q_idx)

            # Recompute S = Q @ K^T across head blocks
            buf_s = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_qk):
                packed_idx_q = tl.cast(tl.load(pidx_q + (pid_bat * s_l_q_b_s + tl.cast(q_seq_block, index_dtype) * s_l_q_r_s + h * s_l_q_c_s)), index_dtype)
                packed_idx_k = tl.cast(tl.load(pidx_k + (pid_bat * s_l_k_b_s + pid_k_seq * s_l_k_r_s + h * s_l_k_c_s)), index_dtype)
                if packed_idx_q >= 0 and packed_idx_k >= 0:
                    blk_q_idx = ((tl.cast(packed_idx_q, index_dtype) * q_b_s) +
                                 (idx_row[:, None] * q_r_s) +
                                 (idx_col[None, :] * q_c_s))
                    blk_q = tl.load(q + blk_q_idx, mask=((blk_q_idx >= 0) & (blk_q_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)), other=0)
                    blk_k_idx = ((tl.cast(packed_idx_k, index_dtype) * k_b_s) +
                                 (idx_row[:, None] * k_r_s) +
                                 (idx_col[None, :] * k_c_s))
                    blk_k = tl.load(k + blk_k_idx, mask=((blk_k_idx >= 0) & (blk_k_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)), other=0)
                    buf_s += tl.dot(blk_q, tl.trans(blk_k))

            buf_s = buf_s * qk_scale

            # Apply bias
            if has_bias:
                packed_idx_bias = tl.cast(tl.load(pidx_bias + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + tl.cast(q_seq_block, index_dtype) * n_seq_blocks_k + pid_k_seq)), index_dtype)
                if packed_idx_bias >= 0:
                    blk_bias_idx = ((tl.cast(packed_idx_bias, index_dtype) * bias_b_s) +
                                    (idx_row[:, None] * bias_r_s) +
                                    (idx_col[None, :] * bias_c_s))
                    blk_bias = tl.load(attention_bias + blk_bias_idx)
                    buf_s = buf_s + blk_bias * 1.4426950408889634

            # Apply the mask last so its ignored-position contract takes
            # precedence over any bias value.
            if has_mask:
                packed_idx_mask = tl.cast(tl.load(pidx_mask + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + tl.cast(q_seq_block, index_dtype) * n_seq_blocks_k + pid_k_seq)), index_dtype)
                if packed_idx_mask >= 0:
                    blk_mask_idx = ((tl.cast(packed_idx_mask, index_dtype) * mask_b_s) +
                                    (idx_row[:, None] * mask_r_s) +
                                    (idx_col[None, :] * mask_c_s))
                    blk_mask = tl.load(attention_mask + blk_mask_idx)
                    buf_s = tl.where(blk_mask != 0, float("-inf") * 1.4426950408889634, buf_s)

            # Recompute P from S and saved LSE
            m = tl.load(
                lse + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + tl.cast(q_seq_block, index_dtype) * TRITON_BLOCK_SIZE + idx_row)
            delta_row = tl.load(
                delta + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + tl.cast(q_seq_block, index_dtype) * TRITON_BLOCK_SIZE + idx_row)

            valid_lse = m != float("-inf")
            safe_m = tl.where(valid_lse, m, 0.0)
            p = tl.math.exp2(buf_s - safe_m[:, None])
            p = tl.where(valid_lse[:, None], p, 0.0)

            # Compute dp = sum_h dO_h @ V_h^T
            dp = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_v):
                packed_idx_o_h = tl.cast(tl.load(pidx_o + (pid_bat * s_l_o_b_s + tl.cast(q_seq_block, index_dtype) * s_l_o_r_s + h * s_l_o_c_s)), index_dtype)
                packed_idx_v_h = tl.cast(tl.load(pidx_v + (pid_bat * s_l_v_b_s + pid_k_seq * s_l_v_r_s + h * s_l_v_c_s)), index_dtype)
                if packed_idx_o_h >= 0 and packed_idx_v_h >= 0:
                    blk_do_idx = ((tl.cast(packed_idx_o_h, index_dtype) * do_b_s) +
                                  (idx_row[:, None] * do_r_s) +
                                  (idx_col[None, :] * do_c_s))
                    blk_do = tl.load(do + blk_do_idx, mask=((blk_do_idx >= 0) & (blk_do_idx < tl.cast(total_o_blocks, index_dtype) * do_b_s)), other=0)
                    blk_v_idx = ((tl.cast(packed_idx_v_h, index_dtype) * v_b_s) +
                                 (idx_row[:, None] * v_r_s) +
                                 (idx_col[None, :] * v_c_s))
                    blk_v = tl.load(v + blk_v_idx, mask=((blk_v_idx >= 0) & (blk_v_idx < tl.cast(total_v_blocks, index_dtype) * v_b_s)), other=0)
                    dp += tl.cast(tl.dot(blk_do, tl.trans(blk_v)), tl.float32)

            # ds = P * (dp - delta_row)
            ds = p * (dp - delta_row[:, None])

            # Accumulate dK across Q/K head blocks
            for h in range(n_head_blocks_qk):
                packed_idx_q_h = tl.cast(tl.load(pidx_q + (pid_bat * s_l_q_b_s + tl.cast(q_seq_block, index_dtype) * s_l_q_r_s + h * s_l_q_c_s)), index_dtype)
                packed_idx_k_h = tl.cast(tl.load(pidx_k + (pid_bat * s_l_k_b_s + pid_k_seq * s_l_k_r_s + h * s_l_k_c_s)), index_dtype)

                if packed_idx_q_h >= 0 and packed_idx_k_h >= 0:
                    blk_q_idx = ((tl.cast(packed_idx_q_h, index_dtype) * q_b_s) +
                                 (idx_row[:, None] * q_r_s) +
                                 (idx_col[None, :] * q_c_s))
                    blk_q = tl.load(q + blk_q_idx, mask=((blk_q_idx >= 0) & (blk_q_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)), other=0)
                    blk_dk_idx = ((tl.cast(packed_idx_k_h, index_dtype) * k_b_s) +
                                  (idx_row[:, None] * k_r_s) +
                                  (idx_col[None, :] * k_c_s))
                    blk_dk = tl.cast(tl.load(dk + blk_dk_idx, mask=((blk_dk_idx >= 0) & (blk_dk_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)), other=0), tl.float32)
                    blk_dk += tl.cast(tl.dot(tl.trans(tl.cast(ds, blk_q.dtype)), blk_q), tl.float32) * scale
                    tl.store(dk + blk_dk_idx, tl.cast(blk_dk, dk.dtype.element_ty))

            # Accumulate dV across V/output head blocks
            for h in range(n_head_blocks_v):
                packed_idx_o_h = tl.cast(tl.load(pidx_o + (pid_bat * s_l_o_b_s + tl.cast(q_seq_block, index_dtype) * s_l_o_r_s + h * s_l_o_c_s)), index_dtype)
                packed_idx_v_h = tl.cast(tl.load(pidx_v + (pid_bat * s_l_v_b_s + pid_k_seq * s_l_v_r_s + h * s_l_v_c_s)), index_dtype)
                if packed_idx_o_h >= 0 and packed_idx_v_h >= 0:
                    blk_do_idx = ((tl.cast(packed_idx_o_h, index_dtype) * do_b_s) +
                                  (idx_row[:, None] * do_r_s) +
                                  (idx_col[None, :] * do_c_s))
                    blk_do = tl.load(do + blk_do_idx, mask=((blk_do_idx >= 0) & (blk_do_idx < tl.cast(total_o_blocks, index_dtype) * do_b_s)), other=0)
                    blk_dv_idx = ((tl.cast(packed_idx_v_h, index_dtype) * v_b_s) +
                                  (idx_row[:, None] * v_r_s) +
                                  (idx_col[None, :] * v_c_s))
                    blk_dv = tl.cast(tl.load(dv + blk_dv_idx, mask=((blk_dv_idx >= 0) & (blk_dv_idx < tl.cast(total_v_blocks, index_dtype) * v_b_s)), other=0), tl.float32)
                    blk_dv += tl.cast(tl.dot(tl.trans(tl.cast(p, blk_do.dtype)), blk_do), tl.float32)
                    tl.store(dv + blk_dv_idx, tl.cast(blk_dv, dv.dtype.element_ty))

            # dBias
            if has_bias:
                packed_idx_bias = tl.cast(tl.load(pidx_bias + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + tl.cast(q_seq_block, index_dtype) * n_seq_blocks_k + pid_k_seq)), index_dtype)
                if packed_idx_bias >= 0:
                    blk_dbias_idx = ((tl.cast(packed_idx_bias, index_dtype) * dbias_b_s) +
                                     (idx_row[:, None] * dbias_r_s) +
                                     (idx_col[None, :] * dbias_c_s))
                    blk_dbias = tl.cast(tl.load(dbias + blk_dbias_idx, mask=((blk_dbias_idx >= 0) & (blk_dbias_idx < tl.cast(total_bias_blocks, index_dtype) * dbias_b_s)), other=0), tl.float32)
                    blk_dbias += ds
                    tl.store(dbias + blk_dbias_idx, tl.cast(blk_dbias, dbias.dtype.element_ty))


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["dq"],
)
@triton.jit
def flash_attention_kernel_bwd_dq(q, q_b_s, q_r_s, q_c_s,
                                  k, k_b_s, k_r_s, k_c_s,
                                  v, v_b_s, v_r_s, v_c_s,
                                  do, do_b_s, do_r_s, do_c_s,
                                  dq,
                                  lse, delta,
                                  attention_mask, mask_b_s, mask_r_s, mask_c_s,
                                  attention_bias, bias_b_s, bias_r_s, bias_c_s,
                                  pidx_q,
                                  s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
                                  pidx_k,
                                  s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
                                  pidx_v,
                                  s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
                                  pidx_o,
                                  s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                  pidx_mask, pidx_bias,
                                  key_indices, key_offsets,
                                  n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                  n_head_blocks_qk, n_head_blocks_v, max_keys_per_query,
                                  total_q_blocks, total_k_blocks,
                                  total_v_blocks, total_o_blocks,
                                  total_mask_blocks, total_bias_blocks,
                                  scale,
                                  has_mask, has_bias,
                                  sparsity_block_size,
                                  USE_INT64: tl.constexpr,
                                  TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_q_seq = tl.cast(tl.program_id(axis=1), index_dtype)

    idx_row = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    idx_col = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    qk_scale = scale * 1.4426950408889634

    m = tl.load(lse + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + idx_row)
    delta_row = tl.load(delta + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + idx_row)

    key_offset_idx = pid_bat * n_seq_blocks_q + pid_q_seq
    key_start = tl.load(key_offsets + key_offset_idx)
    key_end = tl.load(key_offsets + key_offset_idx + 1)
    n_key_blocks = key_end - key_start

    for key_idx in range(max_keys_per_query):
        if key_idx < n_key_blocks:
            k_seq_block = tl.load(key_indices + key_start + key_idx)

            # Recompute S = Q @ K^T across head blocks
            buf_s = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_qk):
                packed_idx_q = tl.cast(tl.load(pidx_q + (pid_bat * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)), index_dtype)
                packed_idx_k = tl.cast(tl.load(pidx_k + (pid_bat * s_l_k_b_s + tl.cast(k_seq_block, index_dtype) * s_l_k_r_s + h * s_l_k_c_s)), index_dtype)
                if packed_idx_q >= 0 and packed_idx_k >= 0:
                    blk_q_idx = ((tl.cast(packed_idx_q, index_dtype) * q_b_s) +
                                 (idx_row[:, None] * q_r_s) +
                                 (idx_col[None, :] * q_c_s))
                    blk_q = tl.load(q + blk_q_idx, mask=((blk_q_idx >= 0) & (blk_q_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)), other=0)
                    blk_k_idx = ((tl.cast(packed_idx_k, index_dtype) * k_b_s) +
                                 (idx_row[:, None] * k_r_s) +
                                 (idx_col[None, :] * k_c_s))
                    blk_k = tl.load(k + blk_k_idx, mask=((blk_k_idx >= 0) & (blk_k_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)), other=0)
                    buf_s += tl.dot(blk_q, tl.trans(blk_k))

            buf_s = buf_s * qk_scale

            if has_bias:
                packed_idx_bias = tl.cast(tl.load(pidx_bias + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))), index_dtype)
                if packed_idx_bias >= 0:
                    blk_bias_idx = ((tl.cast(packed_idx_bias, index_dtype) * bias_b_s) +
                                    (idx_row[:, None] * bias_r_s) +
                                    (idx_col[None, :] * bias_c_s))
                    blk_bias = tl.load(attention_bias + blk_bias_idx)
                    buf_s = buf_s + blk_bias * 1.4426950408889634

            if has_mask:
                packed_idx_mask = tl.cast(tl.load(pidx_mask + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))), index_dtype)
                if packed_idx_mask >= 0:
                    blk_mask_idx = ((tl.cast(packed_idx_mask, index_dtype) * mask_b_s) +
                                    (idx_row[:, None] * mask_r_s) +
                                    (idx_col[None, :] * mask_c_s))
                    blk_mask = tl.load(attention_mask + blk_mask_idx)
                    buf_s = tl.where(blk_mask != 0, float("-inf") * 1.4426950408889634, buf_s)

            valid_lse = m != float("-inf")
            safe_m = tl.where(valid_lse, m, 0.0)
            p = tl.math.exp2(buf_s - safe_m[:, None])
            p = tl.where(valid_lse[:, None], p, 0.0)

            # dp = sum_h dO_h @ V_h^T
            dp = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_v):
                packed_idx_o_h = tl.cast(tl.load(pidx_o + (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)), index_dtype)
                packed_idx_v_h = tl.cast(tl.load(pidx_v + (pid_bat * s_l_v_b_s + tl.cast(k_seq_block, index_dtype) * s_l_v_r_s + h * s_l_v_c_s)), index_dtype)
                if packed_idx_o_h >= 0 and packed_idx_v_h >= 0:
                    blk_do_idx = ((tl.cast(packed_idx_o_h, index_dtype) * do_b_s) +
                                  (idx_row[:, None] * do_r_s) +
                                  (idx_col[None, :] * do_c_s))
                    blk_do = tl.load(do + blk_do_idx, mask=((blk_do_idx >= 0) & (blk_do_idx < tl.cast(total_o_blocks, index_dtype) * do_b_s)), other=0)
                    blk_v_idx = ((tl.cast(packed_idx_v_h, index_dtype) * v_b_s) +
                                 (idx_row[:, None] * v_r_s) +
                                 (idx_col[None, :] * v_c_s))
                    blk_v = tl.load(v + blk_v_idx, mask=((blk_v_idx >= 0) & (blk_v_idx < tl.cast(total_v_blocks, index_dtype) * v_b_s)), other=0)
                    dp += tl.cast(tl.dot(blk_do, tl.trans(blk_v)), tl.float32)

            # ds = P * (dp - delta_row)
            ds = p * (dp - delta_row[:, None])

            # dQ += ds @ K * scale
            for h in range(n_head_blocks_qk):
                packed_idx_q_h = tl.cast(tl.load(pidx_q + (pid_bat * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)), index_dtype)
                packed_idx_k_h = tl.cast(tl.load(pidx_k + (pid_bat * s_l_k_b_s + tl.cast(k_seq_block, index_dtype) * s_l_k_r_s + h * s_l_k_c_s)), index_dtype)
                if packed_idx_q_h >= 0 and packed_idx_k_h >= 0:
                    blk_k_idx = ((tl.cast(packed_idx_k_h, index_dtype) * k_b_s) +
                                 (idx_row[:, None] * k_r_s) +
                                 (idx_col[None, :] * k_c_s))
                    blk_k = tl.load(k + blk_k_idx, mask=((blk_k_idx >= 0) & (blk_k_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)), other=0)
                    blk_dq_idx = ((tl.cast(packed_idx_q_h, index_dtype) * q_b_s) +
                                  (idx_row[:, None] * q_r_s) +
                                  (idx_col[None, :] * q_c_s))
                    blk_dq = tl.cast(tl.load(dq + blk_dq_idx, mask=((blk_dq_idx >= 0) & (blk_dq_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)), other=0), tl.float32)
                    blk_dq += tl.cast(tl.dot(tl.cast(ds, blk_k.dtype), blk_k), tl.float32) * scale
                    tl.store(dq + blk_dq_idx, tl.cast(blk_dq, dq.dtype.element_ty))


def flash_attention_build_layout_cache(attention_layout: Tensor,
                                       sparsity_layout_q: Tensor | None = None,
                                       sparsity_layout_k: Tensor | None = None,
                                       sparsity_layout_v: Tensor | None = None,
                                       n_seq_blocks_q: int | None = None,
                                       n_seq_blocks_k: int | None = None,
                                       n_head_blocks: int | None = None,
                                       sparsity_layout_o: Tensor | None = None,
                                       n_head_blocks_v: int | None = None,
                                       layout_cache: dict | None = None,
                                       sparsity_layout_mask: Tensor | None = None,
                                       sparsity_layout_bias: Tensor | None = None) -> dict:
    """Builds reusable layout metadata for block-sparse Flash Attention.

    Args:
        attention_layout (Tensor): The block attention pattern with shape
            ``(n_batches, n_seq_blocks_q, n_seq_blocks_k)``.
        sparsity_layout_q (Tensor, optional): The sparsity layout of Q (default ``None``).
        sparsity_layout_k (Tensor, optional): The sparsity layout of K (default ``None``).
        sparsity_layout_v (Tensor, optional): The sparsity layout of V (default ``None``).
        n_seq_blocks_q (int, optional): The number of Q sequence blocks (default ``None``).
        n_seq_blocks_k (int, optional): The number of K sequence blocks (default ``None``).
        n_head_blocks (int, optional): The number of Q/K head-dimension blocks (default ``None``).
        sparsity_layout_o (Tensor, optional): The output sparsity layout (default ``None``).
        n_head_blocks_v (int, optional): The number of V/output head-dimension blocks (default ``None``).
        layout_cache (dict, optional): An existing layout metadata cache to extend (default ``None``).
        sparsity_layout_mask (Tensor, optional): The attention-mask sparsity layout (default ``None``).
        sparsity_layout_bias (Tensor, optional): The attention-bias sparsity layout (default ``None``).

    Returns:
        dict: The populated layout metadata cache.

    """
    validate_dimensions(attention_layout)
    validate_contiguous(attention_layout)
    validate_sparsity_layout(attention_layout)

    optional_layouts = tuple(layout for layout in (
        sparsity_layout_q,
        sparsity_layout_k,
        sparsity_layout_v,
        sparsity_layout_o,
        sparsity_layout_mask,
        sparsity_layout_bias,
    ) if layout is not None)
    if optional_layouts:
        validate_dimensions(*optional_layouts)
        validate_contiguous(*optional_layouts)
        validate_device(attention_layout, *optional_layouts)
        validate_sparsity_layout(*optional_layouts)
    else:
        validate_device(attention_layout)

    n_batches, n_seq_blocks_q_layout, n_seq_blocks_k_layout = attention_layout.shape
    if n_seq_blocks_q is None:
        n_seq_blocks_q = n_seq_blocks_q_layout
    elif n_seq_blocks_q != n_seq_blocks_q_layout:
        raise ValueError("n_seq_blocks_q does not match attention_layout")
    if n_seq_blocks_k is None:
        n_seq_blocks_k = n_seq_blocks_k_layout
    elif n_seq_blocks_k != n_seq_blocks_k_layout:
        raise ValueError("n_seq_blocks_k does not match attention_layout")

    if n_head_blocks is None:
        if sparsity_layout_q is not None:
            n_head_blocks = sparsity_layout_q.size(2)
        elif sparsity_layout_k is not None:
            n_head_blocks = sparsity_layout_k.size(2)

    if n_head_blocks_v is None:
        if sparsity_layout_v is not None:
            n_head_blocks_v = sparsity_layout_v.size(2)
        elif sparsity_layout_o is not None:
            n_head_blocks_v = sparsity_layout_o.size(2)
        elif sparsity_layout_q is not None:
            n_head_blocks_v = sparsity_layout_q.size(2)

    if sparsity_layout_q is not None:
        validate_shape(
            sparsity_layout_q,
            (n_batches, n_seq_blocks_q, n_head_blocks),
            "sparsity_layout_q",
        )
    if sparsity_layout_k is not None:
        validate_shape(
            sparsity_layout_k,
            (n_batches, n_seq_blocks_k, n_head_blocks),
            "sparsity_layout_k",
        )
    if sparsity_layout_v is not None:
        validate_shape(
            sparsity_layout_v,
            (n_batches, n_seq_blocks_k, n_head_blocks_v),
            "sparsity_layout_v",
        )
    if sparsity_layout_o is not None:
        validate_shape(
            sparsity_layout_o,
            (n_batches, n_seq_blocks_q, n_head_blocks_v),
            "sparsity_layout_o",
        )
    if sparsity_layout_mask is not None:
        validate_shape(
            sparsity_layout_mask,
            (n_batches, n_seq_blocks_q, n_seq_blocks_k),
            "sparsity_layout_mask",
        )
    if sparsity_layout_bias is not None:
        validate_shape(
            sparsity_layout_bias,
            (n_batches, n_seq_blocks_q, n_seq_blocks_k),
            "sparsity_layout_bias",
        )

    requested_sparsity_layout_o = sparsity_layout_o
    layout_cache = prepare_layout_cache(
        layout_cache,
        "flash_attention",
        attention_layout,
        sparsity_layout_q,
        sparsity_layout_k,
        sparsity_layout_v,
        n_seq_blocks_q,
        n_seq_blocks_k,
        n_head_blocks,
        requested_sparsity_layout_o,
        n_head_blocks_v,
        sparsity_layout_mask,
        sparsity_layout_bias,
    )

    if "sparsity_layout_o" not in layout_cache:
        if requested_sparsity_layout_o is not None:
            layout_cache["sparsity_layout_o"] = as_base_tensor(
                requested_sparsity_layout_o)
        elif sparsity_layout_v is not None:
            layout_cache["sparsity_layout_o"] = build_sparsity_layout_matmul(
                attention_layout, sparsity_layout_v)

    sparsity_layout_o = layout_cache.get("sparsity_layout_o")

    if "key_indices" not in layout_cache:
        key_indices, key_offsets, max_keys_per_query = _build_segmented_indices(
            attention_layout, n_batches, n_seq_blocks_q, n_seq_blocks_k)
        layout_cache["key_indices"] = key_indices
        layout_cache["key_offsets"] = key_offsets
        layout_cache["max_keys_per_query"] = max_keys_per_query

    if "query_indices" not in layout_cache:
        attention_layout_t = as_base_tensor(
            attention_layout.transpose(1, 2).contiguous())
        query_indices, query_offsets, max_queries_per_key = _build_segmented_indices(
            attention_layout_t, n_batches, n_seq_blocks_k, n_seq_blocks_q)
        layout_cache["query_indices"] = query_indices
        layout_cache["query_offsets"] = query_offsets
        layout_cache["max_queries_per_key"] = max_queries_per_key

    if sparsity_layout_q is not None and "packed_indices_q" not in layout_cache:
        layout_cache["packed_indices_q"] = build_packed_indices(sparsity_layout_q)
        layout_cache["n_sparse_blocks_q"] = int(sparsity_layout_q.to(torch.int64).sum().item())

    if sparsity_layout_k is not None and "packed_indices_k" not in layout_cache:
        layout_cache["packed_indices_k"] = build_packed_indices(sparsity_layout_k)

    if sparsity_layout_v is not None and "packed_indices_v" not in layout_cache:
        layout_cache["packed_indices_v"] = build_packed_indices(sparsity_layout_v)

    if sparsity_layout_mask is not None and "packed_indices_mask" not in layout_cache:
        layout_cache["packed_indices_mask"] = build_packed_indices(sparsity_layout_mask)

    if sparsity_layout_bias is not None and "packed_indices_bias" not in layout_cache:
        layout_cache["packed_indices_bias"] = build_packed_indices(sparsity_layout_bias)

    if "packed_indices_o" not in layout_cache:
        if sparsity_layout_o is not None:
            layout_cache["packed_indices_o"] = build_packed_indices(sparsity_layout_o)
            layout_cache["n_sparse_blocks_o"] = int(sparsity_layout_o.to(torch.int64).sum().item())

    layout_cache_tensors = [v for v in layout_cache.values() if isinstance(v, Tensor)]
    if layout_cache_tensors:
        validate_contiguous(*layout_cache_tensors)

    return finalize_layout_cache(layout_cache)


def _build_segmented_indices(attention_layout: Tensor, n_batches: int,
                             n_blocks_row: int, n_blocks_col: int) -> tuple[Tensor, Tensor, int]:
    device = attention_layout.device

    counts = attention_layout.to(torch.int64).sum(dim=2).flatten()
    if counts.numel() == 0:
        offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=torch.int32, device=device)
        key_indices = torch.empty(0, dtype=torch.int32, device=device)
        return key_indices, offsets, 1

    max_blocks_per_row = int(counts.max().item())

    if max_blocks_per_row == 0:
        offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=torch.int32, device=device)
        key_indices = torch.empty(0, dtype=torch.int32, device=device)
        return key_indices, offsets, 1

    n_active_blocks = int(counts.sum().item())
    metadata_dtype = _select_segmented_index_dtype(n_active_blocks, n_blocks_col)

    offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=metadata_dtype, device=device)
    offsets[1:] = counts.cumsum(0).to(metadata_dtype)

    indices = attention_layout.reshape(n_batches * n_blocks_row, n_blocks_col).nonzero(as_tuple=False)
    key_indices = as_base_tensor(indices[:, 1].to(metadata_dtype))

    return key_indices, as_base_tensor(offsets), max_blocks_per_row


def _select_segmented_index_dtype(n_active_blocks: int, n_blocks_col: int) -> torch.dtype:
    """Select the smallest safe dtype for Flash Attention adjacency metadata."""
    max_column_index = n_blocks_col - 1
    if n_active_blocks > INT32_INDEX_MAX or max_column_index > INT32_INDEX_MAX:
        return torch.int64
    return torch.int32

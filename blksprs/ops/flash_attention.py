import math

import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs_exact
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride, build_reverse_lut, can_use_int32_indexing
from blksprs.utils.validation import validate_contiguous, validate_device, validate_dtype_float, \
    validate_dimensions, validate_sparsity, validate_sparsity_block_size, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def flash_attention(q: BlksprsTensor, sparsity_layout_q: Tensor,
                    k: BlksprsTensor, sparsity_layout_k: Tensor,
                    v: BlksprsTensor, sparsity_layout_v: Tensor,
                    attention_layout: Tensor,
                    sparsity_block_size: int,
                    scale: float = None,
                    attention_mask: BlksprsTensor = None, sparsity_layout_mask: Tensor = None,
                    attention_bias: BlksprsTensor = None, sparsity_layout_bias: Tensor = None,
                    lut: dict = None,
                    sparsity_layout_o: Tensor = None) -> BlksprsTensor:
    """Performs block-sparse flash attention on compressed tensors.

    All inputs use the standard blksprs compressed format: tensors have shape
    ``(n_sparse_blocks, sparsity_block_size, sparsity_block_size)`` with an
    accompanying sparsity layout.

    Note:
        The ``attention_mask`` convention used here (``True`` = **masked/ignored**)
        is the same as :meth:`torch.Tensor.masked_fill` but **opposite** to
        :func:`torch.nn.functional.scaled_dot_product_attention`, where a boolean
        ``attn_mask`` of ``True`` means the position **participates** in attention.

    Args:
        q (BlksprsTensor): Query tensor in compressed form.
            Sparsity layout shape: ``(n_batches, seq_q // bs, head_dim // bs)``.
        sparsity_layout_q (Tensor): The sparsity layout of the query tensor.
        k (BlksprsTensor): Key tensor in compressed form.
            Sparsity layout shape: ``(n_batches, seq_k // bs, head_dim // bs)``.
        sparsity_layout_k (Tensor): The sparsity layout of the key tensor.
        v (BlksprsTensor): Value tensor in compressed form.
        sparsity_layout_v (Tensor): The sparsity layout of the value tensor.
        attention_layout (Tensor): Block attention pattern
            ``(n_batches, seq_q // bs, seq_k // bs)`` indicating which Q-K block
            pairs participate in attention.
        sparsity_block_size (int): The size of the sparsity blocks.
        scale (float, optional): Attention scale (default ``1/sqrt(head_dim)``).
        attention_mask (BlksprsTensor, optional): Boolean mask in compressed form
            where ``True`` (non-zero) means *masked* (position ignored). Does not
            participate in gradient computation (default ``None``).
        sparsity_layout_mask (Tensor, optional): The sparsity layout for the mask (default ``None``).
        attention_bias (BlksprsTensor, optional): Additive bias in compressed form,
            added to attention scores before softmax. Supports gradient computation (default ``None``).
        sparsity_layout_bias (Tensor, optional): The sparsity layout for the bias (default ``None``).
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).
        sparsity_layout_o (Tensor, optional): Output sparsity layout.
            Defaults to ``sparsity_layout_q`` if ``d_v == d_att`` (default ``None``).

    Returns:
        BlksprsTensor: The output tensor in compressed form with layout ``sparsity_layout_o``.

    """
    q, k, v, attention_layout = ensure_contiguous(q, k, v, attention_layout)

    validate_dimensions(q, k, v)
    validate_contiguous(q, k, v)
    validate_dtype_float(q, k, v)
    validate_device(q, k, v)
    validate_sparsity(sparsity_block_size, (q, sparsity_layout_q), (k, sparsity_layout_k), (v, sparsity_layout_v))
    validate_sparsity_block_size(sparsity_block_size, q, k, v)

    validate_dimensions(attention_layout)
    validate_contiguous(attention_layout)
    validate_device(attention_layout, q)

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
    if attention_layout.shape != torch.Size(expected_attn_shape):
        raise ValueError(
            f"attention_layout shape {tuple(attention_layout.shape)} doesn't match "
            f"expected {expected_attn_shape}")

    # Resolve output sparsity layout
    if sparsity_layout_o is None:
        if n_head_blocks_v != n_head_blocks_qk:
            raise ValueError("sparsity_layout_o is required when V head dimension differs from Q/K")
        sparsity_layout_o = sparsity_layout_q
    else:
        sparsity_layout_o = ensure_contiguous(sparsity_layout_o)
        validate_dimensions(sparsity_layout_o)
        validate_contiguous(sparsity_layout_o)
        validate_device(sparsity_layout_o, q)

    expected_output_shape = (n_batches, n_seq_blocks_q, n_head_blocks_v)
    if sparsity_layout_o.shape != torch.Size(expected_output_shape):
        raise ValueError(
            f"sparsity_layout_o shape {tuple(sparsity_layout_o.shape)} doesn't match "
            f"expected {expected_output_shape}")

    if scale is None:
        scale = 1.0 / math.sqrt(n_head_blocks_qk * sparsity_block_size)

    lut = flash_attention_build_lut(attention_layout,
                                    sparsity_layout_q, sparsity_layout_k, sparsity_layout_v,
                                    n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk,
                                    sparsity_layout_o=sparsity_layout_o,
                                    n_head_blocks_v=n_head_blocks_v,
                                    lut=lut)

    # Resolve optional mask
    has_mask = attention_mask is not None
    if has_mask:
        if sparsity_layout_mask is None:
            raise ValueError("sparsity_layout_mask is required when attention_mask is provided")
        attention_mask = ensure_contiguous(attention_mask)
        sparsity_layout_mask = ensure_contiguous(sparsity_layout_mask)
        validate_dimensions(attention_mask, sparsity_layout_mask)
        validate_contiguous(attention_mask, sparsity_layout_mask)
        validate_device(attention_mask, sparsity_layout_mask, q)
        validate_sparsity(sparsity_block_size, (attention_mask, sparsity_layout_mask))
    else:
        attention_mask = torch.empty(0, device=q.device, dtype=q.dtype)

    # Resolve optional bias
    has_bias = attention_bias is not None
    if has_bias:
        if sparsity_layout_bias is None:
            raise ValueError("sparsity_layout_bias is required when attention_bias is provided")
        attention_bias = ensure_contiguous(attention_bias)
        sparsity_layout_bias = ensure_contiguous(sparsity_layout_bias)
        validate_dimensions(attention_bias, sparsity_layout_bias)
        validate_contiguous(attention_bias, sparsity_layout_bias)
        validate_device(attention_bias, sparsity_layout_bias, q)
        validate_sparsity(sparsity_block_size, (attention_bias, sparsity_layout_bias))
    else:
        attention_bias = torch.empty(0, device=q.device, dtype=q.dtype)

    # Build reverse LUTs for mask and bias
    if has_mask and "sparsity_reverse_lut_mask" not in lut:
        lut["sparsity_reverse_lut_mask"] = build_reverse_lut(sparsity_layout_mask)
    if has_bias and "sparsity_reverse_lut_bias" not in lut:
        lut["sparsity_reverse_lut_bias"] = build_reverse_lut(sparsity_layout_bias)

    dummy_lut = torch.empty(0, device=q.device, dtype=torch.long)
    sparsity_reverse_lut_mask = lut["sparsity_reverse_lut_mask"] if has_mask else dummy_lut
    sparsity_reverse_lut_bias = lut["sparsity_reverse_lut_bias"] if has_bias else dummy_lut

    return BlksprsTensor.wrap(
        _FlashAttentionAutograd.apply(
            q, k, v,
            attention_mask, attention_bias,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
            lut["sparsity_reverse_lut_q"], lut["sparsity_reverse_lut_k"],
            lut["sparsity_reverse_lut_v"], lut["sparsity_reverse_lut_o"],
            lut["attention_lut"], lut["attention_offsets"],
            lut["reverse_attention_lut"], lut["reverse_attention_offsets"],
            sparsity_reverse_lut_mask, sparsity_reverse_lut_bias,
            sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v,
            lut["max_kv_blocks"], lut["max_q_per_k"],
            lut["n_sparse_blocks_o"],
            scale, has_mask, has_bias,
            n_batches))


class _FlashAttentionAutograd(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, q, k, v, attention_mask, attention_bias,
        sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
        sparsity_reverse_lut_q, sparsity_reverse_lut_k, sparsity_reverse_lut_v, sparsity_reverse_lut_o,
        attention_lut, attention_offsets,
        reverse_attention_lut, reverse_attention_offsets,
        sparsity_reverse_lut_mask, sparsity_reverse_lut_bias,
        sparsity_block_size, n_seq_blocks_q, n_seq_blocks_k,
        n_head_blocks_qk, n_head_blocks_v,
        max_kv_blocks, max_q_per_k,
        n_sparse_blocks_o,
        scale, has_mask, has_bias,
        n_batches,
    ):
        output = torch.zeros(size=(n_sparse_blocks_o, sparsity_block_size, sparsity_block_size),
                             dtype=q.dtype, device=q.device)

        lse = torch.full(size=(n_batches, n_seq_blocks_q, sparsity_block_size),
                         fill_value=float("-inf"), dtype=torch.float32, device=q.device)

        q_b, q_r, q_c = q.size()
        q_b_s, q_r_s, q_c_s = stride(q)
        k_b, k_r, k_c = k.size()
        k_b_s, k_r_s, k_c_s = stride(k)
        v_b, v_r, v_c = v.size()
        v_b_s, v_r_s, v_c_s = stride(v)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
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

        dummy_lut = torch.empty(0, device=q.device, dtype=torch.long)

        triton_grid = lambda meta: [n_batches,
                                    n_seq_blocks_q]

        use_int64 = not can_use_int32_indexing(
            q,
            k,
            v,
            output,
            attention_mask if has_mask else None,
            attention_bias if has_bias else None,
            sparsity_reverse_lut_q,
            sparsity_reverse_lut_k,
            sparsity_reverse_lut_v,
            sparsity_reverse_lut_o,
            sparsity_reverse_lut_mask if has_mask else None,
            sparsity_reverse_lut_bias if has_bias else None,
            attention_lut,
            attention_offsets,
            lse,
        )

        flash_attention_kernel[triton_grid](
            q,
            q_b, q_b_s, q_r_s, q_c_s,
            k,
            k_b, k_b_s, k_r_s, k_c_s,
            v,
            v_b, v_b_s, v_r_s, v_c_s,
            output,
            o_b, o_b_s, o_r_s, o_c_s,
            attention_mask,
            mask_b_s, mask_r_s, mask_c_s,
            attention_bias,
            bias_b_s, bias_r_s, bias_c_s,
            sparsity_reverse_lut_q,
            s_l_q_b, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            sparsity_reverse_lut_k,
            s_l_k_b, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            sparsity_reverse_lut_v,
            s_l_v_b, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            sparsity_reverse_lut_o,
            s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            sparsity_reverse_lut_mask if has_mask else dummy_lut,
            sparsity_reverse_lut_bias if has_bias else dummy_lut,
            attention_lut, attention_offsets,
            lse,
            n_batches, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v, max_kv_blocks,
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            scale,
            has_mask, has_bias,
            sparsity_block_size,
            USE_INT64=use_int64)

        ctx.save_for_backward(
            q, k, v, output, lse,
            sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
            sparsity_reverse_lut_q, sparsity_reverse_lut_k, sparsity_reverse_lut_v, sparsity_reverse_lut_o,
            attention_lut, attention_offsets,
            reverse_attention_lut, reverse_attention_offsets,
            attention_mask if has_mask else torch.empty(0, device=q.device),
            attention_bias if has_bias else torch.empty(0, device=q.device),
            sparsity_reverse_lut_mask if has_mask else torch.empty(0, device=q.device, dtype=torch.long),
            sparsity_reverse_lut_bias if has_bias else torch.empty(0, device=q.device, dtype=torch.long),
        )
        ctx.sparsity_block_size = sparsity_block_size
        ctx.n_seq_blocks_q = n_seq_blocks_q
        ctx.n_seq_blocks_k = n_seq_blocks_k
        ctx.n_head_blocks_qk = n_head_blocks_qk
        ctx.n_head_blocks_v = n_head_blocks_v
        ctx.max_kv_blocks = max_kv_blocks
        ctx.max_q_per_k = max_q_per_k
        ctx.scale = scale
        ctx.has_mask = has_mask
        ctx.has_bias = has_bias
        ctx.n_batches = n_batches

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (q, k, v, o, lse,
         sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_layout_o,
         sparsity_reverse_lut_q, sparsity_reverse_lut_k, sparsity_reverse_lut_v, sparsity_reverse_lut_o,
         attention_lut, attention_offsets,
         reverse_attention_lut, reverse_attention_offsets,
         attention_mask, attention_bias,
         sparsity_reverse_lut_mask, sparsity_reverse_lut_bias,
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

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Precompute delta = rowsum(O * dO)
        delta = torch.zeros(n_batches, n_seq_blocks_q, sparsity_block_size,
                            device=q.device, dtype=torch.float32)

        use_int64_preprocess = not can_use_int32_indexing(
            o,
            grad_output,
            delta,
            sparsity_reverse_lut_o,
        )

        flash_attention_kernel_bwd_preprocess[(n_batches, n_seq_blocks_q)](
            o, grad_output, delta,
            sparsity_reverse_lut_o,
            o.size(0),
            do_b_s, do_r_s, do_c_s,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            n_batches, n_seq_blocks_q, n_head_blocks_v,
            sparsity_block_size,
            USE_INT64=use_int64_preprocess)

        # Allocate dbias if needed
        if has_bias:
            dbias = torch.zeros_like(attention_bias)
            dbias_b_s, dbias_r_s, dbias_c_s = stride(dbias)
        else:
            dbias = torch.empty(0, device=q.device, dtype=q.dtype)
            dbias_b_s = dbias_r_s = dbias_c_s = 0

        dummy_lut = torch.empty(0, device=q.device, dtype=torch.long)

        # dK, dV kernel
        use_int64_dkdv = not can_use_int32_indexing(
            q,
            k,
            v,
            grad_output,
            dk,
            dv,
            dbias if has_bias else None,
            lse,
            delta,
            attention_mask if has_mask else None,
            attention_bias if has_bias else None,
            sparsity_reverse_lut_q,
            sparsity_reverse_lut_k,
            sparsity_reverse_lut_v,
            sparsity_reverse_lut_o,
            sparsity_reverse_lut_mask if has_mask else None,
            sparsity_reverse_lut_bias if has_bias else None,
            reverse_attention_lut,
            reverse_attention_offsets,
        )

        flash_attention_kernel_bwd_dkdv[(n_batches, n_seq_blocks_k)](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            grad_output, do_b_s, do_r_s, do_c_s,
            dk, dv,
            dbias, dbias_b_s, dbias_r_s, dbias_c_s,
            lse, delta,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            sparsity_reverse_lut_q,
            s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            sparsity_reverse_lut_k,
            s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            sparsity_reverse_lut_v,
            s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            sparsity_reverse_lut_o,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            sparsity_reverse_lut_mask if has_mask else dummy_lut,
            sparsity_reverse_lut_bias if has_bias else dummy_lut,
            reverse_attention_lut, reverse_attention_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v, ctx.max_q_per_k,
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
            dq,
            lse,
            delta,
            attention_mask if has_mask else None,
            attention_bias if has_bias else None,
            sparsity_reverse_lut_q,
            sparsity_reverse_lut_k,
            sparsity_reverse_lut_v,
            sparsity_reverse_lut_o,
            sparsity_reverse_lut_mask if has_mask else None,
            sparsity_reverse_lut_bias if has_bias else None,
            attention_lut,
            attention_offsets,
        )

        flash_attention_kernel_bwd_dq[(n_batches, n_seq_blocks_q)](
            q, q_b_s, q_r_s, q_c_s,
            k, k_b_s, k_r_s, k_c_s,
            v, v_b_s, v_r_s, v_c_s,
            grad_output, do_b_s, do_r_s, do_c_s,
            dq,
            lse, delta,
            attention_mask, mask_b_s, mask_r_s, mask_c_s,
            attention_bias, bias_b_s, bias_r_s, bias_c_s,
            sparsity_reverse_lut_q,
            s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
            sparsity_reverse_lut_k,
            s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
            sparsity_reverse_lut_v,
            s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
            sparsity_reverse_lut_o,
            s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
            sparsity_reverse_lut_mask if has_mask else dummy_lut,
            sparsity_reverse_lut_bias if has_bias else dummy_lut,
            attention_lut, attention_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k,
            n_head_blocks_qk, n_head_blocks_v, ctx.max_kv_blocks,
            q.size(0), k.size(0), v.size(0), grad_output.size(0),
            attention_mask.size(0) if has_mask else 0,
            attention_bias.size(0) if has_bias else 0,
            ctx.scale,
            has_mask, has_bias,
            sparsity_block_size,
            USE_INT64=use_int64_dq)

        dbias_out = dbias if has_bias else None

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
                           r_lut_q,
                           s_l_q_b, s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
                           r_lut_k,
                           s_l_k_b, s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
                           r_lut_v,
                           s_l_v_b, s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
                           r_lut_o,
                           s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                           r_lut_mask, r_lut_bias,
                           attention_lut, attention_offsets,
                           lse,
                           n_batches, n_seq_blocks_q, n_seq_blocks_k,
                           n_head_blocks_qk, n_head_blocks_v, max_kv_blocks,
                           total_mask_blocks, total_bias_blocks,
                           scale,
                           has_mask, has_bias,
                           sparsity_block_size,
                           USE_INT64: tl.constexpr,
                           TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_q_seq = tl.cast(tl.program_id(axis=1), index_dtype)

    offs_m = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    offs_d = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    # Online softmax accumulators
    m_i = tl.full([TRITON_BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([TRITON_BLOCK_SIZE], dtype=tl.float32)

    # Get attention LUT for this (batch, q_seq) pair
    attention_offset_idx = pid_bat * n_seq_blocks_q + pid_q_seq
    attention_start = tl.load(attention_offsets + attention_offset_idx)
    attention_end = tl.load(attention_offsets + attention_offset_idx + 1)
    n_kv_blocks = attention_end - attention_start

    # Iterate over K sequence blocks
    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attention_lut + attention_start + kv_idx)

            # Compute S = Q @ K^T by accumulating across head_dim blocks
            buf_s = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)

            for h in range(n_head_blocks_qk):
                rev_idx_spa_q_idx = (pid_bat * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)
                rev_idx_spa_q_msk = ((rev_idx_spa_q_idx >= 0) &
                                     (rev_idx_spa_q_idx < tl.cast(s_l_q_b, index_dtype) * s_l_q_b_s))
                rev_idx_spa_q = tl.cast(tl.load(r_lut_q + rev_idx_spa_q_idx, mask=rev_idx_spa_q_msk), tl.int32)

                rev_idx_spa_k_idx = (pid_bat * s_l_k_b_s + tl.cast(k_seq_block, index_dtype) * s_l_k_r_s + h * s_l_k_c_s)
                rev_idx_spa_k_msk = ((rev_idx_spa_k_idx >= 0) &
                                     (rev_idx_spa_k_idx < tl.cast(s_l_k_b, index_dtype) * s_l_k_b_s))
                rev_idx_spa_k = tl.cast(tl.load(r_lut_k + rev_idx_spa_k_idx, mask=rev_idx_spa_k_msk), tl.int32)

                if rev_idx_spa_q >= 0 and rev_idx_spa_k >= 0:
                    blk_q_idx = ((tl.cast(rev_idx_spa_q, index_dtype) * q_b_s) +
                                 (offs_m[:, None] * q_r_s) +
                                 (offs_d[None, :] * q_c_s))
                    blk_q_msk = ((blk_q_idx >= 0) &
                                 (blk_q_idx < tl.cast(q_b, index_dtype) * q_b_s))
                    blk_q = tl.load(q + blk_q_idx, mask=blk_q_msk)

                    blk_k_idx = ((tl.cast(rev_idx_spa_k, index_dtype) * k_b_s) +
                                 (offs_m[:, None] * k_r_s) +
                                 (offs_d[None, :] * k_c_s))
                    blk_k_msk = ((blk_k_idx >= 0) &
                                 (blk_k_idx < tl.cast(k_b, index_dtype) * k_b_s))
                    blk_k = tl.load(k + blk_k_idx, mask=blk_k_msk)

                    buf_s += tl.dot(blk_q, tl.trans(blk_k))

            # Scale scores
            qk_scale = scale * 1.4426950408889634
            buf_s = buf_s * qk_scale

            # Apply mask if present
            if has_mask:
                rev_idx_spa_mask_idx = (pid_bat * n_seq_blocks_q * n_seq_blocks_k +
                                        pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))
                rev_idx_spa_mask = tl.cast(tl.load(r_lut_mask + rev_idx_spa_mask_idx), tl.int32)
                if rev_idx_spa_mask >= 0:
                    blk_mask_idx = ((tl.cast(rev_idx_spa_mask, index_dtype) * mask_b_s) +
                                    (offs_m[:, None] * mask_r_s) +
                                    (offs_d[None, :] * mask_c_s))
                    blk_mask = tl.load(attention_mask + blk_mask_idx)
                    buf_s = tl.where(blk_mask != 0, float("-inf") * 1.4426950408889634, buf_s)

            # Apply bias if present
            if has_bias:
                rev_idx_spa_bias_idx = (pid_bat * n_seq_blocks_q * n_seq_blocks_k +
                                        pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))
                rev_idx_spa_bias = tl.cast(tl.load(r_lut_bias + rev_idx_spa_bias_idx), tl.int32)
                if rev_idx_spa_bias >= 0:
                    blk_bias_idx = ((tl.cast(rev_idx_spa_bias, index_dtype) * bias_b_s) +
                                    (offs_m[:, None] * bias_r_s) +
                                    (offs_d[None, :] * bias_c_s))
                    blk_bias = tl.load(attention_bias + blk_bias_idx)
                    buf_s = buf_s + blk_bias * 1.4426950408889634

            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(buf_s, axis=1))
            both_neg_inf = (m_i == float("-inf")) & (m_ij == float("-inf"))
            alpha = tl.where(both_neg_inf, 1.0, tl.math.exp2(m_i - m_ij))
            p_raw = tl.math.exp2(buf_s - m_ij[:, None])
            p = tl.where((buf_s == float("-inf")) & (m_ij[:, None] == float("-inf")), 0.0, p_raw)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            # For each V head block, update the output accumulator
            for h in range(n_head_blocks_v):
                rev_idx_spa_o_idx = (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)
                rev_idx_spa_o_msk = ((rev_idx_spa_o_idx >= 0) &
                                     (rev_idx_spa_o_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
                rev_idx_spa_o = tl.cast(tl.load(r_lut_o + rev_idx_spa_o_idx, mask=rev_idx_spa_o_msk), tl.int32)

                rev_idx_spa_v_idx = (pid_bat * s_l_v_b_s + tl.cast(k_seq_block, index_dtype) * s_l_v_r_s + h * s_l_v_c_s)
                rev_idx_spa_v_msk = ((rev_idx_spa_v_idx >= 0) &
                                     (rev_idx_spa_v_idx < tl.cast(s_l_v_b, index_dtype) * s_l_v_b_s))
                rev_idx_spa_v = tl.cast(tl.load(r_lut_v + rev_idx_spa_v_idx, mask=rev_idx_spa_v_msk), tl.int32)

                if rev_idx_spa_o >= 0:
                    blk_o_idx = ((tl.cast(rev_idx_spa_o, index_dtype) * o_b_s) +
                                 (offs_m[:, None] * o_r_s) +
                                 (offs_d[None, :] * o_c_s))
                    blk_o_msk = ((blk_o_idx >= 0) &
                                 (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
                    blk_o = tl.cast(tl.load(o + blk_o_idx, mask=blk_o_msk), tl.float32)
                    blk_o = blk_o * alpha[:, None]

                    if rev_idx_spa_v >= 0:
                        blk_v_idx = ((tl.cast(rev_idx_spa_v, index_dtype) * v_b_s) +
                                     (offs_m[:, None] * v_r_s) +
                                     (offs_d[None, :] * v_c_s))
                        blk_v_msk = ((blk_v_idx >= 0) &
                                     (blk_v_idx < tl.cast(v_b, index_dtype) * v_b_s))
                        blk_v = tl.load(v + blk_v_idx, mask=blk_v_msk)
                        blk_o = blk_o + tl.cast(tl.dot(tl.cast(p, blk_v.dtype), blk_v), tl.float32)

                    tl.store(o + blk_o_idx, tl.cast(blk_o, o.dtype.element_ty), mask=blk_o_msk)

            m_i = m_ij

    # Final normalization
    has_attention = l_i > 0
    l_safe = tl.where(has_attention, l_i, 1.0)

    for h in range(n_head_blocks_v):
        rev_idx_spa_o_idx = (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)
        rev_idx_spa_o_msk = ((rev_idx_spa_o_idx >= 0) &
                             (rev_idx_spa_o_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
        rev_idx_spa_o = tl.cast(tl.load(r_lut_o + rev_idx_spa_o_idx, mask=rev_idx_spa_o_msk), tl.int32)
        if rev_idx_spa_o >= 0:
            blk_o_idx = ((tl.cast(rev_idx_spa_o, index_dtype) * o_b_s) +
                         (offs_m[:, None] * o_r_s) +
                         (offs_d[None, :] * o_c_s))
            blk_o_msk = ((blk_o_idx >= 0) &
                         (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
            blk_o = tl.cast(tl.load(o + blk_o_idx, mask=blk_o_msk), tl.float32)
            blk_o = blk_o / l_safe[:, None]
            blk_o = tl.where(has_attention[:, None], blk_o, 0.0)
            tl.store(o + blk_o_idx, tl.cast(blk_o, o.dtype.element_ty), mask=blk_o_msk)

    # Store LSE
    lse_val = tl.where(has_attention, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(
        lse + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m,
        lse_val,
        mask=offs_m < TRITON_BLOCK_SIZE)


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_exact},
    reset_to_zero=["delta"],
)
@triton.jit
def flash_attention_kernel_bwd_preprocess(o, do, delta,
                                          r_lut_o,
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

    offs_m = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    offs_d = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    delta_acc = tl.zeros([TRITON_BLOCK_SIZE], dtype=tl.float32)

    for h in range(n_head_blocks):
        rev_idx_spa_o_idx = (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)
        rev_idx_spa_o = tl.cast(tl.load(r_lut_o + rev_idx_spa_o_idx), tl.int32)
        if rev_idx_spa_o >= 0:
            blk_idx = ((tl.cast(rev_idx_spa_o, index_dtype) * o_b_s) +
                       (offs_m[:, None] * o_r_s) +
                       (offs_d[None, :] * o_c_s))
            blk_msk = ((blk_idx >= 0) & (blk_idx < tl.cast(total_o_blocks, index_dtype) * o_b_s))
            blk_o = tl.cast(tl.load(o + blk_idx, mask=blk_msk), tl.float32)
            blk_do = tl.cast(tl.load(do + blk_idx, mask=blk_msk), tl.float32)
            delta_acc += tl.sum(blk_o * blk_do, axis=1)

    tl.store(
        delta + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m,
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
                                    r_lut_q,
                                    s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
                                    r_lut_k,
                                    s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
                                    r_lut_v,
                                    s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
                                    r_lut_o,
                                    s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                    r_lut_mask, r_lut_bias,
                                    reverse_attention_lut, reverse_attention_offsets,
                                    n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                    n_head_blocks_qk, n_head_blocks_v, max_q_per_k,
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

    offs_m = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    offs_d = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    qk_scale = scale * 1.4426950408889634

    # Get reverse attention LUT: which Q blocks attend to this K block
    reverse_offset_idx = pid_bat * n_seq_blocks_k + pid_k_seq
    reverse_start = tl.load(reverse_attention_offsets + reverse_offset_idx)
    reverse_end = tl.load(reverse_attention_offsets + reverse_offset_idx + 1)
    n_q_blocks = reverse_end - reverse_start

    for q_idx in range(max_q_per_k):
        if q_idx < n_q_blocks:
            q_seq_block = tl.load(reverse_attention_lut + reverse_start + q_idx)

            # Recompute S = Q @ K^T across head blocks
            buf_s = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_qk):
                rev_idx_spa_q = tl.cast(tl.load(r_lut_q + (pid_bat * s_l_q_b_s + tl.cast(q_seq_block, index_dtype) * s_l_q_r_s + h * s_l_q_c_s)), tl.int32)
                rev_idx_spa_k = tl.cast(tl.load(r_lut_k + (pid_bat * s_l_k_b_s + pid_k_seq * s_l_k_r_s + h * s_l_k_c_s)), tl.int32)
                if rev_idx_spa_q >= 0 and rev_idx_spa_k >= 0:
                    blk_q_idx = ((tl.cast(rev_idx_spa_q, index_dtype) * q_b_s) +
                                 (offs_m[:, None] * q_r_s) +
                                 (offs_d[None, :] * q_c_s))
                    blk_q = tl.load(q + blk_q_idx, mask=((blk_q_idx >= 0) & (blk_q_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)))
                    blk_k_idx = ((tl.cast(rev_idx_spa_k, index_dtype) * k_b_s) +
                                 (offs_m[:, None] * k_r_s) +
                                 (offs_d[None, :] * k_c_s))
                    blk_k = tl.load(k + blk_k_idx, mask=((blk_k_idx >= 0) & (blk_k_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)))
                    buf_s += tl.dot(blk_q, tl.trans(blk_k))

            buf_s = buf_s * qk_scale

            # Apply mask
            if has_mask:
                rev_idx_spa_mask = tl.cast(tl.load(r_lut_mask + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + tl.cast(q_seq_block, index_dtype) * n_seq_blocks_k + pid_k_seq)), tl.int32)
                if rev_idx_spa_mask >= 0:
                    blk_mask_idx = ((tl.cast(rev_idx_spa_mask, index_dtype) * mask_b_s) +
                                    (offs_m[:, None] * mask_r_s) +
                                    (offs_d[None, :] * mask_c_s))
                    blk_mask = tl.load(attention_mask + blk_mask_idx)
                    buf_s = tl.where(blk_mask != 0, float("-inf") * 1.4426950408889634, buf_s)

            # Apply bias
            if has_bias:
                rev_idx_spa_bias = tl.cast(tl.load(r_lut_bias + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + tl.cast(q_seq_block, index_dtype) * n_seq_blocks_k + pid_k_seq)), tl.int32)
                if rev_idx_spa_bias >= 0:
                    blk_bias_idx = ((tl.cast(rev_idx_spa_bias, index_dtype) * bias_b_s) +
                                    (offs_m[:, None] * bias_r_s) +
                                    (offs_d[None, :] * bias_c_s))
                    blk_bias = tl.load(attention_bias + blk_bias_idx)
                    buf_s = buf_s + blk_bias * 1.4426950408889634

            # Recompute P from S and saved LSE
            m = tl.load(
                lse + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + tl.cast(q_seq_block, index_dtype) * TRITON_BLOCK_SIZE + offs_m)
            Di = tl.load(
                delta + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + tl.cast(q_seq_block, index_dtype) * TRITON_BLOCK_SIZE + offs_m)

            valid_lse = m > float("-inf")
            safe_m = tl.where(valid_lse, m, 0.0)
            p = tl.math.exp2(buf_s - safe_m[:, None])
            p = tl.where(valid_lse[:, None], p, 0.0)

            # Compute dp = sum_h dO_h @ V_h^T
            dp = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_v):
                rev_idx_spa_o_h = tl.cast(tl.load(r_lut_o + (pid_bat * s_l_o_b_s + tl.cast(q_seq_block, index_dtype) * s_l_o_r_s + h * s_l_o_c_s)), tl.int32)
                rev_idx_spa_v_h = tl.cast(tl.load(r_lut_v + (pid_bat * s_l_v_b_s + pid_k_seq * s_l_v_r_s + h * s_l_v_c_s)), tl.int32)
                if rev_idx_spa_o_h >= 0 and rev_idx_spa_v_h >= 0:
                    blk_do_idx = ((tl.cast(rev_idx_spa_o_h, index_dtype) * do_b_s) +
                                  (offs_m[:, None] * do_r_s) +
                                  (offs_d[None, :] * do_c_s))
                    blk_do = tl.load(do + blk_do_idx, mask=((blk_do_idx >= 0) & (blk_do_idx < tl.cast(total_o_blocks, index_dtype) * do_b_s)))
                    blk_v_idx = ((tl.cast(rev_idx_spa_v_h, index_dtype) * v_b_s) +
                                 (offs_m[:, None] * v_r_s) +
                                 (offs_d[None, :] * v_c_s))
                    blk_v = tl.load(v + blk_v_idx, mask=((blk_v_idx >= 0) & (blk_v_idx < tl.cast(total_v_blocks, index_dtype) * v_b_s)))
                    dp += tl.cast(tl.dot(blk_do, tl.trans(blk_v)), tl.float32)

            # ds = P * (dp - Di)
            ds = p * (dp - Di[:, None])

            # Accumulate dK across Q/K head blocks
            for h in range(n_head_blocks_qk):
                rev_idx_spa_q_h = tl.cast(tl.load(r_lut_q + (pid_bat * s_l_q_b_s + tl.cast(q_seq_block, index_dtype) * s_l_q_r_s + h * s_l_q_c_s)), tl.int32)
                rev_idx_spa_k_h = tl.cast(tl.load(r_lut_k + (pid_bat * s_l_k_b_s + pid_k_seq * s_l_k_r_s + h * s_l_k_c_s)), tl.int32)

                if rev_idx_spa_q_h >= 0 and rev_idx_spa_k_h >= 0:
                    blk_q_idx = ((tl.cast(rev_idx_spa_q_h, index_dtype) * q_b_s) +
                                 (offs_m[:, None] * q_r_s) +
                                 (offs_d[None, :] * q_c_s))
                    blk_q = tl.load(q + blk_q_idx, mask=((blk_q_idx >= 0) & (blk_q_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)))
                    blk_dk_idx = ((tl.cast(rev_idx_spa_k_h, index_dtype) * k_b_s) +
                                  (offs_m[:, None] * k_r_s) +
                                  (offs_d[None, :] * k_c_s))
                    blk_dk = tl.cast(tl.load(dk + blk_dk_idx, mask=((blk_dk_idx >= 0) & (blk_dk_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s))), tl.float32)
                    blk_dk += tl.cast(tl.dot(tl.trans(tl.cast(ds, blk_q.dtype)), blk_q), tl.float32) * scale
                    tl.store(dk + blk_dk_idx, tl.cast(blk_dk, dk.dtype.element_ty))

            # Accumulate dV across V/output head blocks
            for h in range(n_head_blocks_v):
                rev_idx_spa_o_h = tl.cast(tl.load(r_lut_o + (pid_bat * s_l_o_b_s + tl.cast(q_seq_block, index_dtype) * s_l_o_r_s + h * s_l_o_c_s)), tl.int32)
                rev_idx_spa_v_h = tl.cast(tl.load(r_lut_v + (pid_bat * s_l_v_b_s + pid_k_seq * s_l_v_r_s + h * s_l_v_c_s)), tl.int32)
                if rev_idx_spa_o_h >= 0 and rev_idx_spa_v_h >= 0:
                    blk_do_idx = ((tl.cast(rev_idx_spa_o_h, index_dtype) * do_b_s) +
                                  (offs_m[:, None] * do_r_s) +
                                  (offs_d[None, :] * do_c_s))
                    blk_do = tl.load(do + blk_do_idx, mask=((blk_do_idx >= 0) & (blk_do_idx < tl.cast(total_o_blocks, index_dtype) * do_b_s)))
                    blk_dv_idx = ((tl.cast(rev_idx_spa_v_h, index_dtype) * v_b_s) +
                                  (offs_m[:, None] * v_r_s) +
                                  (offs_d[None, :] * v_c_s))
                    blk_dv = tl.cast(tl.load(dv + blk_dv_idx, mask=((blk_dv_idx >= 0) & (blk_dv_idx < tl.cast(total_v_blocks, index_dtype) * v_b_s))), tl.float32)
                    blk_dv += tl.cast(tl.dot(tl.trans(tl.cast(p, blk_do.dtype)), blk_do), tl.float32)
                    tl.store(dv + blk_dv_idx, tl.cast(blk_dv, dv.dtype.element_ty))

            # dBias
            if has_bias:
                rev_idx_spa_bias = tl.cast(tl.load(r_lut_bias + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + tl.cast(q_seq_block, index_dtype) * n_seq_blocks_k + pid_k_seq)), tl.int32)
                if rev_idx_spa_bias >= 0:
                    blk_dbias_idx = ((tl.cast(rev_idx_spa_bias, index_dtype) * dbias_b_s) +
                                     (offs_m[:, None] * dbias_r_s) +
                                     (offs_d[None, :] * dbias_c_s))
                    blk_dbias = tl.cast(tl.load(dbias + blk_dbias_idx, mask=((blk_dbias_idx >= 0) & (blk_dbias_idx < tl.cast(total_bias_blocks, index_dtype) * dbias_b_s))), tl.float32)
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
                                  r_lut_q,
                                  s_l_q_b_s, s_l_q_r_s, s_l_q_c_s,
                                  r_lut_k,
                                  s_l_k_b_s, s_l_k_r_s, s_l_k_c_s,
                                  r_lut_v,
                                  s_l_v_b_s, s_l_v_r_s, s_l_v_c_s,
                                  r_lut_o,
                                  s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                  r_lut_mask, r_lut_bias,
                                  attention_lut, attention_offsets,
                                  n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                  n_head_blocks_qk, n_head_blocks_v, max_kv_blocks,
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

    offs_m = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    offs_d = tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    qk_scale = scale * 1.4426950408889634

    m = tl.load(lse + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m)
    Di = tl.load(delta + pid_bat * n_seq_blocks_q * TRITON_BLOCK_SIZE + pid_q_seq * TRITON_BLOCK_SIZE + offs_m)

    attention_offset_idx = pid_bat * n_seq_blocks_q + pid_q_seq
    attention_start = tl.load(attention_offsets + attention_offset_idx)
    attention_end = tl.load(attention_offsets + attention_offset_idx + 1)
    n_kv_blocks = attention_end - attention_start

    for kv_idx in range(max_kv_blocks):
        if kv_idx < n_kv_blocks:
            k_seq_block = tl.load(attention_lut + attention_start + kv_idx)

            # Recompute S = Q @ K^T across head blocks
            buf_s = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_qk):
                rev_idx_spa_q = tl.cast(tl.load(r_lut_q + (pid_bat * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)), tl.int32)
                rev_idx_spa_k = tl.cast(tl.load(r_lut_k + (pid_bat * s_l_k_b_s + tl.cast(k_seq_block, index_dtype) * s_l_k_r_s + h * s_l_k_c_s)), tl.int32)
                if rev_idx_spa_q >= 0 and rev_idx_spa_k >= 0:
                    blk_q_idx = ((tl.cast(rev_idx_spa_q, index_dtype) * q_b_s) +
                                 (offs_m[:, None] * q_r_s) +
                                 (offs_d[None, :] * q_c_s))
                    blk_q = tl.load(q + blk_q_idx, mask=((blk_q_idx >= 0) & (blk_q_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s)))
                    blk_k_idx = ((tl.cast(rev_idx_spa_k, index_dtype) * k_b_s) +
                                 (offs_m[:, None] * k_r_s) +
                                 (offs_d[None, :] * k_c_s))
                    blk_k = tl.load(k + blk_k_idx, mask=((blk_k_idx >= 0) & (blk_k_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)))
                    buf_s += tl.dot(blk_q, tl.trans(blk_k))

            buf_s = buf_s * qk_scale

            if has_mask:
                rev_idx_spa_mask = tl.cast(tl.load(r_lut_mask + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))), tl.int32)
                if rev_idx_spa_mask >= 0:
                    blk_mask_idx = ((tl.cast(rev_idx_spa_mask, index_dtype) * mask_b_s) +
                                    (offs_m[:, None] * mask_r_s) +
                                    (offs_d[None, :] * mask_c_s))
                    blk_mask = tl.load(attention_mask + blk_mask_idx)
                    buf_s = tl.where(blk_mask != 0, float("-inf") * 1.4426950408889634, buf_s)

            if has_bias:
                rev_idx_spa_bias = tl.cast(tl.load(r_lut_bias + (pid_bat * n_seq_blocks_q * n_seq_blocks_k + pid_q_seq * n_seq_blocks_k + tl.cast(k_seq_block, index_dtype))), tl.int32)
                if rev_idx_spa_bias >= 0:
                    blk_bias_idx = ((tl.cast(rev_idx_spa_bias, index_dtype) * bias_b_s) +
                                    (offs_m[:, None] * bias_r_s) +
                                    (offs_d[None, :] * bias_c_s))
                    blk_bias = tl.load(attention_bias + blk_bias_idx)
                    buf_s = buf_s + blk_bias * 1.4426950408889634

            valid_lse = m > float("-inf")
            safe_m = tl.where(valid_lse, m, 0.0)
            p = tl.math.exp2(buf_s - safe_m[:, None])
            p = tl.where(valid_lse[:, None], p, 0.0)

            # dp = sum_h dO_h @ V_h^T
            dp = tl.zeros([TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE], dtype=tl.float32)
            for h in range(n_head_blocks_v):
                rev_idx_spa_o_h = tl.cast(tl.load(r_lut_o + (pid_bat * s_l_o_b_s + pid_q_seq * s_l_o_r_s + h * s_l_o_c_s)), tl.int32)
                rev_idx_spa_v_h = tl.cast(tl.load(r_lut_v + (pid_bat * s_l_v_b_s + tl.cast(k_seq_block, index_dtype) * s_l_v_r_s + h * s_l_v_c_s)), tl.int32)
                if rev_idx_spa_o_h >= 0 and rev_idx_spa_v_h >= 0:
                    blk_do_idx = ((tl.cast(rev_idx_spa_o_h, index_dtype) * do_b_s) +
                                  (offs_m[:, None] * do_r_s) +
                                  (offs_d[None, :] * do_c_s))
                    blk_do = tl.load(do + blk_do_idx, mask=((blk_do_idx >= 0) & (blk_do_idx < tl.cast(total_o_blocks, index_dtype) * do_b_s)))
                    blk_v_idx = ((tl.cast(rev_idx_spa_v_h, index_dtype) * v_b_s) +
                                 (offs_m[:, None] * v_r_s) +
                                 (offs_d[None, :] * v_c_s))
                    blk_v = tl.load(v + blk_v_idx, mask=((blk_v_idx >= 0) & (blk_v_idx < tl.cast(total_v_blocks, index_dtype) * v_b_s)))
                    dp += tl.cast(tl.dot(blk_do, tl.trans(blk_v)), tl.float32)

            # ds = P * (dp - Di)
            ds = p * (dp - Di[:, None])

            # dQ += ds @ K * scale
            for h in range(n_head_blocks_qk):
                rev_idx_spa_q_h = tl.cast(tl.load(r_lut_q + (pid_bat * s_l_q_b_s + pid_q_seq * s_l_q_r_s + h * s_l_q_c_s)), tl.int32)
                rev_idx_spa_k_h = tl.cast(tl.load(r_lut_k + (pid_bat * s_l_k_b_s + tl.cast(k_seq_block, index_dtype) * s_l_k_r_s + h * s_l_k_c_s)), tl.int32)
                if rev_idx_spa_q_h >= 0 and rev_idx_spa_k_h >= 0:
                    blk_k_idx = ((tl.cast(rev_idx_spa_k_h, index_dtype) * k_b_s) +
                                 (offs_m[:, None] * k_r_s) +
                                 (offs_d[None, :] * k_c_s))
                    blk_k = tl.load(k + blk_k_idx, mask=((blk_k_idx >= 0) & (blk_k_idx < tl.cast(total_k_blocks, index_dtype) * k_b_s)))
                    blk_dq_idx = ((tl.cast(rev_idx_spa_q_h, index_dtype) * q_b_s) +
                                  (offs_m[:, None] * q_r_s) +
                                  (offs_d[None, :] * q_c_s))
                    blk_dq = tl.cast(tl.load(dq + blk_dq_idx, mask=((blk_dq_idx >= 0) & (blk_dq_idx < tl.cast(total_q_blocks, index_dtype) * q_b_s))), tl.float32)
                    blk_dq += tl.cast(tl.dot(tl.cast(ds, blk_k.dtype), blk_k), tl.float32) * scale
                    tl.store(dq + blk_dq_idx, tl.cast(blk_dq, dq.dtype.element_ty))


def flash_attention_build_lut(attention_layout: Tensor,
                              sparsity_layout_q: Tensor = None,
                              sparsity_layout_k: Tensor = None,
                              sparsity_layout_v: Tensor = None,
                              n_seq_blocks_q: int = None,
                              n_seq_blocks_k: int = None,
                              n_head_blocks: int = None,
                              sparsity_layout_o: Tensor = None,
                              n_head_blocks_v: int = None,
                              lut: dict = None) -> dict:
    """Builds the look-up tables for block-sparse flash attention.

    Args:
        attention_layout (Tensor): Block attention pattern ``(n_batches, n_seq_blocks_q, n_seq_blocks_k)``.
        sparsity_layout_q (Tensor, optional): Sparsity layout for Q (default ``None``).
        sparsity_layout_k (Tensor, optional): Sparsity layout for K (default ``None``).
        sparsity_layout_v (Tensor, optional): Sparsity layout for V (default ``None``).
        n_seq_blocks_q (int, optional): Number of Q sequence blocks (default ``None``).
        n_seq_blocks_k (int, optional): Number of K sequence blocks (default ``None``).
        n_head_blocks (int, optional): Number of Q/K head dimension blocks (default ``None``).
        sparsity_layout_o (Tensor, optional): Output sparsity layout (default ``None``).
        n_head_blocks_v (int, optional): Number of V/output head dimension blocks (default ``None``).
        lut (dict, optional): Existing look-up table dictionary to extend (default ``None``).

    Returns:
        dict: A dictionary containing the look-up tables for the operation.

    """
    if lut is None:
        lut = dict()

    n_batches = attention_layout.shape[0]
    if n_seq_blocks_q is None:
        n_seq_blocks_q = attention_layout.shape[1]
    if n_seq_blocks_k is None:
        n_seq_blocks_k = attention_layout.shape[2]

    if "attention_lut" not in lut:
        attention_lut, attention_offsets, max_kv_blocks = _build_attention_lut(
            attention_layout, n_batches, n_seq_blocks_q, n_seq_blocks_k)
        lut["attention_lut"] = attention_lut
        lut["attention_offsets"] = attention_offsets
        lut["max_kv_blocks"] = max_kv_blocks

    if "reverse_attention_lut" not in lut:
        attention_layout_t = attention_layout.transpose(1, 2).contiguous()
        reverse_attention_lut, reverse_attention_offsets, max_q_per_k = _build_attention_lut(
            attention_layout_t, n_batches, n_seq_blocks_k, n_seq_blocks_q)
        lut["reverse_attention_lut"] = reverse_attention_lut
        lut["reverse_attention_offsets"] = reverse_attention_offsets
        lut["max_q_per_k"] = max_q_per_k

    if sparsity_layout_q is not None and "sparsity_reverse_lut_q" not in lut:
        lut["sparsity_reverse_lut_q"] = build_reverse_lut(sparsity_layout_q)
        lut["n_sparse_blocks_q"] = int(sparsity_layout_q.sum().item())

    if sparsity_layout_k is not None and "sparsity_reverse_lut_k" not in lut:
        lut["sparsity_reverse_lut_k"] = build_reverse_lut(sparsity_layout_k)

    if sparsity_layout_v is not None and "sparsity_reverse_lut_v" not in lut:
        lut["sparsity_reverse_lut_v"] = build_reverse_lut(sparsity_layout_v)

    if "sparsity_reverse_lut_o" not in lut:
        if sparsity_layout_o is not None:
            lut["sparsity_reverse_lut_o"] = build_reverse_lut(sparsity_layout_o)
            lut["n_sparse_blocks_o"] = int(sparsity_layout_o.sum().item())
        elif sparsity_layout_q is not None and (n_head_blocks_v is None or n_head_blocks_v == sparsity_layout_q.size(2)):
            lut["sparsity_reverse_lut_o"] = lut["sparsity_reverse_lut_q"]
            lut["n_sparse_blocks_o"] = lut["n_sparse_blocks_q"]

    lut_tensors = [v for v in lut.values() if isinstance(v, Tensor)]
    if lut_tensors:
        validate_contiguous(*lut_tensors)

    return lut


def _build_attention_lut(attention_layout: Tensor, n_batches: int,
                         n_blocks_row: int, n_blocks_col: int) -> tuple[Tensor, Tensor, int]:
    device = attention_layout.device

    counts = attention_layout.sum(dim=2).flatten()
    max_blocks_per_row = int(counts.max().item())

    if max_blocks_per_row == 0:
        offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=torch.int32, device=device)
        attention_lut = torch.empty(0, dtype=torch.int32, device=device)
        return attention_lut, offsets, 1

    offsets = torch.zeros(n_batches * n_blocks_row + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    indices = attention_layout.reshape(n_batches * n_blocks_row, n_blocks_col).nonzero(as_tuple=False)
    attention_lut = indices[:, 1].to(torch.int32)

    return attention_lut, offsets, max_blocks_per_row

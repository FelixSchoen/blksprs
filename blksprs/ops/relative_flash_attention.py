"""Fused block-sparse Flash attention with query-relative embeddings.

The operation in this module is deliberately representation agnostic.  A
caller supplies one integer relation coordinate per query and key position;
the attention score receives ``q dot R[clip(q_relation - key_relation)]``.
This is the query-relative term used by relative-position attention, but it is
also useful for any ordered/discrete relation.  Crucially, embeddings are
looked up inside the streaming Flash kernels: no ``(sequence, sequence)``
relative-score tensor is materialised.
"""

import math
from numbers import Integral, Real

import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.flash_attention import flash_attention_build_layout_cache
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import cast_for_autocast
from blksprs.utils.validation import (
    ensure_contiguous,
    validate_binary,
    validate_contiguous,
    validate_device,
    validate_dimensions,
    validate_dtype_float,
    validate_shape,
    validate_sparsity,
    validate_sparsity_block_size,
    validate_sparsity_layout,
)

@torch.amp.custom_fwd(device_type="cuda")
def flash_attention_relative_embedding(
        q: BlksprsTensor,
        sparsity_layout_q: Tensor,
        k: BlksprsTensor,
        sparsity_layout_k: Tensor,
        v: BlksprsTensor,
        sparsity_layout_v: Tensor,
        attention_layout: Tensor,
        sparsity_block_size: int,
        query_relations: Tensor,
        key_relations: Tensor,
        relative_embedding: Tensor,
        relation_min: int,
        relation_max: int,
        *,
        scale: float | None = None,
        attention_mask: BlksprsTensor | None = None,
        sparsity_layout_mask: Tensor | None = None,
        layout_cache: dict | None = None,
        sparsity_layout_o: Tensor | None = None,
) -> BlksprsTensor:
    """Compute block-sparse Flash attention with fused relative embeddings.

    ``relative_embedding`` has shape ``(attention_batches, relation_count,
    head_dimension)``.  An *attention batch* commonly represents a
    batch/head pair, which provides independent tables per head without baking
    a head convention into this generic primitive.  The relation table covers
    the inclusive range ``[relation_min, relation_max]``.

    ``query_relations`` and ``key_relations`` have shape ``(attention_batches,
    sequence_length)`` and use arbitrary integral coordinates.  Their
    difference is clipped to the supplied inclusive range before lookup.

    The relative term is added after the usual scaled QK term, matching the
    original query-relative attention formulation::

        score = scale * (q @ k.T) + q @ R[clip(q_relation - key_relation)]

    The result, masking convention, and sparse tensor contracts are otherwise
    the same as :func:`blksprs.ops.flash_attention`.
    """
    q, k, v, attention_layout = ensure_contiguous(q, k, v, attention_layout)
    q, k, v, relative_embedding = cast_for_autocast(
        q, k, v, relative_embedding)
    query_relations = ensure_contiguous(query_relations)
    key_relations = ensure_contiguous(key_relations)
    relative_embedding = ensure_contiguous(relative_embedding)

    validate_dimensions(q, k, v)
    validate_contiguous(q, k, v)
    validate_dtype_float(q, k, v, relative_embedding)
    validate_device(q, k, v, attention_layout, query_relations, key_relations, relative_embedding)
    validate_sparsity(sparsity_block_size, (q, sparsity_layout_q), (k, sparsity_layout_k), (v, sparsity_layout_v))
    validate_sparsity_block_size(sparsity_block_size, q, k, v)
    if sparsity_block_size not in (16, 32, 64):
        raise ValueError("Relative Flash Attention supports block sizes 16, 32, and 64")

    validate_dimensions(attention_layout)
    validate_contiguous(attention_layout)
    validate_sparsity_layout(attention_layout)

    if query_relations.dim() != 2 or key_relations.dim() != 2:
        raise ValueError("query_relations and key_relations must have shape (attention_batches, sequence_length)")
    if query_relations.dtype == torch.bool or torch.is_floating_point(query_relations):
        raise TypeError("query_relations must use an integral dtype")
    if key_relations.dtype == torch.bool or torch.is_floating_point(key_relations):
        raise TypeError("key_relations must use an integral dtype")
    if relative_embedding.dim() != 3:
        raise ValueError("relative_embedding must have shape (attention_batches, relation_count, head_dimension)")
    if not isinstance(relation_min, Integral) or isinstance(relation_min, bool):
        raise TypeError("relation_min must be an integer")
    if not isinstance(relation_max, Integral) or isinstance(relation_max, bool):
        raise TypeError("relation_max must be an integer")
    if relation_min > relation_max:
        raise ValueError("relation_min must not exceed relation_max")

    n_batches = sparsity_layout_q.size(0)
    n_seq_blocks_q = sparsity_layout_q.size(1)
    n_head_blocks_qk = sparsity_layout_q.size(2)
    n_seq_blocks_k = sparsity_layout_k.size(1)
    n_head_blocks_v = sparsity_layout_v.size(2)
    if sparsity_layout_k.size(0) != n_batches or sparsity_layout_k.size(2) != n_head_blocks_qk:
        raise ValueError("K sparsity layout must be compatible with Q")
    if sparsity_layout_v.size(0) != n_batches or sparsity_layout_v.size(1) != n_seq_blocks_k:
        raise ValueError("V sparsity layout must be compatible with K")
    validate_shape(attention_layout, (n_batches, n_seq_blocks_q, n_seq_blocks_k), "attention_layout")

    q_length = n_seq_blocks_q * sparsity_block_size
    k_length = n_seq_blocks_k * sparsity_block_size
    head_dimension = n_head_blocks_qk * sparsity_block_size
    if query_relations.shape != (n_batches, q_length):
        raise ValueError("query_relations does not match Q's padded sequence shape")
    if key_relations.shape != (n_batches, k_length):
        raise ValueError("key_relations does not match K's padded sequence shape")
    if relative_embedding.shape != (n_batches, relation_max - relation_min + 1, head_dimension):
        raise ValueError(
            "relative_embedding must have shape "
            f"({n_batches}, {relation_max - relation_min + 1}, {head_dimension})"
        )
    if relative_embedding.dtype != q.dtype:
        raise ValueError("relative_embedding must use the same dtype as Q")

    if scale is None:
        scale = 1.0 if head_dimension == 0 else 1.0 / math.sqrt(head_dimension)
    elif not isinstance(scale, Real):
        raise TypeError("scale must be a real number or None")
    else:
        scale = float(scale)

    if (attention_mask is None) != (sparsity_layout_mask is None):
        raise ValueError("attention_mask and sparsity_layout_mask must be provided together")
    has_mask = attention_mask is not None
    if has_mask:
        attention_mask = ensure_contiguous(attention_mask)
        sparsity_layout_mask = ensure_contiguous(sparsity_layout_mask)
        validate_dimensions(attention_mask, sparsity_layout_mask)
        validate_contiguous(attention_mask, sparsity_layout_mask)
        validate_device(attention_mask, sparsity_layout_mask, q)
        if attention_mask.dtype != torch.bool:
            validate_dtype_float(attention_mask)
        validate_binary(attention_mask)
        validate_sparsity(sparsity_block_size, (attention_mask, sparsity_layout_mask))
        validate_shape(sparsity_layout_mask, (n_batches, n_seq_blocks_q, n_seq_blocks_k), "sparsity_layout_mask")
    else:
        attention_mask = torch.empty(0, dtype=q.dtype, device=q.device)
        sparsity_layout_mask = None

    if sparsity_layout_o is None:
        sparsity_layout_o = torch.any(attention_layout, dim=-1, keepdim=True).expand(
            -1, -1, n_head_blocks_v).contiguous()
    else:
        sparsity_layout_o = ensure_contiguous(sparsity_layout_o)
        validate_dimensions(sparsity_layout_o)
        validate_contiguous(sparsity_layout_o)
        validate_device(sparsity_layout_o, q)
        validate_sparsity_layout(sparsity_layout_o)
        validate_shape(sparsity_layout_o, (n_batches, n_seq_blocks_q, n_head_blocks_v), "sparsity_layout_o")

    layout_cache = flash_attention_build_layout_cache(
        attention_layout,
        sparsity_layout_q=sparsity_layout_q,
        sparsity_layout_k=sparsity_layout_k,
        sparsity_layout_v=sparsity_layout_v,
        sparsity_layout_o=sparsity_layout_o,
        sparsity_layout_mask=sparsity_layout_mask,
        layout_cache=layout_cache,
    )
    pidx_mask = layout_cache["packed_indices_mask"] if has_mask else torch.empty(
        0, dtype=torch.long, device=q.device)

    return BlksprsTensor.wrap(_RelativeFlashAttention.apply(
        q, k, v, query_relations, key_relations, relative_embedding, attention_mask,
        layout_cache["packed_indices_q"], layout_cache["packed_indices_k"],
        layout_cache["packed_indices_v"], layout_cache["packed_indices_o"],
        pidx_mask, layout_cache["key_indices"], layout_cache["key_offsets"],
        layout_cache["query_indices"], layout_cache["query_offsets"],
        sparsity_block_size, n_batches, n_seq_blocks_q, n_seq_blocks_k,
        n_head_blocks_qk, n_head_blocks_v, layout_cache["max_keys_per_query"],
        layout_cache["max_queries_per_key"], relation_min, relation_max,
        float(scale), has_mask,
    ))


class _RelativeFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, query_relations, key_relations, relative_embedding, attention_mask,
                pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, key_indices, key_offsets,
                query_indices, query_offsets, sparsity_block_size, n_batches, n_seq_blocks_q,
                n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v, max_keys_per_query,
                max_queries_per_key, relation_min, relation_max, scale, has_mask):
        n_sparse_output_blocks = int((pidx_o >= 0).sum().item())
        output = torch.zeros(
            (n_sparse_output_blocks, sparsity_block_size, sparsity_block_size),
            device=q.device, dtype=torch.float32,
        )
        lse = torch.full(
            (n_batches, n_seq_blocks_q, sparsity_block_size), float("-inf"),
            device=q.device, dtype=torch.float32,
        )
        grid = (n_batches, n_seq_blocks_q, sparsity_block_size)
        _relative_flash_attention_forward_kernel[grid](
            q, k, v, output, lse, query_relations, key_relations, relative_embedding,
            attention_mask, pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, key_indices, key_offsets,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v,
            max_keys_per_query, relation_min, relation_max, scale, has_mask,
            BLOCK_SIZE=sparsity_block_size,
        )
        output = output.to(q.dtype)
        ctx.save_for_backward(
            q, k, v, output, lse, query_relations, key_relations, relative_embedding, attention_mask,
            pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, key_indices, key_offsets,
            query_indices, query_offsets,
        )
        ctx.sparsity_block_size = sparsity_block_size
        ctx.n_batches = n_batches
        ctx.n_seq_blocks_q = n_seq_blocks_q
        ctx.n_seq_blocks_k = n_seq_blocks_k
        ctx.n_head_blocks_qk = n_head_blocks_qk
        ctx.n_head_blocks_v = n_head_blocks_v
        ctx.max_keys_per_query = max_keys_per_query
        ctx.max_queries_per_key = max_queries_per_key
        ctx.relation_min = relation_min
        ctx.relation_max = relation_max
        ctx.scale = scale
        ctx.has_mask = has_mask
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        (q, k, v, output, lse, query_relations, key_relations, relative_embedding, attention_mask,
         pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, key_indices, key_offsets,
         query_indices, query_offsets) = ctx.saved_tensors
        bs = ctx.sparsity_block_size
        delta = torch.empty(
            (ctx.n_batches, ctx.n_seq_blocks_q, bs), device=q.device, dtype=torch.float32)
        _relative_flash_attention_delta_kernel[(ctx.n_batches, ctx.n_seq_blocks_q, bs)](
            output, grad_output, delta, pidx_o, ctx.n_batches, ctx.n_seq_blocks_q,
            ctx.n_head_blocks_v, BLOCK_SIZE=bs,
        )
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        drelative_embedding = torch.zeros_like(relative_embedding, dtype=torch.float32)
        _relative_flash_attention_dq_kernel[(ctx.n_batches, ctx.n_seq_blocks_q, bs)](
            q, k, v, grad_output, dq, lse, delta, query_relations, key_relations,
            relative_embedding, attention_mask, pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask,
            key_indices, key_offsets, ctx.n_batches, ctx.n_seq_blocks_q, ctx.n_seq_blocks_k,
            ctx.n_head_blocks_qk, ctx.n_head_blocks_v, ctx.max_keys_per_query,
            ctx.relation_min, ctx.relation_max, ctx.scale, ctx.has_mask, BLOCK_SIZE=bs,
        )
        _relative_flash_attention_dkdv_kernel[(ctx.n_batches, ctx.n_seq_blocks_k, bs)](
            q, k, v, grad_output, dk, dv, drelative_embedding, lse, delta,
            query_relations, key_relations, relative_embedding, attention_mask,
            pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, query_indices, query_offsets,
            ctx.n_batches, ctx.n_seq_blocks_q, ctx.n_seq_blocks_k, ctx.n_head_blocks_qk,
            ctx.n_head_blocks_v, ctx.max_queries_per_key, ctx.relation_min, ctx.relation_max,
            ctx.scale, ctx.has_mask, BLOCK_SIZE=bs,
        )
        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None,
            drelative_embedding.to(relative_embedding.dtype), None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
        )


@triton.jit
def _load_scores(q, k, query_relations, key_relations, relative_embedding,
                 pidx_q, pidx_k, attention_mask, pidx_mask,
                 batch_index, query_block, query_row, key_block,
                 n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk,
                 relation_count, relation_min, relation_max, scale, has_mask,
                 BLOCK_SIZE: tl.constexpr):
    key_rows = tl.arange(0, BLOCK_SIZE)
    dims = tl.arange(0, BLOCK_SIZE)
    query_position = query_block * BLOCK_SIZE + query_row
    key_positions = key_block * BLOCK_SIZE + key_rows
    query_relation = tl.load(query_relations + batch_index * n_seq_blocks_q * BLOCK_SIZE + query_position)
    key_relation = tl.load(key_relations + batch_index * n_seq_blocks_k * BLOCK_SIZE + key_positions)
    relation_indices = tl.maximum(
        tl.minimum(query_relation - key_relation, relation_max), relation_min) - relation_min
    qk_score = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    relative_score = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for head_block in range(n_head_blocks_qk):
        packed_q = tl.load(pidx_q + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_qk + head_block)
        packed_k = tl.load(pidx_k + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_qk + head_block)
        if (packed_q >= 0) & (packed_k >= 0):
            q_values = tl.load(q + packed_q * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims)
            k_values = tl.load(
                k + packed_k * BLOCK_SIZE * BLOCK_SIZE + key_rows[:, None] * BLOCK_SIZE + dims[None, :])
            relative_values = tl.load(
                relative_embedding + batch_index * relation_count * n_head_blocks_qk * BLOCK_SIZE
                + relation_indices[:, None] * n_head_blocks_qk * BLOCK_SIZE
                + head_block * BLOCK_SIZE + dims[None, :])
            qk_score += tl.sum(k_values * q_values[None, :], axis=1)
            relative_score += tl.sum(relative_values * q_values[None, :], axis=1)
    score = qk_score * scale + relative_score
    if has_mask:
        packed_mask = tl.load(
            pidx_mask + (batch_index * n_seq_blocks_q + query_block) * n_seq_blocks_k + key_block)
        if packed_mask >= 0:
            mask_values = tl.load(
                attention_mask + packed_mask * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + key_rows)
            score = tl.where(mask_values != 0, float("-inf"), score)
    return score, relation_indices


@triton.jit
def _relative_flash_attention_forward_kernel(
        q, k, v, output, lse, query_relations, key_relations, relative_embedding, attention_mask,
        pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, key_indices, key_offsets,
        n_batches, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, n_head_blocks_v,
        max_keys_per_query, relation_min, relation_max, scale, has_mask,
        BLOCK_SIZE: tl.constexpr):
    batch_index = tl.program_id(0)
    query_block = tl.program_id(1)
    query_row = tl.program_id(2)
    key_rows = tl.arange(0, BLOCK_SIZE)
    dims = tl.arange(0, BLOCK_SIZE)
    relation_count = relation_max - relation_min + 1
    key_start = tl.load(key_offsets + batch_index * n_seq_blocks_q + query_block)
    key_end = tl.load(key_offsets + batch_index * n_seq_blocks_q + query_block + 1)
    n_key_blocks = key_end - key_start
    maximum = float("-inf")
    normaliser = 0.0
    for key_offset in range(max_keys_per_query):
        if key_offset < n_key_blocks:
            key_block = tl.load(key_indices + key_start + key_offset)
            score, _ = _load_scores(
                q, k, query_relations, key_relations, relative_embedding, pidx_q, pidx_k,
                attention_mask, pidx_mask, batch_index, query_block, query_row, key_block,
                n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, relation_count,
                relation_min, relation_max, scale, has_mask, BLOCK_SIZE,
            )
            maximum_next = tl.maximum(maximum, tl.max(score, axis=0))
            alpha = tl.where(
                (maximum == float("-inf")) & (maximum_next == float("-inf")),
                1.0,
                tl.exp(maximum - maximum_next),
            )
            probabilities = tl.exp(score - maximum_next)
            probabilities = tl.where(
                (score == float("-inf")) & (maximum_next == float("-inf")), 0.0, probabilities)
            normaliser = normaliser * alpha + tl.sum(probabilities, axis=0)
            for value_block in range(n_head_blocks_v):
                packed_o = tl.load(
                    pidx_o + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_v + value_block)
                packed_v = tl.load(
                    pidx_v + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_v + value_block)
                if packed_o >= 0:
                    accumulator = tl.load(
                        output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims).to(tl.float32)
                    accumulator *= alpha
                    if packed_v >= 0:
                        values = tl.load(
                            v + packed_v * BLOCK_SIZE * BLOCK_SIZE + key_rows[:, None] * BLOCK_SIZE + dims[None, :])
                        accumulator += tl.sum(probabilities[:, None] * values, axis=0)
                    tl.store(output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims, accumulator)
            maximum = maximum_next
    has_attention = normaliser != 0
    normaliser_safe = tl.where(has_attention, normaliser, 1.0)
    for value_block in range(n_head_blocks_v):
        packed_o = tl.load(pidx_o + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_v + value_block)
        if packed_o >= 0:
            accumulator = tl.load(
                output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims).to(tl.float32)
            accumulator = tl.where(has_attention, accumulator / normaliser_safe, 0.0)
            tl.store(output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims, accumulator)
    tl.store(
        lse + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row,
        tl.where(has_attention, maximum + tl.log(normaliser_safe), float("-inf")),
    )


@triton.jit
def _relative_flash_attention_delta_kernel(output, grad_output, delta, pidx_o,
                                           n_batches, n_seq_blocks_q, n_head_blocks_v,
                                           BLOCK_SIZE: tl.constexpr):
    batch_index = tl.program_id(0)
    query_block = tl.program_id(1)
    query_row = tl.program_id(2)
    dims = tl.arange(0, BLOCK_SIZE)
    delta_value = 0.0
    for value_block in range(n_head_blocks_v):
        packed_o = tl.load(pidx_o + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_v + value_block)
        if packed_o >= 0:
            offsets = packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims
            delta_value += tl.sum(
                tl.load(output + offsets).to(tl.float32) * tl.load(grad_output + offsets).to(tl.float32), axis=0)
    tl.store(delta + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row, delta_value)


@triton.jit
def _relative_flash_attention_dq_kernel(
        q, k, v, grad_output, dq, lse, delta, query_relations, key_relations,
        relative_embedding, attention_mask, pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask,
        key_indices, key_offsets, n_batches, n_seq_blocks_q, n_seq_blocks_k,
        n_head_blocks_qk, n_head_blocks_v, max_keys_per_query,
        relation_min, relation_max, scale, has_mask, BLOCK_SIZE: tl.constexpr):
    batch_index = tl.program_id(0)
    query_block = tl.program_id(1)
    query_row = tl.program_id(2)
    key_rows = tl.arange(0, BLOCK_SIZE)
    dims = tl.arange(0, BLOCK_SIZE)
    relation_count = relation_max - relation_min + 1
    lse_value = tl.load(lse + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row)
    delta_value = tl.load(delta + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row)
    key_start = tl.load(key_offsets + batch_index * n_seq_blocks_q + query_block)
    key_end = tl.load(key_offsets + batch_index * n_seq_blocks_q + query_block + 1)
    n_key_blocks = key_end - key_start
    for head_block in range(n_head_blocks_qk):
        gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for key_offset in range(max_keys_per_query):
            if key_offset < n_key_blocks:
                key_block = tl.load(key_indices + key_start + key_offset)
                score, relation_indices = _load_scores(
                    q, k, query_relations, key_relations, relative_embedding, pidx_q, pidx_k,
                    attention_mask, pidx_mask, batch_index, query_block, query_row, key_block,
                    n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, relation_count,
                    relation_min, relation_max, scale, has_mask, BLOCK_SIZE,
                )
                probabilities = tl.where(
                    lse_value == float("-inf"), 0.0, tl.exp(score - lse_value))
                dot_product_gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                for value_block in range(n_head_blocks_v):
                    packed_o = tl.load(pidx_o + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_v + value_block)
                    packed_v = tl.load(pidx_v + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_v + value_block)
                    if (packed_o >= 0) & (packed_v >= 0):
                        output_gradient = tl.load(
                            grad_output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims)
                        values = tl.load(
                            v + packed_v * BLOCK_SIZE * BLOCK_SIZE + key_rows[:, None] * BLOCK_SIZE + dims[None, :])
                        dot_product_gradient += tl.sum(values * output_gradient[None, :], axis=1)
                score_gradient = probabilities * (dot_product_gradient - delta_value)
                packed_k = tl.load(pidx_k + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_qk + head_block)
                if packed_k >= 0:
                    keys = tl.load(
                        k + packed_k * BLOCK_SIZE * BLOCK_SIZE + key_rows[:, None] * BLOCK_SIZE + dims[None, :])
                    relatives = tl.load(
                        relative_embedding + batch_index * relation_count * n_head_blocks_qk * BLOCK_SIZE
                        + relation_indices[:, None] * n_head_blocks_qk * BLOCK_SIZE
                        + head_block * BLOCK_SIZE + dims[None, :])
                    gradient += tl.sum(score_gradient[:, None] * (keys * scale + relatives), axis=0)
        packed_q = tl.load(pidx_q + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_qk + head_block)
        if packed_q >= 0:
            tl.store(dq + packed_q * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims, gradient)


@triton.jit
def _relative_flash_attention_dkdv_kernel(
        q, k, v, grad_output, dk, dv, drelative_embedding, lse, delta,
        query_relations, key_relations, relative_embedding, attention_mask,
        pidx_q, pidx_k, pidx_v, pidx_o, pidx_mask, query_indices, query_offsets,
        n_batches, n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk,
        n_head_blocks_v, max_queries_per_key, relation_min, relation_max,
        scale, has_mask, BLOCK_SIZE: tl.constexpr):
    batch_index = tl.program_id(0)
    key_block = tl.program_id(1)
    key_row = tl.program_id(2)
    key_rows = tl.arange(0, BLOCK_SIZE)
    dims = tl.arange(0, BLOCK_SIZE)
    relation_count = relation_max - relation_min + 1
    key_position = key_block * BLOCK_SIZE + key_row
    key_relation = tl.load(key_relations + batch_index * n_seq_blocks_k * BLOCK_SIZE + key_position)
    query_start = tl.load(query_offsets + batch_index * n_seq_blocks_k + key_block)
    query_end = tl.load(query_offsets + batch_index * n_seq_blocks_k + key_block + 1)
    n_query_blocks = query_end - query_start
    for head_block in range(n_head_blocks_qk):
        gradient_key = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for query_offset in range(max_queries_per_key):
            if query_offset < n_query_blocks:
                query_block = tl.load(query_indices + query_start + query_offset)
                for query_row in range(BLOCK_SIZE):
                    score, relation_indices = _load_scores(
                        q, k, query_relations, key_relations, relative_embedding, pidx_q, pidx_k,
                        attention_mask, pidx_mask, batch_index, query_block, query_row, key_block,
                        n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, relation_count,
                        relation_min, relation_max, scale, has_mask, BLOCK_SIZE,
                    )
                    lse_value = tl.load(lse + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row)
                    delta_value = tl.load(delta + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row)
                    probabilities = tl.where(
                        lse_value == float("-inf"), 0.0, tl.exp(score - lse_value))
                    dot_product_gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                    for value_block in range(n_head_blocks_v):
                        packed_o = tl.load(pidx_o + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_v + value_block)
                        packed_v = tl.load(pidx_v + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_v + value_block)
                        if (packed_o >= 0) & (packed_v >= 0):
                            output_gradient = tl.load(
                                grad_output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims)
                            values = tl.load(
                                v + packed_v * BLOCK_SIZE * BLOCK_SIZE + key_rows[:, None] * BLOCK_SIZE + dims[None, :])
                            dot_product_gradient += tl.sum(values * output_gradient[None, :], axis=1)
                    score_gradient = probabilities * (dot_product_gradient - delta_value)
                    key_score_gradient = tl.sum(
                        score_gradient * (key_rows == key_row), axis=0)
                    packed_q = tl.load(pidx_q + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_qk + head_block)
                    if packed_q >= 0:
                        query_values = tl.load(
                            q + packed_q * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims)
                        gradient_key += key_score_gradient * query_values * scale
                        query_position = query_block * BLOCK_SIZE + query_row
                        query_relation = tl.load(
                            query_relations + batch_index * n_seq_blocks_q * BLOCK_SIZE + query_position)
                        relation_index = tl.maximum(
                            tl.minimum(query_relation - key_relation, relation_max), relation_min) - relation_min
                        relative_offsets = (
                            batch_index * relation_count * n_head_blocks_qk * BLOCK_SIZE
                            + relation_index * n_head_blocks_qk * BLOCK_SIZE + head_block * BLOCK_SIZE + dims)
                        tl.atomic_add(
                            drelative_embedding + relative_offsets,
                            key_score_gradient * query_values,
                        )
        packed_k = tl.load(pidx_k + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_qk + head_block)
        if packed_k >= 0:
            tl.store(dk + packed_k * BLOCK_SIZE * BLOCK_SIZE + key_row * BLOCK_SIZE + dims, gradient_key)
    for value_block in range(n_head_blocks_v):
        gradient_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for query_offset in range(max_queries_per_key):
            if query_offset < n_query_blocks:
                query_block = tl.load(query_indices + query_start + query_offset)
                for query_row in range(BLOCK_SIZE):
                    score, _ = _load_scores(
                        q, k, query_relations, key_relations, relative_embedding, pidx_q, pidx_k,
                        attention_mask, pidx_mask, batch_index, query_block, query_row, key_block,
                        n_seq_blocks_q, n_seq_blocks_k, n_head_blocks_qk, relation_count,
                        relation_min, relation_max, scale, has_mask, BLOCK_SIZE,
                    )
                    lse_value = tl.load(lse + (batch_index * n_seq_blocks_q + query_block) * BLOCK_SIZE + query_row)
                    probabilities = tl.where(
                        lse_value == float("-inf"), 0.0, tl.exp(score - lse_value))
                    packed_o = tl.load(pidx_o + (batch_index * n_seq_blocks_q + query_block) * n_head_blocks_v + value_block)
                    if packed_o >= 0:
                        output_gradient = tl.load(
                            grad_output + packed_o * BLOCK_SIZE * BLOCK_SIZE + query_row * BLOCK_SIZE + dims)
                        key_probability = tl.sum(
                            probabilities * (key_rows == key_row), axis=0)
                        gradient_value += key_probability * output_gradient
        packed_v = tl.load(pidx_v + (batch_index * n_seq_blocks_k + key_block) * n_head_blocks_v + value_block)
        if packed_v >= 0:
            tl.store(dv + packed_v * BLOCK_SIZE * BLOCK_SIZE + key_row * BLOCK_SIZE + dims, gradient_value)

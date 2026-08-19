"""Block-sparse Flash attention with learned query-relative embeddings.

Queries are projected against the bounded relation tables with tensor-core
matrix multiplication. The tiled Flash kernels then gather projected scores
directly for active attention blocks. The projection is recomputed during
backward so long-sequence training does not retain a large score table for
every layer.
"""

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch
from torch import Tensor

from blksprs.ops.conversion import to_dense
from blksprs.ops.flash_attention import (
    _flash_attention_projected_relative,
    _normalise_causal_lengths,
    flash_attention_build_layout_cache,
)
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import cast_for_autocast
from blksprs.utils.validation import (
    ensure_contiguous,
    validate_contiguous,
    validate_binary,
    validate_device,
    validate_dimensions,
    validate_dtype_float,
    validate_shape,
    validate_sparsity,
    validate_sparsity_block_size,
    validate_sparsity_layout,
)

MAX_RELATIONS = 8


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
        key_relations_are_unique: bool = False,
        causal_lengths: Tensor | None = None,
        scale: float | None = None,
        attention_mask: BlksprsTensor | None = None,
        sparsity_layout_mask: Tensor | None = None,
        layout_cache: dict | None = None,
        sparsity_layout_o: Tensor | None = None,
) -> BlksprsTensor:
    """Compute Flash attention with one learned query-relative term.

    The relative score is ``q @ R[clip(q_relation - key_relation)]``. See
    :func:`flash_attention_relative_embeddings` for the multi-relation form
    and the complete tensor contracts. Set ``key_relations_are_unique`` only
    when every attention batch has pairwise-distinct valid key coordinates.
    """
    return flash_attention_relative_embeddings(
        q,
        sparsity_layout_q,
        k,
        sparsity_layout_k,
        v,
        sparsity_layout_v,
        attention_layout,
        sparsity_block_size,
        query_relations.unsqueeze(0),
        key_relations.unsqueeze(0),
        relative_embedding,
        ((relation_min, relation_max),),
        key_relations_are_unique=(key_relations_are_unique,),
        causal_lengths=causal_lengths,
        scale=scale,
        attention_mask=attention_mask,
        sparsity_layout_mask=sparsity_layout_mask,
        layout_cache=layout_cache,
        sparsity_layout_o=sparsity_layout_o,
    )


@torch.amp.custom_fwd(device_type="cuda")
def flash_attention_relative_embeddings(
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
        relative_embeddings: Tensor,
        relation_bounds: Sequence[Sequence[int]],
        *,
        key_relations_are_unique: Sequence[bool] | None = None,
        causal_lengths: Tensor | None = None,
        query_relation_validity: Tensor | None = None,
        key_relation_validity: Tensor | None = None,
        scale: float | None = None,
        attention_mask: BlksprsTensor | None = None,
        sparsity_layout_mask: Tensor | None = None,
        layout_cache: dict | None = None,
        sparsity_layout_o: Tensor | None = None,
) -> BlksprsTensor:
    """Compute block-sparse Flash attention with learned relative terms.

    ``query_relations`` and ``key_relations`` have shape
    ``(n_relations, attention_batches, sequence_length)``. The corresponding
    relation tables are concatenated in ``relative_embeddings``, whose shape
    is ``(attention_batches, total_relation_count, head_dimension)``.
    ``relation_bounds`` gives the inclusive minimum and maximum for every
    table, in concatenation order.

    ``query_relation_validity`` and ``key_relation_validity`` are optional
    boolean tensors with the same shapes as their coordinate tensors. When
    supplied, a relation contributes only if both positions are valid. This
    keeps the coordinate domain unrestricted instead of reserving a sentinel.

    ``causal_lengths`` optionally provides the valid self-attention length for
    every flattened attention batch. It applies causal and padding masking
    directly in the kernels and can replace an explicit causal mask.

    ``key_relations_are_unique`` can mark relation-coordinate rows whose valid
    key values are pairwise distinct. This permits a faster, non-atomic
    interior-gradient write. It is a performance assertion: callers must only
    enable a relation when the property holds for every attention batch.

    The score is
    ``scale * (q @ k.T) + sum(q @ R_i[clip(q_relation_i - key_relation_i)])``.

    At most eight independent relations are accepted. This is a compile-time
    guard: Triton unrolls the relation loop, so an unbounded count would create
    impractically large kernels. The result and mask conventions match
    :func:`blksprs.ops.flash_attention`.
    """
    if query_relations.dim() != 3 or key_relations.dim() != 3:
        raise ValueError(
            "query_relations and key_relations must have shape "
            "(n_relations, attention_batches, sequence_length)."
        )
    if query_relations.dtype == torch.bool or torch.is_floating_point(
            query_relations):
        raise TypeError("query_relations must use an integral dtype.")
    if key_relations.dtype == torch.bool or torch.is_floating_point(
            key_relations):
        raise TypeError("key_relations must use an integral dtype.")

    q, k, v, attention_layout = ensure_contiguous(q, k, v, attention_layout)
    q, k, v, relative_embeddings = cast_for_autocast(
        q, k, v, relative_embeddings)
    query_relations = ensure_contiguous(query_relations.to(dtype=torch.int64))
    key_relations = ensure_contiguous(key_relations.to(dtype=torch.int64))
    relative_embeddings = ensure_contiguous(relative_embeddings)

    validate_dimensions(q, k, v, attention_layout)
    validate_contiguous(q, k, v, attention_layout, query_relations,
                        key_relations, relative_embeddings)
    validate_dtype_float(q, k, v, relative_embeddings)
    validate_device(q, k, v, attention_layout, query_relations,
                    key_relations, relative_embeddings)
    validate_sparsity(
        sparsity_block_size,
        (q, sparsity_layout_q),
        (k, sparsity_layout_k),
        (v, sparsity_layout_v),
    )
    validate_sparsity_block_size(sparsity_block_size, q, k, v)
    if sparsity_block_size not in (16, 32, 64):
        raise ValueError(
            "Relative Flash attention supports block sizes 16, 32, and 64.")
    validate_sparsity_layout(attention_layout)

    normalised_bounds = _normalise_relation_bounds(relation_bounds)
    n_relations = len(normalised_bounds)
    unique_key_relations = _normalise_unique_key_relations(
        key_relations_are_unique,
        n_relations,
    )
    n_batches = sparsity_layout_q.size(0)
    n_seq_blocks_q = sparsity_layout_q.size(1)
    n_head_blocks_qk = sparsity_layout_q.size(2)
    n_seq_blocks_k = sparsity_layout_k.size(1)

    if (sparsity_layout_k.size(0) != n_batches
            or sparsity_layout_k.size(2) != n_head_blocks_qk):
        raise ValueError("K sparsity layout must be compatible with Q.")
    if (sparsity_layout_v.size(0) != n_batches
            or sparsity_layout_v.size(1) != n_seq_blocks_k):
        raise ValueError("V sparsity layout must be compatible with K.")
    validate_shape(
        attention_layout,
        (n_batches, n_seq_blocks_q, n_seq_blocks_k),
        "attention_layout",
    )

    q_length = n_seq_blocks_q * sparsity_block_size
    k_length = n_seq_blocks_k * sparsity_block_size
    head_dimension = n_head_blocks_qk * sparsity_block_size
    causal_lengths_value = _normalise_causal_lengths(
        causal_lengths,
        n_batches,
        q_length,
        k_length,
        q.device,
    )
    validate_shape(
        query_relations,
        (n_relations, n_batches, q_length),
        "query_relations",
    )
    validate_shape(
        key_relations,
        (n_relations, n_batches, k_length),
        "key_relations",
    )
    relation_counts = tuple(
        relation_max - relation_min + 1
        for relation_min, relation_max in normalised_bounds
    )
    total_relation_count = sum(relation_counts)
    validate_shape(
        relative_embeddings,
        (n_batches, total_relation_count, head_dimension),
        "relative_embeddings",
    )
    if relative_embeddings.dtype != q.dtype:
        raise ValueError("relative_embeddings must use the same dtype as Q.")

    if scale is None:
        scale = 1.0 if head_dimension == 0 else 1.0 / math.sqrt(head_dimension)
    elif not isinstance(scale, Real):
        raise TypeError("scale must be a real number or None.")
    else:
        scale = float(scale)

    if (attention_mask is None) != (sparsity_layout_mask is None):
        raise ValueError(
            "attention_mask and sparsity_layout_mask must be provided together.")
    has_mask = attention_mask is not None
    attention_mask_value: Tensor
    sparsity_layout_mask_value: Tensor | None
    if attention_mask is not None and sparsity_layout_mask is not None:
        attention_mask_value = ensure_contiguous(attention_mask)
        sparsity_layout_mask_value = ensure_contiguous(sparsity_layout_mask)
        validate_dimensions(attention_mask_value, sparsity_layout_mask_value)
        validate_contiguous(attention_mask_value, sparsity_layout_mask_value)
        validate_device(attention_mask_value, sparsity_layout_mask_value, q)
        if attention_mask_value.dtype != torch.bool:
            validate_dtype_float(attention_mask_value)
        validate_binary(attention_mask_value)
        validate_sparsity(
            sparsity_block_size,
            (attention_mask_value, sparsity_layout_mask_value),
        )
        validate_shape(
            sparsity_layout_mask_value,
            (n_batches, n_seq_blocks_q, n_seq_blocks_k),
            "sparsity_layout_mask",
        )
    else:
        attention_mask_value = torch.empty(
            0, dtype=q.dtype, device=q.device)
        sparsity_layout_mask_value = None
    if (query_relation_validity is None) != (key_relation_validity is None):
        raise ValueError(
            "query_relation_validity and key_relation_validity must be "
            "provided together."
        )
    has_relation_validity = query_relation_validity is not None
    query_relation_validity_value: Tensor
    key_relation_validity_value: Tensor
    if (query_relation_validity is not None
            and key_relation_validity is not None):
        query_relation_validity_value = ensure_contiguous(
            query_relation_validity)
        key_relation_validity_value = ensure_contiguous(
            key_relation_validity)
        validate_device(
            query_relation_validity_value, key_relation_validity_value, q)
        if (query_relation_validity_value.dtype != torch.bool
                or key_relation_validity_value.dtype != torch.bool):
            raise TypeError("Relation-validity tensors must use bool dtype.")
        validate_shape(
            query_relation_validity_value,
            (n_relations, n_batches, q_length),
            "query_relation_validity",
        )
        validate_shape(
            key_relation_validity_value,
            (n_relations, n_batches, k_length),
            "key_relation_validity",
        )
    else:
        query_relation_validity_value = torch.empty(
            0, dtype=torch.bool, device=q.device)
        key_relation_validity_value = torch.empty(
            0, dtype=torch.bool, device=q.device)

    bounds_with_offsets = []
    relation_offset = 0
    for (relation_min, relation_max), relation_count in zip(
            normalised_bounds, relation_counts, strict=True):
        bounds_with_offsets.append(
            (relation_min, relation_max, relation_offset))
        relation_offset += relation_count
    relation_metadata = torch.tensor(
        bounds_with_offsets,
        dtype=torch.int64,
        device=q.device,
    )

    layout_cache = flash_attention_build_layout_cache(
        attention_layout,
        sparsity_layout_q=sparsity_layout_q,
        sparsity_layout_k=sparsity_layout_k,
        sparsity_layout_v=sparsity_layout_v,
        sparsity_layout_o=sparsity_layout_o,
        sparsity_layout_mask=sparsity_layout_mask_value,
        layout_cache=layout_cache,
    )

    dense_q = to_dense(q, sparsity_layout_q, sparsity_block_size)
    return _flash_attention_projected_relative(
        q,
        sparsity_layout_q,
        k,
        sparsity_layout_k,
        v,
        sparsity_layout_v,
        attention_mask_value,
        causal_lengths_value,
        dense_q,
        relative_embeddings,
        query_relations,
        key_relations,
        relation_metadata,
        query_relation_validity_value,
        key_relation_validity_value,
        layout_cache,
        sparsity_block_size,
        n_seq_blocks_q,
        n_seq_blocks_k,
        n_head_blocks_qk,
        sparsity_layout_v.size(2),
        scale,
        has_mask,
        n_batches,
        n_relations,
        has_relation_validity,
        sum(
            int(keys_are_unique) << relation
            for relation, keys_are_unique in enumerate(unique_key_relations)
        ),
    )


def _normalise_relation_bounds(
        relation_bounds: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    if not isinstance(relation_bounds, Sequence):
        raise TypeError("relation_bounds must be a sequence.")
    if not 1 <= len(relation_bounds) <= MAX_RELATIONS:
        raise ValueError(
            f"relation_bounds must contain between 1 and {MAX_RELATIONS} ranges.")

    result: list[tuple[int, int]] = []
    for bounds in relation_bounds:
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(
                "Every relation bound must contain [minimum, maximum].")
        relation_min, relation_max = bounds
        if (not isinstance(relation_min, Integral)
                or isinstance(relation_min, bool)):
            raise TypeError("Relation bounds must be integers.")
        if (not isinstance(relation_max, Integral)
                or isinstance(relation_max, bool)):
            raise TypeError("Relation bounds must be integers.")
        if relation_min > relation_max:
            raise ValueError(
                "A relation minimum must not exceed its maximum.")
        result.append((int(relation_min), int(relation_max)))
    return tuple(result)


def _normalise_unique_key_relations(
        values: Sequence[bool] | None,
        n_relations: int,
) -> tuple[bool, ...]:
    if values is None:
        return (False,) * n_relations
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("key_relations_are_unique must be a sequence of bool values.")
    if len(values) != n_relations:
        raise ValueError(
            "key_relations_are_unique must contain one value per relation."
        )
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("key_relations_are_unique must contain only bool values.")
    return tuple(values)

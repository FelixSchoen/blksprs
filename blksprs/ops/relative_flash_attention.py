import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import NamedTuple, cast

import torch
from torch import Tensor

from blksprs.ops.conversion import to_dense, to_sparse
from blksprs.ops.flash_attention import _FlashAttentionArguments, _RelativeAttentionArguments, _flash_attention, \
    _normalise_causal_lengths, _schedule_group_blocks, flash_attention, flash_attention_build_layout_cache
from blksprs.ops.relative_score_blocks import ReferencedScoreBlocks, compact_score_relations, \
    compress_referenced_score_blocks, mark_referenced_score_blocks, referenced_value_intervals
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.public_op import opaque_public_op
from blksprs.utils.tools import cast_for_autocast, finalize_layout_cache
from blksprs.utils.validation import ensure_contiguous, validate_contiguous, validate_binary, validate_device, \
    validate_dimensions, validate_dtype_float, validate_shape, validate_sparsity, validate_sparsity_block_size, \
    validate_sparsity_layout

MAX_RELATIONS = 8
SCORE_PRECALCULATIONS = ("referenced_blocks", "complete")
# Coordinates of at most this magnitude keep the differences of their block bounds, including the bounds of ignored
# positions, within 64 bits.
_COORDINATE_LIMIT = 2 ** 60
_COORDINATE_CACHE_KEY = "relative_coordinates"


@opaque_public_op
def flash_attention_relative_embedding(q: BlksprsTensor, sparsity_layout_q: Tensor,
                                       k: BlksprsTensor, sparsity_layout_k: Tensor,
                                       v: BlksprsTensor, sparsity_layout_v: Tensor,
                                       attention_layout: Tensor,
                                       sparsity_block_size: int,
                                       query_relations: Tensor, key_relations: Tensor,
                                       relative_embedding: Tensor, relation_min: int, relation_max: int, *,
                                       key_relations_are_unique: bool = False,
                                       key_relations_repeat: bool | None = None,
                                       causal_lengths: Tensor | None = None,
                                       scale: float | None = None,
                                       attention_mask: BlksprsTensor | None = None,
                                       sparsity_layout_mask: Tensor | None = None,
                                       layout_cache: dict | None = None,
                                       sparsity_layout_o: Tensor | None = None,
                                       score_precalculation: str = "referenced_blocks") -> BlksprsTensor:
    """Computes block-sparse Flash Attention with one learned query-relative term.

    The relative score is ``q @ R[clip(q_relation - key_relation)]``. See :func:`flash_attention_relative_embeddings`
    for the multi-relation form and the complete tensor contracts.

    Args:
        q (BlksprsTensor): The query tensor in compressed form.
        sparsity_layout_q (Tensor): The sparsity layout of ``q``.
        k (BlksprsTensor): The key tensor in compressed form.
        sparsity_layout_k (Tensor): The sparsity layout of ``k``.
        v (BlksprsTensor): The value tensor in compressed form.
        sparsity_layout_v (Tensor): The sparsity layout of ``v``.
        attention_layout (Tensor): The block attention pattern ``(n_batches, seq_q // bs, seq_k // bs)``.
        sparsity_block_size (int): The size of the sparsity blocks.
        query_relations (Tensor): The integral relation coordinates of the queries with shape ``(n_batches, seq_q)``.
        key_relations (Tensor): The integral relation coordinates of the keys with shape ``(n_batches, seq_k)``.
        relative_embedding (Tensor): The relation table with shape
            ``(n_batches, relation_max - relation_min + 1, head_dim)``.
        relation_min (int): The inclusive minimum of the relation.
        relation_max (int): The inclusive maximum of the relation.
        key_relations_are_unique (bool, optional): Whether every attention batch has pairwise-distinct valid key
            coordinates (default ``False``).
        key_relations_repeat (bool, optional): Whether consecutive keys mostly share their coordinates, see
            :func:`flash_attention_relative_embeddings` (default ``None``, which is ``not key_relations_are_unique``).
        causal_lengths (Tensor, optional): The valid self-attention length of every attention batch
            (default ``None``).
        scale (float, optional): The real-valued attention scale (default ``1/sqrt(head_dim)``).
        attention_mask (BlksprsTensor, optional): A binary mask in compressed form where a non-zero value marks an
            ignored position (default ``None``).
        sparsity_layout_mask (Tensor, optional): The sparsity layout for the mask (default ``None``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).
        sparsity_layout_o (Tensor, optional): The output sparsity layout (default ``None``).
        score_precalculation (str, optional): The pre-calculation of the projected relative scores, either
            ``"referenced_blocks"`` or ``"complete"`` (default ``"referenced_blocks"``).

    Returns:
        BlksprsTensor: The attention output in compressed form.

    """
    return flash_attention_relative_embeddings(
        q, sparsity_layout_q, k, sparsity_layout_k, v, sparsity_layout_v, attention_layout, sparsity_block_size,
        query_relations.unsqueeze(0), key_relations.unsqueeze(0), relative_embedding, ((relation_min, relation_max),),
        key_relations_are_unique=(key_relations_are_unique,),
        key_relations_repeat=None if key_relations_repeat is None else (key_relations_repeat,),
        causal_lengths=causal_lengths,
        scale=scale,
        attention_mask=attention_mask,
        sparsity_layout_mask=sparsity_layout_mask,
        layout_cache=layout_cache,
        sparsity_layout_o=sparsity_layout_o,
        score_precalculation=score_precalculation)


@opaque_public_op
def flash_attention_relative_embeddings(q: BlksprsTensor, sparsity_layout_q: Tensor,
                                        k: BlksprsTensor, sparsity_layout_k: Tensor,
                                        v: BlksprsTensor, sparsity_layout_v: Tensor,
                                        attention_layout: Tensor,
                                        sparsity_block_size: int,
                                        query_relations: Tensor, key_relations: Tensor,
                                        relative_embeddings: Tensor, relation_bounds: Sequence[Sequence[int]], *,
                                        key_relations_are_unique: Sequence[bool] | None = None,
                                        key_relations_repeat: Sequence[bool] | None = None,
                                        causal_lengths: Tensor | None = None,
                                        query_relation_validity: Tensor | None = None,
                                        key_relation_validity: Tensor | None = None,
                                        scale: float | None = None,
                                        attention_mask: BlksprsTensor | None = None,
                                        sparsity_layout_mask: Tensor | None = None,
                                        layout_cache: dict | None = None,
                                        sparsity_layout_o: Tensor | None = None,
                                        score_precalculation: str = "referenced_blocks") -> BlksprsTensor:
    """Computes block-sparse Flash Attention with learned relative terms.

    The score is ``scale * (q @ k.T) + sum(q @ R_i[clip(q_relation_i - key_relation_i)])``. Queries are projected
    against the bounded relation tables with tensor-core matrix multiplications, and the tiled Flash Attention kernels
    gather the relative terms from these pre-calculated scores ``S_pre = Q E^T`` of the queries and all table rows. The
    projection is recomputed during backward, so that long-sequence training does not retain a large score table for
    every layer.

    With ``score_precalculation="referenced_blocks"``, only the blocks of ``S_pre`` are computed whose relation values
    are referenced by a permitted query-key pair, that is, a pair in the attention layout, within the causal lengths,
    and valid for the relation, see ``blksprs.ops.relative_score_blocks``, together with short gaps between them, so
    that most query blocks address the scores of a relation through one offset. Blocks span ``sparsity_block_size``
    queries and relation values. With ``"complete"``, the complete tables are projected for every query. The
    referenced blocks are derived from the coordinates and stored in the layout cache, so that calls that share a
    layout cache and the same coordinate, validity, and causal-length tensors reuse them.

    Note:
        At most eight independent relations are accepted. This is a compile-time guard: Triton unrolls the relation
        loop, so an unbounded count would create impractically large kernels. The compiled kernels only depend on the
        static properties of a call, such as the number, bounds, and unique and repeated keys of the relations, and
        never on the coordinates, see :func:`flash_attention_warm_up`. Tiles that clip the first relation, or the
        first two, to their maximum take their scores from a single table column, so that the relations that clip
        most often, such as positions and times, should come first. Supported sparsity block sizes are 16,
        32, and 64. The result and mask conventions match :func:`blksprs.ops.flash_attention`.

    Args:
        q (BlksprsTensor): The query tensor in compressed form.
        sparsity_layout_q (Tensor): The sparsity layout of ``q``.
        k (BlksprsTensor): The key tensor in compressed form.
        sparsity_layout_k (Tensor): The sparsity layout of ``k``.
        v (BlksprsTensor): The value tensor in compressed form.
        sparsity_layout_v (Tensor): The sparsity layout of ``v``.
        attention_layout (Tensor): The block attention pattern ``(n_batches, seq_q // bs, seq_k // bs)`` indicating
            which Q-K block pairs participate in attention.
        sparsity_block_size (int): The size of the sparsity blocks.
        query_relations (Tensor): The integral relation coordinates of the queries with shape
            ``(n_relations, n_batches, seq_q)``.
        key_relations (Tensor): The integral relation coordinates of the keys with shape
            ``(n_relations, n_batches, seq_k)``.
        relative_embeddings (Tensor): The concatenated relation tables with shape
            ``(n_batches, total_relation_count, head_dim)``.
        relation_bounds (Sequence[Sequence[int]]): The inclusive minimum and maximum of every table, in concatenation
            order.
        key_relations_are_unique (Sequence[bool], optional): Marks the relations whose valid key coordinates are
            pairwise distinct, which permits a faster, non-atomic interior-gradient write. It is a performance
            assertion: callers must only enable a relation when the property holds for every attention batch
            (default ``None``).
        key_relations_repeat (Sequence[bool], optional): Marks the relations whose consecutive keys mostly share their
            coordinates, such as onsets or bar positions. The query-gradient kernel handles the scores and gradients
            of these relations once per query and run of keys with equal coordinates of all marked relations, in the
            key tiles that hold few such runs, and per pair in all others. Relations whose keys change at most keys,
            such as a pitch class, should not be marked, as their runs would prevent the merge of the others. The
            results do not depend on the marks, up to the order of summation (default ``None``, which marks every
            relation without unique keys).
        causal_lengths (Tensor, optional): The valid self-attention length of every flattened attention batch. It
            applies causal and padding masking directly in the kernels and can replace an explicit causal mask
            (default ``None``).
        query_relation_validity (Tensor, optional): A boolean tensor with the shape of ``query_relations``. When
            supplied, a relation contributes only if both positions are valid, which keeps the coordinate domain
            unrestricted instead of reserving a sentinel (default ``None``).
        key_relation_validity (Tensor, optional): A boolean tensor with the shape of ``key_relations``, supplied
            together with ``query_relation_validity`` (default ``None``).
        scale (float, optional): The real-valued attention scale (default ``1/sqrt(head_dim)``).
        attention_mask (BlksprsTensor, optional): A binary mask in compressed form where a non-zero value marks an
            ignored position. The mask does not participate in gradient computation (default ``None``).
        sparsity_layout_mask (Tensor, optional): The sparsity layout for the mask (default ``None``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).
        sparsity_layout_o (Tensor, optional): The output sparsity layout. When omitted, it is derived from the
            structural product of ``attention_layout`` and ``sparsity_layout_v`` (default ``None``).
        score_precalculation (str, optional): The pre-calculation of the projected relative scores, either
            ``"referenced_blocks"`` or ``"complete"`` (default ``"referenced_blocks"``).

    Returns:
        BlksprsTensor: The attention output in compressed form.

    """
    if query_relations.dim() != 3 or key_relations.dim() != 3:
        raise ValueError("query_relations and key_relations must have shape "
                         "(n_relations, attention_batches, sequence_length)")
    if query_relations.dtype == torch.bool or torch.is_floating_point(query_relations):
        raise TypeError("query_relations must use an integral dtype")
    if key_relations.dtype == torch.bool or torch.is_floating_point(key_relations):
        raise TypeError("key_relations must use an integral dtype")
    if score_precalculation not in SCORE_PRECALCULATIONS:
        raise ValueError(f"score_precalculation must be one of {SCORE_PRECALCULATIONS}")

    q, k, v, attention_layout = ensure_contiguous(q, k, v, attention_layout)
    q, k, v, relative_embeddings = cast_for_autocast(q, k, v, relative_embeddings)
    relative_embeddings = ensure_contiguous(relative_embeddings)

    validate_dimensions(q, k, v, attention_layout)
    validate_contiguous(q, k, v, attention_layout, relative_embeddings)
    validate_dtype_float(q, k, v, relative_embeddings)
    validate_device(q, k, v, attention_layout, query_relations, key_relations, relative_embeddings)
    validate_sparsity(sparsity_block_size, (q, sparsity_layout_q), (k, sparsity_layout_k), (v, sparsity_layout_v))
    validate_sparsity_block_size(sparsity_block_size, q, k, v)
    if sparsity_block_size not in (16, 32, 64):
        raise ValueError("Relative Flash Attention supports block sizes 16, 32, and 64")
    validate_sparsity_layout(attention_layout)

    normalised_bounds = _normalise_relation_bounds(relation_bounds)
    n_relations = len(normalised_bounds)
    unique_key_relations = _normalise_unique_key_relations(key_relations_are_unique, n_relations)
    repeated_key_relations = _normalise_repeated_key_relations(key_relations_repeat, unique_key_relations)
    n_batches = sparsity_layout_q.size(0)
    n_seq_blocks_q = sparsity_layout_q.size(1)
    n_head_blocks_qk = sparsity_layout_q.size(2)
    n_seq_blocks_k = sparsity_layout_k.size(1)

    if sparsity_layout_k.size(0) != n_batches or sparsity_layout_k.size(2) != n_head_blocks_qk:
        raise ValueError("K sparsity layout must be compatible with Q")
    if sparsity_layout_v.size(0) != n_batches or sparsity_layout_v.size(1) != n_seq_blocks_k:
        raise ValueError("V sparsity layout must be compatible with K")
    validate_shape(attention_layout, (n_batches, n_seq_blocks_q, n_seq_blocks_k), "attention_layout")

    q_length = n_seq_blocks_q * sparsity_block_size
    k_length = n_seq_blocks_k * sparsity_block_size
    head_dimension = n_head_blocks_qk * sparsity_block_size
    causal_lengths_value = _normalise_causal_lengths(causal_lengths, n_batches, q_length, k_length, q.device)
    validate_shape(query_relations, (n_relations, n_batches, q_length), "query_relations")
    validate_shape(key_relations, (n_relations, n_batches, k_length), "key_relations")
    relation_counts = tuple(relation_max - relation_min + 1 for relation_min, relation_max in normalised_bounds)
    total_relation_count = sum(relation_counts)
    validate_shape(relative_embeddings, (n_batches, total_relation_count, head_dimension), "relative_embeddings")
    if relative_embeddings.dtype != q.dtype:
        raise ValueError("relative_embeddings must use the same dtype as Q")

    if scale is None:
        scale = 1.0 if head_dimension == 0 else 1.0 / math.sqrt(head_dimension)
    elif not isinstance(scale, Real):
        raise TypeError("scale must be a real number or None")
    else:
        scale = float(scale)

    # Resolve optional mask
    if (attention_mask is None) != (sparsity_layout_mask is None):
        raise ValueError("attention_mask and sparsity_layout_mask must be provided together")

    attention_mask_value: Tensor | None = None
    sparsity_layout_mask_value: Tensor | None = None
    if attention_mask is not None and sparsity_layout_mask is not None:
        attention_mask_value = ensure_contiguous(attention_mask)
        sparsity_layout_mask_value = ensure_contiguous(sparsity_layout_mask)
        validate_dimensions(attention_mask_value, sparsity_layout_mask_value)
        validate_contiguous(attention_mask_value, sparsity_layout_mask_value)
        validate_device(attention_mask_value, sparsity_layout_mask_value, q)
        if attention_mask_value.dtype != torch.bool:
            validate_dtype_float(attention_mask_value)
            validate_binary(attention_mask_value)
        validate_sparsity(sparsity_block_size, (attention_mask_value, sparsity_layout_mask_value))
        validate_shape(sparsity_layout_mask_value, (n_batches, n_seq_blocks_q, n_seq_blocks_k),
                       "sparsity_layout_mask")

    # Resolve optional relation validity
    if (query_relation_validity is None) != (key_relation_validity is None):
        raise ValueError("query_relation_validity and key_relation_validity must be provided together")

    query_relation_validity_value: Tensor | None = None
    key_relation_validity_value: Tensor | None = None
    if query_relation_validity is not None and key_relation_validity is not None:
        query_relation_validity_value = ensure_contiguous(query_relation_validity)
        key_relation_validity_value = ensure_contiguous(key_relation_validity)
        validate_device(query_relation_validity_value, key_relation_validity_value, q)
        if query_relation_validity_value.dtype != torch.bool or key_relation_validity_value.dtype != torch.bool:
            raise TypeError("Relation-validity tensors must use bool dtype")
        validate_shape(query_relation_validity_value, (n_relations, n_batches, q_length), "query_relation_validity")
        validate_shape(key_relation_validity_value, (n_relations, n_batches, k_length), "key_relation_validity")

    layout_cache = flash_attention_build_layout_cache(
        attention_layout,
        sparsity_layout_q=sparsity_layout_q,
        sparsity_layout_k=sparsity_layout_k,
        sparsity_layout_v=sparsity_layout_v,
        sparsity_layout_o=sparsity_layout_o,
        sparsity_layout_mask=sparsity_layout_mask_value,
        layout_cache=layout_cache,
        sparsity_block_size=sparsity_block_size,
    )
    relation_metadata, embedding_columns, n_score_columns = _relation_tables(normalised_bounds, sparsity_block_size,
                                                                             q.device, layout_cache)
    coordinates = _cached_relation_coordinates(
        (query_relations, key_relations, query_relation_validity, key_relation_validity, causal_lengths),
        query_relations, key_relations, normalised_bounds, relation_metadata, n_score_columns, attention_layout,
        sparsity_block_size, causal_lengths_value, query_relation_validity_value, key_relation_validity_value,
        layout_cache, score_precalculation)

    unique_key_bits = sum(int(keys_are_unique) << relation
                          for relation, keys_are_unique in enumerate(unique_key_relations))
    relations = _RelativeAttentionArguments(
        query_relations=coordinates.query_relations,
        key_relations=coordinates.key_relations,
        relation_metadata=relation_metadata,
        query_relation_validity=query_relation_validity_value,
        key_relation_validity=key_relation_validity_value,
        query_invalid_blocks=coordinates.query_invalid_blocks,
        key_invalid_blocks=coordinates.key_invalid_blocks,
        n_score_columns=n_score_columns,
        embedding_columns=embedding_columns,
        score_blocks=coordinates.score_blocks,
        n_relations=n_relations,
        unique_key_relations=unique_key_bits,
        query_coordinate_floors=coordinates.query_floors,
        key_coordinate_ceilings=coordinates.key_ceilings,
        merged_relations=sum(int(keys_repeat) << relation
                             for relation, keys_repeat in enumerate(repeated_key_relations)),
    )
    return _flash_attention(
        q, k, v, None, to_dense(q, sparsity_layout_q, sparsity_block_size), relative_embeddings,
        _FlashAttentionArguments.from_layout_cache(
            layout_cache, sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_block_size, scale,
            attention_mask=attention_mask_value,
            has_bias=False,
            causal_lengths=causal_lengths_value,
            relations=relations))


@opaque_public_op
def flash_attention_warm_up(sparsity_block_size: int, head_dimension: int, dtype: torch.dtype,
                            shapes: Sequence[tuple[int, int]], *,
                            value_dimension: int | None = None,
                            relation_bounds: Sequence[Sequence[int]] | None = None,
                            key_relations_are_unique: Sequence[bool] | None = None,
                            key_relations_repeat: Sequence[bool] | None = None,
                            relation_validity: bool = False,
                            causal_lengths: bool = True,
                            attention_mask: bool | torch.dtype = False,
                            attention_bias: bool | torch.dtype = False,
                            score_precalculation: str = "referenced_blocks",
                            backward: bool = True,
                            device: torch.device | str = "cuda") -> None:
    """Compiles and tunes the Flash Attention kernels of a static call configuration ahead of their first use.

    The kernels compile one variant per autotuning configuration and combination of the static properties of a call:
    the dtype, the head dimensions, the block size, the mask and the bias with their dtypes, the causal lengths, the
    relation validity, the number, bounds, and unique keys of the relations, and the score pre-calculation. The
    coordinates of a call only decide at run time which of the compiled loops and launches a tile takes, so that this
    warm-up compiles every variant that calls with these properties use, all configurations of a variant in parallel,
    and later calls compile nothing. The autotuning key also holds power-of-two buckets of the sequence length and of
    the number of tile-schedule entries, whose first calls only benchmark the compiled configurations; the warm-up
    benchmarks them for full causal self-attention over ``shapes``.

    Note:
        Together with a persistent ``TRITON_CACHE_DIR`` and ``TRITON_CACHE_AUTOTUNING=1``, the compiled kernels and the
        autotuning results are reused by later processes, see the README. For distributed training, warm up before the
        first distributed step, so that no rank compiles while the others wait in a collective. The synthetic inputs
        use coordinates that clip, merge, and address through intervals in only some tiles, which does not affect what
        is compiled. Calls with relation coordinates beyond ``2**29`` in magnitude use 64-bit coordinates, and calls
        whose operands exceed ``2**31`` elements use 64-bit indexing, both of which compile separately. The explicit
        mask and the bias compile separately for every dtype, so that calls with Boolean masks, for instance, are
        warmed up with ``attention_mask=torch.bool``.

    Args:
        sparsity_block_size (int): The size of the sparsity blocks.
        head_dimension (int): The head dimension of the queries and keys.
        dtype (torch.dtype): The dtype of the operands.
        shapes (Sequence[tuple[int, int]]): The numbers of attention batches and sequence lengths to tune for. Lengths
            are rounded up to whole sparsity blocks.
        value_dimension (int, optional): The head dimension of the values (default ``head_dimension``).
        relation_bounds (Sequence[Sequence[int]], optional): The inclusive minimum and maximum of every relation of
            :func:`flash_attention_relative_embeddings`, or ``None`` for :func:`blksprs.ops.flash_attention`
            (default ``None``).
        key_relations_are_unique (Sequence[bool], optional): The unique key relations (default ``None``).
        key_relations_repeat (Sequence[bool], optional): The relations with repeated key coordinates
            (default ``None``).
        relation_validity (bool, optional): Whether the calls pass relation-validity tensors (default ``False``).
        causal_lengths (bool, optional): Whether the calls pass causal lengths (default ``True``).
        attention_mask (bool | torch.dtype, optional): Whether the calls pass an explicit attention mask, given by its
            dtype or by ``True`` for the dtype of the operands (default ``False``).
        attention_bias (bool | torch.dtype, optional): Whether the calls of :func:`blksprs.ops.flash_attention` pass an
            attention bias, given by its dtype or by ``True`` for the dtype of the operands (default ``False``).
        score_precalculation (str, optional): The pre-calculation of the projected relative scores
            (default ``"referenced_blocks"``).
        backward (bool, optional): Whether to warm up the backward kernels (default ``True``).
        device (torch.device | str, optional): The device (default ``"cuda"``).

    """
    if not shapes:
        raise ValueError("shapes must hold at least one shape")
    mask_dtype = _warm_up_dtype(attention_mask, dtype)
    bias_dtype = _warm_up_dtype(attention_bias, dtype)
    if bias_dtype is not None and relation_bounds is not None:
        raise ValueError("Flash Attention with learned relative terms takes no attention bias")
    for n_batches, length in shapes:
        _warm_up_call(sparsity_block_size, head_dimension, value_dimension or head_dimension, dtype, n_batches, length,
                      relation_bounds, key_relations_are_unique, key_relations_repeat, relation_validity,
                      causal_lengths, mask_dtype, bias_dtype, score_precalculation, backward, torch.device(device))()
    torch.cuda.synchronize(device)


def _warm_up_dtype(value: bool | torch.dtype, dtype: torch.dtype) -> torch.dtype | None:
    """Returns the dtype of an optional operand of the warm-up calls, or None if the calls do not pass it.

    A true value other than a dtype stands for the dtype of the operands, ``dtype``.

    """
    if isinstance(value, torch.dtype):
        return value
    return dtype if value else None


def _warm_up_call(sparsity_block_size: int, head_dimension: int, value_dimension: int, dtype: torch.dtype,
                  n_batches: int, length: int, relation_bounds: Sequence[Sequence[int]] | None,
                  key_relations_are_unique: Sequence[bool] | None, key_relations_repeat: Sequence[bool] | None,
                  relation_validity: bool, causal_lengths: bool, mask_dtype: torch.dtype | None,
                  bias_dtype: torch.dtype | None, score_precalculation: str, backward: bool, device: torch.device):
    """Returns a function that runs one synthetic call of ``flash_attention_warm_up`` and its backward."""
    n_blocks = -(-length // sparsity_block_size)
    length = n_blocks * sparsity_block_size
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=device))
    layout_qk = torch.ones((n_batches, n_blocks, -(-head_dimension // sparsity_block_size)), dtype=torch.bool,
                           device=device)
    layout_v = torch.ones((n_batches, n_blocks, -(-value_dimension // sparsity_block_size)), dtype=torch.bool,
                          device=device)
    generator = torch.Generator(device=device).manual_seed(0)

    def operand(layout: Tensor) -> Tensor:
        dense = torch.randn((n_batches, length, layout.size(2) * sparsity_block_size), generator=generator,
                            device=device, dtype=dtype)
        return to_sparse(dense, layout, sparsity_block_size)

    q, k, v = operand(layout_qk), operand(layout_qk), operand(layout_v)
    keywords: dict = {"sparsity_layout_o": layout_v}
    if causal_lengths:
        keywords["causal_lengths"] = torch.full((n_batches,), length, dtype=torch.int64, device=device)
    if mask_dtype is not None:
        # The strict upper triangles of the diagonal blocks, a causal mask within the attention layout.
        positions = torch.arange(length, device=device)
        mask = (positions[None, :] > positions[:, None]).expand(n_batches, -1, -1)
        keywords["attention_mask"] = to_sparse(mask.to(mask_dtype), attention_layout, sparsity_block_size)
        keywords["sparsity_layout_mask"] = attention_layout
    if bias_dtype is not None:
        # One block of the bias per block of the attention layout.
        n_bias_blocks = int(attention_layout.sum().item())
        keywords["attention_bias"] = BlksprsTensor.wrap(torch.randn(
            (n_bias_blocks, sparsity_block_size, sparsity_block_size), generator=generator, device=device,
            dtype=bias_dtype))
        keywords["sparsity_layout_bias"] = attention_layout
    relative_embeddings = None
    if relation_bounds is not None:
        bounds = _normalise_relation_bounds(relation_bounds)
        # Positions and onsets every fourth position, so that some tiles clip, merge runs of equal keys, and address
        # the projected scores through intervals.
        positions = torch.arange(length, device=device)
        coordinates = torch.stack([(positions if relation % 2 == 0 else positions // 4)
                                   for relation in range(len(bounds))])[:, None, :].expand(-1, n_batches, -1)
        keywords.update(query_relations=coordinates, key_relations=coordinates, relation_bounds=bounds,
                        key_relations_are_unique=key_relations_are_unique, key_relations_repeat=key_relations_repeat,
                        score_precalculation=score_precalculation)
        if relation_validity:
            validity = torch.ones(coordinates.shape, dtype=torch.bool, device=device)
            keywords.update(query_relation_validity=validity, key_relation_validity=validity)
        n_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
        relative_embeddings = torch.randn((n_batches, n_rows, head_dimension), generator=generator, device=device,
                                          dtype=dtype)

    def attend() -> None:
        inputs = [cast(BlksprsTensor, tensor.detach().requires_grad_(backward)) for tensor in (q, k, v)]
        if relative_embeddings is None:
            output = flash_attention(inputs[0], layout_qk, inputs[1], layout_qk, inputs[2], layout_v,
                                     attention_layout, sparsity_block_size, **keywords)
        else:
            embeddings = relative_embeddings.detach().requires_grad_(backward)
            output = flash_attention_relative_embeddings(inputs[0], layout_qk, inputs[1], layout_qk, inputs[2],
                                                         layout_v, attention_layout, sparsity_block_size,
                                                         relative_embeddings=embeddings, **keywords)
        if backward:
            output.backward(torch.ones_like(output))

    return attend


def _relation_tables(relation_bounds: tuple[tuple[int, int], ...], sparsity_block_size: int, device: torch.device,
                     layout_cache: dict) -> tuple[Tensor, Tensor, int]:
    """Lays out the relation tables in the columns of the projected scores.

    Every table occupies whole sparsity blocks of columns, so that every value block belongs to a single relation.
    Returns one row (minimum, maximum, first column) per relation, the column of every row of the concatenated tables,
    and the number of columns. The tensors are kept in the layout cache, so that later calls need no host-to-device
    copies, which would synchronise.

    """
    cache_key = f"relative_tables_{relation_bounds}_{sparsity_block_size}"
    tables = layout_cache.get(cache_key)
    if tables is None:
        relation_columns: list[int] = []
        embedding_columns: list[int] = []
        n_score_columns = 0
        for relation_min, relation_max in relation_bounds:
            relation_count = relation_max - relation_min + 1
            relation_columns.append(n_score_columns)
            embedding_columns.extend(range(n_score_columns, n_score_columns + relation_count))
            n_score_columns += math.ceil(relation_count / sparsity_block_size) * sparsity_block_size
        relation_metadata = torch.tensor(
            [(relation_min, relation_max, relation_column)
             for (relation_min, relation_max), relation_column in zip(relation_bounds, relation_columns, strict=True)],
            dtype=torch.int64, device=device)
        tables = (relation_metadata, torch.tensor(embedding_columns, dtype=torch.int64, device=device),
                  n_score_columns)
        layout_cache[cache_key] = tables
        finalize_layout_cache(layout_cache)
    return tables


class _RelationCoordinates(NamedTuple):
    """Relation coordinates and the derived metadata of the kernels.

    ``query_floors`` and ``key_ceilings`` find the tiles in which every pair is clipped to the maximum of a relation,
    whose scores the kernels take from the score column of its maximum, see ``_prepare_relation_coordinates``.
    ``score_blocks`` holds the referenced blocks of the projected scores, or None to project the complete tables. With
    validity masks, ``query_invalid_blocks`` and ``key_invalid_blocks`` mark the blocks with positions without valid
    relations, so that the kernels apply the masks only to the tiles that hold them.

    """

    query_relations: Tensor
    key_relations: Tensor
    query_floors: Tensor
    key_ceilings: Tensor
    score_blocks: ReferencedScoreBlocks | None
    query_invalid_blocks: Tensor | None
    key_invalid_blocks: Tensor | None


def _cached_relation_coordinates(inputs: tuple[Tensor | None, ...], query_relations: Tensor, key_relations: Tensor,
                                 relation_bounds: tuple[tuple[int, int], ...], relation_metadata: Tensor,
                                 n_score_columns: int, attention_layout: Tensor, sparsity_block_size: int,
                                 causal_lengths: Tensor | None, query_relation_validity: Tensor | None,
                                 key_relation_validity: Tensor | None, layout_cache: dict,
                                 score_precalculation: str) -> _RelationCoordinates:
    """Prepares the relation coordinates once per layout cache and inputs.

    ``inputs`` are the coordinate, validity, and causal-length tensors of the caller. The prepared coordinates are
    reused while the layout cache holds the same tensors, or views of them, without in-place changes. The layers of a
    model share these tensors and therefore the preparation. The other arguments are those of
    ``_prepare_relation_coordinates``.

    """
    state = (_tensor_states(inputs), relation_bounds, sparsity_block_size, score_precalculation)
    cached = layout_cache.get(_COORDINATE_CACHE_KEY)
    if cached is not None and cached["state"] == state:
        return cached["coordinates"]
    coordinates = _prepare_relation_coordinates(query_relations, key_relations, relation_bounds, relation_metadata,
                                                n_score_columns, attention_layout, sparsity_block_size,
                                                causal_lengths, query_relation_validity, key_relation_validity,
                                                layout_cache, score_precalculation)
    if None not in state[0]:
        # The cache keeps the inputs alive, so that their storage cannot be reused by other tensors while the state
        # refers to it.
        layout_cache[_COORDINATE_CACHE_KEY] = {
            "inputs": inputs,
            "state": state,
            "coordinates": coordinates,
        }
        finalize_layout_cache(layout_cache)
    return coordinates


def _tensor_states(tensors: tuple[Tensor | None, ...]) -> tuple:
    """Identifies tensors by storage, view, and version, or None for inference tensors."""
    states: list = []
    for tensor in tensors:
        if tensor is None:
            states.append(())
        elif torch.is_inference(tensor):
            states.append(None)
        else:
            states.append((tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
                           tensor.device, tensor._version))
    return tuple(states)


def _prepare_relation_coordinates(query_relations: Tensor, key_relations: Tensor,
                                  relation_bounds: tuple[tuple[int, int], ...], relation_metadata: Tensor,
                                  n_score_columns: int, attention_layout: Tensor, sparsity_block_size: int,
                                  causal_lengths: Tensor | None, query_relation_validity: Tensor | None,
                                  key_relation_validity: Tensor | None, layout_cache: dict,
                                  score_precalculation: str) -> _RelationCoordinates:
    """Narrows the relation coordinates and derives the metadata of the kernels from them.

    The kernels compute clipped coordinate differences for every query-key pair in the dtype of the coordinates, and
    32-bit arithmetic is considerably faster. Coordinates and bounds of at most 2**29 in magnitude keep every
    difference within the 32-bit range.

    Under causal layouts, the keys far from a query often reach the maximum of a relation for every pair of a tile. The
    kernels find the leading tiles that clip each relation from the least query coordinate of every block and all
    later blocks and the greatest key coordinate of every block and all earlier blocks, see ``_clipped_tile_count``.

    With ``score_precalculation="referenced_blocks"``, the blocks of the projected scores referenced by permitted pairs
    are found, see ``blksprs.ops.relative_score_blocks``, and stored with the gaps between them if these are few, so
    that the blocks of every relation form one interval of value blocks in every query block, which the kernels
    address through one offset per relation instead of a map of all value blocks. All decisions share one
    synchronisation, which also narrows the coordinates.

    """
    query_relations = ensure_contiguous(query_relations.to(dtype=torch.int64))
    key_relations = ensure_contiguous(key_relations.to(dtype=torch.int64))
    query_floors = _relation_coordinate_bounds(query_relations, query_relation_validity, causal_lengths,
                                               sparsity_block_size, floors=True)
    key_ceilings = _relation_coordinate_bounds(key_relations, key_relation_validity, causal_lengths,
                                               sparsity_block_size, floors=False)
    device = query_relations.device
    n_relations = len(relation_bounds)
    referenced = intervals = complete = None
    value_blocks = _relation_value_blocks(relation_bounds, sparsity_block_size)
    compact_relations = compact_score_relations(value_blocks)
    if score_precalculation == "referenced_blocks":
        referenced = mark_referenced_score_blocks(
            query_relations, key_relations, relation_metadata, n_score_columns // sparsity_block_size,
            attention_layout, layout_cache[f"query_tile_schedule_{_schedule_group_blocks(sparsity_block_size)}"],
            sparsity_block_size, causal_lengths, query_relation_validity, key_relation_validity)
        intervals, complete = referenced_value_intervals(referenced, value_blocks, compact_relations)

    summary = [
        (torch.zeros(2, dtype=torch.int64, device=device) if referenced is None or complete is None else
         torch.stack([torch.count_nonzero(referenced), complete.to(torch.int64)])),
    ]
    if query_relations.numel() > 0 and key_relations.numel() > 0:
        summary.append(torch.stack([*torch.aminmax(query_relations), *torch.aminmax(key_relations)]))
    values = torch.cat(summary).tolist()
    n_blocks, complete_intervals = values[:2]
    extremes = values[2:] or [0]

    narrow_limit = 2 ** 29
    if (min(extremes) >= -narrow_limit and max(extremes) <= narrow_limit
            and all(abs(bound) <= narrow_limit for bounds in relation_bounds for bound in bounds)):
        query_relations = ensure_contiguous(query_relations.to(dtype=torch.int32))
        key_relations = ensure_contiguous(key_relations.to(dtype=torch.int32))
    # Larger coordinates could overflow the differences of the bounds, which are replaced by bounds whose differences
    # stay below the minimum of every relation, so that no tile clips any relation.
    if min(extremes) < -_COORDINATE_LIMIT or max(extremes) > _COORDINATE_LIMIT:
        query_floors = (relation_metadata[:, 0] - 1).view(n_relations, 1, 1).expand_as(query_floors).contiguous()
        key_ceilings = torch.zeros_like(key_ceilings)
    score_blocks = None
    if referenced is not None and intervals is not None:
        score_blocks = compress_referenced_score_blocks(referenced, n_blocks, intervals, compact_relations,
                                                        bool(complete_intervals))
    invalid_blocks = (_invalid_blocks(query_relation_validity, causal_lengths, sparsity_block_size),
                      _invalid_blocks(key_relation_validity, causal_lengths, sparsity_block_size))
    return _RelationCoordinates(query_relations, key_relations, query_floors, key_ceilings, score_blocks,
                                *invalid_blocks)


def _relation_value_blocks(relation_bounds: tuple[tuple[int, int], ...],
                           sparsity_block_size: int) -> list[tuple[int, int]]:
    """Returns the first and the end value block of every relation table.

    The tables are padded to whole blocks, see ``_relation_tables``.

    """
    value_blocks = []
    first_block = 0
    for relation_min, relation_max in relation_bounds:
        end_block = first_block + math.ceil((relation_max - relation_min + 1) / sparsity_block_size)
        value_blocks.append((first_block, end_block))
        first_block = end_block
    return value_blocks


def _invalid_blocks(validity: Tensor | None, causal_lengths: Tensor | None,
                    sparsity_block_size: int) -> Tensor | None:
    """Marks the blocks with positions without a valid relation.

    Positions beyond the causal length contribute no scores and are ignored.

    """
    if validity is None:
        return None
    invalid = ~validity.all(dim=0)
    if causal_lengths is not None:
        invalid &= torch.arange(invalid.size(1), device=invalid.device) < causal_lengths[:, None]
    return invalid.view(invalid.size(0), -1, sparsity_block_size).any(dim=2).to(torch.int8)


def _relation_coordinate_bounds(coordinates: Tensor, validity: Tensor | None, causal_lengths: Tensor | None,
                                sparsity_block_size: int, *, floors: bool) -> Tensor:
    """Bounds the coordinates of every block and its successors or predecessors.

    With ``floors``, returns the least coordinate of every block and all later blocks, and otherwise the greatest
    coordinate of every block and all earlier blocks. Positions beyond the causal length and positions without a valid
    relation contribute no scores and are ignored.

    """
    n_relations, n_batches, length = coordinates.shape
    ignored = None if validity is None else ~validity
    if causal_lengths is not None:
        beyond = torch.arange(length, device=coordinates.device) >= causal_lengths[:, None]
        ignored = beyond if ignored is None else ignored | beyond
    if ignored is not None:
        coordinates = coordinates.masked_fill(ignored, 2 * _COORDINATE_LIMIT if floors else -2 * _COORDINATE_LIMIT)
    blocks = coordinates.view(n_relations, n_batches, length // sparsity_block_size, sparsity_block_size)
    if floors:
        return blocks.amin(dim=-1).flip(-1).cummin(dim=-1).values.flip(-1).contiguous()
    return blocks.amax(dim=-1).cummax(dim=-1).values


def _normalise_relation_bounds(relation_bounds: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Validates the relation bounds and returns them as a tuple of integer pairs."""
    if not isinstance(relation_bounds, Sequence):
        raise TypeError("relation_bounds must be a sequence")
    if not 1 <= len(relation_bounds) <= MAX_RELATIONS:
        raise ValueError(f"relation_bounds must contain between 1 and {MAX_RELATIONS} ranges")

    result: list[tuple[int, int]] = []
    for bounds in relation_bounds:
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError("Every relation bound must contain [minimum, maximum]")
        relation_min, relation_max = bounds
        if not isinstance(relation_min, Integral) or isinstance(relation_min, bool):
            raise TypeError("Relation bounds must be integers")
        if not isinstance(relation_max, Integral) or isinstance(relation_max, bool):
            raise TypeError("Relation bounds must be integers")
        if relation_min > relation_max:
            raise ValueError("A relation minimum must not exceed its maximum")
        result.append((int(relation_min), int(relation_max)))
    return tuple(result)


def _normalise_repeated_key_relations(values: Sequence[bool] | None,
                                      unique_key_relations: tuple[bool, ...]) -> tuple[bool, ...]:
    """Validates the flags of the relations with repeated key coordinates and returns them as a tuple."""
    if values is None:
        return tuple(not keys_are_unique for keys_are_unique in unique_key_relations)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("key_relations_repeat must be a sequence of bool values")
    if len(values) != len(unique_key_relations):
        raise ValueError("key_relations_repeat must contain one value per relation")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("key_relations_repeat must contain only bool values")
    if any(keys_repeat and keys_are_unique for keys_repeat, keys_are_unique in zip(values, unique_key_relations)):
        raise ValueError("Relations with unique key coordinates cannot repeat them")
    return tuple(values)


def _normalise_unique_key_relations(values: Sequence[bool] | None, n_relations: int) -> tuple[bool, ...]:
    """Validates the flags of the relations with unique key coordinates and returns them as a tuple."""
    if values is None:
        return (False,) * n_relations
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("key_relations_are_unique must be a sequence of bool values")
    if len(values) != n_relations:
        raise ValueError("key_relations_are_unique must contain one value per relation")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("key_relations_are_unique must contain only bool values")
    return tuple(values)

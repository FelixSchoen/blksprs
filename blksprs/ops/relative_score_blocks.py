from typing import NamedTuple

import torch
import triton
from torch import Tensor
from triton import language as tl

# Schedule entries visited by one program of the marking kernel.
_MARKING_CHUNK = tl.constexpr(16)
# Bytes of one row of a head-dimension chunk of the projection kernels. Wider head dimensions are split into chunks so
# that the operand tiles fit into shared memory.
_MAX_HEAD_TILE_BYTES = 512
# The gaps between the referenced blocks of a call are stored as well if they hold at most one in this many of its
# referenced blocks, see ``referenced_value_intervals``.
_GAP_BUDGET = 64
# Relations whose tables span at most this many value blocks always store the gaps between their referenced blocks.
_COMPACT_VALUE_BLOCKS = 8


class ReferencedScoreBlocks(NamedTuple):
    """Layout of the referenced blocks of the projected relative scores in compressed sparse row and column form.

    For every relation ``g``, ``flash_attention_relative_embeddings`` adds the score
    ``q @ R_g[clip(c_g(q) - c_g(k), min_g, max_g) - min_g]`` of query ``q`` and key ``k``. The Flash Attention kernels
    gather these scores from ``S_pre = Q E^T``, whose rows are queries and whose columns are the relation values of all
    relations. Every table is padded to whole blocks of ``sparsity_block_size`` values, so that the value blocks of the
    relations are disjoint. A block of ``S_pre`` is referenced if the clipped relation value of some permitted
    query-key pair lies in it. Permitted pairs lie in the attention layout and within the causal lengths, and their
    positions are valid for the relation.

    The functions of this module build the layout of the referenced blocks exactly, store them together with the
    gaps between them where these are few, see ``referenced_value_intervals``, compute only the stored blocks, and
    compute the gradients of the queries and the tables from the gradients of these blocks. The blocks are stored in
    row-major order of batch, query block, and value block.

    """

    # Every row of (batch, query block) holds the index of every value block in the compressed scores, or -1, followed
    # by the index of the first stored block minus its value block for every relation, see
    # ``referenced_value_intervals``.
    addressing: Tensor
    # The index of every block of (batch, query block, value block) in the compressed scores, or -1, a view of
    # ``addressing``.
    lookup: Tensor
    # Bit mask of the relations whose stored blocks form one interval in every row, so that the index of every stored
    # block is the value block plus the entry of its row and relation. These are the relations with few value blocks,
    # see ``compact_score_relations``, which only depend on the relation bounds.
    interval_relations: int
    # Whether the stored blocks of every relation form one interval in every row, so that all relations are addressed
    # through intervals, see ``referenced_value_intervals``.
    complete_intervals: bool
    # The first block of every row of (batch, query block), followed by the number of blocks.
    row_offsets: Tensor
    # The value block of every block.
    block_columns: Tensor
    # The first entry of every column of (batch, value block) in ``column_blocks``, followed by the number of blocks.
    column_offsets: Tensor
    # The blocks and their query blocks in column-major order.
    column_blocks: Tensor
    column_rows: Tensor
    n_blocks: int


def mark_referenced_score_blocks(query_relations: Tensor, key_relations: Tensor, relation_metadata: Tensor,
                                 n_value_blocks: int, attention_layout: Tensor, query_schedule: dict,
                                 sparsity_block_size: int, causal_lengths: Tensor | None,
                                 query_relation_validity: Tensor | None,
                                 key_relation_validity: Tensor | None) -> Tensor:
    """Marks the referenced blocks of the projected relative scores.

    ``relation_metadata`` holds the minimum, maximum, and first table column of every relation, and
    ``query_schedule`` is the query tile schedule of the layout cache, which lists the key blocks attended by every
    query block. Returns a tensor of shape ``(batch, query blocks, value blocks)`` that is non-zero for the
    referenced blocks.

    Consecutive batches with equal coordinates, validity, causal length, and attention layout, such as the heads of
    one sequence, reference the same blocks, so that only the first batch of every such run is marked.

    """
    n_relations, n_batches, q_length = query_relations.shape
    n_seq_blocks_q = q_length // sparsity_block_size
    n_seq_blocks_k = key_relations.size(2) // sparsity_block_size
    referenced = torch.zeros((n_batches, n_seq_blocks_q, n_value_blocks), dtype=torch.int8,
                             device=query_relations.device)
    if referenced.numel() == 0:
        return referenced
    representatives = _repeated_batch_representatives(
        attention_layout, query_relations, key_relations, query_relation_validity, key_relation_validity,
        causal_lengths)
    empty_boolean = torch.empty(0, dtype=torch.bool, device=query_relations.device)
    referenced_score_blocks_kernel[(n_batches * n_seq_blocks_q, triton.cdiv(n_seq_blocks_k, _MARKING_CHUNK.value))](
        referenced, representatives, query_relations, key_relations, relation_metadata,
        empty_boolean if query_relation_validity is None else query_relation_validity,
        empty_boolean if key_relation_validity is None else key_relation_validity,
        torch.empty(0, dtype=torch.int64, device=query_relations.device) if causal_lengths is None else causal_lengths,
        query_schedule["offsets"], query_schedule["blocks"], query_schedule["bits"],
        n_batches, n_seq_blocks_q, n_seq_blocks_k, n_value_blocks,
        sparsity_block_size,
        SCHEDULE_BLOCKS=query_schedule["group_blocks"],
        N_RELATIONS=n_relations,
        HAS_RELATION_VALIDITY=query_relation_validity is not None,
        HAS_CAUSAL=causal_lengths is not None)
    return referenced.index_select(0, representatives)


def _repeated_batch_representatives(attention_layout: Tensor, query_relations: Tensor, key_relations: Tensor,
                                    query_relation_validity: Tensor | None, key_relation_validity: Tensor | None,
                                    causal_lengths: Tensor | None) -> Tensor:
    """Returns, for every batch, the first of the consecutive batches whose inputs equal its own."""
    n_batches = attention_layout.size(0)
    repeated = (attention_layout[1:] == attention_layout[:-1]).flatten(1).all(dim=1)
    for per_relation in (query_relations, key_relations, query_relation_validity, key_relation_validity):
        if per_relation is not None:
            repeated &= (per_relation[:, 1:] == per_relation[:, :-1]).transpose(0, 1).flatten(1).all(dim=1)
    if causal_lengths is not None:
        repeated &= causal_lengths[1:] == causal_lengths[:-1]
    batches = torch.arange(n_batches, device=attention_layout.device)
    starts = torch.ones(n_batches, dtype=torch.bool, device=attention_layout.device)
    starts[1:] = ~repeated
    return torch.cummax(torch.where(starts, batches, 0), dim=0).values


def referenced_value_intervals(referenced: Tensor, relation_value_blocks: list[tuple[int, int]],
                               compact_relations: int) -> tuple[Tensor, Tensor]:
    """Finds the interval of value blocks that spans the referenced blocks of every relation in every row.

    ``relation_value_blocks`` holds the first and the end value block of every relation. The gaps of unreferenced
    blocks within the intervals of the relations of the bit mask ``compact_relations``, whose tables span few value
    blocks, see ``compact_score_relations``, are marked as referenced, and thus stored. The gaps of all other relations
    are stored as well if they hold at most ``1 / _GAP_BUDGET`` of the referenced blocks, so that the stored blocks of
    every relation form one interval in every row, which the Flash Attention kernels address with one offset per query
    block, see ``_score_offsets``. ``referenced`` is updated in place. Returns the first and the end value block of
    every row of (batch, query block) and relation, zero for rows without referenced blocks, with shape
    ``(batch, query blocks, relations, 2)``, and whether the stored blocks of all relations form one interval in every
    row, as a tensor.

    """
    marked = referenced != 0
    n_value_blocks = marked.size(2)
    value_blocks = torch.arange(n_value_blocks, device=referenced.device)
    spans = []
    gaps = torch.zeros((), dtype=torch.int64, device=referenced.device)
    for relation, (first_block, end_block) in enumerate(relation_value_blocks):
        relation_marked = marked[:, :, first_block:end_block]
        relation_blocks = value_blocks[first_block:end_block]
        counts = relation_marked.sum(dim=2)
        empty = counts == 0
        first = torch.where(empty, 0, torch.where(relation_marked, relation_blocks, n_value_blocks).amin(dim=2))
        end = torch.where(empty, 0, torch.where(relation_marked, relation_blocks, -1).amax(dim=2) + 1)
        spans.append((first, end))
        if not compact_relations >> relation & 1:
            gaps = gaps + (end - first - counts).sum()
    # The decision stays on the device and is read together with the number of blocks, in the one synchronisation of
    # a call that sizes the projected scores.
    complete = gaps * _GAP_BUDGET <= marked.sum()
    for relation, ((first_block, end_block), (first, end)) in enumerate(zip(relation_value_blocks, spans)):
        fill = complete | bool(compact_relations >> relation & 1)
        relation_blocks = value_blocks[first_block:end_block]
        within = (relation_blocks >= first[..., None]) & (relation_blocks < end[..., None]) & fill
        referenced[:, :, first_block:end_block] |= within.to(referenced.dtype)
    intervals = torch.stack([torch.stack([first, end], dim=-1) for first, end in spans], dim=2)
    return intervals, complete


def compact_score_relations(relation_value_blocks: list[tuple[int, int]]) -> int:
    """Returns the bit mask of the relations whose tables span at most ``_COMPACT_VALUE_BLOCKS`` value blocks.

    All gaps between the referenced blocks of these relations are stored, see ``referenced_value_intervals``, so that
    the Flash Attention kernels always address them through intervals. A row holds at most this number of their blocks.

    """
    return sum(1 << relation for relation, (first_block, end_block) in enumerate(relation_value_blocks)
               if end_block - first_block <= _COMPACT_VALUE_BLOCKS)


def compress_referenced_score_blocks(referenced: Tensor, n_blocks: int, intervals: Tensor, interval_relations: int,
                                     complete_intervals: bool) -> ReferencedScoreBlocks:
    """Builds the layout of the referenced blocks marked by ``mark_referenced_score_blocks``.

    ``n_blocks`` must be the number of marked blocks, ``intervals`` and ``complete_intervals`` the intervals of value
    blocks and whether they are all complete, see ``referenced_value_intervals``, and ``interval_relations`` the bit
    mask of the compact relations, so that no synchronisation is needed.

    """
    n_batches, n_seq_blocks_q, n_value_blocks = referenced.shape
    device = referenced.device
    marked = referenced != 0
    flat = marked.flatten()
    block_indices = torch.cumsum(flat, dim=0, dtype=torch.int64) - 1
    lookup = torch.where(flat, block_indices, -1).view(referenced.shape)
    # The blocks of a row are stored in the order of their value blocks, so that the blocks of an interval without
    # unreferenced blocks follow its first one.
    first_blocks = lookup.gather(2, intervals[..., 0])
    bases = torch.where(intervals[..., 1] > intervals[..., 0], first_blocks - intervals[..., 0], 0)
    addressing = torch.cat([lookup, bases], dim=2).to(torch.int32)
    # Unmarked entries scatter into a spare last element.
    targets = torch.where(flat, block_indices, n_blocks)

    row_offsets = torch.zeros(n_batches * n_seq_blocks_q + 1, dtype=torch.int64, device=device)
    row_offsets[1:] = torch.cumsum(marked.sum(dim=2).flatten(), dim=0)
    value_blocks = torch.arange(n_value_blocks, dtype=torch.int32, device=device).repeat(n_batches * n_seq_blocks_q)
    block_columns = torch.empty(n_blocks + 1, dtype=torch.int32, device=device).scatter_(0, targets, value_blocks)

    column_offsets = torch.zeros(n_batches * n_value_blocks + 1, dtype=torch.int64, device=device)
    column_offsets[1:] = torch.cumsum(marked.sum(dim=1).flatten(), dim=0)
    column_ranks = torch.cumsum(marked, dim=1, dtype=torch.int64) - 1
    column_starts = column_offsets[:-1].view(n_batches, 1, n_value_blocks)
    column_targets = torch.where(marked, column_starts + column_ranks, n_blocks).flatten()
    column_blocks = torch.empty(n_blocks + 1, dtype=torch.int64, device=device).scatter_(
        0, column_targets, block_indices)
    query_blocks = torch.arange(n_seq_blocks_q, dtype=torch.int32, device=device).repeat_interleave(
        n_value_blocks).repeat(n_batches)
    column_rows = torch.empty(n_blocks + 1, dtype=torch.int32, device=device).scatter_(
        0, column_targets, query_blocks)

    return ReferencedScoreBlocks(addressing=addressing, lookup=addressing[:, :, :n_value_blocks],
                                 interval_relations=interval_relations, complete_intervals=complete_intervals,
                                 row_offsets=row_offsets,
                                 block_columns=block_columns[:n_blocks], column_offsets=column_offsets,
                                 column_blocks=column_blocks[:n_blocks], column_rows=column_rows[:n_blocks],
                                 n_blocks=n_blocks)


def referenced_score_elements(blocks: ReferencedScoreBlocks, sparsity_block_size: int) -> int:
    """Returns the number of elements of the allocation of the referenced blocks, including their guard blocks."""
    return (blocks.n_blocks + 2 * blocks.lookup.size(2)) * sparsity_block_size * sparsity_block_size


def allocate_referenced_scores(blocks: ReferencedScoreBlocks, sparsity_block_size: int, dtype: torch.dtype,
                               device: torch.device, zero: bool) -> Tensor:
    """Allocates the referenced blocks of the projected scores or their gradients, between zero guard blocks.

    The Flash kernels address the blocks of relations whose referenced blocks form one interval per row without
    checking the interval, see ``_score_offsets``. Pairs outside the permitted ones, whose scores are masked and whose
    gradients are zero, can then address up to one row of value blocks before or after the referenced blocks, which
    the guard blocks hold. Returns the view of the referenced blocks, which are zero with ``zero``. Without referenced
    blocks, the view holds the first trailing guard block, as the data pointer of an empty view is null. The size of
    the allocation is ``referenced_score_elements``.

    """
    n_guard_blocks = blocks.lookup.size(2)
    shape = (blocks.n_blocks + 2 * n_guard_blocks, sparsity_block_size, sparsity_block_size)
    view = slice(n_guard_blocks, n_guard_blocks + max(blocks.n_blocks, 1))
    if zero:
        return torch.zeros(shape, dtype=dtype, device=device)[view]
    scores = torch.empty(shape, dtype=dtype, device=device)
    scores[:n_guard_blocks].zero_()
    scores[n_guard_blocks + blocks.n_blocks:].zero_()
    return scores[view]


def project_referenced_scores(query: Tensor, embeddings: Tensor, blocks: ReferencedScoreBlocks,
                              sparsity_block_size: int) -> Tensor:
    """Computes the referenced blocks of ``query @ embeddings.T`` for dense queries and padded tables."""
    n_batches, q_length, head_dimension = query.shape
    scores = allocate_referenced_scores(blocks, sparsity_block_size, query.dtype, query.device, zero=False)
    if blocks.n_blocks > 0:
        head_tile, n_chunks = _head_tile(head_dimension, query.element_size())
        referenced_score_projection_kernel[(n_batches * (q_length // sparsity_block_size),)](
            query, embeddings, scores, blocks.row_offsets, blocks.block_columns,
            n_batches, q_length, embeddings.size(1), head_dimension,
            sparsity_block_size, HEAD_TILE=head_tile, N_CHUNKS=n_chunks, DOT_PRECISION=_dot_precision(query),
            num_warps=4, num_stages=1)
    return scores


def referenced_score_gradients(query: Tensor, embeddings: Tensor, dscores: Tensor, blocks: ReferencedScoreBlocks,
                               sparsity_block_size: int) -> tuple[Tensor, Tensor]:
    """Returns the gradients of dense queries and padded tables from the float32 gradients of the referenced blocks.

    The gradients of the blocks are rounded to the dtype of the operands before the products, as for the complete
    projection.

    """
    n_batches, q_length, head_dimension = query.shape
    n_score_columns = embeddings.size(1)
    dquery = torch.empty_like(query)
    dembeddings = torch.empty_like(embeddings)
    head_tile, n_chunks = _head_tile(head_dimension, query.element_size())
    if dquery.numel() > 0:
        referenced_query_gradient_kernel[(n_batches * (q_length // sparsity_block_size), n_chunks)](
            dscores, embeddings, dquery, blocks.row_offsets, blocks.block_columns,
            n_batches, q_length, n_score_columns, head_dimension,
            sparsity_block_size, HEAD_TILE=head_tile, DOT_PRECISION=_dot_precision(query), num_warps=4,
            num_stages=1)
    if dembeddings.numel() > 0:
        referenced_embedding_gradient_kernel[(n_batches * (n_score_columns // sparsity_block_size), n_chunks)](
            dscores, query, dembeddings, blocks.column_offsets, blocks.column_blocks, blocks.column_rows,
            n_batches, q_length, n_score_columns, head_dimension,
            sparsity_block_size, HEAD_TILE=head_tile, DOT_PRECISION=_dot_precision(query), num_warps=4,
            num_stages=1)
    return dquery, dembeddings


def _head_tile(head_dimension: int, element_size: int) -> tuple[int, int]:
    """Returns the width of the head-dimension chunks of the projection kernels and the number of chunks."""
    head_tile = min(max(16, triton.next_power_of_2(head_dimension)), _MAX_HEAD_TILE_BYTES // element_size)
    return head_tile, max(1, triton.cdiv(head_dimension, head_tile))


def _dot_precision(operand: Tensor) -> str:
    """Multiplies float32 operands exactly, as the matrix multiplications of the complete projection."""
    return "ieee" if operand.dtype == torch.float32 else "tf32"


@triton.jit(do_not_specialize=["n_batches", "n_seq_blocks_q", "n_seq_blocks_k"])
def referenced_score_blocks_kernel(referenced, representatives, query_relations, key_relations, relation_metadata,
                                   query_relation_validity, key_relation_validity, causal_lengths,
                                   schedule_offsets, schedule_blocks, schedule_bits,
                                   n_batches, n_seq_blocks_q, n_seq_blocks_k, n_value_blocks,
                                   sparsity_block_size: tl.constexpr, SCHEDULE_BLOCKS: tl.constexpr,
                                   N_RELATIONS: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                                   HAS_CAUSAL: tl.constexpr) -> None:
    """Marks the value blocks referenced by the permitted pairs of one query block."""
    # Batches and query blocks share the first grid axis, whose size is not limited to 65,535.
    program = tl.program_id(axis=0).to(tl.int64)
    batch = program % n_batches
    query_block = program // n_batches
    q_length = n_seq_blocks_q * sparsity_block_size
    k_length = n_seq_blocks_k * sparsity_block_size
    schedule_row = batch * tl.cdiv(n_seq_blocks_q, SCHEDULE_BLOCKS) + query_block // SCHEDULE_BLOCKS
    schedule_bit = query_block % SCHEDULE_BLOCKS
    # Every program visits one chunk of the key blocks of its query block, which keeps the loops short. Repeated
    # batches are copied from their representative instead.
    start = tl.load(schedule_offsets + schedule_row) + tl.program_id(axis=1) * _MARKING_CHUNK
    end = tl.minimum(tl.load(schedule_offsets + schedule_row + 1), start + _MARKING_CHUNK)
    end = tl.where(tl.load(representatives + batch) == batch, end, start)
    query_positions = query_block * sparsity_block_size + tl.arange(0, sparsity_block_size)
    referenced_row = referenced + (batch * n_seq_blocks_q + query_block) * n_value_blocks
    marks = tl.full((sparsity_block_size, sparsity_block_size), 1, dtype=tl.int8)

    for entry in range(start, end):
        if (tl.load(schedule_bits + entry) >> schedule_bit) & 1:
            key_positions = tl.load(schedule_blocks + entry).to(tl.int64) * sparsity_block_size + tl.arange(
                0, sparsity_block_size)
            permitted = (query_positions[:, None] >= 0) & (key_positions[None, :] >= 0)
            if HAS_CAUSAL:
                sequence_length = tl.load(causal_lengths + batch)
                permitted = ((key_positions[None, :] <= query_positions[:, None])
                             & (query_positions[:, None] < sequence_length))
            for relation in tl.static_range(N_RELATIONS):
                relation_min = tl.load(relation_metadata + relation * 3)
                relation_span = tl.load(relation_metadata + relation * 3 + 1) - relation_min
                relation_offset = tl.load(relation_metadata + relation * 3 + 2)
                query_coordinates = tl.load(query_relations + (relation * n_batches + batch) * q_length
                                            + query_positions).to(tl.int64)
                key_coordinates = tl.load(key_relations + (relation * n_batches + batch) * k_length
                                          + key_positions).to(tl.int64)
                rows = tl.minimum(tl.maximum((query_coordinates - relation_min)[:, None] - key_coordinates[None, :],
                                             0), relation_span)
                relation_permitted = permitted
                if HAS_RELATION_VALIDITY:
                    query_validity = tl.load(query_relation_validity + (relation * n_batches + batch) * q_length
                                             + query_positions)
                    key_validity = tl.load(key_relation_validity + (relation * n_batches + batch) * k_length
                                           + key_positions)
                    relation_permitted = (relation_permitted & (query_validity != 0)[:, None]
                                          & (key_validity != 0)[None, :])
                tl.store(referenced_row + (relation_offset + rows) // sparsity_block_size, marks,
                         mask=relation_permitted)


@triton.jit(do_not_specialize=["n_batches"])
def referenced_score_projection_kernel(query, embeddings, scores, row_offsets, block_columns,
                                       n_batches, q_length, n_score_columns, head_dimension,
                                       sparsity_block_size: tl.constexpr, HEAD_TILE: tl.constexpr,
                                       N_CHUNKS: tl.constexpr, DOT_PRECISION: tl.constexpr) -> None:
    """Computes the referenced blocks of one query block, see ``project_referenced_scores``.

    Head dimensions of several chunks are accumulated chunk by chunk.

    """
    # Batches and query blocks share the first grid axis, whose size is not limited to 65,535.
    program = tl.program_id(axis=0).to(tl.int64)
    batch = program % n_batches
    query_block = program // n_batches
    row = batch * (q_length // sparsity_block_size) + query_block
    start = tl.load(row_offsets + row)
    end = tl.load(row_offsets + row + 1)

    block_rows = tl.arange(0, sparsity_block_size)
    query_rows = (batch * q_length + query_block * sparsity_block_size + block_rows) * head_dimension
    columns = tl.arange(0, HEAD_TILE)
    if N_CHUNKS == 1:
        column_mask = (columns < head_dimension)[None, :]
        query_tile = tl.load(query + query_rows[:, None] + columns[None, :], mask=column_mask, other=0)
    for block in range(start, end):
        value_block = tl.load(block_columns + block).to(tl.int64)
        embedding_rows = (batch * n_score_columns + value_block * sparsity_block_size + block_rows) * head_dimension
        if N_CHUNKS == 1:
            embedding_tile = tl.load(embeddings + embedding_rows[:, None] + columns[None, :], mask=column_mask,
                                     other=0)
            block_scores = tl.dot(query_tile, tl.trans(embedding_tile), input_precision=DOT_PRECISION)
        else:
            block_scores = tl.zeros((sparsity_block_size, sparsity_block_size), dtype=tl.float32)
            for chunk in range(0, N_CHUNKS):
                chunk_columns = chunk * HEAD_TILE + columns
                chunk_mask = (chunk_columns < head_dimension)[None, :]
                query_chunk = tl.load(query + query_rows[:, None] + chunk_columns[None, :], mask=chunk_mask, other=0)
                embedding_chunk = tl.load(embeddings + embedding_rows[:, None] + chunk_columns[None, :],
                                          mask=chunk_mask, other=0)
                block_scores = tl.dot(query_chunk, tl.trans(embedding_chunk), block_scores,
                                      input_precision=DOT_PRECISION)
        tl.store(scores + block * (sparsity_block_size * sparsity_block_size)
                 + block_rows[:, None] * sparsity_block_size + block_rows[None, :],
                 block_scores.to(scores.dtype.element_ty))


@triton.jit(do_not_specialize=["n_batches"])
def referenced_query_gradient_kernel(dscores, embeddings, dquery, row_offsets, block_columns,
                                     n_batches, q_length, n_score_columns, head_dimension,
                                     sparsity_block_size: tl.constexpr, HEAD_TILE: tl.constexpr,
                                     DOT_PRECISION: tl.constexpr) -> None:
    """Computes one head-dimension chunk of the query gradient of a query block, see ``referenced_score_gradients``."""
    # Batches and query blocks share the first grid axis, whose size is not limited to 65,535.
    program = tl.program_id(axis=0).to(tl.int64)
    batch = program % n_batches
    query_block = program // n_batches
    row = batch * (q_length // sparsity_block_size) + query_block
    start = tl.load(row_offsets + row)
    end = tl.load(row_offsets + row + 1)

    block_rows = tl.arange(0, sparsity_block_size)
    columns = tl.program_id(axis=1) * HEAD_TILE + tl.arange(0, HEAD_TILE)
    column_mask = (columns < head_dimension)[None, :]
    accumulator = tl.zeros((sparsity_block_size, HEAD_TILE), dtype=tl.float32)
    for block in range(start, end):
        value_block = tl.load(block_columns + block).to(tl.int64)
        block_gradients = tl.load(dscores + block * (sparsity_block_size * sparsity_block_size)
                                  + block_rows[:, None] * sparsity_block_size + block_rows[None, :])
        embedding_tile = tl.load(embeddings + ((batch * n_score_columns + value_block * sparsity_block_size
                                                + block_rows) * head_dimension)[:, None] + columns[None, :],
                                 mask=column_mask, other=0)
        accumulator += tl.dot(block_gradients.to(embedding_tile.dtype), embedding_tile,
                              input_precision=DOT_PRECISION)
    tl.store(dquery + ((batch * q_length + query_block * sparsity_block_size + block_rows)
                       * head_dimension)[:, None] + columns[None, :],
             accumulator.to(dquery.dtype.element_ty), mask=column_mask)


@triton.jit(do_not_specialize=["n_batches"])
def referenced_embedding_gradient_kernel(dscores, query, dembeddings, column_offsets, column_blocks, column_rows,
                                         n_batches, q_length, n_score_columns, head_dimension,
                                         sparsity_block_size: tl.constexpr, HEAD_TILE: tl.constexpr,
                                         DOT_PRECISION: tl.constexpr) -> None:
    """Computes one head-dimension chunk of the table gradient of a value block, see ``referenced_score_gradients``."""
    # Batches and value blocks share the first grid axis, whose size is not limited to 65,535.
    program = tl.program_id(axis=0).to(tl.int64)
    batch = program % n_batches
    value_block = program // n_batches
    column = batch * (n_score_columns // sparsity_block_size) + value_block
    start = tl.load(column_offsets + column)
    end = tl.load(column_offsets + column + 1)

    block_rows = tl.arange(0, sparsity_block_size)
    columns = tl.program_id(axis=1) * HEAD_TILE + tl.arange(0, HEAD_TILE)
    column_mask = (columns < head_dimension)[None, :]
    accumulator = tl.zeros((sparsity_block_size, HEAD_TILE), dtype=tl.float32)
    for entry in range(start, end):
        block = tl.load(column_blocks + entry)
        query_block = tl.load(column_rows + entry).to(tl.int64)
        block_gradients = tl.load(dscores + block * (sparsity_block_size * sparsity_block_size)
                                  + block_rows[:, None] * sparsity_block_size + block_rows[None, :])
        query_tile = tl.load(query + ((batch * q_length + query_block * sparsity_block_size + block_rows)
                                      * head_dimension)[:, None] + columns[None, :], mask=column_mask, other=0)
        accumulator += tl.dot(tl.trans(block_gradients.to(query_tile.dtype)), query_tile,
                              input_precision=DOT_PRECISION)
    tl.store(dembeddings + ((batch * n_score_columns + value_block * sparsity_block_size + block_rows)
                            * head_dimension)[:, None] + columns[None, :],
             accumulator.to(dembeddings.dtype.element_ty), mask=column_mask)

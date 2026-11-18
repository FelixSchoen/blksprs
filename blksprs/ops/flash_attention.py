import math
import operator
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real

import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable
from triton import language as tl
from triton.runtime.errors import OutOfResources

from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul
from blksprs.ops.relative_score_blocks import ReferencedScoreBlocks, allocate_referenced_scores, \
    project_referenced_scores, referenced_score_elements, referenced_score_gradients
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs, \
    prune_autotune_configs_spanning, SUPPORTED_AUTOTUNE_MODES
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import as_base_tensor, build_packed_indices, can_use_int32_indexing, cast_for_autocast, \
    prepare_layout_cache, finalize_layout_cache, INT32_INDEX_MAX
from blksprs.utils.validation import validate_contiguous, validate_device, validate_dtype_float, \
    validate_dimensions, validate_sparsity, validate_sparsity_block_size, validate_shape, ensure_contiguous, \
    validate_sparsity_layout, validate_binary

_LOG2E = tl.constexpr(1.4426950408889634)
# Tiles that are examined at once, see ``_clipped_tile_count`` and ``_validity_tiles``.
_TILES_PER_COUNT = tl.constexpr(64)
# Runs of equal key coordinates that a key tile may hold for the merged relations, see ``_key_runs``.
_RUN_CAPACITY = tl.constexpr(16)
# Leading relations that the kernels take from their maximum in the tiles that clip all of them, see
# ``_clipped_tile_count``. A static set of the first two relations matches the per-call choice of clipped relations on
# production batches, whereas loops for further sets cost more registers than they gain.
_CLIPPED_RELATIONS = 2

_TILE_SIZE_NAMES = ("TRITON_BLOCK_SIZE_Q", "TRITON_BLOCK_SIZE_K")
# The best tiles depend on the sequence length and on the density of the attention layout, which enter the key as
# power-of-two buckets of sparsity blocks and of tile schedule entries, so that varying lengths, such as during
# generation, are tuned a bounded number of times, and layouts of one length but different density, such as the local
# and the global attention of one model, are tuned separately. Every compiled variant of a kernel, given by its
# constexpr arguments, differs in its register and shared-memory needs and is tuned separately as well. These arguments
# only depend on the static properties of a call, never on its coordinates, so that the new buckets of a call only
# benchmark compiled variants. The variants that the coordinates of a call select, see ``_variant_launches``, share
# their tuning, and only configurations that fit every variant are tuned, see ``_prune_configs``.
_AUTOTUNE_KEY = ["sparsity_block_size", "n_seq_blocks_bucket", "n_schedule_entries_bucket", "HEAD_TILE_QK",
                 "HEAD_TILE_V", "N_CHUNKS_QK", "N_CHUNKS_V", "ROW_COMPLETE", "USE_INT64", "HAS_MASK", "HAS_BIAS",
                 "HAS_CAUSAL", "HAS_RELATION_VALIDITY", "N_RELATIONS", "CLIPPED_RELATIONS", "SCORE_LAYOUT"]
# Only the query-gradient kernel merges runs of equal keys, see ``_key_runs``.
_AUTOTUNE_KEY_DQ = _AUTOTUNE_KEY + ["MERGED_RELATIONS", "UNIQUE_KEY_RELATIONS"]
# Integer arguments that vary between calls, such as with the lengths and the number of sequences of a batch. Triton
# would otherwise compile a variant for their divisibility by 16.
_UNSPECIALISED = ["n_batches", "n_seq_blocks_q", "n_seq_blocks_k", "n_seq_blocks_bucket", "n_schedule_entries_bucket"]

# The configurations of every call configuration of the Flash kernels, once its variants are compiled, that exceed the
# resources of the GPU for some variant, see ``_launch``.
_UNFIT_CONFIGS: dict = {}
# The arguments of every Flash kernel that enter its call configurations, see ``_call_configuration``, and the keyword
# arguments that do not: those of the variants of a call and of the launch.
_CONFIGURATION_ARGUMENTS: dict = {}
_VARIANT_ARGUMENTS = frozenset(("BLOCK_SPARSE_SCORES", "grid", "warmup"))

# Bytes of one row of a head-dimension chunk. Wider head dimensions are split into chunks so that the tiles of a
# kernel fit into shared memory.
_MAX_HEAD_TILE_BYTES = 256


def _largest_program_tile() -> int:
    """Returns the largest tile that a program owns in any Flash Attention autotuning configuration of any mode."""
    program_tiles = (("flash_attention_forward", "TRITON_BLOCK_SIZE_Q"),
                     ("flash_attention_backward_dq", "TRITON_BLOCK_SIZE_Q"),
                     ("flash_attention_backward_dkdv", "TRITON_BLOCK_SIZE_K"))
    return max(config.kwargs[name]
               for mode in SUPPORTED_AUTOTUNE_MODES
               for profile, name in program_tiles
               for config in get_autotune_configs(profile, mode=mode, block_size_names=_TILE_SIZE_NAMES))


# Tile schedules group program blocks by the largest program tile, so that every autotuning configuration can share
# them. Smaller tiles use the schedule of their group.
_SCHEDULE_TILE_SIZE = _largest_program_tile()


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
                    sparsity_layout_o: Tensor | None = None,
                    causal_lengths: Tensor | None = None) -> BlksprsTensor:
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
        causal_lengths (Tensor, optional): The integral valid sequence lengths for native causal self-attention, one
            per attention batch. Future keys and query/key padding positions are ignored without requiring a
            compressed attention mask (default ``None``).

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
    causal_lengths_value = _normalise_causal_lengths(causal_lengths, n_batches, n_seq_blocks_q * sparsity_block_size,
                                                     n_seq_blocks_k * sparsity_block_size, q.device)

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

    attention_mask_value: Tensor | None = None
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

    # Resolve optional bias
    if (attention_bias is None) != (sparsity_layout_bias is None):
        raise ValueError("attention_bias and sparsity_layout_bias must be provided together")

    attention_bias_value: Tensor | None = None
    if attention_bias is not None and sparsity_layout_bias is not None:
        attention_bias_value = ensure_contiguous(attention_bias)
        sparsity_layout_bias = ensure_contiguous(sparsity_layout_bias)
        validate_dimensions(attention_bias_value, sparsity_layout_bias)
        validate_contiguous(attention_bias_value, sparsity_layout_bias)
        validate_device(attention_bias_value, sparsity_layout_bias, q)
        validate_dtype_float(attention_bias_value)
        validate_sparsity(sparsity_block_size, (attention_bias_value, sparsity_layout_bias))
        validate_shape(sparsity_layout_bias, expected_attn_shape, "sparsity_layout_bias")

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
        sparsity_block_size=sparsity_block_size,
    )

    return _flash_attention(
        q, k, v, attention_bias_value, None, None,
        _FlashAttentionArguments.from_layout_cache(
            layout_cache, sparsity_layout_q, sparsity_layout_k, sparsity_layout_v, sparsity_block_size, scale,
            attention_mask=attention_mask_value,
            has_bias=attention_bias_value is not None,
            causal_lengths=causal_lengths_value))


def _normalise_causal_lengths(causal_lengths: Tensor | None, n_batches: int, q_length: int, k_length: int,
                              device: torch.device) -> Tensor | None:
    """Validates the causal lengths of a call and returns them as a contiguous ``int64`` tensor."""
    if causal_lengths is None:
        return None
    if not isinstance(causal_lengths, Tensor):
        raise TypeError("causal_lengths must be a tensor or None")
    if q_length != k_length:
        raise ValueError("Native causal attention requires equal query and key lengths")
    if causal_lengths.dim() != 1:
        raise ValueError("causal_lengths must be one-dimensional")
    if causal_lengths.dtype == torch.bool or torch.is_floating_point(causal_lengths):
        raise TypeError("causal_lengths must use an integral dtype")
    validate_device(causal_lengths, torch.empty(0, device=device))
    validate_shape(causal_lengths, (n_batches,), "causal_lengths")
    if torch.any((causal_lengths < 0) | (causal_lengths > q_length)):
        raise ValueError(f"causal_lengths values must be between zero and {q_length}")
    return ensure_contiguous(causal_lengths.to(dtype=torch.int64))


@dataclass(frozen=True)
class _RelativeAttentionArguments:
    """Coordinates and relation tables of the learned relative terms, see ``flash_attention_relative_embeddings``."""

    query_relations: Tensor
    key_relations: Tensor
    # One row (minimum, maximum, first column of the projected relative scores) per relation. The tables of the
    # relations are padded to whole sparsity blocks in the columns of the projected scores.
    relation_metadata: Tensor
    query_relation_validity: Tensor | None
    key_relation_validity: Tensor | None
    # The query and key blocks with positions without valid relations within the causal lengths, with validity masks.
    query_invalid_blocks: Tensor | None
    key_invalid_blocks: Tensor | None
    # The columns of the projected scores, and the column of every row of the concatenated tables.
    n_score_columns: int
    embedding_columns: Tensor
    # The referenced blocks of the projected scores, or None to project the complete tables.
    score_blocks: ReferencedScoreBlocks | None
    n_relations: int
    # Bit mask of the relations whose valid key coordinates are pairwise distinct.
    unique_key_relations: int
    # The coordinate bounds that find the leading tiles in which every pair is clipped to the maximum of a relation,
    # see ``_clipped_tile_count``.
    query_coordinate_floors: Tensor | None = None
    key_coordinate_ceilings: Tensor | None = None
    # Bit mask of the relations whose scores and gradients the query-gradient kernel handles once per query and run of
    # keys with equal coordinates, see ``_key_runs``, from the relations declared to repeat their key coordinates.
    merged_relations: int = 0


@dataclass(frozen=True)
class _FlashAttentionArguments:
    """Non-differentiable operands and layout metadata of one Flash Attention call."""

    sparsity_block_size: int
    scale: float
    n_batches: int
    n_seq_blocks_q: int
    n_seq_blocks_k: int
    n_head_blocks_qk: int
    n_head_blocks_v: int
    n_sparse_blocks_o: int
    packed_indices_q: Tensor
    packed_indices_k: Tensor
    packed_indices_v: Tensor
    packed_indices_o: Tensor
    # Whether every sequence block of Q, K, V, and O holds all or none of its head-dimension blocks.
    row_complete: bool
    query_schedule: dict
    key_schedule: dict
    attention_mask: Tensor | None
    packed_indices_mask: Tensor | None
    packed_indices_bias: Tensor | None
    causal_lengths: Tensor | None
    relations: _RelativeAttentionArguments | None

    @staticmethod
    def from_layout_cache(layout_cache: dict, sparsity_layout_q: Tensor, sparsity_layout_k: Tensor,
                          sparsity_layout_v: Tensor, sparsity_block_size: int, scale: float, *,
                          attention_mask: Tensor | None, has_bias: bool, causal_lengths: Tensor | None,
                          relations: _RelativeAttentionArguments | None = None) -> "_FlashAttentionArguments":
        """Collects the arguments of a call from its operands and a layout cache built with its block size."""
        group_blocks = _schedule_group_blocks(sparsity_block_size)
        return _FlashAttentionArguments(
            sparsity_block_size=sparsity_block_size,
            scale=scale,
            n_batches=sparsity_layout_q.size(0),
            n_seq_blocks_q=sparsity_layout_q.size(1),
            n_seq_blocks_k=sparsity_layout_k.size(1),
            n_head_blocks_qk=sparsity_layout_q.size(2),
            n_head_blocks_v=sparsity_layout_v.size(2),
            n_sparse_blocks_o=layout_cache["n_sparse_blocks_o"],
            packed_indices_q=layout_cache["packed_indices_q"],
            packed_indices_k=layout_cache["packed_indices_k"],
            packed_indices_v=layout_cache["packed_indices_v"],
            packed_indices_o=layout_cache["packed_indices_o"],
            row_complete=layout_cache["row_complete"],
            query_schedule=layout_cache[f"query_tile_schedule_{group_blocks}"],
            key_schedule=layout_cache[f"key_tile_schedule_{group_blocks}"],
            attention_mask=attention_mask,
            packed_indices_mask=layout_cache["packed_indices_mask"] if attention_mask is not None else None,
            packed_indices_bias=layout_cache["packed_indices_bias"] if has_bias else None,
            causal_lengths=causal_lengths,
            relations=relations,
        )

    def input_tensors(self) -> list[Tensor]:
        """Returns the non-differentiable input tensors, whose in-place modification invalidates the backward."""
        tensors = [self.attention_mask, self.causal_lengths]
        if self.relations is not None:
            tensors += [self.relations.query_relations, self.relations.key_relations,
                        self.relations.query_relation_validity, self.relations.key_relation_validity]
        return [tensor for tensor in tensors if tensor is not None]


def _flash_attention(q: Tensor, k: Tensor, v: Tensor, attention_bias: Tensor | None,
                     relative_query: Tensor | None, relative_embeddings: Tensor | None,
                     arguments: _FlashAttentionArguments) -> BlksprsTensor:
    """Applies the Flash Attention kernels to validated operands.

    The kernels process tiles of query and key rows that span one or more whole sparsity blocks. Their sizes are
    autotuned independently of the sparsity block size, see ``prune_autotune_configs_spanning``. Every program owns one
    contiguous tile, the query tile in forward and in the query gradient and the key tile in the key and value
    gradients, and visits the union of the blocks that any block of its tile attends to. A tile schedule in the layout
    cache lists this union per group of program blocks in compressed sparse row form, so the visited tiles are gathered
    from blocks that need not be contiguous. The schedule also records, per entry, which program blocks attend to it
    and which of these block pairs hold blocks of the explicit mask or the bias, so that tiles mask pairs outside the
    attention layout and gather mask and bias elements only where they exist. Leading entries that need no masking at
    all are processed by a separate loop without masking. With learned relative terms, the leading ones among them in
    which every pair reaches the maximum of the first two relations are processed by a loop specialised to these
    relations, which takes their scores from the score column of their maximum and sums their gradients per query
    instead of handling every pair, see ``_clipped_tile_count``. With the referenced blocks of the projected scores, a
    call whose query blocks all store every relation as an interval launches the variant of the kernels that addresses
    all of them through intervals, see ``_variant_launches``. The compiled code only depends on the static properties
    of a call; its data only decide which of these loops and variants handle a tile.

    Every operand tile covers its full head dimension, so that the online-softmax numerator and the gradients stay in
    registers. Head dimensions that do not fit into one tile are split into chunks: query-key products accumulate over
    the chunks, and outputs are distributed over programs that each recompute the scores for their chunk.

    The relative terms are given by the dense queries ``relative_query``, the concatenated tables
    ``relative_embeddings``, and ``arguments.relations``.

    """
    return BlksprsTensor.wrap(_FlashAttentionAutograd.apply(
        q, k, v, attention_bias, relative_query, relative_embeddings, arguments))


class _FlashAttentionAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, attention_bias: Tensor | None,
                relative_query: Tensor | None, relative_embeddings: Tensor | None,
                arguments: _FlashAttentionArguments) -> Tensor:
        relative_scores = _project_relative_scores(relative_query, relative_embeddings, arguments)
        output, lse = _flash_attention_forward(q, k, v, attention_bias, relative_scores, arguments)

        ctx.save_for_backward(q, k, v, output, lse, attention_bias, relative_query, relative_embeddings,
                              *arguments.input_tensors())
        ctx.arguments = arguments

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        q, k, v, output, lse, attention_bias, relative_query, relative_embeddings, *_ = ctx.saved_tensors
        arguments: _FlashAttentionArguments = ctx.arguments

        # The projected scores are recomputed so that long-sequence training does not retain them for every layer.
        relative_scores = _project_relative_scores(relative_query, relative_embeddings, arguments)
        dq, dk, dv, dbias, drelative_scores = _flash_attention_backward(
            q, k, v, output, lse, grad_output.contiguous(), attention_bias, relative_scores, arguments)

        drelative_query = drelative_embeddings = None
        relations = arguments.relations
        if drelative_scores is not None and relations is not None:
            # Release the projected scores before using their much larger gradient, so that the CUDA allocator can
            # reuse that storage during long-sequence backward.
            del relative_scores
            padded_embeddings = _pad_relative_embeddings(relative_embeddings, relations)
            if relations.score_blocks is None:
                drelative_scores = drelative_scores.to(relative_query.dtype)
                drelative_query = torch.bmm(drelative_scores, padded_embeddings)
                dpadded_embeddings = torch.bmm(drelative_scores.transpose(1, 2), relative_query)
            else:
                drelative_query, dpadded_embeddings = referenced_score_gradients(
                    relative_query, padded_embeddings, drelative_scores, relations.score_blocks,
                    arguments.sparsity_block_size)
            drelative_embeddings = dpadded_embeddings.index_select(1, relations.embedding_columns)

        return dq, dk, dv, dbias, drelative_query, drelative_embeddings, None


def _project_relative_scores(relative_query: Tensor | None, relative_embeddings: Tensor | None,
                             arguments: _FlashAttentionArguments) -> Tensor | None:
    """Returns the projected relative scores, either complete or only their referenced blocks."""
    relations = arguments.relations
    if relative_query is None or relative_embeddings is None or relations is None:
        return None
    padded_embeddings = _pad_relative_embeddings(relative_embeddings, relations)
    if relations.score_blocks is None:
        return torch.bmm(relative_query, padded_embeddings.transpose(1, 2))
    return project_referenced_scores(relative_query, padded_embeddings, relations.score_blocks,
                                     arguments.sparsity_block_size)


def _pad_relative_embeddings(relative_embeddings: Tensor, relations: _RelativeAttentionArguments) -> Tensor:
    """Pads every relation table with zero rows to whole sparsity blocks.

    The columns of the projected scores then fall into value blocks of a single relation, and the matrix
    multiplications of the complete projection use aligned kernels. The padding rows are never referenced.

    """
    padded = relative_embeddings.new_zeros((relative_embeddings.size(0), relations.n_score_columns,
                                            relative_embeddings.size(2)))
    return padded.index_copy_(1, relations.embedding_columns, relative_embeddings)


def _head_tile(n_head_blocks: int, sparsity_block_size: int, element_size: int) -> tuple[int, int]:
    """Returns the width of the head-dimension chunks of a tile and the number of chunks."""
    head_dimension = n_head_blocks * sparsity_block_size
    head_tile = min(max(triton.next_power_of_2(head_dimension), 16), _MAX_HEAD_TILE_BYTES // element_size)
    return head_tile, max(1, triton.cdiv(head_dimension, head_tile))


def _sequence_bucket(arguments: _FlashAttentionArguments) -> int:
    """Returns the power-of-two bucket of the sequence length in sparsity blocks, which only enters autotuning."""
    return triton.next_power_of_2(max(arguments.n_seq_blocks_q, arguments.n_seq_blocks_k, 1))


def _schedule_bucket(schedule: dict) -> int:
    """Returns the power-of-two bucket of the number of entries of a tile schedule, which only enters autotuning."""
    return triton.next_power_of_2(max(schedule["blocks"].numel(), 1))


def _schedule_group_blocks(sparsity_block_size: int) -> int:
    """Returns the number of sparsity blocks that share one entry of a tile schedule."""
    return max(1, _SCHEDULE_TILE_SIZE // sparsity_block_size)


def _schedule_operands(schedule: dict) -> tuple[Tensor, ...]:
    """Returns the tensors of a tile schedule in the order of the kernel arguments."""
    return (schedule["offsets"], schedule["blocks"], schedule["bits"], schedule["mask_bits"], schedule["bias_bits"],
            schedule["plain"])


def _score_operands(arguments: _FlashAttentionArguments, attention_bias: Tensor | None,
                    relative_scores: Tensor | None, device: torch.device) -> tuple[list, dict]:
    """Returns the kernel arguments that complete query-key products to attention logits, and their flags."""
    relations = arguments.relations
    empty_float = torch.empty(0, device=device, dtype=torch.float32)
    empty_integer = torch.empty(0, device=device, dtype=torch.int64)
    empty_boolean = torch.empty(0, device=device, dtype=torch.bool)
    empty_byte = torch.empty(0, device=device, dtype=torch.int8)

    def present(value: Tensor | None, empty: Tensor) -> Tensor:
        return empty if value is None else value

    operands = [
        present(arguments.attention_mask, empty_boolean),
        present(arguments.packed_indices_mask, empty_integer),
        present(attention_bias, empty_float),
        present(arguments.packed_indices_bias, empty_integer),
        present(arguments.causal_lengths, empty_integer),
        present(relative_scores, empty_float),
        empty_integer if relations is None else relations.query_relations,
        empty_integer if relations is None else relations.key_relations,
        empty_integer if relations is None else relations.relation_metadata,
        empty_integer if relations is None else present(relations.query_coordinate_floors, empty_integer),
        empty_integer if relations is None else present(relations.key_coordinate_ceilings, empty_integer),
        empty_boolean if relations is None else present(relations.query_relation_validity, empty_boolean),
        empty_boolean if relations is None else present(relations.key_relation_validity, empty_boolean),
        empty_byte if relations is None else present(relations.query_invalid_blocks, empty_byte),
        empty_byte if relations is None else present(relations.key_invalid_blocks, empty_byte),
        empty_integer if relations is None or relations.score_blocks is None else relations.score_blocks.addressing,
        arguments.n_batches, arguments.n_seq_blocks_q, arguments.n_seq_blocks_k,
        0 if relations is None else relations.n_score_columns,
        arguments.scale,
    ]
    flags = dict(
        HAS_MASK=arguments.attention_mask is not None,
        HAS_BIAS=attention_bias is not None,
        HAS_CAUSAL=arguments.causal_lengths is not None,
        HAS_RELATIVE=relations is not None,
        HAS_RELATION_VALIDITY=relations is not None and relations.query_relation_validity is not None,
        BLOCK_SPARSE_SCORES=_score_layout(relations),
        SCORE_LAYOUT=_score_layout(relations),
        N_RELATIONS=0 if relations is None else relations.n_relations,
        CLIPPED_RELATIONS=0 if relations is None else (1 << min(relations.n_relations, _CLIPPED_RELATIONS)) - 1,
    )
    return operands, flags


def _score_layout(relations: _RelativeAttentionArguments | None) -> int:
    """Returns the layout of the projected scores as a kernel flag, see ``_score_offsets``.

    The flag is 0 for the complete projection. For the referenced blocks, it is one plus twice the bit mask of the
    relations that are always addressed through intervals of value blocks instead of the map of all value blocks, the
    relations with few value blocks, see ``compact_score_relations``. It only depends on the relation bounds. The
    calls whose query blocks all store every relation as an interval address all of them so, see
    ``_variant_launches``.

    """
    if relations is None or relations.score_blocks is None:
        return 0
    return 1 | (relations.score_blocks.interval_relations << 1)


def _launch(kernel, grid, args: tuple, kwargs: dict, flags: dict, variants: list[dict]) -> None:
    """Launches an autotuned Flash kernel with the variant ``flags`` of a call, see ``_variant_launches``.

    Triton compiles the configurations of a new autotuning key one after the other while benchmarking them, which takes
    minutes for the Flash kernels. The first launch of a call configuration, given by its static compile-time
    arguments and the dtypes of its tensors, compiles all configurations of all its ``variants`` concurrently instead,
    so that no later call of the configuration compiles, whichever of the variants its data select. Configurations
    that exceed the resources of the GPU for some variant are excluded from the tuning of the call configuration, see
    ``_prune_configs``, so that the configuration tuned on one variant also serves the others.

    """
    configuration = _call_configuration(kernel, args, kwargs | variants[0])
    if configuration not in _UNFIT_CONFIGS:
        configs = prune_autotune_configs_spanning(kernel.configs, {"sparsity_block_size": kwargs.get(
            "sparsity_block_size", args[kernel.arg_names.index("sparsity_block_size")])})
        compiled = []
        with ThreadPoolExecutor(min(len(configs) * len(variants), _available_cpus())) as executor, \
                triton.AsyncCompileMode(executor, ignore_errors=True):
            for variant in variants:
                for config in configs:
                    compiled.append((config, kernel.fn.warmup(*args, grid=(1,), **kwargs, **variant,
                                                              **config.all_kwargs())))
        unfit = set()
        for config, future in compiled:
            # Compiled kernels found in memory are returned as such, all others as futures.
            compiled_kernel = future.result(ignore_errors=True) if hasattr(future, "result") else future
            if compiled_kernel is None:
                # The autotuner reports the error of a configuration that does not compile.
                unfit.add(config)
                continue
            try:
                compiled_kernel._init_handles()
            except OutOfResources:
                unfit.add(config)
        _UNFIT_CONFIGS[configuration] = unfit
    kernel[grid](*args, **kwargs, **flags)


def _available_cpus() -> int:
    """Returns the number of CPUs that this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _call_configuration(kernel, arguments: tuple, flags: dict) -> tuple:
    """Returns the static compile-time arguments of a call of a Flash kernel and the dtypes of its tensors.

    The variants of a call, see ``_variant_launches``, share them. ``arguments`` are the positional arguments and
    ``flags`` the keyword arguments. Arguments that vary between calls of one configuration, such as the lengths, and
    the attention scale are left out. The selection of the arguments is prepared once per kernel, as the configuration
    is found for every launch.

    """
    selection = _CONFIGURATION_ARGUMENTS.get(kernel)
    if selection is None:
        static_names = [name for name in kernel.arg_names if name not in _UNSPECIALISED and name != "scale"]
        tensors = [index for index, name in enumerate(kernel.arg_names[:len(arguments)])
                   if name in static_names and isinstance(arguments[index], Tensor)]
        values = [index for index, name in enumerate(kernel.arg_names[:len(arguments)])
                  if name in static_names and not isinstance(arguments[index], Tensor)]
        flag_names = tuple(name for name in flags if name not in _VARIANT_ARGUMENTS)
        selection = _CONFIGURATION_ARGUMENTS[kernel] = (_tuple_getter(tensors), _tuple_getter(values), flag_names)
    tensors, values, flag_names = selection
    with torch._C.DisableTorchFunctionSubclass():
        # The dtypes of compressed tensors are read without the dispatch of their subclass.
        dtypes = tuple(tensor.dtype for tensor in tensors(arguments))
    return kernel.base_fn.__name__, tuple(map(flags.get, flag_names)), values(arguments), dtypes


def _tuple_getter(indices: list[int]):
    """Returns a function that returns the items of a tuple at ``indices`` as a tuple."""
    if len(indices) == 1:
        return lambda items: (items[indices[0]],)
    return operator.itemgetter(*indices) if indices else lambda items: ()


def _prune_configs(configs: list, kernel_args: dict, **kwargs) -> list:
    """Keeps the configurations whose tiles span whole sparsity blocks and that fit every variant of the call.

    The configurations that exceed the resources of the GPU for some variant of the call configuration are found by
    ``_launch``, which compiles them before the first benchmark of the configuration.

    """
    pruned = prune_autotune_configs_spanning(configs, kernel_args, **kwargs)
    kernel = next(kernel for kernel in (flash_attention_kernel, flash_attention_kernel_bwd_dkdv,
                                        flash_attention_kernel_bwd_dq) if kernel.configs is configs)
    unfit = _UNFIT_CONFIGS.get(_call_configuration(kernel, tuple(kernel_args.values()), kwargs), set())
    return [config for config in pruned if config not in unfit] or pruned


def _variant_launches(score_flags: dict, relations: "_RelativeAttentionArguments | None") -> tuple[dict, list[dict]]:
    """Returns the kernel flags of the launch of a Flash kernel for a call and those of all variants of the call.

    With the referenced blocks of the projected scores, a call whose stored blocks of every relation form one interval
    in every query block addresses all relations through intervals, and any other call only the relations that are
    always stored as intervals, see ``_score_layout`` and ``referenced_value_intervals``. The complete projection and
    relations that are all stored as intervals have a single variant.

    """
    layout = score_flags["BLOCK_SPARSE_SCORES"]
    all_intervals = (1 << (score_flags["N_RELATIONS"] + 1)) - 1
    if layout in (0, all_intervals):
        return score_flags, [score_flags]
    variants = [{**score_flags, "BLOCK_SPARSE_SCORES": all_intervals}, score_flags]
    # Only the referenced blocks have a layout other than the complete one, so that ``relations`` holds them.
    return variants[0 if relations.score_blocks.complete_intervals else 1], variants


def _head_settings(arguments: _FlashAttentionArguments, element_size: int) -> tuple[dict, int, int]:
    """Returns the head-dimension tiling of a call as kernel flags, and the numbers of Q/K and V chunks."""
    head_tile_qk, n_chunks_qk = _head_tile(arguments.n_head_blocks_qk, arguments.sparsity_block_size, element_size)
    head_tile_v, n_chunks_v = _head_tile(arguments.n_head_blocks_v, arguments.sparsity_block_size, element_size)
    settings = dict(HEAD_TILE_QK=head_tile_qk, HEAD_TILE_V=head_tile_v,
                    N_CHUNKS_QK=n_chunks_qk, N_CHUNKS_V=n_chunks_v)
    return settings, n_chunks_qk, n_chunks_v


def _indexing_extents(arguments: _FlashAttentionArguments, score_operands: list) -> list[Tensor | int]:
    """Returns the index and metadata tensors of a call and other extents that bound the flat offsets of the kernels.

    The referenced blocks of the projected scores and of their gradients are addressed up to the guard blocks around
    them, see ``allocate_referenced_scores``, so that the size of their allocation enters as well.

    """
    extents: list[Tensor | int] = [
        arguments.packed_indices_q, arguments.packed_indices_k, arguments.packed_indices_v, arguments.packed_indices_o,
        *_schedule_operands(arguments.query_schedule), *_schedule_operands(arguments.key_schedule),
        *(operand for operand in score_operands if isinstance(operand, Tensor))]
    relations = arguments.relations
    if relations is not None and relations.score_blocks is not None:
        extents.append(referenced_score_elements(relations.score_blocks, arguments.sparsity_block_size))
    return extents


def _flash_attention_forward(q: Tensor, k: Tensor, v: Tensor, attention_bias: Tensor | None,
                             relative_scores: Tensor | None,
                             arguments: _FlashAttentionArguments) -> tuple[Tensor, Tensor]:
    sparsity_block_size = arguments.sparsity_block_size
    output = torch.empty((arguments.n_sparse_blocks_o, sparsity_block_size, sparsity_block_size),
                         dtype=q.dtype, device=q.device)
    lse = torch.empty((arguments.n_batches, arguments.n_seq_blocks_q * sparsity_block_size),
                      dtype=torch.float32, device=q.device)

    score_operands, score_flags = _score_operands(arguments, attention_bias, relative_scores, q.device)
    head_settings, _, n_chunks_v = _head_settings(arguments, q.element_size())
    use_int64 = not can_use_int32_indexing(q, k, v, output, lse, *_indexing_extents(arguments, score_operands))

    def triton_grid(meta):
        return [arguments.n_batches,
                triton.cdiv(arguments.n_seq_blocks_q * sparsity_block_size, meta["TRITON_BLOCK_SIZE_Q"]),
                n_chunks_v]

    _launch(flash_attention_kernel, triton_grid,
            (q, k, v, output, lse,
             arguments.packed_indices_q, arguments.packed_indices_k,
             arguments.packed_indices_v, arguments.packed_indices_o,
             *_schedule_operands(arguments.query_schedule),
             *score_operands,
             _sequence_bucket(arguments), _schedule_bucket(arguments.query_schedule),
             arguments.n_head_blocks_qk, arguments.n_head_blocks_v,
             sparsity_block_size),
            dict(SCHEDULE_BLOCKS=arguments.query_schedule["group_blocks"],
                 ROW_COMPLETE=arguments.row_complete,
                 USE_INT64=use_int64,
                 **head_settings),
            *_variant_launches(score_flags, arguments.relations))

    return output, lse


def _flash_attention_backward(q: Tensor, k: Tensor, v: Tensor, output: Tensor, lse: Tensor, grad_output: Tensor,
                              attention_bias: Tensor | None, relative_scores: Tensor | None,
                              arguments: _FlashAttentionArguments) -> tuple:
    sparsity_block_size = arguments.sparsity_block_size
    n_batches = arguments.n_batches
    q_length = arguments.n_seq_blocks_q * sparsity_block_size
    relations = arguments.relations

    score_operands, score_flags = _score_operands(arguments, attention_bias, relative_scores, q.device)
    head_settings, n_chunks_qk, n_chunks_v = _head_settings(arguments, q.element_size())

    # The row-wise sums of the output and its gradient, which recover the softmax gradient from probabilities.
    delta = torch.empty((n_batches, q_length), dtype=torch.float32, device=q.device)
    use_int64_preprocess = not can_use_int32_indexing(output, grad_output, delta, arguments.packed_indices_o)

    def triton_grid_preprocess(meta):
        return [n_batches, triton.cdiv(q_length, meta["TRITON_BLOCK_SIZE"])]

    flash_attention_kernel_bwd_preprocess[triton_grid_preprocess](
        output, grad_output, delta,
        arguments.packed_indices_o,
        arguments.n_seq_blocks_q, arguments.n_head_blocks_v,
        sparsity_block_size,
        HEAD_TILE_V=head_settings["HEAD_TILE_V"],
        ROW_COMPLETE=arguments.row_complete,
        USE_INT64=use_int64_preprocess)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # Blocks of the bias outside the attention layout receive no gradient.
    dbias = None if attention_bias is None else torch.zeros_like(attention_bias)
    drelative_scores = None
    if relative_scores is not None and relations is not None:
        score_blocks = relations.score_blocks
        drelative_scores = (
            torch.zeros_like(relative_scores, dtype=torch.float32) if score_blocks is None else
            allocate_referenced_scores(score_blocks, sparsity_block_size, torch.float32, q.device, zero=True))
    use_int64 = not can_use_int32_indexing(
        q, k, v, grad_output, lse, delta, dq, dk, dv, dbias, drelative_scores,
        *_indexing_extents(arguments, score_operands))

    def triton_grid_dkdv(meta):
        return [n_batches,
                triton.cdiv(arguments.n_seq_blocks_k * sparsity_block_size, meta["TRITON_BLOCK_SIZE_K"]),
                max(n_chunks_qk, n_chunks_v)]

    _launch(flash_attention_kernel_bwd_dkdv, triton_grid_dkdv,
            (q, k, v, grad_output, dk, dv, lse, delta,
             arguments.packed_indices_q, arguments.packed_indices_k,
             arguments.packed_indices_v, arguments.packed_indices_o,
             *_schedule_operands(arguments.key_schedule),
             *score_operands,
             _sequence_bucket(arguments), _schedule_bucket(arguments.key_schedule),
             arguments.n_head_blocks_qk, arguments.n_head_blocks_v,
             sparsity_block_size),
            dict(SCHEDULE_BLOCKS=arguments.key_schedule["group_blocks"],
                 ROW_COMPLETE=arguments.row_complete,
                 USE_INT64=use_int64,
                 **head_settings),
            *_variant_launches(score_flags, relations))

    def triton_grid_dq(meta):
        return [n_batches,
                triton.cdiv(q_length, meta["TRITON_BLOCK_SIZE_Q"]),
                n_chunks_qk]

    _launch(flash_attention_kernel_bwd_dq, triton_grid_dq,
            (q, k, v, grad_output, dq, lse, delta,
             torch.empty(0, device=q.device, dtype=q.dtype) if dbias is None else dbias,
             torch.empty(0, device=q.device, dtype=torch.float32) if drelative_scores is None else drelative_scores,
             arguments.packed_indices_q, arguments.packed_indices_k,
             arguments.packed_indices_v, arguments.packed_indices_o,
             *_schedule_operands(arguments.query_schedule),
             *score_operands,
             _sequence_bucket(arguments), _schedule_bucket(arguments.query_schedule),
             arguments.n_head_blocks_qk, arguments.n_head_blocks_v,
             sparsity_block_size),
            dict(SCHEDULE_BLOCKS=arguments.query_schedule["group_blocks"],
                 ROW_COMPLETE=arguments.row_complete,
                 USE_INT64=use_int64,
                 RELATION_BLOCK_SIZE=max(1, triton.next_power_of_2(score_flags["N_RELATIONS"])),
                 MERGED_RELATIONS=0 if relations is None else relations.merged_relations,
                 UNIQUE_KEY_RELATIONS=0 if relations is None else relations.unique_key_relations,
                 **head_settings),
            *_variant_launches(score_flags, relations))

    return dq, dk, dv, dbias, drelative_scores


@triton.jit
def _sequence_tile(tile, n_seq_blocks,
                   TILE_SIZE: tl.constexpr, sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns the sparsity blocks, sequence positions, and validity of the rows of a contiguous sequence tile."""
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    positions = tl.cast(tile, index_dtype) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    blocks = positions // sparsity_block_size
    return blocks, positions, blocks < n_seq_blocks


@triton.jit
def _scheduled_tile(schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits, start, end, tile,
                    TILE_SIZE: tl.constexpr, sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr,
                    LOAD_BITS: tl.constexpr, LOAD_MASK_BITS: tl.constexpr, LOAD_BIAS_BITS: tl.constexpr):
    """Returns the rows of a tile gathered from consecutive schedule entries.

    Next to the sparsity blocks, sequence positions, and validity of the rows, the activity, mask, and bias bits of
    their entries are returned when ``LOAD_BITS``, ``LOAD_MASK_BITS``, and ``LOAD_BIAS_BITS`` are set.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    rows = tl.arange(0, TILE_SIZE)
    entries = start + tile * (TILE_SIZE // sparsity_block_size) + rows // sparsity_block_size
    valid = entries < end
    blocks = tl.cast(tl.load(schedule_blocks + entries, mask=valid, other=0), index_dtype)
    positions = blocks * sparsity_block_size + rows % sparsity_block_size
    if LOAD_BITS:
        bits = tl.load(schedule_bits + entries, mask=valid, other=0)
    else:
        bits = 0
    if LOAD_MASK_BITS:
        mask_bits = tl.load(schedule_mask_bits + entries, mask=valid, other=0)
    else:
        mask_bits = 0
    if LOAD_BIAS_BITS:
        bias_bits = tl.load(schedule_bias_bits + entries, mask=valid, other=0)
    else:
        bias_bits = 0
    return blocks, positions, valid, bits, mask_bits, bias_bits


@triton.jit
def _operand_offsets(packed_indices, batch, blocks, positions, valid, n_seq_blocks, n_head_blocks: tl.constexpr,
                     chunk, HEAD_TILE: tl.constexpr, sparsity_block_size: tl.constexpr,
                     ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns the offsets and mask of one head-dimension chunk of a compressed operand for a tile of rows.

    The head-dimension blocks of a complete layout row are stored consecutively, so the offsets separate into a row
    and a column term. This lets Triton vectorise and pipeline the loads. Otherwise, every head block is looked up.
    ``chunk`` must be one of the chunks of the head dimension, so that only a partial last chunk needs a column mask.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    block_numel: tl.constexpr = sparsity_block_size * sparsity_block_size
    columns = chunk * HEAD_TILE + tl.arange(0, HEAD_TILE)
    head_blocks = columns // sparsity_block_size
    columns_valid = head_blocks < n_head_blocks
    row_offsets = (positions % sparsity_block_size) * sparsity_block_size
    layout_rows = (batch * n_seq_blocks + blocks) * n_head_blocks

    if ROW_COMPLETE:
        first_blocks = tl.cast(tl.load(packed_indices + layout_rows, mask=valid & (n_head_blocks > 0), other=-1),
                               index_dtype)
        offsets = ((first_blocks * block_numel + row_offsets)[:, None] +
                   (head_blocks * block_numel + columns % sparsity_block_size)[None, :])
        mask = (first_blocks >= 0)[:, None]
        if (n_head_blocks * sparsity_block_size) % HEAD_TILE != 0:
            mask = mask & columns_valid[None, :]
    else:
        blocks_packed = tl.cast(tl.load(packed_indices + layout_rows[:, None] + head_blocks[None, :],
                                        mask=valid[:, None] & columns_valid[None, :], other=-1), index_dtype)
        offsets = blocks_packed * block_numel + row_offsets[:, None] + (columns % sparsity_block_size)[None, :]
        mask = blocks_packed >= 0

    return offsets, mask


@triton.jit
def _head_product(a_tile, a, packed_indices_a, a_blocks, a_positions, a_valid, a_n_seq_blocks,
                  b, packed_indices_b, b_blocks, b_positions, b_valid, b_n_seq_blocks,
                  batch, n_head_blocks: tl.constexpr, chunk,
                  A_ROWS: tl.constexpr, B_ROWS: tl.constexpr, HEAD_TILE: tl.constexpr, N_CHUNKS: tl.constexpr,
                  RETURN_CHUNK: tl.constexpr, sparsity_block_size: tl.constexpr,
                  ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns ``a @ b.T`` over the full head dimension and chunk ``chunk`` of the ``b`` tile.

    Without chunking, ``a_tile`` holds the complete ``a`` tile and the returned ``b`` tile covers the full head
    dimension. With chunking, both operands are loaded chunk by chunk, and the requested chunk of ``b`` is only loaded
    if ``RETURN_CHUNK`` is set.

    """
    if N_CHUNKS > 1:
        product = tl.zeros((A_ROWS, B_ROWS), dtype=tl.float32)
        for head_chunk in range(0, N_CHUNKS):
            a_offsets, a_mask = _operand_offsets(packed_indices_a, batch, a_blocks, a_positions, a_valid,
                                                 a_n_seq_blocks, n_head_blocks, head_chunk,
                                                 HEAD_TILE, sparsity_block_size, ROW_COMPLETE, USE_INT64)
            b_offsets, b_mask = _operand_offsets(packed_indices_b, batch, b_blocks, b_positions, b_valid,
                                                 b_n_seq_blocks, n_head_blocks, head_chunk,
                                                 HEAD_TILE, sparsity_block_size, ROW_COMPLETE, USE_INT64)
            product += tl.dot(tl.load(a + a_offsets, mask=a_mask, other=0),
                              tl.trans(tl.load(b + b_offsets, mask=b_mask, other=0)))
        if RETURN_CHUNK:
            b_offsets, b_mask = _operand_offsets(packed_indices_b, batch, b_blocks, b_positions, b_valid,
                                                 b_n_seq_blocks, n_head_blocks, chunk,
                                                 HEAD_TILE, sparsity_block_size, ROW_COMPLETE, USE_INT64)
            b_tile = tl.load(b + b_offsets, mask=b_mask, other=0)
        else:
            b_tile = 0
    else:
        b_offsets, b_mask = _operand_offsets(packed_indices_b, batch, b_blocks, b_positions, b_valid,
                                             b_n_seq_blocks, n_head_blocks, 0,
                                             HEAD_TILE, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        b_tile = tl.load(b + b_offsets, mask=b_mask, other=0)
        product = tl.dot(a_tile, tl.trans(b_tile))
    return product, b_tile


@triton.jit
def _pair_offsets(batch, query_blocks, query_positions, query_valid, key_blocks, key_positions, key_valid,
                  packed_indices, n_seq_blocks_q, n_seq_blocks_k,
                  sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns the offsets and mask of query-key pairs in a compressed tensor with an attention-shaped layout.

    Query and key operands are broadcast against each other, so queries may lie along either tile dimension.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    blocks_packed = tl.cast(tl.load(packed_indices + (batch * n_seq_blocks_q + query_blocks) * n_seq_blocks_k
                                    + key_blocks, mask=query_valid & key_valid, other=-1), index_dtype)
    offsets = (blocks_packed * (sparsity_block_size * sparsity_block_size)
               + (query_positions % sparsity_block_size) * sparsity_block_size
               + key_positions % sparsity_block_size)
    return offsets, blocks_packed >= 0


@triton.jit
def _score_offsets(score_blocks, relation: tl.constexpr, batch, query_positions, valid, columns, q_length,
                   n_score_columns, sparsity_block_size: tl.constexpr, N_RELATIONS: tl.constexpr,
                   BLOCK_SPARSE_SCORES: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns the offsets of the projected relative scores of queries at table columns, and whether they exist.

    The columns belong to ``relation``. The complete projection holds the scores of all table columns for every query.
    The block-sparse projection only holds the blocks of queries and table columns that are referenced by permitted
    query-key pairs, and every row of ``score_blocks`` maps the value blocks of one query block to their index in the
    compressed projection or to -1, followed by an offset for every relation, see ``ReferencedScoreBlocks``. The
    relations marked in ``BLOCK_SPARSE_SCORES``, see ``_score_layout``, reference one interval of value blocks without
    gaps per query block, and the index of a block is its value block plus the offset of its query block. Their columns
    are not checked against the interval, which would cost a comparison per pair: every permitted pair addresses a
    referenced block, and all other ``valid`` pairs have masked scores and zero gradients, and address guard blocks or
    other blocks, see ``allocate_referenced_scores``. All other columns, and those of several relations with
    ``relation=-1``, are addressed through the map. Only scores of ``valid`` positions are looked up.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    if BLOCK_SPARSE_SCORES:
        n_value_blocks = n_score_columns // sparsity_block_size
        query_blocks = batch * (q_length // sparsity_block_size) + query_positions // sparsity_block_size
        row = score_blocks + query_blocks * (n_value_blocks + N_RELATIONS)
        value_blocks = columns // sparsity_block_size
        if relation >= 0 and (BLOCK_SPARSE_SCORES >> (relation + 1)) & 1:
            # One load per query instead of one per pair.
            blocks = tl.load(row + n_value_blocks + relation, mask=query_positions < q_length, other=0) + value_blocks
            present = valid
        else:
            blocks = tl.load(row + value_blocks, mask=valid, other=-1)
            present = blocks >= 0
        offsets = (tl.cast(blocks, index_dtype) * (sparsity_block_size * sparsity_block_size)
                   + (query_positions % sparsity_block_size) * sparsity_block_size + columns % sparsity_block_size)
    else:
        offsets = (batch * q_length + query_positions) * n_score_columns + columns
        present = valid
    return offsets, present


@triton.jit
def _relation_rows(query_relations, key_relations, relation_metadata, relation, batch,
                   query_positions, query_valid, key_positions, key_valid, n_batches, q_length, k_length,
                   USE_INT64: tl.constexpr):
    """Returns the clipped rows of a relation's table for query-key pairs, the table size minus one, and offset.

    Rows ``0`` and the returned span are the clipped bounds.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    # The coordinates are stored in 32 bits whenever their clipped differences fit, see
    # ``_prepare_relation_coordinates``.
    coordinate_dtype = query_relations.dtype.element_ty
    relation_min = tl.cast(tl.load(relation_metadata + relation * 3), coordinate_dtype)
    relation_span = tl.cast(tl.load(relation_metadata + relation * 3 + 1), coordinate_dtype) - relation_min
    relation_offset = tl.cast(tl.load(relation_metadata + relation * 3 + 2), index_dtype)
    query_coordinates = tl.load(query_relations + (relation * n_batches + batch) * q_length + query_positions,
                                mask=query_valid, other=0)
    key_coordinates = tl.load(key_relations + (relation * n_batches + batch) * k_length + key_positions,
                              mask=key_valid, other=0)
    # Relations are query minus key. Negating both coordinates yields the opposite direction, which is equivalent up
    # to reversed embedding rows.
    differences = (query_coordinates - relation_min) - key_coordinates
    rows = tl.minimum(tl.maximum(differences, 0), relation_span)
    return rows, relation_span, relation_offset


@triton.jit
def _relation_maximum_column(relation_metadata, relation, USE_INT64: tl.constexpr):
    """Returns the column of the clipped maximum of a relation in the projected relative scores."""
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    relation_span = tl.load(relation_metadata + relation * 3 + 1) - tl.load(relation_metadata + relation * 3)
    return tl.cast(tl.load(relation_metadata + relation * 3 + 2) + relation_span, index_dtype)


@triton.jit
def _clipped_tile_count(schedule_blocks, query_floors, key_ceilings, relation_metadata, batch, program_block,
                        start, n_plain, n_batches, n_seq_blocks_q, n_seq_blocks_k,
                        CLIPPED_RELATIONS: tl.constexpr, N_RELATIONS: tl.constexpr,
                        INNER_TILE_SIZE: tl.constexpr, sparsity_block_size: tl.constexpr, BY_KEY: tl.constexpr):
    """Counts the leading plain tiles of a program in which every relation of ``CLIPPED_RELATIONS`` is clipped.

    A relation is clipped in a tile if the coordinate difference of every pair reaches the maximum of the relation,
    so that all pairs share the score column of its maximum. ``query_floors`` holds, for every query block, the least
    coordinate of this and all later query blocks, and ``key_ceilings``, for every key block, the greatest coordinate
    of this and all earlier key blocks. ``program_block`` is the first query block of the program or, with
    ``BY_KEY``, its last key block. Schedule entries are ordered away from the diagonal, so the bound at the last entry
    of a tile covers all of its entries, and the clipped tiles of every relation form a prefix of the schedule.

    """
    inner_blocks: tl.constexpr = INNER_TILE_SIZE // sparsity_block_size
    tiles = tl.arange(0, _TILES_PER_COUNT)
    n_clipped = n_plain
    for relation in tl.static_range(N_RELATIONS):
        if CLIPPED_RELATIONS & (1 << relation):
            relation_max = tl.load(relation_metadata + relation * 3 + 1)
            relation_floors = query_floors + (relation * n_batches + batch) * n_seq_blocks_q
            relation_ceilings = key_ceilings + (relation * n_batches + batch) * n_seq_blocks_k
            n_relation_clipped = 0
            for first_tile in range(0, n_plain, _TILES_PER_COUNT):
                tile_valid = first_tile + tiles < n_plain
                last_blocks = tl.load(schedule_blocks + start + (first_tile + tiles + 1) * inner_blocks - 1,
                                      mask=tile_valid, other=0)
                if BY_KEY:
                    differences = (tl.load(relation_floors + last_blocks, mask=tile_valid, other=0)
                                   - tl.load(relation_ceilings + program_block))
                else:
                    differences = (tl.load(relation_floors + program_block)
                                   - tl.load(relation_ceilings + last_blocks, mask=tile_valid, other=0))
                n_relation_clipped += tl.sum(tl.cast(tile_valid & (differences >= relation_max), tl.int32))
            n_clipped = tl.minimum(n_clipped, n_relation_clipped)
    return n_clipped


@triton.jit
def _validity_tiles(schedule_blocks, other_invalid_blocks, program_invalid_blocks, batch, program_tile, start, end,
                    n_tiles, n_seq_blocks_program, n_seq_blocks_other,
                    PROGRAM_TILE_SIZE: tl.constexpr, INNER_TILE_SIZE: tl.constexpr, sparsity_block_size: tl.constexpr,
                    SUFFIX: tl.constexpr):
    """Returns the bound of the tiles of a program that hold positions without valid relations.

    ``program_invalid_blocks`` and ``other_invalid_blocks`` mark the blocks of the program's and of the other sequence
    that hold such positions within the causal lengths. Tiles of a program whose own blocks hold such positions all
    need the validity masks. Otherwise, these are the tiles up to the last one with such an entry or, with
    ``SUFFIX``, from the first one on, which covers positions at the start of sequences, whose entries come first in
    the schedules of query blocks and last in those of key blocks. Returns the number of leading tiles that need the
    masks or, with ``SUFFIX``, the first tile that needs them.

    """
    program_blocks = (tl.cast(program_tile, tl.int32) * (PROGRAM_TILE_SIZE // sparsity_block_size)
                      + tl.arange(0, PROGRAM_TILE_SIZE // sparsity_block_size))
    program_flagged = tl.max(tl.load(program_invalid_blocks + batch * n_seq_blocks_program + program_blocks,
                                     mask=program_blocks < n_seq_blocks_program, other=0)) != 0
    inner_blocks: tl.constexpr = INNER_TILE_SIZE // sparsity_block_size
    tiles = tl.arange(0, _TILES_PER_COUNT)
    entry_offsets = tl.arange(0, inner_blocks)
    first = n_tiles
    last = n_tiles * 0
    for first_tile in range(0, n_tiles, _TILES_PER_COUNT):
        tile_indices = first_tile + tiles
        entries = start + tile_indices[:, None] * inner_blocks + entry_offsets[None, :]
        entry_valid = (tile_indices[:, None] < n_tiles) & (entries < end)
        blocks = tl.load(schedule_blocks + entries, mask=entry_valid, other=0)
        flagged = tl.max(tl.load(other_invalid_blocks + batch * n_seq_blocks_other + blocks, mask=entry_valid,
                                 other=0), axis=1) != 0
        first = tl.minimum(first, tl.min(tl.where(flagged, tile_indices, n_tiles)))
        last = tl.maximum(last, tl.max(tl.where(flagged, tile_indices + 1, 0)))
    if SUFFIX:
        bound = tl.where(program_flagged, 0, first)
    else:
        bound = tl.where(program_flagged, n_tiles, last)
    return bound


@triton.jit
def _relation_validity(query_relation_validity, key_relation_validity, relation, batch,
                       query_positions, query_valid, key_positions, key_valid, n_batches, q_length, k_length):
    """Returns whether a relation applies to query-key pairs."""
    query_validity = tl.load(query_relation_validity + (relation * n_batches + batch) * q_length + query_positions,
                             mask=query_valid, other=0)
    key_validity = tl.load(key_relation_validity + (relation * n_batches + batch) * k_length + key_positions,
                           mask=key_valid, other=0)
    return (query_validity != 0) & (key_validity != 0)


@triton.jit
def _projected_relative_scores(relative_scores, query_relations, key_relations, relation_metadata,
                               query_relation_validity, key_relation_validity, score_blocks,
                               batch, query_positions, query_valid, key_positions, key_valid,
                               n_batches, q_length, k_length, n_score_columns,
                               N_RELATIONS: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                               BLOCK_SPARSE_SCORES: tl.constexpr, RELATIONS: tl.constexpr, CLIPPED: tl.constexpr,
                               sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Gathers the summed projected relative scores of query-key pairs for the relation bit mask ``RELATIONS``.

    The relations of the bit mask ``CLIPPED`` are clipped to their maximum for every pair, see
    ``_clipped_tile_count``, so that their scores are loaded once per query.

    """
    pair_valid = query_valid & key_valid
    result = tl.where(pair_valid, 0.0, 0.0)
    for relation in tl.static_range(N_RELATIONS):
        if RELATIONS & (1 << relation):
            if CLIPPED & (1 << relation):
                offsets, present = _score_offsets(score_blocks, relation, batch, query_positions, query_valid,
                                                  _relation_maximum_column(relation_metadata, relation, USE_INT64),
                                                  q_length, n_score_columns, sparsity_block_size, N_RELATIONS,
                                                  BLOCK_SPARSE_SCORES, USE_INT64)
                maximum_scores = tl.load(relative_scores + offsets, mask=present, other=0)
                scores = tl.where(pair_valid, tl.cast(maximum_scores, tl.float32), 0.0)
            else:
                rows, _, offset = _relation_rows(query_relations, key_relations, relation_metadata, relation, batch,
                                                 query_positions, query_valid, key_positions, key_valid,
                                                 n_batches, q_length, k_length, USE_INT64)
                offsets, present = _score_offsets(score_blocks, relation, batch, query_positions, pair_valid,
                                                  offset + rows, q_length, n_score_columns, sparsity_block_size,
                                                  N_RELATIONS, BLOCK_SPARSE_SCORES, USE_INT64)
                scores = tl.cast(tl.load(relative_scores + offsets, mask=present, other=0), tl.float32)
            if HAS_RELATION_VALIDITY:
                scores = tl.where(_relation_validity(query_relation_validity, key_relation_validity, relation,
                                                     batch, query_positions, query_valid, key_positions, key_valid,
                                                     n_batches, q_length, k_length), scores, 0.0)
            result += scores
    return result


@triton.jit
def _projected_relative_gradients(gradients, drelative_scores, minimum_gradients, maximum_gradients,
                                  query_relations, key_relations, relation_metadata,
                                  query_relation_validity, key_relation_validity, score_blocks,
                                  batch, query_positions, query_valid, key_positions, key_valid, pair_mask,
                                  n_batches, q_length, k_length, n_score_columns,
                                  N_RELATIONS: tl.constexpr, RELATION_BLOCK_SIZE: tl.constexpr,
                                  UNIQUE_KEY_RELATIONS: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                                  BLOCK_SPARSE_SCORES: tl.constexpr, RELATIONS: tl.constexpr,
                                  CLIPPED: tl.constexpr, sparsity_block_size: tl.constexpr,
                                  USE_INT64: tl.constexpr):
    """Accumulates the gradients of the projected relative scores of a tile for the relation bit mask ``RELATIONS``.

    Queries lie along the rows of the tile. Unclipped pairs are written directly to their table columns. Pairs clipped
    to a relation bound are summed per query and returned, so that every bound is stored once per query after all of its
    keys. The relations of the bit mask ``CLIPPED`` are clipped to their maximum for every pair, see
    ``_clipped_tile_count``.

    The interior gradients of relations with unique key coordinates are stored without atomics. Every table column of
    a query then holds at most one permitted pair, whereas pairs that are not permitted, such as padding keys that
    repeat a coordinate, or that address other blocks through an interval, see ``_score_offsets``, have zero gradients.
    Zero gradients are therefore not stored, as the gradients are zero-initialised, so that they cannot overwrite the
    gradient of a permitted pair.

    """
    relation_slots = tl.arange(0, RELATION_BLOCK_SIZE)
    pair_mask = pair_mask & query_valid & key_valid
    for relation in tl.static_range(N_RELATIONS):
        if RELATIONS & (1 << relation):
            relation_mask = pair_mask
            if HAS_RELATION_VALIDITY:
                relation_mask = relation_mask & _relation_validity(query_relation_validity, key_relation_validity,
                                                                   relation, batch, query_positions, query_valid,
                                                                   key_positions, key_valid, n_batches, q_length,
                                                                   k_length)
            slot = relation_slots[:, None] == relation

            if CLIPPED & (1 << relation):
                maximum_gradients += tl.where(slot, tl.sum(tl.where(relation_mask, gradients, 0.0), axis=1)[None, :],
                                              0.0)
            else:
                rows, span, offset = _relation_rows(query_relations, key_relations, relation_metadata, relation, batch,
                                                    query_positions, query_valid, key_positions, key_valid,
                                                    n_batches, q_length, k_length, USE_INT64)
                minimum_mask = ((rows == 0) | (span == 0)) & relation_mask
                maximum_mask = (rows == span) & (span != 0) & relation_mask
                interior_mask = (rows > 0) & (rows < span) & relation_mask
                minimum_gradients += tl.where(slot, tl.sum(tl.where(minimum_mask, gradients, 0.0), axis=1)[None, :],
                                              0.0)
                maximum_gradients += tl.where(slot, tl.sum(tl.where(maximum_mask, gradients, 0.0), axis=1)[None, :],
                                              0.0)
                offsets, present = _score_offsets(score_blocks, relation, batch, query_positions, interior_mask,
                                                  offset + rows, q_length, n_score_columns, sparsity_block_size,
                                                  N_RELATIONS, BLOCK_SPARSE_SCORES, USE_INT64)
                if UNIQUE_KEY_RELATIONS & (1 << relation):
                    tl.store(drelative_scores + offsets, gradients, mask=present & (gradients != 0))
                else:
                    tl.atomic_add(drelative_scores + offsets, gradients, mask=present, sem="relaxed")
    return minimum_gradients, maximum_gradients


@triton.jit
def _previous_key_rows(schedule_blocks, start, end, tile,
                       TILE_SIZE: tl.constexpr, sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns the position of the preceding row of every row of a key tile, and whether it exists.

    Key tiles are gathered from consecutive schedule entries.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    rows = tl.arange(0, TILE_SIZE)
    previous_entries = start + tile * (TILE_SIZE // sparsity_block_size) + (rows - 1) // sparsity_block_size
    previous_valid = (rows > 0) & (previous_entries < end)
    previous_blocks = tl.cast(tl.load(schedule_blocks + previous_entries, mask=previous_valid, other=0), index_dtype)
    previous_positions = previous_blocks * sparsity_block_size + (rows - 1) % sparsity_block_size
    return previous_positions, previous_valid


@triton.jit
def _key_runs(key_relations, batch, key_positions, key_valid, previous_positions, previous_valid, n_batches,
              k_length, MERGED: tl.constexpr, N_RELATIONS: tl.constexpr, TILE_SIZE: tl.constexpr):
    """Splits the rows of a key tile into runs of consecutive rows with equal coordinates of the merged relations.

    The rows of a run also agree in their validity. Returns the run of every row, whether a row starts its run, and
    the number of runs. Unsorted coordinates only lead to more runs.

    """
    starts = (tl.arange(0, TILE_SIZE) == 0) | (key_valid != previous_valid)
    for relation in tl.static_range(N_RELATIONS):
        if MERGED & (1 << relation):
            relation_row = (relation * n_batches + batch) * k_length
            starts |= (tl.load(key_relations + relation_row + key_positions, mask=key_valid, other=0)
                       != tl.load(key_relations + relation_row + previous_positions, mask=previous_valid, other=0))
    runs = tl.cumsum(tl.cast(starts, tl.int32), axis=0) - 1
    return runs, starts, tl.max(runs) + 1


@triton.jit
def _run_layout(runs, starts, key_valid):
    """Returns the rows of a key tile that belong to and start every run slot, and the slots of valid keys."""
    slots = tl.arange(0, _RUN_CAPACITY)
    members = runs[None, :] == slots[:, None]
    leaders = members & starts[None, :]
    run_valid = tl.max(tl.cast(leaders & key_valid[None, :], tl.int32), axis=1) != 0
    return members, leaders, run_valid


@triton.jit
def _merged_relation_rows(query_relations, key_relations, relation_metadata, relation, batch, query_positions,
                          query_valid, key_positions, key_valid, leaders, run_valid, n_batches, q_length, k_length,
                          USE_INT64: tl.constexpr):
    """Returns the clipped table rows of a relation for every query and run of keys.

    Also returns the table size minus one, the first column, and which query-run pairs are valid.

    """
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    coordinate_dtype = query_relations.dtype.element_ty
    relation_min = tl.cast(tl.load(relation_metadata + relation * 3), coordinate_dtype)
    relation_span = tl.cast(tl.load(relation_metadata + relation * 3 + 1), coordinate_dtype) - relation_min
    relation_offset = tl.cast(tl.load(relation_metadata + relation * 3 + 2), index_dtype)
    relation_row_q = (relation * n_batches + batch) * q_length
    relation_row_k = (relation * n_batches + batch) * k_length
    query_coordinates = tl.load(query_relations + relation_row_q + query_positions, mask=query_valid, other=0)
    run_coordinates = tl.sum(tl.where(leaders, tl.load(key_relations + relation_row_k + key_positions,
                                                       mask=key_valid, other=0)[None, :], 0), axis=1)
    rows = tl.minimum(tl.maximum((query_coordinates - relation_min)[:, None] - run_coordinates[None, :], 0),
                      relation_span)
    valid = query_valid[:, None] & run_valid[None, :]
    return rows, relation_span, relation_offset, valid


@triton.jit
def _merged_relative_scores(relative_scores, query_relations, key_relations, relation_metadata, score_blocks,
                            batch, query_positions, query_valid, key_positions, key_valid, runs, leaders,
                            run_valid, n_batches, q_length, k_length, n_score_columns,
                            MERGED: tl.constexpr, N_RELATIONS: tl.constexpr, BLOCK_SPARSE_SCORES: tl.constexpr,
                            sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Gathers the summed projected scores of the merged relations once per query and run of equal keys.

    The scores are expanded to the query-key pairs of the tile, with queries along its rows, by gathering the score of
    the run of every key.

    """
    result = 0.0
    for relation in tl.static_range(N_RELATIONS):
        if MERGED & (1 << relation):
            rows, _, offset, valid = _merged_relation_rows(
                query_relations, key_relations, relation_metadata, relation, batch, query_positions, query_valid,
                key_positions, key_valid, leaders, run_valid, n_batches, q_length, k_length, USE_INT64)
            offsets, present = _score_offsets(score_blocks, relation, batch, query_positions[:, None], valid,
                                              offset + rows, q_length, n_score_columns, sparsity_block_size,
                                              N_RELATIONS, BLOCK_SPARSE_SCORES, USE_INT64)
            run_scores = tl.cast(tl.load(relative_scores + offsets, mask=present, other=0), tl.float32)
            result += tl.gather(run_scores, runs[None, :] + tl.cast(0 * query_positions[:, None], tl.int32), axis=1)
    return result


@triton.jit
def _merged_relative_gradients(gradients, drelative_scores, minimum_gradients, maximum_gradients,
                               query_relations, key_relations, relation_metadata, score_blocks,
                               batch, query_positions, query_valid, key_positions, key_valid, members, leaders,
                               run_valid, n_batches, q_length, k_length, n_score_columns,
                               MERGED: tl.constexpr, N_RELATIONS: tl.constexpr, RELATION_BLOCK_SIZE: tl.constexpr,
                               BLOCK_SPARSE_SCORES: tl.constexpr, sparsity_block_size: tl.constexpr,
                               USE_INT64: tl.constexpr):
    """Accumulates the gradients of the projected scores of the merged relations once per query and run of keys.

    The gradients of a tile with queries along its rows are summed over every run first. ``query_valid`` marks the
    queries whose gradients are written. Otherwise, as ``_projected_relative_gradients``.

    """
    # The run sums are products with the 0-1 membership matrix. Splitting the gradients into three bfloat16 terms keeps
    # their full float32 precision and the operands in registers.
    members = tl.trans(tl.cast(members, tl.bfloat16))
    gradients_high = tl.cast(gradients, tl.bfloat16)
    residuals = gradients - tl.cast(gradients_high, tl.float32)
    gradients_middle = tl.cast(residuals, tl.bfloat16)
    gradients_low = tl.cast(residuals - tl.cast(gradients_middle, tl.float32), tl.bfloat16)
    run_gradients = tl.dot(gradients_low, members)
    run_gradients = tl.dot(gradients_middle, members, run_gradients)
    run_gradients = tl.dot(gradients_high, members, run_gradients)
    relation_slots = tl.arange(0, RELATION_BLOCK_SIZE)
    for relation in tl.static_range(N_RELATIONS):
        if MERGED & (1 << relation):
            rows, span, offset, valid = _merged_relation_rows(
                query_relations, key_relations, relation_metadata, relation, batch, query_positions, query_valid,
                key_positions, key_valid, leaders, run_valid, n_batches, q_length, k_length, USE_INT64)
            slot = relation_slots[:, None] == relation
            minimum_mask = ((rows == 0) | (span == 0)) & valid
            maximum_mask = (rows == span) & (span != 0) & valid
            interior_mask = (rows > 0) & (rows < span) & valid
            minimum_gradients += tl.where(slot, tl.sum(tl.where(minimum_mask, run_gradients, 0.0), axis=1)[None, :],
                                          0.0)
            maximum_gradients += tl.where(slot, tl.sum(tl.where(maximum_mask, run_gradients, 0.0), axis=1)[None, :],
                                          0.0)
            offsets, present = _score_offsets(score_blocks, relation, batch, query_positions[:, None], interior_mask,
                                              offset + rows, q_length, n_score_columns, sparsity_block_size,
                                              N_RELATIONS, BLOCK_SPARSE_SCORES, USE_INT64)
            tl.atomic_add(drelative_scores + offsets, run_gradients, mask=present, sem="relaxed")
    return minimum_gradients, maximum_gradients


@triton.jit
def _merged_relative_logits(relative_scores, query_relations, key_relations, relation_metadata, score_blocks,
                            schedule_blocks, batch, start, end, key_tile, query_positions, query_valid, key_positions,
                            key_valid, n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                            MERGED: tl.constexpr, N_RELATIONS: tl.constexpr, BLOCK_SPARSE_SCORES: tl.constexpr,
                            KEY_TILE_SIZE: tl.constexpr, sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Returns the base-2 logit terms of the merged relations of a tile with queries along its rows.

    Tiles whose keys form at most ``_RUN_CAPACITY`` runs of equal coordinates gather the scores once per query and
    run, see ``_merged_relative_scores``; other tiles gather them for every pair. Also returns the layout of the runs
    of the keys, see ``_run_layout``, and their number. Merged tiles never need relation-validity masks, see
    ``_validity_tiles``.

    """
    q_length = n_seq_blocks_q * sparsity_block_size
    k_length = n_seq_blocks_k * sparsity_block_size
    previous_positions, previous_valid = _previous_key_rows(schedule_blocks, start, end, key_tile, KEY_TILE_SIZE,
                                                            sparsity_block_size, USE_INT64)
    runs, starts, n_runs = _key_runs(key_relations, batch, key_positions, key_valid, previous_positions,
                                     previous_valid, n_batches, k_length, MERGED, N_RELATIONS, KEY_TILE_SIZE)
    members, leaders, run_valid = _run_layout(runs, starts, key_valid)
    if n_runs <= _RUN_CAPACITY:
        scores = _merged_relative_scores(relative_scores, query_relations, key_relations, relation_metadata,
                                         score_blocks, batch, query_positions, query_valid, key_positions, key_valid,
                                         runs, leaders, run_valid, n_batches, q_length, k_length, n_score_columns,
                                         MERGED, N_RELATIONS, BLOCK_SPARSE_SCORES, sparsity_block_size, USE_INT64)
    else:
        scores = _projected_relative_scores(relative_scores, query_relations, key_relations, relation_metadata,
                                            None, None, score_blocks, batch,
                                            query_positions[:, None], query_valid[:, None],
                                            key_positions[None, :], key_valid[None, :],
                                            n_batches, q_length, k_length, n_score_columns,
                                            N_RELATIONS, False, BLOCK_SPARSE_SCORES, MERGED, 0,
                                            sparsity_block_size, USE_INT64)
    return scores * _LOG2E, members, leaders, run_valid, n_runs


@triton.jit
def _add_merged_logits(logits, merged_logits, MASKED: tl.constexpr, BLOCK_SPARSE_SCORES: tl.constexpr):
    """Adds the logit terms of the merged relations to the logits of a tile.

    Masked pairs keep their logits of -inf, as the merged scores of relations addressed through intervals may come
    from other blocks, see ``_score_offsets``.

    """
    if MASKED and BLOCK_SPARSE_SCORES > 1:
        logits = tl.where(logits == float("-inf"), logits, logits + merged_logits)
    else:
        logits += merged_logits
    return logits


@triton.jit
def _tile_activity(bits, mask_bits, bias_bits, program_sub_blocks,
                   MASKED: tl.constexpr, HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Returns which pairs of a tile lie in the attention layout and in blocks of the explicit mask and the bias.

    Program blocks lie along the tile rows and schedule entries along its columns. Also returns whether any pair lies
    in a block of the explicit mask and of the bias. Without ``MASKED``, every pair is active and the explicit mask is
    not needed, so that placeholders are returned.

    """
    if MASKED:
        active = _pair_bits(bits, program_sub_blocks)
    else:
        active = 0
    if MASKED and HAS_MASK:
        mask_blocks = _pair_bits(mask_bits, program_sub_blocks)
        any_mask_block = tl.max(mask_bits) != 0
    else:
        mask_blocks = 0
        any_mask_block = 0
    if HAS_BIAS:
        bias_blocks = _pair_bits(bias_bits, program_sub_blocks)
        any_bias_block = tl.max(bias_bits) != 0
    else:
        bias_blocks = 0
        any_bias_block = 0
    return active, mask_blocks, any_mask_block, bias_blocks, any_bias_block


@triton.jit
def _pair_bits(bits, program_sub_blocks):
    """Returns the bits of schedule entries along the tile columns for the program blocks along its rows."""
    return ((bits[None, :] >> program_sub_blocks[:, None]) & 1) != 0


@triton.jit
def _attention_logits(products, active, mask_blocks, any_mask_block, bias_blocks, any_bias_block, batch,
                      query_blocks, query_positions, query_valid, key_blocks, key_positions, key_valid,
                      attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                      relative_scores, query_relations, key_relations, relation_metadata,
                      query_relation_validity, key_relation_validity, score_blocks,
                      n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                      scale,
                      HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr,
                      HAS_CAUSAL: tl.constexpr, HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                      BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                      MASKED: tl.constexpr, RELATIONS: tl.constexpr, CLIPPED: tl.constexpr,
                      sparsity_block_size: tl.constexpr, USE_INT64: tl.constexpr):
    """Completes query-key products to base-2 attention logits.

    Query and key operands are broadcast against each other, so queries may lie along either tile dimension. With
    ``MASKED``, pairs outside the attention layout (``active``), future keys, padding positions, and positions of the
    explicit mask are masked; otherwise every pair of the tile must be valid and outside the blocks of the explicit
    mask. ``mask_blocks`` and ``bias_blocks`` mark the pairs within blocks of the explicit mask and the bias, and
    ``any_mask_block`` and ``any_bias_block`` whether there are any. The explicit mask is applied last, so that masked
    positions remain ignored regardless of their bias. Only the relative terms of the bit mask ``RELATIONS`` are added.

    """
    logits = products * (scale * _LOG2E)

    if HAS_BIAS:
        # Tiles without bias blocks skip the bias. The branch also keeps Triton's software pipeliner from turning the
        # gathered loads of sub-word elements into asynchronous copies, which need at least four bytes.
        if any_bias_block:
            bias_offsets, bias_present = _pair_offsets(batch, query_blocks, query_positions, bias_blocks,
                                                       key_blocks, key_positions, key_valid, packed_indices_bias,
                                                       n_seq_blocks_q, n_seq_blocks_k, sparsity_block_size,
                                                       USE_INT64)
            bias = tl.load(attention_bias + bias_offsets, mask=bias_present, other=0)
            logits += tl.cast(bias, tl.float32) * _LOG2E

    if HAS_RELATIVE:
        logits += _projected_relative_scores(relative_scores, query_relations, key_relations, relation_metadata,
                                             query_relation_validity, key_relation_validity, score_blocks, batch,
                                             query_positions, query_valid, key_positions, key_valid,
                                             n_batches, n_seq_blocks_q * sparsity_block_size,
                                             n_seq_blocks_k * sparsity_block_size, n_score_columns,
                                             N_RELATIONS, HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES,
                                             RELATIONS, CLIPPED, sparsity_block_size, USE_INT64) * _LOG2E

    if MASKED:
        keep = active
        if HAS_CAUSAL:
            sequence_length = tl.load(causal_lengths + batch)
            keep = (keep & (key_positions <= query_positions) & (query_positions < sequence_length)
                    & (key_positions < sequence_length))
        logits = tl.where(keep, logits, float("-inf"))

        if HAS_MASK:
            # Compact mask layouts only hold the blocks that refine the attention layout, so most tiles have none.
            # As for the bias, the branch also keeps the gathered loads out of Triton's software pipeliner.
            if any_mask_block:
                mask_offsets, mask_present = _pair_offsets(batch, query_blocks, query_positions, mask_blocks,
                                                           key_blocks, key_positions, key_valid,
                                                           packed_indices_mask, n_seq_blocks_q, n_seq_blocks_k,
                                                           sparsity_block_size, USE_INT64)
                masked = tl.load(attention_mask + mask_offsets, mask=mask_present, other=0)
                logits = tl.where(masked != 0, float("-inf"), logits)

    return logits


@triton.jit
def _schedule_range(schedule_offsets, schedule_plain, batch, tile, n_seq_blocks, causal_lengths,
                    SCHEDULE_BLOCKS: tl.constexpr, TILE_SIZE: tl.constexpr, INNER_TILE_SIZE: tl.constexpr,
                    sparsity_block_size: tl.constexpr, HAS_CAUSAL: tl.constexpr, PLAIN_NEEDS_LENGTH: tl.constexpr):
    """Returns the schedule entries of a program tile, the number of inner tiles, and the number of plain ones.

    Also returns the position of the program blocks within their schedule group, which selects their activity bits.
    ``PLAIN_NEEDS_LENGTH`` restricts plain tiles to programs whose rows lie within the causal length.

    """
    tile_blocks: tl.constexpr = TILE_SIZE // sparsity_block_size
    inner_blocks: tl.constexpr = INNER_TILE_SIZE // sparsity_block_size
    tiles_per_group: tl.constexpr = SCHEDULE_BLOCKS // tile_blocks
    schedule_row = batch * tl.cdiv(n_seq_blocks, SCHEDULE_BLOCKS) + tile // tiles_per_group
    start = tl.load(schedule_offsets + schedule_row)
    end = tl.load(schedule_offsets + schedule_row + 1)
    n_plain = tl.load(schedule_plain + schedule_row * 2 + (1 if HAS_CAUSAL else 0)) // inner_blocks
    if HAS_CAUSAL and PLAIN_NEEDS_LENGTH:
        sequence_length = tl.load(causal_lengths + batch)
        n_plain = tl.where((tile + 1) * TILE_SIZE <= sequence_length, n_plain, 0)
    group_shift = (tile % tiles_per_group) * tile_blocks
    return start, end, tl.cdiv(end - start, inner_blocks), n_plain, group_shift


@triton.jit
def _forward_tile(key_tile, m_i, l_i, accumulator, q_tile, batch, chunk_v, start, end,
                  query_blocks, query_positions, query_valid, query_sub_blocks,
                  q, k, v, packed_indices_q, packed_indices_k, packed_indices_v,
                  schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                  attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                  relative_scores, query_relations, key_relations, relation_metadata,
                  query_relation_validity, key_relation_validity, score_blocks,
                  n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                  scale, n_head_blocks_qk: tl.constexpr, n_head_blocks_v: tl.constexpr,
                  sparsity_block_size: tl.constexpr, HEAD_TILE_QK: tl.constexpr, HEAD_TILE_V: tl.constexpr,
                  N_CHUNKS_QK: tl.constexpr, ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr,
                  HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CAUSAL: tl.constexpr,
                  HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                  BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                  TRITON_BLOCK_SIZE_Q: tl.constexpr, TRITON_BLOCK_SIZE_K: tl.constexpr, MASKED: tl.constexpr,
                  CLIPPED: tl.constexpr):
    """Processes one key tile of the online softmax."""
    key_blocks, key_positions, key_valid, bits, mask_bits, bias_bits = _scheduled_tile(
        schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits, start, end, key_tile,
        TRITON_BLOCK_SIZE_K, sparsity_block_size, USE_INT64, MASKED, MASKED and HAS_MASK, HAS_BIAS)
    products, _ = _head_product(q_tile, q, packed_indices_q, query_blocks, query_positions, query_valid,
                                n_seq_blocks_q, k, packed_indices_k, key_blocks, key_positions, key_valid,
                                n_seq_blocks_k, batch, n_head_blocks_qk, 0,
                                TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, HEAD_TILE_QK, N_CHUNKS_QK, False,
                                sparsity_block_size, ROW_COMPLETE, USE_INT64)
    active, mask_blocks, any_mask_block, bias_blocks, any_bias_block = _tile_activity(
        bits, mask_bits, bias_bits, query_sub_blocks, MASKED, HAS_MASK, HAS_BIAS)
    logits = _attention_logits(products, active, mask_blocks, any_mask_block, bias_blocks, any_bias_block,
                               batch,
                               query_blocks[:, None], query_positions[:, None], query_valid[:, None],
                               key_blocks[None, :], key_positions[None, :], key_valid[None, :],
                               attention_mask, packed_indices_mask, attention_bias, packed_indices_bias,
                               causal_lengths, relative_scores, query_relations, key_relations, relation_metadata,
                               query_relation_validity, key_relation_validity, score_blocks,
                               n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns, scale,
                               HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE,
                               HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES, N_RELATIONS, MASKED,
                               (1 << N_RELATIONS) - 1, CLIPPED, sparsity_block_size, USE_INT64)

    # Online softmax update. Rows without any attended position so far keep a zero numerator and denominator.
    m_ij = tl.maximum(m_i, tl.max(logits, axis=1))
    m_shift = tl.where(m_ij == float("-inf"), 0.0, m_ij)
    alpha = tl.math.exp2(m_i - m_shift)
    p = tl.math.exp2(logits - m_shift[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)

    v_offsets, v_mask = _operand_offsets(packed_indices_v, batch, key_blocks, key_positions, key_valid,
                                         n_seq_blocks_k, n_head_blocks_v, chunk_v,
                                         HEAD_TILE_V, sparsity_block_size, ROW_COMPLETE, USE_INT64)
    v_tile = tl.load(v + v_offsets, mask=v_mask, other=0)
    accumulator = accumulator * alpha[:, None] + tl.dot(tl.cast(p, v_tile.dtype), v_tile)
    return m_ij, l_i, accumulator


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention_forward", block_size_names=_TILE_SIZE_NAMES),
    key=_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit(do_not_specialize=_UNSPECIALISED)
def flash_attention_kernel(q, k, v, o, lse,
                           packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                           schedule_offsets, schedule_blocks, schedule_bits, schedule_mask_bits,
                           schedule_bias_bits, schedule_plain,
                           attention_mask, packed_indices_mask, attention_bias, packed_indices_bias,
                           causal_lengths,
                           relative_scores, query_relations, key_relations, relation_metadata,
                           query_coordinate_floors, key_coordinate_ceilings,
                           query_relation_validity, key_relation_validity,
                           query_invalid_blocks, key_invalid_blocks,
                           score_blocks,
                           n_batches, n_seq_blocks_q, n_seq_blocks_k,
                           n_score_columns, scale, n_seq_blocks_bucket, n_schedule_entries_bucket,
                           n_head_blocks_qk: tl.constexpr, n_head_blocks_v: tl.constexpr,
                           sparsity_block_size: tl.constexpr,
                           SCHEDULE_BLOCKS: tl.constexpr,
                           HEAD_TILE_QK: tl.constexpr, HEAD_TILE_V: tl.constexpr,
                           N_CHUNKS_QK: tl.constexpr, N_CHUNKS_V: tl.constexpr,
                           ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr,
                           HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CAUSAL: tl.constexpr,
                           HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                           BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                           CLIPPED_RELATIONS: tl.constexpr, SCORE_LAYOUT: tl.constexpr,
                           TRITON_BLOCK_SIZE_Q: tl.constexpr, TRITON_BLOCK_SIZE_K: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    # Start with the last query tiles, which attend to the most keys under causal layouts.
    n_tiles = tl.cdiv(n_seq_blocks_q * sparsity_block_size, TRITON_BLOCK_SIZE_Q)
    pid_tile = n_tiles - 1 - tl.cast(tl.program_id(axis=1), index_dtype)
    pid_chunk = tl.program_id(axis=2)

    query_blocks, query_positions, query_valid = _sequence_tile(pid_tile, n_seq_blocks_q, TRITON_BLOCK_SIZE_Q,
                                                                sparsity_block_size, USE_INT64)
    start, end, n_key_tiles, n_plain, group_shift = _schedule_range(
        schedule_offsets, schedule_plain, pid_bat, pid_tile, n_seq_blocks_q, causal_lengths,
        SCHEDULE_BLOCKS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, sparsity_block_size, HAS_CAUSAL, True)
    query_sub_blocks = group_shift + tl.arange(0, TRITON_BLOCK_SIZE_Q) // sparsity_block_size

    if N_CHUNKS_QK > 1:
        q_tile = 0
    else:
        q_offsets, q_mask = _operand_offsets(packed_indices_q, pid_bat, query_blocks, query_positions, query_valid,
                                             n_seq_blocks_q, n_head_blocks_qk, 0,
                                             HEAD_TILE_QK, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        q_tile = tl.load(q + q_offsets, mask=q_mask, other=0)

    m_i = tl.full((TRITON_BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((TRITON_BLOCK_SIZE_Q,), dtype=tl.float32)
    accumulator = tl.zeros((TRITON_BLOCK_SIZE_Q, HEAD_TILE_V), dtype=tl.float32)

    # Tiles with positions without valid relations come first, see ``_validity_tiles``. All others skip the
    # validity masks.
    n_validity = 0
    if HAS_RELATION_VALIDITY:
        n_validity = _validity_tiles(schedule_blocks, key_invalid_blocks, query_invalid_blocks, pid_bat, pid_tile,
                                     start, end, n_key_tiles, n_seq_blocks_q, n_seq_blocks_k,
                                     TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, sparsity_block_size, False)
        for key_tile in range(0, n_validity):
            m_i, l_i, accumulator = _forward_tile(
                key_tile, m_i, l_i, accumulator, q_tile, pid_bat, pid_chunk, start, end,
                query_blocks, query_positions, query_valid, query_sub_blocks,
                q, k, v, packed_indices_q, packed_indices_k, packed_indices_v,
                schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                relative_scores, query_relations, key_relations, relation_metadata,
                query_relation_validity, key_relation_validity, score_blocks,
                n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                scale, n_head_blocks_qk, n_head_blocks_v,
                sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, ROW_COMPLETE, USE_INT64,
                HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES,
                N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, True, 0)

    # Distant plain tiles in which every pair is clipped to the maximum of the clipped relations come first.
    n_clipped = 0
    if CLIPPED_RELATIONS != 0:
        first_query_block = pid_tile * (TRITON_BLOCK_SIZE_Q // sparsity_block_size)
        n_clipped = _clipped_tile_count(schedule_blocks, query_coordinate_floors, key_coordinate_ceilings,
                                        relation_metadata, pid_bat, first_query_block, start, n_plain, n_batches,
                                        n_seq_blocks_q, n_seq_blocks_k, CLIPPED_RELATIONS, N_RELATIONS,
                                        TRITON_BLOCK_SIZE_K, sparsity_block_size, False)
        for key_tile in range(n_validity, n_clipped):
            m_i, l_i, accumulator = _forward_tile(
                key_tile, m_i, l_i, accumulator, q_tile, pid_bat, pid_chunk, start, end,
                query_blocks, query_positions, query_valid, query_sub_blocks,
                q, k, v, packed_indices_q, packed_indices_k, packed_indices_v,
                schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                relative_scores, query_relations, key_relations, relation_metadata,
                query_relation_validity, key_relation_validity, score_blocks,
                n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                scale, n_head_blocks_qk, n_head_blocks_v,
                sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, ROW_COMPLETE, USE_INT64,
                HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
                N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, False, CLIPPED_RELATIONS)
    plain_start = n_clipped
    masked_start = n_plain
    if HAS_RELATION_VALIDITY:
        plain_start = tl.maximum(plain_start, n_validity)
        masked_start = tl.maximum(masked_start, n_validity)
    for key_tile in range(plain_start, n_plain):
        m_i, l_i, accumulator = _forward_tile(
            key_tile, m_i, l_i, accumulator, q_tile, pid_bat, pid_chunk, start, end,
            query_blocks, query_positions, query_valid, query_sub_blocks,
            q, k, v, packed_indices_q, packed_indices_k, packed_indices_v,
            schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
            attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
            relative_scores, query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            scale, n_head_blocks_qk, n_head_blocks_v,
            sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, ROW_COMPLETE, USE_INT64,
            HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
            N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, False, 0)
    for key_tile in range(masked_start, n_key_tiles):
        m_i, l_i, accumulator = _forward_tile(
            key_tile, m_i, l_i, accumulator, q_tile, pid_bat, pid_chunk, start, end,
            query_blocks, query_positions, query_valid, query_sub_blocks,
            q, k, v, packed_indices_q, packed_indices_k, packed_indices_v,
            schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
            attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
            relative_scores, query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            scale, n_head_blocks_qk, n_head_blocks_v,
            sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, ROW_COMPLETE, USE_INT64,
            HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
            N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, True, 0)

    # Final normalisation
    has_attention = l_i != 0
    l_safe = tl.where(has_attention, l_i, 1.0)
    result = tl.where(has_attention[:, None], accumulator / l_safe[:, None], 0.0)
    o_offsets, o_mask = _operand_offsets(packed_indices_o, pid_bat, query_blocks, query_positions, query_valid,
                                         n_seq_blocks_q, n_head_blocks_v, pid_chunk,
                                         HEAD_TILE_V, sparsity_block_size, ROW_COMPLETE, USE_INT64)
    tl.store(o + o_offsets, tl.cast(result, o.dtype.element_ty), mask=o_mask)

    lse_value = tl.where(has_attention, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(lse + pid_bat * n_seq_blocks_q * sparsity_block_size + query_positions, lse_value,
             mask=query_valid & (pid_chunk == 0))


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("row_wise"),
    key=["sparsity_block_size", "HEAD_TILE_V", "ROW_COMPLETE", "USE_INT64"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
)
@triton.jit(do_not_specialize=["n_seq_blocks_q"])
def flash_attention_kernel_bwd_preprocess(o, do, delta,
                                          packed_indices_o,
                                          n_seq_blocks_q, n_head_blocks_v: tl.constexpr,
                                          sparsity_block_size: tl.constexpr,
                                          HEAD_TILE_V: tl.constexpr,
                                          ROW_COMPLETE: tl.constexpr,
                                          USE_INT64: tl.constexpr,
                                          TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_tile = tl.program_id(axis=1)

    query_blocks, query_positions, query_valid = _sequence_tile(pid_tile, n_seq_blocks_q, TRITON_BLOCK_SIZE,
                                                                sparsity_block_size, USE_INT64)
    delta_acc = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)
    for chunk in range(0, tl.cdiv(n_head_blocks_v * sparsity_block_size, HEAD_TILE_V)):
        offsets, mask = _operand_offsets(packed_indices_o, pid_bat, query_blocks, query_positions, query_valid,
                                         n_seq_blocks_q, n_head_blocks_v, chunk,
                                         HEAD_TILE_V, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        blk_o = tl.cast(tl.load(o + offsets, mask=mask, other=0), tl.float32)
        blk_do = tl.cast(tl.load(do + offsets, mask=mask, other=0), tl.float32)
        delta_acc += tl.sum(blk_o * blk_do, axis=1)

    tl.store(delta + pid_bat * n_seq_blocks_q * sparsity_block_size + query_positions, delta_acc, mask=query_valid)


@triton.jit
def _backward_dkdv_tile(query_tile, dk_accumulator, dv_accumulator, k_tile, v_tile, batch, chunk_qk, chunk_v,
                        start, end,
                        key_blocks, key_positions, key_valid, key_sub_blocks,
                        q, k, v, do, lse, delta,
                        packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                        schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                        attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                        relative_scores, query_relations, key_relations, relation_metadata,
                        query_relation_validity, key_relation_validity, score_blocks,
                        n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                        scale, n_head_blocks_qk: tl.constexpr, n_head_blocks_v: tl.constexpr,
                        sparsity_block_size: tl.constexpr, HEAD_TILE_QK: tl.constexpr, HEAD_TILE_V: tl.constexpr,
                        N_CHUNKS_QK: tl.constexpr, N_CHUNKS_V: tl.constexpr,
                        ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr,
                        HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CAUSAL: tl.constexpr,
                        HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                        BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                        TRITON_BLOCK_SIZE_Q: tl.constexpr, TRITON_BLOCK_SIZE_K: tl.constexpr,
                        MASKED: tl.constexpr, CLIPPED: tl.constexpr):
    """Accumulates the key and value gradients of one query tile, with keys along the rows."""
    query_blocks, query_positions, query_valid, bits, mask_bits, bias_bits = _scheduled_tile(
        schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits, start, end, query_tile,
        TRITON_BLOCK_SIZE_Q, sparsity_block_size, USE_INT64, MASKED, MASKED and HAS_MASK, HAS_BIAS)
    products, q_chunk = _head_product(k_tile, k, packed_indices_k, key_blocks, key_positions, key_valid,
                                      n_seq_blocks_k, q, packed_indices_q, query_blocks, query_positions,
                                      query_valid, n_seq_blocks_q, batch, n_head_blocks_qk, chunk_qk,
                                      TRITON_BLOCK_SIZE_K, TRITON_BLOCK_SIZE_Q, HEAD_TILE_QK, N_CHUNKS_QK, True,
                                      sparsity_block_size, ROW_COMPLETE, USE_INT64)
    active, mask_blocks, any_mask_block, bias_blocks, any_bias_block = _tile_activity(
        bits, mask_bits, bias_bits, key_sub_blocks, MASKED, HAS_MASK, HAS_BIAS)
    logits = _attention_logits(products, active, mask_blocks, any_mask_block, bias_blocks, any_bias_block,
                               batch,
                               query_blocks[None, :], query_positions[None, :], query_valid[None, :],
                               key_blocks[:, None], key_positions[:, None], key_valid[:, None],
                               attention_mask, packed_indices_mask, attention_bias, packed_indices_bias,
                               causal_lengths, relative_scores, query_relations, key_relations, relation_metadata,
                               query_relation_validity, key_relation_validity, score_blocks,
                               n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns, scale,
                               HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE,
                               HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES, N_RELATIONS, MASKED,
                               (1 << N_RELATIONS) - 1, CLIPPED, sparsity_block_size, USE_INT64)

    # Recompute the probabilities from the saved log-sum-exp. Queries without attention have none.
    row_offsets = batch * n_seq_blocks_q * sparsity_block_size + query_positions
    m = tl.load(lse + row_offsets, mask=query_valid, other=float("-inf"))
    delta_rows = tl.load(delta + row_offsets, mask=query_valid, other=0.0)
    valid_lse = m != float("-inf")
    p = tl.math.exp2(logits - tl.where(valid_lse, m, 0.0)[None, :])
    p = tl.where(valid_lse[None, :], p, 0.0)

    dp, do_chunk = _head_product(v_tile, v, packed_indices_v, key_blocks, key_positions, key_valid,
                                 n_seq_blocks_k, do, packed_indices_o, query_blocks, query_positions, query_valid,
                                 n_seq_blocks_q, batch, n_head_blocks_v, chunk_v,
                                 TRITON_BLOCK_SIZE_K, TRITON_BLOCK_SIZE_Q, HEAD_TILE_V, N_CHUNKS_V, True,
                                 sparsity_block_size, ROW_COMPLETE, USE_INT64)
    dv_accumulator += tl.dot(tl.cast(p, do_chunk.dtype), do_chunk)
    ds = p * (dp - delta_rows[None, :])
    dk_accumulator += tl.dot(tl.cast(ds, q_chunk.dtype), q_chunk)
    return dk_accumulator, dv_accumulator


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention_backward_dkdv", block_size_names=_TILE_SIZE_NAMES),
    key=_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit(do_not_specialize=_UNSPECIALISED)
def flash_attention_kernel_bwd_dkdv(q, k, v, do, dk, dv, lse, delta,
                                    packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                                    schedule_offsets, schedule_blocks, schedule_bits, schedule_mask_bits,
                                    schedule_bias_bits, schedule_plain,
                                    attention_mask, packed_indices_mask, attention_bias, packed_indices_bias,
                                    causal_lengths,
                                    relative_scores, query_relations, key_relations, relation_metadata,
                                    query_coordinate_floors, key_coordinate_ceilings,
                                    query_relation_validity, key_relation_validity,
                                    query_invalid_blocks, key_invalid_blocks,
                                    score_blocks,
                                    n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                    n_score_columns, scale, n_seq_blocks_bucket, n_schedule_entries_bucket,
                                    n_head_blocks_qk: tl.constexpr, n_head_blocks_v: tl.constexpr,
                                    sparsity_block_size: tl.constexpr,
                                    SCHEDULE_BLOCKS: tl.constexpr,
                                    HEAD_TILE_QK: tl.constexpr, HEAD_TILE_V: tl.constexpr,
                                    N_CHUNKS_QK: tl.constexpr, N_CHUNKS_V: tl.constexpr,
                                    ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr,
                                    HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CAUSAL: tl.constexpr,
                                    HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                                    BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                                    CLIPPED_RELATIONS: tl.constexpr, SCORE_LAYOUT: tl.constexpr,
                                    TRITON_BLOCK_SIZE_Q: tl.constexpr,
                                    TRITON_BLOCK_SIZE_K: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    # Start with the first key tiles, which are attended by the most queries under causal layouts.
    pid_tile = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_chunk = tl.program_id(axis=2)
    # The programs of a key tile cover the chunks of both head dimensions, one of which may have fewer.
    chunk_qk = tl.minimum(pid_chunk, N_CHUNKS_QK - 1)
    chunk_v = tl.minimum(pid_chunk, N_CHUNKS_V - 1)

    key_blocks, key_positions, key_valid = _sequence_tile(pid_tile, n_seq_blocks_k, TRITON_BLOCK_SIZE_K,
                                                          sparsity_block_size, USE_INT64)
    start, end, n_query_tiles, n_plain, group_shift = _schedule_range(
        schedule_offsets, schedule_plain, pid_bat, pid_tile, n_seq_blocks_k, causal_lengths,
        SCHEDULE_BLOCKS, TRITON_BLOCK_SIZE_K, TRITON_BLOCK_SIZE_Q, sparsity_block_size, HAS_CAUSAL, False)
    key_sub_blocks = group_shift + tl.arange(0, TRITON_BLOCK_SIZE_K) // sparsity_block_size

    if N_CHUNKS_QK > 1:
        k_tile = 0
    else:
        k_offsets, k_mask = _operand_offsets(packed_indices_k, pid_bat, key_blocks, key_positions, key_valid,
                                             n_seq_blocks_k, n_head_blocks_qk, 0,
                                             HEAD_TILE_QK, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        k_tile = tl.load(k + k_offsets, mask=k_mask, other=0)
    if N_CHUNKS_V > 1:
        v_tile = 0
    else:
        v_offsets, v_mask = _operand_offsets(packed_indices_v, pid_bat, key_blocks, key_positions, key_valid,
                                             n_seq_blocks_k, n_head_blocks_v, 0,
                                             HEAD_TILE_V, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        v_tile = tl.load(v + v_offsets, mask=v_mask, other=0)

    dk_accumulator = tl.zeros((TRITON_BLOCK_SIZE_K, HEAD_TILE_QK), dtype=tl.float32)
    dv_accumulator = tl.zeros((TRITON_BLOCK_SIZE_K, HEAD_TILE_V), dtype=tl.float32)

    # Tiles with positions without valid relations come last, see ``_validity_tiles``. All others skip the validity
    # masks.
    n_fast = n_query_tiles
    if HAS_RELATION_VALIDITY:
        n_fast = _validity_tiles(schedule_blocks, query_invalid_blocks, key_invalid_blocks, pid_bat, pid_tile, start,
                                 end, n_query_tiles, n_seq_blocks_k, n_seq_blocks_q, TRITON_BLOCK_SIZE_K,
                                 TRITON_BLOCK_SIZE_Q, sparsity_block_size, True)
    # Distant plain tiles in which every pair is clipped to the maximum of the clipped relations come first.
    n_clipped = 0
    if CLIPPED_RELATIONS != 0:
        last_key_block = tl.minimum((pid_tile + 1) * (TRITON_BLOCK_SIZE_K // sparsity_block_size), n_seq_blocks_k) - 1
        n_clipped = _clipped_tile_count(schedule_blocks, query_coordinate_floors, key_coordinate_ceilings,
                                        relation_metadata, pid_bat, last_key_block, start, n_plain, n_batches,
                                        n_seq_blocks_q, n_seq_blocks_k, CLIPPED_RELATIONS, N_RELATIONS,
                                        TRITON_BLOCK_SIZE_Q, sparsity_block_size, True)
        clipped_end = n_clipped
        if HAS_RELATION_VALIDITY:
            clipped_end = tl.minimum(clipped_end, n_fast)
        for query_tile in range(0, clipped_end):
            dk_accumulator, dv_accumulator = _backward_dkdv_tile(
                query_tile, dk_accumulator, dv_accumulator, k_tile, v_tile, pid_bat, chunk_qk, chunk_v, start, end,
                key_blocks, key_positions, key_valid, key_sub_blocks,
                q, k, v, do, lse, delta,
                packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                relative_scores, query_relations, key_relations, relation_metadata,
                query_relation_validity, key_relation_validity, score_blocks,
                n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                scale, n_head_blocks_qk, n_head_blocks_v,
                sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
                HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
                N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, False, CLIPPED_RELATIONS)
    plain_start = n_clipped
    plain_end = n_plain
    if HAS_RELATION_VALIDITY:
        plain_start = tl.minimum(plain_start, n_fast)
        plain_end = tl.minimum(plain_end, n_fast)
    for query_tile in range(plain_start, plain_end):
        dk_accumulator, dv_accumulator = _backward_dkdv_tile(
            query_tile, dk_accumulator, dv_accumulator, k_tile, v_tile, pid_bat, chunk_qk, chunk_v, start, end,
            key_blocks, key_positions, key_valid, key_sub_blocks,
            q, k, v, do, lse, delta,
            packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
            schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
            attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
            relative_scores, query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            scale, n_head_blocks_qk, n_head_blocks_v,
            sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
            HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
            N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, False, 0)
    for query_tile in range(plain_end, n_fast):
        dk_accumulator, dv_accumulator = _backward_dkdv_tile(
            query_tile, dk_accumulator, dv_accumulator, k_tile, v_tile, pid_bat, chunk_qk, chunk_v, start, end,
            key_blocks, key_positions, key_valid, key_sub_blocks,
            q, k, v, do, lse, delta,
            packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
            schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
            attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
            relative_scores, query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            scale, n_head_blocks_qk, n_head_blocks_v,
            sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
            HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
            N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, True, 0)
    if HAS_RELATION_VALIDITY:
        for query_tile in range(n_fast, n_query_tiles):
            dk_accumulator, dv_accumulator = _backward_dkdv_tile(
                query_tile, dk_accumulator, dv_accumulator, k_tile, v_tile, pid_bat, chunk_qk, chunk_v, start, end,
                key_blocks, key_positions, key_valid, key_sub_blocks,
                q, k, v, do, lse, delta,
                packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                relative_scores, query_relations, key_relations, relation_metadata,
                query_relation_validity, key_relation_validity, score_blocks,
                n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                scale, n_head_blocks_qk, n_head_blocks_v,
                sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
                HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES,
                N_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, True, 0)

    dk_offsets, dk_mask = _operand_offsets(packed_indices_k, pid_bat, key_blocks, key_positions, key_valid,
                                           n_seq_blocks_k, n_head_blocks_qk, chunk_qk,
                                           HEAD_TILE_QK, sparsity_block_size, ROW_COMPLETE, USE_INT64)
    tl.store(dk + dk_offsets, tl.cast(dk_accumulator * scale, dk.dtype.element_ty),
             mask=dk_mask & (pid_chunk < N_CHUNKS_QK))
    dv_offsets, dv_mask = _operand_offsets(packed_indices_v, pid_bat, key_blocks, key_positions, key_valid,
                                           n_seq_blocks_k, n_head_blocks_v, chunk_v,
                                           HEAD_TILE_V, sparsity_block_size, ROW_COMPLETE, USE_INT64)
    tl.store(dv + dv_offsets, tl.cast(dv_accumulator, dv.dtype.element_ty),
             mask=dv_mask & (pid_chunk < N_CHUNKS_V))


@triton.jit
def _backward_dq_tile(key_tile, dq_accumulator, minimum_gradients, maximum_gradients, q_tile, do_tile,
                      lse_rows, valid_lse, delta_rows, batch, chunk, start, end,
                      query_blocks, query_positions, query_valid, query_sub_blocks,
                      q, k, v, do, dbias, drelative_scores,
                      packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                      schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                      attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                      relative_scores, query_relations, key_relations, relation_metadata,
                      query_relation_validity, key_relation_validity, score_blocks,
                      n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                      scale, n_head_blocks_qk: tl.constexpr, n_head_blocks_v: tl.constexpr,
                      sparsity_block_size: tl.constexpr, HEAD_TILE_QK: tl.constexpr, HEAD_TILE_V: tl.constexpr,
                      N_CHUNKS_QK: tl.constexpr, N_CHUNKS_V: tl.constexpr,
                      ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr,
                      HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CAUSAL: tl.constexpr,
                      HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                      BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                      RELATION_BLOCK_SIZE: tl.constexpr, UNIQUE_KEY_RELATIONS: tl.constexpr,
                      TRITON_BLOCK_SIZE_Q: tl.constexpr, TRITON_BLOCK_SIZE_K: tl.constexpr,
                      MASKED: tl.constexpr, CLIPPED: tl.constexpr, MERGED: tl.constexpr):
    """Accumulates the query gradient of one key tile and writes the gradients of its scores."""
    key_blocks, key_positions, key_valid, bits, mask_bits, bias_bits = _scheduled_tile(
        schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits, start, end, key_tile,
        TRITON_BLOCK_SIZE_K, sparsity_block_size, USE_INT64, MASKED, MASKED and HAS_MASK, HAS_BIAS)
    products, k_chunk = _head_product(q_tile, q, packed_indices_q, query_blocks, query_positions, query_valid,
                                      n_seq_blocks_q, k, packed_indices_k, key_blocks, key_positions, key_valid,
                                      n_seq_blocks_k, batch, n_head_blocks_qk, chunk,
                                      TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, HEAD_TILE_QK, N_CHUNKS_QK, True,
                                      sparsity_block_size, ROW_COMPLETE, USE_INT64)
    active, mask_blocks, any_mask_block, bias_blocks, any_bias_block = _tile_activity(
        bits, mask_bits, bias_bits, query_sub_blocks, MASKED, HAS_MASK, HAS_BIAS)
    logits = _attention_logits(products, active, mask_blocks, any_mask_block, bias_blocks, any_bias_block,
                               batch,
                               query_blocks[:, None], query_positions[:, None], query_valid[:, None],
                               key_blocks[None, :], key_positions[None, :], key_valid[None, :],
                               attention_mask, packed_indices_mask, attention_bias, packed_indices_bias,
                               causal_lengths, relative_scores, query_relations, key_relations, relation_metadata,
                               query_relation_validity, key_relation_validity, score_blocks,
                               n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns, scale,
                               HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE,
                               HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES, N_RELATIONS, MASKED,
                               (1 << N_RELATIONS) - 1 - MERGED, CLIPPED, sparsity_block_size, USE_INT64)
    if MERGED != 0:
        merged_logits, run_members, run_leaders, run_valid, n_runs = _merged_relative_logits(
            relative_scores, query_relations, key_relations, relation_metadata, score_blocks, schedule_blocks,
            batch, start, end, key_tile, query_positions, query_valid, key_positions, key_valid,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            MERGED, N_RELATIONS, BLOCK_SPARSE_SCORES, TRITON_BLOCK_SIZE_K, sparsity_block_size, USE_INT64)
        logits = _add_merged_logits(logits, merged_logits, MASKED, BLOCK_SPARSE_SCORES)
    p = tl.math.exp2(logits - lse_rows[:, None])
    p = tl.where(valid_lse[:, None], p, 0.0)

    dp, _ = _head_product(do_tile, do, packed_indices_o, query_blocks, query_positions, query_valid,
                          n_seq_blocks_q, v, packed_indices_v, key_blocks, key_positions, key_valid,
                          n_seq_blocks_k, batch, n_head_blocks_v, 0,
                          TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, HEAD_TILE_V, N_CHUNKS_V, False,
                          sparsity_block_size, ROW_COMPLETE, USE_INT64)
    ds = p * (dp - delta_rows[:, None])

    # Score gradients are written by the programs of the first head-dimension chunk. With relations addressed through
    # intervals, they are only written for permitted pairs: the gradients of all other pairs are zero, and their table
    # columns may lie in other blocks, see ``_score_offsets``.
    if MASKED:
        score_mask = active & (chunk == 0)
        if HAS_CAUSAL and BLOCK_SPARSE_SCORES > 1:
            sequence_length = tl.load(causal_lengths + batch)
            score_mask = (score_mask & (key_positions[None, :] <= query_positions[:, None])
                          & (query_positions[:, None] < sequence_length) & (key_positions[None, :] < sequence_length))
    else:
        score_mask = query_valid[:, None] & (chunk == 0)
    if HAS_BIAS:
        if any_bias_block:
            dbias_offsets, dbias_present = _pair_offsets(batch, query_blocks[:, None], query_positions[:, None],
                                                         bias_blocks, key_blocks[None, :], key_positions[None, :],
                                                         key_valid[None, :], packed_indices_bias,
                                                         n_seq_blocks_q, n_seq_blocks_k, sparsity_block_size,
                                                         USE_INT64)
            tl.store(dbias + dbias_offsets, tl.cast(ds, dbias.dtype.element_ty), mask=dbias_present & score_mask)
    if HAS_RELATIVE:
        minimum_gradients, maximum_gradients = _projected_relative_gradients(
            ds, drelative_scores, minimum_gradients, maximum_gradients,
            query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            batch, query_positions[:, None], query_valid[:, None], key_positions[None, :], key_valid[None, :],
            score_mask, n_batches, n_seq_blocks_q * sparsity_block_size, n_seq_blocks_k * sparsity_block_size,
            n_score_columns, N_RELATIONS, RELATION_BLOCK_SIZE, UNIQUE_KEY_RELATIONS, HAS_RELATION_VALIDITY,
            BLOCK_SPARSE_SCORES, (1 << N_RELATIONS) - 1 - MERGED, CLIPPED, sparsity_block_size, USE_INT64)
        if MERGED != 0:
            if n_runs <= _RUN_CAPACITY:
                # Pairs outside the attention layout or the causal past have no gradients.
                minimum_gradients, maximum_gradients = _merged_relative_gradients(
                    ds, drelative_scores, minimum_gradients, maximum_gradients,
                    query_relations, key_relations, relation_metadata, score_blocks,
                    batch, query_positions, query_valid & (chunk == 0), key_positions, key_valid, run_members,
                    run_leaders, run_valid, n_batches, n_seq_blocks_q * sparsity_block_size,
                    n_seq_blocks_k * sparsity_block_size, n_score_columns, MERGED, N_RELATIONS, RELATION_BLOCK_SIZE,
                    BLOCK_SPARSE_SCORES, sparsity_block_size, USE_INT64)
            else:
                minimum_gradients, maximum_gradients = _projected_relative_gradients(
                    ds, drelative_scores, minimum_gradients, maximum_gradients,
                    query_relations, key_relations, relation_metadata,
                    query_relation_validity, key_relation_validity, score_blocks,
                    batch, query_positions[:, None], query_valid[:, None], key_positions[None, :],
                    key_valid[None, :], score_mask, n_batches, n_seq_blocks_q * sparsity_block_size,
                    n_seq_blocks_k * sparsity_block_size, n_score_columns, N_RELATIONS, RELATION_BLOCK_SIZE,
                    UNIQUE_KEY_RELATIONS, HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES, MERGED, 0,
                    sparsity_block_size, USE_INT64)

    dq_accumulator += tl.dot(tl.cast(ds, k_chunk.dtype), k_chunk)
    return dq_accumulator, minimum_gradients, maximum_gradients


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flash_attention_backward_dq", block_size_names=_TILE_SIZE_NAMES),
    key=_AUTOTUNE_KEY_DQ,
    prune_configs_by={"early_config_prune": _prune_configs},
    reset_to_zero=["drelative_scores"],
)
@triton.jit(do_not_specialize=_UNSPECIALISED)
def flash_attention_kernel_bwd_dq(q, k, v, do, dq, lse, delta, dbias, drelative_scores,
                                  packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                                  schedule_offsets, schedule_blocks, schedule_bits, schedule_mask_bits,
                                  schedule_bias_bits, schedule_plain,
                                  attention_mask, packed_indices_mask, attention_bias, packed_indices_bias,
                                  causal_lengths,
                                  relative_scores, query_relations, key_relations, relation_metadata,
                                  query_coordinate_floors, key_coordinate_ceilings,
                                  query_relation_validity, key_relation_validity,
                                  query_invalid_blocks, key_invalid_blocks,
                                  score_blocks,
                                  n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                  n_score_columns, scale, n_seq_blocks_bucket, n_schedule_entries_bucket,
                                  n_head_blocks_qk: tl.constexpr, n_head_blocks_v: tl.constexpr,
                                  sparsity_block_size: tl.constexpr,
                                  SCHEDULE_BLOCKS: tl.constexpr,
                                  HEAD_TILE_QK: tl.constexpr, HEAD_TILE_V: tl.constexpr,
                                  N_CHUNKS_QK: tl.constexpr, N_CHUNKS_V: tl.constexpr,
                                  ROW_COMPLETE: tl.constexpr, USE_INT64: tl.constexpr,
                                  HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CAUSAL: tl.constexpr,
                                  HAS_RELATIVE: tl.constexpr, HAS_RELATION_VALIDITY: tl.constexpr,
                                  BLOCK_SPARSE_SCORES: tl.constexpr, N_RELATIONS: tl.constexpr,
                                  CLIPPED_RELATIONS: tl.constexpr, SCORE_LAYOUT: tl.constexpr,
                                  MERGED_RELATIONS: tl.constexpr,
                                  RELATION_BLOCK_SIZE: tl.constexpr, UNIQUE_KEY_RELATIONS: tl.constexpr,
                                  TRITON_BLOCK_SIZE_Q: tl.constexpr,
                                  TRITON_BLOCK_SIZE_K: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    # Start with the last query tiles, which attend to the most keys under causal layouts.
    n_tiles = tl.cdiv(n_seq_blocks_q * sparsity_block_size, TRITON_BLOCK_SIZE_Q)
    pid_tile = n_tiles - 1 - tl.cast(tl.program_id(axis=1), index_dtype)
    pid_chunk = tl.program_id(axis=2)

    query_blocks, query_positions, query_valid = _sequence_tile(pid_tile, n_seq_blocks_q, TRITON_BLOCK_SIZE_Q,
                                                                sparsity_block_size, USE_INT64)
    start, end, n_key_tiles, n_plain, group_shift = _schedule_range(
        schedule_offsets, schedule_plain, pid_bat, pid_tile, n_seq_blocks_q, causal_lengths,
        SCHEDULE_BLOCKS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, sparsity_block_size, HAS_CAUSAL, False)
    query_sub_blocks = group_shift + tl.arange(0, TRITON_BLOCK_SIZE_Q) // sparsity_block_size

    if N_CHUNKS_QK > 1:
        q_tile = 0
    else:
        q_offsets, q_mask = _operand_offsets(packed_indices_q, pid_bat, query_blocks, query_positions, query_valid,
                                             n_seq_blocks_q, n_head_blocks_qk, 0,
                                             HEAD_TILE_QK, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        q_tile = tl.load(q + q_offsets, mask=q_mask, other=0)
    if N_CHUNKS_V > 1:
        do_tile = 0
    else:
        do_offsets, do_mask = _operand_offsets(packed_indices_o, pid_bat, query_blocks, query_positions,
                                               query_valid, n_seq_blocks_q, n_head_blocks_v, 0,
                                               HEAD_TILE_V, sparsity_block_size, ROW_COMPLETE, USE_INT64)
        do_tile = tl.load(do + do_offsets, mask=do_mask, other=0)

    # Recompute the probabilities from the saved log-sum-exp. Queries without attention have none.
    row_offsets = pid_bat * n_seq_blocks_q * sparsity_block_size + query_positions
    m = tl.load(lse + row_offsets, mask=query_valid, other=float("-inf"))
    delta_rows = tl.load(delta + row_offsets, mask=query_valid, other=0.0)
    valid_lse = m != float("-inf")
    lse_rows = tl.where(valid_lse, m, 0.0)

    dq_accumulator = tl.zeros((TRITON_BLOCK_SIZE_Q, HEAD_TILE_QK), dtype=tl.float32)
    minimum_gradients = tl.zeros((RELATION_BLOCK_SIZE, TRITON_BLOCK_SIZE_Q), dtype=tl.float32)
    maximum_gradients = tl.zeros((RELATION_BLOCK_SIZE, TRITON_BLOCK_SIZE_Q), dtype=tl.float32)

    # Tiles with positions without valid relations come first, see ``_validity_tiles``. All others skip the
    # validity masks.
    n_validity = 0
    if HAS_RELATION_VALIDITY:
        n_validity = _validity_tiles(schedule_blocks, key_invalid_blocks, query_invalid_blocks, pid_bat, pid_tile,
                                     start, end, n_key_tiles, n_seq_blocks_q, n_seq_blocks_k,
                                     TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K, sparsity_block_size, False)
        for key_tile in range(0, n_validity):
            dq_accumulator, minimum_gradients, maximum_gradients = _backward_dq_tile(
                key_tile, dq_accumulator, minimum_gradients, maximum_gradients, q_tile, do_tile,
                lse_rows, valid_lse, delta_rows, pid_bat, pid_chunk, start, end,
                query_blocks, query_positions, query_valid, query_sub_blocks,
                q, k, v, do, dbias, drelative_scores,
                packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                relative_scores, query_relations, key_relations, relation_metadata,
                query_relation_validity, key_relation_validity, score_blocks,
                n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                scale, n_head_blocks_qk, n_head_blocks_v,
                sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
                HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, HAS_RELATION_VALIDITY, BLOCK_SPARSE_SCORES,
                N_RELATIONS, RELATION_BLOCK_SIZE, UNIQUE_KEY_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K,
                True, 0, 0)

    # Distant plain tiles in which every pair is clipped to the maximum of the clipped relations come first.
    n_clipped = 0
    if CLIPPED_RELATIONS != 0:
        first_query_block = pid_tile * (TRITON_BLOCK_SIZE_Q // sparsity_block_size)
        n_clipped = _clipped_tile_count(schedule_blocks, query_coordinate_floors, key_coordinate_ceilings,
                                        relation_metadata, pid_bat, first_query_block, start, n_plain, n_batches,
                                        n_seq_blocks_q, n_seq_blocks_k, CLIPPED_RELATIONS, N_RELATIONS,
                                        TRITON_BLOCK_SIZE_K, sparsity_block_size, False)
        for key_tile in range(n_validity, n_clipped):
            dq_accumulator, minimum_gradients, maximum_gradients = _backward_dq_tile(
                key_tile, dq_accumulator, minimum_gradients, maximum_gradients, q_tile, do_tile,
                lse_rows, valid_lse, delta_rows, pid_bat, pid_chunk, start, end,
                query_blocks, query_positions, query_valid, query_sub_blocks,
                q, k, v, do, dbias, drelative_scores,
                packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
                schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
                attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
                relative_scores, query_relations, key_relations, relation_metadata,
                query_relation_validity, key_relation_validity, score_blocks,
                n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
                scale, n_head_blocks_qk, n_head_blocks_v,
                sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
                HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
                N_RELATIONS, RELATION_BLOCK_SIZE, UNIQUE_KEY_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K,
                False, CLIPPED_RELATIONS, MERGED_RELATIONS - (MERGED_RELATIONS & CLIPPED_RELATIONS))
    plain_start = n_clipped
    masked_start = n_plain
    if HAS_RELATION_VALIDITY:
        plain_start = tl.maximum(plain_start, n_validity)
        masked_start = tl.maximum(masked_start, n_validity)
    for key_tile in range(plain_start, n_plain):
        dq_accumulator, minimum_gradients, maximum_gradients = _backward_dq_tile(
            key_tile, dq_accumulator, minimum_gradients, maximum_gradients, q_tile, do_tile,
            lse_rows, valid_lse, delta_rows, pid_bat, pid_chunk, start, end,
            query_blocks, query_positions, query_valid, query_sub_blocks,
            q, k, v, do, dbias, drelative_scores,
            packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
            schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
            attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
            relative_scores, query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            scale, n_head_blocks_qk, n_head_blocks_v,
            sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
            HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
            N_RELATIONS, RELATION_BLOCK_SIZE, UNIQUE_KEY_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K,
            False, 0, MERGED_RELATIONS)
    for key_tile in range(masked_start, n_key_tiles):
        dq_accumulator, minimum_gradients, maximum_gradients = _backward_dq_tile(
            key_tile, dq_accumulator, minimum_gradients, maximum_gradients, q_tile, do_tile,
            lse_rows, valid_lse, delta_rows, pid_bat, pid_chunk, start, end,
            query_blocks, query_positions, query_valid, query_sub_blocks,
            q, k, v, do, dbias, drelative_scores,
            packed_indices_q, packed_indices_k, packed_indices_v, packed_indices_o,
            schedule_blocks, schedule_bits, schedule_mask_bits, schedule_bias_bits,
            attention_mask, packed_indices_mask, attention_bias, packed_indices_bias, causal_lengths,
            relative_scores, query_relations, key_relations, relation_metadata,
            query_relation_validity, key_relation_validity, score_blocks,
            n_batches, n_seq_blocks_q, n_seq_blocks_k, n_score_columns,
            scale, n_head_blocks_qk, n_head_blocks_v,
            sparsity_block_size, HEAD_TILE_QK, HEAD_TILE_V, N_CHUNKS_QK, N_CHUNKS_V, ROW_COMPLETE, USE_INT64,
            HAS_MASK, HAS_BIAS, HAS_CAUSAL, HAS_RELATIVE, False, BLOCK_SPARSE_SCORES,
            N_RELATIONS, RELATION_BLOCK_SIZE, UNIQUE_KEY_RELATIONS, TRITON_BLOCK_SIZE_Q, TRITON_BLOCK_SIZE_K,
            True, 0, MERGED_RELATIONS)

    dq_offsets, dq_mask = _operand_offsets(packed_indices_q, pid_bat, query_blocks, query_positions, query_valid,
                                           n_seq_blocks_q, n_head_blocks_qk, pid_chunk,
                                           HEAD_TILE_QK, sparsity_block_size, ROW_COMPLETE, USE_INT64)
    tl.store(dq + dq_offsets, tl.cast(dq_accumulator * scale, dq.dtype.element_ty), mask=dq_mask)

    if HAS_RELATIVE:
        # Every query stores the summed gradients of its clipped relation bounds once.
        relation_slots = tl.arange(0, RELATION_BLOCK_SIZE)
        relation_mask = relation_slots < N_RELATIONS
        relation_minimum = tl.load(relation_metadata + relation_slots * 3, mask=relation_mask, other=0)
        relation_maximum = tl.load(relation_metadata + relation_slots * 3 + 1, mask=relation_mask, other=0)
        relation_offset = tl.cast(tl.load(relation_metadata + relation_slots * 3 + 2, mask=relation_mask, other=0),
                                  index_dtype)
        store_mask = relation_mask[:, None] & query_valid[None, :] & (pid_chunk == 0)
        # The bounds of all relations are stored at once and addressed through the map of value blocks.
        minimum_offsets, minimum_present = _score_offsets(
            score_blocks, -1, pid_bat, query_positions[None, :], store_mask, relation_offset[:, None],
            n_seq_blocks_q * sparsity_block_size, n_score_columns, sparsity_block_size, N_RELATIONS,
            BLOCK_SPARSE_SCORES, USE_INT64)
        # A relation with a single row has one column for both bounds, which its clipped tiles reach as the maximum.
        single_row = (relation_maximum == relation_minimum)[:, None]
        tl.store(drelative_scores + minimum_offsets, minimum_gradients + tl.where(single_row, maximum_gradients, 0.0),
                 mask=minimum_present)
        maximum_offsets, maximum_present = _score_offsets(
            score_blocks, -1, pid_bat, query_positions[None, :],
            store_mask & (relation_maximum != relation_minimum)[:, None],
            (relation_offset + tl.cast(relation_maximum - relation_minimum, index_dtype))[:, None],
            n_seq_blocks_q * sparsity_block_size, n_score_columns, sparsity_block_size, N_RELATIONS,
            BLOCK_SPARSE_SCORES, USE_INT64)
        tl.store(drelative_scores + maximum_offsets, maximum_gradients, mask=maximum_present)


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
                                       sparsity_layout_bias: Tensor | None = None,
                                       sparsity_block_size: int | None = None) -> dict:
    """Builds reusable layout metadata for block-sparse Flash Attention.

    The tile schedules of the kernels depend on the sparsity block size. They are built when it is given, and
    otherwise on the first Flash Attention call that uses the cache.

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
        sparsity_block_size (int, optional): The size of the sparsity blocks (default ``None``).

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

    if sparsity_block_size is not None:
        group_blocks = _schedule_group_blocks(sparsity_block_size)
        query_schedule_key = f"query_tile_schedule_{group_blocks}"
        if query_schedule_key not in layout_cache:
            layout_cache[query_schedule_key] = _build_tile_schedule(attention_layout, group_blocks, by_key=False,
                                                                    sparsity_layout_mask=sparsity_layout_mask,
                                                                    sparsity_layout_bias=sparsity_layout_bias)
        key_schedule_key = f"key_tile_schedule_{group_blocks}"
        if key_schedule_key not in layout_cache:
            layout_cache[key_schedule_key] = _build_tile_schedule(attention_layout, group_blocks, by_key=True,
                                                                  sparsity_layout_mask=sparsity_layout_mask,
                                                                  sparsity_layout_bias=sparsity_layout_bias)

    if sparsity_layout_q is not None and "packed_indices_q" not in layout_cache:
        layout_cache["packed_indices_q"] = build_packed_indices(sparsity_layout_q)

    if sparsity_layout_k is not None and "packed_indices_k" not in layout_cache:
        layout_cache["packed_indices_k"] = build_packed_indices(sparsity_layout_k)

    if sparsity_layout_v is not None and "packed_indices_v" not in layout_cache:
        layout_cache["packed_indices_v"] = build_packed_indices(sparsity_layout_v)

    if sparsity_layout_mask is not None and "packed_indices_mask" not in layout_cache:
        layout_cache["packed_indices_mask"] = build_packed_indices(sparsity_layout_mask)

    if sparsity_layout_bias is not None and "packed_indices_bias" not in layout_cache:
        layout_cache["packed_indices_bias"] = build_packed_indices(sparsity_layout_bias)

    if sparsity_layout_o is not None and "packed_indices_o" not in layout_cache:
        layout_cache["packed_indices_o"] = build_packed_indices(sparsity_layout_o)

    # Scalar layout statistics are transferred to the host together.
    statistics = {}
    row_block_counts = {}

    def count_row_blocks(sparsity_layout: Tensor) -> Tensor:
        if id(sparsity_layout) not in row_block_counts:
            row_block_counts[id(sparsity_layout)] = _binary(sparsity_layout).sum(dim=-1)
        return row_block_counts[id(sparsity_layout)]

    if sparsity_layout_q is not None and "n_sparse_blocks_q" not in layout_cache:
        statistics["n_sparse_blocks_q"] = count_row_blocks(sparsity_layout_q).sum()
    if sparsity_layout_o is not None and "n_sparse_blocks_o" not in layout_cache:
        statistics["n_sparse_blocks_o"] = count_row_blocks(sparsity_layout_o).sum()
    operand_layouts = [layout for layout in (sparsity_layout_q, sparsity_layout_k, sparsity_layout_v,
                                             sparsity_layout_o) if layout is not None]
    if len(operand_layouts) == 4 and "row_complete" not in layout_cache:
        # Every sequence block holds all or none of its head-dimension blocks.
        statistics["row_complete"] = torch.stack([
            torch.all((count_row_blocks(layout) == 0) | (count_row_blocks(layout) == layout.size(-1)))
            for layout in dict.fromkeys(operand_layouts)]).all()
    if statistics:
        values = torch.stack([value.to(torch.int64) for value in statistics.values()]).tolist()
        for name, value in zip(statistics, values):
            layout_cache[name] = bool(value) if name == "row_complete" else value

    layout_cache_tensors = [v for v in layout_cache.values() if isinstance(v, Tensor)]
    if layout_cache_tensors:
        validate_contiguous(*layout_cache_tensors)

    return finalize_layout_cache(layout_cache)


def _binary(sparsity_layout: Tensor) -> Tensor:
    """Returns a layout as a Boolean tensor."""
    return sparsity_layout if sparsity_layout.dtype == torch.bool else sparsity_layout != 0


def _build_tile_schedule(attention_layout: Tensor, group_blocks: int, by_key: bool,
                         sparsity_layout_mask: Tensor | None = None,
                         sparsity_layout_bias: Tensor | None = None) -> dict:
    """Builds the tile schedule of the programs that own query tiles or, with ``by_key``, key tiles.

    Program blocks are grouped by ``group_blocks``. For every group, the schedule lists the blocks of the other
    sequence that any block of the group attends to, or is attended by, in compressed sparse row form: the entries of
    group ``g`` of batch ``b`` are ``blocks[offsets[b * n_groups + g]:offsets[b * n_groups + g + 1]]``. Key blocks
    are listed in ascending and query blocks in descending order, so that blocks far from the diagonal of causal
    layouts come first. Bit ``i`` of ``bits`` is set for the entries attended to by, or attending to, block ``i`` of
    the group, and bit ``i`` of ``mask_bits`` and ``bias_bits`` if this block pair also holds a block of the explicit
    mask or of the bias. ``plain`` gives the number of leading entries that need no masking at all: their blocks are
    attended by every block of the group without an explicit mask, and, in its second column, also lie entirely in
    the causal past of the group (or, for key groups, entirely in the causal future).

    """
    device = attention_layout.device
    layout = _binary(attention_layout)
    mask_layout = None if sparsity_layout_mask is None else layout & _binary(sparsity_layout_mask)
    bias_layout = None if sparsity_layout_bias is None else layout & _binary(sparsity_layout_bias)
    if by_key:
        layout = layout.transpose(1, 2)
        mask_layout = None if mask_layout is None else mask_layout.transpose(1, 2)
        bias_layout = None if bias_layout is None else bias_layout.transpose(1, 2)
    n_batches, n_program_blocks, n_other_blocks = layout.shape
    n_groups = triton.cdiv(n_program_blocks, group_blocks)

    bits = _group_bits(layout, n_groups, group_blocks)
    mask_bits = None if mask_layout is None else _group_bits(mask_layout, n_groups, group_blocks)
    bias_bits = None if bias_layout is None else _group_bits(bias_layout, n_groups, group_blocks)
    other_blocks = torch.arange(n_other_blocks, device=device)
    groups = torch.arange(n_groups, device=device)[:, None]
    if by_key:
        bits = bits.flip(-1)
        mask_bits = None if mask_bits is None else mask_bits.flip(-1)
        bias_bits = None if bias_bits is None else bias_bits.flip(-1)
        causal = n_other_blocks - 1 - other_blocks[None, :] >= (groups + 1) * group_blocks
    else:
        causal = other_blocks[None, :] < groups * group_blocks

    attended = bits != 0
    plain = bits == (1 << group_blocks) - 1
    if mask_bits is not None:
        plain &= mask_bits == 0
    entries = attended.flatten().nonzero().squeeze(1)
    index_dtype = _select_segmented_index_dtype(entries.numel(), n_other_blocks)
    columns = entries % n_other_blocks
    blocks = n_other_blocks - 1 - columns if by_key else columns

    offsets = torch.zeros(n_batches * n_groups + 1, dtype=index_dtype, device=device)
    offsets[1:] = attended.sum(dim=-1).flatten().cumsum(0)
    plain_counts = _count_leading(attended, torch.stack([plain, plain & causal]))

    return {
        "group_blocks": group_blocks,
        "offsets": as_base_tensor(offsets),
        "blocks": as_base_tensor(blocks.to(index_dtype)),
        "bits": as_base_tensor(bits.flatten()[entries]),
        "mask_bits": _entry_bits(mask_bits, entries),
        "bias_bits": _entry_bits(bias_bits, entries),
        "plain": as_base_tensor(plain_counts.permute(1, 2, 0).reshape(-1, 2).to(index_dtype).contiguous()),
    }


def _entry_bits(bits: Tensor | None, entries: Tensor) -> Tensor:
    """Returns the bits of the schedule entries, or an empty tensor without bits."""
    if bits is None:
        return torch.empty(0, dtype=torch.int32, device=entries.device)
    return as_base_tensor(bits.flatten()[entries])


def _group_bits(layout: Tensor, n_groups: int, group_blocks: int) -> Tensor:
    """Sets bit ``i`` for the blocks of the other sequence that block ``i`` of every group of rows attends to."""
    n_batches, n_rows, n_columns = layout.shape
    weights = torch.pow(2, torch.arange(group_blocks, dtype=torch.int32, device=layout.device))
    grouped = layout.to(torch.int32)
    if n_groups * group_blocks != n_rows:
        grouped = torch.nn.functional.pad(grouped, (0, 0, 0, n_groups * group_blocks - n_rows))
    return (grouped.view(n_batches, n_groups, group_blocks, n_columns) * weights[:, None]).sum(
        dim=2, dtype=torch.int32)


def _count_leading(attended: Tensor, plain: Tensor) -> Tensor:
    """Counts the attended entries before the first attended entry that is not plain, along the last dimension."""
    return (attended & (torch.cumsum(attended & ~plain, dim=-1) == 0)).sum(dim=-1)


def _select_segmented_index_dtype(n_active_blocks: int, n_blocks_col: int) -> torch.dtype:
    """Selects the smallest safe dtype for Flash Attention adjacency metadata."""
    max_column_index = n_blocks_col - 1
    if n_active_blocks > INT32_INDEX_MAX or max_column_index > INT32_INDEX_MAX:
        return torch.int64
    return torch.int32

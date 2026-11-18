import importlib
import math

import pytest
import torch

import blksprs as bs

DEVICE = torch.device("cuda")


def _brute_force_referenced(relations, bounds, attention_layout, sparsity_block_size, lengths, validity):
    """Marks the blocks of (batch, query block, value block) referenced by a permitted pair.

    The reference marks the blocks referenced by the dense relative-index matrix of all permitted pairs, so that it
    covers clipping, validity masks, causal lengths, several relations, batches, non-causal layouts, and restricted
    layouts independently of the kernels.

    """
    n_batches, length = relations.shape[1:]
    positions = torch.arange(length, device=DEVICE)
    permitted = (attention_layout.repeat_interleave(sparsity_block_size, dim=1)
                 .repeat_interleave(sparsity_block_size, dim=2))
    if lengths is not None:
        within = positions[None, :] < lengths[:, None]
        permitted = (permitted & (positions[:, None] >= positions[None, :])
                     & within[:, :, None] & within[:, None, :])
    value_blocks = [math.ceil((maximum - minimum + 1) / sparsity_block_size) for minimum, maximum in bounds]
    referenced = torch.zeros((n_batches, length // sparsity_block_size, sum(value_blocks)), dtype=torch.bool,
                             device=DEVICE)
    batches = torch.arange(n_batches, device=DEVICE)[:, None, None].expand(-1, length, length)
    query_blocks = (positions // sparsity_block_size)[None, :, None].expand(n_batches, -1, length)
    for relation, (minimum, maximum) in enumerate(bounds):
        indices = (relations[relation][:, :, None] - relations[relation][:, None, :]).clamp(minimum, maximum) - minimum
        relation_permitted = permitted
        if validity is not None:
            relation_permitted = relation_permitted & validity[relation][:, :, None] & validity[relation][:, None, :]
        blocks = sum(value_blocks[:relation]) + indices // sparsity_block_size
        referenced[batches[relation_permitted], query_blocks[relation_permitted], blocks[relation_permitted]] = True
    return referenced


def _stored_blocks(referenced, bounds, sparsity_block_size):
    """Adds the gaps that are stored between the referenced blocks to them.

    The gaps of relations with few value blocks are always stored, and those of the other relations if they hold at
    most one sixty-fourth of the referenced blocks. Also returns whether the stored blocks of every relation form one
    interval in every row afterwards.

    """
    stored = referenced.clone()
    spans = []
    gaps = 0
    first_block = 0
    for minimum, maximum in bounds:
        end_block = first_block + math.ceil((maximum - minimum + 1) / sparsity_block_size)
        relation_referenced = referenced[:, :, first_block:end_block]
        value_blocks = torch.arange(first_block, end_block, device=DEVICE)
        counts = relation_referenced.sum(dim=2)
        first = torch.where(relation_referenced, value_blocks, end_block).amin(dim=2)
        end = torch.where(relation_referenced, value_blocks, -1).amax(dim=2) + 1
        compact = end_block - first_block <= 8
        if not compact:
            gaps += int(torch.where(counts > 0, end - first - counts, 0).sum())
        spans.append((first_block, end_block, first, end, compact))
        first_block = end_block
    complete = gaps * 64 <= int(referenced.sum())
    for first_block, end_block, first, end, compact in spans:
        if complete or compact:
            value_blocks = torch.arange(first_block, end_block, device=DEVICE)
            stored[:, :, first_block:end_block] |= (value_blocks >= first[..., None]) & (value_blocks < end[..., None])
    return stored, complete


def _single_configuration(monkeypatch: pytest.MonkeyPatch):
    """Runs every Flash Attention kernel with its first configuration, so that results compared across calls of
    different layouts of the projected scores use the same tiles, and thus the same order of their float32 products.

    """
    flash_module = importlib.import_module("blksprs.ops.flash_attention")
    for name in ("flash_attention_kernel", "flash_attention_kernel_bwd_dkdv", "flash_attention_kernel_bwd_dq"):
        kernel = getattr(flash_module, name)
        monkeypatch.setattr(kernel, "configs", kernel.configs[:1])
        monkeypatch.setattr(kernel, "cache", {})


def _exact_storage(monkeypatch: pytest.MonkeyPatch):
    """Stores only the referenced blocks, without gaps, so that calls with gaps address relations through the map."""
    blocks_module = importlib.import_module("blksprs.ops.relative_score_blocks")
    monkeypatch.setattr(blocks_module, "_COMPACT_VALUE_BLOCKS", 0)
    monkeypatch.setattr(blocks_module, "_GAP_BUDGET", 2 ** 40)


def _case(name, sparsity_block_size, generator):
    n_batches, n_blocks = 2, 10
    length = n_blocks * sparsity_block_size
    positions = torch.arange(length).repeat(n_batches, 1)
    pitches = torch.randint(0, 12, (n_batches, length), generator=generator)
    relations = torch.stack([positions, positions // 3, pitches]).to(DEVICE)
    # Positions and onsets clip distant keys; the pitches are not monotone and reference sparse value blocks.
    bounds = ((0, 2 * sparsity_block_size + 5), (0, sparsity_block_size // 2), (-3, 40))
    lengths = validity = None
    block_indices = torch.arange(n_blocks)
    if name == "causal":
        attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool))
        lengths = torch.tensor([length - 5, 4 * sparsity_block_size + 3])
        validity = torch.rand(relations.shape, generator=generator) < 0.8
    elif name == "repeated":
        # Repeated batches, such as the heads of one sequence, share their referenced blocks.
        relations = relations[:, :1].expand(-1, n_batches, -1).contiguous()
        attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool))
        lengths = torch.full((n_batches,), length - 5)
        validity = (torch.rand(relations[:, :1].shape, generator=generator) < 0.8).expand(-1, n_batches, -1)
    elif name == "non-causal":
        attention_layout = torch.rand((n_batches, n_blocks, n_blocks), generator=generator) < 0.5
        attention_layout |= torch.eye(n_blocks, dtype=torch.bool)
        relations = relations.flip(-1).contiguous()
        bounds = ((-2 * sparsity_block_size - 5, sparsity_block_size), (-4, 4), (-40, 3))
        validity = torch.rand(relations.shape, generator=generator) < 0.8
    elif name in ("holes", "only holes", "some holes"):
        # Coordinates that take one of two values far apart reference the value blocks of three distant differences,
        # so that the interval of value blocks of a query block also holds unreferenced blocks.
        jumps = (pitches % 2) * 3 * sparsity_block_size
        leaps = (pitches // 6) * 4 * sparsity_block_size
        jump_bounds = (-3 * sparsity_block_size, 3 * sparsity_block_size)
        if name == "holes":
            relations = torch.stack([positions, jumps]).to(DEVICE)
            bounds = (bounds[0], jump_bounds)
        elif name == "some holes":
            # Only the second half of every sequence jumps. Under a causal band of three blocks, the query blocks of
            # the first half attend to keys without gaps, and the key blocks of the first half only to them.
            jumps = torch.where(positions < length // 2, 0, jumps)
            relations = torch.stack([positions, jumps]).to(DEVICE)
            bounds = (bounds[0], jump_bounds)
        else:
            relations = torch.stack([jumps, leaps]).to(DEVICE)
            bounds = (jump_bounds, (-4 * sparsity_block_size, 4 * sparsity_block_size))
        attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool))
        if name == "some holes":
            distance = block_indices[:, None] - block_indices[None, :]
            attention_layout = ((distance >= 0) & (distance < 3)).expand(n_batches, -1, -1).contiguous()
        lengths = torch.tensor([length - 5, length])
    else:
        # A causal band of three blocks, as for local-window attention.
        distance = block_indices[:, None] - block_indices[None, :]
        attention_layout = ((distance >= 0) & (distance < 3)).expand(n_batches, -1, -1).contiguous()
        lengths = torch.tensor([length, length - sparsity_block_size - 1])
    return (relations, bounds, attention_layout.to(DEVICE), None if lengths is None else lengths.to(DEVICE),
            None if validity is None else validity.to(DEVICE))


def _attend(case, sparsity_block_size, inputs, score_precalculation, layout_cache=None):
    relations, bounds, attention_layout, lengths, validity = case
    n_batches, length, head_dimension = inputs[0].shape
    layout = torch.ones((n_batches, length // sparsity_block_size, head_dimension // sparsity_block_size),
                        dtype=torch.bool, device=DEVICE)
    output = bs.ops.flash_attention_relative_embeddings(
        *(operand for tensor in inputs[:3]
          for operand in (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout)),
        attention_layout, sparsity_block_size, relations, relations, inputs[3], bounds,
        causal_lengths=lengths, query_relation_validity=validity, key_relation_validity=validity,
        sparsity_layout_o=layout, layout_cache=layout_cache, score_precalculation=score_precalculation)
    return bs.ops.to_dense(output, layout, sparsity_block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("storage", ["gaps", "exact"])
@pytest.mark.parametrize("sparsity_block_size", [16, 32])
@pytest.mark.parametrize("name", ["causal", "repeated", "non-causal", "band", "holes", "only holes", "some holes"])
def test_relative_score_blocks_match_brute_force(name: str, sparsity_block_size: int, storage: str,
                                                 monkeypatch: pytest.MonkeyPatch):
    """The stored blocks of the projected relative scores match a brute-force reference of the referenced blocks and
    the gaps that are stored between them.

    """
    if storage == "exact":
        _exact_storage(monkeypatch)
    generator = torch.Generator().manual_seed(3)
    case = _case(name, sparsity_block_size, generator)
    relations, bounds, attention_layout, lengths, validity = case
    n_batches, length = relations.shape[1:]
    n_relation_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
    inputs = [torch.randn((n_batches, length, 32), generator=generator).to(DEVICE) for _ in range(3)]
    inputs.append(torch.randn((n_batches, n_relation_rows, 32), generator=generator).to(DEVICE))
    layout_cache = {}
    _attend(case, sparsity_block_size, inputs, "referenced_blocks", layout_cache)
    blocks = layout_cache["relative_coordinates"]["coordinates"].score_blocks

    referenced = blocks.lookup >= 0
    expected_referenced = _brute_force_referenced(relations, bounds, attention_layout, sparsity_block_size, lengths,
                                                  validity)
    if storage == "exact":
        expected, complete = expected_referenced, None
    else:
        expected, complete = _stored_blocks(expected_referenced, bounds, sparsity_block_size)
    assert torch.equal(referenced, expected)
    # Some blocks are not stored, so that the layout is a proper subset.
    assert 0 < blocks.n_blocks < referenced.numel()

    # The compressed rows and columns list the stored blocks in row-major and column-major order.
    entries = referenced.nonzero()
    assert torch.equal(blocks.lookup[referenced].long(), torch.arange(blocks.n_blocks, device=DEVICE))
    assert torch.equal(blocks.block_columns.long(), entries[:, 2])
    assert torch.equal(blocks.row_offsets.diff(), referenced.sum(dim=2).flatten())
    column_entries = referenced.transpose(1, 2).nonzero()
    assert torch.equal(blocks.column_blocks,
                       blocks.lookup[column_entries[:, 0], column_entries[:, 2], column_entries[:, 1]].long())
    assert torch.equal(blocks.column_rows.long(), column_entries[:, 2])
    assert torch.equal(blocks.column_offsets.diff(), referenced.sum(dim=1).flatten())

    # In the rows whose stored blocks of a relation form one interval of value blocks, the index of every stored block
    # of the relation is its value block plus the offset of the row and relation.
    n_value_blocks = expected.size(2)
    first_block = compact_relations = 0
    expected_rows = torch.ones(expected.shape[:2], dtype=torch.bool, device=DEVICE)
    for relation, (minimum, maximum) in enumerate(bounds):
        end_block = first_block + math.ceil((maximum - minimum + 1) / sparsity_block_size)
        if storage == "gaps" and end_block - first_block <= 8:
            compact_relations |= 1 << relation
        relation_stored = expected[:, :, first_block:end_block]
        value_blocks = torch.arange(first_block, end_block, device=DEVICE)
        counts = relation_stored.sum(dim=2)
        first = torch.where(relation_stored, value_blocks, n_value_blocks).amin(dim=2)
        end = torch.where(relation_stored, value_blocks, -1).amax(dim=2) + 1
        interval = torch.where(counts > 0, end - first, 0) == counts
        expected_rows &= interval
        entries = (relation_stored & interval[..., None]).nonzero()
        offsets = blocks.addressing[:, :, n_value_blocks + relation].long()
        assert torch.equal(offsets[entries[:, 0], entries[:, 1]] + first_block + entries[:, 2],
                           blocks.lookup[entries[:, 0], entries[:, 1], first_block + entries[:, 2]].long())
        first_block = end_block
    assert blocks.interval_relations == compact_relations
    if complete is not None:
        assert bool(expected_rows.all()) == complete
    assert blocks.complete_intervals == bool(expected_rows.all())
    if name in ("holes", "only holes", "some holes") and storage == "exact":
        # Calls with gaps address the relations with gaps through the map.
        assert not blocks.complete_intervals


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("storage", ["gaps", "exact"])
@pytest.mark.parametrize("sparsity_block_size", [16, 32])
@pytest.mark.parametrize("name", ["causal", "non-causal", "band", "holes", "only holes", "some holes"])
def test_relative_score_precalculations_agree(name: str, sparsity_block_size: int, storage: str,
                                              monkeypatch: pytest.MonkeyPatch):
    """The referenced blocks give the results of the complete projection up to the order of reductions, whether they
    are addressed through intervals of value blocks, through the map of all value blocks, or both, in programs whose
    query blocks all have complete intervals and in all others.

    """
    if storage == "exact":
        _exact_storage(monkeypatch)
    _single_configuration(monkeypatch)
    generator = torch.Generator().manual_seed(4)
    case = _case(name, sparsity_block_size, generator)
    relations, bounds, *_ = case
    n_batches, length = relations.shape[1:]
    n_relation_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
    base = [torch.randn((n_batches, length, 32), generator=generator).to(DEVICE) for _ in range(3)]
    base.append((torch.randn((n_batches, n_relation_rows, 32), generator=generator) * 0.3).to(DEVICE))
    output_gradient = torch.randn((n_batches, length, 32), generator=generator).to(DEVICE)

    results = []
    layout_cache = {}
    for score_precalculation in ("referenced_blocks", "complete"):
        inputs = [tensor.clone().requires_grad_(True) for tensor in base]
        output = _attend(case, sparsity_block_size, inputs, score_precalculation,
                         layout_cache if score_precalculation == "referenced_blocks" else None)
        output.backward(output_gradient)
        results.append([output.detach()] + [tensor.grad for tensor in inputs])
    complete_intervals = layout_cache["relative_coordinates"]["coordinates"].score_blocks.complete_intervals
    if name in ("holes", "only holes", "some holes") and storage == "exact":
        assert not complete_intervals
    for referenced, complete in zip(*results):
        torch.testing.assert_close(referenced, complete, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("storage", ["gaps", "exact"])
@pytest.mark.parametrize("sparsity_block_size", [16, 32])
@pytest.mark.parametrize("name", ["causal", "repeated", "non-causal", "band", "holes", "only holes"])
def test_relative_score_intervals_stay_within_zero_guard_blocks(name: str, sparsity_block_size: int, storage: str,
                                                                monkeypatch: pytest.MonkeyPatch):
    """The columns of relations addressed through intervals are not checked against the interval.

    Every column that any pair of such a relation can address, including those of the first and last blocks and of
    rows beyond the causal lengths, lies within the zero guard blocks around the referenced blocks. Values read from
    the guard blocks never reach the results: with guard blocks of NaN, the results equal those of the complete
    projection.

    """
    if storage == "exact":
        _exact_storage(monkeypatch)
    _single_configuration(monkeypatch)
    generator = torch.Generator().manual_seed(6)
    case = _case(name, sparsity_block_size, generator)
    relations, bounds, *_ = case
    n_batches, length = relations.shape[1:]
    n_relation_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
    base = [torch.randn((n_batches, length, 32), generator=generator).to(DEVICE) for _ in range(3)]
    base.append((torch.randn((n_batches, n_relation_rows, 32), generator=generator) * 0.3).to(DEVICE))
    output_gradient = torch.randn((n_batches, length, 32), generator=generator).to(DEVICE)

    flash_module = importlib.import_module("blksprs.ops.flash_attention")
    blocks_module = importlib.import_module("blksprs.ops.relative_score_blocks")
    allocate = blocks_module.allocate_referenced_scores
    guards = []

    def poisoned(blocks, block_size, dtype, device, zero):
        scores = allocate(blocks, block_size, dtype, device, zero)
        n_guard_blocks = blocks.lookup.size(2)
        allocation = scores._base
        guards.append(torch.cat([allocation[:n_guard_blocks], allocation[n_guard_blocks + blocks.n_blocks:]]))
        allocation[:n_guard_blocks] = float("nan")
        allocation[n_guard_blocks + blocks.n_blocks:] = float("nan")
        return scores

    monkeypatch.setattr(flash_module, "allocate_referenced_scores", poisoned)
    monkeypatch.setattr(blocks_module, "allocate_referenced_scores", poisoned)
    results = []
    layout_cache = {}
    for score_precalculation in ("referenced_blocks", "complete"):
        inputs = [tensor.clone().requires_grad_(True) for tensor in base]
        output = _attend(case, sparsity_block_size, inputs, score_precalculation,
                         layout_cache if score_precalculation == "referenced_blocks" else None)
        output.backward(output_gradient)
        results.append([output.detach()] + [tensor.grad for tensor in inputs])
    # The scores and their gradients are allocated between zero guard blocks.
    assert len(guards) >= 2 and all(torch.count_nonzero(guard) == 0 for guard in guards)
    for referenced, complete in zip(*results):
        assert torch.isfinite(referenced).all()
        torch.testing.assert_close(referenced, complete, atol=1e-4, rtol=1e-4)

    blocks = layout_cache["relative_coordinates"]["coordinates"].score_blocks
    n_value_blocks = blocks.lookup.size(2)
    first_block = 0
    for relation, (minimum, maximum) in enumerate(bounds):
        end_block = first_block + math.ceil((maximum - minimum + 1) / sparsity_block_size)
        # The value blocks of a relation's columns lie within its table, and every offset of a row is added to them,
        # for every batch and query block, also in rows whose stored blocks of the relation have gaps.
        offsets = blocks.addressing[:, :, n_value_blocks + relation].long()
        assert (offsets + first_block).min() >= -n_value_blocks
        assert (offsets + end_block - 1).max() < blocks.n_blocks + n_value_blocks
        first_block = end_block


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_scores_without_referenced_blocks_match_complete():
    """Without valid relations, as for a sequence of only a start token, no block is referenced, and the interval
    relations still address the zero guard blocks, through a non-empty view.

    """
    generator = torch.Generator().manual_seed(7)
    relations, bounds, attention_layout, lengths, validity = _case("causal", 16, generator)
    case = (relations, bounds, attention_layout, lengths, torch.zeros_like(validity))
    n_batches, length = relations.shape[1:]
    n_relation_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
    base = [torch.randn((n_batches, length, 32), generator=generator).to(DEVICE) for _ in range(3)]
    base.append(torch.randn((n_batches, n_relation_rows, 32), generator=generator).to(DEVICE))
    results = []
    layout_cache = {}
    for score_precalculation in ("referenced_blocks", "complete"):
        inputs = [tensor.clone().requires_grad_(True) for tensor in base]
        output = _attend(case, 16, inputs, score_precalculation,
                         layout_cache if score_precalculation == "referenced_blocks" else None)
        output.sum().backward()
        results.append([output.detach()] + [tensor.grad for tensor in inputs])
    blocks = layout_cache["relative_coordinates"]["coordinates"].score_blocks
    # The relations of the case span few value blocks and are always addressed through intervals.
    assert blocks.n_blocks == 0 and blocks.interval_relations == (1 << len(bounds)) - 1
    for referenced, complete in zip(*results):
        torch.testing.assert_close(referenced, complete, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_score_precalculation_is_validated():
    generator = torch.Generator().manual_seed(5)
    case = _case("causal", 16, generator)
    n_batches, length = case[0].shape[1:]
    n_relation_rows = sum(maximum - minimum + 1 for minimum, maximum in case[1])
    inputs = [torch.randn((n_batches, length, 32), generator=generator).to(DEVICE) for _ in range(3)]
    inputs.append(torch.randn((n_batches, n_relation_rows, 32), generator=generator).to(DEVICE))
    with pytest.raises(ValueError, match="score_precalculation"):
        _attend(case, 16, inputs, "blocks")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype, sparsity_block_size, head_dimension", [
    (torch.float32, 64, 256), (torch.float32, 32, 512), (torch.float16, 64, 512), (torch.float16, 32, 1024)])
def test_relative_score_blocks_chunk_wide_head_dimensions(dtype: torch.dtype, sparsity_block_size: int,
                                                          head_dimension: int):
    """The projection kernels split wide head dimensions into chunks that fit into shared memory, and give the results
    of the complete projection.

    """
    generator = torch.Generator().manual_seed(8)
    n_batches, n_blocks = 1, 3
    length = n_blocks * sparsity_block_size
    relations = torch.arange(length).repeat(n_batches, 1).unsqueeze(0).to(DEVICE)
    bounds = ((-16, 16),)
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    case = (relations, bounds, attention_layout, None, None)
    base = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, dtype)
            for _ in range(3)]
    base.append((torch.randn((n_batches, 33, head_dimension), generator=generator) * 0.3).to(DEVICE, dtype))
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, dtype)
    results = []
    for score_precalculation in ("referenced_blocks", "complete"):
        inputs = [tensor.clone().requires_grad_(True) for tensor in base]
        output = _attend(case, sparsity_block_size, inputs, score_precalculation)
        output.backward(output_gradient)
        results.append([output.detach()] + [tensor.grad for tensor in inputs])
    tolerance = 1e-3 if dtype == torch.float32 else 3e-2
    for referenced, complete in zip(*results):
        torch.testing.assert_close(referenced, complete, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_score_blocks_bound_the_index_width_with_their_guard_blocks(monkeypatch: pytest.MonkeyPatch):
    """Relations addressed through intervals reach the guard blocks around the referenced blocks, so that the size of
    their allocation, not only of the referenced blocks, decides whether the kernels can use 32-bit offsets.

    """
    flash_module = importlib.import_module("blksprs.ops.flash_attention")
    score_module = importlib.import_module("blksprs.ops.relative_score_blocks")
    extents = []
    original = flash_module.can_use_int32_indexing

    def recording(*values):
        extents.append(values)
        return original(*values)

    monkeypatch.setattr(flash_module, "can_use_int32_indexing", recording)
    generator = torch.Generator().manual_seed(9)
    case = _case("causal", 16, generator)
    n_batches, length = case[0].shape[1:]
    n_relation_rows = sum(maximum - minimum + 1 for minimum, maximum in case[1])
    inputs = [torch.randn((n_batches, length, 32), generator=generator).to(DEVICE) for _ in range(3)]
    inputs.append(torch.randn((n_batches, n_relation_rows, 32), generator=generator).to(DEVICE).requires_grad_(True))
    layout_cache = {}
    _attend(case, 16, inputs, "referenced_blocks", layout_cache).sum().backward()
    blocks = layout_cache["relative_coordinates"]["coordinates"].score_blocks
    allocation = score_module.referenced_score_elements(blocks, 16)
    assert allocation > blocks.n_blocks * 16 * 16
    # The checks of the forward and of the backward kernels include the allocation.
    assert sum(any(isinstance(value, int) and value == allocation for value in values) for values in extents) == 2

import importlib
import itertools
import math
from contextlib import contextmanager
from functools import cache

import pytest
import torch
from triton.runtime.errors import OutOfResources

import blksprs as bs
from blksprs.utils.autotuning import get_autotune_configs

DEVICE = torch.device("cuda")
FLASH_ATTENTION_MODULE = importlib.import_module("blksprs.ops.flash_attention")
RELATIVE_MODULE = importlib.import_module("blksprs.ops.relative_flash_attention")
TILE_SIZE_NAMES = ("TRITON_BLOCK_SIZE_Q", "TRITON_BLOCK_SIZE_K")
KERNEL_PROFILES = {
    "flash_attention_kernel": "flash_attention_forward",
    "flash_attention_kernel_bwd_dkdv": "flash_attention_backward_dkdv",
    "flash_attention_kernel_bwd_dq": "flash_attention_backward_dq",
}
DEFAULT_CONFIGS = {
    kernel: get_autotune_configs(profile, mode="DEFAULT", block_size_names=TILE_SIZE_NAMES)
    for kernel, profile in KERNEL_PROFILES.items()
}
# Combinations of configurations whose program and inner tiles of the three kernels span 32, 64, and 128 rows, see
# ``_forced_configs``.
TILE_SIZE_COMBINATIONS = (0, 1, 4, 6)
NAMES = ("output", "query gradient", "key gradient", "value gradient", "table gradient")
# The clipped and the gathered tiles load the same scores. The query and table gradients sum the score gradients in
# another order, which changes single float16 gradients by a unit in the last place, at most a fifth of this tolerance.
# The tolerance is two orders of magnitude below the contribution of a single clipped tile.
REORDERED_TOLERANCE = dict(rtol=2.5e-3, atol_scale=5e-4)
# Float16 operands against a float64 reference, whose errors stay below a fifth of this tolerance.
REFERENCE_TOLERANCE = dict(rtol=1.5e-2, atol_scale=7.5e-3)


@contextmanager
def _forced_configs(combination: int, sparsity_block_size: int):
    """Runs the Flash Attention kernels with one configuration each instead of autotuning them.

    Autotuning skips configurations that exceed the shared memory of the GPU, so the test is skipped for them.

    """
    kernels = {name: getattr(FLASH_ATTENTION_MODULE, name) for name in KERNEL_PROFILES}
    forced = {}
    for name in KERNEL_PROFILES:
        configs = [config for config in DEFAULT_CONFIGS[name]
                   if all(config.kwargs[size] % sparsity_block_size == 0 for size in TILE_SIZE_NAMES)]
        forced[name] = configs[combination % len(configs)]
    saved = {name: (kernel.configs, dict(kernel.cache)) for name, kernel in kernels.items()}
    try:
        for name, kernel in kernels.items():
            kernel.configs = [forced[name]]
            kernel.cache.clear()
        yield
    except OutOfResources as error:
        pytest.skip(f"Configuration exceeds the GPU resources: {error}")
    finally:
        for name, kernel in kernels.items():
            kernel.configs, cache = saved[name]
            kernel.cache.clear()
            kernel.cache.update(cache)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol_scale: float, name: str):
    """Compares at the scale of the expected tensor, as the clipped table rows sum many terms that cancel."""
    torch.testing.assert_close(actual.double(), expected.double(), rtol=rtol,
                               atol=atol_scale * expected.abs().max().item(), msg=lambda message: f"{name}: {message}")


def _query_programs(layout_cache: dict, bounds, sparsity_block_size: int, tile_q: int, tile_k: int) -> list[dict]:
    """Mirrors the tile loops of the programs that own query tiles, see ``flash_attention_kernel``.

    Returns, for every program, its batch, its number of leading tiles with validity masks, its number of leading plain
    tiles that clip the leading relations, and the least coordinate difference of the pairs of every leading plain
    tile for every relation.

    """
    group_blocks = FLASH_ATTENTION_MODULE._schedule_group_blocks(sparsity_block_size)
    schedule = layout_cache[f"query_tile_schedule_{group_blocks}"]
    offsets, blocks = schedule["offsets"].cpu().long(), schedule["blocks"].cpu().long()
    # The leading entries that need no masking under causal lengths.
    plain = schedule["plain"].cpu().long().view(-1, 2)[:, 1]
    coordinates = layout_cache["relative_coordinates"]["coordinates"]
    floors, ceilings = coordinates.query_floors.cpu(), coordinates.key_ceilings.cpu()
    query_invalid = None if coordinates.query_invalid_blocks is None else coordinates.query_invalid_blocks.cpu() != 0
    key_invalid = None if coordinates.key_invalid_blocks is None else coordinates.key_invalid_blocks.cpu() != 0
    n_relations, n_batches, n_blocks = floors.shape
    maxima = torch.tensor([maximum for _, maximum in bounds])[:, None]
    n_clipped_relations = min(n_relations, FLASH_ATTENTION_MODULE._CLIPPED_RELATIONS)
    query_blocks, key_blocks = tile_q // sparsity_block_size, tile_k // sparsity_block_size
    n_groups = -(-n_blocks // group_blocks)
    programs = []
    for batch, tile in itertools.product(range(n_batches), range(-(-n_blocks // query_blocks))):
        row = batch * n_groups + tile * query_blocks // group_blocks
        entries = blocks[offsets[row]:offsets[row + 1]]
        n_tiles = -(-entries.numel() // key_blocks)
        n_plain = int(plain[row]) // key_blocks
        first_block = tile * query_blocks
        last_blocks = entries[(torch.arange(n_plain) + 1) * key_blocks - 1]
        differences = floors[:, batch, first_block, None] - ceilings[:, batch, last_blocks]
        clipping = differences[:n_clipped_relations] >= maxima[:n_clipped_relations]
        n_clipped = int(clipping.sum(dim=1).min()) if n_plain > 0 else 0
        n_validity = 0
        if query_invalid is not None:
            flagged = [bool(key_invalid[batch, entries[index * key_blocks:(index + 1) * key_blocks]].any())
                       for index in range(n_tiles)]
            if query_invalid[batch, first_block:first_block + query_blocks].any():
                n_validity = n_tiles
            elif any(flagged):
                n_validity = max(index for index, value in enumerate(flagged) if value) + 1
        programs.append(dict(batch=batch, n_validity=n_validity, n_clipped=n_clipped, differences=differences))
    return programs


def _program_tiles() -> set[tuple[int, int]]:
    """Returns the tile sizes of the query programs of every configuration that the forward and query-gradient kernels
    may currently select."""
    return {(config.kwargs["TRITON_BLOCK_SIZE_Q"], config.kwargs["TRITON_BLOCK_SIZE_K"])
            for name in ("flash_attention_kernel", "flash_attention_kernel_bwd_dq")
            for config in getattr(FLASH_ATTENTION_MODULE, name).configs}


def _assert_clipped_beyond_validity(layout_cache: dict, bounds, sparsity_block_size: int):
    """Some programs of every configuration reach tiles that clip the leading relations after their validity tiles."""
    for tile_q, tile_k in _program_tiles():
        if tile_q % sparsity_block_size or tile_k % sparsity_block_size:
            continue
        programs = _query_programs(layout_cache, bounds, sparsity_block_size, tile_q, tile_k)
        assert any(program["n_clipped"] > program["n_validity"] for program in programs), (tile_q, tile_k)


def _relative_scores(q, query_relations, key_relations, embeddings, bounds, validity=None):
    """Sums ``q @ R[clip(query_relation - key_relation)]`` over the relations."""
    result = 0.0
    offset = 0
    table = q @ embeddings.transpose(-1, -2)
    for relation, (minimum, maximum) in enumerate(bounds):
        indices = ((query_relations[relation][:, :, None] - key_relations[relation][:, None, :])
                   .clamp(minimum, maximum) - minimum + offset)
        scores = torch.gather(table, 2, indices)
        if validity is not None:
            scores = scores * (validity[relation][:, :, None] & validity[relation][:, None, :])
        result = result + scores
        offset += maximum - minimum + 1
    return result


def _reference(inputs, output_gradient, query_relations, key_relations, bounds, lengths, validity, scale):
    """Dense causal attention with the relative terms in float64, and its gradients."""
    q, k, v, embeddings = [tensor.double().requires_grad_(True) for tensor in inputs]
    length = q.size(1)
    positions = torch.arange(length, device=DEVICE)
    valid = positions[None, :] < lengths[:, None]
    allowed = (positions[:, None] >= positions[None, :]) & valid[:, :, None] & valid[:, None, :]
    scores = (q @ k.transpose(-1, -2)) * scale + _relative_scores(q, query_relations, key_relations, embeddings,
                                                                  bounds, validity)
    has_keys = allowed.any(dim=-1, keepdim=True)
    scores = torch.where(allowed, scores, float("-inf"))
    output = (torch.softmax(torch.where(has_keys, scores, 0.0), dim=-1) * has_keys) @ v
    output.backward(output_gradient.double())
    return [output.detach(), q.grad, k.grad, v.grad, embeddings.grad]


@cache
def _layouts(n_batches: int, n_blocks: int, n_head_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a causal attention layout and a complete operand layout.

    The layouts are shared between calls, as layout caches are tied to the layout tensors.

    """
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    return attention_layout, torch.ones((n_batches, n_blocks, n_head_blocks), dtype=torch.bool, device=DEVICE)


def _attend(inputs, output_gradient, sparsity_block_size, query_relations, key_relations, bounds, layout_cache=None,
            **keywords):
    """Runs relative Flash Attention under a causal layout and returns its output and gradients."""
    inputs = [tensor.clone().requires_grad_(True) for tensor in inputs]
    n_batches, length, head_dimension = inputs[0].shape
    attention_layout, layout = _layouts(n_batches, length // sparsity_block_size,
                                        head_dimension // sparsity_block_size)
    output = bs.ops.flash_attention_relative_embeddings(
        *itertools.chain.from_iterable(
            (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs[:3]),
        attention_layout, sparsity_block_size, query_relations, key_relations, inputs[3], bounds,
        sparsity_layout_o=layout, layout_cache=layout_cache, **keywords)
    output = bs.ops.to_dense(output, layout, sparsity_block_size)
    output.backward(output_gradient)
    return [output.detach()] + [tensor.grad for tensor in inputs]


def _clipped_and_gathered(inputs, output_gradient, sparsity_block_size, query_relations, key_relations, bounds,
                          monkeypatch: pytest.MonkeyPatch, **keywords):
    """Returns the results with clipped tiles, the layout cache of that call, and the results without clipped tiles."""
    layout_cache = {}
    clipped = _attend(inputs, output_gradient, sparsity_block_size, query_relations, key_relations, bounds,
                      layout_cache, **keywords)
    # Coordinates beyond the limit of the coordinate bounds clip no tile.
    with monkeypatch.context() as patch:
        patch.setattr(RELATIVE_MODULE, "_COORDINATE_LIMIT", -1)
        gathered = _attend(inputs, output_gradient, sparsity_block_size, query_relations, key_relations, bounds,
                           **keywords)
    return clipped, layout_cache, gathered


def _assert_clipped_match(clipped, gathered, reference, exact: bool):
    """The clipped tiles match the gathered ones and the float64 reference.

    With ``exact``, the output and the key and value gradients, which sum the same terms in the same order, match
    exactly. Otherwise, the loops of the clipped and the gathered tiles may round their logits differently, which
    changes single float16 elements by a unit in the last place.

    """
    for index, (actual, expected) in enumerate(zip(clipped, gathered)):
        if exact and index not in (1, 4):
            assert torch.equal(actual, expected), NAMES[index]
        else:
            _assert_close(actual, expected, **REORDERED_TOLERANCE, name=NAMES[index])
    for index, (actual, expected) in enumerate(zip(clipped, reference)):
        _assert_close(actual, expected, **REFERENCE_TOLERANCE, name=NAMES[index])


def _onset_case(sparsity_block_size: int, order: str, validity_case: str | None):
    """Positions, onsets, and pitches of two causal sequences, ordered so that the onsets or the positions come first.

    Onsets clip nearer keys than positions, so that the distant tiles clip both, nearer ones only the onsets, and the
    nearest none, whereas pitches are not monotone and never clip whole tiles. Onsets advance every sixth position, so
    that every key tile holds few runs of equal onsets, which the query gradient merges.

    """
    generator = torch.Generator().manual_seed(5)
    n_batches, head_dimension, length = 2, 64, 512
    lengths = torch.tensor([length, length - 3 * sparsity_block_size - 5], dtype=torch.int64, device=DEVICE)
    positions = torch.arange(length, device=DEVICE).repeat(n_batches, 1)
    pitches = torch.randint(0, 12, (n_batches, length), generator=generator).to(DEVICE)
    relations = torch.stack([positions, positions // 6, pitches])
    bounds = ((-128, 128), (0, 16), (-11, 11))
    unique, repeat = (True, False, False), (False, True, False)
    validity = None
    if validity_case == "start token":
        # Only the start token has no valid relations, so that only the first tiles need the validity masks.
        validity = torch.ones(relations.shape, dtype=torch.bool, device=DEVICE)
        validity[:, :, 0] = False
    if order == "onsets first":
        permutation = [1, 0, 2]
        relations = relations[permutation]
        validity = None if validity is None else validity[permutation]
        bounds, unique, repeat = ([values[index] for index in permutation] for values in (bounds, unique, repeat))
    inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
              for _ in range(3)]
    inputs.append((torch.randn((n_batches, 257 + 17 + 23, head_dimension), generator=generator) * 0.3).to(
        DEVICE, torch.float16))
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
    keywords = dict(key_relations_are_unique=tuple(unique), key_relations_repeat=tuple(repeat),
                    causal_lengths=lengths)
    if validity is not None:
        keywords.update(query_relation_validity=validity, key_relation_validity=validity)
    return inputs, output_gradient, relations, tuple(bounds), lengths, validity, keywords


def _assert_merged_runs(relations, repeat, tile_sizes):
    """The repeated relations of every key tile form few enough runs to be merged, see ``_key_runs``."""
    run_capacity = FLASH_ATTENTION_MODULE._RUN_CAPACITY.value
    for relation, keys_repeat in enumerate(repeat):
        if keys_repeat:
            for tile_k in tile_sizes:
                tiles = relations[relation].view(relations.size(1), -1, tile_k)
                assert int(((tiles.diff(dim=-1) != 0).sum(dim=-1) + 1).max()) <= run_capacity


def _run_onset_case(sparsity_block_size: int, order: str, validity_case: str | None,
                    monkeypatch: pytest.MonkeyPatch, exact: bool):
    inputs, output_gradient, relations, bounds, lengths, validity, keywords = _onset_case(sparsity_block_size, order,
                                                                                         validity_case)
    clipped, layout_cache, gathered = _clipped_and_gathered(inputs, output_gradient, sparsity_block_size, relations,
                                                            relations, bounds, monkeypatch, **keywords)
    _assert_clipped_beyond_validity(layout_cache, bounds, sparsity_block_size)
    _assert_merged_runs(relations, keywords["key_relations_repeat"],
                        {tile_k for _, tile_k in _program_tiles() if tile_k % sparsity_block_size == 0})
    # The coordinates are prepared once per layout cache and coordinate tensors.
    with monkeypatch.context() as patch:
        patch.setattr(RELATIVE_MODULE, "_prepare_relation_coordinates", None)
        cached = _attend(inputs, output_gradient, sparsity_block_size, relations, relations, bounds, layout_cache,
                         **keywords)
    for index, (actual, expected) in enumerate(zip(cached, clipped)):
        if index in (1, 4):
            _assert_close(actual, expected, **REORDERED_TOLERANCE, name=NAMES[index])
        else:
            assert torch.equal(actual, expected), NAMES[index]
    reference = _reference(inputs, output_gradient, relations, relations, bounds, lengths, validity,
                           1 / math.sqrt(inputs[0].size(2)))
    _assert_clipped_match(clipped, gathered, reference, exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sparsity_block_size, order, validity_case", [
    (16, "positions first", "start token"),
    (16, "onsets first", None),
    (32, "positions first", None),
    (32, "onsets first", "start token"),
])
def test_flash_attention_relative_clipped_tiles_match_gathered_scores(sparsity_block_size: int, order: str,
                                                                      validity_case: str | None,
                                                                      monkeypatch: pytest.MonkeyPatch):
    """Tiles that clip relations to their maximum use its table row and match gathering it for every pair.

    The tiles that clip both leading relations take them from their maximum in either order, whereas the tiles that
    only clip the onsets gather them, see ``_clipped_tile_count``. Without validity masks, or with an invalid start
    token, the programs reach these tiles after their validity tiles, and the query gradient merges the runs of equal
    onsets in the tiles that do not clip them. The output and the key and value gradients recompute the same scores in
    every tile and match exactly, whereas the query and table gradients agree up to the order of summation of the score
    gradients. Both match a float64 reference.

    """
    _run_onset_case(sparsity_block_size, order, validity_case, monkeypatch, exact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("combination", TILE_SIZE_COMBINATIONS)
def test_flash_attention_relative_clipped_tiles_match_gathered_scores_for_every_tile_size(
        combination: int, monkeypatch: pytest.MonkeyPatch):
    """The clipped and merged tiles of the autotuning configurations, whose tiles span 32 to 128 rows."""
    with _forced_configs(combination, 32):
        _run_onset_case(32, "positions first", "start token", monkeypatch, exact=True)


def _boundary_case(sparsity_block_size: int, arrangement: str):
    """Coordinates whose least difference in some tiles is one below the maximum of a relation and in others equal to it.

    The boundary relation is constant within every sparsity block, so that every pair of two blocks has the same
    difference. The difference of the first query block of a program and the last block of a key tile is a multiple of
    the block size, and the queries of the second sequence are one ahead. The maximum is one above the difference of
    blocks nine apart, which separates a program block from the last block of a key tile in every configuration, so
    that the first sequence holds tiles one below the maximum and the second tiles that reach it exactly. Positions
    with a narrow table clip all of these tiles.

    """
    generator = torch.Generator().manual_seed(31)
    n_batches, head_dimension, length = 2, 64, 512
    positions = torch.arange(length, device=DEVICE).repeat(n_batches, 1)
    block_coordinates = positions // sparsity_block_size * sparsity_block_size
    ahead = torch.tensor([[0], [1]], device=DEVICE)
    boundary_bounds = (-4, 9 * sparsity_block_size + 1)
    position_bounds = (-8, 8)
    if arrangement == "only":
        query_relations, key_relations = (block_coordinates + ahead)[None], block_coordinates[None]
        bounds = (boundary_bounds,)
    else:
        query_relations = torch.stack([block_coordinates + ahead, positions])
        key_relations = torch.stack([block_coordinates, positions])
        bounds = (boundary_bounds, position_bounds)
        if arrangement == "second":
            query_relations, key_relations, bounds = query_relations.flip(0), key_relations.flip(0), bounds[::-1]
    n_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
    inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
              for _ in range(3)]
    inputs.append((torch.randn((n_batches, n_rows, head_dimension), generator=generator) * 0.3).to(DEVICE,
                                                                                                    torch.float16))
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
    lengths = torch.full((n_batches,), length, dtype=torch.int64, device=DEVICE)
    return inputs, output_gradient, query_relations.contiguous(), key_relations.contiguous(), bounds, lengths, \
        bounds.index(boundary_bounds)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sparsity_block_size, arrangement", [(16, "only"), (32, "only"), (16, "second"), (32, "first")])
def test_flash_attention_relative_tiles_clip_from_the_maximum_on(sparsity_block_size: int, arrangement: str,
                                                                monkeypatch: pytest.MonkeyPatch):
    """A tile clips a relation if its least difference reaches the maximum, but not if it is one below it.

    The boundary relation is the only relation, which takes the clipped loops of a single relation, or the first or
    the second of two relations.

    """
    inputs, output_gradient, query_relations, key_relations, bounds, lengths, boundary = _boundary_case(
        sparsity_block_size, arrangement)
    clipped, layout_cache, gathered = _clipped_and_gathered(inputs, output_gradient, sparsity_block_size,
                                                            query_relations, key_relations, bounds, monkeypatch,
                                                            causal_lengths=lengths)
    maximum = bounds[boundary][1]
    for tile_q, tile_k in _program_tiles():
        if tile_q % sparsity_block_size or tile_k % sparsity_block_size:
            continue
        programs = _query_programs(layout_cache, bounds, sparsity_block_size, tile_q, tile_k)
        for batch, difference in ((0, maximum - 1), (1, maximum)):
            assert any((program["differences"][boundary] == difference).any()
                       for program in programs if program["batch"] == batch), (tile_q, tile_k, batch)
        n_clipped = [sum(program["n_clipped"] for program in programs if program["batch"] == batch)
                     for batch in range(2)]
        assert n_clipped[1] > n_clipped[0] > 0, (tile_q, tile_k)
    reference = _reference(inputs, output_gradient, query_relations, key_relations, bounds, lengths, None,
                           1 / math.sqrt(inputs[0].size(2)))
    _assert_clipped_match(clipped, gathered, reference, exact=False)


def _relation_count_case(n_relations: int):
    """Up to eight relations of two causal sequences, the first two of which clip distant tiles.

    The relations are positions, onsets, pitches, positions within and numbers of bars, voices, half positions with a
    table of more than eight value blocks, and a second pitch. Onsets and bars repeat their keys. Only the start token
    has no valid relations.

    """
    generator = torch.Generator().manual_seed(37)
    n_batches, head_dimension, length, sparsity_block_size = 2, 64, 512, 32
    positions = torch.arange(length, device=DEVICE).repeat(n_batches, 1)

    def random(high):
        return torch.randint(0, high, (n_batches, length), generator=generator).to(DEVICE)

    relations = torch.stack([positions, positions // 6, random(12), positions % 48, positions // 48, random(4),
                             positions // 2, random(24)])[:n_relations]
    bounds = ((-100, 100), (0, 20), (-11, 11), (-47, 47), (0, 3), (-3, 3), (-300, 300), (-23, 23))[:n_relations]
    unique = (True,) + (False,) * (n_relations - 1)
    repeat = tuple(relation in (1, 4) for relation in range(n_relations))
    validity = torch.ones(relations.shape, dtype=torch.bool, device=DEVICE)
    validity[:, :, 0] = False
    lengths = torch.tensor([length, length - 2 * sparsity_block_size - 3], dtype=torch.int64, device=DEVICE)
    n_rows = sum(maximum - minimum + 1 for minimum, maximum in bounds)
    inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
              for _ in range(3)]
    inputs.append((torch.randn((n_batches, n_rows, head_dimension), generator=generator) * 0.3).to(DEVICE,
                                                                                                    torch.float16))
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
    keywords = dict(key_relations_are_unique=unique, key_relations_repeat=repeat, causal_lengths=lengths,
                    query_relation_validity=validity, key_relation_validity=validity)
    return inputs, output_gradient, relations, bounds, lengths, validity, keywords


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n_relations", [1, 5, 8])
def test_flash_attention_relative_relation_counts_clip_and_match_reference(n_relations: int,
                                                                           monkeypatch: pytest.MonkeyPatch):
    """One relation, which takes the clipped loops of a single relation, and five and eight relations, which hold
    their bound gradients in eight slots, clip distant tiles and match a float64 reference."""
    sparsity_block_size = 32
    inputs, output_gradient, relations, bounds, lengths, validity, keywords = _relation_count_case(n_relations)
    clipped, layout_cache, gathered = _clipped_and_gathered(inputs, output_gradient, sparsity_block_size, relations,
                                                            relations, bounds, monkeypatch, **keywords)
    _assert_clipped_beyond_validity(layout_cache, bounds, sparsity_block_size)
    reference = _reference(inputs, output_gradient, relations, relations, bounds, lengths, validity,
                           1 / math.sqrt(inputs[0].size(2)))
    _assert_clipped_match(clipped, gathered, reference, exact=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("score_precalculation", ["referenced_blocks", "complete"])
@pytest.mark.parametrize("single_row", ["first", "second"])
def test_flash_attention_relative_single_row_relations_receive_clipped_gradients(single_row: str,
                                                                                score_precalculation: str,
                                                                                monkeypatch: pytest.MonkeyPatch):
    """A relation whose minimum equals its maximum has a single table row for both bounds. The clipped tiles sum its
    gradients as those of the maximum, which the query gradient stores in that row."""
    sparsity_block_size = 16
    generator = torch.Generator().manual_seed(41)
    n_batches, head_dimension, length = 2, 64, 512
    positions = torch.arange(length, device=DEVICE).repeat(n_batches, 1)
    relations = torch.stack([positions, positions // 3])
    bounds = ((0, 0), (-16, 16))
    if single_row == "second":
        relations, bounds = relations.flip(0).contiguous(), bounds[::-1]
    lengths = torch.tensor([length, length - 37], dtype=torch.int64, device=DEVICE)
    inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
              for _ in range(3)]
    inputs.append((torch.randn((n_batches, 1 + 33, head_dimension), generator=generator) * 0.3).to(DEVICE,
                                                                                                  torch.float16))
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
    clipped, layout_cache, gathered = _clipped_and_gathered(inputs, output_gradient, sparsity_block_size, relations,
                                                            relations, bounds, monkeypatch, causal_lengths=lengths,
                                                            score_precalculation=score_precalculation)
    _assert_clipped_beyond_validity(layout_cache, bounds, sparsity_block_size)
    reference = _reference(inputs, output_gradient, relations, relations, bounds, lengths, None,
                           1 / math.sqrt(inputs[0].size(2)))
    _assert_clipped_match(clipped, gathered, reference, exact=False)

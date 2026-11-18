import importlib
import itertools
import math
from contextlib import contextmanager

import pytest
import torch
from triton.runtime.errors import OutOfResources

import blksprs as bs
from blksprs.utils.autotuning import get_autotune_configs

DEVICE = torch.device("cuda")
FLASH_ATTENTION_MODULE = importlib.import_module("blksprs.ops.flash_attention")
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
# Every configuration of every kernel appears in at least one combination.
N_CONFIG_COMBINATIONS = max(len(configs) for configs in DEFAULT_CONFIGS.values())


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


def _expand(layout: torch.Tensor, sparsity_block_size: int) -> torch.Tensor:
    return layout.repeat_interleave(sparsity_block_size, dim=1).repeat_interleave(sparsity_block_size, dim=2)


def _reference(q, k, v, allowed, scale, bias=None, relative=None):
    """Dense attention in float32, where ``allowed`` marks the participating query-key pairs."""
    scores = (q @ k.transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias
    if relative is not None:
        scores = scores + relative
    has_keys = allowed.any(dim=-1, keepdim=True)
    scores = torch.where(allowed, scores, float("-inf"))
    probabilities = torch.softmax(torch.where(has_keys, scores, 0.0), dim=-1) * has_keys
    return probabilities @ v


def _relative_scores(q, query_relations, key_relations, embeddings, bounds, validity=None):
    """Sums ``q @ R[clip(query_relation - key_relation)]`` over the relations."""
    result = 0.0
    offset = 0
    for relation, (minimum, maximum) in enumerate(bounds):
        indices = ((query_relations[relation][:, :, None] - key_relations[relation][:, None, :])
                   .clamp(minimum, maximum) - minimum + offset)
        table = q @ embeddings.transpose(-1, -2)
        scores = torch.gather(table, 2, indices)
        if validity is not None:
            scores = scores * (validity[relation][:, :, None] & validity[relation][:, None, :])
        result = result + scores
        offset += maximum - minimum + 1
    return result


def _random_layouts(n_batches, n_blocks, sparsity_block_size, generator):
    attention_layout = torch.tril(torch.rand((n_batches, n_blocks, n_blocks), generator=generator) < 0.7)
    # Keep the diagonal so that every query attends to itself, and make some rows empty.
    attention_layout |= torch.eye(n_blocks, dtype=torch.bool)
    attention_layout[0, 1] = False
    mask_layout = attention_layout & (torch.rand(attention_layout.shape, generator=generator) < 0.4)
    bias_layout = attention_layout & (torch.rand(attention_layout.shape, generator=generator) < 0.6)
    return attention_layout.to(DEVICE), mask_layout.to(DEVICE), bias_layout.to(DEVICE)


def _run_case(sparsity_block_size: int, *, n_blocks: int, head_dimension: int, value_dimension: int,
              dtype: torch.dtype, use_mask: bool, use_bias: bool, causal: bool, dense: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    n_batches = 2
    length = n_blocks * sparsity_block_size
    if dense:
        attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
        mask_layout = bias_layout = attention_layout
    else:
        attention_layout, mask_layout, bias_layout = _random_layouts(n_batches, n_blocks, sparsity_block_size,
                                                                     generator)
    # Lengths that end inside a sparsity block, inside a tile, and at the end of the sequence.
    lengths = torch.tensor([length - 5, length], dtype=torch.int64, device=DEVICE)
    if n_blocks > 4:
        lengths[1] = 3 * sparsity_block_size + 7

    q, k = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE) for _ in range(2))
    v = torch.randn((n_batches, length, value_dimension), generator=generator).to(DEVICE)
    mask = (torch.rand((n_batches, length, length), generator=generator) < 0.3).to(DEVICE) \
        & _expand(mask_layout, sparsity_block_size)
    bias = (torch.randn((n_batches, length, length), generator=generator).to(DEVICE)
            * _expand(bias_layout, sparsity_block_size))

    allowed = _expand(attention_layout, sparsity_block_size)
    if use_mask:
        allowed = allowed & ~mask
    if causal:
        positions = torch.arange(length, device=DEVICE)
        valid = positions[None, :] < lengths[:, None]
        allowed = allowed & (positions[None, :, None] >= positions[None, None, :]) \
            & valid[:, :, None] & valid[:, None, :]
    scale = 1 / math.sqrt(head_dimension)

    reference_inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, bias)]
    reference = _reference(*reference_inputs[:3], allowed, scale,
                           bias=reference_inputs[3] if use_bias else None)

    layout_qk = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                           device=DEVICE)
    layout_v = torch.ones((n_batches, n_blocks, value_dimension // sparsity_block_size), dtype=torch.bool,
                          device=DEVICE)
    inputs = [tensor.to(dtype).requires_grad_(True) for tensor in (q, k, v)]
    bias_sparse = bs.ops.to_sparse(bias.to(dtype), bias_layout, sparsity_block_size).detach().requires_grad_(True)
    arguments = {}
    if use_mask:
        arguments.update(attention_mask=bs.ops.to_sparse(mask, mask_layout, sparsity_block_size),
                         sparsity_layout_mask=mask_layout)
    if use_bias:
        arguments.update(attention_bias=bias_sparse, sparsity_layout_bias=bias_layout)
    if causal:
        arguments.update(causal_lengths=lengths)
    output = bs.ops.flash_attention(
        bs.ops.to_sparse(inputs[0], layout_qk, sparsity_block_size), layout_qk,
        bs.ops.to_sparse(inputs[1], layout_qk, sparsity_block_size), layout_qk,
        bs.ops.to_sparse(inputs[2], layout_v, sparsity_block_size), layout_v,
        attention_layout, sparsity_block_size, scale=scale, sparsity_layout_o=layout_v, **arguments)
    output = bs.ops.to_dense(output, layout_v, sparsity_block_size)

    output_gradient = torch.randn(reference.shape, generator=generator).to(DEVICE)
    reference.backward(output_gradient)
    output.backward(output_gradient.to(dtype))

    # Float32 operands use TF32 tensor-core products.
    tolerance = dict(atol=2e-2, rtol=2e-2) if dtype != torch.float32 else dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(output.float(), reference, **tolerance)
    for actual, expected in zip(inputs, reference_inputs[:3]):
        torch.testing.assert_close(actual.grad.float(), expected.grad, **tolerance)
    if use_bias:
        expected_bias_gradient = bs.ops.to_sparse(reference_inputs[3].grad, bias_layout, sparsity_block_size)
        torch.testing.assert_close(bias_sparse.grad.float(), expected_bias_gradient, **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("combination", range(N_CONFIG_COMBINATIONS))
@pytest.mark.parametrize("sparsity_block_size", [16, 32, 64])
@pytest.mark.parametrize("use_mask, use_bias, causal, dense", [
    (False, False, True, True),
    (False, False, False, False),
    (True, True, True, False),
])
def test_flash_attention_tiles_match_reference_for_every_config(combination: int, sparsity_block_size: int,
                                                                use_mask: bool, use_bias: bool, causal: bool,
                                                                dense: bool):
    """Flash Attention tiles that span several sparsity blocks match a dense reference.

    The tiles of the Flash Attention kernels gather the blocks that a group of program blocks attends to, so that the
    sub-blocks of one tile can differ in their layout, mask, bias, and causal extent. The tests force every autotuning
    configuration of the kernels, which covers tiles that span one or several sparsity blocks and tiles that cover part
    of a tile schedule group.

    """
    if sparsity_block_size == 64 and combination >= 4:
        pytest.skip("Only a few configurations span whole 64-element blocks")
    with _forced_configs(combination, sparsity_block_size):
        _run_case(sparsity_block_size, n_blocks=320 // sparsity_block_size + 1, head_dimension=64,
                  value_dimension=64, dtype=torch.float16, use_mask=use_mask, use_bias=use_bias, causal=causal,
                  dense=dense, seed=combination)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype, head_dimension, value_dimension", [
    # One head-dimension chunk, several Q/K chunks and V chunks, and chunks of float32 operands.
    (torch.float16, 96, 32),
    (torch.float16, 192, 320),
    (torch.float32, 128, 64),
])
@pytest.mark.parametrize("use_mask, use_bias", [(False, False), (True, True)])
def test_flash_attention_tiles_chunk_wide_head_dimensions(dtype: torch.dtype, head_dimension: int, value_dimension: int,
                                                          use_mask: bool, use_bias: bool):
    _run_case(32, n_blocks=7, head_dimension=head_dimension, value_dimension=value_dimension, dtype=dtype,
              use_mask=use_mask, use_bias=use_bias, causal=True, dense=False, seed=head_dimension)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sparsity_block_size", [16, 32])
def test_flash_attention_tiles_gather_head_sparse_operands(sparsity_block_size: int):
    """Sequence blocks that hold only some of their head-dimension blocks are gathered block by block."""
    generator = torch.Generator().manual_seed(7)
    n_batches, n_blocks, n_head_blocks = 2, 9, 3
    length, head_dimension = n_blocks * sparsity_block_size, n_head_blocks * sparsity_block_size
    attention_layout = torch.tril(torch.rand((n_batches, n_blocks, n_blocks), generator=generator) < 0.8).to(DEVICE)
    layouts = [(torch.rand((n_batches, n_blocks, n_head_blocks), generator=generator) < 0.7).to(DEVICE)
               for _ in range(3)]
    dense = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE)
             * _expand(layout, sparsity_block_size) for layout in layouts]
    inputs = [tensor.clone().requires_grad_(True) for tensor in dense]

    output_layout = bs.layouting.build_sparsity_layout_matmul(attention_layout, layouts[2])
    output = bs.ops.flash_attention(
        *itertools.chain.from_iterable(
            (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout)
            for tensor, layout in zip(inputs, layouts)),
        attention_layout, sparsity_block_size)
    output = bs.ops.to_dense(output, output_layout, sparsity_block_size)

    reference_inputs = [tensor.clone().requires_grad_(True) for tensor in dense]
    reference = _reference(*reference_inputs, _expand(attention_layout, sparsity_block_size),
                           1 / math.sqrt(head_dimension)) * _expand(output_layout, sparsity_block_size)
    output_gradient = torch.randn(reference.shape, generator=generator).to(DEVICE)
    reference.backward(output_gradient)
    output.backward(output_gradient)

    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    for actual, expected, layout in zip(inputs, reference_inputs, layouts):
        torch.testing.assert_close(actual.grad, expected.grad * _expand(layout, sparsity_block_size),
                                   atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("combination", range(N_CONFIG_COMBINATIONS))
@pytest.mark.parametrize("sparsity_block_size", [16, 32])
def test_flash_attention_relative_tiles_match_reference_for_every_config(combination: int, sparsity_block_size: int):
    """Relative terms with repeating key coordinates, validity, and unaligned causal lengths."""
    generator = torch.Generator().manual_seed(combination)
    n_batches, n_blocks, head_dimension = 2, 320 // sparsity_block_size + 1, 64
    length = n_blocks * sparsity_block_size
    attention_layout, _, _ = _random_layouts(n_batches, n_blocks, sparsity_block_size, generator)
    lengths = torch.tensor([length - 5, 3 * sparsity_block_size + 7], dtype=torch.int64, device=DEVICE)
    bounds = ((-8, 8), (0, 5))
    positions = torch.arange(length, device=DEVICE)
    query_relations = torch.stack([positions.repeat(n_batches, 1), (positions // 3).repeat(n_batches, 1)])
    validity = (torch.rand((2, n_batches, length), generator=generator) < 0.8).to(DEVICE)
    embeddings = (torch.randn((n_batches, 17 + 6, head_dimension), generator=generator) * 0.3).to(DEVICE)
    q, k, v = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE) for _ in range(3))

    valid = positions[None, :] < lengths[:, None]
    allowed = (_expand(attention_layout, sparsity_block_size) & (positions[:, None] >= positions[None, :])
               & valid[:, :, None] & valid[:, None, :])
    reference_inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, embeddings)]
    relative = _relative_scores(reference_inputs[0], query_relations, query_relations, reference_inputs[3], bounds,
                                validity)
    reference = _reference(*reference_inputs[:3], allowed, 1 / math.sqrt(head_dimension), relative=relative)

    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                        device=DEVICE)
    inputs = [tensor.to(torch.float16).requires_grad_(True) for tensor in (q, k, v, embeddings)]
    with _forced_configs(combination, sparsity_block_size):
        output = bs.ops.flash_attention_relative_embeddings(
            bs.ops.to_sparse(inputs[0], layout, sparsity_block_size), layout,
            bs.ops.to_sparse(inputs[1], layout, sparsity_block_size), layout,
            bs.ops.to_sparse(inputs[2], layout, sparsity_block_size), layout,
            attention_layout, sparsity_block_size, query_relations, query_relations, inputs[3], bounds,
            key_relations_are_unique=(True, False), causal_lengths=lengths,
            query_relation_validity=validity, key_relation_validity=validity, sparsity_layout_o=layout)
        output = bs.ops.to_dense(output, layout, sparsity_block_size)
        output_gradient = torch.randn(reference.shape, generator=generator).to(DEVICE)
        reference.backward(output_gradient)
        output.backward(output_gradient.to(torch.float16))

    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(inputs, reference_inputs):
        torch.testing.assert_close(actual.grad.float(), expected.grad, atol=5e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_relative_coordinates_wider_than_32_bits():
    """Coordinates beyond 32 bits use 64-bit arithmetic and give the same result, as only clipped differences matter.

    The coordinates of the second sequence jump far ahead within a block, so that the pairs across the jump clip to a
    bound of every relation, in the tiles that clip whole and in those that gather every pair. The wide coordinates are
    offset by more than 2**31 and jump by more than 2**31, so that their differences across the jump do not fit into
    32 bits.

    """
    generator = torch.Generator().manual_seed(11)
    n_batches, sparsity_block_size, n_blocks, head_dimension = 2, 32, 6, 64
    length = n_blocks * sparsity_block_size
    attention_layout, _, _ = _random_layouts(n_batches, n_blocks, sparsity_block_size, generator)
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                        device=DEVICE)
    positions = torch.arange(length, device=DEVICE)
    relations = torch.stack([positions.repeat(n_batches, 1), (positions // 3).repeat(n_batches, 1)])
    jumped = (positions[None, :] >= length // 2 + 5) & (torch.arange(n_batches, device=DEVICE)[:, None] == 1)
    bounds = ((-8, 8), (0, 5))
    q, k, v = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE) for _ in range(3))
    embeddings = (torch.randn((n_batches, 17 + 6, head_dimension), generator=generator) * 0.3).to(DEVICE)
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE)

    results = []
    for coordinate_offset, jump in ((0, 10 ** 6), (3 * 2 ** 30 + 7, 2 ** 31 + 12_345)):
        inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, embeddings)]
        shifted = relations + coordinate_offset + jump * jumped
        layout_cache = {}
        output = bs.ops.flash_attention_relative_embeddings(
            *itertools.chain.from_iterable(
                (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs[:3]),
            attention_layout, sparsity_block_size, shifted, shifted, inputs[3], bounds,
            key_relations_are_unique=(True, False), sparsity_layout_o=layout, layout_cache=layout_cache)
        output = bs.ops.to_dense(output, layout, sparsity_block_size)
        output.backward(output_gradient)
        results.append([output.detach()] + [tensor.grad for tensor in inputs])
        coordinates = layout_cache["relative_coordinates"]["coordinates"]
        assert coordinates.query_relations.dtype == (torch.int32 if coordinate_offset == 0 else torch.int64)

    for narrow, wide in zip(*results):
        torch.testing.assert_close(narrow, wide, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("score_precalculation", ["referenced_blocks", "complete"])
@pytest.mark.parametrize("case", ["padding", "padding with validity", "explicit mask", "lookup", "padding queries"])
def test_flash_attention_relative_unique_keys_ignore_pairs_that_are_not_permitted(case: str,
                                                                                  score_precalculation: str):
    """Relations with unique key coordinates store their interior gradients without atomics. Pairs that are not
    permitted, such as padding keys whose coordinate 0 repeats that of the first key, masked pairs, or padding queries
    in tiles without masking, whose columns may lie in the blocks of other queries, must not overwrite them.

    """
    generator = torch.Generator().manual_seed(17)
    sparsity_block_size, n_batches, head_dimension = 16, 2, 64
    # Coordinates of twice the block size leave every other value block unreferenced, which requires the lookup.
    stride = 2 * sparsity_block_size if case == "lookup" else 1
    n_blocks = 512 // sparsity_block_size if case == "padding queries" else 6
    length = n_blocks * sparsity_block_size
    positions = torch.arange(length, device=DEVICE)
    if case == "padding queries":
        # Many query tiles lie beyond the causal length of the first batch and are processed without masking.
        lengths = torch.tensor([length // 2 - 5, length - 3], dtype=torch.int64, device=DEVICE)
    else:
        lengths = torch.tensor([length - sparsity_block_size // 2 - 3, length - 3], dtype=torch.int64, device=DEVICE)
    valid = positions[None, :] < lengths[:, None]
    coordinates = (positions * stride).repeat(n_batches, 1)
    padded = torch.where(valid, coordinates, 0)
    query_relations = padded.unsqueeze(0)
    key_relations = (coordinates if case == "padding queries" else padded).unsqueeze(0)
    span = stride * length
    bounds = ((-span, span),)
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                        device=DEVICE)
    q, k, v = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE) for _ in range(3))
    embeddings = (torch.randn((n_batches, 2 * span + 1, head_dimension), generator=generator) * 0.3).to(DEVICE)
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE)
    allowed = (_expand(attention_layout, sparsity_block_size) & (positions[:, None] >= positions[None, :])
               & valid[:, :, None] & valid[:, None, :])
    keywords = {"causal_lengths": lengths}
    validity = None
    if case == "padding with validity":
        validity = valid.unsqueeze(0).contiguous()
        keywords.update(query_relation_validity=validity, key_relation_validity=validity)
    elif case == "explicit mask":
        # The future keys and the padding positions are masked explicitly instead of through the causal lengths.
        keywords = {"attention_mask": bs.ops.to_sparse(~allowed, attention_layout, sparsity_block_size),
                    "sparsity_layout_mask": attention_layout}

    reference_inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, embeddings)]
    relative = _relative_scores(reference_inputs[0], query_relations, key_relations, reference_inputs[3], bounds,
                                validity)
    reference = _reference(*reference_inputs[:3], allowed, 1 / math.sqrt(head_dimension), relative=relative)
    reference.backward(output_gradient)

    results = []
    for unique in (True, False):
        layout_cache: dict = {}
        inputs = [tensor.to(torch.float16).requires_grad_(True) for tensor in (q, k, v, embeddings)]
        output = bs.ops.flash_attention_relative_embeddings(
            *itertools.chain.from_iterable(
                (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs[:3]),
            attention_layout, sparsity_block_size, query_relations, key_relations, inputs[3], bounds,
            # Pairwise handling for both, so that the runs differ only in their stores and atomics.
            key_relations_are_unique=(unique,), key_relations_repeat=(False,), sparsity_layout_o=layout,
            layout_cache=layout_cache, score_precalculation=score_precalculation, **keywords)
        output = bs.ops.to_dense(output, layout, sparsity_block_size)
        output.backward(output_gradient.to(torch.float16))
        results.append([output.detach()] + [tensor.grad for tensor in inputs])
        if score_precalculation == "referenced_blocks":
            score_blocks = layout_cache["relative_coordinates"]["coordinates"].score_blocks
            # Without gaps, the relation is addressed through intervals, otherwise looked up.
            assert score_blocks.complete_intervals == (case != "lookup")

    for actual, expected in zip(results[0], [reference.detach()] + [tensor.grad for tensor in reference_inputs]):
        torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=2e-2)
    # Every interior gradient has a single non-zero term, so that stores and atomics give identical results.
    for unique_result, atomic_result in zip(*results):
        assert torch.equal(unique_result, atomic_result)


def test_flash_attention_relative_repeated_key_relations_are_validated():
    """Relations with repeated keys are one flag per relation, and relations with unique keys cannot repeat them."""
    relative_module = importlib.import_module("blksprs.ops.relative_flash_attention")
    normalise = relative_module._normalise_repeated_key_relations
    assert normalise(None, (True, False, False)) == (False, True, True)
    assert normalise([False, True, False], (True, False, False)) == (False, True, False)
    with pytest.raises(ValueError, match="unique"):
        normalise((True, True, False), (True, False, False))
    with pytest.raises(ValueError, match="one value per relation"):
        normalise((True,), (False, False))
    with pytest.raises(TypeError, match="bool"):
        normalise((1, 0), (False, False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_relative_repeated_key_relations_match_pairs():
    """Relations with repeated keys are merged jointly where key tiles hold few runs, and a relation whose keys change
    at almost every key, if marked as well, leaves the tiles per pair. Every choice matches handling every relation
    per pair, up to the order of summation.

    """
    generator = torch.Generator().manual_seed(12)
    n_batches, sparsity_block_size, head_dimension = 3, 32, 64
    length = 12 * sparsity_block_size
    # Onsets with runs of about five keys, bars of about seven onsets, and pitch classes, which rarely repeat.
    onsets = torch.cumsum((torch.rand((n_batches, length), generator=generator) < 0.2).long(), dim=1)
    relations = torch.stack([onsets, onsets // 7, torch.randint(0, 12, (n_batches, length), generator=generator)])
    lengths = torch.tensor([length, 4 * sparsity_block_size + 5, 6 * sparsity_block_size])

    n_blocks = length // sparsity_block_size
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    bounds = ((0, 40), (0, 6), (-11, 11))
    q, k, v = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE) for _ in range(3))
    embeddings = (torch.randn((n_batches, 41 + 7 + 23, head_dimension), generator=generator) * 0.3).to(DEVICE)
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE)

    def attend(key_relations_repeat):
        inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, embeddings)]
        output = bs.ops.flash_attention_relative_embeddings(
            *itertools.chain.from_iterable(
                (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs[:3]),
            attention_layout, sparsity_block_size, relations.to(DEVICE), relations.to(DEVICE), inputs[3], bounds,
            key_relations_repeat=key_relations_repeat, causal_lengths=lengths.to(DEVICE), sparsity_layout_o=layout)
        output = bs.ops.to_dense(output, layout, sparsity_block_size)
        output.backward(output_gradient)
        return [output.detach()] + [tensor.grad for tensor in inputs]

    pairwise = attend((False, False, False))
    for key_relations_repeat in ((True, True, False), None):
        for actual, expected in zip(attend(key_relations_repeat), pairwise):
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# Float32 operands fit the shared memory of common GPUs only with the first configuration, so that float16 operands,
# as in mixed-precision training, cover every configuration.
@pytest.mark.parametrize("dtype, combination", [
    pytest.param(dtype, combination, id=f"{str(dtype).removeprefix('torch.')}-{combination}")
    for dtype, combination in [(torch.float32, 0)] + [(torch.float16, index) for index in range(N_CONFIG_COMBINATIONS)]
])
@pytest.mark.parametrize("invalid_positions", [None, "start", "scattered"])
@pytest.mark.parametrize("score_precalculation", ["referenced_blocks", "complete"])
def test_flash_attention_relative_merged_key_runs_match_pairs(score_precalculation: str, invalid_positions: str | None,
                                                              dtype: torch.dtype, combination: int):
    """Relations without unique keys handled once per run of equal keys match handling every pair, up to the order
    of summation. With validity masks, only the tiles without invalid positions are merged. Invalid positions at
    the start of every sequence, as for a start token, leave most tiles without validity masks.

    """
    generator = torch.Generator().manual_seed(9)
    n_batches, sparsity_block_size, head_dimension = 2, 32, 64
    n_blocks = 768 // sparsity_block_size
    length = n_blocks * sparsity_block_size
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    lengths = torch.tensor([length, length - 2 * sparsity_block_size - 3], dtype=torch.int64, device=DEVICE)
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                        device=DEVICE)
    # Onsets advance at about every fifth token, so that keys form runs of equal onsets and bar positions. The last
    # batch has pitch-like coordinates whose tiles hold too many runs and use the per-pair path.
    positions = torch.arange(length).repeat(n_batches, 1)
    onsets = torch.cumsum((torch.rand((n_batches, length), generator=generator) < 0.2).long() * 6, dim=1)
    onsets[1] = torch.randint(0, 40, (length,), generator=generator)
    relations = torch.stack([positions, onsets, onsets % 48]).to(DEVICE)
    bounds = ((0, 100), (0, 120), (-47, 47))
    validity = None
    if invalid_positions is not None:
        validity = torch.ones(relations.shape, dtype=torch.bool)
        if invalid_positions == "start":
            validity[:, :, 0] = False
        else:
            validity[1] = torch.rand((n_batches, length), generator=generator) < 0.95
        validity = validity.to(DEVICE)
    q, k, v = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, dtype)
               for _ in range(3))
    embeddings = (torch.randn((n_batches, 101 + 121 + 95, head_dimension), generator=generator) * 0.3).to(DEVICE,
                                                                                                          dtype)
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, dtype)

    def attend(key_relations_repeat):
        inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, embeddings)]
        output = bs.ops.flash_attention_relative_embeddings(
            *itertools.chain.from_iterable(
                (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs[:3]),
            attention_layout, sparsity_block_size, relations, relations, inputs[3], bounds,
            key_relations_are_unique=(True, False, False), key_relations_repeat=key_relations_repeat,
            causal_lengths=lengths, query_relation_validity=validity, key_relation_validity=validity,
            sparsity_layout_o=layout, score_precalculation=score_precalculation)
        output = bs.ops.to_dense(output, layout, sparsity_block_size)
        output.backward(output_gradient)
        return [output.detach()] + [tensor.grad for tensor in inputs]

    with _forced_configs(combination, sparsity_block_size):
        merged = attend(None)
        pairwise = attend((False, False, False))
    # Float32 operands enter the products of the kernels as tf32, which amplifies the differences of the summation
    # order of a few logits up to the tf32 precision. Float16 gradients round these differences, and their terms
    # cancel in some table rows.
    tolerance = 2e-3 if dtype == torch.float32 else 1e-2
    for actual, expected in zip(merged, pairwise):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def _music_relations(n_batches: int, length: int, generator: torch.Generator, rest_probability: float):
    """Positions, onsets, and phases within bars of a synthetic piece. Rests of many onsets leave gaps between the
    referenced blocks of the onsets, whose number grows with ``rest_probability``.

    """
    advances = (torch.rand((n_batches, length), generator=generator) < 0.25).long()
    rests = (torch.rand((n_batches, length), generator=generator) < rest_probability).long() * 40
    onsets = torch.cumsum(advances * (1 + rests), dim=1)
    positions = torch.arange(length).repeat(n_batches, 1)
    return torch.stack([positions, onsets, onsets % 24])


def _compiled_variants(kernels) -> int:
    return sum(len(cache[0]) for kernel in kernels for cache in kernel.fn.device_caches.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_relative_batches_of_one_configuration_compile_no_new_variants(monkeypatch: pytest.MonkeyPatch):
    """Batches that differ in their coordinates, causal lengths, and attention layouts, but not in the static
    properties of the call, compile no further kernel variant after the first batch, and their autotuning keys differ
    at most in the buckets of the sequence length and of the schedule entries.

    """
    kernels = [getattr(FLASH_ATTENTION_MODULE, name) for name in KERNEL_PROFILES]
    generator = torch.Generator().manual_seed(21)
    n_batches, sparsity_block_size, head_dimension = 2, 16, 32
    length = 24 * sparsity_block_size
    n_blocks = length // sparsity_block_size
    bounds = ((0, 200), (0, 300), (-23, 23))
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    blocks = torch.arange(n_blocks)
    layouts = {
        "causal": torch.tril(torch.ones((n_blocks, n_blocks), dtype=torch.bool)),
        "band": (blocks[:, None] >= blocks[None, :]) & (blocks[:, None] - blocks[None, :] < 6),
    }

    def attend(rest_probability: float, layout_name: str, lengths: list[int]) -> None:
        relations = _music_relations(n_batches, length, generator, rest_probability).to(DEVICE)
        validity = (relations >= 0) & (torch.rand(relations.shape, generator=generator) < 0.98).to(DEVICE)
        attention_layout = layouts[layout_name].expand(n_batches, -1, -1).contiguous().to(DEVICE)
        inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
                  .requires_grad_(True) for _ in range(3)]
        embeddings = (torch.randn((n_batches, 201 + 301 + 47, head_dimension), generator=generator) * 0.3).to(
            DEVICE, torch.float16).requires_grad_(True)
        output = bs.ops.flash_attention_relative_embeddings(
            *itertools.chain.from_iterable(
                (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs),
            attention_layout, sparsity_block_size, relations, relations, embeddings, bounds,
            key_relations_are_unique=(True, False, False), key_relations_repeat=(False, True, True),
            causal_lengths=torch.tensor(lengths, device=DEVICE), query_relation_validity=validity,
            key_relation_validity=validity, sparsity_layout_o=layout)
        output.sum().backward()

    def static_keys():
        keys = set()
        for kernel in kernels:
            bucket_indices = [kernel.keys.index(name) for name in ("n_seq_blocks_bucket", "n_schedule_entries_bucket")]
            keys |= {(kernel.base_fn.__name__,) + tuple(value for index, value in enumerate(key)
                                                       if index not in bucket_indices) for key in kernel.cache}
        return keys

    attend(0.0, "causal", [length, length - 37])
    compiled, keys = _compiled_variants(kernels), static_keys()
    for rest_probability, layout_name, lengths in [(0.05, "causal", [length, length]), (0.3, "band", [200, 333]),
                                                   (0.0, "band", [length - 1, 17]), (0.6, "causal", [5, length])]:
        attend(rest_probability, layout_name, lengths)
        assert _compiled_variants(kernels) == compiled
        assert static_keys() == keys

    # Configurations that do not fit every variant of a call are not tuned: with all but the last marked as unfit,
    # tuning selects the last one.
    unfit = FLASH_ATTENTION_MODULE._UNFIT_CONFIGS
    assert {configuration[0] for configuration in unfit} >= set(KERNEL_PROFILES)
    with monkeypatch.context() as patch:
        for kernel in kernels:
            patch.setattr(kernel, "cache", {})
        for configuration in list(unfit):
            kernel = getattr(FLASH_ATTENTION_MODULE, configuration[0])
            patch.setitem(unfit, configuration, set(kernel.configs[:-1]))
        attend(0.0, "causal", [length, length - 37])
        for kernel in kernels:
            assert set(kernel.cache.values()) == {kernel.configs[-1]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("storage, rest_probability", [("gaps", 0.0), ("gaps", 0.2), ("exact", 0.2)])
@pytest.mark.parametrize("sparsity_block_size", [16, 32])
def test_flash_attention_relative_dispatch_paths_match_reference(sparsity_block_size: int, storage: str,
                                                                 rest_probability: float,
                                                                 monkeypatch: pytest.MonkeyPatch):
    """Batches that take every path of the kernels match dense attention: the variants that address all relations
    through intervals and those that look up the relations with gaps, the tiles that clip the leading relations,
    merged runs of repeated keys, the per-pair path of tiles with many runs, and the validity masks.

    """
    if storage == "exact":
        blocks_module = importlib.import_module("blksprs.ops.relative_score_blocks")
        monkeypatch.setattr(blocks_module, "_COMPACT_VALUE_BLOCKS", 0)
        monkeypatch.setattr(blocks_module, "_GAP_BUDGET", 2 ** 40)
    generator = torch.Generator().manual_seed(23)
    n_batches, head_dimension = 2, 32
    length = 768 // sparsity_block_size * sparsity_block_size
    n_blocks = length // sparsity_block_size
    # Rests of the second batch leave gaps between the referenced blocks of its onsets.
    relations = torch.cat([_music_relations(1, length, generator, 0.0),
                           _music_relations(1, length, generator, rest_probability)], dim=1)
    # Tables of more than eight value blocks, which are stored without gaps only if their gaps are few.
    bounds = ((0, 300), (0, 400), (-23, 23))
    validity = torch.ones(relations.shape, dtype=torch.bool)
    validity[:, :, 0] = False
    relations, validity = relations.to(DEVICE), validity.to(DEVICE)
    lengths = torch.tensor([length - 9, length - 3 * sparsity_block_size], device=DEVICE)
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                        device=DEVICE)
    q, k, v = (torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE) for _ in range(3))
    embeddings = (torch.randn((n_batches, 301 + 401 + 47, head_dimension), generator=generator) * 0.3).to(DEVICE)
    output_gradient = torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE)

    positions = torch.arange(length, device=DEVICE)
    valid = positions[None, :] < lengths[:, None]
    allowed = (_expand(attention_layout, sparsity_block_size) & (positions[:, None] >= positions[None, :])
               & valid[:, :, None] & valid[:, None, :])
    # A float64 reference, so that the comparison measures the rounding of the kernels alone.
    reference_inputs = [tensor.double().requires_grad_(True) for tensor in (q, k, v, embeddings)]
    relative = _relative_scores(reference_inputs[0], relations, relations, reference_inputs[3], bounds, validity)
    reference = _reference(*reference_inputs[:3], allowed, 1 / math.sqrt(head_dimension), relative=relative)
    reference.backward(output_gradient.double())

    layout_cache = {}
    inputs = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, embeddings)]
    output = bs.ops.flash_attention_relative_embeddings(
        *itertools.chain.from_iterable(
            (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs[:3]),
        attention_layout, sparsity_block_size, relations, relations, inputs[3], bounds,
        key_relations_are_unique=(True, False, False), key_relations_repeat=(False, True, True),
        causal_lengths=lengths, query_relation_validity=validity, key_relation_validity=validity,
        sparsity_layout_o=layout, layout_cache=layout_cache)
    output = bs.ops.to_dense(output, layout, sparsity_block_size)
    output.backward(output_gradient)

    score_blocks = layout_cache["relative_coordinates"]["coordinates"].score_blocks
    if storage == "exact" or rest_probability == 0:
        assert score_blocks.complete_intervals == (storage == "gaps")
    # The gradients of the clipped table rows sum the terms of many pairs, which cancel to small values, so that the
    # TF32 products of the kernels are compared at the scale of every tensor.
    for actual, expected in zip([output] + [tensor.grad for tensor in inputs],
                                [reference] + [tensor.grad for tensor in reference_inputs]):
        torch.testing.assert_close(actual.double(), expected, atol=2e-2 * expected.abs().max().item(), rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("score_precalculation", ["referenced_blocks", "complete"])
def test_flash_attention_warm_up_compiles_every_variant_of_a_configuration(score_precalculation: str):
    """After the warm-up of a static call configuration, calls with it compile no kernel variant, whatever their
    coordinates, lengths, and layouts.

    """
    kernels = [getattr(FLASH_ATTENTION_MODULE, name) for name in KERNEL_PROFILES]
    sparsity_block_size, head_dimension, n_batches = 16, 48, 3
    bounds = ((0, 250), (0, 180), (-23, 23))
    keywords = dict(relation_bounds=bounds, key_relations_are_unique=(True, False, False),
                    key_relations_repeat=(False, True, True), score_precalculation=score_precalculation)
    bs.ops.flash_attention_warm_up(sparsity_block_size, head_dimension, torch.float16, [(n_batches, 320), (6, 128)],
                                   relation_validity=True, **keywords)
    compiled = _compiled_variants(kernels)

    generator = torch.Generator().manual_seed(29)
    for length, rest_probability in [(320, 0.0), (480, 0.3), (96, 0.1)]:
        n_blocks = length // sparsity_block_size
        relations = _music_relations(n_batches, length, generator, rest_probability).to(DEVICE)
        validity = (torch.rand(relations.shape, generator=generator) < 0.97).to(DEVICE)
        attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
        layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool,
                            device=DEVICE)
        inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
                  .requires_grad_(True) for _ in range(3)]
        embeddings = torch.randn((n_batches, 251 + 181 + 47, head_dimension), generator=generator).to(
            DEVICE, torch.float16).requires_grad_(True)
        output = bs.ops.flash_attention_relative_embeddings(
            *itertools.chain.from_iterable(
                (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs),
            attention_layout, sparsity_block_size, relations, relations, embeddings,
            causal_lengths=torch.full((n_batches,), length - 7, device=DEVICE), query_relation_validity=validity,
            key_relation_validity=validity, sparsity_layout_o=layout, **keywords)
        output.sum().backward()
        assert _compiled_variants(kernels) == compiled

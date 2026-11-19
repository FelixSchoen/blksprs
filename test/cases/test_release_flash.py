import importlib
import itertools
from collections import defaultdict

import pytest
import torch
import triton

import blksprs as bs
from blksprs.utils.autotuning import prune_autotune_configs, prune_autotune_configs_spanning

DEVICE = torch.device("cuda")
FLASH_ATTENTION_MODULE = importlib.import_module("blksprs.ops.flash_attention")
# The autotuned Flash kernels with the tile that their grids hold along the second axis, and their pruning.
GRID_KERNELS = {
    "flash_attention_kernel": ("TRITON_BLOCK_SIZE_Q", prune_autotune_configs_spanning),
    "flash_attention_kernel_bwd_preprocess": ("TRITON_BLOCK_SIZE", prune_autotune_configs),
    "flash_attention_kernel_bwd_dkdv": ("TRITON_BLOCK_SIZE_K", prune_autotune_configs_spanning),
    "flash_attention_kernel_bwd_dq": ("TRITON_BLOCK_SIZE_Q", prune_autotune_configs_spanning),
}
# The kernels whose variants a static call configuration compiles, see ``flash_attention_warm_up``.
COMPILED_KERNELS = ["flash_attention_kernel", "flash_attention_kernel_bwd_dkdv", "flash_attention_kernel_bwd_dq"]


def _compiled_variants(kernels) -> int:
    return sum(len(cache[0]) for kernel in kernels for cache in kernel.fn.device_caches.values())


def _attend(n_batches: int, length: int, sparsity_block_size: int, head_dimension: int, dtype: torch.dtype,
            generator: torch.Generator, mask_dtype: torch.dtype | None = None, bias_dtype: torch.dtype | None = None,
            causal: bool = True) -> list[torch.Tensor]:
    """Runs one Flash Attention call with its backward and returns the output and the gradients."""
    n_blocks = length // sparsity_block_size
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    attention_layout[-1, n_blocks // 2:, :n_blocks // 4] = False
    inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, dtype)
              .requires_grad_(True) for _ in range(3)]
    keywords: dict = {"sparsity_layout_o": layout}
    if causal:
        keywords["causal_lengths"] = torch.tensor([length - 5 * index for index in range(n_batches)], device=DEVICE)
    if mask_dtype is not None:
        mask = torch.rand((n_batches, length, length), generator=generator) < 0.1
        keywords.update(attention_mask=bs.ops.to_sparse(mask.to(DEVICE, mask_dtype), attention_layout,
                                                        sparsity_block_size),
                        sparsity_layout_mask=attention_layout)
    bias = None
    if bias_dtype is not None:
        dense_bias = torch.randn((n_batches, length, length), generator=generator).to(DEVICE, bias_dtype)
        bias = bs.ops.to_sparse(dense_bias, attention_layout, sparsity_block_size).detach().requires_grad_(True)
        keywords.update(attention_bias=bias, sparsity_layout_bias=attention_layout)
    output = bs.ops.flash_attention(
        *itertools.chain.from_iterable(
            (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs),
        attention_layout, sparsity_block_size, **keywords)
    output.backward(torch.randn(output.shape, generator=generator).to(DEVICE, dtype))
    return [output.detach()] + [tensor.grad for tensor in inputs] + ([] if bias is None else [bias.grad])


def _attend_relative(generator: torch.Generator) -> list[torch.Tensor]:
    """Runs one learned-relative Flash Attention call with unique keys, whose gradients use no atomics."""
    n_batches, length, sparsity_block_size, head_dimension = 2, 256, 16, 32
    n_blocks = length // sparsity_block_size
    positions = torch.arange(length, device=DEVICE).repeat(1, n_batches, 1)
    layout = torch.ones((n_batches, n_blocks, head_dimension // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.tril(torch.ones((n_batches, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    inputs = [torch.randn((n_batches, length, head_dimension), generator=generator).to(DEVICE, torch.float16)
              .requires_grad_(True) for _ in range(3)]
    embeddings = (torch.randn((n_batches, 65, head_dimension), generator=generator) * 0.3).to(
        DEVICE, torch.float16).requires_grad_(True)
    output = bs.ops.flash_attention_relative_embeddings(
        *itertools.chain.from_iterable(
            (bs.ops.to_sparse(tensor, layout, sparsity_block_size), layout) for tensor in inputs),
        attention_layout, sparsity_block_size, positions, positions, embeddings, ((0, 64),),
        key_relations_are_unique=(True,), causal_lengths=torch.tensor([length, length - 21], device=DEVICE),
        sparsity_layout_o=layout)
    output.backward(torch.randn(output.shape, generator=generator).to(DEVICE, torch.float16))
    return [output.detach()] + [tensor.grad for tensor in inputs] + [embeddings.grad]


def test_flash_attention_grids_fold_only_beyond_the_grid_limit():
    """Sequences whose tiles fit the second grid axis for every configuration that autotuning may select keep their
    grid, and all longer ones fold their tiles into the batch axis.

    """
    for name, (tile_name, prune) in GRID_KERNELS.items():
        kernel = getattr(FLASH_ATTENTION_MODULE, name)
        for sparsity_block_size in (16, 32, 64):
            smallest_tile = min(config.kwargs[tile_name]
                                for config in prune(kernel.configs, {"sparsity_block_size": sparsity_block_size}))
            longest = smallest_tile * 65535
            assert not FLASH_ATTENTION_MODULE._folds_grid(kernel, longest, sparsity_block_size, tile_name, prune)
            assert FLASH_ATTENTION_MODULE._folds_grid(kernel, longest + 1, sparsity_block_size, tile_name, prune)
            for config in prune(kernel.configs, {"sparsity_block_size": sparsity_block_size}):
                grid = FLASH_ATTENTION_MODULE._tile_grid(3, longest, config.kwargs[tile_name], False, 2)
                assert grid[1] <= 65535
                folded_grid = FLASH_ATTENTION_MODULE._tile_grid(3, longest + 1, config.kwargs[tile_name], True, 2)
                assert folded_grid == [3 * triton.cdiv(longest + 1, config.kwargs[tile_name]), 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_folded_grids_match_unfolded_grids(monkeypatch: pytest.MonkeyPatch):
    """Folding the tiles into the batch axis of the grids, which calls beyond the grid limit need, leaves the results
    of every kernel unchanged.

    """
    calls = [lambda generator: _attend(3, 320, 16, 48, torch.float16, generator, torch.bool, torch.float16),
             lambda generator: _attend(2, 256, 32, 64, torch.bfloat16, generator, causal=False),
             _attend_relative]
    expected = [call(torch.Generator().manual_seed(index)) for index, call in enumerate(calls)]

    grids = []
    tile_grid = FLASH_ATTENTION_MODULE._tile_grid

    def recorded_tile_grid(*arguments):
        grid = tile_grid(*arguments)
        grids.append(grid)
        return grid

    with monkeypatch.context() as patch:
        patch.setattr(FLASH_ATTENTION_MODULE, "_GRID_AXIS_LIMIT", 1)
        patch.setattr(FLASH_ATTENTION_MODULE, "_tile_grid", recorded_tile_grid)
        actual = [call(torch.Generator().manual_seed(index)) for index, call in enumerate(calls)]
    assert grids and all(grid[1] == 1 for grid in grids)
    for expected_tensors, actual_tensors in zip(expected, actual, strict=True):
        for expected_tensor, actual_tensor in zip(expected_tensors, actual_tensors, strict=True):
            assert torch.equal(expected_tensor, actual_tensor)


def test_flash_attention_detects_concurrent_compilation(monkeypatch: pytest.MonkeyPatch):
    """Concurrent compilation is used from Triton 3.6 on, whose compilation mode and futures ignore errors on request,
    and Triton versions without either compile sequentially.

    """
    version = tuple(int(part) for part in triton.__version__.split(".")[:2])
    assert FLASH_ATTENTION_MODULE._compiles_concurrently() == (version >= (3, 6))

    class CompileModeWithoutErrors:
        def __init__(self, executor):
            self.executor = executor

    with monkeypatch.context() as patch:
        patch.setattr(triton, "AsyncCompileMode", CompileModeWithoutErrors, raising=False)
        assert not FLASH_ATTENTION_MODULE._compiles_concurrently()
    with monkeypatch.context() as patch:
        patch.delattr(triton, "AsyncCompileMode", raising=False)
        assert not FLASH_ATTENTION_MODULE._compiles_concurrently()


def test_flash_attention_sequential_compilation_returns_none_for_errors():
    """Configurations that do not compile sequentially are marked as unfit, as with concurrent compilation."""

    class FailingFunction:
        @staticmethod
        def warmup(*args, **kwargs):
            raise RuntimeError("does not compile")

    class FailingKernel:
        fn = FailingFunction()

    config = triton.Config({"TRITON_BLOCK_SIZE_Q": 64}, num_warps=4, num_stages=1)
    assert FLASH_ATTENTION_MODULE._compile_sequentially(FailingKernel(), (), {}, {}, config) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_compiles_sequentially_without_concurrent_compilation(monkeypatch: pytest.MonkeyPatch):
    """Without concurrent compilation, the first launch of a call configuration compiles its configurations one
    after the other, finds the same unfit configurations, and gives the same results.

    """
    kernels = [getattr(FLASH_ATTENTION_MODULE, name) for name in COMPILED_KERNELS]

    def attend() -> list[torch.Tensor]:
        return _attend(2, 192, 16, 32, torch.float16, torch.Generator().manual_seed(3), torch.bool)

    def fresh_state(patch: pytest.MonkeyPatch) -> dict:
        unfit: dict = {}
        patch.setattr(FLASH_ATTENTION_MODULE, "_UNFIT_CONFIGS", unfit)
        for kernel in kernels:
            patch.setattr(kernel.fn, "device_caches", defaultdict(kernel.fn.create_binder))
        return unfit

    with monkeypatch.context() as patch:
        concurrent_unfit = fresh_state(patch)
        expected = attend()

    compiled = []
    compile_sequentially = FLASH_ATTENTION_MODULE._compile_sequentially

    def recorded_compile_sequentially(*arguments):
        compiled_kernel = compile_sequentially(*arguments)
        compiled.append(compiled_kernel)
        return compiled_kernel

    def unavailable(*arguments, **keywords):
        raise AssertionError("concurrent compilation is unavailable")

    with monkeypatch.context() as patch:
        sequential_unfit = fresh_state(patch)
        patch.setattr(FLASH_ATTENTION_MODULE, "_CONCURRENT_COMPILATION", False)
        patch.setattr(FLASH_ATTENTION_MODULE, "_compile_sequentially", recorded_compile_sequentially)
        patch.setattr(triton, "AsyncCompileMode", unavailable, raising=False)
        actual = attend()

    assert compiled and all(compiled_kernel is not None for compiled_kernel in compiled)
    assert sequential_unfit == concurrent_unfit
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(expected_tensor, actual_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("warm_up_mask, mask_dtype", [(torch.bool, torch.bool), (True, torch.float16)])
def test_flash_attention_warm_up_compiles_mask_calls_of_their_dtype(warm_up_mask, mask_dtype: torch.dtype):
    """After the warm-up with a mask dtype, or with the dtype of the operands for ``True``, calls with masks of this
    dtype compile no kernel variant.

    """
    kernels = [getattr(FLASH_ATTENTION_MODULE, name) for name in GRID_KERNELS]
    sparsity_block_size, head_dimension = 16, 32
    bs.ops.flash_attention_warm_up(sparsity_block_size, head_dimension, torch.float16, [(2, 256)],
                                   attention_mask=warm_up_mask)
    compiled = _compiled_variants(kernels)
    generator = torch.Generator().manual_seed(5)
    for n_batches, length in [(2, 256), (3, 400)]:
        _attend(n_batches, length, sparsity_block_size, head_dimension, torch.float16, generator, mask_dtype)
        assert _compiled_variants(kernels) == compiled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("warm_up_bias, bias_dtype", [(True, torch.bfloat16), (torch.float32, torch.float32)])
def test_flash_attention_warm_up_compiles_bias_calls_of_their_dtype(warm_up_bias, bias_dtype: torch.dtype):
    """After the warm-up with an attention bias, calls with a bias of its dtype compile no kernel variant."""
    kernels = [getattr(FLASH_ATTENTION_MODULE, name) for name in GRID_KERNELS]
    sparsity_block_size, head_dimension = 32, 64
    bs.ops.flash_attention_warm_up(sparsity_block_size, head_dimension, torch.bfloat16, [(2, 256)],
                                   attention_mask=torch.bool, attention_bias=warm_up_bias)
    compiled = _compiled_variants(kernels)
    generator = torch.Generator().manual_seed(7)
    for n_batches, length in [(2, 256), (1, 416)]:
        _attend(n_batches, length, sparsity_block_size, head_dimension, torch.bfloat16, generator, torch.bool,
                bias_dtype)
        assert _compiled_variants(kernels) == compiled


def test_flash_attention_warm_up_rejects_bias_with_relative_terms():
    with pytest.raises(ValueError, match="attention bias"):
        bs.ops.flash_attention_warm_up(16, 32, torch.float16, [(1, 64)], relation_bounds=((0, 8),),
                                       attention_bias=True)

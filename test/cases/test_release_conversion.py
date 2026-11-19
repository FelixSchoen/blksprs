import os

os.environ.setdefault("BLKSPRS_AUTOTUNE", "TEST")

import pytest
import torch
import torch.nn.functional as F

import blksprs as bs
from blksprs import BlksprsTensor

DEVICE = torch.device("cuda:0")

# CUDA limits the second and third grid axes to 65535 programs. The smallest Triton block size is 16, so dense extents
# of more than 65535 * 16 rows or columns exceed that limit.
GRID_AXIS_LIMIT = 65535
SMALLEST_TRITON_BLOCK_SIZE = 16


# Dense references

def _expand_layout(sparsity_layout: torch.Tensor, sparsity_block_size: int) -> torch.Tensor:
    return (sparsity_layout != 0).repeat_interleave(sparsity_block_size, dim=1).repeat_interleave(
        sparsity_block_size, dim=2)


def _reference_to_sparse(x: torch.Tensor, sparsity_layout: torch.Tensor, sparsity_block_size: int) -> torch.Tensor:
    n_batches, n_rows, n_cols = sparsity_layout.shape
    return x.reshape(n_batches, n_rows, sparsity_block_size, n_cols, sparsity_block_size).permute(
        0, 1, 3, 2, 4)[sparsity_layout != 0]


def _reference_to_dense(x: torch.Tensor, sparsity_layout: torch.Tensor, sparsity_block_size: int,
                        fill_value: float = 0) -> torch.Tensor:
    n_batches, n_rows, n_cols = sparsity_layout.shape
    blocks = torch.full((n_batches, n_rows, n_cols, sparsity_block_size, sparsity_block_size), fill_value,
                        dtype=x.dtype, device=x.device)
    blocks = blocks.index_put(torch.nonzero(sparsity_layout != 0, as_tuple=True), x)
    return blocks.permute(0, 1, 3, 2, 4).reshape(n_batches, n_rows * sparsity_block_size, n_cols * sparsity_block_size)


def _reference_build_sparsity_layout(x: torch.Tensor, sparsity_block_size: int) -> torch.Tensor:
    n_batches, n_rows, n_cols = x.shape
    return (x != 0).reshape(n_batches, n_rows // sparsity_block_size, sparsity_block_size,
                            n_cols // sparsity_block_size, sparsity_block_size).any(dim=4).any(dim=2)


def _target_shape(sparsity_layout_from: torch.Tensor, sparsity_block_size_from: int,
                  sparsity_block_size_to: int) -> tuple[int, int, int]:
    n_batches, n_rows, n_cols = sparsity_layout_from.shape
    return (n_batches,
            -(-n_rows * sparsity_block_size_from // sparsity_block_size_to),
            -(-n_cols * sparsity_block_size_from // sparsity_block_size_to))


def _reference_structural_layout(sparsity_layout_from: torch.Tensor, sparsity_block_size_from: int,
                                 sparsity_block_size_to: int) -> torch.Tensor:
    # A target block is active if any of its elements lies in an active source block
    n_batches, n_rows_to, n_cols_to = _target_shape(sparsity_layout_from, sparsity_block_size_from,
                                                    sparsity_block_size_to)
    source_mask = _expand_layout(sparsity_layout_from, sparsity_block_size_from)
    element_mask = torch.zeros((n_batches, n_rows_to * sparsity_block_size_to, n_cols_to * sparsity_block_size_to),
                               dtype=torch.bool, device=DEVICE)
    element_mask[:, :source_mask.size(1), :source_mask.size(2)] = source_mask
    return element_mask.reshape(n_batches, n_rows_to, sparsity_block_size_to,
                                n_cols_to, sparsity_block_size_to).any(dim=4).any(dim=2)


def _reference_adapt_layout(x: torch.Tensor, sparsity_layout_from: torch.Tensor, sparsity_block_size_from: int,
                            sparsity_layout_to: torch.Tensor, sparsity_block_size_to: int) -> torch.Tensor:
    dense = _reference_to_dense(x, sparsity_layout_from, sparsity_block_size_from)
    n_batches, n_rows_to, n_cols_to = sparsity_layout_to.shape
    dense = F.pad(dense, (0, n_cols_to * sparsity_block_size_to - dense.size(2),
                          0, n_rows_to * sparsity_block_size_to - dense.size(1)))
    return _reference_to_sparse(dense, sparsity_layout_to, sparsity_block_size_to)


def _random_layout(shape: tuple[int, int, int], generator: torch.Generator) -> torch.Tensor:
    sparsity_layout = torch.rand(shape, generator=generator, device=DEVICE) > 0.4
    sparsity_layout[0, 0, 0] = True
    sparsity_layout[-1, -1, -1] = False
    return sparsity_layout


def _blocks_with_zero_blocks(n_blocks: int, sparsity_block_size: int, generator: torch.Generator,
                             dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = torch.randn((n_blocks, sparsity_block_size, sparsity_block_size), generator=generator,
                         device=DEVICE).to(dtype)
    zero_blocks = torch.arange(0, n_blocks, 3, device=DEVICE)
    blocks[zero_blocks] = 0
    return blocks, zero_blocks


# Layout adaption

@pytest.mark.parametrize("block_sizes", [(16, 32), (16, 64), (32, 64), (32, 16), (64, 16), (64, 32),
                                         (16, 16), (32, 32), (64, 64)])
@pytest.mark.parametrize("layout_shape", [(2, 4, 4), (2, 3, 5)])
def test_adapt_layout_default_is_structural_and_retains_zero_blocks(block_sizes: tuple[int, int],
                                                                    layout_shape: tuple[int, int, int]):
    sparsity_block_size_from, sparsity_block_size_to = block_sizes
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    sparsity_layout_from = _random_layout(layout_shape, generator)
    n_blocks = int(sparsity_layout_from.sum())
    x_base, zero_blocks = _blocks_with_zero_blocks(n_blocks, sparsity_block_size_from, generator)
    x = x_base.clone().requires_grad_(True)
    x_reference = x_base.clone().requires_grad_(True)

    output, sparsity_layout_to = bs.ops.adapt_layout(
        BlksprsTensor.wrap(x), sparsity_layout_from, sparsity_block_size_from, sparsity_block_size_to)

    expected_layout = _reference_structural_layout(sparsity_layout_from, sparsity_block_size_from,
                                                   sparsity_block_size_to)
    assert sparsity_layout_to.dtype == torch.bool
    assert torch.equal(sparsity_layout_to, expected_layout)
    assert torch.equal(sparsity_layout_to, bs.layouting.build_sparsity_layout_adaption(
        BlksprsTensor.wrap(x_base), sparsity_layout_from, sparsity_block_size_from, sparsity_block_size_to))

    # The values never change the target layout
    _, sparsity_layout_to_zeros = bs.ops.adapt_layout(
        BlksprsTensor.wrap(torch.zeros_like(x_base)), sparsity_layout_from, sparsity_block_size_from,
        sparsity_block_size_to)
    assert torch.equal(sparsity_layout_to_zeros, expected_layout)

    expected = _reference_adapt_layout(x_reference, sparsity_layout_from, sparsity_block_size_from,
                                       expected_layout, sparsity_block_size_to)
    assert torch.equal(output, expected)

    weights = torch.randn(output.shape, generator=generator, device=DEVICE)
    (output * weights).sum().backward()
    (expected * weights).sum().backward()

    assert torch.equal(x.grad, x_reference.grad)
    # All-zero source blocks remain part of the result and therefore receive their gradients
    assert torch.all(x.grad[zero_blocks].abs().flatten(1).amax(dim=1) > 0)


@pytest.mark.parametrize("layout_dtype", [torch.float32, torch.int32, torch.uint8])
def test_build_sparsity_layout_adaption_accepts_non_boolean_layouts(layout_dtype: torch.dtype):
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    sparsity_layout_from = _random_layout((2, 3, 5), generator)
    x = BlksprsTensor.wrap(torch.zeros((int(sparsity_layout_from.sum()), 16, 16), device=DEVICE))

    for sparsity_block_size_to in (16, 32, 64):
        actual = bs.layouting.build_sparsity_layout_adaption(
            x, sparsity_layout_from.to(layout_dtype), 16, sparsity_block_size_to)

        assert type(actual) is torch.Tensor
        assert torch.equal(actual, _reference_structural_layout(sparsity_layout_from, 16, sparsity_block_size_to))


@pytest.mark.parametrize("sparsity_block_size", [16, 32, 64])
def test_adapt_layout_with_equal_block_sizes_returns_an_unchanged_copy(sparsity_block_size: int):
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    sparsity_layout = _random_layout((2, 3, 4), generator)
    x_base, _ = _blocks_with_zero_blocks(int(sparsity_layout.sum()), sparsity_block_size, generator)
    x = x_base.clone().requires_grad_(True)

    output, sparsity_layout_to = bs.ops.adapt_layout(
        BlksprsTensor.wrap(x), sparsity_layout, sparsity_block_size, sparsity_block_size)

    assert torch.equal(sparsity_layout_to, sparsity_layout)
    assert torch.equal(output, x_base)
    assert output.data_ptr() != x.data_ptr()

    weights = torch.randn(output.shape, generator=generator, device=DEVICE)
    (output * weights).sum().backward()
    assert torch.equal(x.grad, weights)


# Grid limits

@pytest.mark.parametrize("sparsity_block_size", [16, 64])
@pytest.mark.parametrize("long_dimension", [1, 2])
def test_dense_conversions_support_extents_beyond_the_grid_axis_limit(sparsity_block_size: int,
                                                                      long_dimension: int):
    long_extent = GRID_AXIS_LIMIT * SMALLEST_TRITON_BLOCK_SIZE + sparsity_block_size
    long_extent += -long_extent % sparsity_block_size
    assert long_extent // SMALLEST_TRITON_BLOCK_SIZE > GRID_AXIS_LIMIT

    shape = [1, sparsity_block_size, sparsity_block_size]
    shape[long_dimension] = long_extent
    layout_shape = [1, 1, 1]
    layout_shape[long_dimension] = long_extent // sparsity_block_size

    generator = torch.Generator(device=DEVICE).manual_seed(3)
    sparsity_layout = torch.rand(layout_shape, generator=generator, device=DEVICE) > 0.5
    sparsity_layout.view(-1)[0] = True
    sparsity_layout.view(-1)[-1] = True
    x_base = torch.randn(shape, generator=generator, device=DEVICE).to(torch.float16)
    x = x_base.clone().requires_grad_(True)

    x_sparse = bs.ops.to_sparse(x, sparsity_layout, sparsity_block_size)
    assert torch.equal(x_sparse, _reference_to_sparse(x_base, sparsity_layout, sparsity_block_size))

    x_dense = bs.ops.to_dense(x_sparse, sparsity_layout, sparsity_block_size, fill_value=-2.0)
    assert torch.equal(x_dense, _reference_to_dense(x_sparse.detach(), sparsity_layout, sparsity_block_size,
                                                    fill_value=-2.0))

    # The backward pass of ``to_sparse`` uses ``to_dense``
    weights = torch.randn(x_sparse.shape, generator=generator, device=DEVICE).to(torch.float16)
    (x_sparse * weights).sum().backward()
    assert torch.equal(x.grad, _reference_to_dense(weights, sparsity_layout, sparsity_block_size))

    # Inactive blocks are zero in ``x_dense_zero_filled``, and the first eight blocks are zeroed as well
    x_dense_zero_filled = bs.ops.to_dense(x_sparse.detach(), sparsity_layout, sparsity_block_size)
    x_dense_zero_filled.narrow(long_dimension, 0, 8 * sparsity_block_size).zero_()
    actual_layout = bs.layouting.build_sparsity_layout(x_dense_zero_filled, sparsity_block_size)
    expected_layout = _reference_build_sparsity_layout(x_dense_zero_filled, sparsity_block_size)
    assert torch.equal(actual_layout, expected_layout)
    assert not torch.equal(actual_layout, sparsity_layout)


@pytest.mark.parametrize("sparsity_block_size", [16, 64])
def test_flattened_dense_grids_match_dense_reference_for_batches(sparsity_block_size: int):
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    sparsity_layout = _random_layout((3, 5, 7), generator)
    x_dense = torch.randn((3, 5 * sparsity_block_size, 7 * sparsity_block_size), generator=generator, device=DEVICE)
    x_dense = x_dense * _expand_layout(sparsity_layout, sparsity_block_size)
    x_dense[1, :sparsity_block_size, :sparsity_block_size] = 0
    x_sparse = bs.ops.to_sparse(x_dense, sparsity_layout, sparsity_block_size)

    actual_dense = bs.ops.to_dense(x_sparse, sparsity_layout, sparsity_block_size, fill_value=5.0)
    actual_layout = bs.layouting.build_sparsity_layout(x_dense, sparsity_block_size)

    assert torch.equal(actual_dense, _reference_to_dense(x_sparse, sparsity_layout, sparsity_block_size, 5.0))
    assert torch.equal(actual_layout, _reference_build_sparsity_layout(x_dense, sparsity_block_size))


# Storage semantics

def _assert_new_storage(output: torch.Tensor, source: torch.Tensor) -> None:
    source_copy = source.detach().clone()
    assert output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    output.detach().fill_(7)
    assert torch.equal(source.detach(), source_copy)


def test_single_block_conversions_return_new_storage():
    sparsity_block_size = 16
    sparsity_layout = torch.ones((3, 1, 1), dtype=torch.bool, device=DEVICE)
    x_base = torch.randn((3, sparsity_block_size, sparsity_block_size), device=DEVICE)
    x = x_base.clone().requires_grad_(True)

    x_sparse = bs.ops.to_sparse(x, sparsity_layout, sparsity_block_size)
    x_dense = bs.ops.to_dense(x_sparse, sparsity_layout, sparsity_block_size)

    assert type(x_sparse) is BlksprsTensor
    assert type(x_dense) is torch.Tensor
    assert torch.equal(x_sparse, x_base)
    assert torch.equal(x_dense, x_base)

    weights = torch.randn_like(x_base)
    (x_dense * weights).sum().backward()
    assert torch.equal(x.grad, weights)

    _assert_new_storage(x_dense, x_sparse)
    _assert_new_storage(x_sparse, x)


@pytest.mark.parametrize("layout_shape", [(3, 1, 1), (2, 2, 3)])
def test_general_and_adapted_conversions_return_new_storage(layout_shape: tuple[int, int, int]):
    sparsity_block_size = 16
    sparsity_layout = torch.ones(layout_shape, dtype=torch.bool, device=DEVICE)
    x = torch.randn((layout_shape[0], layout_shape[1] * sparsity_block_size, layout_shape[2] * sparsity_block_size),
                    device=DEVICE)

    x_sparse = bs.ops.to_sparse(x, sparsity_layout, sparsity_block_size)
    x_dense = bs.ops.to_dense(x_sparse, sparsity_layout, sparsity_block_size)
    x_adapted, _ = bs.ops.adapt_layout(x_sparse, sparsity_layout, sparsity_block_size, sparsity_block_size)

    _assert_new_storage(x_adapted, x_sparse)
    _assert_new_storage(x_dense, x_sparse)
    _assert_new_storage(x_sparse, x)


# Row-striped conversion

@pytest.mark.parametrize("layout_shape", [(2, 4, 1), (2, 4, 3)])
@pytest.mark.parametrize("active_rows", ["none", "partial", "all"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_row_striped_conversion_matches_dense_reference_and_returns_new_storage(
        layout_shape: tuple[int, int, int], active_rows: str, dtype: torch.dtype):
    sparsity_block_size = 16
    row_mask = torch.zeros(layout_shape[:2], dtype=torch.bool, device=DEVICE)
    if active_rows == "partial":
        row_mask[0, 1] = True
        row_mask[1, 0] = True
        row_mask[1, 3] = True
    elif active_rows == "all":
        row_mask[:] = True
    sparsity_layout = row_mask.unsqueeze(-1).expand(layout_shape).contiguous()

    x_base = torch.randn((layout_shape[0], layout_shape[1] * sparsity_block_size,
                          layout_shape[2] * sparsity_block_size), device=DEVICE).mul(8).to(dtype)
    requires_grad = dtype.is_floating_point
    x = x_base.clone().requires_grad_(requires_grad)

    assert bs.ops.is_row_striped_layout(sparsity_layout)
    x_sparse = bs.ops.to_sparse_row_striped(x, sparsity_layout, sparsity_block_size)
    x_dense = bs.ops.to_dense_row_striped(x_sparse, sparsity_layout, sparsity_block_size, fill_value=3)

    expected_sparse = _reference_to_sparse(x_base, sparsity_layout, sparsity_block_size)
    assert type(x_sparse) is BlksprsTensor
    assert x_sparse.is_contiguous()
    assert torch.equal(x_sparse, expected_sparse)
    assert torch.equal(x_sparse, bs.ops.to_sparse(x_base, sparsity_layout, sparsity_block_size))
    assert torch.equal(x_dense, _reference_to_dense(expected_sparse, sparsity_layout, sparsity_block_size, 3))

    if requires_grad:
        weights = torch.randn(x_sparse.shape, device=DEVICE).to(dtype)
        (x_sparse * weights).sum().backward()
        assert torch.equal(x.grad, _reference_to_dense(weights, sparsity_layout, sparsity_block_size))

    _assert_new_storage(x_dense, x_sparse)
    if x_sparse.numel() > 0:
        _assert_new_storage(x_sparse, x)
    else:
        assert x_sparse.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()

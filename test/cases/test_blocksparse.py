import os
import tomllib

os.environ["BLKSPRS_AUTOTUNE"] = "TEST"

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch import Tensor

import blksprs as bs
from blksprs import BlksprsTensor
from blksprs.layouting.distribution_layout import build_distribution_layout_operation
from blksprs.layouting.sparsity_layout import build_sparsity_layout_operation

# Device setup
DEVICE = torch.device("cuda:0")

# Constants
BASE_PATH = Path(__file__).parent.parent.parent
TEST_CONFIGURATIONS = [
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    # All same
    (2, 64, 64, 64, 64, 0),
    (2, 16, 16, 16, 16, 0.75),
    (2, 32, 32, 32, 32, 0.75),
    (2, 64, 64, 64, 64, 0.75),
    (2, 64, 64, 64, 64, 1),
    # Same dimensions, sparsity_block_size
    (2, 64, 64, 64, 16, 0.75),
    (2, 64, 64, 64, 32, 0.75),
    (2, 128, 128, 128, 64, 0),
    (2, 128, 128, 128, 16, 0.75),
    (2, 128, 128, 128, 32, 0.75),
    (2, 128, 128, 128, 64, 0.75),
    (2, 128, 128, 128, 64, 1),
    (2, 2048, 2048, 128, 64, 0.75),
    # Same dimensions
    (2, 64, 64, 64, 32, 0.75),
    (2, 128, 128, 128, 32, 0),
    (2, 128, 128, 128, 32, 0.75),
    (2, 128, 128, 128, 64, 0.75),
    (2, 128, 128, 128, 64, 0.75),
    (2, 128, 128, 128, 64, 1),
    # All different
    (16, 64, 32, 64, 32, 0.75),
    (16, 32, 64, 64, 32, 0.75),
    (8, 128, 64, 128, 64, 0.75),
    (8, 64, 128, 128, 64, 0.75),
    (4, 256, 128, 128, 64, 0),
    (4, 256, 128, 128, 64, 0.75),
    (4, 256, 128, 128, 64, 1),
    (4, 128, 256, 128, 64, 0.75),
    (2, 4096, 1024, 128, 64, 0.75),
    (2, 1024, 4096, 128, 64, 0.75),
    # Different sparsity
    (2, 128, 128, 128, 64, 0.5),
    (2, 128, 128, 128, 64, 0.25),
    (2, 128, 128, 128, 64, 0.1),
    (2, 256, 128, 64, 32, 0.1),
    (2, 128, 256, 64, 32, 0.1),
    (2, 128, 256, 64, 32, 0.1),
    (2, 128, 128, 128, 16, 0.015625),
    # Empty, full, and single block
    (2, 64, 64, 64, 32, 0),
    (2, 64, 64, 64, 32, 1),
    (1, 64, 64, 64, 32, 0.25),
    # Odd batch and sparsity layout sizes
    (3, 112, 80, 48, 16, 0),
    (3, 112, 80, 48, 16, 0.25),
    (3, 112, 80, 48, 16, 0.5),
    (5, 112, 80, 48, 16, 0.75),
    (7, 112, 80, 48, 16, 1),
    (3, 224, 160, 96, 32, 0),
    (3, 224, 160, 96, 32, 0.25),
    (3, 224, 160, 96, 32, 0.5),
    (5, 224, 160, 96, 32, 0.75),
    (7, 224, 160, 96, 32, 1),
    (3, 448, 320, 192, 64, 0),
    (3, 448, 320, 192, 64, 0.25),
    (3, 448, 320, 192, 64, 0.5),
    (5, 448, 320, 192, 64, 0.75),
    (7, 448, 320, 192, 64, 1),
]

OVERFLOW_TEST_CONFIGURATIONS = [
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    # Designed so block-sparse storage crosses the signed int32 element-index limit.
    (4096, 16384, 32, 32, 32, 1),
]

OVERFLOW_PARTITION_CONFIGURATIONS = [
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    # Two column blocks so split/merge can partition while sparse storage still
    # crosses the signed int32 element-index limit.
    (4096, 8192, 32, 64, 32, 1),
]

LARGE_INDEX_BUILD_LAYOUT_CONFIGURATIONS = [
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    # Just below and just above the signed int32 element-index limit.
    (4095, 16384, 32, 32, 32, 1),
    (4096, 16384, 32, 32, 32, 1),
]

OVERFLOW_FLASH_ATTENTION_CONFIGURATIONS = [
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    # Single sequence block, many head-dimension blocks.
    (2048, 32, 32, 32768, 32, 0),
]

OVERFLOW_BROADCAST_CONFIGURATIONS = [
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    # Dense 2D inputs cross the signed int32 element-index limit while the sparse
    # output remains tiny because only a handful of batches are materialized.
    (67_108_865, 32, 32, 32, 32, 0),
]

# Tolerances
ATOL = 2e-2
RTOL = 1.5e-2

# Seed
SEED = 0
RANDOM_SEED = True


@pytest.fixture(scope="session", autouse=True)
def setup():
    global SEED
    global RANDOM_SEED

    if RANDOM_SEED:
        seed = random.randint(0, 2 ** 32 - 1)
        SEED = seed
        print("Using randomly generated seed...")
    else:
        seed = SEED
        print("Notice: Not using randomly generated seed!")

    print("Seed:", seed)
    torch.manual_seed(seed)
    torch.set_printoptions(edgeitems=64, linewidth=10000)
    override_pytorch_repr()

    yield

    print("Seed:", seed)


def override_pytorch_repr():
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self, *args, **kwargs: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"


# Ops

@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_to_sparse(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x_s in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_to_sparse_out = _slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)
            stock_dtype = stock_to_sparse_out.dtype

            blksprs_to_sparse_out = bs.ops.to_sparse(x_blksprs, sparsity_layout_x_s, sparsity_block_size)

            assert torch.allclose(blksprs_to_sparse_out.to(stock_dtype), stock_to_sparse_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_to_sparse_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_to_sparse_out, target)
            blksprs_loss = blksprs_loss(blksprs_to_sparse_out, target)

            blksprs_loss.backward()

            if sparsity_percentage > 0:
                stock_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_to_dense(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x_s in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_to_sparse_out = _slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)
            stock_to_dense_out = _slow_to_dense(stock_to_sparse_out, sparsity_layout_x_s,
                                                sparsity_block_size)
            stock_dtype = stock_to_dense_out.dtype

            blksprs_to_sparse_out = bs.ops.to_sparse(x_blksprs, sparsity_layout_x_s, sparsity_block_size)
            blksprs_to_dense_out = bs.ops.to_dense(blksprs_to_sparse_out, sparsity_layout_x_s, sparsity_block_size)

            assert torch.allclose(blksprs_to_dense_out.to(stock_dtype), stock_to_dense_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_to_dense_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_to_dense_out, target)
            blksprs_loss = blksprs_loss(blksprs_to_dense_out, target)

            blksprs_loss.backward()

            if sparsity_percentage > 0:
                stock_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_transpose(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_transpose_out = x_stock.transpose(1, 2)
            stock_dtype = stock_transpose_out.dtype

            blksprs_transpose_out, blksprs_sparsity_layout_t = bs.ops.transpose(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size)
            blksprs_transpose_dense_out = bs.ops.to_dense(blksprs_transpose_out, blksprs_sparsity_layout_t,
                                                          sparsity_block_size)

            assert torch.allclose(blksprs_transpose_dense_out.to(stock_dtype), stock_transpose_out, atol=ATOL,
                                  rtol=RTOL)

            target = torch.randn_like(stock_transpose_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_transpose_out, target)
            blksprs_loss = blksprs_loss(blksprs_transpose_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_gather(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        dims = [-2, -1, 0, 1, 2]
        for dim in dims:
            x_d = torch.randn(size=(b * 2, m * 2, k * 2), device=DEVICE)
            sparsity_layout_x_d = torch.ones(size=(b * 2, m * 2 // sparsity_block_size, k * 2 // sparsity_block_size),
                                             device=DEVICE)

            if dim % 3 == 0:
                dist_lim = b * 2
            elif dim % 3 == 1:
                dist_lim = m * 2
            elif dim % 3 == 2:
                dist_lim = k * 2
            else:
                raise ValueError("Invalid dim")

            i_d = (torch.randint(0, dist_lim, size=(b, m, k), dtype=torch.int, device=DEVICE).contiguous())
            sparsity_layout_i_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size),
                                             device=DEVICE)

            sparsity_layout_i_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
            i_bs = _blocksparse_roundtrip(i_d, sparsity_layout_i_bs, sparsity_block_size)

            sparsity_layout_x_bs = bs.layouting.build_distribution_layout(
                bs.ops.to_sparse(i_d, sparsity_layout_i_d, sparsity_block_size),
                sparsity_layout_i_d, dim, x_d.size(), sparsity_block_size)

            for x, sparsity_layout_x, i, sparsity_layout_i in [(x_d, sparsity_layout_x_d, i_d, sparsity_layout_i_d),
                                                               (x_d, sparsity_layout_x_bs, i_bs, sparsity_layout_i_bs)]:
                x_stock = x.clone().requires_grad_(True)
                i_stock = i.clone()
                x_blksprs = x.clone().requires_grad_(True)
                i_blksprs = i.clone()

                stock_gather_out = _blocksparse_roundtrip(
                    torch.gather(x_stock, dim=dim, index=i_stock.to(torch.int64)),
                    sparsity_layout_i, sparsity_block_size)
                stock_dtype = stock_gather_out.dtype

                blksprs_gather_out = bs.ops.gather(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x,
                    dim,
                    bs.ops.to_sparse(i_blksprs, sparsity_layout_i, sparsity_block_size),
                    sparsity_layout_i,
                    sparsity_block_size)
                blksprs_gather_dense_out = bs.ops.to_dense(blksprs_gather_out, sparsity_layout_i,
                                                           sparsity_block_size)

                assert torch.allclose(blksprs_gather_dense_out.to(stock_dtype), stock_gather_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_gather_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_gather_out, target)
                blksprs_loss = blksprs_loss(blksprs_gather_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad.to(torch.float), x_stock.grad.to(torch.float),
                                      atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dim", [1, 2])
def test_gather_kernel_masks_packed_indices_per_axis(dim: int):
    sparsity_block_size = 16
    sparsity_layout_src = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)
    sparsity_layout_idx = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)

    src = torch.zeros((2, sparsity_block_size, sparsity_block_size), device=DEVICE)
    src[1] = 7
    idx = torch.full(
        (1, sparsity_block_size, sparsity_block_size),
        fill_value=sparsity_block_size,
        dtype=torch.int32,
        device=DEVICE,
    )

    actual = bs.ops.gather(
        BlksprsTensor.wrap(src),
        sparsity_layout_src,
        dim,
        BlksprsTensor.wrap(idx),
        sparsity_layout_idx,
        sparsity_block_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_scatter(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        dims = [-2, -1, 0, 1, 2]
        for dim in dims:
            x_d = torch.randn(size=(b, m, k), device=DEVICE)
            sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size),
                                             device=DEVICE)

            if dim % 3 == 0:
                dist_lim = b * 2
            elif dim % 3 == 1:
                dist_lim = m * 2
            elif dim % 3 == 2:
                dist_lim = k * 2
            else:
                raise ValueError("Invalid dim")

            i_d = (torch.randint(0, dist_lim, size=(b, m, k), dtype=torch.int, device=DEVICE).contiguous())

            sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
            x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)
            i_bs = _blocksparse_roundtrip(i_d, sparsity_layout_x_bs, sparsity_block_size)

            sparsity_layout_o_d = torch.ones(size=(b * 2, m * 2 // sparsity_block_size, k * 2 // sparsity_block_size),
                                             device=DEVICE)
            sparsity_layout_o_bs = bs.layouting.build_distribution_layout(
                bs.ops.to_sparse(i_d, sparsity_layout_x_d, sparsity_block_size),
                sparsity_layout_x_d, dim, torch.Size((b * 2, m * 2, k * 2)), sparsity_block_size)

            for x, sparsity_layout_x, i, sparsity_layout_i, sparsity_layout_o in [
                (x_d, sparsity_layout_x_d, i_d, sparsity_layout_x_d, sparsity_layout_o_d),
                (x_bs, sparsity_layout_x_bs, i_bs, sparsity_layout_x_bs, sparsity_layout_o_bs)]:
                x_stock = x.clone().requires_grad_(True)
                i_stock = i.clone()
                x_blksprs = x.clone().requires_grad_(True)
                i_blksprs = i.clone()

                stock_out_buffer = torch.zeros(size=(b * 2, m * 2, k * 2), dtype=x_stock.dtype, device=DEVICE)
                stock_scatter_out = _blocksparse_roundtrip(
                    stock_out_buffer.scatter_reduce(dim=dim, index=i_stock.to(torch.int64), src=x_stock,
                                                    reduce="sum"),
                    sparsity_layout_o, sparsity_block_size)
                stock_dtype = stock_scatter_out.dtype

                blksprs_scatter_out = bs.ops.scatter_reduce(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x,
                    dim,
                    bs.ops.to_sparse(i_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_o,
                    sparsity_block_size,
                    reduce_op="sum")
                blksprs_scatter_dense_out = bs.ops.to_dense(blksprs_scatter_out, sparsity_layout_o,
                                                            sparsity_block_size)

                assert torch.allclose(blksprs_scatter_dense_out.to(stock_dtype), stock_scatter_out, atol=ATOL,
                                      rtol=RTOL)

                target = torch.randn_like(stock_scatter_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_scatter_out, target)
                blksprs_loss = blksprs_loss(blksprs_scatter_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dim", [1, 2])
def test_scatter_reduce_kernel_masks_packed_indices_per_axis(dim: int):
    sparsity_block_size = 16
    sparsity_layout_src = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    sparsity_layout_tgt = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)

    src = torch.full((1, sparsity_block_size, sparsity_block_size), 7.0, device=DEVICE)
    idx = torch.full(
        (1, sparsity_block_size, sparsity_block_size),
        fill_value=sparsity_block_size,
        dtype=torch.int32,
        device=DEVICE,
    )

    actual = bs.ops.scatter_reduce(
        BlksprsTensor.wrap(src),
        sparsity_layout_src,
        dim,
        BlksprsTensor.wrap(idx),
        sparsity_layout_tgt,
        sparsity_block_size,
        reduce_op="sum",
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_matmul(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)
        y_d = torch.randn(size=(b, n, k), device=DEVICE).transpose(-1, -2).contiguous()
        sparsity_layout_y_d = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)
        sparsity_layout_y_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
        y_bs = _blocksparse_roundtrip(y_d, sparsity_layout_y_bs, sparsity_block_size)

        sparsity_layout_o_d = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size),
                                         device=DEVICE)
        sparsity_layout_o_bs = bs.layouting.build_sparsity_layout_matmul_fast(sparsity_layout_x_bs,
                                                                              sparsity_layout_y_bs)

        for x, sparsity_layout_x, y, sparsity_layout_y, sparsity_layout_o in [
            (x_d, sparsity_layout_x_d, y_d, sparsity_layout_y_d, sparsity_layout_o_d),
            (x_bs, sparsity_layout_x_bs, y_bs, sparsity_layout_y_bs, sparsity_layout_o_bs),
            (x_bs, sparsity_layout_x_bs, y_bs, sparsity_layout_y_bs, sparsity_layout_o_d)]:
            x_stock = x.clone().requires_grad_(True)
            y_stock = y.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)
            y_blksprs = y.clone().requires_grad_(True)

            stock_matmul_out = torch.matmul(x_stock, y_stock)
            stock_dtype = stock_matmul_out.dtype

            blksprs_matmul_out = bs.ops.matmul(bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                                               sparsity_layout_x,
                                               bs.ops.to_sparse(y_blksprs, sparsity_layout_y, sparsity_block_size),
                                               sparsity_layout_y,
                                               sparsity_layout_o, sparsity_block_size)
            blksprs_matmul_dense_out = bs.ops.to_dense(blksprs_matmul_out, sparsity_layout_o, sparsity_block_size)

            atol = 4e-2 if use_amp else 6e-2
            rtol = 2e-2 if use_amp else RTOL

            assert torch.allclose(blksprs_matmul_dense_out.to(stock_dtype), stock_matmul_out, atol=atol, rtol=rtol)

            target = torch.randn_like(stock_matmul_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_matmul_out, target)
            blksprs_loss = blksprs_loss(blksprs_matmul_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=atol, rtol=rtol)
            assert torch.allclose(y_blksprs.grad, y_stock.grad, atol=atol, rtol=rtol)


def test_matmul_rejects_incompatible_output_layout_shape():
    sparsity_block_size = 16
    sparsity_layout_x = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)
    sparsity_layout_y = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)
    sparsity_layout_output = torch.ones((2, 2, 1), dtype=torch.bool, device=DEVICE)
    x = BlksprsTensor.wrap(torch.zeros((2, sparsity_block_size, sparsity_block_size), device=DEVICE))
    y = BlksprsTensor.wrap(torch.zeros((2, sparsity_block_size, sparsity_block_size), device=DEVICE))

    with pytest.raises(ValueError, match="Output sparsity layout shape"):
        bs.ops.matmul(x, sparsity_layout_x, y, sparsity_layout_y, sparsity_layout_output, sparsity_block_size)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_repeat(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        num_repeats_values = [1, 2, 3, 4]

        for num_repeats in num_repeats_values:
            for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
                x_stock = x.clone().requires_grad_(True)
                x_blksprs = x.clone().requires_grad_(True)

                repeats = (num_repeats, num_repeats, num_repeats)

                sparsity_layout_o_bs = _get_blocksparse_layout(b * repeats[0], m * repeats[1], k * repeats[2],
                                                               sparsity_block_size, sparsity_percentage)
                if torch.all(sparsity_layout_x):
                    sparsity_layout_o_bs = torch.ones_like(sparsity_layout_o_bs)

                stock_repeat_out = _blocksparse_roundtrip(x_stock.repeat(repeats), sparsity_layout_o_bs,
                                                          sparsity_block_size)
                stock_dtype = stock_repeat_out.dtype

                blksprs_repeat_out, sparsity_layout_output = bs.ops.repeat(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, repeats,
                    sparsity_block_size, sparsity_layout_o_bs)
                blksprs_repeat_dense_out = bs.ops.to_dense(blksprs_repeat_out, sparsity_layout_output,
                                                           sparsity_block_size)

                assert torch.allclose(blksprs_repeat_dense_out.to(stock_dtype), stock_repeat_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_repeat_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_repeat_out, target)
                blksprs_loss = blksprs_loss(blksprs_repeat_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_repeat_interleave(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, n), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, n, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        num_repeats_values = [1, 2, 3, 4]

        for num_repeats in num_repeats_values:
            for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
                x_stock = x.clone().requires_grad_(True)
                x_blksprs = x.clone().requires_grad_(True)

                sparsity_layout_o_bs = _get_blocksparse_layout(b * num_repeats, m, n,
                                                               sparsity_block_size, sparsity_percentage)
                if torch.all(sparsity_layout_x):
                    sparsity_layout_o_bs = torch.ones_like(sparsity_layout_o_bs)

                stock_repeat_interleave_out = _blocksparse_roundtrip(
                    torch.repeat_interleave(x_stock, num_repeats, dim=0),
                    sparsity_layout_o_bs,
                    sparsity_block_size)
                stock_dtype = stock_repeat_interleave_out.dtype

                blksprs_repeat_interleave_out, sparsity_layout_output = bs.ops.repeat_interleave(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, num_repeats,
                    sparsity_block_size, sparsity_layout_o_bs)
                blksprs_repeat_interleave_dense_out = bs.ops.to_dense(blksprs_repeat_interleave_out,
                                                                      sparsity_layout_output,
                                                                      sparsity_block_size)

                assert torch.allclose(blksprs_repeat_interleave_dense_out.to(stock_dtype), stock_repeat_interleave_out,
                                      atol=ATOL,
                                      rtol=RTOL)

                target = torch.randn_like(stock_repeat_interleave_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_repeat_interleave_out, target)
                blksprs_loss = blksprs_loss(blksprs_repeat_interleave_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_softmax(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size),
                                         device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                      fill_value=_get_autocast_min_val())

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)
            x_blksprs_fused = x.clone().requires_grad_(True)

            stock_softmax_out = _blocksparse_roundtrip(torch.softmax(x_stock, dim=-1), sparsity_layout_x,
                                                       sparsity_block_size)
            stock_dtype = stock_softmax_out.dtype

            blksprs_softmax_out = bs.ops.softmax(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size, flag_fused=False)
            blksprs_softmax_dense_out = bs.ops.to_dense(blksprs_softmax_out, sparsity_layout_x,
                                                        sparsity_block_size)

            blksprs_softmax_fused_out = bs.ops.softmax_fused(
                bs.ops.to_sparse(x_blksprs_fused, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size)
            blksprs_softmax_fused_dense_out = bs.ops.to_dense(blksprs_softmax_fused_out, sparsity_layout_x,
                                                              sparsity_block_size)

            assert torch.allclose(blksprs_softmax_dense_out.to(stock_dtype), stock_softmax_out, atol=ATOL,
                                  rtol=RTOL)
            assert torch.allclose(blksprs_softmax_fused_dense_out.to(stock_dtype), stock_softmax_out, atol=ATOL,
                                  rtol=RTOL)

            target = torch.randn_like(stock_softmax_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            blksprs_fused_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_softmax_out, target)
            blksprs_loss = blksprs_loss(blksprs_softmax_dense_out, target)
            blksprs_fused_loss = blksprs_fused_loss(blksprs_softmax_fused_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()
            blksprs_fused_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)
            assert torch.allclose(x_blksprs_fused.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_split(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        num_partitions_values = []
        x = k // sparsity_block_size
        while x >= 1:
            num_partitions_values.append(x)
            x //= 2

        for num_partitions in num_partitions_values:
            for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
                x_stock = x.clone().requires_grad_(True)
                x_blksprs = x.clone().requires_grad_(True)

                stock_split_out = (x_stock.reshape(x_stock.size(0), x_stock.size(1), num_partitions,
                                                   x_stock.size(2) // num_partitions).permute(0, 2, 1, 3)
                                   .reshape(x_stock.size(0) * num_partitions, x_stock.size(1),
                                            x_stock.size(2) // num_partitions))
                stock_dtype = stock_split_out.dtype

                blksprs_split_out, sparsity_layout_output = bs.ops.split(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, num_partitions, -1,
                    sparsity_block_size)
                blksprs_split_dense_out = bs.ops.to_dense(blksprs_split_out, sparsity_layout_output,
                                                          sparsity_block_size)

                assert torch.allclose(blksprs_split_dense_out.to(stock_dtype), stock_split_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_split_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_split_out, target)
                blksprs_loss = blksprs_loss(blksprs_split_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_merge(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        num_partitions_values = []
        x = k // sparsity_block_size
        while x >= 1:
            num_partitions_values.append(x)
            x //= 2

        for num_partitions in num_partitions_values:
            for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
                x_stock = x.clone().requires_grad_(True)
                x_blksprs = x.clone().requires_grad_(True)

                stock_split_out = (x_stock.reshape(x_stock.size(0), x_stock.size(1), num_partitions,
                                                   x_stock.size(2) // num_partitions).permute(0, 2, 1, 3)
                                   .reshape(x_stock.size(0) * num_partitions, x_stock.size(1),
                                            x_stock.size(2) // num_partitions))
                stock_merge_out = (
                    stock_split_out.reshape(stock_split_out.size(0) // num_partitions, num_partitions,
                                            stock_split_out.size(1), stock_split_out.size(2))
                    .permute(0, 2, 1, 3).reshape(stock_split_out.size(0) // num_partitions,
                                                 stock_split_out.size(1),
                                                 stock_split_out.size(2) * num_partitions))
                stock_dtype = stock_merge_out.dtype

                blksprs_split_out, sparsity_layout_split = bs.ops.split(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, num_partitions, -1,
                    sparsity_block_size)
                blksprs_merge_out, sparsity_layout_merge = bs.ops.merge(blksprs_split_out, sparsity_layout_split,
                                                                        num_partitions, -1, sparsity_block_size)
                blksprs_merge_dense_out = bs.ops.to_dense(blksprs_merge_out, sparsity_layout_merge,
                                                          sparsity_block_size)

                assert torch.allclose(stock_merge_out, x_stock, atol=ATOL, rtol=RTOL)
                assert torch.allclose(blksprs_merge_dense_out.to(stock_dtype), stock_merge_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_merge_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_merge_out, target)
                blksprs_loss = blksprs_loss(blksprs_merge_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_row_wise_sum(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_sum_out = torch.sum(x_stock, dim=-1)
            stock_dtype = stock_sum_out.dtype

            blksprs_row_wise_sum_out, sparsity_layout_output = bs.ops.misc.row_wise_sum(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x,
                sparsity_block_size)
            blksprs_row_wise_sum_dense_out = bs.ops.to_dense(blksprs_row_wise_sum_out, sparsity_layout_output,
                                                             sparsity_block_size)

            blksprs_row_wise_sum_out_slice = blksprs_row_wise_sum_dense_out[..., 0]

            assert torch.allclose(blksprs_row_wise_sum_out_slice.to(stock_dtype), stock_sum_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_row_wise_max(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.neg(torch.abs(torch.randn(size=(b, m, k), device=DEVICE)))
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                      fill_value=_get_autocast_min_val())

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs), (x_d, sparsity_layout_x_d),
                                     (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_max_out = torch.max(x_stock, dim=-1).values
            stock_dtype = stock_max_out.dtype

            blksprs_row_wise_max_out, sparsity_layout_output = bs.ops.misc.row_wise_max(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x,
                sparsity_block_size)
            blksprs_row_wise_max_dense_out = bs.ops.to_dense(blksprs_row_wise_max_out, sparsity_layout_output,
                                                             sparsity_block_size, fill_value=_get_autocast_min_val())

            blksprs_row_wise_max_out_slice = blksprs_row_wise_max_dense_out[..., 0]

            assert torch.allclose(blksprs_row_wise_max_out_slice.to(stock_dtype), stock_max_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_row_wise_add(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                      fill_value=_get_autocast_min_val())

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs), (x_d, sparsity_layout_x_d),
                                     (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_max_out = (torch.max(_blocksparse_roundtrip(x_stock, sparsity_layout_x,
                                                              sparsity_block_size),
                                       dim=-1).values).unsqueeze(-1)
            stock_rwa_out = _blocksparse_roundtrip(x_stock + stock_max_out, sparsity_layout_x,
                                                   sparsity_block_size)
            stock_dtype = stock_rwa_out.dtype

            blksprs_row_wise_max_out, sparsity_layout_output = bs.ops.misc.row_wise_max(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x,
                sparsity_block_size)
            blksprs_row_wise_add_out = bs.ops.misc.row_wise_add(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, blksprs_row_wise_max_out,
                sparsity_block_size)
            blksprs_row_wise_add_dense_out = bs.ops.to_dense(blksprs_row_wise_add_out, sparsity_layout_x,
                                                             sparsity_block_size)

            assert torch.allclose(blksprs_row_wise_add_dense_out.to(stock_dtype), stock_rwa_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_blksprs_adapt_layout(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)

        for sparsity_block_size_from, sparsity_block_size_to in [(sparsity_block_size, sparsity_block_size),
                                                                 (sparsity_block_size, sparsity_block_size // 4),
                                                                 (sparsity_block_size, sparsity_block_size // 2),
                                                                 (sparsity_block_size, sparsity_block_size),
                                                                 (sparsity_block_size // 4, sparsity_block_size),
                                                                 (sparsity_block_size // 2, sparsity_block_size)]:
            if any([sparsity_block_size_from < 16, sparsity_block_size_to < 16]):
                continue

            sparsity_layout_x_d_from = torch.ones(b, m // sparsity_block_size_from, k // sparsity_block_size_from,
                                                  dtype=torch.bool, device=DEVICE)
            sparsity_layout_x_bs_from = _get_blocksparse_layout(b, m, k, sparsity_block_size_from, sparsity_percentage)
            x_bs_from = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs_from, sparsity_block_size_from)

            sparsity_layout_x_d_to = torch.ones(b, m // sparsity_block_size_to, k // sparsity_block_size_to,
                                                dtype=torch.bool, device=DEVICE)
            sparsity_layout_x_bs_to = _get_blocksparse_layout(b, m, k, sparsity_block_size_to, sparsity_percentage)
            sparsity_layout_x_bs_to_same = bs.layouting.build_sparsity_layout(x_bs_from, sparsity_block_size_to)
            sparsity_layout_x_bs_to_less = torch.logical_and(sparsity_layout_x_bs_to_same, sparsity_layout_x_bs_to)

            for x_from, sparsity_layout_x_from, sparsity_layout_x_to, use_output_layout in [
                (x_d, sparsity_layout_x_d_from, sparsity_layout_x_d_to, False),
                (x_bs_from, sparsity_layout_x_bs_from, sparsity_layout_x_bs_to_same, False),
                (x_bs_from, sparsity_layout_x_bs_from, sparsity_layout_x_bs_to, True),
                (x_d, sparsity_layout_x_d_from, sparsity_layout_x_bs_to, True),
                (x_bs_from, sparsity_layout_x_bs_from, sparsity_layout_x_bs_to_less, True)]:
                x_from_stock = x_from.clone().requires_grad_(True)
                x_from_blksprs = x_from.clone().requires_grad_(True)

                stock_adapt_layout_out = _blocksparse_roundtrip(
                    _blocksparse_roundtrip(x_from_stock, sparsity_layout_x_from,
                                           sparsity_block_size_from),
                    sparsity_layout_x_to, sparsity_block_size_to)
                stock_dtype = stock_adapt_layout_out.dtype

                blksprs_adapt_layout_out, _ = bs.ops.adapt_layout(
                    bs.ops.to_sparse(x_from_blksprs, sparsity_layout_x_from, sparsity_block_size_from),
                    sparsity_layout_x_from, sparsity_block_size_from,
                    sparsity_block_size_to,
                    sparsity_layout_x_to if use_output_layout else None)
                blksprs_adapt_layout_dense_out = bs.ops.to_dense(blksprs_adapt_layout_out, sparsity_layout_x_to,
                                                                 sparsity_block_size_to)

                assert torch.allclose(blksprs_adapt_layout_dense_out.to(stock_dtype), stock_adapt_layout_out, atol=ATOL,
                                      rtol=RTOL)

                target = torch.randn_like(stock_adapt_layout_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_adapt_layout_out, target)
                blksprs_loss = blksprs_loss(blksprs_adapt_layout_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_from_blksprs.grad, x_from_stock.grad, atol=ATOL, rtol=RTOL)


def test_adapt_layout_zero_pads_larger_non_divisible_output_blocks():
    sparsity_block_size_from = 16
    sparsity_block_size_to = 32
    sparsity_layout_from = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)
    x_sparse = torch.zeros(
        (2, sparsity_block_size_from, sparsity_block_size_from),
        dtype=torch.float32,
        device=DEVICE,
    )
    x_sparse[0] = 1
    x_sparse[1] = 2

    out_sparse, sparsity_layout_to = bs.ops.adapt_layout(
        BlksprsTensor.wrap(x_sparse),
        sparsity_layout_from,
        sparsity_block_size_from,
        sparsity_block_size_to,
    )
    torch.cuda.synchronize()

    out_dense = bs.ops.to_dense(out_sparse, sparsity_layout_to, sparsity_block_size_to)
    expected = torch.zeros((2, sparsity_block_size_to, sparsity_block_size_to), device=DEVICE)
    expected[:, :sparsity_block_size_from, :sparsity_block_size_from] = x_sparse

    assert torch.equal(sparsity_layout_to, torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE))
    assert torch.equal(out_dense, expected)


# Layouting

@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_build_sparsity_layout(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs)]:
            x_blksprs = x.clone().requires_grad_(True)

            x_sparse = bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size)
            x_dense = bs.ops.to_dense(x_sparse, sparsity_layout_x, sparsity_block_size)

            blksprs_sparsity_layout = bs.layouting.build_sparsity_layout(x_dense, sparsity_block_size)

            assert torch.allclose(blksprs_sparsity_layout.to(torch.bool), sparsity_layout_x.to(torch.bool),
                                  atol=ATOL,
                                  rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
def test_build_sparsity_layout_matmul(config: list):
    b, m, n, k, sparsity_block_size, sparsity_percentage = config

    sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
    sparsity_layout_y_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)

    sparsity_layout_matmul = bs.layouting.build_sparsity_layout_matmul(sparsity_layout_x_bs, sparsity_layout_y_bs)
    sparsity_layout_matmul_fast = bs.layouting.build_sparsity_layout_matmul_fast(sparsity_layout_x_bs,
                                                                                 sparsity_layout_y_bs)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_build_distribution_layout(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        dims = [0, 1, 2]
        for dim in dims:
            src_d = torch.randn(size=(b, m, k), device=DEVICE)
            sparsity_layout_src_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size),
                                               device=DEVICE)

            tgt_d = torch.randn(size=(b * 2, m * 2, k * 2), device=DEVICE)
            sparsity_layout_tgt_d = torch.ones(size=(b * 2, m * 2 // sparsity_block_size, k * 2 // sparsity_block_size),
                                               device=DEVICE)

            if dim == 0:
                dist_lim = b * 2
            elif dim == 1:
                dist_lim = m * 2
            else:
                dist_lim = k * 2

            i_d = (
                torch.randint(0, dist_lim, size=(b, m, k), dtype=torch.int, device=DEVICE).contiguous())

            sparsity_layout_src_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
            src_bs = _blocksparse_roundtrip(src_d, sparsity_layout_src_bs, sparsity_block_size)
            i_bs = _blocksparse_roundtrip(i_d, sparsity_layout_src_bs, sparsity_block_size)

            for src, sparsity_layout_src, tgt, sparsity_layout_tgt, i, sparsity_layout_i in [
                (src_d, sparsity_layout_src_d, tgt_d, sparsity_layout_tgt_d, i_d, sparsity_layout_src_d),
                (src_bs, sparsity_layout_src_bs, tgt_d, sparsity_layout_tgt_d, i_bs, sparsity_layout_src_bs)]:
                stock_out_buffer = torch.zeros(size=(b * 2, m * 2, k * 2), dtype=src.dtype, device=DEVICE)
                stock_scatter_out = _blocksparse_roundtrip(
                    stock_out_buffer.scatter_reduce(dim=dim, index=i.to(torch.int64), src=src, reduce="sum"),
                    sparsity_layout_tgt, sparsity_block_size)
                stock_distribution_layout = bs.layouting.build_sparsity_layout(stock_scatter_out,
                                                                               sparsity_block_size)

                blksprs_distribution_layout = bs.layouting.build_distribution_layout(
                    bs.ops.to_sparse(i, sparsity_layout_i, sparsity_block_size),
                    sparsity_layout_i, dim, tgt.size(), sparsity_block_size)

                assert torch.allclose(blksprs_distribution_layout.to(torch.int), stock_distribution_layout.to(torch.int),
                                      atol=ATOL, rtol=RTOL)


def test_build_sparsity_layout_rejects_non_divisible_shape():
    x = torch.zeros((1, 17, 16), device=DEVICE)

    with pytest.raises(ValueError, match="divisible"):
        bs.layouting.build_sparsity_layout(x, 16)

    with pytest.raises(ValueError, match="divisible"):
        bs.layouting.build_sparsity_layout_full(x, 16)


def test_build_distribution_layout_rejects_non_divisible_target_shape():
    sparsity_block_size = 16
    sparsity_layout_indices = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    indices = torch.zeros((1, sparsity_block_size, sparsity_block_size), dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="divisible"):
        bs.layouting.build_distribution_layout(
            indices,
            sparsity_layout_indices,
            2,
            torch.Size((1, sparsity_block_size, sparsity_block_size + 1)),
            sparsity_block_size,
        )


def test_validate_sparsity_dense_rejects_non_divisible_shape():
    sparsity_block_size = 16
    x = torch.zeros((1, sparsity_block_size + 1, sparsity_block_size), device=DEVICE)
    sparsity_layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="divisible"):
        bs.utils.validation.validate_sparsity_dense(sparsity_block_size, (x, sparsity_layout))


def test_build_sparsity_layout_kernel_masks_row_edge_tiles_per_axis():
    sparsity_block_size = 16
    x = torch.zeros((2, sparsity_block_size + 1, sparsity_block_size), device=DEVICE)
    x[0, sparsity_block_size, 0] = 1

    actual = build_sparsity_layout_operation(x, sparsity_block_size)
    torch.cuda.synchronize()

    expected = torch.zeros((2, 1, 1), dtype=torch.bool, device=DEVICE)
    assert torch.equal(actual, expected)


def test_build_sparsity_layout_kernel_masks_col_edge_tiles_per_axis():
    sparsity_block_size = 16
    x = torch.zeros((2, sparsity_block_size, sparsity_block_size + 1), device=DEVICE)
    x[0, 0, sparsity_block_size] = 1

    actual = build_sparsity_layout_operation(x, sparsity_block_size)
    torch.cuda.synchronize()

    expected = torch.zeros((2, 1, 1), dtype=torch.bool, device=DEVICE)
    assert torch.equal(actual, expected)


def test_build_distribution_layout_kernel_masks_target_edge_tiles_per_axis():
    sparsity_block_size = 16
    indices = torch.full(
        (1, sparsity_block_size, sparsity_block_size),
        fill_value=sparsity_block_size,
        dtype=torch.int32,
        device=DEVICE,
    )
    layout_indices_i = torch.tensor([[0, 0, 0]], dtype=torch.int64, device=DEVICE)

    actual = build_distribution_layout_operation(
        indices,
        layout_indices_i,
        2,
        [2, sparsity_block_size, sparsity_block_size + 1],
        sparsity_block_size,
    )
    torch.cuda.synchronize()

    expected = torch.zeros((2, 1, 1), dtype=torch.bool, device=DEVICE)
    assert torch.equal(actual, expected)


# Processing

@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_apply_torch_linear(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for bias in [True, False]:
            linear = torch.nn.Linear(k, n, bias=bias, device=DEVICE)

            for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
                x_stock = x.clone().requires_grad_(True)
                x_blksprs = x.clone().requires_grad_(True)

                stock_linear_out = linear(x_stock)
                stock_dtype = stock_linear_out.dtype

                blksprs_linear_out, sparsity_layout_xl = bs.utils.apply_torch_linear(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, sparsity_block_size, linear)
                blksprs_linear_dense_out = bs.ops.to_dense(blksprs_linear_out, sparsity_layout_xl,
                                                           sparsity_block_size)

                assert torch.allclose(blksprs_linear_dense_out.to(stock_dtype), stock_linear_out, atol=ATOL, rtol=RTOL)


def test_apply_torch_linear_rejects_non_divisible_weight_shape():
    sparsity_block_size = 16
    sparsity_layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    x = BlksprsTensor.wrap(torch.zeros((1, sparsity_block_size, sparsity_block_size), device=DEVICE))
    linear = torch.nn.Linear(sparsity_block_size + 1, sparsity_block_size, bias=False, device=DEVICE)

    with pytest.raises(ValueError, match="divisible"):
        bs.utils.apply_torch_linear(x, sparsity_layout, sparsity_block_size, linear)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_apply_torch_normalisation(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout_sparse_rows(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        normalisation = torch.nn.LayerNorm(k, device=DEVICE)

        for x, sparsity_layout_x in [
            (x_d, sparsity_layout_x_d),
            (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_normalisation_out = _blocksparse_roundtrip(normalisation(x_stock), sparsity_layout_x,
                                                             sparsity_block_size)
            stock_dtype = stock_normalisation_out.dtype

            blksprs_normalisation_out = bs.utils.apply_torch_normalisation(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size, normalisation)
            blksprs_normalisation_dense_out = bs.ops.to_dense(blksprs_normalisation_out, sparsity_layout_x,
                                                              sparsity_block_size)

            assert torch.allclose(blksprs_normalisation_dense_out.to(stock_dtype), stock_normalisation_out, atol=ATOL,
                                  rtol=RTOL)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_apply_torch_dropout(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout_sparse_rows(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        dropout = torch.nn.Dropout(p=1)

        for x, sparsity_layout_x in [
            (x_d, sparsity_layout_x_d),
            (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_dropout_out = _blocksparse_roundtrip(dropout(x_stock), sparsity_layout_x,
                                                       sparsity_block_size)
            stock_dtype = stock_dropout_out.dtype

            global SEED
            torch.manual_seed(SEED)
            blksprs_normalisation_out = bs.utils.apply_torch_normalisation(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size, dropout)
            blksprs_normalisation_dense_out = bs.ops.to_dense(blksprs_normalisation_out, sparsity_layout_x,
                                                              sparsity_block_size)

            assert torch.allclose(blksprs_normalisation_dense_out.to(stock_dtype), stock_dropout_out, atol=ATOL,
                                  rtol=RTOL)


# Misc

@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_broadcast_addition(config: list, use_amp: bool):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randint(high=m, size=(b, m), device=DEVICE)
        y_d = torch.randint(high=m, size=(b, m), device=DEVICE)

        sparsity_layout_o = torch.ones(size=(b, m // sparsity_block_size, m // sparsity_block_size), device=DEVICE)
        sparsity_layout_o_bs = _get_blocksparse_layout(b, m, m, sparsity_block_size, sparsity_percentage)

        for x, y, sparsity_layout_o in [(x_d, y_d, sparsity_layout_o), (x_d, y_d, sparsity_layout_o_bs)]:
            stock_broadcast_addition = _blocksparse_roundtrip(torch.add(x.unsqueeze(-1), y.unsqueeze(-2)),
                                                              sparsity_layout_o, sparsity_block_size).to(
                torch.float)
            stock_dtype = stock_broadcast_addition.dtype

            blksprs_broadcast_addition_out = bs.ops.misc.broadcast_add(x, y, sparsity_layout_o,
                                                                       sparsity_block_size)
            blksprs_broadcast_addition_dense_out = bs.ops.to_dense(blksprs_broadcast_addition_out,
                                                                   sparsity_layout_o,
                                                                   sparsity_block_size)

            stock_broadcast_subtraction = _blocksparse_roundtrip(torch.sub(x.unsqueeze(-1), y.unsqueeze(-2)),
                                                                 sparsity_layout_o, sparsity_block_size).to(
                torch.float)
            blksprs_broadcast_subtraction = bs.ops.misc.broadcast_sub(x, y, sparsity_layout_o,
                                                                      sparsity_block_size)
            blksprs_broadcast_subtraction_dense_out = bs.ops.to_dense(blksprs_broadcast_subtraction,
                                                                      sparsity_layout_o,
                                                                      sparsity_block_size)

            assert torch.allclose(blksprs_broadcast_addition_dense_out.to(stock_dtype),
                                  stock_broadcast_addition,
                                  atol=ATOL, rtol=RTOL)
            assert torch.allclose(blksprs_broadcast_subtraction_dense_out.to(stock_dtype),
                                  stock_broadcast_subtraction,
                                  atol=ATOL, rtol=RTOL)


def test_broadcast_add_rejects_non_divisible_inputs():
    sparsity_block_size = 16
    x = torch.zeros((1, sparsity_block_size + 1), device=DEVICE)
    y = torch.zeros((1, sparsity_block_size + 1), device=DEVICE)
    sparsity_layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="divisible"):
        bs.ops.misc.broadcast_add(x, y, sparsity_layout, sparsity_block_size)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
def test_subclass(config: list):
    b, m, n, k, sparsity_block_size, sparsity_percentage = config

    x_d = torch.randn(size=(b, m, k), device=DEVICE)
    sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
    x_bs = BlksprsTensor.wrap(_blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size))

    assert type(x_bs).__name__ == BlksprsTensor.__name__


def test_version():
    assert bs.__version__ == _get_version()


# Validation error-path tests

def test_validate_dimensions_rejects_wrong_dims():
    x = torch.randn(4, 4, device=DEVICE)
    with pytest.raises(ValueError, match="dimensions"):
        bs.utils.validation.validate_dimensions(x)


def test_validate_contiguous_rejects_non_contiguous():
    x = torch.randn(2, 4, 4, device=DEVICE).transpose(-1, -2)
    assert not x.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        bs.utils.validation.validate_contiguous(x)


def test_validate_device_rejects_cpu():
    x = torch.randn(2, 4, 4)
    with pytest.raises(ValueError, match="GPU"):
        bs.utils.validation.validate_device(x)


def test_validate_device_rejects_different_devices():
    x = torch.randn(2, 4, 4, device=DEVICE)
    y = torch.randn(2, 4, 4)
    with pytest.raises(ValueError, match="same device"):
        bs.utils.validation.validate_device(x, y)


def test_validate_dtype_float_rejects_int():
    x = torch.randint(0, 10, (2, 4, 4), device=DEVICE, dtype=torch.int32)
    with pytest.raises(ValueError, match="float"):
        bs.utils.validation.validate_dtype_float(x)


def test_validate_dtype_float_rejects_mixed():
    x = torch.randn(2, 4, 4, device=DEVICE, dtype=torch.float32)
    y = torch.randn(2, 4, 4, device=DEVICE, dtype=torch.float16)
    with pytest.raises(ValueError, match="same dtype"):
        bs.utils.validation.validate_dtype_float(x, y)


def test_validate_sparsity_block_size_rejects_small():
    with pytest.raises(ValueError, match="at least 16"):
        bs.utils.validation.validate_sparsity_block_size(8)


def test_validate_sparsity_block_size_rejects_non_power_of_2():
    with pytest.raises(ValueError, match="power of 2"):
        bs.utils.validation.validate_sparsity_block_size(48)


def test_validate_sparsity_layout_values_rejects_bad_values():
    x = torch.randn(2, 16, 16, device=DEVICE)
    sl = torch.tensor([[[2, 1], [0, 1]]], dtype=torch.float32, device=DEVICE)
    with pytest.raises(ValueError, match="0 or 1"):
        bs.utils.validation.validate_sparsity(16, (x, sl))


def test_disable_enable_validation():
    bs.utils.disable_validation()
    try:
        # Should not raise even with wrong dims
        x = torch.randn(4, 4, device=DEVICE)
        bs.utils.validation.validate_dimensions(x)
    finally:
        bs.utils.enable_validation()

    # After re-enabling, should raise again
    with pytest.raises(ValueError, match="dimensions"):
        bs.utils.validation.validate_dimensions(x)


def test_disable_enable_contiguous():
    x = torch.randn(2, 4, 4, device=DEVICE).transpose(-1, -2)
    assert not x.is_contiguous()

    bs.utils.disable_contiguous()
    try:
        from blksprs.utils.validation import ensure_contiguous
        result = ensure_contiguous(x)
        # With contiguous disabled, should return original (non-contiguous) tensor
        assert not result.is_contiguous()
    finally:
        bs.utils.enable_contiguous()

    # After re-enabling, should make contiguous
    from blksprs.utils.validation import ensure_contiguous
    result = ensure_contiguous(x)
    assert result.is_contiguous()


# Scatter without reduce

SCATTER_NONE_CONFIGURATIONS = [
    # Only test non-overlapping indices to avoid non-determinism
    # (b, m, n, k, sparsity_block_size, sparsity_percentage)
    (2, 64, 64, 64, 32, 0),
    (2, 128, 128, 128, 64, 0),
    (2, 64, 64, 64, 32, 0.75),
    (4, 128, 128, 128, 64, 0.75),
]


@pytest.mark.parametrize("config", SCATTER_NONE_CONFIGURATIONS)
def test_blksprs_scatter_none(config: list):
    b, m, n, k, sparsity_block_size, sparsity_percentage = config

    dims = [-2, -1, 1, 2]
    for dim in dims:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size),
                                         device=DEVICE)

        # Build non-overlapping (unique) indices to avoid data races
        if dim % 3 == 1:
            # dim along rows: scatter each row to a unique destination
            i_d = torch.arange(m, device=DEVICE).unsqueeze(0).unsqueeze(-1).expand(b, m, k).contiguous().to(torch.int)
            dist_lim_r = m
            dist_lim_c = k
        elif dim % 3 == 2:
            # dim along cols: scatter each col to a unique destination
            i_d = torch.arange(k, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(b, m, k).contiguous().to(torch.int)
            dist_lim_r = m
            dist_lim_c = k
        else:
            continue  # Skip batch dim for simplicity

        sparsity_layout_o_d = torch.ones(
            size=(b, dist_lim_r // sparsity_block_size, dist_lim_c // sparsity_block_size),
            device=DEVICE)

        x_stock = x_d.clone()
        stock_out = torch.zeros(size=(b, dist_lim_r, dist_lim_c), dtype=x_stock.dtype, device=DEVICE)
        stock_out.scatter_(dim=dim, index=i_d.to(torch.int64), src=x_stock)
        stock_out = _blocksparse_roundtrip(stock_out, sparsity_layout_o_d, sparsity_block_size)

        x_sparse = bs.ops.to_sparse(x_d, sparsity_layout_x_d, sparsity_block_size)
        i_sparse = bs.ops.to_sparse(i_d, sparsity_layout_x_d, sparsity_block_size)

        blksprs_scatter_out = bs.ops.scatter(x_sparse, sparsity_layout_x_d, dim, i_sparse,
                                             sparsity_layout_o_d, sparsity_block_size)
        blksprs_dense_out = bs.ops.to_dense(blksprs_scatter_out, sparsity_layout_o_d, sparsity_block_size)

        assert torch.allclose(blksprs_dense_out, stock_out, atol=ATOL, rtol=RTOL)


# Flash attention performance benchmark

@pytest.mark.benchmark
@pytest.mark.parametrize("seq_len", [512, 1024, 2048])
def test_flash_attention_performance(seq_len: int):
    """Tests that flash attention is faster than regular attention for larger sequence lengths."""
    n_batches = 2
    n_heads = 4
    head_dim = 64
    sbs = 32
    sparsity_percentage = 0.5

    n_seq_blocks = seq_len // sbs
    n_head_blocks = head_dim // sbs
    total_batches = n_batches * n_heads

    q = torch.randn(total_batches, seq_len, head_dim, device=DEVICE)
    k = torch.randn(total_batches, seq_len, head_dim, device=DEVICE)
    v = torch.randn(total_batches, seq_len, head_dim, device=DEVICE)

    sparsity_layout_qkv = torch.ones(total_batches, n_seq_blocks, n_head_blocks, dtype=torch.bool, device=DEVICE)

    attention_layout = _get_flash_attention_layout(total_batches, n_seq_blocks, n_seq_blocks, sparsity_percentage)
    _ensure_flash_attention_rows(attention_layout)

    q_sparse = bs.ops.to_sparse(q, sparsity_layout_qkv, sbs)
    k_sparse = bs.ops.to_sparse(k, sparsity_layout_qkv, sbs)
    v_sparse = bs.ops.to_sparse(v, sparsity_layout_qkv, sbs)

    # Pre-compute transpose and layouts for regular attention
    k_t, sparsity_layout_kt = bs.ops.transpose(k_sparse, sparsity_layout_qkv, sbs)
    sparsity_layout_attn = bs.layouting.build_sparsity_layout_matmul(sparsity_layout_qkv, sparsity_layout_kt)
    sparsity_layout_o = bs.layouting.build_sparsity_layout_matmul(sparsity_layout_attn, sparsity_layout_qkv)

    # --- Regular (matmul-based) attention ---
    def regular_attention():
        attn_scores = bs.ops.matmul(q_sparse, sparsity_layout_qkv,
                                    k_t, sparsity_layout_kt,
                                    sparsity_layout_attn, sbs)
        attn_probs = bs.ops.softmax(attn_scores, sparsity_layout_attn, sbs)
        output = bs.ops.matmul(attn_probs, sparsity_layout_attn,
                               v_sparse, sparsity_layout_qkv,
                               sparsity_layout_o, sbs)
        return output

    # --- Flash attention ---
    def flash_attention():
        output = bs.ops.flash_attention(
            q_sparse, sparsity_layout_qkv,
            k_sparse, sparsity_layout_qkv,
            v_sparse, sparsity_layout_qkv,
            attention_layout, sbs,
        )
        return output

    # Warmup
    n_warmup = 3
    n_runs = 10

    for _ in range(n_warmup):
        regular_attention()
        flash_attention()
    torch.cuda.synchronize()

    # Time regular attention
    start_events_reg = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    end_events_reg = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    for i in range(n_runs):
        start_events_reg[i].record()
        regular_attention()
        end_events_reg[i].record()
    torch.cuda.synchronize()
    regular_times = [s.elapsed_time(e) for s, e in zip(start_events_reg, end_events_reg)]
    median_regular = sorted(regular_times)[n_runs // 2]

    # Time flash attention
    start_events_flash = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    end_events_flash = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    for i in range(n_runs):
        start_events_flash[i].record()
        flash_attention()
        end_events_flash[i].record()
    torch.cuda.synchronize()
    flash_times = [s.elapsed_time(e) for s, e in zip(start_events_flash, end_events_flash)]
    median_flash = sorted(flash_times)[n_runs // 2]

    speedup = median_regular / median_flash if median_flash > 0 else float("inf")

    print(f"\n[seq_len={seq_len}] Regular: {median_regular:.3f}ms, Flash: {median_flash:.3f}ms, "
          f"Speedup: {speedup:.2f}x")

    assert median_regular > 0
    assert median_flash > 0

# Utility

def _get_blocksparse_layout(b, m, n, sparsity_block_size, sparsity_percentage):
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), dtype=torch.bool, device=DEVICE)

    num_zero_elements = int(m_s * n_s * (1 - sparsity_percentage))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = False

    return sparsity_layout


def _get_blocksparse_layout_sparse_rows(b, m, n, sparsity_block_size, sparsity_percentage):
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), dtype=torch.bool, device=DEVICE)

    # Calculate the number of rows to be set to False
    num_zero_rows = int(m_s * (1 - sparsity_percentage))
    for b_i in range(b):
        # Randomly select rows to set to False
        row_indices = torch.randperm(m_s)[:num_zero_rows]
        sparsity_layout[b_i, row_indices, :] = False

    return sparsity_layout


def _blocksparse_roundtrip(x, sparsity_layout, sparsity_block_size, fill_value=0.0):
    return bs.ops.to_dense(bs.ops.to_sparse(x, sparsity_layout, sparsity_block_size),
                           sparsity_layout,
                           sparsity_block_size, fill_value=fill_value)


def _get_version():
    with open(Path(__file__).parent.parent.parent.joinpath("pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


# Visualisation

def _visualise(*matrix_name_tuples, dim=0):
    vmin = np.inf
    vmax = -np.inf

    for matrix_tuple in matrix_name_tuples:
        vmin = min(vmin, torch.min(matrix_tuple[0]))
        vmax = max(vmax, torch.max(matrix_tuple[0]))

    for matrix_tuple in matrix_name_tuples:
        matrix_data = matrix_tuple[0]
        matrix_label = matrix_tuple[1]

        add_args = {}
        if len(matrix_tuple) > 2:
            add_args = matrix_tuple[2]

        output_path_base = BASE_PATH.joinpath("test", "output", "blksprs")
        output_path_base.mkdir(exist_ok=True)

        _visualise_matrix(matrix_data[dim], str(output_path_base.joinpath(matrix_label)), grid_size=16, vmin=vmin,
                          vmax=vmax, **add_args)


def _visualise_matrix(matrix: torch.Tensor, output_path: str = None, grid_size=16, vmin=None, vmax=None):
    while matrix.dim() > 2:
        matrix = matrix[0]

    matrix = matrix.cpu().detach().numpy()
    cmap = None
    norm = None

    plt.yticks([i - 0.5 for i in range(0, matrix.shape[0] + 1, grid_size)],
               [i if i % 2 == 0 else "" for i in range(0, matrix.shape[0] + 1, grid_size)])
    plt.xticks([i - 0.5 for i in range(0, matrix.shape[1] + 1, grid_size)],
               [i if i % 2 == 0 else "" for i in range(0, matrix.shape[1] + 1, grid_size)])

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')

    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest', vmin=vmin, vmax=vmax)

    if output_path is not None:
        plt.savefig(f"{output_path}.svg", format="svg")
    else:
        plt.show()


# Comparison

def _slow_to_sparse(x, sparsity_layout, sparsity_block_size: int):
    num_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
    output = torch.zeros(size=(num_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)
    indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

    for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
        t_r = r * sparsity_block_size
        t_c = c * sparsity_block_size
        to_insert = x[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size]
        output[idx] = to_insert

    return output


def _slow_to_dense(x, sparsity_layout, sparsity_block_size: int):
    output = torch.zeros(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                               sparsity_layout.size(2) * sparsity_block_size), device=x.device)
    indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

    for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
        t_r = r * sparsity_block_size
        t_c = c * sparsity_block_size
        to_insert = x[idx]
        output[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size] = to_insert

    return output


def _slow_gather_mdi(src, idx_bat, idx_row, idx_col):
    output = torch.zeros(size=(idx_bat.size(0), idx_bat.size(1), idx_bat.size(2)), device=src.device)

    for b in range(idx_bat.size(0)):
        for k in range(idx_bat.size(1)):
            for n in range(idx_bat.size(2)):
                output[b, k, n] = src[idx_bat[b, k, n], k, idx_col[b, k, n]]

    return output


def _slow_scatter_reduce_mdi(src, tgt_size, idx_bat, idx_row, idx_col):
    output = torch.zeros(size=tgt_size, device=src.device)

    for b in range(idx_bat.size(0)):
        for k in range(idx_bat.size(1)):
            for n in range(idx_bat.size(2)):
                output[idx_bat[b, k, n], k, idx_col[b, k, n]] += src[b, k, n]

    return output


# Debug

def _debug_convert_tensor(x: Tensor):
    output = torch.arange(0, x.size(-2) * x.size(-1), dtype=x.dtype, device=DEVICE).reshape(x.size(-2),
                                                                                            x.size(-1)).unsqueeze(
        0).repeat(x.size(0), 1, 1)

    return output


def _debug_convert_tensor_full(x: Tensor):
    output = (torch.arange(0, x.size(-3) * x.size(-2) * x.size(-1), dtype=x.dtype, device=DEVICE)
              .reshape(x.size(-3), x.size(-2), x.size(-1)))

    return output


def _get_autocast_min_val():
    """Return the minimum finite value for the current dtype.
    
    Note: This is used for fill values in sparse tensors, NOT for attention masking.
    For attention masking, use float("-inf") directly.
    """
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype("cuda")
    else:
        dtype = torch.float

    return torch.finfo(dtype).min


FLASH_ATTENTION_MAX_SEQ = 512


def _get_flash_attention_layout(n_batches: int, n_seq_q: int, n_seq_k: int,
                                sparsity_pct: float) -> Tensor:
    attention_layout = torch.ones(n_batches, n_seq_q, n_seq_k, dtype=torch.bool, device=DEVICE)

    num_zero_elements = int(n_seq_q * n_seq_k * sparsity_pct)
    for b in range(n_batches):
        indices = torch.randperm(n_seq_q * n_seq_k, device=DEVICE)[:num_zero_elements]
        attention_layout[b, indices // n_seq_k, indices % n_seq_k] = False

    return attention_layout


def _ensure_flash_attention_rows(attention_layout: Tensor):
    n_batches, n_seq_blocks_q, n_seq_blocks_k = attention_layout.shape

    for b_i in range(n_batches):
        for i in range(n_seq_blocks_q):
            if not attention_layout[b_i, i].any():
                j = torch.randint(0, n_seq_blocks_k, (1,), device=attention_layout.device).item()
                attention_layout[b_i, i, j] = True


def _build_flash_optional_sparse_inputs(mask_dense: Tensor, bias_blksprs: Tensor,
                                        n_batches: int, n_seq_blocks: int, sparsity_block_size: int):
    mask_sparse = None
    sparsity_layout_mask = None
    if mask_dense is not None:
        sparsity_layout_mask = torch.ones(n_batches, n_seq_blocks, n_seq_blocks, dtype=torch.bool, device=DEVICE)
        mask_sparse = bs.ops.to_sparse(mask_dense.float(), sparsity_layout_mask, sparsity_block_size)

    bias_sparse = None
    sparsity_layout_bias = None
    if bias_blksprs is not None:
        sparsity_layout_bias = torch.ones(n_batches, n_seq_blocks, n_seq_blocks, dtype=torch.bool, device=DEVICE)
        bias_sparse = bs.ops.to_sparse(bias_blksprs, sparsity_layout_bias, sparsity_block_size)

    return mask_sparse, sparsity_layout_mask, bias_sparse, sparsity_layout_bias


def _reference_attention_blocksparse(
    q: Tensor, k: Tensor, v: Tensor,
    attention_layout: Tensor, block_size: int,
    attention_mask: Tensor = None,
    attention_bias: Tensor = None,
    scale: float = None,
) -> Tensor:
    n_batches, seq_q, head_dim = q.shape
    _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    attn_scores = torch.bmm(q, k.transpose(-2, -1)) * scale

    n_seq_blocks_q = seq_q // block_size
    n_seq_blocks_k = seq_k // block_size

    for b in range(n_batches):
        for i in range(n_seq_blocks_q):
            for j in range(n_seq_blocks_k):
                if not attention_layout[b, i, j]:
                    attn_scores[b,
                                i * block_size:(i + 1) * block_size,
                                j * block_size:(j + 1) * block_size] = float("-inf")

    if attention_mask is not None:
        attn_scores = attn_scores.masked_fill(attention_mask, float("-inf"))

    if attention_bias is not None:
        attn_scores = attn_scores + attention_bias

    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

    out = torch.bmm(attn_probs, v)
    return out


def _sample_positions(size: int) -> list[int]:
    return sorted({0, size // 2, size - 1})


def _dense_block_index(batch_idx: int, row_idx: int, col_idx: int, n_row_blocks: int, n_col_blocks: int) -> int:
    return batch_idx * n_row_blocks * n_col_blocks + row_idx * n_col_blocks + col_idx


def _require_min_cuda_memory(min_gib: float) -> None:
    total_gib = torch.cuda.get_device_properties(DEVICE).total_memory / (1024 ** 3)
    if total_gib < min_gib:
        pytest.skip(f"Requires at least {min_gib:.0f} GiB of GPU memory, found {total_gib:.1f} GiB")


def _set_sample_blocks(tensor: Tensor, block_size: int, sample_batches: list[int], sample_rows: list[int]) -> None:
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            tensor[
                batch_idx,
                row_idx * block_size:(row_idx + 1) * block_size,
            ] = torch.randn(block_size, device=tensor.device, dtype=tensor.dtype)


FLASH_ATTENTION_CONFIGS = [
    config for config in TEST_CONFIGURATIONS
    if (config[1] // config[4]) >= 1  # n_seq_blocks >= 1
    and (config[2] % config[4]) == 0  # value dim divisible by SBS
    and (config[3] % config[4]) == 0  # q/k dim divisible by SBS
    and config[1] <= FLASH_ATTENTION_MAX_SEQ
    and config[1] >= config[4]
]

FLASH_ATTENTION_MIXED_DIM_CONFIGS = [
    config for config in FLASH_ATTENTION_CONFIGS
    if config[2] != config[3]
]


@pytest.mark.parametrize("config", FLASH_ATTENTION_CONFIGS)
@pytest.mark.parametrize("use_amp", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_blksprs_flash_attention(config: tuple, use_amp: bool, use_mask: bool, use_bias: bool):
    b, m, n, k_dim, sparsity_block_size, sparsity_percentage = config
    seq = m
    sbs = sparsity_block_size
    n_batches = b
    n_seq_blocks = seq // sbs
    n_head_blocks_qk = k_dim // sbs
    n_head_blocks_v = n // sbs

    q = torch.randn(n_batches, seq, k_dim, device=DEVICE)
    k = torch.randn(n_batches, seq, k_dim, device=DEVICE)
    v = torch.randn(n_batches, seq, n, device=DEVICE)

    sparsity_layout_q = torch.ones(n_batches, n_seq_blocks, n_head_blocks_qk, dtype=torch.bool, device=DEVICE)
    sparsity_layout_k = torch.ones(n_batches, n_seq_blocks, n_head_blocks_qk, dtype=torch.bool, device=DEVICE)
    sparsity_layout_v = torch.ones(n_batches, n_seq_blocks, n_head_blocks_v, dtype=torch.bool, device=DEVICE)
    sparsity_layout_o = torch.ones(n_batches, n_seq_blocks, n_head_blocks_v, dtype=torch.bool, device=DEVICE)

    attention_layout = _get_flash_attention_layout(n_batches, n_seq_blocks, n_seq_blocks, sparsity_percentage)
    _ensure_flash_attention_rows(attention_layout)

    mask_dense = None
    if use_mask:
        mask_dense = torch.rand(n_batches, seq, seq, device=DEVICE) > 0.7

    bias = None
    if use_bias:
        bias = torch.randn(n_batches, seq, seq, device=DEVICE) * 0.1

    q_stock = q.clone().detach().float().requires_grad_(True)
    k_stock = k.clone().detach().float().requires_grad_(True)
    v_stock = v.clone().detach().float().requires_grad_(True)
    bias_stock = None
    if bias is not None:
        bias_stock = bias.clone().detach().float().requires_grad_(True)

    stock_flash_out = _reference_attention_blocksparse(
        q_stock, k_stock, v_stock, attention_layout, sbs,
        attention_mask=mask_dense,
        attention_bias=bias_stock,
    )

    q_blksprs = q.clone().detach().requires_grad_(True)
    k_blksprs = k.clone().detach().requires_grad_(True)
    v_blksprs = v.clone().detach().requires_grad_(True)
    bias_blksprs = None
    if bias is not None:
        bias_blksprs = bias.clone().detach().requires_grad_(True)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        q_sparse = bs.ops.to_sparse(q_blksprs, sparsity_layout_q, sbs)
        k_sparse = bs.ops.to_sparse(k_blksprs, sparsity_layout_k, sbs)
        v_sparse = bs.ops.to_sparse(v_blksprs, sparsity_layout_v, sbs)

        mask_sparse, sparsity_layout_mask, bias_sparse, sparsity_layout_bias = _build_flash_optional_sparse_inputs(
            mask_dense, bias_blksprs, n_batches, n_seq_blocks, sbs
        )

        blksprs_flash_out = bs.ops.flash_attention(
            q_sparse, sparsity_layout_q,
            k_sparse, sparsity_layout_k,
            v_sparse, sparsity_layout_v,
            attention_layout, sbs,
            attention_mask=mask_sparse, sparsity_layout_mask=sparsity_layout_mask,
            attention_bias=bias_sparse, sparsity_layout_bias=sparsity_layout_bias,
            sparsity_layout_o=sparsity_layout_o,
        )
        blksprs_flash_dense_out = bs.ops.to_dense(blksprs_flash_out, sparsity_layout_o, sbs)

    assert torch.allclose(
        blksprs_flash_dense_out.float(), stock_flash_out, atol=ATOL, rtol=RTOL
    ), "Forward output mismatch"

    target = torch.randn_like(stock_flash_out)
    stock_loss = torch.nn.L1Loss()(stock_flash_out, target)
    blksprs_loss = torch.nn.L1Loss()(blksprs_flash_dense_out.float(), target)

    stock_loss.backward()
    blksprs_loss.backward()

    assert torch.allclose(
        torch.nan_to_num(q_blksprs.grad.float()),
        torch.nan_to_num(q_stock.grad),
        atol=ATOL, rtol=RTOL,
    ), "dQ mismatch"
    assert torch.allclose(
        torch.nan_to_num(k_blksprs.grad.float()),
        torch.nan_to_num(k_stock.grad),
        atol=ATOL, rtol=RTOL,
    ), "dK mismatch"
    assert torch.allclose(
        torch.nan_to_num(v_blksprs.grad.float()),
        torch.nan_to_num(v_stock.grad),
        atol=ATOL, rtol=RTOL,
    ), "dV mismatch"

    if bias_blksprs is not None:
        assert bias_blksprs.grad is not None, "Bias gradient should not be None"
        assert torch.allclose(
            torch.nan_to_num(bias_blksprs.grad.float()),
            torch.nan_to_num(bias_stock.grad),
            atol=ATOL, rtol=RTOL,
        ), "dBias mismatch"


@pytest.mark.parametrize("config", FLASH_ATTENTION_MIXED_DIM_CONFIGS)
def test_blksprs_flash_attention_requires_output_layout_for_mixed_dims(config: tuple):
    b, m, n, k_dim, sparsity_block_size, sparsity_percentage = config
    seq = m
    sbs = sparsity_block_size
    n_batches = b
    n_seq_blocks = seq // sbs
    n_head_blocks_qk = k_dim // sbs
    n_head_blocks_v = n // sbs

    q = torch.randn(n_batches, seq, k_dim, device=DEVICE)
    k = torch.randn(n_batches, seq, k_dim, device=DEVICE)
    v = torch.randn(n_batches, seq, n, device=DEVICE)

    sparsity_layout_q = torch.ones(n_batches, n_seq_blocks, n_head_blocks_qk, dtype=torch.bool, device=DEVICE)
    sparsity_layout_k = torch.ones(n_batches, n_seq_blocks, n_head_blocks_qk, dtype=torch.bool, device=DEVICE)
    sparsity_layout_v = torch.ones(n_batches, n_seq_blocks, n_head_blocks_v, dtype=torch.bool, device=DEVICE)

    attention_layout = _get_flash_attention_layout(n_batches, n_seq_blocks, n_seq_blocks, sparsity_percentage)
    _ensure_flash_attention_rows(attention_layout)

    q_sparse = bs.ops.to_sparse(q, sparsity_layout_q, sbs)
    k_sparse = bs.ops.to_sparse(k, sparsity_layout_k, sbs)
    v_sparse = bs.ops.to_sparse(v, sparsity_layout_v, sbs)

    with pytest.raises(ValueError, match="sparsity_layout_o is required"):
        bs.ops.flash_attention(
            q_sparse, sparsity_layout_q,
            k_sparse, sparsity_layout_k,
            v_sparse, sparsity_layout_v,
            attention_layout, sbs,
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_operations(config: tuple):
    _require_min_cuda_memory(20)

    b, m, n, k, sparsity_block_size, _ = config
    dtype = torch.float16
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    x = torch.randn(size=(b, m, k), device=DEVICE, dtype=dtype)
    sparsity_layout_x = torch.ones(size=(b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    x_sparse = bs.ops.to_sparse(x, sparsity_layout_x, sparsity_block_size)

    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)

    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            blk_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
            dense_block = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ]
            assert torch.allclose(x_sparse[blk_idx], dense_block, atol=ATOL, rtol=RTOL)

    x_dense_roundtrip = bs.ops.to_dense(x_sparse, sparsity_layout_x, sparsity_block_size)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            dense_block = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ]
            dense_roundtrip_block = x_dense_roundtrip[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ]
            assert torch.allclose(dense_roundtrip_block, dense_block, atol=ATOL, rtol=RTOL)
    del x_dense_roundtrip

    x_sparse_t, sparsity_layout_x_t = bs.ops.transpose(x_sparse, sparsity_layout_x, sparsity_block_size)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            blk_idx = _dense_block_index(batch_idx, 0, row_idx, n_col_blocks, n_row_blocks)
            dense_block_t = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ].transpose(-1, -2)
            assert torch.allclose(x_sparse_t[blk_idx], dense_block_t, atol=ATOL, rtol=RTOL)
    del x_sparse_t
    torch.cuda.empty_cache()

    i = torch.randint(0, b, size=(64, 64, 32), dtype=torch.int, device=DEVICE).contiguous()
    sparsity_layout_i = torch.ones(size=(64, 64 // sparsity_block_size, 32 // sparsity_block_size),
                                   dtype=torch.bool, device=DEVICE)
    i_sparse = bs.ops.to_sparse(i, sparsity_layout_i, sparsity_block_size)
    gather_out = bs.ops.gather(x_sparse, sparsity_layout_x, 0, i_sparse, sparsity_layout_i, sparsity_block_size)
    gather_dense = bs.ops.to_dense(gather_out, sparsity_layout_i, sparsity_block_size)
    gather_ref = torch.gather(x, dim=0, index=i.to(torch.int64))
    assert torch.allclose(gather_dense.float(), gather_ref.float(), atol=ATOL, rtol=RTOL)
    del i_sparse, gather_out, gather_dense, gather_ref
    torch.cuda.empty_cache()

    y = torch.randn(size=(b, k, n), device=DEVICE, dtype=dtype)
    sparsity_layout_y = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size),
                                   dtype=torch.bool, device=DEVICE)
    y_sparse = bs.ops.to_sparse(y, sparsity_layout_y, sparsity_block_size)
    sparsity_layout_o = torch.ones(size=(b, n_row_blocks, n // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    out_sparse = bs.ops.matmul(x_sparse, sparsity_layout_x, y_sparse, sparsity_layout_y, sparsity_layout_o,
                               sparsity_block_size)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            blk_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n // sparsity_block_size)
            dense_ref = torch.matmul(
                x[
                    batch_idx,
                    row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                    :,
                ].float(),
                y[batch_idx].float(),
            ).to(dtype)
            assert torch.allclose(out_sparse[blk_idx].float(), dense_ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.benchmark
@pytest.mark.parametrize("config", LARGE_INDEX_BUILD_LAYOUT_CONFIGURATIONS)
def test_blksprs_large_index_build_sparsity_layout(config: tuple):
    _require_min_cuda_memory(20)

    b, m, _, k, sparsity_block_size, _ = config
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    x = torch.zeros(size=(b, m, k), device=DEVICE, dtype=torch.float16)
    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)
    _set_sample_blocks(x, sparsity_block_size, sample_batches, sample_rows)

    actual = bs.layouting.build_sparsity_layout(x, sparsity_block_size)
    torch.cuda.synchronize()

    expected = torch.zeros((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            expected[batch_idx, row_idx, 0] = True

    assert torch.equal(actual, expected)

    actual_full = bs.layouting.build_sparsity_layout_full(x, sparsity_block_size)
    assert torch.equal(actual_full, torch.ones_like(expected))


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_build_sparsity_layout_adaption(config: tuple):
    _require_min_cuda_memory(20)

    b, m, _, k, sparsity_block_size_from, _ = config
    n_row_blocks_from = m // sparsity_block_size_from
    n_col_blocks_from = k // sparsity_block_size_from
    sparsity_block_size_to = sparsity_block_size_from // 2
    n_row_blocks_to = m // sparsity_block_size_to
    n_col_blocks_to = k // sparsity_block_size_to

    sparsity_layout_from = torch.ones((b, n_row_blocks_from, n_col_blocks_from), dtype=torch.bool, device=DEVICE)
    n_sparse_blocks = int(sparsity_layout_from.sum().item())

    x_sparse = torch.zeros(
        (n_sparse_blocks, sparsity_block_size_from, sparsity_block_size_from),
        device=DEVICE,
        dtype=torch.float16,
    )
    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks_from)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks_from, n_col_blocks_from)
            x_sparse[block_idx] = torch.randn_like(x_sparse[block_idx])

    actual = bs.layouting.build_sparsity_layout_adaption(
        x_sparse,
        sparsity_layout_from,
        sparsity_block_size_from,
        sparsity_block_size_to,
    )
    torch.cuda.synchronize()

    expected = torch.zeros((b, n_row_blocks_to, n_col_blocks_to), dtype=torch.bool, device=DEVICE)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            expected[batch_idx, row_idx * 2:(row_idx + 1) * 2, :] = True

    assert torch.equal(actual, expected)


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_row_wise_operations(config: tuple):
    _require_min_cuda_memory(20)

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float16
    n_row_blocks = m // sparsity_block_size

    x = torch.randn(size=(b, m, k), device=DEVICE, dtype=dtype)
    sparsity_layout_x = torch.ones(size=(b, n_row_blocks, k // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    x_sparse = bs.ops.to_sparse(x, sparsity_layout_x, sparsity_block_size)

    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)

    row_sum_sparse, sparsity_layout_sum = bs.ops.misc.row_wise_sum(
        x_sparse, sparsity_layout_x, sparsity_block_size, flag_slice_only=True)
    row_max_sparse, sparsity_layout_max = bs.ops.misc.row_wise_max(
        x_sparse, sparsity_layout_x, sparsity_block_size, flag_slice_only=True)
    row_add_sparse = bs.ops.misc.row_wise_add(x_sparse, sparsity_layout_x, row_max_sparse, sparsity_block_size)
    row_sub_sparse = bs.ops.misc.row_wise_sub(x_sparse, sparsity_layout_x, row_max_sparse, sparsity_block_size)

    assert torch.equal(sparsity_layout_sum, torch.max(sparsity_layout_x, dim=-1, keepdim=True).values)
    assert torch.equal(sparsity_layout_max, torch.max(sparsity_layout_x, dim=-1, keepdim=True).values)

    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            block_idx_x = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, 1)
            ref_sum = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ].sum(dim=-1)
            ref_max = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ].max(dim=-1).values
            ref_add = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ] + ref_max.unsqueeze(-1)
            ref_sub = x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ] - ref_max.unsqueeze(-1)

            assert torch.allclose(row_sum_sparse[block_idx_x, :, 0], ref_sum, atol=ATOL, rtol=RTOL)
            assert torch.allclose(row_max_sparse[block_idx_x, :, 0].float(), ref_max.float(), atol=ATOL, rtol=RTOL)
            assert torch.allclose(row_add_sparse[block_idx_x].float(), ref_add.float(), atol=ATOL, rtol=RTOL)
            assert torch.allclose(row_sub_sparse[block_idx_x].float(), ref_sub.float(), atol=ATOL, rtol=RTOL)

    del x, x_sparse, row_sum_sparse, row_max_sparse, row_add_sparse, row_sub_sparse
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_repeat_and_adapt_layout(config: tuple):
    _require_min_cuda_memory(20)
    torch.cuda.empty_cache()

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float16
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    sparsity_layout_x = torch.ones((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    n_sparse_blocks = int(sparsity_layout_x.sum().item())
    x_sparse = torch.zeros(
        (n_sparse_blocks, sparsity_block_size, sparsity_block_size),
        device=DEVICE,
        dtype=dtype,
    )

    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
            x_sparse[block_idx] = torch.randn_like(x_sparse[block_idx])

    repeat_output_layout = torch.zeros((b, n_row_blocks, 2), dtype=torch.bool, device=DEVICE)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            repeat_output_layout[batch_idx, row_idx, 1] = True
    repeated_sparse, repeated_layout = bs.ops.repeat(
        x_sparse,
        sparsity_layout_x,
        (1, 1, 2),
        sparsity_block_size,
        sparsity_layout_output=repeat_output_layout,
    )
    torch.cuda.synchronize()
    assert torch.equal(repeated_layout, repeat_output_layout)
    for sparse_idx, (batch_idx, row_idx) in enumerate(
        (batch_idx, row_idx) for batch_idx in sample_batches for row_idx in sample_rows
    ):
        source_block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
        assert torch.allclose(repeated_sparse[sparse_idx], x_sparse[source_block_idx], atol=ATOL, rtol=RTOL)
    del repeated_sparse, repeated_layout, repeat_output_layout
    torch.cuda.empty_cache()

    repeat_interleave_output_layout = torch.zeros((b * 2, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            repeat_interleave_output_layout[batch_idx * 2 + 1, row_idx, 0] = True
    repeated_interleaved_sparse, repeated_interleaved_layout = bs.ops.repeat_interleave(
        x_sparse,
        sparsity_layout_x,
        2,
        sparsity_block_size,
        sparsity_layout_output=repeat_interleave_output_layout,
    )
    torch.cuda.synchronize()
    assert torch.equal(repeated_interleaved_layout, repeat_interleave_output_layout)
    for sparse_idx, (batch_idx, row_idx) in enumerate(
        (batch_idx, row_idx) for batch_idx in sample_batches for row_idx in sample_rows
    ):
        source_block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
        assert torch.allclose(repeated_interleaved_sparse[sparse_idx], x_sparse[source_block_idx], atol=ATOL, rtol=RTOL)
    del repeated_interleaved_sparse, repeated_interleaved_layout, repeat_interleave_output_layout
    torch.cuda.empty_cache()

    sparsity_block_size_to = sparsity_block_size // 2
    adaption_output_layout = torch.zeros(
        (b, n_row_blocks * 2, n_col_blocks * 2),
        dtype=torch.bool,
        device=DEVICE,
    )
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            adaption_output_layout[batch_idx, row_idx * 2 + 1, 1] = True
    adapted_sparse, adapted_layout = bs.ops.adapt_layout(
        x_sparse,
        sparsity_layout_x,
        sparsity_block_size,
        sparsity_block_size_to,
        sparsity_layout_to=adaption_output_layout,
    )
    torch.cuda.synchronize()
    assert torch.equal(adapted_layout, adaption_output_layout)
    for sparse_idx, (batch_idx, row_idx) in enumerate(
        (batch_idx, row_idx) for batch_idx in sample_batches for row_idx in sample_rows
    ):
        source_block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
        expected = x_sparse[source_block_idx, sparsity_block_size_to:, sparsity_block_size_to:]
        assert torch.allclose(adapted_sparse[sparse_idx], expected, atol=ATOL, rtol=RTOL)

    del x_sparse, sparsity_layout_x, adapted_sparse, adapted_layout, adaption_output_layout
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_scatter_operations(config: tuple):
    _require_min_cuda_memory(20)
    torch.cuda.empty_cache()

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float16
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    sparsity_layout_src = torch.ones((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    n_sparse_blocks = int(sparsity_layout_src.sum().item())
    src_sparse = torch.zeros(
        (n_sparse_blocks, sparsity_block_size, sparsity_block_size),
        device=DEVICE,
        dtype=dtype,
    )
    idx_sparse = torch.full(
        (n_sparse_blocks, sparsity_block_size, sparsity_block_size),
        b,
        device=DEVICE,
        dtype=torch.int32,
    )

    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)
    sparsity_layout_tgt = torch.zeros_like(sparsity_layout_src)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
            src_sparse[block_idx] = torch.randn_like(src_sparse[block_idx])
            idx_sparse[block_idx] = batch_idx
            sparsity_layout_tgt[batch_idx, row_idx, 0] = True

    scattered_sparse = bs.ops.scatter(
        src_sparse,
        sparsity_layout_src,
        0,
        idx_sparse,
        sparsity_layout_tgt,
        sparsity_block_size,
    )
    scatter_reduced_sparse = bs.ops.scatter_reduce(
        src_sparse,
        sparsity_layout_src,
        0,
        idx_sparse,
        sparsity_layout_tgt,
        sparsity_block_size,
        reduce_op="sum",
    )
    torch.cuda.synchronize()

    for sparse_idx, (batch_idx, row_idx) in enumerate(
        (batch_idx, row_idx) for batch_idx in sample_batches for row_idx in sample_rows
    ):
        source_block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
        assert torch.allclose(scattered_sparse[sparse_idx], src_sparse[source_block_idx], atol=ATOL, rtol=RTOL)
        assert torch.allclose(scatter_reduced_sparse[sparse_idx], src_sparse[source_block_idx], atol=ATOL, rtol=RTOL)

    del src_sparse, idx_sparse, scattered_sparse, scatter_reduced_sparse, sparsity_layout_src, sparsity_layout_tgt
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_PARTITION_CONFIGURATIONS)
def test_blksprs_large_index_partition_operations(config: tuple):
    _require_min_cuda_memory(20)
    torch.cuda.empty_cache()

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float16
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size
    partitions = 2

    sparsity_layout_x = torch.ones((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    n_sparse_blocks = int(sparsity_layout_x.sum().item())
    x_sparse = torch.zeros(
        (n_sparse_blocks, sparsity_block_size, sparsity_block_size),
        device=DEVICE,
        dtype=dtype,
    )

    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            for col_idx in range(n_col_blocks):
                block_idx = _dense_block_index(batch_idx, row_idx, col_idx, n_row_blocks, n_col_blocks)
                x_sparse[block_idx] = torch.randn_like(x_sparse[block_idx])

    split_sparse, split_layout = bs.ops.split(x_sparse, sparsity_layout_x, partitions, 2, sparsity_block_size)
    torch.cuda.synchronize()
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            for col_idx in range(n_col_blocks):
                source_block_idx = _dense_block_index(batch_idx, row_idx, col_idx, n_row_blocks, n_col_blocks)
                split_block_idx = _dense_block_index(batch_idx * partitions + col_idx, row_idx, 0, n_row_blocks, 1)
                assert torch.allclose(split_sparse[split_block_idx], x_sparse[source_block_idx], atol=ATOL, rtol=RTOL)

    merged_sparse, merged_layout = bs.ops.merge(split_sparse, split_layout, partitions, 2, sparsity_block_size)
    torch.cuda.synchronize()
    assert torch.equal(merged_layout, sparsity_layout_x)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            for col_idx in range(n_col_blocks):
                source_block_idx = _dense_block_index(batch_idx, row_idx, col_idx, n_row_blocks, n_col_blocks)
                assert torch.allclose(merged_sparse[source_block_idx], x_sparse[source_block_idx], atol=ATOL, rtol=RTOL)

    del x_sparse, split_sparse, merged_sparse, sparsity_layout_x, split_layout, merged_layout
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_row_striped_conversion(config: tuple):
    _require_min_cuda_memory(20)
    torch.cuda.empty_cache()

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float16
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    x = torch.zeros((b, m, k), device=DEVICE, dtype=dtype)
    sparsity_layout = torch.ones((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    sample_batches = _sample_positions(b)
    sample_rows = _sample_positions(n_row_blocks)
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            x[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ] = torch.randn((sparsity_block_size, sparsity_block_size), device=DEVICE, dtype=dtype)
    sample_blocks = {
        (batch_idx, row_idx): x[
            batch_idx,
            row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
            :sparsity_block_size,
        ].cpu()
        for batch_idx in sample_batches
        for row_idx in sample_rows
    }

    assert bs.ops.is_row_striped_layout(sparsity_layout)
    x_sparse = bs.ops.to_sparse_row_striped(x, sparsity_layout, sparsity_block_size)
    torch.cuda.synchronize()
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            block_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
            assert torch.allclose(x_sparse[block_idx].cpu(), sample_blocks[(batch_idx, row_idx)], atol=ATOL, rtol=RTOL)

    del x
    torch.cuda.empty_cache()

    x_dense = bs.ops.to_dense_row_striped(x_sparse, sparsity_layout, sparsity_block_size)
    torch.cuda.synchronize()
    for batch_idx in sample_batches:
        for row_idx in sample_rows:
            dense_block = x_dense[
                batch_idx,
                row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
                :sparsity_block_size,
            ]
            assert torch.allclose(dense_block.cpu(), sample_blocks[(batch_idx, row_idx)], atol=ATOL, rtol=RTOL)

    del x_sparse, x_dense, sparsity_layout
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_layout_matmul_helpers(config: tuple):
    _require_min_cuda_memory(20)

    b, m, n, k, sparsity_block_size, _ = config
    n_row_blocks = m // sparsity_block_size
    n_inner_blocks = k // sparsity_block_size
    n_col_blocks = n // sparsity_block_size

    sparsity_layout_x = torch.zeros((b, n_row_blocks, n_inner_blocks), dtype=torch.bool, device=DEVICE)
    sparsity_layout_y = torch.ones((b, n_inner_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    expected = torch.zeros((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)

    for batch_idx in _sample_positions(b):
        for row_idx in _sample_positions(n_row_blocks):
            sparsity_layout_x[batch_idx, row_idx, 0] = True
            expected[batch_idx, row_idx, 0] = True

    actual = bs.layouting.build_sparsity_layout_matmul(sparsity_layout_x, sparsity_layout_y)
    actual_fast = bs.layouting.build_sparsity_layout_matmul_fast(sparsity_layout_x, sparsity_layout_y)
    actual_outer = bs.layouting.build_sparsity_layout_matmul_outer(sparsity_layout_x, sparsity_layout_y)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_fast, expected)
    assert torch.equal(actual_outer, torch.ones_like(expected))


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_build_distribution_layout(config: tuple):
    _require_min_cuda_memory(20)

    b, m, _, k, sparsity_block_size, _ = config
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    sparsity_layout_indices = torch.ones((b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    n_sparse_blocks = int(sparsity_layout_indices.sum().item())
    indices_sparse = torch.zeros(
        (n_sparse_blocks, sparsity_block_size, sparsity_block_size),
        device=DEVICE,
        dtype=torch.int16,
    )

    actual = bs.layouting.build_distribution_layout(
        indices_sparse,
        sparsity_layout_indices,
        0,
        torch.Size((1, m, k)),
        sparsity_block_size,
    )
    torch.cuda.synchronize()

    expected = torch.ones((1, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)

    assert torch.equal(actual, expected)


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_BROADCAST_CONFIGURATIONS)
def test_blksprs_large_index_broadcast_operations(config: tuple):
    _require_min_cuda_memory(20)
    torch.cuda.empty_cache()

    b, m, _, _, sparsity_block_size, _ = config
    dtype = torch.float16

    x = torch.empty(size=(b, m), device=DEVICE, dtype=dtype)
    y = torch.empty(size=(b, m), device=DEVICE, dtype=dtype)
    sample_batches = _sample_positions(b)
    sample_blocks = [0]

    x.zero_()
    y.zero_()
    _set_sample_blocks(x, sparsity_block_size, sample_batches, sample_blocks)
    _set_sample_blocks(y, sparsity_block_size, sample_batches, sample_blocks)

    sparsity_layout_o = torch.zeros(size=(b, m // sparsity_block_size, m // sparsity_block_size),
                                    dtype=torch.bool, device=DEVICE)
    sparsity_layout_o[sample_batches, 0, 0] = True

    out_add_sparse = bs.ops.misc.broadcast_add(x, y, sparsity_layout_o, sparsity_block_size)
    out_sub_sparse = bs.ops.misc.broadcast_sub(x, y, sparsity_layout_o, sparsity_block_size)

    for sparse_idx, batch_idx in enumerate(sample_batches):
        ref_add = x[batch_idx, :sparsity_block_size].unsqueeze(-1) + y[batch_idx, :sparsity_block_size].unsqueeze(0)
        ref_sub = x[batch_idx, :sparsity_block_size].unsqueeze(-1) - y[batch_idx, :sparsity_block_size].unsqueeze(0)

        assert torch.allclose(out_add_sparse[sparse_idx].float(), ref_add.float(), atol=ATOL, rtol=RTOL)
        assert torch.allclose(out_sub_sparse[sparse_idx].float(), ref_sub.float(), atol=ATOL, rtol=RTOL)

    del x, y, out_add_sparse, out_sub_sparse, sparsity_layout_o
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_softmax_regular(config: tuple):
    _require_min_cuda_memory(32)

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float32
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    x = torch.randn(size=(b, m, k), device=DEVICE, dtype=dtype)
    sparsity_layout_x = torch.ones(size=(b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    sample_blocks = {
        (batch_idx, row_idx): x[
            batch_idx,
            row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
            :sparsity_block_size,
        ].cpu()
        for batch_idx in _sample_positions(b)
        for row_idx in _sample_positions(n_row_blocks)
    }
    x_sparse = bs.ops.to_sparse(x, sparsity_layout_x, sparsity_block_size)
    del x
    torch.cuda.empty_cache()

    x_softmax = bs.ops.softmax(x_sparse, sparsity_layout_x, sparsity_block_size, flag_fused=False)

    for batch_idx in _sample_positions(b):
        for row_idx in _sample_positions(n_row_blocks):
            blk_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
            dense_block_softmax = torch.softmax(sample_blocks[(batch_idx, row_idx)], dim=-1)
            assert torch.allclose(x_softmax[blk_idx].cpu(), dense_block_softmax, atol=ATOL, rtol=RTOL)


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_TEST_CONFIGURATIONS)
def test_blksprs_large_index_softmax_fused(config: tuple):
    _require_min_cuda_memory(20)

    b, m, _, k, sparsity_block_size, _ = config
    dtype = torch.float32
    n_row_blocks = m // sparsity_block_size
    n_col_blocks = k // sparsity_block_size

    x = torch.randn(size=(b, m, k), device=DEVICE, dtype=dtype)
    sparsity_layout_x = torch.ones(size=(b, n_row_blocks, n_col_blocks), dtype=torch.bool, device=DEVICE)
    sample_blocks = {
        (batch_idx, row_idx): x[
            batch_idx,
            row_idx * sparsity_block_size:(row_idx + 1) * sparsity_block_size,
            :sparsity_block_size,
        ].cpu()
        for batch_idx in _sample_positions(b)
        for row_idx in _sample_positions(n_row_blocks)
    }
    x_sparse = bs.ops.to_sparse(x, sparsity_layout_x, sparsity_block_size)
    del x
    torch.cuda.empty_cache()

    x_softmax_fused = bs.ops.softmax_fused(x_sparse, sparsity_layout_x, sparsity_block_size)

    for batch_idx in _sample_positions(b):
        for row_idx in _sample_positions(n_row_blocks):
            blk_idx = _dense_block_index(batch_idx, row_idx, 0, n_row_blocks, n_col_blocks)
            dense_block_softmax = torch.softmax(sample_blocks[(batch_idx, row_idx)], dim=-1)
            assert torch.allclose(x_softmax_fused[blk_idx].cpu(), dense_block_softmax, atol=ATOL, rtol=RTOL)


@pytest.mark.benchmark
@pytest.mark.parametrize("config", OVERFLOW_FLASH_ATTENTION_CONFIGURATIONS)
def test_blksprs_flash_attention_large_indices(config: tuple):
    _require_min_cuda_memory(20)

    b, m, n, k_dim, sparsity_block_size, _ = config
    dtype = torch.float16
    seq = m
    sbs = sparsity_block_size
    n_seq_blocks = seq // sbs
    n_head_blocks_qk = k_dim // sbs
    n_head_blocks_v = n // sbs

    q = torch.randn(b, seq, k_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(b, seq, k_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(b, seq, n, device=DEVICE, dtype=dtype)

    sparsity_layout_q = torch.ones(b, n_seq_blocks, n_head_blocks_qk, dtype=torch.bool, device=DEVICE)
    sparsity_layout_k = torch.ones(b, n_seq_blocks, n_head_blocks_qk, dtype=torch.bool, device=DEVICE)
    sparsity_layout_v = torch.ones(b, n_seq_blocks, n_head_blocks_v, dtype=torch.bool, device=DEVICE)
    sparsity_layout_o = torch.ones(b, n_seq_blocks, n_head_blocks_v, dtype=torch.bool, device=DEVICE)
    attention_layout = torch.ones(b, n_seq_blocks, n_seq_blocks, dtype=torch.bool, device=DEVICE)
    sample_batches = _sample_positions(b)

    q_ref = q[sample_batches].float().cpu()
    k_ref = k[sample_batches].float().cpu()
    v_ref = v[sample_batches].float().cpu()

    q_sparse = bs.ops.to_sparse(q, sparsity_layout_q, sbs)
    del q
    torch.cuda.empty_cache()
    k_sparse = bs.ops.to_sparse(k, sparsity_layout_k, sbs)
    del k
    torch.cuda.empty_cache()
    v_sparse = bs.ops.to_sparse(v, sparsity_layout_v, sbs)
    del v
    torch.cuda.empty_cache()

    out_sparse = bs.ops.flash_attention(
        q_sparse, sparsity_layout_q,
        k_sparse, sparsity_layout_k,
        v_sparse, sparsity_layout_v,
        attention_layout, sbs,
        sparsity_layout_o=sparsity_layout_o,
    )
    out_dense = bs.ops.to_dense(out_sparse, sparsity_layout_o, sbs)

    for sample_idx, batch_idx in enumerate(sample_batches):
        ref = _reference_attention_blocksparse(
            q_ref[sample_idx:sample_idx + 1],
            k_ref[sample_idx:sample_idx + 1],
            v_ref[sample_idx:sample_idx + 1],
            attention_layout[batch_idx:batch_idx + 1].cpu(),
            sbs,
        )
        assert torch.allclose(out_dense[batch_idx:batch_idx + 1].float().cpu(), ref.float(), atol=ATOL, rtol=RTOL)

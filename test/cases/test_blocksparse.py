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

# Tolerances
ATOL = 3e-2
RTOL = 2e-2

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

            assert torch.allclose(blksprs_matmul_dense_out.to(stock_dtype), stock_matmul_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_matmul_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_matmul_out, target)
            blksprs_loss = blksprs_loss(blksprs_matmul_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)
            assert torch.allclose(y_blksprs.grad, y_stock.grad, atol=ATOL, rtol=RTOL)


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

                torch.allclose(blksprs_distribution_layout.to(torch.int), stock_distribution_layout.to(torch.int),
                               atol=ATOL, rtol=RTOL)


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


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
def test_subclass(config: list):
    b, m, n, k, sparsity_block_size, sparsity_percentage = config

    x_d = torch.randn(size=(b, m, k), device=DEVICE)
    sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
    x_bs = BlksprsTensor.wrap(_blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size))

    assert type(x_bs).__name__ == BlksprsTensor.__name__


def test_version():
    assert bs.__version__ == _get_version()


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


# Flash Attention Tests

# Fixed head parameters for flash attention tests
FLASH_ATTENTION_N_HEADS = 2
FLASH_ATTENTION_HEAD_DIM = 32
# Maximum sequence length for flash attention tests — the reference attention materialises the
# full [n_batches, seq, seq] score matrix, so very large configs are too slow and memory-intensive
# for a unit-test run. Configs beyond this limit are skipped.
FLASH_ATTENTION_MAX_SEQ = 512


def _get_flash_attention_layout(n_batches: int, n_seq_q: int, n_seq_k: int,
                                 sparsity_pct: float) -> Tensor:
    """Generate a random attention sparsity layout."""
    attention_layout = torch.ones(n_batches, n_seq_q, n_seq_k,
                                  dtype=torch.bool, device=DEVICE)

    num_zero_elements = int(n_seq_q * n_seq_k * sparsity_pct)
    for b in range(n_batches):
        indices = torch.randperm(n_seq_q * n_seq_k, device=DEVICE)[:num_zero_elements]
        attention_layout[b, indices // n_seq_k, indices % n_seq_k] = False

    return attention_layout


def _reference_attention(q: Tensor, k: Tensor, v: Tensor,
                         attention_layout: Tensor, block_size: int,
                         attention_mask: Tensor = None,
                         attention_bias: Tensor = None,
                         scale: float = None) -> Tensor:
    """Compute reference (non-flash) attention using standard PyTorch ops.

    This is a straightforward matmul-softmax-matmul implementation that serves as the ground-truth
    for verifying the flash attention kernel.  It supports the same attention_mask and
    attention_bias parameters as the flash attention op.
    """
    batch, seq_q, n_heads, head_dim = q.shape
    _, seq_k, _, _ = k.shape

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)

    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Apply block sparsity mask
    attn_layout_expanded = attention_layout.reshape(batch, n_heads,
                                                     seq_q // block_size,
                                                     seq_k // block_size)

    for b in range(batch):
        for h in range(n_heads):
            for i in range(seq_q // block_size):
                for j in range(seq_k // block_size):
                    if not attn_layout_expanded[b, h, i, j]:
                        attn_scores[b, h,
                                    i * block_size:(i + 1) * block_size,
                                    j * block_size:(j + 1) * block_size] = float("-inf")

    # Apply attention mask if provided
    if attention_mask is not None:
        attention_mask_expanded = attention_mask.reshape(batch, n_heads, seq_q, seq_k)
        attn_scores = attn_scores.masked_fill(attention_mask_expanded, float("-inf"))

    # Apply attention bias if provided
    if attention_bias is not None:
        attention_bias_expanded = attention_bias.reshape(batch, n_heads, seq_q, seq_k)
        attn_scores = attn_scores + attention_bias_expanded

    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

    out = torch.matmul(attn_probs, v_t)
    return out.permute(0, 2, 1, 3)


@pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("use_amp", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_blksprs_flash_attention(config: tuple, use_amp: bool, use_mask: bool, use_bias: bool):
    b, m, n, k, sparsity_block_size, sparsity_percentage = config

    # Use m as seq length, fixed n_heads and head_dim
    n_heads = FLASH_ATTENTION_N_HEADS
    head_dim = FLASH_ATTENTION_HEAD_DIM
    seq = m

    # Skip configs that cannot be tested
    if seq > FLASH_ATTENTION_MAX_SEQ:
        pytest.skip("Sequence too long for reference attention (quadratic memory)")
    if seq < sparsity_block_size:
        pytest.skip("Sequence shorter than sparsity block size")

    batch = b
    n_batches = batch * n_heads
    n_seq_blocks = seq // sparsity_block_size

    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k_tensor = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    # Use sparsity_percentage as the fraction of blocks to zero out in the attention layout
    attention_layout = _get_flash_attention_layout(n_batches, n_seq_blocks, n_seq_blocks,
                                                   sparsity_percentage)

    # Attention mask: random boolean mask.  We use a moderate masking rate (30%) so that entire
    # rows are unlikely to be fully masked — fully masked rows make the reference softmax backward
    # produce NaN which is not a real bug but makes numerical comparison impossible.
    attention_mask = None
    if use_mask:
        attention_mask = torch.rand(n_batches, seq, seq, device=DEVICE) > 0.7

    # Attention bias: random float bias (with gradient)
    bias_data = None
    bias_ref = None
    bias_blksprs = None
    if use_bias:
        bias_data = torch.randn(n_batches, seq, seq, device=DEVICE) * 0.1
        bias_ref = bias_data.clone().detach().requires_grad_(True)
        bias_blksprs = bias_data.clone().detach().requires_grad_(True)

    # Reference attention runs in float32 to avoid numerical issues from -inf in float16.
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k_tensor.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)

    ref_out = _reference_attention(q_ref, k_ref, v_ref, attention_layout,
                                   sparsity_block_size,
                                   attention_mask=attention_mask,
                                   attention_bias=bias_ref)

    # Block-sparse flash attention (with AMP if enabled)
    q_blksprs = q.clone().detach().requires_grad_(True)
    k_blksprs = k_tensor.clone().detach().requires_grad_(True)
    v_blksprs = v.clone().detach().requires_grad_(True)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        blksprs_out = bs.ops.flash_attention(
            q_blksprs, k_blksprs, v_blksprs, attention_layout, sparsity_block_size,
            attention_mask=attention_mask, attention_bias=bias_blksprs
        )

    # Forward comparison
    assert torch.allclose(blksprs_out.to(ref_out.dtype), ref_out,
                          atol=ATOL, rtol=RTOL), "Forward output mismatch"

    # Backward comparison
    target = torch.randn_like(ref_out)
    ref_loss = torch.nn.L1Loss()(ref_out, target)
    blksprs_loss = torch.nn.L1Loss()(blksprs_out.to(ref_out.dtype), target)

    ref_loss.backward()
    blksprs_loss.backward()

    # Gradient comparisons — use nan_to_num because when an entire row is fully masked (all -inf),
    # the reference softmax backward produces NaN (mathematically undefined), while the flash
    # attention kernel correctly outputs zero.  Both behaviours are valid for undefined positions.
    assert torch.allclose(torch.nan_to_num(q_blksprs.grad.to(q_ref.grad.dtype)),
                          torch.nan_to_num(q_ref.grad), atol=ATOL, rtol=RTOL), "dQ mismatch"
    assert torch.allclose(torch.nan_to_num(k_blksprs.grad.to(k_ref.grad.dtype)),
                          torch.nan_to_num(k_ref.grad), atol=ATOL, rtol=RTOL), "dK mismatch"
    assert torch.allclose(torch.nan_to_num(v_blksprs.grad.to(v_ref.grad.dtype)),
                          torch.nan_to_num(v_ref.grad), atol=ATOL, rtol=RTOL), "dV mismatch"

    if use_bias:
        assert bias_blksprs.grad is not None, "Bias gradient should not be None"
        assert torch.allclose(torch.nan_to_num(bias_blksprs.grad.to(bias_ref.grad.dtype)),
                              torch.nan_to_num(bias_ref.grad),
                              atol=ATOL, rtol=RTOL), "dBias mismatch"



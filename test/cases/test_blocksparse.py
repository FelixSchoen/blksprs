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
    # (b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage)
    # All same
    (2, 16, 16, 16, 16, 16, 0.75),
    (2, 32, 32, 32, 32, 32, 0.75),
    (2, 64, 64, 64, 64, 64, 0.75),
    # Same dimensions, sparsity_block_size, and triton_block_size
    (2, 64, 64, 64, 16, 16, 0.75),
    (2, 64, 64, 64, 32, 32, 0.75),
    (2, 128, 128, 128, 16, 16, 0.75),
    (2, 128, 128, 128, 32, 32, 0.75),
    (2, 128, 128, 128, 64, 64, 0.75),
    (2, 2048, 2048, 128, 64, 64, 0.75),
    # Same dimensions
    (2, 64, 64, 64, 32, 16, 0.75),
    (2, 128, 128, 128, 32, 16, 0.75),
    (2, 128, 128, 128, 64, 16, 0.75),
    (2, 128, 128, 128, 64, 32, 0.75),
    # All different
    (16, 64, 32, 64, 32, 16, 0.75),
    (16, 32, 64, 64, 32, 16, 0.75),
    (8, 128, 64, 128, 64, 32, 0.75),
    (8, 64, 128, 128, 64, 32, 0.75),
    (4, 256, 128, 128, 64, 32, 0.75),
    (4, 128, 256, 128, 64, 32, 0.75),
    (2, 4096, 1024, 128, 64, 32, 0.75),
    (2, 1024, 4096, 128, 64, 32, 0.75),
    # Different sparsity
    (2, 128, 128, 128, 64, 32, 0.5),
    (2, 128, 128, 128, 64, 32, 0.25),
    (2, 128, 128, 128, 64, 32, 0.1),
    (2, 256, 128, 64, 32, 16, 0.1),
    (2, 128, 256, 64, 32, 16, 0.1),
    (2, 128, 256, 64, 32, 16, 0.1),
    (2, 128, 128, 128, 16, 8, 0.015625),
    # Empty and single block
    (2, 64, 64, 64, 32, 16, 0),
    (1, 64, 64, 64, 32, 16, 0.25),
]

TEST_CONFIGURATIONS_FAST = [
    # (b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage)
    # All same
    (2, 16, 16, 16, 16, 16, 0.75),
    (2, 32, 32, 32, 32, 32, 0.75),
    (2, 64, 64, 64, 64, 64, 0.75),
    # Same dimensions, sparsity_block_size, and triton_block_size
    (2, 64, 64, 64, 16, 16, 0.75),
    (2, 64, 64, 64, 32, 32, 0.75),
    # Same dimensions
    (2, 64, 64, 64, 32, 16, 0.75),
    # All different
    (16, 64, 32, 64, 32, 16, 0.75),
    (16, 32, 64, 64, 32, 16, 0.75),
    # Different sparsity
    (2, 64, 64, 64, 32, 16, 0.5),
    # Empty
    (2, 64, 64, 64, 32, 16, 0),
]

# Settings
BENCHMARK = True  # Whether to run benchmark
BENCHMARK_MATRIX_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Dimensions to benchmark
BENCHMARK_SPARSITY_BLOCK_SIZES = [32, 32, 64, 64, 128, 128, 128, 128]
BENCHMARK_TRITON_BLOCK_SIZES = [16, 16, 32, 32, 64, 64, 64, 64]

# Tolerances
ATOL = 2e-2
RTOL = 1e-2
SEED = 3654566474


@pytest.fixture(scope="session", autouse=True)
def setup():
    global SEED
    use_random_seed = False

    if use_random_seed:
        seed = random.randint(0, 2 ** 32 - 1)
        SEED = seed
    else:
        seed = SEED
        print("NOT USING RANDOM SEED")

    print("Random Seed:", seed)
    torch.manual_seed(seed)
    torch.set_printoptions(edgeitems=64, linewidth=10000)
    override_pytorch_repr()


def override_pytorch_repr():
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self, *args, **kwargs: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"


# Ops


def test_blksprs_to_sparse():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x_s in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_to_sparse_out = slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)

            blksprs_to_sparse_out = bs.ops.to_sparse(x_blksprs, sparsity_layout_x_s, sparsity_block_size)

            assert torch.allclose(blksprs_to_sparse_out, stock_to_sparse_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_to_sparse_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_to_sparse_out, target)
            blksprs_loss = blksprs_loss(blksprs_to_sparse_out, target)

            blksprs_loss.backward()

            if sparsity_percentage > 0:
                stock_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_to_dense():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x_s in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_to_sparse_out = slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)
            stock_to_dense_out = slow_to_dense(stock_to_sparse_out, sparsity_layout_x_s,
                                               sparsity_block_size)

            blksprs_to_sparse_out = bs.ops.to_sparse(x_blksprs, sparsity_layout_x_s, sparsity_block_size)
            blksprs_to_dense_out = bs.ops.to_dense(blksprs_to_sparse_out, sparsity_layout_x_s, sparsity_block_size)

            assert torch.allclose(blksprs_to_dense_out, stock_to_dense_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_to_dense_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_to_dense_out, target)
            blksprs_loss = blksprs_loss(blksprs_to_dense_out, target)

            blksprs_loss.backward()

            if sparsity_percentage > 0:
                stock_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_transpose():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_transpose_out = x_stock.transpose(1, 2)
            blksprs_transpose_out, blksprs_sparsity_layout_t = bs.ops.transpose(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size)
            blksprs_transpose_dense_out = bs.ops.to_dense(blksprs_transpose_out, blksprs_sparsity_layout_t,
                                                          sparsity_block_size)

            assert torch.allclose(blksprs_transpose_dense_out, stock_transpose_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_transpose_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_transpose_out, target)
            blksprs_loss = blksprs_loss(blksprs_transpose_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_gather():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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

                stock_gather_out = _blocksparse_roundtrip(torch.gather(x_stock, dim=dim, index=i_stock.to(torch.int64)),
                                                          sparsity_layout_i, sparsity_block_size)
                blksprs_gather_out = bs.ops.gather(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x,
                    dim,
                    bs.ops.to_sparse(i_blksprs, sparsity_layout_i, sparsity_block_size),
                    sparsity_layout_i,
                    sparsity_block_size)
                blksprs_gather_dense_out = bs.ops.to_dense(blksprs_gather_out, sparsity_layout_i,
                                                           sparsity_block_size)

                assert torch.allclose(blksprs_gather_dense_out.to(torch.float), stock_gather_out.to(torch.float),
                                      atol=ATOL, rtol=RTOL)

                pass

                target = torch.randn_like(stock_gather_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_gather_out, target)
                blksprs_loss = blksprs_loss(blksprs_gather_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad.to(torch.float), x_stock.grad.to(torch.float), atol=ATOL,
                                      rtol=RTOL)


def test_blksprs_scatter():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
                (x_bs, sparsity_layout_x_bs, i_bs, sparsity_layout_x_bs, sparsity_layout_o_bs),
            ]:
                x_stock = x.clone().requires_grad_(True)
                i_stock = i.clone()
                x_blksprs = x.clone().requires_grad_(True)
                i_blksprs = i.clone()

                stock_out_buffer = torch.zeros(size=(b * 2, m * 2, k * 2), device=DEVICE)
                stock_scatter_out = _blocksparse_roundtrip(
                    stock_out_buffer.scatter_reduce(dim=dim, index=i_stock.to(torch.int64), src=x_stock, reduce="sum"),
                    sparsity_layout_o, sparsity_block_size)

                blksprs_scatter_out = bs.ops.scatter_reduce(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x,
                    dim,
                    bs.ops.to_sparse(i_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_o,
                    sparsity_block_size,
                    reduce_op="sum")
                blksprs_scatter_dense_out = bs.ops.to_dense(blksprs_scatter_out, sparsity_layout_o, sparsity_block_size)

                assert torch.allclose(blksprs_scatter_dense_out, stock_scatter_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_scatter_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_scatter_out, target)
                blksprs_loss = blksprs_loss(blksprs_scatter_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_matmul():
    for b, m, n, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)
        y_d = torch.randn(size=(b, n, k), device=DEVICE).transpose(-1, -2).contiguous()
        sparsity_layout_y_d = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)
        sparsity_layout_y_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
        y_bs = _blocksparse_roundtrip(y_d, sparsity_layout_y_bs, sparsity_block_size)

        sparsity_layout_o_d = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size), device=DEVICE)
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
            blksprs_matmul_out = bs.ops.matmul(bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                                               sparsity_layout_x,
                                               bs.ops.to_sparse(y_blksprs, sparsity_layout_y, sparsity_block_size),
                                               sparsity_layout_y,
                                               sparsity_layout_o, sparsity_block_size)
            blksprs_matmul_dense_out = bs.ops.to_dense(blksprs_matmul_out, sparsity_layout_o, sparsity_block_size)

            assert torch.allclose(blksprs_matmul_dense_out, stock_matmul_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_matmul_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_matmul_out, target)
            blksprs_loss = blksprs_loss(blksprs_matmul_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)
            assert torch.allclose(y_blksprs.grad, y_stock.grad, atol=ATOL, rtol=RTOL)


def test_repeat():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
                blksprs_repeat_out, sparsity_layout_output = bs.ops.repeat(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, repeats,
                    sparsity_block_size, sparsity_layout_o_bs)
                blksprs_repeat_dense_out = bs.ops.to_dense(blksprs_repeat_out, sparsity_layout_output,
                                                           sparsity_block_size)

                assert torch.allclose(blksprs_repeat_dense_out, stock_repeat_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_repeat_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_repeat_out, target)
                blksprs_loss = blksprs_loss(blksprs_repeat_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_repeat_interleave():
    for b, m, n, _, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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

                blksprs_repeat_interleave_out, sparsity_layout_output = bs.ops.repeat_interleave(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, num_repeats,
                    sparsity_block_size, sparsity_layout_o_bs)
                blksprs_repeat_interleave_dense_out = bs.ops.to_dense(blksprs_repeat_interleave_out,
                                                                      sparsity_layout_output,
                                                                      sparsity_block_size)

                assert torch.allclose(blksprs_repeat_interleave_dense_out, stock_repeat_interleave_out, atol=ATOL,
                                      rtol=RTOL)

                target = torch.randn_like(stock_repeat_interleave_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_repeat_interleave_out, target)
                blksprs_loss = blksprs_loss(blksprs_repeat_interleave_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_softmax():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size, fill_value=float('-1e12'))

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_softmax_out = torch.softmax(x_stock, dim=-1)
            blksprs_softmax_out = bs.ops.softmax(bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                                                 sparsity_layout_x, sparsity_block_size)
            blksprs_softmax_dense_out = bs.ops.to_dense(blksprs_softmax_out, sparsity_layout_x, sparsity_block_size)

            assert torch.allclose(blksprs_softmax_dense_out, stock_softmax_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_softmax_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_softmax_out, target)
            blksprs_loss = blksprs_loss(blksprs_softmax_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_split():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
                blksprs_split_out, sparsity_layout_output = bs.ops.split(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, num_partitions, -1,
                    sparsity_block_size)
                blksprs_split_dense_out = bs.ops.to_dense(blksprs_split_out, sparsity_layout_output,
                                                          sparsity_block_size)

                assert torch.allclose(blksprs_split_dense_out, stock_split_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_split_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_split_out, target)
                blksprs_loss = blksprs_loss(blksprs_split_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_merge():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
                stock_merge_out = (stock_split_out.reshape(stock_split_out.size(0) // num_partitions, num_partitions,
                                                           stock_split_out.size(1), stock_split_out.size(2))
                                   .permute(0, 2, 1, 3).reshape(stock_split_out.size(0) // num_partitions,
                                                                stock_split_out.size(1),
                                                                stock_split_out.size(2) * num_partitions))
                blksprs_split_out, sparsity_layout_split = bs.ops.split(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, num_partitions, -1,
                    sparsity_block_size)
                blksprs_merge_out, sparsity_layout_merge = bs.ops.merge(blksprs_split_out, sparsity_layout_split,
                                                                        num_partitions, -1, sparsity_block_size)
                blksprs_merge_dense_out = bs.ops.to_dense(blksprs_merge_out, sparsity_layout_merge, sparsity_block_size)

                assert torch.allclose(stock_merge_out, x_stock, atol=ATOL, rtol=RTOL)
                assert torch.allclose(blksprs_merge_dense_out, stock_merge_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_merge_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_merge_out, target)
                blksprs_loss = blksprs_loss(blksprs_merge_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_row_wise_sum():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_sum_out = torch.sum(x_stock, dim=-1)
            blksprs_row_wise_sum_out, sparsity_layout_output = bs.ops.misc.row_wise_sum(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x,
                sparsity_block_size)
            blksprs_row_wise_sum_dense_out = bs.ops.to_dense(blksprs_row_wise_sum_out, sparsity_layout_output,
                                                             sparsity_block_size)

            blksprs_row_wise_sum_out_slice = blksprs_row_wise_sum_dense_out[..., 0]

            assert torch.allclose(blksprs_row_wise_sum_out_slice, stock_sum_out, atol=ATOL, rtol=RTOL)


def test_blksprs_row_wise_max():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                      fill_value=float("-inf"))

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs), (x_d, sparsity_layout_x_d),
                                     (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_max_out = torch.max(x_stock, dim=-1).values
            blksprs_row_wise_max_out, sparsity_layout_output = bs.ops.misc.row_wise_max(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x,
                sparsity_block_size)
            blksprs_row_wise_max_dense_out = bs.ops.to_dense(blksprs_row_wise_max_out, sparsity_layout_output,
                                                             sparsity_block_size, fill_value=float("-inf"))

            blksprs_row_wise_max_out_slice = blksprs_row_wise_max_dense_out[..., 0]

            assert torch.allclose(blksprs_row_wise_max_out_slice, stock_max_out, atol=ATOL, rtol=RTOL)


def test_blksprs_row_wise_add():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                      fill_value=float("-inf"))

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs), (x_d, sparsity_layout_x_d),
                                     (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_max_out = (torch.max(_blocksparse_roundtrip(x_stock, sparsity_layout_x,
                                                              sparsity_block_size),
                                       dim=-1).values).unsqueeze(-1)
            stock_rwa_out = _blocksparse_roundtrip(x_stock + stock_max_out, sparsity_layout_x,
                                                   sparsity_block_size)

            blksprs_row_wise_max_out, sparsity_layout_output = bs.ops.misc.row_wise_max(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x,
                sparsity_block_size)
            blksprs_row_wise_add_out = bs.ops.misc.row_wise_add(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, blksprs_row_wise_max_out,
                sparsity_block_size)
            blksprs_row_wise_add_dense_out = bs.ops.to_dense(blksprs_row_wise_add_out, sparsity_layout_x,
                                                             sparsity_block_size)

            assert torch.allclose(blksprs_row_wise_add_dense_out, stock_rwa_out, atol=ATOL, rtol=RTOL)


def test_blksprs_adapt_layout():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)

        for sparsity_block_size_from, sparsity_block_size_to in [(sparsity_block_size, sparsity_block_size),
                                                                 (sparsity_block_size, sparsity_block_size // 4),
                                                                 (sparsity_block_size, sparsity_block_size // 2),
                                                                 (sparsity_block_size, sparsity_block_size),
                                                                 (sparsity_block_size // 4, sparsity_block_size),
                                                                 (sparsity_block_size // 2, sparsity_block_size)]:
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
                (x_bs_from, sparsity_layout_x_bs_from, sparsity_layout_x_bs_to_less, True),
            ]:
                x_from_stock = x_from.clone().requires_grad_(True)
                x_from_blksprs = x_from.clone().requires_grad_(True)

                stock_adapt_layout_out = _blocksparse_roundtrip(
                    _blocksparse_roundtrip(x_from_stock, sparsity_layout_x_from,
                                           sparsity_block_size_from),
                    sparsity_layout_x_to, sparsity_block_size_to)

                blksprs_adapt_layout_out, _ = bs.ops.adapt_layout(
                    bs.ops.to_sparse(x_from_blksprs, sparsity_layout_x_from, sparsity_block_size_from),
                    sparsity_layout_x_from, sparsity_block_size_from,
                    sparsity_block_size_to,
                    sparsity_layout_x_to if use_output_layout else None)
                blksprs_adapt_layout_dense_out = bs.ops.to_dense(blksprs_adapt_layout_out, sparsity_layout_x_to,
                                                                 sparsity_block_size_to)

                assert torch.allclose(blksprs_adapt_layout_dense_out, stock_adapt_layout_out, atol=ATOL, rtol=RTOL)

                target = torch.randn_like(stock_adapt_layout_out)
                stock_loss = torch.nn.L1Loss()
                blksprs_loss = torch.nn.L1Loss()
                stock_loss = stock_loss(stock_adapt_layout_out, target)
                blksprs_loss = blksprs_loss(blksprs_adapt_layout_dense_out, target)

                stock_loss.backward()
                blksprs_loss.backward()

                assert torch.allclose(x_from_blksprs.grad, x_from_stock.grad, atol=ATOL, rtol=RTOL)


# Layouting


def test_build_sparsity_layout():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size)

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs)]:
            x_blksprs = x.clone().requires_grad_(True)

            x_sparse = bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size)
            x_dense = bs.ops.to_dense(x_sparse, sparsity_layout_x, sparsity_block_size)

            blksprs_sparsity_layout = bs.layouting.build_sparsity_layout(x_dense, sparsity_block_size)

            assert torch.allclose(blksprs_sparsity_layout.to(torch.bool), sparsity_layout_x.to(torch.bool), atol=ATOL,
                                  rtol=RTOL)


def test_build_sparsity_layout_matmul():
    for b, m, n, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        sparsity_layout_y_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)

        sparsity_layout_matmul = bs.layouting.build_sparsity_layout_matmul(sparsity_layout_x_bs, sparsity_layout_y_bs)
        sparsity_layout_matmul_fast = bs.layouting.build_sparsity_layout_matmul_fast(sparsity_layout_x_bs,
                                                                                     sparsity_layout_y_bs)


def test_build_distribution_layout():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
                stock_out_buffer = torch.zeros(size=(b * 2, m * 2, k * 2), device=DEVICE)
                stock_scatter_out = _blocksparse_roundtrip(
                    stock_out_buffer.scatter_reduce(dim=dim, index=i.to(torch.int64), src=src, reduce="sum"),
                    sparsity_layout_tgt, sparsity_block_size)
                stock_distribution_layout = bs.layouting.build_sparsity_layout(stock_scatter_out, sparsity_block_size)

                blksprs_distribution_layout = bs.layouting.build_distribution_layout(
                    bs.ops.to_sparse(i, sparsity_layout_i, sparsity_block_size),
                    sparsity_layout_i, dim, tgt.size(), sparsity_block_size)

                torch.allclose(blksprs_distribution_layout.to(torch.int), stock_distribution_layout.to(torch.int),
                               atol=ATOL, rtol=RTOL)


# Processing

def test_apply_torch_linear():
    for b, m, n, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
                blksprs_linear_out, sparsity_layout_xl = bs.utils.apply_torch_linear(
                    bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                    sparsity_layout_x, sparsity_block_size, linear)
                blksprs_linear_dense_out = bs.ops.to_dense(blksprs_linear_out, sparsity_layout_xl, sparsity_block_size)

                assert torch.allclose(blksprs_linear_dense_out, stock_linear_out, atol=ATOL, rtol=RTOL)


def test_apply_torch_normalisation():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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
            blksprs_normalisation_out = bs.utils.apply_torch_normalisation(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size, normalisation)
            blksprs_normalisation_dense_out = bs.ops.to_dense(blksprs_normalisation_out, sparsity_layout_x,
                                                              sparsity_block_size)

            assert torch.allclose(blksprs_normalisation_dense_out, stock_normalisation_out, atol=ATOL, rtol=RTOL)


def test_apply_torch_dropout():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
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

            stock_normalisation_out = _blocksparse_roundtrip(dropout(x_stock), sparsity_layout_x,
                                                             sparsity_block_size)
            global SEED
            torch.manual_seed(SEED)
            blksprs_normalisation_out = bs.utils.apply_torch_normalisation(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size, dropout)
            blksprs_normalisation_dense_out = bs.ops.to_dense(blksprs_normalisation_out, sparsity_layout_x,
                                                              sparsity_block_size)

            assert torch.allclose(blksprs_normalisation_dense_out, stock_normalisation_out, atol=ATOL, rtol=RTOL)


# Misc

def test_broadcast_addition():
    for b, m, _, k, sparsity_block_size, _, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randint(high=m, size=(b, m), device=DEVICE)
        y_d = torch.randint(high=m, size=(b, m), device=DEVICE)

        sparsity_layout_o = torch.ones(size=(b, m // sparsity_block_size, m // sparsity_block_size), device=DEVICE)
        sparsity_layout_o_bs = _get_blocksparse_layout(b, m, m, sparsity_block_size, sparsity_percentage)

        for x, y, sparsity_layout_o in [(x_d, y_d, sparsity_layout_o), (x_d, y_d, sparsity_layout_o_bs)]:
            stock_broadcast_addition = _blocksparse_roundtrip(torch.add(x.unsqueeze(-1), y.unsqueeze(-2)),
                                                              sparsity_layout_o, sparsity_block_size).to(torch.float)
            blksprs_broadcast_addition_out = bs.ops.misc.broadcast_add(x, y, sparsity_layout_o,
                                                                       sparsity_block_size)
            blksprs_broadcast_addition_dense_out = bs.ops.to_dense(blksprs_broadcast_addition_out, sparsity_layout_o,
                                                                   sparsity_block_size)

            stock_broadcast_subtraction = _blocksparse_roundtrip(torch.sub(x.unsqueeze(-1), y.unsqueeze(-2)),
                                                                 sparsity_layout_o, sparsity_block_size).to(torch.float)
            blksprs_broadcast_subtraction = bs.ops.misc.broadcast_sub(x, y, sparsity_layout_o,
                                                                      sparsity_block_size)
            blksprs_broadcast_subtraction_dense_out = bs.ops.to_dense(blksprs_broadcast_subtraction, sparsity_layout_o,
                                                                      sparsity_block_size)

            assert torch.allclose(blksprs_broadcast_addition_dense_out.to(torch.float),
                                  stock_broadcast_addition.to(torch.float),
                                  atol=ATOL, rtol=RTOL)
            assert torch.allclose(blksprs_broadcast_subtraction_dense_out.to(torch.float),
                                  stock_broadcast_subtraction.to(torch.float),
                                  atol=ATOL, rtol=RTOL)


def test_subclass():
    for b, m, _, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = BlksprsTensor(_blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size, triton_block_size))

        assert type(x_bs).__name__ == BlksprsTensor.__name__


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


def _blocksparse_roundtrip(x, sparsity_layout, sparsity_block_size, triton_block_size=None, fill_value=0.0):
    return bs.ops.to_dense(bs.ops.to_sparse(x, sparsity_layout, sparsity_block_size),
                           sparsity_layout,
                           sparsity_block_size, fill_value=fill_value)


# Visualisation

def _visualise(*matrices, dim=0):
    vmin = np.inf
    vmax = -np.inf

    for matrix_tuple in matrices:
        vmin = min(vmin, torch.min(matrix_tuple[0]))
        vmax = max(vmax, torch.max(matrix_tuple[0]))

    for matrix_tuple in matrices:
        matrix_data = matrix_tuple[0]
        matrix_label = matrix_tuple[1]

        add_args = {}
        if len(matrix_tuple) > 2:
            add_args = matrix_tuple[2]

        output_path_base = BASE_PATH.joinpath("test", "output", "blksprs")
        output_path_base.mkdir(exist_ok=True)

        _visualise_matrix(matrix_data[dim], str(output_path_base.joinpath(matrix_label)), grid_size=1, vmin=vmin,
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

def slow_to_sparse(x, sparsity_layout, sparsity_block_size: int):
    num_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
    output = torch.zeros(size=(num_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)
    indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

    for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
        t_r = r * sparsity_block_size
        t_c = c * sparsity_block_size
        to_insert = x[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size]
        output[idx] = to_insert

    return output


def slow_to_dense(x, sparsity_layout, sparsity_block_size: int):
    output = torch.zeros(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                               sparsity_layout.size(2) * sparsity_block_size), device=x.device)
    indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

    for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
        t_r = r * sparsity_block_size
        t_c = c * sparsity_block_size
        to_insert = x[idx]
        output[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size] = to_insert

    return output


def slow_gather_mdi(src, idx_bat, idx_row, idx_col):
    output = torch.zeros(size=(idx_bat.size(0), idx_bat.size(1), idx_bat.size(2)), device=src.device)

    for b in range(idx_bat.size(0)):
        for k in range(idx_bat.size(1)):
            for n in range(idx_bat.size(2)):
                output[b, k, n] = src[idx_bat[b, k, n], k, idx_col[b, k, n]]

    return output


def slow_scatter_reduce_mdi(src, tgt_size, idx_bat, idx_row, idx_col):
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

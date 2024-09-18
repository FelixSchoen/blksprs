from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt

from blksprs.layouting.distribution_layout import create_distribution_layout
from blksprs.layouting.sparsity_layout import create_sparsity_layout
from blksprs.ops.conversion import to_dense, to_sparse
from blksprs.ops.distribution import scatter_reduce, gather
from blksprs.ops.exp import exp
from blksprs.ops.matmul import matmul
from blksprs.ops.row_wise_sum import row_wise_sum
from blksprs.ops.softmax import softmax
from blksprs.ops.transpose import transpose

# TODO Benchmarking
# TODO Validate sparsity_block_size power of 2
# TODO more test cases with sparsity

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
    # Empty
    (2, 64, 64, 64, 64, 64, 0),
]

# Settings
BENCHMARK = True  # Whether to run benchmark
BENCHMARK_MATRIX_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Dimensions to benchmark
BENCHMARK_SPARSITY_BLOCK_SIZES = [32, 32, 64, 64, 128, 128, 128, 128]
BENCHMARK_TRITON_BLOCK_SIZES = [16, 16, 32, 32, 64, 64, 64, 64]

# Tolerances
ATOL = 2e-2
RTOL = 1e-2


@pytest.fixture(scope="session", autouse=True)
def setup():
    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=64, linewidth=10000)
    override_pytorch_repr()


def override_pytorch_repr():
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"


# Ops

def test_blksprs_matmul_sss():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)
        y = torch.randn(size=(b, n, k), device=DEVICE).transpose(-1, -2).contiguous()
        sparsity_layout_y = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)
        sparsity_layout_y_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
        y_bs = _blocksparse_roundtrip(y, sparsity_layout_y_bs, sparsity_block_size, triton_block_size)

        sparsity_layout_o = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        for x, sparsity_layout_x, y, sparsity_layout_y in [(x, sparsity_layout_x, y, sparsity_layout_y),
                                                           (x_bs, sparsity_layout_x_bs, y_bs, sparsity_layout_y_bs)]:
            x_stock = x.clone().requires_grad_(True)
            y_stock = y.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)
            y_blksprs = y.clone().requires_grad_(True)

            stock_matmul_out = torch.matmul(x_stock, y_stock)
            blksprs_matmul_out = matmul(to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                                        to_sparse(y_blksprs, sparsity_layout_y, sparsity_block_size),
                                        sparsity_layout_x, sparsity_layout_y, sparsity_layout_o,
                                        sparsity_block_size)
            blksprs_matmul_dense_out = to_dense(blksprs_matmul_out, sparsity_layout_o, sparsity_block_size)

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


def test_blksprs_softmax():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size,
                                      fill_value=float('-1e12'))

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_softmax_out = torch.softmax(x_stock, dim=-1)
            blksprs_softmax_out = softmax(to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                                          sparsity_layout_x, sparsity_block_size)
            blksprs_softmax_dense_out = to_dense(blksprs_softmax_out, sparsity_layout_x, sparsity_block_size)

            assert torch.allclose(blksprs_softmax_dense_out, stock_softmax_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_softmax_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_softmax_out, target)
            blksprs_loss = blksprs_loss(blksprs_softmax_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_transpose():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_transpose_out = x_stock.transpose(1, 2)
            blksprs_transpose_out, blksprs_sparsity_layout_t = transpose(
                to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size, triton_block_size)
            blksprs_transpose_dense_out = to_dense(blksprs_transpose_out, blksprs_sparsity_layout_t,
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


def test_blksprs_to_sparse():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x_s in [(x, sparsity_layout_x), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_to_sparse_out = slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)

            blksprs_to_sparse_out = to_sparse(x_blksprs, sparsity_layout_x_s, sparsity_block_size)

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
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x_s in [(x, sparsity_layout_x), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_to_sparse_out = slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)
            stock_to_dense_out = slow_to_dense(stock_to_sparse_out, sparsity_layout_x_s,
                                               sparsity_block_size)

            blksprs_to_sparse_out = to_sparse(x_blksprs, sparsity_layout_x_s, sparsity_block_size)
            blksprs_to_dense_out = to_dense(blksprs_to_sparse_out, sparsity_layout_x_s, sparsity_block_size)

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


def test_blksprs_row_wise_sum():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_sum_out = torch.sum(x_stock, dim=-1)
            blksprs_row_wise_sum_out, sparsity_layout_output = row_wise_sum(
                to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_layout_x, sparsity_block_size)
            blksprs_row_wise_sum_dense_out = to_dense(blksprs_row_wise_sum_out, sparsity_layout_output,
                                                      sparsity_block_size)

            blksprs_row_wise_sum_out_slice = blksprs_row_wise_sum_dense_out[..., 0]

            assert torch.allclose(blksprs_row_wise_sum_out_slice, stock_sum_out, atol=ATOL, rtol=RTOL)


def test_blksprs_exp():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_exp_out = _blocksparse_roundtrip(torch.exp(x_stock), sparsity_layout_x,
                                                   sparsity_block_size, triton_block_size)
            blksprs_exp_out = exp(to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size), sparsity_block_size)
            blksprs_exp_dense_out = to_dense(blksprs_exp_out, sparsity_layout_x, sparsity_block_size)

            assert torch.allclose(blksprs_exp_dense_out, stock_exp_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_exp_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_exp_out, target)
            blksprs_loss = blksprs_loss(blksprs_exp_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_gather():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        m = max(m, n)

        x = torch.randn(size=(b, k, m), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, k // sparsity_block_size, m // sparsity_block_size), device=DEVICE)

        i = (torch.randint(0, n, size=(k, n), dtype=torch.int, device=DEVICE).unsqueeze(0).expand(b, k, n).contiguous())
        sparsity_layout_i = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        sparsity_layout_i_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
        i_bs = _blocksparse_roundtrip(i, sparsity_layout_i_bs, sparsity_block_size, triton_block_size)

        sparsity_layout_x_bs = create_distribution_layout(
            to_sparse(i, sparsity_layout_i, sparsity_block_size, triton_block_size),
            sparsity_layout_i, x.size(), sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x, i, sparsity_layout_i in [(x, sparsity_layout_x, i, sparsity_layout_i),
                                                           (x, sparsity_layout_x_bs, i_bs, sparsity_layout_i_bs)]:
            x_stock = x.clone().requires_grad_(True)
            i_stock = i.clone()
            x_blksprs = x.clone().requires_grad_(True)
            i_blksprs = i.clone()

            stock_gather_out = _blocksparse_roundtrip(torch.gather(x_stock, dim=-1, index=i_stock.to(torch.int64)),
                                                      sparsity_layout_i, sparsity_block_size, triton_block_size)
            blksprs_gather_out = gather(to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size, triton_block_size),
                                        sparsity_layout_x,
                                        to_sparse(i_blksprs, sparsity_layout_i, sparsity_block_size, triton_block_size),
                                        sparsity_layout_i,
                                        sparsity_block_size, triton_block_size)
            blksprs_gather_dense_out = to_dense(blksprs_gather_out, sparsity_layout_i,
                                                sparsity_block_size, triton_block_size=triton_block_size)

            assert torch.allclose(blksprs_gather_dense_out, stock_gather_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_gather_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_gather_out, target)
            blksprs_loss = blksprs_loss(blksprs_gather_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_scatter():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        m = max(m, n)

        x = torch.randn(size=(b, k, n), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        i = (torch.randint(0, m, size=(k, n), dtype=torch.int, device=DEVICE).unsqueeze(0).expand(b, k, n).contiguous())

        sparsity_layout_o = torch.ones(size=(b, k // sparsity_block_size, m // sparsity_block_size), device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)
        i_bs = _blocksparse_roundtrip(i, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        sparsity_layout_o_bs = create_distribution_layout(
            to_sparse(i, sparsity_layout_x, sparsity_block_size, triton_block_size),
            sparsity_layout_x, (b, k, m), sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x, i, sparsity_layout_i, sparsity_layout_o in [
            (x, sparsity_layout_x, i, sparsity_layout_x, sparsity_layout_o),
            (x_bs, sparsity_layout_x_bs, i_bs, sparsity_layout_x_bs, sparsity_layout_o_bs),
        ]:
            x_stock = x.clone().requires_grad_(True)
            i_stock = i.clone()
            x_blksprs = x.clone().requires_grad_(True)
            i_blksprs = i.clone()

            stock_out_buffer = torch.zeros(size=(b, k, m), device=DEVICE)
            stock_scatter_out = _blocksparse_roundtrip(
                stock_out_buffer.scatter_reduce(dim=-1, index=i_stock.to(torch.int64), src=x_stock, reduce="sum"),
                sparsity_layout_o, sparsity_block_size, triton_block_size)

            blksprs_scatter_out = scatter_reduce(
                to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size, triton_block_size),
                sparsity_layout_x,
                to_sparse(i_blksprs, sparsity_layout_x, sparsity_block_size, triton_block_size),
                sparsity_layout_o,
                sparsity_block_size,
                reduce_op="sum", triton_block_size=triton_block_size)
            blksprs_scatter_dense_out = to_dense(blksprs_scatter_out, sparsity_layout_o, sparsity_block_size,
                                                 triton_block_size=triton_block_size)

            assert torch.allclose(blksprs_scatter_dense_out, stock_scatter_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_scatter_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_scatter_out, target)
            blksprs_loss = blksprs_loss(blksprs_scatter_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


# Layouting

def test_create_sparsity_layout():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)

        for x, sparsity_layout_x in [(x_bs, sparsity_layout_x_bs)]:
            x_blksprs = x.clone().requires_grad_(True)

            x_sparse = to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size, triton_block_size=triton_block_size)
            x_dense = to_dense(x_sparse, sparsity_layout_x, sparsity_block_size, triton_block_size=triton_block_size)

            blksprs_sparsity_layout = create_sparsity_layout(x_dense, sparsity_block_size, triton_block_size)

            assert torch.allclose(blksprs_sparsity_layout, sparsity_layout_x.to(torch.int), atol=ATOL, rtol=RTOL)


def test_create_distribution_layout():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        m = max(m, n)

        src = torch.randn(size=(b, k, n), device=DEVICE)
        sparsity_layout_src = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        tgt = torch.randn(size=(b, k, m), device=DEVICE)
        sparsity_layout_tgt = torch.ones(size=(b, k // sparsity_block_size, m // sparsity_block_size), device=DEVICE)

        i = (torch.randint(0, m, size=(k, n), dtype=torch.int, device=DEVICE).unsqueeze(0).expand(b, k, n).contiguous())

        sparsity_layout_src_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
        src_bs = _blocksparse_roundtrip(src, sparsity_layout_src_bs, sparsity_block_size, triton_block_size)
        i_bs = _blocksparse_roundtrip(i, sparsity_layout_src_bs, sparsity_block_size, triton_block_size)

        for src, sparsity_layout_src, tgt, sparsity_layout_tgt, i, sparsity_layout_i in [
            (src, sparsity_layout_src, tgt, sparsity_layout_tgt, i, sparsity_layout_src),
            (src_bs, sparsity_layout_src_bs, tgt, sparsity_layout_tgt, i_bs, sparsity_layout_src_bs)]:
            stock_out_buffer = torch.zeros(size=(b, k, m), device=DEVICE)
            stock_scatter_out = _blocksparse_roundtrip(
                stock_out_buffer.scatter_reduce(dim=-1, index=i.to(torch.int64), src=src, reduce="sum"),
                sparsity_layout_tgt, sparsity_block_size, triton_block_size)
            stock_distribution_layout = create_sparsity_layout(stock_scatter_out, sparsity_block_size,
                                                               triton_block_size)

            blksprs_distribution_layout = create_distribution_layout(
                to_sparse(i, sparsity_layout_i, sparsity_block_size, triton_block_size),
                sparsity_layout_i, tgt.size(), sparsity_block_size, triton_block_size)

            torch.allclose(blksprs_distribution_layout, stock_distribution_layout, atol=ATOL, rtol=RTOL)


# Utility

def _get_blocksparse_layout(b, m, n, sparsity_block_size, sparsity_percentage):
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), device=DEVICE)

    num_zero_elements = int(m_s * n_s * (1 - sparsity_percentage))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = 0

    return sparsity_layout


def _blocksparse_roundtrip(x, sparsity_layout, sparsity_block_size, triton_block_size, fill_value=0.0):
    return to_dense(to_sparse(x, sparsity_layout, sparsity_block_size, triton_block_size), sparsity_layout,
                    sparsity_block_size, fill_value=fill_value, triton_block_size=triton_block_size)


# Visualisation

def _visualise(*matrices, dim=0):
    for matrix_tuple in matrices:
        matrix_data = matrix_tuple[0]
        matrix_label = matrix_tuple[1]

        output_path_base = BASE_PATH.joinpath("test", "output", "blksprs")
        output_path_base.mkdir(exist_ok=True)

        _visualise_matrix(matrix_data[dim], str(output_path_base.joinpath(matrix_label)), grid_size=1)


def _visualise_matrix(matrix: torch.Tensor, output_path: str, grid_size=16):
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

    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

    if output_path is not None:
        plt.savefig(f"{output_path}.svg", format="svg")


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

from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt

from blksprs.ops.exp import BlocksparseExp
from blksprs.ops.row_wise_sum import BlocksparseRowWiseSum
from blksprs.ops.tools import BlocksparseTools
from blksprs.ops.softmax import BlocksparseSoftmax
from blksprs.ops.transpose import BlocksparseTranspose
from blksprs.ops.conversion import BlocksparseToDense, BlocksparseToSparse
from blksprs.ops.matmul_sss import BlocksparseMatmulSSS

# Device setup
DEVICE = torch.device("cuda:0")

# Constants
B, M, N, K = 2, 64, 64, 64
SPARSITY_BLOCK_SIZE = 32
TRITON_BLOCK_SIZE = 16
SPARSITY_LAYOUT_FULL = torch.ones(size=(B, M // SPARSITY_BLOCK_SIZE, K // SPARSITY_BLOCK_SIZE), device=DEVICE)
BASE_PATH = Path(__file__).parent.parent.parent

TEST_CONFIGURATIONS = [
    (2, 64, 64, 64, 32, 16, 0.75),
    (2, 64, 64, 64, 64, 64, 0.75),
    (2, 64, 64, 32, 32, 16, 0.75)
]

# Settings
SPARSITY_PERCENTAGE = 0.75  # Percentage of non-sparse blocks
BENCHMARK = True  # Whether to run benchmark
BENCHMARK_MATRIX_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Dimensions to benchmark
BENCHMARK_SPARSITY_BLOCK_SIZES = [32, 32, 64, 64, 128, 128, 128, 128]
BENCHMARK_TRITON_BLOCK_SIZES = [16, 16, 32, 32, 64, 64, 64, 64]

# Tolerances
ATOL = 1e-2
RTOL = 1e-2


@pytest.fixture(scope="session", autouse=True)
def setup():
    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=64, linewidth=10000)
    override_pytorch_repr()


def override_pytorch_repr():
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"


# Tests

def test_blksprs_matmul_sss():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)
        y = torch.randn(size=(b, n, k), device=DEVICE).transpose(-1, -2).contiguous()
        sparsity_layout_y = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)
    
        x_s, sparsity_layout_x_s = _get_blocksparse_input(b, m, k, sparsity_block_size, sparsity_percentage)
        y_s, sparsity_layout_y_s = _get_blocksparse_input(b, k, n, sparsity_block_size, sparsity_percentage)

        sparsity_layout_o = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

        for x, sparsity_layout_x, y, sparsity_layout_y in [(x, sparsity_layout_x, y, sparsity_layout_y),
                                                           (x_s, sparsity_layout_x_s, y_s, sparsity_layout_y_s)]:
            x_stock = x.clone().requires_grad_(True)
            y_stock = y.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)
            y_blksprs = y.clone().requires_grad_(True)
    
            blksprs_matmul_sss = BlocksparseMatmulSSS(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_dense = BlocksparseToDense(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
    
            stock_matmul_out = torch.matmul(x_stock, y_stock)
            blksprs_matmul_out = blksprs_matmul_sss(blksprs_to_sparse(x_blksprs, sparsity_layout_x),
                                                    blksprs_to_sparse(y_blksprs, sparsity_layout_y),
                                                    sparsity_layout_x, sparsity_layout_y, sparsity_layout_o)
            blksprs_matmul_out_dense = blksprs_to_dense(blksprs_matmul_out, sparsity_layout_o)
    
            assert torch.allclose(blksprs_matmul_out_dense, stock_matmul_out, atol=ATOL, rtol=RTOL)
    
            target = torch.randn_like(stock_matmul_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_matmul_out, target)
            blksprs_loss = blksprs_loss(blksprs_matmul_out_dense, target)
    
            stock_loss.backward()
            blksprs_loss.backward()
    
            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)
            assert torch.allclose(y_blksprs.grad, y_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_softmax():
    TEST_CONFIGURATIONS = [(2, 64, 64, 64, 32, 32, 0.75)]

    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        x_s, sparsity_layout_x_s = _get_blocksparse_input(b, m, k, sparsity_block_size, sparsity_percentage)

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_s, sparsity_layout_x_s)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            blksprs_softmax = BlocksparseSoftmax(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_dense = BlocksparseToDense(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)

            stock_out = blksprs_to_dense(blksprs_to_sparse(torch.softmax(x_stock, dim=-1), sparsity_layout_x),
                                         sparsity_layout_x)
            blksprs_softmax_out = blksprs_softmax(blksprs_to_sparse(x_blksprs, sparsity_layout_x), sparsity_layout_x)
            blksprs_softmax_dense_out = blksprs_to_dense(blksprs_softmax_out, sparsity_layout_x)

            _visualise((blksprs_softmax_dense_out, "blksprs_softmax_dense_out"), (stock_out, "stock_out"), dim=1)

            assert torch.allclose(blksprs_softmax_dense_out, stock_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_out, target)
            blksprs_loss = blksprs_loss(blksprs_softmax_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            _visualise((x_blksprs.grad, "blksprs_softmax_dense_grad_out"), (x_stock.grad, "stock_grad_out"))

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_transpose():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        x_s, sparsity_layout_x_s = _get_blocksparse_input(b, m, k, sparsity_block_size, sparsity_percentage)

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_s, sparsity_layout_x_s)]:
            blksprs_transpose = BlocksparseTranspose(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_dense = BlocksparseToDense(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)

            blksprs_to_sparse_out = blksprs_to_sparse(x, sparsity_layout_x)
            stock_transpose_out = x.transpose(1, 2)

            blksprs_transpose, blksprs_sparsity_layout_t = blksprs_transpose(blksprs_to_sparse_out, sparsity_layout_x)

            blksprs_to_dense_out_t = blksprs_to_dense(blksprs_transpose, blksprs_sparsity_layout_t)

            assert torch.allclose(blksprs_to_dense_out_t, stock_transpose_out, atol=ATOL, rtol=RTOL)


def test_blksprs_to_sparse():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        x_s, sparsity_layout_x_s = _get_blocksparse_input(b, m, k, sparsity_block_size, sparsity_percentage)

        for x, sparsity_layout_x_s in [(x, sparsity_layout_x), (x_s, sparsity_layout_x_s)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)

            stock_to_sparse_out = BlocksparseTools.slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)

            blksprs_to_sparse_out = blksprs_to_sparse(x_blksprs, sparsity_layout_x_s)

            assert torch.allclose(blksprs_to_sparse_out, stock_to_sparse_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_to_sparse_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_to_sparse_out, target)
            blksprs_loss = blksprs_loss(blksprs_to_sparse_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_to_dense():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        x_s, sparsity_layout_x_s = _get_blocksparse_input(b, m, k, sparsity_block_size, sparsity_percentage)

        for x, sparsity_layout_x_s in [(x, sparsity_layout_x), (x_s, sparsity_layout_x_s)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            blksprs_to_dense = BlocksparseToDense(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)

            stock_to_sparse_out = BlocksparseTools.slow_to_sparse(x_stock, sparsity_layout_x_s, sparsity_block_size)
            stock_to_dense_out = BlocksparseTools.slow_to_dense(stock_to_sparse_out, sparsity_layout_x_s,
                                                                sparsity_block_size)

            blksprs_to_sparse_out = blksprs_to_sparse(x_blksprs, sparsity_layout_x_s)
            blksprs_to_dense_out = blksprs_to_dense(blksprs_to_sparse_out, sparsity_layout_x_s)

            assert torch.allclose(blksprs_to_dense_out, stock_to_dense_out, atol=ATOL, rtol=RTOL)

            target = torch.randn_like(stock_to_dense_out)
            stock_loss = torch.nn.L1Loss()
            blksprs_loss = torch.nn.L1Loss()
            stock_loss = stock_loss(stock_to_dense_out, target)
            blksprs_loss = blksprs_loss(blksprs_to_dense_out, target)

            stock_loss.backward()
            blksprs_loss.backward()

            assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blksprs_row_wise_sum():
    for b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage in TEST_CONFIGURATIONS:
        x = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)

        x_s, sparsity_layout_x_s = _get_blocksparse_input(b, m, k, sparsity_block_size, sparsity_percentage)

        for x, sparsity_layout_x in [(x, sparsity_layout_x), (x_s, sparsity_layout_x_s)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            blksprs_row_wise_sum = BlocksparseRowWiseSum(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)
            blksprs_to_dense = BlocksparseToDense(sparsity_block_size, DEVICE, triton_block_size=triton_block_size)

            stock_out = torch.sum(x_stock, dim=-1)
            blksprs_row_wise_sum_out, sparsity_layout_output = blksprs_row_wise_sum(
                blksprs_to_sparse(x_blksprs, sparsity_layout_x), sparsity_layout_x)
            blksprs_row_wise_sum_out_dense = blksprs_to_dense(blksprs_row_wise_sum_out, sparsity_layout_output)

            blksprs_row_wise_sum_out_slice = blksprs_row_wise_sum_out_dense[..., 0]

            assert torch.allclose(blksprs_row_wise_sum_out_slice, stock_out, atol=ATOL, rtol=RTOL)


def test_blksprs_exp():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x in [(x_s, sparsity_layout_x), (x, SPARSITY_LAYOUT_FULL), (x_s, sparsity_layout_x)]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        blksprs_exp = BlocksparseExp(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)

        stock_out = blksprs_to_dense(blksprs_to_sparse(torch.exp(x_stock), sparsity_layout_x), sparsity_layout_x)
        blksprs_exp_out = blksprs_exp(blksprs_to_sparse(x_blksprs, sparsity_layout_x))
        blksprs_exp_dense_out = blksprs_to_dense(blksprs_exp_out, sparsity_layout_x)

        assert torch.allclose(blksprs_exp_dense_out, stock_out, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(stock_out)
        stock_loss = torch.nn.L1Loss()
        blksprs_loss = torch.nn.L1Loss()
        stock_loss = stock_loss(stock_out, target)
        blksprs_loss = blksprs_loss(blksprs_exp_dense_out, target)

        stock_loss.backward()
        blksprs_loss.backward()

        assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


# Utility


def _get_blocksparse_input(b, m, n, sparsity_block_size, sparsity_percentage):
    x = torch.randn(size=(b, m, n), device=DEVICE)
    sparsity_layout = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size), device=DEVICE)
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    num_zero_elements = int(m_s * n_s * (1 - sparsity_percentage))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = 0

    blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, DEVICE)
    blksprs_to_dense = BlocksparseToDense(sparsity_block_size, DEVICE)

    return blksprs_to_dense(blksprs_to_sparse(x, sparsity_layout), sparsity_layout), sparsity_layout


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

from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt

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
    x = torch.randn(size=(B, M, K), device=DEVICE)
    y = torch.randn(size=(B, N, K), device=DEVICE).transpose(-1, -2).contiguous()

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    y_s, sparsity_layout_y = _get_blocksparse_input(B, K, N, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x, y, sparsity_layout_y in [(x, SPARSITY_LAYOUT_FULL, y, SPARSITY_LAYOUT_FULL),
                                                       (x_s, sparsity_layout_x, y_s, sparsity_layout_y)]:
        x_stock = x.clone().requires_grad_(True)
        y_stock = y.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)
        y_blksprs = y.clone().requires_grad_(True)

        blksprs_matmul_sss = BlocksparseMatmulSSS(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

        stock_matmul_out = torch.matmul(x_stock, y_stock)
        blksprs_matmul_out = blksprs_matmul_sss(blksprs_to_sparse(x_blksprs, sparsity_layout_x),
                                                blksprs_to_sparse(y_blksprs, sparsity_layout_y),
                                                sparsity_layout_x, sparsity_layout_y, SPARSITY_LAYOUT_FULL)
        blksprs_matmul_out_dense = blksprs_to_dense(blksprs_matmul_out, SPARSITY_LAYOUT_FULL)

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
    M = 4
    K = 4
    SPARSITY_BLOCK_SIZE = 2
    TRITON_BLOCK_SIZE = 1

    x = torch.randn(size=(B, M, K), device=DEVICE)
    x = torch.arange(0, K).unsqueeze(0).expand(M, K).unsqueeze(0).expand(B, M, K).to(DEVICE).float()
    x = torch.arange(0, B * M * K, device=DEVICE, dtype=torch.float).reshape(B, M, K)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x in [(x, SPARSITY_LAYOUT_FULL), (x_s, sparsity_layout_x)]:
        blksprs_softmax = BlocksparseSoftmax(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)
        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)

        stock_softmax_out = torch.softmax(x, dim=-1)
        stock_softmax_out_t = blksprs_to_dense(blksprs_to_sparse(stock_softmax_out, sparsity_layout_x),
                                               sparsity_layout_x, fill_value=float("-inf"))

        blksprs_softmax_out = blksprs_softmax(blksprs_to_sparse(x, sparsity_layout_x), sparsity_layout_x,
                                              fill_value_output=0)
        blksprs_to_dense_out = blksprs_to_dense(blksprs_softmax_out, sparsity_layout_x, fill_value=float("-inf"))

        _visualise((blksprs_to_dense_out, "blksprs_softmax_out"), (stock_softmax_out, "stock_softmax_out"))

        # assert torch.allclose(blksprs_to_dense_out, stock_softmax_out_t, atol=ATOL, rtol=RTOL)


def test_blksprs_transpose():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x in [(x, SPARSITY_LAYOUT_FULL), (x_s, sparsity_layout_x)]:
        blksprs_transpose = BlocksparseTranspose(SPARSITY_BLOCK_SIZE, DEVICE)
        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

        blksprs_to_sparse_out = blksprs_to_sparse(x, sparsity_layout_x)
        stock_transpose_out = x.transpose(1, 2)

        blksprs_transpose, blksprs_sparsity_layout_t = blksprs_transpose(blksprs_to_sparse_out, sparsity_layout_x)

        blksprs_to_dense_out_t = blksprs_to_dense(blksprs_transpose, blksprs_sparsity_layout_t)

        assert torch.allclose(blksprs_to_dense_out_t, stock_transpose_out, atol=ATOL, rtol=RTOL)


def test_blocksparse_to_sparse():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x in [(x, SPARSITY_LAYOUT_FULL), (x_s, sparsity_layout_x)]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)

        stock_to_sparse_out = BlocksparseTools.slow_to_sparse(x_stock, sparsity_layout_x, SPARSITY_BLOCK_SIZE)

        blksprs_to_sparse_out = blksprs_to_sparse(x_blksprs, sparsity_layout_x)

        assert torch.allclose(blksprs_to_sparse_out, stock_to_sparse_out, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(stock_to_sparse_out)
        stock_loss = torch.nn.L1Loss()
        blksprs_loss = torch.nn.L1Loss()
        stock_loss = stock_loss(stock_to_sparse_out, target)
        blksprs_loss = blksprs_loss(blksprs_to_sparse_out, target)

        stock_loss.backward()
        blksprs_loss.backward()

        assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blocksparse_to_dense():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x in [(x, SPARSITY_LAYOUT_FULL), (x_s, sparsity_layout_x)]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)

        stock_to_sparse_out = BlocksparseTools.slow_to_sparse(x_stock, sparsity_layout_x, SPARSITY_BLOCK_SIZE)
        stock_to_dense_out = BlocksparseTools.slow_to_dense(stock_to_sparse_out, sparsity_layout_x, SPARSITY_BLOCK_SIZE)

        blksprs_to_sparse_out = blksprs_to_sparse(x_blksprs, sparsity_layout_x)
        blksprs_to_dense_out = blksprs_to_dense(blksprs_to_sparse_out, sparsity_layout_x)

        assert torch.allclose(blksprs_to_dense_out, stock_to_dense_out, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(stock_to_dense_out)
        stock_loss = torch.nn.L1Loss()
        blksprs_loss = torch.nn.L1Loss()
        stock_loss = stock_loss(stock_to_dense_out, target)
        blksprs_loss = blksprs_loss(blksprs_to_dense_out, target)

        stock_loss.backward()
        blksprs_loss.backward()

        assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blocksparse_row_wise_sum():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x, sparsity_layout_x in [(x, SPARSITY_LAYOUT_FULL), (x_s, sparsity_layout_x)]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        blksprs_row_wise_sum = BlocksparseRowWiseSum(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)

        stock_out = torch.sum(x_stock, dim=-1)
        blksprs_row_wise_sum_out, sparsity_layout_output = blksprs_row_wise_sum(
            blksprs_to_sparse(x_blksprs, sparsity_layout_x), sparsity_layout_x)
        blksprs_row_wise_sum_out_dense = blksprs_to_dense(blksprs_row_wise_sum_out, sparsity_layout_output)

        blksprs_row_wise_sum_out_slice = blksprs_row_wise_sum_out_dense[..., 0]

        assert torch.allclose(blksprs_row_wise_sum_out_slice, stock_out, atol=ATOL, rtol=RTOL)


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


def _visualise(*matrices):
    for matrix_tuple in matrices:
        matrix_data = matrix_tuple[0]
        matrix_label = matrix_tuple[1]

        output_path_base = BASE_PATH.joinpath("test", "output", "blksprs")
        output_path_base.mkdir(exist_ok=True)

        _visualise_matrix(matrix_data[0], str(output_path_base.joinpath(matrix_label)), grid_size=1)


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

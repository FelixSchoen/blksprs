from pathlib import Path
import torch
from matplotlib import pyplot as plt

from blksprs.ops.blocksparse import BlocksparseMatmulSSS, BlocksparseToDense, BlocksparseToSparse, BlocksparseSoftmax, \
    BlocksparseTranspose, BlocksparseTools, BaseBlocksparse
from blksprs.utils.benchmarking import benchmark

# Device setup
DEVICE = torch.device("cuda:0")

# Constants
B, M, N, K = 2, 64, 64, 64
SPARSITY_BLOCK_SIZE = 32
TRITON_BLOCK_SIZE = 16
SPARSITY_LAYOUT = torch.ones(size=(B, M // SPARSITY_BLOCK_SIZE, K // SPARSITY_BLOCK_SIZE), device=DEVICE)
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


# Tests

def test_blksprs_matmul_sss():
    x = torch.randn(size=(B, M, K), device=DEVICE)
    y = torch.randn(size=(B, N, K), device=DEVICE).transpose(-1, -2).contiguous()

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    y_s, sparsity_layout_y = _get_blocksparse_input(B, K, N, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    # _visualise((stock_matmul_out, "stock_matmul_out"), (blksprs_dense, "blksprs_dense"))

    for x, y in [(x, y), (x_s, y_s)]:
        x_stock = x.clone().requires_grad_(True)
        y_stock = y.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)
        y_blksprs = y.clone().requires_grad_(True)

        blksprs_matmul_sss = BlocksparseMatmulSSS(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

        stock_matmul_out = torch.matmul(x_stock, y_stock)
        blksprs_matmul_out = blksprs_matmul_sss(blksprs_to_sparse(x_blksprs, SPARSITY_LAYOUT),
                                                blksprs_to_sparse(y_blksprs, SPARSITY_LAYOUT),
                                                SPARSITY_LAYOUT, SPARSITY_LAYOUT, SPARSITY_LAYOUT)
        blksprs_matmul_out_dense = blksprs_to_dense(blksprs_matmul_out, SPARSITY_LAYOUT)

        assert torch.allclose(blksprs_matmul_out_dense, stock_matmul_out, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(stock_matmul_out)
        stock_loss = torch.nn.L1Loss()
        stock_loss = stock_loss(stock_matmul_out, target)

        stock_loss.backward()

        pass

    # x, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    # y, sparsity_layout_y = _get_blocksparse_input(B, K, N, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    #
    # blksprs_matmul_sss = BlocksparseMatmulSSS(SPARSITY_BLOCK_SIZE, DEVICE)
    # blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    # blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)
    #
    # stock_matmul_out = torch.matmul(x, y)
    # blksprs_matmul_out = blksprs_matmul_sss(blksprs_to_sparse(x, sparsity_layout_x),
    #                                     blksprs_to_sparse(y, sparsity_layout_y),
    #                                     sparsity_layout_x, sparsity_layout_y, SPARSITY_LAYOUT)
    # blksprs_dense = blksprs_to_dense(blksprs_matmul_out, SPARSITY_LAYOUT)
    #
    # assert torch.allclose(blksprs_dense, stock_matmul_out, atol=ATOL, rtol=RTOL)

    # Benchmark
    if BENCHMARK:
        method_labels = ["pytorch", "blksprs"]
        func_input_generator = lambda mat_size, spar_blk_size, trit_blk_size: {
            "_": (blksprs_to_sparse := BlocksparseToSparse(spar_blk_size, DEVICE)),
            "x": (x := torch.randn(size=(B, mat_size, mat_size), device=DEVICE)),
            "y": (y := torch.randn(size=(B, mat_size, mat_size), device=DEVICE)),
            "sparsity_layout": (sparsity_layout := torch.ones(
                device=DEVICE, size=(B, mat_size // spar_blk_size, mat_size // spar_blk_size))),
            "x_s": blksprs_to_sparse(x, sparsity_layout),
            "y_s": blksprs_to_sparse(y, sparsity_layout),
            "func": BlocksparseMatmulSSS(spar_blk_size, DEVICE, triton_block_size=trit_blk_size)}
        func_test_subject_0 = lambda func, x, y, **kwargs,: torch.matmul(x, y)
        func_test_subject_1 = lambda func, x_s, y_s, sparsity_layout, **kwargs: func(x_s,
                                                                                     y_s,
                                                                                     sparsity_layout,
                                                                                     sparsity_layout,
                                                                                     sparsity_layout)
        benchmark(method_labels, func_input_generator,
                  BENCHMARK_MATRIX_SIZES, BENCHMARK_SPARSITY_BLOCK_SIZES, BENCHMARK_TRITON_BLOCK_SIZES,
                  func_test_subject_0, func_test_subject_1,
                  y_lim_top=150)


def test_blksprs_softmax():
    x = torch.randn(size=(B, M, K), device=DEVICE)
    blksprs_softmax = BlocksparseSoftmax(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)

    stock_softmax = torch.softmax(x, dim=-1)

    blksprs_softmax_o = blksprs_softmax(blksprs_to_sparse(x, SPARSITY_LAYOUT), SPARSITY_LAYOUT)
    blksprs_dense = blksprs_to_dense(blksprs_softmax_o, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_dense, stock_softmax, atol=ATOL, rtol=RTOL)


def test_blksprs_transpose():
    x = torch.randn(size=(B, M, K), device=DEVICE)
    blksprs_transpose = BlocksparseTranspose(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    blksprse_sparse = blksprs_to_sparse(x, SPARSITY_LAYOUT)
    stock_transpose = x.transpose(1, 2)

    blksprse_transpose, blocksprse_sparsity_layout_t = blksprs_transpose(blksprse_sparse, SPARSITY_LAYOUT)

    blksprse_dense_t = blksprs_to_dense(blksprse_transpose, blocksprse_sparsity_layout_t)

    assert torch.allclose(blksprse_dense_t, stock_transpose, atol=ATOL, rtol=RTOL)

    x, _ = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    blksprs_transpose = BlocksparseTranspose(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    blksprse_sparse = blksprs_to_sparse(x, SPARSITY_LAYOUT)
    stock_transpose = x.transpose(1, 2)

    blksprse_transpose, blocksprse_sparsity_layout_t = blksprs_transpose(blksprse_sparse, SPARSITY_LAYOUT)

    blksprse_dense_t = blksprs_to_dense(blksprse_transpose, blocksprse_sparsity_layout_t)

    assert torch.allclose(blksprse_dense_t, stock_transpose, atol=ATOL, rtol=RTOL)


def test_blocksparse_to_sparse():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    # _visualise((stock_matmul_out, "stock_matmul_out"), (blksprs_dense, "blksprs_dense"))

    for x in [x, x_s]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)

        stock_sparse_out = BlocksparseTools.to_sparse(x_stock, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)

        blksprs_sparse_out = blksprs_to_sparse(x_blksprs, SPARSITY_LAYOUT)

        assert torch.allclose(blksprs_sparse_out, stock_sparse_out, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(stock_sparse_out)
        stock_loss = torch.nn.L1Loss()
        blksprs_loss = torch.nn.L1Loss()
        stock_loss = stock_loss(stock_sparse_out, target)
        blksprs_loss = blksprs_loss(blksprs_sparse_out, target)

        stock_loss.backward()
        blksprs_loss.backward()

        assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


def test_blocksparse_to_dense():
    x = torch.randn(size=(B, M, K), device=DEVICE)

    x_s, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    for x in [x, x_s]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)
        blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE, triton_block_size=TRITON_BLOCK_SIZE)

        stock_sparse_out = BlocksparseTools.to_sparse(x_stock, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)
        stock_dense_out = BlocksparseTools.to_dense(stock_sparse_out, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)

        blksprs_sparse_out = blksprs_to_sparse(x_blksprs, SPARSITY_LAYOUT)
        blksprs_dense_out = blksprs_to_dense(blksprs_sparse_out, SPARSITY_LAYOUT)

        assert torch.allclose(blksprs_dense_out, stock_dense_out, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(stock_dense_out)
        stock_loss = torch.nn.L1Loss()
        blksprs_loss = torch.nn.L1Loss()
        stock_loss = stock_loss(stock_dense_out, target)
        blksprs_loss = blksprs_loss(blksprs_dense_out, target)

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

    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)

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

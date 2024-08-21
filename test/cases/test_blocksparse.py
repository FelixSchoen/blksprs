from pathlib import Path
import torch
import triton
from matplotlib import pyplot as plt

from blksprs.blocksparse import BlocksparseMatmulSSS, BlocksparseToDense, BlocksparseToSparse, BlocksparseSoftmax, \
    BlocksparseTranspose, BlocksparseTools

# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants
B, M, N, K = 2, 64, 64, 64
SPARSITY_BLOCK_SIZE = 32
SPARSITY_LAYOUT = torch.ones(size=(B, M // SPARSITY_BLOCK_SIZE, K // SPARSITY_BLOCK_SIZE), device=DEVICE)
BASE_PATH = Path(__file__).parent.parent.parent

# Settings
SPARSITY_PERCENTAGE = 0.75  # Percentage of non-sparse blocks
BENCHMARK = False  # Whether to run benchmark
BENCHMARK_DIMS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Dimensions to benchmark

# Tolerances
ATOL = 1e-2
RTOL = 1e-2


# Tests

def test_blksprs_matmul_sss():
    x = torch.randn(size=(B, M, K), device=DEVICE)
    y = torch.randn(size=(B, N, K), device=DEVICE).transpose(-1, -2).contiguous()

    blksprs_matmul_sss = BlocksparseMatmulSSS(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    stock_matmul = torch.matmul(x, y)
    blksprs_matmul = blksprs_matmul_sss(blksprs_to_sparse(x, SPARSITY_LAYOUT), blksprs_to_sparse(y, SPARSITY_LAYOUT),
                                        SPARSITY_LAYOUT, SPARSITY_LAYOUT, SPARSITY_LAYOUT)
    blksprs_dense = blksprs_to_dense(blksprs_matmul, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_dense, stock_matmul, atol=ATOL, rtol=RTOL)

    x, sparsity_layout_x = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    y, sparsity_layout_y = _get_blocksparse_input(B, K, N, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)

    blksprs_matmul_sss = BlocksparseMatmulSSS(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    stock_matmul = torch.matmul(x, y)
    blksprs_matmul = blksprs_matmul_sss(blksprs_to_sparse(x, sparsity_layout_x),
                                        blksprs_to_sparse(y, sparsity_layout_y),
                                        sparsity_layout_x, sparsity_layout_y, SPARSITY_LAYOUT)
    blksprs_dense = blksprs_to_dense(blksprs_matmul, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_dense, stock_matmul, atol=ATOL, rtol=RTOL)

    # Benchmark
    if BENCHMARK:
        method_labels = ["pytorch", "blksprs"]
        func_input_generator = lambda mat_size: {"x": torch.randn(size=(B, mat_size, mat_size), device=DEVICE),
                                                 "y": torch.randn(size=(B, mat_size, mat_size), device=DEVICE),
                                                 "sparsity_layout": torch.ones(device=DEVICE,
                                                                               size=(
                                                                                   B, mat_size // SPARSITY_BLOCK_SIZE,
                                                                                   mat_size // SPARSITY_BLOCK_SIZE))}
        func_test_subject_0 = lambda x, y, sparsity_layout: torch.matmul(x, y)
        func_test_subject_1 = lambda x, y, sparsity_layout: blksprs_matmul_sss(blksprs_to_sparse(x, sparsity_layout),
                                                                               blksprs_to_sparse(y, sparsity_layout),
                                                                               sparsity_layout, sparsity_layout,
                                                                               sparsity_layout)
        _benchmark(method_labels, func_input_generator, func_test_subject_0, func_test_subject_1)


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
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    stock_sparse = BlocksparseTools.to_sparse(x, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)

    blksprs_sparse = blksprs_to_sparse(x, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_sparse, stock_sparse, atol=ATOL, rtol=RTOL)

    x, _ = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    stock_sparse = BlocksparseTools.to_sparse(x, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)

    blksprs_sparse = blksprs_to_sparse(x, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_sparse, stock_sparse, atol=ATOL, rtol=RTOL)


def test_blocksparse_to_dense():
    x = torch.randn(size=(B, M, K), device=DEVICE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    blksprs_sparse = blksprs_to_sparse(x, SPARSITY_LAYOUT)
    stock_sparse = BlocksparseTools.to_sparse(x, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)
    stock_dense = BlocksparseTools.to_dense(stock_sparse, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)

    blksprs_dense = blksprs_to_dense(blksprs_sparse, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_dense, stock_dense, atol=ATOL, rtol=RTOL)

    x, _ = _get_blocksparse_input(B, M, K, SPARSITY_BLOCK_SIZE, SPARSITY_PERCENTAGE)
    blksprs_to_dense = BlocksparseToDense(SPARSITY_BLOCK_SIZE, DEVICE)
    blksprs_to_sparse = BlocksparseToSparse(SPARSITY_BLOCK_SIZE, DEVICE)

    blksprs_sparse = blksprs_to_sparse(x, SPARSITY_LAYOUT)
    stock_sparse = BlocksparseTools.to_sparse(x, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)
    stock_dense = BlocksparseTools.to_dense(stock_sparse, SPARSITY_LAYOUT, SPARSITY_BLOCK_SIZE)

    blksprs_dense = blksprs_to_dense(blksprs_sparse, SPARSITY_LAYOUT)

    assert torch.allclose(blksprs_dense, stock_dense, atol=ATOL, rtol=RTOL)


# Utility

def _benchmark(method_labels, func_input_generator, *funcs_test_subject):
    quantiles = [0.5, 0.2, 0.8]
    results = {}

    for benchmark_dim in BENCHMARK_DIMS:
        arguments = func_input_generator(benchmark_dim)

        for i, func_test_subject in enumerate(funcs_test_subject):
            func_ms_avg, func_ms_min, func_ms_max = triton.testing.do_bench(
                lambda: func_test_subject(**arguments), quantiles=quantiles)
            results.setdefault(i, []).append((func_ms_avg, func_ms_min, func_ms_max))

    plt.figure(dpi=300)
    for key_method, value_method in results.items():
        ms_method_avg, ms_method_min, ms_method_max = zip(*value_method)
        plt.plot(BENCHMARK_DIMS, ms_method_avg, label=method_labels[key_method])
        plt.fill_between(BENCHMARK_DIMS, ms_method_min, ms_method_max, alpha=0.2)

    plt.xlabel("Matrix size")
    plt.ylabel("Time (ms)")
    plt.ylim(bottom=0, top=None)
    plt.grid(True)
    plt.legend()
    plt.show()


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

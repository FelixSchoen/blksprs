from typing import Callable

import triton
from matplotlib import pyplot as plt


def benchmark(method_labels: list[str], func_input_generator: Callable,
              matrix_sizes: list[int], sparsity_block_sizes: list[int], triton_block_sizes: list[int],
              *funcs_test_subject: Callable, y_lim_top: int = None):
    quantiles = [0.5, 0.2, 0.8]
    results = {}

    for matrix_size, sparsity_block_size, triton_block_size in zip(matrix_sizes, sparsity_block_sizes, triton_block_sizes):
        arguments = func_input_generator(matrix_size, sparsity_block_size, triton_block_size)

        for i, func_test_subject in enumerate(funcs_test_subject):
            func_ms_avg, func_ms_min, func_ms_max = triton.testing.do_bench(
                lambda: func_test_subject(**arguments), quantiles=quantiles)
            results.setdefault(i, []).append((func_ms_avg, func_ms_min, func_ms_max))

    plt.figure(dpi=300)
    for key_method, value_method in results.items():
        ms_method_avg, ms_method_min, ms_method_max = zip(*value_method)
        plt.plot(matrix_sizes, ms_method_avg, label=method_labels[key_method])
        plt.fill_between(matrix_sizes, ms_method_min, ms_method_max, alpha=0.2)

    plt.xlabel("Matrix size")
    plt.ylabel("Time (ms)")
    plt.ylim(bottom=0, top=y_lim_top)
    plt.grid(True)
    plt.legend()
    plt.show()

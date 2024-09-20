# blksprs

## Overview

A lightweight and efficient library for operations on block-sparse matrices in PyTorch using Triton.

Currently supported operations (includes gradient calculation):

- Sparse matrix multiplication (_supports any combination of sparse and dense matrices due to support
  for `sparse = sparse @ sparse` matmul_)
- Softmax
- Transposition
- Gather
- Scatter (_supports either no reduction or summation, gradients are only available for summation_)
- Conversion to and from sparse form
- Conversion to different sparsity layouts and different sparsity block sizes

As with this library sparse matrices are represented using a tuple of `(matrix, sparsity_layout, sparsity_block_size)`,
any element-wise operations can be applied in regular torch-like fashion.
These include, e.g.,

- Element-wise addition and subtraction
- Element-wise multiplication and division
- Element-wise exponentiation
- ...

Note that in order to correctly apply element-wise operations between two sparse tensors their sparsity layouts have to
match.

Furthermore, the library provides a set of utility functions for the creation of sparsity layouts based on existing
dense tensors.

## Installation

Note that due to the dependency on [Triton](https://github.com/triton-lang/triton) this library is only compatible with
the Linux platform.

We recommend installing blksprs from [PyPI](https://pypi.org/project/blksprs/) using pip:

```pip install blksprs```

### Dependencies

- [PyTorch](https://pytorch.org/) (built with v2.4.0)
- _[Triton](https://github.com/triton-lang/triton) (included with PyTorch)_

## Changelog

See [`CHANGELOG.md`](https://github.com/FelixSchoen/blksprs/blob/main/CHANGELOG.md) for a detailed changelog.

## Usage

We provide an example below to demonstrate the usage of the library.
For more detailed examples, please refer to
the [test cases](https://github.com/FelixSchoen/blksprs/blob/main/test/cases/test_blocksparse.py) which cover all
implemented operations and functions.
The example below can also be found in
the [test cases](https://github.com/FelixSchoen/blksprs/blob/main/test/cases/test_readme.py).

```python
import torch

from blksprs.layouting.sparsity_layout import build_sparsity_layout
from blksprs.ops.conversion import to_sparse, to_dense
from blksprs.ops.matmul import matmul
from blksprs.ops.row_wise_sum import row_wise_sum
from blksprs.ops.softmax import softmax
from blksprs.ops.transpose import transpose
from blksprs.utils.tools import do_shape_blocksparse, undo_shape_blocksparse


def test_readme():
    # Set up parameters (batch size, number of heads, dimensions for matrices (m, k) and (n, k))
    b, h, m, n, k = 2, 4, 64, 64, 16

    # Percentage of blocks that will be sparse in the output for demonstration purposes
    sparsity_percentage = 25

    # Must be a power of two, greater than or equal to 16 for matmul, and divide m, n, and k
    sparsity_block_size = 16

    # Must be a power of two and smaller than or equal to sparsity_block_size
    # If it is set to ``none`` a value will be chosen automatically
    triton_block_size = None

    # Initialise random (dense) tensors
    x = torch.randn(size=(b, h, m, k), device="cuda")
    y = torch.randn(size=(b, h, n, k), device="cuda").transpose(-1, -2).contiguous()

    # Convert tensors to three-dimensional (dense) tensors since Triton can only handle tensors of exactly three dimensions
    x_dense, x_shape_original = do_shape_blocksparse(x)
    y_dense, y_shape_original = do_shape_blocksparse(y)

    # Create sparsity layouts from existing tensors
    sparsity_layout_x = build_sparsity_layout(x_dense, sparsity_block_size, triton_block_size=triton_block_size)
    sparsity_layout_y = build_sparsity_layout(y_dense, sparsity_block_size, triton_block_size=triton_block_size)

    # Create random sparsity layout for output tensor
    sparsity_layout_o = _get_random_sparsity_layout(b * h, m, n, sparsity_block_size, sparsity_percentage)

    # Convert tensors to sparse tensors for matrix multiplication
    x_sparse = to_sparse(x_dense, sparsity_layout_x, sparsity_block_size, triton_block_size=triton_block_size)
    y_sparse = to_sparse(y_dense, sparsity_layout_y, sparsity_block_size, triton_block_size=triton_block_size)

    # Perform matrix multiplication
    o_sparse = matmul(x_sparse, sparsity_layout_x, y_sparse, sparsity_layout_y, sparsity_layout_o, sparsity_block_size,
                      triton_block_size=triton_block_size)
    o_dense = to_dense(o_sparse, sparsity_layout_o, sparsity_block_size, triton_block_size=triton_block_size)

    # Sanity check
    o_torch = torch.matmul(x_dense, y_dense)

    # Perform round trip to set sparse blocks to 0
    o_torch_round_trip = to_dense(
        to_sparse(o_torch, sparsity_layout_o, sparsity_block_size, triton_block_size=triton_block_size),
        sparsity_layout_o, sparsity_block_size, fill_value=0, triton_block_size=triton_block_size)

    # Assert that the output is correct
    assert torch.allclose(o_dense, o_torch_round_trip, atol=2e-2)  # Note that small numerical differences are expected

    # Assert that the output has the correct sparsity layout
    actual_sparsity_layout_o = build_sparsity_layout(o_dense, sparsity_block_size, triton_block_size=triton_block_size)
    assert torch.allclose(actual_sparsity_layout_o, sparsity_layout_o)

    # Convert output tensor back to original shape
    o = undo_shape_blocksparse(o_dense, x_shape_original)

    # Other available functions
    transpose(o_sparse, sparsity_layout_o, sparsity_block_size, triton_block_size=triton_block_size)
    softmax(o_sparse, sparsity_layout_o, sparsity_block_size, triton_block_size=triton_block_size)
    row_wise_sum(o_sparse, sparsity_layout_o, sparsity_block_size, triton_block_size=triton_block_size)


def _get_random_sparsity_layout(b, m, n, sparsity_block_size, sparsity_percentage):
    """Helper function, creates a random sparsity layout for a given shape with a given percentage of blocks marked as sparse.

    """
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), device="cuda", dtype=torch.int)

    num_zero_elements = int(m_s * n_s * (sparsity_percentage / 100))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = 0

    return sparsity_layout
```
# 🧊 blksprs

[![GitHub Release](https://img.shields.io/github/v/release/FelixSchoen/blksprs?include_prereleases&label=Latest%20Release)](https://github.com/FelixSchoen/blksprs/releases)
[![Python 3.11](https://img.shields.io/badge/Python%20Version-3.11-blue)](https://www.python.org/downloads/release/python-3119/)
[![Python 3.12](https://img.shields.io/badge/Python%20Version-3.12-blue)](https://www.python.org/downloads/release/python-31210/)
[![Python 3.13](https://img.shields.io/badge/Python%20Version-3.13-blue)](https://www.python.org/downloads/)
[![Python 3.14](https://img.shields.io/badge/Python%20Version-3.14-blue)](https://www.python.org/downloads/)

## 📖 Overview

A lightweight and efficient library for operations on block-sparse matrices in PyTorch using Triton.

Core operations support gradient calculation unless noted otherwise:

- Matrix multiplication
- Softmax
- Transpose
- Gather
- Scatter (_supports either no reduction or summation; gradients are only available for summation_)
- Repeat (_supports target sparsity layout_)
- Repeat-interleave (_supports target sparsity layout_)
- Splitting and merging of matrices (_currently restricted to the last dimension_)
- Conversion between dense and compressed form
- Conversion to different sparsity layouts and different sparsity block sizes
- Flash Attention (_supports custom masks and cross-attention_)

In this library, sparse matrices are represented by a tuple of
`(matrix, sparsity_layout, sparsity_block_size)`, so element-wise operations can be applied in regular PyTorch fashion.
These include, for example:

- Element-wise addition and subtraction
- Element-wise multiplication and division
- Element-wise exponentiation
- Other element-wise PyTorch operations

The sparsity layouts of two compressed tensors must match when applying an element-wise operation between them.

The ``bs.ops.misc`` module also provides non-differentiable helpers for:

- Row-wise sum, max, addition, and subtraction
- Broadcast addition and subtraction between slices

The library also provides utility functions:

- Creating sparsity layouts from dense tensors and distribution indices in ``bs.layouting``
- Applying ``nn.Linear``, ``nn.Dropout``, and row-wise normalisation modules to compressed tensors
- Shaping tensors and controlling input validation in ``bs.utils``

See the [Roadmap](#roadmap) for the current development scope.

## 🛠️ Installation

BLK-SPRS requires Linux, an NVIDIA GPU supported by CUDA, a compatible NVIDIA driver, and a CUDA-enabled PyTorch
installation. There is no CPU fallback. Triton is installed as a PyTorch dependency on supported Linux platforms.

We recommend installing blksprs from [PyPI](https://pypi.org/project/blksprs/) using pip:

```bash
pip install blksprs
```

### Dependencies

- Python 3.11 to 3.14
- [PyTorch](https://pytorch.org/) 2.8 or newer with CUDA support
- [Triton](https://github.com/triton-lang/triton), included with supported PyTorch installations

The exact PyTorch, Triton, CUDA, and driver versions must be mutually compatible. Follow the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for the appropriate CUDA build on your system.

### Supported dtypes

| Operation family | Supported dtypes |
| --- | --- |
| Matrix multiplication, Flash Attention, and row-wise helpers | `float16`, `bfloat16`, `float32` |
| Regular and fused softmax | `float32` |
| Gather and scatter indices | `int32`, `int64` |
| Flash Attention masks | `bool`, `float16`, `bfloat16`, `float32`; values must be binary |
| Conversion, transpose, repeat, partitioning, gather, and non-reducing scatter data | `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64`; input dtype is preserved |
| Summation scatter data | `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64` |
| Broadcast addition operands | Matching dtypes from the generic data-movement set above |
| Broadcast subtraction operands | Matching numeric dtypes from the generic data-movement set above |
| Sparsity and attention layouts | `bool` is recommended; integer and real floating-point layouts must contain only `0` and `1`; complex layouts are rejected |

Automatic mixed precision is supported by the operations decorated for CUDA autocast. Inputs to an individual
floating-point operation must satisfy that operation's dtype validation; in particular, matrix multiplication and
the Q, K, and V inputs of Flash Attention must use matching dtypes. As in PyTorch, CUDA autocast does not downcast
``float64`` inputs.

The sparse linear helpers follow ``torch.nn.Linear`` dtype semantics: outside CUDA autocast, their compressed input,
weight, and optional bias must have the same supported floating-point dtype. CUDA autocast may cast these operands to
its configured dtype.

Repeat counts are non-negative. A zero in ``repeat()`` or a zero count in ``repeat_interleave()`` returns an empty
compressed tensor and the correspondingly empty layout. Broadcast addition and subtraction accept independent slice
lengths ``M`` and ``N`` for inputs shaped ``(B, M)`` and ``(B, N)`` and produce a possibly rectangular ``(B, M, N)``
result; each length must be divisible by the sparsity block size.

## 📝 Changelog

See [`CHANGELOG.md`](https://github.com/FelixSchoen/blksprs/blob/main/CHANGELOG.md) for a detailed changelog.

## 📄 License

BLK-SPRS is licensed under the [MIT License](https://github.com/FelixSchoen/blksprs/blob/main/LICENSE.md).

## 🗺️ Roadmap

Since the library covers our current needs, it is in a **bugfix-only** state. There are no plans to add new features,
such as support for dimensions other than the last in the ``split`` and ``merge`` operations. We will continue to
maintain the library and fix issues that arise.

If you find a bug, please open an [issue](https://github.com/FelixSchoen/blksprs/issues).
We also encourage [pull requests](https://github.com/FelixSchoen/blksprs/pulls).

This scope may change for future projects; as of August 2026, the library meets our current requirements.

## ⚠️ Known Limitations and Issues

- Block-sparse and dense reference operations can differ slightly because their reduction order, precision, and kernel
  implementations differ. Appropriate tolerances depend on the dtype, tensor sizes, operation, and application; validate
  numerical error against the requirements of your workload.

- Compressed and dense data operands accepted by automatic contiguous conversion are normalised before kernel dispatch.
  Sparsity-layout tensors are structural metadata and must be contiguous when supplied by the caller.

- Floating-point operations that use atomic GPU reductions, including summation scatter with overlapping indices, may
  not be bitwise deterministic because the accumulation order can vary.

- Public operation wrappers are compatible with the default, graph-break-tolerant ``torch.compile`` mode, but are not
  compatible with ``fullgraph=True``. Validation and layout-cache preparation deliberately run as eager Python code, so
  compiling an individual BLK-SPRS wrapper may introduce graph breaks without improving performance.

## 💻 Usage

### Terminology

Block-sparse tensors are stored in compressed form: only active blocks are present in the tensor data. The accompanying
``sparsity_layout`` describes the full block layout.

Derived layout metadata is passed through an optional ``layout_cache`` dictionary. Internally, ``layout_indices`` map
compressed block positions to coordinates in the full sparsity layout, while ``packed_indices`` map full sparsity-layout
positions back to compressed block positions or ``-1`` for inactive blocks. Flash Attention additionally caches
``key_indices``/``key_offsets`` and ``query_indices``/``query_offsets`` for the attention pattern.

### Layout caches

A layout cache is a caller-owned mutable dictionary. Use a separate cache for each repeatedly invoked operation:

```python
layout_cache = {}
o_sparse = bs.ops.matmul(
    x_sparse, sparsity_layout_x,
    y_sparse, sparsity_layout_y,
    sparsity_layout_o, sparsity_block_size,
    layout_cache=layout_cache,
)
```

BLK-SPRS ties cached metadata to the operation, tensor identity, in-place tensor version, shape, stride, dtype, device,
gradient requirement, and scalar parameters used to construct it. It also snapshots derived cache contents after
construction. Reusing the dictionary with different inputs, after changing whether a cached parameter requires
gradients, or after mutating a source or derived layout safely clears and rebuilds the cached metadata. Newly derived
layouts do not alias their source layouts. The cache retains references to source tensors to prevent allocator identity
reuse from returning stale metadata. Consequently, a long-lived cache also keeps those tensors alive; call
``layout_cache.clear()`` when the cache is no longer required. Tensors created inside ``torch.inference_mode()`` do not
provide version counters, so their derived metadata is rebuilt on each call to preserve correctness after possible
in-place mutations.

### Softmax dispatch

``bs.ops.softmax()`` prefers the fused implementation. The fused kernel rounds the largest number of active blocks in a
row up to a power of two and processes that vector for each row element. BLK-SPRS uses it only when the resulting vector
contains at most 131,072 elements; larger rows automatically use the regular implementation to avoid pathological Triton
compilation and execution costs. Calling ``bs.ops.softmax_fused()`` explicitly enforces the same limit and raises a clear
``ValueError`` instead of falling back. Pass ``flag_fused=False`` to ``bs.ops.softmax()`` to request regular execution
directly.

### Layout adaptation and row-wise reductions

``bs.ops.adapt_layout()`` preserves the logical values when changing block
sizes. If a row or column extent is not divisible by a larger target block,
the returned layout covers the next complete block and the trailing values are
zero-padded. Backward propagation maps gradients only to the original logical
extent.

``bs.ops.misc.row_wise_sum()`` and ``row_wise_max()`` return square compressed
``BlksprsTensor`` blocks by default. With ``flag_slice_only=True``, they instead
return a regular ``torch.Tensor`` containing one value per row with shape
``(active_block_rows, sparsity_block_size, 1)``. This compact auxiliary representation is
accepted by ``row_wise_add()`` and ``row_wise_sub()`` but is not a general
block-sparse tensor and cannot be passed to operations such as ``to_dense()``. In the default square representation, the
first column contains the reductions; the remaining columns are padding with values of ``0`` for sums and ``-inf`` for
maxima.

### Flash Attention layouts and masks

Flash Attention uses compressed Q, K, and V tensors together with their individual sparsity layouts. The separate
``attention_layout`` has shape ``(batch, query_blocks, key_blocks)`` and marks the Q-K block pairs that participate in
attention. It is a block-level pattern rather than an element-level mask. Flash Attention supports sparsity block sizes
16, 32, and 64; larger blocks are rejected because the kernel cannot fit them portably into GPU shared memory.

An element-level ``attention_mask`` is also stored in compressed block-sparse form. It may use ``bool`` or a supported
floating-point dtype, its values must be binary, and a non-zero value means that the position is masked and does not
participate in attention. This convention matches
``Tensor.masked_fill`` and is the opposite of the boolean-mask convention of
``torch.nn.functional.scaled_dot_product_attention``. Optional tensors and layouts are pairs:

- provide both ``attention_mask`` and ``sparsity_layout_mask``, or neither;
- provide both ``attention_bias`` and ``sparsity_layout_bias``, or neither.

The mask is not differentiable. The additive attention bias is differentiable. The mask is applied after the bias so a
masked position remains ignored regardless of its bias value. When ``sparsity_layout_o`` is omitted, BLK-SPRS derives
the conservative structural output layout as
``build_sparsity_layout_matmul(attention_layout, sparsity_layout_v)``. This follows the attended V blocks rather than the
Q sparsity pattern and works when the V/output head dimension differs from the Q/K head dimension. Callers that need the
layout for a later operation can compute it explicitly with the same helper or read ``sparsity_layout_o`` from a supplied
Flash Attention layout cache.

### Validation and contiguous conversion

Input validation and automatic contiguous conversion for supported data operands are enabled by default and configured
per thread. Sparsity layouts are caller-managed structural metadata and must be contiguous. These safeguards can be
disabled for carefully measured hot paths, but must always be restored with ``try``/``finally``:

```python
bs.utils.disable_validation()
bs.utils.disable_contiguous()
try:
    # Inputs must now satisfy all documented shape, dtype, device, layout, and contiguity requirements.
    result = bs.ops.transpose(x_sparse, sparsity_layout_x, sparsity_block_size)
finally:
    bs.utils.enable_contiguous()
    bs.utils.enable_validation()
```

Disabling these safeguards is unsafe for untrusted or dynamically shaped inputs. Invalid indices or stale layout
metadata can otherwise reach low-level GPU kernels.

### ``torch.compile``

BLK-SPRS kernels are compiled by Triton independently of ``torch.compile``. Public wrappers can be called from a function
compiled with the default ``torch.compile`` settings, and BLK-SPRS tests verify that this graph-break-tolerant path remains
numerically correct. The wrappers also perform Python-side validation and prepare mutable layout caches; these steps cause
graph breaks and mean that ``fullgraph=True`` is not supported.

In eager execution, compressed outputs are marked with the ``BlksprsTensor`` subclass. During Dynamo tracing, compressed
outputs may instead be base ``torch.Tensor`` instances because constructing tensor subclasses inside the compiled graph
is unsupported. This does not change their compressed representation, and all BLK-SPRS operations accept either form.

Use eager BLK-SPRS wrappers by default. If the surrounding PyTorch model already benefits from ``torch.compile``, it can
continue to contain BLK-SPRS calls, but benchmark the complete workload rather than assuming that compiling an individual
wrapper improves its performance.

### Autotuning

``BLKSPRS_AUTOTUNE`` selects the Triton autotuning profile. Set it before importing ``blksprs``:

```bash
BLKSPRS_AUTOTUNE=DEFAULT python train.py
BLKSPRS_AUTOTUNE=TEST python -m pytest test/cases
```

``DEFAULT`` explores the production configuration set. ``TEST`` uses a smaller configuration set to reduce test startup
time. Any other value is rejected during import.

The following example is verified by
[`test_readme.py`](https://github.com/FelixSchoen/blksprs/blob/main/test/cases/test_readme.py). See
[`test_blocksparse.py`](https://github.com/FelixSchoen/blksprs/blob/main/test/cases/test_blocksparse.py) for examples of
the other operation families.

```python
import blksprs as bs
import torch


def test_readme():
    # Set up parameters (batch size, number of heads, dimensions for matrices (m, k) and (n, k))
    b, h, m, n, k = 2, 4, 64, 64, 16

    # Percentage of blocks that will be sparse in the output for demonstration purposes
    sparsity_percentage = 25

    # Must be a power of two, greater than or equal to 16 for matmul, and divide m, n, and k
    sparsity_block_size = 16

    # Initialise random (dense) tensors
    x = torch.randn(size=(b, h, m, k), device="cuda")
    y = torch.randn(size=(b, h, n, k), device="cuda").transpose(-1, -2).contiguous()

    # Flatten leading dimensions because BLK-SPRS operations accept three-dimensional tensors
    x_dense, x_shape_original = bs.utils.do_shape_blocksparse(x)
    y_dense, _ = bs.utils.do_shape_blocksparse(y)

    # Create sparsity layouts from existing tensors
    sparsity_layout_x = bs.layouting.build_sparsity_layout(x_dense, sparsity_block_size)
    sparsity_layout_y = bs.layouting.build_sparsity_layout(y_dense, sparsity_block_size)

    # Create random sparsity layout for output tensor
    sparsity_layout_o = _get_random_sparsity_layout(b * h, m, n, sparsity_block_size, sparsity_percentage)

    # Convert tensors to sparse tensors for matrix multiplication
    x_sparse = bs.ops.to_sparse(x_dense, sparsity_layout_x, sparsity_block_size)
    y_sparse = bs.ops.to_sparse(y_dense, sparsity_layout_y, sparsity_block_size)

    # Default torch.compile mode is supported for correctness. Public wrapper
    # validation and cache preparation may graph-break, so benchmark the full workload.
    matmul_compiled = torch.compile(bs.ops.matmul)

    # Perform matrix multiplication
    o_sparse = matmul_compiled(x_sparse, sparsity_layout_x,
                               y_sparse, sparsity_layout_y,
                               sparsity_layout_o, sparsity_block_size)

    # Apply element-wise operation
    o_sparse = torch.add(o_sparse, 1)

    o_dense = bs.ops.to_dense(o_sparse, sparsity_layout_o, sparsity_block_size)

    # Sanity check
    o_torch = torch.matmul(x_dense, y_dense)
    o_torch = torch.add(o_torch, 1)

    # Perform round trip to set sparse blocks to 0
    o_torch_round_trip = bs.ops.to_dense(
        bs.ops.to_sparse(o_torch, sparsity_layout_o, sparsity_block_size),
        sparsity_layout_o, sparsity_block_size, fill_value=0)

    # Assert that the output is correct
    assert torch.allclose(o_dense, o_torch_round_trip, atol=2e-2)  # Note that small numerical differences are expected

    # Assert that the output has the correct sparsity layout
    actual_sparsity_layout_o = bs.layouting.build_sparsity_layout(o_dense, sparsity_block_size)
    assert torch.equal(actual_sparsity_layout_o, sparsity_layout_o)

    # Convert output tensor back to original shape
    o = bs.utils.undo_shape_blocksparse(o_dense, x_shape_original)
    assert o.shape == (b, h, m, n)

    # Other available functions
    bs.ops.transpose(o_sparse, sparsity_layout_o, sparsity_block_size)
    bs.ops.softmax(o_sparse, sparsity_layout_o, sparsity_block_size, flag_fused=False)
    bs.ops.softmax_fused(o_sparse, sparsity_layout_o,
                         sparsity_block_size)  # Explicit fused execution; raises above 131,072 padded row elements
    bs.ops.softmax(o_sparse, sparsity_layout_o,
                   sparsity_block_size)  # Fused by default with automatic fallback for oversized rows
    bs.ops.misc.row_wise_sum(o_sparse, sparsity_layout_o, sparsity_block_size)
    bs.ops.misc.row_wise_max(o_sparse, sparsity_layout_o, sparsity_block_size)

    # Flash Attention
    seq_len, head_dim = 512, 64
    sparsity_block_size_attn = 64

    q = torch.randn(b, seq_len, h, head_dim, device="cuda")
    k = torch.randn(b, seq_len, h, head_dim, device="cuda")
    v = torch.randn(b, seq_len, h, head_dim, device="cuda")

    # Flash Attention expects (batch * heads, seq_len, head_dim)
    q_dense = q.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()
    k_dense = k.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()
    v_dense = v.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()

    n_batches_attn = b * h
    n_seq_blocks = seq_len // sparsity_block_size_attn
    n_head_blocks = head_dim // sparsity_block_size_attn

    sparsity_layout_qkv = torch.ones(
        n_batches_attn, n_seq_blocks, n_head_blocks,
        device="cuda", dtype=torch.bool,
    )
    attention_layout = torch.tril(torch.ones(
        n_batches_attn, n_seq_blocks, n_seq_blocks,
        device="cuda", dtype=torch.bool,
    ))
    sparsity_layout_o = bs.layouting.build_sparsity_layout_matmul(attention_layout, sparsity_layout_qkv)

    q_sparse = bs.ops.to_sparse(q_dense, sparsity_layout_qkv, sparsity_block_size_attn)
    k_sparse = bs.ops.to_sparse(k_dense, sparsity_layout_qkv, sparsity_block_size_attn)
    v_sparse = bs.ops.to_sparse(v_dense, sparsity_layout_qkv, sparsity_block_size_attn)

    # Pre-build reusable layout cache data for repeated calls with the same layouts
    flash_layout_cache = bs.ops.flash_attention_build_layout_cache(
        attention_layout,
        sparsity_layout_q=sparsity_layout_qkv,
        sparsity_layout_k=sparsity_layout_qkv,
        sparsity_layout_v=sparsity_layout_qkv,
        n_seq_blocks_q=n_seq_blocks,
        n_seq_blocks_k=n_seq_blocks,
        n_head_blocks=n_head_blocks,
        sparsity_layout_o=sparsity_layout_o,
    )

    attn_out_sparse = bs.ops.flash_attention(
        q_sparse, sparsity_layout_qkv,
        k_sparse, sparsity_layout_qkv,
        v_sparse, sparsity_layout_qkv,
        attention_layout, sparsity_block_size_attn,
        layout_cache=flash_layout_cache,
        sparsity_layout_o=sparsity_layout_o,
    )
    attn_out_dense = bs.ops.to_dense(attn_out_sparse, sparsity_layout_o, sparsity_block_size_attn)
    attn_out = attn_out_dense.reshape(b, h, seq_len, head_dim).transpose(1, 2).contiguous()

    assert attn_out.shape == (b, seq_len, h, head_dim)


def _get_random_sparsity_layout(b, m, n, sparsity_block_size, sparsity_percentage):
    """Creates a random sparsity layout with the requested percentage of inactive blocks."""
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), device="cuda", dtype=torch.bool)

    num_zero_elements = int(m_s * n_s * (sparsity_percentage / 100))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = False

    return sparsity_layout
```

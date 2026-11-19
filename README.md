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
- Flash Attention (_supports custom masks, cross-attention, and fused query-relative embeddings_)

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
| Regular and fused softmax | `float32`; under CUDA autocast, `float16` and `bfloat16` inputs are computed and returned in `float32` |
| Gather and scatter indices | `int32`, `int64` |
| Flash Attention masks | `bool`, `float16`, `bfloat16`, `float32`; values must be binary |
| Conversion, transpose, repeat, partitioning, gather, and non-reducing scatter data | `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64`; input dtype is preserved |
| Summation scatter data | `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64` |
| Broadcast addition operands | Matching dtypes from the generic data-movement set above; `bool` addition is the logical or |
| Broadcast subtraction operands | Matching numeric dtypes from the generic data-movement set above |
| Sparsity and attention layouts | `bool` is recommended; integer and real floating-point layouts must contain only `0` and `1`; complex layouts are rejected |

Under CUDA autocast, matrix multiplication, broadcast arithmetic, and Flash Attention cast their ``float16``,
``bfloat16``, and ``float32`` inputs to the autocast dtype, softmax computes in ``float32`` as listed above, and
all other operations keep the dtype of their inputs. Inputs to an individual
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

The library covers our current needs. Version 2.6.0 adds block-sparse Flash Attention with learned relative embeddings,
native causal lengths, and causal layout builders; beyond these, there are no plans to add new features, such as support
for dimensions other than the last in the ``split`` and ``merge`` operations. We will continue to maintain the library
and fix issues that arise.

If you find a bug, please open an [issue](https://github.com/FelixSchoen/blksprs/issues).
We also encourage [pull requests](https://github.com/FelixSchoen/blksprs/pulls).

This scope may change for future projects; as of November 2026, the library meets our current requirements.

## ⚠️ Known Limitations and Issues

- Block-sparse and dense reference operations can differ slightly because their reduction order, precision, and kernel
  implementations differ. Appropriate tolerances depend on the dtype, tensor sizes, operation, and application; validate
  numerical error against the requirements of your workload.

- Compressed and dense data operands accepted by automatic contiguous conversion are normalised before kernel dispatch.
  Sparsity-layout tensors are structural metadata and must be contiguous when supplied by the caller.

- Floating-point operations that use atomic GPU reductions, including summation scatter with overlapping indices, may
  not be bitwise deterministic because the accumulation order can vary.

- ``torch.compile`` is supported in its default, graph-break-tolerant mode, but not with ``fullgraph=True``. Validation,
  layout-cache preparation, and Flash Attention run eagerly and break the graph, so compiling a model that calls
  BLK-SPRS does not necessarily improve its performance.

## 💻 Usage

### Terminology

Block-sparse tensors are stored in compressed form: only active blocks are present in the tensor data. The accompanying
``sparsity_layout`` describes the full block layout.

Derived layout metadata is passed through an optional ``layout_cache`` dictionary. Internally, ``layout_indices`` map
compressed block positions to coordinates in the full sparsity layout, while ``packed_indices`` map full sparsity-layout
positions back to compressed block positions or ``-1`` for inactive blocks. Flash Attention additionally caches tile
schedules of the attention pattern, and learned relative attention the metadata derived from its coordinates.

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

Changes are detected through PyTorch's version counters. In-place writes that bypass them, such as writes through
``tensor.data`` or by custom kernels that write the raw storage, are not detected: the cache then returns stale
metadata, for example the prepared coordinates of learned relative attention. Pass new tensors or call
``layout_cache.clear()`` after such writes.

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
16, 32, and 64; larger blocks are rejected because the kernel cannot fit them portably into GPU shared memory. The
kernels autotune their query and key tiles independently of the sparsity block size: a tile spans one or more sparsity
blocks and gathers the blocks that its queries attend to, so that sparse layouts and short sparsity blocks retain the
efficiency of large tiles.

For standard padded causal self-attention, ``bs.layouting`` provides
``build_causal_self_attention_layout()`` and
``build_causal_self_attention_mask()``. Their ``valid_lengths`` input preserves
each sequence's independent valid prefix, while the compact mask stores only
blocks that need causal or padding refinement. The corresponding
``build_causal_window_self_attention_layout()`` and
``build_causal_window_self_attention_mask()`` helpers implement an exact
token-local causal window. These helpers describe sequence structure only and
are usable with any compatible block-sparse attention model.

Flash Attention can apply full causal and padding refinement directly in the
kernel by receiving ``causal_lengths``, one integral valid length per flattened
attention batch. In that form, use ``build_causal_self_attention_layout()`` for
the block pattern and omit the causal ``attention_mask`` entirely. This avoids
mask storage and lookup while preserving zero output for padded query rows.
``causal_lengths`` can also be combined with an explicit mask when the caller
needs additional restrictions; causal-window attention still needs its compact
window-boundary mask.

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

``bs.ops.flash_attention_relative_embedding()`` adds one learned relation term. Supply one integral coordinate per query
and key position plus an embedding table for an inclusive relation range; the operation adds
``q @ R[clip(query_relation - key_relation)]`` to the usual scaled QK score. The implementation projects queries against
the bounded relation table with tensor-core matrix multiplications, and the tiled Flash kernels gather those scores
directly for active blocks. It never constructs a dense sequence-by-sequence tensor. By default
(``score_precalculation="referenced_blocks"``), it computes only the blocks of queries and relation values that a
permitted query-key pair references, that is, a pair in the attention layout, within the causal lengths, and valid for
the relation; ``score_precalculation="complete"`` projects the complete table for every query. The gaps between the
referenced blocks of a query block and relation are stored and computed as well for relations whose tables span at
most eight value blocks, and for all relations if the gaps of a call hold at most one sixty-fourth of its referenced
blocks. The kernels then address such relations through one offset per query block instead of a lookup per query-key
pair; for causal music models, this stored at most 0.1 % additional blocks. The projection is recomputed during
backward so a large projected-score table is not retained for every layer. The reusable Flash Attention layout cache
retains the referenced blocks for calls with the same coordinate, validity, and causal-length tensors, such as the
layers of a model.

``bs.ops.flash_attention_relative_embeddings()`` is the generic multi-relation
form. Coordinates have shape ``(n_relations, attention_batches, sequence)`` and
the relation tables are concatenated along their relation-range dimension. Up
to eight terms are supported. Optional per-position validity tensors suppress a
term when either coordinate is unavailable without reserving a sentinel value
from the integral coordinate domain. Both forms share the compressed-tensor,
mask, output-layout, cache, gradient, and CUDA-autocast contracts of
``flash_attention()``. The flattened attention batch remains model agnostic,
so callers can use independent per-head tables without imposing a head
convention on BLK-SPRS.

When valid key coordinates are pairwise distinct, set
``key_relations_are_unique=True`` for the single-relation form or mark the
corresponding entries in ``key_relations_are_unique`` for the multi-relation
form. This avoids atomic writes for unclipped embedding gradients. These flags
are performance assertions: enabling one for a relation with duplicate valid
key coordinates produces incorrect gradients. Only the keys of permitted pairs count: masked keys and keys beyond the
causal lengths, such as padding keys, may repeat a coordinate.

``key_relations_repeat`` marks the relations whose consecutive keys mostly share their coordinates, such as onsets or
bar positions. The query-gradient kernel handles these relations once per query and run of keys with equal
coordinates of all marked relations, in the key tiles that hold few such runs, and per pair in all others. By default,
every relation without unique keys is marked. A relation whose keys change at almost every key, such as a pitch class,
should not be marked, since its runs would leave the tiles to the per-pair path; the results do not depend on the
marks beyond the order of summation.

Distant key tiles of causal layouts often clip a relation to its maximum for every pair, so that the kernels take its
score from a single table column. The kernels handle the tiles that clip the first two relations, or the only one,
with a specialised loop, which takes these from their maximum and gathers the others. List the relations that clip
most often first, such as positions and times before phases or pitch classes.

For float32 operands, the Flash Attention kernels compute their matrix products with TF32 tensor cores, Triton's
default, regardless of ``torch.backends.cuda.matmul.allow_tf32``; float16 and bfloat16 operands use their native
products with float32 accumulation. The projection of learned relative scores uses full float32 products for float32
operands, as ``torch.bmm`` does by default. ``bs.ops.matmul`` follows PyTorch's float32 precision setting like
``torch.matmul``: float32 products use full precision by default and TF32 tensor cores once TF32 is allowed, for example
with ``torch.set_float32_matmul_precision("high")``.

### Validation and contiguous conversion

Input validation and automatic contiguous conversion for supported data operands are enabled by default and configured
process-wide, so that the settings also apply to autograd backward passes. Sparsity layouts are caller-managed structural metadata and must be contiguous. These safeguards can be
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

BLK-SPRS operations are registered as PyTorch Triton operators, so that ``torch.compile`` traces their kernels into the
compiled graph. In a function compiled in the default mode, every public operation matches eager execution in its
forward and backward pass, also after recompilations for new shapes, layouts, or block sizes, and in-place updates of a
layout between compiled calls take effect. Validation and the preparation of layout caches run eagerly and break the
graph, so ``fullgraph=True`` is not supported. The Flash Attention functions are not traced: they run eagerly within
compiled functions, with the kernels that Triton compiles for them as in eager execution. Backward passes that
synchronise with the host fall back to eager execution, which PyTorch logs as a graph break.

In eager execution, compressed outputs are marked with the ``BlksprsTensor`` subclass. In compiled code, compressed
outputs may instead be regular ``torch.Tensor`` instances because tensor subclasses cannot be constructed inside the
compiled graph; gradients are always regular tensors. This does not change their compressed representation, and all
BLK-SPRS operations accept either form.

Use eager BLK-SPRS operations by default. If the surrounding PyTorch model already benefits from ``torch.compile``, it
can continue to contain BLK-SPRS calls, but benchmark the complete workload rather than assuming that compiling improves
its performance.

### Autotuning

``BLKSPRS_AUTOTUNE`` selects the Triton autotuning profile. Set it before importing ``blksprs``:

```bash
BLKSPRS_AUTOTUNE=DEFAULT python train.py
BLKSPRS_AUTOTUNE=TEST python -m pytest test/cases
```

``DEFAULT`` explores the production configuration set. ``TEST`` uses a smaller configuration set to reduce test startup
time. Any other value is rejected during import. The test suite deselects the high-memory and high-runtime
``benchmark`` tests by default; select them with ``-m benchmark``.

Triton compiles every kernel variant on first use, which takes seconds to minutes for the Flash Attention kernels, and
benchmarks the configurations of an autotuned kernel for every new autotuning key. The Flash Attention kernels compile
one variant per configuration and combination of the static properties of a call: the dtype, the head dimensions, the
block size, the mask and bias, the causal lengths, the relation validity, the number of relations, their unique and
repeated keys, and the score pre-calculation. The data of a call, such as its lengths, coordinates, and layouts, do
not change the compiled code: they only decide at run time which of the compiled variants a call launches, whether
its relations are all addressed through intervals, and which of the compiled loops a tile takes, so that a model
compiles a fixed set of variants. The autotuning key also holds power-of-two buckets of the sequence
length and of the number of tile-schedule entries, which grows with the number and the density of the attention
batches of a call, whose first occurrences benchmark the compiled configurations again without compiling. Three
properties of the data still change the compiled code: relation coordinates beyond ``2**29`` in magnitude, which use
64-bit coordinates; calls whose operands, including the projected relative scores, exceed ``2**31`` elements, which
use 64-bit indexing; and operand layouts in which a sequence block holds only some of its head-dimension blocks.

``bs.ops.flash_attention_warm_up()`` compiles every variant of a static call configuration and tunes it for given
shapes ahead of training. For production training, and in particular for distributed training, where a rank that
compiles or tunes stalls the others:

```bash
# A persistent Triton cache shared by all ranks and jobs, so that compiled kernels are reused across processes.
export TRITON_CACHE_DIR=/persistent/shared/triton-cache
# Persist autotuning results in this cache, so that later processes skip the benchmarking.
export TRITON_CACHE_AUTOTUNING=1
```

and warm up before the first distributed step on every rank, each on its own and without collectives, so that no rank
waits in a collective, with its timeout, while another compiles; the ranks load what another rank has compiled from
the shared cache. The warm-up needs the static configuration of the calls of training, including the dtype of an
explicit mask (``attention_mask=torch.bool``) and whether calls pass an attention bias (``attention_bias``); a forward
and backward pass of the model on one real batch covers it as well, but only tunes the buckets of that batch:

```python
import torch

import blksprs as bs

# The static configuration of the calls of a model: 64-dimensional heads in float16 under autocast, three relations,
# causal lengths and relation validity, for the attention batches (sequences times heads) and lengths of training.
bs.ops.flash_attention_warm_up(
    32, 64, torch.float16, [(8 * 16, 1024), (8 * 4, 4096), (8, 16384)],
    relation_bounds=[(0, 2048), (0, 3072), (-96, 96)], key_relations_are_unique=[True, False, False],
    key_relations_repeat=[False, True, True], relation_validity=True)
```

Autotuning results are keyed by the Triton version, the GPU architecture, the kernel source, and the autotuning key,
so that they are reused only for identical kernels and GPUs of the same architecture.

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
        sparsity_block_size=sparsity_block_size_attn,
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

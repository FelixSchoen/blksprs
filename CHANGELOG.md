# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.0] - 2026-08-03

### Added

- Add package namespace exports, deterministic regression tests, and installed-artifact validation

### Changed

- Harden public shape, device, dtype, dimension, index, layout, and contiguity validation
- Invalidate conversion, linear, and Flash Attention layout caches after input changes while retaining source tensors to
  prevent allocator identity reuse
- Preserve 64-bit indices throughout kernels and dynamically use 64-bit Flash Attention adjacency metadata when required
- Limit Flash Attention sparsity blocks to the portable supported range of 16 to 64 elements
- Publish packages through pinned GitHub Actions and PyPI trusted publishing
- Gate PyPI publication on static checks, minimum-PyTorch compatibility, and installed distribution tests across all
  supported Python versions, and publish the exact artifacts that passed those checks
- Expand API, cache, mask, dtype, installation, and safety documentation
- Clarify that public wrappers support graph-break-tolerant ``torch.compile`` execution rather than ``fullgraph=True``
- Test compatibility with the declared minimum PyTorch release and verify versions from built release artifacts
- Benchmark repeated attention calls with reusable layout caches and skip overflow tests unless sufficient memory is free
- Reduce regular-softmax peak storage by normalising its exponentiated output in place
- Preserve positional compatibility for the existing Flash Attention cache argument while extending optional metadata
- Make the Ruff rule selection explicit so release linting remains stable across tooling updates
- Polish README, package metadata, and API documentation wording and formatting

### Fixed

- Preserve float32 Flash Attention accumulators across key/query blocks to prevent long-sequence reduced-precision
  forward corruption and gradient saturation
- Preserve float32 accumulators for row-wise sums, scatter reductions, and repeated-tensor gradients to prevent
  reduced-precision saturation on large reductions
- Preserve autograd through row-striped conversions with no active blocks
- Ensure Flash Attention masks remain authoritative over non-finite additive bias values
- Rebuild cached sparse linear parameters when switching between gradient-disabled evaluation and training
- Match ``torch.nn.Linear`` mixed-dtype behavior outside autocast and clarify compiled compressed-tensor return types
- Preserve ``float64`` and unsupported floating dtypes at the autocast boundary instead of silently downcasting them
- Apply automatic contiguous conversion to public layout-builder data operands and document structural-layout requirements
- Document that non-reducing scatter does not support a backward pass
- Fix stale layout metadata after in-place mutations or cache reuse across operations
- Invalidate cached sparse linear parameters when their gradient requirements change
- Fix stale cache metadata after mutating derived layouts and prevent transformed output layouts from aliasing inputs
- Fix non-contiguous output gradients across differentiable operations
- Fix Flash Attention mask, bias, sparse Q/K/V, mixed-dimension, and empty-row forward and backward behavior
- Fix the default Flash Attention output layout silently discarding V blocks that are absent from the Q layout
- Fix scatter compilation for non-reducing Boolean and compact-integer inputs and reject unsupported summation dtypes clearly
- Fix gather and scatter silently accepting oversized non-indexed dimensions
- Reject negative distribution-layout target dimensions before kernel allocation
- Fix inference-created tensors failing layout-cache preparation and rebuild their metadata safely
- Fix large numeric layouts overflowing block counts in conversion and Flash Attention caches
- Reject complex-valued sparsity layouts explicitly instead of warning during metadata conversion
- Fix sparse linear bias addition promoting autocast output back to full precision
- Fix row-wise maxima for all-negative-infinity rows and reject silent mixed-dtype row-wise arithmetic
- Fall back to regular softmax before fused row sizes cause pathological Triton compilation
- Fix softmax for empty compressed inputs and dense conversion returning the compressed tensor subclass
- Fix layout matmul, row-wise reductions, and default Flash Attention scaling for empty logical dimensions
- Fix layout-matmul helpers silently broadcasting incompatible batches or accepting invalid layouts
- Fix public layout and shape helpers accepting malformed dimensions or CPU layouts, failing incidentally, or rejecting
  empty leading dimensions
- Fix zero-count sparse repeats and their backward passes
- Fix biased sparse linear projections with empty input features, rows, or batches
- Fix row-wise arithmetic on layouts with no column blocks and support rectangular broadcast outputs
- Ensure public layout builders, transformations, reductions, and cache metadata return regular tensors when passed
  block-sparse tensor subclasses
- Forward optional layout caches through ``to_blksprs()``
- Fix the Nix GPU shell overriding PyTorch's bundled CUDA libraries and claiming stale virtual environments were active
- Preserve NaN-bearing blocks when building or adapting sparsity layouts and propagate NaNs through row-wise maxima
- Propagate non-finite Flash Attention inputs and scales instead of silently replacing failed rows with zeros, and reject
  non-real scales at the public boundary
- Validate row-wise callable inputs and outputs and reject inconsistent benchmark argument lengths before execution
- Preserve gradients when layout coarsening zero-pads non-divisible logical extents
- Return slice-only row-wise reductions as regular tensors rather than marking their non-square auxiliary representation
  as a general block-sparse tensor
- Allow an explicitly selected production autotuning profile to pass through the test configuration

## [2.5] - 2026-05-20

### Added

- Add large-index regression coverage for repeat, partitioning, scatter, row-striped conversion, layout helpers, and edge tiles

### Changed

- Rename optional operation cache arguments to `layout_cache`
- Rename operation cache builders to the `*_build_layout_cache()` suffix
- Rename derived layout metadata to `layout_indices`, `packed_indices`, and attention `key_indices`/`query_indices`
- Refactor shared shape, divisibility, and positive-integer validation helpers

### Removed

- Remove the former cache/key naming aliases without compatibility shims

### Fixed

- Fix non-divisible dense shapes being silently truncated by layout builders
- Fix kernel edge-tile masks and masked loads to avoid cross-row and cross-batch accesses
- Fix remaining public CUDA paths for overflow-scale indexing

## [2.4] - 2026-05-07

### Added

- Add large-index regression coverage for sparse conversion, row-wise ops, broadcast ops, layout builders, softmax, and flash attention

### Changed

- Update README usage example and README test to document `torch.compile` usage explicitly
- Harden public sparse, misc, and layout-building kernels with guarded `int32`/`int64` indexing dispatch
- Isolate hardware-sensitive performance and overflow checks behind the `benchmark` pytest marker

### Fixed

- Fix address-overflow bugs in block-sparse kernels for large flattened index spaces
- Fix mixed-length batched Longformer evaluation corruption caused by incorrect sparse mask conversion at overflow scale
- Fix silent CUDA public-API downcasting so caller dtypes are preserved unless autocast is active
- Fix row-wise, broadcast, and layout-building helper correctness for overflow-scale inputs
- Fix release verification to use the in-tree `2.4` package state consistently

## [2.3.2] - 2026-04-08

### Added

- Add shaped conversion convenience wrappers `to_sparse_shaped()` and `to_dense_shaped()`
- Add row-striped conversion helpers `is_row_striped_layout()`, `to_sparse_row_striped()`, and `to_dense_row_striped()`
- Add `apply_torch_linear_cached()` utility for repeated sparse linear projections with cached packed weights

### Changed

- Expose the new conversion helpers via `bs.ops`
- Expose `apply_torch_linear_cached()` via `bs.utils`
- Optimize dense-to-sparse and sparse-to-dense conversion for row-striped sequence-feature layouts

## [2.3.1] - 2026-02-19

### Added

- Expose `enable_contiguous()` and `enable_validation()` in `bs.utils` to re-enable checks after disabling

### Changed

- Expand `flash_attention()` API documentation with detailed argument and mask semantics
- Update README dependency note for PyTorch (`built with v2.10.0`, requires `>= v2.8.0`)

### Fixed

- Fix additive bias gradient scaling in flash attention backward pass
- Fix uninitialized output values in `flow_pull_forward()` by zero-initializing the output buffer
- Fix `softmax` backward wrapper autograd return arity
- Add missing dimension validation checks in `broadcast_add()`

## [2.3] - 2026-02-10

### Added

- Add support for mixed Q/K and V dimensions in block-sparse flash attention
- Add operation-specific autotuning profiles across all kernels

### Changed

- Refactor autotuning prune helpers to reduce duplication

### Fixed

- Update README flash attention test usage to current API
- Fix flash attention output layout handling for mixed model dimensions

## [2.2] - 2026-02-04

### Added

- Implement block-sparse flash attention

### Fixed

- Fix small issues

## [2.1.10] - 2026-01-30

### Added

- Add flake.nix file for Nix users
- Add support for `dim==1` for `stride()` helper

## [2.1.9] - 2025-08-11

### Fixed

- Fix Triton deprecation warnings

## [2.1.8] - 2025-08-07

### Changed

- Rework wrapper class `BlksprsTensor` to only wrap when not using `torch.compile()`

### Fixed

- Remove workaround for fix introduced in Triton v2.4.0, see [triton-lang/triton#6376](https://github.com/triton-lang/triton/issues/6376)

## [2.1.7] - 2025-08-01

### Fixed

- Fix `ensure_contiguous()` when skipping

## [2.1.6] - 2025-08-01

### Fixed

- Fix functions not making use of `ensure_contiguous()`

## [2.1.5] - 2025-07-30

### Added

- Add `disable_contiguous()` to `bs.utils`

### Changed

- Rework sparsity packed-index access
- Refactor `disable_validation()` to `bs.utils`

### Fixed

- Fix layout cache validation for repeat functions

## [2.1.4] - 2025-06-27

### Changed

- Update README.me

### Fixed

- Fix wrong version number for `bs.__version`

## [2.1.3] - 2025-06-25

### Fixed

- Fix `softmax_fused()` function not working correctly with sparsity layouts with odd dimensions

## [2.1.2] - 2025-06-22

### Changed

- Improve speed of `softmax_fused()` function by computing the length of the longest needed chain of non-sparse blocks

## [2.1.1] - 2025-06-20

### Added

- Add `softmax_fused()` function for matrices whose rows fit into memory

## [2.0] - 2025-06-09

### Added

- Add option to pre-build layout cache data for most operations, improving performance for repeated operations with same sparsity
  layouts

### Changed

- Rework all operations to use new `torch.library.triton_op()` approach, allowing for JIT compilation and better
  compatibility
- Rework kernels to work with triton block sizes larger than sparsity block sizes via masking
- Rework kernels to use automatic tuning of triton block sizes rather than fixed block sizes
- Rework operations to support dtype autocasting

### Removed

- Remove manual specification of triton block sizes

## [1.11] - 2025-03-10

### Added

- Add ``build_layout_cache()`` method for most operations, allowing for faster execution when precomputing layout cache data

### Changed

- Adapt transpose to use flow kernel

## [1.10.2] - 2024-12-28

### Fixed

- Fix small performance issues

## [1.10.1] - 2024-12-17

### Fixed

- Fix return type of ``to_dense``

## [1.10] - 2024-12-14

### Changed

- Change dtype of sparsity layouts from ``int32`` to ``int1``
- Change signature of ``split`` and ``merge`` to contain placeholder ``dim`` parameter

### Fixed

- Fix masking bounds of kernels not being tight

### Removed

- Remove ``bs.ops.experimental`` module

## [1.9.3] - 2024-12-03

### Fixed

- Fix deprecation warning for ``resize``
- Fix representation of ``BlksprsTensor``

## [1.9.1] - 2024-11-07

### Added

- Add ``bs.utils.apply_torch_normalisation`` wrapper function
- Add ``bs.utils.apply_torch_dropout`` wrapper function
- Add ``bias`` parameter to ``apply_torch_linear``
- Implement target layout specification for ``adapt_layout``

### Fixed

- Fix ``scatter`` not passing ``dim``

## [1.9] - 2024-11-04

### Added

- Add ``dim`` parameter for ``gather``, ``scatter_reduce``, and ``build_distribution_layout`` functions

### Fixed

- Add masks for packed block index access, fixing potential memory leaks

## [1.8.3] - 2024-10-31

### Added

- Expose validation functions
- Add ``bs.utils.apply_torch_linear`` function

### Changed

- Move operations to ``bs.ops`` module
- Move miscellaneous operations to ``bs.ops.misc`` module
- Move experimental operations to ``bs.ops.experimental`` module
- Rename ``bs.layout`` to ``bs.layouting``
- Rename ``bs.utils`` to ``bs.utils``

## [1.8.2] - 2024-10-31

### Added

- Add ``BlksprsTensor`` wrapper class to indicate block-sparse tensors

## [1.8.1] - 2024-10-29

### Fixed

- Fix ``build_sparsity_layout_fast`` not being exposed

## [1.8] - 2024-10-28

### Added

- Add validation for ``to_sparse`` for input dense tensors
- Add alias for ``to_sparse`` and ``to_dense`` functions
- Add documentation for ``repeat``
- Add gradient calculation for ``repeat_interleave``

### Changed

- Change ``repeat_interleave`` to use same flow kernels as ``repeat`` does

## [1.7] - 2024-10-28

### Added

- Add ``repeat`` function

### Fixed

- Fix kernels not returning on illegal sparse blocks
- Fix stride not being calculated correctly in some cases

## [1.6.1] - 2024-10-23

### Added

- Add ``build_sparsity_layout_matmul`` function
- Add ``build_sparsity_layout_matmul_fast`` function

### Fixed

- Fixed project version number

## [1.6] - 2024-10-22

### Added

- Add ``split`` function
- Add ``merge`` function

### Fixed

- Fixed ``repeat_interleave`` always using 3 repetitions instead of the specified amount

## [1.5] - 2024-10-21

### Added

- Add ``gather_mdi`` and ``scatter_reduce_mdi`` experimental functions

### Changed

- Rename ``gather_3d`` to ``gather_mdi``

## [1.4.3] - 2024-10-18

### Added

- Add experimental ``gather_3d`` function

## [1.4.2] - 2024-10-17

### Fixed

- Fixed output dtype differing from input dtype

## [1.4.1] - 2024-10-08

### Added

- Add ``disable_validation()`` function to disable validation

### Changed

- Change output dtype of sparsity layouts from ``int32`` to ``bool``
- Apply ``.contiguous()`` to tensors for all ops

## [1.4] - 2024-10-02

### Added

- Add ``row_wise_max`` function
- Add ``row_wise_add`` function
- Add ``row_wise_sub`` wrapper function
- Expose functions in ``__init__.py``

### Changed

- Refactor ``row_wise_sum`` function to ``misc`` module
- Change ``row_wise_sum`` to not make use of ``torch.autograd.Function``
- Rename ``broadcast_addition`` module to ``broadcast_ops``
- Rename ``broadcast_addition`` and ``broadcast_subtraction`` to ``broadcast_add`` and ``broadcast_sub``

### Fixed

- Fix ``softmax`` shift trick not using per-row maximum values

## [1.3] - 2024-09-26

### Added

- Add ``repeat_interleave`` function

### Fixed

- Fix ``undo_shape_blocksparse`` function checking the incorrect dimensions

## [1.2.1] - 2024-09-25

### Changed

- Downgrade Python version to 3.11

## [1.2] - 2024-09-20

### Added

- Add ``build_sparsity_layout_adaption`` function to create sparsity layout for adaption of sparsity block size
- Add ``adapt_layout`` function to adapt block-sparse tensor to new sparsity layout and sparsity block size

### Changed

- ``to_sparse`` and ``to_dense`` will no longer return a copy of the input tensor if the input tensor is already in the
  desired format

## [1.1] - 2024-09-19

### Added

- Add validation for sparsity of input for all applicable functions
- Add forward and backward functions for block-sparse ``gather`` operation
- Add forward and backward functions for block-sparse ``scatter_reduce`` operation (gradients only computable with
  ``reduce_op = 'sum'``)
- Add wrapper function ``scatter`` (applies ``scatter_reduce`` with ``reduce_op = 'none'``)
- Add ``build_distribution_layout`` function to create sparsity layout for distribution operations
- Add ``broadcast_addition`` and ``broadcast_subtraction`` functions

### Changed

- Rename ``matmul_sss`` to ``matmul``
- Improve documentation

### Fixed

- Fix memory leaks of backward passes

## [1.0] - 2024-09-13

- Initial release

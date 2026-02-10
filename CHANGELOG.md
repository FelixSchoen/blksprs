# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- Rework lookup of sparsity LUT values
- Refactor `disable_validation()` to `bs.utils`

### Fixed

- Fix LUT validation for repeat functions

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

- Add option to pre-build LUTs for most operations, improving performance for repeated operations with same sparsity
  layouts

### Changed

- Rework all operations to use new `torch.library.triton_op()` approach, allowing for JIT compilation and better
  compatability
- Rework kernels to work with triton block sizes larger than sparsity block sizes via masking
- Rework kernels to use automatic tuning of triton block sizes rather than fixed block sizes
- Rework operations to support dtype autocasting

### Removed

- Remove manual specification of triton block sizes

## [1.11] - 2025-03-10

### Added

- Add ``build_lut()`` method for most operations, allowing for faster execution when precomputing lookup tables

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

- Add masks for reverse block index lookup, fixing potential memory leaks

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

# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
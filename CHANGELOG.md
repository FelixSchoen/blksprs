# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1] - 2024-09-19

### Added

- Add validation for sparsity of input for all applicable functions
- Add forward and backward functions for block-sparse ``gather`` operation
- Add forward and backward functions for block-sparse ``scatter_reduce`` operation (gradients only computable with ``reduce_op = 'sum'``)
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
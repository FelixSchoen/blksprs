"""Block-sparse tensor operations for PyTorch and Triton."""

from blksprs.utils.blksprs_tensor import BlksprsTensor

from . import layouting, ops, utils

__version__ = "2.6.0"

__all__ = ["BlksprsTensor", "layouting", "ops", "utils", "__version__"]

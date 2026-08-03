"""Utilities for block-sparse tensors, validation, and PyTorch modules."""

from . import validation
from .processing import (
    apply_function_applicable_row_wise,
    apply_torch_dropout,
    apply_torch_linear,
    apply_torch_linear_cached,
    apply_torch_normalisation,
)
from .tools import do_shape_blocksparse, undo_shape_blocksparse
from .validation import disable_contiguous, disable_validation, enable_contiguous, enable_validation

__all__ = [
    "apply_function_applicable_row_wise",
    "apply_torch_dropout",
    "apply_torch_linear",
    "apply_torch_linear_cached",
    "apply_torch_normalisation",
    "disable_contiguous",
    "disable_validation",
    "do_shape_blocksparse",
    "enable_contiguous",
    "enable_validation",
    "undo_shape_blocksparse",
    "validation",
]

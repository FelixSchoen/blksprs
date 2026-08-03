"""Operations on compressed block-sparse tensors."""

from . import misc
from .conversion import (
    adapt_layout,
    from_blksprs,
    is_row_striped_layout,
    to_blksprs,
    to_dense,
    to_dense_row_striped,
    to_dense_shaped,
    to_sparse,
    to_sparse_row_striped,
    to_sparse_shaped,
)
from .distribution import gather, scatter, scatter_reduce
from .flash_attention import flash_attention, flash_attention_build_layout_cache
from .matmul import matmul
from .partitioning import merge, split
from .repeat import repeat, repeat_interleave
from .softmax import softmax, softmax_fused
from .transpose import transpose

__all__ = [
    "adapt_layout",
    "flash_attention",
    "flash_attention_build_layout_cache",
    "from_blksprs",
    "gather",
    "is_row_striped_layout",
    "matmul",
    "merge",
    "misc",
    "repeat",
    "repeat_interleave",
    "scatter",
    "scatter_reduce",
    "softmax",
    "softmax_fused",
    "split",
    "to_blksprs",
    "to_dense",
    "to_dense_row_striped",
    "to_dense_shaped",
    "to_sparse",
    "to_sparse_row_striped",
    "to_sparse_shaped",
    "transpose",
]

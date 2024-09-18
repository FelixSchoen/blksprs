import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.transpose import transpose
from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size

def broadcast_addition(x: Tensor, y: Tensor,)
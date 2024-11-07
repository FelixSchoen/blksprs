import math

import torch
import triton
from torch import Tensor
from torch.xpu import device
from triton import language as tl

from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_triton_block_size, validate_dimensions, validate_device, \
    validate_contiguous, validate_sparsity, validate_sparsity_block_size


def build_full_sparsity_layout(x: Tensor, sparsity_block_size: int) -> Tensor:
    return torch.ones(size=(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size),
                      dtype=torch.bool, device=x.device)

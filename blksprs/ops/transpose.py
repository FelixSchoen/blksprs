import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.flow import flow_forward_pull
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size


def transpose(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int, triton_block_size: int = None,
              lut: dict = None) -> (BlksprsTensor, Tensor):
    """Transposes a block-sparse tensor in compressed form.

    Note:
         Returns the transposed tensor and the sparsity layout of the transposed tensor.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The transposed block-sparse tensor in compressed form.
        Tensor: The sparsity layout of the transposed tensor.

    """
    x = x.contiguous()
    x_t = x.transpose(-1, -2).contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    lut = _BlocksparseTranspose.build_lut(lut, sparsity_layout)

    return BlksprsTensor(
        _BlocksparseTranspose.apply(x_t, lut["sparsity_layout_t"], lut["sparsity_lut"], lut["sparsity_reverse_lut"],
                                    sparsity_block_size,
                                    lut["n_sparse_blocks"], triton_block_size)), lut["sparsity_layout_t"]


class _BlocksparseTranspose(torch.autograd.Function):

    @staticmethod
    def build_lut(lut: dict, sparsity_layout: Tensor):
        if lut is None:
            lut = dict()

        if "sparsity_layout_t" not in lut:
            sparsity_layout_t = sparsity_layout.transpose(-1, -2).contiguous()
            lut["sparsity_layout_t"] = sparsity_layout_t

        if "sparsity_lut" not in lut:
            sparsity_lut = torch.nonzero(lut["sparsity_layout_t"]).contiguous()
            lut["sparsity_lut"] = sparsity_lut

        if "sparsity_reverse_lut" not in lut:
            sparsity_layout_flat = sparsity_layout.reshape(-1)
            sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                     (sparsity_layout_flat == 1) -
                                     (1 * (sparsity_layout_flat == 0)))
                                    .reshape(sparsity_layout.size()).transpose(-1, -2).contiguous().reshape(-1))
            lut["sparsity_reverse_lut"] = sparsity_reverse_lut

        if "n_sparse_blocks" not in lut:
            n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
            lut["n_sparse_blocks"] = n_sparse_blocks

        validate_contiguous(lut["sparsity_layout_t"], lut["sparsity_lut"], lut["sparsity_reverse_lut"])

        return lut

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_block_size: int,
                n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        ctx.save_for_backward(sparsity_layout_o)

        return flow_forward_pull(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut,
                                 sparsity_block_size, n_sparse_blocks, triton_block_size)

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.saved_tensors[0]
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return transpose(grad_output, sparsity_layout, sparsity_block_size, triton_block_size)[
            0], None, None, None, None, None, None

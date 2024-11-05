import torch
import triton
from torch import Tensor

from blksprs.ops.flow import kernel_blocksparse_flow_push, flow_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size


def repeat(x: BlksprsTensor, sparsity_layout_x: Tensor, repeats: tuple[int, int, int],
           sparsity_block_size: int, sparsity_layout_output: Tensor = None, triton_block_size: int = None) -> (
        BlksprsTensor, Tensor):
    """Repeats a block-spare tensor in compressed form according to the given repeats.
    
    Repeats is a 3-tuple of integers, where each integer represents the number of times the tensor should be repeated in
        the first, second and third dimension respectively.
        
    Note:
        An output sparsity layout can be provided, in which case only the indicated blocks are filled. This may result
        in blocks not being present in the output that were present in the input if the output sparsity layout indicates
        them to be sparse.
    
    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (tuple[int, int, int]): The number of times the tensor should be repeated in the first, second and
            third dimension respectively.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor): The desired sparsity layout of the output tensor (default ``None``).
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        BlksprsTensor: A block-sparse tensor in compressed form containing the repeated values.
        Tensor: The sparsity layout of the resulting output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_o = sparsity_layout_x.repeat(repeats[0], repeats[1], repeats[2])

    if sparsity_layout_output is not None:
        sparsity_layout_o = torch.logical_and(sparsity_layout_o, sparsity_layout_output)

    sparsity_lut = torch.nonzero(sparsity_layout_o).contiguous()

    sparsity_layout_flat = sparsity_layout_x.reshape(-1)
    sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                             (sparsity_layout_flat == 1) -
                             (1 * (sparsity_layout_flat == 0)))
                            .reshape(sparsity_layout_x.size())
                            .repeat(repeats[0], repeats[1], repeats[2])
                            .reshape(-1).contiguous())

    n_sparse_blocks = torch.sum(sparsity_layout_o.to(torch.int)).item()

    validate_contiguous(sparsity_layout_o, sparsity_lut, sparsity_reverse_lut)

    return BlksprsTensor(
        _BlocksparseRepeat.apply(x, sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut,
                                 sparsity_block_size, n_sparse_blocks, triton_block_size)), sparsity_layout_o


def repeat_interleave(x: BlksprsTensor, sparsity_layout_x: Tensor, repeats: int,
                      sparsity_block_size: int, sparsity_layout_output: Tensor = None,
                      triton_block_size: int = None) -> (
        BlksprsTensor, Tensor):
    """Repeats and interleaves the block-sparse tensor in compressed form.

    Repeats each matrix contained in the tensors by ``repeats`` amount and places them consecutively in the output
        tensor.

    Note:
        In similar fashion to the regular ``repeat`` an output sparsity layout can be provided. In this case only
        non-sparse blocks will be filled.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (int): The number of times to repeat the matrices.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor): The desired sparsity layout of the output tensor (default ``None``).
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        BlksprsTensor: A block-sparse tensor in compressed form containing the repeated and interleaved matrices.
        Tensor: The sparsity layout of the resulting output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_o = torch.repeat_interleave(sparsity_layout_x, repeats, dim=0).contiguous()

    if sparsity_layout_output is not None:
        sparsity_layout_o = torch.logical_and(sparsity_layout_o, sparsity_layout_output)

    sparsity_lut = torch.nonzero(sparsity_layout_o).contiguous()

    sparsity_layout_flat = sparsity_layout_x.reshape(-1)
    sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                             (sparsity_layout_flat == 1) -
                             (1 * (sparsity_layout_flat == 0)))
                            .reshape(sparsity_layout_x.size())
                            .repeat_interleave(repeats, dim=0)
                            .reshape(-1).contiguous())

    n_sparse_blocks = torch.sum(sparsity_layout_o.to(torch.int)).item()

    validate_contiguous(sparsity_layout_o, sparsity_lut, sparsity_reverse_lut)

    return BlksprsTensor(
        _BlocksparseRepeat.apply(x, sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut,
                                 sparsity_block_size, n_sparse_blocks, triton_block_size)), sparsity_layout_o


class _BlocksparseRepeat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor,
                sparsity_reverse_lut: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int,
                triton_block_size: int) -> Tensor:
        ctx.save_for_backward(sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut)
        ctx.x_size = x.size()
        ctx.x_stride = stride(x)

        return flow_forward(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                            n_sparse_blocks, triton_block_size)

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut = ctx.saved_tensors
        x_size = ctx.x_size
        x_stride = ctx.x_stride
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        n_sparse_blocks = torch.sum(sparsity_layout_x.to(torch.int)).item()

        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=grad_output.dtype, device=grad_output.device)

        x_b, x_r, x_c = grad_output.size()
        x_b_s, x_r_s, x_c_s = stride(grad_output)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_o.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_o)
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
        o_b, o_r, o_c = x_size
        o_b_s, o_r_s, o_c_s = x_stride

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        (kernel_blocksparse_flow_push[triton_grid]
         (grad_output,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          triton_block_size))

        return output, None, None, None, None, None, None, None

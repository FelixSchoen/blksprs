import torch
import triton
from triton import language as tl
from torch import Tensor

from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size


def repeat(x: Tensor, sparsity_layout_x: Tensor, repeats: tuple[int, int, int],
           sparsity_block_size: int, sparsity_layout_output: Tensor = None, triton_block_size: int = None) -> (
        Tensor, Tensor):
    """Repeats a block-spare tensor in compressed form according to the given repeats.
    
    Repeats is a 3-tuple of integers, where each integer represents the number of times the tensor should be repeated in
        the first, second and third dimension respectively.
        
    Note:
        An output sparsity layout can be provided, in which case only the indicated blocks are filled. This may result
        in blocks not being present in the output that were present in the input if the output sparsity layout indicates
        them to be sparse.
    
    Args:
        x (Tensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (tuple[int, int, int]): The number of times the tensor should be repeated in the first, second and
            third dimension respectively.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor): The desired sparsity layout of the output tensor (default ``None``).
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        Tensor: A block-sparse tensor in compressed form containing the repeated values.
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

    return _BlocksparseRepeat.apply(x, sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut,
                                    sparsity_block_size, n_sparse_blocks, triton_block_size), sparsity_layout_o


def repeat_interleave(x: Tensor, sparsity_layout_x: Tensor, repeats: int,
                      sparsity_block_size: int, sparsity_layout_output: Tensor = None,
                      triton_block_size: int = None) -> (
        Tensor, Tensor):
    """Repeats and interleaves the block-sparse tensor in compressed form.

    Repeats each matrix contained in the tensors by ``repeats`` amount and places them consecutively in the output
        tensor.

    Note:
        In similar fashion to the regular ``repeat`` an output sparsity layout can be provided. In this case only
        non-sparse blocks will be filled.

    Args:
        x (Tensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (int): The number of times to repeat the matrices.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor): The desired sparsity layout of the output tensor (default ``None``).
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        Tensor: A block-sparse tensor in compressed form containing the repeated and interleaved matrices.
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

    return _BlocksparseRepeat.apply(x, sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut,
                                    sparsity_block_size, n_sparse_blocks, triton_block_size), sparsity_layout_o


class _BlocksparseRepeat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor,
                sparsity_reverse_lut: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int,
                triton_block_size: int) -> Tensor:
        ctx.save_for_backward(sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut)
        ctx.x_size = x.size()
        ctx.x_stride = stride(x)

        return forward_flow(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
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


@triton.jit
def kernel_blocksparse_flow_pull(x,
                                 x_b, x_b_s, x_r_s, x_c_s,
                                 o,
                                 o_b, o_b_s, o_r_s, o_c_s,
                                 s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                 s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                 r_lut,
                                 TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get sparsity index of current output block consisting of its batch, row, and column index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
    spa_col_msk = (spa_col_idx < s_lut_r * s_lut_r_s)
    spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

    # Get reverse sparsity index
    rev_idx_spa_idx = (spa_bat * s_l_o_b_s +
                       spa_row * s_l_o_r_s +
                       spa_col * s_l_o_c_s)
    rev_idx_spa_msk = (rev_idx_spa_idx < s_l_o_b * s_l_o_b_s)
    rev_idx_spa = tl.load(r_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    if rev_idx_spa == -1:
        tl.device_assert(False)
        return

    blk_x_idx = (rev_idx_spa * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    blk_o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = (blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


@triton.jit
def kernel_blocksparse_flow_push(x,
                                 x_b, x_b_s, x_r_s, x_c_s,
                                 s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                                 s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                 r_lut,
                                 o,
                                 o_b, o_b_s, o_r_s, o_c_s,
                                 TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get sparsity index of current input block consisting of its batch, row, and column index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
    spa_col_msk = (spa_col_idx < s_lut_r * s_lut_r_s)
    spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

    # Get reverse sparsity index
    rev_idx_spa_idx = (spa_bat * s_l_x_b_s +
                       spa_row * s_l_x_r_s +
                       spa_col * s_l_x_c_s)
    rev_idx_spa_msk = (rev_idx_spa_idx < s_l_x_b * s_l_x_b_s)
    rev_idx_spa = tl.load(r_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    if rev_idx_spa == -1:
        tl.device_assert(False)
        return

    blk_x_idx = (pid_blk * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    blk_o_idx = (rev_idx_spa * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = (blk_o_idx < o_b * o_b_s)
    tl.atomic_add(o + blk_o_idx, blk_x, mask=blk_o_msk)


def forward_flow(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                 sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
    output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                         dtype=x.dtype, device=x.device)
    output = torch.zeros_like(output)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = stride(x)
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = stride(output)
    s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
    s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)
    s_lut_r, s_lut_c = sparsity_lut.size()
    s_lut_r_s, s_lut_c_s = stride(sparsity_lut)

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    triton_grid = lambda meta: [o_b,
                                triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_blocksparse_flow_pull[triton_grid]
     (x,
      x_b, x_b_s, x_r_s, x_c_s,
      output,
      o_b, o_b_s, o_r_s, o_c_s,
      s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
      sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
      sparsity_reverse_lut,
      triton_block_size))

    # Save for backward pass
    ctx.sparsity_block_size = sparsity_block_size
    ctx.triton_block_size = triton_block_size

    return output

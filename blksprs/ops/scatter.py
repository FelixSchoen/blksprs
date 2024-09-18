import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.gather import gather
from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_dtype_float, validate_device, \
    validate_sparsity, validate_dtype_int


def scatter(x: Tensor, sparsity_layout_x: Tensor,
            i: Tensor, sparsity_layout_i: Tensor,
            sparsity_layout_output: Tensor,
            sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    return scatter_reduce(x, sparsity_layout_x,
                          i, sparsity_layout_i,
                          sparsity_layout_output,
                          sparsity_block_size,
                          reduce_op="none", triton_block_size=triton_block_size)


def scatter_reduce(x: Tensor, sparsity_layout_x: Tensor,
                   i: Tensor, sparsity_layout_i: Tensor,
                   sparsity_layout_output: Tensor,
                   sparsity_block_size: int,
                   reduce_op: str = "sum", triton_block_size: int = None) -> Tensor:
    validate_dimensions(x, i)
    validate_contiguous(x, i)
    validate_dtype_float(x)
    validate_dtype_int(i)
    validate_device(x, i)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x), (i, sparsity_layout_x))

    if reduce_op not in ["none", "sum"]:
        raise ValueError(f"Reduction operation '{reduce_op}' is not supported")

    sparsity_lut_x = torch.nonzero(sparsity_layout_x).contiguous()

    sparsity_layout_o_flat = sparsity_layout_output.reshape(-1)
    sparsity_reverse_lut_o = ((torch.cumsum(sparsity_layout_o_flat, dim=-1) - 1) *
                              (sparsity_layout_o_flat == 1) -
                              (1 * (sparsity_layout_o_flat == 0)))

    n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout_x, sparsity_lut_x,
                        sparsity_layout_output, sparsity_reverse_lut_o)

    return _BlocksparseScatterReduce.apply(x, sparsity_lut_x,
                                           i, sparsity_layout_i,
                                           sparsity_layout_output, sparsity_reverse_lut_o,
                                           sparsity_block_size, n_sparse_blocks,
                                           reduce_op, triton_block_size)


class _BlocksparseScatterReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_lut_x: Tensor,
                i: Tensor, sparsity_layout_i: Tensor,
                sparsity_layout_o: Tensor, sparsity_reverse_lut_o: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int,
                reduce_op: str, triton_block_size: int) -> Tensor:
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_lut_x_r, s_lut_x_c = sparsity_lut_x.size()
        s_lut_x_r_s, s_lut_x_c_s = sparsity_lut_x.stride()
        i_b, i_r, i_c = i.size()
        i_b_s, i_r_s, i_c_s = i.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = sparsity_layout_o.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        reduce_op_ind = 0
        if reduce_op == "sum":
            reduce_op_ind = 1

        (_BlocksparseScatterReduce.kernel_blocksparse_scatter[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
          i,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
          sparsity_reverse_lut_o,
          reduce_op_ind,
          sparsity_block_size,
          triton_block_size))

        ctx.save_for_backward(i)
        ctx.sparsity_layout_i = sparsity_layout_i
        ctx.sparsity_layout_o = sparsity_layout_o
        ctx.sparsity_block_size = sparsity_block_size
        ctx.reduce_op = reduce_op
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sparsity_layout_i = ctx.sparsity_layout_i
        sparsity_layout_o = ctx.sparsity_layout_o
        sparsity_block_size = ctx.sparsity_block_size
        reduce_op = ctx.reduce_op
        triton_block_size = ctx.triton_block_size

        if reduce_op == "sum":
            gather(grad_output, sparsity_layout_o, i, sparsity_layout_i, sparsity_block_size, triton_block_size=triton_block_size)
        else:
            raise ValueError(f"Reduction operation '{reduce_op}' does not support backward pass")

    @staticmethod
    @triton.jit
    def kernel_blocksparse_scatter(x,
                                   x_b, x_b_s, x_r_s, x_c_s,
                                   s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                                   i,
                                   i_b, i_b_s, i_r_s, i_c_s,
                                   o,
                                   o_b, o_b_s, o_r_s, o_c_s,
                                   s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                   r_lut_o,
                                   reduce_op_ind,
                                   sparsity_block_size,
                                   TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch, row, and column index
        spa_bat_x_idx = (pid_blk * s_lut_x_r_s + 0 * s_lut_x_c_s)
        spa_bat_x_msk = (spa_bat_x_idx < s_lut_x_r * s_lut_x_r_s)
        spa_bat_x = tl.load(s_lut_x + spa_bat_x_idx, mask=spa_bat_x_msk)

        spa_row_x_idx = (pid_blk * s_lut_x_r_s + 1 * s_lut_x_c_s)
        spa_row_x_msk = (spa_row_x_idx < s_lut_x_r * s_lut_x_r_s)
        spa_row_x = tl.load(s_lut_x + spa_row_x_idx, mask=spa_row_x_msk)

        # Load x values
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Load index values
        blk_i_idx = ((pid_blk * i_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_i_msk = (blk_i_idx < i_b * i_b_s)
        blk_i = tl.load(i + blk_i_idx, mask=blk_i_msk).to(tl.int32)

        # Get positions of sparsity blocks
        pos_spa_blk_o = blk_i // sparsity_block_size
        pos_spa_col_o = blk_i % sparsity_block_size

        # Load reverse sparsity indices for o
        rev_idx_spa_o_idx = ((spa_bat_x * s_l_o_b_s) +
                             (spa_row_x * s_l_o_r_s) +
                             (pos_spa_blk_o * s_l_o_c_s))
        rev_idx_spa_o_msk = (rev_idx_spa_o_idx < s_l_o_b * s_l_o_b_s)
        rev_idx_spa_o = tl.load(r_lut_o + rev_idx_spa_o_idx, mask=rev_idx_spa_o_msk).to(tl.int32)

        # Store output
        blk_o_idx = ((rev_idx_spa_o * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     (pos_spa_col_o * o_c_s))
        blk_o_msk = (blk_o_idx < o_b * o_b_s)

        if reduce_op_ind == 0:
            tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)
        elif reduce_op_ind == 1:
            tl.atomic_add(o + blk_o_idx, blk_x, mask=blk_o_msk)

import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.transpose import transpose
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size, validate_dtype_float


def matmul(x: BlksprsTensor, sparsity_layout_x: Tensor,
           y: BlksprsTensor, sparsity_layout_y: Tensor,
           sparsity_layout_output: Tensor,
           sparsity_block_size: int, triton_block_size: int = None) -> BlksprsTensor:
    """Performs matrix multiplication between two block-sparse tensors.

    The sparsity layout of the output tensor is used to only calculate blocks that will be present in the output.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        y (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the first block-sparse tensor.
        sparsity_layout_y (Tensor): The sparsity layout of the second block-sparse tensor.
        sparsity_layout_output (Tensor): The sparsity layout of the output tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int, optional): The block size to use for the triton kernel (default ``None``).

    Returns:
        BlksprsTensor: The result of the matrix multiplication as a block-sparse tensor in compressed form.

    """
    x = x.contiguous()
    y = y.contiguous()

    validate_dimensions(x, y)
    validate_contiguous(x, y)
    validate_dtype_float(x, y)
    validate_device(x, y)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x), (y, sparsity_layout_y))
    if sparsity_layout_x.size(-1) != sparsity_layout_y.size(-2):
        raise ValueError("Inner dimensions of tensors must match")
    validate_sparsity_block_size(sparsity_block_size, x, y)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_x_flat = sparsity_layout_x.reshape(-1)
    sparsity_reverse_lut_x = ((torch.cumsum(sparsity_layout_x_flat, dim=-1) - 1) *
                              (sparsity_layout_x_flat == 1) -
                              (1 * (sparsity_layout_x_flat == 0)))

    sparsity_layout_y_flat = sparsity_layout_y.reshape(-1)
    sparsity_reverse_lut_y = ((torch.cumsum(sparsity_layout_y_flat, dim=-1) - 1) *
                              (sparsity_layout_y_flat == 1) -
                              (1 * (sparsity_layout_y_flat == 0)))

    sparsity_lut_o = torch.nonzero(sparsity_layout_output).contiguous()

    n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout_x, sparsity_reverse_lut_x,
                        sparsity_layout_y, sparsity_reverse_lut_y,
                        sparsity_layout_output, sparsity_lut_o)

    return BlksprsTensor(_BlocksparseMatmulSSS.apply(x, y,
                                                     sparsity_layout_x, sparsity_reverse_lut_x,
                                                     sparsity_layout_y, sparsity_reverse_lut_y,
                                                     sparsity_layout_output, sparsity_lut_o,
                                                     sparsity_block_size,
                                                     n_sparse_blocks,
                                                     triton_block_size))


class _BlocksparseMatmulSSS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor,
                sparsity_layout_x: Tensor, sparsity_reverse_lut_x: Tensor,
                sparsity_layout_y: Tensor, sparsity_reverse_lut_y: Tensor,
                sparsity_layout_o: Tensor, sparsity_lut_o: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_x)
        y_b, y_r, y_c = y.size()
        y_b_s, y_r_s, y_c_s = stride(y)
        s_l_y_b, s_l_y_r, s_l_y_c = sparsity_layout_y.size()
        s_l_y_b_s, s_l_y_r_s, s_l_y_c_s = stride(sparsity_layout_y)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_lut_o_r, s_lut_o_c = sparsity_lut_o.size()
        s_lut_o_r_s, s_lut_o_c_s = stride(sparsity_lut_o)

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseMatmulSSS.kernel_blocksparse_matmul_sss[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
          sparsity_reverse_lut_x,
          y,
          y_b, y_b_s, y_r_s, y_c_s,
          s_l_y_b, s_l_y_b_s, s_l_y_r_s, s_l_y_c_s,
          sparsity_reverse_lut_y,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_o,
          s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
          sparsity_block_size,
          triton_block_size))

        ctx.save_for_backward(x, sparsity_layout_x, y, sparsity_layout_y, sparsity_layout_o)
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, sparsity_layout_x, y, sparsity_layout_y, sparsity_layout_o = ctx.saved_tensors
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        x_t, sparsity_layout_x_t = transpose(x, sparsity_layout_x, sparsity_block_size, triton_block_size)
        y_t, sparsity_layout_y_t = transpose(y, sparsity_layout_y, sparsity_block_size, triton_block_size)

        grad_x = matmul(grad_output, sparsity_layout_o, y_t, sparsity_layout_y_t, sparsity_layout_x,
                        sparsity_block_size, triton_block_size)
        grad_y = matmul(x_t, sparsity_layout_x_t, grad_output, sparsity_layout_o, sparsity_layout_y,
                        sparsity_block_size, triton_block_size)

        return grad_x, grad_y, None, None, None, None, None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_matmul_sss(x,
                                      x_b, x_b_s, x_r_s, x_c_s,
                                      s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
                                      r_lut_x,
                                      y,
                                      y_b, y_b_s, y_r_s, y_c_s,
                                      s_l_y_b, s_l_y_b_s, s_l_y_r_s, s_l_y_c_s,
                                      r_lut_y,
                                      o,
                                      o_b, o_b_s, o_r_s, o_c_s,
                                      s_lut_o,
                                      s_lut_o_r, s_lut_o_r_s,
                                      s_lut_o_c_s,
                                      sparsity_block_size,
                                      TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch, row, and column index
        spa_bat_o_idx = (pid_blk * s_lut_o_r_s + 0 * s_lut_o_c_s)
        spa_bat_o_msk = (spa_bat_o_idx >= 0 and spa_bat_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_bat_o = tl.load(s_lut_o + spa_bat_o_idx, mask=spa_bat_o_msk)

        spa_row_o_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_o_msk = (spa_row_o_idx >= 0 and spa_row_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_row_o = tl.load(s_lut_o + spa_row_o_idx, mask=spa_row_o_msk)

        spa_col_o_idx = (pid_blk * s_lut_o_r_s + 2 * s_lut_o_c_s)
        spa_col_o_msk = (spa_col_o_idx >= 0 and spa_col_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_col_o = tl.load(s_lut_o + spa_col_o_idx, mask=spa_col_o_msk)

        # Setup buffer
        buf = tl.zeros(shape=(TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), dtype=tl.float32)

        # Slide over triton block sized segments of input tensors
        for i_seg_tri in range(0, tl.cdiv(s_l_x_c * sparsity_block_size, TRITON_BLOCK_SIZE)):
            # Convert to segment index of sparsity layout
            i_seg_spa = (i_seg_tri * TRITON_BLOCK_SIZE) // sparsity_block_size
            # Calculate the triton segment index within a block
            i_seg_tri_mod = i_seg_tri % (sparsity_block_size // TRITON_BLOCK_SIZE)

            # Get reverse sparsity indices for input tensors x and y
            # These are either -1 if the block is empty or equal to the index of the block in the sparse tensor

            # Get reverse sparsity indices for x
            rev_idx_spa_x_idx = (spa_bat_o * s_l_x_b_s +
                                 spa_row_o * s_l_x_r_s +
                                 i_seg_spa * s_l_x_c_s)
            rev_idx_spa_x_msk = (rev_idx_spa_x_idx >= 0 and rev_idx_spa_x_idx < s_l_x_b * s_l_x_b_s)
            rev_idx_spa_x = tl.load(r_lut_x + rev_idx_spa_x_idx, mask=rev_idx_spa_x_msk).to(tl.int32)

            # Get reverse sparsity indices for y
            rev_idx_spa_y_idx = (spa_bat_o * s_l_y_b_s + i_seg_spa * s_l_y_r_s + spa_col_o * s_l_y_c_s)
            rev_idx_spa_y_msk = (rev_idx_spa_y_idx >= 0 and rev_idx_spa_y_idx < s_l_y_b * s_l_y_b_s)
            rev_idx_spa_y = tl.load(r_lut_y + rev_idx_spa_y_idx, mask=rev_idx_spa_y_msk).to(tl.int32)

            # If both blocks are present commence calculation
            if rev_idx_spa_x >= 0 and rev_idx_spa_y >= 0:
                blk_x_idx = ((rev_idx_spa_x * x_b_s) +
                             ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                             ((i_seg_tri_mod * TRITON_BLOCK_SIZE +
                               tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
                blk_x_msk = (blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
                blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

                blk_y_idx = ((rev_idx_spa_y * y_b_s) +
                             ((i_seg_tri_mod * TRITON_BLOCK_SIZE +
                               tl.arange(0, TRITON_BLOCK_SIZE)) * y_r_s)[:, None] +
                             ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * y_c_s)[None, :])
                blk_y_msk = (blk_y_idx >= 0 and blk_y_idx < y_b * y_b_s)
                blk_y = tl.load(y + blk_y_idx, mask=blk_y_msk)

                # Perform matrix multiplication
                buf += tl.dot(blk_x, blk_y, input_precision="tf32")

        # Store output
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, buf, mask=blk_o_msk)

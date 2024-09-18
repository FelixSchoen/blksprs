import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size


def row_wise_sum(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int,
                 flag_slice_only: bool = False, triton_block_size: int = None) -> tuple[Tensor, Tensor]:
    """Computes the row-wise sum of a block-sparse tensor.

    Returns a block-sparse tensor in compressed form with only one block per row, where the first entry contains the sum
        of the corresponding row.

    Note:
        If ``flag_slice_only`` is set the output will be of shape ``[x.size(0), x.size(1), 1]``.

    Args:
        x (Tensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        flag_slice_only (bool, optional): If set the output will be of shape ``[x.size(0), x.size(1), 1]``
            (default ``False``).
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        tuple[Tensor, Tensor]: A tuple containing a block-sparse tensor in compressed form containing the row-wise sum
            of the input and the sparsity layout of the output tensor.

    """
    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_lut = torch.nonzero(sparsity_layout).contiguous()
    sparsity_layout_flat = sparsity_layout.reshape(-1)
    sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                            (sparsity_layout_flat == 1) -
                            (1 * (sparsity_layout_flat == 0)))

    sparsity_layout_output, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
    sparsity_lut_output = torch.nonzero(sparsity_layout_output).contiguous()
    sparsity_layout_output_flat = sparsity_layout_output.reshape(-1)
    sparsity_reverse_lut_output = ((torch.cumsum(sparsity_layout_output_flat, dim=-1) - 1) *
                                   (sparsity_layout_output_flat == 1) -
                                   (1 * (sparsity_layout_output_flat == 0)))

    n_sparse_blocks_output = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout, sparsity_lut, sparsity_reverse_lut,
                        sparsity_layout_output, sparsity_lut_output, sparsity_reverse_lut_output)

    return (_BlocksparseRowWiseSum.apply(x,
                                         sparsity_layout, sparsity_lut, sparsity_reverse_lut,
                                         sparsity_layout_output, sparsity_lut_output, sparsity_reverse_lut_output,
                                         n_sparse_blocks_output,
                                         flag_slice_only,
                                         sparsity_block_size, triton_block_size),
            sparsity_layout_output)


class _BlocksparseRowWiseSum(torch.autograd.Function):
    IMPLEMENTATION = "atomic_add"

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_layout_output: Tensor, sparsity_lut_output: Tensor, sparsity_reverse_lut_output: Tensor,
                n_sparse_blocks_output: int,
                flag_slice_only: bool,
                sparsity_block_size: int, triton_block_size: int) -> Tensor:
        output = torch.zeros(size=(n_sparse_blocks_output,
                                   sparsity_block_size,
                                   1 if flag_slice_only else sparsity_block_size),
                             device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = sparsity_layout.stride()
        s_lut_x_r, s_lut_x_c = sparsity_lut.size()
        s_lut_x_r_s, s_lut_x_c_s = sparsity_lut.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_output.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = sparsity_layout_output.stride()
        s_lut_o_r, s_lut_o_c = sparsity_lut_output.size()
        s_lut_o_r_s, s_lut_o_c_s = sparsity_lut_output.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        if _BlocksparseRowWiseSum.IMPLEMENTATION == "basic":
            triton_grid = lambda meta: [o_b,
                                        triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"])]

            (_BlocksparseRowWiseSum.kernel_blocksparse_row_wise_sum[triton_grid]
             (x,
              x_b, x_b_s, x_r_s, x_c_s,
              s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
              sparsity_reverse_lut,
              output,
              o_b, o_b_s, o_r_s,
              sparsity_lut_output, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
              sparsity_block_size,
              triton_block_size))
        elif _BlocksparseRowWiseSum.IMPLEMENTATION == "atomic_add":
            triton_grid = lambda meta: [x_b,
                                        triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                        triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

            (_BlocksparseRowWiseSum.kernel_blocksparse_row_wise_sum_atomic_add[triton_grid]
             (x,
              x_b, x_b_s, x_r_s, x_c_s,
              sparsity_lut, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
              output,
              o_b, o_b_s, o_r_s,
              s_l_o_b, s_l_o_b_s, s_l_o_r_s,
              sparsity_reverse_lut_output,
              triton_block_size))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @staticmethod
    @triton.jit
    def kernel_blocksparse_row_wise_sum(x,
                                        x_b, x_b_s, x_r_s, x_c_s,
                                        s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
                                        r_lut_x,
                                        o,
                                        o_b, o_b_s, o_r_s,
                                        s_lut_o, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
                                        sparsity_block_size,
                                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_o_r_s + 0 * s_lut_o_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_o_r * s_lut_o_r_s)
        spa_bat = tl.load(s_lut_o + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_msk = (spa_row_idx < s_lut_o_r * s_lut_o_r_s)
        spa_row = tl.load(s_lut_o + spa_row_idx, mask=spa_row_msk)

        buf = tl.zeros(shape=(TRITON_BLOCK_SIZE, 1), dtype=tl.float32)

        # Slide over triton block sized segments of input tensor
        for i_seg_tri in range(0, tl.cdiv(s_l_x_c * sparsity_block_size, TRITON_BLOCK_SIZE)):
            # Convert to segment index of sparsity layout
            i_seg_spa = (i_seg_tri * TRITON_BLOCK_SIZE) // sparsity_block_size
            # Calculate the triton segment index within a block
            i_seg_tri_mod = i_seg_tri % (sparsity_block_size // TRITON_BLOCK_SIZE)

            # Load reverse sparsity index for current block
            rev_idx_spa_idx = (spa_bat * s_l_x_b_s +
                               spa_row * s_l_x_r_s +
                               i_seg_spa * s_l_x_c_s)
            rev_idx_spa_msk = (rev_idx_spa_idx < s_l_x_b * s_l_x_b_s)
            rev_idx_spa = tl.load(r_lut_x + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

            # If block is present commence operations
            if rev_idx_spa >= 0:
                blk_idx = ((rev_idx_spa * x_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                           ((i_seg_tri_mod * TRITON_BLOCK_SIZE +
                             tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
                blk_msk = (blk_idx < x_b * x_b_s)
                blk = tl.load(x + blk_idx, mask=blk_msk)

                buf = buf + tl.reshape(tl.sum(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

        o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 (tl.arange(0, 1))[None, :])
        o_msk = (o_idx < o_b * o_b_s)
        tl.store(o + o_idx, buf, o_msk)

    @staticmethod
    @triton.jit
    def kernel_blocksparse_row_wise_sum_atomic_add(x,
                                                   x_b, x_b_s, x_r_s, x_c_s,
                                                   s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                                                   o,
                                                   o_b, o_b_s, o_r_s,
                                                   s_l_o_b, s_l_o_b_s, s_l_o_r_s,
                                                   r_lut_o,
                                                   TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_x_r_s + 0 * s_lut_x_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_x_r * s_lut_x_r_s)
        spa_bat = tl.load(s_lut_x + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_x_r_s + 1 * s_lut_x_c_s)
        spa_row_msk = (spa_row_idx < s_lut_x_r * s_lut_x_r_s)
        spa_row = tl.load(s_lut_x + spa_row_idx, mask=spa_row_msk)

        # Load reverse sparsity index for current block
        rev_idx_spa_idx = (spa_bat * s_l_o_b_s +
                           spa_row * s_l_o_r_s)
        rev_idx_spa_msk = (rev_idx_spa_idx < s_l_o_b * s_l_o_b_s)
        rev_idx_spa = tl.load(r_lut_o + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

        blk_idx = ((pid_blk * x_b_s) +
                   ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                   ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_msk = (blk_idx < x_b * x_b_s)
        blk = tl.load(x + blk_idx, mask=blk_msk)

        buf = tl.reshape(tl.sum(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

        o_idx = (rev_idx_spa * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 (tl.arange(0, 1))[None, :])
        o_msk = (o_idx < o_b * o_b_s)
        tl.atomic_add(o + o_idx, buf, o_msk)

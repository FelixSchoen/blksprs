import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_device, \
    validate_sparsity_block_size, validate_triton_block_size, validate_dimensions


def repeat_interleave(x: Tensor, sparsity_layout: Tensor, repeats: int,
                      sparsity_block_size: int, triton_block_size: int = None) -> tuple[Tensor, Tensor]:
    """Repeats and interleaves the block-sparse tensor in compressed form.

    Repeats each matrix contained in the tensors by ``repeats`` amount and places them consecutively in the output
        tensor.

    Args:
        x (Tensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (int): The number of times to repeat the matrices.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        Tensor: A block-sparse tensor in compressed form containing the repeated and interleaved matrices.
        Tensor: The sparsity layout of the resulting output tensor.

    """
    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_output = torch.repeat_interleave(sparsity_layout, 3, dim=0).contiguous()

    sparsity_lut = torch.nonzero(sparsity_layout).contiguous()

    sparsity_layout_output_flat = sparsity_layout_output.reshape(-1)
    sparsity_output_reverse_lut = ((torch.cumsum(sparsity_layout_output_flat, dim=-1) - 1) *
                                   (sparsity_layout_output_flat == 1) -
                                   (1 * (sparsity_layout_output_flat == 0)))

    n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

    validate_contiguous(sparsity_layout, sparsity_lut, sparsity_layout_output, sparsity_output_reverse_lut)

    output = torch.empty(n_sparse_blocks * repeats, sparsity_block_size, sparsity_block_size,
                         dtype=x.dtype, device=x.device)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = x.stride()
    s_lut_r, s_lut_c = sparsity_lut.size()
    s_lut_r_s, s_lut_c_s = sparsity_lut.stride()
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = output.stride()
    s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_output.size()
    s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = sparsity_layout_output.stride()

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    triton_grid = lambda meta: [x_b,
                                triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_repeat_interleave[triton_grid]
     (x,
      x_b, x_b_s, x_r_s, x_c_s,
      sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
      output,
      o_b, o_b_s, o_r_s, o_c_s,
      s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
      sparsity_output_reverse_lut,
      repeats,
      triton_block_size))

    return output, sparsity_layout_output


@triton.jit
def kernel_repeat_interleave(x,
                             x_b, x_b_s, x_r_s, x_c_s,
                             s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                             o,
                             o_b, o_b_s, o_r_s, o_c_s,
                             s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                             r_lut_o,
                             repeats,
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

    # Load block
    blk_x_idx = ((pid_blk * x_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    for repeat in range(repeats):
        # Get reverse sparsity index
        rev_idx_spa_idx = ((spa_bat * repeats + repeat) * s_l_o_b_s +
                           spa_row * s_l_o_r_s +
                           spa_col * s_l_o_c_s)
        rev_idx_spa_msk = (rev_idx_spa_idx < s_l_o_b * s_l_o_b_s)
        rev_idx_spa = tl.load(r_lut_o + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

        # Store block
        blk_o_idx = ((rev_idx_spa * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)

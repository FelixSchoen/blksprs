import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.validation import validate_contiguous, validate_device, \
    validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def broadcast_add(x: Tensor, y: Tensor, sparsity_layout_output: Tensor,
                  sparsity_block_size: int) -> BlksprsTensor:
    """Performs a broadcast and subsequent addition of two dense tensors x and y. Returns a block-sparse tensor in
        compressed form.

    Args:
        x (Tensor): A dense input tensor.
        y (Tensor): A dense input tensor.
        sparsity_layout_output (Tensor): The sparsity layout of the output tensor.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        BlksprsTensor: The result of the operation as a block-sparse tensor in compressed form. Each element o(i, j) of the
            output tensor corresponds to x(i) + y(j).

    """
    x = x.contiguous()
    y = y.contiguous()

    validate_device(x, y)
    validate_contiguous(x, y)
    if x.size(-1) != y.size(-1):
        raise ValueError("Dimensions of tensors must match")
    validate_sparsity_block_size(sparsity_block_size)

    sparsity_lut_o = torch.nonzero(sparsity_layout_output).contiguous()

    n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout_output, sparsity_lut_o)

    return BlksprsTensor(broadcast_add_forward(x, y, sparsity_lut_o, sparsity_block_size, n_sparse_blocks))


def broadcast_sub(x: Tensor, y: Tensor, sparsity_layout_output: Tensor,
                  sparsity_block_size: int) -> BlksprsTensor:
    """Wrapper for ``broadcast_add`` with negated y.

    """
    return broadcast_add(x, torch.neg(y), sparsity_layout_output, sparsity_block_size)


@triton_op("blksprs::broadcast_add_forward", mutates_args={})
def broadcast_add_forward(x: Tensor, y: Tensor,
                          sparsity_lut_o: Tensor,
                          sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(n_sparse_blocks, sparsity_block_size, sparsity_block_size, dtype=x.dtype, device=x.device)

        x_b, x_c = x.size()
        x_b_s, x_c_s = stride(x)
        y_b, y_c = y.size()
        y_b_s, y_c_s = stride(y)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_lut_o_r, s_lut_o_c = sparsity_lut_o.size()
        s_lut_o_r_s, s_lut_o_c_s = stride(sparsity_lut_o)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(broadcast_add_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_c_s,
          y,
          y_b, y_b_s, y_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_o, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
          sparsity_block_size))

        return output


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def broadcast_add_kernel(x,
                         x_b, x_b_s, x_c_s,
                         y,
                         y_b, y_b_s, y_c_s,
                         o,
                         o_b, o_b_s, o_r_s, o_c_s,
                         s_lut_o, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
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

    # Load x block
    blk_x_idx = (spa_bat_o * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + spa_row_o * sparsity_block_size +
                   tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and
                 blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    # Load y block
    blk_y_idx = (spa_bat_o * y_b_s +
                 ((pid_col * TRITON_BLOCK_SIZE + spa_col_o * sparsity_block_size +
                   tl.arange(0, TRITON_BLOCK_SIZE)) * y_c_s)[None, :])
    blk_y_msk = (blk_y_idx >= 0 and
                 blk_y_idx < y_b * y_b_s)
    blk_y = tl.load(y + blk_y_idx, mask=blk_y_msk)

    # Compute sum
    blk_x, blk_y = tl.broadcast(tl.trans(blk_x), blk_y)
    buf = blk_x + blk_y

    # Store result
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = (blk_o_idx >= 0 and
                 blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, buf, mask=blk_o_msk)

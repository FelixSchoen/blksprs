import pdb

import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.ops.misc.row_wise import row_wise_sum, row_wise_max, row_wise_sub
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride, ceil_pow2
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_dtype_float_32


def softmax(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int, flag_fused: bool = True,
            lut: dict = None) -> BlksprsTensor:
    """Wrapper for :func:`softmax_regular` and :func:`softmax_fused` based on the ``flag_fused`` parameter.

    """
    if flag_fused:
        return softmax_fused(x, sparsity_layout, sparsity_block_size, lut)
    else:
        return softmax_regular(x, sparsity_layout, sparsity_block_size, lut)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def softmax_regular(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                    lut: dict = None) -> BlksprsTensor:
    """Computes the softmax of a block-sparse tensor in compressed form.

    Note:
        Sparse blocks are not considered for the calculation of the softmax, i.e., all values are assumed to be ``-inf``.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The result of the softmax operation as a block-sparse tensor in compressed form.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_float_32(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = softmax_build_lut(lut, sparsity_layout)

    return BlksprsTensor(softmax_forward(x, sparsity_layout,
                                         lut["sparsity_lut"],
                                         lut["sparsity_reverse_lut_rws"],
                                         sparsity_block_size))


@triton_op("blksprs::softmax_forward", mutates_args={})
def softmax_forward(x: Tensor, sparsity_layout: Tensor,
                    sparsity_lut: Tensor,
                    sparsity_reverse_lut_rws: Tensor,
                    sparsity_block_size: int) -> Tensor:
    output = torch.zeros_like(x)

    x_row_wise_max, sparsity_layout_rwm = row_wise_max(x, sparsity_layout, sparsity_block_size,
                                                       flag_slice_only=True)
    x_scaled = row_wise_sub(x, sparsity_layout, x_row_wise_max, sparsity_block_size)
    x_exp = torch.exp(x_scaled)
    x_exp_row_wise_sum, sparsity_layout_rws = row_wise_sum(x_exp, sparsity_layout, sparsity_block_size,
                                                           flag_slice_only=True)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = stride(x)
    s_lut_r, s_lut_c = sparsity_lut.size()
    s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
    o_b, o_r, o_c = output.size()
    s_b, s_r, s_c = x_exp_row_wise_sum.shape
    s_b_s, s_r_s, s_c_s = stride(x_exp_row_wise_sum)
    s_l_s_b, s_l_s_r, s_l_s_c = sparsity_layout_rws.shape
    s_l_s_b_s, s_l_s_r_s, s_l_s_c_s = stride(sparsity_layout_rws)

    triton_grid = lambda meta: [o_b,
                                triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

    (wrap_triton(softmax_kernel)[triton_grid]
     (x_exp,
      x_b, x_b_s, x_r_s, x_c_s,
      sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
      x_exp_row_wise_sum, s_b, s_b_s, s_r_s, s_c_s,
      s_l_s_b, s_l_s_b_s, s_l_s_r_s,
      sparsity_reverse_lut_rws,
      output,
      sparsity_block_size))

    return output


def softmax_backward_wrapper(ctx, grad_output):
    o, sparsity_layout, sparsity_lut = ctx.saved_tensors
    sparsity_block_size = ctx.sparsity_block_size

    return softmax_backward(grad_output, o, sparsity_lut, sparsity_layout,
                            sparsity_block_size), None, None, None, None, None


@triton_op("blksprs::softmax_backward", mutates_args={})
def softmax_backward(grad_output: Tensor, o: Tensor, sparsity_lut: Tensor, sparsity_layout: Tensor,
                     sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        grad_x = torch.zeros_like(o, dtype=torch.float)

        s, sparsity_layout_s = row_wise_sum(grad_output * o, sparsity_layout, sparsity_block_size, flag_slice_only=True)

        sparsity_layout_s_flat = sparsity_layout_s.reshape(-1)
        sparsity_reverse_lut_s = ((torch.cumsum(sparsity_layout_s_flat, dim=-1) - 1) *
                                  (sparsity_layout_s_flat == 1) -
                                  (1 * (sparsity_layout_s_flat == 0)))

        o_b, o_r, o_c = o.size()
        o_b_s, o_r_s, o_c_s = stride(o)
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
        s_b, s_r, s_c = s.size()
        s_b_s, s_r_s, s_c_s = stride(s)
        s_l_s_b, s_l_s_r, s_l_s_c = sparsity_layout_s.size()
        s_l_s_b_s, s_l_s_r_s, s_l_s_c_s = stride(sparsity_layout_s)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(softmax_kernel_grad)[triton_grid]
         (grad_output,
          o_b, o_b_s, o_r_s, o_c_s,
          o,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          s,
          s_b, s_b_s, s_r_s, s_c_s,
          s_l_s_b, s_l_s_b_s, s_l_s_r_s,
          sparsity_reverse_lut_s,
          grad_x,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size))

        return grad_x


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def softmax_kernel(x,
                   x_b, x_b_s, x_r_s, x_c_s,
                   s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                   s, s_b, s_b_s, s_r_s, s_c_s,
                   s_l_s_b, s_l_s_b_s, s_l_s_r_s,
                   r_lut_s,
                   o,
                   sparsity_block_size,
                   TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch and row index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    # Get reverse sparsity indices for s
    rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                         spa_row * s_l_s_r_s)
    rev_idx_spa_s_msk = (rev_idx_spa_s_idx >= 0 and rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s)
    rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

    if rev_idx_spa_s >= 0:
        # Load x block
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx >= 0 and
                     blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Load sum block
        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx >= 0 and
                     blk_s_idx < s_b * s_b_s)
        blk_s = tl.load(s + blk_s_idx, mask=blk_s_msk)

        # Compute softmax
        buf = tl.div_rn(blk_x, blk_s)

        # Store output
        tl.store(o + blk_x_idx, buf, mask=blk_x_msk)


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def softmax_kernel_grad(g,
                        g_b, g_b_s, g_r_s, g_c_s,
                        x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                        s,
                        s_b, s_b_s, s_r_s, s_c_s,
                        s_l_s_b, s_l_s_b_s, s_l_s_r_s,
                        r_lut_s,
                        o,
                        o_b, o_b_s, o_r_s, o_c_s,
                        sparsity_block_size,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch and row index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                         spa_row * s_l_s_r_s)
    rev_idx_spa_s_msk = (rev_idx_spa_s_idx >= 0 and rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s)
    rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

    if rev_idx_spa_s >= 0:
        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx >= 0 and
                     blk_s_idx < s_b * s_b_s)
        blk_s = tl.load(s + blk_s_idx, mask=blk_s_msk)

        blk_g_idx = ((pid_blk * g_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * g_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * g_c_s)[None, :])
        blk_g_msk = (blk_g_idx >= 0 and
                     blk_g_idx < g_b * g_b_s)
        blk_g = tl.load(g + blk_g_idx, mask=blk_g_msk)

        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx >= 0 and
                     blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        buf = blk_x * (blk_g - blk_s)

        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx >= 0 and
                     blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, buf, mask=blk_o_msk)


def softmax_build_lut(lut: dict, sparsity_layout: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(sparsity_layout).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "sparsity_reverse_lut_rws" not in lut:
        sparsity_layout_rws, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
        sparsity_layout_rws_flat = sparsity_layout_rws.reshape(-1)
        sparsity_reverse_lut_rws = ((torch.cumsum(sparsity_layout_rws_flat, dim=-1) - 1) *
                                    (sparsity_layout_rws_flat == 1) -
                                    (1 * (sparsity_layout_rws_flat == 0)))
        lut["sparsity_reverse_lut_rws"] = sparsity_reverse_lut_rws

    validate_contiguous(sparsity_layout, lut["sparsity_lut"], lut["sparsity_reverse_lut_rws"])

    return lut


# noinspection PyUnusedLocal
def softmax_setup_context(ctx, inputs, output):
    (_, sparsity_layout, sparsity_lut, _, sparsity_block_size) = inputs

    ctx.save_for_backward(output, sparsity_layout, sparsity_lut)
    ctx.sparsity_block_size = sparsity_block_size


softmax_forward.register_autograd(softmax_backward_wrapper, setup_context=softmax_setup_context)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def softmax_fused(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                  lut: dict = None) -> BlksprsTensor:
    """Computes the softmax fused for each row of a block-sparse tensor in compressed form.

    Note:
        This softmax implementation is a fused version that loads the entire row of a block-sparse tensor into memory.
        See :func:`softmax` for a true block-wise softmax implementation.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The result of the softmax operation as a block-sparse tensor in compressed form.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_float_32(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = softmax_fused_build_lut(lut, sparsity_layout)

    return BlksprsTensor(softmax_fused_forward(x, sparsity_layout,
                                               lut["sparsity_reverse_lut_sorted"],
                                               lut["max_blocks_line"],
                                               sparsity_block_size))


@triton_op("blksprs::softmax_fused_forward", mutates_args={})
def softmax_fused_forward(x: Tensor, sparsity_layout: Tensor,
                          sparsity_reverse_lut_sorted: Tensor,
                          max_blocks_line: int,
                          sparsity_block_size: int) -> Tensor:
    output = torch.zeros_like(x)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = stride(x)
    s_l_b, s_l_r, s_l_c = sparsity_layout.size()
    s_l_b_s, s_l_r_s, s_l_c_s = stride(sparsity_layout)

    triton_grid = lambda meta: [s_l_b,
                                s_l_r,
                                sparsity_block_size]

    (wrap_triton(softmax_fused_kernel)[triton_grid]
     (x,
      x_b, x_b_s, x_r_s, x_c_s,
      output,
      s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
      sparsity_reverse_lut_sorted,
      max_blocks_line,
      sparsity_block_size))

    return output


def softmax_fused_backward_wrapper(ctx, grad_output):
    o, sparsity_layout, sparsity_reverse_lut_sorted = ctx.saved_tensors
    max_blocks_line = ctx.max_blocks_line
    sparsity_block_size = ctx.sparsity_block_size

    return softmax_fused_backward(grad_output, o, sparsity_reverse_lut_sorted, sparsity_layout,
                                  max_blocks_line, sparsity_block_size), None, None, None, None


@triton_op("blksprs::softmax_fused_backward", mutates_args={})
def softmax_fused_backward(grad_output: Tensor,
                           o: Tensor,
                           sparsity_reverse_lut_sorted: Tensor,
                           sparsity_layout: Tensor,
                           max_blocks_line: int,
                           sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        grad_x = torch.zeros_like(o)

        g_b, g_r, g_c = grad_output.size()
        g_b_s, g_r_s, g_c_s = stride(grad_output)
        o_b, o_r, o_c = o.size()
        o_b_s, o_r_s, o_c_s = stride(o)
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = stride(sparsity_layout)

        triton_grid = lambda meta: [s_l_b,
                                    s_l_r,
                                    sparsity_block_size]

        (wrap_triton(softmax_fused_kernel_grad)[triton_grid]
         (grad_output,
          g_b, g_b_s, g_r_s, g_c_s,
          o,
          o_b, o_b_s, o_r_s, o_c_s,
          s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
          sparsity_reverse_lut_sorted,
          grad_x,
          max_blocks_line,
          sparsity_block_size))

        return grad_x


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def softmax_fused_kernel(x,
                         x_b, x_b_s, x_r_s, x_c_s,
                         o,
                         s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
                         r_lut_s,
                         mbs: tl.constexpr,
                         sparsity_block_size: tl.constexpr,
                         TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_bat = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_lin = tl.program_id(axis=2)

    # Load reverse sparsity indices of row
    blk_rev_idx = (pid_bat * s_l_b_s +
                   pid_row * s_l_r_s +
                   (tl.arange(0, mbs) * s_l_c_s))
    blk_rev_msk = (blk_rev_idx >= 0 and blk_rev_idx < s_l_b * s_l_b_s)
    blk_rev = tl.load(r_lut_s + blk_rev_idx, mask=blk_rev_msk).to(tl.int32)

    if (not (tl.min(blk_rev) == -1 and
             tl.max(blk_rev) == -1)):
        # Extend sparsity indices to cover sparsity blocks
        blk_rev_ext = tl.expand_dims(blk_rev, -1)
        blk_rev_ext = tl.broadcast_to(blk_rev_ext, (mbs, sparsity_block_size))
        blk_rev_ext = tl.reshape(blk_rev_ext, (mbs * sparsity_block_size))

        # Load line of x
        blk_x_idx = (blk_rev_ext * x_b_s +
                     pid_lin * x_r_s +
                     (tl.arange(0, mbs * sparsity_block_size) % sparsity_block_size) * x_c_s)
        blk_x_mask = ((blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
                      and blk_rev_ext != -1)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_mask, other=float("-inf"))

        # Compute softmax
        blk_x_softmax = tl.softmax(blk_x)

        # Store output
        tl.store(o + blk_x_idx, blk_x_softmax, mask=blk_x_mask)


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def softmax_fused_kernel_grad(g,
                              g_b, g_b_s, g_r_s, g_c_s,
                              x,
                              x_b, x_b_s, x_r_s, x_c_s,
                              s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
                              r_lut_s,
                              o,
                              mbs: tl.constexpr,
                              sparsity_block_size: tl.constexpr,
                              TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_bat = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_lin = tl.program_id(axis=2)

    # Load reverse sparsity indices of row
    blk_rev_idx = (pid_bat * s_l_b_s +
                   pid_row * s_l_r_s +
                   (tl.arange(0, mbs) * s_l_c_s))
    blk_rev_msk = (blk_rev_idx >= 0 and blk_rev_idx < s_l_b * s_l_b_s)
    blk_rev = tl.load(r_lut_s + blk_rev_idx, mask=blk_rev_msk).to(tl.int32)

    if (not (tl.min(blk_rev) == -1 and
             tl.max(blk_rev) == -1)):
        # Extend sparsity indices to cover sparsity blocks
        blk_rev_ext = tl.expand_dims(blk_rev, -1)
        blk_rev_ext = tl.broadcast_to(blk_rev_ext, (mbs, sparsity_block_size))
        blk_rev_ext = tl.reshape(blk_rev_ext, (mbs * sparsity_block_size))

        # Load line of g
        blk_g_idx = (blk_rev_ext * g_b_s +
                     pid_lin * g_r_s +
                     (tl.arange(0, mbs * sparsity_block_size) % sparsity_block_size) * g_c_s)
        blk_g_mask = ((blk_g_idx >= 0 and blk_g_idx < g_b * g_b_s)
                      and blk_rev_ext != -1)
        blk_g = tl.load(g + blk_g_idx, mask=blk_g_mask)

        # Load line of x
        blk_x_idx = (blk_rev_ext * x_b_s +
                     pid_lin * x_r_s +
                     (tl.arange(0, mbs * sparsity_block_size) % sparsity_block_size) * x_c_s)
        blk_x_mask = ((blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
                      and blk_rev_ext != -1)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_mask)

        # Compute gradients
        blk_grad = blk_x * (blk_g - tl.sum(blk_x * blk_g))

        # Store output
        tl.store(o + blk_x_idx, blk_grad, mask=blk_x_mask)


def softmax_fused_build_lut(lut: dict, sparsity_layout: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_reverse_lut_sorted" not in lut:
        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut_sorted = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                        (sparsity_layout_flat == 1) -
                                        (1 * (sparsity_layout_flat == 0)))
                                       .reshape(sparsity_layout.size())
                                       .sort(descending=True, dim=-1)[0]
                                       .reshape(-1).contiguous())
        lut["sparsity_reverse_lut_sorted"] = sparsity_reverse_lut_sorted

    if "max_blocks_line" not in lut:
        sparsity_reverse_lut_sorted = lut["sparsity_reverse_lut_sorted"]
        max_blocks_line = ((torch.reshape(sparsity_reverse_lut_sorted, (-1, sparsity_layout.size(-1)))
                            != -1)
                           .sum(dim=-1)
                           .max()
                           .item())
        lut["max_blocks_line"] = min(ceil_pow2(max(max_blocks_line, 2)), sparsity_layout.size(-1))

    validate_contiguous(sparsity_layout, lut["sparsity_reverse_lut_sorted"])

    return lut


# noinspection PyUnusedLocal
def softmax_fused_setup_context(ctx, inputs, output):
    (_, sparsity_layout, sparsity_reverse_lut_sorted, max_blocks_line, sparsity_block_size) = inputs

    ctx.save_for_backward(output, sparsity_layout, sparsity_reverse_lut_sorted)
    ctx.max_blocks_line = max_blocks_line
    ctx.sparsity_block_size = sparsity_block_size


softmax_fused_forward.register_autograd(softmax_fused_backward_wrapper, setup_context=softmax_fused_setup_context)

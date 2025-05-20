import math

import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs, prune_autotune_configs_conversion
from blksprs.utils.validation import validate_dimensions, validate_device, \
    validate_contiguous, validate_sparsity, validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def build_sparsity_layout(x: Tensor, sparsity_block_size: int) -> Tensor:
    """Builds the sparsity layout of a dense tensor in regular form covering its sparse blocks.

    Args:
        x (Tensor): A block-sparse (or dense) tensor in regular form.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        Tensor: The sparsity layout of the input block-sparse (or dense) tensor.

    """
    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)

    return build_sparsity_layout_operation(x, sparsity_block_size)


@triton_op("blksprs::build_sparsity_layout", mutates_args={})
def build_sparsity_layout_operation(x: Tensor, sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size,
                             dtype=torch.bool, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(build_sparsity_layout_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size))

        return output


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def build_sparsity_layout_kernel(x,
                                 x_b, x_b_s, x_r_s, x_c_s,
                                 o,
                                 o_b, o_b_s, o_r_s, o_c_s,
                                 sparsity_block_size,
                                 TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_bat = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Load x values
    blk_x_idx = (pid_bat * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and
                 blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    # Store sparsity layout value
    if tl.min(blk_x) != 0 or tl.max(blk_x) != 0:
        blk_o_idx = (pid_bat * o_b_s +
                     (((pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size) * o_r_s +
                      ((pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size) * o_c_s))
        blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, 1, mask=blk_o_msk)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def build_sparsity_layout_adaption(x: BlksprsTensor, sparsity_layout_from: Tensor,
                                   sparsity_block_size_from: int, sparsity_block_size_to: int) -> Tensor:
    """Builds the sparsity layout of a block-sparse tensor in compressed form if a different sparsity block size were
        used.
        
    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_from (Tensor): The sparsity layout of the input block-sparse tensor.
        sparsity_block_size_from (int): The size of the sparsity blocks of the input tensor.
        sparsity_block_size_to (int): The desired size of the sparsity blocks for the resulting layout.

    Returns:
        Tensor: The sparsity layout in regular form using the new sparsity block size of the input block-sparse tensor
            in compressed form.
    
    """
    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout_from)
    validate_device(x)
    validate_sparsity(sparsity_block_size_from, (x, sparsity_layout_from))
    validate_sparsity_block_size(sparsity_block_size_from, x)
    validate_sparsity_block_size(sparsity_block_size_to)

    sparsity_lut = torch.nonzero(sparsity_layout_from).contiguous()

    validate_contiguous(sparsity_layout_from, sparsity_lut)

    return build_sparsity_layout_adaption_operation(x, sparsity_layout_from, sparsity_lut,
                                                    sparsity_block_size_from, sparsity_block_size_to)


@triton_op("blksprs::build_sparsity_layout_adaption", mutates_args={})
def build_sparsity_layout_adaption_operation(x: Tensor, sparsity_layout_from: Tensor, sparsity_lut: Tensor,
                                             sparsity_block_size_from: int, sparsity_block_size_to: int) -> Tensor:
    with torch.no_grad():
        o_b = sparsity_layout_from.size(0)
        o_r = math.ceil(sparsity_layout_from.size(1) * sparsity_block_size_from // sparsity_block_size_to)
        o_c = math.ceil(sparsity_layout_from.size(2) * sparsity_block_size_from // sparsity_block_size_to)

        output = torch.zeros(o_b, o_r, o_c, dtype=torch.bool, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(build_sparsity_layout_adaption_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size_from,
          sparsity_block_size_to))

        return output


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size_from", "sparsity_block_size_to"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_conversion},
    reset_to_zero=["o"]
)
@triton.jit
def build_sparsity_layout_adaption_kernel(x,
                                          x_b, x_b_s, x_r_s, x_c_s,
                                          s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                          o,
                                          o_b, o_b_s, o_r_s, o_c_s,
                                          sparsity_block_size_from,
                                          sparsity_block_size_to,
                                          TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get sparsity index of current output block consisting of its batch, row, and column index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
    spa_col_msk = (spa_col_idx >= 0 and spa_col_idx < s_lut_r * s_lut_r_s)
    spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

    # Load x values
    blk_x_idx = ((pid_blk * x_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and
                 blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    # Store sparsity layout value
    if tl.min(blk_x) != 0 or tl.max(blk_x) != 0:
        blk_o_idx = ((spa_bat * o_b_s) +
                     (((pid_row * TRITON_BLOCK_SIZE + spa_row * sparsity_block_size_from)
                       // sparsity_block_size_to) * o_r_s) +
                     (((pid_col * TRITON_BLOCK_SIZE + spa_col * sparsity_block_size_from)
                       // sparsity_block_size_to) * o_c_s))
        blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, 1, mask=blk_o_msk)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def build_sparsity_layout_matmul(sparsity_layout_x: Tensor, sparsity_layout_y: Tensor) -> Tensor:
    """Builds the precise sparsity layout of the result of a matrix multiplication between the two input tensors.

    Args:
        sparsity_layout_x (Tensor): The sparsity layout of the first block-sparse tensor.
        sparsity_layout_y (Tensor): The sparsity layout of the second block-sparse tensor.

    Returns:
        Tensor: The precise sparsity layout of the result of a matrix multiplication between the two input tensors.

    """
    return torch.matmul(sparsity_layout_x.to(torch.float), sparsity_layout_y.to(torch.float)).to(torch.bool)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def build_sparsity_layout_matmul_fast(sparsity_layout_x: Tensor, sparsity_layout_y: Tensor):
    """Builds the approximate sparsity layout of the result of a matrix multiplication between the two input tensors.

    Note:
        This function is faster than the ``build_sparsity_layout_matmul`` function due to the fact that it only checks
            whether at least one of the blocks in either of the vectors participating in the matmul is non-sparse. The
            resulting sparsity layout may thus overestimate the actual sparsity of the result.

    Args:
        sparsity_layout_x (Tensor): The sparsity layout of the first block-sparse tensor.
        sparsity_layout_y (Tensor): The sparsity layout of the second block-sparse tensor.

    Returns:
        Tensor: The approximate sparsity layout of the result of a matrix multiplication between the two input tensors.

    """
    sparsity_layout_x_slice = torch.max(sparsity_layout_x, dim=-1).values.unsqueeze(-1)
    sparsity_layout_y_slice = torch.max(sparsity_layout_y, dim=-2).values.unsqueeze(1)

    return torch.logical_or(sparsity_layout_x_slice, sparsity_layout_y_slice)


def build_sparsity_layout_full(x: Tensor, sparsity_block_size: int) -> Tensor:
    return torch.ones(size=(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size),
                      dtype=torch.bool, device=x.device)

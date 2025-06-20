import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.layouting.sparsity_layout import build_sparsity_layout_adaption
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs, prune_autotune_configs_conversion
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_sparsity_dense


def to_blksprs(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int) -> BlksprsTensor:
    """Wrapper for :func:`to_sparse`.

    """
    return to_sparse(x, sparsity_layout, sparsity_block_size)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def to_sparse(x: Tensor, sparsity_layout: Tensor,
              sparsity_block_size: int, lut: dict = None) -> BlksprsTensor:
    """Converts a block-sparse tensor in regular form to a block-sparse tensor in compressed form based on the given
    sparsity layout.

        Args:
        x (Tensor): A block-sparse tensor in regular form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The block-sparse tensor converted to compressed form.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity_dense(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = to_sparse_build_lut(lut, sparsity_layout)

    if sparsity_layout.size(1) == 1 and sparsity_layout.size(2) == 1 and torch.all(sparsity_layout):
        return BlksprsTensor(x)

    return BlksprsTensor(to_sparse_forward(x, sparsity_layout,
                                           lut["sparsity_lut"], sparsity_block_size, lut["n_sparse_blocks"]))


@triton_op("blksprs::to_sparse_forward", mutates_args={})
def to_sparse_forward(x: Tensor, _: Tensor,
                      sparsity_lut: Tensor, sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(to_sparse_kernel)[triton_grid]
         (x, x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          output, o_b_s, o_r_s, o_c_s,
          sparsity_block_size))

        return output


def to_sparse_wrapper_backward(ctx, grad_output):
    sparsity_layout = ctx.saved_tensors[0]
    sparsity_block_size = ctx.sparsity_block_size

    return to_dense(grad_output, sparsity_layout, sparsity_block_size), None, None, None, None


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def to_sparse_kernel(x,
                     x_b, x_b_s, x_r_s, x_c_s,
                     s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                     o,
                     o_b_s, o_r_s, o_c_s,
                     sparsity_block_size,
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

    # Load block from dense tensor
    blk_d_idx = (spa_bat * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + spa_row * sparsity_block_size +
                   tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + spa_col * sparsity_block_size +
                   tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_d_msk = (blk_d_idx >= 0 and
                 blk_d_idx < x_b * x_b_s)
    blk_d = tl.load(x + blk_d_idx, mask=blk_d_msk)

    # Store block in sparse tensor
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE) * o_c_s))[None, :])
    blk_o_msk = (blk_o_idx >= 0 and
                 blk_o_idx < (pid_blk + 1) * o_b_s)
    tl.store(o + blk_o_idx, blk_d, mask=blk_o_msk)


def to_sparse_build_lut(lut: dict, sparsity_layout: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(sparsity_layout).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout, lut["sparsity_lut"])

    return lut


# noinspection PyUnusedLocal
def to_sparse_setup_context(ctx, inputs, output):
    (_, sparsity_layout, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout, )
    ctx.sparsity_block_size = sparsity_block_size


to_sparse_forward.register_autograd(to_sparse_wrapper_backward, setup_context=to_sparse_setup_context)


def from_blksprs(x: BlksprsTensor, sparsity_layout: Tensor,
                 sparsity_block_size: int, fill_value: float = 0, lut: dict = None) -> Tensor:
    """Wrapper for :func:`to_dense`.

    """
    return to_dense(x, sparsity_layout, sparsity_block_size, fill_value=fill_value, lut=lut)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def to_dense(x: BlksprsTensor, sparsity_layout: Tensor,
             sparsity_block_size: int, fill_value: float = 0, lut: dict = None) -> Tensor:
    """Converts a block-sparse tensor in compressed form to a block-sparse tensor in regular form based on the given
        sparsity layout.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        fill_value (float): The value to fill the resulting dense tensor with where the block-sparse tensor is not
            present (default ``0``).
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        Tensor: The block-sparse tensor converted to regular form.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = to_dense_build_lut(lut, sparsity_layout)

    if sparsity_layout.size(1) == 1 and sparsity_layout.size(2) == 1 and torch.all(sparsity_layout):
        return x

    return Tensor(to_dense_forward(x, sparsity_layout,
                            lut["sparsity_reverse_lut"], sparsity_block_size, fill_value))


@triton_op("blksprs::to_dense_forward", mutates_args={})
def to_dense_forward(x: Tensor, sparsity_layout: Tensor,
                     sparsity_reverse_lut: Tensor,
                     sparsity_block_size: int, fill_value: float) -> Tensor:
    with torch.no_grad():
        output = torch.full(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                  sparsity_layout.size(2) * sparsity_block_size), fill_value=fill_value,
                            dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = stride(sparsity_layout)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(to_dense_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size))

        return output


def to_dense_wrapper_backward(ctx, grad_output):
    sparsity_layout = ctx.saved_tensors[0]
    sparsity_block_size = ctx.sparsity_block_size

    return to_sparse(grad_output, sparsity_layout, sparsity_block_size), None, None, None, None


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    restore_value=["o"]
)
@triton.jit
def to_dense_kernel(x,
                    x_b, x_b_s, x_r_s, x_c_s,
                    s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
                    sparsity_reverse_lut,
                    o,
                    o_b, o_b_s, o_r_s, o_c_s,
                    sparsity_block_size,
                    TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get sparsity index of current block
    spa_row = (pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size
    spa_col = (pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size

    # Get reverse sparsity index for current block
    rev_idx_spa_idx = (pid_blk * s_l_b_s + spa_row * s_l_r_s + spa_col * s_l_c_s)
    rev_idx_spa_msk = (rev_idx_spa_idx >= 0 and rev_idx_spa_idx < s_l_b * s_l_b_s)
    rev_idx_spa = tl.load(sparsity_reverse_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    # If block is present commence operations
    if rev_idx_spa >= 0:
        blk_idx = (rev_idx_spa * x_b_s +
                   (((pid_row % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                     tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                   (((pid_col % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                     tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_msk = (blk_idx >= 0 and
                   blk_idx < x_b * x_b_s)
        blk = tl.load(x + blk_idx, mask=blk_msk)

        o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        o_msk = (o_idx >= 0 and o_idx < o_b * o_b_s)
        tl.store(o + o_idx, blk, o_msk)


def to_dense_build_lut(lut: dict, sparsity_layout: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_reverse_lut" not in lut:
        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                (sparsity_layout_flat == 1) -
                                (1 * (sparsity_layout_flat == 0)))
        lut["sparsity_reverse_lut"] = sparsity_reverse_lut

    validate_contiguous(lut["sparsity_reverse_lut"])

    return lut


# noinspection PyUnusedLocal
def to_dense_setup_context(ctx, inputs, output):
    (_, sparsity_layout, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout)
    ctx.sparsity_block_size = sparsity_block_size


to_dense_forward.register_autograd(to_dense_wrapper_backward, setup_context=to_dense_setup_context)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def adapt_layout(x: BlksprsTensor, sparsity_layout_from: Tensor, sparsity_block_size_from: int,
                 sparsity_block_size_to: int, sparsity_layout_to: Tensor = None) -> (BlksprsTensor, Tensor):
    """Adapts the sparsity layout of a block-sparse tensor, resulting in a new block-sparse tensor in compressed form
        conforming to the new sparsity layout (and sparsity block size) definition.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_from (Tensor): The sparsity layout of the input block-sparse tensor.
        sparsity_block_size_from (int): The size of the sparsity blocks of the input sparsity layout.
        sparsity_block_size_to (int): The size of the sparsity blocks of the output sparsity layout.
        sparsity_layout_to (Tensor): The sparsity layout of the output block-sparse tensor (default ``None``).

    Returns:
        BlksprsTensor: The block-sparse tensor in compressed form with the adapted sparsity layout and sparsity block size.
        Tensor: The sparsity layout of the resulting output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout_from)
    validate_device(x)
    validate_sparsity(sparsity_block_size_from, (x, sparsity_layout_from))
    validate_sparsity_block_size(sparsity_block_size_from, x)
    validate_sparsity_block_size(sparsity_block_size_to)

    sparsity_layout_from_flat = sparsity_layout_from.reshape(-1)
    sparsity_reverse_lut_from = ((torch.cumsum(sparsity_layout_from_flat, dim=-1) - 1) *
                                 (sparsity_layout_from_flat == 1) -
                                 (1 * (sparsity_layout_from_flat == 0)))

    if sparsity_layout_to is None:
        sparsity_layout_to = build_sparsity_layout_adaption(x, sparsity_layout_from,
                                                            sparsity_block_size_from, sparsity_block_size_to)

    sparsity_lut_to = torch.nonzero(sparsity_layout_to).contiguous()

    n_sparse_blocks_to = torch.sum(sparsity_layout_to.to(torch.int)).item()

    validate_contiguous(sparsity_reverse_lut_from, sparsity_layout_to, sparsity_lut_to)

    if (sparsity_block_size_from == sparsity_block_size_to) and torch.equal(sparsity_layout_from, sparsity_layout_to):
        return BlksprsTensor(x), sparsity_layout_to

    return BlksprsTensor(adapt_layout_forward(x,
                                              sparsity_layout_from, sparsity_reverse_lut_from,
                                              sparsity_block_size_from,
                                              sparsity_layout_to, sparsity_lut_to,
                                              sparsity_block_size_to,
                                              n_sparse_blocks_to)), sparsity_layout_to


@triton_op("blksprs::adapt_layout_forward", mutates_args={})
def adapt_layout_forward(x: Tensor,
                         sparsity_layout_from: Tensor, sparsity_reverse_lut_from: Tensor,
                         sparsity_block_size_from: int,
                         _: Tensor, sparsity_lut_to: Tensor,
                         sparsity_block_size_to: int,
                         n_sparse_blocks_to: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size=(n_sparse_blocks_to, sparsity_block_size_to, sparsity_block_size_to),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_from.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_from)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_lut_o_r, s_lut_o_c = sparsity_lut_to.size()
        s_lut_o_r_s, s_lut_o_c_s = stride(sparsity_lut_to)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(adapt_layout_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          sparsity_reverse_lut_from,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_to, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
          sparsity_block_size_from,
          sparsity_block_size_to))

        return output


def adapt_layout_wrapper_backward(ctx, grad_output):
    x, sparsity_layout_from, sparsity_layout_to = ctx.saved_tensors
    sparsity_block_size_from = ctx.sparsity_block_size_from
    sparsity_block_size_to = ctx.sparsity_block_size_to

    return adapt_layout(
        grad_output, sparsity_layout_to, sparsity_block_size_to, sparsity_block_size_from,
        sparsity_layout_to=sparsity_layout_from)[0], None, None, None, None, None, None, None


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size_from", "sparsity_block_size_to"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_conversion},
    reset_to_zero=["o"]
)
@triton.jit
def adapt_layout_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                        r_lut_x,
                        o,
                        o_b, o_b_s, o_r_s, o_c_s,
                        s_lut_o, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
                        sparsity_block_size_from,
                        sparsity_block_size_to,
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

    # Get equivalent sparsity block in from layout
    spa_bat_x = spa_bat_o
    spa_row_x = (spa_row_o * sparsity_block_size_to + pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size_from
    spa_col_x = (spa_col_o * sparsity_block_size_to + pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size_from

    # Get reverse sparsity indices for x
    rev_idx_spa_x_idx = (spa_bat_x * s_l_x_b_s +
                         spa_row_x * s_l_x_r_s +
                         spa_col_x * s_l_x_c_s)
    rev_idx_spa_x_msk = (rev_idx_spa_x_idx >= 0 and rev_idx_spa_x_idx < s_l_x_b * s_l_x_b_s)
    rev_idx_spa_x = tl.load(r_lut_x + rev_idx_spa_x_idx, mask=rev_idx_spa_x_msk).to(tl.int32)

    # If block is present commence operations
    if rev_idx_spa_x >= 0:
        # Calculate triton block size shifts
        shift_row_x = ((spa_row_o * sparsity_block_size_to + pid_row * TRITON_BLOCK_SIZE)
                       % sparsity_block_size_from) // TRITON_BLOCK_SIZE
        shift_col_x = ((spa_col_o * sparsity_block_size_to + pid_col * TRITON_BLOCK_SIZE)
                       % sparsity_block_size_from) // TRITON_BLOCK_SIZE

        # Load x values
        blk_x_idx = ((rev_idx_spa_x * x_b_s) +
                     ((shift_row_x * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((shift_col_x * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx >= 0 and
                     blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Store output
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx >= 0 and
                     blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


# noinspection PyUnusedLocal
def adapt_layout_setup_context(ctx, inputs, output):
    (x, sparsity_layout_from, _, sparsity_block_size_from, sparsity_layout_to, _, sparsity_block_size_to, _) = inputs

    ctx.save_for_backward(x, sparsity_layout_from, sparsity_layout_to)
    ctx.sparsity_block_size_from = sparsity_block_size_from
    ctx.sparsity_block_size_to = sparsity_block_size_to


adapt_layout_forward.register_autograd(adapt_layout_wrapper_backward, setup_context=adapt_layout_setup_context)

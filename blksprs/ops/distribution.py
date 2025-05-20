import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_dtype_int, validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def gather(src: BlksprsTensor, sparsity_layout_src: Tensor,
           dim: int,
           idx: BlksprsTensor, sparsity_layout_idx: Tensor,
           sparsity_block_size: int, lut: dict = None) -> BlksprsTensor:
    """Applies a gather operation on a block-sparse tensor in compressed form.

    Args:
        src (BlksprsTensor): The source block-sparse tensor in compressed form to gather from.
        sparsity_layout_src (Tensor): The sparsity layout of the source block-sparse tensor.
        dim (int): The dimension along which to gather.
        idx (BlksprsTensor): The block-sparse indices tensor in compressed form specifying how to gather from the source tensor.
        sparsity_layout_idx (Tensor): The sparsity layout of the indices block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The result of the gather operation as a block-sparse tensor in compressed form.

    """
    src = src.contiguous()
    idx = idx.contiguous()

    validate_dimensions(src, idx)
    validate_contiguous(src, idx)
    validate_dtype_int(idx)
    validate_device(src, idx)
    validate_sparsity(sparsity_block_size, (src, sparsity_layout_src), (idx, sparsity_layout_idx))
    validate_sparsity_block_size(sparsity_block_size, src, idx)

    adjusted_dim = dim % 3

    lut = gather_build_lut(lut, sparsity_layout_src, sparsity_layout_idx)

    return BlksprsTensor(gather_forward(src, sparsity_layout_src, lut["sparsity_reverse_lut_x"],
                                        adjusted_dim, idx, sparsity_layout_idx, lut["sparsity_lut_i"],
                                        sparsity_block_size))


@triton_op("blksprs::gather_forward", mutates_args={})
def gather_forward(x: Tensor, sparsity_layout_x: Tensor, sparsity_reverse_lut_x: Tensor,
                   dim: int, i: Tensor, _: Tensor, sparsity_lut_i: Tensor,
                   sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros_like(i, dtype=x.dtype)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_x)
        i_b, i_r, i_c = i.size()
        i_b_s, i_r_s, i_c_s = stride(i)
        s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
        s_lut_i_r_s, s_lut_i_c_s = stride(sparsity_lut_i)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(gather_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          sparsity_reverse_lut_x,
          dim,
          i,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_i, s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
          sparsity_block_size))

        return output


def gather_wrapper_backward(ctx, grad_output):
    sparsity_layout_x, i, sparsity_layout_i = ctx.saved_tensors
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size

    return scatter_reduce(grad_output, sparsity_layout_i,
                          dim, i,
                          sparsity_layout_x, sparsity_block_size,
                          reduce_op="sum"), None, None, None, None, None, None, None


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def gather_kernel(x,
                  x_b, x_b_s, x_r_s, x_c_s,
                  s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                  r_lut_x,
                  dim,
                  i,
                  i_b, i_b_s, i_r_s, i_c_s,
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

    # Load index values
    blk_i_idx = ((pid_blk * i_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
    blk_i_msk = (blk_i_idx >= 0 and
                 blk_i_idx < i_b * i_b_s)
    blk_i = tl.load(i + blk_i_idx, mask=blk_i_msk).to(tl.int32)

    # Get indices of sparsity blocks and positions within the blocks
    pos_spa_blk_x = blk_i // sparsity_block_size
    pos_spa_int_x = blk_i % sparsity_block_size

    rev_dst_bat_x = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_bat_o, dtype=tl.int32)
    rev_dst_row_x = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_row_o, dtype=tl.int32)
    rev_dst_col_x = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_col_o, dtype=tl.int32)
    dst_row_x = (((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    dst_col_x = (((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    if dim == 0:
        rev_dst_bat_x = blk_i
    elif dim == 1:
        rev_dst_row_x = pos_spa_blk_x
        dst_row_x = pos_spa_int_x * x_r_s
    elif dim == 2:
        rev_dst_col_x = pos_spa_blk_x
        dst_col_x = pos_spa_int_x * x_c_s

    # Load reverse sparsity indices for x
    rev_idx_spa_x_idx = ((rev_dst_bat_x * s_l_x_b_s) +
                         (rev_dst_row_x * s_l_x_r_s) +
                         (rev_dst_col_x * s_l_x_c_s))
    rev_idx_spa_x_msk = (rev_idx_spa_x_idx >= 0 and
                         rev_idx_spa_x_idx < s_l_x_b * s_l_x_b_s)
    rev_idx_spa_x = tl.load(r_lut_x + rev_idx_spa_x_idx, mask=rev_idx_spa_x_msk).to(tl.int32)

    # Load x values
    blk_x_idx = ((rev_idx_spa_x * x_b_s) +
                 dst_row_x +
                 dst_col_x)
    blk_x_msk = ((blk_x_idx >= 0 and
                  blk_x_idx < x_b * x_b_s) and
                 rev_idx_spa_x_msk != -1)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    # Store output
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = ((blk_o_idx >= 0 and
                  blk_o_idx < o_b * o_b_s) and
                 rev_idx_spa_x_msk != -1)
    tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


def gather_build_lut(lut: dict, sparsity_layout_src: Tensor, sparsity_layout_idx: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_reverse_lut_x" not in lut:
        sparsity_layout_x_flat = sparsity_layout_src.reshape(-1)
        sparsity_reverse_lut_x = ((torch.cumsum(sparsity_layout_x_flat, dim=-1) - 1) *
                                  (sparsity_layout_x_flat == 1) -
                                  (1 * (sparsity_layout_x_flat == 0)))
        lut["sparsity_reverse_lut_x"] = sparsity_reverse_lut_x

    if "sparsity_lut_i" not in lut:
        sparsity_lut_i = torch.nonzero(sparsity_layout_idx).contiguous()
        lut["sparsity_lut_i"] = sparsity_lut_i

    validate_contiguous(sparsity_layout_src, lut["sparsity_reverse_lut_x"],
                        sparsity_layout_idx, lut["sparsity_lut_i"])

    return lut


# noinspection PyUnusedLocal
def gather_setup_context(ctx, inputs, output):
    (_, sparsity_layout_x, _, dim, i, sparsity_layout_i, _, sparsity_block_size) = inputs

    ctx.save_for_backward(sparsity_layout_x, i, sparsity_layout_i)
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size


gather_forward.register_autograd(gather_wrapper_backward, setup_context=gather_setup_context)


def scatter(src: BlksprsTensor, sparsity_layout_src: Tensor,
            dim: int,
            idx: BlksprsTensor,
            sparsity_layout_tgt: Tensor,
            sparsity_block_size: int, lut: dict = None) -> BlksprsTensor:
    """Wrapper for ``scatter_reduce`` with ``reduce_op="none"``.

    """
    return scatter_reduce(src, sparsity_layout_src,
                          dim,
                          idx,
                          sparsity_layout_tgt,
                          sparsity_block_size,
                          reduce_op="none", lut=lut)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def scatter_reduce(src: BlksprsTensor, sparsity_layout_src: Tensor,
                   dim: int,
                   idx: BlksprsTensor,
                   sparsity_layout_tgt: Tensor,
                   sparsity_block_size: int,
                   reduce_op: str = "sum", lut: dict = None) -> BlksprsTensor:
    """Applies a scatter operation on a block-sparse tensor in compressed form.

    Args:
        src (BlksprsTensor): The source block-sparse tensor in compressed form to scatter from.
        sparsity_layout_src (Tensor): The sparsity layout of the source block-sparse tensor.
        dim (int): The dimension along which to scatter.
        idx (BlksprsTensor): The block-sparse indices tensor in compressed form specifying how to scatter to the target tensor.
        sparsity_layout_tgt (Tensor): The sparsity layout of the target block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        reduce_op (str, optional): The reduction operation to apply during the scatter operation (default ``"sum"``).
            Supported operations are ``"none"`` and ``"sum"``.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The result of the scatter operation as a block-sparse tensor in compressed form.

    """
    src = src.contiguous()
    idx = idx.contiguous()

    validate_dimensions(src, idx)
    validate_contiguous(src, idx)
    validate_dtype_int(idx)
    validate_device(src, idx)
    validate_sparsity(sparsity_block_size, (src, sparsity_layout_src), (idx, sparsity_layout_src))
    validate_sparsity_block_size(sparsity_block_size, src, idx)

    if reduce_op not in ["none", "sum"]:
        raise ValueError(f"Reduction operation '{reduce_op}' is not supported")

    adjusted_dim = dim % 3

    lut = scatter_reduce_build_lut(lut, sparsity_layout_src, sparsity_layout_tgt)

    return BlksprsTensor(scatter_reduce_forward(src, sparsity_layout_src, lut["sparsity_lut_x"],
                                                adjusted_dim, idx,
                                                sparsity_layout_tgt, lut["sparsity_reverse_lut_o"],
                                                sparsity_block_size, lut["n_sparse_blocks"],
                                                reduce_op))


@triton_op("blksprs::scatter_reduce_forward", mutates_args={})
def scatter_reduce_forward(x: Tensor, _: Tensor, sparsity_lut_x: Tensor,
                           dim: int, i: Tensor,
                           sparsity_layout_o: Tensor, sparsity_reverse_lut_o: Tensor,
                           sparsity_block_size: int, n_sparse_blocks: int,
                           reduce_op: str) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_x_r, s_lut_x_c = sparsity_lut_x.size()
        s_lut_x_r_s, s_lut_x_c_s = stride(sparsity_lut_x)
        i_b, i_r, i_c = i.size()
        i_b_s, i_r_s, i_c_s = stride(i)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        reduce_op_ind = 0
        if reduce_op == "sum":
            reduce_op_ind = 1

        (wrap_triton(scatter_reduce_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
          dim,
          i,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
          sparsity_reverse_lut_o,
          reduce_op_ind,
          sparsity_block_size))

        return output


def scatter_reduce_wrapper_backward(ctx, grad_output):
    sparsity_layout_x, i, sparsity_layout_o = ctx.saved_tensors
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size
    reduce_op = ctx.reduce_op

    if reduce_op == "sum":
        return gather(grad_output, sparsity_layout_o, dim, i, sparsity_layout_x,
                      sparsity_block_size), None, None, None, None, None, None, None, None, None
    else:
        raise ValueError(f"Reduction operation '{reduce_op}' does not support backward pass")


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def scatter_reduce_kernel(x,
                          x_b, x_b_s, x_r_s, x_c_s,
                          s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                          dim,
                          i,
                          i_b, i_b_s, i_r_s, i_c_s,
                          o,
                          o_b, o_b_s,
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
    spa_bat_x_msk = (spa_bat_x_idx >= 0 and spa_bat_x_idx < s_lut_x_r * s_lut_x_r_s)
    spa_bat_x = tl.load(s_lut_x + spa_bat_x_idx, mask=spa_bat_x_msk)

    spa_row_x_idx = (pid_blk * s_lut_x_r_s + 1 * s_lut_x_c_s)
    spa_row_x_msk = (spa_row_x_idx >= 0 and spa_row_x_idx < s_lut_x_r * s_lut_x_r_s)
    spa_row_x = tl.load(s_lut_x + spa_row_x_idx, mask=spa_row_x_msk)

    spa_col_x_idx = (pid_blk * s_lut_x_r_s + 2 * s_lut_x_c_s)
    spa_col_x_msk = (spa_col_x_idx >= 0 and spa_col_x_idx < s_lut_x_r * s_lut_x_r_s)
    spa_col_x = tl.load(s_lut_x + spa_col_x_idx, mask=spa_col_x_msk)

    # Load x values
    blk_x_idx = ((pid_blk * x_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and
                 blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    # Load index values
    blk_i_idx = ((pid_blk * i_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
    blk_i_msk = (blk_i_idx >= 0 and
                 blk_i_idx < i_b * i_b_s)
    blk_i = tl.load(i + blk_i_idx, mask=blk_i_msk).to(tl.int32)

    # Get indices of sparsity blocks and positions within the blocks
    pos_spa_blk_x = blk_i // sparsity_block_size
    pos_spa_int_x = blk_i % sparsity_block_size

    rev_dst_bat_o = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_bat_x, dtype=tl.int32)
    rev_dst_row_o = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_row_x, dtype=tl.int32)
    rev_dst_col_o = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_col_x, dtype=tl.int32)
    dst_row_o = (((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    dst_col_o = (((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    if dim == 0:
        rev_dst_bat_o = blk_i
    elif dim == 1:
        rev_dst_row_o = pos_spa_blk_x
        dst_row_o = pos_spa_int_x * x_r_s
    elif dim == 2:
        rev_dst_col_o = pos_spa_blk_x
        dst_col_o = pos_spa_int_x * x_c_s

    # Load reverse sparsity indices for o
    rev_idx_spa_o_idx = ((rev_dst_bat_o * s_l_o_b_s) +
                         (rev_dst_row_o * s_l_o_r_s) +
                         (rev_dst_col_o * s_l_o_c_s))
    rev_idx_spa_o_msk = (rev_idx_spa_o_idx >= 0 and
                         rev_idx_spa_o_idx < s_l_o_b * s_l_o_b_s)
    rev_idx_spa_o = tl.load(r_lut_o + rev_idx_spa_o_idx, mask=rev_idx_spa_o_msk).to(tl.int32)

    # Store output
    blk_o_idx = ((rev_idx_spa_o * o_b_s) +
                 dst_row_o +
                 dst_col_o)
    blk_o_msk = ((blk_o_idx >= 0 and
                  blk_o_idx < o_b * o_b_s) and
                 rev_idx_spa_o_msk != -1)

    if reduce_op_ind == 0:
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)
    elif reduce_op_ind == 1:
        tl.atomic_add(o + blk_o_idx, blk_x, mask=blk_o_msk)


def scatter_reduce_build_lut(lut: dict, sparsity_layout_src: Tensor, sparsity_layout_tgt: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_lut_x" not in lut:
        sparsity_lut_x = torch.nonzero(sparsity_layout_src).contiguous()
        lut["sparsity_lut_x"] = sparsity_lut_x

    if "sparsity_reverse_lut_o" not in lut:
        sparsity_layout_o_flat = sparsity_layout_tgt.reshape(-1)
        sparsity_reverse_lut_o = ((torch.cumsum(sparsity_layout_o_flat, dim=-1) - 1) *
                                  (sparsity_layout_o_flat == 1) -
                                  (1 * (sparsity_layout_o_flat == 0)))
        lut["sparsity_reverse_lut_o"] = sparsity_reverse_lut_o

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(sparsity_layout_tgt.to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout_src, lut["sparsity_lut_x"],
                        sparsity_layout_tgt, lut["sparsity_reverse_lut_o"])

    return lut


# noinspection PyUnusedLocal
def scatter_reduce_setup_context(ctx, inputs, output):
    (_, sparsity_layout_x, _, dim, i, sparsity_layout_o, _, sparsity_block_size, _, reduce_op) = inputs

    ctx.save_for_backward(sparsity_layout_x, i, sparsity_layout_o)
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size
    ctx.reduce_op = reduce_op


scatter_reduce_forward.register_autograd(scatter_reduce_wrapper_backward, setup_context=scatter_reduce_setup_context)

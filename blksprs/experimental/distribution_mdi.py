import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_dtype_int, validate_sparsity_block_size, validate_triton_block_size


def gather_mdi(src: Tensor, sparsity_layout_src: Tensor,
               idx_bat: Tensor,
               idx_row: Tensor,
               idx_col: Tensor,
               sparsity_layout_idx: Tensor,
               sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    src = src.contiguous()
    idx_bat = idx_bat.contiguous()
    idx_col = idx_col.contiguous()

    validate_dimensions(src, idx_bat, idx_col)
    validate_contiguous(src, idx_bat, idx_col)
    validate_dtype_int(idx_bat, idx_col)
    validate_device(src, idx_bat, idx_col)
    validate_sparsity(sparsity_block_size, (src, sparsity_layout_src),
                      (idx_bat, sparsity_layout_idx), (idx_col, sparsity_layout_idx))
    validate_sparsity_block_size(sparsity_block_size, src, idx_bat, idx_col)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_x_flat = sparsity_layout_src.reshape(-1)
    sparsity_reverse_lut_x = ((torch.cumsum(sparsity_layout_x_flat, dim=-1) - 1) *
                              (sparsity_layout_x_flat == 1) -
                              (1 * (sparsity_layout_x_flat == 0)))

    sparsity_lut_i = torch.nonzero(sparsity_layout_idx).contiguous()

    validate_contiguous(sparsity_layout_src, sparsity_reverse_lut_x,
                        sparsity_layout_idx, sparsity_lut_i)

    return _BlocksparseGatherMDI.apply(src, sparsity_layout_src, sparsity_reverse_lut_x,
                                       idx_bat, idx_col, sparsity_layout_idx, sparsity_lut_i,
                                       sparsity_block_size, triton_block_size)


class _BlocksparseGatherMDI(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_x: Tensor, sparsity_reverse_lut_x: Tensor,
                idx_bat: Tensor, idx_col: Tensor, sparsity_layout_i: Tensor, sparsity_lut_i: Tensor,
                sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
        output = torch.empty_like(idx_col, dtype=x.dtype)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_x)
        i_b, i_r, i_c = idx_col.size()
        i_b_s, i_r_s, i_c_s = stride(idx_col)
        s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
        s_lut_i_r_s, s_lut_i_c_s = stride(sparsity_lut_i)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseGatherMDI.kernel_blocksparse_gather_mdi[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          sparsity_reverse_lut_x,
          idx_bat,
          idx_col,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_i, s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
          sparsity_block_size,
          triton_block_size))

        ctx.save_for_backward(sparsity_layout_x, idx_bat, idx_col, sparsity_layout_i)
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout_x, idx_bat, idx_col, sparsity_layout_i = ctx.saved_tensors
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return scatter_reduce_mdi(grad_output, sparsity_layout_i,
                                  idx_bat,
                                  None,
                                  idx_col,
                                  sparsity_layout_x,
                                  sparsity_block_size,
                                  reduce_op="sum",
                                  triton_block_size=triton_block_size), None, None, None, None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_gather_mdi(x,
                                      x_b, x_b_s, x_r_s, x_c_s,
                                      s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                                      r_lut_x,
                                      idx_bat,
                                      idx_col,
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

        # Load batch index values
        blk_idx_bat_idx = ((pid_blk * i_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                           ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_idx_bat_msk = (blk_idx_bat_idx < i_b * i_b_s)
        blk_idx_bat = tl.load(idx_bat + blk_idx_bat_idx, mask=blk_idx_bat_msk).to(tl.int32)

        # Get position of current sparsity block row
        spa_row_o_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_o_msk = (spa_row_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_row_o = tl.load(s_lut_o + spa_row_o_idx, mask=spa_row_o_msk)

        # Load column index values
        blk_idx_col_idx = ((pid_blk * i_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                           ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_idx_col_msk = (blk_idx_col_idx < i_b * i_b_s)
        blk_idx_col = tl.load(idx_col + blk_idx_col_idx, mask=blk_idx_col_msk).to(tl.int32)

        # Get positions of sparsity blocks
        pos_spa_blk_x = blk_idx_col // sparsity_block_size
        pos_spa_col_x = blk_idx_col % sparsity_block_size

        # Load reverse sparsity indices for x
        rev_idx_spa_x_idx = ((blk_idx_bat * s_l_x_b_s) +
                             (spa_row_o * s_l_x_r_s) +
                             (pos_spa_blk_x * s_l_x_c_s))
        rev_idx_spa_x_msk = (rev_idx_spa_x_idx < s_l_x_b * s_l_x_b_s)
        rev_idx_spa_x = tl.load(r_lut_x + rev_idx_spa_x_idx, mask=rev_idx_spa_x_msk).to(tl.int32)

        # Load x values
        blk_x_idx = ((rev_idx_spa_x * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     (pos_spa_col_x * x_c_s))
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Store output
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


def scatter_reduce_mdi(src: Tensor, sparsity_layout_src: Tensor,
                       idx_bat: Tensor,
                       idx_row: Tensor,
                       idx_col: Tensor,
                       sparsity_layout_tgt: Tensor,
                       sparsity_block_size: int,
                       reduce_op: str = "sum", triton_block_size: int = None) -> Tensor:
    src = src.contiguous()
    idx_bat = idx_bat.contiguous()
    idx_col = idx_col.contiguous()

    validate_dimensions(src, idx_bat, idx_col)
    validate_contiguous(src, idx_bat, idx_col)
    validate_dtype_int(idx_bat, idx_col)
    validate_device(src, idx_bat, idx_col)
    validate_sparsity(sparsity_block_size, (src, sparsity_layout_src),
                      (idx_bat, sparsity_layout_src),
                      (idx_col, sparsity_layout_src))
    validate_sparsity_block_size(sparsity_block_size, src, idx_bat, idx_col)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    if reduce_op not in ["none", "sum"]:
        raise ValueError(f"Reduction operation '{reduce_op}' is not supported")

    sparsity_lut_x = torch.nonzero(sparsity_layout_src).contiguous()

    sparsity_layout_o_flat = sparsity_layout_tgt.reshape(-1)
    sparsity_reverse_lut_o = ((torch.cumsum(sparsity_layout_o_flat, dim=-1) - 1) *
                              (sparsity_layout_o_flat == 1) -
                              (1 * (sparsity_layout_o_flat == 0)))

    n_sparse_blocks = torch.sum(sparsity_layout_tgt.to(torch.int)).item()

    validate_contiguous(sparsity_layout_src, sparsity_lut_x,
                        sparsity_layout_tgt, sparsity_reverse_lut_o)

    return _BlocksparseScatterReduceMDI.apply(src, sparsity_layout_src, sparsity_lut_x,
                                              idx_bat,
                                              idx_col,
                                              sparsity_layout_tgt, sparsity_reverse_lut_o,
                                              sparsity_block_size, n_sparse_blocks,
                                              reduce_op, triton_block_size)


class _BlocksparseScatterReduceMDI(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_x: Tensor, sparsity_lut_x: Tensor,
                idx_bat: Tensor,
                idx_col: Tensor,
                sparsity_layout_o: Tensor, sparsity_reverse_lut_o: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int,
                reduce_op: str, triton_block_size: int) -> Tensor:
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_x_r, s_lut_x_c = sparsity_lut_x.size()
        s_lut_x_r_s, s_lut_x_c_s = stride(sparsity_lut_x)
        i_b, i_r, i_c = idx_col.size()
        i_b_s, i_r_s, i_c_s = stride(idx_col)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        reduce_op_ind = 0
        if reduce_op == "sum":
            reduce_op_ind = 1

        (_BlocksparseScatterReduceMDI.kernel_blocksparse_scatter_mdi[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
          idx_bat,
          idx_col,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
          sparsity_reverse_lut_o,
          reduce_op_ind,
          sparsity_block_size,
          triton_block_size))

        ctx.save_for_backward(sparsity_layout_x, idx_bat, idx_col, sparsity_layout_o)
        ctx.sparsity_block_size = sparsity_block_size
        ctx.reduce_op = reduce_op
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout_x, idx_bat, idx_col, sparsity_layout_o = ctx.saved_tensors
        sparsity_block_size = ctx.sparsity_block_size
        reduce_op = ctx.reduce_op
        triton_block_size = ctx.triton_block_size

        if reduce_op == "sum":
            return gather_mdi(grad_output, sparsity_layout_o,
                              idx_bat,
                              None,
                              idx_col,
                              sparsity_layout_x, sparsity_block_size,
                              triton_block_size=triton_block_size), None, None, None, None, None, None, None, None, None, None
        else:
            raise ValueError(f"Reduction operation '{reduce_op}' does not support backward pass")

    @staticmethod
    @triton.jit
    def kernel_blocksparse_scatter_mdi(x,
                                       x_b, x_b_s, x_r_s, x_c_s,
                                       s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                                       idx_bat,
                                       idx_col,
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

        # Load batch index values
        blk_idx_bat_idx = ((pid_blk * i_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                           ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_idx_bat_msk = (blk_idx_bat_idx < i_b * i_b_s)
        blk_idx_bat = tl.load(idx_bat + blk_idx_bat_idx, mask=blk_idx_bat_msk).to(tl.int32)

        # Get position of current sparsity block row
        spa_row_x_idx = (pid_blk * s_lut_x_r_s + 1 * s_lut_x_c_s)
        spa_row_x_msk = (spa_row_x_idx < s_lut_x_r * s_lut_x_r_s)
        spa_row_x = tl.load(s_lut_x + spa_row_x_idx, mask=spa_row_x_msk)

        # Load x values
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Load column index values
        blk_idx_col_idx = ((pid_blk * i_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                           ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_idx_col_msk = (blk_idx_col_idx < i_b * i_b_s)
        blk_idx_col = tl.load(idx_col + blk_idx_col_idx, mask=blk_idx_col_msk).to(tl.int32)

        # Get positions of sparsity blocks
        pos_spa_blk_o = blk_idx_col // sparsity_block_size
        pos_spa_col_o = blk_idx_col % sparsity_block_size

        # Load reverse sparsity indices for o
        rev_idx_spa_o_idx = ((blk_idx_bat * s_l_o_b_s) +
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


def build_distribution_layout_mdi(idx_bat: Tensor, idx_row: Tensor, idx_col: Tensor, sparsity_layout_idx: Tensor,
                                  size_target: torch.Size,
                                  sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    validate_dimensions(idx_bat, idx_col)
    validate_contiguous(idx_bat, idx_col)
    validate_device(idx_bat, idx_col)

    sparsity_lut_i = torch.nonzero(sparsity_layout_idx).contiguous()

    output = torch.zeros(size_target[0], size_target[1] // sparsity_block_size, size_target[2] // sparsity_block_size,
                         dtype=torch.bool, device=idx_col.device)

    i_b, i_r, i_c = idx_col.size()
    i_b_s, i_r_s, i_c_s = stride(idx_col)
    s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
    s_lut_i_r_s, s_lut_i_c_s = stride(sparsity_lut_i)
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = stride(output)

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    validate_triton_block_size(triton_block_size, sparsity_block_size)

    triton_grid = lambda meta: [i_b,
                                triton.cdiv(i_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(i_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_distribution_layout_mdi[triton_grid]
     (idx_bat,
      idx_col,
      i_b, i_b_s, i_r_s, i_c_s,
      sparsity_lut_i,
      s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
      output,
      o_b, o_b_s, o_r_s, o_c_s,
      sparsity_block_size,
      triton_block_size))

    return output


@triton.jit
def kernel_distribution_layout_mdi(idx_bat,
                                   idx_col,
                                   i_b, i_b_s, i_r_s, i_c_s,
                                   s_lut_i,
                                   s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
                                   o,
                                   o_b, o_b_s, o_r_s, o_c_s,
                                   sparsity_block_size,
                                   TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Load batch index values
    blk_idx_bat_idx = ((pid_blk * i_b_s) +
                       ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                       ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
    blk_idx_bat_msk = (blk_idx_bat_idx < i_b * i_b_s)
    blk_idx_bat = tl.load(idx_bat + blk_idx_bat_idx, mask=blk_idx_bat_msk).to(tl.int32)

    # Get position of current sparsity block row
    spa_row_i_idx = (pid_blk * s_lut_i_r_s + 1 * s_lut_i_c_s)
    spa_row_i_msk = (spa_row_i_idx < s_lut_i_r * s_lut_i_r_s)
    spa_row_i = tl.load(s_lut_i + spa_row_i_idx, mask=spa_row_i_msk)

    blk_i_idx = (pid_blk * i_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
    blk_i_msk = (blk_i_idx < i_b * i_b_s)
    blk_i = tl.load(idx_col + blk_i_idx, mask=blk_i_msk)

    blk_i = blk_i // sparsity_block_size
    blk_v = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), 1, dtype=tl.int32)

    blk_o_idx = ((blk_idx_bat * o_b_s) +
                 (spa_row_i * o_r_s) +
                 (blk_i * o_c_s))
    blk_o_msk = (blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, blk_v, mask=blk_o_msk)

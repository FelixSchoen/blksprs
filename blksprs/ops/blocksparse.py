from abc import ABC

import torch
import triton
import triton.language as tl
from torch import Tensor, Size
from torch.nn import Module


class BaseBlocksparse(Module, ABC):
    _validate = None

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__()

        self.sparsity_block_size = sparsity_block_size
        self.device = device

        self.triton_block_size = triton_block_size

        if BaseBlocksparse._validate is None:
            BaseBlocksparse._validate = True
            print(
                f"{'\033[93m'}Blocksparse validation is activated. Consider deactivating for production use.{'\033[0m'}")

    def validate_tensors(self, *tensors: Tensor, flag_dim: bool = True, flag_contiguous: bool = True,
                         flag_dtype: bool = True,
                         flag_device: bool = True) -> None:
        if not BaseBlocksparse._validate:
            return

        for tensor in tensors:
            if flag_dim:
                assert tensor.dim() == 3, "Input tensors must have 3 dimensions"
            if flag_contiguous:
                assert tensor.is_contiguous(), "Input tensors must be contiguous"
            if flag_dtype:
                assert tensor.dtype == torch.float32, "Input tensors must be of type float32"
            if flag_device:
                assert tensor.device == self.device, "Input tensors must be on the same device"

    def validate_sparsity(self, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
        if not BaseBlocksparse._validate:
            return

        for tensor_sparsity_layout_tuple in tensor_sparsity_layout_tuples:
            tensor, sparsity_layout = tensor_sparsity_layout_tuple

            assert tensor.size(-1) == tensor.size(
                -2) == self.sparsity_block_size, "Tensor not conforming to sparsity specification"
            assert tensor.size(0) == torch.sum(sparsity_layout.reshape(-1))

    @staticmethod
    def get_triton_block_size(sparsity_block_size: int, limit: int = 128):
        return min(sparsity_block_size, limit)

    @staticmethod
    def disable_validation():
        BaseBlocksparse._validate = False


# --- Matmul SSS ---

class BlocksparseMatmulSSS(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, y: Tensor,
                sparsity_layout_x: Tensor, sparsity_layout_y: Tensor, sparsity_layout_output: Tensor) -> Tensor:
        self.validate_tensors(x, y)
        self.validate_sparsity((x, sparsity_layout_x), (y, sparsity_layout_y))
        assert x.size(2) == y.size(1), "Inner dimensions must match"

        o_n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

        sparsity_layout_x_flat = sparsity_layout_x.reshape(-1)
        sparsity_reverse_lut_x = ((torch.cumsum(sparsity_layout_x_flat, dim=-1) - 1) *
                                  (sparsity_layout_x_flat == 1) -
                                  (1 * (sparsity_layout_x_flat == 0)))

        sparsity_layout_y_flat = sparsity_layout_y.reshape(-1)
        sparsity_reverse_lut_y = ((torch.cumsum(sparsity_layout_y_flat, dim=-1) - 1) *
                                  (sparsity_layout_y_flat == 1) -
                                  (1 * (sparsity_layout_y_flat == 0)))

        sparsity_lut_o = torch.nonzero(sparsity_layout_output)

        return _BlocksparseMatmulSSS.apply(x, y,
                                           sparsity_layout_x, sparsity_reverse_lut_x,
                                           sparsity_layout_y, sparsity_reverse_lut_y,
                                           sparsity_layout_output, sparsity_lut_o,
                                           self.sparsity_block_size,
                                           o_n_sparse_blocks,
                                           self.triton_block_size,
                                           self.device)


class _BlocksparseMatmulSSS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor,
                sparsity_layout_x: Tensor, sparsity_reverse_lut_x: Tensor,
                sparsity_layout_y: Tensor, sparsity_reverse_lut_y: Tensor,
                sparsity_layout_o: Tensor, sparsity_lut_o: Tensor,
                sparsity_block_size: int, o_n_sparse_blocks: int, triton_block_size: int,
                device: torch.device) -> Tensor:
        output = torch.zeros(size=(o_n_sparse_blocks, sparsity_block_size, sparsity_block_size), device=device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = sparsity_layout_x.stride()
        y_b, y_r, y_c = y.size()
        y_b_s, y_r_s, y_c_s = y.stride()
        s_l_y_b, s_l_y_r, s_l_y_c = sparsity_layout_y.size()
        s_l_y_b_s, s_l_y_r_s, s_l_y_c_s = sparsity_layout_y.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_lut_o_r, s_lut_o_c = sparsity_lut_o.size()
        s_lut_o_r_s, s_lut_o_c_s = sparsity_lut_o.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseMatmulSSS.kernel_blocksparse_matmul_sss[triton_grid]
         (x,
          x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
          sparsity_reverse_lut_x,
          y,
          y_b, y_b_s, y_r, y_r_s, y_c, y_c_s,
          s_l_y_b, s_l_y_b_s, s_l_y_r, s_l_y_r_s, s_l_y_c, s_l_y_c_s,
          sparsity_reverse_lut_y,
          output,
          o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
          sparsity_lut_o,
          s_lut_o_r, s_lut_o_r_s,
          s_lut_o_c, s_lut_o_c_s,
          sparsity_block_size,
          triton_block_size))

        return output

    @staticmethod
    @triton.jit
    def kernel_blocksparse_matmul_sss(x,
                                      x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                                      s_l_x_b, s_l_x_b_s, s_l_x_r, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
                                      r_lut_x,
                                      y,
                                      y_b, y_b_s, y_r, y_r_s, y_c, y_c_s,
                                      s_l_y_b, s_l_y_b_s, s_l_y_r, s_l_y_r_s, s_l_y_c, s_l_y_c_s,
                                      r_lut_y,
                                      o,
                                      o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                                      s_lut_o,
                                      s_lut_o_r, s_lut_o_r_s,
                                      s_lut_o_c, s_lut_o_c_s,
                                      sparsity_block_size,
                                      TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch, row, and column index
        spa_bat_o_idx = (pid_blk * s_lut_o_r_s + 0 * s_lut_o_c_s)
        spa_bat_o_msk = (spa_bat_o_idx < s_lut_o_r * s_lut_o_r_s + s_lut_o_c * s_lut_o_c_s)
        spa_bat_o = tl.load(s_lut_o + spa_bat_o_idx, mask=spa_bat_o_msk)

        spa_row_o_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_o_msk = (spa_row_o_idx < s_lut_o_r * s_lut_o_r_s + s_lut_o_c * s_lut_o_c_s)
        spa_row_o = tl.load(s_lut_o + spa_row_o_idx, mask=spa_row_o_msk)

        spa_col_o_idx = (pid_blk * s_lut_o_r_s + 2 * s_lut_o_c_s)
        spa_col_o_msk = (spa_col_o_idx < s_lut_o_r * s_lut_o_r_s + s_lut_o_c * s_lut_o_c_s)
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
            rev_idx_spa_x_idx = (spa_bat_o * s_l_x_b_s + spa_row_o * s_l_x_r_s + i_seg_spa * s_l_x_c_s)
            rev_idx_spa_x_msk = (rev_idx_spa_x_idx < s_l_x_b * s_l_x_b_s + s_l_x_r * s_l_x_r_s + s_l_x_c * s_l_x_c_s)
            rev_idx_spa_x = tl.load(r_lut_x + rev_idx_spa_x_idx, mask=rev_idx_spa_x_msk).to(tl.int32)

            # Get reverse sparsity indices for y
            rev_idx_spa_y_idx = (spa_bat_o * s_l_y_b_s + i_seg_spa * s_l_y_r_s + spa_col_o * s_l_y_c_s)
            rev_idx_spa_y_msk = (rev_idx_spa_y_idx < s_l_y_b * s_l_y_b_s + s_l_y_r * s_l_y_r_s + s_l_y_c * s_l_y_c_s)
            rev_idx_spa_y = tl.load(r_lut_y + rev_idx_spa_y_idx, mask=rev_idx_spa_y_msk).to(tl.int32)

            # If both blocks are present commence calculation
            if rev_idx_spa_x >= 0 and rev_idx_spa_y >= 0:
                blk_x_idx = ((rev_idx_spa_x * x_b_s) +
                             ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                             ((i_seg_tri_mod * TRITON_BLOCK_SIZE +
                               tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
                blk_x_msk = (blk_x_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
                blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

                blk_y_idx = ((rev_idx_spa_y * y_b_s) +
                             ((i_seg_tri_mod * TRITON_BLOCK_SIZE +
                               tl.arange(0, TRITON_BLOCK_SIZE)) * y_r_s)[:, None] +
                             ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * y_c_s)[None, :])
                blk_y_msk = (blk_y_idx < y_b * y_b_s + y_r * y_r_s + y_c * y_c_s)
                blk_y = tl.load(y + blk_y_idx, mask=blk_y_msk)

                # Perform matrix multiplication
                buf += tl.dot(blk_x, blk_y)

        # Store output
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
        tl.store(o + blk_o_idx, buf, mask=blk_o_msk)


# --- Softmax ---

class BlocksparseSoftmax(BaseBlocksparse):
    # TODO At the moment uses standard softmax instead of blocksparse improvements

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

        self.blksprs_to_dense = BlocksparseToDense(sparsity_block_size, device)
        self.blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, device)

    def forward(self, x: Tensor, sparsity_layout: Tensor) -> Tensor:
        self.validate_tensors(x)

        x_dense = self.blksprs_to_dense(x, sparsity_layout, fill_value=float('-inf'))
        x_softmax = torch.softmax(x_dense, dim=-1)
        x_sparse = self.blksprs_to_sparse(x_softmax, sparsity_layout)

        return x_sparse


# --- Transpose ---

class BlocksparseTranspose(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor, shuffle_blocks: bool = True) -> (Tensor, Tensor):
        self.validate_tensors(x)

        x_t = x.transpose(1, 2).contiguous()
        sparsity_layout_t = sparsity_layout.transpose(-1, -2).contiguous()

        shuffle_layout = (torch.cumsum(sparsity_layout.reshape(-1), dim=-1)
                          .reshape(sparsity_layout.size()).transpose(-1, -2)
                          .reshape(-1).to(torch.int) - 1)

        x_t = x_t[shuffle_layout, :, :]

        return x_t, sparsity_layout_t


# --- To Dense ---

class BlocksparseToDense(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor, fill_value: int = 0) -> Tensor:
        self.validate_tensors(x)

        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                (sparsity_layout_flat == 1) -
                                (1 * (sparsity_layout_flat == 0)))

        return _BlocksparseToDense.apply(x,
                                         sparsity_layout, sparsity_reverse_lut,
                                         self.sparsity_block_size, fill_value,
                                         self.triton_block_size, self.device)


class _BlocksparseToDense(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_block_size: int, fill_value: int,
                triton_block_size: int, device: torch.device) -> Tensor:
        output = torch.full(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                  sparsity_layout.size(2) * sparsity_block_size), fill_value=fill_value,
                            dtype=x.dtype, device=device)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = sparsity_layout.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseToDense.kernel_blocksparse_to_dense[triton_grid]
         (x,
          x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          s_l_b, s_l_b_s, s_l_r, s_l_r_s, s_l_c, s_l_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
          sparsity_block_size,
          triton_block_size))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @staticmethod
    @triton.jit
    def kernel_blocksparse_to_dense(x,
                                    x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                                    s_l_b, s_l_b_s, s_l_r, s_l_r_s, s_l_c, s_l_c_s,
                                    sparsity_reverse_lut,
                                    o,
                                    o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                                    sparsity_block_size,
                                    TRITON_BLOCK_SIZE: tl.constexpr):
        # Get triton block indices
        pid_bat = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get sparsity index of current block
        spa_row = (pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size
        spa_col = (pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size

        # Get reverse sparsity index for current block
        rev_idx_spa_idx = (pid_bat * s_l_b_s + spa_row * s_l_r_s + spa_col * s_l_c_s)
        rev_idx_spa_msk = (rev_idx_spa_idx < s_l_b * s_l_b_s + s_l_r * s_l_r_s + s_l_c * s_l_c_s)
        rev_idx_spa = tl.load(sparsity_reverse_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

        # If block is present commence operations
        if rev_idx_spa >= 0:
            blk_idx = (rev_idx_spa * x_b_s +
                       (((pid_row % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                         tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                       (((pid_col % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                         tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
            blk_msk = (blk_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
            blk = tl.load(x + blk_idx, mask=blk_msk)

            o_idx = (pid_bat * o_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
            o_msk = (o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
            tl.store(o + o_idx, blk, o_msk)


# --- To Sparse ---

class BlocksparseToSparse(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor) -> Tensor:
        self.validate_tensors(x)

        sparsity_lut = torch.nonzero(sparsity_layout)
        n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

        return _BlocksparseToSparse.apply(x,
                                          sparsity_layout, sparsity_lut,
                                          self.sparsity_block_size, n_sparse_blocks,
                                          self.triton_block_size, self.device)


class _BlocksparseToSparse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int, device: torch.device) -> Tensor:
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size), device=device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseToSparse.kernel_blocksparse_to_sparse[triton_grid]
         (x, x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c,
          s_lut_c_s,
          output, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          triton_block_size))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @staticmethod
    @triton.jit
    def kernel_blocksparse_to_sparse(x,
                                     x_b, x_b_s, x_r, x_r_s, x_c: tl.constexpr, x_c_s,
                                     s_lut, s_lut_r, s_lut_r_s, s_lut_c, s_lut_c_s,
                                     o,
                                     o_b_s, o_r_s, o_c_s,
                                     sparsity_block_size,
                                     TRITON_BLOCK_SIZE: tl.constexpr):
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get sparsity index of current output block consisting of its batch, row, and column index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
        spa_col_msk = (spa_col_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

        # Load block from dense tensor
        blk_d_idx = (spa_bat * x_b_s +
                     ((spa_row * sparsity_block_size + pid_row * TRITON_BLOCK_SIZE +
                       tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((spa_col * sparsity_block_size + pid_col * TRITON_BLOCK_SIZE +
                       tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_d_msk = (blk_d_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
        blk_d = tl.load(x + blk_d_idx, mask=blk_d_msk)

        # Store block in sparse tensor
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE) * o_c_s))[None, :])
        blk_o_msk = (blk_o_idx < (pid_blk + 1) * o_b_s)
        tl.store(o + blk_o_idx, blk_d, mask=blk_o_msk)


class BlocksparseTools:

    @staticmethod
    def do_shape_blocksparse(x: Tensor):
        if x.dim() == 3:
            return x

        return x.reshape(-1, x.size(-2), x.size(-1))

    @staticmethod
    def undo_shape_blocksparse(x: Tensor, shape: Size):
        if x.dim() == 3:
            return x

        return x.reshape(shape)

    @staticmethod
    def to_dense(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
        output = torch.zeros(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                   sparsity_layout.size(2) * sparsity_block_size), device=x.device)
        indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

        for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
            t_r = r * sparsity_block_size
            t_c = c * sparsity_block_size
            to_insert = x[idx]
            output[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size] = to_insert

        return output

    @staticmethod
    def to_sparse(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
        indices_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        output = torch.zeros(size=(indices_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)
        indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

        for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
            t_r = r * sparsity_block_size
            t_c = c * sparsity_block_size
            to_insert = x[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size]
            output[idx] = to_insert

        return output

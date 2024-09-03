import torch
from torch import Tensor, Size


class BlocksparseTools:

    # --- Shaping functions ---

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

    # --- Verification functions ---

    @staticmethod
    def slow_to_dense(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
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
    def slow_to_sparse(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
        indices_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        output = torch.zeros(size=(indices_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)
        indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

        for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
            t_r = r * sparsity_block_size
            t_c = c * sparsity_block_size
            to_insert = x[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size]
            output[idx] = to_insert

        return output

import torch
from torch import Tensor, Size


def do_shape_blocksparse(x: Tensor):
    if x.dim() == 3:
        return x, x.size()

    return x.reshape(-1, x.size(-2), x.size(-1)), x.size()


def undo_shape_blocksparse(x: Tensor, shape: Size):
    if x.shape[-2:] == shape[-2:]:
        return x

    return x.reshape((*shape[:-2], *x.shape[-2:]))

def get_triton_block_size(sparsity_block_size: int, limit: int = 128):
    return min(sparsity_block_size, limit)

#

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

def slow_to_sparse(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
    num_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
    output = torch.zeros(size=(num_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)
    indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

    for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
        t_r = r * sparsity_block_size
        t_c = c * sparsity_block_size
        to_insert = x[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size]
        output[idx] = to_insert

    return output
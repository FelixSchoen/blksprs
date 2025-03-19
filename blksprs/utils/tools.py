import torch
import triton
from torch import Tensor, Size


def do_shape_blocksparse(x: Tensor):
    if x.dim() == 3:
        return x.contiguous(), x.size()

    return x.reshape(-1, x.size(-2), x.size(-1)).contiguous(), x.size()


def undo_shape_blocksparse(x: Tensor, shape: Size):
    if x.shape[:-2] == shape[:-2]:
        return x

    return x.reshape((*shape[:-2], *x.shape[-2:]))


def get_triton_block_size(sparsity_block_size: int, limit: int = 128):
    return min(sparsity_block_size, limit)


def stride(x: Tensor):
    if x.dim() == 2:
        return x.size(1), 1
    elif x.dim() == 3:
        return x.size(1) * x.size(2), x.size(2), 1
    else:
        raise NotImplementedError


@torch.compile
def get_autotune_configs():
    configs = []
    config_parameters = [
        (16, 3, 8),
        (16, 4, 4),
        (16, 4, 8),
        (16, 5, 2),
        (16, 5, 4),
        (16, 5, 8),

        (32, 3, 8),
        (32, 4, 4),
        (32, 4, 8),
        (32, 5, 2),
        (32, 5, 4),

        (64, 3, 8),
        (64, 4, 4),
        (64, 4, 8),
        (64, 5, 2),

        (128, 3, 8),
        (128, 4, 4),
        (128, 5, 2),
    ]

    for block_size, num_stages, num_warps in config_parameters:
        configs.append(triton.Config({"TRITON_BLOCK_SIZE": block_size}, num_stages=num_stages, num_warps=num_warps))

    return configs

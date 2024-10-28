from torch import Tensor, Size

from blksprs.utils.validation import _set_skip_validation


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


def disable_validation():
    _set_skip_validation(True)

def stride(x: Tensor):
    return x.view(x.shape).stride()
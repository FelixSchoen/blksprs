import torch
from torch import Tensor, Size


def build_reverse_lut(sparsity_layout: Tensor) -> Tensor:
    """Builds a reverse look-up table from a sparsity layout.

    Maps each position in the flattened sparsity layout to its index among the non-zero elements. Positions
    corresponding to zero entries are mapped to ``-1``.

    Args:
        sparsity_layout (Tensor): A sparsity layout tensor containing only ``0`` and ``1`` values.

    Returns:
        Tensor: A 1D tensor of the same length as the flattened input, where each position holds the cumulative
            index of the non-zero entry, or ``-1`` if the entry is zero.

    """
    sparsity_layout_flat = sparsity_layout.reshape(-1)
    return ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
            (sparsity_layout_flat == 1) -
            (1 * (sparsity_layout_flat == 0)))


def do_shape_blocksparse(x: Tensor) -> tuple[Tensor, Size]:
    if x.dim() == 3:
        return x.contiguous(), x.size()

    return x.reshape(-1, x.size(-2), x.size(-1)).contiguous(), x.size()


def undo_shape_blocksparse(x: Tensor, shape: Size | tuple[int, ...]) -> Tensor:
    if x.shape[:-2] == shape[:-2]:
        return x

    return x.reshape((*shape[:-2], *x.shape[-2:]))


def stride(x: Tensor):
    if x.dim() == 1:
        return 1
    elif x.dim() == 2:
        return x.size(1), 1
    elif x.dim() == 3:
        return x.size(1) * x.size(2), x.size(2), 1
    else:
        raise NotImplementedError(f"stride() not implemented for {x.dim()}-dimensional tensors")


def ceil_pow2(x: int) -> int:
    """Returns the smallest power of 2 that is greater than or equal to x."""
    if x <= 0:
        raise ValueError("Input must be a positive integer.")
    return 1 << (x - 1).bit_length()

import torch
from torch import Tensor


def validate_dimensions(*tensors: Tensor) -> None:
    if _skip_validation():
        return

    for tensor in tensors:
        if tensor.dim() != 3:
            raise ValueError("Tensor must have 3 dimensions")


def validate_contiguous(*tensors: Tensor) -> None:
    if _skip_validation():
        return

    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")


def validate_dtype_float(*tensors: Tensor) -> None:
    if _skip_validation():
        return

    for tensor in tensors:
        if tensor.dtype != torch.float32:
            raise ValueError("Tensor must have float32 dtype")


def validate_dtype_int(*tensors: Tensor) -> None:
    if _skip_validation():
        return

    for tensor in tensors:
        if tensor.dtype != torch.int32 and tensor.dtype != torch.int64:
            raise ValueError("Tensor must have int32 or int64 dtype")


def validate_device(*tensors: Tensor) -> None:
    if _skip_validation():
        return

    device = None

    for i, tensor in enumerate(tensors):
        if i == 0:
            device = tensor.device

            if not device.type == 'cuda':
                raise ValueError("Tensors must be on GPU")

        if tensor.device != device:
            raise ValueError("Tensors must be on same device")


def validate_sparsity(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    if _skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if not (tensor.size(-1) == tensor.size(-2) == sparsity_block_size):
            raise ValueError("Blocks not conforming to sparsity block size")
        if not tensor.size(0) == torch.sum(sparsity_layout.reshape(-1)):
            raise ValueError("Mismatch between sparsity layout and blocks")


def _validate_sparsity_layout_values(sparsity_layout: Tensor):
    if not torch.all(torch.logical_or(sparsity_layout == 0, sparsity_layout == 1)):
        raise ValueError("Sparsity layout values must be either 0 or 1")

def validate_sparsity_block_size(sparsity_block_size: int, *tensors):
    if _skip_validation():
        return

    if not (sparsity_block_size & (sparsity_block_size - 1)) == 0:
        raise ValueError("Sparsity block size must be a power of 2")

    for tensor in tensors:
        if not (tensor.size(-1) % sparsity_block_size == 0 and tensor.size(-2) % sparsity_block_size == 0):
            raise ValueError("Tensor sizes must be divisible by sparsity block size")

def validate_triton_block_size(triton_block_size: int, sparsity_block_size: int):
    if _skip_validation():
        return

    if triton_block_size is None:
        return

    if triton_block_size > sparsity_block_size:
        raise ValueError("Triton block size cannot be larger than sparsity block size")

def _skip_validation():
    return False
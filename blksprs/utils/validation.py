import torch
from torch import Tensor

VALIDATION = True


def validate_dimensions(*tensors: Tensor, dims=3) -> None:
    if _check_skip_validation():
        return

    for tensor in tensors:
        if tensor.dim() != dims:
            raise ValueError(f"Tensor must have {dims} dimensions")


def validate_contiguous(*tensors: Tensor) -> None:
    if _check_skip_validation():
        return

    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")


def validate_dtype_float(*tensors: Tensor) -> None:
    if _check_skip_validation():
        return

    dtype = None

    for i, tensor in enumerate(tensors):
        if i == 0:
            dtype = tensor.dtype

        if tensor.dtype != torch.float16 and tensor.dtype != torch.float32:
            raise ValueError("Tensor must have either float16 or float32 dtype")

        if tensor.dtype != dtype:
            raise ValueError("Tensors must have same dtype")


def validate_dtype_float_32(*tensors: Tensor) -> None:
    if _check_skip_validation():
        return

    for tensor in tensors:
        if tensor.dtype != torch.float32:
            raise ValueError("Tensor must have float32 dtype")


def validate_dtype_int(*tensors: Tensor) -> None:
    if _check_skip_validation():
        return

    for tensor in tensors:
        if (tensor.dtype !=
                torch.int32 and tensor.dtype != torch.int64):
            raise ValueError("Tensor must have either int32 or int64 dtype")


def validate_device(*tensors: Tensor) -> None:
    if _check_skip_validation():
        return

    device = None

    for i, tensor in enumerate(tensors):
        if i == 0:
            device = tensor.device

            if not device.type == "cuda":
                raise ValueError("Tensors must be on GPU")

        if tensor.device != device:
            raise ValueError("Tensors must be on same device")


def validate_sparsity(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    if _check_skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        if not (tensor.size(-1) == tensor.size(-2) == sparsity_block_size):
            raise ValueError("Blocks not conforming to sparsity block size")
        if not tensor.size(0) == torch.sum(sparsity_layout.reshape(-1)):
            raise ValueError("Mismatch between sparsity layout and blocks")


def validate_sparsity_dense(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    if _check_skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        if not (tensor.size(-1) // sparsity_block_size == sparsity_layout.size(-1) and
                tensor.size(-2) // sparsity_block_size == sparsity_layout.size(-2)):
            raise ValueError("Tensor not conforming to sparsity layout")


def _validate_sparsity_layout_values(sparsity_layout: Tensor):
    if not torch.all(torch.logical_or(sparsity_layout == 0, sparsity_layout == 1)):
        raise ValueError("Sparsity layout values must be either 0 or 1")


def validate_sparsity_block_size(sparsity_block_size: int, *tensors):
    if _check_skip_validation():
        return

    if not sparsity_block_size >= 16:
        raise ValueError("Sparsity block size must be at least 16")

    if not (sparsity_block_size & (sparsity_block_size - 1)) == 0:
        raise ValueError("Sparsity block size must be a power of 2")

    for tensor in tensors:
        if not (tensor.size(-1) % sparsity_block_size == 0 and tensor.size(-2) % sparsity_block_size == 0):
            raise ValueError("Tensor sizes must be divisible by sparsity block size")


def _check_skip_validation():
    return not VALIDATION


def _set_skip_validation(skip_validation: bool):
    global VALIDATION
    VALIDATION = not skip_validation


def disable_validation():
    _set_skip_validation(True)

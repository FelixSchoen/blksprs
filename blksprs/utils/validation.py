import threading

import torch
from torch import Tensor

_thread_local = threading.local()

_CONTIGUOUS_DEFAULT = True
_VALIDATION_DEFAULT = True


def _get_contiguous():
    return getattr(_thread_local, "contiguous", _CONTIGUOUS_DEFAULT)


def _get_validation():
    return getattr(_thread_local, "validation", _VALIDATION_DEFAULT)


def ensure_contiguous(*tensors: Tensor) -> tuple[Tensor, ...] | Tensor:
    transformed = tensors

    if _check_contiguous():
        transformed = tuple(tensor.contiguous() for tensor in tensors)

    return transformed[0] if len(transformed) == 1 else transformed


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

        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Tensor must have either float16, bfloat16, or float32 dtype")

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


def validate_shape(tensor: Tensor, expected_shape, name: str = "Tensor") -> None:
    if _check_skip_validation():
        return

    expected_shape = torch.Size(expected_shape)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} doesn't match expected {tuple(expected_shape)}")


def validate_positive_integer(value: int, name: str) -> None:
    if _check_skip_validation():
        return

    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_positive_integer_tuple(values: tuple[int, ...], length: int, name: str) -> None:
    if _check_skip_validation():
        return

    if len(values) != length or any(value < 1 for value in values):
        raise ValueError(f"{name} must contain {length} positive integers")


def validate_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    if _check_skip_validation():
        return

    _validate_divisible(value, divisor, value_name, divisor_name)


def _validate_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{value_name} must be divisible by {divisor_name}")


def validate_sparsity(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    if _check_skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        if not (tensor.size(-1) == tensor.size(-2) == sparsity_block_size):
            raise ValueError("Blocks not conforming to sparsity block size")
        if not tensor.size(0) == torch.sum(sparsity_layout.reshape(-1).to(torch.int)):
            raise ValueError("Mismatch between sparsity layout and blocks")


def validate_sparsity_dense(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    if _check_skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        _validate_divisible(tensor.size(-1), sparsity_block_size, "Tensor sizes", "sparsity block size")
        _validate_divisible(tensor.size(-2), sparsity_block_size, "Tensor sizes", "sparsity block size")
        if not (tensor.size(-1) // sparsity_block_size == sparsity_layout.size(-1) and
                tensor.size(-2) // sparsity_block_size == sparsity_layout.size(-2)):
            raise ValueError("Tensor not conforming to sparsity layout")


def validate_sparsity_layout(*sparsity_layouts: Tensor) -> None:
    if _check_skip_validation():
        return

    for sparsity_layout in sparsity_layouts:
        _validate_sparsity_layout_values(sparsity_layout)

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")


def _validate_sparsity_layout_values(sparsity_layout: Tensor):
    if sparsity_layout.dtype == torch.bool:
        return

    if not torch.all(torch.logical_or(sparsity_layout == 0, sparsity_layout == 1)):
        raise ValueError("Sparsity layout values must be either 0 or 1")


def validate_sparsity_block_size(sparsity_block_size: int, *values):
    if _check_skip_validation():
        return

    if not sparsity_block_size >= 16:
        raise ValueError("Sparsity block size must be at least 16")

    if not (sparsity_block_size & (sparsity_block_size - 1)) == 0:
        raise ValueError("Sparsity block size must be a power of 2")

    for value in values:
        size = value if isinstance(value, (torch.Size, tuple, list)) else value.size()
        _validate_divisible(size[-1], sparsity_block_size, "Tensor sizes", "sparsity block size")
        _validate_divisible(size[-2], sparsity_block_size, "Tensor sizes", "sparsity block size")


def _check_contiguous():
    return _get_contiguous()


def disable_contiguous():
    """Disables automatic contiguous conversion for the current thread."""
    _thread_local.contiguous = False


def enable_contiguous():
    """Enables automatic contiguous conversion for the current thread."""
    _thread_local.contiguous = True


def _check_skip_validation():
    return not _get_validation()


def disable_validation():
    """Disables input validation for the current thread."""
    _thread_local.validation = False


def enable_validation():
    """Enables input validation for the current thread."""
    _thread_local.validation = True

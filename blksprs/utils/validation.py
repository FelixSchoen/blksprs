import operator
import threading
from typing import TypeVar, overload

import torch
from torch import Tensor

_thread_local = threading.local()

_CONTIGUOUS_DEFAULT = True
_VALIDATION_DEFAULT = True

TensorT = TypeVar("TensorT", bound=Tensor)
TensorT2 = TypeVar("TensorT2", bound=Tensor)
TensorT3 = TypeVar("TensorT3", bound=Tensor)
TensorT4 = TypeVar("TensorT4", bound=Tensor)


def _get_contiguous():
    return getattr(_thread_local, "contiguous", _CONTIGUOUS_DEFAULT)


def _get_validation():
    return getattr(_thread_local, "validation", _VALIDATION_DEFAULT)


@overload
def ensure_contiguous(tensor: TensorT, /) -> TensorT:
    ...


@overload
def ensure_contiguous(tensor: TensorT, tensor_2: TensorT2, /) -> tuple[TensorT, TensorT2]:
    ...


@overload
def ensure_contiguous(tensor: TensorT, tensor_2: TensorT2,
                      tensor_3: TensorT3, /) -> tuple[TensorT, TensorT2, TensorT3]:
    ...


@overload
def ensure_contiguous(tensor: TensorT, tensor_2: TensorT2, tensor_3: TensorT3,
                      tensor_4: TensorT4, /) -> tuple[TensorT, TensorT2, TensorT3, TensorT4]:
    ...


@overload
def ensure_contiguous(*tensors: Tensor) -> tuple[Tensor, ...]:
    ...


def ensure_contiguous(*tensors: Tensor) -> tuple[Tensor, ...] | Tensor:
    """Returns contiguous tensors when automatic contiguous conversion is enabled."""
    transformed = tensors

    if _check_contiguous():
        transformed = tuple(tensor.contiguous() for tensor in tensors)

    return transformed[0] if len(transformed) == 1 else transformed


def validate_dimensions(*tensors: Tensor, dims=3) -> None:
    """Validates that every tensor has the requested number of dimensions."""
    if _check_skip_validation():
        return

    for tensor in tensors:
        if tensor.dim() != dims:
            raise ValueError(f"Tensor must have {dims} dimensions")


def validate_contiguous(*tensors: Tensor) -> None:
    """Validates that every tensor is contiguous."""
    if _check_skip_validation():
        return

    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")


def validate_dtype_float(*tensors: Tensor) -> None:
    """Validates matching ``float16``, ``bfloat16``, or ``float32`` dtypes."""
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
    """Validates that every tensor uses the ``float32`` dtype."""
    if _check_skip_validation():
        return

    for tensor in tensors:
        if tensor.dtype != torch.float32:
            raise ValueError("Tensor must have float32 dtype")


def validate_dtype_int(*tensors: Tensor) -> None:
    """Validates that every tensor uses the ``int32`` or ``int64`` dtype."""
    if _check_skip_validation():
        return

    for tensor in tensors:
        if (tensor.dtype !=
                torch.int32 and tensor.dtype != torch.int64):
            raise ValueError("Tensor must have either int32 or int64 dtype")


def validate_dtype_integral(*tensors: Tensor) -> None:
    """Validates that every tensor uses a supported integer dtype."""
    if _check_skip_validation():
        return

    for tensor in tensors:
        if tensor.dtype not in (
                torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Tensor must have an integer dtype")


def validate_dtype_supported(*tensors: Tensor) -> None:
    """Validates dtypes supported by the generic Triton data-movement kernels."""
    if _check_skip_validation():
        return

    supported_dtypes = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )

    for tensor in tensors:
        if tensor.dtype not in supported_dtypes:
            raise ValueError("Tensor dtype is not supported")


def validate_same_dtype(*tensors: Tensor) -> None:
    """Validates that every tensor uses the same dtype."""
    if _check_skip_validation() or not tensors:
        return

    dtype = tensors[0].dtype

    for tensor in tensors[1:]:
        if tensor.dtype != dtype:
            raise ValueError("Tensors must have same dtype")


def validate_binary(*tensors: Tensor) -> None:
    """Validates tensors whose values represent a binary mask."""
    if _check_skip_validation():
        return

    for tensor in tensors:
        if not torch.all(torch.logical_or(tensor == 0, tensor == 1)):
            raise ValueError("Tensor values must be either 0 or 1")


def validate_device(*tensors: Tensor) -> None:
    """Validates that every tensor is on the same CUDA device."""
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
    """Validates a tensor against an expected shape."""
    if _check_skip_validation():
        return

    expected_shape = torch.Size(expected_shape)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} doesn't match expected {tuple(expected_shape)}")


def validate_positive_integer(value: int, name: str) -> None:
    """Validates that a value is a positive integer."""
    if _check_skip_validation():
        return

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_non_negative_integer(value: int, name: str) -> None:
    """Validates that a value is a non-negative integer."""
    if _check_skip_validation():
        return

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc

    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def validate_positive_integer_tuple(values: tuple[int, ...], length: int, name: str) -> None:
    """Validates the length and positive values of an integer tuple."""
    if _check_skip_validation():
        return

    if len(values) != length:
        raise ValueError(f"{name} must contain {length} positive integers")

    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain integers")

        try:
            value = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} must contain integers") from exc

        if value < 1:
            raise ValueError(f"{name} must contain {length} positive integers")


def validate_non_negative_integer_tuple(values: tuple[int, ...], length: int, name: str) -> None:
    """Validates the length and non-negative values of an integer tuple."""
    if _check_skip_validation():
        return

    if len(values) != length:
        raise ValueError(f"{name} must contain {length} non-negative integers")

    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain integers")

        try:
            value = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} must contain integers") from exc

        if value < 0:
            raise ValueError(f"{name} must contain {length} non-negative integers")


def validate_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    """Validates that one integer is divisible by another."""
    if _check_skip_validation():
        return

    _validate_divisible(value, divisor, value_name, divisor_name)


def _validate_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{value_name} must be divisible by {divisor_name}")


def validate_sparsity(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    """Validates compressed tensors and their corresponding sparsity layouts."""
    if _check_skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if tensor.device != sparsity_layout.device:
            raise ValueError("Tensor and sparsity layout must be on same device")
        if not sparsity_layout.is_contiguous():
            raise ValueError("Sparsity layout must be contiguous")

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        if not (tensor.size(-1) == tensor.size(-2) == sparsity_block_size):
            raise ValueError("Blocks not conforming to sparsity block size")
        if not tensor.size(0) == torch.sum(sparsity_layout.reshape(-1).to(torch.int)):
            raise ValueError("Mismatch between sparsity layout and blocks")


def validate_sparsity_dense(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    """Validates dense tensors against their corresponding sparsity layouts."""
    if _check_skip_validation():
        return

    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        _validate_sparsity_layout_values(sparsity_layout)

        if tensor.device != sparsity_layout.device:
            raise ValueError("Tensor and sparsity layout must be on same device")
        if not sparsity_layout.is_contiguous():
            raise ValueError("Sparsity layout must be contiguous")

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        if tensor.size(0) != sparsity_layout.size(0):
            raise ValueError("Tensor batch dimension does not match sparsity layout")
        _validate_divisible(tensor.size(-1), sparsity_block_size, "Tensor sizes", "sparsity block size")
        _validate_divisible(tensor.size(-2), sparsity_block_size, "Tensor sizes", "sparsity block size")
        if not (tensor.size(-1) // sparsity_block_size == sparsity_layout.size(-1) and
                tensor.size(-2) // sparsity_block_size == sparsity_layout.size(-2)):
            raise ValueError("Tensor not conforming to sparsity layout")


def validate_sparsity_layout(*sparsity_layouts: Tensor) -> None:
    """Validates three-dimensional, contiguous, real-valued binary CUDA sparsity layouts."""
    if _check_skip_validation():
        return

    for sparsity_layout in sparsity_layouts:
        _validate_sparsity_layout_values(sparsity_layout)

        if not sparsity_layout.dim() == 3:
            raise ValueError("Sparsity layout must have exactly 3 dimensions")
        if not sparsity_layout.is_contiguous():
            raise ValueError("Sparsity layout must be contiguous")
        if sparsity_layout.device.type != "cuda":
            raise ValueError("Sparsity layout must be on GPU")


def _validate_sparsity_layout_values(sparsity_layout: Tensor):
    if sparsity_layout.is_complex():
        raise ValueError("Sparsity layout must not have a complex dtype")

    if sparsity_layout.dtype == torch.bool:
        return

    if not torch.all(torch.logical_or(sparsity_layout == 0, sparsity_layout == 1)):
        raise ValueError("Sparsity layout values must be either 0 or 1")


def validate_sparsity_block_size(sparsity_block_size: int, *values):
    """Validates a block size and its divisibility into tensors or shapes."""
    if _check_skip_validation():
        return

    if isinstance(sparsity_block_size, bool) or not isinstance(sparsity_block_size, int):
        raise TypeError("Sparsity block size must be an integer")

    if not sparsity_block_size >= 16:
        raise ValueError("Sparsity block size must be at least 16")

    if not (sparsity_block_size & (sparsity_block_size - 1)) == 0:
        raise ValueError("Sparsity block size must be a power of 2")

    for value in values:
        size = value if isinstance(value, (torch.Size, tuple, list)) else value.size()
        _validate_divisible(size[-1], sparsity_block_size, "Tensor sizes", "sparsity block size")
        _validate_divisible(size[-2], sparsity_block_size, "Tensor sizes", "sparsity block size")


def validate_dimension(dim: int, dimensions: int = 3) -> int:
    """Validates and normalises a possibly negative tensor dimension."""
    if _check_skip_validation():
        return operator.index(dim) % dimensions

    try:
        dim = operator.index(dim)
    except TypeError as exc:
        raise TypeError("Dimension must be an integer") from exc

    if dim < -dimensions or dim >= dimensions:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-dimensions}, {dimensions - 1}], but got {dim})"
        )

    return dim % dimensions


def validate_indices(indices: Tensor, dim: int, target_size: torch.Size | tuple[int, ...]) -> None:
    """Validates gather/scatter indices against a logical target shape."""
    if _check_skip_validation() or indices.numel() == 0:
        return

    lower, upper = torch.aminmax(indices)
    lower_value = int(lower.item())
    upper_value = int(upper.item())
    dimension_size = int(target_size[dim])

    if lower_value < 0 or upper_value >= dimension_size:
        invalid_value = lower_value if lower_value < 0 else upper_value
        raise IndexError(
            f"Index {invalid_value} is out of bounds for dimension {dim} "
            f"with size {dimension_size}"
        )


def validate_distribution_shape(index_size: torch.Size | tuple[int, ...],
                                target_size: torch.Size | tuple[int, ...],
                                dim: int) -> None:
    """Validates gather/scatter sizes outside the indexed dimension."""
    if _check_skip_validation():
        return

    index_size = torch.Size(index_size)
    target_size = torch.Size(target_size)

    if len(index_size) != 3 or len(target_size) != 3:
        raise ValueError("Distribution tensor sizes must have exactly 3 dimensions")

    for axis, (index_axis_size, target_axis_size) in enumerate(zip(index_size, target_size)):
        if axis != dim and index_axis_size > target_axis_size:
            raise ValueError(
                "Index tensor size must not exceed target tensor size "
                f"in dimension {axis}"
            )


def _check_contiguous():
    return _get_contiguous()


def disable_contiguous() -> None:
    """Disables automatic contiguous conversion for the current thread."""
    _thread_local.contiguous = False


def enable_contiguous() -> None:
    """Enables automatic contiguous conversion for the current thread."""
    _thread_local.contiguous = True


def _check_skip_validation():
    return not _get_validation()


def disable_validation() -> None:
    """Disables input validation for the current thread."""
    _thread_local.validation = False


def enable_validation() -> None:
    """Enables input validation for the current thread."""
    _thread_local.validation = True

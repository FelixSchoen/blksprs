from torch import Tensor


def validate_contiguous(*tensors: Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")
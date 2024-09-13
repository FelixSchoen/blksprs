import torch
from torch import Tensor


def validate_dimensions(*tensors: Tensor) -> None:
    for tensor in tensors:
        if tensor.dim() != 3:
            raise ValueError("Tensor must have 3 dimensions")


def validate_contiguous(*tensors: Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous")


def validate_dtype_float(*tensors: Tensor) -> None:
    for tensor in tensors:
        if tensor.dtype != torch.float32:
            raise ValueError("Tensor must have float32 dtype")


def validate_device(*tensors: Tensor) -> None:
    device = None

    for i, tensor in enumerate(tensors):
        if i == 0:
            device = tensor.device

            if not device.type == 'cuda':
                raise ValueError("Tensors must be on GPU")

        if tensor.device != device:
            raise ValueError("Tensors must be on same device")

def validate_sparsity(sparsity_block_size: int, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
    for (tensor, sparsity_layout) in tensor_sparsity_layout_tuples:
        if not (tensor.size(-1) == tensor.size(-2) == sparsity_block_size):
            raise ValueError("Blocks not conforming to sparsity block size")
        if not tensor.size(0) == torch.sum(sparsity_layout.reshape(-1)):
            raise ValueError("Mismatch between sparsity layout and blocks")

def validate_triton_block_size(triton_block_size: int, sparsity_block_size: int):
    if triton_block_size > sparsity_block_size:
        raise ValueError("Triton block size cannot be larger than sparsity block size")
from abc import ABC

import torch
from torch import Tensor
from torch.nn import Module


class BaseBlocksparse(Module, ABC):
    _validate = None

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__()

        self.sparsity_block_size = sparsity_block_size
        self.device = device

        self.triton_block_size = triton_block_size

        if BaseBlocksparse._validate is None:
            BaseBlocksparse._validate = True
            # print(
            #     f"{'\033[93m'}Blocksparse validation is activated. Consider deactivating for production use.{'\033[0m'}")

    def validate_tensors(self, *tensors: Tensor, flag_dim: bool = True, flag_contiguous: bool = True,
                         flag_dtype: bool = True,
                         flag_device: bool = True) -> None:
        if not BaseBlocksparse._validate:
            return

        for tensor in tensors:
            if flag_dim:
                assert tensor.dim() == 3, "Input tensors must have 3 dimensions"
            if flag_contiguous:
                assert tensor.is_contiguous(), "Input tensors must be contiguous"
            if flag_dtype:
                assert tensor.dtype == torch.float32, "Input tensors must be of type float32"
            if flag_device:
                assert tensor.device == self.device, "Input tensors must be on the same device"

    def validate_sparsity(self, *tensor_sparsity_layout_tuples: tuple[Tensor, Tensor]) -> None:
        if not BaseBlocksparse._validate:
            return

        for tensor_sparsity_layout_tuple in tensor_sparsity_layout_tuples:
            tensor, sparsity_layout = tensor_sparsity_layout_tuple

            assert tensor.size(-1) == tensor.size(-2) == self.sparsity_block_size, \
                "Tensor not conforming to sparsity specification"
            assert tensor.size(0) == torch.sum(sparsity_layout.reshape(-1))

    @staticmethod
    def get_triton_block_size(sparsity_block_size: int, limit: int = 128):
        return min(sparsity_block_size, limit)

    @staticmethod
    def disable_validation():
        BaseBlocksparse._validate = False

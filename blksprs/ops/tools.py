from abc import ABC

import torch
from torch import Tensor, Size
from torch.nn import Module

# TODO Add type hints for all methods
# TODO Remove pid_row % where it is not needed

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


class BlocksparseTools:

    # --- Shaping functions ---

    @staticmethod
    def do_shape_blocksparse(x: Tensor):
        if x.dim() == 3:
            return x

        return x.reshape(-1, x.size(-2), x.size(-1))

    @staticmethod
    def undo_shape_blocksparse(x: Tensor, shape: Size):
        if x.dim() == 3:
            return x

        return x.reshape(shape)

    # --- Verification functions ---

    @staticmethod
    def slow_to_dense(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
        output = torch.zeros(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                   sparsity_layout.size(2) * sparsity_block_size), device=x.device)
        indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

        for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
            t_r = r * sparsity_block_size
            t_c = c * sparsity_block_size
            to_insert = x[idx]
            output[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size] = to_insert

        return output

    @staticmethod
    def slow_to_sparse(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int):
        indices_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        output = torch.zeros(size=(indices_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)
        indices_sparse_blocks = sparsity_layout.nonzero(as_tuple=True)

        for idx, (b, r, c) in enumerate(zip(*indices_sparse_blocks)):
            t_r = r * sparsity_block_size
            t_c = c * sparsity_block_size
            to_insert = x[b, t_r:t_r + sparsity_block_size, t_c:t_c + sparsity_block_size]
            output[idx] = to_insert

        return output

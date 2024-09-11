from abc import ABC

import torch
from torch import Tensor, Size
from torch.nn import Module


# TODO Remove object creation from conversion and matmul backwards?
# TODO Implement Triton kernels for transpose?
# TODO Replace object creation in backward passes?

class BaseBlocksparse(Module, ABC):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__()

        self.sparsity_block_size = sparsity_block_size
        self.device = device

        self.triton_block_size = triton_block_size
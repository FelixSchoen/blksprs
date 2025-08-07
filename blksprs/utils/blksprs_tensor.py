from typing import Union

import torch
from torch import Tensor


class BlksprsTensor(Tensor):
    """A wrapper class representing a block-sparse tensor in compressed form.
    """

    def __repr__(self):
        return f"BlksprsTensor({torch.Tensor(self).__repr__()})"

    @staticmethod
    def wrap(tensor: Tensor) -> Union[Tensor, "BlksprsTensor"]:
        if torch._dynamo.is_compiling():
            return tensor
        else:
            return BlksprsTensor(tensor)

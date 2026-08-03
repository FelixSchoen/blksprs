from typing import Any, cast

import torch
from torch import Tensor


class BlksprsTensor(Tensor):
    """A tensor subclass marking a block-sparse tensor in compressed form."""

    def __repr__(self, *, tensor_contents=None):
        tensor_repr = cast(Any, torch.Tensor(self).__repr__)(tensor_contents=tensor_contents)
        return f"BlksprsTensor({tensor_repr})"

    @staticmethod
    def wrap(tensor: Tensor) -> "BlksprsTensor":
        """Marks a compressed tensor outside Dynamo tracing.

        Dynamo tracing retains a base :class:`torch.Tensor` because constructing
        tensor subclasses inside a compiled graph is unsupported. Both forms use
        the same compressed representation and are accepted by BLK-SPRS operations.
        """
        if torch._dynamo.is_compiling():
            return cast(BlksprsTensor, tensor)
        else:
            return BlksprsTensor(tensor)

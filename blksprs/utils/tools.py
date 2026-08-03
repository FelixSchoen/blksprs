from typing import Any

import torch
from torch import Tensor, Size

INT32_INDEX_MAX = 2_147_483_647

LayoutCache = dict[str, Any]

_LAYOUT_CACHE_KEY = "_blksprs_cache_key"
_LAYOUT_CACHE_REFS = "_blksprs_cache_refs"
_LAYOUT_CACHE_STATE = "_blksprs_cache_state"
_LAYOUT_CACHE_METADATA = frozenset({
    _LAYOUT_CACHE_KEY,
    _LAYOUT_CACHE_REFS,
    _LAYOUT_CACHE_STATE,
})


def as_base_tensor(value: Tensor) -> Tensor:
    """Returns ``value`` as an exact :class:`torch.Tensor`.

    PyTorch propagates tensor subclasses through most operations. Layouts and
    index metadata are regular tensors in the BLK-SPRS API, so strip any
    subclass without copying their storage.

    """
    if type(value) is Tensor:
        return value
    return Tensor(value)


def build_layout_indices(sparsity_layout: Tensor) -> Tensor:
    """Builds contiguous base-tensor indices for active layout blocks."""
    return as_base_tensor(torch.nonzero(sparsity_layout).contiguous())


def build_packed_indices(sparsity_layout: Tensor) -> Tensor:
    """Builds packed indices from a sparsity layout.

    Maps each position in the flattened sparsity layout to its index among the non-zero elements. Positions
    corresponding to zero entries are mapped to ``-1``.

    Args:
        sparsity_layout (Tensor): A sparsity layout tensor containing only ``0`` and ``1`` values.

    Returns:
        Tensor: A 1D tensor of the same length as the flattened input, where each position holds the cumulative
            index of the non-zero entry, or ``-1`` if the entry is zero.

    """
    sparsity_layout_flat = sparsity_layout.reshape(-1).to(torch.int64)
    return as_base_tensor(torch.where(
        sparsity_layout_flat == 1,
        torch.cumsum(sparsity_layout_flat, dim=-1) - 1,
        -1,
    ))


def prepare_layout_cache(layout_cache: LayoutCache | None, operation: str,
                         *values: object) -> LayoutCache:
    """Prepares an operation-specific cache and invalidates stale metadata.

    Explicit caches are tied to the exact tensor objects, their in-place
    versions, gradient requirements, and all scalar parameters used to derive
    cached metadata. A cache passed to another operation or reused after an
    input state change is cleared before it can return stale values.

    """
    if layout_cache is None:
        layout_cache = {}

    tensor_references: list[Tensor] = []
    inference_cache_token = object()

    def cache_key(value: object):
        if isinstance(value, Tensor):
            tensor_references.append(value)
            # Inference tensors do not expose a version counter. Force their
            # metadata to be rebuilt on every call so that in-place mutations
            # cannot reuse stale cached indices.
            version = inference_cache_token if torch.is_inference(value) else int(value._version)
            return (
                "tensor",
                id(value),
                version,
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
                value.device,
                value.requires_grad,
            )
        if isinstance(value, (tuple, list, torch.Size)):
            return tuple(cache_key(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, cache_key(item)) for key, item in value.items()))
        return value

    expected_key = (operation, tuple(cache_key(value) for value in values))
    cached_state = layout_cache.get(_LAYOUT_CACHE_STATE)
    current_state = _layout_cache_state(layout_cache) if cached_state is not None else None
    if (layout_cache.get(_LAYOUT_CACHE_KEY) != expected_key or
            cached_state is None or current_state != cached_state):
        layout_cache.clear()
        layout_cache[_LAYOUT_CACHE_KEY] = expected_key
        # Retaining the source tensors prevents Python/CUDA allocator identity
        # reuse from making a different layout appear to have the same key.
        layout_cache[_LAYOUT_CACHE_REFS] = tuple(tensor_references)

    return layout_cache


def finalize_layout_cache(layout_cache: LayoutCache) -> LayoutCache:
    """Records the current cache contents so caller mutations invalidate them.

    Cache builders must call this function after populating their operation-specific
    metadata. The next call to :func:`prepare_layout_cache` compares tensor identity
    and version information as well as non-tensor values against this snapshot.

    """
    layout_cache[_LAYOUT_CACHE_STATE] = _layout_cache_state(layout_cache)
    return layout_cache


def _layout_cache_state(layout_cache: LayoutCache) -> tuple:
    return tuple(
        (key, _layout_cache_value_state(value))
        for key, value in sorted(layout_cache.items())
        if key not in _LAYOUT_CACHE_METADATA
    )


def _layout_cache_value_state(value: object):
    if isinstance(value, Tensor):
        # Inference tensors do not expose version counters. A fresh sentinel
        # deliberately makes caches containing them non-reusable.
        version = object() if torch.is_inference(value) else int(value._version)
        return (
            "tensor",
            id(value),
            version,
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
            value.requires_grad,
        )
    if isinstance(value, dict):
        return tuple(
            (key, _layout_cache_value_state(item))
            for key, item in sorted(value.items())
            if key not in _LAYOUT_CACHE_METADATA
        )
    if isinstance(value, (tuple, list, torch.Size)):
        return tuple(_layout_cache_value_state(item) for item in value)
    return value


def do_shape_blocksparse(x: Tensor) -> tuple[Tensor, Size]:
    """Flattens leading dimensions into the batch dimension used by BLK-SPRS.

    Args:
        x (Tensor): A tensor with at least two dimensions.

    Returns:
        tuple[Tensor, Size]: A contiguous three-dimensional tensor and the original shape for use with
            :func:`undo_shape_blocksparse`.

    """
    if x.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    if x.dim() == 3:
        return x.contiguous(), x.size()

    batch_size = 1
    for dimension_size in x.shape[:-2]:
        batch_size *= dimension_size

    return x.reshape(batch_size, x.size(-2), x.size(-1)).contiguous(), x.size()


def undo_shape_blocksparse(x: Tensor, shape: Size | tuple[int, ...]) -> Tensor:
    """Restores leading dimensions flattened by :func:`do_shape_blocksparse`.

    The final two dimensions are taken from ``x`` so that shape-changing matrix operations can be restored as well.

    Args:
        x (Tensor): A three-dimensional tensor produced by a BLK-SPRS operation.
        shape (Size or tuple): The original shape returned by :func:`do_shape_blocksparse`.

    Returns:
        Tensor: The tensor with its original leading dimensions restored.

    """
    if x.dim() != 3:
        raise ValueError("Tensor must have exactly 3 dimensions")
    if len(shape) < 2:
        raise ValueError("Original shape must have at least 2 dimensions")

    expected_batches = 1
    for dimension_size in shape[:-2]:
        expected_batches *= dimension_size
    if x.size(0) != expected_batches:
        raise ValueError("Tensor batch dimension does not match original shape")

    if x.shape[:-2] == shape[:-2]:
        return x

    return x.reshape((*shape[:-2], *x.shape[-2:]))


def stride(x: Tensor):
    if x.dim() == 1:
        return 1
    elif x.dim() == 2:
        return x.size(1), 1
    elif x.dim() == 3:
        return x.size(1) * x.size(2), x.size(2), 1
    else:
        raise NotImplementedError(f"stride() not implemented for {x.dim()}-dimensional tensors")


def ceil_pow2(x: int) -> int:
    """Returns the smallest power of two greater than or equal to ``x``."""
    if x <= 0:
        raise ValueError("Input must be a positive integer.")
    return 1 << (x - 1).bit_length()


def can_use_int32_indexing(*values: Tensor | int | None) -> bool:
    """Returns whether all indexed tensors fit within signed int32 address space.

    The Triton kernels in this package operate on contiguous tensors and derive
    flat element offsets from block indices and contiguous strides. Under that
    constraint, the largest reachable flat index is bounded by ``numel() - 1``
    for tensors and by the raw element count for explicit integer sizes.

    Args:
        *values: Tensors or integer element counts participating in flat index
            computations. ``None`` values are ignored.

    Returns:
        bool: ``True`` if all participating buffers fit within the signed
            ``int32`` range, ``False`` otherwise.

    """
    for value in values:
        if value is None:
            continue

        if isinstance(value, Tensor):
            count = value.numel()
        else:
            count = int(value)

        if count > INT32_INDEX_MAX:
            return False

    return True


def cast_for_autocast(*values: Tensor):
    """Casts eligible CUDA tensors to the active autocast dtype.

    PyTorch autocast leaves ``float64`` and unsupported floating formats
    unchanged. Preserve that boundary so operation-specific validation can
    reject unsupported dtypes instead of silently reducing their precision.

    """
    if not torch.is_autocast_enabled():
        if len(values) == 1:
            return values[0]
        return values

    autocast_dtype = torch.get_autocast_dtype("cuda")
    autocast_input_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    casted = tuple(
        value.to(autocast_dtype)
        if value.is_cuda and value.dtype in autocast_input_dtypes
        else value
        for value in values
    )

    if len(casted) == 1:
        return casted[0]
    return casted

import os

import triton

SUPPORTED_AUTOTUNE_MODES = ("DEFAULT", "TEST")
blksprs_autotune_mode = os.getenv("BLKSPRS_AUTOTUNE", "DEFAULT").upper()

if blksprs_autotune_mode not in SUPPORTED_AUTOTUNE_MODES:
    raise NotImplementedError(f"Unknown autotune mode: {blksprs_autotune_mode}")

# Named autotune profiles allow each operation family to expose its own
# configuration space while keeping a small, test-friendly mode.
_AUTOTUNE_PARAMETER_PROFILES: dict[str, dict[str, list[tuple[int, ...]]]] = {
    "generic": {
        "DEFAULT": [
            (16, 3, 8),
            (16, 4, 4),
            (16, 5, 2),

            (32, 3, 8),
            (32, 4, 4),
            (32, 5, 2),

            (64, 3, 8),
            (64, 4, 4),
            (64, 4, 8),

            (128, 3, 8),
            (128, 4, 4),
            (128, 4, 8),
        ],
        "TEST": [
            (16, 3, 8),
            (32, 3, 8),
            (64, 3, 8),
        ],
    },
    "matmul": {
        "DEFAULT": [
            (16, 3, 4),
            (16, 4, 8),
            (16, 5, 8),

            (32, 3, 4),
            (32, 4, 4),
            (32, 5, 8),

            (64, 3, 4),
            (64, 4, 8),
            (64, 5, 8),

            (128, 3, 4),
            (128, 4, 8),
        ],
        "TEST": [
            (16, 3, 4),
            (32, 3, 4),
            (64, 3, 4),
        ],
    },
    "flow": {
        "DEFAULT": [
            (16, 2, 2),
            (16, 3, 4),

            (32, 2, 2),
            (32, 3, 4),

            (64, 2, 4),
            (64, 3, 4),

            (128, 2, 4),
            (128, 3, 8),
        ],
        "TEST": [
            (16, 2, 2),
            (32, 2, 2),
            (64, 2, 4),
        ],
    },
    # Attention profiles list query and key tile sizes instead of a single block size. Tiles span whole sparsity
    # blocks, see ``prune_autotune_configs_spanning``. The forward and key/value-gradient profiles start with program
    # tiles of single sparsity blocks, which suit sparse layouts whose block rows attend to few, scattered blocks. The
    # last entry of every profile fits into the shared memory of current GPUs for every supported dtype, head
    # dimension, and optional input.
    "flash_attention_forward": {
        "DEFAULT": [
            (32, 64, 2, 4),

            (64, 32, 3, 4),
            (64, 64, 2, 4),
            (64, 64, 3, 4),

            (128, 32, 3, 4),
            (128, 64, 2, 4),
            (128, 64, 3, 4),
            (128, 64, 3, 8),

            (64, 64, 1, 4),
        ],
        "TEST": [
            (64, 64, 2, 4),
            (64, 64, 1, 4),
        ],
    },
    "flash_attention_backward_dkdv": {
        "DEFAULT": [
            (32, 32, 2, 4),
            (64, 32, 2, 4),

            (32, 64, 3, 4),
            (32, 128, 2, 4),
            (32, 128, 3, 4),

            (64, 64, 3, 4),
            (64, 128, 2, 8),

            (64, 64, 1, 4),
        ],
        "TEST": [
            (64, 64, 2, 4),
            (64, 64, 1, 4),
        ],
    },
    "flash_attention_backward_dq": {
        "DEFAULT": [
            (32, 32, 3, 4),
            (32, 64, 2, 4),

            (64, 32, 3, 4),
            (64, 64, 3, 4),

            (128, 32, 3, 4),
            (128, 64, 2, 4),
            (128, 64, 2, 8),

            (64, 64, 1, 4),
        ],
        "TEST": [
            (64, 64, 2, 4),
            (64, 64, 1, 4),
        ],
    },
    "conversion": {
        "DEFAULT": [
            (16, 2, 2),
            (16, 3, 4),

            (32, 2, 2),
            (32, 3, 4),

            (64, 2, 4),
            (64, 3, 4),

            (128, 2, 4),
        ],
        "TEST": [
            (16, 2, 2),
            (32, 2, 2),
            (64, 2, 4),
        ],
    },
    "distribution": {
        "DEFAULT": [
            (16, 2, 2),
            (16, 3, 4),

            (32, 2, 4),
            (32, 3, 4),

            (64, 2, 4),
            (64, 3, 8),

            (128, 2, 8),
        ],
        "TEST": [
            (16, 2, 2),
            (32, 2, 4),
            (64, 2, 4),
        ],
    },
    "softmax": {
        "DEFAULT": [
            (16, 3, 4),
            (16, 4, 4),

            (32, 3, 4),
            (32, 4, 8),

            (64, 3, 8),
            (64, 4, 8),

            (128, 3, 8),
        ],
        "TEST": [
            (16, 3, 4),
            (32, 3, 4),
            (64, 3, 8),
        ],
    },
    "row_wise": {
        "DEFAULT": [
            (16, 2, 2),
            (16, 3, 4),

            (32, 2, 2),
            (32, 3, 4),

            (64, 2, 4),
            (64, 3, 4),

            (128, 2, 4),
        ],
        "TEST": [
            (16, 2, 2),
            (32, 2, 2),
            (64, 2, 4),
        ],
    },
    "broadcast": {
        "DEFAULT": [
            (16, 2, 2),
            (16, 3, 4),

            (32, 2, 2),
            (32, 3, 4),

            (64, 2, 4),
            (64, 3, 4),

            (128, 2, 4),
        ],
        "TEST": [
            (16, 2, 2),
            (32, 2, 2),
            (64, 2, 4),
        ],
    },
}


def get_autotune_parameters(profile: str = "generic", mode: str | None = None):
    mode = blksprs_autotune_mode if mode is None else mode.upper()
    profile = profile.lower()

    if mode not in SUPPORTED_AUTOTUNE_MODES:
        raise NotImplementedError(f"Unknown autotune mode: {mode}")
    if profile not in _AUTOTUNE_PARAMETER_PROFILES:
        raise NotImplementedError(f"Unknown autotune profile: {profile}")

    return _AUTOTUNE_PARAMETER_PROFILES[profile][mode]


def prune_autotune_configs(autotune_configs, kernel_args, **kwargs):
    sparsity_block_size = kernel_args["sparsity_block_size"]
    return _prune_autotune_configs_by_block_size(
        autotune_configs,
        sparsity_block_size,
        predicate=lambda triton_block_size, sbs: triton_block_size <= sbs,
    )


def prune_autotune_configs_spanning(autotune_configs, kernel_args, **kwargs):
    """Keeps configurations whose tiles span whole sparsity blocks in every tiled dimension."""
    sparsity_block_size = kernel_args["sparsity_block_size"]
    pruned_configs = [
        config for config in autotune_configs
        if all(block_size % sparsity_block_size == 0
               for name, block_size in config.kwargs.items() if name.startswith("TRITON_BLOCK_SIZE"))
    ]

    assert len(pruned_configs) > 0, (
        f"No valid autotune configs found for sparsity block size {sparsity_block_size}"
    )

    return pruned_configs


def prune_autotune_configs_conversion(autotune_configs, kernel_args, **kwargs):
    sparsity_block_size_from = kernel_args["sparsity_block_size_from"]
    sparsity_block_size_to = kernel_args["sparsity_block_size_to"]
    sparsity_block_size = min(sparsity_block_size_from, sparsity_block_size_to)

    return _prune_autotune_configs_by_block_size(
        autotune_configs,
        sparsity_block_size,
        predicate=lambda triton_block_size, sbs: triton_block_size <= sbs,
    )


def _prune_autotune_configs_by_block_size(autotune_configs, sparsity_block_size, predicate):
    pruned_configs = []

    for config in autotune_configs:
        if predicate(config.kwargs["TRITON_BLOCK_SIZE"], sparsity_block_size):
            pruned_configs.append(config)

    assert len(pruned_configs) > 0, (
        f"No valid autotune configs found for sparsity block size {sparsity_block_size}"
    )

    return pruned_configs


def get_autotune_configs(profile: str = "generic", mode: str | None = None,
                         block_size_names: tuple[str, ...] = ("TRITON_BLOCK_SIZE",)):
    """Builds the Triton configurations of a profile.

    Every profile entry lists one block size per name in ``block_size_names``, followed by the number of pipeline
    stages and warps.

    """
    autotune_parameters = get_autotune_parameters(profile=profile, mode=mode)

    autotune_configs = []

    for *block_sizes, num_stages, num_warps in autotune_parameters:
        if len(block_sizes) != len(block_size_names):
            raise ValueError(f"Autotune profile {profile} does not provide the block sizes {block_size_names}")
        autotune_configs.append(
            triton.Config(dict(zip(block_size_names, block_sizes)), num_stages=num_stages, num_warps=num_warps))

    return autotune_configs

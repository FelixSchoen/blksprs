import os

import triton

_SUPPORTED_AUTOTUNE_MODES = ("DEFAULT", "TEST")
blksprs_autotune_mode = os.getenv("BLKSPRS_AUTOTUNE", "DEFAULT").upper()

if blksprs_autotune_mode not in _SUPPORTED_AUTOTUNE_MODES:
    raise NotImplementedError(f"Unknown autotune mode: {blksprs_autotune_mode}")

# Named autotune profiles allow each operation family to expose its own
# configuration space while keeping a small, test-friendly mode.
_AUTOTUNE_PARAMETER_PROFILES = {
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
    "flash_attention": {
        "DEFAULT": [
            (16, 3, 4),
            (16, 4, 4),

            (32, 3, 4),
            (32, 4, 8),

            (64, 3, 8),
            (64, 4, 8),

            (128, 3, 8),
            (128, 4, 8),
        ],
        "TEST": [
            (16, 3, 4),
            (32, 3, 4),
            (64, 3, 8),
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

    if mode not in _SUPPORTED_AUTOTUNE_MODES:
        raise NotImplementedError(f"Unknown autotune mode: {mode}")
    if profile not in _AUTOTUNE_PARAMETER_PROFILES:
        raise NotImplementedError(f"Unknown autotune profile: {profile}")

    return _AUTOTUNE_PARAMETER_PROFILES[profile][mode]


# Backward compatibility for callers that still read this module-level value.
autotune_parameters = get_autotune_parameters()


def prune_autotune_configs(autotune_configs, kernel_args, **kwargs):
    sparsity_block_size = kernel_args["sparsity_block_size"]
    return _prune_autotune_configs_by_block_size(
        autotune_configs,
        sparsity_block_size,
        predicate=lambda triton_block_size, sbs: triton_block_size <= sbs,
    )


def prune_autotune_configs_exact(autotune_configs, kernel_args, **kwargs):
    sparsity_block_size = kernel_args["sparsity_block_size"]
    return _prune_autotune_configs_by_block_size(
        autotune_configs,
        sparsity_block_size,
        predicate=lambda triton_block_size, sbs: triton_block_size == sbs,
    )


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


def get_autotune_configs(profile: str = "generic", mode: str | None = None):
    autotune_parameters = get_autotune_parameters(profile=profile, mode=mode)

    autotune_configs = []

    for block_size, num_stages, num_warps in autotune_parameters:
        autotune_configs.append(
            triton.Config({"TRITON_BLOCK_SIZE": block_size}, num_stages=num_stages, num_warps=num_warps))

    return autotune_configs

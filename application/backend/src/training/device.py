# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning accelerator/strategy resolution for training runs.

One preference order for every runner. Callers that used to reimplement this
(the studio backend and the standalone trainer service each had their own
version, which disagreed) resolve through here instead so a job trains on the
same device whether it runs in-process or on a remote trainer.

Distinct from :mod:`physicalai.devices`, which resolves ``torch.device``
objects for inference and tensor movement: this module answers the narrower
question of what to pass to :class:`physicalai.train.Trainer`.
"""

from __future__ import annotations

from typing import Any

ACCELERATOR_PREFERENCE: tuple[str, ...] = ("xpu", "cuda", "mps", "cpu")
"""Auto-detection order. XPU first: this is the primary supported accelerator."""


def resolve_accelerator(device_type: str | None = None) -> str:
    """Return the Lightning accelerator string for a training run.

    Args:
        device_type: Explicit accelerator (e.g. ``"xpu"``, ``"cuda"``, ``"cpu"``).
            When None the best available accelerator is auto-detected following
            :data:`ACCELERATOR_PREFERENCE`.

    Returns:
        The accelerator name to pass to the trainer.

    Example:
        >>> resolve_accelerator("cpu")
        'cpu'
    """
    if device_type is not None:
        return device_type

    import torch

    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def resolve_strategy(device_type: str | None = None, *, cpu_offload: bool = False) -> str | Any:
    """Return the Lightning strategy for a training run.

    XPU needs its own single-device strategy. ``cpu_offload`` requests DeepSpeed
    ZeRO Stage 3 with parameter and optimizer offload, so models that don't fit
    in GPU VRAM can still train by spilling frozen weights and optimizer state
    to CPU RAM; it only applies on CUDA (DeepSpeed's offload path isn't wired up
    for XPU/MPS/CPU here), and is silently ignored elsewhere. Everything else is
    covered by ``"auto"``.

    Args:
        device_type: Explicit accelerator, or None to auto-detect.
        cpu_offload: Whether to offload parameters/optimizer state to CPU RAM.

    Returns:
        The strategy to pass to the trainer: a string name, or a configured
        ``DeepSpeedStrategy`` instance when ``cpu_offload`` is requested.

    Example:
        >>> resolve_strategy("cpu")
        'auto'
    """
    accelerator = resolve_accelerator(device_type)
    if accelerator == "xpu":
        return "xpu_single"
    if accelerator == "cuda" and cpu_offload:
        return _build_cpu_offload_strategy()
    return "auto"


def _build_cpu_offload_strategy() -> Any:
    """Build a DeepSpeed ZeRO Stage 3 strategy with parameter/optimizer offload.

    Policies here bring their own ``torch.optim.AdamW`` (via
    ``configure_optimizers``), not DeepSpeed's fused CPU-Adam. DeepSpeed refuses
    that combination by default (it's typically slow), raising instead of
    training. ``zero_force_ds_cpu_optimizer`` isn't a constructor kwarg on
    ``DeepSpeedStrategy`` — the only way to set it is by mutating the generated
    config dict before Lightning invokes the strategy, which is exactly what's
    done here: build the strategy normally so every offload/stage kwarg gets
    its default config shape, then flip the one flag that lets a plain AdamW
    run under ZeRO-Offload anyway. Slower than DeepSpeedCPUAdam, but correct
    and dependency-free.

    Returns:
        A DeepSpeedStrategy configured for stage-3 parameter and optimizer
        offload to pinned CPU memory, accepting a non-DeepSpeed optimizer.
    """
    from lightning.pytorch.strategies import DeepSpeedStrategy

    strategy = DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        pin_memory=True,
    )
    strategy.config["zero_force_ds_cpu_optimizer"] = False
    return strategy


def resolve_devices(device_index: int | None = None) -> list[int] | int:
    """Return the Lightning ``devices`` value for a training run.

    Args:
        device_index: Zero-based index of the accelerator to train on, or None
            to let Lightning pick one device.

    Returns:
        ``[device_index]`` when an index is given, otherwise ``1``.

    Example:
        >>> resolve_devices(2)
        [2]
        >>> resolve_devices()
        1
    """
    return [device_index] if device_index is not None else 1

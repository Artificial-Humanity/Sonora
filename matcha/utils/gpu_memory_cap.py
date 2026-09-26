"""A ceiling on the caching allocator, so fragmentation cannot take the host's memory.

Why this exists (2026-09-26): vat7_dit_spike's MemoryTrace showed the trainer never
ALLOCATES more than 45.5 GiB, but the caching allocator's RESERVED memory ratchets to
80-88 GiB over about two hours, as variable-length batches leave cached blocks no later
batch can reuse. On this APU reserved memory is GTT, taken from host RAM, and no container
limit covers it, so the ratchet ended five segments in a host OOM. With ttm.pages_limit at
90 GiB the driver refused at the cap, the allocator released its cache and retried, and the
run survived -- but with the host down to 16 GiB available.

`torch.cuda.set_per_process_memory_fraction` makes that release-and-retry happen at a
ceiling this process chooses, well inside the box's. It limits future allocations only, so
applying it at `setup` is in time: nothing the model holds then is near the ceiling.
"""
from __future__ import annotations

import logging

import torch
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)

_GIB = 1024**3


class GpuMemoryCap(Callback):
    """Cap this process's caching allocator at `cap_gib` of device 0.

    Args:
        cap_gib: the ceiling in GiB. None -> no cap. A value at or above the device's
            memory is refused with a warning rather than clamped: it would be a no-op, and a
            no-op that reads as a cap is the failure this exists to prevent.
    """

    def __init__(self, cap_gib: float | None = None) -> None:
        super().__init__()
        self.cap_gib = cap_gib
        self.fraction: float | None = None

    def setup(self, trainer, pl_module, stage: str) -> None:  # noqa: ANN001
        if self.cap_gib is None or not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = self.cap_gib * _GIB / total
        if fraction >= 1.0:
            log.warning("GpuMemoryCap: cap %.1f GiB is not below the device's %.1f GiB; "
                        "no cap applied", self.cap_gib, total / _GIB)
            return
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        self.fraction = fraction
        log.info("GpuMemoryCap: caching allocator capped at %.1f GiB of %.1f GiB (fraction %.4f)",
                 self.cap_gib, total / _GIB, fraction)

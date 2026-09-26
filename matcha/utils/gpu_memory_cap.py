"""A ceiling on the caching allocator, so fragmentation cannot take the host's memory.

Why this exists (2026-09-26): vat7_dit_spike's MemoryTrace showed the trainer never
ALLOCATES more than 45.5 GiB, but the caching allocator's RESERVED memory reaches 80-88 GiB,
as variable-length batches leave cached blocks no later batch can reuse. Reserved is not a
steady ratchet -- it collapsed to 1-2 GiB about 130 times in the first ~2,000 steps, a flush
from outside this repo that counts no retry -- but those collapses thin out and the
high-water mark climbs, reaching the host's limit about two hours in. On this APU reserved
memory is GTT, taken from host RAM, and no container limit covers it, so that ended five
segments in a host OOM. With ttm.pages_limit at 90 GiB the driver refused at the cap, the
allocator released its cache and retried, and the run survived -- but with the host down
to 16 GiB available. After each of those retries, a near-worst batch (2,048 frames) sat at
42.8-47.2 GiB reserved: that is what a step actually needs, and 64 GiB leaves ~17 GiB over it.

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
        if self.cap_gib is None:
            return
        if not torch.cuda.is_available():
            log.warning("GpuMemoryCap: cap %.1f GiB configured but no GPU is visible; "
                        "no cap applied", self.cap_gib)
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

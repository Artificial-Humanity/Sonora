"""Per-step memory record, written so that a run the OOM killer takes still leaves it.

Why this exists (2026-09-26): five segments of vat7_dit_spike died of HOST memory
exhaustion by GTT -- the unified-memory pool the GPU allocates from on this APU, which no
container limit covers -- at the same process age (2 h 08 - 2 h 15) but at different steps.
The only memory number the run kept was ThroughputProbe's per-epoch peak, and no segment
finished an epoch. So this writes one JSONL line per step and flushes it at once.

Each line carries what tells the candidate causes apart:
  * `age_s` against `step`: the failures tracked process age, not step count;
  * `allocated_gib` vs `reserved_gib` and `inactive_split_gib`: caching-allocator
    fragmentation grows reserved while allocated stays flat;
  * `alloc_retries`, `num_device_alloc`/`num_device_free`: the allocator freeing its cache
    and asking the driver again, which is how a fragmented pool turns into a burst;
  * `host_gtt_gib`, `host_avail_gib`: what the kernel saw, which torch cannot report;
  * `frames`, `batch`, `step_s`: the batch in hand and whether the step stalled.

No torch.cuda.synchronize(): the allocator counters are host-side bookkeeping and cost
microseconds, and a sync every step would change the thing being measured.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback

_GIB = 1024**3


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _mem_available_kb(path: str) -> int | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


class MemoryTrace(Callback):
    """One flushed JSONL line per training step, next to the run.

    Args:
        out_name: the JSONL file, in the trainer's log dir.
        gtt_path: sysfs file holding the GPU's GTT bytes in use. Absent -> null.
        meminfo_path: read for MemAvailable. Absent -> null.
    """

    def __init__(
        self,
        out_name: str = "memory_trace.jsonl",
        gtt_path: str = "/sys/class/drm/card1/device/mem_info_gtt_used",
        meminfo_path: str = "/proc/meminfo",
    ) -> None:
        super().__init__()
        self.out_name = out_name
        self.gtt_path = gtt_path
        self.meminfo_path = meminfo_path
        # Instantiated with the other callbacks, before the datamodule and model are built,
        # so this is within seconds of process start.
        self._t0 = time.monotonic()
        self._fh = None
        self._t_step_start: float | None = None

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        out_dir = Path(trainer.default_root_dir or ".")
        try:
            if trainer.log_dir:
                out_dir = Path(trainer.log_dir)
        except Exception:  # pragma: no cover - logger-dependent
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (out_dir / self.out_name).open("a", buffering=1)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001
        self._t_step_start = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        if self._fh is None:
            return
        now = time.monotonic()
        y = batch.get("y") if isinstance(batch, dict) else None
        rec: dict[str, Any] = {
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "age_s": round(now - self._t0, 1),
            "step": int(trainer.global_step),
            "step_s": round(now - self._t_step_start, 3) if self._t_step_start else None,
            "batch": int(y.shape[0]) if y is not None else None,
            "frames": int(y.shape[-1]) if y is not None else None,
            "allocated_gib": None,
            "reserved_gib": None,
        }
        if torch.cuda.is_available():
            s = torch.cuda.memory_stats()
            rec.update({
                "allocated_gib": round(s.get("allocated_bytes.all.current", 0) / _GIB, 3),
                "reserved_gib": round(s.get("reserved_bytes.all.current", 0) / _GIB, 3),
                "peak_allocated_gib": round(s.get("allocated_bytes.all.peak", 0) / _GIB, 3),
                "inactive_split_gib": round(s.get("inactive_split_bytes.all.current", 0) / _GIB, 3),
                "alloc_retries": s.get("num_alloc_retries", 0),
                "num_ooms": s.get("num_ooms", 0),
                "num_device_alloc": s.get("num_device_alloc", 0),
                "num_device_free": s.get("num_device_free", 0),
                "segments": s.get("segment.all.current", 0),
            })
        gtt = _read_int(self.gtt_path)
        avail = _mem_available_kb(self.meminfo_path)
        rec["host_gtt_gib"] = round(gtt / _GIB, 3) if gtt is not None else None
        rec["host_avail_gib"] = round(avail / 1024**2, 3) if avail is not None else None
        self._fh.write(json.dumps(rec) + "\n")

    def teardown(self, trainer, pl_module, stage: str) -> None:  # noqa: ANN001
        if self._fh is not None:
            self._fh.close()
            self._fh = None

"""Per-step memory record, written so that a run the OOM killer takes still leaves it.

Why this exists (2026-09-26): five segments of vat7_dit_spike died of HOST memory
exhaustion by GTT -- the unified-memory pool the GPU allocates from on this APU, which no
container limit covers -- at the same process age (2 h 08 - 2 h 15) but at different steps.
The only memory number the run kept was ThroughputProbe's per-epoch peak, and no segment
finished an epoch. So this writes JSONL, line-buffered: every line is in the kernel's page
cache when it is written, and a SIGKILL cannot lose it.

Two lines per step. `"phase": "start"` carries the batch about to run, because the kill lands
inside forward/backward and an end-only record never shows the batch that caused it.
`"phase": "end"` carries what tells the candidate causes apart:
  * `age_s` against `step`: the failures tracked process age, not step count. The gap between
    one line's `age_s` and the next, less `step_s`, is time spent waiting on the loader;
  * `allocated_gib` vs `reserved_gib` and `inactive_split_gib`: caching-allocator
    fragmentation grows reserved while allocated stays flat. The `peak_*` fields are LIFETIME
    high-water marks (resetting them would corrupt ThroughputProbe's per-epoch peak), so a
    rise marks the step that set a new record, not that step's own peak;
  * `alloc_retries`, `num_device_alloc`/`num_device_free`: the allocator freeing its cache
    and asking the driver again, which is how a fragmented pool turns into a burst;
  * `pinned_gib`: the pinned host allocator (pin_memory), which on an APU is GTT too and
    sits outside the device allocator's counters;
  * `proc_gtt_gib` (this process, from DRM fdinfo) beside `host_gtt_gib` (the whole box):
    the box total alone cannot say whose memory grew;
  * `host_avail_gib`: what the OOM killer acts on.

No torch.cuda.synchronize(): the allocator counters are host-side bookkeeping, and a sync
every step would change the thing being measured.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)

_GIB = 1024**3


def _read_int(path: str | None) -> int | None:
    if not path:
        return None
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


def _fdinfo_fields(path: str) -> dict[str, str]:
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def _kib(value: str) -> int | None:
    # fdinfo reports e.g. "2097152 KiB"; older kernels may omit the unit (bytes).
    try:
        num, *unit = value.split()
        n = int(num)
    except (ValueError, AttributeError):
        return None
    scale = {"": 1 / 1024, "KiB": 1, "MiB": 1024, "GiB": 1024**2}.get(unit[0] if unit else "")
    return None if scale is None else int(n * scale)


class MemoryTrace(Callback):
    """Two flushed JSONL lines per training step, next to the run.

    Args:
        out_name: the JSONL file, in the trainer's log dir.
        gtt_path: sysfs file holding the GPU's GTT bytes in use. None -> found by `gtt_glob`.
        gtt_glob: where to look for it. The card index is not stable across kernels.
        meminfo_path: read for MemAvailable. Absent -> null.
        fdinfo_dir: this process's fdinfo, for its own GTT. No DRM client -> null.
    """

    def __init__(
        self,
        out_name: str = "memory_trace.jsonl",
        gtt_path: str | None = None,
        gtt_glob: str = "/sys/class/drm/card*/device/mem_info_gtt_used",
        meminfo_path: str = "/proc/meminfo",
        fdinfo_dir: str = "/proc/self/fdinfo",
    ) -> None:
        super().__init__()
        self.out_name = out_name
        self.gtt_path = gtt_path
        self.gtt_glob = gtt_glob
        self.meminfo_path = meminfo_path
        self.fdinfo_dir = fdinfo_dir
        # matcha/train.py builds the datamodule and model first, then the callbacks, so this
        # is about a second after Hydra's first log line -- the same baseline the failure
        # ages were read against.
        self._t0 = time.monotonic()
        self._fh = None
        self._drm_fds: list[str] = []
        self._t_step_start: float | None = None

    # ---- lifecycle ---------------------------------------------------------

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        out_dir = Path(trainer.default_root_dir or ".")
        try:
            if trainer.log_dir:
                out_dir = Path(trainer.log_dir)
        except Exception:  # pragma: no cover - logger-dependent
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (out_dir / self.out_name).open("a", buffering=1)

        if self.gtt_path is None:
            for cand in sorted(glob.glob(self.gtt_glob)):
                if _read_int(cand) is not None:
                    self.gtt_path = cand
                    break
        if _read_int(self.gtt_path) is None:
            log.warning("MemoryTrace: no readable GTT file (%s); host_gtt_gib will be null",
                        self.gtt_path or self.gtt_glob)

        # The GPU is initialised by now, so its DRM clients are open. One per client id:
        # several fds can share a client, and each reports the same totals.
        seen = set()
        try:
            fds = sorted(os.listdir(self.fdinfo_dir), key=lambda s: int(s) if s.isdigit() else -1)
        except OSError:
            fds = []
        for fd in fds:
            f = _fdinfo_fields(os.path.join(self.fdinfo_dir, fd))
            cid = f.get("drm-client-id")
            if cid is not None and cid not in seen:
                seen.add(cid)
                self._drm_fds.append(os.path.join(self.fdinfo_dir, fd))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001
        self._t_step_start = time.monotonic()
        y = batch.get("y") if isinstance(batch, dict) else None
        self._write({
            "phase": "start",
            "age_s": round(self._t_step_start - self._t0, 1),
            "step": int(trainer.global_step),
            "batch": int(y.shape[0]) if y is not None else None,
            "frames": int(y.shape[-1]) if y is not None else None,
        })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        if self._fh is None:
            return
        try:
            self._write(self._end_record(trainer, batch))
        except Exception as e:  # noqa: BLE001 - a trace must never end the run
            log.warning("MemoryTrace: disabled after %s: %s", type(e).__name__, e)
            self._fh = None

    def teardown(self, trainer, pl_module, stage: str) -> None:  # noqa: ANN001
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---- records -----------------------------------------------------------

    def _write(self, rec: dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001 - ENOSPC/EIO on /data must not end a 53 h run
            log.warning("MemoryTrace: disabled after %s: %s", type(e).__name__, e)
            self._fh = None

    def _end_record(self, trainer, batch) -> dict[str, Any]:  # noqa: ANN001
        now = time.monotonic()
        y = batch.get("y") if isinstance(batch, dict) else None
        rec: dict[str, Any] = {
            "phase": "end",
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
                "peak_reserved_gib": round(s.get("reserved_bytes.all.peak", 0) / _GIB, 3),
                "inactive_split_gib": round(s.get("inactive_split_bytes.all.current", 0) / _GIB, 3),
                "alloc_retries": s.get("num_alloc_retries", 0),
                "num_ooms": s.get("num_ooms", 0),
                "num_device_alloc": s.get("num_device_alloc", 0),
                "num_device_free": s.get("num_device_free", 0),
                "segments": s.get("segment.all.current", 0),
            })
            try:
                hs = torch.cuda.host_memory_stats()
                rec["pinned_gib"] = round(hs.get("reserved_bytes.current", 0) / _GIB, 3)
            except Exception:  # noqa: BLE001 - absent on older torch
                rec["pinned_gib"] = None
        gtt = _read_int(self.gtt_path)
        avail = _mem_available_kb(self.meminfo_path)
        rec["host_gtt_gib"] = round(gtt / _GIB, 3) if gtt is not None else None
        rec["host_avail_gib"] = round(avail / 1024**2, 3) if avail is not None else None
        rec["proc_gtt_gib"] = self._proc_gtt_gib()
        return rec

    def _proc_gtt_gib(self) -> float | None:
        total, found = 0, False
        for path in self._drm_fds:
            kib = _kib(_fdinfo_fields(path).get("drm-memory-gtt", ""))
            if kib is not None:
                total, found = total + kib, True
        return round(total / 1024**2, 3) if found else None

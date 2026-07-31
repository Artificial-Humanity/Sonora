"""Throughput instrumentation for the local-vs-rented-GPU decision.

The question this exists to answer: would renting a faster GPU actually cut
wall-clock on a Sonora run, or is the run limited by how fast the dataloader
can turn wavs into mels?

It deliberately does NOT try to attribute time *within* a training step. Async
GPU execution makes that unreliable unless you synchronise every iteration,
which perturbs the very thing you are measuring. Instead this records the one
number that needs no attribution -- observed steps/sec -- and leaves the
loader's ceiling to scripts/bench_loader.py. Comparing the two IS the
diagnosis:

    loader ceiling >> observed rate   ->  GPU-bound. A faster GPU helps.
    loader ceiling ~= observed rate   ->  loader-bound. A faster GPU does
                                          nothing, and a 5-vCPU pod is WORSE
                                          than this 16-core box.

Stall time (the gap between one step ending and the next beginning) is
recorded too. With prefetching a healthy GPU-bound run shows stall ~= 0; a
rising stall fraction is the loader starving the GPU. It corroborates the
ceiling comparison rather than replacing it.

Overhead: a few microseconds per step, plus one torch.cuda.synchronize() every
`sync_every` steps.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


def _pct(values: list[float], q: float) -> float:
    """Percentile without a numpy dependency. `values` need not be sorted."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "sum": sum(values),
    }


class ThroughputProbe(Callback):
    """Per-epoch throughput and stall statistics, written next to the run.

    Args:
        out_name: JSONL file, one line per epoch, written incrementally so a
            killed run still leaves usable data.
        summary_name: JSON written at train end (and at teardown, so a crash
            or Ctrl-C still produces it).
        sync_every: synchronise and time a full step this often. Keep it
            coarse; the point is a representative sample, not every step.
        warmup_steps: steps to ignore at the start of each epoch. Worker
            spin-up under the "spawn" context (an AMD fork-wedge workaround,
            see text_mel_datamodule.py) makes the first steps unrepresentative.
    """

    def __init__(
        self,
        out_name: str = "throughput_probe.jsonl",
        summary_name: str = "throughput_summary.json",
        sync_every: int = 50,
        warmup_steps: int = 25,
    ) -> None:
        super().__init__()
        self.out_name = out_name
        self.summary_name = summary_name
        self.sync_every = max(1, int(sync_every))
        self.warmup_steps = max(0, int(warmup_steps))

        self._out_path: Path | None = None
        self._summary_path: Path | None = None
        self._context: dict[str, Any] = {}
        self._epochs: list[dict[str, Any]] = []

        self._t_prev_end: float | None = None
        self._t_step_start: float | None = None
        self._epoch_t0: float = 0.0
        self._stalls: list[float] = []
        self._computes: list[float] = []
        self._steps: int = 0

    # ---- lifecycle ---------------------------------------------------------

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        out_dir = Path(trainer.default_root_dir or ".")
        try:
            if trainer.log_dir:
                out_dir = Path(trainer.log_dir)
        except Exception:  # pragma: no cover - logger-dependent
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_path = out_dir / self.out_name
        self._summary_path = out_dir / self.summary_name

        dm = getattr(trainer, "datamodule", None)
        hp = getattr(dm, "hparams", {}) if dm is not None else {}
        device = "cpu"
        if torch.cuda.is_available():
            try:
                device = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover
                device = "cuda (name unavailable)"

        self._context = {
            "device": device,
            # torch.version.hip is set on ROCm builds and None on CUDA ones --
            # this is what tells a Strix Halo run apart from a Runpod one.
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "batch_size": hp.get("batch_size"),
            "num_workers": hp.get("num_workers"),
            "pin_memory": hp.get("pin_memory"),
            "precision": str(getattr(trainer, "precision", "?")),
            "world_size": getattr(trainer, "world_size", 1),
            "sync_every": self.sync_every,
            "warmup_steps": self.warmup_steps,
        }

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        self._epoch_t0 = time.perf_counter()
        self._stalls, self._computes, self._steps = [], [], 0
        self._t_prev_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001
        now = time.perf_counter()
        if self._t_prev_end is not None and batch_idx >= self.warmup_steps:
            # Gap between the previous step finishing and this one starting.
            # With prefetching this is the un-overlapped share of data loading.
            self._stalls.append(now - self._t_prev_end)
        self._t_step_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        sampled = (batch_idx % self.sync_every == 0) and batch_idx >= self.warmup_steps
        if sampled and torch.cuda.is_available():
            # Only on sampled steps: make the async queue drain so the number
            # below is real GPU time rather than host launch time.
            torch.cuda.synchronize()
        now = time.perf_counter()
        if sampled and self._t_step_start is not None:
            self._computes.append(now - self._t_step_start)
        if batch_idx >= self.warmup_steps:
            self._steps += 1
        self._t_prev_end = now

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        wall = time.perf_counter() - self._epoch_t0
        bs = self._context.get("batch_size") or 0
        rec: dict[str, Any] = {
            "epoch": int(trainer.current_epoch),
            "wall_s": wall,
            "timed_steps": self._steps,
            "steps_per_s": (self._steps / wall) if wall > 0 else None,
            "samples_per_s": (self._steps * bs / wall) if (wall > 0 and bs) else None,
            "stall_s": _stats(self._stalls),
            "step_s_synced": _stats(self._computes),
        }
        # The headline: what share of wall-clock was spent waiting for data.
        # Near 0 => GPU-bound. Climbing => the loader is starving the GPU.
        rec["stall_fraction"] = (sum(self._stalls) / wall) if wall > 0 else None
        if torch.cuda.is_available():
            try:
                rec["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            except Exception:  # pragma: no cover
                pass

        self._epochs.append(rec)
        if self._out_path is not None:
            with self._out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def on_train_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        self._write_summary()

    def teardown(self, trainer, pl_module, stage: str) -> None:  # noqa: ANN001
        # Also on crash/interrupt -- a run that died at epoch 40 still carries
        # the answer we are after.
        if stage == "fit":
            self._write_summary()

    # ---- summary -----------------------------------------------------------

    def _write_summary(self) -> None:
        if self._summary_path is None or not self._epochs:
            return
        rates = [e["steps_per_s"] for e in self._epochs if e.get("steps_per_s")]
        stalls = [e["stall_fraction"] for e in self._epochs if e.get("stall_fraction") is not None]
        walls = [e["wall_s"] for e in self._epochs]
        synced = [e["step_s_synced"].get("p50") for e in self._epochs if e["step_s_synced"].get("n")]

        summary = {
            "context": self._context,
            "epochs_recorded": len(self._epochs),
            "steps_per_s_median": _pct(rates, 0.5) if rates else None,
            "epoch_wall_s_median": _pct(walls, 0.5) if walls else None,
            "stall_fraction_median": _pct(stalls, 0.5) if stalls else None,
            "step_s_synced_median": _pct([s for s in synced if s], 0.5) if synced else None,
            "note": (
                "Compare steps_per_s_median against the loader ceiling from "
                "scripts/bench_loader.py. Ceiling much higher => GPU-bound => a "
                "faster rented GPU helps. Ceiling close => loader-bound => it "
                "will not, and fewer pod vCPUs makes it worse. Feed both files "
                "to scripts/project_pod_speedup.py."
            ),
        }
        self._summary_path.write_text(json.dumps(summary, indent=2) + "\n")

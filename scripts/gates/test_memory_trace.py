"""Gates for MemoryTrace (matcha/utils/memory_trace.py), the per-step memory record.

IN-CONTAINER (needs torch):  scripts/stages/run_in_rocm.sh scripts/gates/test_memory_trace.py

WHY IT EXISTS (2026-09-26). Five segments of vat7_dit_spike died of HOST memory exhaustion by
GTT, at the same process age (2 h 08 - 2 h 15) but at different steps, and none left a memory
record: ThroughputProbe writes once per epoch and no segment finished one. A trace that only
exists after a clean exit is no use against the OOM killer, so the gates below hold it to:

  1. one line per step, readable BEFORE on_train_end (a killed run never gets one);
  2. batch shape, wall time and process age on every line;
  3. host GTT and MemAvailable read from their files;
  4. absent host files recorded as null, never an exception that kills the run;
  5. allocator state (allocated, reserved, retries, device allocs/frees, inactive split) on a GPU;
  6. null allocator fields on a CPU run;
  7. the vat7_dit_spike config composes WITH it and still carries the default callbacks
     (a `callbacks:` key in an experiment merges; a mistake there would silently replace).
"""
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _SONORA_REPO not in _sys.path:
    _sys.path.insert(0, _SONORA_REPO)

import json
import tempfile
import types
from pathlib import Path

import torch

from matcha.utils.memory_trace import MemoryTrace

_FAILURES = []


def check(name, ok, detail=""):
    if not ok:
        _FAILURES.append(name)
    print(f"{name}: {detail}  {'PASS' if ok else 'FAIL'}")
    return ok


def trainer(d, step=0):
    return types.SimpleNamespace(log_dir=str(d), default_root_dir=str(d), global_step=step)


def batch(frames=172, b=4):
    return {"y": torch.zeros(b, 80, frames), "y_lengths": torch.full((b,), frames)}


def host_files(d, gtt_bytes=5 * 2**30, avail_kb=90 * 2**20):
    (d / "gtt_used").write_text(f"{gtt_bytes}\n")
    (d / "meminfo").write_text(f"MemTotal: 127000000 kB\nMemAvailable: {avail_kb} kB\n")
    return str(d / "gtt_used"), str(d / "meminfo")


def lines(d):
    p = d / "memory_trace.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def one_step(cb, t, b, idx=0):
    cb.on_train_batch_start(t, None, b, idx)
    cb.on_train_batch_end(t, None, None, b, idx)


def fresh():
    return Path(tempfile.mkdtemp(prefix="memtrace_"))


# 1. one flushed line per step, with no on_train_end
d = fresh()
gtt, mi = host_files(d)
cb, t = MemoryTrace(gtt_path=gtt, meminfo_path=mi), trainer(d)
cb.on_train_start(t, None)
for i in range(3):
    t.global_step = i
    one_step(cb, t, batch(), i)
steps = [r.get("step") for r in lines(d)]
check("1 one flushed line per step", steps == [0, 1, 2], f"steps={steps}")

# 2. batch shape, wall time, process age
d = fresh()
gtt, mi = host_files(d)
cb, t = MemoryTrace(gtt_path=gtt, meminfo_path=mi), trainer(d, step=7)
cb.on_train_start(t, None)
one_step(cb, t, batch(frames=2064, b=32))
rec = (lines(d) or [{}])[0]
check("2 shape and times", rec.get("frames") == 2064 and rec.get("batch") == 32
      and rec.get("age_s", -1) >= 0 and rec.get("step_s", -1) >= 0 and "utc" in rec,
      f"{ {k: rec.get(k) for k in ('frames', 'batch', 'age_s', 'step_s', 'utc')} }")

# 3. host GTT and MemAvailable
d = fresh()
gtt, mi = host_files(d, gtt_bytes=43 * 2**30, avail_kb=53 * 2**20)
cb, t = MemoryTrace(gtt_path=gtt, meminfo_path=mi), trainer(d)
cb.on_train_start(t, None)
one_step(cb, t, batch())
rec = (lines(d) or [{}])[0]
check("3 host gtt and avail", rec.get("host_gtt_gib") == 43.0 and rec.get("host_avail_gib") == 53.0,
      f"gtt={rec.get('host_gtt_gib')} avail={rec.get('host_avail_gib')}")

# 4. absent host files -> null, no exception
d = fresh()
cb, t = MemoryTrace(gtt_path=str(d / "absent"), meminfo_path=str(d / "absent2")), trainer(d)
try:
    cb.on_train_start(t, None)
    one_step(cb, t, batch())
    rec = (lines(d) or [{"host_gtt_gib": "missing line"}])[0]
    check("4 absent host files are null", rec.get("host_gtt_gib") is None
          and rec.get("host_avail_gib") is None, f"gtt={rec.get('host_gtt_gib')!r}")
except Exception as e:  # noqa: BLE001
    check("4 absent host files are null", False, f"raised {type(e).__name__}: {e}")

# 5. allocator state on a GPU
if torch.cuda.is_available():
    d = fresh()
    gtt, mi = host_files(d)
    cb, t = MemoryTrace(gtt_path=gtt, meminfo_path=mi), trainer(d)
    cb.on_train_start(t, None)
    held = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")  # 256 MiB
    one_step(cb, t, batch())
    rec = (lines(d) or [{}])[0]
    keys = ("alloc_retries", "num_device_alloc", "num_device_free", "inactive_split_gib")
    check("5 allocator state on GPU", (rec.get("allocated_gib") or 0) >= 0.25
          and (rec.get("reserved_gib") or 0) >= rec.get("allocated_gib", 1)
          and all(k in rec for k in keys),
          f"alloc={rec.get('allocated_gib')} reserved={rec.get('reserved_gib')} "
          f"missing={[k for k in keys if k not in rec]}")
    del held
else:
    print("5 allocator state on GPU: SKIPPED (no GPU)")

# 6. CPU run -> null allocator fields
_real = torch.cuda.is_available
torch.cuda.is_available = lambda: False
try:
    d = fresh()
    gtt, mi = host_files(d)
    cb, t = MemoryTrace(gtt_path=gtt, meminfo_path=mi), trainer(d)
    cb.on_train_start(t, None)
    one_step(cb, t, batch())
    rec = (lines(d) or [{}])[0]
    check("6 CPU run nulls allocator fields", "allocated_gib" in rec
          and rec["allocated_gib"] is None and rec.get("reserved_gib") is None,
          f"alloc={rec.get('allocated_gib', 'absent')!r}")
finally:
    torch.cuda.is_available = _real

# 7. the spike config composes with the trace and keeps the defaults
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

with initialize_config_dir(version_base="1.3", config_dir=_os.path.join(_SONORA_REPO, "configs")):
    cfg = compose(config_name="train.yaml", overrides=["experiment=vat7_dit_spike"])
cbs = cfg.get("callbacks") or {}
built = instantiate(cbs["memory_trace"]) if "memory_trace" in cbs else None
check("7 spike config carries the trace and keeps the defaults",
      isinstance(built, MemoryTrace) and {"model_checkpoint", "throughput_probe"} <= set(cbs),
      f"callbacks={sorted(cbs)}")

print()
if _FAILURES:
    print(f"FAILED: {', '.join(_FAILURES)}")
    raise SystemExit(1)
print("all memory trace gates PASS")

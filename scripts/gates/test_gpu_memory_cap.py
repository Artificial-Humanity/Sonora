"""Gates for GpuMemoryCap (matcha/utils/gpu_memory_cap.py), the caching-allocator ceiling.

IN-CONTAINER (needs torch):  scripts/stages/run_in_rocm.sh scripts/gates/test_gpu_memory_cap.py

WHY IT EXISTS (2026-09-26). vat7_dit_spike's MemoryTrace showed the trainer never ALLOCATES
more than 45.5 GiB, but the caching allocator's RESERVED memory ratchets to 80-88 GiB over
two hours as variable-length batches fragment it. On this APU reserved memory is GTT taken
from host RAM, so the ratchet ended five segments in a host OOM. Once ttm.pages_limit was cut
to 90 GiB the driver refused at the cap, the allocator freed its cache and retried, and the
run survived. The cap below does the same thing earlier, inside the process, so the host keeps
its margin. The GPU gates stay under 1 GiB so they can run beside a live trainer.

  1. no cap configured -> nothing is set;
  2. the cap is enforced: an allocation larger than it raises OutOfMemoryError;
  3. THE MECHANISM: a pattern that fragments the cache past the cap is absorbed by a cache
     release and retry -- reserved stays under the cap, nothing raises, a retry is counted;
  4. a cap at or above the device's memory is refused with a warning, not silently clamped;
  5. vat7_dit_spike composes with the cap and still carries the default callbacks.
"""
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _SONORA_REPO not in _sys.path:
    _sys.path.insert(0, _SONORA_REPO)

import logging
import types

import torch

from matcha.utils.gpu_memory_cap import GpuMemoryCap

_FAILURES = []
_GIB = 1024**3


def check(name, ok, detail=""):
    if not ok:
        _FAILURES.append(name)
    print(f"{name}: {detail}  {'PASS' if ok else 'FAIL'}")
    return ok


class _Catch(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


TRAINER = types.SimpleNamespace()

# 1. no cap -> nothing set
cb = GpuMemoryCap(cap_gib=None)
cb.setup(TRAINER, None, "fit")
check("1 no cap sets nothing", cb.fraction is None, f"fraction={cb.fraction}")

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory

    # 2. enforced
    torch.cuda.empty_cache()
    cb = GpuMemoryCap(cap_gib=0.5)
    cb.setup(TRAINER, None, "fit")
    try:
        x = torch.empty(int(0.75 * _GIB), dtype=torch.uint8, device="cuda")
        del x
        check("2 cap is enforced", False, "0.75 GiB allocated under a 0.5 GiB cap")
    except torch.OutOfMemoryError:
        check("2 cap is enforced", True, f"fraction={cb.fraction:.5f} of {total / _GIB:.1f} GiB")
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    torch.cuda.empty_cache()

    # 3. the mechanism: fragment past the cap, absorbed by release-and-retry.
    # POSITIVE CONTROL first: uncapped, the same pattern must overshoot 0.75 GiB, or the
    # capped run below proves nothing about the cap.
    torch.cuda.reset_peak_memory_stats()
    for gib in (0.40, 0.45, 0.50):
        t = torch.empty(int(gib * _GIB), dtype=torch.uint8, device="cuda")
        del t
    control = torch.cuda.memory_stats().get("reserved_bytes.all.peak", 0) / _GIB
    check("3a control: uncapped, the pattern overshoots 0.75 GiB", control > 0.75,
          f"peak_reserved={control:.3f} GiB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    retries0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    cb = GpuMemoryCap(cap_gib=0.75)
    cb.setup(TRAINER, None, "fit")
    try:
        # Each size is larger than every cached block, so none can be reused: uncapped, the
        # cache would hold 0.40 + 0.45 + 0.50 = 1.35 GiB reserved for 0.50 GiB in use.
        for gib in (0.40, 0.45, 0.50):
            t = torch.empty(int(gib * _GIB), dtype=torch.uint8, device="cuda")
            del t
        s = torch.cuda.memory_stats()
        peak_res = s.get("reserved_bytes.all.peak", 0) / _GIB
        retries = s.get("num_alloc_retries", 0) - retries0
        check("3 fragmentation absorbed under the cap", peak_res <= 0.75 + 1e-6 and retries >= 1,
              f"peak_reserved={peak_res:.3f} GiB retries={retries}")
    except torch.OutOfMemoryError as e:
        check("3 fragmentation absorbed under the cap", False, f"raised OOM: {e}"[:200])
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    torch.cuda.empty_cache()

    # 4. a cap at or above the device is refused, loudly
    catch = _Catch()
    lg = logging.getLogger("matcha.utils.gpu_memory_cap")
    lg.addHandler(catch)
    cb = GpuMemoryCap(cap_gib=total / _GIB + 1)
    cb.setup(TRAINER, None, "fit")
    lg.removeHandler(catch)
    check("4 cap above the device is refused with a warning",
          cb.fraction is None and any("cap" in m.lower() for m in catch.msgs),
          f"fraction={cb.fraction} warnings={catch.msgs}")
else:
    print("2-4 GPU gates: SKIPPED (no GPU)")

# 5. the spike config carries the cap and keeps the defaults
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

with initialize_config_dir(version_base="1.3", config_dir=_os.path.join(_SONORA_REPO, "configs")):
    cfg = compose(config_name="train.yaml", overrides=["experiment=vat7_dit_spike"])
cbs = cfg.get("callbacks") or {}
built = instantiate(cbs["gpu_memory_cap"]) if "gpu_memory_cap" in cbs else None
check("5 spike config carries the cap and keeps the defaults",
      isinstance(built, GpuMemoryCap) and built.cap_gib == 64
      and {"model_checkpoint", "throughput_probe", "memory_trace"} <= set(cbs),
      f"callbacks={sorted(cbs)} cap_gib={getattr(built, 'cap_gib', None)}")

print()
if _FAILURES:
    print(f"FAILED: {', '.join(_FAILURES)}")
    raise SystemExit(1)
print("all GPU memory cap gates PASS")

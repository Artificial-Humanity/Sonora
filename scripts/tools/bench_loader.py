"""Loader ceiling: how many batches/sec can the dataloader produce, GPU idle?

Half of the local-vs-Runpod decision. matcha/utils/throughput_probe.py records
what a real run achieves; this records what the DATALOADER alone could sustain
if the GPU were infinitely fast. The gap between them is the answer:

    ceiling >> achieved   ->  GPU-bound. A faster rented GPU cuts wall-clock.
    ceiling ~= achieved   ->  loader-bound. It does not, and a 5-vCPU pod is
                              worse than this 16-core box.

Why this matters more than it looks: TextMelDataset.get_mel does sf.read plus
a mel_spectrogram per ITEM, inside the worker. The loader's cost therefore
scales with worker count, and worker count is exactly what collapses when you
move from a 16-core box (num_workers=8) to a 5-vCPU pod (num_workers=4). The
sweep below measures that scaling directly instead of assuming it is linear.

Runs on CPU. Never touches the model. Spin the inference engines down first
anyway (AI-Lab-AMD/scripts/inference-engines.sh stop) or you are timing
against contended cores and the numbers mean nothing.

Usage:
    python scripts/tools/bench_loader.py --experiment vat3_finetune \
        --workers 0,1,2,4,8,16 --batches 60 --out loader_bench.json
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# configs/paths/default.yaml resolves ${oc.env:PROJECT_ROOT}; train.py normally
# sets it via rootutils. Do the same so compose() can resolve.
os.environ.setdefault("PROJECT_ROOT", REPO)


def bench_one(cfg, workers: int, batches: int, warmup: int) -> dict:
    """Time `batches` batches at a given num_workers, after `warmup` batches."""
    cfg.data.num_workers = workers
    dm = instantiate(cfg.data)
    dm.setup("fit")
    loader = dm.train_dataloader()

    it = iter(loader)
    # Worker spin-up under the "spawn" context is slow and one-off -- each
    # worker re-imports torch. Excluding it is the whole point of the warmup.
    t_first = time.perf_counter()
    for _ in range(warmup):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
    warm_s = time.perf_counter() - t_first

    per_batch = []
    t0 = time.perf_counter()
    for _ in range(batches):
        t_a = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
        per_batch.append(time.perf_counter() - t_a)
    wall = time.perf_counter() - t0

    del it, loader, dm
    gc.collect()

    bs = int(cfg.data.batch_size)
    ordered = sorted(per_batch)
    return {
        "num_workers": workers,
        "batch_size": bs,
        "batches_timed": batches,
        "wall_s": wall,
        "batches_per_s": batches / wall if wall > 0 else None,
        "samples_per_s": batches * bs / wall if wall > 0 else None,
        "per_batch_s_p50": ordered[len(ordered) // 2],
        "per_batch_s_p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "warmup_s": warm_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="vat3_finetune")
    ap.add_argument("--workers", default="0,1,2,4,8,16",
                    help="comma-separated num_workers values to sweep")
    ap.add_argument("--batches", type=int, default=60, help="batches timed per setting")
    ap.add_argument("--warmup", type=int, default=15, help="batches discarded first")
    ap.add_argument("--out", default="loader_bench.json")
    args = ap.parse_args()

    config_dir = os.path.join(_SONORA_REPO, "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={args.experiment}"])

    workers = [int(w) for w in args.workers.split(",") if w.strip() != ""]
    results = []
    for w in workers:
        print(f"[bench] num_workers={w} ...", flush=True)
        r = bench_one(cfg, w, args.batches, args.warmup)
        results.append(r)
        print(f"        {r['batches_per_s']:.2f} batches/s  "
              f"({r['samples_per_s']:.1f} samples/s)", flush=True)

    # Per-worker marginal throughput, measured rather than assumed. Used by
    # project_pod_speedup.py to extrapolate to a pod's worker count. If this
    # falls off badly at high worker counts you are memory-bandwidth or
    # spawn-overhead bound, and a pod's fewer-but-faster cores may do better
    # than a naive linear scaling suggests.
    parallel = [r for r in results if r["num_workers"] >= 1 and r["batches_per_s"]]
    per_worker = [r["batches_per_s"] / r["num_workers"] for r in parallel]

    out = {
        "experiment": args.experiment,
        "batch_size": int(cfg.data.batch_size),
        "host_cpu_count": os.cpu_count(),
        "results": results,
        "batches_per_s_per_worker_median": statistics.median(per_worker) if per_worker else None,
        "best": max(results, key=lambda r: r["batches_per_s"] or 0) if results else None,
        "note": (
            "Compare 'best' against steps_per_s_median from the run's "
            "throughput_summary.json. Feed both to scripts/tools/project_pod_speedup.py."
        ),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[bench] wrote {args.out}")
    if results:
        b = out["best"]
        print(f"[bench] ceiling: {b['batches_per_s']:.2f} batches/s at "
              f"num_workers={b['num_workers']}")


if __name__ == "__main__":
    main()

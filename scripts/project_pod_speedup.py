"""Would renting a GPU actually cut wall-clock, and what would it cost?

Consumes the two measurements:
  - throughput_summary.json   (matcha/utils/throughput_probe.py, from a real run)
  - loader_bench.json         (scripts/bench_loader.py)

THE MODEL. With prefetching, a training step costs roughly

    step = max(gpu_compute, time_for_the_loader_to_supply_one_batch)

because the two overlap. That single max() is what makes this decision
non-obvious: renting a faster GPU shrinks the first term and does nothing to
the second, and moving to a pod makes the second term WORSE (fewer vCPUs, so
fewer mel workers). Past the crossover you are paying by the hour for a GPU
that waits on soundfile.

So the projection reports two things, and the second matters more:
  1. projected wall-clock and cost at an assumed GPU speedup
  2. the FLOOR -- wall-clock with an infinitely fast GPU, which is set purely
     by the pod's loader. No amount of money buys past it.

CALIBRATION. gpu_compute is taken from step_s_synced_median (the probe's
GPU-synchronised step sample). If that is missing it falls back to the
observed step time, which OVERSTATES gpu_compute whenever the run was
loader-bound -- and overstating it flatters the rented GPU. The output says
which path was used.

Speedup factors are yours to supply and are the weakest input here. Bandwidth
ratios are a poor proxy for training throughput on a small model. Treat the
output as a bracket, and if the bracket straddles your decision threshold,
measure one epoch on a real pod instead of arguing about the multiplier.

Usage:
    python scripts/project_pod_speedup.py \
        --summary logs/train/vat3_finetune/runs/<ts>/throughput_summary.json \
        --loader-bench loader_bench.json \
        --epochs 100 --pod-workers 4 --speedups 2,3,4 --price 0.99
"""

import argparse
import json


def fmt_h(hours: float) -> str:
    if hours != hours:  # NaN
        return "?"
    if hours < 1:
        return f"{hours*60:.0f} min"
    return f"{hours:.1f} h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="throughput_summary.json from a run")
    ap.add_argument("--loader-bench", required=True, help="loader_bench.json")
    ap.add_argument("--epochs", type=int, default=100, help="epochs in the projected run")
    ap.add_argument("--pod-workers", type=int, default=4,
                    help="num_workers on the pod (1x RTX4090/5090 = 5-6 vCPU)")
    ap.add_argument("--core-ratio", type=float, default=1.0,
                    help="pod per-core speed / local per-core speed. 1.0 unless measured.")
    ap.add_argument("--speedups", default="2,3,4",
                    help="comma-separated assumed GPU speedup factors vs local")
    ap.add_argument("--price", type=float, default=0.99, help="$/hr for the pod")
    args = ap.parse_args()

    summary = json.load(open(args.summary))
    bench = json.load(open(args.loader_bench))

    steps_per_s = summary.get("steps_per_s_median")
    epoch_wall = summary.get("epoch_wall_s_median")
    stall_frac = summary.get("stall_fraction_median")
    synced = summary.get("step_s_synced_median")
    if not steps_per_s or not epoch_wall:
        raise SystemExit("summary lacks steps_per_s_median / epoch_wall_s_median -- "
                         "was the probe running for a full epoch?")

    step_local = 1.0 / steps_per_s
    if synced:
        gpu_step, calib = float(synced), "step_s_synced_median (measured)"
    else:
        gpu_step, calib = step_local, "observed step time (FALLBACK -- flatters the pod)"

    per_worker = bench.get("batches_per_s_per_worker_median")
    best = bench.get("best") or {}
    ceiling_local = best.get("batches_per_s")
    if not per_worker or not ceiling_local:
        raise SystemExit("loader_bench.json lacks a usable ceiling -- rerun bench_loader.py")

    loader_pod_bps = per_worker * args.pod_workers * args.core_ratio
    loader_pod_step = 1.0 / loader_pod_bps

    local_hours = epoch_wall * args.epochs / 3600.0

    print("=" * 74)
    print("MEASURED")
    print("=" * 74)
    print(f"  device                  {summary.get('context', {}).get('device', '?')}")
    print(f"  observed                {steps_per_s:.3f} steps/s  ({step_local*1000:.1f} ms/step)")
    print(f"  gpu compute est.        {gpu_step*1000:.1f} ms/step   [{calib}]")
    print(f"  data stall              {stall_frac*100:.1f}% of wall" if stall_frac is not None
          else "  data stall              n/a")
    print(f"  loader ceiling (local)  {ceiling_local:.2f} batches/s "
          f"at num_workers={best.get('num_workers')}")
    print(f"  per-worker              {per_worker:.3f} batches/s/worker")
    print(f"  -> {args.epochs} epochs locally = {fmt_h(local_hours)}")

    headroom = ceiling_local / steps_per_s if steps_per_s else float("nan")
    print()
    print(f"  VERDICT: loader ceiling is {headroom:.2f}x the achieved rate.")
    if headroom >= 2.0:
        print("           GPU-BOUND. The loader has real headroom; a faster GPU should pay.")
    elif headroom >= 1.25:
        print("           MIXED. Some headroom, but a big GPU speedup will hit the loader.")
    else:
        print("           LOADER-BOUND. A faster GPU buys ~nothing. Fix the loader first")
        print("           (more workers, cache mels, or precompute them offline).")

    print()
    print("=" * 74)
    print(f"PROJECTED ON A POD  (num_workers={args.pod_workers}, "
          f"loader ceiling {loader_pod_bps:.2f} batches/s)")
    print("=" * 74)
    print(f"  {'gpu speedup':>12} {'step':>10} {'wall':>10} {'vs local':>10} {'cost':>10}")
    for s in [float(x) for x in args.speedups.split(",") if x.strip()]:
        gpu_pod = gpu_step / s
        step_pod = max(gpu_pod, loader_pod_step)   # <- the whole decision
        hours = local_hours * (step_pod / step_local)
        print(f"  {s:>11.1f}x {step_pod*1000:>9.1f}ms {fmt_h(hours):>10} "
              f"{local_hours/hours if hours else 0:>9.2f}x ${hours*args.price:>9.2f}")

    floor_hours = local_hours * (loader_pod_step / step_local)
    print()
    print(f"  FLOOR (infinitely fast GPU): {fmt_h(floor_hours)}  "
          f"= {local_hours/floor_hours:.2f}x, ${floor_hours*args.price:.2f}")
    print("  The pod's loader sets this. No GPU purchase goes below it.")
    print()
    print("  Assumptions: perfect prefetch overlap; loader scales linearly in")
    print(f"  workers at {per_worker:.3f} batches/s/worker; pod core-ratio "
          f"{args.core_ratio}.")
    print("  Linear worker scaling is the shakiest of these -- check the sweep in")
    print("  loader_bench.json for fall-off before trusting a large extrapolation.")


if __name__ == "__main__":
    main()

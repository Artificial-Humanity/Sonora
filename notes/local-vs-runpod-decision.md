# Local vs Runpod: measure it during the next run

Owner's criterion (2026-07-31): *"The only reason I'd pay for Runpod training
time is if it can actually reduce training time significantly."* 48 h → 6 h for
$6 is obviously worth it; 48 h → 14 h is worth considering; 48 h → 40 h is not.

Next training session runs **local**. The instrumentation below rides along so
the decision after it is made on data instead of spec sheets.

## The one equation

With prefetching, a step costs about

```
step = max( gpu_compute , time for the loader to supply one batch )
```

That `max()` is the whole decision. Renting shrinks the first term and does
nothing to the second — and moving to a pod makes the second term **worse**,
because `TextMelDataset.get_mel` does `sf.read` + `mel_spectrogram` per item
inside a worker, and worker count collapses from 8 (16 cores) to 4 (5 vCPU).

Past the crossover you are paying by the hour for a GPU that waits on
soundfile.

## What runs, and when

**1. During training — automatic.** `ThroughputProbe`
(`matcha/utils/throughput_probe.py`) is now in `configs/callbacks/default.yaml`.
Costs microseconds per step plus one `cuda.synchronize()` every 50. Writes into
the run dir, incrementally, so a killed run still leaves data:

- `throughput_probe.jsonl` — one line per epoch
- `throughput_summary.json` — written at train end *and* on crash/Ctrl-C

Disable with `~callbacks.throughput_probe` if a run must be pristine.

**2. Once, separately — the loader ceiling.** Answers "how fast could the
loader go if the GPU were infinite?"

```
AI-Lab-AMD/scripts/inference-engines.sh stop     # or you time contended cores
python scripts/bench_loader.py --experiment vat3_finetune \
    --workers 0,1,2,4,8,16 --batches 60 --out loader_bench.json
```

CPU-only, never touches the model. The sweep matters as much as the peak: it
measures how throughput scales with workers rather than assuming linearity,
which is the extrapolation the projection leans on hardest.

**3. After — the projection.**

```
python scripts/project_pod_speedup.py \
    --summary logs/train/vat3_finetune/runs/<ts>/throughput_summary.json \
    --loader-bench loader_bench.json \
    --epochs 100 --pod-workers 4 --speedups 2,3,5 --price 0.99
```

## How to read it

The headline is `loader ceiling / achieved rate`:

| ratio | meaning |
| --- | --- |
| ≥ 2× | GPU-bound. A faster rented GPU should pay. |
| 1.25–2× | Mixed. A big GPU speedup will run into the loader. |
| < 1.25× | Loader-bound. A faster GPU buys ~nothing. |

Then read the **FLOOR** line — wall-clock with an infinitely fast GPU, set
purely by the pod's loader. No amount of money goes below it. If the floor is
already above your threshold, stop; no GPU choice rescues it.

**A worked trap** (illustrative numbers, not measurements): a run that is
GPU-bound locally with 2× loader headroom, moved to a pod with half the
workers, projects **1.00× speedup for $49.50**. Local headroom does not
survive the vCPU cut. This is the specific failure this note exists to
prevent.

## If it comes out loader-bound

Renting is the wrong lever; the fix is to stop recomputing mels every epoch:

- precompute mels offline and load tensors instead of decoding wavs
- or cache them on first epoch
- `--workers` sweep will also show whether local `num_workers=8` is even optimal

Either makes the loader term small, at which point renting a faster GPU starts
to pay — and the projection can be re-run to confirm before spending.

## Known-weak inputs

- **GPU speedup factor** is supplied by hand and is the weakest input.
  Bandwidth ratios are a poor proxy for training throughput on an ~18M-param
  model. Treat output as a bracket; if the bracket straddles the decision,
  measure one epoch on a real pod (`sky/smoke-runpod.yaml` is the cheap harness).
- **Linear worker scaling** on the pod. Check `loader_bench.json` for fall-off.
- **Per-core speed ratio** defaults to 1.0 (`--core-ratio`); unmeasured.
- The probe calibrates `gpu_compute` from `step_s_synced_median`. If that is
  missing it falls back to observed step time, which *overstates* gpu_compute
  and flatters the pod. The output says which path it used.

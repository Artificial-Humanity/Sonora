# Local vs Runpod: measure it during the next run

Owner's criterion (2026-07-31): *"The only reason I'd pay for Runpod training
time is if it can actually reduce training time significantly."* 48 h → 6 h for
$6 is obviously worth it; 48 h → 14 h is worth considering; 48 h → 40 h is not.

Next training session runs **local**. The instrumentation below rides along so
the decision after it is made on data instead of spec sheets.

> **STATUS 2026-08-06 — the run happened, the data is there, the projection has not been
> run.** `vat3c` (run `2026-08-05_15-02-57`, 100 epochs, exit 0) carried the probe and left
> **51 epochs** in `throughput_probe.jsonl`. Headline from it: **stall_fraction ≈ 0.005** —
> the loader is supplying batches with the GPU waiting on it about half a percent of the
> time, i.e. this run was firmly **GPU-bound** locally, `steps_per_s` 1.26 → 1.48 and
> `gpu_mem_peak` 18.9 GB.
>
> ⚠ **Two things block the projection as documented below.** (1) `throughput_summary.json`
> was **never written** — not for the clean exit-0 run and not for any of the four crashed
> ones — so `project_pod_speedup.py --summary` has no input. Either the end-of-train hook
> does not fire under this launch path, or it fails silently; fix that before trusting
> "a killed run still leaves data." (2) The loader-ceiling sweep (step 2) has not been run,
> and it is the term the whole decision turns on. **GPU-bound locally is exactly the case
> the worked trap below is about**: local headroom does not survive the pod's vCPU cut.
>
> ✅ **PARTLY UNBLOCKED 2026-08-09 (PR-M5), and the headline answer is: rung 3 does not
> need this decision.** The `throughput_summary.json` gap is real and still open, but it
> obscured the fact that **the per-epoch probe DID write** — 48 epochs of
> `throughput_probe.jsonl` sit in the v5 run directory, and they carry the terms the
> projection needs. Measured over the 47 steady-state epochs (epoch 0 is the 3,941 s
> MIOpen ramp, as documented):
>
> | term | v5, measured |
> |---|---|
> | samples/s | **39.80** mean · 40.26 p50 · 35.80 min |
> | wall per epoch | **1,015 s** (16.9 min) over 41,138 rows |
> | **loader stall fraction** | **0.47%** mean, 0.57% max |
> | GPU mem peak | 18.9 GB (batch ≈ 32) |
>
> **Rung 3 projected at that rate:** v7 is ~615 h ≈ 322k rows, so **2.25 h/epoch — about
> ONE DAY of GPU for the ~10 epochs this lineage actually converges in**, and 4.5 days for
> a full 48. Not multi-week. **So the local-vs-cloud call does not have to precede the v7
> corpus build**, which was the specific worry: an interrupted multi-week run is a very
> different risk from an interrupted one-day run. Even if the loader degrades **3×** at 8×
> the corpus — well past anything the 0.47% stall fraction suggests — v7's converged
> portion is ~2.8 days.
>
> ⚠ **What this does NOT answer, and it is still step 2's job:** whether the loader holds
> at 8× the rows. 0.47% stall says there is large headroom *at v5's size*; it does not say
> where the ceiling is, and the pod's vCPU cut is applied to the ceiling, not to the
> achieved rate. The sweep is still the right thing to run before *renting* anything — it
> is simply no longer blocking rung 3.

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
.venv/bin/python scripts/bench_loader.py --experiment vat3c_finetune \
    --workers 0,1,2,4,8,16 --batches 60 --out loader_bench.json
```

CPU-only, never touches the model. The sweep matters as much as the peak: it
measures how throughput scales with workers rather than assuming linearity,
which is the extrapolation the projection leans on hardest.

**3. After — the projection.**

```
.venv/bin/python scripts/project_pod_speedup.py \
    --summary logs/train/vat3c_finetune/runs/<ts>/throughput_summary.json \
    --loader-bench loader_bench.json \
    --epochs 100 --pod-workers 4 --speedups 2,3,5 --price 0.99
```

(Host scripts run as `.venv/bin/python`, never `uv run` — [AGENTS.md](../AGENTS.md) §3.)

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

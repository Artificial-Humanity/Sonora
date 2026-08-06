# Changelog

This document tracks technical changes, refactoring milestones, and build-system adjustments
for Project Sonora (the training pipeline and the teacher-synthesis lane).

> **Maintenance:** Required by [AGENTS.md](../AGENTS.md) §4, which is the rule of record —
> append an entry **after committing** code work (source, configs, dependency manifests;
> docs-only commits are exempt), always carrying the short 7-character commit SHA. The log is
> append-only within a release cycle: entries are pruned **only** when a new version of the
> overall project is tagged, at which point they collect under that version's heading. New
> entries go at the top under the current date, using the `Added` / `Changed` / `Fixed` /
> `Removed` structure.
>
> Maintained since **2026-08-06**, when the changelog + code-review cycle (AGENTS.md §4–§5)
> was adopted here to match the convention Prosodia already runs. History before that date
> lives in `git log` and [STATE.md](STATE.md).

---

## 2026-08-06

### Fixed

- **`e5e5ed1` — E-M5: the logged diffusion loss is masked.** `BASECFM.compute_loss` summed
  its residual over the full padded tensor. The decoder masks its own output, but the
  target residual `u = x1 - (1-σ)z` does not (`z` is drawn `randn_like` across the padding),
  so each batch's logged `diff_loss` carried a floor proportional to its padding fraction.
  Train batches have been length-bucketed since 2026-08-01 and val batches are not, so the
  floor fell on val and presented as a 3.2× train/val gap — vat3c epoch 1 logged diff 0.646
  train vs 2.069 val while `dur_loss` and `prior_loss` matched to three decimals.
  Gradients are unchanged (the padded positions carried none); **only the logged value
  moves, and it moves down.** `diff_loss` and `loss/train` / `loss/val` therefore do not
  compare across this commit — a second scale break in the metric after the bucketing
  change. Documented at every site that teaches someone to read the curve.
- **`e5e5ed1` — E-M6: the RoPE cache is built outside inference mode.**
  `on_validation_end` synthesises under `@torch.inference_mode()`, so a
  `RotaryPositionalEmbeddings` cache first built or grown during validation was made of
  inference tensors; the next training step with text no longer than the cached length
  reused them and killed the run with "Inference tensors cannot be saved for backward".
  The build now nests `torch.inference_mode(False)`, so the cache is an ordinary detached
  tensor regardless of which mode first demanded it.

### Added

- **`e5e5ed1` — `tests/test_training_seams.py`**, regression coverage for both of the
  above. Verified to fail against pre-fix source (4.643 vs 1.643 logged loss at 75%
  padding; the RoPE case reaching the real backward crash). The tests need the model's
  dependency stack, so they `importorskip` on the host venv and run in the ROCm training
  container:
  `docker compose run --rm --entrypoint pytest sonora_training tests/test_training_seams.py`

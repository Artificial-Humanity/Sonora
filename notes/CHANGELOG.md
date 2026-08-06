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

### Added

- **`f015c8e` — `scripts/score_holdout.py` + `score_holdout.sh`, the never-trained holdout
  (Phase 0a).** Scores a checkpoint teacher-forced per clip with **paired** noise draws, so
  two checkpoints see identical timesteps and noise on identical clips; clip-to-clip
  variance dwarfs the effect being measured, and an unpaired comparison of two 5,463-clip
  means reads as noise. Runs in a throwaway ROCm container as `ai-mgr`, building
  `monotonic_align` in a `/tmp` copy so the owner's checkout is neither written to nor
  littered with build artifacts. **No `phonemizer` in the eval deps** — `matcha.text.cleaners`
  imports it lazily, the filelists are already IPA, so pulling GPL in to satisfy an import
  that never fires would be the G-3/G-4 mistake voluntarily.
- **`f015c8e` — `configs/data_licenses.yaml` declares `libritts_r_holdout_devclean`.**
  Without it `enforce()` refuses to load the filelist — the same trap that made v3/v3b/v3c
  structurally unrunnable. Declaring it is **not** permission to train on it, and the wall
  will not stop such a run; the guards are `--assert-disjoint-from` (any shared clip
  basename stops the run) and the deletion of the derive's `train_op.txt`/`val_op.txt` after
  concatenation into `holdout.txt`, so no file in the directory carries a name a training
  config would accept.

### Changed

- **`f015c8e` — `vat3c` ep099 is retired; `vat3-24k` ep099 is the base.** The holdout says
  the v3c fine-tune was a **regression, not a no-op**: +0.0164 against its own warm start
  over 5,463 unseen clips, all three loss terms worse, better on only 39.1% of clips, and
  +0.0443 worse on v3c's *own* val split. Not a normalisation artifact (+0.0172 under the
  v2 constants it trained with). The ear had already said "no audible change"; this
  sharpens the direction to *down*. **Phase 0b — the clean-lineage retrain from
  `matcha_vctk` — is not indicated**, because the v2 fine-tune shows a real gain on unseen
  audio (`diff` −0.0241, better on 78.7% of clips) and a compromised lineage could not
  produce that. Owner's call to ratify. Recorded in `quality-gap-plan.md` § 0a,
  `STATE.md` and `training-sources.md`.

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

# Registry & Artifact-Flow Housekeeping (deferred)

_Captured 2026-07-13. Status (end of day): **items 1–4 DONE**, and the layout question is
**settled: end-state C implemented** — the registry clone lives at the workspace-root
`Registry/Sonora/`, a sibling of `Reference/` (after briefly living at `Sonora/model/` earlier
the same day under an initial status-quo call). Implemented on both machines 2026-07-13.
This note is now a record, not a queue._

## Why this note exists

The current Sonora artifact setup works but "feels muddy" (owner's words): "Sonora" names both
the GitHub training repo (the factory) and the HF model registry (the warehouse), the warehouse
is checked out *inside* the factory (`Sonora/model/`, gitignored), and a blessed copy of one
artifact is materialized a third place (`Reference/models/sonora.tflite`, refreshed by
`bootstrap.sh`). Before changing anything, we surveyed how the ecosystem handles
model-in-training projects. Verdict: the current shape is **within normal variance of the
standard** (code in git / weights in a registry / linked by reference — cf. Piper TTS:
GitHub engine + per-voice HF repos), so none of this is urgent.

## Cheap improvements worth doing under ANY layout

These are the real housekeeping items — they carry over regardless of the end-state below.
_All four executed 2026-07-13 (registry commit `a889bf0`; umbrella bootstrap/snapshot updates)._

1. ✅ **Model card on the HF registry.** Standard HF README with YAML metadata: license, base
   model (Matcha-TTS), dataset (LJSpeech), eval metrics (deterministic ONNX↔TFLite cosine
   1.0000, ASR WER 0.000), intended use, known limits (50-token static shape, 22.05 kHz,
   fp16-I/O variant unusable on the current engine). Cheap, and it is exactly the provenance
   record the AGENTS §8 mandate keeps re-learning the value of.
2. ✅ **Provenance convention per promotion.** Every registry commit records which training-repo
   SHA and which MLflow run produced the artifact (one-line commit-message convention, or a
   small JSON beside the checkpoint). Natural extension of `snapshot_versions.sh`/VERSIONS.md —
   the registry SHA is one more row.
3. ✅ **`config.json` into the registry.** The engine-contract config (locked 178-symbol vocab +
   `sample_rate`) currently has no remote source; committing it next to the float32 export makes
   the artifact self-describing and turns the `bootstrap.sh` warning into a fetch.
4. ✅ **Consumers pin a registry revision.** `bootstrap.sh`'s cmp-copy is a hand-rolled version of
   this; the mature form records the HF revision SHA it materialized (lockfile-style) instead of
   "whatever the checkout has." _Implemented via `snapshot_versions.sh`: VERSIONS.md carries a
   `Sonora/model` row and `bootstrap.sh --at-versions` restores the registry pin alongside the
   code repos._

## Layout end-states (only if the muddiness starts costing something)

* **A. Status quo** — registry clone at `Sonora/model/` (gitignored). Keeps git semantics for
  promotion (commit messages, local diff, NFS/offline workflow). The nesting is the only wart.
  *(Initially affirmed 2026-07-13, superseded by C the same day.)*
* **B. HF-native, no checkout** — delete `Sonora/model`; promote via `hf upload`/`push_to_hub`
  from training scripts; consumers fetch pinned revisions via `huggingface_hub` into the global
  cache. Conceptually cleanest; costs the git-style local workflow.
* **C. Registry as sibling** — **CHOSEN & implemented 2026-07-13 (Mac).** Checkout moved to the
  workspace-root `Registry/Sonora`, parallel to how the umbrella treats GitHub repos as gitignored
  siblings. Kills the repo-in-repo nesting, keeps git semantics. (The pre-2026-07-13 `Models/Sonora`
  was this pattern; its actual flaw was the misleading `Models` parent, since fixed by
  `Reference/` for the true reference clones.)

All three are small migrations from the status quo. Items 1–4 above are the ones with real
payoff; the A/B/C choice is taste until some concrete friction (e.g. multi-machine promotion
conflicts, or registry LFS bloat in the checkout) picks one.

## Cross-references

* AGENTS.md §8 (Reference Directory Mandate) — read-only reference library rules.
* [STATE.md](STATE.md) Current State 2026-07-13 bullets — the relayout + rename this builds on.
* `bootstrap.sh` (umbrella root) — current materialization of the blessed actor artifact.
* Notes/Ai-Lab-0 — the box mirrors the full layout (2026-07-13): reference layer behind
  `Reference/models` → `/data/reference/models`, and its registry clone at `Registry/Sonora`
  in its workspace checkout (moved out of the Sonora checkout's `model/` the same day).

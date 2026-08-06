# Closing the quality gap — the ordered plan

**Scope: what we do next to close the synthetic-vs-real gap, and in what order.**
Measurement repair → data sequencing → decoder spike. This file owns the *sequencing
and the gates between phases*; it does not restate the plans it sequences. The DiT
design itself is [model-decisions.md § Decoder v2](model-decisions.md); source state is
[training-sources.md](training-sources.md) (SSOT); run mechanics are
[training-operations.md](training-operations.md).

Written 2026-08-06, immediately after the diagnostics below. **Nothing here has started.**

---

## What scoped this (2026-08-06)

`vat3c_finetune` ran 100 epochs off the v2-trained warmstart onto v3c's clean IPA
(run `2026-08-05_15-02-57`, exit 0). Three measurements came out of it, and together
they say where the remaining quality lives.

**1. The G2P fix was the whole fix — retraining was never needed.** Per-clip *paired*
scoring (same torch seed per clip/draw across both checkpoints) of `vat3c_init` vs
`epoch=099`, over the 2,784 clips whose phonemes changed between v3 and v3c, against
2,782 speaker/length-matched controls:

| term | corrected | control | attributable to the fix |
|---|---|---|---|
| prior | −0.560% | −0.427% | **−0.133%** |
| dur | −5.568% | −4.158% | **−1.411%** |
| diff | −2.554% | −2.242% | **−0.311%** |

All three significant (bootstrap CIs exclude zero), all perceptually negligible. Blind
A/B renders confirmed **no audible difference**. The poisoning was text→phoneme only;
the model's phoneme→audio mapping was always sound, so the warmstart — which never
trained on a single corrected contraction — already says `I'm` and `that's` correctly.
Fixing `op_g2p` closed it. *Generalizes: check the front end before spending a run.*

**2. The vocoder is not the bottleneck.** Copy-synthesis (real audio → mel with exact
training params → `cp_hifigan_24k/g_02510000`, no acoustic model in the path) gives mean
**mel L1 = 0.2447**, uniform across 10 clips = **10.2% of this corpus's mel_std (2.389)**.
Owner's verdict: *"similar to ref, if not identical."* This **rules out** vocoder
training, vocoder replacement and sample-rate work as quality levers.

**3. The gap belongs entirely to the acoustic model's predicted mel.** That is what both
phases below target, and it is why they are the right two phases.

**What this does NOT support:** starting the project over. The architecture responds to
conditioning, the corpus is sound, the 1,802 human ratings are untouched, and no
checkpoint was ever selected by the broken metric (`monitor: epoch`, not val loss).

Harness kept at `/data/model-training/sonora/eval_phoneme_fix/` —
`build_sets.py`, `eval_phoneme_fix.py`, `render_ab.py`, `copy_synth.py`.

### Two traps this exposed, for any future comparison

- **Always randomise A/B assignment per pair.** The owner's "the second one sounds
  better" tracked *position*, not checkpoint — per-pair randomisation caught it because
  A/B was reversed on the pairs heard. A fixed A=old/B=new layout would have recorded a
  false positive.
- **Copy-synthesis must skip `normalize()`.** `TextMelDataset.get_mel` normalises for
  *model* input, but `MatchaTTS.synthesise` returns `denormalize(...)` before vocoding —
  **the vocoder consumes RAW log-mel.** Normalising first produces garbage. Assert the
  vocoder's `config.json` against the data config's mel params first (they agree:
  24000/80/256/1024/0/12000).

---

## Phase 0 — make it measurable

Everything downstream is wasted if we cannot tell whether it worked. Today we cannot:
`loss/val_epoch` has never been a generalization measure, because splits were re-drawn
per corpus version while runs warm-started from the previous version. **93–97% of v3c's
val clips were trained on under an earlier corpus.** Cross-run val comparisons — the
3.18 (07-20) vs 3.48 (08-05) pairing in particular — are invalid.

Going forward the split itself is fine: `derive_vat_corpus._in_val` hashes the wav
basename (blake2b, `SPLIT_SALT="sonora-vat-split-20260802"`) as of 2026-08-02, and adding
2,000 rows moved 0 of 960 val clips. **The contamination is historical**, inherited
through the warm-start chain. It does not fix itself.

### 0a — the never-trained holdout (do this; it is the instrument)

Pull **LibriTTS-R `dev-clean`** (~5.4 h / 40 speakers, ~1 GB — we hold only
`train-clean-100`), derive it as a scoring-only set, and score every retained checkpoint
against it. Same corpus family, same recording conditions, same license entry, and *no
checkpoint in the lineage has ever seen a frame of it.*

- **Cost: one small download and a derive pass. No training.**
- Retroactive: it re-measures checkpoints we already have, so the last six weeks of runs
  become comparable for the first time.
- Gates as a corpus even though it never trains: it needs a `data_licenses.yaml` entry
  or `enforce()` refuses the run (this is exactly the v3 trap), and phonemes must come
  from the **fixed** `op_g2p`.
- Scoring-only means **no** `data_statistics` re-measure — normalise it with the
  training corpus's constants, or the numbers are not comparable to anything.

### 0b — clean-lineage restart (decide after 0a, not before)

If 0a shows the fine-tune lineage is genuinely compromised rather than merely
unmeasured, the clean restart is warm-starting from `matcha_vctk.ckpt` — VCTK-trained,
never saw LibriTTS-R, so v3c's frozen split is honest against it from the first step.

**Be honest about the cost: this is a retrain, not a fine-tune.** `matcha_vctk` is
22.05 kHz, VCTK speaker set, no VAT trunk — the decoder is effectively starting over at
24 kHz with our conditioning. It buys a clean measurement, not a free one. *(An earlier
verbal framing of this as "near-zero cost" was about abandoning the fine-tune lineage,
which is cheap; the retrain is not.)* Defer it until 0a says whether it is needed.

---

## Phase 1 — data sequencing

Cheapest first, so the expensive item is decided by evidence rather than assumption.

| # | Source | State | Adds | Work |
|---|---|---|---|---|
| 1 | **Emilia-YODAS keeps** | READY, 24 kHz, license-cleared | 13,141 clips (**+43%**) | `eiv_merge_corpus.py` — merge only |
| 2 | **sonora-expressive-registers** | DERIVED, 1,071 keeps | small, ear-certified | close **+116** (Neutral +81, Documentary +35) → 1,156, then merge |
| 3 | **LibriTTS-R, the other 90%** | NOT PULLED | **~10×**, ~2,400 speakers | download + full corpus build |
| 4 | **Hi-Fi TTS v1** | RAW, 40 GB parquet *with audio* | 292 h / 10 speakers | parquet → wav + filelist |

**Why this order.** #1 is nearly free and already cleared — the fastest available test of
whether volume moves quality *at all*. #2 is the first time expressive, ear-certified
material enters the training path; today the corpus we spend ear time on and the corpus
we train on have never been joined, which is the largest structural oddity in
`training-sources.md`. #3 is the real lever but the largest effort, and it is the one
worth gating on #1's result. #4 plays a different role — few voices, deep hours — so it
is casting anchors and speaker-consistent long-form prosody, not volume; it is
independent of #3 and can slot in whenever the parquet conversion is worth doing.

### Per-corpus checklist — every one of these has bitten

- **`configs/data_licenses.yaml` entry**, or `enforce()` refuses the run. This is what
  silently made v3/v3b/v3c unrunnable; "nothing has trained on v3c" was structurally
  guaranteed, not merely true. **Check this file first whenever a new corpus dir appears.**
- **VAT labels** via `derive_vat_corpus.py`.
- **Phonemes** via `rephonemize_corpus.py` on the **fixed** `op_g2p`.
- **`data_statistics` re-measured in-container** for the new mix
  (`matcha/utils/generate_data_statistics.py`). Do not inherit the previous version's
  mel mean/std silently — v2→v3c moved them by 0.0203/0.0024, a constant shift on every
  normalised mel.
- **Explicit corpus bind in compose.** The `data ->` symlink is untracked and does not
  survive a clone.
- **Split is automatic** — the hash split needs no attention, which is the point of it.
- **QC gate** after every generation pass, and **spin down all inference engines**
  before any run.

### Closed 2026-08-06, before the phase started

`train_diff_loss` 0.643 vs `val_diff_loss` 2.085 — the **3.2× gap** — was E-M5. The
diffusion loss summed its residual unmasked while the decoder masks its output, so every
batch paid a floor proportional to its padding; train batches are length-bucketed and val
batches are not, which is the whole of the "gap". `out_size` and clip-length distribution
had already been ruled out and were right to be. `compute_loss` masks now, gradients
unchanged, and `tests/test_training_seams.py` holds the curve honest. **Cost:** a second
scale break in logged loss (after 2026-08-01's bucketing change) — nothing before this commit compares to anything after it,
which is one more reason Phase 0a's holdout scoring is the real measurement.

---

## Phase 2 — the DiT decoder spike

Design, gates and known risks are already written: **[model-decisions.md § Decoder v2](model-decisions.md)**
(the content of the former `decoder-v2-dit-spike.md`, folded in during the 2026-08-02
consolidation — the standalone file no longer exists). Start from StableTTS's 31M
`DiTConVBlock` shape, which retires the spike's named tiny-scale risk in public MIT code
([matcha-siblings-study.md](matcha-siblings-study.md)).

What this file adds is **when**, and one amendment:

- **After Phase 1 lands and Phase 0 works.** Architecture work is CPU-side until its
  de-risk run, so authoring can overlap; the *gate* cannot.
- **The parity gate must run against a U-Net baseline trained on the same expanded
  corpus.** Otherwise data and architecture confound each other and the run teaches
  nothing about either. Freeze that baseline as the last act of Phase 1.
- Adopt iff the gate passes; **stall → the stock decoder runs the corpus** and the spike
  parks with its findings. The schedule never waits on the swap.

This remains a spike, not a pivot: the decoder is ~7.6% of the codebase.

---

## Gates between phases

| gate | question | consequence of the answer |
|---|---|---|
| after **0a** | do retained checkpoints separate at all on never-trained audio? | if not, the instrument is wrong before the model is — fix it before spending anything |
| after **1.1** | does +43% move the clean holdout? | no movement ⇒ likely capacity-limited, not data-limited ⇒ **promote Phase 2 ahead of the 10×** |
| after **Phase 1** | is the vocoder still transparent? | re-run `copy_synth.py`; a better acoustic model can walk into the vocoder's ceiling unnoticed |
| before **Phase 2** | is there a same-corpus U-Net baseline? | if not, the parity gate is unreadable — do not start |

## Why data before decoder

Data first is lower-risk, needs no architecture work, and answers the question that
governs everything else: whether ~30k clips was ever the constraint. Doing the spike
first would mean evaluating a new decoder against a corpus we are about to change —
the same confound that made the vat3c run so hard to read.

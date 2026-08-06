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

Everything downstream is wasted if we cannot tell whether it worked. Until 2026-08-06 we
could not: `loss/val_epoch` has never been a generalization measure, because splits were
re-drawn per corpus version while runs warm-started from the previous version. **93–97%
of v3c's val clips were trained on under an earlier corpus.** Cross-run val comparisons —
the 3.18 (07-20) vs 3.48 (08-05) pairing in particular — are invalid.

**0a closed that.** There is now a never-trained holdout and every retained checkpoint
has a number on it; read 0a below before reading any MLflow curve. The rest of this
section is kept because the defect it describes is permanent — the historical
contamination does not heal, so `loss/val_epoch` stays unusable for cross-run
comparison even now.

Going forward the split itself is fine: `derive_vat_corpus._in_val` hashes the wav
basename (blake2b, `SPLIT_SALT="sonora-vat-split-20260802"`) as of 2026-08-02, and adding
2,000 rows moved 0 of 960 val clips. **The contamination is historical**, inherited
through the warm-start chain. It does not fix itself.

### 0a — the never-trained holdout — **DONE, and it reported. 2026-08-06**

**LibriTTS-R `dev-clean`**, derived scoring-only as
`data/libritts_r_holdout_devclean`: 5,463 clips / 8.7 h / **40 speakers, zero overlap**
with the corpus's 247. Every retained VAT checkpoint scored teacher-forced with four
paired noise draws per clip — 43,704 per-clip rows, `scripts/score_holdout.{py,sh}`,
raw at `/data/model-training/sonora/holdout_eval/lineage.csv`. No training was done.

| checkpoint | dur | prior | diff | **total** |
|---|---|---|---|---|
| `vat3_init` (warm start of the v2 run) | 0.5369 | 1.0351 | 0.2903 | 1.8623 |
| **`vat3_ep099`** (v2 corpus) | 0.5466 | 1.0384 | 0.2662 | **1.8512** |
| `vat3c_init` | 0.5466 | 1.0384 | 0.2662 | 1.8512 |
| **`vat3c_ep009`** | 0.5463 | 1.0380 | 0.2663 | **1.8506** |
| `vat3c_ep029` | 0.5533 | 1.0393 | 0.2679 | 1.8604 |
| `vat3c_ep049` | 0.5468 | 1.0393 | 0.2693 | 1.8554 |
| `vat3c_ep069` | 0.5540 | 1.0389 | 0.2678 | 1.8607 |
| `vat3c_ep099` (shipped as the v3c result) | 0.5562 | 1.0400 | 0.2714 | 1.8676 |

**1. The gate passes: checkpoints separate on never-trained audio.** `vat3_ep099` beats
its own warm start by **−0.0111** [−0.0131, −0.0091]. The instrument resolves the
lineage, so the speaker-conditioning floor (dev-clean's 40 speakers land on arbitrary
trained embeddings — see `score_holdout.py`) did **not** swamp the signal, and the ECAPA
nearest-voice mapping held in reserve for that case is **not needed**.

**2. The v2 fine-tune genuinely generalized.** Its gain is concentrated exactly where it
should be: `diff` — the flow-matching objective that actually generates the mel —
improved **−0.0241, on 78.7% of clips**, while `dur`/`prior` gave back a little.

**3. The v3c fine-tune was a regression, not a no-op.** `vat3c_init → ep099` is
**+0.0164** [+0.0148, +0.0180], better on only **39.1%** of clips, and **all three
terms worsened** — there is no component it bought. It is monotone-ish from `ep009`
onward, and `ep009` is statistically tied with the init it started from (−0.0006,
CI spans zero). *The best checkpoint in the entire lineage is `vat3_ep099`.*

**4. Same verdict on v3c's own val split**, which it trained on: `vat3c_ep099` loses to
`vat3_ep099` by **+0.0443**, better on just **14.5%** of 960 clips. So this is not
holdout-specific and not a distribution artifact — the run is worse everywhere under
teacher-forced loss.

**5. It is not a normalisation artifact.** Re-scored under the **v2** constants the
checkpoint was trained with, the gap is +0.0172 against +0.0164 under v3c's. The
asymmetry also ran *against* the conclusion — `vat3_ep099` won while being scored under
constants it never saw.

**6. `vat3c_init` is bit-identical to `vat3_ep099`** across all 5,463 clips (Δ = 0.0000,
every clip). The warm start is now verified by measurement rather than by a 338/338 log
line, which also means a checkpoint's per-clip numbers are reproducible within a run.

This is consistent with the ear ("no audible change") and sharpens it: the ear could not
resolve a 0.9% move, and the direction is *down*. It also lines up with the two findings
that bracket it — the vocoder is transparent, so the gap is the acoustic model's; and
the v2→v3c label change was tiny (corr ≥ 0.9993, a phoneme fix on 6.4% of rows). **100
more epochs against ~30k clips a model was already fit to had almost no new signal to
learn, so it sharpened to the training set instead.** That is the evidence Phase 1 was
assuming: more epochs on this corpus is not the lever; more corpus is.

**Actions.** Retire `vat3c_ep099` — do not stage or export it. `vat3_ep099`
(= `vat3c_init`) stays the base and the warm start for Phase 1. Score every future
checkpoint on the holdout as a matter of course; it costs ~100 min for eight checkpoints
on an idle card and it is now the only honest number we have.

### 0b — clean-lineage restart — **NOT INDICATED. Owner's call to ratify.**

0b was conditional: *"if 0a shows the fine-tune lineage is genuinely compromised rather
than merely unmeasured."* It does not. The lineage **generalizes** — finding 2 is a real
gain on audio no checkpoint had seen, which is precisely what a compromised lineage
could not produce. What 0a found is narrower and cheaper to fix: **one wasted run**, not
a poisoned ancestry. Dropping `vat3c_ep099` recovers everything the restart would have.

That matters because 0b is **a retrain, not a fine-tune**: `matcha_vctk` is 22.05 kHz,
VCTK speaker set, no VAT trunk, so the decoder starts over at 24 kHz with our
conditioning. *(An earlier verbal framing of this as "near-zero cost" was about
abandoning the fine-tune lineage, which is cheap; the retrain is not.)* On this evidence
that cost buys nothing the holdout does not already give us. **Recommendation: close 0b
and go straight to Phase 1** from `vat3_ep099`. Reopen it only if a Phase 1 run with
materially more data also fails to move the holdout, which would point at the ancestry
rather than the corpus.

### What the holdout is, and the one way to destroy it

`data/libritts_r_holdout_devclean` is worth exactly one thing: no checkpoint has trained
on it. One epoch ends that permanently, and there is no second dev-clean. The directory
therefore ships with a `README.md` saying so, and `derive_vat_corpus.py`'s `train_op.txt`
/ `val_op.txt` were **deleted** after concatenation into `holdout.txt`, so no file in it
carries a name a training config would accept. Note that `configs/data_licenses.yaml`
*does* declare it — the wall gates on provenance, not intent, and would happily let a run
train on it. The naming is the guard.

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
| ~~after **0a**~~ **PASSED 2026-08-06** | do retained checkpoints separate at all on never-trained audio? | **yes** — `vat3_ep099` − `vat3_init` = −0.0111, CI excludes zero. Instrument is sound; 0b closed, Phase 1 proceeds from `vat3_ep099` |
| after **1.1** | does +43% move the clean holdout? | no movement ⇒ likely capacity-limited, not data-limited ⇒ **promote Phase 2 ahead of the 10×** |
| after **Phase 1** | is the vocoder still transparent? | re-run `copy_synth.py`; a better acoustic model can walk into the vocoder's ceiling unnoticed |
| before **Phase 2** | is there a same-corpus U-Net baseline? | if not, the parity gate is unreadable — do not start |

## Why data before decoder

Data first is lower-risk, needs no architecture work, and answers the question that
governs everything else: whether ~30k clips was ever the constraint. Doing the spike
first would mean evaluating a new decoder against a corpus we are about to change —
the same confound that made the vat3c run so hard to read.

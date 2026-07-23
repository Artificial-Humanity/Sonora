# §7 De-risk Verdict — Energy Channel: PASS (2026-07-16)

_The formal record of the north-star §7 de-risk experiment's outcome. The experiment asked one
question: **does the FiLM/VAT conditioning architecture actually produce directable speech**, on
one cheap-to-label channel, before we commit to the much larger investment of full 3-channel VAT
training (which needs valence/tension-labeled data that doesn't exist yet in a shippable form)?
Answer: **yes, decisively.**_

## Pre-registered criteria (set before training, unchanged)

From the eval harness (`Sonora/scripts/eval_harness.py`, thresholds in code) and recorded in
[training-operations.md](training-operations.md) before launch:

| Criterion | Threshold | Measures |
|---|---|---|
| Controllability | Spearman \|ρ\| ≥ 0.9 per sweep group | requested energy vs produced LUFS loudness |
| Identity leakage | ≤ 0.2 | max ECAPA cosine drift across the sweep, normalized by the real inter-speaker gap |
| Intelligibility guardrail | WER Δ ≤ +0.10 | worst sweep row vs baseline row, faster-whisper base.en |

## Setup

* **Model:** Matcha-TTS + FiLM VAT conditioning in both text encoder and CFM decoder (22.6M
  params), `configs/experiment/derisk_energy.yaml`. One live channel: energy in the arousal slot;
  V = T = 0 in all corpus labels.
* **Corpus:** LibriTTS-R train-clean-100 @ native 24 kHz, 247 speakers, 29,441 train clips.
  Energy label = per-speaker LUFS z-score clamped [−1, 1] @ 2σ (`scripts/derive_vat_corpus.py`).
  **License-clean: CC-BY-4.0 only, no Expresso — the artifact is promotable.**
* **Warm start:** `derisk_energy_init.ckpt` built by `scripts/make_warmstart.py` from the
  upstream multi-speaker VCTK checkpoint (FiLM tensors zero-init → bit-identical synthesis at
  init, verified by `scripts/test_vat_identity.py`).
* **Training:** `sonora_training` on ai-lab-0 (MLflow run `nimble-duck-437`), launched
  2026-07-15. Steady-state ~1.24 s/step (batch 32) after the usual ~2.7 h MIOpen warm-up.
  Gated checkpoint: **epoch 099** (~92k steps, ~35 h wall-clock).
* **Evaluation:** automated convergence watcher (§Stop signal in training-operations.md):
  20 renders per checkpoint (energy ∈ {−1, −0.5, 0, +0.5, +1} × 4 held-out val clips, 4 distinct
  speakers), vocoded with the promoted 24 kHz HiFi-GAN (`g_02510000`), fixed seed so sweep rows
  differ only in the vat vector.

## Result (checkpoint_epoch=099, evaluated 2026-07-16 11:11 UTC)

| Group (speaker/clip) | Energy ρ | Produced range (LUFS) | WER base → worst (Δ) | Leakage |
|---|---|---|---|---|
| 1355_39947 | 1.000 | −30.4 → −24.6 | 0.000 → 0.000 (0.000) | 0.069 |
| 229_130880 | 1.000 | −30.5 → −24.9 | 0.043 → 0.043 (0.000) | 0.058 |
| 2416_152137 | 1.000 | −32.2 → −25.1 | 0.154 → 0.154 (0.000) | 0.091 |
| 2836_5354 | 1.000 | −26.1 → −22.0 | 0.208 → 0.250 (0.042) | 0.051 |

Every group passes every criterion with real margin — this was not a borderline verdict:

* **Controllability:** perfectly monotonic (ρ ≈ 1.0) in all four speakers; the −1→+1 sweep moves
  produced loudness ~4–7 LUFS.
* **Identity:** max leakage 0.091 vs the 0.2 ceiling (inter-speaker gap denominator 0.971) — the
  energy knob is not dragging the voice toward another speaker.
* **Intelligibility:** WER unchanged in 3 of 4 groups; worst Δ 0.042 vs the 0.10 guardrail.
  (Absolute baseline WERs of 0.15/0.21 on two clips are whisper-vs-clip difficulty, present at
  energy 0 too — the criterion is deliberately the *delta*.)
* **Audition (2026-07-16, owner):** energy/loudness change across the sweep judged clearly
  discernible and natural-sounding. The criterion is necessary-not-sufficient; ears confirmed it.

Zero-init baseline for contrast (the watcher's plumbing-test entry): at init the sweep renders
are bit-identical (ρ undefined → FAIL), confirming the gate cannot pass on an inert channel —
after the tie-averaged-ranks fix (Sonora `16c1183`) closed exactly that false-PASS hole.

## What this does and does not prove

**Proven:** the FiLM/VAT plumbing (trunk + per-block FiLM in encoder and decoder) learns a
continuous, monotonic, speaker-preserving, intelligibility-preserving control from weak per-
utterance labels in ~35 GPU-hours on one consumer-class GPU. The architecture question §7 asked
is closed.

**Not proven:** (a) valence/tension behave as well as energy — energy/loudness is the *easiest*
channel (acoustically simple, trivially labelable); V/T are perceptually subtler and their
labeling is the actual open problem. (b) Channel independence — with V=T=0 everywhere, nothing
tested whether channels interfere when driven together. (c) Long-form/expressive behavior —
sweep clips are single utterances ≤ ~8 s.

## Consequences

1. **Full 3-channel VAT is now worth the investment** — and its blocker is a **corpus decision**,
   not architecture: no shippable valence/tension-labeled data exists (Expresso is NC).
2. **Artifacts promoted** to `Sonora/huggingface/derisk-energy-24k/` (checkpoint, eval report, sweep
   WAVs, gate history) — staged locally, public HF push pending a deliberate call.
3. **Export-harness adaptation unblocked** — there is now a 24 kHz/multi-speaker/VAT checkpoint
   worth exporting; `/data/toolchain/litert-conversion/` needs `spks` + `vat` wrapper inputs and
   the 24k vocoder pairing.
4. **The watcher pattern is validated twice** (vocoder + this run) as the standard for unbounded
   fine-tunes, including its "test the gate against a checkpoint that SHOULD fail" rule.

Linked from: [STATE.md](STATE.md) §3 · [training-operations.md](training-operations.md) ·
[vat-conditioning-design.md](vat-conditioning-design.md) ·
[model-size-target-decision.md](model-size-target-decision.md)

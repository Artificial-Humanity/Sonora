# Model decisions — shape, size, rate, decoder

> The settled answers to *what shape is the model*, and only what they still direct.
> Live architecture canon is [ARCHITECTURE.md](ARCHITECTURE.md); training operations are in
> [training-operations.md](../notes/training-operations.md). The decisions here were made between
> 2026-06-14 and 2026-08-02 and the reasoning that produced them is in git history.

## The size ladder — mini / mid / heavy

Tiers are **separate trainings of a shared recipe**, not one model grown. No model family
carries weights across tiers, so "will we retrain for mid" is not a risk — it is the plan.
The only real question is whether the *architecture* would have to be rebuilt, and the
assets that prevent that are deliberately scale-free:

| asset | scale-free? |
|---|---|
| corpus + label pipeline (licence wall, z-score weak labels, EIV/phonation derivation) | ✅ data outlives architectures |
| eval regime (pre-registered gates, holdout scoring, human audit) | ✅ bigger models meet the same or tighter thresholds |
| Director↔Actor contract (V/A/T + delivery, speaker-as-vector, text lane, chunking) | ✅ tiers are implementations of it |
| export/gate discipline (split graphs, GPU-clean rules, parity gates) | ✅ as method; shapes are per-tier parameters |
| vocoder | separable — already its own graph, swaps at the mel interchange |
| backbone + weights | ❌ per-tier, and that is expected |

**Naming: the `-mini` suffix waits until a second tier exists.** Publish the current
lineage unsuffixed.

The DiT spike below is what would collapse mini/mid/heavy onto **one config-scalable
backbone** — mini ≈ 8 layers narrow, mid ≈ 16 wide, same codebase.

### Two standing insurances — do not weaken either

1. **Speaker is a 64-dim VECTOR, never an id.** The 247-row table is a roster, not a
   representation, and a dead end at scale. The export graphs already take a vector; the
   lookup is host-side. A future speaker encoder produces the same vector and nothing
   downstream moves.
2. **V/A/T semantics and units are versioned contract items**, owner-gated. If channels
   mean subtly different things per tier, tiers stop being interchangeable and every
   consumer forks.

## The size target — anchored on hardware, not on a parameter count

**150M is a loose maximum, and the real anchor is mobile capability: iPhone 17+ and
equivalent Android.** Legacy devices are explicitly not targeted. Current trajectory lands
a full-featured end-to-end graph around **40–50M**, well under the ceiling.

**Where to spend headroom, highest leverage first:**

1. ~~Vocoder fidelity~~ — **struck 2026-08-06.** Copy-synthesis measured the 24 kHz
   HiFi-GAN as perceptually transparent, so a wider vocoder buys nothing audible. The gap
   is entirely the acoustic model's predicted mel.
2. **A real speaker encoder (ECAPA-class)** over the 247-row lookup, if voice
   generalization beyond the training roster matters.
3. **Acoustic backbone width — LAST, with a caveat.** Matcha's decoder cost scales with
   ODE step count more than raw width, and a wider decoder on-device (CPU, per the Pixel 8a
   delegate placement) can hurt real-time factor more than it helps quality.

**Revisit the budget when the DiT spike picks a block shape** — that is the decision this
ceiling actually binds.

## Why not Kokoro, StyleTTS2 or a GAN stack — the criteria, which still apply

- **Non-directability is the origin reason for this project.** Static or reference-derived
  style control makes programmatic mood/inflection control impossible. Sonora exists
  because Kokoro — otherwise excellent — could not be *directed*.
- **LSTM architectures are export-hostile** (mobile-delegate CPU fallback); Matcha's
  transformer + flow-matching decoder exports at corr ≈ 1.000000.
- **Heavy PL-BERT-class dependencies cost startup latency** Matcha does not pay — phoneme
  ids go straight in.
- **Reference-derived style priors we do not control** are measurably risky: the teacher
  campaign found excursion-driven voice splits and 45% gender flips on cloning. That is
  what retired the StyleTTS2-Lite re-platform as the quality-ceiling contingency; the
  escape hatch is scaling the flow-matching backbone instead.
- **Standing licence rule: repo licence ≠ weights licence.** Check weights and
  training-data terms separately (Coqui XTTS is MPL-2.0 with CPML non-commercial weights).
- **Make the first success boring** — and do not make an adversarial GAN stack the second
  project either.

## Sample rate — 24 kHz native, 80 bands

Native 24 kHz end to end, no resampling: LibriTTS-R is native 24 kHz and the Rust engine's
`StageAudioSink` is too, so the actor's resample step disappears.

**The mel contract, locked:** `sr 24000 · n_fft 1024 · hop 256 · win 1024 · n_mels 80 ·
f_min 0 · f_max 12000`.

- **`n_feats` stays 80.** It preserves the warm-start shapes, the engine mel contract and
  the proven LiteRT export lane. Vocos (ISTFT head, no standard TFLite lowering) and
  BigVGAN (~110M) were rejected on export-lane risk and budget, not on quality.
- **f_max 12000, not 8000** — above f_max the acoustic model cannot specify content, and
  staying at 8000 wastes the point of 24 kHz. 80 bands over 0–12 kHz is coarser than the
  100-band configs at this rate; accepted for warm-start compatibility.
- **Frame arithmetic: hop 256 @ 24 kHz = 93.75 fps**, so the 512-frame split graph spans
  **~5.46 s**.
- ⚠ **The engine's `config.json` `sample_rate` flips 22050 → 24000 only when a genuine
  24 kHz model ships**, at which point the actor's resample path becomes a no-op.

## Long-form — chunking is mandatory, and the seam is unsolved

**The export lane is fixed-shape: 256 phonemes / 512 mel frames ≈ 5.46 s.** A single call
cannot exceed it regardless of quality, so **Director-side chunking is mandatory** and the
chunk size IS the export ceiling. Chunk on natural sentence/clause breaks.

**Quality is not the binding constraint.** Swept to ~4,100 tokens (≈2 min continuous):
WER 0.00–0.04 at every length, speaking rate flat per speaker, loudness stable across
thirds. A parallel duration predictor has no drift mechanism, so Kokoro's long-passage
failure mode does not carry over — despite training on ≤16 s clips.

⚠ **The cross-chunk SEAM is a separate, open problem.** Bounding chunk length fixes
degradation; it does nothing for continuity across the cut (consistent pacing and prosody,
or a deliberate pause rather than a hard join). The silence-padding pattern from
[high-ambition-2-dramatic-reader.md](../notes/high-ambition-2-dramatic-reader.md) is the precedent
worth reusing.

**To raise the ceiling — two-tier export is the recommended default:** keep 256/512 for
turn-based synthesis and export a second larger graph (~1024 frames / ~450 tokens, ~11 s)
used only for Director-initiated long-form. Additive, same recipe, gates reused.

- **Grow mel frames before phoneme slots.** Mel frames bind first *and* are the cheaper
  dimension: the CFM decoder is a conv U-Net (linear in frames per ODE step) while the text
  encoder has real self-attention (roughly quadratic in phoneme slots). The phoneme
  dimension carries ~25% slack at this corpus's 37.4 tokens/sec.
- **Dynamic TFLite shapes are the fallback, not the default** — fixed shapes are why the
  Pixel 8a delegate placement works, and dynamic ones risk CPU fallback or per-call graph
  rebuild.
- ⚠ **Raising the technical ceiling does not prove the model sounds good out there.** Any
  new ceiling needs its own length-vs-quality sweep, and a corpus capped at 16 s clips may
  need a longer-clip tier to have taught that regime at all.

## Decoder v2 — the staged DiT spike (LIVE)

Make the mel decoder a config-scalable DiT, staged behind a parity gate against the U-Net
baseline. Keeping the ceiling fluid is the point.

| kept (Matcha) | replaced |
|---|---|
| text encoder, duration predictor + MAS, length regulator | mel decoder backbone: conv U-Net → **DiT blocks** (adaLN-Zero) |
| OT-CFM objective, few-step ODE, losses, Lightning harness | decoder weights (encoder/duration may warm-start) |
| vocoder + 24 kHz/80-band interchange, 178-vocab, split-graph export | — |
| Director↔Actor contract v2 | — |

**adaLN-Zero is a fit, not a port.** DiT's native conditioning — per-block zero-initialized
affine modulation from a conditioning vector — is the same mechanism as our zero-init FiLM,
promoted to a first-class citizen. Conditioning vector = concat(V/A/T, delivery embedding,
speaker 64-d).

**Start from StableTTS's 31M `DiTConVBlock`, not a plain DiT** — adaLN-Zero with a fully
convolutional FFN inside every block and U-Net-style long skips, keeping MAS and the
duration predictor. That retires this spike's one named risk (tiny plain transformers lose
the fine spectral texture convs get free) in public MIT code rather than in our run.

**Carry the CFG lever into scope** (conditioning dropout 0.15, adopted 2026-07-16): it was
never written into this spike, StableTTS confirms it survives a DiT decoder, and it buys an
inference-time direction-strength dial. **Consistency-FM (RapFlow-TTS, ~2 NFE) is a 5–10×
throughput lever that changes the objective — it gets its OWN de-risk cycle, not a ride on
this one.**

### The gate — adopt/reject is measured, not judged

Train DiT-mini and hold it to the same gates the U-Net decoder passed, plus export:

- energy channel ρ ≥ 0.9 (U-Net measured ≈ 1.000)
- leakage ≤ 0.2 · WER Δ ≤ +0.10
- export parity through the split-graph path — transformer-at-fixed-shape is the favourable
  case, but decoder graph re-validation is mandatory, not assumed
- the owner's ear on samples: the instrument gates cannot hear texture

⚠ **The parity gate must run against a U-Net baseline trained on the SAME expanded
corpus**, frozen as the last act of Phase 1 — otherwise data and architecture confound each
other and the run teaches nothing about either. Sequencing lives in
[quality-gap-plan.md § Phase 2](../notes/quality-gap-plan.md).

**Pass → decoder v2 is the family backbone. Stall → the stock decoder runs the corpus** and
the spike parks with its findings. The schedule never waits on the swap.

## The base model — Matcha, reaffirmed

**Matcha-TTS (MIT).** It won on the axes that carry first-run risk: single-stage
non-adversarial training, an export-clean transformer/conv architecture, an official ONNX
exporter matching the validated backbone, and speaker-embedding conditioning the FiLM/VAT
layer extends additively.

**Reaffirmed 2026-07-29** after directing seven teacher engines, on that campaign's central
lesson: *an engine's usable range is set by its conditioning interface, not its raw
quality.* That argues for a base whose conditioning surface we define. The differentiating
work — VAT directability, casting, export discipline — was never the base model, and
transfers to any tier of the family.

# High-Ambition 1 — 🎙️ Matcha-TTS Directable Production Actor

> **Sequence:** 1 of 5 high-ambition goals — the **first actor model we actually ship** and the
> foundation the others build on. Next: [2 — Dramatic Reader](high-ambition-2-dramatic-reader.md) ·
> [3 — Child Voices](../Prosodia/high-ambition-3-child-voices.md) ·
> [4 — Multilingual G2P](../Prosodia/high-ambition-4-multilingual-g2p.md) ·
> [5 — StyleTTS2-Lite](high-ambition-5-styletts2-lite.md) (the higher-ceiling re-platform, if Matcha
> quality falls short). Model decision + training/hardware detail:
> [actor-model-and-training.md](actor-model-and-training.md).

This note is the production design for a **Matcha-TTS-based** Prosodia actor: a small, fast,
**directable** and **castable** on-device TTS model. Matcha is the chosen *first* base because it
trains in a single stable stage (no GAN), ships an official ONNX exporter that matches our validated
`torch → ONNX` backbone, and uses export-friendly transformer/conv ops (no LSTM CPU fallback on
mobile delegates). The hard, novel work is **not** the base model — it is the **directability and
casting layer we add on top**, which transfers to StyleTTS2-Lite later if we graduate to it.

---

## 🎯 Objective

Ship a ~20–40 MB on-device actor that:
1. Synthesizes natural 24 kHz mono speech in real time on iOS / Android / desktop.
2. Is **directable** — renders Valence/Arousal/Tension (VAT) emotion codes from the Gemma Director
   as audible prosody (pitch, energy, rate).
3. Is **castable** — exposes the same continuous voice space (age / masculinity / strain) the
   existing `voice_loader` + casting grid drive, so the Director's casting payload still works.
4. Exports cleanly to the on-device runtime via `torch → ONNX → TFLite` (or ONNX Runtime).

---

## 🏛️ Pipeline (Matcha base + Prosodia additions)

```
            phonemes (Rust G2P)              VAT [V,A,T]        casting payload
                  │                              │            (age/masc/strain)
                  ▼                              ▼                   │
        ┌───────────────────┐                    │                  ▼
        │ Transformer Text  │◀──── FiLM/AdaLN ────┤        ┌───────────────────┐
        │     Encoder       │     (our addition)  │        │  Casting grid →   │
        └─────────┬─────────┘                     │        │ speaker embedding │
                  ▼                                │        └─────────┬─────────┘
        ┌───────────────────┐                     │                  │
        │ Duration Predictor│   (monotonic align) │                  │
        └─────────┬─────────┘                     ▼                  ▼
                  ▼              ┌──────────────────────────────────────────┐
        ┌───────────────────┐   │  Flow-Matching Mel Decoder (few-step ODE) │
        │  length regulate  │──▶│  conditioned on speaker emb + VAT (AdaLN) │
        └───────────────────┘   └─────────────────────┬────────────────────┘
                                                       ▼
                                        ┌───────────────────────────┐
                                        │ HiFi-GAN vocoder (pretr.) │ → 24 kHz PCM
                                        └───────────────────────────┘
```

Vanilla Matcha = transformer text encoder → duration predictor (monotonic alignment) → flow-matching
mel decoder (OT-CFM, default 5 ODE steps) → separate pretrained HiFi-GAN vocoder. Speaker identity in
vanilla Matcha enters as a **speaker embedding** added to the encoder/decoder. **Everything outside
the dashed conditioning paths is stock Matcha; the additions below are ours.**

---

## 🧠 The three things we build (the actual high-ambition work)

### 1. VAT directability conditioning
* **Problem:** stock Matcha has no emotion control — speaker embedding sets *who*, not *how*.
* **Approach:** condition the **text encoder** and the **flow-matching decoder** on the Director's
  `[V, A, T]` vector via **FiLM / AdaLN** (feature-wise affine modulation), the same mechanism Matcha
  already uses for the timestep/speaker conditioning — so it's an additive change, not surgery.
* **Training signal:** pair training utterances with VAT labels (derive from prosodic features —
  pitch range, energy, rate — and/or LLM annotation; the Strix Halo can host a big labeler locally).
  This is the same VAT contract the [Tuner calibration](../Prosodia/voicing-synthesis-and-tuning.md) and Director
  prompts already speak.
* **Done looks like:** audibly distinct delivery across VAD slider settings in the Tuner.

### 2. Casting / blend layer (re-derive in speaker-embedding space)
* **Key difference from StyleTTS2:** StyleTTS2 gives a disentangled 64/128-d **style vector** that our
  `voice_loader` interpolates directly. Matcha conditions on a **speaker embedding** instead. The
  *machinery ports unchanged* — the bilinear age/masculinity grid and the gruff/raspiness texture
  blend in [voicing-synthesis-and-tuning.md](../Prosodia/voicing-synthesis-and-tuning.md) operate on whatever
  vector conditions the decoder; only the **vector's semantics** change (speaker embedding vs style
  vector).
* **Work:** define the anchor embeddings (train a small speaker-embedding LUT, or extract embeddings
  for anchor voices), then point `voice_loader`'s interpolation at that space. Validate that
  interpolation stays on-manifold (Matcha speaker embeddings may be less linearly interpolable than
  StyleTTS2 style vectors — A/B the midpoints).
* **Done looks like:** the existing casting payload (`age_profile`, `masculinity`, `strain_or_rasp`)
  produces a smooth, identity-stable continuous voice space.

### 3. On-device export
* **Path:** official exporter `python -m matcha.onnx.export` (can embed the HiFi-GAN vocoder) →
  **`onnx2tf` → TFLite** (keeps the LiteRT runtime + Rust `tflite.rs` actor) **or** ship ONNX via
  ONNX Runtime. The `onnx2tf` leg is the one still-unproven step — spike it early (see
  [next-steps.md §1](../Prosodia/next-steps.md)). *(Update 2026-07-12: both legs since proven, and
  the priority reversed — **Plan A is now the `litert-torch` fixed-shape split-graph export**
  (textenc/decoder/vocoder + host ODE, verified at parity on Epoch 199); the `onnx2tf` monolith is
  the Plan B fallback. ONNX Runtime was never adopted. See the 📌 callout in
  [next-steps.md](../Prosodia/next-steps.md).)*
* **I/O contract:** must match `crates/actor/src/engine.rs::forward_impl` — phonemes `[1,N] i32`,
  speaker/style `f32`, speed scalar `f32`, optional VAT `f32[3]`, output PCM `f32` mono 24 kHz. The
  contract is documented in [next-steps.md](../Prosodia/next-steps.md) and is identical whether the base is Matcha
  or StyleTTS2 — so the Rust actor doesn't care which model produced the graph.

---

## 🧪 Phase 0 — Untuned Matcha discovery spike (**before any training**)

Run **stock, pretrained Matcha** end to end *before* training your own. It's cheap, low-risk, and —
crucially — **its outputs are training inputs**: it forces you to lock the contracts you must train
*against* and de-risks the export runtime before you've spent anything on a model. This is a
de-risking spike, **not a prerequisite** — training isn't sequentially blocked on the app running;
this is just the cheapest forcing function to settle the decisions below.

**What runs untrained vs what doesn't:**

| Layer | Stock pretrained Matcha | Needs training |
|---|---|---|
| Base TTS (text → natural single voice) | ✅ out of the box | — |
| Prosodia integration (G2P → actor FFI → audio sink → coordinator/Tuner) | ⚠️ after adapting the I/O contract | — |
| Directability (VAT → prosody) | ❌ nowhere for VAT to go | ✅ FiLM/AdaLN conditioning |
| Continuous casting (age/masculinity/strain) | ❌ 1 speaker (LJSpeech) / discrete IDs (VCTK) | ✅ speaker-embedding re-derivation |

**Spike steps:**
1. **Stock Matcha standalone on the M1 Max** — synthesize your own sample passages, A/B against
   StyleTTS2 / Kokoro. *Hours.* Confirms the quality bar clears *before* spending a dollar (if it
   disappoints on expressiveness, that's the real signal to weight
   [5 — StyleTTS2-Lite](high-ambition-5-styletts2-lite.md) sooner).
2. **Export spike** — stock checkpoint → official ONNX export → `onnx2tf` → `.tflite`, run it from the
   Rust actor. *A few days.* This is the unproven leg, and it's where the contract mismatches below
   surface.
3. **Minimal end-to-end through the Tuner** with **neutral** synthesis — validates the Director →
   payload → FFI → audio-sink seam. The existing **DSP-level** knobs (speed, `f0_bias`, pause /
   expressiveness calibration) still shape output even though the stock model is emotion-blind, so the
   plumbing is exercised without learned directability.

> ⚠️ **Don't over-polish the stock integration** — the input contract *changes* once VAT conditioning
> is added (a new tensor + retrain), so scope steps 2–3 as a throwaway-friendly spike, not production
> wiring.

### 🔒 Contract-lock checklist (must be settled here — you train against these)

> **Scope note (reconciled 2026-06-18):** these are the *training-time* contracts the model commits
> to. Their *discovery-spike / runtime-bridge* counterparts — the `map_styletts2_to_matcha_ipa` remap,
> the 22.05→24 kHz runtime resample, the `[V,A,T]` payload representation, and the TFLite/`onnx2tf`
> runtime — are already locked and checked in [next-steps.md](../Prosodia/next-steps.md). They are *not* the same
> items: the spike proves the **stock** model runs through the actor; the rows below are what must be
> fixed **before training**. Three of the four rows below are now resolved (vocab + sample rate via
> commit `7143617`, 2026-06-18; export/runtime via the `onnx2tf` spike) — only **data + label format**
> remains genuinely open.

- [x] **G2P / phoneme contract.** ✅ **Locked (Gate 1, commit `7143617`, 2026-06-18):** a
  Matcha-specific vocab is defined — `config.json` + `symbols.py` deduped to exactly **178 unique
  symbols** (removed the duplicate `'`, added the modifier-letter schwa `ᵊ`), and an `is_matcha_ipa`
  flag lets direct Matcha models bypass `map_styletts2_to_matcha_ipa` at runtime. The model must still
  be *trained* against this vocab, but the contract (G2P ↔ training ↔ `config.json` lockstep) is
  settled — keep them in lockstep if the symbol set ever changes (see [next-steps.md](../Prosodia/next-steps.md)).
- [x] **Sample rate.** ✅ **Decided (Gate 2, commit `7143617`, 2026-06-18):** native **24 kHz** — the
  preferred path that keeps the `StageAudioSink` / coordinator contract. `config.json` declares
  `"sample_rate": 24000`, and `get_model_sample_rate` in `engine.rs` reads the native rate and bypasses
  resampling when the model already outputs 24 kHz (stock 22.05 kHz checkpoints still resample).
- [ ] **Data + label format.** Lock the filelist schema and **how VAT labels are represented** before
  collecting/labeling, so the directability fine-tune (Phase 3) has consistent supervision.
- [x] **Export/runtime decision.** ✅ **Resolved (2026-06-17):** keep **TFLite via `onnx2tf`**. The
  custom conversion was validated on `model_e2e.onnx` and the FFI contract locked — ONNX Runtime is not
  needed (see [STATE.md](../Prosodia/STATE.md) and [next-steps.md](../Prosodia/next-steps.md)).

---

## 🔄 Phased training plan

Mirror the de-risking order in [actor-model-and-training.md §3](actor-model-and-training.md) — make
the *first* success boring, then add novelty. (Phase 0 above runs first.)

1. **Plain fine-tune** a pretrained Matcha checkpoint on a small clean single-speaker set (LibriTTS
   speaker or LJSpeech) on a RunPod 4090 — purely to learn the loop and confirm quality clears your
   bar by ear. (~hours, a few dollars.)
2. **Export spike** — official ONNX export → `onnx2tf` → a real on-device `.tflite`; confirm the Rust
   actor speaks it. This decides keep-TFLite vs ONNX-Runtime before more model work.
3. **Add VAT conditioning** (FiLM/AdaLN) and retrain with VAT labels; verify directability in the Tuner.
4. **Add the casting layer** — anchor embeddings + `voice_loader` interpolation in speaker-embedding
   space; verify the casting grid.
5. **Multi-speaker / expressive data** for range, then iterate on quality.

> **Data licensing stays permissive:** LibriTTS (CC-BY-4.0), LJSpeech (public domain). G2P uses our
> Rust permissive lexicon, **not** espeak-ng, in the shipping path.

---

## 🧭 Relationship to the rest of the sequence

* **[2 — Dramatic Reader](high-ambition-2-dramatic-reader.md)** (full-cast) builds directly on this
  actor's directability + casting. Its multi-voice design was first written against StyleTTS2's style
  space; the Matcha analog is multiple speaker embeddings + the conditioning above.
* **[3 — Child Voices](../Prosodia/high-ambition-3-child-voices.md)** extends the casting range; its DSP and
  fine-tune options are base-agnostic, its embedding-blend/extract options map onto the
  speaker-embedding space.
* **[5 — StyleTTS2-Lite](high-ambition-5-styletts2-lite.md)** is the **optional quality-ceiling
  re-platform**, attempted only if Matcha's naturalness/expressiveness falls short (the dramatic
  axis is where StyleTTS2's higher ceiling shows most). **Crucially, the work here transfers:** the
  VAT directability design, the casting machinery, the data pipeline, and the `torch→ONNX→TFLite`
  export are all reused — so Matcha-first is not a throwaway even if we later graduate.

**What transfers vs what's Matcha-specific**

| Asset | Transfers to StyleTTS2-Lite? |
|---|---|
| VAT/FiLM directability design | ✅ same conditioning idea (StyleTTS2 = its style predictor) |
| Casting grid + `voice_loader` interpolation | ✅ (vector semantics differ; machinery same) |
| Data prep + VAT labeling pipeline | ✅ |
| `torch → ONNX → TFLite` export + Rust I/O contract | ✅ |
| Flow-matching decoder / OT-CFM specifics | ❌ Matcha-only (StyleTTS2 uses diffusion/GAN) |
| Speaker-embedding conditioning point | ⚠️ re-targets to StyleTTS2's style vector |

# Model decisions — shape, size, rate, decoder

> **Consolidated 2026-07-31** from `model-family-strategy.md` + `model-size-target-decision.md`
> + `sample-rate-24khz-decision.md` + `decoder-v2-dit-spike.md`; **§5 (base model) folded in
> 2026-08-02** from the retired `actor-model-and-training.md`. Closed briefs that all answer
> one question — *what shape is the model* — kept largely verbatim because each records an
> owner decision and the reasoning that produced it.
>
> Live architecture canon is [ARCHITECTURE.md](ARCHITECTURE.md); training operations are in
> [training-operations.md](training-operations.md).

## Contents

1. **The size ladder — mini / mid / heavy** — from `model-family-strategy.md`
2. **The size target — mobile-anchored, 150M loose max** — from `model-size-target-decision.md`
3. **Sample rate — 24 kHz native** — from `sample-rate-24khz-decision.md`
4. **Decoder v2 — staged DiT spike** — from `decoder-v2-dit-spike.md`
5. **The base model — Matcha, reaffirmed** — from `actor-model-and-training.md`

---

# The size ladder — mini / mid / heavy

_Source: `model-family-strategy.md`_

_Owner intent: a size ladder like other model families — **sonora-mini** (the current 150M-ceiling
on-device commitment), **sonora-mid**, **sonora-heavy** — chosen per target hardware, without the
current work training us into a corner where a bigger tier means starting over. The pinned answer
lives in [ARCHITECTURE.md](ARCHITECTURE.md) (canon); this note is the rationale._

## The framing

No model family carries weights across tiers — Gemma 12B is not a grown 2B; they are separate
trainings of a shared recipe. So the corner-risk is never "will we retrain for mid" (we will);
it's "would the *architecture* have to be rebuilt." The durable assets are deliberately
scale-free:

| Asset | Scale-free? | Why |
|---|---|---|
| Corpus + label pipeline (license wall, z-score weak labels, EIV/phonation derivation) | ✅ | Data outlives architectures; re-running the automated pipeline over more hours IS the data scaling story. |
| Eval regime (§7 pre-registered gates, watchers, human-audit workflow) | ✅ | Bigger models are held to the same or tighter thresholds. |
| Director↔Actor contract (V/A/T semantics, speaker-as-vector, text lane, chunking) | ✅ | Pinned in ARCHITECTURE.md §1; tiers are implementations of it. |
| Export/gate discipline (split graphs, GPU-clean rules, parity gates) | ✅ method | Fixed shapes/budgets are per-tier parameters of the same method. |
| Vocoder | per-tier, separable | Already a separate graph; swaps independently at the mel interchange. |
| Backbone + weights | ❌ per-tier | Expected. Matcha's encoder/decoder plausibly stretches to a ~100–300M mid (trainable on ai-lab-0 with patience); a heavy tier likely wants a DiT-style flow-matching backbone — still the same contract, mel interchange, and gates. **Ratified 2026-07-29:** this scale-up path is now also the *named quality-ceiling contingency* for mini — the StyleTTS2-Lite re-platform is retired (rationale in §5 below; the full high-ambition-5 design is in git history). **Advanced 2026-07-30 (owner: keep the ceiling fluid):** the DiT decoder moves from contingency to *staged spike NOW* — mini/mid/heavy collapse onto one config-scalable backbone if it passes the parity gate vs the U-Net baseline; plan in **§ Decoder v2** (below). |

## The two real corner-risks and their insurance (both cheap, both taken)

1. **Speaker representation.** The 247-slot embedding table is a roster, not a representation —
   a dead end at scale. Insurance: the contract pins **speaker = 64-dim vector, never an id**
   (the export graphs already take a vector; the table lookup is host-side). A future speaker
   encoder (zero-shot voices, mid-tier territory) produces the same vector and nothing
   downstream moves.
2. **Semantic drift between tiers.** If V/A/T mean subtly different things per tier's corpus,
   tiers stop being interchangeable and every consumer forks. Insurance: channel semantics and
   units are contract items (versioned, owner-gated changes), pinned before `derive_vat_corpus.py
   v1` gets written.

## Consequences already adopted

* Corpus accumulation is a **standing activity** (Emilia-YODAS tail mining moved in-scope for the
  first 3-channel run) — tier ambitions raise the stakes on hours, and the label extremes are
  what deeper emotional conveyance is bounded by.
* The **expressive fine-tune** stays on the radar as a *stage* any tier can receive (nonverbal
  vocal events, micro-prosody beyond the 3-dim control space), not a fork of the family.
* Compute reality: mini and plausibly mid train on ai-lab-0; heavy (≥1B-class) implies rented
  compute — a budget decision for its day, not an architectural one.
* Naming: keep publishing the current lineage unsuffixed; the `-mini` suffix gets applied when a
  second tier actually exists (owner: "we don't need to add -mini yet").

Linked from: [ARCHITECTURE.md](ARCHITECTURE.md) ·
**§ The size target** (below) · [STATE.md](STATE.md) §3.

---

# The size target — mobile-anchored, 150M loose max

_Source: `model-size-target-decision.md`_

_Refined 2026-07-23 (owner): **150M stands as the maximum — a loose limit** ("I find it hard
to believe we'd push MatchaTTS to that size, in any case"). The real anchor is **mobile
hardware capability, not a parameter number: baseline = iPhone 17+ and equivalent Android** —
legacy devices are explicitly not targeted. (Clarifies in passing: 18M is just what Phase-0
happened to weigh, never a target.)_

_Owner call, informal (not yet load-bearing on a config): the acoustic model + vocoder together
should aim for **Kokoro+** quality/capability at a **150M-parameter ceiling** (not a target to
spend in full). Explicitly **not** driven by Kokoro's own footprint — Kokoro is ~82M and that was
never the problem. Mobile pairing target: alongside a Gemma-class on-device Director (STATE.md
§3 calls it "Gemma 4 Director"; the actor-model doc's measured comfort case is Gemma E2B
≈1.5–3 GB quantized + TTS actor ≈0.1–0.3 GB on 32 GB dev hardware — §5's measured comfort case)._

## Why Kokoro was left (the actual shortcomings — none are about size)

1. **Non-directability — the origin reason.** "Sonora exists because Kokoro — otherwise
   fantastic — could not be *directed*" (high-ambition-6-audience-conveyance-stt.md:26). Static/
   implicit style control (target reference audio or averaged style matrices) makes programmatic
   mood/inflection control impossible (high-ambition-5-styletts2-lite.md:122).
2. **LSTM architecture** — export/mobile-delegate hostile (§5),
   unlike Matcha's transformer + flow-matching decoder, already proven to export at
   corr ≈ 1.000000 (litert-conversion).
3. **Heavy PL-BERT dependency** — startup latency cost Matcha doesn't have (no BERT layer;
   phoneme IDs go straight in).
4. **Choppy long-form flow, confirmed 2026-07-15.** Previously documented as "BERT-based TTS
   tends to synthesize in isolated sentence blocks, yielding a choppy, disconnected flow"
   (high-ambition-5-styletts2-lite.md:113,119). Added detail from this conversation: Kokoro's
   quality **degrades over long passages** — not just choppy joins between sentence-level
   renders, but drift/quality loss within continuous long-form reading.
5. **82M size** was cited only as an aspirational precedent to replicate (LiteRT-compiled
   footprint worth matching), never a shortcoming (high-ambition-5-styletts2-lite.md:168).

## Where the size budget likely goes

Current trajectory, from measured checkpoints: Phase 0 (single-speaker, no directability)
18.2M → derisk_energy (multi-speaker + one VAT channel) 22.6M. FiLM/VAT conditioning is cheap
(~4.4M for spk_emb + one channel's trunk+heads) against a backbone that dominates the count
(encoder 10.6M + decoder 12M) — full 3-channel VAT is unlikely to push far past ~25–28M. Add the
24 kHz HiFi-GAN vocoder (V1 config, ~14M generator params). **A full-featured e2e graph on the
current architecture lands around 40–50M** — well under the 150M ceiling.

Priority for spending the remaining headroom (highest leverage first):

1. **Vocoder fidelity.** Perceived audio quality is disproportionately vocoder-driven. Something
   between HiFi-GAN V1 (~14M) and the rejected BigVGAN (~110M, rejected for TFLite export-lane
   risk, not raw size — see sample-rate-24khz-decision.md) — a wider HiFi-GAN config, still in
   the proven-exportable family.
2. **A real speaker encoder (ECAPA-class) over the current 247-row lookup table**, if voice
   generalization/cloning beyond the training roster matters — also a direct capability edge over
   Kokoro's static style-matrix approach.
3. **Acoustic backbone width, last, with a caveat.** Matcha's decoder cost scales with
   reverse-ODE step count more than raw width; a wider decoder run on-device (CPU, per the
   Pixel 8a delegate-placement finding — decoder placed on CPU due to a Mali transformer-fusion
   bug) can hurt real-time factor more than it helps quality. Don't spend headroom here first.

## Open risk: long-passage handling — solve via Director-side chunking, not model capacity

Not a parameter-budget problem, and not even purely a data problem: the Director can chunk text
before it ever reaches the model, so long-form reading doesn't require the acoustic model to
generalize past whatever chunk length is safe. This also isn't a free design choice — the mobile
export lane already forces a ceiling:

* **Hard technical ceiling (already fixed, not tunable):** the `litert-split/` lane — the one
  planned for the actor — is a **fixed-shape** graph: 256 phonemes / 512 mel frames per inference
  call (Sonora/huggingface/README.md item 5). At 24 kHz (hop 256, 93.75 fps) that's **~5.46 s**,
  roughly one sentence. A single call cannot exceed this regardless of what quality would
  tolerate — chunking is mandatory on this export lane independent of any quality finding.
* **ANSWERED (2026-07-16, `scripts/chunk_size_sweep.py`):** quality holds far past the window.
  Swept one fresh prose passage at cumulative prefixes of ~300 → ~4,100 tokens (≈8 s → ≈2 min of
  continuous speech), 3 speakers, torch pipeline (unclipped): WER stays 0.00–0.04 at every
  length (0.02–0.035 at the ~2 min mark), speaking rate flat per speaker across all buckets,
  loudness stable across each render's thirds. No inflection point exists in the tested range —
  the parallel duration-predictor architecture has no drift mechanism, so Kokoro's long-passage
  failure mode simply doesn't carry over, despite utterance-level (≤16 s) training clips.
  **Consequence: the Director's chunk size IS the export ceiling** (256 tokens on the current
  single-tier lane; the two-tier option below buys longer), pending only the human audit of the
  sweep WAVs (`/data/model-training/sonora/chunk_size_sweep/derisk_epoch099`, report.json
  alongside) for prosodic quality WER can't see.
* **A separate problem chunking alone doesn't solve: the seam between chunks.** Kokoro's
  documented shortcomings include *both* long-passage degradation *and* "choppy, disconnected
  flow" from isolated sentence-block synthesis (high-ambition-5-styletts2-lite.md:113,119) — two
  different failure modes. Bounding chunk length fixes the first; cross-chunk continuity
  (consistent pacing/prosody across the cut, maybe a deliberate pause rather than a hard join) is
  a separate design question. There's already a precedent for the pattern needed here — silence
  padding between voice transitions — in the dramatic-reader note
  (high-ambition-2-dramatic-reader.md:118); worth reusing for single-narrator long-form chunk
  boundaries too, chunked on natural sentence/clause breaks within the 256-phoneme budget.

## Extending the 5.46 s ceiling (options, 2026-07-15)

Measured against the actual corpus rather than assumed: sampled 60 training clips from
`data/libritts_r_vat/train_op.txt`, paired blank-interspersed token counts against real audio
duration. At this corpus's average rate (**37.4 tokens/sec**), the 256-token phoneme budget would
allow ~6.84 s of speech — **but the 512-frame mel budget caps out first, at 5.46 s.** The mel-frame
dimension is the true bottleneck today, with the phoneme dimension carrying ~25% slack. That
matters for which lever to pull, since the two dimensions aren't equally expensive to grow: the
text encoder has real self-attention (`encoder.encoder.attn_layers`), so its cost scales roughly
quadratically with phoneme-slot count; the CFM decoder is a conv U-Net (`down_blocks`/`mid_blocks`/
`up_blocks`, no attention), so its cost scales roughly linearly with mel-frame count per ODE step.
Mel frames being both the binding constraint *and* the cheaper dimension to grow is a favorable
combination.

Options, ranked:

1. **Bump the fixed shapes and re-export (simplest, lowest risk).** The `litert-torch` recipe
   (`build_matcha.py`/`convert_final.py` in `/data/toolchain/litert-conversion/`) already produces
   three independent graphs (textenc/decoder/vocoder), each with its own fixed shape — retuning is
   a parameter change to a proven pipeline, not a redesign. Given the measurement above, prioritize
   mel frames (e.g. 1024 frames ≈ 10.9 s) over phoneme slots (modest bump, e.g. 384–448, to hold
   the ~25% margin rather than doubling both blindly). Cost: every call, including short
   utterances, pays the bigger fixed shape's compute — the runtime mask zeroes padding but doesn't
   skip work.
2. **Two-tier export (best ROI, recommended default).** Keep 256/512 as the default for normal
   turn-based synthesis (fast, cheap, good RTF); export a second, larger fixed-shape graph (e.g.
   ~1024 frames / ~450 tokens, ~11 s) used only for Director-initiated long-form narration. Additive
   — no regression to the common case, same recipe and parity/ASR gates reused.
3. **Dynamic/signature-ranged TFLite shapes instead of fixed (higher risk, not recommended by
   default).** Removes the ceiling entirely, but fixed shapes are specifically why the Pixel 8a
   delegate placement works (decoder CPU, textenc+vocoder GPU); dynamic shapes risk forcing CPU
   fallback or per-call graph rebuild — reopening the same export-lane-risk axis that already
   rejected Vocos (ISTFT lowering) and BigVGAN. Fallback only if 1/2 prove insufficient.
4. **Native cross-window continuity (architecturally deeper, not needed yet).** Carry
   decoder/duration state across successive fixed-shape calls so chunks stitch seamlessly instead
   of Director-level chunking + pauses. Solves the ceiling and the cross-chunk seam problem
   together, but it's a real architecture change, not a re-export. Reach for this only if
   pause-based chunking sounds bad in practice.

**Caveat that travels with all four:** raising the technical ceiling doesn't prove the model
sounds good out that far — it only makes a longer single call *possible*. The length-vs-quality
sweep (above) needs to target whichever new ceiling gets picked, not the old 5.46 s number, and if
quality needs to hold meaningfully past 5.46 s, the training corpus (capped at 16 s clips) may need
a longer-clip tier to have actually taught the model that regime.

## Status

Ballpark, not yet wired into any config or eval-harness threshold — still true as of 2026-08-06.

The revisit trigger this section named (the derisk_energy §7 verdict + export-harness adaptation)
**both landed on 2026-07-16**, and the budget was not re-opened then. Two later measurements
change what the revisit should conclude:

- **Vocoder fidelity is no longer the top place to spend headroom.** This section ranks it #1 on
  the reasoning that perceived quality is disproportionately vocoder-driven. Copy-synthesis on
  2026-08-06 measured the current 24 kHz HiFi-GAN as **perceptually transparent** (mel L1 = 10.2%
  of mel_std; owner: "similar to ref, if not identical"), so a wider vocoder buys nothing audible
  today. The gap is entirely the acoustic model's predicted mel.
- **Which leaves data and decoder**, which is exactly the order
  [quality-gap-plan.md](quality-gap-plan.md) sequences. Revisit the parameter budget when the
  DiT spike picks a block shape — that is the decision this ceiling actually binds.

Linked from: **§ Sample rate** (below) (vocoder size
precedent), **§5** (Kokoro license/architecture table; the `high-ambition-5-styletts2-lite.md:N`
citations above reference the retired design, now in git history) and
[high-ambition-6-audience-conveyance-stt.md](high-ambition-6-audience-conveyance-stt.md) (Kokoro
shortcomings), [STATE.md](STATE.md) §3 (VAT goal, Gemma 4 Director pairing).

---

# Sample rate — 24 kHz native

_Source: `sample-rate-24khz-decision.md`_

_Owner call: the milestone-3 corpus and all subsequent training run at **native 24 kHz** — no
resampling of LibriTTS-R — honoring the original Gate 2 decision (2026-06-18) that Phase 0
pragmatically deviated from (LJSpeech @ 22.05 kHz + `hifigan_T2_v1`). Rationale: resampling has
its own risks/artifacts; LibriTTS-R is native 24 kHz; and the Rust engine's `StageAudioSink` is
native 24 kHz, so the actor's current 22.05→24 resample step disappears — model samples reach
the sink untouched._

## Vocoder: fine-tune HiFi-GAN to 24 kHz / 80-band (owner call, same conversation)

Chosen over Vocos-24k and BigVGAN-v2-24k (both pretrained, both 100-band): keeping **n_feats =
80** preserves the Phase-0 warm start (encoder/decoder shapes unchanged), the engine mel
contract, and the proven litert export lane — Vocos additionally has an ISTFT head with no
standard TFLite lowering (export-lane risk), and BigVGAN (~110M params) is outside the
on-device budget. Cost accepted: a HiFi-GAN fine-tune (from the universal checkpoint) on
LibriTTS-R, days of hands-off wall-clock on the Strix Halo.

## Consequences / task list

1. **Vocoder fine-tune is now on the critical path to the §7 de-risk verdict** — the eval
   harness needs rendered audio. Start it early; it's hands-off compute that runs while the
   corpus pipeline is built.
2. **Verify the vocoder standalone before the de-risk run** (copy-synthesis: ground-truth mel →
   vocoder → parity/ASR vs the recording). Two novelties are entering at once (24 kHz +
   conditioning) against the "make the first success boring" rule — this check keeps a bad
   vocoder from confounding the conditioning verdict.
3. **Mel spec to lock when writing the vocoder config:** sr 24000, n_fft 1024, hop 256,
   win 1024, n_mels 80, f_min 0 — and **f_max open**: 8000 maximizes transfer from the
   universal checkpoint; **12000 (lean)** actually uses the new bandwidth (above f_max the
   acoustic model can't specify content — staying at 8000 wastes part of the point of 24 kHz).
   Decide with a quick A/B at vocoder-config time. 80 bands over 0–12 kHz is coarser than the
   100-band configs Vocos/BigVGAN use at 24 kHz; accepted to keep warm-start compatibility.
4. **Frame arithmetic shifts:** hop 256 @ 24 kHz = 93.75 fps (was 86.13); the 512-frame fixed
   split graph now spans ~5.46 s (was ~5.94 s). Chunking budgets in the runtime notes stay
   valid; re-derive exact numbers at export time.
5. **Corpus stats:** mel mean/std recomputed for the 24 kHz multi-speaker corpus (Phase 0's
   −5.54/2.12 are LJSpeech@22.05).
6. **Engine `config.json` `sample_rate` flips 22050 → 24000 only when a genuine 24 kHz model
   ships** (the standing STATE warning) — and the actor's resample path becomes a no-op for it.
7. Warm start across the rate change is approximate for the decoder (mel content shifts) but
   full for the text encoder/durations; expected and acceptable for a fine-tune.

Linked from: [next-steps §A Gate 2](../../../Prosodia/notes/next-steps.md),
[vat-channels.md](vat-channels.md) (sequencing note),
[dataset-landscape.md](dataset-landscape.md).

---

# Decoder v2 — staged DiT spike

_Source: `decoder-v2-dit-spike.md`_

_Owner decision 2026-07-30: keep the ceiling fluid by making the mel decoder a
config-scalable DiT **now**, staged behind a parity gate against the U-Net baseline.
Supersedes "DiT only if/when mini disappoints" from
**§ The size ladder** (below); this is the concrete plan._

## What changes / what doesn't

| kept (Matcha) | replaced |
|---|---|
| text encoder, duration predictor + MAS, length regulator | mel decoder backbone: U-Net-style conv blocks → **DiT blocks** (transformer, adaLN-Zero) |
| OT-CFM objective, few-step ODE, losses, Lightning harness | decoder weights (no warm-start; encoder/duration may warm-start) |
| vocoder + 24 kHz/80-band mel interchange, 178-vocab, split-graph export pattern | — |
| Director↔Actor contract v2 (V/A/T + delivery + 64-d speaker) | — |

Ceiling fluidity is the point: mini ≈ 8 layers × narrow (~20M) and mid ≈ 16 × wide
(~150–300M) become **configs of one codebase**, not an architecture migration. The
family strategy's mini/mid/heavy tiers collapse onto one backbone.

## Why adaLN-Zero is a fit, not a port

DiT's native conditioning — per-block zero-initialized affine modulation from a
conditioning vector — is the same mechanism as our zero-init FiLM, as the architecture's
first-class citizen. Conditioning vector = concat(V/A/T, delivery embedding, speaker
64-d). Contract v2's delivery channel already obligates a §7 de-risk cycle; the decoder
swap rides the same cycle — two changes, one de-risk.

## The gate (adopt/reject is measured, not judged)

Train DiT-mini on the existing de-risk corpus (derisk-energy-24k lineage) and hold it to
the SAME gates the U-Net decoder already passed, plus export:

- energy channel ρ ≥ 0.9 (U-Net measured ≈ 1.000)
- leakage ≤ 0.2 · WER Δ ≤ +0.10
- export parity through the split-graph path (litert-torch Plan A; ONNX fallback) —
  transformer-at-fixed-shape is the favorable case, but decoder graph re-validation is
  mandatory, not assumed
- owner's ear on samples (the instrument gates cannot hear texture)

**Pass → decoder v2 is the family backbone** for the direction-taking run on the
certified corpus. **Stall → stock decoder runs the corpus** and the spike parks with its
findings; the schedule never waits on the swap.

## Known risk

Tiny-scale DiT is less proven than tiny-scale U-Nets on fine spectral texture (~20M
plain transformers can lose local detail that convs get free). Mitigations if the gate
or the ear catches it: local/windowed attention, or a shallow conv stem in the patch
embedding. Reference implementations: F5-TTS (flow matching on DiT, publicly available)
— consult for block shape, do not import its no-alignment training style; we keep MAS.

**Answered 2026-08-01 by [matcha-siblings-study.md](matcha-siblings-study.md).** This
risk is already retired in public MIT code: StableTTS runs a 31M `DiTConVBlock` —
adaLN-Zero, **fully convolutional FFN inside every block**, U-Net-style long skips across
the stack — while keeping MAS, a duration predictor and a length regulator. Start the
spike from that block shape rather than a plain DiT. Two amendments follow from the
study: (a) carry the **CFG amplification lever already adopted 2026-07-16**
([STATE.md](STATE.md): conditioning dropout 0.15) into this spike's scope — it was never
written into it, and StableTTS confirms it survives a DiT decoder, buying an
inference-time *direction-strength* dial per [[vocalizer-vetting-surface]];
(b) consistency-FM training (RapFlow-TTS, ~2 NFE,
Apache-2.0 code **and** weights) is a 5–10× throughput lever but changes the objective —
it gets its OWN de-risk cycle later, not a ride on this one.

## Sequencing

> **Superseded 2026-08-06 — the *when* now lives in
> [quality-gap-plan.md § Phase 2](quality-gap-plan.md).** delivery-v1 is complete, so
> "runs parallel to corpus assembly" no longer describes anything. Two amendments from
> that plan bind this design: the spike runs **after** the Phase 1 data work lands, and
> **its parity gate must run against a U-Net baseline trained on the same expanded
> corpus** — otherwise data and architecture confound each other and the run teaches
> nothing about either. Freeze that baseline as the last act of Phase 1.

Architecture work is CPU-side until the de-risk training run; that run obeys
[[spin-down-inference-before-training]] and waits for a render-idle window. The
direction-taking run starts on whichever decoder holds the gate when the corpus is folded —
**stall → the stock decoder runs the corpus** and the schedule never waits on the swap.

---

# The base model — Matcha over StyleTTS2-Lite (2026-06-14, reaffirmed 2026-07-29)

_Distilled 2026-08-02 from the retired `actor-model-and-training.md` (full June record —
decision matrix, training 101, RunPod-vs-local hardware guide — in git history; the
hardware question is now governed by [local-vs-runpod-decision.md](local-vs-runpod-decision.md)
and all runs to date have been local)._

**The decision:** Sonora's base is **Matcha-TTS** (MIT). It won on the axes that carry
first-run risk: single-stage non-adversarial training (no GAN), an export-clean
transformer/conv architecture (no LSTM CPU-fallback on mobile delegates), an official
ONNX exporter matching the validated `torch → ONNX` backbone, and speaker-embedding
conditioning that the FiLM/VAT layer extends additively. StyleTTS2's higher raw
ceiling routes through multi-stage adversarial training and a reference-derived style
prior we don't control — "Lite" meant re-architecture, not fine-tuning.

**Reaffirmed (owner, 2026-07-29)** after a month of directing seven teacher engines:
the campaign's central lesson — *an engine's usable range is set by its conditioning
interface, not its raw quality* — argues **for** a base whose conditioning surface we
define, and the measured reference-style failure modes (excursion-driven voice splits,
45% gender flips on cloning, synthetic ref pools) argue **against** reference-derived
style machinery. **StyleTTS2-Lite is retired as the quality-ceiling contingency**; the
escape hatch is scaling the flow-matching backbone (§1/§4). Kokoro stays out: frozen
voicepacks, no training recipe, LSTM export hostility, heavy PL-BERT startup —
non-directability is the origin reason (§2 records the full shortcoming list).

**License verification (GitHub API, 2026-06-14):** StyleTTS2 / PL-BERT / HiFi-GAN /
**Matcha-TTS** / VITS / Piper / MeloTTS / F5-TTS all MIT ✅ · Kokoro Apache-2.0 ✅ ·
espeak-ng **GPL-3.0** ⚠️ (removed from the training path 2026-07-14; banned from
runtime) · Coqui XTTS repo MPL-2.0 but **weights CPML non-commercial** ❌ ·
fish-speech custom ❌. Standing lesson: **repo license ≠ weights license** — always
check weights and training-data terms separately ([[nc-license-stance]] stop-me rule).

**The guiding principle, kept:** make the first success boring — and after the teacher
campaign, don't make an adversarial GAN stack the second project either. The
differentiating work (VAT directability, casting, export discipline) was never the base
model; it transfers to any tier of the family.

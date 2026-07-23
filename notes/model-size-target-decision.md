# Model size target — "Kokoro+" ballpark, 150M loose max, mobile-anchored (2026-07-23)

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
≈1.5–3 GB quantized + TTS actor ≈0.1–0.3 GB on 32 GB dev hardware, actor-model-and-training.md:302-303)._

## Why Kokoro was left (the actual shortcomings — none are about size)

1. **Non-directability — the origin reason.** "Sonora exists because Kokoro — otherwise
   fantastic — could not be *directed*" (high-ambition-6-audience-conveyance-stt.md:26). Static/
   implicit style control (target reference audio or averaged style matrices) makes programmatic
   mood/inflection control impossible (high-ambition-5-styletts2-lite.md:122).
2. **LSTM architecture** — export/mobile-delegate hostile (actor-model-and-training.md:143),
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

Ballpark, not yet wired into any config or eval-harness threshold. Revisit when the derisk_energy
§7 verdict lands and export-harness adaptation begins — that's the natural point to pick an actual
target vocoder config and speaker-conditioning approach against this budget.

Linked from: [sample-rate-24khz-decision.md](sample-rate-24khz-decision.md) (vocoder size
precedent), [actor-model-and-training.md](actor-model-and-training.md) (Kokoro license/
architecture table), [high-ambition-5-styletts2-lite.md](high-ambition-5-styletts2-lite.md) and
[high-ambition-6-audience-conveyance-stt.md](high-ambition-6-audience-conveyance-stt.md) (Kokoro
shortcomings), [STATE.md](STATE.md) §3 (VAT goal, Gemma 4 Director pairing).

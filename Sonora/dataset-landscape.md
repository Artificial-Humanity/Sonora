# Dataset Landscape — Training Options for Sonora

_Surveyed 2026-07-13 (HF Hub, licenses verified per-repo unless marked unverified). The governing
rule comes from [open-decision-licensing.md](../Prosodia/open-decision-licensing.md) tightening #3:
Sonora is Apache-2.0 "for everyone," so **every training input must be permissive (CC-BY-4.0 or
freer)** — no NC/ND anywhere in the lineage. Roles reference the roadmap: Gate-2 24 kHz fine-tune,
Phase-1 casting grid / multi-speaker, milestone-3 VAT directability corpus, and
[high-ambition-6](high-ambition-6-audience-conveyance-stt.md) dual-use._

## 📝 Standing data-prep rule: force-align text to audio, ASR is fallback-only (owner, 2026-07-18)

**Wherever a canonical text for an audio source exists or can be sourced, PREFER force-aligning that
text to the audio over transcribing with Whisper/ASR — for every source, not just LibriVox.** Forced
alignment against the true text yields verbatim, punctuation-correct, alignment-grade transcripts;
ASR introduces substitution/hallucination error that then propagates into every downstream label and
duration/F0 target. **ASR is the fallback** — used only when no canonical text exists (in-the-wild
sources like Emilia). Applies to: raw LibriVox (→ its Gutenberg source text), the owner's
audiobooks (→ owned DRM-free ebook, see §owner's-audiobooks), and any future audio source.
Corollary: for corpora already delivered *with* aligned text (LibriTTS-R, MLS), the alignment is
done — don't re-transcribe, and don't re-synthesize that text (see [book-prose-synthesis-spike.md
§ settled Q3](book-prose-synthesis-spike.md): real audio wherever it exists gets real/aligned text;
synthesis is reserved for text with no real audio).

## ✅ Cleared for training (license verified)

| Dataset | License | What it is | Role |
|---|---|---|---|
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | Public domain | 24 h, single speaker, 22.05 kHz | **Phase 0 (done)** — pipeline de-risking; the current v1 voice |
| [LibriTTS-R](https://hf.co/datasets/mythicinfinity/libritts_r) | CC-BY-4.0 | 585 h, ~2,400 speakers, **24 kHz**, quality-restored | **Gate-2 24 kHz fine-tune + Phase-1 casting grid** (timbre anchors, multi-speaker) |
| [parler-tts/libritts-r-filtered-speaker-descriptions](https://hf.co/datasets/parler-tts/libritts-r-filtered-speaker-descriptions) | CC-BY-4.0 | LibriTTS-R with per-utterance natural-language annotations (pace, pitch, expressivity, quality) | **Milestone-3 labeling shortcut** — map annotations → V/A/T instead of hand-labeling |
| [cdminix/libritts-r-aligned](https://hf.co/datasets/cdminix/libritts-r-aligned) | CC-BY-4.0 | LibriTTS-R with forced alignments + extracted prosody measures (per-token pitch/energy/duration) | **VAT-labeling substrate** — arousal/tension correlate with exactly these measures; also the per-token duration/F0 training targets |
| [Emilia — **Emilia-YODAS subset only**](https://hf.co/datasets/amphion/Emilia-Dataset) | CC-BY-4.0 (this subset) | ~114k h in-the-wild, emotionally diverse speech (YODAS-sourced); gated repo (click-through) | **Expressivity mining** for milestone 3 — filter for high-arousal/high-variance segments with the same measure tooling; too big to use whole |
| [GLOBE V2](https://hf.co/datasets/MushanW/GLOBE_V2) | CC0 | 44.1 kHz *nominal* — **supersampled Common Voice** (crowd mics upsampled; true bandwidth lower), worldwide accents | **Phase-1 casting variety / accents** — zero license anxiety; keep away from the quality bar |
| [VCTK](https://hf.co/datasets/CSTR-Edinburgh/vctk) | CC-BY-4.0 (verified 2026-07-19) | 110 speakers, ~44 h, 48 kHz studio, neutral read sentences | **Phase-1 casting/accent variety** — timbre anchors; no conveyance value |
| [Hi-Fi TTS](https://hf.co/datasets/MikhailT/hifi-tts) | CC-BY-4.0 | LibriVox audiobooks, 10 speakers × deep hours (~292 h), 44.1 kHz | **Casting anchors + speaker-consistent long-form prosody** — few voices, lots of each |
| [MLS English](https://hf.co/datasets/parler-tts/mls_eng) | CC-BY-4.0 | ~44.5k h LibriVox audiobooks — but **16 kHz** | Pretraining-scale only; sample rate disqualifies it for the quality fine-tune |

## 🧪 Evaluation & methodology (not training data)

| Dataset | License | Use |
|---|---|---|
| [E-VOC](https://hf.co/datasets/wizzzzzzzzz/E-VOC) | CC-BY-4.0 | Human-ratings corpus on the **instruction↔perception gap** in expressive TTS — "did the listener hear the emotion the director asked for" is literally Prosodia's success metric; mine it for eval design |
| [MANGO](https://hf.co/datasets/ai4bharat/MANGO) | CC-BY-4.0 | Large-scale MUSHRA listening-test methodology — how to run perceptual evals at scale when the awe question gets serious |

## ❌ Excluded from training (keep as listening/design reference only)

| Dataset | Why excluded |
|---|---|
| [Expresso](https://hf.co/datasets/ylacombe/expresso) | **CC-BY-NC-4.0.** Was penciled into the roadmap (casting grid / milestone 3) — replaced per open-decision tightening #3. Still the best *design reference* for expressive style taxonomies (8 read + 26 improvised styles, 4 speakers, 48 kHz). Gray-area tiers + quality-weighted deliberation: [nc-gray-area-and-candidate-quality.md](nc-gray-area-and-candidate-quality.md) |
| [EmoV-DB](https://github.com/numediart/EmoV-DB) | **NC** (LICENSE.md: "Non-commercial Purposes" only — resolved 2026-07-19). Rare acted classes (amused-with-laughter, sleepy) make it reference-only alongside Expresso |
| Emilia — original 101k-h subset | CC-BY-NC-4.0 (only the YODAS portion is CC-BY; the repo's license tag reflects the newer subset — easy to misread, verified 2026-07-13) |
| AniSpeech / Hailuo-derived sets, etc. | Carry MIT labels but are scraped/synthetic from copyrighted or closed-model sources — **provenance risk**; the "for everyone" promise needs clean lineage, and synthetic-from-closed-APIs additionally raises ToS questions (the Kokoro data caveat) |

## 🌍 Multilingual — deferred to its own vetting surface

This doc is **English-only by scope**. Non-English sources — full MLS language splits, MLCommons
People's Speech, the MLCommons/Common Voice relationship — are surveyed separately in
[multilingual-dataset-sources.md](multilingual-dataset-sources.md) so this file stays the
cleared-for-English SSOT. The same two constraints carry over unchanged: the CC-BY-4.0-or-freer
license wall (tightening #3) and the 24 kHz quality bar (which already disqualifies MLS-English and
would hit every 16 kHz multilingual split identically). Do not ingest anything from that file until
its per-split license/sample-rate checks are done.

## 🔍 Candidates for a later vetting pass (NOT yet license/provenance-verified)

~~VCTK~~ (verified CC-BY-4.0 2026-07-19 → promoted to the cleared table above) · ~~EmoV-DB~~
(verified **NC** 2026-07-19 → moved to the excluded table above) ·
[MLCommons People's Speech](https://hf.co/datasets/MLCommons/peoples_speech) (~30k h, English;
**mixed CC-BY-4.0 + CC-BY-SA-4.0** subsets — the SA portion's share-alike is arguably *not* "freer
than CC-BY," so it must be split out, not ingested wholesale; ASR-grade audio, sample rate/quality
need per-clip checking before any TTS use) ·
[KRAFTON Raon-OpenTTS-Pool](https://hf.co/datasets/KRAFTON/Raon-OpenTTS-Pool) (license "other",
per-source mix — **read its tech report as a curation recipe** rather than ingesting; 615k h from
11 sources is the modern data-centric playbook, the Kokoro lesson at scale). Verify each before any
training use; do not trust this paragraph's parentheticals.

## ⭐ Standouts for the dual goal: emotional conveyance × long-form reading (2026-07-13)

The two goals pull apart: read-audiobook corpora have **narrative continuity** but are prosodically
narrow (calm narration dominates — the V/A/T *tails* are undersampled); in-the-wild corpora have
the emotional range but only in **short, contextless segments**. Ranking against both:

1. **LibriTTS-R + its two derivative layers** — the backbone. The only cleared corpus that is
   *natively long-form narration* (LibriVox audiobooks) with **chapter/utterance ordering
   preserved** — which is what enables training/evaluating *cross-sentence* prosody, the thing the
   Director needs the Actor to sustain. Parler annotations + aligned measures add the conveyance
   labels. Use the measures to *select the expressive subset* (high pitch/energy variance readers).
2. **Hi-Fi TTS** — same audiobook domain, but 10 speakers × ~30 h each at 44.1 kHz: deep
   per-speaker hours are exactly what casting anchors and speaker-consistent long-form prosody
   want.
3. **LibriVox itself (public domain, the sleeper)** — the most *theatrical* LibriVox narrators
   (character voices inside narration, hushes, builds) are where genuine conveyance-in-narration
   lives, and the raw source is license-perfect. Curating a "dramatic narrator" subset with our own
   pipeline (Emilia-Pipe-style) is the Kokoro-grade data-craft move and the true Expresso
   replacement for narration. **Text: force-align to the Gutenberg source edition, not Whisper**
   (per the standing data-prep rule above) — LibriVox links its source text, so alignment-grade
   ground truth is always available here; ASR would only add error.
4. **Emilia-YODAS** — the conveyance-range donor: mine it for the V/A/T tails (anger, fear,
   excitement) that audiobooks undersample. Short segments — it contributes *range*, not
   *continuity*.

Recipe implied: LibriTTS-R/Hi-Fi backbone for narrative flow → measure-driven expressive-subset
selection → LibriVox-dramatic + Emilia-YODAS mining to fill the V/A/T tails.

## 🎧 The owner's DRM-free audiobooks (personal collection)

Recurring idea, analyzed 2026-07-13. **The missing-text problem is solved**; the license problem is
not — so these split cleanly by use:

* **Text is recoverable, two ways.** (a) ASR: whisper-class transcription on clean professional
  audio is near-verbatim (our own fidelity gates already run faster-whisper). (b) Better: where the
  matching **DRM-free ebook** is owned — the Prosodia thesis pairing — align audio to the *true*
  book text (audiobook↔ebook forced alignment, the same trick that built LibriSpeech from
  Gutenberg; `folioparser` already parses the book side). Owned audiobook+ebook pairs are perfectly
  parallel corpora: ground-truth text, professional performance, chapter-length continuity —
  literally the ideal *shape* of data for conveyance × long-form.
* **❌ Never in the public Sonora lineage.** DRM-free ≠ licensed for model training or
  redistribution: these are copyrighted performances (author, publisher, narrator; plus
  voice-likeness/right-of-publicity concerns for identifiable narrators). Weights trained on them
  cannot honestly carry the Apache "for everyone" promise or the auditable-provenance model card.
  This is a hard line regardless of how fair-use litigation eventually lands.
* **✅ Legitimate private uses (not redistributed):**
  1. **Measurement corpus** — the highest-value use: extract prosody statistics (V/A/T
     distributions, per-token F0/duration dynamics, pause structure, quote-vs-narration deltas)
     from professional dramatic narration. *Statistics and insights are not copies.* This teaches
     us what "performance-grade" looks like in contract-space and directly guides curation of the
     clean LibriVox-dramatic corpus (⭐ standout #3) — i.e., use them to learn what to look for,
     then source it permissively.
  2. **Private eval gold standard** — the quality bar the directable actor is judged against
     (A/B listening: our render vs. the pro narrator on the same owned passage).
  3. **Private prototype experiments** — e.g., validating that VAT conditioning works at all
     before spending on the clean corpus. Throwaway weights, never promoted to the registry.

## 🧭 Strategy (the derivation-pipeline thesis)

The milestone-3 expressive corpus mostly **already exists as permissive derivatives of
LibriTTS-R**: Parler's descriptions + cdminix's aligned prosody measures supply labels; Emilia-YODAS
supplies expressive raw material minable with the same measure tooling; GLOBE adds casting variety.
The work is a *derivation pipeline* (map annotations/measures → V/A/T; filter; keep lineage CC-BY-
or-freer), not recording or hand-labeling from scratch. Two compounding effects:

1. **Dual-use:** the same corpus read as (audio → labels) trains the
   [high-ambition-6 "Audience"](high-ambition-6-audience-conveyance-stt.md) listener — every
   labeling hour funds both directions.
2. **Provenance:** record each source + license in the registry model card per the promotion
   convention, so the Apache claim on future weights is auditable.

Cross-refs: [STATE roadmap §3 (VAT)](STATE.md) · [high-ambition-1 §Contract-lock](high-ambition-1-matcha-actor.md) ·
[open-decision-licensing.md tightening #3](../Prosodia/open-decision-licensing.md).

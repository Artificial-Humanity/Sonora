# What the teachers trained on — and what is actually copyable

_Researched 2026-08-08, prompted by the owner: "record knowledge of what Qwen Voice used
for its training data along with Chatterbox. Some research there may go a long way to build
a more complete training corpus for us."_

**The headline is a negative, and it is more useful than a corpus list would have been:
neither model discloses its training data.** There is nothing to copy at the level of
"which datasets". What IS disclosed — scale, staging, and mixture balance — is worth more,
and some of it we are already doing by convergence rather than by design.

This file is **reference**, not a licence ruling. Licence rulings live in
[dataset-landscape.md](dataset-landscape.md); what we train on lives in
[training-sources.md](training-sources.md).

---

## Qwen3-TTS-12Hz-1.7B-VoiceDesign

Weights **Apache-2.0**. Technical report [arXiv:2601.15621](https://arxiv.org/abs/2601.15621).
Our gold-standard teacher ([teacher-synthesis-portfolio], confirmed at equal loudness).

**Data, in full, as disclosed:**

> "Trained on over 5 million hours of speech data spanning 10 languages"

That is the entire statement. **No corpus is named. The ten languages are not enumerated in
the abstract or in the body I could read.** The only licensing sentence in the report is
about the release — *"we release both tokenizers and models under the Apache 2.0 license"* —
which is about the weights, not the data. The named datasets in the report (Seed-TTS test
set, CommonVoice, Fleurs, LibriSpeech, InstructTTSEval) are **evaluation only**.

**What IS methodologically disclosed — three-stage pre-training:**

| stage | what it trains on | stated purpose |
|---|---|---|
| S1 general | "over 5 million hours of multilingual speech data" | basic text→speech mapping |
| S2 high-quality | a stratified subset — *"We stratify data quality with a dedicated pipeline and perform continual pre-training"* | *"alleviate hallucinations caused by noisy data"* |
| S3 long-context | max token length 8,192 → 32,768; *"upsample long speech in the training data"* | long-form |

The stratification **mechanism is not described** — no MOS/DNSMOS, ASR, or diarization
criteria given. So the *shape* is copyable and the *recipe* is not.

**VoiceDesign specifically:** *"we introduce a probabilistically activated thinking pattern
during training to improve instruction following."* **No detail on how the
instruction / voice-description training pairs were created** — which is exactly the part we
would most want, since that is the analogue of our Director→Actor channel.

**Tokenizers** (relevant to us because they set the frame rate the acoustic model works at):
- 25 Hz single-codebook — Qwen2-Audio encoder with an inserted VQ layer; reconstructs mel
  via flow matching + BigVGAN.
- **12 Hz multi-codebook** (the one we render with) — 16 quantizers at 12.5 Hz / 80 ms,
  semantic–acoustic decomposition under a WavLM teacher, 15-layer residual VQ, causal
  ConvNet for streaming.

## Chatterbox (Resemble AI)

Weights **MIT**. Classic English 0.5B — a Llama backbone.

**Data, in full, as disclosed in the repo:**

> "Prompts are sourced from freely available data on the internet."

That is the only provenance statement in the README. **No hours, no corpora, no filtering,
no per-language breakdown.** Resemble's own marketing claims **0.5 M hours of "cleaned
data"** for the English model — widely repeated, but it is a company claim, not a
documented figure, and it appears nowhere in the repo.

**The multilingual line is more forthcoming, and it is the useful part.** For the 23-language
model Resemble states the mixture went **25.6 k h → 36.7 k h** with

> "improved balancing across **script reading, narration, and conversational** data to
> better match real-world usage patterns"

and a "stronger focus on high-quality, expressive, conversational data, with more emphasis
on priority languages and regional variants."

Every output carries Resemble's **PerTh** inaudible watermark. Worth holding beside
[lucida-media-generation]'s note that Google output carries SynthID + C2PA: **our
teacher-render corpus is watermarked at the source**, by two different schemes, and nobody
has checked what our resampling and loudness normalisation do to those marks.

---

## What this actually tells us

### 1 · The scale gap is four to five orders of magnitude, and it is not closeable

| | hours |
|---|---:|
| Qwen3-TTS | ~5,000,000 |
| Chatterbox (English, company claim) | ~500,000 |
| Chatterbox (multilingual mixture) | 36,700 |
| **Sonora v5, the corpus we just built** | **78.5** |
| Sonora, everything cleared and reachable (LibriTTS-R full + Hi-Fi TTS + Emilia-YODAS EN) | ~115,000 + |

The prosody we admire in Qwen is **not** coming from a clever corpus we could reproduce. It
is coming from scale we will not reach with a licence wall set at unrestricted open
redistribution. **This validates the teacher-synthesis lane rather than redirecting it** —
distilling their output *is* the mechanism by which that scale reaches us, and it is why the
portfolio exists. It also puts a ceiling on "just get more data" as a strategy: rung 3 of
the ladder (LibriTTS-R full, ~615 h) is a 10× on us and a 0.01% on them.

### 2 · Three methods worth copying — we already do two by accident

- **Quality stratification as a CURRICULUM, not just a gate (Qwen S2).** We stratify — QC
  gate, DNSMOS floors, [audit-trust-tiers] — but we use it to *admit or reject*. Qwen uses
  it to *stage*: train broad, then continue on the clean tier. **We have never trained in
  stages.** This is the one genuinely new idea in the research and it costs no new data.
  Filed, not scheduled.
- **Mixture balance across MODES OF ADDRESS (Chatterbox).** "Script reading, narration,
  conversational" is very nearly our delivery vocabulary — Newscaster / Documentary /
  Neutral / Dialogue / Speech. Two teams arrived at the same axis independently, which is
  the strongest outside evidence the delivery contract has. Their rebalance was toward
  *conversational*; our corpus is 64% Dialogue ([audio-review-directory]), so we are
  already on the side they moved toward.
- **Long-form upsampling (Qwen S3).** We raised `MAX_SECONDS` 16 → 22 for arc diversity on
  2026-08-01 — same instinct, and their staging says it is worth doing deliberately rather
  than as a filter threshold.

### 3 · The provenance question, stated plainly rather than discovered later

Our bar is stricter than either teacher's: the 2026-08-01 no-patent / fully-Apache posture
sets it at **unrestricted open redistribution**. Both teachers ship **permissive weights
over undisclosed data**.

Our recorded position already handles the sharp version of this and is coherent:
`dataset-landscape.md` blocks AniSpeech/Hailuo-derived sets as "scraped/synthetic from
copyrighted or **closed-model** sources — provenance risk", noting that synthetic-from-
closed-APIs additionally raises ToS questions. **Our teachers are open-weight (Apache-2.0
and MIT) and licensed for exactly this use, so the ToS half does not apply.**

What is true, and should be written down once rather than rediscovered: **"every clip traces
to a licensed recording" holds for the LibriTTS and Emilia-YODAS halves of our corpus and
does NOT hold for the teacher-render half.** Those clips trace to a licensed *model* whose
own training data is undisclosed. That is a weaker claim than the one we make about the real
audio, it is already implicitly accepted by using these teachers at all, and it is not a
reason to stop — it is a reason for the model card to say two different things about two
halves of the corpus instead of one thing about all of it.

---

## The multilingual plan

_Owner, 2026-08-08: "Even though we are targeting English, we should still maintain a
multilingual plan and I'm sure those have trailblazers a trail to follow."_ They do, and the
trail has a clear shape.

**What the trailblazers actually did: English deep, multilingual comparatively thin, and
LATER.** Chatterbox is ~0.5 M h English against a **36.7 k h** mixture for 23 languages —
roughly 1.5 k h per language, ~7% of the English budget for 23× the language coverage. Qwen
spans 10 languages inside one 5 M h pool. Neither built multilingual first. **So the
sequencing we already have is the trail: English to quality, then a thin balanced mixture
on top — not a parallel effort.**

**The single most actionable fact, and it is close to free:** we already hold the pipeline.
`mine_emilia_probe` → `mine_emilia_keeps` → `process_emilia_tail` → `merge_emilia_corpus`
is **language-agnostic except for the G2P and the ASR cross-check**, and
**Emilia-YODAS is six languages — `zh, en, ja, fr, de, ko` — of which we have mined
English only.** Five more languages are reachable with the tooling built this week and no
new licence work, since the CC-BY-4.0 YODAS portion is already declared in
`configs/data_licenses.yaml`.

⚠ **The trap is already known and already written down:** the `amphion/Emilia-Dataset` repo
tag reads `cc-by-4.0`, but **Emilia-Large mixes the CC-BY 114 k-h YODAS portion with the
101 k-h ORIGINAL subset, which is CC-BY-NC-4.0.** Only the YODAS portion clears our bar,
in every language. `emilia_original` is declared `nc` in the licence wall for exactly this
reason. Verify per shard, never per repo tag.

**Permissive multilingual candidates, NOT yet vetted per-split — do not ingest on this
paragraph:**

| source | licence | scale / languages | the catch |
|---|---|---|---|
| **Emilia-YODAS** | CC-BY-4.0 (YODAS portion ONLY) | 114 k h · zh/en/ja/fr/de/ko | already cleared, already tooled — **start here** |
| YODAS2 | CC-BY-3.0 | ~500 k h · 100+ languages | the upstream Emilia-YODAS was mined from; raw YouTube, needs the whole pipeline |
| MLS | CC-BY-4.0 | 8 languages | **16 kHz — fails our quality bar**, same as MLS English. Pretraining-scale only |
| Granary | check | 25 European languages | ASR/AST dataset; licence and TTS-suitability both unverified |
| GLOBE V2 / Common Voice | CC0 | worldwide accents | supersampled — true bandwidth below nominal |
| WorldSpeech | check | multilingual | new (2026); nothing verified |

**What a multilingual rung would cost us that English does not**, recorded so it is a
decision and not a surprise:
1. **G2P per language.** `op_g2p` is the English espeak-free lane; every language needs its
   own dictionary + OOV path, and gate **G7** holds the device front end to phoneme parity —
   so each language doubles that surface ([device-g2p-parity-gate]).
2. **The ASR cross-check.** `ASR_MAX_WER` against `base.en` — English-only by name. A
   multilingual merge needs a multilingual ASR or the caption filter silently stops working.
3. **The VAT labels are language-blind by construction and that is a claim, not a fact.**
   V/A/T are acoustic (LUFS, phonation, EIV heads); nothing in them assumes English, but
   nothing has tested that the composites mean the same thing across languages either.
4. **Delivery vocabulary.** The five lanes are modes of address in English discourse. Whether
   "Newscaster" partitions the same way in Japanese is an open question, and the contract
   says vocabulary changes need an owner call.

**Sequencing: multilingual is Phase 1+, after the English ladder.** It is not scheduled and
should not be until rung 3, because every item above is work that competes directly with the
volume lever we have not yet pulled. What this file buys is that the plan exists, the trail
is identified, and the five free languages sitting in an already-cleared, already-tooled
corpus are on record instead of forgotten.

---

## Sources

- [Qwen3-TTS Technical Report, arXiv:2601.15621](https://arxiv.org/abs/2601.15621) ·
  [model card](https://hf.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [resemble-ai/chatterbox (GitHub)](https://github.com/resemble-ai/chatterbox) ·
  [model card](https://hf.co/ResembleAI/chatterbox) ·
  [Chatterbox Multilingual announcement](https://www.resemble.ai/introducing-chatterbox-multilingual-open-source-tts-for-23-languages/) ·
  [Chatterbox Multilingual v3](https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages) ·
  [Resemble model page](https://www.resemble.ai/learn/models/chatterbox)
- [Emilia dataset](https://hf.co/datasets/amphion/Emilia-Dataset) ·
  [Emilia paper, arXiv:2501.15907](https://arxiv.org/abs/2501.15907)

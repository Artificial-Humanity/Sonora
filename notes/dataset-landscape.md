# Dataset Landscape — Training Options for Sonora

> **This file evaluates CANDIDATES. For what is on disk and what a training run
> actually consumes, see [training-sources.md](training-sources.md) — that is the
> SSOT for a source's STATE, this one is the SSOT for its LICENSE and its role.**
> They were the same file in spirit and neither answered "what are we training on
> today", which is how we reached 2026-08-02 with ~60 GB of cleared, downloaded,
> unused audio and every checkpoint tracing to a tenth of LibriTTS-R.
>
> **Three documents live here, folded in and kept as parts:** the English survey
> (below), **§ NC licensing — the two-fence ruling** (was
> `nc-gray-area-and-candidate-quality.md`), and **§ Multilingual sources** (was
> `multilingual-dataset-sources.md`). Where those two used to point at this file by
> name they now point at the part; a July rename pass had turned those references
> into links to this file itself (fixed 2026-08-06).

_Surveyed 2026-07-13 (HF Hub, licenses verified per-repo unless marked unverified). The governing
rule comes from [open-decision-licensing.md](../../../Prosodia/notes/open-decision-licensing.md) tightening #3:
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
§ settled Q3](book-prose-lane.md): real audio wherever it exists gets real/aligned text;
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
| [Expresso](https://hf.co/datasets/ylacombe/expresso) | ⚠ **THIS RULING IS CONTESTED — see [training-sources.md § The Expresso conflict](training-sources.md). The 2026-08-01 no-patent / fully-Apache-2.0 posture sets the corpus bar at *unrestricted open redistribution* and says explicitly it is STRICTER than the old commercial test, which CC-BY-NC does not clear. Unresolved; owner's call. Nothing has trained on it.** — **CC-BY-NC-4.0 — status changed 2026-07-19.** Bare NC (no click-through agreement executed) is **risk-accepted for training use** under the two-fence ruling below; tainted-lineage bookkeeping applies and the ship/don't-ship call defers to promotion time. Agreement-walled NC (EmoV-DB) stays fully excluded. Still the best *design reference* for expressive style taxonomies (8 read + 26 improvised styles, 4 speakers, 48 kHz). Gray-area tiers + quality-weighted deliberation: **§ NC licensing** below |
| [EmoV-DB](https://github.com/numediart/EmoV-DB) | **NC** (LICENSE.md: "Non-commercial Purposes" only — resolved 2026-07-19). Rare acted classes (amused-with-laughter, sleepy) make it reference-only alongside Expresso |
| Emilia — original 101k-h subset | CC-BY-NC-4.0 (only the YODAS portion is CC-BY; the repo's license tag reflects the newer subset — easy to misread, verified 2026-07-13) |
| AniSpeech / Hailuo-derived sets, etc. | Carry MIT labels but are scraped/synthetic from copyrighted or closed-model sources — **provenance risk**; the "for everyone" promise needs clean lineage, and synthetic-from-closed-APIs additionally raises ToS questions (the Kokoro data caveat) |

## 🌍 Multilingual — deferred to its own vetting surface

This doc is **English-only by scope**. Non-English sources — full MLS language splits, MLCommons
People's Speech, the MLCommons/Common Voice relationship — are surveyed separately in
**§ Multilingual sources** (below) so this file stays the
cleared-for-English SSOT. The same two constraints carry over unchanged: the CC-BY-4.0-or-freer
license wall (tightening #3) and the 24 kHz quality bar (which already disqualifies MLS-English and
would hit every 16 kHz multilingual split identically). Do not ingest anything from that file until
its per-split license/sample-rate checks are done.

> **The multilingual PLAN — what to do and when — lives in
> [teacher-training-data.md § the multilingual plan](teacher-training-data.md)** (owner asked
> for one 2026-08-08). This section stays the *licence* surface; that one carries the
> sequencing, the per-language costs, and what the trailblazers actually did. Two facts from
> it belong here because they are licence facts:
>
> - **Emilia-YODAS is SIX languages — `zh, en, ja, fr, de, ko` — and we have mined English
>   only.** The CC-BY-4.0 YODAS portion is already declared in `configs/data_licenses.yaml`,
>   and the mining pipeline is language-agnostic apart from G2P and the ASR cross-check. Five
>   languages are reachable with zero new licence work.
> - ⚠ **The `amphion/Emilia-Dataset` repo tag reads `cc-by-4.0` and that is a trap in every
>   language, not just English.** Emilia-Large mixes the CC-BY 114k-h YODAS portion with the
>   101k-h ORIGINAL subset, which is **CC-BY-NC-4.0**. Verify per shard, never per repo tag —
>   `emilia_original` is declared `nc` in the wall for exactly this reason.

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
[open-decision-licensing.md tightening #3](../../../Prosodia/notes/open-decision-licensing.md).

---

# NC licensing — the two-fence ruling (folded in 2026-07-26)

*(From the retired `nc-gray-area-and-candidate-quality.md`. That doc called itself "the
deliberation surface" and the survey above "the verdict table"; the 2026-07-19 ruling closed the
deliberation, and keeping a second place to look for licence facts had already produced a
disagreement about Expresso. **The verdict table is the survey above; this part is why it says
what it says.**)*

_Owner request 2026-07-19: a write-up of (a) the CC-BY-NC on-disk options and the gray-area
decisions around them, and (b) the surveyed-but-unverified candidates — **quality-weighted**,
because the quality of the permissive options bears directly on whether the NC gray area is ever
worth entering._

## 0 · Corrections to the record (2026-07-19)

* **JL-Corpus is CC0, not NC** (a session summary misreported it). It has been used as the
  tension-calibration anchor all along ([vat-channels.md](../docs/vat-channels.md),
  ARCHITECTURE §labels). On disk: `/data/model-training/datasets/JL-Corpus` (459 MB, 2,400 acted
  utterances, 4 NZ-English speakers, 5 primary + 5 secondary emotions, perception-verified by a
  120-participant study). Gap: it is **not declared in `Sonora/configs/data_licenses.yaml`** — the
  wall would refuse it in a filelist (fail-safe, correct); declare it `permissive` if it ever
  graduates from calibration to training.
* **EmoV-DB is NC** — resolved 2026-07-19: its LICENSE.md conditions use on "Non-commercial
  Purposes" (research). It moves from "candidates" to the NC bucket below. Not on disk.
* **VCTK license verified CC-BY-4.0** (HF `CSTR-Edinburgh/vctk` card, 2026-07-19) — was
  "reportedly"; now cleared for the candidate table below.

## 1 · What is actually on disk, license-wise

| On disk | License | Size | Status |
|---|---|---|---|
| `datasets/expresso` | **CC-BY-NC-4.0** | 1.8 GB — 11,615 read-speech clips (the read portion of the 40 h set) | Reference-only today; the subject of §2 |
| `datasets/JL-Corpus` | **CC0** | 459 MB, 2,400 utts | Calibration anchor (in use, clean) |
| Owner's DRM-free audiobooks | Copyrighted performances | — | Hard line + three sanctioned private uses, already settled — see **§ The owner's DRM-free audiobooks** above and [audiobook-corpus-policy.md](../docs/audiobook-corpus-policy.md) (the policy SSOT) |
| Emilia original 101k-h subset | CC-BY-NC-4.0 | not on disk (only the CC-BY YODAS keeps were mined: 13,141 clips in `emilia_kept`) | Nothing to ponder — we never pulled the NC portion |

So the on-disk NC question is **really only about Expresso**.

## 2 · Expresso — the NC crown jewel, and the actual gray area

**What it is (quality):** Meta's 48 kHz studio expressive corpus. 4 professional actors (2M/2F);
40 h total = 11 h expressively-read speech across **8 styles** + 30 h improvised dialogue across
**26 styles**; includes whisper, laughter, non-verbal vocalizations, child-directed speech. Its
two properties nothing surveyed replicates:

1. **Style-parallel same-text renders** — the same sentence performed in multiple styles by the
   same actor = direct style-contrast supervision with text and speaker held constant.
2. **Studio-grade acted extremes** — whisper/laughter/NV at anechoic quality; audiobook corpora
   have none of this, in-the-wild corpora have it only noisy and unlabeled.

**The gray-area tiers.** The repo already implies a three-tier policy; making it explicit:

| Tier | Use | Standing |
|---|---|---|
| 1. Reference / measurement | Style taxonomy design; extract statistics, distributions, thresholds ("statistics are not copies") | ✅ Sanctioned today (landscape "design reference"; same rationale as the audiobook measurement-corpus use) |
| 2. Private de-risk training | Throwaway experiments validating an architecture question (e.g. "does style/VAT conditioning learn at all"), run under `SONORA_LICENSE_WALL=derisk`, run **tainted**, never promoted to Registry, never used for release decisions marketed as data-clean | ⚠️ The wall supports it by design (`data_licenses.yaml` class `nc`). Legal exposure of private research training is the classic TDM/fair-use gray zone; the *binding* constraint is our own Apache "for everyone" promise, which tainted throwaway weights do not touch. **Owner call, case-by-case.** |
| 3. Production lineage (any released weights/dataset) | — | ❌ Hard no (open-decision tightening #3). Includes **laundering**: a model trained on NC data generating "clean" synthetic data inherits the restriction in spirit — treat distill-through as tier 3, not tier 2. |

**Owner position (2026-07-19) — "walls" are really "low fences."** Sonora began dual-licensed
and the owner deliberately chose full Apache-2.0 ([open-decision-licensing.md](../../../Prosodia/notes/open-decision-licensing.md)).
Their read of the NC landscape: the liability theory for *downstream commercial usage* of
weights trained on NC data is dubious and highly debated — in fair-use/TDM-exception
jurisdictions the NC condition may simply never trigger for training, because no licensed right
is being exercised. Two counterweights keep the fence standing even under this view:

1. **Contract ≠ copyright.** Some datasets (EmoV-DB's "By downloading or using… you agree"
   phrasing) are click-through *agreements* — a breach-of-contract theory needs no copyright
   trigger. Bare CC-BY-NC (Expresso) is the weaker fence; terms-conditioned downloads are the
   taller one. Worth distinguishing per-source.
2. **The binding constraint was never liability.** Tightening #3 exists to make the Apache
   "for everyone" claim *auditably true* on the model card — a trust/provenance promise, not a
   legal-risk hedge. That promise binds released weights (tier 3) regardless of how the law
   settles.

Net effect of the owner's position: tier 1 is unquestioned, **tier 2 (tainted private
de-risk) is comfortably available** when it buys something, and tier 3 remains closed as a
matter of the project's own promise rather than fear of liability.

**Owner RULING (2026-07-19, same day) — the two-fence taxonomy is now policy:**

1. **Agreement-walled NC** (click-through / "by downloading you agree" terms — e.g. EmoV-DB):
   stays **fully walled off**. Not downloaded, not ingested, not declared in
   `data_licenses.yaml`. The contract theory is the fence we respect.
2. **Bare CC-BY-NC** (no agreement executed — Expresso on disk; Emilia-original if ever
   pulled): **risk-accepted for training use.** The owner accepts the (dubious, debated)
   downstream-liability theory as a tolerable risk. Practically this collapses the tier-2
   ceremony: Expresso experiments no longer need to be framed as throwaway.
3. **Bookkeeping survives the ruling:** keep the `nc` class + `SONORA_LICENSE_WALL=derisk` +
   run-taint mechanics — no longer as prohibition but as **lineage audit**, so at any future
   release the model card can state exactly what is in the weights. Risk-exercise plus
   disclosure keeps the "for everyone" promise honest; silence would break it. Whether
   NC-lineage weights ship in a public release stays a **flagged decision at promotion time**,
   made with the audit trail in hand. Distill-laundering remains a hard no.

**What tier 2 would actually buy:** de-risking conditioning mechanics *before* the permissive
expressive stack matures — i.e., prove FiLM/VAT/style conditioning converges on data where the
style signal is guaranteed strong, so a null result on our own corpus can be attributed to the
data, not the architecture. That is a real, bounded value. The cost: taint discipline forever
(tracking that nothing downstream of the run escapes), for a corpus whose irreplaceable slice is
shrinking (§4).

## 3 · Candidate quality dossier (license status as of 2026-07-19)

| Dataset | License | Scale | True audio quality | Expressive range | Continuity | Verdict / role |
|---|---|---|---|---|---|---|
| **VCTK** | CC-BY-4.0 ✅ verified | 110 speakers, ~44 h (~25 min/speaker) | 48 kHz studio (Edinburgh); good, some takes noisy/clipped | Neutral read newspaper sentences — none | None (isolated sentences) | **Casting/accent variety**, timbre anchors. No conveyance value |
| **EmoV-DB** | ❌ NC (LICENSE.md non-commercial) | 4 EN speakers, ~7,000 utts | Anechoic-chamber studio | 5 classes incl. **amused-with-laughter**, sleepy/drowsy — rare acted classes | None | Joins Expresso as reference-only; do not ingest |
| **GLOBE V2** | CC0 ✅ | ~535 h, tens of thousands of speakers | ⚠️ **"44.1 kHz" is supersampled Common Voice** — crowdsourced mics upsampled; nominal rate ≠ true bandwidth. Volume/alignment-cleaned (~5% dropped) | Conversational-neutral | None | Accent/casting breadth only; keep it away from the quality bar. Its zero-anxiety license is its whole charm |
| **Hi-Fi TTS** | CC-BY-4.0 ✅ | 292 h, 10 speakers (~30 h each) | 44.1 kHz, bandwidth-filtered (≥13 kHz) LibriVox — genuinely high | Audiobook narration — moderate, narrow tails | **Chapter-length** | **Strongest permissive quality set.** Casting anchors + speaker-consistent long-form prosody |
| **MLS English** | CC-BY-4.0 | ~44.5k h | 16 kHz — below the 24 kHz bar | Narration | Long-form | Disqualified for fine-tunes; pretraining-scale only if ever needed |
| **People's Speech** | Mixed CC-BY-4.0 + **CC-BY-SA-4.0** | ~30k h | ASR-grade, highly variable | In-the-wild | Variable | SA share-alike arguably fails "CC-BY or freer" — would need subset split; quality disqualifies it for TTS regardless. Skip |
| **Raon-OpenTTS-Pool** | "other" (per-source mix, 11 sources) | 615k h / 239.7M segments | Mixed by construction | Mixed | Mixed | **Read the tech report as a curation recipe; do not ingest the pool.** The Kokoro lesson at industrial scale |

## 4 · Does the permissive stack close the Expresso gap?

Mapping Expresso's roles onto what we already have or can build clean:

| Expresso role | Permissive coverage | Residual gap |
|---|---|---|
| Style taxonomy / design language | Tier-1 reference use (free, sanctioned) | none |
| Expressive V/A/T tails | Emilia-YODAS mining (13,141 keeps already) + LibriVox dramatic-narrator curation (landscape ⭐ #3) | partial — coverage grows with mining effort |
| **Style-parallel same-text contrast** | **The expressive-registers synthesis lane is exactly this** — same line rendered across registers/voices/seeds, now 193 owner-certified clips (v1, 2026-07-19) and growing per the standing directive | shrinking as the lane scales; synthetic rather than acted, but instrument-verified + owner-audited |
| Whisper / laughter / non-verbal, studio-grade acted | ❌ Nothing surveyed covers this permissively (EmoV-DB's laughter is NC too) | **the genuine irreplaceable slice.** Options: mine Emilia-YODAS for whisper/laughter segments; add whisper/laugh registers to the synthesis lane (teacher models can render whisper); tiny in-house recording session; or a tier-2 Expresso de-risk before investing |

**The shape of the decision (superseded by the §2 ruling, kept for the record):** Expresso's
irreplaceable slice has shrunk to *acted-studio whisper/laughter/non-verbals* plus *human (vs.
synthetic) style-parallel data*. Hi-Fi TTS + LibriTTS-R carry the quality bar, GLOBE/VCTK carry
breadth, Emilia-YODAS + our own synthesis lane carry expressivity.

**Post-ruling landing spot:** Expresso (bare NC, risk-accepted) is now directly usable for the
style-conditioning and whisper/laughter work — its unique slice no longer gates on a permissive
replacement maturing. The permissive stack remains the **public-track backbone** (it is what an
undisclosed-clean release would rest on), tainted-lineage bookkeeping continues on every
NC-touching run, and the ship/don't-ship-NC-lineage call is deferred to promotion time with the
audit trail in hand. Emilia-original (bare NC, ~101k h) is now *eligible* for tail-mining if
YODAS runs dry — same bookkeeping.

Cross-refs: **§ Cleared for training** above (the verdict table) ·
[training-sources.md](training-sources.md) (what is actually on disk, and the live Expresso
conflict) ·
[open-decision-licensing tightening #3](../../../Prosodia/notes/open-decision-licensing.md) ·
`Sonora/configs/data_licenses.yaml` (the wall) ·
[vat-channels.md](../docs/vat-channels.md) (JL-Corpus calibration) ·
[synthesis-pipeline.md](synthesis-pipeline.md) + [book-prose-lane.md § Part 1 — Operations](book-prose-lane.md)
(the permissive expressive lanes).

---

# Multilingual sources — survey (folded in 2026-07-31)

> Was `multilingual-dataset-sources.md`. Folded here because it is a survey of the same kind
> this document exists to hold, and it explicitly deferred to this file's discipline — kept
> separate it read as a cleared list rather than the unverified one it says it is.

_Drafted 2026-07-18. **This is an UNVERIFIED survey**, not a cleared list. Every entry below still
owes a per-repo license check and a per-split sample-rate/quality check before a single file enters
any pipeline — exactly the discipline the English survey at the top of this file applies. Do not
trust the parentheticals here as clearance; they are leads to verify._

## Governing constraints (unchanged, cross-lingual)

The multilanguage phase does **not** relax either wall:

1. **License wall — CC-BY-4.0 or freer, no NC/ND anywhere in the lineage**
   ([open-decision-licensing.md](../../../Prosodia/notes/open-decision-licensing.md) tightening #3;
   [ARCHITECTURE.md](../docs/ARCHITECTURE.md) §2). Note the sharp edge: **CC-BY-SA is arguably *not*
   "freer" than CC-BY** — share-alike adds a copyleft obligation. Any SA-licensed subset must be
   split out and treated as excluded until we decide share-alike is acceptable in the public
   lineage (it currently is not established that it is). CC0 is freer; that's clean.
2. **24 kHz quality bar for the fine-tune.** Almost every large multilingual corpus is **16 kHz**
   (ASR heritage). 16 kHz sources are *pretraining-scale only* — same verdict MLS-English already
   earned. Bandwidth extension / restoration (LibriTTS-R was the "-R" restoration of LibriTTS) is a
   separate research bet, not an assumption.

## 🎯 Primary: Multilingual LibriSpeech (MLS)

The natural first target — same LibriVox provenance as the English backbone, so the license story
is the most tractable of the multilingual options.

| Split | Reported license | Notes to verify |
|---|---|---|
| MLS **English** | CC-BY-4.0 (verified 2026-07-13 via `parler-tts/mls_eng`) | Already in the English landscape; **16 kHz**, pretraining-scale only |
| MLS **German, Dutch, French, Spanish, Italian, Portuguese, Polish** | Corpus-level CC-BY-4.0 (MLS is released CC-BY-4.0 as a whole — **verify the specific HF repo you pull each split from carries it through**) | All **16 kHz**; hours are very uneven across languages (German/Spanish deep, Polish/Portuguese thin) |

- Canonical source: OpenSLR SLR94 (the original MLS release). HF mirrors exist per language
  (`facebook/multilingual_librispeech` and community re-hosts) — **the license/quality guarantee
  lives with the specific repo you download from, not with "MLS" the name.** Re-verify per repo.
- Provenance is the strength: LibriVox (public-domain audio) + Project Gutenberg (public-domain
  text), the same clean lineage that makes the English side publishable. This is why MLS ranks
  above web-scraped multilingual sets.
- Weakness: read-audiobook narration → prosodically narrow in *every* language, the same tail
  problem called out for English. Multilingual expressivity tails are an open sourcing question
  (there is no clean Emilia-YODAS equivalent surveyed yet — flag for a later pass).

## 🌐 MLCommons sources

Two distinct things wear the MLCommons name; keep them separate.

1. **Common Voice** (Mozilla, stewarded under the MLCommons umbrella) — **CC0** (public-domain
   dedication). License-perfect. Massive language coverage, crowd-read short utterances.
   - We already ingest a *derivative*: **GLOBE V2** (CC0) is Common Voice-derived and is in the
     English cleared list for accents. For multilanguage, Common Voice itself is the direct source.
   - Caveats to verify: variable recording quality (consumer mics), sample rates vary by clip and
     are often low; short contextless utterances (range, not narrative continuity); per-language
     hours extremely uneven. Curation/filtering is mandatory, not optional.
2. **MLCommons People's Speech** — ~30k h, **English**, **mixed CC-BY-4.0 + CC-BY-SA-4.0**. The
   **SA portion trips the share-alike edge above** and must be split out, not ingested wholesale.
   ASR-grade audio; sample rate/quality need per-clip checking before any TTS use. (Listed as an
   English vetting-pass candidate in **§ Candidates for a later vetting pass** above; repeated
   here because it arrives via the MLCommons question.)

## 🔭 Other multilingual candidates to survey later (NOT yet checked)

- **VoxPopuli** (European Parliament recordings; permissive-leaning license — **verify**, 16 kHz,
  parliamentary register is its own prosodic bias) ·
- **CML-TTS** (Multilingual LibriSpeech-derived, explicitly built *for TTS* — check license + rate;
  potentially higher-value than raw MLS if it clears) ·
- **Emilia non-English subsets** — only the **YODAS** portion is CC-BY (the original subset is
  CC-BY-NC-4.0; the exact same misread trap flagged for English — verify per language) ·
- Fleurs, BibleTTS, and other OpenSLR entries — per-language license lottery; verify each.

## 🧭 Recommended posture when the phase starts

1. **Lead with MLS non-English splits + Common Voice (CC0).** Cleanest provenance; establishes the
   multilingual pipeline on defensible license ground before touching anything murkier.
2. **Treat everything as 16 kHz pretraining-scale** until proven otherwise; do not assume a
   multilingual 24 kHz fine-tune corpus exists — sourcing or restoring one is its own line item.
3. **Split share-alike out.** People's Speech CC-BY-SA and any SA multilingual subset stay excluded
   until/unless the license wall is deliberately revisited for SA — an owner decision, not a
   default.
4. **Record source + license + sample rate per split in the registry model card**, same promotion
   convention as English, so the Apache "for everyone" claim stays auditable across languages.

Cross-refs: **§ Cleared for training** at the top of this file (the English licence SSOT) ·
[open-decision-licensing.md tightening #3](../../../Prosodia/notes/open-decision-licensing.md) ·
[audiobook-corpus-policy.md](../docs/audiobook-corpus-policy.md) (the private-lineage boundary, unchanged
cross-lingually) · [STATE.md](STATE.md).


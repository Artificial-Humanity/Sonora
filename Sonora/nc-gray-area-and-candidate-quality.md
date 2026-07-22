# NC Gray Area & Candidate Quality — Ponder Dossier (2026-07-19)

_Owner request 2026-07-19: a write-up of (a) the CC-BY-NC on-disk options and the gray-area
decisions around them, and (b) the surveyed-but-unverified candidates — **quality-weighted**,
because the quality of the permissive options bears directly on whether the NC gray area is ever
worth entering. Companion to [dataset-landscape.md](dataset-landscape.md) (the license SSOT);
this doc is the deliberation surface, that one is the verdict table._

## 0 · Corrections to the record (2026-07-19)

* **JL-Corpus is CC0, not NC** (a session summary misreported it). It has been used as the
  tension-calibration anchor all along ([vat-corpus-decision-brief.md](vat-corpus-decision-brief.md),
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
| Owner's DRM-free audiobooks | Copyrighted performances | — | Hard line + three sanctioned private uses, already settled — see [dataset-landscape.md §owner's-audiobooks](dataset-landscape.md) |
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
and the owner deliberately chose full Apache-2.0 ([open-decision-licensing.md](../Prosodia/open-decision-licensing.md)).
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

Cross-refs: [dataset-landscape.md](dataset-landscape.md) ·
[open-decision-licensing tightening #3](../Prosodia/open-decision-licensing.md) ·
`Sonora/configs/data_licenses.yaml` (the wall) ·
[vat-corpus-decision-brief.md](vat-corpus-decision-brief.md) (JL-Corpus calibration) ·
[synthesis-pipeline.md](synthesis-pipeline.md) + [book-prose-operations.md](book-prose-operations.md)
(the permissive expressive lanes).

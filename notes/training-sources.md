# Training Sources — SSOT

**Scope: what audio Sonora trains on, where it came from, and what state it is in.**
Nothing else. This file exists because that answer was spread across
`dataset-landscape.md` (candidates + licensing), `book-prose-lane.md` (one lane),
`STATE.md` (runs) and several campaign notes, and no single place said *what is
actually feeding a training run today*. When those disagree with this file about a
source's **state**, this file is right; when they disagree about **licensing or
candidate evaluation**, `dataset-landscape.md` is right.

Deliberately NOT here: license deliberation (→ `dataset-landscape.md`), the
multilingual survey (→ same, § Multilingual), teacher-engine selection (→
`teacher-tts-audition-shortlist.md`), audit policy (→ `delivery-mix-campaign.md`).

Last verified against disk and configs: **2026-08-06**. The *order* in which the READY /
NOT PULLED / RAW rows below get taken up is not decided here — it is
[quality-gap-plan.md](quality-gap-plan.md).

---

## The table

Status vocabulary is deliberately narrow, because "we have it" and "we train on it"
had been blurring into each other:

- **TRAINED** — a checkpoint exists that consumed it
- **READY** — on disk, license-cleared, correctly formatted, never used
- **RAW** — on disk but not usable as-is (archive, metadata-only, unprocessed)
- **NOT PULLED** — cleared, never downloaded
- **BLOCKED** — a decision or a license stands in the way
- **DERIVED** — produced by us from something above

| Source | Origin | Status in our training | Next action |
|---|---|---|---|
| **LJSpeech-1.1** | keithito, public domain. 13,100 clips / 24 h / 1 speaker / 22.05 kHz | **TRAINED** — Phase 0, `ljspeech` runs 2026-07-10 and 07-11 (ep199). The current v1 voice | None. Carries the contraction-poisoned IPA at ~1.2% of rows and was NOT rebuilt — it feeds the Phase-0 warm start, not the VAT lane. Rebuild only if Phase 0 is re-run |
| **LibriTTS-R `train-clean-100`** | Google, CC-BY-4.0, quality-restored LibriTTS. 257 speaker dirs, 9.0 GB on disk → 31,445 rows / **51.3 h** | **TRAINED** — the only VAT source. `derisk_energy` (v1), `vat3_finetune` ep099 (v2), `vat3c_finetune` ep099 (v3c) | None outstanding. See § The VAT corpus lineage |
| **LibriTTS-R `dev-clean`** | Same, CC-BY-4.0. 5,463 clips / **8.7 h** / 40 speakers | **NEVER TRAINED ON, DELIBERATELY, AND IT MUST STAY THAT WAY.** Pulled 2026-08-06 and derived scoring-only as `data/libritts_r_holdout_devclean`; its 40 speakers are disjoint from the corpus's 247 | Nothing. It is the standing instrument — score every new checkpoint with `scripts/score_holdout.sh` and normalise with the *training* corpus's `data_statistics`, never a re-measure. **Training on it destroys it permanently; there is no second dev-clean.** The wall declares it (so `enforce()` loads it) and therefore will *not* stop such a run — the deleted `train_op.txt`/`val_op.txt` and the directory README are the guard. Results: [quality-gap-plan.md § 0a](quality-gap-plan.md) |
| **LibriTTS-R (the other 90%)** | Same, CC-BY-4.0. Full set ≈ 585 h / ~2,400 speakers | **NOT PULLED** — we hold `train-clean-100` only | The 10×. Gated on whether Emilia's +43% moves the holdout — [quality-gap-plan.md § Phase 1](quality-gap-plan.md) |
| **`parler-tts/libritts-r-filtered-speaker-descriptions`** | CC-BY-4.0. LibriTTS-R + per-utterance natural-language annotations (pace, pitch, expressivity, quality) | **NOT PULLED** | Evaluate against our derived V/A/T. Cleared as the "labeling shortcut" and we hand-built that substrate instead |
| **`cdminix/libritts-r-aligned`** | CC-BY-4.0. LibriTTS-R + forced alignments and per-token pitch/energy/duration | **NOT PULLED** | Same evaluation. Its prosody measures overlap what `derive_vat_corpus.py` computes; worth knowing whether it is better |
| **Hi-Fi TTS (v1)** | NVIDIA, CC-BY-4.0, LibriVox-sourced. ~292 h / 10 speakers / 44.1 kHz. **40 GB of parquet WITH audio** | **RAW** — largest corpus on disk, never touched | Convert parquet → wav + filelist. Role: casting anchors and speaker-consistent long-form prosody (few voices, deep hours) |
| **HiFiTTS-2** | NVIDIA, CC-BY-4.0, LibriVox-sourced. 36.7 k h @22 kHz / 31.7 k h @44 kHz, ~5 k speakers | **RAW — METADATA ONLY.** 16.6 GB of manifests and chapter lists (8.9 GB 22 kHz + 7.7 GB 44 kHz). NVIDIA does not redistribute the audio | Decide before any download: **2.8 TB for 22 kHz, 4.0 TB for 44 kHz**, fetched from LibriVox via NeMo-speech-data-processor. A bandwidth-filtered subset is the realistic version |
| **VCTK** | CSTR Edinburgh, CC-BY-4.0 (verified 2026-07-19). 110 speakers / ~44 h / 48 kHz studio | **RAW** — still `VCTK-Corpus-0.92.zip`, 4.3 GB, unextracted | Unzip and build a filelist if the casting/accent grid needs it. Neutral read sentences — no conveyance value |
| **Emilia-YODAS keeps** | Amphion, CC-BY-4.0 (**YODAS subset only**). **13,141 clips**, mined and resampled to 24 kHz, 5.4 GB | **READY** — the closest thing to usable that is not in a filelist | Merge into the VAT filelists (`eiv_merge_corpus.py`). This is the v1.1 volume lane and per the mining verdict likely holds our only REAL newscaster/documentary audio |
| **Emilia-YODAS probe shards** | Same. 9.8 GB of raw shards | **RAW** — the pool the keeps were mined from | Keep or delete. Mining is done; this is the working set behind it |
| **JL-Corpus** | CC0. 2,400 utterances, 459 MB | **READY** — used as a calibration anchor, not as training audio | None. Correctly scoped as an instrument |
| **Expresso** | Meta, **CC-BY-NC-4.0**. 11,615 read-speech clips, 48 kHz, 1.8 GB | **BLOCKED** — see § The Expresso conflict | Owner call. Two ratified decisions disagree about it |
| **GLOBE V2** | CC0. Supersampled Common Voice, worldwide accents | **NOT PULLED** | Low priority. Nominal 44.1 kHz but true bandwidth is lower — keep away from the quality bar |
| **MLS English** | CC-BY-4.0, ~44.5 k h LibriVox | **BLOCKED** — 16 kHz | None. The sample rate disqualifies it for a 24 kHz fine-tune. Pretraining-scale only |
| **EmoV-DB** | NC behind a click-through agreement | **BLOCKED** | None. Agreement-walled NC stays fully excluded |
| **Emilia (original 101 k-h subset)** | CC-BY-NC-4.0 | **BLOCKED** | None. Never pulled — only the CC-BY YODAS portion was mined |
| **AniSpeech / Hailuo-derived** | MIT labels over scraped or closed-model-synthetic audio | **BLOCKED** | None. Provenance risk; clean lineage is the promise |
| **Owner's DRM-free audiobooks** | Copyrighted performances | **BLOCKED** | None. Hard line, three sanctioned private uses, settled |

### Derived by us

These are ours, built from the sources above or synthesized. None is in a training
filelist yet.

| Source | Origin | Status in our training | Next action |
|---|---|---|---|
| **sonora-expressive-registers** | Teacher renders (qwen · chatterbox · zonos · orpheus · moss_vg) + real keeps. **1,071 fold-eligible keeps** | **DERIVED** — the expressive corpus, ear-certified via `ratings.csv` | Close the last **+116** (Neutral +81, Documentary +35), then merge. Target 1,156 |
| **book-prose** | Standard Ebooks CC0 text → director-labelled banks. 21 books, 856 rendered wavs | **DERIVED** — the text shelf that feeds synthesis | None outstanding. The 4 owner-approved titles queued 2026-08-02 (+`conan-stories`) are ingested; delivery-v1 closed on them. ⚠ `book_ingest` hardcodes provenance "Standard Ebooks CC0" even for PG sources — false licence metadata in every derived clip's paper trail (A-M6, [todo.md §5](todo.md)) |
| **librivox-v2** | Force-aligned LibriVox audio (Uneasy Money). 1,366 clips / 3.13 h | **DERIVED** — better cut than v1, nothing staged from it | Stage when re-earning v1's 12 ear verdicts is worth it |
| **librivox-v1** | Same book, pre-fix cut. 664 clips | **DERIVED** — kept for its 12 valid ear verdicts | Retire once v2 is staged |
| **librivox-speech-v1** | Dickens *Speeches*, re-cut 2026-08-02. 79 clips | **DERIVED** — Speech lane | None; lane is at target |
| **Campaign dirs** | `delivery-v1-narration`, `newscaster-v1`, `teacher-ab-v1`, `revisit-v1`, `stress-v1/v2`, probes | **DERIVED** — render + audit workspaces | Keeps flow into sonora-expressive-registers; the dirs are working state |

---

## What a training run actually consumes today

One corpus. `configs/experiment/vat4_finetune.yaml` points at
`libritts_r_vat_v4.yaml`, whose filelists are **100 % LibriTTS-R** — as were v3c's and
v2's before it (verified 2026-08-02, zero non-LibriTTS rows). Every expressive clip we
have rendered and auditioned sits outside the training path.

That gap is the single most important fact in this file. The corpus we spend ear
time on and the corpus we train on have never been joined.

**v4 does not close it, and it is worth being precise about how little it moves.** The
8-wide derivation joins delivery labels from `ratings.csv`, but only **48 of 31,445 clips**
match — the `audit-markup-v0` rows, which are LibriTTS clips that happened to be
auditioned. The 1,189 delivery keeps the campaign produced are in
`sonora-expressive-registers`, still outside the training path. So v4 widens the wire and
leaves the gap exactly where it was: **Phase 1 is what closes it.**

## The VAT corpus lineage

All five derive from the same LibriTTS-R `train-clean-100` audio; they differ in
labels, phonemes and split.

| version | rows | what changed | trained? |
|---|---|---|---|
| `libritts_r_vat` (v1) | 30,351 | first VAT derivation | yes — `derisk_energy` |
| `_v2` | 30,351 | soft-json harshness repair; independence gate PASS | **yes — `vat3_finetune` ep099** |
| `_v3` | 31,445 | `MAX_SECONDS` 16 → 22 (owner 2026-08-01) | no |
| `_v3b` | 31,445 | apostrophe-clean IPA (v1–v3 carry ~6.4 % poisoned rows) | no |
| `_v3c` | 31,445 | **per-clip hash split** — 30,485 train / 960 val | yes — `vat3c_finetune` ep099, 2026-08-06, **retired as a regression** |
| **`_v4`** | 31,445 | **8-wide** (V/A/T + one-hot delivery), D-L2's corrected z guard, D-M4 homographs ON | **no — the one to train on.** `vat4_finetune` |

Both blockers are cleared: `configs/experiment/vat3c_finetune.yaml` points at it, and
`data_statistics` were re-measured **inside the training container** on the 30,485-clip
split (`mel_mean −5.525067 / mel_std 2.388571`). The v2 values were measured on the
29,441-clip split and must never be inherited silently — the delta is a constant shift on
every normalised mel.

**What the v3c run proved** (2026-08-06): the phoneme fix was worth **−1.411% dur** vs
matched controls and **nothing audible** — the `op_g2p` repair alone closed it, and the
run was not needed. Scored the same day against the never-trained holdout it turned out
to be **worse than not running it at all**: +0.0164 against its own warm start, every
loss term worse, better on only 39.1% of 5,463 unseen clips, and +0.0443 worse on v3c's
own val split too. **`vat3_finetune` ep099 is the base going forward, not `vat3c`.**

Read that as a verdict on the *run*, not on `_v3c`: the corpus is correct and is what
Phase 1 builds on. What it says is that ~30k clips a model is already fit to has no
100-epoch's worth of signal left in it. Full measurements and the plan that follows:
[quality-gap-plan.md § 0a](quality-gap-plan.md).

## The Expresso conflict — needs the owner

Two ratified decisions point opposite ways and both are in the notes:

- **2026-07-26**, folded into `dataset-landscape.md`: bare NC with no click-through
  agreement executed is *risk-accepted for training use* under the two-fence
  ruling; tainted-lineage bookkeeping applies and the ship/don't-ship call defers
  to promotion time.
- **2026-08-01**, the no-patent / fully-Apache-2.0 posture: the corpus bar is
  **unrestricted open redistribution** — explicitly *stricter* than the old
  commercial test, because there is no longer a copyleft or commercial side that
  could legitimately absorb a restricted asset.

CC-BY-NC-4.0 does not clear unrestricted open redistribution. The later decision
appears to supersede the earlier one, which would move Expresso from
risk-accepted to excluded. **Recorded as unresolved rather than decided here**:
it is 1.8 GB of the best expressive material available to us, and the call is the
owner's. Nothing has trained on it either way.

## Text sources (not audio)

Synthesis needs text, and it comes from a different wall: **Standard Ebooks (CC0)**
and **Project Gutenberg** (header stripped), routed by
`scripts/synthesis/book_router.py` with `books_ledger.json` as the record of
record. LibriVox URLs route to the force-align lane instead. Full lane docs:
`book-prose-lane.md`.

## Conventions that apply to every source here

- **License wall**: CC-BY-4.0 or freer, no exceptions, and no non-commercial tool
  anywhere in the lineage either.
- **Force-align first**: canonical text aligned to real audio; ASR is
  fallback-only; synthesize only where no real audio exists.
- **24 kHz quality bar**: disqualifies MLS-English at 16 kHz regardless of scale.
- **QC gate is mandatory** after every generation pass.

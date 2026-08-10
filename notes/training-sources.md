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

Last verified against disk and configs: **2026-08-09** (v5 landed 08-08; Emilia moved READY -> TRAINED, expressive-registers rescoped). The *order* in which the READY /
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
| **LibriTTS-R (the other 90%)** | Same, CC-BY-4.0. Full set ≈ 585 h / ~2,400 speakers | **NOT PULLED** — we hold `train-clean-100` only | **The 10×, and it is UNGATED as of 2026-08-09** — Emilia's **+36%** (not the planned +43%) moved the clean holdout by **−0.0606**, so the corpus is data-limited and rung 3 proceeds. It sequences after rung 2, not behind another gate — [quality-gap-plan.md § Phase 1](quality-gap-plan.md) |
| **`parler-tts/libritts-r-filtered-speaker-descriptions`** | CC-BY-4.0. LibriTTS-R + per-utterance natural-language annotations (pace, pitch, expressivity, quality) | **NOT PULLED** | Evaluate against our derived V/A/T. Cleared as the "labeling shortcut" and we hand-built that substrate instead |
| **`cdminix/libritts-r-aligned`** | CC-BY-4.0. LibriTTS-R + forced alignments and per-token pitch/energy/duration | **NOT PULLED** | Same evaluation. Its prosody measures overlap what `derive_vat_corpus.py` computes; worth knowing whether it is better |
| **Hi-Fi TTS (v1)** | NVIDIA, CC-BY-4.0, LibriVox-sourced. ~292 h / 10 speakers / 44.1 kHz. **40 GB of parquet WITH audio** | **RAW** — largest corpus on disk, never touched | Convert parquet → wav + filelist. Role: casting anchors and speaker-consistent long-form prosody (few voices, deep hours) |
| **HiFiTTS-2** | NVIDIA, CC-BY-4.0, LibriVox-sourced. 36.7 k h @22 kHz / 31.7 k h @44 kHz, ~5 k speakers | **RAW — METADATA ONLY.** 16.6 GB of manifests and chapter lists (8.9 GB 22 kHz + 7.7 GB 44 kHz). NVIDIA does not redistribute the audio | Decide before any download: **2.8 TB for 22 kHz, 4.0 TB for 44 kHz**, fetched from LibriVox via NeMo-speech-data-processor. A bandwidth-filtered subset is the realistic version |
| **VCTK** | CSTR Edinburgh, CC-BY-4.0 (verified 2026-07-19). 110 speakers / ~44 h / 48 kHz studio | **RAW** — still `VCTK-Corpus-0.92.zip`, 4.3 GB, unextracted | Unzip and build a filelist if the casting/accent grid needs it. Neutral read sentences — no conveyance value |
| **Emilia-YODAS keeps** | Amphion, CC-BY-4.0 (**YODAS subset only**). Mined and resampled to 24 kHz, 5.4 GB | **TRAINED** (2026-08-08) — merged as **10,997 keeps** of `libritts_r_emilia_vat_v5` — **10,653 train + 344 val** — taking the corpus to **41,138 train / 1,304 val** (42,442 total), 78.5 h, 2,500 speakers. The planned 13,141 became 10,997 on contact: 1,676 dropped on digits (D-M3) and 468 on the shared `ASR_MAX_WER`. ⚠ **ALWAYS SAY WHICH SPLIT.** 10,653 and 10,997 are both correct and measure different things — `derivation_report.json` carries `emilia.kept` **10,997**, `emilia.train` **10,653**, `emilia.val` **344**. Quoting one against the other has already produced a wrong "correction" once | None. ⚠ It did **not** bring real newscaster/documentary audio — the mining verdict expected that and it did not survive: **100% of Emilia rows carry blank delivery.** Those lanes still have no real audio from any source |
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
| **sonora-expressive-registers** | Teacher renders (qwen · chatterbox · zonos · orpheus · moss_vg) + real keeps. **846 rows to append** — 1,004 eligible of 1,279 (902 delivery-labelled + 102 blank; scoped 2026-08-09) less 158 duplicate-audio rows (2026-08-10), leaving 836 labelled + 10 blank | **DERIVED** — the expressive corpus, ear-certified via `ratings.csv`. **The only delivery-label signal in existence:** v5 carries 48 labelled rows in 42,442 (0.11%), and Documentary/Newscaster/Speech have zero | Merge as **v6**. Eligibility filter and the accepted provenance confound: [quality-gap-plan.md § Rung 2](quality-gap-plan.md) |
| **book-prose** | Standard Ebooks CC0 text → director-labelled banks. 21 books, 856 rendered wavs | **DERIVED** — the text shelf that feeds synthesis | None outstanding. The 4 owner-approved titles queued 2026-08-02 (+`conan-stories`) are ingested; delivery-v1 closed on them. ⚠ `book_ingest` hardcodes provenance "Standard Ebooks CC0" even for PG sources — false licence metadata in every derived clip's paper trail (A-M6, [todo.md §5](todo.md)) |
| **librivox-v2** | Force-aligned LibriVox audio (Uneasy Money). 1,366 clips / 3.13 h | **DERIVED** — better cut than v1, nothing staged from it | Stage when re-earning v1's 12 ear verdicts is worth it |
| **librivox-v1** | Same book, pre-fix cut. 664 clips | **DERIVED** — kept for its 12 valid ear verdicts | Retire once v2 is staged |
| **librivox-speech-v1** | Dickens *Speeches*, re-cut 2026-08-02. 79 clips | **DERIVED** — Speech lane | None; lane is at target |
| **Campaign dirs** | `delivery-v1-narration`, `newscaster-v1`, `teacher-ab-v1`, `revisit-v1`, `stress-v1/v2`, probes | **DERIVED** — render + audit workspaces | Keeps flow into sonora-expressive-registers; the dirs are working state |

---

## What a training run actually consumes today

**Two datasets, since 2026-08-08.** `configs/experiment/vat5_finetune.yaml` points at
`libritts_r_emilia_vat_v5.yaml` — 41,138 train / 1,304 val, **78.5 h, 2,500 speakers** —
which is v4's LibriTTS-R rows verbatim plus **10,997 Emilia-YODAS keeps**. Every corpus
before it was 100% LibriTTS-R (v3c's and v2's verified 2026-08-02, zero non-LibriTTS rows).

**The ear gap is still open, and v5 does not close it either.** The corpus we spend ear
time on and the corpus we train on have still never been joined: v5's new material was
mined on ACOUSTICS and nobody has heard a clip of it. That gap remains the single most
important fact in this file.

- **v4** widened the wire and nothing else. Its derivation joins delivery labels from
  `ratings.csv` and only **48 of 31,445 clips** match — the `audit-markup-v0` rows, which
  are LibriTTS clips that happened to be auditioned.
- **v5** adds volume, and volume is precisely what Phase 1 #1 exists to test. Its Emilia
  rows are delivery-`unknown` by construction, so the delivery count is still 48.
- **Phase 1 #2** is the one that closes the gap: `sonora-expressive-registers`, **846 rows
  to append** (owner-scoped 2026-08-09 at 1,004 eligible; 158 duplicate-audio rows removed
  2026-08-10), still outside the training path.

  ⚠ **Six sizes for this corpus circulate, and they are not contradictions — they are six
  different filters.** In order:
  **1,279** total keeps → **1,189** delivery-labelled (the delivery-v1 campaign's own count)
  → **1,156** the *planned* close of +116 that the owner's scoping superseded → **1,071** an
  intermediate ear-certified count → **1,004 eligible before dedup**: 1,279 minus licence
  and standing-policy exclusions (VibeVoice/Dia 133 benched, moss85 83, longcat 45, higgs3 8
  NC) → **846 appended**, the v6 scope: 1,004 minus 158 rows whose audio is already in v5.
  **Quote 846 for v6 and name the filter whenever another number is used.**

So the order is deliberate — v5 asks "does volume move quality at all?" with material that
costs no ear time, and #2 spends ear time only if the answer is yes.

## The VAT corpus lineage

v1–v4 all derive from the same LibriTTS-R `train-clean-100` audio and differ only in
labels, phonemes and split. **v5 is the first that adds audio.**

| version | rows | what changed | trained? |
|---|---|---|---|
| `libritts_r_vat` (v1) | 30,351 | first VAT derivation | yes — `derisk_energy` |
| `_v2` | 30,351 | soft-json harshness repair; independence gate PASS | **yes — `vat3_finetune` ep099** |
| `_v3` | 31,445 | `MAX_SECONDS` 16 → 22 (owner 2026-08-01) | no |
| `_v3b` | 31,445 | apostrophe-clean IPA (v1–v3 carry ~6.4 % poisoned rows) | no |
| `_v3c` | 31,445 | **per-clip hash split** — 30,485 train / 960 val | yes — `vat3c_finetune` ep099, 2026-08-06, **retired as a regression** |
| `_v4` | 31,445 | **8-wide** (V/A/T + one-hot delivery), D-L2's corrected z guard, D-M4 homographs ON | smoke only — superseded by v5 before it ran |
| **`libritts_r_emilia_vat_v5`** | **42,442** | **+10,997 Emilia-YODAS keeps**; 247 → 2,500 speakers; 51.3 → 78.5 h | **YES — trained 2026-08-08/09.** `vat5_finetune`, 48 epochs, holdout-scored, **`ep019` selected** (converged by epoch 9). The warm-start donor for v6 |

**v5 is a merge, not a derivation** (`scripts/merge_emilia_corpus.py`), and the two halves
carry labels on different scales *on purpose*. LibriTTS keeps its per-speaker z; Emilia
gets a **global anchor** against v4's own distribution, because 2,408 speakers at a median
of 3 clips have no per-speaker statistic and re-centring a tail-selected clip on its own
mean would hand 756 one-clip speakers a label of exactly 0.0 — clips selected FOR being
extreme, trained as neutral. Owner's option 1, 2026-08-08. The full argument is in
`scripts/anchor_emilia_labels.py`; the semantic cost (a global anchor leaves some speaker
identity in the affect channels) is stated in the data config rather than hidden.

2,144 of the 13,141 keeps did not make it: **1,676 carry digits** (D-M3 — the tokenizer
deletes them, and "1 Chronicles" is *First* Chronicles, so normalising is a guess not a
table) and **468 exceed the shared `ASR_MAX_WER` of 0.35** against their YODAS caption.
`n_spks` is 2,500 rather than the 2,655 the plan predicted, because 155 speakers lost every
clip — derived from the corpus, never assumed.

`data_statistics` are re-measured **inside the training container** for every version whose
audio or split moves, and v5 is the version that proves the rule was not pedantry:

| | mel_mean | mel_std | split |
|---|---|---|---|
| v2 | — | — | 29,441 clips |
| v3c / v4 | −5.525067 | 2.388571 | 30,485 (v4 inherited these, against a verified set-equal split) |
| **v5** | **−5.683762** | **2.709323** | 41,138 |

v2 → v3c moved the mean by 0.0203. **v4 → v5 moves it by 0.1587 and the std by 0.3208** —
adding 27.2 h of podcast/YouTube audio to 51.3 h of studio reading is a different mel
distribution by construction. Inheriting there would have put a constant offset and a 12%
scale error on every normalised mel in the corpus, on the one run whose entire purpose is
to read a quality difference.

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

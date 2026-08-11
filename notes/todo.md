# Open items

What is actually open. **Landed items are deleted, not struck** — git history is the
archive — and a section is deleted when it empties. Letter codes (`F-H2`, `D-C1`, `A-M2`…)
are kept where they appear so the git-history record stays joinable.

Sequencing lives in [quality-gap-plan.md](quality-gap-plan.md); this is the item list, not
the plan.

## 0 · Owned follow-ups, blocked on sequencing

- [ ] **`derive_a_channel_stats.py`: `{e:+.0f}` collapses two design cells into one column
      heading.** On the `{−1, 0, +0.5, +1}` energy design, `+0.5` and `0` both print as `A=+0`,
      so the section shows two adjacent cell-mean columns with identical headings — unreadable
      rather than merely ugly. Found by the review lane on PR #62 and deliberately left
      unfixed there: pre-existing, and outside that PR's reviewed range.
      **Owner: the lab-manager session. Blocked until #62 merges** — that PR is currently the
      sole editor of the file, and a concurrent edit re-creates the same-file collision class
      that cost two rounds on `pyproject.toml`. The defect is unreachable on the artifact
      anyone runs, so waiting costs nothing.

## 1 · Before the next training run

- [ ] **Score it on the holdout, not on `loss/val_epoch`.** A standing rule, listed here
      because it is the step most easily skipped: `scripts/score_holdout.sh` against
      `data/libritts_r_holdout_devclean`, ~100 min for eight checkpoints on an idle card.
      MLflow val is permanently unusable for cross-run comparison (historical split
      contamination), and there are **two** scale breaks in logged `diff_loss` — 2026-08-01
      bucketing and 2026-08-06 masking. Deciding a run by its curve is deciding it by an
      artefact. Stratify by channel with `scripts/stratify_holdout_sweep.py`; the aggregate
      hides a channel-specific regression.

## 2 · QC / audit / staging

- [ ] **A pause-POSITION measure.** `worst_pause` knows how long a silence is and nothing
      about whether it falls where a reader would breathe. A 2.1 s pause at a paragraph
      break is fine; a 1.4 s one mid-clause is not. This is a new measure, not a new
      constant — the phrase boundary is already available from the CTC alignment A-M2
      fixed. Not urgent: the advisory band catches these today at the cost of an audition.
      ⚠ n = 2, and both from one voice in one campaign.

## 3 · Conditioning axes — one approved, one open

The approved position is canon in [ARCHITECTURE.md § 1](ARCHITECTURE.md) and is not
repeated here.

- [ ] **Embodiment bank across the current five survivor engines**, folded into a narration
      round rather than run as its own campaign. It settles whether the mean of 3.71 is the
      MODE or was just the old engines — of the four engines in those 28 rows (qwen 8,
      longcat 8, libritts-r 10, scm-spike 2) only qwen is still in the portfolio, so we have
      almost no result rather than a bad one.
- [ ] **Span-FiLM phase:** delivery varying across an utterance on the same zero-init path,
      expanded through the duration alignment. The 17 keeps are the pilot set. Its
      prerequisite is met — the clip-level channel shipped 2026-08-07 — so this is sequenced
      behind evidence (the bank above) rather than behind plumbing.

⚠ **Standing rule, not an open item:** embodiment clips stay **delivery-blank** and outside
the 50/30/8/6/6 percentages, so a future balance pass cannot "fix" the blanks. Enforced at
the encoding — `matcha.delivery` refuses any label outside the closed five, so an
"Embodiment" lane cannot arrive by accident, only by a contract change.

### The categorical emotion block — OPEN DECISION, not scheduled

*Isn't 3 + 8 the standard?* It is — dimensional-plus-categorical (3 continuous VAD + 8
Plutchik or 6 Ekman) is a common shape in emotional TTS. We do not have one, and whether we
should is genuinely undecided. Recorded so the answer is a decision rather than an omission.

**Why not today.** Affect is carried entirely by the three continuous channels. The
categorical vocabulary that exists — the 47-label register lexicon — is **Director-side by
contract** (the Actor never sees a register id) and compiles to V/A/T + delivery + text
before the model sees anything. Delivery's five lanes are *modes of address*, not emotions.

**The case for it is not weak.** That compile-down is lossy in a documented way: valence is
the one dimension acoustics alone cannot reach — a sob and a laugh have similar energy — and
categorical affects can sit at nearly the same point in a low-dimensional continuous space
(anger and fear are close in V/A; only the discrete label separates them). Valence is exactly
the channel that **FAILED its standing test** on vat3-24k.

- [ ] **Gate on Phase 1.** That failure is diagnosed today as a **corpus-label limit, not
      architectural** — and Phase 1 is the experiment that separates them. **If valence still
      fails after the corpus grows, the representation is the suspect and this becomes a real
      spike; if volume moves it, the question is answered and this stays closed.** Do not
      spike it before that read: a categorical block trained on a corpus that cannot support
      the continuous channel would confound the two.

If it is ever spiked, two things are settled in advance: it **APPENDS** as channels 8+ on
the same zero-init FiLM path, so `unknown` stays all-zero and every existing checkpoint and
filelist keeps its meaning (**reordering is the one edit that must never happen** — position
is the wire format); and the labels **do not exist yet in that form** — delivery shipped with
**1,189 labelled keeps; an 8-way emotion block starts at zero**, and folding the 47-label lexicon
down is Director-authored *intent* rather than an ear verdict on the render. Costing that
fold is part of the spike, not a preliminary.

## 4 · The score scale has a ceiling, pinned at human audio

**The 1–5 scale is saturated, and that is a finding about the INSTRUMENT.** Measured over
the corpus: 62 texts are rendered by ≥3 engines with ear verdicts (478 clips), and **46 of
those 62 groups have ≥3 DIFFERENT engines all scoring 5** — 229 clips tied at the ceiling.
On three-quarters of the controlled comparisons the scale cannot separate them. `librivox`
— real human recordings — means **5.00**, so the top of the scale is "indistinguishable
from a human read" and six synthetic engines are sitting on it.

⚠ **NEVER RANK ENGINES BY MEAN SCORE.** They cluster in a 0.14-point band and the scale
places chatterbox *above* the engine the owner calls the gold standard — an inversion
produced by compression, not by quality. Mean-score tables anywhere in these notes read as
"all of these clear the bar", never as an order. Tier calls rest on keep RATE and defect
characterisation, which are binary and survive this.

**It is a corpus problem, not only a bookkeeping one.** Sonora trains on keeps, so if
"magnificent" and "acceptable" both label 5, the corpus's ceiling is set by the weakest 5 in
it — the "clinging to a teacher's style" risk arriving through the labels instead of the
engine mix.

- [ ] **A forced-ranking pass over the 46 tied groups** (229 clips). Same text, ≥3 engines,
      all at 5 — rank within the group instead of scoring absolutely. It is the protocol the
      owner already used informally, needs no scale change, invalidates no existing verdict,
      and yields the comparative signal the absolute scale structurally cannot. Bounded: 46
      sittings of 3–5 clips.

⚠ **Anchor exemplars are the EAR's, never a measure's.** Nothing computes one — a computed
anchor re-anchors the scale to whatever the measure already believes. The bar ships with one
entry (`tab_14_resentment_QWN`) and the other four **unset rather than guessed**. State
lives in `anchors.json`, never in a clip column: the Qwen/VibeVoice A/B parked prior scores
in `note`, the app overwrote them, and 17 of 33 were lost.

## 5 · Parked dataset decisions

SSOT is [training-sources.md](training-sources.md) — not duplicated here. Headlines: the
Expresso two-ruling conflict (owner call), the other 90% of LibriTTS-R, Hi-Fi TTS
parquet→wav conversion, VCTK unzip, HiFiTTS-2's 2.8 TB fetch decision, and retiring
`librivox-v1` once `librivox-v2` is staged and its 12 ear verdicts re-earned.

# Open items

What is actually open. **Landed items are deleted, not struck** — git history is the
archive — and a section is deleted when it empties. Letter codes (`F-H2`, `D-C1`, `A-M2`…)
are kept where they appear so the git-history record stays joinable.

Sequencing lives in [quality-gap-plan.md](quality-gap-plan.md); this is the item list, not
the plan.

## 1 · Before the next training run

- [ ] **Score it on the holdout, not on `loss/val_epoch`.** A standing rule, listed here
      because it is the step most easily skipped: `scripts/stages/score_holdout.sh` against
      `data/libritts_r_holdout_devclean`, ~100 min for eight checkpoints on an idle card.
      MLflow val is permanently unusable for cross-run comparison (historical split
      contamination), and there are **two** scale breaks in logged `diff_loss` — 2026-08-01
      bucketing and 2026-08-06 masking. Deciding a run by its curve is deciding it by an
      artefact. Stratify by channel with `scripts/tools/stratify_holdout_sweep.py`; the aggregate
      hides a channel-specific regression.

## 2 · QC / audit / staging

- [ ] **A pause-POSITION measure.** `worst_pause` knows how long a silence is and nothing
      about whether it falls where a reader would breathe. A 2.1 s pause at a paragraph
      break is fine; a 1.4 s one mid-clause is not. This is a new measure, not a new
      constant — the phrase boundary is already available from the CTC alignment A-M2
      fixed. Not urgent: the advisory band catches these today at the cost of an audition.
      ⚠ n = 2, and both from one voice in one campaign.

## 3 · Conditioning axes — one approved, one open

The approved position is canon in [ARCHITECTURE.md § 1](../docs/ARCHITECTURE.md) and is not
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
  - ⚠⚠ **BLOCKED 2026-08-25: THE GROUPING RULE THAT PRODUCED "46" IS NOT RECORDED, AND THE
    COUNTS DO NOT REPRODUCE.** `ratings.csv` has no text column, so "same text" has to be
    derived from the `id`, and nothing in the repo does it — the analysis was ad hoc and only
    its outputs survived. Four readings tried against today's file, none matching 62 texts /
    478 clips / 46 tied / 229 clips:

    | grouping | texts ≥3 engines | clips | tied groups | ceiling clips |
    |---|---|---|---|---|
    | `id` minus last token | 66 | 308 | 19 strict · 30 loose | 67 · 136 |
    | campaign + `id` minus last | 58 | 276 | 14 strict | 50 |
    | text token, cross-campaign | 72 | 695 | 15 strict · 63 loose | 75 · 484 |

    *strict* = every clip in the group scored 5; *loose* = ≥3 distinct engines each scored 5.
    **The likely cause is drift, not error: `ratings.csv` is a LIVE WRITER** and those counts
    are a snapshot of it — 57 clips have been relabeled and 253 dropped since. So "46" was
    probably true when written and is not true now.
    ⚠ **Do not prepare a set by picking the closest number.** The cost of being wrong is 46
    sittings of owner time spent measuring something other than what this line claims. The
    unblock is one owner sentence naming the grouping rule, after which the rule goes in a
    script — not in prose — so the count is derived and cannot drift again. Same failure
    class as the ep010 probe: the outputs survived, the design did not
    (`probe_delivery_intercept.py` was written for exactly that reason).

⚠ **Anchor exemplars are the EAR's, never a measure's.** Nothing computes one — a computed
anchor re-anchors the scale to whatever the measure already believes. The bar ships with one
entry (`tab_14_resentment_QWN`) and the other four **unset rather than guessed**. State
lives in `anchors.json`, never in a clip column: the Qwen/VibeVoice A/B parked prior scores
in `note`, the app overwrote them, and 17 of 33 were lost.

## 4b · Dataset Listening app — PARKED 2026-08-25, findings recorded so the work starts informed

Renamed from "Auditions" that day (owner): an audition is a casting decision, and most of
what this surface does is vet output. Live at `listen.ai-lab-0.mcfarlin.family`; the old
host 301s. The **service, container, deploy target and `/data/services/audition` path are
still `audition`** — renaming those moves the deployed copy and its MIRRORS entry, so it is
one coordinated change, not a find-and-replace.

⚠ **The first pass renamed the RUNNING SURFACE and stopped there** (`app/main.py`,
`app/static/index.html`, the Caddy route), which left `audition/README.md` making a false
claim about a file it names — line 106 told a reader to look in `index.html` for a
`Dataset Auditions` tile that had just been renamed out of it. **#304**, and the enumeration
above is what made it look deliberate: a README is not a service, a container, a deploy
target or a `/data` path, so it fell in the gap between "renamed" and "deliberately kept".
Swept 2026-08-26 across **14 files** — `audition/README.md`, `AGENTS.md`,
`notes/{training-sources,book-prose-lane,casting-attribute-norms-brief}.md`,
`docs/{markup-schema-brief,direction-interface-brief}.md`,
`scripts/teacher_audition/README.md`, `scripts/lib/synth_common.py`,
`scripts/tools/{tag_spike,label_expressive_registers,measure_expressive_registers}.py`,
`scripts/stages/register_audition.py`, `tests/test_ratings_transaction.py` — plus the
`audition.ai-lab-0` URLs inside them. **Filenames and directory names were NOT touched**
(`register_audition.py`, `scripts/teacher_audition/`): those are plumbing, and this list is
the surface. `audition/app/main.py`'s docstring keeps the old name on purpose — it is the
record of the rename, not an instance of it.

⚠ **THE GAP THAT BIT THREE TIMES IN ONE SESSION: there is no COMPARATIVE mode.** Every
input is absolute and per-clip, and the app only serves clips registered in `ratings.csv`.
So the delivery ear test, the forced-ranking pass (§4) and blind A/B are each *N clips
judged against each other*, each unreachable through the app, and each falls out to a
directory on the box — which the owner does not work from. The stopgap is a plain file
route at `listen.../probes/*` (marked in the Caddyfile to be **deleted rather than grown**).
⚠ **Do NOT close the gap by registering probe clips in `ratings.csv`** — it is the dataset
ledger and a live writer; a comparative probe is not a dataset row.

**Two more gaps, from reading the 466 human-written notes:**

* **Direction adherence has nowhere to go, and it is the most common thing the notes say** —
  "Younger than requested", "Everything is accurate except gender", "not British at all",
  "Poor instruction following". The app records what was HEARD (gender/age/accent) and never
  whether it MATCHED what was asked. That comparison happens in the owner's head and lands
  as prose. It is also a first-class variable: some engines have no text-instruction slot at
  all, so adherence is what separates them.
* **Audio quality is fused into a prosody score.** `score` is vocals/prosody only by rule, so
  "slight scratchiness but not enough to detract" has no home and the owner computes an
  override in prose. A good read in a bad recording and a flat read in a clean one collapse
  to one number.

**Under all three: `note` does at least four jobs** — machine markers, the script text, the
direction text, and the ear's observation. ⚠ **That is why term-frequency analysis of it is
not currently possible, and two attempts to do it were WRONG before this was noticed** (the
regexes matched `"verb"` inside `"Whatever"` and the `[was: breathy]` register-rename tag).
Do not re-apply the "17 notes / 4 campaigns" precedent that justified `delivery` until the
column is split; the denominator is not what it looks like.

**Fill rates, 1,802 rows:** status 100% · register 94% · delivery 91% · gender 87% ·
score 87% · **age 49%** · **accent 49%** · note 39%. Accent is 76% `US - General` where
filled and age is 87% adult+middle-aged — near-constant as always-on dials, decisive only in
probes. They read like campaign-level properties wanting an occasional per-clip override.

**Smaller, unrelated:** the default view is `todo` (unrated) and every clip is rated, so the
app opens on an empty list with no hint that 1,802 clips sit behind the "Rated" chip. It
reads as broken. A fallback or an empty-state message is cheap.

## 5 · Parked dataset decisions

SSOT is [training-sources.md](training-sources.md) — not duplicated here. Headlines: the
Expresso two-ruling conflict (owner call), the other 90% of LibriTTS-R, Hi-Fi TTS
parquet→wav conversion, VCTK unzip, HiFiTTS-2's 2.8 TB fetch decision, and retiring
`librivox-v1` once `librivox-v2` is staged and its 12 ear verdicts re-earned.

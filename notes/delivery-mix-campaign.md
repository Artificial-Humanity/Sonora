# delivery-v1 — the dataset-assembly rebalance campaign

_Owner ratified 2026-07-30: the **book-actor mix** — Dialogue 50% / Neutral 30% /
Documentary 8% / Newscaster 6% / Speech 6%. This note is the SSOT for the push._

## Baseline (measured 2026-07-30, after cleanup)

Fold-eligible certified keeps (audit-* campaigns excluded — they are agreement-rate
instruments, never folded): **642**, after retiring **52 fragment keeps** via the app's
drop action (completeness gate re-derived over every bank/manifest text; the old "21"
figure predated the quote-pilot duplicate sweeps — two fragment lines had been rendered
by many engines). 14 `embodiment` keeps stay delivery-blank on purpose (genuinely
narration+quote) and sit outside the percentages.

| lane | have | want | gap |
|---|---|---|---|
| Dialogue | **500** | 500 (anchor, 50%) | 0 |
| Neutral | 106 | 300 | **+194** |
| Documentary | 7 | 80 | **+73** |
| Newscaster | 8 | 60 | **+52** |
| Speech | 7 | 60 | **+53** |

Target corpus: **~1,000 certified keeps**. New keeps needed: **+372** ≈ **420–450
renders** at measured keep rates (survivors' Neutral keep 94%; assume ~85% for the
untested minor lanes). Separate story: `emilia_kept_24k` holds 13,141 real clips for the
v1.1 volume lane — its delivery composition should be estimated by sampling, never
ear-tagged exhaustively; per the mining verdict it likely holds the only REAL
newscaster/documentary audio we have.

## Text supply per lane

- **Neutral / Documentary — already on the shelf.** ~500 narration (`_nar_`) lines are
  minted across 17 ingested books in `/data/model-training/datasets/book-prose/`; only
  baseball-club, brewsters-millions and apple-cart were ever rendered (~90 lines), so
  ~440 unrendered narration lines exist. Use distinct lines per lane (Documentary is a
  RENDER style on narration text — nonfiction flavor is nice, not necessary). ⚠ The
  existing bank `direction` fields predate the survivor portfolio AND the narration
  skill-file lanes — every line must be **re-directed** (Gemma two-pass, current skill
  files) before rendering. Do not render stale directions.
- **Speech — source real oratory.** Public-domain speeches (PG has whole collections:
  world-famous orations, Lincoln, Henry, Webster, Bryan). Complete-utterance excerpts,
  ≥92 chars (the Zonos rate-16 floor), through `is_complete_utterance()` like every
  minted line. The 7 existing Speech keeps are stress-batch oratory — exactly this mode,
  already proven on all four stress engines.
- **Newscaster — author bulletin copy.** News-register lines written for the lane
  (precedent: the register-lexicon lines were authored). Period-neutral bulletins;
  remember the standing terminology rule (newsreel ≠ Mid-Atlantic ≠ old-radio EQ), and
  that Newscaster is a VOCAL style, never a medium description (no "as heard on the
  radio" — Qwen/MOSS rule).

## Engine routing (survivors only; VV/Dia stay set aside)

| lane | engines | notes |
|---|---|---|
| Neutral | zonos · qwen · moss_vg · chatterbox · orpheus | zonos first choice (numeric, repeatable: pitch_std 20–35 / rate 13–15 / no emotion); skill-file lanes are provisional — first ~20 renders per engine are the narration audition |
| Documentary | zonos (30–45 / 12–14) · chatterbox (0.4/0.5) · qwen · moss_vg | |
| Newscaster | zonos (35–45 / 15–16) · qwen (vocal style) · moss_vg (anchor persona) | |
| Speech | chatterbox · orpheus · qwen · zonos | stress-proven at high arousal; expect the known stochastic defects — reroll, don't re-cast |
| (Dialogue) | closed for this campaign | at target; book dialogue lines keep accruing via the normal book lane afterward |

LongCat: excluded until the affect-transfer experiment passes (standing rule).

## Discipline carried over

- Per-narrator identity: pin ONE reference (zonos/chatterbox) or freeze the persona
  clause / voice keys (moss_vg/qwen) per book — a book's narration is one voice.
- Gender: pool skews Female (391/280) — bias narrator casting male-ward to converge.
- Audit: **risk-stratified sampling** (owner 2026-07-30: "I can't realistically audition
  every processed file"). `pick_audit_subset.py` queues the high-risk subset and parks
  the rest as status `deferred` (new app filter). Soundness rests on the frozen-voice
  group design: one voice per (book, lane) means systematic defects are perfectly
  correlated within a group — hearing a few clips certifies the group; unheard clips
  carry only the stress-bounded per-clip stochastic risk. Certification counts by
  engine risk: moss_vg 4/group (no stress bounds), orpheus 3 (1/20 split), others 2
  (0/20); plus every objective-QC-flagged clip; plus a seeded 10% tail (sampled per
  engine, so a thin tail can't be diluted by other engines' share of the batch). A group
  with any drop gets its deferred clips promoted back to the queue. Certified groups' unheard
  clips ride into the fold via `group_certification.json` with provenance
  `owner_audit=group-sampled` — ratings.csv scores stay ear-only, and the certified
  register lexicon never counts unheard clips. Expect ~1/3 of a first-of-its-kind batch
  in the queue, dropping toward ~15% once a lane's voices are proven.
- **Render allocation is PER ENGINE, and a tier is a tag — not a bucket** (owner
  2026-07-31). The two questions are separate: `ref_select.ENGINE_MIX` says how much
  each engine RENDERS, `pick_audit_subset.TIERS` says how much gets AUDITIONED. The
  owner caught why they must not be fused: the standard tier holds exactly one engine,
  so a "50/30/20 by tier" split hands orpheus a larger share (30%) than either trusted
  engine (25% each) — the thinnest evidence base taking the biggest slice. Before this
  there was nowhere in the tree for allocation to live, which is the real defect: the
  round-1 mix (52% trusted / 9% standard / 39% scrutinized) was never decided, it
  accumulated.

  | engine | tier | share | audited | note |
  |---|---|---|---|---|
  | qwen | trusted | 27.5% | 11% | earned across many campaigns |
  | chatterbox | trusted | 27.5% | 11% | **provisional** — 38 clips, one campaign; SPLIT is ear-only |
  | zonos | scrutinized | 20% | **100%** | only engine with numeric prosody dials |
  | orpheus | standard | 15% | 27% | 5% rests on n=19; earns more by surviving a round |
  | moss_vg | scrutinized | 10% | **100%** | overlaps qwen's niche at 24% vs qwen's ~1% |

  Shares are ordered by measured PRODUCTION failure rate with probe/stress campaigns
  excluded — those are adversarial by construction and libel every engine (chatterbox
  reads 0% production but 35% probe). On a 342-clip batch this is **137 auditions
  versus the 195 actually done in round 1**: less listening, and nothing rendered by a
  scrutinized engine is wasted, because all of it is heard.
- **Every QC failure is auditioned, at every tier, on every engine** (owner
  2026-07-31). `qc_gate` now runs as a mandatory step in `synth_bank.sh` — if it fails,
  the batch is NOT registered. Findings attach as a note ("transcribed only 20 words of
  47 — CHECK THE END") and never change a clip's status: the instrument tells the ear
  what to look for, it does not decide. Vindicated on its first run — it caught a
  `deferred` **qwen** clip (`the-return_nar_0043_neu_QWE`, 52 words against a 37-word
  passage) that trusted-tier sampling would have folded unheard.
- **Trust tiers** (owner 2026-07-31): an engine with a measured track record moves to
  spot-check — 1 clip/group and a 3% tail instead of 2 and 10%. **qwen is the first**
  (`TRUSTED_ENGINES` in `pick_audit_subset.py`), on 37/37 keeps with zero drops in
  delivery-v1-narration, 19/19 all-5 in the round-1 narration audit, and 18/20 at mean
  4.94 in teacher-ab. Batch coverage for qwen: 25% → 10%; other engines unchanged.
  What does NOT relax is the QC-flag path — every flagged clip still reaches the ear on
  any engine, which is where the ear's ceded coverage is bought back (the round-1 flags
  predicted the actual failures). Trust is revocable: `certify` prints `TRUST BROKEN`
  and records `trust_revoked` in `group_certification.json` if a trusted engine drops a
  clip under spot-check coverage — the response is to pull the tier and re-`select`, not
  merely promote the one failed group.
- QC before audition: loudnorm + objective gate (4 s floor via Silero, WER deletions,
  DEAD/TRUNC flags) — the Silero cartoon-timbre blind spot means real-energy/zero-speech
  clips get REVIEW, not auto-drop.
- Render batches of ~100–150 so audition and reroll cycles interleave; register each
  batch via `register_audition.py` (guarded append).

## Why the lanes are uneven on purpose (owner 2026-08-01)

> "There will just be more dialogue and narration since it's the majority of actual
> content. This is fine since the final model will also read those things a majority of
> the time."

The 50/30/8/6/6 mix is **not a balancing target** — it is an estimate of what books
actually contain, and therefore of what the production model will spend its life reading.
Shaping the corpus like the job is the point; flattening the lanes would train for a
distribution that never occurs at inference.

Two consequences worth stating, because both have been mis-reasoned once already:

1. **Dialogue is the ANCHOR, not the surplus.** Measured 2026-08-01: Dialogue 579 of 919
   keeps (63%). That reads "over target" only if the other lanes are assumed fixed. Held
   as the 50% anchor instead, the corpus completes at **1158** and needs +80 Neutral,
   +35 Documentary, +61 Newscaster, +61 Speech — **and no further dialogue at all.**
   The remedy for an over-weight lane here is growth elsewhere, never reduction.
2. **The minority lanes are a CAPABILITY requirement, not a volume one.** Newscaster and
   Speech at 6% exist so the model can deliver them on request, not so it delivers them
   often.

**Open question the ratio cannot answer.** [[direction-contract-v2]] makes delivery the
4th FiLM channel (5 lanes + unknown≡zero). "What share of the corpus" and "how many
examples before a FiLM channel VALUE separates" are different questions, and only the
first is settled. If ~69 clips proves to be under the floor for a channel value to learn
a direction, the fix is to grow the corpus until 6% of it clears that floor — not to
reshape the mix away from what books are actually made of.

## Sequence

1. **delivery-v1-narration** — Neutral (+194→~230 renders) + Documentary (+73→~90):
   select unrendered `_nar_` lines, Gemma re-direction with current skill files, render,
   QC, audition.
2. **delivery-v1-speech** — source PG oratory, mint (completeness + ≥92 chars), direct,
   render (+53→~65).
3. **delivery-v1-news** — author bulletin lines, direct, render (+52→~65).
4. Fold, then recompute this table; publish tier per [[expressive-registers-dataset]]
   conventions (higgs3-NC keeps stay out of the CC-BY publish set).

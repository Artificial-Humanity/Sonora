# delivery-v1 — the dataset-assembly rebalance campaign

_Owner ratified 2026-07-30: the **book-actor mix** — Dialogue 50% / Neutral 30% /
Documentary 8% / Newscaster 6% / Speech 6%. This note is the SSOT for the push._

## Where it stands — measured 2026-08-02 (supersedes the 2026-07-30 baseline below)

**Fold-eligible = every keep except the three instrument campaigns**
(`audit-tension-v2`, `audit-valence-v1`, `audit-emilia-keeps-v1` — the set
`seed_delivery.SKIP_CAMPAIGNS` already excludes). **`audit-markup-v0` DOES fold**
(owner, 2026-08-02): it is authored expressive material with curated registers, and
`seed_delivery.py` has always treated it as corpus. The blanket "audit-* campaigns
excluded" phrasing in the 2026-07-30 baseline below was wrong, and it mattered — the
89 keeps it excluded are 79 Dialogue, and Dialogue is the anchor the whole table
scales from.

**1,071 fold-eligible keeps.** Dialogue 578 held as the 50% anchor → the corpus
completes at **1,156**.

| lane | have | want | gap |
|---|---|---|---|
| Dialogue | **578** | 578 | anchor — no further dialogue needed |
| Neutral | 266 | 347 | **+81** |
| Documentary | 57 | 92 | **+35** |
| Newscaster | 84 | 69 | **−15 — DONE** |
| Speech | 69 | 69 | **DONE** |

**Remaining: +116, both lanes narration.** Newscaster and Speech closed when
newscaster-v1 landed (78 clips, qwen 27/27 · zonos 29/31 · moss_vg 20/20).

### CLOSED 2026-08-04 — delivery-v1-narration-r2 landed

| lane | have | want | gap |
|---|---|---|---|
| Dialogue | 578 | 578 | anchor |
| **Neutral** | **324** | 347 | −23 |
| **Documentary** | **87** | 92 | −5 |
| Newscaster | 84 | 69 | done |
| Speech | 69 | 69 | done |
| (delivery blank) | 108 | — | 91 are the three instrument campaigns, which do not fold |

Corpus total **1,250** heard-or-folded, past the 1,156 the campaign was sized for.
r2 contributed 88 heard keeps (58 Neutral, 30 Documentary) plus 13 group-certified
riders across 5 groups; 2 dropped. Both lanes are inside a rounding error of target
and the remaining −23/−5 does not justify another round on its own — take it from
the real-audio lane, where clips are already cleared and unused.

#### EXCEPTION: chatterbox's `TRUST BROKEN` alarm on this campaign was overridden

`certify` raised **`TRUST BROKEN: chatterbox`** and it was **not acted on** (owner,
2026-08-04). Chatterbox stays in the trusted tier.

The alarm fired on `voyage-of-the-beagle_r2_0034_CHA`, dropped because *"this passage
is making references to illustrations not visible here."* That is a **text-selection**
defect: the passage narrates figures that do not exist in audio, and no engine or
setting renders it usably. Chatterbox read it faithfully. Acting on the alarm would
have moved the engine to standard coverage — 2 per group + 10% instead of 1 + 3% —
roughly tripling its audit load on evidence that says nothing about the engine.

The same gap makes `voyage-of-the-beagle_r2_0018_ZON` count as a drop against zonos,
when that row is a bookkeeping retirement (re-cast to qwen in round 1, wav superseded
before the ear heard it) rather than a verdict. Neither group held deferred clips, so
neither false failure blocked anything.

**Root cause, deliberately left in place:** `certify` treats every drop as evidence
against the engine, but drops have causes — engine-attributable (mangled audio,
artifacts, truncation) versus source-attributable (bad text, invisible references,
bookkeeping). Typing them was considered and declined as not worth the machinery yet.
The mint-time `references_the_invisible` gate now catches this particular class before
it renders, so the text defect should not recur; the **attribution gap will**, the next
time a clip is dropped for its content rather than its audio. Read the drop's note
before believing a trust alarm.

### The last −23/−5, taken from the real-audio lane (2026-08-04)

The close-out above said to take the residual from the real-audio lane, "where clips
are already cleared and unused." Checked, and the two lanes came apart.

**Documentary stays at 87/92 — CLOSED at 95% of target.** There is no documentary
real audio segmented anywhere. The pool holds exactly two books: *Uneasy Money*
(fiction) and *Speeches: Literary and Social* (17 clips left, a lane already closed).
`book-prose` is an empty pool. The `lv:the-voyage-of-the-beagle` ledger entry is still
`pending book_ingest` — Darwin reached delivery-v1 as *text* for synthesis, never as a
LibriVox reading. Closing −5 would mean a fresh fetch + force-align + new-reader ear
pass for five clips, which is more than the gap is worth. Owner call: leave it, record
why. If Documentary reopens, the guide's other candidates (White *Selborne* `pg:1408`,
Muir, Kingsley) are the place to start; *Origin of Species* was considered and has
never been used in any lane.

**Neutral −23 comes from `librivox-v2`, through the ear.** Three things had to be
established first:

1. **The ear evidence does not survive a re-cut.** `librivox-v2` is the canonical
   *Uneasy Money* pool (1,366 clips / 25 sections against v1's 664 / 15), but **none
   of the 12 audited v1 ids exist in it** — ids encode section and sentence index and
   the duration-split fix moved both. Attributes still transfer, because
   `reader_profiles.json` is keyed on (reader, title) and `confirmed_tags` reads it
   that way; delivery does not, because `--mark-delivery` matches the heard set by id.
2. **A homogeneity mark would have been wrong.** This edition quotes dialogue with
   **single** quotes — 818 of them against 26 doubles — so the double-quote test reads
   a Wodehouse novel as pure narration. Corrected, **165 of 1,366 clips carry
   dialogue**. The title is heterogeneous, the per-clip rule for novels holds, and a
   title-level mark would have tagged those 165 Neutral. See A-M7 in
   [todo.md](todo.md) — this is that finding firing on real data, not a latent risk.
3. **`--seed-ear` could not reach them.** It tests confirmation with `confirmed_tags`,
   which covers gender/age/accent only, so the pair reads as confirmed and it seeds
   nothing — while no v2 clip can reach the ear to establish delivery. The deadlock it
   was written to break, reappearing in a dimension it does not look at.

So `stage_pool --delivery-ear` was added: the same linear take a normal staging run
makes, entered **unaudited** with attributes pre-filled, so the ear answers one
question instead of four. Linear rather than narration-filtered on purpose — at 12%
dialogue density a contiguous run costs almost no Neutral yield, and it keeps the
continuous-reading bet the pool/staging split was built on.

**Staged and heard 2026-08-04: 30 clips, reading-order 0..29 — 30/30 keeps, 30/30
Neutral, every one scored 5.** 1,336 still pooled. **Neutral closes at 354/347.**

### delivery-v1 is COMPLETE (2026-08-04)

| lane | have | want | | share |
|---|---|---|---|---|
| Dialogue | 578 | 578 | anchor | 49.3% |
| Neutral | **354** | 347 | **+7** | 30.2% |
| Documentary | 87 | 92 | −5, closed by owner call | 7.4% |
| Newscaster | 84 | 69 | +15 | 7.2% |
| Speech | 69 | 69 | +0 | 5.9% |
| (delivery blank) | 17 | — | the embodiment span-FiLM pilot | — |

**1,189 fold-eligible keeps.** Measured mix 49.3 / 30.2 / 7.4 / 7.2 / 5.9 against the
ratified 50 / 30 / 8 / 6 / 6. Every lane is at or above target except Documentary,
which is closed deliberately at 95%.

**One result cuts against the reasoning that produced it, and is worth recording.**
All three dialogue-quoting clips in the take were called **Neutral** — this reader
does not shift manner for quoted speech. That is mild evidence *Uneasy Money* may be
delivery-homogeneous after all, i.e. that a mark would have been safe and 22 auditions
cheaper. It is **not** grounds to mark it now: the sample is 30 *contiguous* clips from
section 1 of 25, which is precisely the coverage weakness that makes the mark risky
(see todo §7). If Neutral ever needs growing from this book again, settle it properly
first — a spread sample across sections is cheap and would answer it.

### The text shelf was empty, and that was the real blocker

The "~440 unrendered narration lines" below is **stale — it was 20**. Of 503 minted
narration lines across 21 books, delivery-v1 consumed all but 20. Ingesting is now
the rate limiter for the last two lanes, not rendering.

Refilled 2026-08-02. `conan-stories` had been routed SYNTHESIZE on 2026-08-01 and
never ingested; four more titles were owner-approved, chosen against the lane guide:

| book | lane | narration lines |
|---|---|---|
| conan-stories | Neutral | 40 |
| voyage-of-the-beagle (Darwin) | Documentary | 45 |
| up-from-slavery (Washington) | Neutral | 45 |
| franklin-autobiography | Neutral | 45 |
| walden (Thoreau) | Neutral | 45 |

⚠ The router flagged **Voyage of the Beagle** and **Franklin's Autobiography** as
already present in LibriTTS-R train-clean-100 (matched on title, different PG
edition). It annotates rather than skips, which is right — we are taking the text,
not the audio. But if the expressive corpus is ever merged into the VAT filelists,
those sentences would sit on both sides of the merge. Check before that merge, not
before the render.

### RESOLVED (2026-08-02): three-layer allocation replaces the flat mix

The per-line-choice vs `ENGINE_MIX` contradiction was dissolved by measuring first
(`ref_select.py`, commit `1faac46`; owner: "we know various engines have their
strengths and perhaps we can play to them better"):

1. **Capability veto** — `ENGINE_CHANNELS` makes the direction-relay audit executable
   (`route_engines(..., requires={...})` drops engines that cannot hear what a line
   needs; vibevoice = reference-only is the load-bearing entry).
2. **Measured per-lane weights** — `ENGINE_MIX_BY_LANE` replaces the flat
   27.5/27.5/20/15/10, built from heard PRODUCTION verdicts only. The measurement that
   reframed it: zonos conditioned correctly is a **93.7% engine (74/79)** across all
   three narration lanes (its 31% headline averaged a pre-fix era), and moss_vg is
   lane-SHAPED (Newscaster 95% / Documentary 56%) — one global share both overpaid and
   underpaid it.
3. **Diversity floor** — `MIN_SHARE` keeps every eligible engine present (a 0% share
   can never produce the evidence that would raise it) and `MAX_SHARE_NARRATION` caps
   any one engine: max keep-rate is NOT the objective; a one-timbre teacher corpus has
   failed at its job.

The director's pass-1 roster now derives from `ENGINE_MIX − SET_ASIDE` (the frozen
2026-07-25 prompt roster is gone), `_validate_mixes()` runs at import, and Dialogue
keeps the ratified mix near-unchanged — that lane is at target.

---

## Baseline (measured 2026-07-30, after cleanup) — HISTORICAL

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
- **Render allocation is PER ENGINE (per lane), and a tier is a tag — not a bucket**
  (owner 2026-07-31). The two questions stay separate: `ref_select.ENGINE_MIX_BY_LANE`
  says how much each engine RENDERS in each lane (three-layer allocation, see the
  RESOLVED section above — the flat table that used to sit here is superseded),
  `pick_audit_subset.TIERS` says how much gets AUDITIONED. Tiers as of 2026-08-02:
  trusted (qwen; chatterbox **provisional** — 38 clips, one campaign, SPLIT is
  ear-only) · normal (orpheus — explicit; its record was dragged by the now-banned
  `tara` fallback: non-tara 80.0% keep / mean 4.49 vs tara 9.5%; **zonos** joined
  2026-08-04 on r2's 37/38) · scrutinized (moss_vg — **100% heard**).
  Unknown engines default to scrutinized. Shares derive from measured PRODUCTION
  failure rates with probe/stress campaigns excluded — those are adversarial by
  construction and libel every engine (chatterbox reads 0% production but 35% probe).
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

### What text suits which lane (sourcing guide, owner 2026-08-01)

`seed_delivery.py` sets the rule that matters: Dialogue/Neutral is a property of the
**text**, Newscaster/Documentary a property of the **render**, and Speech of **both**. So
"find Documentary text" is the wrong request — what you can source is text that INVITES
the delivery, and then direct (synthesis) or hear it (real audio).

| lane | text that suits it | examples |
|---|---|---|
| **Speech** | addresses an audience — rally, sermon, toast, oration | Cicero *Against Catilina* (pg:39355); Dickens *Speeches: Literary and Social* (pg:824) |
| **Documentary** | third-person exposition | natural history, travel writing — Darwin *Beagle* (pg:944), White *Selborne* (pg:1408), Muir, Kingsley |
| **Documentary** | **biography** — owner 2026-08-01: "biographies can also be categorized as documentaries. The delivery is much the same" | any third-person *Life of…* |
| **Neutral** | first-person recollection | **AUTOBIOGRAPHY — explicitly NOT documentary** (owner, same): "those tend to be more narrative" |

The biography/autobiography split is grammatical stance, not subject matter. Third-person
exposition about a subject stands on the same footing as natural history, which is why
the delivery transfers intact. First-person recollection is narration that happens to be
about a real life — same subject, different footing, different read.

Caveat for the real-audio lane: none of this decides the tag. Whether a LibriVox reading
of Darwin IS Documentary depends on the reader's manner, not the book — some deliver it
measured and expository, others flat, which is Neutral. The text choice raises the odds;
the ear still calls it, once per (reader, title).

## Sequence

1. ~~**delivery-v1-narration** — Neutral + Documentary~~ — round 1 done (342 renders);
   **reopened**, see step 5. The lanes did not close.
2. ~~**delivery-v1-speech**~~ — **DONE.** Speech 69/69, sourced from PG oratory +
   the Dickens *Speeches* force-align lane.
3. ~~**delivery-v1-news**~~ — **DONE.** Newscaster 84/69, closed by newscaster-v1
   (78 clips: qwen 27/27 · zonos 29/31 · moss_vg 20/20).
4. ~~**delivery-v1-narration round 2**~~ — **DONE 2026-08-04**, see "CLOSED" above.
   88 heard keeps + 13 certified riders; Neutral 324/347, Documentary 87/92.
   It also settled the **zonos tier test** it was carrying: every prior zonos
   narration bank had `emotion: null` hand-patched, so the director path had never
   run end to end in production. This one did — **44/44 lines directed with emotion
   off**, 37 keeps from 38 heard. **Zonos promoted scrutinized → normal.**
5. ~~Fold, recompute the table~~ — **DONE.** Publish tier per
   [[expressive-registers-dataset]] conventions still open (higgs3-NC keeps stay out
   of the CC-BY publish set).

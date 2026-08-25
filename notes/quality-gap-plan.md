# Closing the quality gap — the ordered plan

**Scope: what we do next to close the synthetic-vs-real gap, and in what order.**
Measurement repair → data sequencing → decoder spike. This file owns the *sequencing
and the gates between phases*; it does not restate the plans it sequences. The DiT
design itself is [model-decisions.md § Decoder v2](../docs/model-decisions.md); source state is
[training-sources.md](training-sources.md) (SSOT); run mechanics are
[training-operations.md](training-operations.md).

Written 2026-08-06, immediately after the diagnostics below. **Phase 0 is done and Phase 1
rung 1 is built** — see the pathway below for where the front actually is.

---

## The pathway — one table, the QUALITY ladder

_Added 2026-08-08 at the owner's request. This is the INDEX, not the detail: every row
links to the section that owns it. If a row and its section disagree, the section wins._

⚠ **This is the quality ladder — the route to a model worth casting — not the whole route
to the product** (owner, 2026-08-09; PR-M3). It said "whole route" and it does not contain
a casting rung, while the repo leads with "directable, **castable**, and mobile-friendly"
and goal 2 (Dramatic Reader) depends on casting more than on mel loss. Every step below
ladders to *quality*; nothing here ladders to *castability*. That is a deliberate scope,
now stated rather than implied — see [§ Parked — the casting/blend layer](#parked--the-castingblend-layer).
**Where the front is: Phase 1, rung 2 — v5 trained and scored, `ep019` selected; v6 is
scoped, labelled and decided but NOT built.** The append set is settled at **832 rows**
(822 delivery-labelled + 10 delivery-blank), and as of **2026-08-10 all three
prerequisites are closed**: V from the 12-head EIV pass, **A and T from a new acoustic
pass** (846/846 measured, 0 unmeasurable), and the labelling lane ratified as the **global
anchor with A centred per campaign**. What remains is the **build script itself**, which
does not exist — plus one open owner decision that can still move the count: **14 clips
exceeded `MAX_SECONDS` (22 s) and were **dropped** (owner, 2026-08-10), so the append is
**832**.
See [§ Rung 2 build decisions](#rung-2-build-decisions--recorded-2026-08-09-corpus-not-built-no-run-queued).

| | step | status | gated on | detail |
|---|---|---|---|---|
| **P0** | 0a — never-trained holdout | ✅ **DONE 2026-08-06** | — | [§ 0a](#0a--the-never-trained-holdout--done-and-it-reported-2026-08-06) |
| | 0b — clean-lineage restart | ⛔ **NOT INDICATED** | — | [§ 0b](#0b--clean-lineage-restart--not-indicated-owners-call-to-ratify) |
| **P1** | **rung 1 — v5, +Emilia (78.5 h, 2,500 spk)** | ✅ **DONE 2026-08-08/09** — 48 epochs, holdout-scored, **`ep019` selected**; converged by epoch 9 | — | [§ the ladder](#the-ladder--a-strictly-growing-corpus-one-lever-per-rung) |
| | **rung 2 — v6, +expressive-registers (826 rows appended, 832 staged)** | ✅ **DONE 2026-08-10/11** — built, trained 10 epochs, holdout-scored, **`ep008` selected** (`logs/train/vat6_finetune/SELECTED.md`); flat as pre-registered, and **the delivery block is live** | — | [§ the ladder](#the-ladder--a-strictly-growing-corpus-one-lever-per-rung) · ⚠ **do not select this run on `total`** and the A-frame question is **open**, not closed — [direction-contract-v3-proposal.md § 3b](direction-contract-v3-proposal.md) |
| | rung 3 — v7, +LibriTTS-R full (~564 h measured) | 🔄 **UNGATED — rung 1 passed. AUDIO ON DISK, EIV PASS RUNNING 2026-08-25.** ~**2.25 h/epoch, ~1 day** to convergence (est. 2026-08-09 against ~330,000 rows — **not re-measured against the 345,600 now expected**) | rung 1's holdout ✅ | ditto |
| | rung 4 — v8, +Hi-Fi TTS (292 h) + VCTK (44 h) | ⏸ **both ON DISK, unconverted** | independent — slot in when converted | ditto |
| | rung 5 — v9, +more Emilia-YODAS shards | ⏸ 9 of ~114,000 h probed | rung 3's holdout | ditto |
| | freeze a same-corpus U-Net baseline | ⏸ | **the last act of P1** | [§ Phase 2](#phase-2--the-dit-decoder-spike) |
| | vocoder transparency re-check | ⏸ | after P1 | [§ gates](#gates-between-phases) |
| **P1S** | mass synthetic production, Qwen-primary | ⏸ **plan written, 3 prerequisites open** | **rungs 1–5** (owner: exhaust public sources first) | [§ Phase 1S](#phase-1s--mass-synthetic-production-qwen-primary) |
| **P2** | DiT decoder spike | ⏸ design written, not started | P1 landing + the frozen baseline | [model-decisions.md § Decoder v2](../docs/model-decisions.md) |

### Parked — the casting/blend layer

**Parked, not dropped** (owner, 2026-08-09). It is half of goal 1 and
`high-ambition-1-matcha-actor.md` names the casting/blend layer one of "the three things we
build", so its absence from the ladder above needs to be a recorded decision rather than an
oversight. This is the same treatment the emotion block got, and for the same reason: an
unscheduled ambition with no end-condition drifts into a forgotten one.

- **Why parked.** Rungs 1–5 each move exactly one lever, and that lever is quality. Casting
  needs its own instrument and none exists — the clean holdout is teacher-forced loss on
  audiobook narration and cannot see whether a directed cast matched its brief. Scheduling
  a rung before its instrument exists is what this plan refuses everywhere else.
- **What would end it.** (1) The measurable gender/age/accent norm set —
  `casting-attribute-norms-brief.md` item 3, "the actual goal", currently **logged, not
  scheduled**; and (2) a casting eval that is not the holdout. Rung 4 already tags Hi-Fi
  TTS as "casting anchors" and no scheduled work consumes them, which is the cheapest place
  to start when this un-parks.
- **Reconcile the vocabulary FIRST.** Two casting vocabularies are in circulation and they
  disagree: `high-ambition-1-matcha-actor.md` says **age / masculinity / strain**, while the
  owner-scoped norms brief says **gender / age / accent**. Building an instrument against
  the wrong one is the expensive version of this mistake — and "define our own measurable
  norms, never inherit another model's gender conflation" is the standing ruling that the
  brief's vocabulary is the live one.
- **Checked:** 2026-08-09.
| **P3+** | conditioning: embodiment bank → span-FiLM | ⏸ deferred chain | evidence, not plumbing | [todo.md §3](todo.md) |
| | categorical emotion block (3+8) | ⏸ **open decision** | P1's valence read | [todo.md §3](todo.md) |
| | multilingual | ⏸ **plan only** | after rung 3 | [teacher-training-data.md § multilingual](teacher-training-data.md) |

**Running alongside, needing no GPU:** the audit-quality pair in [todo.md §4](todo.md) — a
forced-ranking pass over the 46 ceiling-tied groups, and anchor exemplars in the audition
app. Both came out of the 2026-08-08 finding that the 1–5 scale is saturated (`librivox`
real audio means 5.00, and mean score ranks chatterbox above Qwen). They gate nothing, but
rungs 2+ depend on ear verdicts and the scale those verdicts are made on is currently
compressed — so this is the cheapest thing that improves every rung above 1.

### The three standing rules that outlive every phase

1. **Score on the never-trained holdout, never `loss/val_epoch`.** MLflow val is
   permanently unusable for cross-run comparison, and from v5 on the val set mixes two
   domains. One epoch on `libritts_r_holdout_devclean` destroys it permanently and there is
   no second dev-clean.
2. **Every rung ADDS rows without re-rolling what came before.** That is what makes rung
   *n*'s holdout number comparable to rung *n−1*'s, and a re-derivation destroys it
   silently. v5 proves it is achievable; the per-corpus checklist below is how.
3. **Spin down all inference engines before any run**, and remember that
   `compose up -d sonora_training` *starts* one.

### What is deliberately NOT on this pathway

- **De-skewing the Emilia mining** (widening past the tail-selection). Recorded under Phase 1
  with its cost; it must not run before rung 1's holdout, because it would change two things
  at once and cost us the test.
- **A staged-quality curriculum** — train broad, then continue on the clean tier. New idea,
  from reading what Qwen actually disclosed
  ([teacher-training-data.md](teacher-training-data.md)); costs no new data. Filed, not
  scheduled, because it competes with the volume lever we have not yet pulled.
- **Out-scaling the teachers.** Qwen trained on ~5,000,000 h and Chatterbox on ~500,000 h;
  everything cleared and reachable for us is ~115,000 h, and v5 is 78.5. Distillation is how
  that scale reaches us — Phase 1S — and no amount of corpus work substitutes for it.
- **Mass synthetic generation before the public sources are exhausted** (owner, 2026-08-08).
  Not a judgement on the idea: real audio yields ~100% against Qwen's 89.3% and costs no ear
  time, so render-hours before conversion-hours spends the expensive currency first. **The
  synthetic we ALREADY have is not deferred** — it is rung 2.

---

## What scoped this — three measured facts that still direct the work

1. **The vocoder is NOT the bottleneck.** Copy-synthesis puts mel L1 at 10.2% of the
   corpus's mel_std; the owner heard "similar to ref, if not identical". **This rules out
   vocoder training, vocoder replacement and sample-rate work as quality levers.** The gap
   belongs entirely to the acoustic model's predicted mel, which is what both phases below
   target. ⚠ The vocoder consumes **RAW log-mel** — copy-synthesis must skip `normalize()`,
   or it produces garbage.
2. **Check the front end before spending a run.** `vat3c`'s G2P fix turned out to be the
   whole fix: retraining bought a perceptually negligible change and blind A/B heard no
   difference. The model's phoneme→audio mapping was always sound; only text→phoneme was
   poisoned.
3. **Always randomise A/B assignment per pair.** "The second one sounds better" tracked
   *position*, not checkpoint. A fixed A=old/B=new layout records false positives.

---

## Phase 0 — the measurement, and what it fixed

**`loss/val_epoch` is not a generalization measure and never will be.** Splits were
re-drawn per corpus version while runs warm-started from the previous one, so 93–97% of
v3c's val clips had been trained on. The contamination is historical and does not heal.
**Score on the holdout; do not compare val curves across runs.**

**The instrument: `data/libritts_r_holdout_devclean`** — LibriTTS-R dev-clean, 5,463 clips
/ 8.7 h / 40 speakers disjoint from the corpus. Score with `scripts/score_holdout.{py,sh}`,
stratify by conditioning channel with `scripts/tools/stratify_holdout_sweep.py`. **Score every
retained checkpoint as a matter of course** — ~100 min for eight on an idle card.

- **Resolution floor: −0.0111.** That is the smallest movement this instrument has
  resolved (0a's `vat3_ep099` gain, CI excluding zero). Treat anything of that order as the
  floor, not as a result.
- **Its one failure mode is training on it**, which is permanent and has no second
  dev-clean. Now enforced in code — `license_wall.refuse_holdout` refuses any train or val
  filelist whose path names a holdout.
- **`vat3c_ep099` is RETIRED** — do not stage or export it. Its run was a measured
  regression on all three loss terms, which is what established that more epochs on a fixed
  corpus is not the lever; more corpus is. Rung 1 has since confirmed that at −0.0606.

**0b — clean-lineage restart: CLOSED, not indicated.** The lineage generalizes, so what 0a
found was one wasted run rather than a poisoned ancestry, and 0b is a full retrain from
22.05 kHz rather than a fine-tune. **Reopen only if a run with materially more data also
fails to move the holdout** — that would point at the ancestry rather than the corpus.

---

## Phase 1 — data sequencing

Cheapest first, so the expensive item is decided by evidence rather than assumption.

| # | Source | State | Adds | Work |
|---|---|---|---|---|
| 1 | **Emilia-YODAS keeps** | **MERGED 2026-08-08** → `libritts_r_emilia_vat_v5` | 10,997 clips / +27.2 h (**+36%**) | done; needs a GPU slot |
| 2 | **sonora-expressive-registers** | DERIVED, 1,004 eligible keeps → **832 to append** after dedup + duration filter | small, ear-certified | **merge as v6 at 832** (1,004 owner-scoped 2026-08-09, less 158 duplicate-audio rows found 2026-08-10, less 14 over `MAX_SECONDS` dropped 2026-08-10). ⚠ Do not re-open the **Documentary close** (87/92 — closed because *no documentary real audio exists*) |
| 3 | **LibriTTS-R, the other 90%** | NOT PULLED | **~10×**, ~2,400 speakers | download + full corpus build |
| 4 | **Hi-Fi TTS v1** | RAW, 40 GB parquet *with audio* | 292 h / 10 speakers | parquet → wav + filelist |

**Why this order.** #1 is nearly free and already cleared — the fastest available test of
whether volume moves quality *at all*. #2 is the first time expressive, ear-certified
material enters the training path; today the corpus we spend ear time on and the corpus
we train on have never been joined, which is the largest structural oddity in
`training-sources.md`. #3 is the real lever but the largest effort, and it is the one
worth gating on #1's result. #4 plays a different role — few voices, deep hours — so it
is casting anchors and speaker-consistent long-form prosody, not volume; it is
independent of #3 and can slot in whenever the parquet conversion is worth doing.

### The ladder — a strictly growing corpus, one lever per rung

**Owner's framing, 2026-08-08: each iteration fine-tunes against a LARGER set than the
last.** That is now the shape of Phase 1 rather than a hope about it, and it costs
something to hold: every rung must add rows without re-rolling what came before, which the
merge already proves is possible (v5 carries v4's rows byte-identically, its speaker
indices untouched, and its held-out clips still held out). Keep that property at every rung
— it is what makes rung *n*'s holdout number comparable to rung *n−1*'s, and it is the one
thing a re-derivation destroys silently.

**EXHAUST THE WELL-KNOWN PUBLIC SOURCES FIRST — owner, 2026-08-08.** That is the ordering
principle above everything below, and the measured case for it is strong: **real audio
yields ~100% and costs no ear time.** `librivox` keeps 99.0% of what is heard (103/104) and
`emilia-yodas` 100% (48/48), against 89.3% for our best synthetic engine. Real audio needs
no rendering, no direction pass, no QA pruning and no listening — **every hour of it is
cheaper than every hour of synthetic**, and roughly 870 of them are cleared and unused.

| rung | corpus | train rows | hours | speakers | the one lever it moves | gate to the next |
|---|---|---|---|---|---|---|
| — | `libritts_r_vat_v4` | 30,485 | 51.3 | 247 | width only (8-wide) | smoked, superseded |
| **1** | **`libritts_r_emilia_vat_v5`** | **41,138** | **78.5** | **2,500** | **volume, +36%** | holdout vs `vat3_ep099` |
| 2 | v6 = +expressive-registers | ~41,980 | ~81 | ~3,350 | **the delivery channel's only training signal** — see below | delivery channel finally has signal |
| 3 | v7 = +LibriTTS-R full | **~345,600** | **~564** | **~5,414** | **10× volume, same domain** | the real lever |
| 4 | v8 = +Hi-Fi TTS v1 + VCTK | +? | +336 | +120 | **depth per voice** + studio timbre breadth | independent; slot in when converted |
| 5 | v9 = +more Emilia-YODAS shards | +? | unbounded | +thousands | **in-the-wild expressivity at scale** | the mining pipeline already exists |
| 6+ | the synthetic lane | — | — | — | see **§ Phase 1S** below | gated on 1–5 |

### Rung 2 is not a volume rung — it is the only delivery signal that exists

Measured 2026-08-09 against the shipped v5 filelists. **The delivery channel is untrained:**

```
v5 (41,138 train + 1,304 val):  48 rows carry ANY delivery label  — 0.11%
   Dialogue 39 · Neutral 9 · Documentary 0 · Newscaster 0 · Speech 0
```

Three of the five lanes have **zero positive examples**, so the 8-wide conditioning vector
that shipped in contract v2 is, today, five channels the model has never seen fire. Any
delivery audition run against a v5 checkpoint measures an untrained channel and will read
as an architecture failure — it isn't one.

**v6 scope: 1,004 eligible keeps before dedup → 846 appended (owner 2026-08-09; dedup
2026-08-10).** The 1,004 — 902 delivery-labelled + 102 delivery-blank — is what survives
the licence and standing-policy filter over 1,279 total keeps, all four exclusions honoured
as written: VibeVoice/Dia 133 (benched), moss85 83 (un-SFT'd base), longcat 51 (benched),
higgs3-NC 8 (**NC — never trains**). Per lane at that stage: Dialogue 402 · Neutral 270 ·
Documentary 85 · Newscaster 76 · Speech 69. **158 of those rows duplicate audio already in
v5** (three classes, § Rung 2 prerequisite 3), leaving 846 — less **14 over-length rows**
dropped on the owner's call (2026-08-10) for **832 to append**. Quote **832** as the v6
figure, 846 as "after dedup, before the duration filter", and 1,004 only as "eligible
before dedup".

⚠ **Known confound, accepted deliberately rather than bought off.** Dialogue, Documentary
and Newscaster have **no real audio at all** — no documentary real audio exists, and the
same holds for the other two — so after the merge, *delivery-labelled ⇒ ~91% synthetic*
while *blank ⇒ ~99.8% real*. The model can satisfy the loss by learning "labelled means
render like a teacher engine" instead of a manner of address. Removing the confound is not
purchasable: matching the labelled-rate across provenance needs **~36,700 blank synthetic
renders ≈ 336 GPU-hours**, and would make the corpus ~47% synthetic — contradicting the
ordering principle above. So we **measure it instead**: direct a real, well-represented
speaker through each lane and check whether manner changes or only timbre. Predict before
training, not after.

⚠ **The two percentages above were measured against the superseded 1,004-row scope and
have NOT been recomputed for the 832.** They are stale in a direction that matters:

- A closing sentence — *"The 57 synthetic blank keeps in the merge help slightly and do
  not close it"* — has been **struck**, not merely restated. The append set holds **10**
  blank rows in total, so 57 synthetic blanks cannot be among them.
- 91% / 99.8% was measured over 902 labelled + 102 blank; the population is now 836 + 10,
  and after the 14-row duration drop 822 + 10; the 158 removals were **not** provenance-neutral (class 1 is 91 real `libritts-r` /
  `emilia-yodas` rows, class 2 is 45 real LibriTTS-R rows). The labelled half loses 66
  rows, most of them real, so the labelled set's synthetic share moves **up**.
- Internal contradiction, exposed by the drop: if 57 of the 102 blanks were synthetic then
  at most 45 blanks were real — yet class 1 is 91 real rows described as all
  delivery-blank. Both cannot hold.

The confound is the reason this rung ships with a pre-registered measurement, so its
magnitude is load-bearing. **Recompute both percentages and the synthetic-blank count
against the 832 before the build.**

#### Rung 2 build decisions — recorded 2026-08-09, CORPUS NOT BUILT, NO RUN QUEUED

The v6 corpus does not exist yet and no training run is queued. An EIV scoring pass *has*
run (prerequisite 1 below) and its artifacts are on `/data`; nothing downstream of it has.
What follows is settled so the next session starts from a decision rather than a
re-investigation.

**SPEAKER IDENTITY — one speaker id per clip (owner, 2026-08-09).** Each merged clip gets
its own id appended after v5's 2,500, taking the table to ~3,500. It makes no false claim,
and the precedent is already in the corpus: v5 carries **756 one-clip speakers**. Rejected:
*one id per engine* (10 ids — asserts all 147 zonos clips are the same person, which is the
speaker table lying); *recover true identity from the manifests* (real archaeology, see
prerequisite 2 — the information largely is not there).

**Three prerequisites, measured 2026-08-09 — scope and duplicates closed 2026-08-10; the
labelling half of 1 and 2 is still open.** Two of the three numbers below were wrong when
filed, in opposite directions.

**v6 appends 832 rows = 1,004 eligible − 158 duplicate − 14 over-length.** The middle
figure cross-checks on both halves: 1,004 − 158 = **846** = **836** delivery-labelled
(902 − 66) + **10** blank (102 − 92), with 66 + 92 = 158. **The 14** exceed
`derive_vat_corpus.MAX_SECONDS` (22 s, up to 64.2 s) and were **dropped on the owner's
call, 2026-08-10** — all 14 are delivery-labelled (Neutral 10 · Documentary 3 · Dialogue 1),
so the blank half is untouched.
**Final append: 832 = 822 delivery-labelled + 10 blank.** Per lane: Dialogue 344 ·
Neutral 251 · Documentary 82 · Newscaster 76 · Speech 69.
⚠ Dropping them moved nothing else: the label distribution is unchanged to two decimals
(V +0.005 / A +0.024 / T +0.261, clamped 21.6% / 2.4% / 23.1%), and **no campaign crossed
the `MIN_CAMPAIGN_N` floor**, so no clip changed offset lane.
⚠ **Do not derive 846 from the 923 scoring scope.** 923 is the count of eligible rows that
needed an EIV pass (1,004 less the rows already carrying `measured_z`), not the append set,
and the "923 − 158 + 60" reading recorded on 2026-08-10 gives **825**. Whether those 60
`v1/metadata.jsonl`-only rows are inside or outside the 1,004 is unrecorded; the append set
is defined by the 1,004 − 158 line above regardless.

1. ✅ **CLOSED 2026-08-10 — V from the 12-head EIV pass, A and T from a new acoustic
   pass. 846/846 measured, 846/846 labelled, 0 unmeasurable.**
   **The scope was 923, not ~810.** The ~810 assumed the whole 193-row v1 slice was
   eligible; it is not — that slice's keeps are `qwen` 82 · `longcat` 51 · `moss85` 42 ·
   `dia` 5 (**180 of the 193**; the remaining 13 are unexplained and worth a recount),
   and three of those four engines are excluded by the filter above. The prerequisite as
   filed recorded **81** eligible rows carrying `measured_z`, which is one short of the
   82 `qwen` keeps the same sentence gives — if a qwen row is excluded for a second
   reason, it is not written down, and 923 = 1,004 − 81 moves to 922 if the answer is 82.
   Raw: `eiv_scores/expressive_registers_v6.jsonl` (846 rows); derived:
   `corpus_soft_v6.json` / `corpus_valence_combo_v6.json` (**31,197 entries = the
   v4-lineage 30,351 + these 846** — not the merged v6 corpus, whose Emilia half is
   labelled on the global anchor and is not in these maps); env captured per D-M2.
   **On the add-rows-without-re-rolling rule: this append did not test it.** The merge
   printed `corr=1.000000, mean shift +0.0000`, but `eiv_merge_corpus.combo` z-scores
   within path-derived speaker groups (`wav.split("/")[-3]`, line 76) and the 846 clips
   live under the expressive-registers tree — they join **new** groups, so they cannot
   move an existing clip's mean or sd. That output is what the script prints whether or
   not the new scores are sane. Nor does it meet D-L2's precedent on its own terms:
   ≤0.0008 was a per-row **maximum** (D-L2's own entry, in the retired changelog — recover
   the text with `git show "$(git rev-list -1 HEAD -- notes/CHANGELOG.md)^:notes/CHANGELOG.md"`,
   which resolves the deleting commit itself and so survives a squash merge), while
   `mean shift` is
   `np.abs(a - b).mean()` over ~30k clips (line 138). To claim the precedent, record
   `np.abs(a - b).max()`; to show the 846 are on-scale, record the new-clip valence
   mean/sd vs corpus that the script already prints at lines 141-147.
   ⚠ **Score with all 12 heads, never the default 4.** `eiv_score.py:94` defaults to
   `Valence,Arousal,Distress,Soft_vs._Harsh`, but `valence_combo_v1.json` weights **9**
   heads, 8 of them family heads — the default set silently omits 8 of the 9 and
   `eiv_merge_corpus.py:84` then hard-exits on "raw scores missing weighted heads".
   **A wrong-head run is repaired by deleting the output and rescoring, not by
   re-running:** `eiv_score.py:126-132` resumes on wav path alone, with no regard for
   which heads a row already holds, so a second pass skips every row already present and
   leaves a mixed head set. `eiv_score.sh` advertises that resume as the recovery path,
   which makes this easy to walk into.
   **What is NOT closed by this pass — only V comes from the EIV heads.** Per
   `derive_vat_corpus.py:97-115`, **A** is the per-speaker z of **LUFS** and **T** is the
   `alpha_db`/`cpp`/`−h1h2` composite with the `Soft_vs._Harsh` head repairing the pressed
   end — one of four terms. Those four measures come from an acoustic pass writing
   `measures.jsonl`, and **nothing in this repo records one having run for the 846 clips.**
   `derive_vat_corpus.py:536-546` hard-aborts with *"label sources do not cover every kept
   clip"* rather than writing zeros, so if that pass has not run the build fails at derive
   time.
   ✅ **It had not run, and now has** (`scripts/tools/measure_expressive_registers.py`, 2026-08-10
   → `expressive_registers_measures/probe_measures.jsonl` + manifest). Verified first that
   there was no third path: every `measures.jsonl` on disk is LibriTTS-R lineage, and
   Emilia's A/T come from `probe_measures.jsonl` written by the mining pass. Neither
   producer fits this bank — `derive_vat_corpus` walks a speaker/chapter tree,
   `mine_emilia_probe` walks extracted Emilia tars — so the new script is the **third
   producer for the third corpus shape**, writing the row shape `anchor_emilia_labels`
   already consumes. `phonation_measures` is **imported**, never reimplemented: a second
   implementation of the tension composite would put this half of the corpus on a silently
   different scale.
   ⚠ **Two properties of this bank that the other two sources do not have**, both handled
   explicitly rather than by inheriting a filter written for a different corpus.
   **148 of the 846 are 44.1 kHz** and `derive_vat_corpus.measure_clip` rejects
   `sr != 24000` outright, so reusing it directly would have dropped 17.5% of the set
   without comment; they are resampled as `mine_emilia_probe` does, and their LUFS matches
   the 24 kHz clips to within 0.08 dB, so the resample is not moving loudness.
   **14 clips exceed `MAX_SECONDS` (22.0 s, up to 64.2 s)** — measured and flagged
   (`over_max_seconds`), **not** dropped, because dropping them silently moves the append
   count from 846 to 832. ✅ **DROPPED — owner's call, 2026-08-10.** All 14 are
   delivery-labelled (Neutral 10 · Documentary 3 · Dialogue 1), so the blank half is
   untouched, and nothing else moved: the label distribution is unchanged to two decimals
   and no campaign crossed the `MIN_CAMPAIGN_N` floor. **Append set: 832.**
2. ✅ **CLOSED 2026-08-10 — identity is not recoverable, and the lane that implies is now
   ratified: GLOBAL ANCHOR, with A centred per campaign (owner).**
   880 of 1,004 ids carry no `_sSEED` suffix and only 114 rows in any manifest carry a
   reference/voice field, so identity is not recoverable; parsing it out of ids collapses
   the bank into per-engine buckets. That is *why* one-id-per-clip is the honest answer
   rather than merely the easy one.
   ⚠ **The consequence this prerequisite does not record.** One id per clip means every
   appended row is **n = 1** under `derive_vat_corpus.per_spk_z` (lines 503-525), which
   re-centres each clip on its own speaker's mean — so V, A and T all come out **exactly
   0.000**, and a 0.0 label is indistinguishable from a real at-speaker-mean one. This
   corpus has already been bitten by it: `merge_emilia_corpus.py:20-29` exists precisely
   because per-speaker z *"hands 756 one-clip speakers a label of exactly 0.0"*, which is
   why Emilia is labelled on the **global anchor** (`anchor_emilia_labels.libritts_anchor`).
   Note also that `eiv_merge_corpus.combo` z-scores within `wav.split("/")[-3]` groups
   (line 76), so a group of one there zeroes the combo before `derive_vat_corpus` ever
   sees it.
   ✅ **Ratified: the global anchor, same lane as Emilia** (`scripts/label_expressive_
   registers.py`, 2026-08-10). `anchor_emilia_labels.py` is deliberately **not modified** —
   it is the shipped v5 labelling path, so the offset is applied to the row before
   `label()` sees it, and every row records `lufs_native` / `lufs_offset` /
   `lufs_adjusted`.

   ⚠ **The dry run is what forced the one departure, and it is why "record the
   distribution before the build" was worth insisting on.** Plain global anchor:

   | | V | A | T |
   |---|---|---|---|
   | v5 LibriTTS half (30,485) | +0.008 / .42 / 7.2% | −0.001 / .48 / 4.9% | +0.004 / .47 / 4.9% |
   | v5 Emilia half (10,653) | +0.307 / .61 / 30.0% | −0.003 / .59 / 11.5% | +0.747 / .36 / 54.0% |
   | the 846, plain anchor | +0.004 / .58 / 21.4% | **−0.970 / .14 / 94.4%** | +0.262 / .58 / 22.8% |

   V and T sit inside precedent — and note **T at 22.8% is less than half the Emilia
   half's 54.0%**, so the append set does not repeat rung 1's T saturation. **A was pinned
   at −1 on 94.4% of rows**, and the cause is not performance: this bank is loudness-
   normalised (**median exactly −23.00 LUFS**) and LibriTTS-R is not (−18.16, sd 1.86). A
   5.18 dB gap is **2.8 anchor sd**, which `clamp2` maps to −1.39 and clips. Emilia's mean
   (−17.73) lands near LibriTTS's by accident of source — *that* is why the plain anchor
   worked there and not here.

   ⚠ **Centring per BANK was the first fix and it was not enough.** The bank holds at
   least **three loudness targets** — −23.0 (general), −20.4 (`quote-pilot-*`), −26.3 to
   −27.1 (`book-librivox-*`) — so one offset leaves each displaced by up to 6.7 dB and the
   103 librivox rows still came out at **A = −0.835**: the same defect one level down.
   **Centring per campaign** (`MIN_CAMPAIGN_N = 10`; 25 rows in 8 thin campaigns take the
   bank-wide fallback, because centring a lone clip on itself is the n = 1 trap again)
   removes it wherever it occurs:

   | | plain | bank-wide | **per-campaign** |
   |---|---|---|---|
   | A mean | −0.970 | +0.011 | **+0.024** |
   | A sd | 0.135 | 0.385 | **0.217** |
   | A clamped | 94.4% | 8.7% | **2.4%** |
   | librivox 103, A mean | −0.917 | −0.835 | **+0.000** |

   **A constant A is the CORRECT answer for most of these rows, and is not a failure.**
   638 of 846 sit within 0.05 dB of −23.00: loudnorm already destroyed their performance
   loudness and no offset recovers it. But the model is trained to reproduce the
   *normalised* audio, so "unremarkable loudness" is a true statement about the target.
   The real variation survives where loudnorm did not flatten it — and it is now the
   **real-audio librivox clips that carry the most A signal** (sd 0.309 against the
   register-labelled 0.201), which is the right way round.
   ⚠ Only the **centre** is corrected, never the spread: rescaling the sd would assert
   that this bank's loudness variation means what LibriTTS-R's does, which nothing has
   measured. **V and T are not offset at all** — an offset there would be a correction
   with no defect to correct.
   ⚠ **Pre-loudnorm loudness is recoverable for 104 of the 846 only** (`v1/audio/
   loudnorm.jsonl` carries `lufs_in`, `_pre_loudnorm_v1/` holds the originals). Using it
   would put 104 rows on a fourth scale, so it is unused — and it is the one lane that
   could carry real A signal if that sidecar were ever backfilled across the bank.
3. ✅ **CLOSED — 158 rows duplicate audio, not 91; three disjoint classes needing opposite
   treatment** (verified no overlap):
   - **91** audit-set copies (`libritts-r` 43, `emilia-yodas` 48, ids encoding `spk122`/
     `spk136`), **all delivery-blank** → **dropped**; they carry no signal v5 lacks.
   - **45** rows whose **engine tag** is `scm-spike-v1` (the *markup* campaign that
     produced them, path `audit-markup-v0/scm-spike` — not the audio's origin) →
     **excluded, and they needed nothing.** The audio is real LibriTTS-R
     `train-clean-100`, already in `train_op.txt`. They were merged into v5 **with their
     labels**: v5's 48 delivery-labelled rows (39 Dialogue + 9 Neutral) are **44 of these
     clips plus 4 others**, and `ratings.csv` agrees with v5 on all 44 (0 disagreements).
     Appending them would have handed a voice a second speaker row for labels v5 already
     holds.
     ⚠ **44 ≠ 45 — the 45th row is unaccounted for.** The exclusion rests on "v5 already
     holds their labels", and that claim demonstrably covers only 44. Delivery labels are
     the scarcest signal in this corpus (48 in 42,442), so before the build: confirm the
     45th is delivery-blank, or resolve it back to its existing v5 id rather than dropping
     it.
   - **22 `mk_` twins dropped** — **22 pairs (44 rows)** where two ids share one wav; the
     original-campaign row is kept and the `mk_` twin dropped, so class 3 contributes 22
     to the 158. (Classes 1 and 2 count *all* affected rows; class 3 counts only the
     dropped side. 91 + 45 + 22 = 158.) 21 pairs agreed on delivery; **1 contradicted** —
     `neutral_narration_00_narratorFM_s1234`, `Neutral` (bulk1/qwen) vs `Dialogue`
     (audit-markup-v0/scm-spike). **Owner heard it 2026-08-10: Neutral.** The text is
     narration prose with no quoted speech, and the markup campaign appears to have
     annotated emotional register through a field that means mix balance.

   The 103 `librivox` keeps are ours and genuinely new.

   **The lesson worth keeping: an engine tag is not audio provenance.** Class 2 was
   invisible to the engine-name rule that found class 1, and class 3 was *noticed* because
   `collect_wavs` deduped by path and returned 786 for a 787-row list — that is one
   collapsed path, not the measurement. What sized the class was a full path-multiplicity
   count over the eligible set. Check where the audio actually lives.

**STEP 0 — PARTLY DONE 2026-08-09.** The checkpoint half ran; the Qwen half has not.

- ✅ **ep009 vs ep019 — no discernible difference (owner's ear).** The pick is **ep019**,
  taken on total. The null is the result: neither loss margin is perceptually real, so
  ep009's 0.0013 `diff` edge is not grounds to reopen it. Recorded in
  `logs/train/vat5_finetune/SELECTED.md`.
- ✅ **The V/A/T dials work** — small but real effect, confirmed the same session. Worth
  recording because the first read was "the dials do nothing", which was the Synthesize
  button not being pressed. Before treating a dead dial as a model defect, confirm the
  render actually re-ran: at `guidance` 1.0 (CFG off) and 10 ODE steps the effect is subtle
  by design, and temperature 0.667 means two renders at identical settings already differ.
- ⏸ **Sonora vs Qwen — WANTED, but held until the model is far enough along to learn from
  it (owner, 2026-08-09).** Not a deferral for want of time: run today it measures a gap
  dominated by scale — **78.5 h against Qwen's ~5,000,000 h** — and returns the answer we
  already have, "more data." It starts saying something about the *model* rather than the
  *corpus* only once the volume lever has actually been pulled and a gap still remains.
  **Proposed trigger (assistant's, unless the owner sets otherwise): after rung 3's holdout
  lands** — v7, LibriTTS-R full, ~615 h, the 10×. Does not gate rung 2.
- ❌ **A pre-v6 baseline render is NOT needed** (owner, 2026-08-09), and the argument for it
  was simply wrong. It was justified as "cheap now, impossible later" — but **`ep019` is
  retained on disk** and v6 trains into a *new* run directory from its own warm start, so
  nothing consumes or overwrites it. The checkpoint IS the baseline: load it in the
  Vocalizer whenever a before/after is wanted. Generalises — *before treating an artifact as
  perishable, check whether the thing that produces it is retained.*

**No pre-v6 audition is required. This does not gate the build.** Step 0 above settled it:
`ep019` is the pick, the Qwen comparison is held until after rung 3, and `ep019` is retained
on disk so the checkpoint IS the baseline — nothing about v6 consumes or overwrites it.

*(A "NOT OPTIONAL" block stood here until 2026-08-09 requiring ep009 · ep019 · Qwen in front
of the ear before v6. Its three justifications had each been retired by the records above it;
it is in git history and in the 2026-08-09 review. The lesson worth keeping: **before treating
an artifact as perishable, check whether the thing that produces it is retained.**)*

**The protocol below survives** — it is how the Qwen comparison should be run whenever it
does happen, at rung 3. Delivery-blank on purpose: the channel is untrained (48 labelled rows in 42,442), so
directing it would measure nothing and read as an architecture failure. Run it blind and
order-randomised, **forced-choice rather than scored** (the scale is saturated — real
LibriVox audio means 5.00 and six engines tie at 5), loudness-normalised (the
Qwen loudness confound ran backwards once already), and with **one pinned speaker id**
across every clip, or delivery differences confound with voice.

**Then the build:** dedup (3) → EIV pass (1) → merge with v5 rows byte-identical and ids
appended per (2)+decision → hash split → measure `data_statistics` → licence entry → config
→ warm start from `vat5_finetune` **`ep019`** — the pick, reaffirmed by the owner
2026-08-09. Follow `merge_emilia_corpus.py` verbatim; its contiguity, collision and
val-nonempty guards are the ones that matter.

#### Rung 3 build decisions — recorded 2026-08-25, AUDIO ON DISK, EIV PASS RUNNING, CORPUS NOT BUILT

The other 90% of LibriTTS-R was never on the box. Both tarballs were fetched on 2026-08-25
(`train_clean_360` 28.95 GB + `train_other_500` 46.84 GB, from `www.openslr.org` — the
default `us.` mirror is down, http=000) and extracted to exact advertised byte counts. What
follows is measured on the extracted tree, not estimated from the corpus papers.

| subset | speakers | clips | in v6? |
|---|---|---|---|
| `train-clean-100` | 257 | 33,232 | yes |
| `train-clean-360` | **904** | **116,462** | new |
| `train-other-500` | **1,160** | **205,035** | new |
| `dev-clean` (holdout source) | 40 | — | never |

**THE STRICTLY-GROWING RULE SURVIVES RUNG 3 ONLY BECAUSE THE NEW SPEAKERS ARE DISJOINT, AND
THAT WAS TESTED RATHER THAN ASSUMED.** `eiv_merge_corpus.py` recomputes *every* clip's combo
on purpose — the value is `dot(weights, per-speaker z(head))`, so adding clips to a speaker
moves that speaker's mean and sd and silently rewrites labels already shipped. Its own
header states the consequence: *"v2 labels are not comparable to v3 labels and the vat3
checkpoint cannot be fine-tuned onto them."* That is the warm start and the holdout
comparability in one sentence. Measured under `LC_ALL=C` with a positive control (the
first run gave a **false 0** — `comm` bailed with "not in sorted order" on locale-collated
input and still exited clean):

```
train-other-500 ∩ train-clean-100 : 0      train-clean-360 ∩ train-clean-100 : 0
train-other-500 ∩ train-clean-360 : 0      train-clean-360 ∩ dev-clean       : 0
train-other-500 ∩ dev-clean       : 0      control (dev ∩ dev)               : 40 ✅
```

So no existing speaker's population changes, v6's rows reproduce byte-identically, and the
`--donor-speakers` prefix proof still holds. **The Emilia half is safe for a second,
independent reason:** `anchor_emilia_labels.py` reads `CORPUS = "data/libritts_r_vat_v4"` —
a *named, frozen* corpus, not "whatever LibriTTS rows exist now" — so a 10× LibriTTS
population cannot move the global anchor those rows were labelled against.

**THE NEW EIV PASS MUST SCORE EXACTLY THE 12 HEADS THE OLD RAW FILES CARRY.** `combo()`
keeps only heads present in *every* row (`all(h in v for v in raw.values())`), so a pass
that scores a different set would silently drop a weighted head **for the whole corpus,
including clips already labelled**. The set is `corpus_v1.jsonl`'s 4 (Valence, Arousal,
Distress, Soft_vs._Harsh) + `corpus_families.jsonl`'s 8 (Bitterness, Amusement, Sadness,
Elation, Hope_Enthusiasm_Optimism, Shame, Fear, Contentment) — covering all 9 nonzero combo
weights plus the T channel's `Soft_vs._Harsh`.

**Scope: 303,638 clips, not 321,497 — score the keeps, which is what the corpus already
did.** `corpus_v1.jsonl` holds 30,351 rows against train-clean-100's 33,232 clips, so the
convention is a duration-filtered list. Applying `derive_vat_corpus`'s own gate
(`MIN_SECONDS=1.0`, `MAX_SECONDS=22.0`, 24 kHz) over all 321,497 new clips: **kept 303,638
(94.45%)**, dropped 15,618 under 1 s and 2,241 over 22 s, **0 wrong sample rate and 0
unreadable** — which is also the extraction's integrity check. List at
`eiv_scores/libritts_r_full_v7_wavs.txt`. ⚠ The 4 s floor is the *audition-bank* rule and
does not apply here; this lane's floor is 1 s.

**Throughput is measured, and batch size is a real lever.** Isolation verified (exactly one
probe container during each window — an earlier batch-64 reading of 1.07 and a second of
2.13 were both contaminated by an overlapping batch-32 probe, and a naive
"two-equal-samples" settle check is what let the second one through):

| `--batch-size` | clips/sec | 303,638-clip pass |
|---|---|---|
| 8 (the default) | 2.00 | 42.2 h |
| **32 — chosen** | **3.20** | **26.4 h** |
| 64 | 2.84 | 29.7 h |

Running since 2026-08-25T02:44:58Z at a steady **3.73 clips/sec** (ETA ~22.5 h) to
`eiv_scores/libritts_r_full_v7.jsonl`. It appends and skips wavs it already holds, so it is
safe to interrupt and resume — **it is not a job that has to be protected**.

⚠ **`tar` extracted the audio owner-only (`0600` files, `0700` dirs) and the EIV pass died
at `PermissionError` on its first clip.** The containers run as `ai-mgr` (105:109) through
the `datashare` group, and the tarball's own mode bits carry none. Fixed to `2775`/`0664`
across 973,994 files and 6,823 dirs, scoped to the two new subsets. **Any future corpus
unpacked from an upstream tarball has this defect** — the existing subsets do not, because
they were unpacked before the container ran as a service account.

**The ladder table above now carries these measured figures**, replacing the estimates it
shipped with: train rows **~330,000 → ~345,600** (v6's ~41,980 + 303,638 before the
text-side drops); hours **~615 → ~564** (new kept audio measures ~483 h against v6's ~81 h;
the 4,000-clip sample gives median 4.28 s / mean 5.58 s, all 24 kHz); speakers
**~4,900 → ~5,414** (v6's ~3,350 + 2,064 new). ⚠ The **~2.25 h/epoch** in the pathway table
is untouched and is now the *stale* half of this pair — it was derived against ~330,000
rows and nothing has re-measured it. The row count went **up** and the hours **down** — LibriTTS-R
utterances are shorter than the estimate assumed, so this rung buys more rows per hour than
planned and the epoch cost should be read off rows, not hours.

**Then the build:** finish the EIV pass → `eiv_merge_corpus.py --add` to rebuild the combo
and soft maps corpus-wide (it recomputes every clip by design; the disjointness above is
what makes that a no-op for existing rows, and **that is a claim to re-verify against the
shipped v6 files, not to trust from this note**) → `derive_vat_corpus.py` over the two new
roots on per-speaker z (viable here, unlike Emilia: ~2,064 new speakers with a median in
the hundreds of clips) → merge with v6's rows byte-identical and ids appended → hash split
→ `data_statistics` re-measured **in-container** → `configs/data_licenses.yaml` already
classifies both subsets under `libritts_r` (verified by calling `classify_path`, so no
manifest edit is needed) → warm start from `vat6_finetune` **`ep008`**. Follow
`merge_emilia_corpus.py` verbatim, as rung 2 did.

### The low-hanging fruit, itemised — ~870 h before anything is rendered

Everything here is **licence-cleared** and costs conversion work rather than GPU-hours.
Hours are the sources' own figures (`training-sources.md`), not measured by us.

| source | state | hours | what it actually costs |
|---|---|---|---|
| **Hi-Fi TTS v1** | **40 GB of parquet WITH audio, ON DISK, never opened** | **292** | parquet → wav + filelist. The largest corpus we own and the only one needing no download |
| **LibriTTS-R, the other 90%** | not pulled — we hold `train-clean-100` only | **+534** | download + corpus build, same pipeline as v4 verbatim |
| **VCTK** | **4.3 GB zip ON DISK, still unextracted** | **44** | `unzip` + filelist. 110 speakers, 48 kHz studio. Casting/accent breadth only — no conveyance value |
| Emilia-YODAS, more shards | 9 shards probed of ~114,000 h licensed | unbounded | the mining pipeline exists and ran this week |
| sonora-expressive-registers | derived, 1,004 eligible ear-certified keeps (scoped 2026-08-09) → **832 to append** (dedup + duration filter, 2026-08-10) | small | merge as v6 — see § Rung 2 below for the eligibility filter and why this rung is not optional |
| librivox-v2 | derived, 1,366 clips, nothing staged | 3.1 | stage; re-earn v1's 12 ear verdicts |

**Two are already on the disk and merely unconverted** — Hi-Fi TTS and VCTK, **336 hours
between them, 4× the entire corpus v5 just became**, and neither has ever been opened.
That is the definition of low-hanging.

⚠ **THE SYNTHETIC WE ALREADY HAVE IS NOT WHAT IS BEING DEFERRED** (owner, 2026-08-08:
*"I'm not counting our existing synth. Let's use that now."*). `sonora-expressive-registers`
is **rung 2 and stays at rung 2**, immediately after v5 — it is already rendered, already
ear-certified, and already paid for in ear time, so using it costs nothing but the merge.
It is also the **only** thing that gives the delivery channel real signal: v5 carries 45
delivery labels in 41,138 rows, and the 1,189 delivery keeps all live here. What waits
behind the real-audio rungs is **mass production of NEW synthetic** — Phase 1S.

⚠ Deliberately NOT on this list: **MLS English** (CC-BY-4.0 but 16 kHz — the quality bar
disqualifies it), **HiFiTTS-2** (metadata only; a 2.8 TB fetch is a decision, not fruit),
**GLOBE V2** (CC0 but supersampled — true bandwidth below nominal), **Expresso** (NC,
unresolved owner call).

Rung sizes past 1 are estimates from `training-sources.md`, not measurements — every one
of them gets derived from the corpus before it is quoted. Rung 1 already made that point
three times over: 13,141 rows became 10,997, `n_spks` 2,655 became 2,500, and
`data_statistics` moved ten times the precedent.

**THE ORDER IS 1 → 2 → 3 → 4 → 5, then Phase 1S.** Each rung is gated on the previous
rung's **holdout** number, never on `loss/val_epoch` — which is now doubly unusable, since
v5's val set mixes two domains and a move in it could be either.

### ⚠ How much is left on the table, measured 2026-08-08

Asked directly by the owner, because v5's drop counts read like the corpus had been
whittled down. It has been, and **the merge filters are the least selective stage in the
chain by an order of magnitude**:

| stage | in | out | kept |
|---|---|---|---|
| Emilia-YODAS, licensed (CC-BY) | **~114,000 h** | — | — |
| shards actually pulled | 9 | 238,203 clips / 8.4 GB | — |
| probe: sampled + base filters (duration, DNSMOS ≥ 3.0) | 238,203 | 66,217 clips / **159.8 h** | 27.8% |
| **mining tail-selection** | 66,217 | 13,141 | **19.8%** |
| merge (digits, caption WER) | 13,141 | 10,997 | 83.7% |

LibriTTS-R is the same shape: **51.3 h of ~585 h**, 8.8%.

**Nothing is wasted — but the corpus IS skewed, and the skew is deliberate.** The 53,076
probed-but-unkept clips are still on disk *with their acoustic measures already computed*,
so re-mining costs EIV + ASR, not re-measuring. What matters more is *why* they were
unkept: the criteria keep a clip only if it sits in a tail (`T > p90`, `V > p95`, `V < p5`,
`A > p95`), and the T threshold is **5.75 σ** — 5,002 keeps qualify on T+ alone. So v5's
Emilia half is not 27 hours of speech, it is **27 hours of extremes**, against a LibriTTS
centre, at roughly 3:1.

**That is what T's 53.6% saturation IS** — not a labelling bug but the mining criteria
arriving at the label. It is also why the prediction block below is worth more than it
looks: it is the instrument that tells us whether the skew hurt.

**Two options exist for de-skewing, and neither is scheduled:**
- **Widen the mining to ~p75** and re-mine the 53k already-measured clips — 3–4× the Emilia
  half, de-saturates T. Costs EIV + ASR on 53k clips, days of CPU, plus a re-derivation.
- **Pull more shards.** 160 h probed out of ~114,000 h licensed. Effectively unbounded.
  ⚠ **The FULL pool is storage-barred** (owner ruling 2026-08-25: viable, not committed):
  extrapolating the probe (160 h = 8.4 GB raw) puts all ~114,000 h at **~6 TB**, which the
  3.7 TB `/data` drive cannot hold — an external drive is the unlock, recorded in
  [training-sources.md](training-sources.md) beside HiFiTTS-2's identical bar. Incremental
  shard pulls are NOT barred: 10× the current holding is ~85 GB and fits trivially.

**Neither runs before rung 1's holdout.** Widening now would spend days removing a confound
we have not shown exists *and* change two things at once, which costs us the test. The
prediction disambiguates it for free: **if V and A improve, volume is the lever regardless
of what T does.** If T regresses in the predicted signature, that is when widening is
indicated — and the plan already names the fix (C-soft).

### What the Emilia merge established, and what it directs

- **A licence-manifest entry is required before a corpus can be merged**, not after —
  `classify_path` matches exact path COMPONENTS, so a directory the manifest does not name
  refuses the whole run. Verify provenance before declaring; `emilia_original` next door is
  a different licence.
- **Emilia labels use a GLOBAL anchor, not per-speaker z.** 46.6% of its speakers have
  under 10 clips, where per-speaker z is fixed by arithmetic rather than measured — it
  would hand 756 one-clip speakers a label of exactly 0.0, i.e. clips selected FOR being
  extreme, trained as neutral. The semantic cost (a global anchor leaves some speaker
  identity in the affect channels) is stated in the data config rather than hidden.
- **The speaker table widens by APPENDING.** LibriTTS speakers keep their indices and new
  ones append, so rows 0–246 keep their meaning — unlike the vctk 109 → 247 case
  `make_warmstart._WIDENABLE` refuses, where row *i* is a different person. The warm start
  verifies the index map rather than assuming it.

**T's saturation was accepted on a pre-registered bet, and the bet paid.** Read
2026-08-09 from `v5_ckpt_sweep` — no T-specific regression on the holdout: T's extremes
improved exactly as much as its centre (gap −0.0006, CI [−0.0065, +0.0055], straddling
zero). **The shortcut did not form; saturation is a non-issue at this mix.**

- **Do not re-open `tanh(z/2)` (C-soft) on this evidence.** It stays available if a later
  rung produces the signature; it is a contract change and a full relabel, so it rides a
  re-derivation rather than forcing one.
- ⚠ **PRE-REGISTERED FOR v6: `A`'s extremes lagged its centre** (+0.0133, CI excludes
  zero) — nobody predicted this and it is not acted on. **If v6's sweep shows A lagging
  again it is a real channel effect, and the loudness normalisation is the first thing to
  check** (A is loudness-derived, and Emilia's loudness distribution differs). If it does
  not reappear, drop it rather than carry it.
- The ear still owes T's perceptual legs — does T = +1 sound like *Emilia-domain audio*
  rather than like tension? Not urgent, since a loss cannot settle it and it cannot
  trigger C-soft alone, but a "T sounds like a podcast" verdict would be its own finding.

Stratify any rung's sweep by channel with `scripts/tools/stratify_holdout_sweep.py` — the
aggregate hides exactly this question.

⚠ **Separately, and it IS a blocker: `n_spks`.** The model's speaker table is **247** rows
and Emilia brings **2,408 new speakers**, so a merged corpus needs `n_spks: 2655` and a
`spk_emb.weight` widened 247 → 2655. That widening is *safe here and only here*: LibriTTS
speakers keep their existing indices and the new ones append, so rows 0–246 keep their
meaning — unlike the vctk 109 → 247 case `make_warmstart._WIDENABLE` deliberately refuses,
where row *i* is a different person. It needs the derivation to preserve the index map and
the warm start to verify that rather than assume it.

### Per-corpus checklist — every one of these has bitten

- **`configs/data_licenses.yaml` entry**, or `enforce()` refuses the run. This is what
  silently made v3/v3b/v3c unrunnable; "nothing has trained on v3c" was structurally
  guaranteed, not merely true. **Check this file first whenever a new corpus dir appears.**
- **VAT labels** via `derive_vat_corpus.py`.
- **Phonemes** via `rephonemize_corpus.py` on the **fixed** `op_g2p`.
- **`data_statistics` re-measured in-container** for the new mix
  (`matcha/utils/generate_data_statistics.py`). Do not inherit the previous version's
  mel mean/std silently — v2→v3c moved them by 0.0203/0.0024, a constant shift on every
  normalised mel.
- **Explicit corpus bind in compose.** The `data ->` symlink is untracked and does not
  survive a clone.
- **Split is automatic** — the hash split needs no attention, which is the point of it.
- **QC gate** after every generation pass, and **spin down all inference engines**
  before any run.

## Phase 1S — mass synthetic production, Qwen-primary

**Owner goal, 2026-08-08.** Generate training audio at a scale the licence wall cannot
reach through real sources, on a repeatable pipeline: Gemma directs → TTS renders → machine
QA admits, with the owner's ear out of the per-clip loop. *"Qwen has proven to be
trustworthy and high quality. Frankly, its output is shockingly realistic. I think we can
[get] high yields from it alone without a lot of QA pruning or 'dirty' clips."*

**SEQUENCED BEHIND RUNGS 1–5, by owner decision.** Not because the idea is unsound — the
measured case for Qwen is good — but because ~870 hours of cleared real audio yield ~100%
with zero ear time, and spending render-hours before conversion-hours is spending the
expensive currency first.

**IT HAS TWO USES AND THE SECOND IS THE BETTER ONE.** Bulk volume (§ Text supply) is what
the phase is named for; **targeted remediation** (§ below) — generating exactly the material
that fills a measured weakness in Sonora's output — is what it is *for*. The bulk lane asks
the model to absorb more; the targeted lane asks it to rebalance, which is cheaper, survives
the capacity ceiling, and is the only lever that can fill a corpus region real audio does not
happen to contain. Build the machinery once; the second use is where it pays.

### The measured case FOR it

**Qwen is the highest-yield synthetic engine we have, and it is not close on the axis that
matters.** Over every ear verdict in `ratings.csv`:

| engine | heard | keeps | **keep rate** |
|---|---|---|---|
| `librivox` (real) | 104 | 103 | **99.0%** |
| `emilia-yodas` (real) | 48 | 48 | **100.0%** |
| **`qwen`** | **293** | **260** | **89.3%** |
| `orpheus` | 130 | 99 | 86.1 |
| `zonos` | 183 | 147 | 82.6 |
| `chatterbox` | 160 | 125 | 81.2 |
| `vibevoice` | 116 | 76 | 65.5 |
| `dia` | 68 | 34 | 52.3 |

Qwen is `trusted` tier (1 clip/group + 3% heard), was **confirmed superior at equal
loudness** when the loudness confound was tested and ran backwards, and is the owner's
stated gold standard. An 89.3% keep rate is a ~11% prune, not "a lot of QA pruning" — the
owner's read is supported.

**Qwen is also the right engine for VARIETY, which is the non-obvious part.** VoiceDesign
takes **no reference audio** — timbre comes from the words alone, across twelve free-text
axes (gender, pitch, speed, volume, age, clarity, fluency, accent, texture, emotion, tone,
personality). Its speaker space is continuous and directable rather than bounded by a
reference bank, which is the opposite of Chatterbox, whose diversity is capped by our
reference pool (small, and carrying a blacklist). Leaning on Qwen for variety is the
mechanically sound call, not the risky one.

### The corpus becomes a RECIPE, not a pile of wavs

Owner, 2026-08-08: *"I don't think we need to carry all of it on disk. That would assume a
single training pass."* Correct, and it removes the storage wall entirely — 24 kHz/16-bit
mono is **172.8 MB per audio-hour**, so 100,000 h would be 17.3 TB against 3.1 TB free,
while a *streamed* 500 h shard is 86 GB.

**The two prerequisites already exist**, which is what makes this real rather than
appealing: renders are **seeded** (`make_bulk_bank` expands registers × line pools × job
templates × seeds; `seed` lands in every manifest beside text, direction, engine and
`weights_source`), and the environment is **digest-pinned** (`container_env.sh` pins
`rocm/pytorch@sha256:…` and every wheel — written for campaign reproducibility, which is
exactly what recipe-addressed data needs).

So a synthetic corpus is `manifest + pin`, regenerable on demand, and only the shard being
trained on has to exist. A manifest is also diffable, reviewable and version-controllable
in a way that 17 TB of wav is not.

⚠ **It trades storage for compute, and the exchange rate is the number of passes.**
Discard-and-regenerate pays render cost *per epoch*. At 100 epochs that is prohibitive; at
one-to-few passes it is the standard large-scale recipe. **That is a training-regime
change, not a data change** — we currently run `max_epochs: -1` with warm-started
100-epoch fine-tunes.

**The design that falls out: resident real corpus + streamed synthetic.** Keep the
real-audio rungs (~615 h LibriTTS-R full ≈ 106 GB) permanently on disk as the anchor and
stream synthetic shards over it. Bounded storage, no forgetting of the real base across
shards, and the holdout is untouched either way.

### Three prerequisites, none of which is the pipeline itself

⚠ **Resolve the Expresso conflict BEFORE this phase's register specs are written** (PR-L1).
It blocks nothing today and is well-recorded in
[training-sources.md § The Expresso conflict](training-sources.md) — two ratified decisions
pointing opposite ways, needing the owner. It becomes load-bearing exactly here: Expresso is
the best design reference we have for expressive style taxonomies (8 read + 26 improvised
styles), and the whisper/laughter gap this phase would remediate is precisely what it
covers. Deciding it after the specs are written means either rewriting them or quietly
not using the reference.

- [ ] **1 · Measure render throughput.** **No synth script records elapsed time** — checked
      2026-08-08 across every `synth_*.py`, and no manifest carries a timing field. The
      entire feasibility of this phase is arithmetic in a number nobody has ever measured.
      One afternoon. Do it first, and record it per engine per clip so it stays measured.
      Under streaming it matters *more*, since the cost may be paid repeatedly.
- [ ] **2 · The voice-design diversity probe.** The open question is not whether Qwen's
      design space is large — it is whether the model *realises* it. Many distinct
      descriptions collapsing to few distinct voices is the failure mode, and gate **G6**
      already encodes this exact logic for delivery ("the lanes must be mutually
      distinguishable, since five inputs on one summing junction pass the per-channel probe
      five times"). Same test, applied to voice designs.
      **Cheap version first**, no new dependency: render a designed grid of instructs and
      measure spread with what we already compute — F0 median and excursion, the phonation
      composite (`alpha_db`, `cpp`, `h1h2`), LUFS — against the spread of LibriTTS's 247
      real speakers as the yardstick. Only if that says *collapsed* do we add a
      speaker-verification embedder, which we do not currently have.

      **AND THE PROBE ONE LEVEL UP — engine STYLE, not just voice (owner, 2026-08-09;
      PR-M4).** The test above asks whether Qwen collapses to few TIMBRES across many
      designs. It cannot see the risk that actually threatens this phase: whether a
      **Qwen-primary corpus collapses to one STYLE** against the five-engine mix. Those are
      different failures — a corpus can hold two hundred distinct voices that all *phrase*
      the same way, and phrasing is what a prosody model learns. Run the same forced-ranking
      instrument the scale-saturation work produced, engine-blind, comparing a Qwen-primary
      sample against a mixed-engine sample on identical text.
- [ ] **3 · The direction-adherence gate — the one QA idea that reaches past structural.**
      Every synthetic clip already carries **both** Gemma's `intended` V/A/T (in the
      manifest) and the acoustically-measured V/A/T derived from the render. **Nothing
      compares them.** Their disagreement is a free, fully-automated expressivity check
      that needs no ear: directed high arousal + measured low arousal = the render did not
      follow direction, a defect no current gate sees. The signal is known to be large —
      the VAT audit measured markup at 93% but **valence at 62%**. Calibratable without the
      owner's ears against **E-VOC** (CC-BY-4.0, human ratings on precisely the
      instruction↔perception gap).

### The Qwen-primary exception, recorded rather than assumed (PR-M4)

**Owner, 2026-08-09.** Concentrating mass production on one engine is knowingly in tension
with an owner-ratified principle, and the tension is written down here so it cannot be
mistaken for an oversight later.

**The principle:** *"max keep-rate is not the objective; a one-timbre teacher corpus has
failed at its job."* It is why `MIN_SHARE` and `MAX_SHARE_NARRATION` exist, and why the
allocation layer carries a diversity floor and cap at all. **The distillation framing makes
this a first-order risk, not a corner case:** the standing lesson from the teacher-data
work is *copy their METHOD, not their sources* — and a single-teacher corpus copies one
teacher's style along with its method.

**The exception, and its reasoning.** Qwen-primary is a YIELD decision — 89.3% keep against
a mix whose other engines yield materially lower — and Phase 1S is the one lane where
throughput is the binding constraint, since it sits behind ~870 h of cleared real audio on
a single shared GPU. Diversity within the phase is bought by voice design rather than by
engine mix.

**What makes it revisable rather than a bet.** No `MAX_SHARE_QWEN` is set *yet*, and that
is deliberate: a cap chosen now would be a number with no measurement behind it, which is
the failure this plan names everywhere else. Prerequisite 2's widened probe is what supplies
the measurement. **If the style probe fires, a share cap analogous to
`MAX_SHARE_NARRATION` follows on evidence** — and the cap belongs in
`ref_select.ENGINE_MIX_BY_LANE` with the numbers that chose it beside it, like every other
constant here.

⚠ **Read this against the phase's real purpose.** Phase 1S's point is **targeted
remediation, not bulk** — filling measured gaps that no real audio covers. A remediation
lane is *less* exposed to monoculture than a bulk lane, because its output is a minority of
a corpus by construction. That is a reason the exception is defensible; it is not a reason
to skip the probe, since "minority" is an assumption that stops being true if the lane ever
grows.

### The Gemma split — settled, and it is variant-based not task-based

**`gemma-4-31b-qat-spec` writes direction; `gemma-4-e4b-qat-spec` does volume.** Recorded
in [book-prose-lane.md § Director model](book-prose-lane.md) and STATE.md. For a
variety-driven pipeline the decisive column is casting diversity: on 24 narration passages
31b produced **16 distinct castings** and E4B **5**, and E4B disobeyed the skill file on 19
of 24 lines. **E4B is fast and clean, not obedient.**

The cost objection does not bind, because **the direction pass is per-PASSAGE, not
per-clip** — run 31b over the passages once and reuse each casting across many renders and
seeds, so its 4× per-call cost lands on a small denominator. E4B keeps the high-count
mechanical jobs (`judge_passages`, the markup labeler), which is the `e4b-labels /
31b-judges` shape the markup spike already runs.

⚠ Terminology trap, since it has already caused one: *"e4b for volume, 31b for judgement"*
means the **director's** judgement. `judge_passages.py` runs on **e4b**.

### What the ear is still for, and why that does not scale with corpus size

The bottleneck is real but it is not where it looks. A `trusted` engine is already sampled
at **1 clip per group + 3%** — the owner is already not listening to 97% of Qwen output.
What the ear is actually for is **calibration and novelty**, and neither scales with hours;
they scale with the number of distinct **defect modes**. 100,000 hours from one engine at
fixed settings needs roughly the ear time of 1,000 hours from that engine.

⚠ **The perceptual-QA ceiling is a property of the measures, not of effort, and it stands
at any scale.** Seven documented attempts to replace the ear on quality have failed, two of
them *anti-correlated*: DNSMOS cannot separate expressive from broken in its 2.0–2.6 band;
room reverb defeated **four** detector attempts; `radio_score` misses the worst offender
(0.3299 against a 0.10 bar, 1 of 20 flags); `head_ok` runs opposite the ear; `PAUSE_HARD_MAX`
cannot be tightened because duration is the wrong axis; clip incompleteness is undetectable
by design; and the 1–5 scale itself saturated. **Structural failure is machine-detectable;
perceptual quality is not.**

The sharp consequence for mass production: **sampling catches prevalence, not severity.** A
defect at 0.5% prevalence in 100,000 hours is 500 hours of poisoned corpus that no
affordable sample rate will see. Sonora trains on keeps, so anything the detectors cannot
see is in the corpus at scale — this is the ceiling-set-by-the-weakest-keep problem arriving
through volume instead of through labels.

⚠ **And the unnamed risk underneath all of it: there is exactly ONE ear** (recorded
2026-08-09, PR-L1). Every scale plan here calibrates against a single rater. The docs are
scrupulous about the *instrument's* ceiling and silent about the *rater's* — availability,
drift over months, and a bus factor of one. Two mitigations already exist and neither
addresses singularity: forced ranking (which fixes scale saturation, not drift) and anchor
exemplars (which fix drift, not availability). **What would actually reduce it:** a small
held-back re-rate set the owner scores again at intervals, so drift becomes measurable
rather than assumed absent; and writing the *verdict rationale* down often enough that a
second rater could be calibrated against it later. Named as a risk rather than solved —
naming it is what makes it schedulable.

### Two risks specific to scale, recorded before they bite

- **PerTh watermarking.** Chatterbox embeds an inaudible watermark in **every** output. At
  1,000 h that is noise; at 100,000 h of Chatterbox it could be a dominant corpus signal,
  and nobody has checked whether it survives our resample and loudnorm — let alone whether
  Sonora would learn to reproduce it. Qwen-primary largely sidesteps this; it becomes live
  the moment Chatterbox is used at volume.
- **Capacity, which may make the whole phase moot.** The acoustic model is **22.7M
  parameters**. LibriTTS-R full is 585 h and trains good TTS; even 5,000–20,000 h is 10–35×
  that into a model sized for hundreds. **Rung 1's holdout is the instrument** — if +36%
  does not move it, the diagnosis is capacity-limited rather than data-limited and mass
  generation buys nothing until the model grows, which puts the Phase 2 decoder spike on the
  critical path rather than after it.

### Text supply — the whole Standard Ebooks library

Owner, 2026-08-08: *"We could theoretically run the entire Standard Ebooks library through
Qwen in a variety of narrator voices and variations of embodiment. Even if it overlaps
existing LibriTTS-R, I don't think we would lose out."*

**The overlap worry is a rounding error, and the proposal is stronger than "no loss."**
Standard Ebooks is ~1,500 titles. At ~90k words per novel and ~150 wpm that is **~10 h per
book, ~15,000 h per single-voice pass.** All of LibriTTS-R full is **585 h ≈ 5.3M words ≈ 59
average novels** — so **Standard Ebooks carries ~25× more distinct text than the entire
LibriTTS-R corpus**, and the overlap is at most a few percent of the pool. This is a large
net gain in text diversity that happens to *include* some repetition, not repetition with
acceptable waste.

The overlap is not waste either: an identical line read by a real human *and* rendered by
Qwen is the cleanest signal we could have that prosody is separable from text, and nothing
else in the corpus provides it.

**The lane is mostly built.** `books_ledger.json` already tracks **56 entries** across `se:`,
`pg:` and `lv:` keys, behind `book_router` and `book_ingest`.

Licence is clean throughout — Standard Ebooks' own contributions are CC0, the works are
public domain, and Qwen's weights are Apache-2.0.

- [ ] ⚠ **PREREQUISITE — close A-M6 before any mass ingest.** `book_ingest` hardcodes
      provenance `"Standard Ebooks CC0"` **even for Project Gutenberg sources**. At 21 books
      that is a paperwork defect; at 1,500 it stamps false licence metadata across the
      largest corpus we would own, in every derived clip's paper trail, and it is not
      fixable after the fact. Filed in [todo.md §5](todo.md); it is now load-bearing.
- [ ] ⚠ **THE HOLDOUT EXCLUSION GUARD — this is the trap the idea walks into.**
      `data/libritts_r_holdout_devclean` is worth exactly one thing: no checkpoint has
      trained on it. Rendering **dev-clean's source books** through Qwen and training on the
      result is a subtler destruction of it — the model never hears those recordings but it
      learns those exact phoneme sequences and their durations, so duration loss especially
      comes back optimistically biased and nothing downstream flags it.
      **It is precisely fixable and the metadata exists:** LibriTTS-R ships `*.book.tsv` per
      chapter, dev-clean has **96 distinct chapters**, and their titles are recoverable from
      the first utterance of each (`Renaissance in Italy: The Catholic Reaction`, …). Build
      the exclusion list once and have the render lane **refuse** a title on it — same shape
      as the licence wall, an allowlist that refuses what it cannot classify, not a filter
      that silently drops.
- [ ] ⚠ **Shard format is a WRITE-TIME decision, not a repackaging step.** HF caps repos at
      **<100k files** and hard-caps **10k entries per folder**. 15,000 h at ~12 s/clip is
      **~4.5M files — 45× the recommended cap.** So the render lane must emit **WebDataset or
      Parquet shards** from the start; that is what Emilia itself ships as
      (`format:webdataset`) and what HF recommends. Storage is *not* the constraint —
      at 172.8 MB/audio-hour a PRO account's 10 TB public allowance is **~58,000 audio-hours**,
      more than the whole real-audio ladder plus a Standard Ebooks pass.

⚠ **Embodiment variations are renderable but not yet LABELABLE.** Embodiment is a clip whose
delivery changes partway through; it stays delivery-blank by contract until span-FiLM ships
(the 17 keeps are its pilot set). Rendering embodiment variety now is fine and the clips are
useful, but they enter as `unknown` and the capability to condition on them does not exist
yet — do not let the render spec imply otherwise.

### Targeted remediation — the pipeline's second and better use

Owner, 2026-08-08: *"As fine tuning continued in cycles, we will eventually reach a point
where we will be filling in weaknesses in Sonora's output. We can use this proposed pipeline
as a means of generating training data to precisely target those weaknesses."*

**This is the stronger justification for building the pipeline, and it should be read as the
destination rather than a later bonus.** Bulk generation asks the model to absorb more;
targeted generation asks it to *rebalance* what it sees, which is a much cheaper request and
answers two of this phase's own objections:

- **It survives the capacity ceiling.** If rung 1's holdout says capacity-limited rather than
  data-limited, bulk volume buys nothing — but filling a specific gap can still move a
  specific channel, because it changes the distribution rather than the quantity.
- **It is the direct remedy for corpus skew.** T saturates at 53.6% on the Emilia half and
  delivery is empty on 41,093 of 41,138 rows. Targeted rendering is exactly how you fill a
  V/A/T region or a delivery lane the corpus does not cover, and it is the *only* lever that
  can, since real audio arrives with whatever distribution it has.

**Two thirds of the instrument already exists.** `score_holdout.py` scores every checkpoint
per clip across 5,463 unseen clips — that is already a per-clip weakness map. The standing
perceptual tests give a channel-level one (vat3-24k: energy PASS, tension near-pass,
**valence FAIL**). `render_vat_sweep.py` renders across the VAT space. **What is missing is
the loop**: nothing takes a scoring result and emits a render spec, and that mapping is the
whole build.

- [ ] ⚠ **DO NOT DIAGNOSE ON THE HOLDOUT.** This is the methodological trap, and it is the
      same class of error as training on it. If weaknesses are identified from the holdout
      and then data is generated to fix precisely what the holdout reports, we are fitting to
      the holdout and it stops being an unbiased measure — silently, and permanently, with no
      second dev-clean to fall back on. **Diagnose on a separate diagnostic split, on the ear,
      and on the standing perceptual tests; use the holdout ONLY as the confirmation that the
      remediation worked.** Carving that diagnostic split is a prerequisite of this lane, not
      a detail of it.

---

## Phase 2 — the DiT decoder spike

Design, gates and known risks are already written: **[model-decisions.md § Decoder v2](../docs/model-decisions.md)**
(the content of the former `decoder-v2-dit-spike.md`, folded in during the 2026-08-02
consolidation — the standalone file no longer exists). Start from StableTTS's 31M
`DiTConVBlock` shape, which retires the spike's named tiny-scale risk in public MIT code
([matcha-siblings-study.md](matcha-siblings-study.md)).

What this file adds is **when**, and one amendment:

- **After Phase 1 lands and Phase 0 works.** Architecture work is CPU-side until its
  de-risk run, so authoring can overlap; the *gate* cannot.
- **The parity gate must run against a U-Net baseline trained on the same expanded
  corpus.** Otherwise data and architecture confound each other and the run teaches
  nothing about either. Freeze that baseline as the last act of Phase 1.
- Adopt iff the gate passes; **stall → the stock decoder runs the corpus** and the spike
  parks with its findings. The schedule never waits on the swap.

This remains a spike, not a pivot: the decoder is ~7.6% of the codebase.

---

## Gates between phases

| gate | question | consequence of the answer |
|---|---|---|
| ~~after **0a**~~ **PASSED 2026-08-06** | do retained checkpoints separate at all on never-trained audio? | **yes** — `vat3_ep099` − `vat3_init` = −0.0111, CI excludes zero. Instrument is sound; 0b closed, Phase 1 proceeds from `vat3_ep099` |
| ~~after **1.1**~~ **PASSED 2026-08-09** | does **+36%** move the clean holdout? | **yes** — `ep019` − `vat5_init` = **−0.0606**, improved on 82.1% of 5,463 clips, roughly 5× the −0.0111 that cleared gate 0a. Data-limited, not capacity-limited: **the 10× proceeds.** (**+36% is what shipped**; +43% was the planned merge, before 1,676 digit rows and 468 WER rows dropped out. Quote the shipped figure — a gate naming a number the corpus never had cannot be checked against it) |
| after **rung 2 (v6)** | does the **delivery channel** show signal? | see the pre-registered criterion below — the holdout **cannot** answer this one |
| after **Phase 1** | is the vocoder still transparent? | re-run `copy_synth.py`; a better acoustic model can walk into the vocoder's ceiling unnoticed |
| before **Phase 2** | is there a same-corpus U-Net baseline? | if not, the parity gate is unreadable — do not start |

**Gates need an effect size written BEFORE the run, and most of these do not have one
(PR-M2).** Gate 0a did — it was adjudicated on bootstrap CIs — and the T-saturation
prediction did, which is exactly why both could be read as pass or fail rather than
narrated afterwards. "Does it move the holdout" without a criterion is a question that
cannot be failed. Proposed below (assistant's, unless the owner sets otherwise); the point
is that a number exists in advance, not that it is this number.

- **A rung PASSES its holdout gate** when its `pick − init` mean Δtotal is negative and its
  95% bootstrap CI excludes zero, computed init-relative under that rung's own constants
  (`scripts/tools/stratify_holdout_sweep.py` reports both). **Init-relative is not a detail** —
  SELECTED.md records that absolute holdout numbers do NOT compare across a
  normalization-constant change, and every rung re-measures `data_statistics`, so a gate
  written on absolutes would compare noise across rungs and read it as a result. What
  carries across rungs is the DELTA plus byte-identical row carry-forward.
- **A rung is INCONCLUSIVE, not passed,** when the CI straddles zero. Gate 0a's −0.0111 is
  the smallest movement the instrument has resolved on this holdout; treat anything of that
  order as the floor rather than as a result.
- **⚠ Structural limit, and it is not a threshold problem.** The holdout is clean audiobook
  narration scored on teacher-forced loss. Rung 2's lever is **delivery**, rung 4's is
  **depth per voice**, rung 5's is **in-the-wild expressivity** — *the instrument cannot see
  any of them directly.* Rung 2's gate column read "the build itself", which is a
  description, not a measurement. **The right instrument already exists in this plan: the
  manner-vs-timbre test** — direct one real speaker through each of the five lanes and have
  the ear say whether the manner changed while the voice did not. That is what should gate
  rung 2, and `stratify_holdout_sweep.py --metric diff` is the supporting read, not the
  verdict. Wiring it in is open work.
- **Pre-registered now, because it will otherwise be misread (PR-M2).** Rungs 3–5 dilute the
  delivery-labelled share from ~2.3% (v6) to ~0.3% (v7). **A channel that learns at v6 and
  then washes out at v7 is dilution, not breakage**, and the two are indistinguishable
  after the fact. Record v6's per-lane delivery result before v7 trains, and read any v7
  regression against it — the same inoculation this plan already wrote for the *untrained*
  channel.

## Why data before decoder

Data first is lower-risk, needs no architecture work, and answers the question that
governs everything else: whether ~30k clips was ever the constraint. Doing the spike
first would mean evaluating a new decoder against a corpus we are about to change —
the same confound that made the vat3c run so hard to read.

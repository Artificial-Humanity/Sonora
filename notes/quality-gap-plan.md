# Closing the quality gap — the ordered plan

**Scope: what we do next to close the synthetic-vs-real gap, and in what order.**
Measurement repair → data sequencing → decoder spike. This file owns the *sequencing
and the gates between phases*; it does not restate the plans it sequences. The DiT
design itself is [model-decisions.md § Decoder v2](model-decisions.md); source state is
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
scoped and decided but NOT built.**

| | step | status | gated on | detail |
|---|---|---|---|---|
| **P0** | 0a — never-trained holdout | ✅ **DONE 2026-08-06** | — | [§ 0a](#0a--the-never-trained-holdout--done-and-it-reported-2026-08-06) |
| | 0b — clean-lineage restart | ⛔ **NOT INDICATED** | — | [§ 0b](#0b--clean-lineage-restart--not-indicated-owners-call-to-ratify) |
| **P1** | **rung 1 — v5, +Emilia (78.5 h, 2,500 spk)** | ✅ **DONE 2026-08-08/09** — 48 epochs, holdout-scored, **`ep019` selected**; converged by epoch 9 | — | [§ the ladder](#the-ladder--a-strictly-growing-corpus-one-lever-per-rung) |
| | **rung 2 — v6, +expressive-registers (1,004 eligible keeps)** | 🔨 **decided, NOT built** — 3 prerequisites open | the build itself (the Qwen audition was **deferred**, owner 2026-08-09, and does not gate this) | ditto |
| | rung 3 — v7, +LibriTTS-R full (~615 h) | ⏸ **UNGATED — rung 1 passed.** ~**2.25 h/epoch, ~1 day** to convergence (measured 2026-08-09) | rung 1's holdout ✅ | ditto |
| | rung 4 — v8, +Hi-Fi TTS (292 h) + VCTK (44 h) | ⏸ **both ON DISK, unconverted** | independent — slot in when converted | ditto |
| | rung 5 — v9, +more Emilia-YODAS shards | ⏸ 9 of ~114,000 h probed | rung 3's holdout | ditto |
| | freeze a same-corpus U-Net baseline | ⏸ | **the last act of P1** | [§ Phase 2](#phase-2--the-dit-decoder-spike) |
| | vocoder transparency re-check | ⏸ | after P1 | [§ gates](#gates-between-phases) |
| **P1S** | mass synthetic production, Qwen-primary | ⏸ **plan written, 3 prerequisites open** | **rungs 1–5** (owner: exhaust public sources first) | [§ Phase 1S](#phase-1s--mass-synthetic-production-qwen-primary) |
| **P2** | DiT decoder spike | ⏸ design written, not started | P1 landing + the frozen baseline | [model-decisions.md § Decoder v2](model-decisions.md) |

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
| **P3+** | conditioning: embodiment bank → span-FiLM | ⏸ deferred chain | evidence, not plumbing | [todo.md §4](todo.md) |
| | categorical emotion block (3+8) | ⏸ **open decision** | P1's valence read | [todo.md §4](todo.md) |
| | multilingual | ⏸ **plan only** | after rung 3 | [teacher-training-data.md § multilingual](teacher-training-data.md) |

**Running alongside, needing no GPU:** the audit-quality pair in [todo.md §6](todo.md) — a
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

## What scoped this (2026-08-06)

`vat3c_finetune` ran 100 epochs off the v2-trained warmstart onto v3c's clean IPA
(run `2026-08-05_15-02-57`, exit 0). Three measurements came out of it, and together
they say where the remaining quality lives.

**1. The G2P fix was the whole fix — retraining was never needed.** Per-clip *paired*
scoring (same torch seed per clip/draw across both checkpoints) of `vat3c_init` vs
`epoch=099`, over the 2,784 clips whose phonemes changed between v3 and v3c, against
2,782 speaker/length-matched controls:

| term | corrected | control | attributable to the fix |
|---|---|---|---|
| prior | −0.560% | −0.427% | **−0.133%** |
| dur | −5.568% | −4.158% | **−1.411%** |
| diff | −2.554% | −2.242% | **−0.311%** |

All three significant (bootstrap CIs exclude zero), all perceptually negligible. Blind
A/B renders confirmed **no audible difference**. The poisoning was text→phoneme only;
the model's phoneme→audio mapping was always sound, so the warmstart — which never
trained on a single corrected contraction — already says `I'm` and `that's` correctly.
Fixing `op_g2p` closed it. *Generalizes: check the front end before spending a run.*

**2. The vocoder is not the bottleneck.** Copy-synthesis (real audio → mel with exact
training params → `cp_hifigan_24k/g_02510000`, no acoustic model in the path) gives mean
**mel L1 = 0.2447**, uniform across 10 clips = **10.2% of this corpus's mel_std (2.389)**.
Owner's verdict: *"similar to ref, if not identical."* This **rules out** vocoder
training, vocoder replacement and sample-rate work as quality levers.

**3. The gap belongs entirely to the acoustic model's predicted mel.** That is what both
phases below target, and it is why they are the right two phases.

**What this does NOT support:** starting the project over. The architecture responds to
conditioning, the corpus is sound, the 1,802 human ratings are untouched, and no
checkpoint was ever selected by the broken metric (`monitor: epoch`, not val loss).

Harness kept at `/data/model-training/sonora/eval_phoneme_fix/` —
`build_sets.py`, `eval_phoneme_fix.py`, `render_ab.py`, `copy_synth.py`.

### Two traps this exposed, for any future comparison

- **Always randomise A/B assignment per pair.** The owner's "the second one sounds
  better" tracked *position*, not checkpoint — per-pair randomisation caught it because
  A/B was reversed on the pairs heard. A fixed A=old/B=new layout would have recorded a
  false positive.
- **Copy-synthesis must skip `normalize()`.** `TextMelDataset.get_mel` normalises for
  *model* input, but `MatchaTTS.synthesise` returns `denormalize(...)` before vocoding —
  **the vocoder consumes RAW log-mel.** Normalising first produces garbage. Assert the
  vocoder's `config.json` against the data config's mel params first (they agree:
  24000/80/256/1024/0/12000).

---

## Phase 0 — make it measurable

Everything downstream is wasted if we cannot tell whether it worked. Until 2026-08-06 we
could not: `loss/val_epoch` has never been a generalization measure, because splits were
re-drawn per corpus version while runs warm-started from the previous version. **93–97%
of v3c's val clips were trained on under an earlier corpus.** Cross-run val comparisons —
the 3.18 (07-20) vs 3.48 (08-05) pairing in particular — are invalid.

**0a closed that.** There is now a never-trained holdout and every retained checkpoint
has a number on it; read 0a below before reading any MLflow curve. The rest of this
section is kept because the defect it describes is permanent — the historical
contamination does not heal, so `loss/val_epoch` stays unusable for cross-run
comparison even now.

Going forward the split itself is fine: `derive_vat_corpus._in_val` hashes the wav
basename (blake2b, `SPLIT_SALT="sonora-vat-split-20260802"`) as of 2026-08-02, and adding
2,000 rows moved 0 of 960 val clips. **The contamination is historical**, inherited
through the warm-start chain. It does not fix itself.

### 0a — the never-trained holdout — **DONE, and it reported. 2026-08-06**

**LibriTTS-R `dev-clean`**, derived scoring-only as
`data/libritts_r_holdout_devclean`: 5,463 clips / 8.7 h / **40 speakers, zero overlap**
with the corpus's 247. Every retained VAT checkpoint scored teacher-forced with four
paired noise draws per clip — 43,704 per-clip rows, `scripts/score_holdout.{py,sh}`,
raw at `/data/model-training/sonora/holdout_eval/lineage.csv`. No training was done.

| checkpoint | dur | prior | diff | **total** |
|---|---|---|---|---|
| `vat3_init` (warm start of the v2 run) | 0.5369 | 1.0351 | 0.2903 | 1.8623 |
| **`vat3_ep099`** (v2 corpus) | 0.5466 | 1.0384 | 0.2662 | **1.8512** |
| `vat3c_init` | 0.5466 | 1.0384 | 0.2662 | 1.8512 |
| **`vat3c_ep009`** | 0.5463 | 1.0380 | 0.2663 | **1.8506** |
| `vat3c_ep029` | 0.5533 | 1.0393 | 0.2679 | 1.8604 |
| `vat3c_ep049` | 0.5468 | 1.0393 | 0.2693 | 1.8554 |
| `vat3c_ep069` | 0.5540 | 1.0389 | 0.2678 | 1.8607 |
| `vat3c_ep099` (shipped as the v3c result) | 0.5562 | 1.0400 | 0.2714 | 1.8676 |

**1. The gate passes: checkpoints separate on never-trained audio.** `vat3_ep099` beats
its own warm start by **−0.0111** [−0.0131, −0.0091]. The instrument resolves the
lineage, so the speaker-conditioning floor (dev-clean's 40 speakers land on arbitrary
trained embeddings — see `score_holdout.py`) did **not** swamp the signal, and the ECAPA
nearest-voice mapping held in reserve for that case is **not needed**.

**2. The v2 fine-tune genuinely generalized.** Its gain is concentrated exactly where it
should be: `diff` — the flow-matching objective that actually generates the mel —
improved **−0.0241, on 78.7% of clips**, while `dur`/`prior` gave back a little.

**3. The v3c fine-tune was a regression, not a no-op.** `vat3c_init → ep099` is
**+0.0164** [+0.0148, +0.0180], better on only **39.1%** of clips, and **all three
terms worsened** — there is no component it bought. It is monotone-ish from `ep009`
onward, and `ep009` is statistically tied with the init it started from (−0.0006,
CI spans zero). *The best checkpoint in the entire lineage is `vat3_ep099`.*

**4. Same verdict on v3c's own val split**, which it trained on: `vat3c_ep099` loses to
`vat3_ep099` by **+0.0443**, better on just **14.5%** of 960 clips. So this is not
holdout-specific and not a distribution artifact — the run is worse everywhere under
teacher-forced loss.

**5. It is not a normalisation artifact.** Re-scored under the **v2** constants the
checkpoint was trained with, the gap is +0.0172 against +0.0164 under v3c's. The
asymmetry also ran *against* the conclusion — `vat3_ep099` won while being scored under
constants it never saw.

**6. `vat3c_init` is bit-identical to `vat3_ep099`** across all 5,463 clips (Δ = 0.0000,
every clip). The warm start is now verified by measurement rather than by a 338/338 log
line, which also means a checkpoint's per-clip numbers are reproducible within a run.

This is consistent with the ear ("no audible change") and sharpens it: the ear could not
resolve a 0.9% move, and the direction is *down*. It also lines up with the two findings
that bracket it — the vocoder is transparent, so the gap is the acoustic model's; and
the v2→v3c label change was tiny (corr ≥ 0.9993, a phoneme fix on 6.4% of rows). **100
more epochs against ~30k clips a model was already fit to had almost no new signal to
learn, so it sharpened to the training set instead.** That is the evidence Phase 1 was
assuming: more epochs on this corpus is not the lever; more corpus is.

**Actions.** Retire `vat3c_ep099` — do not stage or export it. `vat3_ep099`
(= `vat3c_init`) stays the base and the warm start for Phase 1. Score every future
checkpoint on the holdout as a matter of course; it costs ~100 min for eight checkpoints
on an idle card and it is now the only honest number we have.

### 0b — clean-lineage restart — **NOT INDICATED. Owner's call to ratify.**

0b was conditional: *"if 0a shows the fine-tune lineage is genuinely compromised rather
than merely unmeasured."* It does not. The lineage **generalizes** — finding 2 is a real
gain on audio no checkpoint had seen, which is precisely what a compromised lineage
could not produce. What 0a found is narrower and cheaper to fix: **one wasted run**, not
a poisoned ancestry. Dropping `vat3c_ep099` recovers everything the restart would have.

That matters because 0b is **a retrain, not a fine-tune**: `matcha_vctk` is 22.05 kHz,
VCTK speaker set, no VAT trunk, so the decoder starts over at 24 kHz with our
conditioning. *(An earlier verbal framing of this as "near-zero cost" was about
abandoning the fine-tune lineage, which is cheap; the retrain is not.)* On this evidence
that cost buys nothing the holdout does not already give us. **Recommendation: close 0b
and go straight to Phase 1** from `vat3_ep099`. Reopen it only if a Phase 1 run with
materially more data also fails to move the holdout, which would point at the ancestry
rather than the corpus.

### What the holdout is, and the one way to destroy it

`data/libritts_r_holdout_devclean` is worth exactly one thing: no checkpoint has trained
on it. One epoch ends that permanently, and there is no second dev-clean. The directory
therefore ships with a `README.md` saying so, and `derive_vat_corpus.py`'s `train_op.txt`
/ `val_op.txt` were **deleted** after concatenation into `holdout.txt`, so no file in it
carries a name a training config would accept. Note that `configs/data_licenses.yaml`
*does* declare it — the wall gates on provenance, not intent, and would happily let a run
train on it. The naming is the guard.

---

## Phase 1 — data sequencing

Cheapest first, so the expensive item is decided by evidence rather than assumption.

| # | Source | State | Adds | Work |
|---|---|---|---|---|
| 1 | **Emilia-YODAS keeps** | **MERGED 2026-08-08** → `libritts_r_emilia_vat_v5` | 10,997 clips / +27.2 h (**+36%**) | done; needs a GPU slot |
| 2 | **sonora-expressive-registers** | DERIVED, **1,004 eligible keeps** | small, ear-certified | **merge as v6 at 1,004** (owner-scoped 2026-08-09). ⚠ This cell said "close **+116** (Neutral +81, Documentary +35) → 1,156, then merge" until 2026-08-09 — superseded, and it silently re-opened the owner's **Documentary close** (87/92, closed because *no documentary real audio exists*) |
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
| 2 | v6 = +expressive-registers | ~42,140 | ~81 | ~2,510 | **the delivery channel's only training signal** — see below | delivery channel finally has signal |
| 3 | v7 = +LibriTTS-R full | ~330,000 | ~615 | ~4,900 | **10× volume, same domain** | the real lever |
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

**v6 scope (owner, 2026-08-09): 1,004 eligible keeps** — 902 delivery-labelled + 102
delivery-blank. Filtered from 1,279 total keeps by licence and standing policy, all four
exclusions honoured as written: VibeVoice/Dia 133 (benched), moss85 83 (un-SFT'd base),
longcat 45 (benched), higgs3-NC 8 (**NC — never trains**). Per lane after filtering:
Dialogue 402 · Neutral 270 · Documentary 85 · Newscaster 76 · Speech 69.

⚠ **Known confound, accepted deliberately rather than bought off.** Dialogue, Documentary
and Newscaster have **no real audio at all** — no documentary real audio exists, and the
same holds for the other two — so after the merge, *delivery-labelled ⇒ ~91% synthetic*
while *blank ⇒ ~99.8% real*. The model can satisfy the loss by learning "labelled means
render like a teacher engine" instead of a manner of address. Removing the confound is not
purchasable: matching the labelled-rate across provenance needs **~36,700 blank synthetic
renders ≈ 336 GPU-hours**, and would make the corpus ~47% synthetic — contradicting the
ordering principle above. So we **measure it instead**: direct a real, well-represented
speaker through each lane and check whether manner changes or only timbre. Predict before
training, not after. The 57 synthetic blank keeps in the merge help slightly and do not
close it.

#### Rung 2 build decisions — recorded 2026-08-09, NOT BUILT, NOT LAUNCHED

The corpus does not exist yet and no run is queued. What follows is settled so the next
session starts from a decision rather than a re-investigation.

**SPEAKER IDENTITY — one speaker id per clip (owner, 2026-08-09).** Each merged clip gets
its own id appended after v5's 2,500, taking the table to ~3,500. It makes no false claim,
and the precedent is already in the corpus: v5 carries **756 one-clip speakers**. Rejected:
*one id per engine* (10 ids — asserts all 147 zonos clips are the same person, which is the
speaker table lying); *recover true identity from the manifests* (real archaeology, see
prerequisite 2 — the information largely is not there).

**Three prerequisites, measured 2026-08-09.** None was in the 1,004-row scope, and each is
work rather than a lookup:

1. **V/A/T is not derived for the bank.** Only the 193-row v1 slice carries `measured_z`;
   the remaining ~810 eligible keeps have none, and `eiv_scores/` covers the corpus lanes
   with **zero id overlap** with this bank. Needs an EIV pass before any merge.
2. **Voice identity is not recoverable from what was recorded.** **880 of 1,004** ids carry
   no `_sSEED` suffix (the naming convention changed across campaigns) and only 114 rows in
   any manifest carry a reference/voice field. Parsing identity out of ids collapses the
   bank into per-engine buckets — which is exactly the false claim decision 1 avoids. This
   prerequisite is *why* one-per-clip is the honest answer, not merely the easy one.
3. **~91 rows are copies of audio already in v5.** `libritts-r` (43) and `emilia-yodas`
   (48) keeps are audit-set copies (`audit-sets/tension-v2/…`, ids encoding `spk122`,
   `spk136`) of corpora v5 already contains. Merging them would hand a voice that already
   owns a speaker row a **second** one — the precise failure `merge_emilia_corpus.py`'s
   collision check exists to stop ("one voice would get two rows"). Resolve them back to
   their existing ids or drop them; do not append. The 103 `librivox` keeps are ours and
   genuinely new.

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

**DISCHARGED 2026-08-09 — this block no longer gates v6.** It read "WHY THIS COMES
FIRST, AND IT IS NOT OPTIONAL: before v6 trains, put ep009 · ep019 · Qwen in front of the
ear", and all three of its justifications were retired by the Step 0 records directly
above — written seven minutes apart and never reconciled, so the section argued against
itself for a day. Kept as a record rather than deleted, because *why* each leg fell is
worth more than the instruction was:

1. ~~Settles ep009 vs ep019.~~ **Settled**: the ear heard no difference and the owner took
   **`ep019`** on total. The null IS the result — neither loss margin is perceptually real,
   so ep009's 0.0013 `diff` edge is not grounds to reopen it.
2. ~~Measures the Sonora↔Qwen gap.~~ **Held until after rung 3**, on model maturity rather
   than time: run today it measures 78.5 h against ~5,000,000 h and returns "more data",
   which is the answer we already have. Does not gate rung 2.
3. ~~Captures what Sonora sounds like before the delivery channel exists, which cannot be
   reconstructed once v6 trains.~~ **Simply wrong**: `ep019` is retained on disk and v6
   trains into a new run directory from its own warm start, so nothing consumes or
   overwrites it. The checkpoint IS the baseline.

A session following the old text literally would have blocked v6 on a listen the owner
waived and re-run a comparison already settled. **The generalisable lesson is #3's:**
before treating an artifact as perishable, check whether the thing that produces it is
retained.

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

### The low-hanging fruit, itemised — ~870 h before anything is rendered

Everything here is **licence-cleared** and costs conversion work rather than GPU-hours.
Hours are the sources' own figures (`training-sources.md`), not measured by us.

| source | state | hours | what it actually costs |
|---|---|---|---|
| **Hi-Fi TTS v1** | **40 GB of parquet WITH audio, ON DISK, never opened** | **292** | parquet → wav + filelist. The largest corpus we own and the only one needing no download |
| **LibriTTS-R, the other 90%** | not pulled — we hold `train-clean-100` only | **+534** | download + corpus build, same pipeline as v4 verbatim |
| **VCTK** | **4.3 GB zip ON DISK, still unextracted** | **44** | `unzip` + filelist. 110 speakers, 48 kHz studio. Casting/accent breadth only — no conveyance value |
| Emilia-YODAS, more shards | 9 shards probed of ~114,000 h licensed | unbounded | the mining pipeline exists and ran this week |
| sonora-expressive-registers | derived, **1,004 eligible ear-certified keeps** (scoped 2026-08-09) | small | merge as v6 — see § Rung 2 below for the eligibility filter and why this rung is not optional |
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

**Neither runs before rung 1's holdout.** Widening now would spend days removing a confound
we have not shown exists *and* change two things at once, which costs us the test. The
prediction disambiguates it for free: **if V and A improve, volume is the lever regardless
of what T does.** If T regresses in the predicted signature, that is when widening is
indicated — and the plan already names the fix (C-soft).

### ⚠ #1 is not a merge. Prep pass 2026-08-08, before any GPU time.

Two things would have gone wrong silently, and the second invalidates the experiment
rather than breaking it.

**The licence wall would have refused it** (fixed, `configs/data_licenses.yaml`). The keeps
live in `emilia_kept/` and `emilia_kept_24k/`; the manifest declared only `emilia_yodas`
and `Emilia-YODAS`, and `classify_path` matches exact path COMPONENTS — so no keep matched
anything. Provenance was verified before declaring, because `emilia_original` next door is
NC: all 13,141 manifest entries carry the YODAS CC-BY-4.0 note and trace to the nine
`EN-B0000xx` shards in `emilia_yodas_probe`.

**Per-speaker z would destroy the exact property the corpus was mined for.** This is the
one to read twice, because the run would complete, the loss would look ordinary, and the
conclusion — "volume does not move quality" — would be wrong.

The keeps are **deliberately tail-selected**: `mine_emilia_keeps.py`'s pre-registered
criteria keep a clip only if `T_full > p90`, `V_combo > p95`, `V_combo < p5` or
`EIV-Arousal > p95`. As mined, on the LibriTTS-anchored global scale, they look like it —
**A mean +1.168, T mean +4.541, 91.9% of clips beyond |T| = 1**.

But the corpus lane computes V/A/T as a **per-speaker z**, and Emilia is
**13,141 clips across 2,408 speakers — a median of 3 clips each**, against LibriTTS's 127:

| clips/speaker | speakers | clips | share |
|---|---|---|---|
| = 1 | 756 (31.4%) | 756 | 5.8% |
| ≤ 2 | 1,170 (48.6%) | 1,584 | 12.1% |
| ≤ 5 | 1,744 (72.4%) | 3,749 | 28.5% |
| ≤ 10 | 2,057 (**85.4%**) | 6,126 | **46.6%** |

Re-centring each speaker on their own kept clips takes V from mean **+0.387 / sd 0.741** to
mean **+0.000 / sd 0.971** — the tail richness is annihilated by construction — and **756
clips selected FOR being extreme come out labelled exactly 0.0**, which in this
representation means *at the speaker mean*, i.e. neutral. That is D-M1's principle biting a
second time: a manufactured `0.0` is indistinguishable from a measured one, which is how
1,094 clips shipped mislabelled in an earlier pass.

`derive_vat_corpus` already reports thin speakers (`!! N speaker(s) with <10 clips … whose
z is fixed by arithmetic rather than measured`). On LibriTTS that warning covers 80 of
31,445 clips (0.25%). On Emilia it would cover **46.6%**.

**So #1 needs a scale decision before it is a merge.** The mined T/V/A are already
LibriTTS-anchored — `mine_emilia_keeps` says so explicitly ("global mean/std from the v2
corpus measures + scores, so T_full/V_combo mean the same thing they meant in the probe") —
but they are on the RAW anchored scale, while corpus labels are clamped at 2σ and halved
into [−1, 1] (v4 measures V/A/T sd 0.42–0.48). The options, none free:
1. **Use the anchored values, mapped into [−1, 1]** — preserves the tails, needs the
   anchor re-derived against v4 rather than v2, and bypasses per-speaker z for this corpus.
2. **Per-speaker z with a minimum-clips floor** — honest but expensive: ≥10 clips/speaker
   keeps 7,015 of 13,141 (+23% instead of +43%), and still re-centres the tails it keeps.
3. **Merge as-is** — cheapest, and it answers a question nobody asked.

**Owner took option 1, 2026-08-08**, and it is implemented in
`scripts/anchor_emilia_labels.py`: every channel z-scored against the **v4 corpus's global
distribution of the same underlying measure** — same components, same signs, same second
normalisation, same `clamp2` — with `per_spk_z` swapped for the global anchor at each step.
The three EIV heads Emilia lacks all carry weight 0.0, so their absence changes nothing.

It does what it was chosen for. **All 13,141 keeps label, and 0 come out all-zero** against
756 under per-speaker z:

| | Emilia mean | sd | \|v\| > 0.5 | at the ±1 rail | LibriTTS rail |
|---|---|---|---|---|---|
| V | +0.305 | 0.597 | 53.0% | 28.4% | 7.04% |
| A | +0.014 | 0.598 | 48.7% | 11.1% | 4.66% |
| T | **+0.749** | 0.359 | 76.3% | **54.0%** | 4.69% |

⚠ **The tails survive, and T now SATURATES — 54% of Emilia clips sit pinned at |T| = 1
against LibriTTS's 4.69%.** That is not a bug in the anchoring; it is the mining criteria
being honest. `T_full > p90` was one of the four keep rules, the corpus's representable
range stops at 2σ, and clips selected for exceeding it land on the rail. But at 30.1% of a
merged 43,626-clip corpus it means roughly a sixth of all training rows carry "maximum
tension" with no gradation inside it, and the model has an easy shortcut available:
Emilia-like acoustics ⇒ T = 1. Three ways out, and this is the next decision, not a
blocker:
1. **Accept it** — honest labels, and the run measures whether the shortcut actually hurts.
2. **Re-balance the merge** — take fewer of the most extreme keeps, trading volume (the
   thing Phase 1 is testing) for gradation.
3. **Widen the clamp for the merged corpus** — the most tempting and the most expensive:
   ±1 is documented in the control contract as *the edge of the TRAINED range*, so
   re-scaling changes what every existing checkpoint, filelist and exported `config.json`
   means. That is a contract change and an owner call (ARCHITECTURE §1), not a constant.

**Owner took option 1 (accept) on 2026-08-08, and the prediction is recorded HERE, before
the run, so it is a test rather than a story told afterwards.** The whole value of
accepting is that the holdout gets to answer; that only works if the expected failure
signature is written down first.

> **PREDICTION, 2026-08-08, Emilia merge with saturated T.** If the shortcut is real, the
> merged run should show: (a) T **worse** on the never-trained holdout than the
> LibriTTS-only baseline, while V and A are unchanged or better — a channel-specific
> regression, not a general one; (b) T's standing perceptual test moving from near-pass
> toward fail; and (c) renders at T = +1 sounding like *Emilia-domain audio* (podcast/
> YouTube timbre, room, mic character) rather than like tension, which is the diagnostic
> an ear can settle and a loss cannot. If instead T holds or improves while V/A move, the
> shortcut did not form and saturation is a non-issue at this mix.
>
> **If (a)+(c) both land, the answer is C-soft** — `tanh(z/2)` for the whole corpus, which
> maps the saturated span 2.00–5.76σ to 0.762–0.994 instead of collapsing it to 1.0 — and
> it will then be justified by evidence rather than by anticipation. It is a contract
> change and a full relabel either way, so it wants to ride a re-derivation, not force one.

**READ-OUT, 2026-08-09 (PR-H3). Leg (a) did NOT land: there is no T-specific regression.**

Recomputed from artifacts already on disk — `v5_ckpt_sweep` holds per-clip losses for
`vat5_init` and `ep019` under identical constants, and the holdout filelist holds the
labels — so this cost no GPU and could have been read at any point since the run landed.
Reproduce with:

    .venv/bin/python scripts/stratify_holdout_sweep.py \
        --sweep /data/model-training/sonora/holdout_eval/v5_ckpt_sweep.csv \
        --filelist data/libritts_r_holdout_devclean/holdout_8w.txt \
        --baseline vat5_init --pick vat5_ep019

Per clip, `Δ = loss(ep019) − loss(init)`; each channel split at the top and bottom quintile
of |label|. **Positive "gap" = that channel's extremes improved LESS than its centre**,
which is the regression shape leg (a) describes. 95% CI, 2,000 bootstrap resamples, seed 42
— the same adjudication contrast 0a used.

| channel | extreme | central | gap | 95% CI | verdict |
|---|---|---|---|---|---|
| **T (tension)** | −0.0616 | −0.0610 | **−0.0006** | [−0.0065, +0.0055] | **crosses zero — no effect** |
| V (valence) | −0.0655 | −0.0498 | −0.0157 | [−0.0212, −0.0098] | excludes zero — extremes improved MORE |
| A (energy) | −0.0546 | −0.0679 | **+0.0133** | [+0.0068, +0.0195] | excludes zero — **extremes lagged** |

Overall Δtotal −0.0606, improved on 82.1% of 5,463 clips.

**The prediction named T, and T is the one channel that shows nothing.** Its CI straddles
zero, which is an ABSENT effect rather than a small one. By the pre-registration's own
terms — *"if instead T holds or improves while V/A move, the shortcut did not form and
saturation is a non-issue at this mix"* — **the shortcut did not form.** T's 53.6%
saturation was accepted on a bet, and the bet paid: the mining criteria arriving at the
label did not become a shortcut the model could take.

**C-soft is therefore NOT triggered.** It needs (a)+(c) together and (a) is out. Do not
re-open `tanh(z/2)` on this evidence; it remains available if a later rung produces the
signature.

⚠ **Two things this does not settle, and they are the honest residue.**

1. **Legs (b) and (c) are still owed the ear** — the standing perceptual test, and whether
   T = +1 renders sound like *Emilia-domain audio* rather than like tension. A
   teacher-forced loss cannot see either. Leg (a) is the cheap leg and it is now closed;
   the ear legs are a Vocalizer session on `ep019`. Since (a) is out, they are no longer
   urgent — they cannot trigger C-soft alone — but a "T sounds like a podcast" verdict
   would be a finding in its own right.
2. **A NEW signature appeared that nobody predicted: `A`'s extremes lagged its centre**
   (+0.0133, CI excludes zero). Small, and the direction is the one leg (a) described —
   for the wrong channel. Recorded here rather than acted on: n is large but this is one
   contrast on one rung, the effect is ~20% of the overall gain, and the obvious
   explanation is benign (A is loudness-derived, and the Emilia half arrives at a
   different loudness distribution than LibriTTS-R). **Pre-registered follow-up: if v6's
   sweep shows A lagging again, it is a real channel effect and wants the loudness
   normalisation checked before anything else.** If it does not reappear, this was one
   rung's noise and should be dropped rather than carried.

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

### Closed 2026-08-06, before the phase started

`train_diff_loss` 0.643 vs `val_diff_loss` 2.085 — the **3.2× gap** — was E-M5. The
diffusion loss summed its residual unmasked while the decoder masks its output, so every
batch paid a floor proportional to its padding; train batches are length-bucketed and val
batches are not, which is the whole of the "gap". `out_size` and clip-length distribution
had already been ruled out and were right to be. `compute_loss` masks now, gradients
unchanged, and `tests/test_training_seams.py` holds the curve honest. **Cost:** a second
scale break in logged loss (after 2026-08-01's bucketing change) — nothing before this commit compares to anything after it,
which is one more reason Phase 0a's holdout scoring is the real measurement.

---

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

Design, gates and known risks are already written: **[model-decisions.md § Decoder v2](model-decisions.md)**
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
| ~~after **1.1**~~ **PASSED 2026-08-09** | does **+36%** move the clean holdout? | **yes** — `ep019` − `vat5_init` = **−0.0606**, improved on 82.1% of 5,463 clips, an order of magnitude past the −0.0111 that cleared gate 0a. Data-limited, not capacity-limited: **the 10× proceeds.** (The question read "+43%" until 2026-08-09; +43% was the *planned* merge and **+36%** is what shipped, after 1,676 digit rows and 468 WER rows dropped out. A gate quoting a number the corpus never had cannot be checked against it) |
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
  (`scripts/stratify_holdout_sweep.py` reports both). **Init-relative is not a detail** —
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

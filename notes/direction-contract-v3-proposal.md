# Direction contract v3 — a proposal, not a decision

**Status: PROPOSAL, 2026-08-10.** Nothing here is ratified and nothing here is built.
It exists so the direction surface is decided *before* rung 2 trains and long before
Phase 1S mass-produces clips against it — the owner's sequencing, and the right one.
The governing rule from [`director_skills/sonora.md`](../scripts/synthesis/director_skills/sonora.md)
is unchanged and this proposal is subordinate to it: **if a field cannot be verified
after the render, it does not belong in the contract.**

Contract v2 (shipped): `vat[3]` = valence · energy · tension, plus a five-wide
one-hot delivery block. `vat_dim` 8.

---

## 0. What Documentary cost, and the rule it buys

**154 rated clips** carried a lane that no engine renders distinctly — 82 of them in
the v6 append set. Every one of them is synthetic, so this is a property of the
**brief**, not of any audio: the direction was authored, rendered, audited and
labelled before anyone asked whether it named a distinct render.

⚠ **Corrected 2026-08-11.** This paragraph read "82 append rows … they split
**bimodally** toward Neutral and Newscaster with an empty middle — 78/22 by mixture
against the two reference lanes". **It did not split.** Reassigned on the ear, **all
154 went to `Neutral`**: against `ratings.csv.bak-20260810-documentary-retire`,
Neutral goes 524 → 678 (+154) and **Newscaster is 87 before and after**. v6's
derivation report says the same from the other side — Newscaster is 76 on both sides
of the retirement while Neutral goes 251 → 328. The 78/22 was an acoustic *mixture*
reading against two centroids, which says where the audio sits, not where the labels
went; it never supported the Newscaster half and is not carried forward. A rationale
for a vocabulary change that nobody differenced against the file it describes is the
argument for the bar below, not a counterexample to it.

> **STANDING BAR (proposed): a direction that no engine renders distinctly is not a
> direction.** Test it before it enters a campaign, not after 154 clips exist.

At rung-2 scale that error cost 154 rated clips (82 of them in the append set) and one
afternoon. At Phase 1S scale — mass synthesis against a direction vocabulary — the
same error is thousands of clips, and they will all be *self-consistently*
mislabelled, which is the kind of error a holdout cannot see.

---

## 1. The 8 emotion heads: SWAP for V, do not ADD to it

The parked plan entry reads "categorical emotion block (3+8)". **Measured, 3+8 is
the wrong shape**, and the measurement is cheap enough that it should have been run
before the entry was written.

Every clip in the corpus already carries eight family heads — `Amusement`, `Elation`,
`Contentment`, `Hope_Enthusiasm_Optimism`, `Sadness`, `Fear`, `Bitterness`, `Shame` —
scored by the same EIV pass that produces V. There is **no new labelling work**: the
v4 lineage (30,351) and the v6 append set both have them.

Regressing the shipped channels on those eight, over the append set:

| channel | R² on the 8 family heads |
|---|---|
| **V** | **0.891** |
| T | 0.156 |
| A | 0.027 |

**V is 89% a linear function of the emotion block.** A `3+8` vector would spend a
channel on a near-duplicate of the other eight, and the corpus's own independence
gate — corr(T,A) = −0.059, corr(T,V) = −0.066, corr(V,A) = +0.027, all PASS — is
exactly the test that should refuse it.

Worse, the projection is **lossy in a biased direction**. `valence_combo_v1` weights
`Bitterness` at −0.221 and `Contentment` at **+0.018** — and Contentment has the
*widest* spread of any head in the append set (45.4% of clips beyond |0.5|). The one
number we keep is the one that throws away the head with the most signal.

**Proposed affect block: `A · T · [8 emotion heads]` = 10 dims.** V is not lost; it
is recoverable as the same weighted sum, Director-side, exactly as `register`
compiles away today.

⚠ **Two honest costs.** (1) The eight heads carry ~6 effective dimensions (90% of
variance in 6 of 8 eigenvalues; max pairwise |corr| 0.786), so 8 is mildly
over-parameterised — kept anyway because **named heads are directable and principal
components are not**. Gemma can emit `Bitterness 0.7`; it cannot emit `PC3 0.7`, and
the Director is an LLM writing direction. (2) These are **EIV classifier outputs, not
ear verdicts** — training on them teaches Sonora to reproduce EIV's opinion of
emotion. That is the same bargain as teacher distillation, defensible but it must be
a *stated* choice. V already carries this property; the block does not add it.

---

## 2. Make delivery MEASURABLE — the highest-leverage change here

Delivery is the scarcest signal in the corpus and it **cannot scale**, because it is
an ear verdict. v5 carries 48 labelled rows in 42,442. Even after v6 it is 822 in
~43,000 — under 2%, and every new row costs owner ear time.

Every other channel is derived from the audio by an instrument. Delivery is not, and
that single difference is why it is rare, why it does not survive a re-cut, and why
Documentary was able to hide in it for months.

**Proposal: build a delivery instrument** and let the ear verdicts become its
*training set* rather than the corpus's only supply.

This is not speculative. The nearest-centroid probe run against Documentary today —
five non-temporal features (V, T, alpha_db, cpp, h1h2), no pace, no phrase length,
no final lengthening — already reaches **~75% self-consistency** on Neutral and
Newscaster. That is a floor, built from measures we compute anyway. Addressee
geometry has obvious acoustic correlates we are not yet extracting:

- **projection** — alpha ratio (have it), plus long-term spectral tilt
- **pace and its variance** — phones/sec from the existing alignment
- **phrase length and pause structure** — from the same alignment
- **final lengthening** — the strongest known correlate of read-vs-spoken

If a delivery instrument reaches useful agreement with the 822 ear verdicts, delivery
becomes available on **every clip in the corpus** — LibriTTS-R, Emilia, and every
future synthetic batch — instead of on 2% of it. It also becomes a **post-render
verifier**, which is what the governing rule demands and what would have caught
Documentary on day one.

⚠ The instrument must be validated against held-out ear verdicts, never against
itself, and it must never overwrite an ear verdict — the ear is ground truth, the
instrument is coverage.

---

## 3. Dynamic direction — the conditioning path is ALREADY time-varying

**Span-FiLM is not the architecture project this plan files it as.** Checked
2026-08-10, and the finding reframes the whole ambition.

`matcha/models/components/film.py` — `VATTrunk` takes **`[B, vat_dim, T]`** and
*explicitly refuses* a per-utterance vector:

> `VATTrunk expects [B, vat_dim, T]; got …. Per-utterance (B, vat_dim) tensors must
> be expanded by the caller.`

`flow_matching.py:45` documents the decoder's input the same way — *"frame-level VAT
conditioning (batch_size, vat_dim, mel_timesteps)"*. **The model already consumes a
time-varying conditioning sequence; we broadcast a constant along T.** Feeding it
something that varies needs **zero model changes**, the zero-init heads keep a warm
start exactly equal to the unconditioned checkpoint at step 0, and the export path
was chosen to survive it (pointwise Conv1d + mul/add, "the op families already proven
through the litert-torch fixed-shape export path").

So this is a **labels-and-compiler project, not a modelling one.**

⚠ **Correction to an earlier framing in this file's first draft.** "Mean + slope" was
proposed as a cheap alternative that would double the affect block, 10 → 20. That is
wrong. A slope is a **linear ramp along T in the channels that already exist** — no
new width, no new parameters. Mean+slope is not an alternative to span conditioning;
it is its smallest case, and it is reachable today without touching the model.

### What is actually missing

| piece | state |
|---|---|
| frame-level conditioning path | ✅ **built** |
| frame-level **A** (LUFS) | ✅ computed per frame, then averaged — un-averaging deletes a line |
| frame-level **T** (alpha_db / cpp / h1h2) | ✅ same; all three are frame-wise before the mean |
| frame-level **emotion** | ❌ EIV is utterance-level. A real gap. |
| forced alignment (token ↔ frame) | ❌ not on disk — `derive_markup_measures.py` names the cleared source (cdminix/libritts-r-aligned, CC-BY-4.0) |
| Director → frames compiler | ❌ does not exist |

### The decision that determines whether it works

At training a trajectory can be **measured** from the audio. At inference the Director
must **supply** one — and no LLM can author 500 frames of a 10-dim vector. Train on
rich measured trajectories, infer from sparse authored ones, and the two sides speak
different languages. That is the classic prosody-transfer failure and it is how this
kind of project usually dies.

The answer is already ratified: **SCM** ([markup-schema-brief.md](markup-schema-brief.md)),
closed vocabularies — `emphasis` 1-3, `pause_after`, `pace`, `pitch_move`, `nonverbal`.

> **Train on the COMPILED SCM, never on raw measured frames.** Derive SCM *from* audio
> for training; compile SCM *to* conditioning for inference. Both sides then speak one
> language, and the closed vocabulary stays checkable — which raw frame trajectories
> never are.

### Why the data argument runs the other way here

Span conditioning is **more** data-efficient than clip conditioning, not less.
Clip-level labels give ~43,000 supervision events in total; frame-level gives 43,000 ×
hundreds. The scarcity that cripples delivery (§2) does not apply, because these
labels come from the audio rather than from the ear.

It also **un-parks embodiment for free**: those 17 clips are blank BY RULE because
their delivery changes partway through. Under a time-varying channel they stop being
an unlabellable special case and become the most informative rows in the set.

### ⚠ The two real risks

- **Duration coupling.** Matcha predicts durations, so token-level conditioning
  reaches the decoder through *predicted* durations at inference and *ground-truth*
  ones at training. This mismatch is the genuine technical hazard and the spike below
  exists to answer it before anything is committed.
- **Export.** `convert_vat.py` already refuses a wider checkpoint and F-H2 is open
  because a mobile host told nothing about the channels will interpolate them. A
  time-varying channel makes that story harder, not easier.

### THE SPIKE — frame-level A, one channel, nothing else

The cheapest decisive experiment available, and it fits between rungs rather than
blocking one:

- **No new labels.** Per-frame LUFS already exists inside the measure pass.
- **No alignment needed.** Energy is frame-level, not token-level — the token↔frame
  substrate is only required once SCM spans enter.
- **No new width.** `vat_dim` stays 8; only the A row stops being constant.
- **It already has a verifier.** Energy conditioning checks out at **ρ ≈ 1.000**, so
  "did the model follow a *time-varying* target?" is answerable with **zero ear time**.

**Read it as:** if frame-level energy is not followed, nothing more ambitious will be,
and we learn that from one run on data we already hold. If it is followed, the
duration-coupling question is answered empirically and everything after it is
label-building rather than architecture.

⚠ Do not let the spike quietly become the feature. It conditions on a *measured*
trajectory, which is the train/inference mismatch above; it is a test of the model's
capacity to follow, **not** a shippable direction surface. The SCM compiler is what
makes it shippable. Mean+slope is a step on the path, not a replacement for it.

---

## 3b. A has THREE reference frames in one corpus — the contract question v6 forces

Filed as **issue #14**, deferred out of PR #10 by the owner ("change neither and let the
diagnostics decide"), and recorded here because it is a property of the *lineage*, not of
any one script.

`data/libritts_r_emilia_expressive_vat_v6` z-scores the same channel against three
different things:

| rows | A is z-scored against |
|---|---|
| LibriTTS-R | the **speaker's** own distribution (`per_spk_z`) |
| Emilia | the **global anchor** |
| expressive append | the mean of the **rendering campaign** the clip was in |

At inference the Director supplies one absolute A, and the trunk has been trained on three
references for it. The first two predate this work and were an accepted trade (§ v5's data
config states the semantic cost). **The third is new, and it is different in kind**: a
campaign is a batch identifier, so it is the first reference frame with *no acoustic
meaning at all*. A clip's A now depends on which rendering run happened to contain it.

⚠ **The defence is real but narrow.** Per-campaign centring was not a preference — the
uncorrected label pinned **A at −1 on 94.4%** of the append set, and per-*bank* centring
still left the 103 librivox rows at −0.835, because the bank holds three loudness targets.
Campaign was the grouping that removed the artefact at every level it occurred. It is the
best available proxy for "recording chain", and it is a proxy, which is exactly the
complaint.

**This is an argument FOR §1's direction, not against it.** A ten-dim affect block whose
channels are named, instrument-measured and independently verifiable is what makes "which
frame is A in?" answerable per channel instead of per corpus generation. The single
squashed A is what allows three frames to hide inside one number.

### 2026-08-11 — the table has been read. The centring is confirmed; the frame question is NOT

⚠ **This section was headed "RESOLVED" until the numbers were re-derived on 2026-08-11.** Two
different questions were being answered as one. *Does per-campaign centring eat real lane
structure?* — **yes, read and confirmed below.** *Do the three frames disagree about what
`A = 0` means?* — **still open**, and the evidence once cited as falsifying it measures a
different quantity (§ "What A *means* at inference"). v6 still ships as built; the reason is
cost, not resolution.

The `bb76085` diagnostic ran when v6 was built, but its output is stdout-only and was lost
when an OOM crash-loop killed the terminal. It was reconstructed read-only from
`/data/model-training/sonora/expressive_registers_measures/labels_v6.jsonl`, measured on LUFS
rather than on A so `clamp2` pinning is not credited to the centring.

📐 **Every number in this section regenerates from
`scripts/derive_a_channel_stats.py`** (read-only; `.venv/bin/python
scripts/derive_a_channel_stats.py`). Writing the numbers into a note is not the same as
committing the derivation, and the first version of this section did only the former — so
the next challenge to any of them would have cost what the first recovery cost. The script
is the method, written down; it is not a gate and asserts nothing, because its inputs live
under `/data` and a clean checkout does not have them.

**The concern was real and is confirmed.** Per-campaign centring removes **94.3%** of the
between-lane loudness structure:

| | value |
|---|---|
| sd of lane-mean LUFS, before centring | **1.4065 dB** |
| sd of lane-mean LUFS, after centring | **0.0807 dB** |
| between-lane structure surviving | **5.7%** |

The mechanism is that **campaign is a near-perfect proxy for lane**: 13 of the 20 campaigns
with n ≥ 10 are ≥90% one delivery lane, and **one** lane/campaign pair is one-to-one in both
directions — `newscaster-v1` is 100% of Newscaster *and* Newscaster is 100% of
`newscaster-v1`, 76 rows either way. Centring per campaign therefore subtracts that lane's
own mean by construction. Newscaster's A is consequently a **constant**: mean −0.0000, sd
**0.0094**, range [−0.040, +0.052]. Two causes worth keeping apart — the *between*-lane
offset was removed by the centring, but the *within*-lane variance was never there, because
`newscaster-v1`'s native LUFS sd is 0.035 dB. The whole campaign was rendered to one
loudness target, and A is loudness-derived.

⚠ **Only Newscaster, and this note said "two lanes" until 2026-08-11.** Speech is the near
miss and the direction of the miss matters: `book-librivox-speech-v1` is 100% *Speech*, but
Speech is **not** 100% *that campaign* — it draws 61 of its 69 rows from it (**88.4%**), plus
`stress-v2` 7 and `delivery-v1-narration` 1. A proxy claim has two directions and only one of
them was checked. The 94.3% figure above is unaffected — it is measured on lane means, not on
this crosstab — but the mechanism is one clean case and a set of strong tendencies, not two
clean cases.

**Two refinements to the table above, and they point in opposite directions.**

1. **Four centres, not three — the table understates the fragmentation.**
   `MIN_CAMPAIGN_N = 10` sends 25 rows across 8 campaigns to the bank-wide offset instead of
   their own, so `labels_v6.jsonl` carries **21 distinct offsets** (20 campaign offsets + 1
   bank offset over 28 campaigns). More frames than three are in play. The table is
   conservative here.

2. **The campaign supplies only the *centre* — and that makes the divergence smaller, not
   larger.** The global anchor still supplies the **scale**, deliberately un-rescaled. §3b's
   table says the append rows are "z-scored against the mean of the rendering campaign",
   which reads as the campaign supplying the whole reference; it does not. The append rows
   and the Emilia rows therefore **share a frame component**, so the two frames diverge
   *less* than the table implies. This is a **clarification, not a widening**, and the
   earlier claim that the table was "conservative on both counts" was wrong on this one.

What survives is narrower and stranger than "more frames": the append rows sit in a frame
that is **not one of the three listed at all** — a centre from the campaign, a scale from the
anchor. The table undercounts the *mixing* while overstating the *distance*.

### What A *means* at inference

**A is within-source relative loudness, in all three frames.** They differ only in what
"source" denotes — the speaker, the corpus, or the recording chain proxied by the campaign.
Every frame answers the same question, *"louder or quieter than this clip's own reference
population?"*, and none of them ever meant absolute dB. So the Director supplying one
absolute A is not incoherent: it requests a **relative displacement**, which the trunk applies
against whatever reference the speaker embedding and delivery one-hot place the clip in.

**Part of that is measured. The load-bearing part is not, and this section claimed otherwise
until 2026-08-11.** The probe establishes **dial gain — a slope — and nothing else.**

#### Why an ep010 probe is quoted for ep008's dial

`ep008` is the selected checkpoint (`logs/train/vat6_finetune/SELECTED.md`, owner
2026-08-11); the probe was rendered on **`ep010`**, two epochs later, before the selection was
made. The gap is not nothing and is stated rather than smoothed over: on the holdout the two
differ by 0.0098 on `total` and by 0.0012 on `diff`, and ep010 is the run's **best** `diff`
checkpoint while ep008 is its best `total`. Neither term is loudness. Nothing in the run
touched the A pathway between the two — `loss/val_epoch` was a flat basin from epoch 0 and
both curves had flattened by ep6 — so the dial gain measured at ep010 is taken to describe
ep008's, and that is an **inference from run flatness, not a measurement of ep008**. Re-running
the probe on ep008 is cheap (90 renders, no training) and is the honest way to close it.

#### What the probe measured: SLOPE

90 renders, 2 texts × 5 lanes × A ∈ {−1, 0, +1} × 3 repeats, true LUFS. Moving A from −1 to
+1, output loudness moves:

| lane | Δ LUFS | 95% CI |
|---|---|---|
| Dialogue | +4.96 | [+3.99, +5.94] |
| **Newscaster** | **+4.68** | **[+3.71, +5.65]** |
| Neutral | +4.41 | [+3.44, +5.39] |
| Speech | +5.58 | [+4.61, +6.55] |
| `unknown` | +4.40 | [+3.43, +5.37] |

**The lane whose training A carried no variance still obeys the dial**, and its interval
excludes zero comfortably. A's meaning is carried by the ~42,000 v5 rows that do vary; the 832
append rows do not have to re-teach it. That much stands.

⚠ **The noise floor was misreported.** This section quoted "pooled within-cell noise floor
**0.554 dB**". That number is **not pooled** — it is the *mean of the 30 per-cell population
sds* (ddof = 0, n = 3), which is biased low twice. The pooled within-cell sd over the same 30
(text, lane, A) cells is **0.7616 dB** (ddof = 1, df = 60). Comparisons *between* cell means
need the (lane, A) pooling instead — 15 cells of 6, **0.8457 dB**, df = 75 — giving
**se = 0.4883 dB** for a difference of two 6-render means. Every CI above uses that se.

The correction does **not** rescue the "every lane obeys the dial equally" reading, and it
does not condemn it either. The 1.18 dB spread between the fastest lane (Speech +5.58) and
the slowest (`unknown` +4.40) is a **maximum over ten lane pairs**, not one chosen pair, so
its null distribution is a studentised range and not a `t`: **q = 1.18 / 0.4883 = 2.42**
(k = 5, df = 75), **p = 0.434**. Not distinguishable from noise at this n, and not excluded
either.

⚠ **This line read "1.71 se, p ≈ 0.09" until 2026-08-11, and that number was ~5× too small
in the direction that matters.** It divided by the se of a *difference of two slopes*
(0.6905 dB) and referred it to a two-sided `t(75)` — the null of one **pre-specified** pair.
The spread is a max-minus-min, so that reference is far too narrow and the `p` overstated
evidence *against* equal gain, which is exactly the reading the paragraph goes on to refuse.
`derive_a_channel_stats.py` § 7 now computes the range statistic, so this sentence
regenerates; the hedge below was right the whole time and the number is what disagreed
with it. (Comparing that 1.18 against a *per-render* sd, as an
earlier reading of this table did, is the wrong denominator: it asks whether the spread
exceeds single-render noise, when the quantities being differenced are six-render means.) The
supportable claim is **"nonzero gain in every lane"**. "Equal gain" is not established.

#### What the probe did NOT measure: INTERCEPT — and the data leans the other way

The three-frames concern is about **intercept**: whether `A = 0` denotes the same absolute
loudness in two lanes trained under different reference frames. A slope measured *inside* a
lane cannot answer that. The `A = 0` cells the probe already holds say this.

**And the lane axis happens to be close to the frame axis, which is what makes the comparison
worth reading at all.** The named lanes live almost entirely in the 832-row append set — the
campaign-centred frame. `unknown` is LibriTTS-R and Emilia — per-speaker z and the global
anchor. So "named lane vs `unknown` at `A = 0`" is, to a first approximation, "the third frame
vs the first two". It is also precisely why the confound below bites so hard: the same split
separates the frames *and* separates expressive-register renders from audiobook narration.

| lane at A = 0 | LUFS | vs `unknown` | in se |
|---|---|---|---|
| `unknown` | −28.51 | — | — |
| Speech | −30.37 | **−1.86** | 3.8 |
| Newscaster | −30.99 | **−2.49** | 5.1 |
| Neutral | −31.55 | **−3.04** | 6.2 |
| Dialogue | −32.08 | **−3.57** | 7.3 |

At `A = 0` every named lane renders **1.86 to 3.57 dB quieter than `unknown`** — 3.8 to 7.3
se, **all four the same direction** — and the named lanes differ from **each other** by
**1.71 dB**. That is the shape of the feared intercept displacement, and it **leans against
the previous conclusion rather than for it**.

⚠ **It is not proof either, because it is confounded.** A named lane rendering quieter at
`A = 0` is exactly what a *working* delivery block should also produce: Newscaster and
narration genuinely are quieter deliveries than an unconditioned average. **That probe** has
no term for the second reading, so it cannot separate them.

## ✅ RUN 2026-08-12 — the confound is resolved, and it does NOT explain the displacement

The outstanding test was: compare rendered `A = 0` loudness per lane against **that lane's
training-set mean**. It is now `scripts/derive_a_channel_stats.py` **§ 9**, joining
`labels_v6.jsonl` (corpus rows with |A| ≤ 0.25) to the same 90 renders. Both profiles are
centred on their own named-lane mean, so the checkpoint's **−7.376 dB** global offset from
the corpus cancels and only the between-lane *shape* is compared.

| lane | n | train @ A≈0 | render @ A=0 | train dev | render dev | discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| Dialogue | 312 | −22.920 | −32.081 | **+0.952** | **−0.833** | −1.785 (4.4 se) |
| Neutral | 288 | −23.162 | −31.549 | +0.710 | −0.300 | −1.011 (2.4 se) |
| Newscaster | 76 | −23.009 | −30.994 | +0.863 | +0.255 | −0.608 (1.5 se) |
| Speech | 48 | −26.399 | −30.370 | **−2.526** | **+0.878** | **+3.404 (9.0 se)** |

**The rendered profile is anti-correlated with the training profile: Spearman ρ = −0.80**,
and the order is very nearly inverted — Speech is the corpus's quietest lane by 3.4 dB and
renders as the *loudest* of the four. Training spread 3.48 dB, rendered spread 1.71 dB.
⚠ **Insensitive to the band**: ρ = −0.80 at |A| ≤ 0.10, 0.25, 0.50, 1.00 and unbanded, with
Speech's discrepancy between **+3.16 and +3.55 dB** throughout.

**What that settles.** "The named lanes are genuinely quieter deliveries" predicts a
*positive* correlation — quiet lanes render quiet. It comes out negative, so **that reading
does not explain § 8's displacement** and the confound above is closed.

⚠ **What it does NOT settle, and must not be quoted as settling.** A negative correlation is
*consistent* with the three-frames mechanism — per-campaign centring removed the lane-specific
part of `A` (§ 1: 94.3%), and `A` is a globally shared FiLM channel that cannot carry a
per-lane intercept — but **"the model has not learned lane loudness at ep010" predicts the
same table.** Separating those needs a second checkpoint or a re-cut with target-clustered
offsets. **Limits: 4 lanes, 2 texts, 6 renders per cell, one checkpoint.**

The correct statement is now: **the dial works; `A = 0` does not mean the same thing across
lanes; and the reason is not that the lanes are quieter.**

⚠ **Retracted:** this section previously called the probe "**the falsification of the feared
consequence**", and the ship decision below rested on that. It is withdrawn. A slope
measurement cannot falsify an intercept claim, and the intercept data that does exist points
the wrong way for it.

### Which frame wins in v7: a declared `loudness_target`, not `campaign` and not the anchor

The obvious answer is the global anchor — the only frame whose reference population is a
fixed, re-derivable artifact (`libritts_anchor` over v4's 31,445 clips, LUFS mean −18.156 /
sd 1.862), where per-speaker z cannot survive a median of 3 clips per speaker and `campaign`
cannot be supplied at inference at all.

**The obvious answer is wrong on its own, because unification RENAMES the offset rather than
deleting it.** The expressive bank is loudnorm'd to −23 LUFS and the anchor is not, so
re-anchoring the append set reintroduces exactly the artefact the centring was introduced to
remove: A pinned at −1 on **94.4%** of those rows. Anchoring alone converts a batch-identifier
offset into a corpus-wide one; it does not make A a measurement.

**So v7 replaces `campaign` with a declared `loudness_target` / recording-chain id** — the
same arithmetic, keyed on something with acoustic meaning rather than on which batch a clip
happened to be in. This keeps the anchor as the *scale* while giving the *centre* a declared,
inference-supplyable meaning.

#### The chain set, derived rather than enumerated

`label_expressive_registers.py:44` says the bank holds "**at least THREE** loudness targets"
and lists −23.0, −20.4 and −26.3..−27.1. That hedge is deliberate and this section closed it
to "three known targets" until 2026-08-11. **The hedge was right and the data is blunter than
either reading**: measured on `lufs_native` over all 832 rows,

| candidate | rows within 0.05 dB | verdict |
|---|---|---|
| **−23.0 LUFS** | **625 / 832 (75.1%)** | a real declared target — loudnorm |
| −20.4 LUFS | **2 / 832 (0.2%)** | **not a target.** Two rows is a coincidence |
| −26.3..−27.1 | 28 inside the literal band | **not a target** — a *range*, and the campaigns behind it (`book-librivox-*`) are simply un-normalised: sd 0.8–1.6 dB, and 0–3.3% of their rows sit at their own median |

**147 of 832 rows (17.7%) fall outside all three** even when the third is generously widened
by ±0.5 dB — 177 (21.3%) on the literal band. Cutting the same question per campaign, using
the modal test (*does most of the campaign sit on one value?*) rather than dispersion, the
28 campaigns split cleanly with nothing between 33% and 73%:

- **16 campaigns / 682 rows (82.0%)** — normalised to **−23.0 LUFS**
- **10 campaigns / 148 rows (17.8%)** — **un-normalised**; their LUFS is a *measurement*
- 2 campaigns / 2 rows — n = 1, undecidable

(The 147 and the 148 are two different cuts that land one row apart, not a typo: the first is
row-level against the three named values, the second is campaign-level against each
campaign's own mode.)

**So the bank has ONE loudness target, not three** — and roughly a sixth of it has no target
at all. That changes the v7 design rather than merely restating it:

1. **`loudness_target` must be nullable, and "un-normalised" must be a first-class declared
   value**, not a nearest-target rounding. Forcing the 148 un-normalised rows onto −23.0 or
   onto a fictional −26.7 reintroduces exactly the defect the key exists to remove: a
   convention wearing the costume of a measurement.
2. **For the un-normalised rows the honest centre is the anchor itself**, because their LUFS
   *is* real performance loudness — which is the one slice where A can carry signal today.
3. The cardinality worry that motivated the check resolves in the safe direction: the set is
   **smaller** than three, not larger, so the key is cheap to declare. The expensive part was
   never the count — it was assuming every row has a target.

**What it costs.** A-only relabelling of **~42,442 v5 rows (41,138 train + 1,304 val)** plus
832 append rows — arithmetic on stored LUFS, not a re-measure. Re-rolling every A breaks the
rung-over-rung holdout comparison the merge discipline protects
(`merge_expressive_registers.py:12` keeps v5 rows byte-identical for exactly this reason).
**It therefore belongs at a version bump with its own before/after, never as a patch.**

⚠ **This line quoted `~41,138 v5 rows` until 2026-08-11 — that is v5's TRAIN split, not v5.**
It under-scoped the relabel by the **1,304 val rows** and paired a train-only figure with an
append count that is train+val, in one sentence. An A-only relabel re-rolls A on *every* row
that has one, so **the val split is in scope** — and the val split is precisely what carries
the rung-over-rung comparability this paragraph argues must be protected, so it is the worst
possible split to lose track of. `notes/training-sources.md:44` carries the standing warning
(**"ALWAYS SAY WHICH SPLIT"**) and `scripts/test_doc_claims.py` registers v5 TRAIN, VAL and
TOTAL as three separate facts for this reason; the phrasing `~41,138 v5 rows` matched none of
the gate's patterns, so it was a silent miss rather than a caught one.

The 104 rows with recoverable pre-loudnorm LUFS (`v1/audio/loudnorm.jsonl`) are the only
slice where real performance loudness could be restored; backfilling that sidecar bank-wide
is the one change that would make A on the append set a measurement rather than a convention.
Note this now has a second beneficiary: the **148 un-normalised rows** above already carry
real performance loudness and need no sidecar at all.

### v6 ships as built — a pre-commitment met and deliberately overridden

**The pre-commitment, verbatim, as it stood before this section replaced it** (`66f8709`,
§3b's closing paragraph):

> **Not resolved here, deliberately.** The `bb76085` diagnostic prints per-campaign mean LUFS
> with dominant lane and lane share, plus the per-lane sd of lane means before and after
> centring — measured on LUFS, not on A, so `clamp2` pinning is not credited to the centring.
> If that table shows the centring is eating real lane structure, re-cutting v6 with
> target-clustered offsets is its own commit with its own before/after. **Read the table
> before taking a position**; that is the same discipline the T-saturation prediction block
> gets.

**Its condition was met.** The table was read and the centring is eating real lane structure:
94.3% of the between-lane loudness structure is gone. The trigger was a plain conditional on
the table's contents, and the table's contents fired it.

**Its consequence was not carried out. Owner call, 2026-08-11: v6 ships as built.** The
reason is cost against evidence, not a re-reading of the trigger:

- re-cutting re-rolls every A, which **destroys the rung-2 holdout comparison** the merge
  discipline exists to protect — the ladder's one lever per rung;
- **`ep008` is already selected and trained** (`logs/train/vat6_finetune/SELECTED.md`), so
  the re-cut is a re-cut *and* a re-run;
- the dial demonstrably works at inference (nonzero gain in all five lanes), so there is no
  *measured* inference defect to buy with that cost.

⚠ **This override is weaker than the version first written here, and the weakening is the
point.** That version said the pre-commitment "was aimed at a consequence that has since been
measured and falsified". Two things were wrong with it. The trigger was written as a
condition on *lane structure*, not on a consequence — the re-description was doing contested
work. And nothing was falsified: the probe measured **slope**, the concern is **intercept**,
and the intercept data that exists (§ above) leans *against* the frames agreeing. The
override therefore rests on **cost and the absence of a demonstrated inference defect**, not
on the concern having been answered. **It has not been answered.**

**What the override buys and what it defers.** It buys rung 2's comparability. It defers the
intercept test, which is cheap (90 renders, no training) and belongs before v7's
`loudness_target` re-key is designed on top of an untested premise. Recorded here so the
declined pre-commitment is readable beside its trigger, rather than reachable only through
git history.

---

## 4. The gate that ties these together

Every proposal above is a channel. Proposed promotion rule — a channel enters the
conditioning vector only when all four hold:

1. **An instrument produces it** from the audio, for a stated fraction of the corpus.
2. **It is independent** of the channels already there (the existing corr gate).
3. **A dial test moves the render measurably** — the V/A/T dials passed this on
   2026-08-09; energy verifies at ρ ≈ 1.000.
4. **It has a post-render verifier**, so direction-following is scoreable with no ear
   time.

Rule 4 is the one that does not exist yet and matters most for Phase 1S. Mass
synthesis without an automatic direction-following score is mass production of
unverified labels.

⚠ **A channel whose measure is touched by the production pipeline is not safe.**
The A channel was pinned at −1 on 94.4% of the append set purely because the bank is
loudnorm'd to −23 LUFS; the label recorded our encoder's target as a property of the
voice. Any new channel must be checked against the pipeline that produced its audio,
not only against the corpus statistics.

---

## Sequencing

| | change | when | why then |
|---|---|---|---|
| now | four delivery lanes, Documentary retired | **done 2026-08-10** | `vat_dim` unchanged at 8; `ep019` warm-starts v6 cleanly |
| rung 2 | **no contract change** | — | v6 exists to test volume + delivery, one lever |
| after rung 2 | delivery instrument (§2) | needs no GPU | unlocks delivery on 43,000 rows instead of 822 |
| after rung 2 | **frame-level A spike (§3)** | **no width change, no new labels** | answers duration coupling for one run; has a verifier already |
| rung 3+ | affect block A·T·8 (§1) | `vat_dim` 8 → 14 | new warm start + seam pass; do it once |
| after the spike | SCM compiler + alignment (§3) | no width change | the substrate that makes span conditioning *shippable* rather than merely possible |
| gate | verifier rule (§4) | **before Phase 1S** | 1S without it mass-produces unverified labels |

Nothing here touches v6. That is deliberate: rung 2's whole purpose is one lever, and
a contract change mid-rung destroys the comparison the ladder is built on.

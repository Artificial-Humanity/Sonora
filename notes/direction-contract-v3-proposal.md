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

82 append rows carried a lane that no engine renders distinctly. They split
**bimodally** toward Neutral and Newscaster with an empty middle — 78/22 by mixture
against the two reference lanes — and every one of them is synthetic, so this is a
property of the **brief**, not of any audio. The direction was authored, rendered,
audited and labelled before anyone asked whether it named a distinct render.

> **STANDING BAR (proposed): a direction that no engine renders distinctly is not a
> direction.** Test it before it enters a campaign, not after 82 clips exist.

At rung-2 scale that error cost 82 rows and one afternoon. At Phase 1S scale — mass
synthesis against a direction vocabulary — the same error is thousands of clips, and
they will all be *self-consistently* mislabelled, which is the kind of error a
holdout cannot see.

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

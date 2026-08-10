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

## 3. Dynamic direction without span-FiLM: mean + slope

The owner wants direction that *moves*. Clip-level conditioning cannot express it,
and `embodiment` is parked for exactly this reason — those 17 clips are blank BY RULE
because their delivery **changes partway through**, and a clip-level channel has no
way to say so.

Span-FiLM is the eventual answer and it is a real architecture project. **There is a
much cheaper intermediate that gets most of the value:** condition on each affect
channel's **mean and slope** across the clip rather than its mean alone.

- Doubles the affect block (10 → 20) instead of building a span architecture.
- **Backwards compatible in the way this corpus already trusts**: a static clip has
  slope 0, which is the same "absence of a value, not a value" property that makes
  `unknown` ≡ the zero vector work for delivery. A corpus with no slopes reproduces
  v2 conditioning exactly.
- **Costs no new labels.** A (LUFS) and T (alpha/cpp/h1h2) are already windowable —
  they are computed per-frame and averaged. Only the EIV heads need a second pass,
  scored over clip halves.
- **It un-parks embodiment for free.** Those 17 clips are precisely the ones where
  slope ≠ 0. They stop being an unlabellable special case and become the most
  informative rows in the set.
- It is directly what Qwen's own demo asks for in prose — *"Initially commanding,
  shifting to narrative amusement"* — expressed as something measurable.

⚠ Slope over a whole clip is a crude trajectory: it cannot express "quiet, loud,
quiet". That is the honest limit, and it is the point at which span-FiLM stops being
deferrable. Mean+slope is a step on the path, not a replacement for it.

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
| rung 3+ | affect block A·T·8 (§1) | `vat_dim` 8 → 14 | new warm start + seam pass; do it once |
| later | mean+slope (§3) | 14 → 24 | after the affect block proves out |
| gate | verifier rule (§4) | **before Phase 1S** | 1S without it mass-produces unverified labels |

Nothing here touches v6. That is deliberate: rung 2's whole purpose is one lever, and
a contract change mid-rung destroys the comparison the ladder is built on.

# North Star — Project Sonora

> The map, not the implementation: the *why* and the standing decisions the build serves. When a
> branch, experiment, or detour raises "wait, what are we doing again?" — this is the document that
> answers it. Sibling: [Prosodia's north star](../../../Prosodia/notes/architecture-north-star.md),
> which describes a **consumer** of what this repo produces. Related:
> [high-ambition-index.md](high-ambition-index.md) · [model-decisions.md](model-decisions.md) ·
> [STATE.md](STATE.md)
>
> ⚠ **§1 IS THE OWNER'S AND WAS RATIFIED 2026-08-20.** Every other section is assembled from this
> repo's measured record and standing rulings. §1 could not be — it is the statement of intent that
> all the others are judged against — so an agent drafted it, marked it unratified, and the owner
> rewrote it. **Keep that order for any future edit to §1.** The failure this procedure exists to
> prevent is on the record: an agent once wrote "a deliberate owner ruling" about a decision nobody
> could later verify, and it read as settled fact for two days.

---

## 1. The Vision (the thing everything serves)

**A voice actor that takes direction.** A voice that not only sounds good — a voice that performs *the
reading you asked for*, and a different one when you ask differently.

Sonora produces the actor. It does not produce the reader, the app, or the book. What it ships is
a capability, and the capability is **directability**.

The acceptance test is the ear, not a metric:

> Hand the model one line and two directions. If the two renders differ in the way the direction
> named — and are recognisably the same voice — the actor works. If they differ in some other way,
> or not at all, it does not, whatever the loss says.

⚠ **RATIFIED BY THE OWNER 2026-08-20**, who rewrote it. An agent drafted this section and marked
it unratified; that warning is gone because it is no longer true, and leaving it would have made a
settled statement read as provisional. The supporting record is
[high-ambition-1](high-ambition-1-matcha-actor.md) ("the first actor model we actually ship and the
foundation the others build on"), `AGENTS.md`'s statement of purpose, and the direction contract.

---

## 2. What Is Actually Ours (and what is rented)

**Rented — and deliberately so.** Matcha-TTS is the acoustic backbone; HiFi-GAN is the vocoder;
LibriTTS-R is the base corpus. Replacing them is a decision with a record
([model-decisions.md §5](model-decisions.md)), not a default. The StyleTTS2-Lite escape hatch was
**retired 2026-07-29** in favour of a scaled flow-matching backbone.

**Ours, and the reason the repo exists:**

* **The direction contract** — V/A/T as continuous FiLM channels, plus delivery as a fourth, with a
  controlled register lexicon. This is the interface between "what a director asked for" and "what
  the model conditions on", and nothing upstream has it.
* **The expressive-registers corpus** — grown and refined continuously, certified keeps published
  **CC-BY-4.0**, versioned per campaign. A published artifact in its own right.
* **The measurement discipline.** The holdout instrument, the ear as final arbiter, and the standing
  refusal to rank on a single scalar.
* **The published actors** — promoted into the sibling `Sonora/huggingface` checkout. That is the
  deliverable, and it is consumable by anyone, not only by Prosodia.

⚠ **Prosodia depends on Sonora; Sonora does not depend on Prosodia.** Ambitions 3 and 4 live there
because they change what an app does with a model it already has. Sonora's value does not require
Prosodia to exist.

---

## 3. The One Organizing Principle

**Measure before believing, and check that the instrument ran.**

This is not a virtue statement; it is the repo's most expensive recurring lesson, and it is written
into the guards rather than the culture:

* the vocoder was assumed to be the bottleneck and **measured not to be** — the gap is the acoustic
  model's, so the levers are data, decoder and capacity;
* runs are **data-limited, not epoch-limited** — a v5 run took 100% of its gain in the first 10 of
  39 epochs;
* `loss/val_epoch` is **not** a generalisation measure here, because the val split is contaminated;
* four instruments disagreed on one checkpoint selection, and diff/mel loss may be *anti-correlated*
  with naturalness — so **never select on a single scalar**, and hand the owner the disagreement
  rather than averaging it away;
* a green check that never ran is worse than a red one. `AGENTS.md` §5b exists because a gate can
  stop enforcing **without going red**.

---

## 4. Load-Bearing Constraints

These are settled and should not be re-litigated without a record:

| constraint | why |
|---|---|
| **Apache-2.0, no patent track** | defensive posture; every new repo and crate ships Apache-2.0 unless the owner says otherwise |
| **Corpus bar = unrestricted open redistribution** | stricter than the old commercial test; NC licences do not clear it |
| **A licence rejection is permanent; a quality rejection is not** | revisit engines on quality, never on licence |
| **Minimum 4 s of speech per clip** | owner floor; the keep-rate cliff is measured exactly there |
| **A clip's text must be a complete utterance** | the score cannot detect this defect |
| **Unrequested singing is a defect** | singing is ambition 7, and wanting it does not make an unasked-for performance correct |
| **Every new capability ships with a Vocalizer dial** | the standing vetting surface |
| **Execute from the repo; `/data` holds data, not source** | with the deploy clone the single documented exception |

---

## 5. The Real Bottleneck

**Data, not architecture, and not the vocoder.**

Measured: the acoustic model carries the gap; Emilia's **+36%** moved the clean holdout by
**−0.0606**, which says the corpus is data-limited and the next rung proceeds
([quality-gap-plan.md](quality-gap-plan.md) §Phase 1, rung 1 — both figures are cited there rather
than restated here, because a number in two places drifts). The scale gap against the
teachers is not closeable by copying their sources — those are undisclosed — which is what makes
**teacher distillation** the strategy rather than a shortcut.

⚠ The synthetic-scale lane exists for **targeted remediation, not bulk**, and sequences behind the
cleared real audio.

---

## One-Breath Summary

Sonora trains and publishes a **directable voice actor** and the **expressive corpus** it learns
from. Everything else — the contracts, the gates, the holdout, the ear — exists so that a direction
given is a direction performed, and so that we can tell the difference between believing that and
having measured it.

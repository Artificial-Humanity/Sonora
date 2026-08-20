# North Star — Project Sonora

> The map, not the implementation: the *why*, the **arc it is aimed along**, and the standing
> decisions the build serves. When a branch, experiment, or detour raises "wait, what are we doing
> again?" — this is the document that answers it. Sibling: [Prosodia's north
> star](../../../Prosodia/notes/architecture-north-star.md), which describes a **consumer** of what
> this repo produces.
>
> ⚠ **THIS FILE OWNS NOTHING BUT §1.** Every target, rung and constraint below is held somewhere
> else, and the owning file is named at each one. **If this file and the owning file disagree, the
> owning file wins** and this one is the defect. That rule is what lets a north star carry real
> numbers instead of vague ambitions — the alternative, a second copy nobody compares, is the
> failure mode this repo has paid for repeatedly. Volatile figures (row counts, holdout deltas, run
> status) are deliberately **not** restated here at all; they are pointed at.
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

## 2. The Arc — the seven ambitions

The vision is one sentence; the **ambitions are what it decomposes into**, and they are numbered
because the numbers are referenced across two repositories. Map and full status:
[high-ambition-index.md](high-ambition-index.md) — ⚠ **it is the SSOT for standing; the column below
is orientation, not a second status board.**

| # | ambition | home | what it is, in one line |
|---|---|---|---|
| **1** | [Matcha-TTS directable production actor](high-ambition-1-matcha-actor.md) | Sonora | **the project** — the first actor we ship, and the foundation of every other number |
| **2** | [Dramatic Reader / full-cast audiobooks](high-ambition-2-dramatic-reader.md) | Sonora | many identities in one performance, stable across chapters — needs 1 first |
| 3 | [Child voices](../../../Prosodia/notes/high-ambition-3-child-voices.md) | Prosodia | casting range downward |
| 4 | [Multilingual G2P](../../../Prosodia/notes/high-ambition-4-multilingual-g2p.md) | Prosodia | after English production quality, not beside it |
| ~~5~~ | StyleTTS2-Lite re-platform | git history | **RETIRED 2026-07-29** — the quality-ceiling escape hatch is now a scaled flow-matching backbone ([model-decisions.md §5](model-decisions.md)) |
| 6 | [The Audience — conveyance-aware STT](high-ambition-6-audience-conveyance-stt.md) | Sonora | the **reverse lane**: perceive prosody rather than dictate it |
| 7 | [Singing](high-ambition-7-singing.md) | Sonora | down the road, behind the quality ladder |

Four things about this arc that are load-bearing, and are not obvious from the table:

* **1 and 2 are the same capability at two scales.** Goal 2 is not "add voices" — it is deliberately
  crossing the line single-narrator reading must never cross, because a large enough excursion in
  speaker space *is* a cast change. The contract's separation of identity (a vector) from emotion
  (V/A/T) is what makes that a threshold policy rather than a hope. ⚠ **And it is two capabilities,
  not one — see below.**
* **3 and 4 change the MODEL but live in Prosodia.** That is the one genuinely blurred spot in the
  split, and it is recorded rather than tidied — they accumulated their history there. If the series
  is ever reorganised, those two move.
* **6 is not a detour.** It is the same control contract read in the other direction: the typed
  payload works as an *annotation* format exactly as well as a *dictation* format. The symmetry is
  the origin story repeated — Sonora exists because Kokoro could not be **directed**; the Audience
  exists because STT cannot **perceive** direction.
* ⚠ **7 has a defect on the other side of the same behaviour.** *Unrequested* singing is a measured
  defect class **today** and stays one; requested singing is the ambition. VibeVoice and Dia were
  benched for **uncontrollability, not inability** — a capability you cannot withhold is not a
  capability — so progress on 7 is never evidence about them.

### Not a focus — cloning real people

⚠ **Cloning real people is not a focus of the owner's** (2026-08-20) — and the exact words matter,
because this is the kind of note that gets hardened into a ruling by whoever reads it next:
**not forbidden, not off the table, and not a goal.** Those are three separate statements and
**none of them may be upgraded into the others.**

It is written down because it is an *absence*, and an absence is the one thing a reader cannot check
by looking: someone scanning the seven ambitions for cloning and not finding it cannot tell
"ruled out" from "set aside" from "nobody has raised it". It is none of those. It is **unprioritised
attention**, which is the weakest and most reversible form — if it ever becomes a focus, that is a
change of interest, not a reversal of a decision, and it needs no repeal.

Two inferences to refuse:

* **This is not a policy.** There is no ban, and there is no consent or likeness position — writing
  one would be its own work, and nobody has done it. Do not cite this paragraph as if a rule existed.
* **This is not about the teacher engines.** Several of them clone, and one is evaluated precisely as
  a clone-multiplier. That is a **tool pointed at a corpus**, not a capability being built into the
  actor, and its standing is unaffected by anything here.

What the project does pursue instead is the positive form of the same sentence: casting is
**synthesised from our own measured norms** — not derived from a target person, and not inherited
from another model's habits.

### The distinction inside goal 2 — portrayal is not casting

⚠ **The owner's two-layer model (2026-07-22), and the smaller layer is the one that comes first.**
[casting-attribute-norms-brief.md § Identity vs. portrayal](casting-attribute-norms-brief.md) owns it.

Most books are single-narrator, and **the narrator's identity persists *through* cross-gender
portrayal**: a female narrator deepens her pitch and sounds bigger for a male character's quoted
lines, and the voice remains female. So every clip carries **two casting layers, never conflated**:

| layer | what it is | how it moves |
|---|---|---|
| **identity** | the narrator's own gender / age / accent | the 64-d speaker vector; **constant across a whole book** |
| **portrayal** | the character being *emulated within* that voice | a **delivery modification** — pitch, size, register — **not an identity change** |

* **The two are separated by magnitude, and that threshold does not exist yet.** Portrayal is a
  *bounded* excursion in speaker space; a cast change is an unbounded one. Goal 2 needs an explicit
  threshold policy, and writing it is work nobody has done.
* **One-voice-many-characters is what an audiobook narrator actually does**, so it is reachable
  before full cast is — and it is what makes goal 2 an *evolution* of goal 1 rather than a
  different model.
* ⚠ **A one-layer instrument reads portrayal as noise.** The valence audit flagged exactly this
  phenomenon ("acted character voices") as confusion, and it stops being noise only once the two
  layers are separate metadata. That is a corpus-labelling consequence, not a modelling one.
* **Narrated dialogue is embodiment in the wild** — narrator-within-narrator, with the shift
  boundary already marked by the quote marks — which is the same thing the span-FiLM deferral in §4
  is waiting on evidence for. The training material for this exists license-clean: the owner-pinned
  full-cast LibriVox readings and stage plays.

---

## 3. The Envelope — what the actor has to fit inside

**The target is not a parameter count; it is a phone.** The anchor is **mobile capability — iPhone
17+ and equivalent Android** — and legacy devices are explicitly not targeted. The numbers below
follow from that anchor rather than the other way round.
⚠ **[model-decisions.md](model-decisions.md) owns every figure in this section.**

| target | value | note |
|---|---|---|
| size ceiling | **150M**, a *loose maximum* | binds the DiT block-shape decision more than anything else |
| current trajectory | **40–50M** for a full-featured end-to-end graph | well under the ceiling |
| today's acoustic model | **22.7M parameters** | measured; [quality-gap-plan.md § Phase 1S](quality-gap-plan.md) states the consequence — a model sized for hundreds of hours being fed thousands |
| shipping artifact | **~20–40 MB**, real-time on iOS / Android / desktop | [high-ambition-1 § Objective](high-ambition-1-matcha-actor.md) |
| audio | **24 kHz mono native**, no resample | LibriTTS-R is native 24 kHz and so is the Rust `StageAudioSink` |
| mel contract (locked) | `n_fft 1024 · hop 256 · win 1024 · n_mels 80 · f_min 0 · f_max 12000` | **`n_feats` stays 80** — it preserves warm-start shapes, the engine mel contract and the proven LiteRT export lane |
| export window (fixed shape) | **256 phonemes / 512 mel frames ≈ 5.46 s** | hop 256 @ 24 kHz = 93.75 fps |

**Three tiers — mini / mid / heavy — are separate trainings of a shared recipe, not one model
grown.** No weights cross a tier, and that is the plan rather than a risk: corpus, label pipeline,
eval regime and the Director↔Actor contract are all deliberately **scale-free**, so only the
backbone is per-tier. The `-mini` suffix waits until a second tier actually exists; publish the
current lineage unsuffixed.

**Two standing insurances — do not weaken either:**

1. **Speaker is a 64-dim VECTOR, never an id.** The speaker table is a roster, not a representation,
   and a dead end at scale. The export graphs already take a vector; the lookup is host-side.
2. **V/A/T semantics and units are versioned contract items, owner-gated.** If the channels mean
   subtly different things per tier, tiers stop being interchangeable and every consumer forks.

**Where headroom gets spent, highest leverage first:**

1. ~~vocoder fidelity~~ — **struck 2026-08-06.** Copy-synthesis measured the 24 kHz HiFi-GAN as
   perceptually transparent, so a wider vocoder buys nothing audible.
2. **a real speaker encoder (ECAPA-class)** over the row lookup, if voice generalisation beyond the
   training roster matters.
3. **acoustic backbone width — LAST**, and with a caveat: decoder cost scales with ODE step count
   more than raw width, and a wider decoder on CPU can cost more real-time factor than it buys.

⚠ **The fixed-shape window makes Director-side chunking mandatory** — the chunk size *is* the export
ceiling. Quality is not what binds (swept to ~2 min continuous with flat WER and stable rate); **the
cross-chunk seam is the open problem**, and raising the ceiling is a second, larger graph
(~1024 frames ≈ 11 s) for long-form rather than a dynamic shape.

---

## 4. The Route — one ladder, one lever per rung

**Phase 0 (measurement repair) → Phase 1 (data, rungs 1–5) → Phase 1S (synthetic) → Phase 2 (the DiT
decoder spike).** [quality-gap-plan.md](quality-gap-plan.md) is the SSOT for sequencing and holds
the live status; [STATE.md](STATE.md) says where the front is. ⚠ **Neither is summarised here** — a
status in two places is a status that drifts, and this file would be the stale copy.

What is structural about the route, and does not move:

* **Every rung ADDS rows without re-rolling what came before.** That property is the only reason
  rung *n*'s holdout number is comparable to rung *n−1*'s, and a re-derivation destroys it silently.
* **Exhaust the cleared public real audio first** (owner). Real audio yields ~100% and costs no ear
  time; synthetic costs rendering, direction, pruning and listening. Every hour of real is cheaper
  than every hour of synthetic.
* **The gate is always the never-trained holdout**, never `loss/val_epoch`. One epoch on the holdout
  destroys it permanently and there is no second dev-clean.
* **The synthetic lane's real purpose is targeted remediation, not bulk** — filling corpus regions
  real audio does not reach.
* **Out-scaling the teachers is not on the route at all.** They trained on undisclosed corpora
  orders of magnitude past everything reachable to us, which is what makes **distillation the
  strategy rather than a shortcut**.

⚠ **This ladder is the route to a model worth casting — not the whole route to the product.** Every
rung ladders to *quality*; **nothing on it ladders to castability**, and goal 2 depends on casting
more than on mel loss.

**Parked, with an end-condition — the casting/blend layer.** It is half of goal 1, so its absence
from the ladder is a recorded decision rather than an oversight. It is parked because **no
instrument exists**: the clean holdout is teacher-forced loss on audiobook narration and cannot see
whether a directed cast matched its brief, and scheduling a rung before its instrument exists is
what the plan refuses everywhere else. Ending the park needs (1) the measurable gender/age/accent
norm set and (2) a casting eval that is not the holdout. ⚠ **Reconcile the vocabulary first** — two
are in circulation (*age/masculinity/strain* vs *gender/age/accent*) and the standing ruling that we
define our own measurable norms makes
[casting-attribute-norms-brief.md](casting-attribute-norms-brief.md)'s the live one.

**Deferred with a trigger, not forgotten — the conditioning chain past the contract.** Each of these
is decided far enough that the *decision* is recorded; none is scheduled, and the trigger is written
down so it becomes an answer rather than an omission ([todo.md § 3](todo.md)):

* **Embodiment → span-FiLM** — delivery that *changes partway through an utterance*, on the same
  zero-init path expanded through the duration alignment. Its plumbing prerequisite is already met,
  so it waits on **evidence** (the embodiment bank across the surviving engines), not on code.
  ⚠ Standing rule: embodiment clips stay **delivery-blank** and outside the lane percentages, and
  the encoding refuses any label outside the closed five — so the lane cannot arrive by accident.
* **A categorical emotion block (3 continuous + 8 categorical)** — genuinely **undecided**, and
  gated on Phase 1's valence read: valence is the one dimension acoustics alone cannot reach, it
  failed its standing test once, and the corpus growth is the experiment that separates a
  label limit from an architectural one. **If it is ever spiked it APPENDS as channels 8+** —
  reordering is the one edit that must never happen, because position is the wire format.
* **Multilingual** — plan only, and sequenced *after* English production quality rather than beside
  it. Neither teacher built multilingual first either.

---

## 5. What Is Actually Ours (and what is rented)

**Rented — and deliberately so.** Matcha-TTS is the acoustic backbone; HiFi-GAN is the vocoder;
LibriTTS-R is the base corpus. Replacing them is a decision with a record
([model-decisions.md §5](model-decisions.md)), not a default. Matcha was **reaffirmed 2026-07-29**
on the teacher campaign's central lesson: *an engine's usable range is set by its conditioning
interface, not its raw quality* — which argues for a base whose conditioning surface we define.

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

## 6. The Real Bottleneck

**Data, not architecture, and not the vocoder.**

Measured: the acoustic model carries the gap, and Emilia's volume increase moved the clean holdout
by an order of magnitude more than the threshold that cleared the previous gate — so the corpus is
**data-limited, not capacity-limited**, and the next rung proceeds ([quality-gap-plan.md](quality-gap-plan.md)
§ Phase 1, rung 1 — both figures are cited there rather than restated here, because a number in two
places drifts).

⚠ That verdict is **a measurement with an expiry, not a law.** The same instrument is what would say
"capacity-limited" at a later rung, and if it does, the Phase 2 decoder spike moves onto the
critical path rather than sitting after it. Read the instrument each rung; do not inherit the answer.

---

## 7. The One Organizing Principle

**Measure before believing, and check that the instrument ran.**

This is not a virtue statement; it is the repo's most expensive recurring lesson, and it is written
into the guards rather than the culture:

* the vocoder was assumed to be the bottleneck and **measured not to be** — the gap is the acoustic
  model's, so the levers are data, decoder and capacity;
* runs are **data-limited, not epoch-limited** — a v5 run took its entire gain in the first handful
  of epochs and every epoch after that was net worse ([STATE.md](STATE.md) has the counts);
* `loss/val_epoch` is **not** a generalisation measure here, because the val split is contaminated;
* four instruments disagreed on one checkpoint selection, and diff/mel loss may be *anti-correlated*
  with naturalness — so **never select on a single scalar**, and hand the owner the disagreement
  rather than averaging it away;
* a green check that never ran is worse than a red one. `AGENTS.md` §5b exists because a gate can
  stop enforcing **without going red**.

---

## 8. Load-Bearing Constraints

These are settled and should not be re-litigated without a record:

| constraint | why |
|---|---|
| **Apache-2.0, no patent track** | defensive posture; every new repo and crate ships Apache-2.0 unless the owner says otherwise |
| **Corpus bar = unrestricted open redistribution** | stricter than the old commercial test; NC licences do not clear it |
| **A licence rejection is permanent; a quality rejection is not** | revisit engines on quality, never on licence |
| **Repo licence ≠ weights licence** | check the weights and the training-data terms separately — the trap is real and named in the record |
| **Minimum 4 s of speech per clip** | owner floor; the keep-rate cliff is measured exactly there |
| **A clip's text must be a complete utterance** | the score cannot detect this defect |
| **Unrequested singing is a defect** | singing is ambition 7, and wanting it does not make an unasked-for performance correct |
| **Every new capability ships with a Vocalizer dial** | the standing vetting surface |
| **Execute from the repo; `/data` holds data, not source** | with the deploy clone the single documented exception |
| **Make the first success boring** | the ordering rule behind base choice, phase 0 and every spike gate |

---

## One-Breath Summary

Sonora trains and publishes a **directable voice actor** — small enough for a current phone, aimed
along a seven-goal arc whose first rung is the only one live — and the **expressive corpus** it
learns from. Everything else — the contracts, the gates, the holdout, the ear — exists so that a
direction given is a direction performed, and so that we can tell the difference between believing
that and having measured it.

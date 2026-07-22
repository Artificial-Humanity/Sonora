# High-Ambition 6 — 👂 The "Audience": Conveyance-Aware Speech-to-Text

> **Sequence:** 6. The *reverse* of the actor lane: instead of dictating prosody into speech,
> perceive prosody out of it. Sibling notes:
> [1 — Matcha-TTS actor](high-ambition-1-matcha-actor.md) ·
> [2 — Dramatic Reader](high-ambition-2-dramatic-reader.md) ·
> [5 — StyleTTS2-Lite](high-ambition-5-styletts2-lite.md).
> Intended landing zone: the **conversation piece** —
> [voice-interruption-and-discussion.md](../Prosodia/voice-interruption-and-discussion.md) ("Solo
> Book Club") and [director-narrative-memory.md](../Prosodia/director-narrative-memory.md) — where
> the user talks *back* to the system.

_Captured 2026-07-13 from the owner's framing. Status: **vision note** — parked deliberately for
when the conversation features approach; nothing scheduled._

---

## 🎯 The idea (owner's scenario)

A user tells a future embodied assistant: *"Wait, I didn't mean for you to hit the button THAT
hard!"* — and the system understands the prosodic weight of the spoken **"THAT"** (emphasis, pitch
spike, duration stretch) as *meaning*, distinct from flat "that," and replies with matching
conveyance. Voice conversation with AI that carries emotional weight and prosody **in both
directions**.

Origin symmetry: Sonora exists because Kokoro — otherwise fantastic — could not be *directed*.
The Audience exists because conventional STT — otherwise excellent — cannot *perceive direction*:
transcription discards exactly the channel Prosodia is built to control.

## 🧩 The core insight — the control contract is already bidirectional

We do not need a new representation. The typed control contract
(`crates/stage/src/prosody_payload.rs`; publicly disclosed in the
[defensive publication](../../Prosodia/Docs/defensive-publication-expressive-control.md)) works as
an *annotation* format exactly as well as a *dictation* format:

```
[V: -0.30 A: 0.80 T: 0.90 FB: ...,+3.1,... DS: ...,1.6,...] Wait, I didn't mean for you to hit the button THAT hard!
```

Same V/A/T channels, same per-token F0/duration tracks — except emitted by a **listener** that
*detected* them rather than a director that *dictated* them ("THAT" carries the measured pitch
spike and duration stretch). The full conversational loop becomes:

**hear → contract → think → contract → speak**

The director-LLM reasons over text *plus* conveyance annotations (input side), and answers through
the actor with dictated conveyance (output side). One interpretable interchange format, symmetric.
The theater troupe gains its missing role: **Director, Actor, Stage — and now the Audience.**

## 🥊 Differentiation vs. the field

* **Frontier voice modes** (GPT-4o-style, full-duplex speech-to-speech models like Moshi) *do*
  perceive prosody — but end-to-end, inside the black box, typically cloud-side, uninspectable.
* **Academic speech-emotion recognition / emphasis detection** exists but emits coarse categorical
  labels that nobody wires into a dialogue loop.
* **The Audience's angle is the same one Prosodia stakes out for synthesis: small, on-device, and
  typed.** Conveyance as an explicit, auditable annotation a developer can read, log, and route —
  not a vibe dissolved into hidden states. "Dictation, not suggestion" inverted:
  **transcription *with* conveyance, not transcription *despite* it.**

## ♻️ The corpus is dual-use (why this costs less than it looks)

The milestone-3 VAT-labeled expressive corpus is the same asset read in two directions:

* **(text + labels → audio)** — trains the directable actor (the current plan).
* **(audio → labels)** — trains the Audience.

Every hour invested in that corpus silently funds the reverse model. The tooling overlaps too: the
forced alignments and prosody measures used to *derive* VAT labels (cf.
`cdminix/libritts-r-aligned`, the Parler annotations — see
[dataset-landscape.md](dataset-landscape.md)) are precisely the feature
extractors a listener needs. Architecturally, a plausible v0 is not even a new model: a compact
ASR (whisper-class or smaller) plus a lightweight prosody head over alignments emitting
V/A/T + per-token tracks into the contract.

## ⚠️ IP note

The **listening direction is NOT covered by the 2026-07-13 defensive publication** (which discloses
the dictation/synthesis direction and its contract). If/when the Audience becomes real work, make a
fresh, explicit IP-posture decision (publish defensively vs. file) **before** the mechanism lands in
a public repo — same discipline as [open-decision-licensing.md](../Prosodia/open-decision-licensing.md).

## 🛑 What this note is not

Not scheduled, not a milestone, and explicitly **not** a reason to defer milestone 3 — the
directable actor and its corpus come first and are the enabling investment for this note anyway.
Revisit when the conversation features ("Solo Book Club" / discussion mode) move from notes to
build.

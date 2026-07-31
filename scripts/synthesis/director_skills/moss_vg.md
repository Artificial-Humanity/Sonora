# Target: MOSS-VoiceGenerator

## What this engine actually accepts
`build_user_message(text=..., instruction=...)` — **both required**. No reference
audio; timbre is generated from the instruction text alone.

This is the instruction-tuned MOSS. Do not confuse it with the MOSS-TTS 8.5B
flagship, which is an un-SFT'd base model that does not list `instruction` as an
input and reads the prompt aloud when the line is short.

## How to write the instruction
MOSS wants **one free-form sentence**, roughly 6–30 words, fusing persona, emotion
and prosody. It has **no key-value schema** — do not use the key: value style that
Qwen wants. These are MOSS's own published examples:

> An elderly female voice, slightly nasal and soft, speaking in a frail, polite
> British tone, conveying subtle discomfort with gentle hesitation.

> Mom scolding kid for breaking a vase, then seeing he cut himself, shifting to
> concern

> Hearty, jovial tavern owner's voice, loud and welcoming with a slightly gruff,
> friendly tone in American English, radiating warmth and hospitality.

> Little girl, innocent and curious, high-pitched and adorable

Note the shape: persona first, then delivery, as one flowing description. Situation
descriptions ("Mom scolding kid, then shifting to concern") work well — MOSS
handles narrative framing better than attribute lists.

## Speaking rate
There is no rate parameter. Pace must be described in words inside the instruction
("slow and deliberate", "fast, tumbling over the words").

## Narration (Neutral / Newscaster / Documentary delivery)

With VibeVoice and Dia set aside (2026-07-29), narration renders on this portfolio. MOSS
takes narration the way it takes everything — one persona sentence. Describe a **narrator
as a person**, not a medium (the broadcast-media rule below applies; "news anchor" is a
persona and fine, "as heard on the radio" is a medium and is not). Starting shapes —
**provisional, not yet auditioned as narration**:

> A composed adult male narrator's voice, even and unhurried, reading descriptive prose
> in American English with quiet authority.

> A professional news anchor's voice, female, brisk and crisp in American English,
> delivering information with detached urgency.

> A warm, measured documentary narrator's voice, middle-aged male, unhurried and
> explanatory in American English, with quiet fascination.

Narrator identity must hold across a book: no reference audio means identity lives
entirely in the sentence, so **freeze the persona clause per book** and vary only the
delivery tail.

## Narration failure modes (measured, delivery-v1-narration 2026-07-30)

Three stochastic defects surfaced at 10/33 audited narration clips — none is steerable
from the instruction, all are reroll-on-new-seed cases, and none is QC-detectable, so
they are what the ear samples are FOR:

1. **Early stop mid-sentence** ("stops after 'among the rest'") — EOS fired at a clause
   boundary on long prose. WER deletions catch the severe cases; reroll.

   ⚠ **Two different faults hide under this one symptom (measured 2026-07-31).** Part of
   it was never stochastic at all: `max_new_tokens` defaulted to 1000 in MOSS's own
   `generate()` and the renderer never overrode it, capping EVERY read at **~31 s**
   (~32.4 Hz frame rate). Any passage needing more than that was cut deterministically —
   `the-return_nar_0050_doc_MOS`, 840 chars, stopped at 30.88 s. That is fixed in
   `synth_moss_vg._token_budget`, which sizes the budget to the passage; **reseeding a
   long-passage truncation before 2026-07-31 could never have worked.**

   What remains genuinely stochastic is early EOS on SHORT passages — 113 to 266 chars,
   nowhere near the ceiling. Those are still reroll-on-new-seed. When a truncation
   appears, check the passage length first: over ~600 chars suspect the budget, well
   under it suspect the model.
2. **"Mid-1900s radio transmission" vocal effect** — an era/medium timbre the model
   drifted into on its own on Documentary personas, with NO media words in the
   instruction (the ban above was respected; the drift is the model's prior on
   documentary-flavored persona language). Reroll; if a persona draws it twice,
   reword the persona away from documentary-associated phrasing.
3. **Robotic IVR-like flow** — human timbre, machine cadence. Reroll.

## Accent
**Not supported.** The maintainers state on the record: "MOSS-TTS currently does not
support specifying accents in user input." Their one British example phrases it as a
"frail, polite British **tone**" among voice-quality adjectives — a timbre
description, not a phonological claim. Anchor the language ("in American English")
because their own examples do, but do not expect a regional accent to be executed.

## Do not
- **Never use square or curly brackets.** This model's input normalizer deletes
  anything inside `[...]` or `{...}` outright, silently.
- Do not use newlines — they are flattened.
- Do not describe rooms, microphones, eras or broadcast media. Always a dry,
  close-mic modern studio.
- Do not write a key: value attribute list. That is Qwen's format, not this one.

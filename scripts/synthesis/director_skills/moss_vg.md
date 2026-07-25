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

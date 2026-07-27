# Target: Qwen3-TTS-12Hz-1.7B-VoiceDesign

## What this engine actually accepts
`generate_voice_design(text=..., instruct=..., language="English")`.

There is exactly **one** direction slot: `instruct`. There is **no**
`voice_description` parameter — a previous version of this pipeline passed one and
it was silently swallowed by `**kwargs`, so the voice design never reached the
model at all. Everything you want the model to know must be inside `instruct`.

No reference audio. Timbre comes from your words alone.

## How to write the instruct
Qwen's own published demo directions use a fixed 12-key format, in this order:

`gender, pitch, speed, volume, age, clarity, fluency, accent, texture, emotion, tone, personality`

Example of the shape (Qwen's own):

> gender: Male. pitch: Low male pitch, generally stable. speed: Deliberate pace,
> slowing slightly after the initial exclamation. volume: Starts loud, then
> transitions to a projected conversational volume. age: Middle-aged adult.
> clarity: High clarity with distinct pronunciation. fluency: Highly fluent.
> accent: American English. texture: Resonant and slightly gravelly. emotion:
> Initially commanding, shifting to narrative amusement. tone: Authoritative
> start, moving to an engaging, descriptive tone. personality: Confident and
> performative.

**Use this 12-key format.** Fill every key. Keep each value short. This matches the
attribute schema Qwen is benchmarked on, so it is the most in-distribution phrasing
available — though Qwen does not formally document it as required, so treat it as
the strong default rather than a hard rule.

## Known behaviour to compensate for
- **Renders younger and higher than described.** For a mature voice, exaggerate:
  ask for older and lower than you actually want.
- Free-form prose also works, but is less reliable under high arousal. Anchor the
  voice (gender + age + timbre) before any delivery instruction, or the model
  drifts cartoonish.

## Accent
Qwen populates an `accent:` key in its own demos, but **only at national-variety
granularity** — "American English", "British English", "General American English".
No regional English accent (Scottish, Southern, Irish, RP, Australian) appears in
any Qwen material, there is no published accent evaluation, and accent has no
token-level representation in the model. Chinese dialects exist only as speaker IDs
in a different checkpoint, and this checkpoint's dialect table is empty.

So: fill the `accent:` key and keep it to a national variety. Whether Qwen actually
executes it is **untested** — our one attempt (teacher-ab-v1, 2026-07-26) was judged
without attending to accent and proved nothing either way. Treat it as unverified rather
than either working or inert, and do not promise an accent you cannot demonstrate.

## Do not
- Do not describe rooms, microphones, eras or broadcast media. The recording style
  is always a dry, close-mic modern studio.
- Do not write a separate design field. One `instruct` string only.

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

### ⚠ Bright / teen female casting: hold it out of "fairyland"
Qwen is one of three homes for bright and teen female casting (with Zonos and, since its
guard moved to pitch excursion on 2026-07-29, Chatterbox). Qwen does this casting **well**
— with one standing caveat in the owner's words: it is good *"when it's constrained from
going too far into fairyland."*

Understand the severity: the failure mode is not a mild twee lean you can live with. In
the owner's words it is **"truly the stuff of cartoon voicing by a specialized actor"** —
a fairy/sprite character voice, and **not usable in an adult non-fantasy novel at all**.
A clip that lands there is a DROP, not a low score. Since our corpus is overwhelmingly
adult literary prose, that makes this a category error rather than a matter of taste.

Left alone, a young + bright + high-arousal instruct collapses straight into it, and this
casting is where it bites hardest because every attribute you are filling in pushes the
same way: young age, high pitch, bright timbre and lifted emotion all compound.

Counter it inside the 12-key instruct — not with a "do not" sentence, which Qwen does not
take, but by writing the keys AGAINST the drift:

- `tone:` and `personality:` — name something grounded (`matter-of-fact`, `direct`,
  `unaffected`, `dry`). Never `whimsical`, `magical`, `playful`, `sparkling`, `bubbly`.
- `pitch:` — say **stable** or `level`. Unqualified "high" invites the sing-song contour.
- `fluency:`/`clarity:` — `natural`, `conversational`. "Crisp/precise" reads as elocution
  and pushes twee.
- `emotion:` — keep it the register's actual emotion. Do not add brightness on top of a
  bright voice; you are already there via `age` and `pitch`.

A young voice should still sound like a person talking, not a narrator performing a
fairy. If the read comes back sing-song, flatten `pitch:` and ground `personality:`
before touching anything else.

**This is measured, not asserted** (artifact-probe-routing, 2026-07-29). Five lines were
rendered twice each — once with a naive bright instruct, once written against the drift as
above — same casting, same text, only the instruct differing:

| arm | mean score | fairy-zone notes |
|---|---|---|
| naive | **2.0** | 4 of 5 ("firmly in the fairy zone", "fairy or child voice") |
| grounded | **5.0** | **0 of 5** |

Four of five naive reads landed at or over the line; none of the grounded ones did. Qwen
with a grounded instruct is currently the **best** home for bright/teen female casting in
the portfolio. Write the keys this way every time — the difference is the whole clip.

## Narration (Neutral / Newscaster / Documentary delivery)

With VibeVoice and Dia set aside (2026-07-29), narration renders on this portfolio. Qwen's
audited Neutral record is 5/5, and the 12-key format extends to narration directly — the
lane lives in `speed`, `emotion`, `tone` and `personality`. Starting points —
**provisional, not yet auditioned as narration**:

| lane | the keys that carry it |
|---|---|
| Neutral | `speed:` Steady, unhurried. `emotion:` Calm, even. `tone:` Even, narratorial. `personality:` Composed and unobtrusive. |
| Documentary | `speed:` Measured, deliberate. `emotion:` Quiet interest. `tone:` Warm, explanatory. `personality:` Knowledgeable, unhurried. |
| Newscaster | `speed:` Brisk, clipped. `clarity:` High, crisp diction. `emotion:` Detached urgency. `tone:` Authoritative, informative. `personality:` Professional, direct. |

- **Convey Newscaster as a VOCAL style, never as a medium.** The existing rule below —
  no rooms, no microphones, no broadcast media — applies with extra force here, because
  "like a news broadcast" is the obvious phrase and it is exactly the one that asks for
  old-radio EQ artifacts instead of a delivery. Describe the speaker, not the radio.
- The anti-fairyland discipline doubles as narration discipline: a grounded
  `personality:` is what keeps long stretches of narration from drifting performative.
- Narrator identity must hold across a book. Qwen has no reference audio, so identity
  lives entirely in the instruct — **freeze the narrator's 9 voice keys per book** (all
  but `speed`/`emotion`/`tone`) and vary only the delivery keys between passages.

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

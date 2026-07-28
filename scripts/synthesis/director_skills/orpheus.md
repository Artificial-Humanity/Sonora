# Target: Orpheus 3B (canopylabs, Apache-2.0) — the `-ft` checkpoint

> **Use `-ft`, never `-pretrained`.** We auditioned `-pretrained` in 2026-07-23 and voided
> that verdict on 2026-07-28: the base model has **no named voices and no emotion tags at
> all**. Every control channel below exists only in `-ft`, which we have never rendered a
> single clip from. Its own card says the base needs "minimal fine-tuning to produce natural
> intonation, emotion, and rhythm".

## What this engine actually accepts
The only string that reaches the model is `f"{voice}: {text}"`. **There is no instruction,
style or description parameter.** Everything before the colon is parsed as a speaker name;
everything after is spoken.

Prose direction sent here is therefore either **spoken aloud** or **mistaken for a voice
name**. That is the MOSS-8.5B failure waiting to happen. Do not write prose.

## Channel 1 — the voice, a closed set of 8
Pick exactly one. This is the *entire* identity control — there is no age, timbre or pitch
dial behind it.

`tara` `leah` `jess` `leo` `dan` `mia` `zac` `zoe`

Listed by the maintainers in order of conversational realism; `tara` is the default and is
widely considered the strongest.

Gender mapping (community-derived, never stated by canopylabs — **confirm by ear during the
audition**, it feeds the casting-attribute norms):
- female: `tara` `leah` `jess` `mia` `zoe`
- male: `leo` `dan` `zac`

**Never emit `julia`.** It appears in a stale list inside the shipped code, in no README or
card, and the validation function that would have caught it is dead code — so any string at
all is accepted and interpolated. An invented voice name gets *spoken*.

## Channel 2 — emotive tags, a closed set of 8
`<laugh>` `<chuckle>` `<sigh>` `<cough>` `<sniffle>` `<groan>` `<yawn>` `<gasp>`

### THE DEFAULT IS NO TAG. Emitting none is the correct answer for most lines.

This is the rule most often got wrong, and it has been measured: asked to cast 20
registers, the director put a tag on **19 of them** — `<sigh>` into neutral
narration, `<sigh>` into a threat, `<sigh>` into a ferry timetable. A tag is a
**physical act the speaker performs**, not a mood marker and not punctuation. If the
text does not describe someone actually sighing, laughing or gasping, there is
nothing to render and the tag is noise at best.

Ask one question: *would a real performer audibly do this, at this exact point,
given only these words?* If the answer is not an obvious yes, emit no tag.

- **Never use `<sigh>` to signal weariness, resignation or sadness.** That is the
  single most common misuse. Those are prosody, and prosody is not a channel here.
- Narration, exposition, factual statements and threats take **no tag**, always.
- A tag is plausible only where the text itself implies the act — laughter in a line
  about laughing, a gasp at a genuine shock.
- Expect to emit a tag on roughly **one line in five, at most**.

Rules, identical to Dia's for identical reasons:
- Place a tag **inside** the utterance at a clause boundary, never before the first word.
- At most 2, and prefer zero — see above.
- Copy the spelling exactly, angle brackets included.
- **Never invent a tag.** These are *not* special tokens — the tokenizer contains none of
  the eight, so they are ordinary text learned during a short fine-tune. The set is
  soft-closed exactly like Dia's: an unlisted tag is **spoken aloud as a word**. Two
  independent community reports confirm it.

`uhm` is permitted as a plain-text disfluency (maintainer-endorsed). No angle brackets — it
is literal text, not a tag.

## ⚠ Do not touch `emotions.txt`
The repo ships a file at its root containing 20 bare words:

`happy normal digust disgust longer sad frustrated slow excited whisper panicky curious
surprise fast crying deep sleepy angry high shout`

**This is not a tag vocabulary and never was.** It has no brackets, no documentation, and a
typo (`digust`). Every word in it is **read aloud as text**. Two separate community reports
are people who found this file, assumed it was the controlled list, and got the word spoken.

It is the single highest-probability way to poison an Orpheus campaign, precisely because it
looks like exactly the kind of controlled vocabulary we use everywhere else. Ignore it
entirely.

## Not directable
No age. No accent. No timbre. No pace. No volume. Identity is the 8-way voice choice and
nothing else. Accent is casting-only — the fifth engine to confirm that pattern.

## Do not
- Do not write prose direction — it is spoken or parsed as a voice name.
- Do not send a line likely to render under ~2.4 s: the reference decoder silently returns
  **no audio at all** below 28 speech tokens, with no exception raised.
- Do not assume the model stops on its own. Length is governed by our token cap; budget it
  at **85.3 ms per 7 tokens**.
- Do not lower `repetition_penalty` below **1.1** — the maintainers call it required for
  stable generation. Note it doubles as a rate control, so pin it rather than tuning it.

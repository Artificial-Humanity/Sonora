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

### ⚠ `tara` carries room tone (measured 2026-07-28) — do not cast her
revisit-v1 split by voice with no exceptions: **all 16 `tara` clips** drew the owner's
"slight reverb, like a small enclosed room with hard floors" verdict; **all 4 `dan` clips**
were clean. Measured, it is genuine reverberation and not a mix artifact — C50 median
11.7 dB against 15.4 dB for the clean clips, with a decay slope roughly half as steep.
It does not depend on the text, the register or the emotive tags, only on the voice.

The 6-voice probe rendered all six voices over the same five lines, `tara` as positive
control and `dan` as negative; the retake bank then put 16 more clips through jess/mia/leah.
**Owner-audited verdicts pooled over both, 2026-07-29:**

| voice | defective | verdict |
|---|---|---|
| `jess` | **0 / 14** | **the only reliable female voice — use it** |
| `dan`  | 0 / 5 | clean (male); its one 0 was an accent leak, not reverb |
| `zoe`  | 0 / 5 | no reverb, but weakest scores — unproven, not preferred |
| `leah` | 3 / 8 | 1 clear layering ("two different voices"), 1 electronic, probe 1 |
| `mia`  | 4 / 9 | 1 layering, 2 electronic, probe 1 |
| `tara` | **5 / 5** | **REVERB + white-noise hiss. Never cast.** |

**Cast `jess`.** An earlier version of this file called `mia` and `leah` "good, one clip
layered" on 5 samples each; at 8-9 samples they fail 35-45% of the time. Do not use them
for anything that has to land first time.

**But `jess` is rare, not immune (stress-v2, 2026-07-30).** After 0/24 across the probe,
retake and stress-v1, `s2_05_scornful_dis_ORP` broke into layering on a single word —
owner: *"breaks into layering at 'plan'; the pitch level at 'plan' is significantly higher
than the rest of the passage."* That is the same emphasis-peak split as Chatterbox, on an
engine with **no reference audio** — the excursion is produced by the render itself, so no
casting rule can guard it. Pooled `jess` rate: **1/34 (~3%)**. Treatment: stochastic —
screen-and-REROLL on a new seed, exactly the Chatterbox workflow; lines with a scornful /
sarcastic pitch spike on one word are where to listen.

### The "electronic" precursor (owner, 2026-07-29)
Between clean and layered there is a third state the owner describes as **sounding
electronically conveyed, "like a voice through a phone call" — on the verge of breaking
into layering without doing so.** It appears on `mia` and `leah`, never on `jess`.

Treat a clip like that as a REROLL even though nothing has audibly split: it is the same
failure caught early, and it is not what a literary corpus should sound like.

The owner's sharpest description: *"not like range is stifled — more like the loudness
edges are starting to break, similar to light crackle of static. Jess output is smooth."*
That reads as AMPLITUDE-domain breakup at transients. **But do not assume it is the same
mechanism as Chatterbox's emphasis-peak split** — an earlier version of this file said so
and the data does not support it. On Chatterbox the owner hears splits at emphasis peaks;
on Orpheus the two LAYERED clips in the retake bank are the two LOWEST-arousal lines
(A -0.2, "Everything is quite in hand", "The ferry goes at six") while the crackle clips
are the high-arousal ones. Arousal does not predict, and the fastest clip in the set is
clean.

An open hypothesis, not a finding: the split may occur at similar rates regardless of
delivery but be MASKED in loud energetic passages and EXPOSED in quiet sustained ones. If
so, calm narration is where to listen hardest, not where to relax. n=7 — do not build on it.

One incidental agreement worth recording: `rev_05_whimsy_ORP2_leah`, which the owner called
"notably smooth", is also the lowest-roughness clip of all 16 measured, and the longest and
slowest-paced (10.8 ch/s). Within `leah` the ordering is monotonic. Within `mia` it reverses.

Two measurements of it have been tried and BOTH came out backwards — bandwidth (those
clips carry *more* HF energy, not less) and envelope roughness in the 30-300 Hz crackle
band (the smoothest voice scored the *highest*). Ear only, like everything else in this
family; see the failure list in qc_artifacts.py before trying a third.

⚠ **A C50 measurement got this wrong and the ear corrected it.** The measured ranking put
`zoe` at 9.2 dB — worse than `tara` — and this file said so. The owner then dropped none of
`zoe`'s five clips. C50 is an early-to-late energy ratio, so a naturally breathy or soft
voice reads as "reverberant" without being so. It identified `tara` correctly and condemned
`zoe` falsely; treat per-voice C50 as a hint that needs an ear behind it, never a verdict.

It is also NOT a pitch effect: `leah` (224 Hz) and `mia` (220 Hz) are the highest-pitched
and both are fine, while `tara` (192 Hz) sits mid-range and is the one that fails. The SNAC
high-pitch weakness in canopylabs discussion #23 is real but is not what we are hearing.

The layering `mia` and `leah` show is the same emphasis-peak split seen on Chatterbox and
Zonos — this defect is portfolio-wide, not one engine's quirk. Expect it occasionally
anywhere and re-roll.

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

### ⚠ `<sigh>` renders badly — prefer `<chuckle>` / `<laugh>` (audit, 2026-07-29)
Sorting every tagged Orpheus clip by which tag it carried, `<sigh>` draws a complaint on
three different voices while the other two do not:

- `dan`  — dropped: sounds like "an English shipwright who, when sighing, gives away that
  he was actually an android" (the ACCENT was praised; the sigh is what killed the clip)
- `mia`  — "the sigh is weird. Sounds like a long worded out 'uuhhhaahh'"
- `jess` — scored 3, its lowest anywhere; every other jess clip is 4-5

`<chuckle>` scores 5,5,5,4 with "good chuckle" and "realistic low-key conversational
laugh"; `<laugh>` is clean wherever the voice itself is. These are not special tokens —
they are ordinary text learned in a short fine-tune, so per-tag quality varies. The
default is still NO TAG; when a line genuinely calls for one, `<sigh>` is the one to
avoid. `jess` did land a 5 with `<sigh>` once, so this is a caution, not a ban.

`uhm` is permitted as a plain-text disfluency (maintainer-endorsed). No angle brackets — it
is literal text, not a tag.

## Narration (Neutral / Newscaster / Documentary delivery)

With VibeVoice and Dia set aside (2026-07-29), narration renders on this portfolio.
Orpheus's surface for it is small and that is fine: **`jess`, no tag, and pacing carried
by punctuation** — the "narration takes no tag, always" rule above already covers most of
what there is to say. Sentence length and comma placement are the only pace controls that
exist here; a Newscaster read is short declarative sentences, a Documentary read is longer
clauses with room to breathe. **Provisional, not yet auditioned as narration** (its 7/9
Neutral record predates the jess-only rule).

⚠ **Calm narration is where to LISTEN HARDEST, not where to relax.** The open hypothesis
above — that the electronic/layering defect occurs at similar rates everywhere but is
MASKED by loud energetic delivery and EXPOSED in quiet sustained passages — makes
narration precisely the exposed case. `jess` has never failed (0/14), which is why she is
the only narration voice; but audition narration clips with that failure mode explicitly
in mind, because these are the clips where it would be audible.

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

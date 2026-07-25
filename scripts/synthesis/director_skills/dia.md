# Target: Dia-1.6B-0626

## What this engine actually accepts
Text only. `generate()` takes the transcript, an optional audio prompt, and
sampling knobs. There is **no instruction, style, emotion or accent parameter of
any kind**, and no technical report. Voice is drawn at random per seed unless an
audio prompt is supplied.

So there is almost nothing for a director to write here — with exactly one
exception.

## The one channel you do control: non-verbal tags
Dia is trained on a closed set of 21 non-verbal tags, placed inline in the
transcript. This is the complete list; anything outside it is accepted silently by
the byte tokenizer and misbehaves:

`(laughs)` `(clears throat)` `(sighs)` `(gasps)` `(coughs)` `(singing)` `(sings)`
`(mumbles)` `(beep)` `(groans)` `(sniffs)` `(claps)` `(screams)` `(inhales)`
`(exhales)` `(applause)` `(burps)` `(humming)` `(sneezes)` `(chuckle)` `(whistles)`

Rules:
- **Choose at most 2, and prefer none.** nari-labs: "Use non-verbal tags sparingly…
  Overusing or using unlisted non-verbals may cause weird artifacts."
- Copy the tag string **exactly**, including the parentheses.
- The tag is prepended to the line, so only pick something that would plausibly
  happen *before* the first word.
- Never invent a tag. `(whispers)`, `(scoffs)`, `(pauses)` are not in the set.

## Structure handled by the pipeline, not by you
- The transcript begins with `[S1]` and repeats a trailing speaker tag at the end
  (nari-labs' documented fix for Dia improvising tails).
- Input length is targeted at 5–20 seconds of speech. Under ~5 s "will sound
  unnatural"; over 20 s "will make the speech unnaturally fast."

## Accent, emotion, delivery
None are directable. Punctuation and capitalisation have some emergent effect
(Dia's own demo uses `!!!!!` for shouting) but this is undocumented and unreliable.

## Do not
- Do not assign Dia a line whose meaning depends on a specific delivery. It reads
  the text and nothing more. Neutral narration is its lane.
- Do not write a voice design for it. Nobody reads it.

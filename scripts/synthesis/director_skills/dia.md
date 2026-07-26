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
- The tag is placed **inside** the utterance at a natural clause boundary, never before
  the first word. A leading bare tag has no preceding speech to bound it and Dia renders
  it without duration constraint — that produced an "extreme, non-human" sigh and the
  worst tail in the batch (2026-07-26). nari-labs' own examples put tags mid-line.
- Never invent a tag. `(whispers)`, `(scoffs)`, `(pauses)` are not in the set.

## Structure handled by the pipeline, not by you
- The transcript begins with `[S1]` and repeats a trailing speaker tag at the end. That is
  nari-labs' documented tail mitigation, but it does **not** work — Dia rarely emits its own
  end token. The real bound is `synth_dia.token_budget()` (1.35x + 1.0 s + 1.2 s per tag).
  Keep the tag count low: each one buys real budget.
- The pipeline fixes temperature at **1.8**. Do not suggest lowering it — 1.5 collapses
  7/20 clips into white noise.
- Input length is targeted at 5–20 seconds of speech. Under ~5 s "will sound
  unnatural"; over 20 s "will make the speech unnaturally fast."

## Accent, emotion, delivery
There is no instruction slot, but Dia is **NOT a flat reader** — it is markedly expressive,
and punctuation and capitalisation are a real, working delivery channel (exclamation,
ellipsis, caps for stress; nari-labs' own demo uses `!!!!!` for shouting). Write the
punctuation you want performed. Accent remains undirectable — that is a casting property.

## Do not
- Dia is the **weakest of the four engines on measured quality** (median DNSMOS 2.53;
  12/20 pass in teacher-ab-v1, 2026-07-26). Prefer another engine when the line matters.
  Do not assume its failure mode is flatness — it is instability.
- Do not write a voice design for it. Nobody reads it.

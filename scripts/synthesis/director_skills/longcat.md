# Target: LongCat-AudioDiT-3.5B (Meituan, MIT)

> **There is nothing for the director to say to this engine.** Do not run a casting pass
> for LongCat's payload. This file exists to record *why*, and to state the one decision
> that does matter.

## What this engine actually accepts
`forward(input_ids, attention_mask, text_embedding, prompt_audio, duration, steps,
cfg_strength, guidance_method, return_dict)`.

Three things condition the output, and only three:

1. **`prompt_audio`** — the reference waveform. This is the sole carrier of voice, timbre
   and whatever affect the output has. It is pinned across every diffusion step.
2. **The text**, formed as `f"{prompt_text} {text}"` — the reference's own transcript
   prepended to the line. The transcript is not metadata; the model uses it to align the
   reference latent to its words.
3. **`duration`** — output length in latent frames, and therefore speech rate.

**No instruction slot, no style field, no emotion input.** The negative branch is a zero
tensor, not a negative prompt — there is not even a place to put one. Prose sent here is
**synthesised as speech**.

## The only decision that matters: which reference
Voice, timbre, inherited affect and (through the duration ratio) pace all come from the
reference clip. If the director has any role at all, it is **choosing a reference from the
audited bank to match the line's register** — a casting call, not a performance note.

## Constraints that bite
- **`prompt_text` must be the reference's exact transcript, and non-empty.** A wrong
  transcript corrupts the clone; an empty one raises `IndexError`. Our bank stores this as
  `ref_text` — keep it force-aligned.
- **Reference length is charged against the 60 s cap.** `duration + prompt_duration <= 60 s`.
  A long reference silently steals length from the line. Keep references **5–10 s**.
- **Duration is estimated, not learned** — roughly 12.2 English characters per second.
  Under-budget truncates; over-budget invites trailing artefacts.
- **A slow reference stretches output up to 1.5×; a fast one cannot compress below 1.0×.**
  Asymmetric, and the nearest thing to a pace control — which makes it a reference-selection
  criterion, not a dial.
- Use `guidance_method="apg"`, not the `"cfg"` code default. Every official example uses
  APG; it is the paper's headline contribution. `cfg_strength=4.0`, `steps=16`.

## ⚠ It mutates your text before synthesis
`normalize_text()` runs unconditionally and:
- **lowercases everything** — any casing-based emphasis is destroyed;
- **replaces all double and curly single quotes with spaces.**

For a book-prose lane full of dialogue that is a real content mutation, and it collides
directly with the SCM `quote` tag. It does **not** strip brackets, so `[stage directions]`
would survive and be spoken.

## Accent
None. Fifth engine, same answer. Accent transfers only insofar as it is present in the
reference clip, which makes it a property of the bank rather than a control.

## Narration (Neutral / Newscaster / Documentary delivery)

Same answer as everything else on this engine: **it is a reference-selection question.**
Clone from a certified keep that already carries the target delivery — a Neutral-labelled
keep for Neutral, etc.; the delivery column in ratings.csv is the shopping list. Two
cautions specific to narration: the 45% gender-flip rate is intolerable for a book
narrator (one voice across hundreds of clips — audit the first clip of any narrator
before rendering the rest), and the affect-transfer experiment below is a prerequisite
here too. Do not build a narration lane on LongCat before it passes.

## Measured standing (2026-07-28) — read before spending render budget
Its stated value is **label multiplication**: clone a certified performance onto new text
and inherit the human-validated V/A/T. That targets our real bottleneck, since valence
failed at vat3 on labels rather than training.

But: **25 of its 55 clips (45%) were renamed for gender mismatch, and every one is a flip**
— 13 intended-female rendered male, 11 intended-male rendered female. And its published
benchmark is CER/WER/SIM only, with **no expressiveness metric anywhere**, so "affect rides
through cloning" is an assumption nobody has tested.

**Settle it with one experiment before committing more budget:** clone N certified clips
onto new text and measure whether the output's instrument-derived V/A/T matches the
reference's. If affect transfers, the gender instability is tolerable (we audit gender
anyway). If it does not, this is an expensive way to generate arbitrary voices and VibeVoice
does casting better on the same conditioning surface.

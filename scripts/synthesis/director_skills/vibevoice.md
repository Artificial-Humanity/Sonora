# Target: VibeVoice (Large-Q8)

## What this engine actually accepts
The model receives **only** the text plus a reference wav. Its processor takes
`text` and `voice_samples` and nothing else — there is no instruction, style or
accent parameter anywhere in the API, and the technical report contains zero
occurrences of "accent", "instruct" or "style".

**Your words are never spoken to this model.** Your `voice_design` is consumed by
our own casting code (`ref_select.select_reference`) to pick which reference clip
to clone. You are not directing a performance here — you are writing a casting
call.

## What actually gets read out of your voice_design
Exactly two things, by regex:

1. **Gender** — matched on the words `female`, `woman`, `girl`, `feminine`
   (otherwise it falls through to Male). Always use one of those words, or an
   explicit `male`/`man`.
2. **Age band** — matched on exactly one of: `child`, `teen`, `young`,
   `middle-aged`, `mature`, `elderly`. Use one of these literal words. This maps to
   an F0 percentile target within gender.

Everything else you write — timbre, accent, emotion, personality — is **discarded
before it can affect anything**. Reference selection then scores on intended V/A/T
proximity, duration and F0 age, not on your prose.

## So write
A short casting line that leads with the two words that matter:

> middle-aged woman, warm even timbre

The timbre clause costs nothing and documents intent for the audit card, but do not
spend effort on it and do not rely on it.

## Accent
Impossible to direct. Accent is a property of whichever reference clip is selected.
If a particular accent is required, that is a casting-pool problem, not a direction
problem — say so rather than writing accent words that will be dropped.

## Do not
- Do not describe rooms, microphones, reverb, eras or broadcast media. VibeVoice's
  background ambience is content-aware and emergent — the maintainers state it
  cannot be directly controlled, and a reference clip containing ambience makes
  ambience *more* likely in the output. Never invite it.
- Do not write delivery instructions expecting them to be performed. Emit an
  `instruct` for the audit record if useful, but know the model never sees it.
- Do not omit the age word. Without one, casting falls back to no age target at all.

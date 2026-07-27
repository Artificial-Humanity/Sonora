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

1. **Gender** — matched by regex on exactly these four words: `female`, `woman`,
   `maternal`, `girl`. Anything else falls through to **Male**. Use one of those four
   literal words, or an explicit `male`/`man`. **`feminine` is NOT matched** and will
   cast a male reference.
2. **Age band** — matched by regex to an F0-percentile target within gender. Safe
   literal choices: `child` (0.95), `teen` (0.80), `young` (0.70), `middle-aged`
   (0.30), `elderly` (0.10). Note the word you WRITE is `young`; the band it records
   for training attribution is **`adult`**, matching the owner's five-band taxonomy.
   Omitting an age word entirely records no band at all — say one. Careful: **`mature` and `matronly` map to the SAME band
   as `middle-aged`, not to elderly** — say `elderly` if you mean old. The pool is
   also filtered to a 4.0-10.0 s duration window and skews young, so an `elderly`
   request is best-effort.

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

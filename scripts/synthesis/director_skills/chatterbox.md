# Target: Chatterbox (Resemble, classic English 0.5B, MIT)

> **CLASSIC, not Turbo.** Our notes said "Chatterbox-Turbo" for months; we were always
> running `ChatterboxTTS`. Turbo's `generate()` *accepts* `exaggeration` and then discards
> it with a log warning — the dial does not exist there, whatever Resemble's marketing
> says. If a renderer ever switches to Turbo, everything below stops working silently.

## What this engine actually accepts
`generate(text, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=1.2,
min_p=0.05, top_p=1.0, audio_prompt_path=None)`.

**There is no natural-language instruction slot.** None. Prose direction has nowhere to go
— do not write any. Your output is two numbers and, optionally, tags inlined in the text.

## Channel 1 — `exaggeration`, the arousal dial

A single scalar through a learned linear layer, which is why it behaves monotonically. Our
own 2026-07-17 audition mapped it by ear and that remains the only calibration in existence:

| value | what it gives |
|---|---|
| 0.25 | deadpan, dry, sarcastic |
| **0.5** | neutral (the default) |
| 0.7–0.9 | dramatic |
| 1.0 | genuinely, openly happy |

**Cap at 1.0.** Higher is reachable (nothing in the code clamps it) but is documented as
unstable, and unstable output costs more than the extra intensity is worth.

## Channel 2 — `cfg_weight`, and it is NOT optional

`exaggeration` and `cfg_weight` are **one control with two knobs**. Raising exaggeration
alone speeds the speech up and reads rushed. The documented pairing:

- neutral: `exaggeration 0.5`, `cfg_weight 0.5`
- dramatic: `exaggeration >= 0.7`, `cfg_weight ~0.3`

**Always emit both.** Emitting exaggeration alone is a defect.

## Channel 3 — paralinguistic tags (verify before relying on)

The tokenizer carries 34 atomic tokens. Use the **classic** spellings only:

`[UH]` `[UM]` `[giggle]` `[laughter]` `[guffaw]` `[inhale]` `[exhale]` `[sigh]` `[cry]`
`[singing]` `[whistle]` `[humming]` `[gasp]` `[groan]` `[whisper]` `[mumble]` `[sniff]`
`[sneeze]` `[cough]` `[snore]` `[clear_throat]` `[kiss]` `[shhh]` `[gibberish]`

Rules, and they are Dia's rules again:
- Place a tag **inside** the utterance at a clause boundary, never before the first word.
- At most 2. Prefer none.
- **Do not use Turbo spellings** — `[laugh]`, `[chuckle]`, `[clear throat]` (with a space)
  belong to a different, disjoint vocabulary. Cross-writing them shreds the tag into
  individual characters, which are then pronounced.
- Never invent one. Unknown brackets are shredded and spoken, exactly like Dia.

> **STATUS: the tag channel is UNVERIFIED.** The tokens tokenize atomically — that is
> proven — but vocabulary presence is not evidence of trained behaviour. Until the
> `[whisper]` smoke test passes, prefer none.

## Voice identity is casting, not direction
Set by `audio_prompt_path` (a reference clip). **Only the first 6 seconds** reach the
speaker prompt and the first 10 the vocoder — longer references are truncated, not
averaged, so a long clip wastes its own best material. You do not pick the voice; the
casting bank does.

Every clip in our 2026-07-17 audition used the single built-in fallback voice because no
reference was passed. Casting has never actually been exercised on this engine.

## Accent
Unsupported, and worse than merely absent: reference-clip accent **leaks uncontrollably**
(spontaneous British roughly 1 in 5 by community report), and the only official mention
treats accent as a defect to suppress. Never request one.

## Do not
- **Do not write `...` for a pause.** `punc_norm()` rewrites input before synthesis and
  converts ellipses to `, `. Write the comma you actually mean.
- **Do not send a line under ~20 characters.** Short lines hallucinate — "Hi!", "Yes" —
  and no combination of cfg/exaggeration/temperature/seed fixes it.
- **Do not exceed ~300 characters** without chunking, and never expect more than **40
  seconds** of audio: there is a hard 1000-token ceiling.
- **Do not send V−/low-arousal grief here.** Our audition found it renders as casual chat.
  The dial raises intensity; it cannot lower valence. That failure is probably genuine.
- Do not write prose. There is no slot for it.

## ⚠ Publication constraint — unresolved (owner 2026-07-28)

Chatterbox watermarks every output with Perth, unconditionally: no flag, no config, no
env var. The native component is unavailable on this box, so our renderers **disable it**
with a no-op patch. That creates a fork in the road, and both branches matter:

- **Keep the watermark** → Sonora trains on watermarked audio and may learn to reproduce
  the artifact, so our own student would emit Resemble's mark. Contamination.
- **Strip it** (what we do) → clean for training, but the clips are then unmarked synthetic
  speech, and our expressive-registers dataset is **published CC-BY-4.0 on HuggingFace**.

Owner's position (2026-07-28): *"easily defensible for training but potentially problematic
for public release… we may need to leave that out of the published dataset."*

**The mechanism to honour that does not exist yet.** Per-clip metadata carries a flat
`license: CC-BY-4.0` with no publication tier and no `engine_license`, so a Chatterbox clip
folded into `v1` would be published by default. Until a train-only tier exists, **no
Chatterbox output should be promoted into the published dataset.**

Note this is a responsible-AI measure, not a licence term — the MIT licence permits
stripping it. That makes it a decision we own rather than one we can point at.

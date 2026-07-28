# Target: Zonos v0.1 transformer (Zyphra, Apache-2.0)

Of everything in the portfolio this is the most **numerically** directable engine — and we
auditioned it with its main expressiveness knob at the bottom of the neutral band and no
voice set at all. Its 2026-07-17 verdicts are void.

## What this engine actually accepts
Conditioning is a dict, not prose. **There is no instruction slot** — a contributor puts it
plainly: *"you can't steer intonation with extra symbols/commands."* Do not write prose.

Our checkpoint (transformer) has exactly **7 conditioners**. Emit only these:

| field | range | note |
|---|---|---|
| `espeak` | — | the phonemes; REQUIRED |
| `speaker` | 128-d | from a 5–30 s reference clip. **Never leave unset.** |
| `emotion` | 8 floats | see below |
| `pitch_std` | 0–400 | **the expressiveness dial** |
| `speaking_rate` | 0–40 | phonemes per **second** |
| `fmax` | 0–24000 | 22050 for cloning |
| `language_id` | — | derived from `language` |

## Channel 1 — `pitch_std`, the highest-leverage control

**20–45 = normal speech. 60–150 = expressive.** Above that, the docs promise "crazier
samples".

Our entire 2026-07-17 audition ran at the **default 20.0** — victory, grief and threat all
rendered at the floor of the neutral band. If a single change explains why Zonos read flat,
it is this one.

## Channel 2 — the emotion vector

Eight floats in **fixed order**:

`[Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]`

**Emit a SHAPE, not magnitudes.** The vector is silently L1-normalised before it reaches the
model, so only proportions survive. Asking for "happiness 0.85" is not a thing that can
happen — in our audition, 0.85 arrived as 0.616. Think "happiness dominant, surprise
secondary", then write proportions that express that ordering.

**Emotion cannot fight the text.** It is documented as entangled with sentiment: angry text
conditions to anger easily, but pushing sad text toward anger works poorly. Choose a vector
that **agrees** with the line rather than one that contradicts it.

> Zyphra's own Gradio UI warns that emotion conditioning can destabilise the model, and
> ships with emotion switched **off** by default. Its best feature is also its main
> instability source. If a line comes back with dropped words or long silences, drop the
> emotion vector before blaming anything else.

## Channel 3 — `speaking_rate`
Phonemes per **second** (the docstring says "per minute" and is wrong; the README is
right). ~15 normal, 30 very fast. This is a genuine pacing control that Chatterbox lacks.

Keep `phonemes ÷ speaking_rate` under **30 seconds** — that is a hard cap and a training
limit. The docs are explicit that cutting the text is better than slowing the rate to fit.

## Voice identity is casting
A 5–30 s reference clip becomes a 128-d embedding. **Never leave `speaker` unset** — the
model then draws an arbitrary unconditional voice per seed, which is exactly what produced
the "gender-coverage gap" in our audition.

## Accent — the one genuine maybe, still unproven
Zonos is the first engine where accent is not a clean "no". It accepts `en-us`, `en-gb`,
`en-gb-scotland`, `en-gb-x-rp` with **distinct language ids**, and the phonemes really do
change — rhoticity and the TRAP–BATH split differ measurably.

**But the evidence says accent does not follow.** Phonemizer coverage is not training
coverage; a controlled `pt` vs `pt-br` test showed the code change and the output not; the
cards claim no dialect granularity. **Treat as UNVERIFIED and probably ineffective. Do not
build a campaign on it.** Accent remains casting.

## Do not
- **Never emit `vqscore_8`, `dnsmos_ovrl`, `speaker_noised` or `ctc_loss`.** Those
  conditioners exist on the *hybrid* checkpoint, not ours — passing them is silently
  ignored, the `**kwargs` trap again.
- Do not write prose, stage directions or bracketed markup. No slot exists.
- Do not exceed 30 s of phonemes.
- Do not set `cfg_scale=1` — it raises an assertion.

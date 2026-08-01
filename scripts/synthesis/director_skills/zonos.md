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
right). ~15 normal. This is a genuine pacing control that Chatterbox lacks.

**Ceiling: 16. Never write more.** The renderer clamps anything higher, so a larger
number buys you nothing but a warning line. Two measured failure points, one per ceiling:
at **24** (revisit-v1, 2026-07-28) both clips came back with a doubled, "ghostly" second
voice — past ~21 the model must fit the phonemes into fewer frames than it can
articulate, and the overlap is audible. At **18** (stress-v1, 2026-07-29) the audio is
clean but the delivery is not: 5 of 10 clips dropped as *"spoken fairly quickly and
without any noticeable emotional conveyance... dry like an actor reading a line
half-heartedly."* Rate 18 is articulable but not performable — the voice spends the line
keeping up instead of acting. It also under-runs the 4 s clip floor on any text shorter
than ~92 characters. Useful band: **12 slow / 14 neutral / 16 fast**.

Urgency is `emotion` and `pitch_std`, not rate. If a line needs to feel hurried, raise
arousal — do not raise the rate past 18 to get there.

Keep `phonemes ÷ speaking_rate` under **30 seconds** — that is a hard cap and a training
limit. The docs are explicit that cutting the text is better than slowing the rate to fit.

## Narration (Neutral / Newscaster / Documentary delivery)

With VibeVoice and Dia set aside (2026-07-29), the narration lanes render on this
portfolio — and Zonos is the natural first choice, because narration is the one job where
its measured weakness is the requirement. stress-v1 dropped 5 of 10 expressive-dialogue
clips as *"spoken fairly quickly and without any noticeable emotional conveyance... dry
like an actor reading a line half-heartedly"* — for a victory line that is a defect; for
an audiobook narrator baseline it is close to the spec. The numeric dials mean narration
here is repeatable in a way no prose-directed engine can match.

Starting points — **now auditioned** (delivery-v1-narration round 1 + the 2026-07-31
reroll; the rate floor and the emotion column below are measured, not inherited):

| lane | `pitch_std` | `speaking_rate` | `emotion` |
|---|---|---|---|
| Neutral | 20–35 | 14–15 | **omit** |
| Documentary | 30–45 | **14** | **omit** |
| Newscaster | 35–45 | 15–16 | omit |

- **Do not go below `speaking_rate` 14 on prose.** Rate 13 stretches pauses at commas
  and colons out of proportion to the rest of the read: it failed 37% vs 29% at rate 14,
  and — the part that is not a book confound — three *lovecraft* Documentary clips at 13
  independently earned the identical note *"Comma pauses are a bit too long but not
  terrible"* and were kept, while two castle-of-otranto clips at 13 tipped over the same
  edge into rerolls (*"Long pause between 'victim' and 'and vowing'"*, *"between
  'promised:' and 'but Isabella'"*). Different book, same rate, same complaint: it is the
  rate, and 13 sits right at the threshold where a long clause tips it. The 12 that used
  to open this range is now out of bounds entirely.
- The Documentary row used to read *"omit, or Neutral-dominant"*. That option is
  withdrawn — a Neutral-dominant vector is the precise thing measured below as the cause
  of the pause-and-rush instability, so offering it here contradicted the bullet that
  follows. `emotion` is **omit** on every narration lane, without exception.

- **Omit the emotion vector for narration — and "omit" means UNCONDITIONAL, not a
  neutral vector.** Measured the hard way (delivery-v1-narration, 2026-07-30): a
  neutral-dominant vector still actively conditions the emotion channel
  (`make_cond_dict`'s default `unconditional_keys` is only `{vqscore_8, dnsmos_ovrl}`;
  Zyphra's UI turns emotion "off" by ADDING it to that set). Five of nine narration
  groups carrying a neutral-dominant vector showed the documented instability on long
  prose: **3–5 s pause at a clause boundary, then rushed resumption, often skipping
  words or truncating the end.** Emit `emotion: null` — the renderer compiles that to
  true unconditional. On long narration also expect, occasionally: inserted
  disfluencies ("umm", "Hmm" — the owner rated one a 4 as realism, another a 2;
  double-edged), and rare hallucinated background sound (violin, noise) — reroll those.
  The **breath intakes at commas are prized** (owner: "very good for generating
  realism") — do not fight them.
- Newscaster pace is rate, which this engine genuinely has — 15–16 is the brisk end.
  The hurried-and-flat verdict at 18 (now clamped away) applies to any delivery.
- Pin ONE reference per narrator (`pinned_reference`), don't re-cast per clip: a book's
  narration must be one voice across hundreds of clips, and per-clip casting consults the
  pool every time.

## Quality softens at loudness peaks under maximum expressiveness
At the riskiest legal settings (pitch_std 85 + a high-arousal emotion vector), expect
occasional **slight quality degradation at the loudness edges** — the owner's stress-v2
verdict on 2 of 10 such clips: *"not enough for layering but enough to rate the general
quality lower"* (scores 2–3, still keeps). This is the precursor state to the split, not
the split itself; it costs score, not keeps. For lines that must land at full quality,
either lower pitch_std toward the 60s or plan a reroll pass. No action needed for
ordinary expressive work.

## Voice identity is casting
**Zonos is NOT immune to the emphasis-peak voice split.** It was briefly treated as the
safe haven for bright/teen female casting, on the strength of a single revisit-v1 clip
that had not been flagged — the same weak evidence that misled us on Chatterbox. The
routing probe (2026-07-29) settled it: `rt_news_ZON` was dropped for layering, on the
highest-arousal line at `pitch_std` 85. **1 of 5** against roughly 4 of 8 on Chatterbox —
a lower rate, not immunity.

`MAX_REF_EXCURSION = 240` therefore applies here too: swooping references break, steady
ones do not, and median pitch predicts nothing. For bright/teen female casting **Qwen with
a grounded instruct is currently the strongest option** (mean 5.0 in that same probe); use
Zonos when you need reference-based identity rather than a described voice.

A 5–30 s reference clip becomes a 128-d embedding. **Never leave `speaker` unset** — the
model then draws an arbitrary unconditional voice per seed, which is exactly what produced
the "gender-coverage gap" in our audition.

Your `voice_design` is consumed by our own casting code (`ref_select.select_reference`),
never by the model — it selects that reference clip. Exactly two things are read out of
it, by regex:

1. **Gender** — matched on exactly these four words: `female`, `woman`, `maternal`,
   `girl`. Anything else falls through to **Male**. **`feminine` is NOT matched.**
2. **Age band** — safe literal choices: `child`, `teen`, `young`, `middle-aged`,
   `elderly`. Omitting an age word records no band at all — say one. **`mature` and
   `matronly` map to the SAME band as `middle-aged`**, not to elderly.

Write a casting call that fits **this line's speaker**. The speaker varies, so the
casting must vary — any of these shapes:

> young man, bright open timbre
> elderly woman, dry and unhurried
> teen girl, quick and breathless
> middle-aged man, gravelled and tired

Everything past the gender and age words is discarded before it can affect anything,
so spend no effort on the timbre clause. But **do not reuse one casting call across
different registers**: identical casting for a victory line and a grief line means
the reference pool was never really consulted.

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

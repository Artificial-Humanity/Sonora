# Target: Chatterbox (Resemble, classic English 0.5B, MIT)

> **CLASSIC, not Turbo.** Our notes said "Chatterbox-Turbo" for months; we were always
> running `ChatterboxTTS`. Turbo's `generate()` *accepts* `exaggeration` and then discards
> it with a log warning — the dial does not exist there, whatever Resemble's marketing
> says. If a renderer ever switches to Turbo, everything below stops working silently.

## What this engine actually accepts
`generate(text, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=1.2,
min_p=0.05, top_p=1.0, audio_prompt_path=None)`.

**There is no natural-language instruction slot.** None. Prose direction has nowhere to go
— do not write any.

**Your entire output for this engine is a casting call and two numbers: `voice_design`,
`exaggeration` and `cfg_weight`.** That is the whole adapter — and the `voice_design` is
never spoken to the model, it only picks the reference clip (see Casting below). See the
refuted tag channel below before adding anything else.

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

### Choosing a value — start from the line's arousal, then adjust

You are given the line's arousal as fixed context. Map it first. This is the table
above expressed on the A axis — not a new measurement, just the same calibration in
the terms you are actually handed:

| the line's arousal | `exaggeration` | `cfg_weight` |
|---|---|---|
| A ≥ +0.6 — elated, urgent, shouting | 0.8–1.0 | 0.3 |
| A +0.2 to +0.6 — animated, insistent | 0.6–0.7 | 0.4 |
| A −0.2 to +0.2 — level, conversational | 0.5 | 0.5 |
| A ≤ −0.2 — subdued, weary, flat | 0.25–0.4 | 0.5 |

**The sarcasm rule below is an exception for ironic lines, not the default.** If you
are writing 0.25 for a line that is neither ironic nor subdued, you have mistaken the
exception for the rule — that failure is real and measured: across 20 registers the
director once emitted (0.25, 0.3) for 18 of them, victory and urgency included.

### What the dial actually is: a RATE PROFILE, not a register selector

This is the key to using it well, and it was established by ear on 2026-07-28.

Lower exaggeration **slows** the line; higher **speeds it up** (the docs say as much for the
upper end). It is prosodic intensity and pace — *not* an emotion control. That is why it
cannot reach negative valence: it changes how hard and how fast, never how the speaker
feels.

**Consequence for sarcasm — do NOT hard-map it to a value.** Which setting reads as ironic
is **situational by text**, and it depends on where the sarcastic weight sits:

| the line's shape | choose | why |
|---|---|---|
| irony **front-loaded**, with a throwaway tail | **0.5** | the tail speeds up and lands dismissive — something to get past |
| irony **end-loaded**, the punchline last | **0.25** | the ending slows and gets room to land |

Owner's own examples, same test: *"Oh, brilliant. That is exactly what I needed today,
**thank you so much**"* is more sarcastic at **0.5** — the tail is sped up and turns
dismissive. *"Sure. That sounds like a completely reasonable plan **with no obvious
problems**"* is more sarcastic at **0.25** — the punchline is slowed and gets focus.

So the director's question is not "how intense is this line" but **"where does the weight
need to fall, and does that phrase need time or contempt?"** Read the line, find the ironic
payload, then pick.

Confirming measurement: every 0.25 render ran longer than its 0.5 control (up to +0.7 s on a
3.5 s line, ~20%). The dial does real prosodic work — unlike the refuted tag channel below,
where changed clips came out *shorter* and sounded identical.

## Channel 2 — `cfg_weight`, and it is NOT optional

`exaggeration` and `cfg_weight` are **one control with two knobs**. Raising exaggeration
alone speeds the speech up and reads rushed. The documented pairing:

- neutral: `exaggeration 0.5`, `cfg_weight 0.5`
- dramatic: `exaggeration >= 0.7`, `cfg_weight ~0.3`

**Always emit both.** Emitting exaggeration alone is a defect.

## ~~Channel 3 — paralinguistic tags~~ — TESTED AND REFUTED (2026-07-28)

**There is no tag channel. Never emit one.**

The tokenizers do carry 34 atomic tokens (`[whisper]`, `[sigh]`, `[laughter]`, …) and they
genuinely tokenize as single units — that much is code-verified. But a smoke test rendering
the identical line, same seed, plain vs `[whisper]` vs `[sigh]` produced audio the owner
judged **identical to plain**. Vocabulary presence is not trained behaviour, and here it is
demonstrably not.

Corroborating detail: both tagged renders came out *shorter* than plain (2.8 s and 2.9 s vs
3.1 s). A realised sigh adds time; consumed-and-discarded tag text removes it.

So emitting a tag is at best inert and at worst harmful — a misspelling (Turbo's
`[laugh]` for classic's `[laughter]`, say) is shredded into characters and **pronounced**.
All risk, no upside.

## Narration (Neutral / Newscaster / Documentary delivery)

With VibeVoice and Dia set aside (2026-07-29), narration renders on this portfolio.
Chatterbox's audited Neutral record is 7/7 — it has simply almost never been asked.
Remember the dial is a RATE PROFILE (see above): for narration you are choosing a pace,
not an emotion. Starting points — **provisional, not yet auditioned as narration**:

| lane | `exaggeration` | `cfg_weight` | why |
|---|---|---|---|
| Neutral | 0.5 | 0.5 | the documented neutral — do NOT drift lower; low exag reads subdued/ironic, not neutral |
| Documentary | 0.4 | 0.5 | measured, slightly slower — the 0.25 end is sarcasm territory, stay off it |
| Newscaster | 0.6 | 0.4 | brisk and projected via the rate profile |

- **No tag channel, no prose** — unchanged; narration gives you nothing new to emit.
- **Cast a steady, LOW-EXCURSION reference and pin it per narrator/book.** Narration is
  hundreds of clips in one voice: per-clip casting consults the pool every time, and any
  splitting would surface repeatedly in the one place a listener spends the most time.
  Steady references are also exactly the split-safe end of the pool, so narration casting
  is naturally conservative.
- The V−/grief prohibition below still applies: somber documentary passages about loss
  are a poor fit — route those lines to Qwen or Orpheus.

## Voice identity is casting, not direction
Set by `audio_prompt_path` (a reference clip). **Only the first 6 seconds** reach the
speaker prompt and the first 10 the vocoder — longer references are truncated, not
averaged, so a long clip wastes its own best material.

Every clip in our 2026-07-17 audition used the single built-in fallback voice because no
reference was passed. Casting has never actually been exercised on this engine.

Your `voice_design` is consumed by our own casting code (`ref_select.select_reference`),
never by the model. Exactly two things are read out of it, by regex — identical to
VibeVoice, because it is the same function:

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

### The voice splits at emphasis peaks — casting handles it, you do not
**Cast bright and teen female voices normally.** An engine-level ban existed briefly and
was withdrawn: a blind 16-clip audition (2026-07-29) showed median pitch does not predict
the split at all — a 227 Hz reference split and a 335 Hz one did not — while pitch
**excursion** does. `MAX_REF_EXCURSION` now refuses swooping references at casting time
and leaves steady-but-high ones available, so the constraint costs 39% of the female pool
instead of the 62% a pitch ceiling cost.

What this means in practice: **lilty, singsong, whimsical and keening references are the
dangerous ones** — three of the four blind failures were exactly those registers. A
steady-but-high voice is comparatively safe at any median pitch. Casting enforces this;
nothing is required of you.

Two things you should know rather than act on:

**It is stochastic.** The same reference at a different seed renders clean or splits
(`cbx_A_f258_s8899` clean, `cbx_A_f258_s1234` split). The excursion ceiling lowers the
rate; it does not eliminate it. A split clip is a re-roll, not a casting mistake.

**No measurement catches it — audition is the gate.** Four detectors have failed
(whole-clip envelope autocorrelation, local CPP dip, emphasis-subharmonic delta, and
median-F0 as a proxy). In the blind test the owner caught 4 of 8 with **zero** false
alarms, so the ear is both the detector and the specificity check. Never treat a clean
`qc_artifacts.py` run as evidence a Chatterbox bank is free of splitting.

The mechanism, in the owner's words: *"at certain emphasis points (either higher pitch,
loudness or the combination), the vocal splits into layering."* The break is at PEAK
pitch, not median — so the median ceiling is a proxy, and a crude one.

**`exaggeration` modulates the break but never gates it** (sweep 0.1→0.75 on the 266 Hz
reference, 2026-07-29): minimal at 0.3, worse at 0.5, and at 0.75 it spreads across the
whole line instead of one phrase. Below 0.3 it does not clear — it simply moves to a
different phrase, and costs ~30% pacing. So there is a severity floor near 0.3 and no
setting that removes it.

The harm is an **interaction**, not an exaggeration effect: seven revisit-v1 clips ran at
0.75 on low-F0 references and were all clean. On a reference under the ceiling, use the
full `exaggeration` range as documented above — nothing here constrains you. It is only
high emphasis ON a high-pitched reference that breaks.

**Do not rely on `qc_artifacts.py` to catch this.** Three detectors have failed on it
(whole-clip envelope autocorrelation, local CPP dip, emphasis-subharmonic delta — the
last reached 20% precision on 40 labelled clips). A clean QC run is not evidence that a
Chatterbox bank is free of splitting. This defect is currently ear-gated only.

One reference is blacklisted in `synth_chatterbox.REF_BLACKLIST`
(`protective_urgency_00_urgentF_s1234`, worst in every condition tried).

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

## ⚠ Publication constraint — RESOLVED (owner 2026-07-28): train-only

Chatterbox watermarks every output with Perth, unconditionally: no flag, no config, no
env var. The native component is unavailable on this box, so our renderers **disable it**
with a no-op patch. That creates a fork in the road, and both branches matter:

- **Keep the watermark** → Sonora trains on watermarked audio and may learn to reproduce
  the artifact, so our own student would emit Resemble's mark. Contamination.
- **Strip it** (what we do) → clean for training, but the clips are then unmarked synthetic
  speech, and our expressive-registers dataset is **published CC-BY-4.0 on HuggingFace**.

**Owner's decision (2026-07-28): strip the watermark, and use these clips for TRAINING
ONLY — they never enter the published dataset.** So Chatterbox is fully available as a
teacher; only republication is closed.

The mechanism now exists. Every metadata row carries `engine_license` and `publish`, and
`scripts/synthesis/publish_tier.py` sets `publish: false` for every Chatterbox clip and
**fails** if one is ever marked publishable. Run it before any release build.

Note this is a responsible-AI measure, not a licence term — the MIT licence permits
stripping it. That makes it a decision we own rather than one we can point at, which is
exactly why it is enforced in code and not just written down here.

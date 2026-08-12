# Target: Sonora (our own Actor) — WIP, NOT LOADED

> **STATUS: WIP — NOT LOADED. DO NOT WIRE THIS INTO THE DIRECTOR YET.**
> `load_skill()` refuses any file carrying this banner (see the guard in
> `book_ingest.py`). Sonora has no renderer in this pipeline and is not a valid
> `engine` value in any bank.
>
> This file exists **before** the interface does, deliberately. Every failure of the
> 2026-07-25 relay audit happened because nobody had written down what actually
> reached each engine — the direction was authored, and its arrival was assumed. For
> Sonora we are writing the director-facing contract *first*, so the implementation
> has a target to satisfy rather than a description to be reverse-engineered from.
> When Sonora reaches the directable phase, this becomes the spec to tune Gemma
> against, and the diff between what is written here and what shipped is itself the
> review.

## The governing rule

**If a field cannot be verified after the render, it does not belong in the contract.**

This is the one lesson worth more than the rest. Prose direction is unfalsifiable —
you cannot tell from the audio whether "deliver with quiet menace" was honoured,
ignored, or silently dropped into `**kwargs`. Every teacher engine we run fails this
test, and it is why a whole campaign's direction went into a hole undetected.

Sonora's channels are numeric and instrument-checkable end to end (energy already
verifies at rho ~ 1.000). Keep it that way. Prefer a channel you can measure over an
expressive one you cannot.

## What Sonora accepts (pinned contract, ARCHITECTURE.md §1)

Nothing here is prose. The Director emits a typed object; the host compiles it.

| input | type | notes |
|---|---|---|
| phonemes | int[] | 178-symbol locked IPA vocab, intersperse-0 |
| `speaker` | **float[64]** | a vector, NEVER an id and never a description |
| `vat` | float[3] in [-1,1] | slot map `{valence: 0, energy: 1, tension: 2}` |
| `length_scale` | float | **host-side**, not learned |
| guidance `s` | float | optional CFG |

## How to direct it

**1. Label the line, engine-agnostically.** V/A/T floats plus a `register` copied from
the controlled lexicon (`register_lexicon.json`). Register is metadata and audit
vocabulary — it is *not* a conditioning input, and must not be invented.

**2. Cast by vector, never by adjective.** Describing a voice in prose failed on every
engine we tested, including the ones that accept prose. Emit a casting *selection* —
a character key resolving to a 64-dim vector, or typed attributes (gender, age band)
that the casting layer resolves against measured norms. See
[casting-attribute-norms-brief.md](../../../notes/casting-attribute-norms-brief.md).

**3. Accent is casting, never direction.** No engine we run represents accent at any
level, and Sonora will not either. It is a property of the speaker vector, realised by
selection against measurable norms — never a fourth channel.

**4. Do not ask the model for what is free at inference.** Pace is host-side
`length_scale`; per-frame loudness is a host-side dB bias. Both were measured free and
dB-exact (2026-07-14; the measurement note was folded away, git history has it, and the
resulting rule is pinned in `notes/ARCHITECTURE.md` §1) — WER 0 at -12 dB. Asking
the model to learn them wastes capacity that valence needs.

**5. Span markup, when the span layer lands, uses CLOSED vocabularies only** — SCM's
`emphasis` (level 1-3), `pause_after` (micro/short/med/long), `pace` (slow/fast),
`pitch_move` (rise/fall), `nonverbal`. Dia's 21-tag set is the model here, and it is
its best feature precisely because a closed set is checkable. Enumerate; never accept
free text.

## Carried forward from the teacher survey (2026-07-26)

- **Structured attributes beat prose.** Qwen's own demos impose a 12-key block
  (gender/pitch/speed/volume/age/clarity/fluency/accent/texture/emotion/tone/
  personality) *inside* its free-text field — evidence the free text was a concession,
  not a design. Sonora should take those as typed fields or not at all.
- **Situational framing is a Director-side strength, not a model input.** MOSS-VG
  responds well to "Mom scolding kid, then shifting to concern". Let Gemma *think* in
  scenes; let the Actor receive only numbers. The scene compiles to V/A/T + casting.
- **Both prose engines take exactly ONE string.** Our `design`/`instruct` split was our
  abstraction, not theirs, and it is where the voice design got lost. Do not reinvent a
  two-field text interface for Sonora.
- **Identity by reference is the one thing cloning engines do well.** VibeVoice's
  casting fidelity is real; its problem is that its reference pool is 100%
  own-synthesis. Sonora's speaker vector is the same idea with a measurable space
  instead of a clip.

## Open questions (resolve before this file is loaded)

1. **Is `register` ever a conditioning input, or permanently metadata?** Today it is
   metadata. A register embedding is possible but unmotivated while valence is unlearned.
2. **How does the Director address a character across a book?** A stable character key
   resolving to a vector, presumably host-side today and a speaker encoder later — but
   the addressing scheme is unpinned.
3. **Does span-level conditioning arrive as per-token VAT, or as a separate span
   channel?** ARCHITECTURE pins per-token FiLM; SCM spans are authored per token range.
   The compile step between them is unspecified.
4. **What is the verification instrument for each new field?** Per the governing rule, a
   field without one should not ship. Energy has LUFS, tension has the phonation
   composite, valence has `valence_combo_v1` and it is the weak one. Anything new needs
   an answer here first.
5. **Loudness normalisation policy** — currently unresolved for the teacher corpus, and
   it directly affects what the energy channel learns.

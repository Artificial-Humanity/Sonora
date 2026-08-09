# Teacher engines — license verdicts and measured standing

The record of which large TTS models may teach Sonora (license wall: outputs trainable
only from Apache-2.0/MIT weights run locally), which are excluded and why, and where
the live portfolio stands. The 2026-07-17 per-clip audition narrative and the
superseded portfolio proposals were cut 2026-08-02 (git history has them); what
remains here is what stays decision-relevant.

**Current portfolio: chatterbox · qwen · zonos · orpheus · moss_vg.** Benched:
**VibeVoice · Dia · LongCat** — status, evidence and end-conditions in
[§ Benched engines](#benched-engines) (one record; do not restate elsewhere).
Allocation is three-layer
(`ref_select.ENGINE_MIX_BY_LANE`, see [delivery-mix-campaign.md](delivery-mix-campaign.md));
onboarding pattern + interface gotchas: [tts-engine-onboarding.md](tts-engine-onboarding.md).

## What "teacher" means (three escalating uses)

1. **Benchmark** — A/B references for human audits (zero risk; NC models allowed here).
2. **Director vocabulary study** — how strong models realize "tense"/"grieving"/"elated".
3. **Synthetic manufacture** (the strategic one) — prompt permissive teachers for the
   scarce extremes, gate with the standing instruments + ears, merge as a labeled
   bounded-minority slice. Synthetic keeps face the same real-data-calibrated gates as
   everything else.

## License verdicts (verified against primary sources on the dates noted)

### Eligible (Apache-2.0 / MIT weights)

| Model | Params | Standing |
|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | 1.9B | Apache-2.0. **Trusted tier** — the portfolio's gold standard (measured 2026-07-27) |
| **MOSS-VoiceGenerator** (`moss_vg`) | 2.1B | Apache-2.0. THE MOSS for directed work; scrutinized tier |
| **MOSS-TTS 8.5B** (`moss85`) | 8.5B | Apache-2.0 — but an **un-SFT'd BASE model**: no instruction input; reads prompts aloud on short lines. Cloning-only candidate via its unused `reference` slot; NOT a directed generator |
| **Zonos-v0.1-transformer** | 1.6B | Apache-2.0. Numeric prosody dials (the closest native V/A/T interface); **normal** tier 2026-08-04 (r2: 44/44 directed emotion-off, 37/38 keeps) |
| **Orpheus-3B `-ft`** | 3B | Apache-2.0. Normal tier; `tara` banned in code (room reverb) |
| **Chatterbox** (classic, NOT Turbo) | 0.5B | MIT. Trusted-provisional; exaggeration dial is real (Turbo's `generate()` accepts and then discards it — verified in shipped code 2026-07-28) |
| **LongCat-AudioDiT-3.5B** | 3.5B | MIT. **BENCHED** — see [§ Benched engines](#benched-engines) |
| **Maya1** | 3B | Apache-2.0 verified on the weights repo. Un-auditioned; enters at onboarding step 1 if the field ever wants another description-driven engine |
| VibeVoice-Large | — | MIT. **BENCHED** — see [§ Benched engines](#benched-engines) |
| Dia-1.6B-0626 | 1.6B | Apache-2.0. **BENCHED** — see [§ Benched engines](#benched-engines) |

### Excluded on license (never teachers; verified live tags on dates noted)

| Model | Why |
|---|---|
| **Higgs TTS 3** (Boson, 4.7B) | NC license (2026-07-17). **Benchmark shelf** — its renders set the audit bar; never trains, never calibrates a detector |
| Mistral Voxtral-4B-TTS · Spark-TTS · F5/MaskGCT lineage · Fish/OpenAudio s1-mini · ChatTTS | NC / NC-SA weights (2026-07-17) |
| **ElevenLabs** (all plan tiers, 2026-07-18) | Prohibited Use Policy bans output as training input (§9k), in any training dataset (§9l), and competing-model development (§9j) — a flat contractual ban, stricter than NC. Never a teacher, never a benchmark render source |
| **IndexTTS-2** | bilibili license §3(c) + §1.5: outputs are Derivative Works, barred from improving other models. Textbook wall case (2026-07-17) |
| **Step-Audio-EditX** | Code Apache-2.0 but weights repo carries no license — unstated = fails the wall until StepFun clarifies |
| **DramaBox** (Resemble, LTX-2 Community License) | Conditional (revenue cap, no-training clause, disclosure rider) — not CC-BY-4.0-or-freer. Owner: **on ice**; private-lineage-only if ever revisited |

Watch list (deliberately deferred): `scripts/synthesis/teacher_audition/README.md` —
GLM-TTS waits on its unreleased post-RL checkpoint (repo dormant since 2026-01);
Kimi-Audio is an instrument candidate, not an actor. Revisit scope is **quality
rejections only, never license rejections** ([tts-engine-onboarding.md](tts-engine-onboarding.md)).

## Standing rule

> No TTS model enters the portfolio without a studied interface — the actual renderer
> call signature, verified at the call site — and a Gemma skill-file adapter in
> `scripts/synthesis/director_skills/`. If we cannot say what reaches the model, we
> cannot direct it, and we must not grade it.

## Measured standing

Allocation and audit shares are **code, not this file**: render share lives in
`ref_select.ENGINE_MIX_BY_LANE` (per-lane weights from heard production verdicts,
capability veto via `ENGINE_CHANNELS`, diversity floor), audit share in
`pick_audit_subset.TIERS`. A tier is a tag on an engine, not an allocation bucket
(owner 2026-07-31).

Two standing measurement lessons, both paid for twice:

- **An engine's failure rate is a property of the INTERFACE until proven otherwise.**
  Qwen was nearly discarded pre-fix; zonos read 31% across mixed pre/post-fix
  populations and 6.5–12% conditioned correctly. Before assigning a rate, check every
  clip behind it was rendered through the interface now believed correct.
- **Probe/stress campaigns are adversarial by construction and libel every engine**
  (chatterbox: 0% production, 35% probe). Production and adversarial rates are kept
  apart everywhere.

Latest clean comparison (newscaster-v1, 2026-08-02, same texts, matched refs): qwen
27/27 · zonos 29/31 (emotion truly off) · moss_vg 20/20 (confounded — the register
flatters its failure mode). Ears queue and pending tier decisions: [todo.md §4](todo.md).

## Benched engines

Not retired — all three carry usable licences. They are out of *this* game.

**Rule: a bench needs a written end-condition and a last-checked date, or it drifts.**
LongCat proved it — for three weeks it was simultaneously "excluded until the
affect-transfer experiment passes" (this file), *retired* (`ref_select.py`), and quietly
passed (the experiment ran 2026-07-18 and was never written up). Change a bench **here and
nowhere else**.

| Engine | Benched | Why — all structural, none a quality verdict | What would end it | Checked |
|---|---|---|---|---|
| **LongCat-AudioDiT-3.5B** (MIT) | 2026-08-09 | No instruction slot — clone-only, so no skill file is possible | Upstream ships a text-instruct interface, **or** we adopt it as a pure clone-multiplier (needs a *label-fidelity* test) | 2026-08-09 |
| **VibeVoice-Large** (MIT) | 2026-07-29 | ① no instruct slot (`instruct` never reaches the model) ② **sings / adds music, uncontrollably** | An answer to ②. ⚠ see trap below | 2026-08-09 |
| **Dia-1.6B-0626** (Apache-2.0) | 2026-07-29 | ① no instruct slot ② same music/flair failure ③ weakest quality: 12/20 QC, DNSMOS 2.53, hard temp floor 1.8 | Nothing identified | 2026-08-09 |

### The unrequested-flair failure is exclusive to two engines

Measured 2026-08-09 over the expressive-registers bank, from auditor notes on dropped
clips (*"a singing segment with music and multiple singers"*, *"Noise, semi-musical"*):

| Engine | Clips | Noted music/song | Share |
|---|---|---|---|
| vibevoice | 116 | 7 | **6.0%** |
| dia | 69 | 4 | **5.8%** |
| qwen · chatterbox · zonos · orpheus · moss_vg · longcat | 1,107 | **0** | **0.0%** |

A floor, not a rate — it counts clips where an auditor wrote the reason down. Cause and
upstream quotes: [tts-engine-onboarding.md § VibeVoice sings, and you cannot stop it](tts-engine-onboarding.md).
Emergent, never trained for, content-aware, and per the maker *"we can't directly control
whether they are generated or not"*.

### Two traps

**VibeVoice's "reversible" covers only one blocker.** The recorded condition — *its ceiling
was bounded by a 100%-synthetic reference pool, and the pool is real speech now* — is true,
and fixes casting quality. **A better pool does not stop it singing**, and a prompt
containing music makes it worse. Re-testing on pool grounds alone would rediscover the 6%
and call it new.

**And a third, which will arrive later: "but we WANT singing now."** Singing is
[high ambition 7](high-ambition-7-singing.md) as of 2026-08-09 — a real goal, deliberately
down the road. When it opens, these two engines will look like a head start and they are
not one. **They were benched for uncontrollability, not for inability**: per the maker,
*"we can't directly control whether they are generated or not."* A capability that cannot be
withheld is not a capability — for a teacher it is a contaminant, because every clip it
touches must then be checked for a behaviour nobody asked for. The end-condition in the
table above is unchanged by goal 7: **an answer to ②**, meaning it can be told *not* to.

**LongCat's gate passed on the wrong axis.** `transfer1` (2026-07-18) scored 51/55 keeps
(92.7%) — trusted-tier. But keep-rate measures whether the audio is *good*, not whether the
*affect transferred*, which is the entire question for a clone-multiplier with inherited
labels. Its 55 clips stay out of training corpora while benched (v6, 2026-08-09).

> **Why none of these is "the model is too narrow":** apparent narrowness is usually a
> missing director skill file — the survivors narrate at 94%, the same as VibeVoice. Each
> bench above is for something the model *is*, not something it failed to do.

**Referenced from** (these point here, never restate a status):
`scripts/synthesis/teacher_audition/README.md`, `notes/delivery-mix-campaign.md`,
`scripts/synthesis/ref_select.py`.

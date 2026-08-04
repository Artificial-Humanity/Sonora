# Teacher engines — license verdicts and measured standing

The record of which large TTS models may teach Sonora (license wall: outputs trainable
only from Apache-2.0/MIT weights run locally), which are excluded and why, and where
the live portfolio stands. The 2026-07-17 per-clip audition narrative and the
superseded portfolio proposals were cut 2026-08-02 (git history has them); what
remains here is what stays decision-relevant.

**Current portfolio: chatterbox · qwen · zonos · orpheus · moss_vg** (VibeVoice + Dia
set aside 2026-07-29, reversible). Allocation is three-layer
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
| **LongCat-AudioDiT-3.5B** | 3.5B | MIT. **Affect-transfer multiplier only** — clone from OUR labeled synthetic anchors onto new text, labels inherited, blind QC per output; standard (promptless) mode unused. Excluded from campaigns until the affect-transfer experiment passes |
| **Maya1** | 3B | Apache-2.0 verified on the weights repo. Un-auditioned; enters at onboarding step 1 if the field ever wants another description-driven engine |
| VibeVoice-Large | — | MIT, **set aside 2026-07-29**: no instruction slot (casting = reference clip only), stages scenes on dialogue; its ceiling was bounded by a 100%-synthetic reference pool, not the engine |
| Dia-1.6B-0626 | 1.6B | Apache-2.0, **set aside 2026-07-29**: no instruct slot, weakest measured quality (12/20 QC pass, DNSMOS 2.53); temp 1.8 floor is a hard constraint |

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

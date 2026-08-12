# Teacher-engine audition renders

One render script per candidate teacher TTS engine, all driven by the shared
`stress_script.json` line set, used for the engine auditions that ratified the
teacher-synthesis portfolio (2026-07-17) and the later quote-pilot benchmark passes.

> ## 📌 THIS DIRECTORY IS PROVENANCE, NOT A LANE — and it is kept on purpose (owner, 2026-08-12)
>
> Nothing invokes these scripts and nothing should. **That is not evidence they are dead**, and
> `scripts/`'s ordinary rule — superseded code is deleted, git history is the archive — does not
> reach here. Two concrete reasons:
>
> 1. **Clips rendered by these files are still rated in `ratings.csv`.** `4e87241` exists for no
>    other purpose than to preserve four of them (`render_orpheus_pt_v3d.py`,
>    `render_vibevoice_{15b,q8,v3d}.py`, `realtime_05b_study.sh`) as *"provenance for rated
>    clips"*. An ear verdict whose renderer is only in git history cannot be reproduced by
>    anyone who does not already know to go looking for it.
> 2. **The table below is the interface record, and its rows describe these files.** The
>    per-engine facts — qwen's single merged `instruct` string with no `voice_description`
>    parameter, Dia's temperature floor of 1.8, VibeVoice having no instruction slot at all —
>    are what the next engine-onboarding pass reads. Deleting the subjects leaves a maintained
>    document describing files that are gone.
>
> Issue #26 proposed deleting all 13 renderers as one-shot campaign tooling. **Declined**, and
> recorded here so the question is not re-litigated at every audit. The two shells in this
> directory carry the same note in `scripts/pipeline_manifest.py`.

> ⚠️ **These are AUDITION scripts, not the production lane.** Production renders go through
> `../synth_<engine>.py` + `synth_bank.sh`, which is where loudness normalisation, the QC gate
> and manifest registration live. The "verdict" column below is a **pointer**, not the record:
> the live portfolio and its tiers are
> [notes/teacher-tts-audition-shortlist.md](../../../notes/teacher-tts-audition-shortlist.md),
> and allocation is code (`ref_select.ENGINE_MIX_BY_LANE`). Refreshed 2026-08-06 — the previous
> version still called VibeVoice "premier" and Chatterbox/Zonos/Orpheus "not adopted," both of
> which reversed in late July.

**Live portfolio: chatterbox · qwen · zonos · orpheus · moss_vg.** Everything else is
**benched**, and the bench is recorded in exactly one place —
[teacher-tts-audition-shortlist.md § Benched engines](../../../notes/teacher-tts-audition-shortlist.md#benched-engines),
which carries each engine's end-condition and last-checked date. The rows below name
INTERFACE facts only and must not restate a status: `23c6af3` moved LongCat to that rule
and left VibeVoice and Dia restating theirs, VibeVoice still carrying the stale
one-blocker wording the commit's own message names as the trap (CL-L2).

| Script | Engine | Standing |
|---|---|---|
| `render_qwen.py` | Qwen3-TTS VoiceDesign | **LIVE — trusted tier, the portfolio's gold standard** (measured 2026-07-27). Directable via exactly ONE merged `instruct` string: `generate_voice_design(text, instruct, language)`. There is **no** `voice_description` parameter; a separate design key is silently dropped. Young/bright skew (renders younger than described); gender infidelity ~10% on pre-2026-07-26 banks. |
| `render_moss_vg.py` (prod: `synth_moss_vg.py`) | **MOSS-VoiceGenerator** (`moss_vg`) | **LIVE — scrutinized tier, 100% heard.** THE MOSS for all directed work. Reads ONE merged `instruct` string (`text` + `instruction` both required, no reference audio). Lane-shaped: Newscaster 95% / Documentary 56%. |
| `render_chatterbox.py` | Chatterbox (classic, NOT Turbo) | **LIVE — trusted-provisional.** Its 2026-07-17 rejection was **void**: all 12 clips used the built-in `conds.pt` fallback, so casting was never exercised. Re-auditioned through `revisit-v1` 2026-07-28. |
| `render_zonos.py` | Zonos-v0.1-transformer | **LIVE — normal tier since 2026-08-04.** Its rejection was also void (rendered at the default `pitch_std` 20.0, i.e. flat, with no speaker). The instability was the neutral-dominant emotion vector, not the engine: 93.5% keep with `emotion: null`. |
| `render_orpheus.py` | Orpheus-3B `-ft` | **LIVE — normal tier.** The 2026-07-23 `-pretrained` verdict was void — wrong checkpoint, and the prompt omitted `128261` + `128257`. `tara` is banned in code (room reverb). |
| `render_dia.py` | Dia-1.6B | **BENCHED** — status, evidence and the condition that would un-bench it: [teacher-tts-audition-shortlist.md § Benched engines](../../../notes/teacher-tts-audition-shortlist.md#benched-engines). Do not restate it here. *Interface facts, which the bench does not change:* `render_text` is the only surface (punctuation/capitalisation plus a closed 21-tag non-verbal set); temperature **1.8 — never lower** (1.5 collapses 7/20 clips to white noise). |
| `synth_vibevoice.py` | VibeVoice-Large (aoi-ot mirror, MIT) | **BENCHED** — status, evidence and the condition that would un-bench it: [teacher-tts-audition-shortlist.md § Benched engines](../../../notes/teacher-tts-audition-shortlist.md#benched-engines). Do not restate it here. *Interface facts, which the bench does not change:* no natural-language instruction slot at all — `design` feeds `ref_select.py` to pick a reference clip and `instruct` never reaches the model; it stages scenes on dialogue. |
| `render_moss_tts.py` / `render_moss_anchors.py` | MOSS-TTS 8.5B flagship (`moss85`) | **NOT a directed teacher** — un-SFT'd base model; its card lists no `instruction` input and it reads prompts aloud on short lines. Retained only as a possible CLONING engine via its unused `reference` slot. |
| `render_higgs.py` | Higgs TTS 3 | **NC-walled** — benchmark shelf only, never trains, never calibrates a detector. |
| `render_longcat.sh` | LongCat | **BENCHED 2026-08-09** — no instruction slot, so no skill file is possible and it stays at onboarding step 3. Status, evidence and the condition that would un-bench it: [teacher-tts-audition-shortlist.md § Benched engines](../../../notes/teacher-tts-audition-shortlist.md#benched-engines). Do not restate it here. |
| `coach_dia_threat.py` | — | Dia direction-sensitivity probe (collapse-class investigation) |

**teacher-ab-v1 (2026-07-26) — pass rates:** qwen 19/20 · vibevoice 19/20 · moss_vg 18/20 · dia 12/20.
**Median DNSMOS:** qwen 3.46 · vibevoice 3.40 · moss_vg 3.27 · dia 2.53.
⚠ These were rendered **before loudness normalisation**, across a 5.99 dB engine spread — the
qwen-vs-VibeVoice ranking in particular is confounded and the re-test is still open
([notes/todo.md §4](../../../notes/todo.md)).

**Loudness IS normalised in production, and is NOT here.** `normalize_loudness.py` (−23 LUFS) has
been wired into `synth_bank.sh` ahead of QC since 2026-07-28 and its failure is fatal. These
audition scripts predate that and write raw renders, so a ~5 dB RMS spread between engines is
expected — normalise before any A/B listening test, or the louder engine wins on volume alone.

**Standing rule:** no TTS model enters the portfolio without a studied interface (verified at the
renderer call site, not from the README) and a Gemma skill-file adapter in
`../director_skills/`. Pattern and known gotchas:
[notes/tts-engine-onboarding.md](../../../notes/tts-engine-onboarding.md). The watch-list triggers
below are subject to it.

## Provenance

Migrated into source control 2026-07-22 from `/data/toolchain/teacher-audition/` on
ai-lab-0, where the working directory (rendered `out/` audio, engine weights) remains.
Verdict ledger: the teacher-synthesis-portfolio notes and the Dataset Auditions app
ratings are the source of truth; these scripts are the reproducible render side.

## Watch list

| Model | Why waiting | Trigger to audition |
|---|---|---|
| `zai-org/GLM-TTS` (MIT, zero-shot cloning, EN supported/CN-primary) | Released weights are the PRE-RL base; the headline GRPO multi-reward emotion pass ("RL-optimized weights") is unreleased — auditioning now would score a floor on exactly our decisive axis (expressiveness). **The 8.3 GB pre-RL base IS on disk** at `/data/models/zai-org/GLM-TTS` (pulled 2026-07-23, verified 2026-08-06) — an earlier "deliberately not kept on disk" line here was wrong. Having it changes nothing: the trigger is the checkpoint, not the download. | RL-optimized checkpoint lands on the HF repo (watch its "News": github.com/zai-org/GLM-TTS). **Long-dated** — the repo has not moved since 12 Jan 2026. Re-audition = the standing quote-pilot bank (v3d protocol: same 10 lines + audited reference clips); the pre/post-RL delta is itself useful evidence for a future Sonora expressiveness-RL lane. |
| `moonshotai/Kimi-Audio-7B-Instruct` (MIT) | Not an actor (no casting interface — audio-LM). Shelved as candidate SER/valence instrument for the LibriVox lane. | Quote-mining lane reaches scale labeling and wants a fourth agreement vote. |
| *(concept)* Quantized variant of any adopted heavy engine (e.g. VibeVoice-Large Q8-class quants) | Bulk-render throughput option — NOT for auditions (quantization confounds engine verdicts; audition at full precision). Specific repo TBD at need, not pinned. | An adopted engine's render throughput becomes the campaign bottleneck; evaluate quant vs bf16 with the bf16 renders as reference standard. |

# Teacher-engine audition renders

One render script per candidate teacher TTS engine, all driven by the shared
`stress_script.json` line set, used for the engine auditions that ratified the
teacher-synthesis portfolio (2026-07-17) and the later quote-pilot benchmark passes.

| Script | Engine | Portfolio verdict |
|---|---|---|
| `render_dia.py` | Dia-1.6B | in portfolio — **markedly expressive, NOT neutral-only**. Consumes only `render_text`, but that is a real control surface: punctuation/capitalisation (undocumented) plus a **closed 21-tag** non-verbal set, which is its one director-drivable channel. Temperature **1.8 — never lower** (1.5 collapses 7/20 clips to white noise). Weakest measured quality of the four: 12/20 pass, median DNSMOS 2.53. |
| `render_qwen.py` | Qwen3-TTS VoiceDesign | in portfolio — directable via exactly ONE merged `instruct` string: `generate_voice_design(text, instruct, language)`. There is **no** `voice_description` parameter; a separate design key is silently dropped. Young/bright skew (renders younger than described); gender infidelity ~10% on pre-2026-07-26 banks. Best measured quality: 19/20 pass, median DNSMOS 3.46. |
| `render_moss_vg.py` (prod: `synth_moss_vg.py`) | **MOSS-VoiceGenerator** (`moss_vg`) | in portfolio — **THE MOSS for all directed work.** Reads ONE merged `instruct` string (`text` + `instruction` both required, no reference audio); dark/force/oratory and situational framing. 18/20 pass, median DNSMOS 3.27. |
| `render_moss_tts.py` / `render_moss_anchors.py` | MOSS-TTS 8.5B flagship (`moss85`) | **NOT a directed teacher** — un-SFT'd PRE-TRAINED BASE model; its card lists no `instruction` input and it reads prompts aloud on short lines. Retained only as a possible CLONING engine via its unused `reference` slot. |
| `render_higgs.py` | Higgs TTS 3 | NC-walled — benchmark shelf only, never train/calibrate |
| `render_chatterbox.py`, `render_orpheus.py`, `render_zonos.py` | Chatterbox / Orpheus / Zonos | auditioned, not adopted |
| `render_longcat.sh` | LongCat | transfer stage of the portfolio |
| `coach_dia_threat.py` | — | Dia direction-sensitivity probe (collapse-class investigation) |
| *(v3d, scripts in scratchpad-era)* | Orpheus-3B-pretrained | auditioned 2026-07-23, NOT adopted — top-tier peak quality but 4/10 net yield (systematic off-script drift, double-fail on re-roll) |
| `synth_vibevoice.py` | VibeVoice-Large (aoi-ot mirror, MIT) | **ADOPTED — premier casting engine.** No natural-language instruction slot at all: `design` feeds `ref_select.py` (gender word + age band) to pick a reference clip; `instruct` is audit-only and never reaches the model. **Reference pool is 100% OWN-SYNTHESIS — 193 clips, no real speech — so casting quality is bounded by the pool, not the engine.** 19/20 pass, median DNSMOS 3.40. |

**teacher-ab-v1 (2026-07-26) — pass rates:** qwen 19/20 · vibevoice 19/20 · moss_vg 18/20 · dia 12/20.
**Median DNSMOS:** qwen 3.46 · vibevoice 3.40 · moss_vg 3.27 · dia 2.53.

**Loudness is NOT normalised across engines** — a ~5 dB RMS spread between engines is expected in
raw renders (nothing is clipping; verified zero flat-topping). Normalise before any A/B listening
test, or the louder engine wins on volume alone.

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
| `zai-org/GLM-TTS` (MIT, zero-shot cloning, EN supported/CN-primary) | Released weights are the PRE-RL base; the headline GRPO multi-reward emotion pass ("RL-optimized weights") is unreleased — auditioning now would score a floor on exactly our decisive axis (expressiveness). Weights deliberately not kept on disk. | RL-optimized checkpoint lands on the HF repo (watch its "News": github.com/zai-org/GLM-TTS). Re-audition = the standing quote-pilot bank (v3d protocol: same 10 lines + audited reference clips); the pre/post-RL delta is itself useful evidence for a future Sonora expressiveness-RL lane. |
| `moonshotai/Kimi-Audio-7B-Instruct` (MIT) | Not an actor (no casting interface — audio-LM). Shelved as candidate SER/valence instrument for the LibriVox lane. | Quote-mining lane reaches scale labeling and wants a fourth agreement vote. |
| *(concept)* Quantized variant of any adopted heavy engine (e.g. VibeVoice-Large Q8-class quants) | Bulk-render throughput option — NOT for auditions (quantization confounds engine verdicts; audition at full precision). Specific repo TBD at need, not pinned. | An adopted engine's render throughput becomes the campaign bottleneck; evaluate quant vs bf16 with the bf16 renders as reference standard. |

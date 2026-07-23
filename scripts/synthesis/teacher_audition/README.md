# Teacher-engine audition renders

One render script per candidate teacher TTS engine, all driven by the shared
`stress_script.json` line set, used for the engine auditions that ratified the
teacher-synthesis portfolio (2026-07-17) and the later quote-pilot benchmark passes.

| Script | Engine | Portfolio verdict |
|---|---|---|
| `render_dia.py` | Dia-1.6B | in portfolio — neutral/long narration only (undirectable; consumes only `render_text`) |
| `render_qwen.py` | Qwen3-TTS VoiceDesign | in portfolio — directable casting, young/bright skew, ~10% gender infidelity |
| `render_moss_tts.py` / `render_moss_vg.py` / `render_moss_anchors.py` | MOSS-TTSD 8.5B | in portfolio — obeys casting; dark/oratory charter |
| `render_higgs.py` | Higgs TTS 3 | NC-walled — benchmark shelf only, never train/calibrate |
| `render_chatterbox.py`, `render_orpheus.py`, `render_zonos.py` | Chatterbox / Orpheus / Zonos | auditioned, not adopted |
| `render_longcat.sh` | LongCat | transfer stage of the portfolio |
| `coach_dia_threat.py` | — | Dia direction-sensitivity probe (collapse-class investigation) |

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

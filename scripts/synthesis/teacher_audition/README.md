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

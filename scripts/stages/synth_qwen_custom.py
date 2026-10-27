"""Stage-1 renderer: Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache-2.0).

Reads the script bank, renders every engine=="qwen_custom" line, writes wavs +
qwen_custom_manifest.jsonl (full provenance per clip) to --out.

Interface (CODE-VERIFIED against qwen-tts 0.1.1, the pinned wheel):
`generate_custom_voice(text, speaker, language, instruct)` — a speaker from the
checkpoint's closed nine-entry `spk_id` table (checked case-insensitively by the
shipped code, which raises on an unlisted name) plus ONE short delivery clause.
There is no reference audio and no voice description: the preset IS the voice.
VoiceDesign's 12-key instruct format is UNVERIFIED on this checkpoint and is
never sent (director_skills/qwen_custom.md). Only ryan and aiden are
English-native presets; every female preset speaks English as a non-native
language, which Step 1c-ii records as a limit.

The speaker guard below duplicates book_ingest's closed set deliberately (the
orpheus precedent): this renderer must be safe against a hand-authored bank,
which is exactly the path that bypasses build_direction's validation. An
unlisted speaker skips its clip rather than spending the generate call to learn
what the shipped check would raise.

No `seed` argument exists; `torch.manual_seed(job["seed"])` sits immediately
before the generate call, at batch size 1. The checkpoint loads through
`qwen_load.load_qwen`, so the float32 waveform decoder refusal applies here too.

Container recipe: synth_bank.sh's qwen_custom block (the qwen block's recipe:
apt sox; --no-deps qwen-tts==0.1.1; transformers==4.57.3, soundfile, sox,
onnxruntime, einops, librosa; --no-deps accelerate==1.12.0).
"""
import argparse
import json
import os
import time

import soundfile as sf  # noqa: F401  (container import check, as in synth_qwen)
import torch

import sys as _sys  # noqa: E402
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from qwen_load import TOKENIZER_DTYPE, load_qwen  # noqa: E402
from synth_common import render_timing, write_wav_atomic  # noqa: E402

MODEL_DIR = "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# The closed speaker set, duplicated from book_ingest deliberately: this renderer
# must be safe when run standalone against a hand-authored bank, which is exactly
# the path that bypasses the director's validation. See
# director_skills/qwen_custom.md. Gender feeds the manifest's intended_gender,
# which the audition prefill keys on ("Female"/"Male" spelled out).
FEMALE_SPEAKERS = {"vivian", "serena", "ono_anna", "sohee"}
MALE_SPEAKERS = {"ryan", "aiden", "dylan", "eric", "uncle_fu"}
SPEAKERS = FEMALE_SPEAKERS | MALE_SPEAKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "qwen_custom"]
    # Return BEFORE the shard load when the bank holds no qwen_custom lines
    # (synth_qwen's B-L3 lesson: an empty job list must not spend a model load).
    if not jobs:
        print("no qwen_custom jobs in bank")
        return
    model = load_qwen(MODEL_DIR)
    print(f"loaded ({len(jobs)} jobs)", flush=True)

    manifest_path = os.path.join(args.out, "qwen_custom_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            d = job["direction"]
            # Renderer-side guard (see the speaker-set note above). The shipped
            # check is case-insensitive, so normalise the same way; the payload
            # and the call carry the canonical lowercase spelling.
            speaker = str(d.get("speaker") or "").strip().lower()
            if speaker not in SPEAKERS:
                print(job["id"], f"SKIP (unknown speaker {d.get('speaker')!r}: "
                                 f"not one of the closed nine)", flush=True)
                continue
            torch.manual_seed(job["seed"])
            t1 = time.monotonic()
            wavs, sr = model.generate_custom_voice(
                text=job["text"], speaker=speaker, language="English",
                instruct=d.get("instruct") or None)
            torch.cuda.synchronize()
            render_sec = time.monotonic() - t1
            wav = wavs[0]
            write_wav_atomic(os.path.join(args.out, f"{job['id']}.wav"), wav, sr)

            row = dict(job)
            row.update({
                "engine": "qwen_custom", "wav": f"{job['id']}.wav", "sr": sr,
                "engine_license": "Apache-2.0 (Qwen3-TTS-12Hz-1.7B-CustomVoice)",
                "speech_tokenizer_dtype": TOKENIZER_DTYPE,
                "bank_version": bank["version"], "campaign": bank["campaign"],
                "intended_gender": "Female" if speaker in FEMALE_SPEAKERS else "Male",
                # The canonical spelling that actually ran, which a bank's
                # mixed-case emission would otherwise misdescribe.
                "applied_speaker": speaker,
                **render_timing(0.0, render_sec, len(wav) / sr),
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/sr:.1f}s speaker={speaker}", flush=True)
    print("SYNTH-QWEN-CUSTOM-DONE")


if __name__ == "__main__":
    main()

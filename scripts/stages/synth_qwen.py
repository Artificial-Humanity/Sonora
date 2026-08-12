"""Stage-1 renderer: Qwen3-TTS VoiceDesign (Apache-2.0).

Reads the script bank, renders every engine=="qwen" line, writes wavs +
qwen_manifest.jsonl (full provenance per clip) to --out.

Container recipe (audition-proven): apt sox; pip --no-deps qwen-tts;
transformers==4.57.3, soundfile, sox, onnxruntime, einops; --no-deps accelerate==1.12.0.
"""
import argparse
import json
import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

import sys as _sys
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from synth_common import write_wav_atomic  # noqa: E402

MODEL_DIR = "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "qwen"]
    # B-L3. The model was loaded BEFORE this check, so a bank with no qwen lines spent
    # the full shard load and VRAM to then iterate an empty list. Every other renderer
    # already returns here; qwen is the one that did not.
    if not jobs:
        print("no qwen jobs in bank")
        return
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0",
                                          dtype=torch.bfloat16)
    gen = getattr(model, "generate_voice_design", None) or model.generate_custom_voice

    manifest_path = os.path.join(args.out, "qwen_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            torch.manual_seed(job["seed"])
            # `generate_voice_design(text, instruct, language)` — there is no
            # `voice_description` parameter in any published qwen-tts release.
            # The old call passed one; it landed in **kwargs and was silently
            # discarded (no TypeError, so the fallback never fired), meaning the
            # voice design never reached the model. build_direction() now merges
            # design + instruct into this single string. (owner finding 2026-07-25)
            wavs, sr = gen(text=job["text"], language="English",
                           instruct=job["direction"]["instruct"])
            name = f"{job['id']}.wav"
            write_wav_atomic(os.path.join(args.out, name), wavs[0], sr)
            # dict(job) first: passthrough fields (pair_key, probe, and now
            # intended_delivery/book for the delivery-mix campaigns) must reach
            # the manifest — register_audition prefill and the fold read them
            # from here (2026-07-30; the explicit-keys version silently dropped
            # every bank field it hadn't been taught about).
            row = dict(job)
            row.update({
                "engine": "qwen", "wav": name, "sr": sr,
                "engine_license": "Apache-2.0 (Qwen3-TTS-VoiceDesign)",
                "bank_version": bank["version"], "campaign": bank["campaign"],
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wavs[0])/sr:.1f}s", flush=True)
    print("SYNTH-QWEN-DONE")


if __name__ == "__main__":
    main()

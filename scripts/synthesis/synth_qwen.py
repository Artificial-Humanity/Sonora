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

MODEL_DIR = "/data/reference/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "qwen"]
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
            kwargs = dict(text=job["text"], language="English",
                          instruct=job["direction"]["instruct"])
            try:
                wavs, sr = gen(voice_description=job["direction"]["design"], **kwargs)
            except TypeError:
                wavs, sr = gen(speaker=job["direction"]["design"], **kwargs)
            name = f"{job['id']}.wav"
            sf.write(os.path.join(args.out, name), wavs[0], sr)
            mf.write(json.dumps({
                "id": job["id"], "engine": "qwen", "wav": name,
                "register": job["register"], "intended": job["intended"],
                "text": job["text"], "direction": job["direction"],
                "seed": job["seed"], "sr": sr,
                "engine_license": "Apache-2.0 (Qwen3-TTS-VoiceDesign)",
                "bank_version": bank["version"], "campaign": bank["campaign"],
            }) + "\n")
            print(job["id"], f"{len(wavs[0])/sr:.1f}s", flush=True)
    print("SYNTH-QWEN-DONE")


if __name__ == "__main__":
    main()

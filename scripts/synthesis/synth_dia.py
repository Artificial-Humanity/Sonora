"""Stage-1 renderer: Dia-1.6B-0626 (Apache-2.0), transformers-native.

Engine floors (owner-audited coaching experiment 2026-07-17): temperature
>= 1.3 (below collapses to white noise), guidance_scale 3.0 (higher bends
register toward the text's literal semantics). Single-speaker lines only in
production; scene staging reserved until a segmentation step exists. Dia
improvises tails — the QC gate's duration-vs-text check is the catch net.

Container recipe: pip transformers soundfile.
"""
import argparse
import json
import os

import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

MODEL_DIR = "/data/reference/models/nari-labs/Dia-1.6B-0626"
TEMP_FLOOR = 1.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "dia"]

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = DiaForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda")

    manifest_path = os.path.join(args.out, "dia_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for job in jobs:
            temp = max(job["direction"].get("temperature", 1.4), TEMP_FLOOR)
            torch.manual_seed(job["seed"])
            inputs = processor(text=[job["direction"]["render_text"]],
                               padding=True, return_tensors="pt").to("cuda")
            audio_tokens = model.generate(
                **inputs, max_new_tokens=1536,
                guidance_scale=job["direction"].get("guidance", 3.0),
                temperature=temp, top_p=0.90, top_k=45)
            decoded = processor.batch_decode(audio_tokens)
            name = f"{job['id']}.wav"
            processor.save_audio(decoded, os.path.join(args.out, name))
            mf.write(json.dumps({
                "id": job["id"], "engine": "dia", "wav": name,
                "register": job["register"], "intended": job["intended"],
                "text": job["text"], "direction": {**job["direction"], "temperature": temp},
                "seed": job["seed"], "sr": 44100,
                "engine_license": "Apache-2.0 (Dia-1.6B-0626)",
                "bank_version": bank["version"], "campaign": bank["campaign"],
            }) + "\n")
            print(job["id"], "done", flush=True)
    print("SYNTH-DIA-DONE")


if __name__ == "__main__":
    main()

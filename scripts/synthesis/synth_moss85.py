"""Stage-1 renderer: MOSS-TTS 8.5B flagship (Apache-2.0).

Anchored instructions per line (the 2.1B recipe template). Reminder: the
flagship drifts from anchors (owner-observed gender flip / accent drift) —
labels are verified downstream by instrument, never trusted from instruct.

Container recipe: pip transformers soundfile; pip --no-deps accelerate.
"""
import argparse
import json
import os

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL_DIR = "/data/reference/models/OpenMOSS-Team/MOSS-TTS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "moss85"]

    device = "cuda"
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16).to(device).eval()
    sr = processor.model_config.sampling_rate

    manifest_path = os.path.join(args.out, "moss85_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf, torch.no_grad():
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            torch.manual_seed(job["seed"])
            msg = processor.build_user_message(text=job["text"],
                                               instruction=job["direction"]["instruct"],
                                               quality=job["direction"].get("quality"))
            batch = processor([[msg]], mode="generation")
            outputs = model.generate(input_ids=batch["input_ids"].to(device),
                                     attention_mask=batch["attention_mask"].to(device),
                                     max_new_tokens=4096)
            for message in processor.decode(outputs):
                audio = message.audio_codes_list[0].float().cpu().numpy()
                name = f"{job['id']}.wav"
                sf.write(os.path.join(args.out, name), audio, sr)
                mf.write(json.dumps({
                    "id": job["id"], "engine": "moss85", "wav": name,
                    "register": job["register"], "intended": job["intended"],
                    "text": job["text"], "direction": job["direction"],
                    "seed": job["seed"], "sr": sr,
                    "engine_license": "Apache-2.0 (MOSS-TTS 8.5B)",
                    "bank_version": bank["version"], "campaign": bank["campaign"],
                }) + "\n")
                print(job["id"], f"{len(audio)/sr:.1f}s", flush=True)
    print("SYNTH-MOSS85-DONE")


if __name__ == "__main__":
    main()

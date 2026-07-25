#!/usr/bin/env python3
"""Stage-1 renderer: MOSS-VoiceGenerator (Apache-2.0), bank-consuming.

Replaces the moss85 flagship for any DIRECTED work. The 8.5B flagship is an
un-SFT'd pre-trained foundation model — its card does not list `instruction`
among its input types, and OpenMOSS maintainers state instruction control lives
in this sibling instead. That mismatch is the source of the prompt-read-aloud
leakage the owner found (the base model continues its prompt; when the text is
short next to the instruction it vocalises the instruction). See the owner
finding 2026-07-25.

Interface (model card §Input Types): `text` and `instruction` are BOTH required.
No reference audio — timbre comes from the instruction text alone. The card warns
the model is "sensitive to decoding hyperparameters"; the documented defaults for
this checkpoint are used below and should not be changed casually.

Container recipe: pip transformers soundfile.

Usage:
    python scripts/synthesis/synth_moss_vg.py --bank <bank.json> --out <dir>
"""
import argparse
import json
import os

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL_DIR = "/data/models/OpenMOSS-Team/MOSS-VoiceGenerator"

# Model-card defaults for THIS checkpoint (README §1.3). The flagship's defaults
# differ sharply (1.7/0.8/25/1.0) — do not cross-apply them.
AUDIO_TEMPERATURE = 1.5
AUDIO_TOP_P = 0.6
AUDIO_TOP_K = 50
AUDIO_REPETITION_PENALTY = 1.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "moss_vg"]
    print(f"{len(jobs)} moss_vg jobs -> {args.out}", flush=True)
    if not jobs:
        print("SYNTH-MOSS-VG-DONE (nothing to do)")
        return

    device = "cuda"
    # normalize_inputs=True enables the card's normalize_instruction(): it flattens
    # newlines and DELETES anything inside [...] or {...}, so never put bracketed
    # stage directions in an instruct string for this model.
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                              normalize_inputs=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16).to(device).eval()
    sr = processor.model_config.sampling_rate

    manifest_path = os.path.join(args.out, "moss_vg_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf, torch.no_grad():
        for job in jobs:
            name = f"{job['id']}.wav"
            if os.path.exists(os.path.join(args.out, name)):
                print(job["id"], "exists, skip", flush=True)
                continue
            torch.manual_seed(job["seed"])
            msg = processor.build_user_message(
                text=job["text"], instruction=job["direction"]["instruct"])
            batch = processor([[msg]], mode="generation")
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                audio_temperature=AUDIO_TEMPERATURE,
                audio_top_p=AUDIO_TOP_P,
                audio_top_k=AUDIO_TOP_K,
                audio_repetition_penalty=AUDIO_REPETITION_PENALTY,
            )
            for message in processor.decode(outputs):
                audio = message.audio_codes_list[0].float().cpu().numpy()
                sf.write(os.path.join(args.out, name), audio, sr)
                mf.write(json.dumps({
                    "id": job["id"], "engine": "moss_vg", "wav": name,
                    "register": job["register"], "intended": job["intended"],
                    "text": job["text"], "direction": job["direction"],
                    "seed": job["seed"], "sr": sr,
                    "engine_license": "Apache-2.0 (MOSS-VoiceGenerator)",
                    "decoding": {"audio_temperature": AUDIO_TEMPERATURE,
                                 "audio_top_p": AUDIO_TOP_P,
                                 "audio_top_k": AUDIO_TOP_K,
                                 "audio_repetition_penalty": AUDIO_REPETITION_PENALTY},
                    "bank_version": bank["version"], "campaign": bank["campaign"],
                    "pair_key": job.get("pair_key"), "probe": job.get("probe"),
                }) + "\n")
                mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
                print(job["id"], f"{len(audio)/sr:.1f}s", flush=True)
    print("SYNTH-MOSS-VG-DONE")


if __name__ == "__main__":
    main()

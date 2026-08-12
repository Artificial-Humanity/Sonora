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
from synth_common import write_wav_atomic  # noqa: E402

MODEL_DIR = "/data/models/OpenMOSS-Team/MOSS-TTS"


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
            # B-L6, the same clobber as moss_vg: this looped over EVERY decoded message
            # writing the same `name`, so a multi-message decode overwrote the wav once
            # per message and wrote one manifest row per message, all claiming one id.
            decoded = list(processor.decode(outputs))
            if len(decoded) > 1:
                print(job["id"], f"WARN decode returned {len(decoded)} messages; "
                                 "keeping the first", flush=True)
            for message in decoded[:1]:
                audio = message.audio_codes_list[0].float().cpu().numpy()
                name = f"{job['id']}.wav"
                write_wav_atomic(os.path.join(args.out, name), audio, sr)
                # B-M6: `dict(job)` first, so passthrough fields (intended_delivery, book,
                # ref_id, source_ref …) reach the manifest. Naming keys explicitly dropped
                # them, and register_audition's prefill and the corpus fold read them from
                # here. The other six renderers were fixed; dia and moss85 were missed.
                row = dict(job)
                row.update({
                    "engine": "moss85", "wav": name, "sr": sr,
                    "engine_license": "Apache-2.0 (MOSS-TTS 8.5B)",
                    "bank_version": bank["version"], "campaign": bank["campaign"],
                })
                mf.write(json.dumps(row) + "\n")
                mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
                print(job["id"], f"{len(audio)/sr:.1f}s", flush=True)
    print("SYNTH-MOSS85-DONE")


if __name__ == "__main__":
    main()

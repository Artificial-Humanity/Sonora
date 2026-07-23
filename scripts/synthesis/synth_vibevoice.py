"""Stage-1 renderer: VibeVoice-Large-Q8 — the PREMIER engine (owner hierarchy
2026-07-23). MIT weights (FabioSarracino selective 8-bit of the aoi-ot mirror;
official MS repo delisted — bf16 source kept at /data/models/aoi-ot).

Reads the script bank, renders every engine=="vibevoice" line, writes wavs +
vibevoice_manifest.jsonl to --out. Casting = reference cloning: the direction's
`design` selects an audited keep via ref_select.py; the reference's voice
carries gender/age/timbre (v3d-proven: 100% gender fidelity).

Container recipe (v3d/v3e-proven): uv pip install
'git+https://github.com/vibevoice-community/VibeVoice.git' soundfile
bitsandbytes accelerate   # community fork — MS repo restructured away the TTS module
NOTE: bnb 8-bit shard load ≈7.5 min on gfx1151 — amortize over big banks.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_select import select_reference  # noqa: E402

MODEL_DIR = "/data/models/FabioSarracino/VibeVoice-Large-Q8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "vibevoice"]
    if not jobs:
        print("no vibevoice jobs in bank"); return

    processor = VibeVoiceProcessor.from_pretrained(MODEL_DIR)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_DIR, device_map="cuda", attn_implementation="sdpa").eval()
    model.set_ddpm_inference_steps(num_steps=10)
    print("loaded", flush=True)

    used = set()
    manifest_path = os.path.join(args.out, "vibevoice_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            torch.manual_seed(job["seed"])
            ref_wav, ref_text, ref_meta = select_reference(
                job["direction"].get("design", ""), job.get("intended", {}), used)
            inputs = processor(text=[f"Speaker 0: {job['text']}"],
                               voice_samples=[[ref_wav]], padding=True,
                               return_tensors="pt", return_attention_mask=True)
            inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=None, cfg_scale=1.3,
                                     tokenizer=processor.tokenizer,
                                     generation_config={"do_sample": False}, verbose=False)
            wav = out.speech_outputs[0].squeeze().float().cpu().numpy()
            if wav is None or len(wav) < 24000:
                print(job["id"], "FAILED (short/empty)", flush=True)
                continue
            sf.write(os.path.join(args.out, f"{job['id']}.wav"), wav, 24000)
            row = dict(job)
            row.update({"engine_license": "mit",
                        "weights_source": "FabioSarracino/VibeVoice-Large-Q8 (8-bit of aoi-ot mirror)",
                        "sr": 24000, "wav": f"{job['id']}.wav",
                        "ref": ref_meta | {"ref_text": ref_text}})
            mf.write(json.dumps(row) + "\n")
            print(job["id"], f"{len(wav)/24000:.1f}s ref={ref_meta['id']}", flush=True)
    print("VIBEVOICE-RENDERS-DONE", flush=True)


if __name__ == "__main__":
    main()

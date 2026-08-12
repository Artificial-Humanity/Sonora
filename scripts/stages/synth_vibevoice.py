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
from synth_common import rebuild_used_set, write_wav_atomic  # noqa: E402
from ref_select import (  # noqa: E402
    MAX_REF_EXCURSION,
    REF_BLACKLIST,
    design_age_band,
    pinned_reference,
    select_reference,
)

MODEL_DIR = "/data/models/FabioSarracino/VibeVoice-Large-Q8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, nargs="+",
                    help="one or more script_bank.json paths; wavs land in <bankdir>/audio "
                         "unless a single --out is given with a single bank")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out and len(args.bank) > 1:
        raise SystemExit("--out only valid with a single --bank")

    processor = VibeVoiceProcessor.from_pretrained(MODEL_DIR)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_DIR, device_map="cuda", attn_implementation="sdpa").eval()
    model.set_ddpm_inference_steps(num_steps=10)
    print("loaded", flush=True)

    for bank_path in args.bank:
        bank = json.load(open(bank_path, encoding="utf-8"))
        jobs = [l for l in bank["lines"] if l["engine"] == "vibevoice"]
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(bank_path)), "audio")
        os.makedirs(out, exist_ok=True)
        print(f"== bank {bank.get('campaign','?')}: {len(jobs)} vibevoice jobs -> {out}", flush=True)
        if not jobs:
            continue
        render_bank(jobs, out, processor, model, bank)
    print("VIBEVOICE-RENDERS-DONE", flush=True)


def render_bank(jobs, out, processor, model, bank):
    manifest_path = os.path.join(out, "vibevoice_manifest.jsonl")
    # B-M3: the variety-bias `used` set started EMPTY on every resume, because a job whose
    # wav already exists skips before select_reference() runs. A rerolled clip therefore
    # cast as if the reference pool were untouched — so a resumed campaign quietly loses the
    # diversity guarantee, and the casting is not reproducible from the bank either.
    # Rebuilt from the manifest rows that already exist (synth_common, shared with the
    # renderers that already did this).
    used = rebuild_used_set(manifest_path)
    if used:
        print(f"resume: {len(used)} reference(s) already spent", flush=True)
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            torch.manual_seed(job["seed"])
            # B-L4. This called `select_reference(design, intended, used)` and nothing
            # else — no `max_excursion`, no `exclude=REF_BLACKLIST`, no `ref_id` pinning.
            # Both guards encode MEASURED findings (the excursion ceiling is the Chatterbox
            # split-artifact result; the blacklist is per-clip evidence), and this renderer
            # opted out of both by forking the call. VibeVoice is SET_ASIDE, so nothing is
            # rendering through here today — which is exactly why it is a reinstatement
            # trap, the same class as B-M6/M7.
            try:
                if job.get("ref_id"):
                    ref_wav, ref_text, ref_meta = pinned_reference(job["ref_id"])
                else:
                    ref_wav, ref_text, ref_meta = select_reference(
                        job["direction"].get("design", ""), job.get("intended", {}), used,
                        max_excursion=MAX_REF_EXCURSION, exclude=REF_BLACKLIST)
            except (LookupError, ValueError) as exc:
                print(job["id"], f"CASTING FAILED ({type(exc).__name__}: {exc}) "
                                 "— continuing", flush=True)
                continue
            inputs = processor(text=[f"Speaker 0: {job['text']}"],
                               voice_samples=[[ref_wav]], padding=True,
                               return_tensors="pt", return_attention_mask=True)
            inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=None, cfg_scale=1.3,
                                     tokenizer=processor.tokenizer,
                                     generation_config={"do_sample": False}, verbose=False)
            wav = gen.speech_outputs[0].squeeze().float().cpu().numpy()
            if wav is None or len(wav) < 24000:
                print(job["id"], "FAILED (short/empty)", flush=True)
                continue
            write_wav_atomic(os.path.join(out, f"{job['id']}.wav"), wav, 24000)
            row = dict(job)
            row.update({"engine_license": "mit",
                        # the bank LINE carries no campaign (it lives at bank
                        # top level); without this, register_audition falls back
                        # to "book-<slug>" and splits the campaign in two.
                        "campaign": bank["campaign"],
                        "bank_version": bank["version"],
                        "intended_age": design_age_band(job["direction"].get("design", "")),
                        "intended_gender": ref_meta["gender"],
                        "weights_source": "FabioSarracino/VibeVoice-Large-Q8 (8-bit of aoi-ot mirror)",
                        "sr": 24000, "wav": f"{job['id']}.wav",
                        "ref": ref_meta | {"ref_text": ref_text}})
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/24000:.1f}s ref={ref_meta['id']}", flush=True)


if __name__ == "__main__":
    main()

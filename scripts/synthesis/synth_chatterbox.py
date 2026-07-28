"""Stage-1 renderer: Chatterbox (Resemble, classic English 0.5B, MIT).

Reads the script bank, renders every engine=="chatterbox" line, writes wavs +
chatterbox_manifest.jsonl to --out.

CLASSIC, NOT TURBO — `ChatterboxTTS`, loaded from the local English weights
(t3_cfg/ve/s3gen/tokenizer/conds). Turbo's `generate()` accepts `exaggeration`
and discards it with a log warning, so switching this import silently removes
the only control channel the engine has. Do not.

Two controls, and they are one control with two knobs: `exaggeration` is a rate
profile (low = slower/deadpan, high = faster/dramatic), and raising it without
dropping `cfg_weight` reads rushed. build_direction() always emits both.

Casting: `audio_prompt_path`, a reference clip chosen by ref_select from the
certified pool. The 2026-07-17 audition passed NONE, so all 12 clips used the
built-in conds.pt fallback voice and casting was never exercised — that is the
single biggest difference between this renderer and that void verdict. Only the
first 6 s of a reference reach the speaker prompt (10 s the vocoder), and the
pool window is 4-10 s, so references are used whole or truncated, never averaged.

Publication: every output is Perth-watermarked unconditionally and the native
component is unavailable on this box, so the watermarker is patched to a no-op
below. Owner decision 2026-07-28: strip it, TRAIN on these clips, never publish
them. publish_tier.py sets publish=false for engine=="chatterbox" and fails if
one is ever marked publishable.

Container recipe (smoke-proven 2026-07-28, the first ever recorded for this
engine): pip --no-deps chatterbox-tts; then librosa s3tokenizer diffusers
resemble-perth conformer transformers safetensors soundfile **omegaconf**.
omegaconf is the one --no-deps drops that actually breaks the import chain
(chatterbox/models/s3gen/flow.py), and it fails at import, not at render.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch

import perth


class _NoopWatermarker:
    """perth's native component is unavailable here; see the publication note above."""

    def apply_watermark(self, wav, sample_rate=None, **kw):
        return wav


perth.PerthImplicitWatermarker = _NoopWatermarker

from chatterbox.tts import ChatterboxTTS  # noqa: E402  (must follow the perth patch)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_select import select_reference, design_age_band  # noqa: E402

MODEL_DIR = "/data/models/ResembleAI/chatterbox"

# Below ~20 characters Chatterbox hallucinates, and no combination of
# cfg/exaggeration/temperature/seed fixes it (skill file, "Do not"). The 4 s
# owner floor already gates most of these; this catches the rest rather than
# rendering a clip that will be dropped in audit.
MIN_CHARS = 20
MAX_CHARS = 300     # beyond this, chunk — there is a hard 1000-token / ~40 s ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "chatterbox"]
    if not jobs:
        print("no chatterbox jobs in bank")
        return
    model = ChatterboxTTS.from_local(MODEL_DIR, device="cuda")
    print(f"loaded ({len(jobs)} jobs)", flush=True)

    used = set()
    manifest_path = os.path.join(args.out, "chatterbox_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            text = job["text"]
            if len(text) < MIN_CHARS:
                print(job["id"], f"SKIP ({len(text)} chars < {MIN_CHARS}: hallucinates)", flush=True)
                continue
            if len(text) > MAX_CHARS:
                print(job["id"], f"WARN {len(text)} chars > {MAX_CHARS} (40 s ceiling)", flush=True)

            d = job["direction"]
            ref_wav, ref_text, ref_meta = select_reference(
                d.get("design", ""), job.get("intended", {}), used)
            torch.manual_seed(job["seed"])
            wav = model.generate(text, audio_prompt_path=ref_wav,
                                 exaggeration=d["exaggeration"],
                                 cfg_weight=d["cfg_weight"])
            wav = wav.squeeze(0).cpu().numpy()
            sf.write(os.path.join(args.out, f"{job['id']}.wav"), wav, model.sr)

            row = dict(job)
            row.update({
                "wav": f"{job['id']}.wav", "sr": model.sr,
                "engine_license": "MIT (ResembleAI/chatterbox, classic English)",
                "weights_source": "ResembleAI/chatterbox (t3_cfg + ve + s3gen)",
                "watermark": "perth stripped (no-op patch) — TRAIN ONLY, never publish",
                "campaign": bank["campaign"], "bank_version": bank["version"],
                "intended_age": design_age_band(d.get("design", "")),
                "intended_gender": ref_meta["gender"],
                "ref": ref_meta | {"ref_text": ref_text},
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/model.sr:.1f}s exag={d['exaggeration']} "
                             f"cfg={d['cfg_weight']} ref={ref_meta['id']}", flush=True)
    print("SYNTH-CHATTERBOX-DONE")


if __name__ == "__main__":
    main()

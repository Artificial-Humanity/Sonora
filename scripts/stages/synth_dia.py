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
from synth_common import save_via_atomic  # noqa: E402

MODEL_DIR = "/data/models/nari-labs/Dia-1.6B-0626"
# Pilot owner-audit 2026-07-17: 2/5 collapses at temp 1.3-1.4 (white noise /
# wordless output). The audition's good renders used 1.8. 1.3 is the cliff,
# not a floor — register control belongs to text/staging, not temperature.
# 1.5 is NOT a safe floor: at 1.5, 7/20 clips collapsed to noise (2026-07-26).
# 1.8 is the validated operating point; treat this as a hard floor, not a target.
TEMP_FLOOR = 1.8
TOKENS_PER_SEC = 86            # Dia audio-token frame rate
CHARS_PER_SEC = 14.0           # mid-rate English speech estimate


HEADROOM = 1.35                # was 1.8; 1.35 leaves room for Dia's real ~11.6 ch/s rate
GRACE_SEC = 1.0                # was 2.0
TAG_SEC = 1.2                  # allowance per non-verbal tag


def token_budget(text):
    """Bound Dia's generation to what the script actually needs.

    Pilot QC finding (2026-07-17): bare lines run to whatever cap they get,
    improvising tails that fail the duration gate AND drag whole-clip DNSMOS down.
    That is still true — and the 2026-07-26 audit showed the old 1.8x + 2 s budget
    was itself the cap being run to. Observed durations matched the formula almost
    exactly (a 6.6 s script budgeted 13.9 s and rendered 14.9 s; a 9.2 s script
    budgeted 18.6 s and rendered 19.6 s), while a line that chose to stop finished
    at 7.7 s against a 13.5 s budget. Lowering temperature to 1.5 changed nothing,
    because temperature was never the constraint.

    So: tight headroom, and an explicit per-tag allowance instead of blanket grace.
    Well-behaved lines emit their end token early and are unaffected; runaway lines
    get cut off before they can improvise a second clip's worth of audio.
    """
    est = len(text) / CHARS_PER_SEC
    tags = text.count("(")            # (sighs), (laughs) … each needs real time
    return int((est * HEADROOM + GRACE_SEC + tags * TAG_SEC) * TOKENS_PER_SEC)


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
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            temp = max(job["direction"].get("temperature", 1.8), TEMP_FLOOR)
            torch.manual_seed(job["seed"])
            inputs = processor(text=[job["direction"]["render_text"]],
                               padding=True, return_tensors="pt").to("cuda")
            audio_tokens = model.generate(
                **inputs, max_new_tokens=token_budget(job["direction"]["render_text"]),
                guidance_scale=job["direction"].get("guidance", 3.0),
                temperature=temp, top_p=0.90, top_k=45)
            decoded = processor.batch_decode(audio_tokens)
            name = f"{job['id']}.wav"
            save_via_atomic(processor.save_audio,
                            os.path.join(args.out, name), decoded)
            # B-M6. This named its keys explicitly, so every passthrough field on the bank
            # line — intended_delivery, book, ref_id, source_ref, chunk_type, pair_key
            # extras — was DROPPED. `register_audition`'s prefill and the corpus fold read
            # those from the manifest, so the clip arrives at the audit surface stripped of
            # the metadata that decides how it is queued and folded. The other six
            # renderers were moved to `dict(job)` first; dia and moss85 were missed. A
            # SET_ASIDE-reinstatement trap: nothing renders through here today, which is
            # precisely why it would come back unnoticed.
            row = dict(job)
            row.update({
                "engine": "dia", "wav": name, "sr": 44100,
                "direction": {**job["direction"], "temperature": temp},
                "engine_license": "Apache-2.0 (Dia-1.6B-0626)",
                "bank_version": bank["version"], "campaign": bank["campaign"],
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], "done", flush=True)
    print("SYNTH-DIA-DONE")


if __name__ == "__main__":
    main()

"""Phase 1S Step 1c-0: the Qwen VoiceDesign anchor range calibration (60 clips).

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § 1c-0 (owner, 2026-10-21).
The owner hears a pure-pitch ladder (7 `pitch:` rungs per gender, every other key fixed) and six
niche personas, three authored sentences each, and marks every clip core / niche / exclude. From
those verdicts come the measured F0 bands that gate any Qwen anchor, the qwen.md casting rules,
and 1c-i's two anchors. This tool renders the clips and measures them; it decides nothing.

    # host: print the 60 items (no GPU, no model)
    .venv/bin/python scripts/tools/render_qwen_1c0.py plan

    # container (GPU, a JARVIS slot), from a clean checkout of the registered commit, in
    # $SONORA_ROCM_IMAGE with /data and the checkout mounted. Install exactly synth_bank.sh's qwen
    # recipe with the studied release pinned: apt sox; --no-deps qwen-tts==0.1.1;
    # $SONORA_PIN_TRANSFORMERS soundfile sox onnxruntime einops librosa;
    # --no-deps accelerate==1.12.0; plus pyloudnorm. Then:
    python scripts/tools/render_qwen_1c0.py render --commit <sha> \\
        --out /data/model-training/datasets/probes/qwen-1c0

    # host: F0 / HNR / phonation per clip, beside the clips
    .venv/bin/python scripts/tools/render_qwen_1c0.py measure \\
        --out /data/model-training/datasets/probes/qwen-1c0

The probe folder sits under `datasets/probes/`, which the audition app mounts and serves in its
probe mode (`manifest.jsonl` in listening order, `measures.jsonl` joined by `id`); nothing here
touches `ratings.csv`. Nothing here pre-fills an attribute from the prompt either: a pitch value
taken from the rung would tell the listener the answer.

⚠ BATCH SIZE 1, ONE SEED PER CLIP. `qwen_tts` has no seed argument; a clip is reproducible only
from `torch.manual_seed(seed)` immediately before its own call. A batched call draws every clip
from one RNG stream, so the registered per-clip seeds would no longer name the audio.

⚠ F0 IS MEASURED UP TO 700 Hz HERE. The repo's speaker estimators stop at 350 Hz
(`measure_speaker_hnr`) and 400 Hz (`phonation_measures`); child and extreme-high rungs can sit
above that, and an estimator that cannot see the top of the range would report the very clips the
owner is judging as lower than they are. CPP / H1-H2 / alpha keep `phonation_measures`' 60-400 Hz
cepstral band and say so in each row.
"""
import argparse
import json
import os
import sys
import time

import sys as _sys  # noqa: E402
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

MODEL_DIR = "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_TTS_VERSION = "0.1.1"
LUFS = -23.0
F0_MIN, F0_MAX = 50.0, 700.0
LANGUAGE = "English"

# Qwen's 12-key instruct order (director_skills/qwen.md).
KEY_ORDER = ("gender", "pitch", "speed", "volume", "age", "clarity", "fluency", "accent",
             "texture", "emotion", "tone", "personality")
# Every key but `pitch` is held here for the ladder: the delivery-v1-narration-r2 narrator keys
# (the voyage-of-the-beagle voice) with `texture` made neutral.
FIXED = {"speed": "Steady, unhurried", "volume": "Moderate conversational",
         "age": "Middle-aged adult", "clarity": "High clarity with distinct pronunciation",
         "fluency": "Highly fluent", "accent": "American English",
         "texture": "Clear and natural", "emotion": "Calm, even",
         "tone": "Even, narratorial", "personality": "Composed and unobtrusive"}
GENDERS = (("M", "Male", "male", 1), ("F", "Female", "female", 2))
RUNGS = ("Extremely low", "Very low", "Low", "Medium", "High", "Very high", "Extremely high")
# Each persona changes only gender, pitch, age and texture. `band` is the owner's age taxonomy
# (casting-attribute-norms-brief.md); `age` is Qwen's wording for it.
PERSONAS = (
    {"id": "P1", "band": "Child girl", "gender": "Female", "pitch": "High child pitch, stable",
     "age": "Child, about eight years old", "texture": "Light and clear"},
    {"id": "P2", "band": "Young girl", "gender": "Female", "pitch": "High, stable",
     "age": "Young teenager, about thirteen", "texture": "Light and natural"},
    {"id": "P3", "band": "Child boy", "gender": "Male", "pitch": "High child pitch, stable",
     "age": "Child, about nine years old", "texture": "Light and clear"},
    {"id": "P4", "band": "Man", "gender": "Male", "pitch": "Very low male pitch, stable",
     "age": "Adult", "texture": "Deep, full and chest-resonant"},
    {"id": "P5", "band": "Elderly man", "gender": "Male", "pitch": "Low male pitch, stable",
     "age": "Elderly, about seventy-five",
     "texture": "Slightly hoarse, with light age strain"},
    {"id": "P6", "band": "Elderly woman", "gender": "Female", "pitch": "Low female pitch, stable",
     "age": "Elderly, about seventy-five",
     "texture": "Slightly hoarse, with light age strain"},
)
SENTENCES = (
    "By the time the rain stopped, the harbour lights were coming on one by one along the far "
    "side of the bay.",
    "She folded the letter twice, set it on the table beside the lamp, and went out without a "
    "word.",
    "The road climbed slowly through the pines, and from the top of the ridge the whole valley "
    "lay open below.",
)


def build_instruct(keys):
    """{key: value} -> Qwen's 12-key instruct, every key filled, in KEY_ORDER."""
    missing = [k for k in KEY_ORDER if not keys.get(k)]
    if missing or set(keys) - set(KEY_ORDER):
        raise ValueError("instruct keys must be exactly %s (missing %s)" % (KEY_ORDER, missing))
    return " ".join("%s: %s." % (k, keys[k].rstrip(".")) for k in KEY_ORDER)


def ladder():
    """The 60 items in listening order: M1..M7, F1..F7, then P1..P6; sentences 1..3 each.
    Seeds: 9GRS (G 1 = M / 2 = F, rung R, sentence S) and 93PS (persona P)."""
    items = []
    for g, gender, word, gd in GENDERS:
        for r, rung in enumerate(RUNGS, 1):
            keys = dict(FIXED, gender=gender, pitch="%s %s pitch, stable" % (rung, word))
            for s, text in enumerate(SENTENCES, 1):
                items.append({"id": "%s%d-s%d" % (g, r, s), "group": "%s ladder" % gender,
                              "rung": r, "label": rung, "band": "Man" if g == "M" else "Woman",
                              "seed": int("9%d%d%d" % (gd, r, s)), "sentence": s, "text": text,
                              "instruct": build_instruct(keys)})
    for n, p in enumerate(PERSONAS, 1):
        keys = dict(FIXED, **{k: p[k] for k in ("gender", "pitch", "age", "texture")})
        for s, text in enumerate(SENTENCES, 1):
            items.append({"id": "%s-s%d" % (p["id"], s), "group": "niche personas",
                          "rung": None, "label": p["id"], "band": p["band"],
                          "seed": int("93%d%d" % (n, s)), "sentence": s, "text": text,
                          "instruct": build_instruct(keys)})
    return items


def require_qwen_tts(version_of):
    """Refuse any qwen-tts but the studied one (`version_of` is importlib.metadata.version)."""
    got = version_of("qwen-tts")
    if got != QWEN_TTS_VERSION:
        raise SystemExit("REFUSING: qwen-tts %s installed, %s is the studied release "
                         "(phase-1s-step1-preregistration.md § Onboarding)." % (got, QWEN_TTS_VERSION))
    return got


def f0_profile(x, sr):
    """Voiced-frame F0 (Hz) of a mono clip with this probe's wide band: (median, p10, p90, n)."""
    import numpy as np
    from periodicity import frames
    f0, hnr = frames(np.asarray(x, dtype=np.float64), sr, F0_MIN, F0_MAX, 0.01, 0.3)
    if not len(f0):
        return None
    return {"f0_median": float(np.median(f0)), "f0_p10": float(np.percentile(f0, 10)),
            "f0_p90": float(np.percentile(f0, 90)),
            "f0_excursion": float(np.percentile(f0, 90) - np.percentile(f0, 10)),
            "hnr": float(np.median(hnr)), "voiced_frames": int(len(f0))}


def cmd_plan(args):
    items = ladder()
    for it in items:
        print("%-7s seed %d  %-15s %s" % (it["id"], it["seed"], it["band"], it["instruct"][:96]))
    print("%d items" % len(items))


def cmd_render(args):
    import importlib.metadata
    import hashlib

    import torch
    from qwen_tts import Qwen3TTSModel

    from ear_bench import match_loudness
    from synth_common import write_json_atomic, write_wav_atomic

    version = require_qwen_tts(importlib.metadata.version)
    clips = os.path.join(args.out, "clips")
    os.makedirs(clips, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        done = {json.loads(line)["id"] for line in open(manifest, encoding="utf-8") if line.strip()}
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0", dtype=torch.bfloat16)
    h = hashlib.sha256()
    with open(os.path.join(MODEL_DIR, "model.safetensors"), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    weights = h.hexdigest()
    limited = 0
    with open(manifest, "a", encoding="utf-8") as mf:
        for it in ladder():
            if it["id"] in done:
                continue
            torch.manual_seed(it["seed"])
            t0 = time.monotonic()
            wavs, sr = model.generate_voice_design(text=it["text"], language=LANGUAGE,
                                                   instruct=it["instruct"])
            wall = time.monotonic() - t0
            audio, bound = match_loudness(wavs[0], sr, LUFS)
            limited += int(bound)
            write_wav_atomic(os.path.join(clips, it["id"] + ".wav"), audio, sr)
            row = dict(it, file="clips/%s.wav" % it["id"], sr=sr,
                       seconds=round(len(audio) / sr, 3), wall_s=round(wall, 2),
                       peak_limited=bool(bound))
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            print("%-7s %5.1fs audio  %5.1fs wall" % (it["id"], row["seconds"], wall), flush=True)
    write_json_atomic(os.path.join(args.out, "render_meta.json"), {
        "tool": "scripts/tools/render_qwen_1c0.py", "commit": args.commit, "model_dir": MODEL_DIR,
        "qwen_tts": version, "lufs": LUFS, "batch_size": 1, "language": LANGUAGE,
        "items": len(ladder()), "peak_limited_this_run": limited, "weights_sha256": weights})
    print("RENDER-QWEN-1C0-DONE")


def cmd_measure(args):
    import soundfile as sf

    from derive_vat_corpus import phonation_measures
    from synth_common import write_text_atomic

    rows = [json.loads(line) for line in open(os.path.join(args.out, "manifest.jsonl"),
                                              encoding="utf-8") if line.strip()]
    if len(rows) != len(ladder()):
        raise SystemExit("REFUSING: %d of %d clips rendered." % (len(rows), len(ladder())))
    out = []
    for r in rows:
        x, sr = sf.read(os.path.join(args.out, r["file"]), dtype="float32")
        f0 = f0_profile(x, sr)
        ph = phonation_measures(x, sr)
        out.append(dict({"id": r["id"], "f0_band_hz": [F0_MIN, F0_MAX],
                         "phonation_f0_band_hz": [60.0, 400.0]},
                        **(f0 or {"f0_median": None}), **(ph or {})))
    write_text_atomic(os.path.join(args.out, "measures.jsonl"),
                      "".join(json.dumps(o) + "\n" for o in out))
    print("%d clips measured; %d without a voiced frame" % (len(out),
                                                           sum(o["f0_median"] is None for o in out)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    r.add_argument("--commit", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"plan": cmd_plan, "render": cmd_render, "measure": cmd_measure}[args.cmd](args)


if __name__ == "__main__":
    main()

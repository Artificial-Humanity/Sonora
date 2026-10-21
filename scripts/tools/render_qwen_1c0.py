"""Phase 1S Step 1c-0: the Qwen VoiceDesign anchor range calibration (60 clips).

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § 1c-0 (owner, 2026-10-21).
The owner hears a pure-pitch ladder (7 `pitch:` rungs per gender, every other key fixed) and six
niche personas, three authored sentences each, and marks every clip core / niche / exclude. From
those verdicts come the measured F0 bands that gate any Qwen anchor, the qwen.md casting rules,
and 1c-i's two anchors. This tool renders the clips and measures them; it decides nothing.

    # host: print the 60 items (no GPU, no model)
    .venv/bin/python scripts/tools/render_qwen_1c0.py plan

    # host: the container runs as ai-mgr and cannot create a directory in probes/
    install -d -m 2775 /data/model-training/datasets/probes/qwen-1c0

    # container (GPU, a JARVIS slot), in bash from the root of a clean checkout of the registered
    # commit: synth_bank.sh's run() and qwen block, with the studied release pinned and pyloudnorm
    # added. The --device ... video flags are synth_bank.sh's $GPU.
    . scripts/container_env.sh
    docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \\
      --group-add video -e SONORA_ROCM_IMAGE="$SONORA_ROCM_IMAGE" \\
      -v /data:/data -v "$PWD":/sonora "$SONORA_ROCM_IMAGE" bash -c "
      $SONORA_UV_BOOTSTRAP;
      apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y sox >/dev/null 2>&1;
      $SONORA_UVPIP --no-deps qwen-tts==0.1.1 >/dev/null 2>&1;
      $SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile sox onnxruntime einops librosa >/dev/null 2>&1;
      $SONORA_UVPIP --no-deps accelerate==1.12.0 >/dev/null 2>&1;
      $SONORA_UVPIP pyloudnorm >/dev/null 2>&1;
      bash /sonora/scripts/container_as_ai_mgr.sh &&
      $(capture_env_cmd qwen-1c0 /data/model-training/datasets/probes/qwen-1c0/_env)
      runuser -u ai-mgr -- bash -c 'umask 002; python /sonora/scripts/tools/render_qwen_1c0.py \\
        render --commit <sha> --out /data/model-training/datasets/probes/qwen-1c0'"

    # host: F0 / HNR / phonation per clip, beside the clips
    .venv/bin/python scripts/tools/render_qwen_1c0.py measure \\
        --out /data/model-training/datasets/probes/qwen-1c0

The probe folder is laid out for the audition app's probe mode (`manifest.jsonl` in listening
order, `measures.jsonl` joined by `id`). That mode is pending in a separate PR: at this commit
nothing serves the folder. Nothing here touches `ratings.csv`. Nothing here pre-fills an attribute
from the prompt either: a pitch value taken from the rung would tell the listener the answer.

⚠ THE STEP 1B SPEAKER EMBEDDING IS A SEPARATE STEP. The spec's per-clip measures include the Zonos
`SpeakerEmbeddingLDA` embedding; this tool does not compute it and takes no Zonos dependency. The
Step 1b extractor computes it, on CPU in an environment with torch and /data/toolchain/Zonos:
`/data/model-training/sonora/analysis/step1b_speaker_consistency_2026-10-21/features.py`. Its
inputs are Step 1b's campaigns, so point it at this probe folder's clips. 1c-i's anchor choice
needs that embedding beside the verdicts and these measures.

⚠ BATCH SIZE 1, ONE SEED PER CLIP. `qwen_tts` has no seed argument; a clip is reproducible only
from `torch.manual_seed(seed)` immediately before its own call. A batched call draws every clip
from one RNG stream, so the registered per-clip seeds would no longer name the audio.

⚠ A RESUME MUST NOT MIX TWO RENDERS. `render_meta.json` is written before the first clip. With a
manifest already present, `render` refuses unless that meta's commit, weights hashes
(`model.safetensors` and every file under `speech_tokenizer/`) and qwen-tts version equal this
run's, and unless every rendered row is its ladder item (seed, text, instruct, commit) with its wav
on disk. Only a truncated final manifest line, a render killed mid-write, is dropped (with a
warning) and re-rendered. When nothing remains it exits before loading the model.

⚠ F0 IS TWO PASSES OF `periodicity.frames`, NOT ONE WIDE SEARCH. A 50-700 Hz search starts at a
34-sample lag at 24 kHz, where a voice below about 80 Hz has not decorrelated yet, so the peak pins
to the search edge and the clip reads about 706 Hz. Pass 1 is the repo's speaker band (the 350 Hz
ceiling of `measure_speaker_f0` and `measure_harmonicity`, floor 50 Hz). Above 350 Hz it reads a
voice at half its F0: the period is shorter than the band's shortest lag and twice the period is
inside it. So `HIGH_BAND` re-measures when pass 1's median is pressed against its top, or when pass
1 reads above half its ceiling and the high band is more periodic (median HNR higher by
`HNR_MARGIN_DB`) without being pinned to its own floor. A subharmonic is always less periodic than
the true period; the formant lags a low voice shows in the high band are far less periodic than
its own period. Each row says
which band it used (`f0_band`), the fraction of voiced frames whose lag sits at either search edge
(`f0_edge_frac`: a reading at an edge is the edge, not the voice) and `f0_above_ceiling`. Above
about 740 Hz the high band locks to a subharmonic without touching an edge, so a 700-1400 Hz probe
that finds the voice, unpinned and more periodic by the same margin, sets `f0_above_ceiling`. A voice at the 50 Hz
floor or near 700 Hz pins both bands at their shortest lag and shows only as `f0_edge_frac` near 1,
whichever end it is. Either flag means the clip's F0 is not measured.

⚠ HNR IS DESCRIPTIVE ONLY. `periodicity.frames` normalises a biased autocorrelation, so the peak at
the period shrinks with the lag: a perfectly periodic 1/k² harmonic stack reads about 2.5 dB at
70 Hz and about 14 dB at 650 Hz. Across this ladder HNR tracks F0, not breathiness. Each row carries
`hnr_note`; do not compare or gate HNR across rungs.

⚠ CPP / H1-H2 / ALPHA COME FROM `phonation_measures`, WHOSE CEPSTRAL F0 SEARCH IS
`derive_vat_corpus.F0_MIN`-`F0_MAX`. A clip measured in the high band or with F0 p90 above that
ceiling gets nulls and `phonation_null_reason`, rather than a measure of the wrong peak.
"""
import argparse
import hashlib
import json
import math
import os
import time

import sys as _sys  # noqa: E402
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

TOOL = "scripts/tools/render_qwen_1c0.py"
MODEL_DIR = "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_TTS_VERSION = "0.1.1"
LUFS = -23.0
LANGUAGE = "English"
# `render_meta.json` keys a resume must match exactly.
RESUME_KEYS = ("commit", "weights_sha256", "qwen_tts")

# F0 search bands (Hz) for `periodicity.frames`, with the repo estimators' voicing thresholds.
LOW_BAND = (50.0, 350.0)
HIGH_BAND = (250.0, 700.0)
CEILING_PROBE = (700.0, 1400.0)
TOP_OF_LOW = 0.9        # a pass-1 median above 0.9 x 350 Hz is pressed against the band's top
# A periodic voice's true period reads 3.2-3.4 dB more HNR than its octave error below 700 Hz and
# 1.6-2 dB more than its subharmonic lock above (biased autocorrelation, 40 ms window). A voice
# whose even harmonics dominate reads within 0.5 dB at twice its F0, so the switch needs a margin.
HNR_MARGIN_DB = 1.0
RMS_FLOOR, PERIODICITY = 0.01, 0.3
HNR_NOTE = ("descriptive only: periodicity.frames' biased autocorrelation reads a periodic voice "
            "higher the higher its F0; do not compare or gate HNR across rungs")

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
# (casting-attribute-norms-brief.md); `age` is Qwen's wording for it. The spec's "Man (large)" is
# band `Man`: the taxonomy has no size band, size is a separate channel, so it lives in `label`.
PERSONAS = (
    {"id": "P1", "label": "P1 child girl", "band": "Child girl", "gender": "Female",
     "pitch": "High child pitch, stable", "age": "Child, about eight years old",
     "texture": "Light and clear"},
    {"id": "P2", "label": "P2 young girl", "band": "Young girl", "gender": "Female",
     "pitch": "High, stable", "age": "Young teenager, about thirteen",
     "texture": "Light and natural"},
    {"id": "P3", "label": "P3 child boy", "band": "Child boy", "gender": "Male",
     "pitch": "High child pitch, stable", "age": "Child, about nine years old",
     "texture": "Light and clear"},
    {"id": "P4", "label": "P4 large man", "band": "Man", "gender": "Male",
     "pitch": "Very low male pitch, stable", "age": "Adult",
     "texture": "Deep, full and chest-resonant"},
    {"id": "P5", "label": "P5 elderly man", "band": "Elderly man", "gender": "Male",
     "pitch": "Low male pitch, stable", "age": "Elderly, about seventy-five",
     "texture": "Slightly hoarse, with light age strain"},
    {"id": "P6", "label": "P6 elderly woman", "band": "Elderly woman", "gender": "Female",
     "pitch": "Low female pitch, stable", "age": "Elderly, about seventy-five",
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
                          "rung": None, "label": p["label"], "band": p["band"],
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


def _read_band(x, sr, band):
    """`periodicity.frames` over one band -> its voiced frames' f0 and hnr, and the fraction whose
    lag is the band's shortest (`at_top`, fmax) or longest (`at_floor`, fmin). None if unvoiced."""
    import numpy as np
    from periodicity import frames
    f0, hnr = frames(x, sr, band[0], band[1], RMS_FLOOR, PERIODICITY)
    if not len(f0):
        return None
    lag = np.rint(sr / f0).astype(int)          # frames reports sr / lag exactly
    return {"band": band, "f0": f0, "hnr": float(np.median(hnr)),
            "at_top": float(np.mean(lag == int(sr / band[1]))),
            "at_floor": float(np.mean(lag == int(sr / band[0]) - 1))}


def f0_profile(x, sr):
    """Voiced-frame F0 of a mono clip, two-pass (module docstring): the band used, median, p10,
    p90, excursion in Hz and semitones, edge fraction, the ceiling flag, median HNR. None if no
    frame is voiced."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    use = low = _read_band(x, sr, LOW_BAND)
    if low is None:
        return None
    m_low = float(np.median(low["f0"]))
    above = False
    if m_low > 0.5 * LOW_BAND[1]:               # pass 1 may be reading a voice above 350 Hz at half
        high = _read_band(x, sr, HIGH_BAND)
        if high is not None and (m_low > TOP_OF_LOW * LOW_BAND[1]
                                 or (high["hnr"] > low["hnr"] + HNR_MARGIN_DB
                                     and high["at_floor"] < 0.5)):
            use = high
            probe = _read_band(x, sr, CEILING_PROBE)
            above = bool(probe is not None and len(probe["f0"]) >= 0.5 * len(high["f0"])
                         and probe["hnr"] > high["hnr"] + HNR_MARGIN_DB
                         and probe["at_top"] + probe["at_floor"] < 0.5)
    f0 = use["f0"]
    p10, p90 = float(np.percentile(f0, 10)), float(np.percentile(f0, 90))
    return {"f0_band": "high" if use is not low else "low", "f0_band_hz": list(use["band"]),
            "f0_median": float(np.median(f0)), "f0_p10": p10, "f0_p90": p90,
            "f0_excursion": p90 - p10, "f0_excursion_st": 12.0 * math.log2(p90 / p10),
            "f0_edge_frac": use["at_top"] + use["at_floor"], "f0_above_ceiling": above,
            "hnr": use["hnr"], "voiced_frames": int(len(f0))}


def phonation_row(x, sr, f0):
    """`phonation_measures` (CPP, H1-H2, alpha), or nulls and the reason when the clip's F0 sits
    above its cepstral band."""
    import derive_vat_corpus as dvc
    row = {"phonation_f0_band_hz": [dvc.F0_MIN, dvc.F0_MAX], "alpha_db": None, "cpp": None,
           "h1h2": None, "phonation_null_reason": None}
    if f0 is not None and (f0["f0_band"] == "high" or f0["f0_p90"] > dvc.F0_MAX):
        row["phonation_null_reason"] = (
            "F0 above phonation_measures' %g-%g Hz cepstral band (f0_band %s, p90 %.0f Hz)"
            % (dvc.F0_MIN, dvc.F0_MAX, f0["f0_band"], f0["f0_p90"]))
        return row
    ph = dvc.phonation_measures(x, sr)
    if ph is None:
        row["phonation_null_reason"] = "phonation_measures found too few voiced frames"
        return row
    return dict(row, **ph)


def read_manifest(path):
    """-> (rows, repaired text or None, dropped tail or None). Only an unparseable final line
    without its newline (a render killed mid-write) is dropped; any other unparseable line
    refuses. `repaired` is what the file must become before a row is appended to it."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.split("\n")
    tail = lines.pop()                          # "" when the file ends in a newline
    rows = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            raise SystemExit("REFUSING: %s line %d is not JSON and is not the final line." % (path, n))
    if not tail.strip():
        return rows, None, None
    try:
        rows.append(json.loads(tail))           # a whole row that lost only its newline
        return rows, text + "\n", None
    except ValueError:
        return rows, text[:len(text) - len(tail)], tail


def check_rows(rows, out, commit=None):
    """Refuse a manifest row that is not its ladder item, repeats an id, or has no wav."""
    items = {it["id"]: it for it in ladder()}
    seen = set()
    for r in rows:
        it = items.get(r.get("id"))
        if it is None or r["id"] in seen:
            raise SystemExit("REFUSING: manifest id %r is %s." % (
                r.get("id"), "repeated" if it is not None else "not a ladder id"))
        seen.add(r["id"])
        bad = [k for k in ("seed", "text", "instruct") if r.get(k) != it[k]]
        bad += ["file"] if r.get("file") != "clips/%s.wav" % r["id"] else []
        bad += ["commit"] if commit is not None and r.get("commit") != commit else []
        if bad:
            raise SystemExit("REFUSING: manifest row %s differs from its ladder item in %s."
                             % (r["id"], ", ".join(bad)))
        if not os.path.exists(os.path.join(out, r["file"])):
            raise SystemExit("REFUSING: manifest row %s has no wav at %s." % (r["id"], r["file"]))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def weights_sha256(model_dir):
    """{relative path: sha256} for `model.safetensors` and every file under `speech_tokenizer/`."""
    tok = os.path.join(model_dir, "speech_tokenizer")
    rels = ["model.safetensors"] + sorted(
        os.path.relpath(os.path.join(root, n), model_dir)
        for root, _dirs, names in os.walk(tok) for n in names)
    if len(rels) == 1 or not os.path.isfile(os.path.join(model_dir, rels[0])):
        raise SystemExit("REFUSING: %s needs model.safetensors and a non-empty speech_tokenizer/."
                         % model_dir)
    return {rel: _sha256(os.path.join(model_dir, rel)) for rel in rels}


def resume_state(out, meta):
    """-> the ids already rendered into `out`, after refusing a resume that would mix renders."""
    from synth_common import write_text_atomic

    manifest = os.path.join(out, "manifest.jsonl")
    if not os.path.exists(manifest):
        return set()
    try:
        with open(os.path.join(out, "render_meta.json"), encoding="utf-8") as fh:
            old = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit("REFUSING: %s exists but its render_meta.json cannot be read (%s)."
                         % (manifest, e))
    differ = [k for k in RESUME_KEYS if old.get(k) != meta[k]]
    if differ:
        raise SystemExit("REFUSING: resuming %s, but this run's %s differ from its "
                         "render_meta.json. Render into a new --out." % (out, ", ".join(differ)))
    rows, repaired, dropped = read_manifest(manifest)
    check_rows(rows, out, commit=meta["commit"])
    if dropped is not None:
        print("WARNING: dropping a truncated final manifest line (%d chars); that clip is "
              "rendered again." % len(dropped), flush=True)
    if repaired is not None:
        write_text_atomic(manifest, repaired)
    return {r["id"] for r in rows}


def cmd_plan(args):
    items = ladder()
    for it in items:
        print("%-7s seed %d  %-16s %-13s %s" % (it["id"], it["seed"], it["label"], it["band"],
                                                 it["instruct"][:80]))
    print("%d items" % len(items))


def cmd_render(args):
    import importlib.metadata

    from loudness import match_loudness
    from synth_common import write_json_atomic, write_wav_atomic

    version = require_qwen_tts(importlib.metadata.version)
    meta = {"tool": TOOL, "commit": args.commit, "model_dir": MODEL_DIR, "qwen_tts": version,
            "weights_sha256": weights_sha256(MODEL_DIR), "lufs": LUFS, "batch_size": 1,
            "language": LANGUAGE, "items": len(ladder())}
    done = resume_state(args.out, meta)
    todo = [it for it in ladder() if it["id"] not in done]
    if not todo:
        print("all %d clips already rendered; the model was not loaded." % len(done))
        print("RENDER-QWEN-1C0-DONE")
        return
    clips = os.path.join(args.out, "clips")
    os.makedirs(clips, exist_ok=True)
    if not done:
        write_json_atomic(os.path.join(args.out, "render_meta.json"), meta)

    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0", dtype=torch.bfloat16)
    with open(os.path.join(args.out, "manifest.jsonl"), "a", encoding="utf-8") as mf:
        for it in todo:
            torch.manual_seed(it["seed"])
            t0 = time.monotonic()
            wavs, sr = model.generate_voice_design(text=it["text"], language=LANGUAGE,
                                                   instruct=it["instruct"])
            wall = time.monotonic() - t0
            audio, bound = match_loudness(wavs[0], sr, LUFS)
            write_wav_atomic(os.path.join(clips, it["id"] + ".wav"), audio, sr)
            row = dict(it, file="clips/%s.wav" % it["id"], sr=sr,
                       seconds=round(len(audio) / sr, 3), wall_s=round(wall, 2),
                       peak_limited=bool(bound), commit=args.commit)
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            print("%-7s %5.1fs audio  %5.1fs wall" % (it["id"], row["seconds"], wall), flush=True)
    print("RENDER-QWEN-1C0-DONE")


F0_KEYS = ("f0_band", "f0_band_hz", "f0_median", "f0_p10", "f0_p90", "f0_excursion",
           "f0_excursion_st", "f0_edge_frac", "f0_above_ceiling", "hnr", "voiced_frames")


def cmd_measure(args):
    import soundfile as sf

    from synth_common import write_text_atomic

    manifest = os.path.join(args.out, "manifest.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit("REFUSING: no manifest at %s; render first." % manifest)
    rows, _repaired, dropped = read_manifest(manifest)
    if dropped is not None:
        raise SystemExit("REFUSING: %s ends in a truncated line; resume the render." % manifest)
    check_rows(rows, args.out)
    want = [it["id"] for it in ladder()]
    if {r["id"] for r in rows} != set(want):
        raise SystemExit("REFUSING: %d of %d ladder clips rendered (missing %s)." % (
            len(rows), len(want), sorted(set(want) - {r["id"] for r in rows})))
    by_id = {r["id"]: r for r in rows}
    out = []
    for i in want:
        x, sr = sf.read(os.path.join(args.out, by_id[i]["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0 = f0_profile(x, sr)
        out.append(dict({"id": i}, **(f0 or dict.fromkeys(F0_KEYS)), hnr_note=HNR_NOTE,
                        **phonation_row(x, sr, f0)))
    write_text_atomic(os.path.join(args.out, "measures.jsonl"),
                      "".join(json.dumps(o) + "\n" for o in out))
    print("%d clips measured; %d without a voiced frame; %d without a phonation measure; %d with "
          "F0 at a search edge (> 10%% of frames) or above the %g Hz ceiling" % (
              len(out), sum(o["f0_median"] is None for o in out),
              sum(o["cpp"] is None for o in out),
              sum(bool(o["f0_above_ceiling"]) or (o["f0_edge_frac"] or 0) > 0.1 for o in out),
              HIGH_BAND[1]))


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

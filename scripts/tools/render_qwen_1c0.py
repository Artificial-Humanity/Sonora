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

    # container (GPU, a JARVIS slot), in bash from the root of the checkout of the registered
    # commit: synth_bank.sh's run() and qwen block, with the studied release pinned and pyloudnorm
    # added. The --device ... video flags are synth_bank.sh's $GPU. It runs only from a checkout
    # with no uncommitted change and no untracked file under scripts/, so the --commit it passes
    # names the code that runs. `$(git ... rev-parse HEAD)` expands on the host because the
    # bash -c string is double-quoted.
    if git -C "$PWD" diff --quiet HEAD &&
       [ -z "$(git -C "$PWD" ls-files -o --exclude-standard scripts/)" ]; then
    . scripts/container_env.sh
    docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \\
      --group-add video -e SONORA_ROCM_IMAGE="$SONORA_ROCM_IMAGE" \\
      -v /data:/data -v "$PWD":/sonora "$SONORA_ROCM_IMAGE" bash -c "
      $SONORA_UV_BOOTSTRAP;
      apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y sox >/dev/null 2>&1;
      $SONORA_UVPIP --no-deps qwen-tts==0.1.1 >/dev/null 2>&1;
      $SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile sox onnxruntime einops librosa >/dev/null 2>&1;
      $SONORA_UVPIP --no-deps accelerate==1.12.0 >/dev/null 2>&1;
      $SONORA_UVPIP $SONORA_PIN_PYLOUDNORM >/dev/null 2>&1;
      bash /sonora/scripts/container_as_ai_mgr.sh &&
      $(capture_env_cmd qwen-1c0 /data/model-training/datasets/probes/qwen-1c0/_env)
      runuser -u ai-mgr -- bash -c 'umask 002; python /sonora/scripts/tools/render_qwen_1c0.py \\
        render --commit $(git -C "$PWD" rev-parse HEAD) \\
        --out /data/model-training/datasets/probes/qwen-1c0'"
    else echo "REFUSING: $PWD has uncommitted changes or untracked files under scripts/."; fi

    # host: F0 / HNR / phonation per clip, beside the clips (librosa's pYIN: the host venv)
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

⚠ F0 IS LIBROSA'S pYIN OVER 50-700 Hz, NOT THE REPO'S AUTOCORRELATION. `periodicity.frames` takes
the highest peak in one fixed lag band. Over 50-700 Hz at 24 kHz its shortest lag (34 samples) is
where a voice below about 80 Hz has not decorrelated yet, so the clip reads about 706 Hz; over the
repo's 50-350 Hz band a voice above 350 Hz reads at half its F0; and a per-clip rule choosing between
bands read noisy high voices at half with no flag (a 600 Hz stack at 7 dB SNR as 300 Hz, a real Qwen
clip that pYIN reads at 371 Hz as 235 Hz). pYIN weighs every candidate period in every frame and
tracks the contour, so one search covers the range. Its frame (`pyin_frame`) is the smallest power of
two holding two periods of the 50 Hz floor, librosa's condition for reading the floor accurately:
1024 samples (42.7 ms) at 24 kHz, close to `periodicity.frames`' 40 ms. librosa's default 2048 reads
the test file's voices within 0.5 semitone of it, with half the time resolution on the contour
whose p10-p90 is the excursion. The hop is 10 ms. Over pYIN's voiced frames each row reports:

* `f0_edge_frac`: the fraction within `EDGE_TOL` of 50 or 700 Hz. A reading at a limit is the limit,
  not the voice: a voice below 50 Hz reads 50 Hz.
* `f0_above_range_frac`: the fraction of a second pYIN pass, its ceiling an octave higher
  (`F0_EXTENDED_FMAX`), that reads above 700 Hz. The 50-700 Hz pass reads a voice above 700 Hz at
  half its F0, off both edges, so only this pass sees it.
* `f0_out_of_range`: either fraction above `OUT_OF_RANGE_FRAC`. The clip's F0 is then not measured.
* `f0_median_acf` and `f0_methods_disagree`: the repo autocorrelation's median over 50-350 Hz, and
  whether it is more than `METHODS_DISAGREE_ST` from pYIN's. pYIN reads a voice whose second
  harmonic exceeds its first by about 16 dB an octave up (a synthetic 220 Hz voice reads 441 Hz;
  from about 350 Hz only the extended pass does, which sets `f0_out_of_range` on a voice in range).
  The autocorrelation reads such a voice at its F0, so a misread is visible. Above 350 Hz the
  autocorrelation reads a subharmonic and the two disagree by construction; there a flagged clip
  needs the ear.

⚠ HNR IS DESCRIPTIVE ONLY. It is `periodicity.frames` over the repo's 50-350 Hz band, the pass that
gives `f0_median_acf`. That estimator normalises a biased autocorrelation, so the peak at the period
shrinks with the lag: a perfectly periodic 1/k² harmonic stack reads about 2.5 dB at 70 Hz and
about 11 dB at 330 Hz, and above 350 Hz HNR is read at a subharmonic's lag. Across this ladder HNR
tracks F0, not breathiness. Each row carries `hnr_note`; do not compare or gate HNR across rungs.

⚠ CPP / H1-H2 / ALPHA COME FROM `phonation_measures`, WHOSE CEPSTRAL F0 SEARCH IS
`derive_vat_corpus.F0_MIN`-`F0_MAX`. A clip with F0 p10 below that floor or p90 above that ceiling,
or with `f0_out_of_range`, gets nulls and `phonation_null_reason`, rather than a measure of the wrong
peak.
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

# F0 (module docstring): pYIN's search range (Hz), and the extended pass that sees a voice above it.
F0_RANGE = (50.0, 700.0)
F0_EXTENDED_FMAX = 2 * F0_RANGE[1]
EDGE_TOL = 0.03         # a reading within 3% of a search limit is the limit
OUT_OF_RANGE_FRAC = 0.1
METHODS_DISAGREE_ST = 1.0
# The repo autocorrelation (`periodicity.frames`), kept for HNR and the `f0_median_acf` audit only:
# the speaker band of `measure_speaker_f0` and `measure_harmonicity`, with their voicing thresholds.
ACF_BAND = (50.0, 350.0)
RMS_FLOOR, PERIODICITY = 0.01, 0.3
HNR_NOTE = ("descriptive only: periodicity.frames' biased autocorrelation over %g-%g Hz reads a "
            "periodic voice higher the higher its F0, and a voice above %g Hz at a subharmonic; "
            "do not compare or gate HNR across rungs" % (ACF_BAND + ACF_BAND[1:]))

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


def pyin_frame(sr):
    """pYIN's frame in samples: the smallest power of two holding two periods of the F0 floor
    (librosa warns below that). 1024 samples, 42.7 ms, at 24 kHz."""
    return 1 << math.ceil(math.log2(2 * sr / F0_RANGE[0] + 2))


def _pyin_voiced(x, sr, fmax):
    """librosa's pYIN from the F0 floor to `fmax` -> the F0 (Hz) of its voiced frames."""
    import librosa
    import numpy as np
    f0, voiced, _prob = librosa.pyin(x.astype(np.float32), fmin=F0_RANGE[0], fmax=fmax, sr=sr,
                                     frame_length=pyin_frame(sr), hop_length=int(0.010 * sr))
    return f0[voiced & np.isfinite(f0)]


def f0_profile(x, sr):
    """Voiced-frame F0 of a mono clip from pYIN (module docstring): median, p10, p90, excursion in
    Hz and semitones, the edge and above-range fractions, the out-of-range flag, the autocorrelation
    median it is audited against, and median HNR. None if pYIN finds no voiced frame."""
    import numpy as np
    from periodicity import frames
    x = np.asarray(x, dtype=np.float64)
    f0 = _pyin_voiced(x, sr, F0_RANGE[1])
    if not len(f0):
        return None
    wide = _pyin_voiced(x, sr, F0_EXTENDED_FMAX)
    acf, hnr = frames(x, sr, ACF_BAND[0], ACF_BAND[1], RMS_FLOOR, PERIODICITY)
    lo, hi = F0_RANGE
    median = float(np.median(f0))
    p10, p90 = float(np.percentile(f0, 10)), float(np.percentile(f0, 90))
    edge = float(np.mean((f0 < lo * (1 + EDGE_TOL)) | (f0 > hi / (1 + EDGE_TOL))))
    above = float(np.mean(wide > hi)) if len(wide) else 0.0
    acf_median = float(np.median(acf)) if len(acf) else None
    return {"f0_median": median, "f0_p10": p10, "f0_p90": p90, "f0_excursion": p90 - p10,
            "f0_excursion_st": 12.0 * math.log2(p90 / p10), "voiced_frames": int(len(f0)),
            "f0_range_hz": list(F0_RANGE), "f0_edge_frac": edge, "f0_above_range_frac": above,
            "f0_out_of_range": edge > OUT_OF_RANGE_FRAC or above > OUT_OF_RANGE_FRAC,
            "f0_median_acf": acf_median,
            "f0_methods_disagree": None if acf_median is None else
            abs(12.0 * math.log2(median / acf_median)) > METHODS_DISAGREE_ST,
            "hnr": float(np.median(hnr)) if len(hnr) else None}


def phonation_row(x, sr, f0):
    """`phonation_measures` (CPP, H1-H2, alpha), or nulls and the reason when the clip's F0 is not
    measured or leaves its cepstral band at either end."""
    import derive_vat_corpus as dvc
    row = {"phonation_f0_band_hz": [dvc.F0_MIN, dvc.F0_MAX], "alpha_db": None, "cpp": None,
           "h1h2": None, "phonation_null_reason": None}
    if f0 is not None and f0["f0_out_of_range"]:
        row["phonation_null_reason"] = (
            "F0 not measured: at a pYIN search limit or above %g Hz" % F0_RANGE[1])
        return row
    if f0 is not None and (f0["f0_p90"] > dvc.F0_MAX or f0["f0_p10"] < dvc.F0_MIN):
        row["phonation_null_reason"] = (
            "F0 p10-p90 %.0f-%.0f Hz leaves phonation_measures' %g-%g Hz cepstral band"
            % (f0["f0_p10"], f0["f0_p90"], dvc.F0_MIN, dvc.F0_MAX))
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


F0_KEYS = ("f0_median", "f0_p10", "f0_p90", "f0_excursion", "f0_excursion_st", "voiced_frames",
           "f0_range_hz", "f0_edge_frac", "f0_above_range_frac", "f0_out_of_range",
           "f0_median_acf", "f0_methods_disagree", "hnr")


def cmd_measure(args):
    import soundfile as sf

    from synth_common import write_text_atomic

    manifest = os.path.join(args.out, "manifest.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit("REFUSING: no manifest at %s; render first." % manifest)
    rows, _repaired, dropped = read_manifest(manifest)
    if dropped is not None:
        raise SystemExit("REFUSING: %s ends in a truncated line; resume the render." % manifest)
    try:
        with open(os.path.join(args.out, "render_meta.json"), encoding="utf-8") as fh:
            commit = json.load(fh)["commit"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit("REFUSING: %s has no readable render_meta.json commit (%r)."
                         % (args.out, e))
    check_rows(rows, args.out, commit=commit)
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
          "F0 out of pYIN's %g-%g Hz range; %d where pYIN and the autocorrelation disagree" % (
              len(out), sum(o["f0_median"] is None for o in out),
              sum(o["cpp"] is None for o in out), sum(bool(o["f0_out_of_range"]) for o in out),
              F0_RANGE[0], F0_RANGE[1], sum(bool(o["f0_methods_disagree"]) for o in out)))


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

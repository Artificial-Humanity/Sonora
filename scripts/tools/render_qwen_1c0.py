"""Phase 1S Steps 1c-0 and 1c-0b: Qwen VoiceDesign probes, rendered and measured.

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § 1c-0 (owner, 2026-10-21) and
§ 1c-0b (owner, 2026-10-25). `--probe` picks the plan; `1c0` is the default.

* **1c0, the anchor range calibration (60 clips).** The owner hears a pure-pitch ladder (7 `pitch:`
  rungs per gender, every other key fixed) and six niche personas, three authored sentences each,
  and marks every clip core / niche / exclude. From those verdicts come the measured F0 bands that
  gate any Qwen anchor, the qwen.md casting rules, and 1c-i's two anchors.
* **1c0b, what moves Qwen's pitch (66 clips).** Per gender, 1c-0's rung 4 as a baseline and five
  methods (numeric pitch, descriptive pitch, age, vocal weight, a free-text instruct) at a low and a
  high pitch pole, three seeds each, all reading 1c-0's sentence 2. `steer` reads the measured
  folder: a method steers when every high-pole clip's F0 median is above every low-pole clip's and
  the mean shift is at least `STEER_SHIFT_ST` semitones.

This tool renders and measures the clips and applies 1c-0b's registered reading; what follows from
them is the owner's. The commands below are 1c-0's; 1c-0b's follow them.

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

    # a subset of a plan, e.g. 1c-i's anchors from 1c-0's: `render` and `measure` with
    # `--ids M4-s3 F4-s3 F6-s2` after `--out`, into a new folder of their own
    # 1c-0b: the same plan, install -d, render and measure steps with `--probe 1c0b` after each
    # subcommand and probes/qwen-1c0b in place of probes/qwen-1c0 (the install -d, the render's
    # --out, capture_env_cmd's name and _env path, the measure's --out); then, on the host, the
    # reading (numpy only), printed and written to steer.json beside the measures:
    .venv/bin/python scripts/tools/render_qwen_1c0.py steer \\
        --out /data/model-training/datasets/probes/qwen-1c0b

The probe folder is what the audition app's probe mode reads (`audition/app/main.py`,
`_load_probe_manifest`): `$AUDITION_DATA_ROOT/probes/<name>/` (by default
/data/model-training/datasets/probes/qwen-1c0), its `manifest.jsonl` in listening order with each
clip at `clips/<id>.wav`, its `measures.jsonl` joined to the manifest by `id` and shown as the
clip's `acoustics`, and its `render_meta.json`. The card shows the row's `gender` (the prompted one)
and, from `acoustics`, `f0_median_hz`, `f0_excursion_hz` and `hnr_db`; `cpp_db` rides along unshown.
The owner's verdicts go to that folder's `verdicts.csv`. Nothing here touches `ratings.csv`. Nothing
here pre-fills an attribute from the prompt either: a pitch value taken from the rung would tell the
listener the answer. The row's `instruct` stays in the manifest (and in the API's clip JSON) but the
card does not render it, for the same reason.

⚠ THE STEP 1B SPEAKER EMBEDDING IS A SEPARATE STEP. The spec's per-clip measures include the Zonos
`SpeakerEmbeddingLDA` embedding; this tool does not compute it and takes no Zonos dependency. The
Step 1b extractor computes it, on CPU in an environment with torch and /data/toolchain/Zonos:
`/data/model-training/sonora/analysis/step1b_speaker_consistency_2026-10-21/features.py`. Its
inputs are Step 1b's campaigns, so point it at this probe folder's clips. 1c-i's anchor choice
needs that embedding beside the verdicts and these measures. Its rows (`id`, `emb`) in the folder's
`speaker_embeddings.jsonl` give `steer` its pole cosine; without that file the cosine is null.

⚠ THE WAVEFORM DECODER RUNS IN FLOAT32. `render` loads the model through `qwen_load.load_qwen`,
whose docstring says why: the language model in bfloat16, the speech tokenizer in float32, any
other decoder dtype refused. `render_meta.json` and every manifest row record
`speech_tokenizer_dtype`, so a folder whose meta does not (one rendered with a bf16 decoder) refuses
a resume.

⚠ BATCH SIZE 1, ONE SEED PER CLIP. `qwen_tts` has no seed argument; a clip is reproducible only
from `torch.manual_seed(seed)` immediately before its own call. A batched call draws every clip
from one RNG stream, so the registered per-clip seeds would no longer name the audio.

⚠ A RESUME MUST NOT MIX TWO RENDERS. `render_meta.json` is written before the first clip. With a
manifest already present, `render` refuses unless that meta's commit, weights hashes
(`model.safetensors` and every file under `speech_tokenizer/`), qwen-tts version, speech tokenizer
dtype, probe and id subset equal this run's (`RESUME_KEYS`), and unless every rendered row is its
plan item (seed, text, instruct, gender, commit) with its wav on disk. Only a truncated final
manifest line, a render killed mid-write, is dropped (with a warning) and re-rendered. When nothing
remains it exits before loading the model.

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
whose p10-p90 is the excursion. The hop is `PYIN_HOP_S`, 10 ms. Over the frames the 50-700 Hz pass
voices, each row reports:

* `f0_edge_frac`: the fraction within `EDGE_TOL` of 50 or 700 Hz. A reading at a limit is the limit,
  not the voice: a voice below 50 Hz reads 50 Hz.
* `f0_above_range_frac`: the fraction that a second pYIN pass, its ceiling an octave higher
  (`F0_EXTENDED_FMAX`), on the same frames also voices and reads above 700 Hz. The 50-700 Hz pass
  reads a voice above 700 Hz at half its F0, off both edges, so only the second pass sees it. A frame
  only the second pass voices does not count: on real Qwen clips that pass voices low-probability
  noise near 1 kHz the first pass leaves unvoiced (a real clip at 246 Hz read 0.22 above range
  counting it, 0 without). A steady tone above 700 Hz that the first pass voices at half still
  counts, at any level: 0.3 s of a 980 Hz tone at -20 dB, in noise 10 dB below it, after 1.2 s of a
  300 Hz voice reads about 0.13 and is flagged.
* `f0_out_of_range`: either fraction above `OUT_OF_RANGE_FRAC`. The clip's F0 is then not measured:
  its row nulls `f0_median`, `f0_p10`, `f0_p90`, `f0_excursion`, `f0_excursion_st` and their `_hz`
  copies, so a band derived from the rows cannot take in a misread.
* `f0_median_acf` and `f0_methods_disagree`: the repo autocorrelation's median over 50-350 Hz, and
  whether it is more than `METHODS_DISAGREE_ST` from pYIN's. pYIN reads a voice whose second
  harmonic exceeds its first by about 16 dB an octave up (a synthetic 220 Hz voice reads 441 Hz;
  from about 350 Hz only the extended pass does, which sets `f0_out_of_range` on a voice in range).
  The autocorrelation reads such a 220 Hz voice at its F0, so that misread is visible. It does not
  catch every octave error: at 110 Hz with the second harmonic 14 dB over the first, pYIN reads an
  octave up, about 220 Hz, and so does the autocorrelation, so `f0_methods_disagree` stays False.
  Above 350 Hz the autocorrelation reads a subharmonic and the two disagree by construction; there a
  flagged clip needs the ear.

`--ids` RENDERS PART OF A PLAN. `render --ids ID [ID ...]` renders only the named items of the
`--probe` plan, in plan order, with their registered seeds and instructs; an id not in the plan, or
one named twice, refuses. `render_meta.json` records the subset as `ids` in plan order (null for the
whole plan), so a whole-plan folder refuses a subset resume and a subset folder refuses a whole-plan
or another subset's resume. An `--ids` that names every plan id IS the whole plan and is recorded as
null, so it resumes, measures and steers as a whole-plan folder does. A `render_meta.json` without an
`ids` key (one written before `--ids` existed, such as the 1c-0 probe's) reads as the whole plan.
`measure` takes the same `--ids` and refuses a folder whose `render_meta.json` records another subset
or the whole plan; a clip missing from the subset refuses, and the rest of the plan is not looked
for. `steer` reads a whole 1c0b render only.

`measure` also places each clip in Qwen's F0 admission bands (`scripts/lib/qwen_bands.py`): `zone`
is `qwen_bands.admit` on the row's `f0_median_hz` and `f0_excursion_hz` for the manifest row's
`gender` (the prompted one), "core", "niche" or "unmeasured"; a clip whose gender is not "Male" or
"Female" is "unmeasured". `zone_provisional` is `qwen_bands.provisional` for that zone (False for
"unmeasured"). It is a measure, not a verdict: the owner's zone is in `verdicts.csv`. `measure` prints
the count of each zone, and `measure_meta.json` records the bands' `VERDICT_ZONES_SHA256` as
`zone_verdicts_sha256`.

`measure` writes `measure_meta.json` beside `measures.jsonl`: the measuring commit (`git rev-parse
HEAD` of this tool's checkout) and whether that tree is clean (`git status --porcelain` empty), the
probe and id subset, the render's commit, the librosa and numpy versions, every F0 and HNR setting
above, and the UTC time. `steer` refuses a folder whose `measure_meta.json` is missing, records a
probe other than 1c0b, or measured another render (its `render_commit` is not `render_meta.json`'s
`commit`). It writes `steer.json` beside them: its own commit and cleanliness, `render_commit` (the
render's commit), `measure_commit` (`measure_meta.json`'s `commit`), `STEER_SHIFT_ST`, and
`steer_readout`'s rows.

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
RESUME_KEYS = ("commit", "weights_sha256", "qwen_tts", "speech_tokenizer_dtype", "probe", "ids")
PROBES = ("1c0", "1c0b")

# F0 (module docstring): pYIN's search range (Hz), and the extended pass that sees a voice above it.
F0_RANGE = (50.0, 700.0)
F0_EXTENDED_FMAX = 2 * F0_RANGE[1]
PYIN_HOP_S = 0.010
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

# 1c-0b: every clip reads 1c-0's sentence 2, three seeds per instruct, against 1c-0's rung 4.
STEER_SENTENCE = 2
STEER_SEEDS = 3
BASELINE_RUNG = 4
# A method steers on complete separation of its poles and a mean shift of at least this.
STEER_SHIFT_ST = 4.0
# 1c-0b's methods in listening order: (id part, method, the key the poles set or None for a
# free-text instruct that replaces the 12 keys, {gender: (low pole, high pole)}). Every other key
# is the baseline's. The poles are pitch poles: Qwen renders younger voices higher (§ 1c-0), so
# `Elderly` is age's low pole and `Young` its high one.
STEER_METHODS = (
    ("num", "numeric pitch", "pitch",
     {"Male": ("Very low, around 80 Hz, stable", "Very high, around 160 Hz, stable"),
      "Female": ("Very low, around 150 Hz, stable", "Very high, around 300 Hz, stable")}),
    ("desc", "descriptive pitch", "pitch",
     {"Male": ("Deep bass, far below an average man's voice, stable",
               "Light and high, far above an average man's voice, stable"),
      "Female": ("Deep bass, far below an average woman's voice, stable",
                 "Light and high, far above an average woman's voice, stable")}),
    ("age", "age", "age",
     {"Male": ("Elderly man", "Young man"), "Female": ("Elderly woman", "Young woman")}),
    ("weight", "vocal weight", "texture",
     dict.fromkeys(("Male", "Female"),
                   ("Deep, heavy and chest-resonant", "Light, thin and head-voiced"))),
    ("free", "free-text instruct", None,
     {g: ("%s with a very deep, low-pitched voice, calm and composed, reading narration in "
          "American English." % who,
          "%s with an unusually high-pitched voice, calm and composed, reading narration in "
          "American English." % who)
      for g, who in (("Male", "A man"), ("Female", "A woman"))}),
)
POLES = ("low", "high")


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
                              "gender": gender, "seed": int("9%d%d%d" % (gd, r, s)),
                              "sentence": s, "text": text,
                              "instruct": build_instruct(keys)})
    for n, p in enumerate(PERSONAS, 1):
        keys = dict(FIXED, **{k: p[k] for k in ("gender", "pitch", "age", "texture")})
        for s, text in enumerate(SENTENCES, 1):
            items.append({"id": "%s-s%d" % (p["id"], s), "group": "niche personas",
                          "rung": None, "label": p["label"], "band": p["band"],
                          "gender": p["gender"], "seed": int("93%d%d" % (n, s)),
                          "sentence": s, "text": text,
                          "instruct": build_instruct(keys)})
    return items


def plan_1c0b():
    """1c-0b's 66 items in listening order: Male then Female; within each, the baseline (1c-0's rung
    4 instruct), then every method's low then high pole; seeds 1..3 each, all reading sentence 2.
    Seeds: 95GMPK (G 1 = M / 2 = F, method M 0 = baseline or 1-5, pole P 0 = baseline, 1 = low,
    2 = high, seed K)."""
    items = []
    for g, gender, word, gd in GENDERS:
        base = dict(FIXED, gender=gender,
                    pitch="%s %s pitch, stable" % (RUNGS[BASELINE_RUNG - 1], word))
        arms = [("base", "baseline", None, 0, 0, "baseline (1c-0 rung %d)" % BASELINE_RUNG,
                 build_instruct(base))]
        for m, (part, method, key, poles) in enumerate(STEER_METHODS, 1):
            for p, (pole, value) in enumerate(zip(POLES, poles[gender]), 1):
                arms.append(("%s-%s" % (part, pole[:2]), method, pole, m, p,
                             "%s, %s" % (method, pole),
                             value if key is None else build_instruct(dict(base, **{key: value}))))
        for arm, method, pole, m, p, label, instruct in arms:
            for k in range(1, STEER_SEEDS + 1):
                items.append({"id": "%s-%s-k%d" % (g, arm, k), "group": "%s 1c-0b" % gender,
                              "method": method, "pole": pole, "label": label,
                              "band": "Man" if g == "M" else "Woman", "gender": gender,
                              "seed": int("95%d%d%d%d" % (gd, m, p, k)),
                              "sentence": STEER_SENTENCE, "text": SENTENCES[STEER_SENTENCE - 1],
                              "instruct": instruct})
    return items


def plan_items(probe):
    """The probe's items in listening order."""
    return {"1c0": ladder, "1c0b": plan_1c0b}[probe]()


def select_ids(items, ids):
    """`items` restricted to `ids`, in plan order; None keeps them all. Refuses an id that is not in
    `items` and an id named twice."""
    if ids is None:
        return items
    known = {it["id"] for it in items}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise SystemExit("REFUSING: --ids %s %s not in the plan." % (
            ", ".join(unknown), "is" if len(unknown) == 1 else "are"))
    if len(set(ids)) != len(ids):
        raise SystemExit("REFUSING: --ids names %s more than once."
                         % ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    return [it for it in items if it["id"] in set(ids)]


def subset_ids(items, ids):
    """The ids `render_meta.json` records for `--ids`: plan order, or None for the whole plan. An
    `ids` that names every item is the whole plan, so it is None too: a folder's record does not
    depend on whether the whole plan was named or implied."""
    if ids is None:
        return None
    subset = [it["id"] for it in select_ids(items, ids)]
    return None if len(subset) == len(items) else subset


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


def _pyin(x, sr, fmax):
    """librosa's pYIN from the F0 floor to `fmax` -> (F0 in Hz, voiced) for every frame. The frame
    and hop do not depend on `fmax`, so two passes share one frame grid."""
    import librosa
    import numpy as np
    f0, voiced, _prob = librosa.pyin(x.astype(np.float32), fmin=F0_RANGE[0], fmax=fmax, sr=sr,
                                     frame_length=pyin_frame(sr), hop_length=int(PYIN_HOP_S * sr))
    return f0, voiced & np.isfinite(f0)


def f0_profile(x, sr):
    """Voiced-frame F0 of a mono clip from pYIN (module docstring): median, p10, p90, excursion in
    Hz and semitones, the edge and above-range fractions, the out-of-range flag, the autocorrelation
    median it is audited against, and median HNR. None if pYIN finds no voiced frame. The F0
    statistics are what the 50-700 Hz pass read even when the flag is set; `measure_row` nulls them."""
    import numpy as np
    from periodicity import frames
    x = np.asarray(x, dtype=np.float64)
    track, voiced = _pyin(x, sr, F0_RANGE[1])
    if not voiced.any():
        return None
    wide, wide_voiced = _pyin(x, sr, F0_EXTENDED_FMAX)
    if len(wide) != len(track):
        raise ValueError("the two pYIN passes are on different frame grids (%d, %d frames)"
                         % (len(track), len(wide)))
    f0 = track[voiced]
    acf, hnr = frames(x, sr, ACF_BAND[0], ACF_BAND[1], RMS_FLOOR, PERIODICITY)
    lo, hi = F0_RANGE
    median = float(np.median(f0))
    p10, p90 = float(np.percentile(f0, 10)), float(np.percentile(f0, 90))
    edge = float(np.mean((f0 < lo * (1 + EDGE_TOL)) | (f0 > hi / (1 + EDGE_TOL))))
    with np.errstate(invalid="ignore"):         # NaN in the unvoiced frames compares False
        above = float(np.mean(wide_voiced[voiced] & (wide[voiced] > hi)))
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


def check_rows(rows, out, items, commit=None):
    """Refuse a manifest row that is not its item in `items` (the probe's plan), repeats an id, or
    has no wav."""
    items = {it["id"]: it for it in items}
    seen = set()
    for r in rows:
        it = items.get(r.get("id"))
        if it is None or r["id"] in seen:
            raise SystemExit("REFUSING: manifest id %r is %s." % (
                r.get("id"), "repeated" if it is not None else "not a plan id"))
        seen.add(r["id"])
        bad = [k for k in ("seed", "text", "instruct", "gender") if r.get(k) != it[k]]
        bad += ["file"] if r.get("file") != "clips/%s.wav" % r["id"] else []
        bad += ["commit"] if commit is not None and r.get("commit") != commit else []
        if bad:
            raise SystemExit("REFUSING: manifest row %s differs from its plan item in %s."
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
    check_rows(rows, out, select_ids(plan_items(meta["probe"]), meta["ids"]),
               commit=meta["commit"])
    if dropped is not None:
        print("WARNING: dropping a truncated final manifest line (%d chars); that clip is "
              "rendered again." % len(dropped), flush=True)
    if repaired is not None:
        write_text_atomic(manifest, repaired)
    return {r["id"] for r in rows}


def cmd_plan(args):
    items = plan_items(args.probe)
    if args.probe == "1c0":
        for it in items:
            print("%-7s seed %d  %-16s %-13s %s" % (it["id"], it["seed"], it["label"], it["band"],
                                                     it["instruct"][:80]))
    else:
        # 1c-0b's instructs differ past their first 80 characters, so they print whole.
        for it in items:
            print("%-14s seed %d  %-26s %s" % (it["id"], it["seed"], it["label"], it["instruct"]))
    print("%d items" % len(items))


def cmd_render(args):
    import importlib.metadata

    from loudness import match_loudness
    from qwen_load import TOKENIZER_DTYPE, load_qwen
    from synth_common import write_json_atomic, write_wav_atomic

    plan = plan_items(args.probe)
    items = select_ids(plan, args.ids)
    version = require_qwen_tts(importlib.metadata.version)
    # Written before the model loads; load_qwen refuses any decoder dtype but TOKENIZER_DTYPE, so
    # the meta's speech_tokenizer_dtype holds for every clip rendered under it.
    meta = {"tool": TOOL, "probe": args.probe, "commit": args.commit, "model_dir": MODEL_DIR,
            "qwen_tts": version, "speech_tokenizer_dtype": TOKENIZER_DTYPE,
            "weights_sha256": weights_sha256(MODEL_DIR), "lufs": LUFS, "batch_size": 1,
            "language": LANGUAGE, "ids": subset_ids(plan, args.ids), "items": len(items)}
    done = resume_state(args.out, meta)
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print("all %d clips already rendered; the model was not loaded." % len(done))
        print("RENDER-QWEN-1C0-DONE")
        return
    clips = os.path.join(args.out, "clips")
    os.makedirs(clips, exist_ok=True)
    if not done:
        write_json_atomic(os.path.join(args.out, "render_meta.json"), meta)

    import torch

    model = load_qwen(MODEL_DIR)
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
                       peak_limited=bool(bound), commit=args.commit,
                       speech_tokenizer_dtype=TOKENIZER_DTYPE)
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            print("%-7s %5.1fs audio  %5.1fs wall" % (it["id"], row["seconds"], wall), flush=True)
    print("RENDER-QWEN-1C0-DONE")


F0_KEYS = ("f0_median", "f0_p10", "f0_p90", "f0_excursion", "f0_excursion_st", "voiced_frames",
           "f0_range_hz", "f0_edge_frac", "f0_above_range_frac", "f0_out_of_range",
           "f0_median_acf", "f0_methods_disagree", "hnr")
# The F0 statistics a row nulls when `f0_out_of_range` is set.
F0_STATS = ("f0_median", "f0_p10", "f0_p90", "f0_excursion", "f0_excursion_st")
# The audition app's probe card reads these names from `acoustics` (module docstring); each is a
# copy of the row's own key, which stays.
READER_ALIASES = (("f0_median_hz", "f0_median"), ("f0_excursion_hz", "f0_excursion"),
                  ("hnr_db", "hnr"), ("cpp_db", "cpp"))


def measure_row(clip_id, x, sr):
    """One `measures.jsonl` row for a mono clip: F0, HNR and phonation, the F0 statistics nulled
    when `f0_out_of_range` is set, and the audition app's aliases."""
    f0 = f0_profile(x, sr)
    row = dict({"id": clip_id}, **(f0 or dict.fromkeys(F0_KEYS)), hnr_note=HNR_NOTE,
               **phonation_row(x, sr, f0))
    if f0 is not None and f0["f0_out_of_range"]:
        row.update(dict.fromkeys(F0_STATS))
    row.update({alias: row[key] for alias, key in READER_ALIASES})
    return row


def admission(gender, row):
    """{"zone", "zone_provisional"} for a `measure_row` row: `qwen_bands.admit` on its
    `f0_median_hz` and `f0_excursion_hz` for `gender`, the manifest row's prompted gender. A gender
    that is not one of `qwen_bands.GENDERS` (missing, or anything else) is "unmeasured", not an
    error."""
    import qwen_bands

    if gender not in qwen_bands.GENDERS:
        return {"zone": qwen_bands.UNMEASURED, "zone_provisional": False}
    zone = qwen_bands.admit(gender, row["f0_median_hz"], row["f0_excursion_hz"])
    return {"zone": zone, "zone_provisional": qwen_bands.provisional(gender, zone)}


def tool_commit():
    """{"commit": this checkout's HEAD, "tree_clean": `git status --porcelain` empty}, each None
    when git cannot say."""
    import subprocess

    def git(*argv):
        try:
            return subprocess.run(["git", "-C", _SONORA_REPO, *argv], capture_output=True,
                                  text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": head.strip() if head else None,
            "tree_clean": None if status is None else not status.strip()}


def measure_meta(render_commit, srs, probe="1c0", ids=None):
    """`measure_meta.json`: the code, library versions and settings that produced the measures;
    `ids` is the render's id subset (None for the whole plan)."""
    import datetime

    import derive_vat_corpus as dvc
    import librosa
    import numpy as np
    import qwen_bands

    return {"tool": TOOL, **tool_commit(), "probe": probe, "ids": ids,
            "zone_verdicts_sha256": qwen_bands.VERDICT_ZONES_SHA256,
            "render_commit": render_commit, "librosa": librosa.__version__, "numpy": np.__version__,
            "f0_range_hz": list(F0_RANGE), "f0_extended_fmax_hz": F0_EXTENDED_FMAX,
            "pyin_frame_samples": {str(sr): pyin_frame(sr) for sr in sorted(srs)},
            "pyin_hop_s": PYIN_HOP_S, "edge_tol": EDGE_TOL, "out_of_range_frac": OUT_OF_RANGE_FRAC,
            "methods_disagree_st": METHODS_DISAGREE_ST, "acf_band_hz": list(ACF_BAND),
            "acf_rms_floor": RMS_FLOOR, "acf_periodicity": PERIODICITY,
            "phonation_f0_band_hz": [dvc.F0_MIN, dvc.F0_MAX],
            "measured_utc":
                datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def rendered(out, probe, ids=None):
    """-> (the manifest's rows in plan order, render_meta.json) of a finished render of `probe`, or
    of its `ids` subset, after refusing a missing or truncated manifest, a meta without a commit,
    recording another probe or another id subset, a row that is not its item, and a missing clip. A
    meta that records no probe is checked by its rows alone: the two plans share no id."""
    manifest = os.path.join(out, "manifest.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit("REFUSING: no manifest at %s; render first." % manifest)
    rows, _repaired, dropped = read_manifest(manifest)
    if dropped is not None:
        raise SystemExit("REFUSING: %s ends in a truncated line; resume the render." % manifest)
    try:
        with open(os.path.join(out, "render_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        commit = meta["commit"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit("REFUSING: %s has no readable render_meta.json commit (%r)." % (out, e))
    if meta.get("probe", probe) != probe:
        raise SystemExit("REFUSING: %s's render_meta.json records probe %s, not %s."
                         % (out, meta["probe"], probe))
    plan = plan_items(probe)
    subset = subset_ids(plan, ids)
    if meta.get("ids") != subset:
        raise SystemExit("REFUSING: %s's render_meta.json records %s, not %s; measure the ids it "
                         "rendered." % (out, _ids_text(meta.get("ids")), _ids_text(subset)))
    items = select_ids(plan, ids)
    check_rows(rows, out, items, commit=commit)
    want = [it["id"] for it in items]
    if {r["id"] for r in rows} != set(want):
        raise SystemExit("REFUSING: %d of %d %s clips rendered (missing %s)." % (
            len(rows), len(want), probe, sorted(set(want) - {r["id"] for r in rows})))
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in want], meta


def _ids_text(ids):
    return "the whole plan" if ids is None else "--ids " + " ".join(ids)


def cmd_measure(args):
    import soundfile as sf

    from synth_common import write_json_atomic, write_text_atomic

    rows, meta = rendered(args.out, args.probe, args.ids)
    out, srs = [], set()
    for r in rows:
        x, sr = sf.read(os.path.join(args.out, r["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        srs.add(sr)
        row = measure_row(r["id"], x, sr)
        out.append(dict(row, **admission(r.get("gender"), row)))
    write_text_atomic(os.path.join(args.out, "measures.jsonl"),
                      "".join(json.dumps(o) + "\n" for o in out))
    write_json_atomic(os.path.join(args.out, "measure_meta.json"),
                      measure_meta(meta["commit"], srs, args.probe, meta.get("ids")))
    print("%d clips measured; %d without a voiced frame; %d without a phonation measure; %d with "
          "F0 out of pYIN's %g-%g Hz range; %d where pYIN and the autocorrelation disagree" % (
              len(out), sum(o["voiced_frames"] is None for o in out),
              sum(o["cpp"] is None for o in out), sum(bool(o["f0_out_of_range"]) for o in out),
              F0_RANGE[0], F0_RANGE[1], sum(bool(o["f0_methods_disagree"]) for o in out)))
    zones = [o["zone"] for o in out]
    print("zones: %d core, %d niche (%d provisional), %d unmeasured" % (
        zones.count("core"), zones.count("niche"), sum(o["zone_provisional"] for o in out),
        zones.count("unmeasured")))


def _cosine(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def steer_readout(rows, measures, embeddings=None):
    """1c-0b's reading (spec § 1c-0b) from its manifest rows, {id: measures row} and optionally
    {id: speaker embedding}. Per gender, the baseline's F0 medians by seed and their median; per
    gender and method, the low- and high-pole F0 medians by seed, `complete_separation` (every
    high-pole median above every low-pole one), `mean_shift_st` (the mean F0 in semitones re 1 Hz
    of the high-pole clips measured minus that of the low-pole clips measured, i.e. 12 log2 of the
    ratio of their geometric means; a pole's seeds are independent draws, so no low clip pairs with
    a high one) and `steers` (complete separation and a mean shift of at least STEER_SHIFT_ST). A
    clip whose F0 is not measured (null `f0_median`) separates nothing and is listed in
    `unmeasured`. `pole_cosine` is the mean embedding cosine over the 3 x 3 low-high pairs, None
    without embeddings."""
    import numpy as np

    out = {"baseline": {}, "methods": []}
    for _g, gender, _word, _gd in GENDERS:
        mine = sorted((r for r in rows if r["gender"] == gender), key=lambda r: r["seed"])
        f0 = {r["id"]: measures[r["id"]].get("f0_median") for r in mine}
        base = [r["id"] for r in mine if r["method"] == "baseline"]
        got = [f0[i] for i in base if f0[i] is not None]
        out["baseline"][gender] = {
            "ids": base, "f0_median_hz": [f0[i] for i in base],
            "median_hz": float(np.median(got)) if got else None,
            "unmeasured": [i for i in base if f0[i] is None]}
        for _part, method, _key, _poles in STEER_METHODS:
            lo, hi = ([r["id"] for r in mine if r["method"] == method and r["pole"] == pole]
                      for pole in POLES)
            lo_f0, hi_f0 = [f0[i] for i in lo], [f0[i] for i in hi]
            separated = None not in lo_f0 + hi_f0 and min(hi_f0) > max(lo_f0)
            lo_st, hi_st = ([12.0 * math.log2(v) for v in pole if v is not None]
                            for pole in (lo_f0, hi_f0))
            shift = float(np.mean(hi_st) - np.mean(lo_st)) if lo_st and hi_st else None
            cosine = None
            if embeddings is not None:
                cosine = float(np.mean([_cosine(embeddings[a], embeddings[b])
                                        for a in lo for b in hi]))
            out["methods"].append({
                "gender": gender, "method": method, "low_ids": lo, "high_ids": hi,
                "low_f0_median_hz": lo_f0, "high_f0_median_hz": hi_f0,
                "complete_separation": separated, "mean_shift_st": shift,
                "steers": separated and shift is not None and shift >= STEER_SHIFT_ST,
                "unmeasured": [i for i in lo + hi if f0[i] is None], "pole_cosine": cosine})
    return out


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_steer(args):
    from synth_common import write_json_atomic

    try:
        with open(os.path.join(args.out, "render_meta.json"), encoding="utf-8") as fh:
            probe = json.load(fh).get("probe")
    except (OSError, ValueError, AttributeError) as e:
        raise SystemExit("REFUSING: %s has no readable render_meta.json (%r)." % (args.out, e))
    if probe != "1c0b":
        raise SystemExit("REFUSING: %s's render_meta.json records probe %s; steer reads a 1c0b "
                         "render." % (args.out, probe))
    rows, meta = rendered(args.out, "1c0b")
    path = os.path.join(args.out, "measures.jsonl")
    measures = {m["id"]: m for m in _read_jsonl(path)} if os.path.exists(path) else {}
    missing = [r["id"] for r in rows if r["id"] not in measures]
    if missing:
        raise SystemExit("REFUSING: %s lacks %d of %d clips (%s); measure first."
                         % (path, len(missing), len(rows), missing[:3]))
    try:
        with open(os.path.join(args.out, "measure_meta.json"), encoding="utf-8") as fh:
            measured = json.load(fh)
        measured_probe, measured_render = measured.get("probe"), measured.get("render_commit")
    except (OSError, ValueError, AttributeError) as e:
        raise SystemExit("REFUSING: %s has no readable measure_meta.json (%r); measure first."
                         % (args.out, e))
    if measured_probe != "1c0b":
        raise SystemExit("REFUSING: %s's measure_meta.json records probe %s; steer reads 1c0b "
                         "measures." % (args.out, measured_probe))
    if measured_render != meta["commit"]:
        raise SystemExit("REFUSING: %s's measure_meta.json measured render %s, not "
                         "render_meta.json's %s; measure again."
                         % (args.out, measured_render, meta["commit"]))
    emb_path = os.path.join(args.out, "speaker_embeddings.jsonl")
    embeddings = None
    if os.path.exists(emb_path):
        embeddings = {e["id"]: e["emb"] for e in _read_jsonl(emb_path)}
        absent = [r["id"] for r in rows if r["id"] not in embeddings]
        if absent:
            raise SystemExit("REFUSING: %s lacks %d of %d clips (%s)."
                             % (emb_path, len(absent), len(rows), absent[:3]))
    res = steer_readout(rows, measures, embeddings)

    def hz(values):
        return " ".join("null" if v is None else "%.1f" % v for v in values)
    for gender, b in res["baseline"].items():
        print("%-6s baseline           F0 %s Hz, median %s" % (
            gender, hz(b["f0_median_hz"]),
            "null" if b["median_hz"] is None else "%.1f Hz" % b["median_hz"]))
    print("%-6s %-18s %-20s %-20s %8s %9s %6s %6s" % (
        "gender", "method", "low F0 (Hz)", "high F0 (Hz)", "shift st", "separated", "steers",
        "cosine"))
    for m in res["methods"]:
        print("%-6s %-18s %-20s %-20s %8s %9s %6s %6s" % (
            m["gender"], m["method"], hz(m["low_f0_median_hz"]), hz(m["high_f0_median_hz"]),
            "null" if m["mean_shift_st"] is None else "%+.2f" % m["mean_shift_st"],
            m["complete_separation"], m["steers"],
            "null" if m["pole_cosine"] is None else "%.3f" % m["pole_cosine"]))
    unmeasured = [i for m in res["methods"] for i in m["unmeasured"]] + \
        [i for b in res["baseline"].values() for i in b["unmeasured"]]
    if unmeasured:
        print("F0 not measured (out of range or unvoiced), so not separating: %s"
              % ", ".join(unmeasured))
    if embeddings is None:
        print("no speaker_embeddings.jsonl: the pole cosine is null")
    write_json_atomic(os.path.join(args.out, "steer.json"), dict(
        {"tool": TOOL}, **tool_commit(), render_commit=meta["commit"],
        measure_commit=measured.get("commit"), steer_shift_st=STEER_SHIFT_ST,
        speaker_embeddings=embeddings is not None, **res))
    print("%d of %d methods steer" % (sum(m["steers"] for m in res["methods"]),
                                      len(res["methods"])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    probe = dict(choices=PROBES, default="1c0", help="the plan (default 1c0)")
    ids = dict(nargs="+", metavar="ID", default=None,
               help="only these items of the plan, in plan order (default: all)")
    sub.add_parser("plan").add_argument("--probe", **probe)
    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--probe", **probe)
    r.add_argument("--ids", **ids)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    m.add_argument("--probe", **probe)
    m.add_argument("--ids", **ids)
    s = sub.add_parser("steer", help="1c-0b's reading of a measured 1c0b folder")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"plan": cmd_plan, "render": cmd_render, "measure": cmd_measure,
     "steer": cmd_steer}[args.cmd](args)


if __name__ == "__main__":
    main()

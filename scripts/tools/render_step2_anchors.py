"""Phase 1S Step 2a: the anchor pool for Arm Q's 85 new `qwen_base` voices.

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § Step 2 — Arm Q (owner, approved
2026-10-29): "Anchor pool for the 85 new qwen_base voices" and step 1 of the budget sequence.

* `plan` prints the 114 candidates: 57 Female and 57 Male grounded adult core-band VoiceDesign
  instructs. Deterministic; seeds 961000-961056 (Male) and 962000-962056 (Female).
* `render` renders them with Qwen3-TTS-12Hz-1.7B-VoiceDesign, batch size 1, one seed before each
  clip, -23.0 LUFS, into a probe folder `audition/app` reads (manifest.jsonl, clips/, render_meta.json).
* `measure` runs render_qwen_1c0's pYIN F0 and phonation measures and `qwen_bands.admit` on each clip.
  On a whole-plan render it also writes `qwen_voicedesign_manifest.jsonl`, the campaign manifest
  `qc_gate.py` reads (`wav` beside `file`).
* `embed` extracts Zonos SpeakerEmbeddingLDA and SpeechBrain ECAPA-TDNN embeddings of every rendered
  clip and of the locked anchors into `speaker_embeddings.jsonl`. It runs under the litert venv
  (`/data/toolchain/litert-conversion/.venv/bin/python`), which carries torch and speechbrain.
* `admit-filter` applies the three automated gates: QC hard-pass, core F0 band, and distinctness
  (cosine < 0.85 on both models against the locked anchors of the same gender and every earlier
  admitted candidate of that gender, in plan order). It writes `admission_report.json` and
  `manifest.json`, which the audition app serves: the admitted clips only. `manifest.jsonl` keeps
  all 114 raw clips. The owner's ear check (`hum == 0`, core verdict) is the fourth gate and happens
  outside this tool.

Commands (GPU render: run it in a JARVIS slot, as render_qwen_1c0.py documents):

    .venv/bin/python scripts/tools/render_step2_anchors.py plan
    install -d -m 2775 /data/model-training/datasets/probes/qwen-step2-anchors
    # container, from a clean checkout of the registered commit, as render_qwen_1c0.py's block:
    #   python /sonora/scripts/tools/render_step2_anchors.py render --commit <HEAD> \\
    #     --out /data/model-training/datasets/probes/qwen-step2-anchors
    .venv/bin/python scripts/tools/render_step2_anchors.py measure \\
        --out /data/model-training/datasets/probes/qwen-step2-anchors
    # the QC gate, against the campaign manifest measure wrote (GPU, ASR):
    #   python scripts/stages/qc_gate.py --campaign-dir /data/model-training/datasets/probes/qwen-step2-anchors
    /data/toolchain/litert-conversion/.venv/bin/python scripts/tools/render_step2_anchors.py embed \\
        --out /data/model-training/datasets/probes/qwen-step2-anchors
    .venv/bin/python scripts/tools/render_step2_anchors.py admit-filter \\
        --out /data/model-training/datasets/probes/qwen-step2-anchors

`speaker_embeddings.jsonl` rows are {"id", "zonos_lda": [floats], "ecapa": [floats]}: every rendered
clip, and the locked anchors M4-s3 and F6-s2.
"""
import argparse
import json
import math
import os
import posixpath
import sys as _sys  # noqa: E402
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib", "tools"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import render_qwen_1c0 as rq  # noqa: E402

TOOL = "scripts/tools/render_step2_anchors.py"
PROBE = "step2a"
# Seeds 96GIII: G 1 = Male, 2 = Female (as render_qwen_1c0's GENDERS), III the candidate's index.
SEED_BASE = 960000
PER_GENDER = 57
# The pre-registered cosine bound: a candidate is distinct only when every cosine is strictly below.
DISTINCT_COS = 0.85
EMBEDDING_KEYS = ("zonos_lda", "ecapa")
# The two locked anchors Step 1c-i and 1c-ii already use, with the gender each was prompted as.
LOCKED_ANCHORS = {"M4-s3": "Male", "F6-s2": "Female"}
# How many new anchors each gender needs (pre-registration: 44 Female and 43 Male rows with the
# locked anchors, so 43 and 42 new).
NEED = {"Female": 43, "Male": 42}
GENDERS = ("Male", "Female")

# Adult core-band age bands: (owner's band prefix, Qwen `age:` wording). An empty prefix is the
# middle band, which is the owner's "Woman" / "Man" taxonomy.
AGE_BANDS = (("Young", "Young adult, mid-twenties"), ("", "Middle-aged adult"),
             ("Elderly", "Older adult, around sixty"))
# Grounded `texture:` wordings, none of them a fairy or performed register (qwen.md).
TEXTURES = ("Clear and natural", "Warm and resonant", "Smooth and steady",
            "Slightly husky and grounded")
# Grounded (`tone`, `personality`) pairs, from qwen.md's neutral and grounded wordings.
TONES = (("Even, narratorial", "Composed and unobtrusive"),
         ("Calm and direct", "Matter-of-fact and steady"))
# Middle pitch rungs only: rung 3, 4 and 5. Rungs 1 and 7 are never used.
PITCH_RUNGS = ("Low", "Medium", "High")
_WORD = {"Male": "male", "Female": "female"}
_NOUN = {"Male": "man", "Female": "woman"}

# The embedding models and the audio root the audition app resolves links against (its DATA_ROOT).
ZONOS_PATH = "/data/toolchain/Zonos"
ECAPA_SNAP = (
    "/home/lmcfarlin/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb"
    "/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
)
ECAPA_EXPECTED_SHA256 = {
    "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
}
DEFAULT_ANCHORS_DIR = "/data/model-training/datasets/probes/qwen-1c0-anchors"
AUDITION_DATA_ROOT = "/data/model-training/datasets"
QC_MANIFEST = "qwen_voicedesign_manifest.jsonl"
SPEAKER_EMBEDDINGS = "speaker_embeddings.jsonl"


def grid_axes(i):
    """(age, pitch rung, texture, tone) indices of candidate `i`: mixed-radix over AGE_BANDS,
    PITCH_RUNGS, TEXTURES and TONES, so the first 3 * 3 * 4 * 2 = 72 indices are distinct
    combinations and age varies fastest."""
    age, rest = i % len(AGE_BANDS), i // len(AGE_BANDS)
    rung, rest = rest % len(PITCH_RUNGS), rest // len(PITCH_RUNGS)
    texture, rest = rest % len(TEXTURES), rest // len(TEXTURES)
    return age, rung, texture, rest % len(TONES)


def candidate_keys(gender, i):
    """The 12-key instruct's keys for candidate `i` of `gender`, its owner age band and pitch rung."""
    a, r, t, o = grid_axes(i)
    prefix, age = AGE_BANDS[a]
    rung = PITCH_RUNGS[r]
    tone, personality = TONES[o]
    keys = dict(rq.FIXED, gender=gender, age=age, texture=TEXTURES[t], tone=tone,
                personality=personality, pitch="%s %s pitch, stable" % (rung, _WORD[gender]))
    band = ("%s %s" % (prefix, _NOUN[gender])).strip().capitalize()
    return keys, band, rung


def sentence_index(i):
    """0-based index into rq.SENTENCES. Offsetting by the age band's cycle gives every age band all
    three sentences, not one each."""
    return (i + i // len(AGE_BANDS)) % len(rq.SENTENCES)


def plan_items():
    """The 114 candidates in plan order: Male then Female, index 1..57 within each."""
    items = []
    for g, gender, _word, gd in rq.GENDERS:
        for i in range(PER_GENDER):
            keys, band, rung = candidate_keys(gender, i)
            s = sentence_index(i)
            items.append({"id": "s2a-%s%02d" % (g, i + 1), "group": "step2 anchor pool",
                          "rung": None, "label": "%s, %s pitch" % (band, rung.lower()),
                          "band": band, "gender": gender,
                          "seed": SEED_BASE + 1000 * gd + i,
                          "sentence": s + 1, "text": rq.SENTENCES[s],
                          "instruct": rq.build_instruct(keys)})
    return items


def cmd_plan(args):
    items = plan_items()
    for it in items:
        print("%-8s seed %d  %-22s %s" % (it["id"], it["seed"], it["label"], it["instruct"]))
    print("%d items" % len(items))
    if args.json:
        from synth_common import write_json_atomic
        write_json_atomic(args.json, {"tool": TOOL, "probe": PROBE, "items": items})


def resume_state(out, meta, plan):
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
    differ = [k for k in rq.RESUME_KEYS if old.get(k) != meta[k]]
    if differ:
        raise SystemExit("REFUSING: resuming %s, but this run's %s differ from its "
                         "render_meta.json. Render into a new --out." % (out, ", ".join(differ)))
    rows, repaired, dropped = rq.read_manifest(manifest)
    rq.check_rows(rows, out, rq.select_ids(plan, meta["ids"]), commit=meta["commit"])
    if dropped is not None:
        print("WARNING: dropping a truncated final manifest line (%d chars); that clip is "
              "rendered again." % len(dropped), flush=True)
    if repaired is not None:
        write_text_atomic(manifest, repaired)
    return {r["id"] for r in rows}


def manifest_row(it, *, n_samples, sr, wall, peak_limited, commit, dtype):
    """The manifest row of one rendered clip. Its audio key is `file` alone, as render_qwen_1c0
    writes it. Do not add a `wav` key: audition/app/main.py's `_resolve_audio` resolves `wav`
    against RATINGS_DIR and DATA_ROOT only, so a `clips/...` value 404s there, and the probe
    fallback that finds `clips/<id>.wav` under the probe folder runs only when `wav` is absent."""
    return dict(it, file="clips/%s.wav" % it["id"], sr=sr, seconds=round(n_samples / sr, 3),
                wall_s=round(wall, 2), peak_limited=bool(peak_limited), commit=commit,
                speech_tokenizer_dtype=dtype)


def cmd_render(args):
    import importlib.metadata

    from loudness import match_loudness
    from qwen_load import TOKENIZER_DTYPE, load_qwen
    from synth_common import write_json_atomic, write_wav_atomic

    plan = plan_items()
    items = rq.select_ids(plan, args.ids)
    version = rq.require_qwen_tts(importlib.metadata.version)
    meta = {"tool": TOOL, "probe": PROBE, "commit": args.commit, "model_dir": rq.MODEL_DIR,
            "qwen_tts": version, "speech_tokenizer_dtype": TOKENIZER_DTYPE,
            "weights_sha256": rq.weights_sha256(rq.MODEL_DIR), "lufs": rq.LUFS, "batch_size": 1,
            "language": rq.LANGUAGE, "ids": rq.subset_ids(plan, args.ids), "items": len(items)}
    done = resume_state(args.out, meta, plan)
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print("all %d clips already rendered; the model was not loaded." % len(done))
        print("RENDER-STEP2-ANCHORS-DONE")
        return
    clips = os.path.join(args.out, "clips")
    os.makedirs(clips, exist_ok=True)
    if not done:
        write_json_atomic(os.path.join(args.out, "render_meta.json"), meta)

    import time

    import torch

    model = load_qwen(rq.MODEL_DIR)
    with open(os.path.join(args.out, "manifest.jsonl"), "a", encoding="utf-8") as mf:
        for it in todo:
            torch.manual_seed(it["seed"])
            t0 = time.monotonic()
            wavs, sr = model.generate_voice_design(text=it["text"], language=rq.LANGUAGE,
                                                   instruct=it["instruct"])
            wall = time.monotonic() - t0
            audio, bound = match_loudness(wavs[0], sr, rq.LUFS)
            write_wav_atomic(os.path.join(clips, it["id"] + ".wav"), audio, sr)
            row = manifest_row(it, n_samples=len(audio), sr=sr, wall=wall, peak_limited=bound,
                               commit=args.commit, dtype=TOKENIZER_DTYPE)
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            print("%-8s %5.1fs audio  %5.1fs wall" % (it["id"], row["seconds"], wall), flush=True)
    print("RENDER-STEP2-ANCHORS-DONE")


def rendered(out, ids=None):
    """-> (the manifest's rows in plan order, render_meta.json) of a finished render, or of its
    `ids` subset. Refuses a missing or truncated manifest, another probe or id subset, and any row
    that is not its plan item."""
    manifest = os.path.join(out, "manifest.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit("REFUSING: no manifest at %s; render first." % manifest)
    rows, _repaired, dropped = rq.read_manifest(manifest)
    if dropped is not None:
        raise SystemExit("REFUSING: %s ends in a truncated line; resume the render." % manifest)
    try:
        with open(os.path.join(out, "render_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        commit = meta["commit"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit("REFUSING: %s has no readable render_meta.json commit (%r)." % (out, e))
    if meta.get("probe") != PROBE:
        raise SystemExit("REFUSING: %s's render_meta.json records probe %s, not %s."
                         % (out, meta.get("probe"), PROBE))
    plan = plan_items()
    if meta.get("ids") != rq.subset_ids(plan, ids):
        raise SystemExit("REFUSING: %s's render_meta.json records ids %s; measure those ids."
                         % (out, meta.get("ids")))
    items = rq.select_ids(plan, ids)
    rq.check_rows(rows, out, items, commit=commit)
    want = [it["id"] for it in items]
    if {r["id"] for r in rows} != set(want):
        raise SystemExit("REFUSING: %d of %d clips rendered (missing %s)." % (
            len(rows), len(want), sorted(set(want) - {r["id"] for r in rows})))
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in want], meta


def qc_manifest_rows(rows):
    """The campaign manifest `qc_gate.py` reads: each manifest row with `wav` beside `file`. qc_gate
    resolves `wav` against the campaign dir, which is this probe's folder."""
    return [dict(r, wav=r["file"]) for r in rows]


def cmd_measure(args):
    import soundfile as sf

    from synth_common import write_json_atomic, write_text_atomic

    rows, meta = rendered(args.out, args.ids)
    out, srs = [], set()
    for r in rows:
        x, sr = sf.read(os.path.join(args.out, r["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        srs.add(sr)
        row = rq.measure_row(r["id"], x, sr)
        out.append(dict(row, gender=r.get("gender"), **rq.admission(r.get("gender"), row)))
    write_text_atomic(os.path.join(args.out, "measures.jsonl"),
                      "".join(json.dumps(o) + "\n" for o in out))
    if args.ids is None:
        write_text_atomic(os.path.join(args.out, QC_MANIFEST),
                          "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in qc_manifest_rows(rows)))
    m = rq.measure_meta(meta["commit"], srs, PROBE, meta.get("ids"))
    write_json_atomic(os.path.join(args.out, "measure_meta.json"), dict(m, tool=TOOL))
    zones = [o["zone"] for o in out]
    print("%d clips measured; %d with F0 out of pYIN's range; zones: %d core, %d niche, "
          "%d unmeasured" % (len(out), sum(bool(o["f0_out_of_range"]) for o in out),
                             zones.count("core"), zones.count("niche"),
                             zones.count("unmeasured")))


def verify_ecapa_hashes(snap, expected):
    """Refuses an ECAPA checkpoint whose bytes differ from the pinned sha256 (as the 1c-i extraction
    does), so an embedding never comes from an unpinned model."""
    import hashlib

    for fname, want in expected.items():
        with open(os.path.join(snap, fname), "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            raise SystemExit("REFUSING: ECAPA %s sha256 %s, pinned %s." % (fname, got, want))


def embed_clips(rows, out, anchors_dir):
    """-> [(id, wav path)] for every rendered clip of `rows` and each locked anchor. Refuses a
    missing wav, so an embedding file is never written with a hole in it."""
    clips = [(r["id"], os.path.join(out, r["file"])) for r in rows]
    clips += [(cid, os.path.join(anchors_dir, "clips", cid + ".wav")) for cid in LOCKED_ANCHORS]
    missing = [p for _cid, p in clips if not os.path.isfile(p)]
    if missing:
        raise SystemExit("REFUSING: %d wav(s) missing for embedding, e.g. %s." % (len(missing),
                                                                                 missing[0]))
    return clips


def embed_rows(clips, lda, ecapa):
    """-> one {"id", "zonos_lda", "ecapa"} row per (id, path), in order. `lda` and `ecapa` take a
    wav path and return its embedding as a list of floats."""
    return [{"id": cid, "zonos_lda": lda(path), "ecapa": ecapa(path)} for cid, path in clips]


def load_embedders():
    """The two embedding models, loaded lazily: the planner, the gates and their tests import none
    of torch, soundfile, speechbrain or Zonos. Pinned to the same ECAPA snapshot the 1c-i
    extraction used (extract_and_analyse.py), and refuses any other checkpoint bytes."""
    import soundfile as sf
    import torch
    import torchaudio

    verify_ecapa_hashes(ECAPA_SNAP, ECAPA_EXPECTED_SHA256)
    _sys.path.insert(0, ZONOS_PATH)
    from speechbrain.inference.speaker import EncoderClassifier
    from zonos.speaker_cloning import SpeakerEmbeddingLDA

    lda_model = SpeakerEmbeddingLDA(device="cpu")
    ecapa_model = EncoderClassifier.from_hparams(source=ECAPA_SNAP, savedir=ECAPA_SNAP,
                                                 run_opts={"device": "cpu"})

    def lda(path):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        with torch.no_grad():
            _, emb = lda_model(torch.from_numpy(x.T), sr)
        return [round(float(v), 6) for v in emb.flatten()]

    def ecapa(path):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = torch.from_numpy(x.mean(axis=1)).unsqueeze(0)
        if sr != 16000:
            mono = torchaudio.functional.resample(mono, sr, 16000)
        with torch.no_grad():
            emb = ecapa_model.encode_batch(mono)
        return [round(float(v), 6) for v in emb.flatten()]

    return lda, ecapa


def file_sha256(path):
    import hashlib

    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def cmd_embed(args):
    from synth_common import write_text_atomic

    rows, _meta = rendered(args.out)
    clips = embed_clips(rows, args.out, args.anchors_dir)
    lda, ecapa = load_embedders()
    out = embed_rows(clips, lda, ecapa)
    # Provenance: which bytes of each wav were embedded, so admission can show it.
    out = [dict(o, wav_sha256=file_sha256(path)) for o, (_cid, path) in zip(out, clips)]
    write_text_atomic(os.path.join(args.out, SPEAKER_EMBEDDINGS),
                      "".join(json.dumps(o) + "\n" for o in out))
    print("%d embeddings (%d rendered, %d locked anchors) -> %s"
          % (len(out), len(rows), len(LOCKED_ANCHORS), SPEAKER_EMBEDDINGS))


def admit(candidates, qc, embeddings, locked):
    """The three automated gates, in plan order. Returns the report's dict.

    `candidates`: measures rows (plan order), each with `id`, `gender`, `zone`, `f0_out_of_range`.
    `qc`: {id: qc_measures row} (`hard_pass`). `embeddings`: {id: {"zonos_lda", "ecapa"}}.
    `locked`: {id: gender} of the locked anchors, which every candidate is compared against.

    A candidate is admitted only if its cosine to every reference of its gender (the locked anchors
    and every earlier admitted candidate) is strictly below DISTINCT_COS on both models. The
    reference that comes closest is reported when a candidate is refused for distinctness."""
    refs = {g: [] for g in GENDERS}
    for cid, g in locked.items():
        refs[g].append(cid)
    admitted = {g: [] for g in GENDERS}
    filtered, counts = [], {"candidates": 0, "passed_qc": 0, "passed_f0": 0}
    for c in candidates:
        cid, g = c["id"], c["gender"]
        counts["candidates"] += 1
        q = qc.get(cid)
        if q is None or q.get("hard_pass") is not True:
            filtered.append({"id": cid, "gender": g, "stage": "qc",
                             "reason": "no qc_measures row" if q is None else "hard_pass is not true"})
            continue
        counts["passed_qc"] += 1
        if c.get("zone") != "core" or c.get("f0_out_of_range"):
            filtered.append({"id": cid, "gender": g, "stage": "f0",
                             "reason": "zone %s, f0_out_of_range %s"
                                       % (c.get("zone"), bool(c.get("f0_out_of_range")))})
            continue
        counts["passed_f0"] += 1
        if cid not in embeddings:
            raise SystemExit("REFUSING: no embedding for candidate %s." % cid)
        worst = None
        for ref in refs[g]:
            if ref not in embeddings:
                raise SystemExit("REFUSING: no embedding for reference %s." % ref)
            for key in EMBEDDING_KEYS:
                cos = _cosine(embeddings[cid][key], embeddings[ref][key],
                              "candidate %s vs %s (%s)" % (cid, ref, key))
                if worst is None or cos > worst[0]:
                    worst = (cos, ref, key)
        # Gated on "not below", so a NaN cosine (which compares false to everything) can never admit.
        if worst is not None and not worst[0] < DISTINCT_COS:
            filtered.append({"id": cid, "gender": g, "stage": "distinct",
                             "reason": "cosine %.4f to %s on %s (bound < %g)"
                                       % (worst[0], worst[1], worst[2], DISTINCT_COS)})
            continue
        refs[g].append(cid)
        admitted[g].append(cid)
    recommended = {g: admitted[g][:NEED[g]] for g in GENDERS}
    return {"admitted": admitted, "recommended": recommended, "filtered": filtered,
            "recommended_rule": "the first %s admitted per gender, in plan order" % NEED,
            "counts": counts,
            "shortfall": {g: max(0, NEED[g] - len(admitted[g])) for g in GENDERS}}


def _cosine(a, b, who):
    """The cosine of two embeddings. `who` names the pair in any refusal.

    Refuses a non-finite value: a NaN compares false against every bound, so it would otherwise
    read as distinct. Refuses a zero-length vector too. rq's `_cosine` has no such guard (it
    divides by zero), so the admission gates use this one."""
    if len(a) != len(b):
        raise SystemExit("REFUSING: %s: embeddings differ in length (%d vs %d)."
                         % (who, len(a), len(b)))
    if not all(math.isfinite(x) for x in list(a) + list(b)):
        raise SystemExit("REFUSING: %s: an embedding has a non-finite value." % who)
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        raise SystemExit("REFUSING: %s: a zero-length embedding has no cosine." % who)
    cos = sum(x * y for x, y in zip(a, b)) / (na * nb)
    if not math.isfinite(cos):
        raise SystemExit("REFUSING: %s: the cosine is not finite." % who)
    return cos


def plan_measures(measures, meta, plan):
    """-> the measures rows of the whole plan, in plan order.

    Refuses a measure of a subset (`measure --ids`): admit-filter would then report the candidates
    it never measured as absent rather than as skipped, and a shortfall would read as a verdict.
    Refuses a plan id with no row, and a duplicated row."""
    if meta.get("ids") is not None:
        raise SystemExit("REFUSING: measure_meta.json records ids %s; admit-filter needs a "
                         "whole-plan measure (ids null)." % (meta.get("ids"),))
    by_id = {}
    for m in measures:
        if m["id"] in by_id:
            raise SystemExit("REFUSING: measures.jsonl has two rows for %s." % m["id"])
        by_id[m["id"]] = m
    missing = [it["id"] for it in plan if it["id"] not in by_id]
    if missing:
        raise SystemExit("REFUSING: %d of %d plan ids have no measures row (missing %s)."
                         % (len(missing), len(plan), ", ".join(missing[:5])
                            + (" ..." if len(missing) > 5 else "")))
    return [by_id[it["id"]] for it in plan]


def audition_relpath(out):
    """`out`'s path relative to the audition app's DATA_ROOT, which the admitted manifest's audio
    links are built from. Refuses a folder outside it: its links would not resolve."""
    rel = os.path.relpath(os.path.abspath(out), AUDITION_DATA_ROOT)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        raise SystemExit("REFUSING: %s is outside the audition data root %s; the admitted "
                         "manifest's audio links would not resolve." % (out, AUDITION_DATA_ROOT))
    return rel.replace(os.sep, "/")


def probe_manifest(report, plan, rows_by_id, measures_by_id, probe_id, data_rel):
    """The audition app's manifest of the ADMITTED clips only, in plan order, in the shape
    `_load_probe_manifest` serves from a manifest.json. Each clip is its manifest row with `group`,
    `wav` (relative to the audition data root) and `acoustics` (its measures row)."""
    admitted = set(report["admitted"]["Male"]) | set(report["admitted"]["Female"])
    clips = []
    for it in plan:
        cid = it["id"]
        if cid not in admitted:
            continue
        rec = rows_by_id[cid]
        clips.append(dict(rec, group=rec.get("group") or rec.get("rung") or "",
                          wav=posixpath.join(data_rel, rec["file"]),
                          acoustics=measures_by_id[cid]))
    return {"probe": probe_id, "title": "Calibration Probe — %s (admitted)" % probe_id,
            "clips": clips}


def cmd_admit_filter(args):
    from synth_common import write_json_atomic

    measures_path = os.path.join(args.out, "measures.jsonl")
    if not os.path.exists(measures_path):
        raise SystemExit("REFUSING: no %s; run measure first." % measures_path)
    meta_path = os.path.join(args.out, "measure_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("REFUSING: no %s; run measure first." % meta_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    qc_path = args.qc_jsonl or os.path.join(args.out, "qc_measures.jsonl")
    if not os.path.exists(qc_path):
        raise SystemExit("REFUSING: no QC measures at %s; run the QC gate first." % qc_path)
    embeddings_path = args.embeddings_jsonl or os.path.join(args.out, SPEAKER_EMBEDDINGS)
    if not os.path.exists(embeddings_path):
        raise SystemExit("REFUSING: no embeddings at %s; run embed first." % embeddings_path)
    plan = plan_items()
    candidates = plan_measures(rq._read_jsonl(measures_path), meta, plan)
    rows, _meta = rendered(args.out)
    data_rel = audition_relpath(args.out)
    qc = {r["id"]: r for r in rq._read_jsonl(qc_path)}
    embeddings = {r["id"]: r for r in rq._read_jsonl(embeddings_path)}
    report = admit(candidates, qc, embeddings, LOCKED_ANCHORS)
    write_json_atomic(os.path.join(args.out, "admission_report.json"), dict(
        report, tool=TOOL, probe=PROBE, distinct_cos_bound=DISTINCT_COS,
        embedding_models=list(EMBEDDING_KEYS), locked_anchors=LOCKED_ANCHORS, need=NEED,
        inputs={"measures_sha256": rq._sha256(measures_path),
                "qc_sha256": rq._sha256(qc_path),
                "embeddings_sha256": rq._sha256(embeddings_path),
                "anchors": LOCKED_ANCHORS}))
    write_json_atomic(os.path.join(args.out, "manifest.json"), probe_manifest(
        report, plan, {r["id"]: r for r in rows}, {m["id"]: m for m in candidates},
        os.path.basename(os.path.abspath(args.out)), data_rel))
    print("candidates %d; passed QC %d; passed F0 band %d; admitted Female %d, Male %d"
          % (report["counts"]["candidates"], report["counts"]["passed_qc"],
             report["counts"]["passed_f0"], len(report["admitted"]["Female"]),
             len(report["admitted"]["Male"])))
    for g in GENDERS:
        if report["shortfall"][g]:
            print("SHORTFALL: %s needs %d more than admitted." % (g, report["shortfall"][g]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="print the 114 candidates")
    p.add_argument("--json", help="also write the plan to this path")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("render", help="render the candidates (GPU)")
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True, help="the checkout's HEAD the render runs under")
    p.add_argument("--ids", nargs="+", help="render only these ids")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("measure", help="pYIN F0, phonation and the F0 band")
    p.add_argument("--out", required=True)
    p.add_argument("--ids", nargs="+", help="measure only the ids the render recorded")
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("embed", help="Zonos LDA and ECAPA-TDNN embeddings (litert venv)")
    p.add_argument("--out", required=True)
    p.add_argument("--anchors-dir", default=DEFAULT_ANCHORS_DIR,
                   help="the folder holding clips/M4-s3.wav and clips/F6-s2.wav")
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("admit-filter", help="the three automated gates")
    p.add_argument("--out", required=True)
    p.add_argument("--qc-jsonl", help="qc_measures.jsonl (default: OUT/qc_measures.jsonl)")
    p.add_argument("--embeddings-jsonl",
                   help="speaker embeddings (default: OUT/speaker_embeddings.jsonl)")
    p.set_defaults(fn=cmd_admit_filter)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

"""Phase 1S Step 2c: the 90-speaker synthetic corpus for Arm Q (`q_synth_22k`).

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § Step 2 — Arm Q, step 2c (owner,
approved 2026-10-29). The speakers are the 90 locked in the table below, the text is the 36,000-line
Step 2b bank (sha256 `TEXT_SHA`), and every line is rendered once, in dataset mode.

Subcommands, in order:

* `lock-speakers`   writes speakers.json (90) — or speakers_locked5.json (speakers 0, 1, 87, 88, 89,
                    which need no owner rating) — from the Step 2a admission report and the owner's
                    verdicts. Refuses if a gender has fewer passing anchors than it needs.
* `plan`            writes spk_NN/bank.json and plan.json per speaker, from the Step 2b speaker files.
* `render`          prints (or with --run executes) the synth_bank.sh call for each speaker, in
                    DATASET mode: SYNTH_BANK_PROBE=0, which keeps the direction check, the defect
                    detectors and the QC gate that a probe skips. The GPU is booked separately.
* `build-corpus`    verifies every speaker has >= 310 QC-passing clips, then writes `q_synth_22k`:
                    22.05 kHz PCM_16 copies, train_op.txt and val_op.txt, speakers.json, and
                    provenance.json. mel_statistics.json comes from `mel-stats`, which needs torch
                    (the litert interpreter).

The phonemizer is `matcha.text.op_g2p.OpenPhonemizerG2P`, the repo's G2P. Its output is checked against
`baseline_bench.original_matcha_symbols()`; a symbol outside that table refuses the build.
"""
import argparse
import json
import math
import os as _os
import random
import shlex
import subprocess
import sys as _sys
import datetime

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib", "tools"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import render_1ci as r1  # noqa: E402
import render_1cii as r2  # noqa: E402
from synth_common import write_json_atomic, write_text_atomic  # noqa: E402

TOOL = "scripts/tools/render_step2_corpus.py"
VERSION = "step2c-v1"
TEXT_SHA = "44f970bc60abfbc8f5b866aa7561e50ad123361fc39662c17591a3f0721d4bdd"
TEXT_DIR = "/data/model-training/datasets/probes/qwen-step2-text"
ANCHORS_DIR = "/data/model-training/datasets/probes/qwen-step2-anchors"
LOCKED_DIR = "/data/model-training/datasets/probes/qwen-1c0-anchors"
RAW_ROOT = "/data/model-training/datasets/q_synth_raw"
OUT_DIR = "/data/model-training/sonora/data/q_synth_22k"
N_SPEAKERS = 90
NEED = {"Male": 42, "Female": 43}                 # new anchors per gender (Step 2a's need)
LOCKED_SPEAKERS = (0, 1, 87, 88, 89)              # the five that need no owner rating
PRIMARY = 360
SEED_BASE = 2_000_000                             # seed = SEED_BASE + 1000 * s + speaker line index
SPLIT_SEED = 1234
VAL_PER_SPEAKER = 10
MIN_QC = 310                                      # QC-passing clips per speaker, before the split
SAMPLE_RATE = 22050
CUSTOM_INSTRUCT = r2.CUSTOM_INSTRUCT
ORPHEUS_VOICE = "jess"
ENGINE_BY_KIND = {"qwen_base": "qwen_base", "qwen_custom": "qwen_custom", "orpheus": "orpheus"}


def refuse(msg):
    print(msg, file=_sys.stderr)
    raise SystemExit(2)


def spk_name(s):
    return "spk_%02d" % s


def _file_sha(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ref_of(row, dirpath, gender):
    """A qwen_base reference clip: path, sha256, prompt text, id and gender, as render_1cii's."""
    path = _os.path.join(dirpath, row["file"])
    return {"path": path, "sha256": _file_sha(path), "text": row["text"], "id": row["id"],
            "gender": gender}


def build_speakers(male_refs, female_refs, locked):
    """The 90-speaker table: 0 M4-s3, 1 F6-s2, 2..43 the 42 Male, 44..86 the 43 Female, 87 ryan,
    88 aiden (qwen_custom, Male), 89 jess (orpheus, Female). Refuses any count that is not the
    registered one."""
    if len(male_refs) != NEED["Male"] or len(female_refs) != NEED["Female"]:
        refuse("REFUSING: %d Male and %d Female anchors; the table needs %d and %d."
               % (len(male_refs), len(female_refs), NEED["Male"], NEED["Female"]))
    table = [{"index": 0, "kind": "qwen_base", "gender": "Male", "anchor": locked["M4-s3"]},
             {"index": 1, "kind": "qwen_base", "gender": "Female", "anchor": locked["F6-s2"]}]
    table += [{"index": 2 + i, "kind": "qwen_base", "gender": "Male", "anchor": r}
              for i, r in enumerate(male_refs)]
    table += [{"index": 44 + i, "kind": "qwen_base", "gender": "Female", "anchor": r}
              for i, r in enumerate(female_refs)]
    table += [{"index": 87, "kind": "qwen_custom", "gender": "Male", "speaker": "ryan",
               "instruct": CUSTOM_INSTRUCT},
              {"index": 88, "kind": "qwen_custom", "gender": "Male", "speaker": "aiden",
               "instruct": CUSTOM_INSTRUCT},
              {"index": 89, "kind": "orpheus", "gender": "Female", "voice": ORPHEUS_VOICE}]
    for t in table:
        t["name"] = spk_name(t["index"])
        t["engine"] = ENGINE_BY_KIND[t["kind"]]
    assert [t["index"] for t in table] == list(range(N_SPEAKERS))
    return table


def _admitted_passing(admitted, verdicts):
    """The admitted ids (plan order) whose verdict is core with no hum, in that order."""
    out = []
    for cid in admitted:
        v = verdicts.get(cid)
        if v and v.get("zone") == "core" and int(v.get("hum") or 0) == 0:
            out.append(cid)
    return out


def _read_verdicts(path):
    import csv
    if not _os.path.exists(path):
        refuse("REFUSING: no verdicts at %s; the owner's ear check has not run." % path)
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def cmd_lock_speakers(args):
    locked_rows = _manifest_rows(args.locked_dir)
    locked = {}
    for cid in ("M4-s3", "F6-s2"):
        if cid not in locked_rows:
            refuse("REFUSING: %s has no %s row." % (args.locked_dir, cid))
        gender = "Male" if cid == "M4-s3" else "Female"
        locked[cid] = ref_of(locked_rows[cid], args.locked_dir, gender)
    if args.locked_five_only:
        # The 42 + 43 new anchors are not needed for these five, so placeholders stand in for them.
        full = build_speakers([None] * NEED["Male"], [None] * NEED["Female"], locked)
        five = [t for t in full if t["index"] in LOCKED_SPEAKERS]
        write_json_atomic(_os.path.join(args.out_root, "speakers_locked5.json"),
                          {"version": VERSION, "speakers": five,
                           "locked_speakers": list(LOCKED_SPEAKERS)})
        print("wrote speakers_locked5.json (speakers %s)" % (", ".join(map(str, LOCKED_SPEAKERS))))
        return 0
    with open(_os.path.join(args.anchors_dir, "admission_report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    verdicts = _read_verdicts(_os.path.join(args.anchors_dir, "verdicts.csv"))
    anchor_rows = _manifest_rows(args.anchors_dir)
    refs = {}
    for gender in ("Male", "Female"):
        passing = _admitted_passing(report["admitted"][gender], verdicts)
        if len(passing) < NEED[gender]:
            refuse("REFUSING: %s has %d passing anchors (core, hum 0), needs %d."
                   % (gender, len(passing), NEED[gender]))
        refs[gender] = [ref_of(anchor_rows[cid], args.anchors_dir, gender)
                        for cid in passing[:NEED[gender]]]
    table = build_speakers(refs["Male"], refs["Female"], locked)
    write_json_atomic(_os.path.join(args.out_root, "speakers.json"),
                      {"version": VERSION, "speakers": table})
    print("wrote speakers.json (%d speakers)" % len(table))
    return 0


def load_speakers(out_root, subset=None):
    full = _os.path.join(out_root, "speakers.json")
    if _os.path.exists(full):
        table = json.load(open(full, encoding="utf-8"))["speakers"]
    else:
        table = json.load(open(_os.path.join(out_root, "speakers_locked5.json"),
                               encoding="utf-8"))["speakers"]
    if subset is not None:
        table = [t for t in table if t["index"] in subset]
    return table


def bank_lines(speaker, speaker_rows):
    """The bank lines for one speaker: one per text row, in speaker line order. The direction
    comes from build_direction through render_1cii's directions_for, so the plan refuses any
    direction the tag does not set."""
    cell = {"name": speaker["name"], "engine": speaker["engine"],
            "tag": ({} if speaker["kind"] == "qwen_base" else
                    {"speaker": speaker["speaker"], "instruct": speaker["instruct"]}
                    if speaker["kind"] == "qwen_custom" else {"voice": speaker["voice"]})}
    dirs, summary = r2.directions_for(cell, speaker_rows)
    lines = []
    for row, d in zip(speaker_rows, dirs):
        j = row["speaker_line_index"]
        line = {"id": "%s-%03d" % (speaker["name"], j), "engine": speaker["engine"],
                "render_index": j, "text_id": row["id"], "text": row["text"],
                "seed": SEED_BASE + 1000 * speaker["index"] + j, "direction": dict(d)}
        if speaker["kind"] == "qwen_base":
            line["ref_file"] = dict(speaker["anchor"])
        lines.append(line)
    return lines, summary


def speaker_bank(speaker, text_dir, include_reserve):
    """(bank dict, the text rows the bank holds, direction summary) for one speaker."""
    path = _os.path.join(text_dir, "speakers", "%s.jsonl" % speaker["name"])
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not include_reserve:
        rows = [r for r in rows if r["role"] == "primary"]
    lines, summary = bank_lines(speaker, rows)
    bank = {"version": VERSION, "campaign": "step2c-" + speaker["name"], "lines": lines}
    return bank, rows, summary


def cmd_plan(args):
    text_meta = json.load(open(_os.path.join(args.text_dir, "meta.json"), encoding="utf-8"))
    if text_meta.get("sha256") != TEXT_SHA:
        refuse("REFUSING: %s/meta.json sha256 %s; the frozen bank is %s."
               % (args.text_dir, text_meta.get("sha256"), TEXT_SHA))
    git = r1.tool_commit()
    r1._require_clean(git, args.allow_dirty)
    for speaker in load_speakers(args.out_root, args.speakers):
        bank, rows, summary = speaker_bank(speaker, args.text_dir, args.include_reserve)
        cdir = _os.path.join(args.out_root, speaker["name"])
        meta = {"tool": TOOL, "version": VERSION, "commit": git["commit"],
                "tree_clean": git["tree_clean"], "speaker": speaker,
                "text_sha256": TEXT_SHA, "include_reserve": bool(args.include_reserve),
                "lines": len(bank["lines"]), "direction": summary,
                "engine_pin": r2.engine_pin(speaker["engine"]),
                "bank_sha256": _file_sha_of_json(bank),
                "seed_rule": "seed = %d + 1000 * speaker index + speaker line index"
                             % SEED_BASE}
        _refuse_existing(cdir, bank, meta)
        _os.makedirs(cdir, exist_ok=True)
        write_json_atomic(_os.path.join(cdir, "bank.json"), bank)
        write_json_atomic(_os.path.join(cdir, "plan.json"), meta)
        print("%s: %d lines (%s)" % (speaker["name"], len(bank["lines"]), speaker["engine"]))
    return 0


def _file_sha_of_json(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _refuse_existing(cdir, bank, meta):
    if not _os.path.exists(_os.path.join(cdir, "bank.json")):
        return
    try:
        on_disk = json.load(open(_os.path.join(cdir, "bank.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        refuse("REFUSING: %s/bank.json cannot be read (%s)." % (cdir, e))
    if on_disk != bank:
        refuse("REFUSING: %s/bank.json differs from this plan; plan into a new --out-root."
               % cdir)


def render_argv(out_root, speaker):
    cdir = _os.path.join(out_root, speaker["name"])
    return [r1.SYNTH_BANK, _os.path.join(cdir, "bank.json"), _os.path.join(cdir, "audio"),
            speaker["engine"]]


def cmd_render(args):
    git = r1.tool_commit()
    r1._require_clean(git, args.allow_dirty)
    out_root = _os.path.abspath(args.out_root)
    for speaker in load_speakers(out_root, args.speakers):
        cdir = _os.path.join(out_root, speaker["name"])
        try:
            with open(_os.path.join(cdir, "plan.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            with open(_os.path.join(cdir, "bank.json"), encoding="utf-8") as fh:
                bank = json.load(fh)
        except (OSError, ValueError) as e:
            refuse("REFUSING: %s has no readable plan (%s). Run `plan` first." % (cdir, e))
        changed = r2.render_inputs_changed(meta["commit"], git["commit"])
        if changed:
            refuse("REFUSING: %s was planned at %s, HEAD is %s; render-affecting files differ: %s."
                   % (cdir, meta["commit"], git["commit"], ", ".join(changed)))
        if meta["engine_pin"] != r2.engine_pin(speaker["engine"]):
            refuse("REFUSING: %s's engine pin changed since its plan." % cdir)
        if _file_sha_of_json(bank) != meta["bank_sha256"]:
            refuse("REFUSING: %s/bank.json changed since its plan." % cdir)
        argv = render_argv(out_root, speaker)
        env = {"SYNTH_BANK_PROBE": "0"}           # dataset mode: never a probe
        print("SYNTH_BANK_PROBE=0 " + " ".join(shlex.quote(a) for a in argv))
        if not args.run:
            continue
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        code = subprocess.run(argv, env=dict(_os.environ, **env)).returncode
        with open(_os.path.join(cdir, "render_log.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"started_utc": started, "argv": argv, "exit_code": code,
                                 "head_commit": git["commit"]}) + "\n")
        if code != 0:
            refuse("REFUSING: synth_bank.sh exited %d for %s." % (code, speaker["name"]))
    if not args.run:
        print("(printed only; --run executes each. The GPU is booked separately.)")
    return 0


def split_rows(ids, seed=SPLIT_SEED, n_val=VAL_PER_SPEAKER):
    """-> (train ids, val ids): `n_val` ids drawn deterministically from the sorted ids."""
    ids = sorted(ids)
    if len(ids) < n_val + 1:
        raise ValueError("%d clips cannot give %d validation rows" % (len(ids), n_val))
    val = set(random.Random(seed).sample(ids, n_val))
    return [i for i in ids if i not in val], [i for i in ids if i in val]


def qc_passing(spk_dir, engine):
    """The speaker's QC-passing clips: (id, wav path, text), in id order."""
    qc = {}
    with open(_os.path.join(spk_dir, "qc_measures.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                qc[r["id"]] = r
    man = _manifest_rows(_os.path.join(spk_dir, "audio"), name="%s_manifest.jsonl" % engine)
    out = []
    for cid, row in man.items():
        q = qc.get(cid)
        if q and q.get("hard_pass") is True:
            out.append((cid, _os.path.join(spk_dir, "audio", row["wav"]), row["text"]))
    return sorted(out)


def _manifest_rows(dirpath, name="manifest.jsonl"):
    rows = {}
    with open(_os.path.join(dirpath, name), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def resample_to(wav_in, dst):
    """A 24 kHz wav -> 22.05 kHz PCM_16, as prep_staged_corpora.resample_to; returns 1 if clipped."""
    import numpy as np
    import soundfile as sf
    import soxr
    x, sr = sf.read(wav_in, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = soxr.resample(np.asarray(x, dtype="float64"), sr, SAMPLE_RATE)
    _os.makedirs(_os.path.dirname(dst), exist_ok=True)
    sf.write(dst, np.clip(y, -1.0, 1.0), SAMPLE_RATE, "PCM_16")
    return int(np.abs(y).max() > 1.0)


def phonemes_checked(g2p, text, allowed):
    """The G2P phonemes of `text`, refusing any symbol outside `allowed`."""
    ph = g2p.phonemize(text).strip()
    bad = sorted({c for c in ph if c not in allowed})
    if not ph or bad:
        raise ValueError("phonemes %r for %r have symbols outside the table: %s"
                         % (ph, text[:60], bad))
    return ph


def cmd_build_corpus(args):
    import soundfile  # noqa: F401  (resample_to needs it; fail early)
    from baseline_bench import original_matcha_symbols
    from matcha.text.op_g2p import OpenPhonemizerG2P

    speakers = load_speakers(args.raw_root)
    short = {}
    per = {}
    for s in speakers:
        spk_dir = _os.path.join(args.raw_root, s["name"])
        try:
            kept = qc_passing(spk_dir, s["engine"])
        except (OSError, KeyError) as e:
            refuse("REFUSING: %s has no readable audio or QC (%s)." % (spk_dir, e))
        per[s["index"]] = kept
        if len(kept) < MIN_QC:
            short[s["name"]] = len(kept)
    if short:
        refuse("REFUSING: %d speaker(s) under %d QC-passing clips: %s. Top them up with "
               "--include-reserve first." % (len(short), MIN_QC,
                                             ", ".join("%s=%d" % kv for kv in short.items())))
    allowed = set(original_matcha_symbols())
    g2p = OpenPhonemizerG2P()
    out = _os.path.abspath(args.out_dir)
    train_lines, val_lines, counts = [], [], {"clipped": 0, "train": 0, "val": 0}
    for s in speakers:
        kept = per[s["index"]]
        train_ids, val_ids = split_rows([k[0] for k in kept], seed=SPLIT_SEED)
        by_id = {k[0]: k for k in kept}
        for split, ids in (("train", train_ids), ("val", val_ids)):
            for cid in ids:
                _cid, wav, text = by_id[cid]
                dst = _os.path.join(out, "wavs", s["name"], cid + ".wav")
                counts["clipped"] += resample_to(wav, dst)
                try:
                    ph = phonemes_checked(g2p, text, allowed)
                except ValueError as e:
                    refuse("REFUSING: %s" % e)
                (train_lines if split == "train" else val_lines).append(
                    "%s|%d|%s" % (dst, s["index"], ph))
                counts[split] += 1
    _os.makedirs(out, exist_ok=True)
    write_text_atomic(_os.path.join(out, "train_op.txt"), "\n".join(train_lines) + "\n")
    write_text_atomic(_os.path.join(out, "val_op.txt"), "\n".join(val_lines) + "\n")
    write_json_atomic(_os.path.join(out, "speakers.json"),
                      {"version": VERSION, "n_spks": N_SPEAKERS, "speakers": speakers})
    write_json_atomic(_os.path.join(out, "provenance.json"), {
        "tool": TOOL, "version": VERSION, "text_sha256": TEXT_SHA, "rows": counts,
        "speakers": N_SPEAKERS, "min_qc": MIN_QC, "val_per_speaker": VAL_PER_SPEAKER,
        "split_seed": SPLIT_SEED, "sample_rate": SAMPLE_RATE,
        "phonemizer": "matcha.text.op_g2p.OpenPhonemizerG2P",
        "steps_per_epoch": math.ceil(counts["train"] / 32)})
    print("q_synth_22k: train %d, val %d rows; %d clips clipped; %d steps/epoch"
          % (counts["train"], counts["val"], counts["clipped"],
             math.ceil(counts["train"] / 32)))
    return 0


def cmd_mel_stats(args):
    from matcha.data.text_mel_datamodule import TextMelDataModule
    from matcha.utils.generate_data_statistics import compute_data_statistics
    from lib import staged_data as sd

    out = _os.path.abspath(args.out_dir)
    dm = TextMelDataModule(
        name="q_synth_22k_stats", train_filelist_path=_os.path.join(out, "train_op.txt"),
        valid_filelist_path=_os.path.join(out, "val_op.txt"), batch_size=256,
        num_workers=args.workers, pin_memory=False, cleaners=["no_cleaners"], add_blank=True,
        n_spks=N_SPEAKERS, data_statistics=None, seed=1234, load_durations=False,
        load_vat=False, vat_dim=None, **dict(zip(sd.MEL_KEYS, sd.STOCK_MEL)))
    dm.setup()
    params = compute_data_statistics(dm.train_dataloader(), sd.STOCK_MEL[1])
    write_json_atomic(_os.path.join(out, "mel_statistics.json"), params)
    print("mel_statistics:", json.dumps(params))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lock-speakers", help="write speakers.json (or speakers_locked5.json)")
    p.add_argument("--anchors-dir", default=ANCHORS_DIR)
    p.add_argument("--locked-dir", default=LOCKED_DIR)
    p.add_argument("--out-root", default=RAW_ROOT)
    p.add_argument("--locked-five-only", action="store_true")
    p.set_defaults(fn=cmd_lock_speakers)

    p = sub.add_parser("plan", help="write spk_NN/bank.json and plan.json")
    p.add_argument("--text-dir", default=TEXT_DIR)
    p.add_argument("--out-root", default=RAW_ROOT)
    p.add_argument("--speakers", nargs="+", type=int)
    p.add_argument("--include-reserve", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("render", help="print (or with --run execute) synth_bank.sh per speaker")
    p.add_argument("--out-root", default=RAW_ROOT)
    p.add_argument("--speakers", nargs="+", type=int)
    p.add_argument("--run", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("build-corpus", help="write q_synth_22k from the QC-passing clips")
    p.add_argument("--raw-root", default=RAW_ROOT)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.set_defaults(fn=cmd_build_corpus)

    p = sub.add_parser("mel-stats", help="mel_statistics.json (needs torch)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_mel_stats)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    _sys.exit(main())

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "scipy"]
# ///
"""Score the two artifact probes built by make_artifact_probe.py.

Reuses qc_artifacts.py's measurements so the probe is judged by the same instrument
that flagged the original campaign — a probe scored with a bespoke metric proves
nothing about the campaign that motivated it.

  orpheus     ROOM per voice. `tara` is the positive control and must come back
              reverberant or the probe did not reproduce the defect and its verdict
              on the other four voices is worthless. `dan` is the negative control.
  chatterbox  DOUBLE per condition.
              Arm A answers "is it deterministic?" — if env_ac swings with the seed,
              MAX_REF_F0 is the wrong instrument regardless of what arm B shows.
              Arm B answers "can parameters rescue a bright reference?" — a cell that
              is clean across all four high-F0 references is a way to keep bright
              casting and drop the ceiling.

Usage:
    .venv/bin/python scripts/synthesis/analyze_artifact_probe.py --dir <probe_campaign_dir>
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_artifacts import DOUBLE_AC, decay_metrics, doubling, load  # noqa: E402


def measure(d):
    audio = os.path.join(d, "audio")
    root = audio if os.path.isdir(audio) else d
    man = {}
    for mf in glob.glob(os.path.join(root, "*_manifest.jsonl")):
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
                man[r["wav"]] = r
            except Exception:
                pass
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*.wav"))):
        r = man.get(os.path.basename(p))
        if not r:
            continue
        x, sr = load(p)
        slope, c50 = decay_metrics(x, sr)
        ac, lag = doubling(x, sr)
        out.append({**r, "c50": c50, "decay": slope, "env_ac": ac,
                    "lag": lag, "dur": len(x) / sr})
    return out


def report_orpheus(rows):
    by = {}
    for r in rows:
        by.setdefault(r["direction"]["voice"], []).append(r)
    print("=== ORPHEUS: reverberation by voice ===")
    print("(C50 low + decay shallow = reverberant. tara is the positive control.)\n")
    print(f"{'voice':8s} {'n':>3s} {'C50 dB':>8s} {'decay dB/s':>11s}   verdict")
    ctl = [r["c50"] for r in by.get("dan", []) if r["c50"] is not None]
    base = st.median(ctl) if ctl else None
    for v, rs in sorted(by.items(), key=lambda kv: st.median(
            [x["c50"] for x in kv[1] if x["c50"] is not None] or [0])):
        c = [x["c50"] for x in rs if x["c50"] is not None]
        d = [x["decay"] for x in rs if x["decay"] is not None]
        if not c:
            continue
        m = st.median(c)
        tag = ""
        if base is not None:
            # dan is the only voice with prior evidence of being clean, so it is the
            # yardstick; >2 dB below it is the gap tara showed against its peers.
            tag = "REVERBERANT" if m < base - 2 else "clean"
            if v == "tara":
                tag += "  (positive control — must read REVERBERANT)"
            if v == "dan":
                tag = "baseline (negative control)"
        print(f"{v:8s} {len(c):3d} {m:8.1f} {st.median(d):11.0f}   {tag}")
    if base is not None and by.get("tara"):
        tc = st.median([x["c50"] for x in by["tara"] if x["c50"] is not None])
        if not tc < base - 2:
            print("\n  ⚠ POSITIVE CONTROL FAILED — tara did not reproduce the defect. "
                  "Do not trust this run's verdict on the other voices.")


def report_chatterbox(rows):
    print("\n=== CHATTERBOX ===")
    A = [r for r in rows if r.get("probe") == "seed"]
    B = [r for r in rows if r.get("probe") == "params"]
    C = [r for r in rows if r.get("probe") == "control"]

    print("\n-- Arm A: is the doubling deterministic? (same ref+params, 3 seeds) --")
    print(f"{'ref F0':>7s}  " + "  ".join(f"{'env_ac':>8s}" for _ in range(3)) + "   spread")
    by = {}
    for r in A:
        by.setdefault(r.get("ref_f0_measured"), []).append(r)
    for f0, rs in sorted(by.items(), reverse=True):
        v = [r["env_ac"] for r in rs if r["env_ac"] is not None]
        if not v:
            continue
        hits = sum(1 for x in v if x > DOUBLE_AC)
        print(f"{f0:7d}  " + "  ".join(f"{x:8.3f}" for x in v)
              + f"   {max(v)-min(v):.3f}  ({hits}/{len(v)} over threshold)")
    allv = [r["env_ac"] for r in A if r["env_ac"] is not None]
    if allv:
        hit = sum(1 for x in allv if x > DOUBLE_AC)
        print(f"\n  {hit}/{len(allv)} seed-sweep renders doubled. "
              + ("Consistent -> the reference drives it, MAX_REF_F0 is the right lever."
                 if hit >= 0.7 * len(allv) else
                 "INCONSISTENT -> partly stochastic; a pitch ceiling cannot fully fix it."
                 if hit > 0.3 * len(allv) else
                 "Mostly CLEAN -> the original defect may not reproduce; re-check before "
                 "trusting MAX_REF_F0."))

    print("\n-- Arm B: can (exaggeration, cfg_weight) rescue a high-F0 reference? --")
    cells, f0s = {}, sorted({r.get("ref_f0_measured") for r in B}, reverse=True)
    for r in B:
        k = (r["direction"]["exaggeration"], r["direction"]["cfg_weight"])
        cells.setdefault(k, {})[r.get("ref_f0_measured")] = r["env_ac"]
    print(f"{'exag':>5s} {'cfg':>5s}  " + "  ".join(f"{f}Hz".rjust(8) for f in f0s) + "   clean?")
    for k in sorted(cells):
        row = cells[k]
        vals = [row.get(f) for f in f0s]
        ok = all(v is not None and v <= DOUBLE_AC for v in vals)
        print(f"{k[0]:5.2f} {k[1]:5.2f}  "
              + "  ".join("     -  " if v is None else f"{v:8.3f}" for v in vals)
              + f"   {'ALL CLEAN' if ok else ''}")
    winners = [k for k, row in cells.items()
               if all(row.get(f) is not None and row[f] <= DOUBLE_AC for f in f0s)]
    print("\n  " + (f"RESCUED at {winners} — bright/teen casting can be kept; "
                    f"set these params and drop MAX_REF_F0." if winners else
                    "No setting is clean across all high-F0 references. "
                    "Parameters do not rescue bright casting; MAX_REF_F0 stands."))

    if C:
        print("\n-- Arm C: low-F0 negative control (must be clean) --")
        for r in C:
            d = r["direction"]
            flag = "  <-- CONTROL FAILED" if (r["env_ac"] or 0) > DOUBLE_AC else ""
            print(f"  f0={r.get('ref_f0_measured')}Hz exag={d['exaggeration']} "
                  f"cfg={d['cfg_weight']}  env_ac={r['env_ac']}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    rows = measure(args.dir)
    if not rows:
        sys.exit(f"no rendered clips with manifests under {args.dir}")
    print(f"{len(rows)} clips measured from {args.dir}\n")
    if any(r["engine"] == "orpheus" for r in rows):
        report_orpheus([r for r in rows if r["engine"] == "orpheus"])
    if any(r["engine"] == "chatterbox" for r in rows):
        report_chatterbox([r for r in rows if r["engine"] == "chatterbox"])


if __name__ == "__main__":
    main()

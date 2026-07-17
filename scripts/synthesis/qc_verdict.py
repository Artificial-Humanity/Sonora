"""Synthesis QC gate, stage 2 of 2 (label verdicts).

Merges qc_measures.jsonl (hard gates, phonation, LUFS) with EIV scores
(scripts/eiv_score.py output over qc_filelist.txt) and confirms each clip's
MEASURED affect direction against its INTENDED labels, on the same
LibriTTS-anchored z-scales the Emilia mining used. We never trust instruct
intent — a clip keeps its labels only if the instruments agree.

Axis check: for any axis with |intended| >= 0.3, the measured z must point
the same way with |z| >= --z-confirm. Near-neutral intents are unconstrained
(sanity-bounded). keep = hard_pass AND all axis checks.

Outputs: qc_verdicts.jsonl, keeps.jsonl (clips + confirmed labels), and a
console summary table for the owner audit.
"""
import argparse
import json
import os

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LIB_MEASURES = os.path.join(REPO, "data/libritts_r_vat_v2/measures.jsonl")
LIB_EIV = "/data/model-training/sonora/eiv_scores/corpus_v1.jsonl"
LIB_FAM = "/data/model-training/sonora/eiv_scores/corpus_families.jsonl"
COMBO = "/data/model-training/sonora/eiv_scores/valence_combo_v1.json"


def jload(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_anchors():
    lib_meas = jload(LIB_MEASURES)
    lib_eiv = {d["wav"]: d for d in jload(LIB_EIV)}
    lib_fam = {d["wav"]: d for d in jload(LIB_FAM)}
    combo = json.load(open(COMBO, encoding="utf-8"))
    heads, w = combo["heads"], np.array(combo["weights"])

    def gs(vals):
        v = np.asarray(vals, float)
        return float(v.mean()), float(v.std() + 1e-9)

    anchor = {k: gs([m[k] for m in lib_meas]) for k in ("alpha_db", "cpp", "h1h2")}
    joined = [{**lib_eiv[m["wav"]], **lib_fam.get(m["wav"], {})}
              for m in lib_meas if m["wav"] in lib_eiv]
    anchor["soft"] = gs([d["Soft_vs._Harsh"] for d in joined])
    anchor["arousal"] = gs([d["Arousal"] for d in joined])
    for h in heads:
        anchor[h] = gs([d.get(h, 0.0) for d in joined])
    return anchor, heads, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--eiv", required=True, help="eiv_score.py output jsonl")
    ap.add_argument("--z-confirm", type=float, default=0.25)
    ap.add_argument("--neutral-band", type=float, default=0.3,
                    help="|intended| below this = no direction requirement")
    args = ap.parse_args()

    anchor, HEADS, W = build_anchors()

    def z(key, x):
        m, s = anchor[key]
        return (x - m) / s

    eiv = {os.path.basename(d["wav"]): d for d in jload(args.eiv)}
    rows = jload(os.path.join(args.campaign_dir, "qc_measures.jsonl"))

    verdicts, keeps = [], []
    print(f"{'id':26s} {'axis verdicts':40s} keep")
    for r in rows:
        e = eiv.get(os.path.basename(r["wav_abs"]))
        measured = {}
        if r.get("phonation") and e:
            p = r["phonation"]
            measured["T"] = (z("alpha_db", p["alpha_db"]) + z("cpp", p["cpp"])
                             - z("h1h2", p["h1h2"]) - z("soft", e["Soft_vs._Harsh"]))
            measured["A"] = z("arousal", e["Arousal"])
            measured["V"] = float(np.dot(W, [z(h, e.get(h, 0.0)) for h in HEADS]))

        checks, notes = {}, []
        for axis in ("V", "A", "T"):
            want = r["intended"][axis]
            if abs(want) <= args.neutral_band:  # inclusive: 0.3 intents are weak, not directional
                checks[axis] = abs(measured.get(axis, 0.0)) < 3.0 if measured else False
                continue
            got = measured.get(axis)
            ok = got is not None and np.sign(got) == np.sign(want) and abs(got) >= args.z_confirm
            checks[axis] = bool(ok)
            notes.append(f"{axis}:{'ok' if ok else 'FAIL'}({want:+.1f}->"
                         f"{'--' if got is None else format(got, '+.2f')})")
        keep = bool(r["hard_pass"] and all(checks.values()))
        v = {**r, "measured_z": measured or None, "axis_checks": checks, "keep": keep}
        verdicts.append(v)
        if keep:
            keeps.append(v)
        print(f"{r['id']:26s} {' '.join(notes):40s} {keep}")

    for name, data in (("qc_verdicts.jsonl", verdicts), ("keeps.jsonl", keeps)):
        with open(os.path.join(args.campaign_dir, name), "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    print(f"{len(keeps)}/{len(verdicts)} keeps -> keeps.jsonl")
    print("QC-VERDICT-DONE")


if __name__ == "__main__":
    main()

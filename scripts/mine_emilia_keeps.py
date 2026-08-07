"""Apply the pre-registered Emilia mining criteria and copy the keeps.

Criteria (emilia-mining-and-verdict.md, pre-registered 2026-07-17): over the scored
batch, keep any clip with T_full > p90, V_combo > p95, V_combo < p5, or
EIV-Arousal > p95 (Emilia-internal percentiles; measures already filtered to
DNSMOS >= 3.3, 1-16 s). Per-speaker-cluster cap 40, most-extreme first.
Scales are LibriTTS-anchored (global mean/std from the v2 corpus measures +
scores) so T_full/V_combo mean the same thing they meant in the probe.

Keeps are COPIED to --out with a provenance manifest (source tar, id, text,
speaker cluster, scores, criteria hit, license note). Also emits an
owner-audit sample (~50 clips across the four criteria groups).

Usage:
    python scripts/mine_emilia_keeps.py \
        --mining-dir /data/model-training/sonora/emilia_mining \
        --out /data/model-training/datasets/emilia_kept
"""

import argparse
import json
import os
import random
import shutil

import numpy as np

# v2 ON PURPOSE — do not "update" it to v3c. The mining criteria are pre-registered
# percentiles against a fixed LibriTTS anchor (emilia-mining-and-verdict.md, 2026-07-17);
# re-anchoring would silently change what T_full/V_combo mean and invalidate the
# pre-registration. This is the opposite case to D-L5's stale v2 path in
# derive_markup_measures.py, which was frozen by accident rather than by design.
LIB_MEASURES = "data/libritts_r_vat_v2/measures.jsonl"
LIB_EIV = "/data/model-training/sonora/eiv_scores/corpus_v1.jsonl"
LIB_FAM = "/data/model-training/sonora/eiv_scores/corpus_families.jsonl"
COMBO = "/data/model-training/sonora/eiv_scores/valence_combo_v1.json"


def jload(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def missing_weighted_heads(record, heads, weights):
    """Weighted EIV heads absent from `record`. D-M1.

    Only WEIGHTED heads matter: `Thankfulness_Gratitude`, `Disappointment` and `Distress`
    carry weight 0.0, so their absence cannot move the dot product — and two of them are
    legitimately absent from the raw score files. Demanding all twelve would fail on data
    that is entirely correct, which is how a guard gets disabled.
    """
    return sorted(h for h, w in zip(heads, weights) if w and h not in record)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mining-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-pct", type=float, default=90.0)
    ap.add_argument("--v-hi-pct", type=float, default=95.0)
    ap.add_argument("--v-lo-pct", type=float, default=5.0)
    ap.add_argument("--a-pct", type=float, default=95.0)
    ap.add_argument("--cluster-cap", type=int, default=40)
    ap.add_argument("--v-min-cluster", type=int, default=10,
                    help="V criteria use WITHIN-CLUSTER z (owner audit 2026-07-17: "
                         "global scales select timbre, not affect); clusters "
                         "smaller than this are exempt from V mining")
    ap.add_argument("--audit-n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # LibriTTS anchor scales.
    lib_meas = jload(LIB_MEASURES)
    lib_eiv = {d["wav"]: d for d in jload(LIB_EIV)}
    lib_fam = {d["wav"]: d for d in jload(LIB_FAM)}
    combo = json.load(open(COMBO, encoding="utf-8"))
    HEADS, W = combo["heads"], np.array(combo["weights"])

    def gs(vals):
        v = np.asarray(vals, float)
        return float(v.mean()), float(v.std() + 1e-9)

    anchor = {k: gs([m[k] for m in lib_meas]) for k in ("alpha_db", "cpp", "h1h2")}
    joined = [{**lib_eiv[m["wav"]], **lib_fam.get(m["wav"], {})}
              for m in lib_meas if m["wav"] in lib_eiv]
    anchor["soft"] = gs([d["Soft_vs._Harsh"] for d in joined])
    anchor["arousal"] = gs([d["Arousal"] for d in joined])

    # D-M1. These used to be `d.get(h, 0.0)` / `e.get(h, 0.0)`, and a missing head is the
    # one thing that must NOT default: 0.0 is a raw score, and `z()` maps it to
    # `(0 - mean) / std` — a systematically nonzero contribution, weighted into every
    # clip's valence, in the same units as a real measurement. Nothing downstream can tell
    # the difference between "this clip is unusually flat" and "this head was never
    # scored".
    #
    # It is safe today only because the Emilia and LibriTTS passes happen to carry the same
    # 12 heads. Run the EIV scorer with its DEFAULT head set (4) and every clip in the
    # corpus silently acquires a wrong valence. `eiv_merge_corpus` already refuses this
    # (its `ragged` guard); this is the same rule at the other end of the same pipeline.
    #
    # Only WEIGHTED heads are required: `Thankfulness_Gratitude`, `Disappointment` and
    # `Distress` carry weight 0.0, so their absence cannot move the dot product — and two
    # of them are legitimately absent from the raw files. Demanding all 12 would fail on
    # data that is entirely correct.
    missing = sorted({h for d in joined for h in missing_weighted_heads(d, HEADS, W)})
    if missing:
        raise SystemExit(
            f"LibriTTS anchor is missing weighted EIV head(s): {missing}\n"
            f"  Anchors come from {LIB_EIV} + {LIB_FAM}. Re-score with the full head set;\n"
            "  imputing 0.0 would z-score NONZERO and bias every anchor scale."
        )
    for h in HEADS:
        present = [d[h] for d in joined if h in d]
        # An unweighted head may legitimately be absent; its z-score is multiplied by 0.0
        # either way, so a neutral scale keeps the dot product well-defined without
        # inventing a measurement.
        anchor[h] = gs(present) if present else (0.0, 1.0)

    def z(key, x):
        m, s = anchor[key]
        return (x - m) / s

    # Emilia batch.
    rows = []
    for tar in sorted(os.listdir(args.mining_dir)):
        tdir = os.path.join(args.mining_dir, tar)
        mfile = os.path.join(tdir, "probe_measures.jsonl")
        efile = os.path.join(tdir, "eiv_scores.jsonl")
        if not (os.path.exists(mfile) and os.path.exists(efile)):
            continue
        eiv = {d["wav"]: d for d in jload(efile)}
        # D-M1, the half that reaches the corpus. Checked once per tar rather than per
        # clip: a head set is a property of the SCORING RUN, so if it is short here it is
        # short for every row in the file, and reporting it 20,000 times helps nobody.
        sample = next(iter(eiv.values()), None)
        if sample is not None:
            short = missing_weighted_heads(sample, HEADS, W)
            if short:
                raise SystemExit(
                    f"{efile}\n  was scored without weighted EIV head(s): {short}\n"
                    "  Re-run the EIV pass with the full head set (see eiv_score.sh --heads).\n"
                    "  Defaulting them to 0.0 would z-score NONZERO, so every clip in this\n"
                    "  tar would get a plausible, wrong valence that nothing downstream can\n"
                    "  distinguish from a real measurement."
                )
        for m in jload(mfile):
            e = eiv.get(m["wav"])
            if not e:
                continue
            t_full = (z("alpha_db", m["alpha_db"]) + z("cpp", m["cpp"])
                      - z("h1h2", m["h1h2"]) - z("soft", e["Soft_vs._Harsh"]))
            v_c = float(np.dot(W, [z(h, e.get(h, 0.0)) for h in HEADS]))
            a_z = z("arousal", e["Arousal"])
            rows.append({"wav": m["wav"], "tar": tar, "speaker": m.get("speaker"),
                         "text": m.get("text", ""), "duration": m.get("duration"),
                         "dnsmos": m.get("dnsmos"), "T": t_full, "V": v_c, "A": a_z})
    print(f"{len(rows)} scored clips in batch")

    # V: within-cluster z — a clip qualifies by being unusually dark/bright
    # FOR ITS OWN SPEAKER (global scales pick timbre, not affect).
    by_spk_v = {}
    for r in rows:
        by_spk_v.setdefault(r["speaker"], []).append(r["V"])
    for r in rows:
        vs = by_spk_v[r["speaker"]]
        if len(vs) >= args.v_min_cluster:
            r["Vz"] = (r["V"] - float(np.mean(vs))) / (float(np.std(vs)) + 1e-6)
        else:
            r["Vz"] = None
    vz_all = np.array([r["Vz"] for r in rows if r["Vz"] is not None])
    print(f"V-eligible clips (cluster >= {args.v_min_cluster}): {len(vz_all)}")

    T = np.array([r["T"] for r in rows]); A = np.array([r["A"] for r in rows])
    thr = {"T_hi": float(np.percentile(T, args.t_pct)),
           "V_hi": float(np.percentile(vz_all, args.v_hi_pct)),
           "V_lo": float(np.percentile(vz_all, args.v_lo_pct)),
           "A_hi": float(np.percentile(A, args.a_pct))}
    print("thresholds:", {k: round(v, 3) for k, v in thr.items()})

    for r in rows:
        crit = []
        if r["T"] > thr["T_hi"]: crit.append("T+")
        if r["Vz"] is not None and r["Vz"] > thr["V_hi"]: crit.append("V+")
        if r["Vz"] is not None and r["Vz"] < thr["V_lo"]: crit.append("V-")
        if r["A"] > thr["A_hi"]: crit.append("A+")
        r["criteria"] = crit
        margins = [r["T"] - thr["T_hi"], r["A"] - thr["A_hi"]]
        if r["Vz"] is not None:
            margins += [r["Vz"] - thr["V_hi"], thr["V_lo"] - r["Vz"]]
        r["margin"] = max(margins)

    cand = [r for r in rows if r["criteria"]]
    by_cluster = {}
    for r in sorted(cand, key=lambda r: -r["margin"]):
        by_cluster.setdefault(r["speaker"], []).append(r)
    keeps, capped = [], 0
    for spk, rs in by_cluster.items():
        keeps += rs[: args.cluster_cap]
        capped += max(0, len(rs) - args.cluster_cap)
    print(f"candidates {len(cand)} -> keeps {len(keeps)} "
          f"({capped} dropped by cluster cap; {len(by_cluster)} clusters)")

    hours = sum(r["duration"] or 0 for r in keeps) / 3600
    by_crit = {}
    for r in keeps:
        for c in r["criteria"]:
            by_crit.setdefault(c, []).append(r["duration"] or 0)
    print(f"kept hours: {hours:.1f}")
    for c in sorted(by_crit):
        print(f"  {c}: {len(by_crit[c])} clips, {sum(by_crit[c])/3600:.1f} h")
    clusters20 = sum(1 for rs in by_cluster.values() if min(len(rs), args.cluster_cap) >= 20)
    print(f"clusters with >=20 keeps: {clusters20}")

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for r in keeps:
        name = os.path.basename(r["wav"])
        shutil.copy2(r["wav"], os.path.join(args.out, name))
        manifest.append({**{k: r[k] for k in ("tar", "speaker", "text", "duration",
                                              "dnsmos", "T", "V", "A", "criteria")},
                         "file": name,
                         "license": "Emilia-YODAS (CC-BY-4.0 subset of amphion/Emilia-Dataset)"})
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"criteria": vars(args), "thresholds": thr, "keeps": manifest}, f, indent=2)

    # Owner audit sample: spread across criteria groups.
    random.seed(args.seed)
    audit_dir = os.path.join(args.out, "audit_sample")
    os.makedirs(audit_dir, exist_ok=True)
    per_group = max(args.audit_n // 4, 1)
    for c in ("T+", "V+", "V-", "A+"):
        grp = [r for r in keeps if c in r["criteria"]]
        for i, r in enumerate(random.sample(grp, min(per_group, len(grp)))):
            tag = c.replace("+", "hi").replace("-", "lo")
            shutil.copy2(r["wav"], os.path.join(audit_dir, f"{tag}_{i:02d}_{os.path.basename(r['wav'])}"))
    print(f"keeps + manifest + audit_sample -> {args.out}")


if __name__ == "__main__":
    main()

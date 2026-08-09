# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Label the Emilia-YODAS keeps on a GLOBAL anchor derived from the LibriTTS corpus.

Owner decision 2026-08-08, Phase 1 #1 option 1: keep the anchored values and map them
into [-1, 1], rather than run Emilia through the corpus lane's per-speaker z.

WHY NOT PER-SPEAKER Z — the thing this script exists to avoid
------------------------------------------------------------
`derive_vat_corpus` computes V/A/T as a per-speaker z-score, which is right for LibriTTS
(247 speakers, 127 clips each) and wrong for Emilia in a way that destroys the corpus.

The keeps are TAIL-SELECTED by pre-registered criteria (`mine_emilia_keeps.py`: keep iff
`T_full > p90`, `V_combo > p95`, `V_combo < p5` or `EIV-Arousal > p95`), and as mined they
look like it — A mean +1.168, T mean +4.541, 91.9% of clips beyond |T| = 1. But Emilia is
13,141 clips across **2,408 speakers, a median of 3 clips each**; 85.4% of speakers hold
fewer than 10 clips, covering 46.6% of the corpus.

Re-centring each speaker on their own kept clips takes V from mean +0.387 / sd 0.741 to
mean **+0.000 / sd 0.971**, and hands **756 one-clip speakers a label of exactly 0.0** —
which in this representation means *at the speaker mean*, i.e. neutral. Clips selected FOR
being extreme would train as neutral. That is D-M1's principle a second time: a
manufactured 0.0 is indistinguishable from a measured one, and it is how 1,094 clips
shipped mislabelled once already.

WHAT THIS DOES INSTEAD
----------------------
Every channel is z-scored against the **LibriTTS corpus's own global distribution of the
same underlying measure**, then put through the identical `clamp2` (÷2, clamp to [-1, 1])
the corpus lane ends with. The anchor is re-derived from the v4 corpus rather than
inherited from v2, which is what `mine_emilia_keeps` originally anchored to.

  A  = clamp2( zg(lufs) )
  T  = clamp2( zg( zg(alpha_db) + zg(cpp) - zg(h1h2) - zg(soft) ) )
  V  = clamp2( zg( dot(weights, zg(head)) ) )

`zg` is always "z against the LibriTTS global mean/sd of that quantity". The composite is
the same one `derive_vat_corpus` builds — same components, same signs, same second
normalisation — with `per_spk_z` swapped for the global anchor at every step.

⚠ THE SEMANTIC DIFFERENCE, STATED RATHER THAN HIDDEN. LibriTTS labels mean "relative to
THIS SPEAKER"; these mean "relative to the corpus". Per-speaker z exists to keep speaker
identity out of the affect channels — a constitutionally loud reader is not high-arousal —
and a global anchor leaves some of that identity in. With a median of 3 clips per speaker
there is no per-speaker statistic to be had, so the choice is between a global anchor and
a manufactured zero. This is the former, and the speaker embedding is what the model has
to disentangle the residue with.

Three EIV heads in the weight spec are absent from Emilia's scores
(`Thankfulness_Gratitude`, `Disappointment`, `Distress`). All three carry weight **0.0**,
so their absence changes nothing — the same note `eiv_merge_corpus` makes.

Usage:
    .venv/bin/python scripts/anchor_emilia_labels.py            # report only
    .venv/bin/python scripts/anchor_emilia_labels.py --out <path.jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

CORPUS = "data/libritts_r_vat_v4"
EIV = "/data/model-training/sonora/eiv_scores"
RAW_HEADS = ("corpus_v1.jsonl", "corpus_families.jsonl")
MINING = "/data/model-training/sonora/emilia_mining"
KEEPS = "/data/model-training/datasets/emilia_kept/manifest.json"
KEPT_24K = "/data/model-training/datasets/emilia_kept_24k"
SOFT_HEAD = "Soft_vs._Harsh"
COMPONENTS = (("alpha_db", 1.0), ("cpp", 1.0), ("h1h2", -1.0))


class LabelError(ValueError):
    """A clip whose label cannot be trusted. Never silently degraded into a number."""


def clamp2(z):
    """The corpus lane's final mapping, unchanged — see derive_vat_corpus."""
    return np.clip(z / 2.0, -1.0, 1.0)


def _stats(values):
    a = np.asarray(values, float)
    # Same 1e-6 floor as the corpus lane (D-L2): a constant measure must read as zero,
    # not as dust divided by dust.
    return float(a.mean()), float(a.std() + 1e-6)


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def libritts_anchor():
    """Global mean/sd of every quantity the labels are built from, over the v4 corpus."""
    kept = {ln.split("|")[0] for p in ("train_op.txt", "val_op.txt")
            for ln in open(os.path.join(CORPUS, p), encoding="utf-8")}
    measures = {r["wav"]: r for r in _jsonl(os.path.join(CORPUS, "measures.jsonl"))
                if r["wav"] in kept}
    heads: dict[str, dict] = {}
    for name in RAW_HEADS:
        for r in _jsonl(os.path.join(EIV, name)):
            heads.setdefault(r["wav"], {}).update({k: v for k, v in r.items() if k != "wav"})
    heads = {w: h for w, h in heads.items() if w in kept}
    spec = json.load(open(os.path.join(EIV, "valence_combo_v1.json"), encoding="utf-8"))
    weights = {h: w for h, w in zip(spec["heads"], spec["weights"]) if w}

    anchor = {"n_clips": len(measures), "n_heads": len(heads), "weights": weights}
    for name, _ in COMPONENTS:
        anchor[name] = _stats([m[name] for m in measures.values()])
    anchor["lufs"] = _stats([m["lufs"] for m in measures.values()])
    for head in list(weights) + [SOFT_HEAD]:
        anchor[f"head:{head}"] = _stats([h[head] for h in heads.values() if head in h])

    # The two composites need their own anchor, because each is z-scored a SECOND time.
    order = sorted(set(measures) & set(heads))
    t_raw, v_raw = [], []
    for w in order:
        m, h = measures[w], heads[w]
        t = sum(sign * (m[n] - anchor[n][0]) / anchor[n][1] for n, sign in COMPONENTS)
        s = anchor[f"head:{SOFT_HEAD}"]
        t -= (h[SOFT_HEAD] - s[0]) / s[1]
        t_raw.append(t)
        v_raw.append(sum(wt * (h[hd] - anchor[f"head:{hd}"][0]) / anchor[f"head:{hd}"][1]
                         for hd, wt in weights.items() if hd in h))
    anchor["t_composite"] = _stats(t_raw)
    anchor["v_composite"] = _stats(v_raw)
    return anchor


def emilia_rows():
    """-> {wav_basename: {measures + heads}} for every mined keep."""
    measures, heads = {}, {}
    for p in sorted(glob.glob(os.path.join(MINING, "*", "probe_measures.jsonl"))):
        for r in _jsonl(p):
            measures[os.path.basename(r["wav"])] = r
    for p in sorted(glob.glob(os.path.join(MINING, "*", "eiv_scores.jsonl"))):
        for r in _jsonl(p):
            heads[os.path.basename(r["wav"])] = r
    keeps = json.load(open(KEEPS, encoding="utf-8"))["keeps"]
    out, missing = {}, []
    for k in keeps:
        base = os.path.splitext(k["file"])[0]
        src = base + ".mp3"
        m, h = measures.get(src), heads.get(src)
        if m is None or h is None:
            missing.append(base)
            continue
        out[base] = {**m, **{f"head:{kk}": vv for kk, vv in h.items() if kk != "wav"},
                     "speaker": k.get("speaker"), "text": k.get("text")}
    return out, missing


def label(row, anchor):
    """One clip's (V, A, T) on the global anchor, in the corpus's own [-1, 1].

    Raises `LabelError` on a non-finite result or a missing weighted head (TR-M2). The
    derive lane aborts on exactly this and this lane did not, though it produces labels
    for the same three channels of the same corpus.

    **Why a NaN here is worse than a crash.** `np.clip` propagates it, the filelist
    carries the string `"nan"`, and `float("nan")` parses cleanly at load
    (`text_mel_datamodule`), so a poisoned conditioning vector reaches the FiLM trunk with
    nothing having failed anywhere. The artifact test catches the all-three-NaN case; a
    single-channel NaN passes every test we have. The shipped v5 corpus was verified clean
    (0 non-finite / out-of-range rows) — this is the guard for the NEXT anchor run.

    The missing-head check is the quieter half. `label()` used to skip a weighted head it
    could not find, which does not fail — it biases V toward 0 in units indistinguishable
    from a real measurement. `mine_emilia_keeps` checks one sample row per tar, so an EIV
    run that degrades partway through a file is invisible to it. Silence is the failure
    mode; the count is the fix.
    """
    z = lambda key, val: (val - anchor[key][0]) / anchor[key][1]
    a = clamp2(z("lufs", row["lufs"]))
    t = sum(sign * z(n, row[n]) for n, sign in COMPONENTS)
    t -= z(f"head:{SOFT_HEAD}", row[f"head:{SOFT_HEAD}"])
    t = clamp2((t - anchor["t_composite"][0]) / anchor["t_composite"][1])
    missing = [h for h in anchor["weights"] if f"head:{h}" not in row]
    if missing:
        raise LabelError(f"{len(missing)} weighted head(s) absent from this row, "
                         f"e.g. {sorted(missing)[:3]} — V would be biased toward 0 in "
                         f"units indistinguishable from a measurement")
    v = sum(w * z(f"head:{h}", row[f"head:{h}"])
            for h, w in anchor["weights"].items() if f"head:{h}" in row)
    v = clamp2((v - anchor["v_composite"][0]) / anchor["v_composite"][1])
    out = (float(v), float(a), float(t))
    bad = [n for n, x in zip("VAT", out) if not np.isfinite(x)]
    if bad:
        raise LabelError(f"non-finite label on channel(s) {bad} — this would reach the "
                         f"filelist as 'nan', parse cleanly at load, and poison the FiLM "
                         f"trunk mid-run")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="write per-clip labels as jsonl")
    args = ap.parse_args()

    anchor = libritts_anchor()
    print(f"anchor: {anchor['n_clips']} LibriTTS clips, {anchor['n_heads']} with EIV heads, "
          f"{len(anchor['weights'])} weighted heads")
    rows, missing = emilia_rows()
    print(f"emilia: {len(rows)} keeps labelled" + (f", {len(missing)} unmatched" if missing else ""))
    if missing:
        print(f"  !! {len(missing)} keep(s) have no probe measure or EIV score: "
              f"{missing[:3]} — they cannot be labelled and must not be written")

    lab, unlabelable = {}, []
    for b, r in rows.items():
        try:
            lab[b] = label(r, anchor)
        except LabelError as e:
            unlabelable.append((b, str(e)))
    if unlabelable:
        # Reported here rather than raised, because this script is the REPORT: it prints
        # the distribution the owner reads before the merge consumes the same labels. A
        # crash would hide the other 41k rows' summary; a silent skip would misstate it.
        # The merge drops these clips (TR-M2).
        print(f"\n!! {len(unlabelable)} clip(s) CANNOT be labelled and must not be "
              f"written:")
        for b, why in unlabelable[:3]:
            print(f"     {b}: {why[:100]}")
    arr = np.array(list(lab.values()))
    ref = np.array([[float(x) for x in ln.split("|")[3].split(",")[:3]]
                    for ln in open(os.path.join(CORPUS, "train_op.txt"), encoding="utf-8")])
    print(f"\n{'channel':8s} {'emilia mean':>12} {'sd':>7} {'|v|>0.5':>8} {'rail':>7}   "
          f"{'libritts mean':>14} {'sd':>7}")
    for i, name in enumerate(("V", "A", "T")):
        e, r = arr[:, i], ref[:, i]
        print(f"{name:8s} {e.mean():+12.3f} {e.std():7.3f} {(np.abs(e)>0.5).mean()*100:7.1f}% "
              f"{(np.abs(e)>=1.0).mean()*100:6.1f}%   {r.mean():+14.4f} {r.std():7.3f}")
    zeros = int((np.abs(arr) < 1e-9).all(axis=1).sum())
    print(f"\nclips labelled all-zero (the per-speaker-z failure mode): {zeros}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for base, (v, a, t) in sorted(lab.items()):
                fh.write(json.dumps({"wav": os.path.join(KEPT_24K, "wavs", base + ".wav"),
                                     "speaker": rows[base]["speaker"],
                                     "text": rows[base]["text"],
                                     "V": round(v, 4), "A": round(a, 4),
                                     "T": round(t, 4)}) + "\n")
        print(f"wrote {len(lab)} rows -> {args.out}")


if __name__ == "__main__":
    main()

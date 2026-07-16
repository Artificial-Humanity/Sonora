"""Assemble the human-audit set for the tension label (calibration step 2,
tension-definition-brief.md).

Picks the strongest +T (pressed?) and -T (breathy?) clips from a v1-derived
filelist at MATCHED LOUDNESS (|A| below a threshold), so the audit hears
phonation, not volume, plus neutral controls. Copies WAVs into an audit dir
with self-describing names (rank_Tlabel_Alabel_spk.wav) and writes a
manifest with the transcript for each.

The audit question, verbatim from the brief: "does +T sound *strained* and
-T sound *breathy*, at matched loudness?" If it fails, fall back to
dominance (option B) rather than stretching the definition.

Usage:
    python scripts/make_tension_audit_set.py \
        [--filelists data/libritts_r_vat_v1/train_op.txt data/libritts_r_vat_v1/val_op.txt] \
        [--out /data/model-training/sonora/tension_audit] \
        [--n 20] [--a-max 0.3]
"""

import argparse
import json
import os
import shutil


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) != 4:
                    continue
                wav, spk, _ipa, vat = parts
                v, a, t = (float(x) for x in vat.split(","))
                rows.append({"wav": wav, "spk": int(spk), "v": v, "a": a, "t": t})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelists", nargs="+",
                    default=["data/libritts_r_vat_v1/train_op.txt",
                             "data/libritts_r_vat_v1/val_op.txt"])
    ap.add_argument("--out", default="/data/model-training/sonora/tension_audit")
    ap.add_argument("--n", type=int, default=20, help="clips per extreme")
    ap.add_argument("--n-neutral", type=int, default=10)
    ap.add_argument("--a-max", type=float, default=0.3,
                    help="only clips with |A| below this (matched loudness)")
    ap.add_argument("--per-speaker-cap", type=int, default=3,
                    help="max clips per speaker per group, for variety")
    args = ap.parse_args()

    rows = load_rows(args.filelists)
    quiet = [r for r in rows if abs(r["a"]) <= args.a_max]
    print(f"{len(rows)} rows, {len(quiet)} at |A|<={args.a_max}")

    def pick(pool, n):
        out, per_spk = [], {}
        for r in pool:
            if per_spk.get(r["spk"], 0) >= args.per_speaker_cap:
                continue
            per_spk[r["spk"]] = per_spk.get(r["spk"], 0) + 1
            out.append(r)
            if len(out) == n:
                break
        return out

    high = pick(sorted(quiet, key=lambda r: -r["t"]), args.n)
    low = pick(sorted(quiet, key=lambda r: r["t"]), args.n)
    mid = pick(sorted(quiet, key=lambda r: abs(r["t"])), args.n_neutral)

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for group, clips in (("pressed", high), ("breathy", low), ("neutral", mid)):
        for i, r in enumerate(clips):
            txt_path = r["wav"].rsplit(".", 1)[0] + ".normalized.txt"
            text = open(txt_path, encoding="utf-8").read().strip() \
                if os.path.exists(txt_path) else ""
            name = f"{group}_{i:02d}_T{r['t']:+.2f}_A{r['a']:+.2f}_spk{r['spk']}.wav"
            shutil.copy2(r["wav"], os.path.join(args.out, name))
            manifest.append({"file": name, "group": group, "text": text, **r})
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(manifest)} clips -> {args.out}")
    print("Audit question: do 'pressed_*' sound strained/tight and 'breathy_*' "
          "sound soft/airy, relative to 'neutral_*'? (Loudness is matched.)")


if __name__ == "__main__":
    main()

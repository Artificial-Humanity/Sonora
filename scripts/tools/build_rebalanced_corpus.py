#!/usr/bin/env python3
"""Build a REBALANCED copy of a VAT corpus by oversampling its minority sources.

Why this exists (2026-08-29). Rung 3 added 303,627 rows and the model reached its final
holdout value in 3,505 optimizer steps -- a third of one epoch. The measured reason is
composition, not size: v7 is 96.6% LibriTTS-R read speech, and the 480 hours it added were
all the domain the model already had. The lever the ladder has not pulled is DIVERSITY.

This pulls it without collecting anything. The minority rows already exist; they are simply
outvoted 28-to-1. Repeating them in the filelist raises their share of every batch, which is
what a weighted sampler would do, except that the result is a FILE you can diff, hash and
hand to someone else. No training code changes.

WHY OVERSAMPLE THE MINORITY RATHER THAN DOWNSAMPLE THE MAJORITY -- both reach the same
per-batch ratio, and downsampling would be smaller and faster. Downsampling fixes the
LibriTTS rows to one subset, so the run sees less of the majority domain as well as
proportionally less of it. That confounds "more diverse" with "less data". Oversampling
changes exactly one thing.

⚠ THE DUPLICATES ARE REAL ROWS, NOT WEIGHTS. Each appears in the epoch that many times.
For a short probe this is the same thing; over many epochs it is a memorisation risk on the
smallest source, so check the repeat count this prints before running long.

Usage:
    build_rebalanced_corpus.py --source <corpus-dir> --out <corpus-dir> --factor 9
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

# Which corpus a row came from is recoverable from its audio path, because every source
# lives under its own directory in /data/model-training/datasets. Deriving it beats carrying
# a source column the filelist format has no room for.
MAJORITY = "LibriTTS_R"


def source_of(line: str) -> str:
    parts = line.split("|", 1)[0].split("/")
    # /data/model-training/datasets/<SOURCE>/...
    return parts[4] if len(parts) > 4 else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="existing corpus directory")
    ap.add_argument("--out", required=True, type=Path, help="corpus directory to write")
    ap.add_argument("--factor", required=True, type=int,
                    help="how many times each minority row is repeated")
    args = ap.parse_args()

    train_in = args.source / "train_op.txt"
    rows = train_in.read_text().splitlines()
    by_source = Counter(source_of(r) for r in rows)

    # ⚠ REFUSE ON AN UNEXPECTED LAYOUT rather than silently classifying every row as
    # minority. If the majority source is not present, the path convention this depends on
    # has changed, and the "rebalanced" corpus would be 9 copies of everything -- which is
    # just v7 with a longer epoch, and would look like a valid experiment.
    if MAJORITY not in by_source:
        raise SystemExit(f"REFUSING: no {MAJORITY} rows in {train_in}. Sources found: "
                         f"{dict(by_source)}. The path convention this reads has changed.")
    if args.factor < 1:
        raise SystemExit("REFUSING: --factor must be at least 1.")

    out_rows = []
    for r in rows:
        out_rows.append(r)
        if source_of(r) != MAJORITY:
            out_rows.extend([r] * (args.factor - 1))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_op.txt").write_text("\n".join(out_rows) + "\n")

    # Val and the speaker map pass through UNCHANGED and this is deliberate. The speaker
    # table must stay index-identical or every warm start from a v7 checkpoint is wrong, and
    # a rebalanced val split would not be comparable with the run this is measured against.
    for name in ("val_op.txt", "speakers.json"):
        src = args.source / name
        if src.exists():
            shutil.copy2(src, args.out / name)

    after = Counter(source_of(r) for r in out_rows)
    total_before, total_after = len(rows), len(out_rows)
    min_before = total_before - by_source[MAJORITY]
    min_after = total_after - after[MAJORITY]

    report = {
        "built_from": str(args.source),
        "factor": args.factor,
        "majority_source": MAJORITY,
        "rows_before": total_before,
        "rows_after": total_after,
        "minority_share_before": round(min_before / total_before, 6),
        "minority_share_after": round(min_after / total_after, 6),
        "by_source_before": dict(by_source),
        "by_source_after": dict(after),
        "train_op_sha256": hashlib.sha256(
            (args.out / "train_op.txt").read_bytes()).hexdigest(),
    }
    (args.out / "rebalance_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"  rows      {total_before:,} -> {total_after:,}")
    print(f"  minority  {min_before/total_before:.2%} -> {min_after/total_after:.2%}")
    for s, n in sorted(after.items(), key=lambda kv: -kv[1]):
        rep = args.factor if s != MAJORITY else 1
        print(f"    {s:<28} {n:>8,}  ({n/total_after:6.2%})  each row appears {rep}x")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

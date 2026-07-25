#!/usr/bin/env python3
"""Derive the controlled register lexicon from the audition SSOT.

markup-schema-brief.md §Field semantics: `utterance.register` is a **controlled
lexicon** = the expressive-registers vocabulary, "grown only via the
Recategorize/relabel flow — the app is the lexicon's governance surface". The
director must PICK from it, never invent.

That was never enforced. As of 2026-07-25 ratings.csv carries 138 distinct register
labels across 554 certified keeps, and a single director pass over 20 lines emitted
54 labels — the same threat line came back as low_menace / menacing_threat /
quiet_menace depending on which engine it was writing for. Free-form labels
fragment the training set.

Regenerate whenever the owner has recategorized enough new material to promote a
label (default threshold: 3 certified keeps).

Usage:
    python scripts/synthesis/build_register_lexicon.py [--min-keeps 3]
"""
import argparse
import collections
import csv
import json
import os

RATINGS = "/data/model-training/datasets/sonora-expressive-registers/ratings.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "register_lexicon.json")


def build(min_keeps: int):
    with open(RATINGS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keeps = [r for r in rows if r.get("status") == "keep" and r.get("register")]
    counts = collections.Counter(r["register"] for r in keeps)
    lexicon = sorted(k for k, v in counts.items() if v >= min_keeps)
    return lexicon, counts, len(keeps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-keeps", type=int, default=3,
                    help="certified keeps required to promote a label")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    lexicon, counts, n_keeps = build(args.min_keeps)
    payload = {
        "source": RATINGS,
        "min_keeps": args.min_keeps,
        "n_certified_keeps": n_keeps,
        "n_distinct_labels_seen": len(counts),
        "lexicon": lexicon,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")
    print(f"  {len(lexicon)} labels promoted from {len(counts)} seen "
          f"across {n_keeps} certified keeps (threshold {args.min_keeps})")
    demoted = [k for k, v in counts.most_common() if v < args.min_keeps]
    print(f"  {len(demoted)} below threshold (not in lexicon), e.g. {demoted[:8]}")


if __name__ == "__main__":
    main()

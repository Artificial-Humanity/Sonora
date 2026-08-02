"""Re-partition an existing VAT corpus onto the per-clip hash split.

Owner decision 2026-08-02. `derive_vat_corpus.py` used to assign train/val by
shuffling the row list, which makes membership a property of a clip's POSITION —
so adding one clip re-rolls every clip after it. Measured on the corpora this was
found in: v2-val and v3-val share 19 clips of ~910, and 889 of v3's val clips sit
in v2's TRAIN set. Cross-version val-loss comparison is the entire point of a
relabel, and the split silently made it impossible.

The fix lives in `derive_vat_corpus._in_val`, so a fresh derivation already comes
out right. This script exists for the corpus that is already built: it moves rows
between train_op.txt and val_op.txt and changes NOTHING else — same rows, same
phonemes, same labels, same speaker index. Re-deriving to change only which side
of a line each row sits on would mean re-running phonemization over 31k rows to
get a byte-identical result.

    .venv/bin/python scripts/resplit_corpus.py --src data/libritts_r_vat_v3b \\
        --out data/libritts_r_vat_v3c

Refuses to write over the source: the old split is the record of what a previous
checkpoint trained on, and overwriting it in place would erase the evidence for
the very leak this fixes.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = ("train_op.txt", "val_op.txt")
CARRY = ("speakers.json", "measures.jsonl", "derivation_report.json")


def _load_in_val():
    """Import the split rule from derive_vat_corpus without running it.

    Imported rather than copied on purpose. A second implementation of the hash
    is a second thing that can drift, and a drifted split is invisible — it
    produces a perfectly plausible val set that just is not the same one.
    """
    spec = importlib.util.spec_from_file_location(
        "derive_vat_corpus", os.path.join(HERE, "derive_vat_corpus.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["derive_vat_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod._in_val, mod.VAL_FRACTION, mod.SPLIT_SALT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="existing corpus dir")
    ap.add_argument("--out", required=True, help="new corpus dir (must not be --src)")
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = ap.parse_args()

    src, out = os.path.abspath(args.src), os.path.abspath(args.out)
    if src == out:
        sys.exit("refusing to resplit in place — pass a new --out")

    in_val, fraction, salt = _load_in_val()

    rows, was_val = [], set()
    for part in PARTS:
        path = os.path.join(src, part)
        if not os.path.isfile(path):
            sys.exit(f"missing {path}")
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line)
            if part == "val_op.txt":
                was_val.add(line)

    val = [r for r in rows if in_val(r)]
    train = [r for r in rows if not in_val(r)]
    if not val:
        sys.exit(f"ABORT: hash split put 0 of {len(rows)} rows in val")

    moved_in = len(set(val) - was_val)
    moved_out = len(was_val - set(val))
    print(f"src   {src}")
    print(f"rows  {len(rows)}   was {len(was_val)} val   now {len(val)} val "
          f"({len(val) / len(rows):.2%}, target {fraction:.0%})")
    print(f"moved {moved_in} into val, {moved_out} out of it   salt={salt}")
    if not args.apply:
        print("\nreport only — pass --apply to write")
        return

    os.makedirs(out, exist_ok=True)
    for name, part in (("train_op.txt", train), ("val_op.txt", val)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        print(f"wrote {len(part):6d} rows -> {os.path.join(out, name)}")
    for name in CARRY:
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out, name))
            print(f"carried {name}")

    # Say in the corpus itself how its split was made. A directory that cannot
    # explain its own membership is how the leak survived three versions.
    report_path = os.path.join(out, "derivation_report.json")
    report = {}
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    report["split"] = {
        "method": "per-clip blake2b hash of the wav basename",
        "salt": salt,
        "val_fraction_target": fraction,
        "val_fraction_actual": round(len(val) / len(rows), 5),
        "n_val": len(val),
        "n_train": len(train),
        "resplit_from": os.path.basename(src),
        "moved_into_val": moved_in,
        "moved_out_of_val": moved_out,
        "note": ("Membership is a property of the clip, not its position, so "
                 "growing the corpus never re-rolls it. Splits made before "
                 "2026-08-02 shuffled the row list and are NOT comparable to "
                 "this one."),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote split provenance -> {report_path}")


if __name__ == "__main__":
    main()

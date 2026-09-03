#!/usr/bin/env python3
"""Measure how far a corpus's delivery labels are confounded with speaker identity.

WHY THIS EXISTS (2026-09-03)
----------------------------
Three explanations for the delivery channel degrading the audio are dead on measurement —
the sampler, row duplication, and simple lane starvation (`configs/experiment/
vat7r_rebalance.yaml` § LANE CONTROL). None of them looked at how the labels are DISTRIBUTED
over speakers, and that turns out to be the whole story: in v7 almost every labelled speaker
id carries exactly one lane, so "Newscaster" and "the 76 voices that only ever appear under
Newscaster" are the same fact to the model. A channel cannot be learned as a factor unless
something else holds still while it changes.

This is the design of that measurement, written down rather than left in a transcript. The
ep010 probe lost its design that way and `probe_delivery_intercept.py` exists because of it.

THE HEADLINE NUMBER is `crossed_fraction`: of the rows that carry a lane, the share whose
speaker id carries at least two DIFFERENT lanes somewhere in the corpus. That is the only
population from which the model can separate delivery from voice. It is the bar the
remediation bank in `notes/delivery-lane-remediation.md` is built to move.

⚠ AN EMPTY POPULATION IS NOT A CLEAN RESULT. A corpus with no delivery labels at all would
otherwise report a tidy 0.0% confounded, which reads like a measurement and is not one. Zero
labelled rows exits 3 and says so.

⚠ The lane vocabulary is READ FROM `matcha.delivery`, never restated here. `lane_of_vector`
is also the validating reader — it refuses a block that is not one-hot, so a merged or an
interpolated label fails loudly instead of being counted as whichever lane came first.

⚠ The `source` column is path-derived and best-effort. It is reported because it is what
shows a lane resting on one dataset, and no other number here depends on it.

Run:  .venv/bin/python scripts/tools/measure_delivery_confound.py --corpus data/libritts_r_full_vat_v7
Exit: 0 measured; 2 the input is not a corpus this can read (no filelists, a malformed row,
      or a filelist whose conditioning width is not `VAT_DIM`); 3 nothing carried a lane.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import collections
import json
import sys

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO,):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from matcha.delivery import (  # noqa: E402
    ACTIVE_DELIVERY_LANES,
    DELIVERY_LANES,
    DELIVERY_UNKNOWN,
    VAT_DIM,
    lane_of_vector,
)

FILELISTS = ("train_op.txt", "val_op.txt")

# ⚠ `raise SystemExit("text")` exits 1, not the code the docstring names. Two distinct
# refusals matter to a caller — "this is not a corpus I can read" and "there was nothing to
# measure" — so they get their own codes and go through here rather than being spelled at
# each site, where they drifted from the docstring on the first draft of this file.
EXIT_UNREADABLE = 2
EXIT_NO_POPULATION = 3


def _refuse(code, message):
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _source_of(audio_path):
    """Best-effort dataset name for a clip. Reporting only — see the module docstring."""
    if "/datasets/" in audio_path:
        return audio_path.split("/datasets/", 1)[1].split("/", 1)[0]
    return _os.path.basename(_os.path.dirname(audio_path)) or "?"


def read_rows(corpus_dir):
    """Yield (source, speaker_id, lane) for every row in the corpus's filelists.

    `lane` is `DELIVERY_UNKNOWN` for a row whose delivery block is all zeros, which is a
    real value in the contract and not a missing one.
    """
    present = [f for f in FILELISTS if _os.path.exists(_os.path.join(corpus_dir, f))]
    if not present:
        _refuse(EXIT_UNREADABLE,
                f"ABORT: {corpus_dir} holds none of {list(FILELISTS)} — that is not a corpus "
                f"directory, or the layout changed. Nothing measured.")
    for name in present:
        path = _os.path.join(corpus_dir, name)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                parts = line.rstrip("\n").split("|")
                if len(parts) < 4:
                    _refuse(EXIT_UNREADABLE,
                            f"ABORT: {path}:{lineno} has {len(parts)} fields, expected at "
                            f"least 4 (audio|speaker|phonemes|vat).")
                vec = [float(x) for x in parts[3].split(",")]
                if len(vec) != VAT_DIM:
                    _refuse(EXIT_UNREADABLE,
                            f"ABORT: {path}:{lineno} carries {len(vec)} conditioning "
                            f"channels, not {VAT_DIM}. A 3-wide filelist has no delivery "
                            f"block to measure — see scripts/gates/test_vat_dim_seams.py.")
                yield _source_of(parts[0]), parts[1], lane_of_vector(vec)


def measure(corpus_dir):
    rows_by_spk = collections.Counter()
    lanes_by_spk = collections.defaultdict(set)
    labelled_by_spk = collections.Counter()
    lane_rows = collections.Counter()
    lane_spk = collections.defaultdict(set)
    lane_src = collections.defaultdict(collections.Counter)
    total = 0

    for source, spk, lane in read_rows(corpus_dir):
        total += 1
        rows_by_spk[spk] += 1
        if lane == DELIVERY_UNKNOWN:
            continue
        labelled_by_spk[spk] += 1
        lanes_by_spk[spk].add(lane)
        lane_rows[lane] += 1
        lane_spk[lane].add(spk)
        lane_src[lane][source] += 1

    labelled = sum(lane_rows.values())
    if labelled == 0:
        _refuse(EXIT_NO_POPULATION,
                f"ABORT (nothing to measure): {total} rows in {corpus_dir} and not one "
                f"carries a delivery lane. This is NOT '0% confounded' — there is no "
                f"population. If that is expected for this corpus, it has no delivery "
                f"signal to confound.")

    crossed_rows = sum(n for spk, n in labelled_by_spk.items() if len(lanes_by_spk[spk]) > 1)
    active = set(ACTIVE_DELIVERY_LANES)
    return {
        "corpus": corpus_dir,
        "rows_total": total,
        "rows_labelled": labelled,
        # THE HEADLINE. Rows whose speaker also appears under another lane, over all
        # labelled rows. This is the only population that teaches delivery as a factor.
        "crossed_rows": crossed_rows,
        "crossed_fraction": crossed_rows / labelled,
        "speakers_labelled": len(labelled_by_spk),
        "speakers_by_lane_count": dict(sorted(
            collections.Counter(len(v) for v in lanes_by_spk.values()).items())),
        "speakers_covering_all_active_lanes": sum(
            1 for v in lanes_by_spk.values() if active <= v),
        # A speaker seen once is memorised, not learned, whatever its label says.
        "labelled_speakers_holding_one_row": sum(
            1 for spk in labelled_by_spk if rows_by_spk[spk] == 1),
        "lanes": {
            lane: {
                "rows": lane_rows[lane],
                "speakers": len(lane_spk[lane]),
                "sources": dict(lane_src[lane].most_common()),
            }
            for lane in DELIVERY_LANES if lane_rows[lane]
        },
        "active_lanes": list(ACTIVE_DELIVERY_LANES),
    }


def report(m):
    print(f"corpus            : {m['corpus']}")
    print(f"rows              : {m['rows_total']} total, {m['rows_labelled']} carry a lane")
    print()
    print(f"{'lane':14s}{'rows':>7s}{'speakers':>10s}   sources")
    for lane, d in m["lanes"].items():
        srcs = ", ".join(f"{k}:{v}" for k, v in d["sources"].items())
        print(f"{lane:14s}{d['rows']:7d}{d['speakers']:10d}   {srcs}")
    print()
    print(f"speakers carrying a lane            : {m['speakers_labelled']}")
    print(f"  by number of DIFFERENT lanes held : {m['speakers_by_lane_count']}")
    print(f"  covering all {len(m['active_lanes'])} assignable lanes    : "
          f"{m['speakers_covering_all_active_lanes']}")
    print(f"  holding exactly one row in total  : {m['labelled_speakers_holding_one_row']}")
    print()
    print(f"CROSSED ROWS      : {m['crossed_rows']} of {m['rows_labelled']} "
          f"= {m['crossed_fraction'] * 100:.2f}%")
    print("  the share of the delivery signal from which lane can be told apart from voice")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True,
                    help="a corpus directory holding train_op.txt / val_op.txt")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the measurement as JSON, for a note or a gate to cite")
    args = ap.parse_args()

    m = measure(args.corpus)
    report(m)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

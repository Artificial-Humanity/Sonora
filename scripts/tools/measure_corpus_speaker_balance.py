#!/usr/bin/env python3
"""Measure how a corpus's rows are distributed over speakers and over source partitions.

WHY THIS EXISTS (2026-09-18)
----------------------------
The rung 3 selection probe could not separate its two checkpoints, and the free text said
why: both arms were judged robotic on both voices. The question that turns that from a
complaint into a lead is *which* voices were being judged — and the answer was not in any
note, because nothing measured it.

They were ranks 1 and 2 of 5,332. The corpus median is 7 rows. A ladder whose next rung
adds volume is answered very differently depending on whether the ear was listening to the
thick end of that distribution or the thin end, and until this ran, nobody could say.

THE TWO HEADLINE NUMBERS
------------------------
`rows_per_speaker` percentiles say whether "more data" means more speakers or more rows for
the speakers already here. A median of 7 against a max of 678 is not a corpus of 5,332
voices in any sense the model can use.

`half_the_rows` is the speaker count holding 50% of all rows. It is the concentration
figure, and it is the one to quote: percentiles hide how few speakers the mass sits on.

The partition composition is the second axis. LibriTTS's own `clean`/`other` split is a
recording-quality judgement made by the corpus authors, so a corpus that is mostly `other`
has a quality story its row count does not tell.

⚠ AN EMPTY CORPUS IS NOT A BALANCED ONE. Zero rows would otherwise report tidy percentiles
over an empty list and a concentration of "0 speakers hold half the rows", which reads like
a measurement. Zero rows exits 3 and says so.

⚠ THE DEFAULT MEASURES `train`, WHICH IS NOT THE CORPUS. That is the right population for
a question about what the model was exposed to, but v7's val filelist holds 53 speakers the
train filelist does not, so the two disagree on the speaker count (5,332 against 5,385). The
header says which filelists were read and prints how many speakers the split leaves out.

⚠ THE PARTITION KEY IS PATH-DERIVED AND THE DERIVATION IS PRINTED. Every row's path is
taken relative to the longest directory prefix they all share, and the key is its first
`--depth` components. That is a convention about how the datasets happen to be laid out on
this box, not a fact about the corpus, so the tool prints the root it found and the keys it
built rather than asking to be believed. Paths with no shared root at all exit 2.

⚠ DEPTH IS REFUSED AT BOTH ENDS. `--depth 2` under a dataset that puts speaker directories
at that level produces one group per speaker, which is not a partition breakdown and should
not be printed as one; more than `--max-groups` groups exits 2, having SEARCHED for a
shallower depth that fits rather than naming depth 1 on faith. Below 1 is refused outright:
depth 0 keys every row on the empty string and prints one nameless group holding 100% of the
corpus, and a negative depth slices from the end of the path and names speaker directories.

⚠ THE CONDITIONING WIDTH IS CHECKED, NOT ASSUMED. A filelist is `vat_dim`-coupled and a
3-wide one will parse cleanly here and mean something else entirely, so every row's vector
is counted against `VAT_DIM` and a mismatch exits 2.

Run:  .venv/bin/python scripts/tools/measure_corpus_speaker_balance.py \
          --corpus data/libritts_r_full_vat_v7 --speakers 4896,4899
Exit: 0 measured; 2 the input is not a corpus this can read (no filelist, a malformed row,
      a conditioning width that is not VAT_DIM, no shared path root, a depth below 1 or one
      that does not name partitions, every row unkeyed, a row count that disagrees with the
      file's own line count, or a --speakers index the SPLIT does not hold); 3 no rows.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import collections
import json
import os
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

from matcha.delivery import VAT_DIM  # noqa: E402

FILELISTS = ("train_op.txt", "val_op.txt")

# ⚠ `raise SystemExit("text")` exits 1, not the code the docstring names. Two distinct
# helpers so the exit code is chosen at the call site and cannot drift from the docstring.
def _refuse(msg):
    print("REFUSED: %s" % msg, file=sys.stderr)
    raise SystemExit(2)


def _empty(msg):
    print("REFUSED: %s" % msg, file=sys.stderr)
    raise SystemExit(3)


def count_rows(path):
    """Non-blank lines in a filelist, counted in its own pass.

    Deliberately independent of `read_rows`: it shares no state, no parsing and no
    accumulator with it, so comparing the two is a control rather than a tautology. An
    earlier version compared `len(rows)` to a total derived from `rows`, which no input
    could ever violate — the shape this repo files under "empty enumeration".
    """
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_rows(path):
    """Yield (speaker_index, audio_path) for each row, refusing anything malformed.

    The refusal names the file and the 1-based line, because the only thing worse than a
    corpus this cannot read is a corpus it reads wrong and reports percentiles for.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 4:
                _refuse("%s line %d states %d `|`-separated fields and a filelist row has 4"
                        % (path, n, len(parts)))
            try:
                spk = int(parts[1])
            except ValueError:
                _refuse("%s line %d states speaker %r, which is not an index"
                        % (path, n, parts[1]))
            width = len(parts[3].split(","))
            if width != VAT_DIM:
                _refuse("%s line %d states a %d-wide conditioning vector and `matcha.delivery` "
                        "defines VAT_DIM as %d — this filelist was built for a different model"
                        % (path, n, width, VAT_DIM))
            rows.append((spk, parts[0]))
    return rows


def common_root(paths):
    """The longest directory prefix every path shares, as a list of components.

    Returns None when they share nothing, which is a refusal rather than a root of `/`:
    keying on the first component of an unshared path gives one group per dataset root and
    reads as a partition breakdown without being one.
    """
    split = [p.strip("/").split("/") for p in paths]
    if not split:
        return None
    root = split[0][:-1]
    # ⚠ Checked BEFORE the loop as well as inside it. A single row whose path has no
    # directory part never entered the loop, so a one-row corpus escaped the refusal the
    # docstring promises and reported `ROOT /` with every row unkeyed.
    if not root:
        return None
    for parts in split[1:]:
        keep = 0
        for a, b in zip(root, parts[:-1]):
            if a != b:
                break
            keep += 1
        root = root[:keep]
        if not root:
            return None
    return root


def partition_of(path, root, depth):
    """The first `depth` path components below `root`, joined — or None if there are none."""
    parts = path.strip("/").split("/")
    rest = parts[len(root):-1]
    if not rest:
        return None
    return "/".join(rest[:depth])


def percentiles(values, points):
    """Nearest-rank percentiles over a sorted copy. `values` must be non-empty.

    Nearest rank is `ceil(p/100 * n)`, 1-based. `int(n * p / 100)` is the FLOOR, which
    agrees only when the product is not an integer — on 1..100 it shifts every percentile
    by one (p50 -> 51). It did not move the published v7 figures, because the ranks that
    landed on integers there happened to sit inside runs of equal values. That is luck,
    not correctness.
    """
    v = sorted(values)
    n = len(v)
    return {p: v[min(n - 1, max(0, -(-n * p // 100) - 1))] for p in points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="corpus directory holding train_op.txt / val_op.txt")
    ap.add_argument("--split", default="train", choices=("train", "val", "both"),
                    help="which filelist to measure (default: train)")
    ap.add_argument("--depth", type=int, default=1,
                    help="path components below the shared root that name a partition")
    ap.add_argument("--max-groups", type=int, default=50,
                    help="refuse a --depth that produces more groups than this")
    ap.add_argument("--speakers", default="",
                    help="comma-separated speaker indices to rank explicitly")
    ap.add_argument("--json", default="", help="also write the report here")
    args = ap.parse_args()

    # ⚠ ARGUMENTS ARE VALIDATED BEFORE ANY MEASUREMENT IS PRINTED. A bad `--speakers` used
    # to refuse at the very end, after a complete and correct report had already gone to
    # stdout under a non-zero exit — so stdout said the run happened and the exit code said
    # it had not, and `--json` was never written.
    if args.depth < 1:
        _refuse("--depth %d is not a partition depth. Depth 0 keys every row on the empty "
                "string and prints one nameless group holding 100%% of the corpus; a "
                "negative depth slices from the END of the path and names speaker "
                "directories. The shallowest real partition is --depth 1." % args.depth)
    wanted_speakers = []
    for token in args.speakers.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            wanted_speakers.append(int(token))
        except ValueError:
            _refuse("--speakers states %r, which is not an index" % token)

    wanted = {"train": ("train_op.txt",), "val": ("val_op.txt",),
              "both": FILELISTS}[args.split]
    rows = []
    read = []
    declared = 0
    for name in wanted:
        path = os.path.join(args.corpus, name)
        if not os.path.exists(path):
            _refuse("%s holds no %s — the corpus directories this reads are the ones "
                    "`derive_vat_corpus.py` writes, which always contain %s"
                    % (args.corpus, name, " and ".join(FILELISTS)))
        declared += count_rows(path)
        rows.extend(read_rows(path))
        read.append(name)

    # ⚠ POSITIVE CONTROL, against a count taken in a SEPARATE PASS over the file. The
    # populations that can actually go missing here are the ones a parse loop drops, and a
    # check derived from the parsed rows cannot see them. A refusal rather than an
    # `assert`, because `python -O` strips asserts and this one guards the denominator of
    # every number below it.
    if len(rows) != declared:
        _refuse("read %d rows but %s holds %d non-blank lines — rows were dropped between "
                "the file and the report, and every figure below divides by that count"
                % (len(rows), ", ".join(read), declared))

    if not rows:
        _empty("%s states 0 rows across %s — an empty corpus has no distribution to report, "
               "and reporting one would read as a measurement" % (args.corpus, ", ".join(read)))

    root = common_root([p for _, p in rows])
    if root is None:
        _refuse("the %d audio paths share no directory prefix, so there is no root to take a "
                "partition key relative to. First three: %s"
                % (len(rows), ", ".join(p for _, p in rows[:3])))

    per_spk = collections.Counter(spk for spk, _ in rows)
    groups = collections.Counter()
    unkeyed = 0
    for _, p in rows:
        key = partition_of(p, root, args.depth)
        if key is None:
            unkeyed += 1
        else:
            groups[key] += 1
    if not groups:
        _refuse("every one of the %d rows sits directly in %s, so there is no path component "
                "left to key a partition on. An empty partition table is not a corpus with "
                "one partition" % (len(rows), "/" + "/".join(root)))

    if len(groups) > args.max_groups:
        # ⚠ The advice is SEARCHED, not hardcoded to depth 1. It used to name depth 1
        # unconditionally, so refusing AT depth 1 printed "--depth 1 produces 60" as the
        # remedy for "--depth 1 produces 60".
        remedy = None
        for d in range(args.depth - 1, 0, -1):
            n = len({partition_of(p, root, d) for _, p in rows} - {None})
            if n <= args.max_groups:
                remedy = (d, n)
                break
        _refuse("--depth %d produces %d groups, which is a per-speaker split rather than a "
                "partition breakdown. %s"
                % (args.depth, len(groups),
                   "--depth %d produces %d." % remedy if remedy else
                   "No shallower depth gets under --max-groups %d either, so this corpus "
                   "has more top-level groups than the cap." % args.max_groups))

    total = len(rows)
    counts = list(per_spk.values())
    pcts = percentiles(counts, (1, 5, 10, 25, 50, 75, 90, 95, 99))

    ranked = per_spk.most_common()
    cum = 0
    half = 0
    for i, (_, n) in enumerate(ranked, 1):
        cum += n
        if cum * 2 >= total:
            half = i
            break

    # ⚠ The grouping loop is the one place a row can still go missing, since it is the only
    # one that can decline to bucket a row. Checked as a refusal, not an `assert`, for the
    # same reason as the row count above.
    if sum(groups.values()) + unkeyed != total:
        _refuse("partition groups hold %d of %d rows"
                % (sum(groups.values()) + unkeyed, total))

    # ⚠ WHAT THIS SPLIT LEAVES OUT, stated rather than left to the reader. The default
    # measures `train`, which is the right population for a question about what the model
    # was exposed to — but it is not the corpus, and quoting its speaker count AS the
    # corpus's is how "5,332 speakers" got written down for a corpus of 5,385.
    elsewhere = 0
    if args.split != "both":
        other = [n for n in FILELISTS if n not in read]
        seen = set(per_spk)
        for name in other:
            path = os.path.join(args.corpus, name)
            if os.path.exists(path):
                elsewhere = len({s for s, _ in read_rows(path)} - seen)

    for spk in wanted_speakers:
        if spk not in per_spk:
            _refuse("--speakers states %d and the %s split holds no such speaker — its "
                    "indices run %d to %d. A speaker can be real and still be absent here, "
                    "so check --split before concluding it does not exist"
                    % (spk, args.split, min(per_spk), max(per_spk)))

    print("CORPUS   %s  (%s)" % (args.corpus, ", ".join(read)))
    print("ROWS     %d over %d speakers — mean %.1f" % (total, len(per_spk), total / len(per_spk)))
    print("ROOT     /%s   partition key = the next %d component%s"
          % ("/".join(root), args.depth, "" if args.depth == 1 else "s"))
    if unkeyed:
        print("         %d rows sit directly in the root and carry no partition key" % unkeyed)
    if elsewhere:
        print("         ⚠ %d further speakers appear ONLY in the filelist this split "
              "excludes — these figures are the %s split, not the corpus"
              % (elsewhere, args.split))

    print("\n=== rows per speaker ===")
    for p in sorted(pcts):
        print("  p%-3d %6d" % (p, pcts[p]))
    print("  max  %6d" % max(counts))
    print("\n  %d speakers (%.1f%%) hold half of all rows"
          % (half, 100 * half / len(per_spk)))

    print("\n=== rows by partition ===")
    for k, n in groups.most_common():
        print("  %-28s %8d  %5.1f%%" % (k, n, 100 * n / total))

    report = {
        "corpus": args.corpus, "filelists": read, "rows": total,
        "speakers": len(per_spk), "root": "/" + "/".join(root), "depth": args.depth,
        "rows_per_speaker": {"p%d" % p: v for p, v in pcts.items()} | {"max": max(counts)},
        "half_the_rows": half,
        "partitions": dict(groups.most_common()),
        "unkeyed_rows": unkeyed,
    }

    if wanted_speakers:
        pos = {spk: i + 1 for i, (spk, _) in enumerate(ranked)}
        print("\n=== named speakers ===")
        named = {}
        for spk in wanted_speakers:
            subs = sorted({partition_of(p, root, args.depth) or "(root)"
                           for s, p in rows if s == spk})
            print("  index %-6d %5d rows   rank %d of %d (top %.2f%%)   %s"
                  % (spk, per_spk[spk], pos[spk], len(per_spk),
                     100 * pos[spk] / len(per_spk), ", ".join(subs)))
            named[str(spk)] = {"rows": per_spk[spk], "rank": pos[spk],
                               "of": len(per_spk), "partitions": subs}
        report["named_speakers"] = named

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()

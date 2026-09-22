"""Stage an HNR-CONTRASTED, PITCH-MATCHED filelist as MATCHED PAIRS of speakers.

THE QUESTION THIS EXISTS TO SETTLE
----------------------------------
`measure_harmonicity.py` found the mechanism behind the robotic hum: the model compresses
the periodicity range toward its middle. Regressing the rendered HNR on the real speaker's
HNR gives a slope of 0.728 where the vocoder alone manages 0.929, so low-HNR voices come
out too periodic and high-HNR voices not periodic enough. A checkpoint ladder through
epoch 5 left that slope flat, so it is not a training-maturity problem.

What that result could NOT separate is pitch from periodicity. In the 36-speaker sample
the two correlated at +0.922, and the full-corpus survey of 1526 speakers put it at
+0.905 — so "the model renders low voices too periodically" and "the model renders
low-pitched voices too periodically" were the same sentence. The partial correlation said
pitch explains nothing once HNR is held fixed (+0.004), but a partial correlation between
two near-collinear predictors is exactly the case where that number is least trustworthy.

⚠⚠ THE OFF-DIAGONAL DRAW THIS REPLACES DOES NOT EXIST IN THIS CORPUS. The first plan was
to sample speakers from the low-F0/high-HNR and high-F0/low-HNR quadrants. The survey
killed it: the off-diagonal count is 7% at 400 speakers and still 7% at all 1526, and
every one of those speakers is off-diagonal by a hair — the most extreme sits 31 Hz below
the median pitch and 0.29 dB above the median HNR. Widening the search found no hidden
tail, because 7% is a property of the corpus rather than of the draw.

⚠ THE MEDIAN SPLIT WAS THE WRONG LENS, AND THAT IS WHY THIS TOOL MATCHES INSTEAD OF BINS.
Regressing HNR on F0 leaves a residual spread of 0.713 dB with tails past 3 dB, and those
tails hold what a quadrant count cannot see: a speaker at 129.7 Hz with an HNR of 0.37 dB
is strongly low-pitched AND strongly rough, but it lands in the low/low cell like any
ordinary voice. Pairing speakers who share a pitch and differ in periodicity holds F0
fixed BY CONSTRUCTION rather than hoping a sample decorrelates it. 134 such pairs survive
every control below.

THE FOUR CONTROLS, ALL WITHIN THE PAIR
---------------------------------------
  1. PITCH. The two speakers sit within --max-f0-gap Hz of each other. This is the whole
     point: whatever the model does differently to them cannot be a pitch effect.
  2. RECORDING CONDITIONS. Same LibriTTS-R partition unless --no-match-partition.
     train-other-500 is the noisier half and holds 58% of the eligible speakers, and HNR
     measured off a recording moves with the room as well as with the voice.
  3. TRAINING VOLUME. Row counts within --row-ratio of each other, so the rougher speaker
     is not also the underfit one.
  4. UTTERANCE LENGTH. Reported per group and refused on, as in the pitch builder.

⚠⚠ THE CONTROLS ARE VERIFIED AFTER THE DRAW, NOT ASSUMED FROM IT. Every pair being within
5 Hz does not make the two GROUPS pitch-matched: if the rougher speaker were reliably the
lower-pitched of its pair, the groups would still differ systematically, by a few Hz in
one direction, and the sign is what a correlation reads. The selection balances that sign
and then `--f0-balance` refuses if the achieved group medians still part.

⚠⚠ NOT A HOLDOUT, for the same reason the pitch filelist is not. These are clips the
checkpoint trained on, chosen deliberately, because the question is what the model does
with ITS OWN speakers' embeddings.

Usage:
    python scripts/tools/build_hnr_error_filelist.py \
        --corpus data/libritts_r_full_vat_v7 \
        --hnr-json /data/model-training/sonora/pitch_error/speaker_hnr_all.json \
        --out /data/model-training/sonora/pitch_error/hnr_clips.txt
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import hnr_pairs                                     # noqa: E402
from lib.corpus_filelist import dataset_of, partition_of, read_corpus   # noqa: E402
from matcha.delivery import VAT_DIM                              # noqa: E402


def candidate_pairs(spk, eligible, args):
    """Every disjoint (rough, clean) pair passing all four within-pair controls.

    ⚠ THE PAIRING ITSELF LIVES IN `lib/hnr_pairs.py`, shared with the ear bench that
    serves these same speakers. A contrast set and its control must be built by one piece
    of code or the control can differ from the contrast in more than the thing under test.
    """
    pairs = hnr_pairs.matched_pairs(
        {s: spk[s] for s in eligible}, lambda s: len(eligible[s]),
        args.max_f0_gap, args.min_hnr_gap, float("inf"), args.row_ratio,
        args.match_partition)
    return [{"rough": r, "clean": c, "hnr_gap": g, "f0": f, "d_f0": d}
            for r, c, g, f, d in pairs]


def select(pairs, n, rng):
    """`n` pairs spread over the F0 range, with the within-pair pitch sign balanced.

    Two things are being bought here. Spreading over F0 keeps pitch READABLE as a second
    axis instead of piling every pair at the corpus median. Balancing the sign of `d_f0`
    keeps the rough and clean GROUPS pitch-matched: each pair is within a few Hz either
    way, but a consistent direction would sum into a real group-level gap.
    """
    if len(pairs) < n:
        raise SystemExit(
            "REFUSING: %d pair(s) pass the controls and --pairs is %d. Widen "
            "--max-f0-gap, lower --min-hnr-gap, or drop --match-partition — but each of "
            "those is a control, so state which one was relaxed alongside the result."
            % (len(pairs), n))
    lo = min(p["f0"] for p in pairs)
    hi = max(p["f0"] for p in pairs)
    targets = [lo + (hi - lo) * k / (n - 1) for k in range(n)] if n > 1 else [(lo + hi) / 2]
    taken, out, running = set(), [], 0.0
    for t in targets:
        # nearest unused pair to the F0 target; ties broken toward cancelling the running
        # signed pitch imbalance rather than by whatever order the list happens to be in
        best = None
        for i, p in enumerate(pairs):
            if i in taken:
                continue
            key = (abs(p["f0"] - t), abs(running + p["d_f0"]))
            if best is None or key < best[0]:
                best = (key, i)
        _, i = best
        taken.add(i)
        running += pairs[i]["d_f0"]
        out.append(pairs[i])
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hnr-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--split", default="train", choices=["train", "val", "both"])
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--clips-per-speaker", type=int, default=8)
    ap.add_argument("--max-f0-gap", type=float, default=5.0,
                    help="max Hz between the two speakers of a pair")
    ap.add_argument("--min-hnr-gap", type=float, default=2.0,
                    help="min dB of periodicity contrast within a pair")
    ap.add_argument("--row-ratio", type=float, default=2.0,
                    help="max ratio between the two speakers' row counts")
    ap.add_argument("--no-match-partition", dest="match_partition", action="store_false",
                    help="allow a pair to straddle train-clean and train-other")
    ap.add_argument("--f0-balance", type=float, default=3.0,
                    help="max Hz between the rough and clean GROUPS' median F0")
    ap.add_argument("--length-tolerance", type=float, default=0.35)
    ap.add_argument("--dataset", default="LibriTTS_R")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.pairs < 6:
        raise SystemExit("REFUSING: --pairs %d. The comparison is between two GROUPS of "
                         "speakers, and a handful of pairs makes the group medians a "
                         "restatement of two or three voices." % args.pairs)
    if args.min_hnr_gap <= 0:
        raise SystemExit("REFUSING: --min-hnr-gap %.2f admits pairs with no periodicity "
                         "contrast, which is the independent variable."
                         % args.min_hnr_gap)

    spk = {int(k): v for k, v in
           json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    rows = read_corpus(args.corpus, args.split, VAT_DIM)

    all_datasets = Counter(dataset_of(r[1]) for r in rows)
    if args.dataset:
        rows = [r for r in rows if dataset_of(r[1]) == args.dataset]
        if not rows:
            raise SystemExit("REFUSING: --dataset %r matched no row. Present: %s"
                             % (args.dataset, ", ".join("%s=%d" % kv for kv in
                                                        sorted(all_datasets.items()))))
        print("dataset: %s only (corpus holds %s)"
              % (args.dataset, ", ".join("%s=%d" % kv
                                         for kv in sorted(all_datasets.items()))))
    elif len(all_datasets) > 1:
        print("⚠ MIXED DRAW ALLOWED over %d datasets: %s — recording conditions are now a "
              "free variable alongside periodicity."
              % (len(all_datasets), ", ".join("%s=%d" % kv
                                              for kv in sorted(all_datasets.items()))))

    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r[2]].append(r)
    eligible = {s: rs for s, rs in by_spk.items()
                if s in spk and len(rs) >= args.clips_per_speaker}
    if len(eligible) < 2 * args.pairs:
        raise SystemExit("REFUSING: %d speakers carry a measured HNR and at least %d "
                         "rows; %d pairs needs %d."
                         % (len(eligible), args.clips_per_speaker, args.pairs,
                            2 * args.pairs))

    rng = random.Random(args.seed)
    chosen = select(candidate_pairs(spk, eligible, args), args.pairs, rng)

    served, key = [], {}
    for i, p in enumerate(chosen):
        for side in ("rough", "clean"):
            s = p[side]
            for line, wav, _, nphon in rng.sample(by_spk[s], args.clips_per_speaker):
                served.append(line)
                key[Path(wav).name] = {
                    "spk": s, "pair": i, "side": side,
                    "f0": round(spk[s]["f0"], 2), "hnr": round(spk[s]["hnr"], 3),
                    "hnr_gap": round(p["hnr_gap"], 3), "rows": len(by_spk[s]),
                    "phonemes": nphon, "partition": partition_of(wav)}

    groups = {side: [v for v in key.values() if v["side"] == side]
              for side in ("rough", "clean")}
    med_all = statistics.median([v["phonemes"] for v in key.values()])
    print("\n%d pairs, %d speakers, %d clips. Within-pair controls: <=%.1f Hz, >=%.2f dB, "
          "rows <=%.1fx, partition %s"
          % (len(chosen), 2 * len(chosen), len(served), args.max_f0_gap, args.min_hnr_gap,
             args.row_ratio, "matched" if args.match_partition else "FREE"))
    for side in ("rough", "clean"):
        g = groups[side]
        spks = {v["spk"] for v in g}
        print("  %-5s  %2d spk  %3d clips  F0 med %6.1f  HNR med %5.2f  rows med %4d  "
              "phon med %3d  %s"
              % (side, len(spks), len(g),
                 statistics.median([v["f0"] for v in g]),
                 statistics.median([v["hnr"] for v in g]),
                 statistics.median([v["rows"] for v in g]),
                 statistics.median([v["phonemes"] for v in g]),
                 " ".join("%s=%d" % kv
                          for kv in sorted(Counter(v["partition"] for v in g).items()))))

    # ⚠⚠ THE POSTCONDITIONS. Each is the claim the docstring makes, tested on the draw
    # that actually came out rather than on the construction that was meant to produce it.
    d_f0 = (statistics.median([v["f0"] for v in groups["rough"]]) -
            statistics.median([v["f0"] for v in groups["clean"]]))
    if abs(d_f0) > args.f0_balance:
        raise SystemExit(
            "REFUSING: the rough group's median F0 sits %+.1f Hz from the clean group's, "
            "past --f0-balance %.1f. Every pair is pitch-matched and the GROUPS are not, "
            "which means the rougher speaker is systematically the lower-pitched one and "
            "pitch is back in the measurement." % (d_f0, args.f0_balance))
    d_hnr = (statistics.median([v["hnr"] for v in groups["clean"]]) -
             statistics.median([v["hnr"] for v in groups["rough"]]))
    if d_hnr < args.min_hnr_gap:
        raise SystemExit("REFUSING: the two groups' median HNR differ by only %.2f dB. "
                         "The independent variable did not survive the draw." % d_hnr)
    for side in ("rough", "clean"):
        med = statistics.median([v["phonemes"] for v in groups[side]])
        if abs(med - med_all) / med_all > args.length_tolerance:
            raise SystemExit(
                "REFUSING: the %s group's median phoneme count (%d) departs from the "
                "sample's (%d) by more than --length-tolerance %.2f. Utterance length "
                "would ride along with periodicity."
                % (side, med, med_all, args.length_tolerance))
    print("  postconditions: group F0 gap %+.1f Hz (<= %.1f), group HNR gap %.2f dB "
          "(>= %.2f)  ✓" % (d_f0, args.f0_balance, d_hnr, args.min_hnr_gap))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(served) + "\n", encoding="utf-8")
    key_out = Path(args.key_out or (str(out.with_suffix("")) + ".key.json"))
    key_out.write_text(json.dumps(
        {"filelist": str(out), "corpus": args.corpus, "split": args.split,
         "pairs": args.pairs, "max_f0_gap": args.max_f0_gap,
         "min_hnr_gap": args.min_hnr_gap, "row_ratio": args.row_ratio,
         "match_partition": args.match_partition, "dataset": args.dataset,
         "seed": args.seed, "group_f0_gap_hz": round(d_f0, 2),
         "group_hnr_gap_db": round(d_hnr, 3), "is_holdout": False,
         "clips": key}, indent=2), encoding="utf-8")
    print("\nSTAGED %d clips from %d speakers -> %s\n  key -> %s"
          % (len(served), 2 * len(chosen), out, key_out))
    print("\n⚠ NOT A HOLDOUT — these are trained clips, chosen for their embeddings.")


if __name__ == "__main__":
    main()

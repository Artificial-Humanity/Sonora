"""Stage a PITCH-STRATIFIED, CONFOUND-MATCHED filelist for teacher-forced mel scoring.

THE QUESTION, AND WHY IT NEEDS NO EARS
--------------------------------------
Five blind tests put the robotic hum in the acoustic model and nowhere else. The training
audio is clean (24 pairs, both null controls 3/3 ties). The mel/vocoder round trip is
audibly lossy but PITCH-NEUTRAL — detections at 93.8, 114.3, 201.7 and 260.9 Hz, misses
scattered between — while the hum is pitch-locked (12 LOW / 0 HIGH at gaps >= 25 Hz,
p = 0.0005). What is left is the model's own mel PREDICTION.

That is measurable without a listener. `scripts/stages/score_holdout.py` already runs the
training objective teacher-forced: hand it a real clip and it returns how badly the model
predicts that clip's true mel. This tool picks WHICH clips, so that the only thing varying
across the sample is the speaker's pitch.

⚠⚠ THIS IS NOT A HOLDOUT AND MUST NEVER BE READ AS ONE. These are clips the checkpoint
trained on, chosen on purpose: the question is whether the model reproduces ITS OWN
low-pitched speakers badly, which requires their real speaker embeddings. dev-clean's 40
speakers are not in the 5385-entry table at all, so a holdout run would hand every clip an
arbitrary voice and measure that instead. score_holdout.py must therefore be invoked with
--scoring-trained-clips, which refuses to run alongside --assert-disjoint-from and stamps
`is_holdout: false` into its own report.

THE THREE CONFOUNDS, AND WHAT IS DONE ABOUT EACH
------------------------------------------------
A raw "loss vs F0" scatter would be worthless, because low-pitched speakers differ from
high-pitched ones in more than pitch:

  1. TRAINING VOLUME. A speaker with 12 rows is underfit whatever their pitch, and pitch
     coverage is thin at the bottom — exactly where the hypothesis predicts trouble. So
     speakers are matched ACROSS BINS on row count: row counts are split into terciles over
     the eligible population and every F0 bin draws the same number from each tercile. A bin
     that cannot fill its quota refuses rather than silently drawing a lopsided sample.

  2. RECORDING CONDITIONS. train-other-500 is noisier than train-clean-100 and holds 55.8%
     of the corpus. If the low bin were mostly `other` and the high bin mostly `clean`, the
     gradient would be a microphone gradient. --partition restricts to one, and the
     composition of every bin is printed whether or not it is used.

  3. UTTERANCE LENGTH. Longer clips carry more frames and a different loss profile. Phoneme
     count is free (it is field 3 of the filelist) and is reported per bin; a bin whose
     median length departs from the sample's by more than --length-tolerance refuses.

⚠ THE PERMUTATION NULL LIVES DOWNSTREAM, IN THE ANALYSER, AND IS SPEAKER-LEVEL. Clips from
one speaker are not independent observations of that speaker's pitch. Shuffling clip labels
would manufacture significance out of a handful of speakers; the analyser shuffles F0
across SPEAKERS, which is the unit the hypothesis is about.

Usage:
    python scripts/tools/build_pitch_error_filelist.py \
        --corpus data/libritts_r_full_vat_v7 \
        --f0-json /tmp/speaker_f0.json \
        --out /data/model-training/sonora/pitch_error/clips.txt \
        --key-out /data/model-training/sonora/pitch_error/clips.key.json
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
from lib.corpus_filelist import dataset_of, partition_of, read_corpus   # noqa: E402
from matcha.delivery import VAT_DIM                              # noqa: E402


def terciles(values):
    v = sorted(values)
    n = len(v)
    return v[max(0, -(-n * 33 // 100) - 1)], v[max(0, -(-n * 67 // 100) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--f0-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--split", default="train", choices=["train", "val", "both"])
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--speakers-per-bin", type=int, default=12)
    ap.add_argument("--clips-per-speaker", type=int, default=8)
    ap.add_argument("--partition", default=None,
                    help="restrict to one partition, e.g. train-other-500")
    ap.add_argument("--dataset", default="LibriTTS_R",
                    help="restrict to one source dataset; '' to allow a mixed draw")
    ap.add_argument("--length-tolerance", type=float, default=0.35,
                    help="max fractional departure of a bin's median phoneme count")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.bins < 2:
        raise SystemExit("REFUSING: --bins %d is not a gradient." % args.bins)
    if args.speakers_per_bin < 3:
        raise SystemExit("REFUSING: --speakers-per-bin %d cannot carry a within-bin "
                         "spread, so a bin mean would be one or two voices."
                         % args.speakers_per_bin)
    if args.speakers_per_bin % 3:
        raise SystemExit("REFUSING: --speakers-per-bin %d is not divisible by 3, and the "
                         "row-count control draws EQUALLY from three terciles. An uneven "
                         "quota would reintroduce the volume confound it exists to remove."
                         % args.speakers_per_bin)

    f0 = {int(k): v["f0"] for k, v in
          json.loads(Path(args.f0_json).read_text())["speakers"].items()}
    rows = read_corpus(args.corpus, args.split, VAT_DIM)
    # ⚠⚠ ONE SOURCE DATASET BY DEFAULT. A corpus that merges LibriTTS-R with Emilia and the
    # expressive-registers bank mixes three recording chains, and "loss rises as pitch
    # falls" would be unreadable if the low bin happened to draw the noisier source. The
    # first run of this tool came out 288/288 LibriTTS-R by luck of the draw, not by
    # construction, and nothing would have said so.
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
              "free variable alongside pitch."
              % (len(all_datasets), ", ".join("%s=%d" % kv
                                              for kv in sorted(all_datasets.items()))))
    if args.partition:
        rows = [r for r in rows if partition_of(r[1]) == args.partition]
        if not rows:
            seen = sorted({partition_of(r[1]) for r in rows} |
                          {partition_of(r[1]) for r in read_corpus(args.corpus, args.split, VAT_DIM)})
            raise SystemExit("REFUSING: --partition %r matched no row. Present: %s"
                             % (args.partition, ", ".join(seen)))

    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r[2]].append(r)
    eligible = {s: rs for s, rs in by_spk.items()
                if s in f0 and len(rs) >= args.clips_per_speaker}
    if len(eligible) < args.bins * args.speakers_per_bin:
        raise SystemExit(
            "REFUSING: %d speakers have a measured F0 and at least %d rows; %d bins x %d "
            "speakers needs %d. Widen --f0-json, lower --clips-per-speaker, or drop "
            "--partition." % (len(eligible), args.clips_per_speaker, args.bins,
                              args.speakers_per_bin, args.bins * args.speakers_per_bin))

    lo = min(f0[s] for s in eligible)
    hi = max(f0[s] for s in eligible)
    edges = [lo + (hi - lo) * k / args.bins for k in range(args.bins + 1)]
    edges[-1] += 1e-6
    t1, t2 = terciles([len(rs) for rs in eligible.values()])

    def tercile_of(n):
        return 0 if n <= t1 else (1 if n <= t2 else 2)

    binned = defaultdict(lambda: defaultdict(list))
    for s, rs in eligible.items():
        for b in range(args.bins):
            if edges[b] <= f0[s] < edges[b + 1]:
                binned[b][tercile_of(len(rs))].append(s)
                break

    rng = random.Random(args.seed)
    per_tercile = args.speakers_per_bin // 3
    chosen = {}
    for b in range(args.bins):
        for t in range(3):
            pool = sorted(binned[b].get(t, []))
            if len(pool) < per_tercile:
                raise SystemExit(
                    "REFUSING: F0 bin %d (%.1f-%.1f Hz) has %d speaker(s) in row-count "
                    "tercile %d and the quota is %d.\n"
                    "  The terciles are the VOLUME CONTROL: without an equal draw from "
                    "each, a pitch gradient and a training-data gradient are the same "
                    "number. Lower --speakers-per-bin, or --bins."
                    % (b, edges[b], edges[b + 1], len(pool), t, per_tercile))
            for s in rng.sample(pool, per_tercile):
                chosen[s] = b

    served, key = [], {}
    for s, b in sorted(chosen.items()):
        for line, wav, _, nphon in rng.sample(by_spk[s], args.clips_per_speaker):
            served.append(line)
            key[Path(wav).name] = {"spk": s, "f0": round(f0[s], 2), "bin": b,
                                   "rows": len(by_spk[s]), "phonemes": nphon,
                                   "partition": partition_of(wav)}

    med_all = statistics.median([v["phonemes"] for v in key.values()])
    print("F0 bins over %d eligible speakers (%.1f-%.1f Hz), row terciles at %d / %d:"
          % (len(eligible), lo, hi, t1, t2))
    for b in range(args.bins):
        ks = [v for v in key.values() if v["bin"] == b]
        spks = sorted({v["spk"] for v in ks})
        med = statistics.median([v["phonemes"] for v in ks])
        parts = Counter(v["partition"] for v in ks)
        print("  bin %d  %5.1f-%5.1f Hz  %2d spk  %3d clips  rows med %4d  "
              "phon med %3d  %s"
              % (b, edges[b], edges[b + 1], len(spks), len(ks),
                 statistics.median([v["rows"] for v in ks]), med,
                 " ".join("%s=%d" % kv for kv in sorted(parts.items()))))
        if abs(med - med_all) / med_all > args.length_tolerance:
            raise SystemExit(
                "REFUSING: bin %d's median phoneme count (%d) departs from the sample's "
                "(%d) by more than --length-tolerance %.2f. Utterance length would ride "
                "along with pitch." % (b, med, med_all, args.length_tolerance))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(served) + "\n", encoding="utf-8")
    key_out = Path(args.key_out or (str(out.with_suffix("")) + ".key.json"))
    key_out.write_text(json.dumps(
        {"filelist": str(out), "corpus": args.corpus, "split": args.split,
         "bins": args.bins, "edges": edges, "row_terciles": [t1, t2],
         "partition": args.partition, "dataset": args.dataset, "seed": args.seed,
         "is_holdout": False, "clips": key}, indent=2), encoding="utf-8")
    print("\nSTAGED %d clips from %d speakers -> %s\n  key -> %s"
          % (len(served), len(chosen), out, key_out))
    print("\n⚠ NOT A HOLDOUT. Score with:\n"
          "    --scoring-trained-clips 'pitch-dependent mel error needs trained embeddings'")


if __name__ == "__main__":
    main()

"""Join teacher-forced losses to speaker pitch and report whether mel error tracks F0.

Reads `score_holdout.py`'s per-clip CSV and the key written by
`build_pitch_error_filelist.py`, and answers one question: does the acoustic model predict
a low-pitched speaker's mel worse than a high-pitched one's?

⚠⚠ THE UNIT OF ANALYSIS IS THE SPEAKER, NOT THE CLIP, AND THIS IS THE WHOLE STATISTICS OF
IT. F0 is a property of a voice. Eight clips from one speaker are eight measurements of
that ONE voice, not eight independent draws on pitch — treating them as independent would
multiply the apparent sample size eightfold and turn a handful of unusual speakers into a
p-value. Every correlation and every permutation below is computed over speaker means.

⚠ THE NULL IS PERMUTED, NOT ASSUMED. A Spearman coefficient over 36 speakers has a tidy
analytic p, but it assumes a null this design does not obviously satisfy. Shuffling the F0
labels ACROSS SPEAKERS and recomputing gives the distribution this sample actually
produces under "pitch is irrelevant", which is the claim being tested.

⚠ dur_loss IS THE NEGATIVE CONTROL and it is reported beside the others for that reason.
Duration prediction runs off the text encoder and the speaker embedding; it has no mel
reconstruction in it. If `dur` tracks F0 as strongly as `diff` and `prior` do, then what
is being measured is "some speakers are harder" — data volume, recording conditions,
reading style — and not a pitch-specific failure of mel prediction. A result that moves
all three equally is a refutation, not a finding.

Usage:
    python scripts/tools/analyse_pitch_mel_error.py \
        --key /data/model-training/sonora/pitch_error/clips.key.json \
        --per-clip /data/model-training/sonora/pitch_error/per_clip.csv
"""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

TERMS = ["prior", "diff", "dur", "total"]


def ranks(xs):
    """Average ranks, so ties do not bias the coefficient."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def permutation_p(f0, loss, rng, n_perm):
    """Two-sided: how often does a shuffled pitch label produce |rho| this large?"""
    obs = spearman(f0, loss)
    shuffled = list(f0)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if abs(spearman(shuffled, loss)) >= abs(obs) - 1e-12:
            hits += 1
    # +1/+1 so a p of exactly zero is never reported from a finite number of shuffles.
    return obs, (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--per-clip", required=True)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    clips = key["clips"]
    if key.get("is_holdout", False):
        raise SystemExit("REFUSING: this key claims to be a holdout. The measurement needs "
                         "the model's own speaker embeddings; see the tool docstring.")

    with open(args.per_clip, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ckpts = sorted({r["ckpt"] for r in rows})
    unknown = {r["clip"] for r in rows} - set(clips)
    if unknown:
        raise SystemExit("REFUSING: %d scored clip(s) are not in the key, e.g. %s. The "
                         "CSV and the key are from different stagings."
                         % (len(unknown), sorted(unknown)[:3]))
    missing = set(clips) - {r["clip"] for r in rows}
    if missing:
        raise SystemExit("REFUSING: %d staged clip(s) were never scored, e.g. %s. A "
                         "partial score silently reweights the bins."
                         % (len(missing), sorted(missing)[:3]))

    rng = random.Random(args.seed)
    report = {"key": args.key, "per_clip": args.per_clip, "perms": args.perms,
              "checkpoints": {}}

    for ck in ckpts:
        per_spk = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["ckpt"] != ck:
                continue
            spk = clips[r["clip"]]["spk"]
            for t in TERMS:
                per_spk[spk][t].append(float(r[t]))
        spks = sorted(per_spk)
        f0 = [clips_f0(clips, s) for s in spks]
        nrows = [clips_rows(clips, s) for s in spks]
        nphon = [statistics.mean([clips[c]["phonemes"] for c in clips
                                  if clips[c]["spk"] == s]) for s in spks]

        print("\n=== %s ===  %d speakers, %d clips" % (ck, len(spks), len(rows) // len(ckpts)))
        print("\n  per F0 bin (mean over speaker means):")
        print("    %-16s %5s %8s %8s %8s %8s" % ("F0 band", "spk", "prior", "diff", "dur", "total"))
        bins = sorted({clips[c]["bin"] for c in clips})
        edges = key["edges"]
        for b in bins:
            bs = [s for s in spks if any(clips[c]["bin"] == b and clips[c]["spk"] == s
                                         for c in clips)]
            line = "    %5.1f-%5.1f Hz  %5d" % (edges[b], edges[b + 1], len(bs))
            for t in TERMS:
                line += " %8.4f" % statistics.mean(
                    [statistics.mean(per_spk[s][t]) for s in bs])
            print(line)

        print("\n  Spearman over SPEAKER MEANS vs F0  (negative = worse at low pitch):")
        res = {}
        for t in TERMS:
            loss = [statistics.mean(per_spk[s][t]) for s in spks]
            rho, p = permutation_p(f0, loss, rng, args.perms)
            flag = "  <-- NEGATIVE CONTROL" if t == "dur" else ""
            print("    %-6s rho %+0.3f   permutation p = %.4f%s" % (t, rho, p, flag))
            res[t] = {"rho_f0": round(rho, 4), "p_perm": round(p, 5)}

        print("\n  confound checks (these should be near zero — the sample was matched):")
        for label, xs in (("rows", nrows), ("phonemes", nphon)):
            line = "    %-9s" % label
            for t in TERMS:
                loss = [statistics.mean(per_spk[s][t]) for s in spks]
                line += "  %s rho %+0.3f" % (t, spearman(xs, loss))
            print(line)
            for t in TERMS:
                res[t]["rho_%s" % label] = round(
                    spearman(xs, [statistics.mean(per_spk[s][t]) for s in spks]), 4)
        report["checkpoints"][ck] = {"speakers": len(spks), "terms": res}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\n  report -> %s" % args.json_out)


def clips_f0(clips, spk):
    for c in clips.values():
        if c["spk"] == spk:
            return c["f0"]
    raise KeyError(spk)


def clips_rows(clips, spk):
    for c in clips.values():
        if c["spk"] == spk:
            return c["rows"]
    raise KeyError(spk)


if __name__ == "__main__":
    main()

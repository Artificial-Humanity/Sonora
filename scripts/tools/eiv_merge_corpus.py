# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""eiv_merge_corpus — fold a new EIV scoring pass into the corpus-wide derived files.

Written 2026-08-01 to close the v3 coverage gap. Raising MAX_SECONDS 16 -> 22 admitted
1,094 LibriTTS-R utterances that the original labelling pass never saw, so v3 would have
derived them with the tension blend's fourth component and the whole valence channel
absent while the other 30,351 clips had both. All 1,094 sit at >=16 s — the gap is
exactly the band the cap change opened, which is the confirmation that nothing else is
missing.

WHY THIS RECOMPUTES EVERY CLIP, NOT JUST THE NEW ONES
-----------------------------------------------------
The valence combo is `dot(weights, per-speaker z(head))`. A per-speaker z is defined
against that speaker's population, so adding clips to a speaker moves the mean and sd and
therefore changes the combo value of clips that were already scored. Appending only the
new rows would leave the corpus in two inconsistent halves — the exact defect this run
exists to fix, one level up. The RAW head scores are immutable; everything derived from
them is rebuilt from scratch here.

The same property is why this is a corpus version bump. `derive_vat_corpus` then takes
another per-speaker z over these values, so v2 labels are not comparable to v3 labels and
the vat3 checkpoint cannot be fine-tuned onto them.

RECIPE, verified against the stored v1 files before writing this
---------------------------------------------------------------
Population sd (ddof=0), no clamping, speaker = the LibriTTS directory component. That
reproduces `corpus_valence_combo.json` at corr 0.9988; ddof=1 and clamping at 2 or 3
sigma both fit worse, so this is the recipe the original pass used. The residual 0.0012
is not chased: every value is regenerated here under one recipe, so an imperfect match to
a file this run supersedes is not a defect to fix.

`Thankfulness_Gratitude` and `Disappointment` appear in the weight spec but not in the
raw score files. Both carry weight 0.0, so their absence changes nothing — worth stating
because a reader checking the spec against the data will otherwise see two heads missing
and assume a hole.

Usage:
    .venv/bin/python scripts/tools/eiv_merge_corpus.py --add gap_16_22s.jsonl --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

EIV = pathlib.Path("/data/model-training/sonora/eiv_scores")
RAW = ("corpus_v1.jsonl", "corpus_families.jsonl")
SPEC = EIV / "valence_combo_v1.json"
SOFT_HEAD = "Soft_vs._Harsh"

# eiv_score.py writes `{"wav": ..., "wav_mtime": ..., "<head>": score, ...}` — its module
# header says so, and `row = {"wav": w, "wav_mtime": mtimes[i]}` is where it happens. Only
# `wav` was ever stripped below, because `wav_mtime` did not exist when this was written; it
# arrived later as the per-row resume stamp that replaced a whole-file mtime check (#56).
#
# The cost was not a mislabelling. `load_raw` promises `wav -> {head: score}`, so a stray
# bookkeeping key becomes a thirteenth "head" for exactly the files that carry it, and the
# ragged check in main() then REFUSES the merge — measured 303,638 of 333,989 clips reading
# as a different head set on the v7 pass. The guard did its job; the writer and the reader
# had simply disagreed about the format for as long as both existed, and nothing exercised
# the pair until a raw file written by the new writer met the old reader.
#
# Keep this a NAMED SET, not an inline pop: test_eiv_merge_corpus.py reads the writer's row
# literal by AST and fails if it grows a key this tuple does not list. A future stamp is
# then a red test rather than a refused build.
NON_HEAD_KEYS = ("wav_mtime",)


def load_raw(paths) -> dict[str, dict]:
    """wav -> {head: score}, merged across every raw file (heads were split over two)."""
    out: dict[str, dict] = collections.defaultdict(dict)
    for p in paths:
        p = pathlib.Path(p)
        if not p.is_absolute():
            p = EIV / p
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            wav = r.pop("wav")
            for k in NON_HEAD_KEYS:
                r.pop(k, None)
            out[wav].update(r)
            n += 1
        print(f"  {p.name}: {n} rows")
    return dict(out)


def speaker(wav: str) -> str:
    return wav.split("/")[-3]


def combo(raw: dict[str, dict], weights: dict[str, float]) -> dict[str, float]:
    heads = [h for h, w in weights.items() if w]
    have = [h for h in heads if all(h in v for v in raw.values())]
    if set(have) != set(heads):
        raise SystemExit(f"raw scores missing weighted heads: {sorted(set(heads) - set(have))}")
    by = collections.defaultdict(list)
    for w in raw:
        by[speaker(w)].append(w)
    out: dict[str, float] = collections.defaultdict(float)
    for wavs in by.values():
        for h in have:
            v = np.array([raw[w][h] for w in wavs])
            # D-L2. `v.std() or 1.0` only guards a std of EXACTLY zero, and that is not
            # the case that occurs. A head can be CONSTANT for a speaker — the EIV scorer
            # returns one identical value for all 106 of speaker 6531's clips on
            # `Amusement`, and likewise 8095/Valence and 909/Valence — and then `v - mean`
            # and `std` are both floating-point dust around 1e-21. Dividing dust by dust
            # gives max|z| = 1.0: a full-scale, weighted contribution to that speaker's
            # valence, built entirely out of rounding error and indistinguishable from a
            # measurement. Dividing by a 1e-6 floor gives ~1e-15, i.e. zero, which is what
            # a constant head actually tells you about any individual clip.
            #
            # This is the guard derive_vat_corpus already used; the two implementations
            # disagreeing was the finding, and this was the wrong half.
            sd = v.std() + 1e-6
            for w, z in zip(wavs, (v - v.mean()) / sd):
                out[w] += weights[h] * float(z)
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="new raw jsonl to fold in")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    ap.add_argument("--suffix", default="_v2", help="written as corpus_soft{suffix}.json etc")
    args = ap.parse_args()

    print("raw scores:")
    old = load_raw(RAW)
    merged = load_raw(list(RAW) + args.add) if args.add else old
    new_wavs = set(merged) - set(old)
    print(f"{len(old)} -> {len(merged)} clips ({len(new_wavs)} new)")

    ragged = {w for w, v in merged.items() if len(v) != len(next(iter(merged.values())))}
    if ragged:
        print(f"!! {len(ragged)} clips carry a different head set — refusing")
        for w in list(ragged)[:3]:
            print("   ", w, sorted(merged[w]))
        return 1

    weights = dict(zip(*(json.loads(SPEC.read_text())[k] for k in ("heads", "weights"))))
    val = combo(merged, weights)
    soft = {w: v[SOFT_HEAD] for w, v in merged.items()}

    prev = json.loads((EIV / "corpus_valence_combo.json").read_text())
    shared = [w for w in val if w in prev]
    a = np.array([val[w] for w in shared])
    b = np.array([prev[w] for w in shared])
    print(f"\nvalence combo recomputed over {len(val)} clips")
    print(f"  vs the {len(prev)}-clip version, on {len(shared)} shared clips: "
          f"corr={np.corrcoef(a, b)[0, 1]:.6f}  mean shift={np.abs(a - b).mean():+.4f}")
    print("  (existing clips MOVE — per-speaker z is defined against a population that "
          "just grew; that is the point, not a bug)")
    if new_wavs:
        na = np.array([val[w] for w in new_wavs])
        print(f"  new clips: valence mean {na.mean():+.3f} vs corpus {a.mean():+.3f}, "
              f"sd {na.std():.3f}")
        ns = np.array([soft[w] for w in new_wavs])
        os_ = np.array([soft[w] for w in shared])
        print(f"  new clips: soft mean {ns.mean():+.3f} vs corpus {os_.mean():+.3f}")

    if not args.apply:
        print("\nDRY RUN — pass --apply to write")
        return 0
    for name, obj in ((f"corpus_soft{args.suffix}.json", soft),
                      (f"corpus_valence_combo{args.suffix}.json", val)):
        (EIV / name).write_text(json.dumps(obj), encoding="utf-8")
        print(f"wrote {EIV / name} ({len(obj)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

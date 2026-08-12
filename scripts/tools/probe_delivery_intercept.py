#!/usr/bin/env python3
"""Render the A-dial delivery probe for one checkpoint, and measure its true LUFS.

WHY THIS EXISTS (2026-08-12, issue #33)
---------------------------------------
The ep010 probe under `/data/model-training/sonora/probes/delivery_ep010/` is 90 wav files
and a `measures.csv`, and **the script that produced them was never committed** — the same
failure `derive_a_channel_stats.py` was written about, one layer down. The renders survived;
the design did not. Its two texts, its speaker id and its sampler settings are not recorded
anywhere in the repo, which means the probe could be quoted but not repeated, and a second
checkpoint could not be compared against it on equal terms.

So this file is the design, written down. Everything that could differ between two runs is a
constant here or a flag with a stated default, and the header of `measures.csv` records what
was used.

WHAT IT IS FOR. `derive_a_channel_stats.py` § 9 shows the rendered between-lane loudness at
A = 0 is ANTI-correlated with the corpus's own (Spearman -0.80 on ep010). Two explanations
predict that table:

  (a) STRUCTURAL — the three reference frames disagree about what A = 0 denotes. Per-campaign
      centring removed 94.3% of the between-lane structure from A (§ 1), and A is a globally
      shared FiLM channel that cannot carry a per-lane intercept.
  (b) DEVELOPMENTAL — the model has not learned lane loudness yet at this checkpoint.

They separate on the EPOCH TREND, which is why this takes a checkpoint argument: if |rho|
stays flat from an early checkpoint to a late one the cause is structural; if it shrinks with
training the cause is developmental and a v6 re-cut buys nothing. Neither is established by
one checkpoint, which is the limit § 9 states about itself.

⚠ THE RENDERS ARE NOT LOUDNESS-NORMALISED, and must not be. The measurement IS the loudness.

⚠ Reproducibility is bounded by the sampler. `temperature` 0.667 with 3 repeats per cell is
the design's own noise term — that is what the (text, lane, A) cells of 3 exist to estimate —
so two runs of this script agree on cell MEANS, not on individual files.

TWO PASSES, AND THE SPLIT IS NOT ARBITRARY. Rendering needs torch and the ROCm stack, which
live in the `sonora_vocalizer` container; measuring needs `pyloudnorm`, which lives in the
repo venv and is deliberately NOT in that container (adding it would drift the image against
its Dockerfile, which this lab has already paid for once). So:

    docker exec sonora_vocalizer python /tmp/probe.py --render --checkpoint X --out D
    .venv/bin/python scripts/tools/probe_delivery_intercept.py --measure --out D

The renders are the expensive artifact and the measurement is cheap and repeatable, so this
also means a measurement can be redone — on a different loudness definition, say — without
re-rendering anything.

Run:  see above; `--render` writes wavs + design.json, `--measure` writes measures.csv
Exit: 0 on success; 2 if an input is missing; 1 on a refusal.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "pyloudnorm"]
# ///
import argparse
import csv
import json
import os
import sys

# The design. 2 texts x 5 lanes x A in {-1, 0, +1} x 3 repeats = 90 renders, matching the
# ep010 probe's shape so the two are comparable cell-for-cell.
#
# ⚠ THE TEXTS ARE NEW, and that is deliberate rather than careless: the originals are not
# recorded anywhere, so there was no way to match them. Consequence, stated once so nobody
# reconstructs it later — a run of this script is comparable to ANOTHER run of this script,
# and NOT to `delivery_ep010/measures.csv`. To compare checkpoints, render them all here.
TEXTS = {
    # Ordinary declarative prose, no punctuation that drives an engine's expressive channel
    # (see the Dia characterisation: punctuation is a real expressive channel and would
    # confound a loudness measurement).
    "t1": "The committee met on Thursday and agreed to postpone the vote until spring.",
    "t2": "She walked the length of the platform twice before the train finally arrived.",
}
LEVELS = (-1.0, 0.0, 1.0)
REPEATS = 3

# ⚠ THE LANE LIST IS READ FROM THE CONTRACT, NEVER SPELLED OUT HERE. The first version of
# this file restated the five-lane tuple and `test_the_vocabulary_has_exactly_one_definition`
# refused it within the minute — correctly: position IS the wire format of the one-hot block,
# so a second copy that gained a lane would silently reinterpret every filelist ever written.
# Imported inside `do_render` rather than at module scope because `--repo` decides which
# checkout supplies it, and that is not known until the arguments are parsed.
#
# ACTIVE, not all five: `Documentary` is retired and `check_assignable` refuses it on a new
# row, so probing it would measure a lane nothing may be labelled with.
#
# `unknown` is REQUIRED and is not a lane — it is the absence of one, and § 8 measures every
# named lane against it. The empty string is the wire format; the filename and csv say
# "unknown".
def probe_lanes(delivery):
    return ("",) + tuple(delivery.ACTIVE_DELIVERY_LANES)

# Sampler settings, pinned to the Vocalizer's defaults so a probe and an ear check are the
# same act. V and T are held at 0 because this probe moves A and only A.
N_TIMESTEPS, TEMPERATURE, LENGTH_SCALE, GUIDANCE = 10, 0.667, 1.0, 1.0
VALENCE, TENSION = 0.0, 0.0
SPK_ID = 0


def do_render(args):
    import soundfile as sf

    # ⚠ EXPLICIT, NOT INFERRED FROM `__file__`. The render pass runs inside the
    # `sonora_vocalizer` container, where this file is copied in rather than checked out —
    # so `dirname(dirname(__file__))` resolved to `/` and the import failed with a
    # `ModuleNotFoundError` naming `vocalizer`, which reads like a broken environment rather
    # than a wrong path. `--repo` defaults to the checkout when the script IS in one.
    repo = args.repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isfile(os.path.join(repo, "vocalizer.py")):
        print(f"no vocalizer.py under --repo {repo}. Pass --repo pointing at the Sonora "
              f"checkout this render should use (inside the container that is /workspace).",
              file=sys.stderr)
        return 2
    sys.path.insert(0, repo)
    import vocalizer                                                    # noqa: E402
    from matcha import delivery                                         # noqa: E402

    lanes = probe_lanes(delivery)

    os.makedirs(args.out, exist_ok=True)
    design, n = [], 0
    total = len(TEXTS) * len(lanes) * len(LEVELS) * args.repeats
    print(f"probing {os.path.basename(args.checkpoint)}: {total} renders -> {args.out}")

    for tname, text in TEXTS.items():
        for lane in lanes:
            label = lane or "unknown"
            for a in LEVELS:
                for rep in range(args.repeats):
                    _out, wav = vocalizer.render(
                        args.checkpoint, text, N_TIMESTEPS, TEMPERATURE, LENGTH_SCALE,
                        args.spk, VALENCE, a, TENSION, GUIDANCE, lane)
                    y = wav.cpu().numpy()
                    name = f"{tname}_{label}_E{a:+.0f}_r{rep}.wav"
                    sf.write(os.path.join(args.out, name), y, vocalizer.sample_rate,
                             "PCM_24")
                    design.append({"text": tname, "lane": label, "energy": a, "rep": rep,
                                   "file": name})
                    n += 1
                    print(f"  {n:3d}/{total} {name}")

    # The DESIGN goes on disk beside the audio, so the measure pass reads it rather than
    # re-parsing filenames — and so a probe whose renders survive its script stays legible,
    # which is the failure this file exists to end.
    with open(os.path.join(args.out, "design.json"), "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": args.checkpoint, "spk": args.spk,
                   "n_timesteps": N_TIMESTEPS, "temperature": TEMPERATURE,
                   "length_scale": LENGTH_SCALE, "guidance": GUIDANCE,
                   "valence": VALENCE, "tension": TENSION, "sample_rate":
                   vocalizer.sample_rate, "texts": TEXTS, "lanes": list(lanes),
                   "rows": design}, fh, indent=2)
    print(f"wrote design.json ({n} renders). Now run --measure on this directory.")
    return 0


def do_measure(args):
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf

    dpath = os.path.join(args.out, "design.json")
    if not os.path.exists(dpath):
        print(f"no design.json in {args.out} — run --render first, or point --out at a "
              f"directory this script rendered.", file=sys.stderr)
        return 2
    with open(dpath, encoding="utf-8") as fh:
        design = json.load(fh)

    meter, rows = None, []
    for r in design["rows"]:
        y, sr = sf.read(os.path.join(args.out, r["file"]), dtype="float64")
        if meter is None:
            meter = pyln.Meter(sr)
        rows.append({**{k: r[k] for k in ("text", "lane", "energy", "rep")},
                     "lufs": round(float(meter.integrated_loudness(y)), 3),
                     "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)), 3),
                     "dur_s": round(float(len(y) / sr), 3), "file": r["file"]})

    # The provenance goes in the file, not in a message that scrolls away — this whole
    # script exists because the last probe's design did exactly that.
    csv_path = os.path.join(args.out, "measures.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# checkpoint={design['checkpoint']}\n")
        fh.write(f"# spk={design['spk']} n_timesteps={design['n_timesteps']} "
                 f"temperature={design['temperature']} "
                 f"length_scale={design['length_scale']} guidance={design['guidance']} "
                 f"valence={design['valence']} tension={design['tension']}\n")
        for k, v in design["texts"].items():
            fh.write(f"# text {k}={v}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true", help="needs torch (the container)")
    mode.add_argument("--measure", action="store_true", help="needs pyloudnorm (repo venv)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True, help="directory for the wavs and measures.csv")
    ap.add_argument("--spk", type=int, default=SPK_ID)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--repo", help="Sonora checkout to import vocalizer from")
    args = ap.parse_args()

    if args.render:
        if not args.checkpoint:
            print("--render needs --checkpoint", file=sys.stderr)
            return 1
        if not os.path.exists(args.checkpoint):
            print(f"missing checkpoint: {args.checkpoint}", file=sys.stderr)
            return 2
        return do_render(args)
    return do_measure(args)


if __name__ == "__main__":
    sys.exit(main())

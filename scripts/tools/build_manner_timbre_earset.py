#!/usr/bin/env python3
"""build_manner_timbre_earset — prepare the blind delivery ear test from a rendered probe.

WHY THIS EXISTS (2026-08-25, PR-M2)
-----------------------------------
quality-gap-plan.md names the manner-vs-timbre test as the instrument that should GATE
rung 2 — *direct one real speaker through each of the five lanes and have the ear say
whether the manner changed while the voice did not* — and then says "wiring it in is open
work". This is the wiring. It renders nothing: `probe_delivery_intercept.py` already
produced the clips, and re-rendering them would have changed the checkpoint, the speaker
and the sampler settings all at once.

⚠ WHY THE PROBE'S WAVS CANNOT BE HANDED OVER AS THEY ARE. That probe is a LOUDNESS
measurement and is deliberately not normalised — its own header says so. Across the A = 0
cells of `intercept_ep008` the integrated loudness spans 7.1 dB. Play those to an ear and
the loudest lane is the one that sounds most "delivered", so the test would measure the
thing it is supposed to hold constant. Everything here exists to remove that confound
without introducing a second one.

⚠ LINEAR GAIN ONLY, NEVER `loudnorm`. ffmpeg's single-pass `loudnorm` applies dynamic range
compression, and compression IS a manner cue — it would manufacture the very difference the
ear is being asked to judge. This measures integrated LUFS with pyloudnorm and applies one
scalar per file. Nothing is limited and nothing is compressed; if a target would clip, the
target moves rather than the peaks.

THE DESIGN, and why each part is there
--------------------------------------
* **One repeat per lane** is the judgement set. The lane is the variable; the repeat index
  is sampler noise.
* **A DUPLICATE LANE is included in every text group, and it is the control.** Two clips of
  the SAME lane, different repeats, sit in the group under different blind letters. If the
  ear separates them as confidently as it separates two different lanes, the lane effect is
  not above sampler noise and a "the lanes differ" verdict means nothing. Without this a
  blind grouping task cannot fail, which is the defect that makes most listening tests
  unfalsifiable.
* **Blind letters and a shuffled order**, seeded so the assignment is reproducible and
  recorded in a key file the listener is asked not to open first.
* **Forced grouping, not scoring** — the 5-point scale is saturated on this project and
  cannot separate what it is asked to separate.

Usage:
    .venv/bin/python scripts/tools/build_manner_timbre_earset.py \
        --probe /data/model-training/sonora/probes/intercept_ep008 \
        --out   /data/model-training/sonora/probes/ear_delivery_v6_ep008
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "pyloudnorm"]
# ///
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import shutil
import string

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# The probe holds three A levels; the ear test wants delivery to be the ONLY thing moving,
# so it reads ONE plane and nothing else. A = 0 is the default because it is the cleanest
# statement of "delivery alone".
#
# ⚠ BUT A = 0 CANNOT ANSWER EVERY QUESTION, AND THE FIRST RUN FOUND THE LIMIT. The owner
# heard no manner difference at A = 0 and named what was missing: "there's an energy level to
# Speech that is not present". `Speech` is defined as public address — PROJECTED rather than
# conversational — so projection is its defining quality. If projection is carried by the A
# channel rather than by the delivery one-hot, then at A = 0 the lane has no way to express
# it, and this test returns a null whether or not the lane works. That is a confound in the
# test, not a finding about the model. Hence the flag: render the same comparison at A = +1
# and the two readings separate the explanations.
ENERGY_PLANE = 0.0
# Mid-range of what the probe actually produced. A target near the quietest clip would ask
# for large positive gain on the rest and risk clipping; near the loudest it would attenuate
# everything and lose headroom for no benefit.
DEFAULT_TARGET_LUFS = -26.0
PEAK_CEILING = 0.97  # leave a little room; we move the target rather than limit


def read_measures(probe: pathlib.Path) -> list[dict]:
    path = probe / "measures.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, type=pathlib.Path,
                    help="a probe_delivery_intercept.py output directory")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--target-lufs", type=float, default=DEFAULT_TARGET_LUFS)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--energy", type=float, default=ENERGY_PLANE,
                    help="which A plane to read (default %(default)s). Use +1 to ask whether "
                         "a lane's manner appears once the energy channel is allowed to move.")
    args = ap.parse_args()

    rows = [r for r in read_measures(args.probe) if float(r["energy"]) == args.energy]
    if not rows:
        have = sorted({r["energy"] for r in read_measures(args.probe)})
        raise SystemExit(f"no A={args.energy} rows in {args.probe}/measures.csv (have {have})")

    texts = sorted({r["text"] for r in rows})
    lanes = sorted({r["lane"] for r in rows})
    rng = random.Random(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    meter = pyln.Meter(int(float(rows[0].get("sr", 24000)) or 24000))

    key: list[dict] = []
    gains: list[float] = []
    for text in texts:
        # One repeat per lane, plus a second repeat of ONE lane as the control pair.
        picked: list[tuple[str, dict]] = []
        for lane in lanes:
            cells = [r for r in rows if r["text"] == text and r["lane"] == lane]
            if not cells:
                raise SystemExit(f"lane {lane!r} missing from text {text!r}")
            picked.append((lane, rng.choice(cells)))

        control_lane = rng.choice(lanes)
        pool = [r for r in rows
                if r["text"] == text and r["lane"] == control_lane
                and r["file"] != dict(picked)[control_lane]["file"]]
        if not pool:
            raise SystemExit(
                f"no second repeat available for the control lane {control_lane!r} in "
                f"{text!r} — the probe must carry >1 repeat per cell for this test")
        picked.append((control_lane, rng.choice(pool)))

        rng.shuffle(picked)
        letters = string.ascii_uppercase[:len(picked)]
        for letter, (lane, cell) in zip(letters, picked):
            src = args.probe / cell["file"]
            y, sr = sf.read(str(src))
            measured = meter.integrated_loudness(y)
            gain = 10.0 ** ((args.target_lufs - measured) / 20.0)
            peak = float(np.max(np.abs(y))) * gain
            if peak > PEAK_CEILING:  # move the target, never limit the peaks
                gain *= PEAK_CEILING / peak
            out_name = f"{text}_{letter}.wav"
            sf.write(str(args.out / out_name), y * gain, sr)
            gains.append(20.0 * np.log10(gain))
            key.append({"file": out_name, "text": text, "letter": letter,
                        "lane": lane, "source": cell["file"],
                        "measured_lufs": round(measured, 3),
                        "applied_gain_db": round(20.0 * float(np.log10(gain)), 3),
                        "is_control_pair_member": lane == control_lane})

    # The key is what makes the test scoreable, and it is the one file the listener must
    # not read first. It is written beside the clips on purpose — a key kept somewhere
    # else is a key that is lost by the time the answers come back.
    (args.out / "KEY.json").write_text(
        json.dumps({"probe": str(args.probe), "seed": args.seed, "energy": args.energy,
                    "target_lufs": args.target_lufs, "clips": key}, indent=2),
        encoding="utf-8")

    groups = {t: sorted(k["letter"] for k in key if k["text"] == t) for t in texts}
    controls = {t: sorted({k["lane"] for k in key
                           if k["text"] == t and k["is_control_pair_member"]})
                for t in texts}
    (args.out / "README.md").write_text(f"""# Delivery ear test — manner vs timbre

Blind. Do not open `KEY.json` until the answers are written down.

Every clip is the SAME synthetic speaker saying the SAME sentence, from one checkpoint,
with valence and tension held at zero and **energy (A) held at {args.energy:+g}**. **The only thing that differs between
clips in a group is the delivery lane** — except that one lane appears TWICE in each group,
which is the control.

Loudness has been equalised to {args.target_lufs:.1f} LUFS by a single gain per file, so
"louder" is not available as a cue. No compression or limiting was applied.

## Groups

{chr(10).join(f"- **{t}** — clips {', '.join(groups[t])} ({len(groups[t])} clips)" for t in texts)}

## For each group, in this order

1. Listen to all of them once before writing anything.
2. **Group them.** Which clips share a delivery? Say which letters go together.
3. **Voice check.** Is it the same person throughout, or does the voice itself change?
4. **Then** say whether any grouping was a guess.

## What the answers mean

Each group contains **one duplicate lane** — two clips that differ only by sampler noise.
If those two do not land in the same group, the delivery channel is not separable above
noise at this checkpoint, and a confident "the lanes differ" answer elsewhere in the same
group cannot be trusted. That is the point of the control, and it is why a clean-looking
result here is worth something.

If manner changes while the voice stays put, the delivery channel is doing its job. If the
voice moves with it, delivery is entangled with timbre and that is a finding.
""", encoding="utf-8")

    print(f"wrote {len(key)} clips to {args.out}")
    for t in texts:
        print(f"  {t}: {len(groups[t])} clips, control lane {controls[t][0]!r}")
    print(f"  gain applied: {min(gains):+.2f} to {max(gains):+.2f} dB")
    print(f"  key: {args.out / 'KEY.json'} (blind — do not read first)")


if __name__ == "__main__":
    main()

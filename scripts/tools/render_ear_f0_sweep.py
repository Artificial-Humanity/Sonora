"""Render a BLIND ear test that holds the checkpoint fixed and varies the SPEAKER'S PITCH.

THE HYPOTHESIS THIS TESTS, and whose it is
------------------------------------------
The owner's, offered from the ear on 2026-09-19 before anything here was measured: the
robotic hum is a FAILED EMULATION of a deep voice. The deep readers carry a natural low
vibration, the model has barely been shown one, and what it produces reaching for that
register is a buzz that resembles it.

Two legs of that already stand. The hum is NOT inherited — a blind test over source
recordings came back all ties on both same-speaker controls, with the "robotic" speaker's
own clips called pleasant and clear. And the coverage is thin: `measure_speaker_f0.py`
puts 4899 at ~93 Hz, below 99% of sampled speakers, with under 100 Hz holding ~2% of rows.

What is missing is the DOSE RESPONSE. One speaker sounding robotic is an anecdote. If the
hum is a pitch-coverage failure, then severity should rise as F0 falls, across speakers
nobody has listened to yet — and if it does not, the hypothesis is wrong and the ear said
so before a single GPU-hour went into acting on it.

⚠ THE F0 GAP IS THE INDEPENDENT VARIABLE, AND NEAR-ZERO GAPS ARE THE CONTROL. Pairs are
drawn across the whole range of gaps, including pairs of speakers at nearly the same
pitch. Those are the null: two voices the hypothesis says should sound equally clean. A
listener who splits them as hard as they split a 150 Hz gap is hearing voice identity, not
pitch, and the result means nothing. This is the same instrument check that validated the
source-audio test, built in rather than bolted on.

⚠ ONE SET, DELIBERATELY. Binning the pairs into "small gap" and "large gap" tabs would
tell the listener which is which, and the whole question is whether they can hear the
difference without being told. Every pair sits in one undifferentiated set and the gaps
live only in the key.

⚠ VOICE IDENTITY IS THE STANDING CONFOUND AND IT IS MANAGED, NOT ELIMINATED. Two speakers
differ in more than pitch. The defence is breadth — many distinct speakers rather than one
pair judged repeatedly — so that identity varies freely while pitch varies systematically.
A result carried by two or three speakers is not a dose response, and the unblinder should
report per-speaker as well as per-gap.

Usage:
    python scripts/tools/render_ear_f0_sweep.py \
        --ckpt /path/to/vat7_best.ckpt \
        --f0-json /tmp/speaker_f0.json \
        --out /data/model-training/sonora/eartest/f0_sweep
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import ear_bench                                     # noqa: E402
from matcha import delivery                                   # noqa: E402

TEXTS = ear_bench.NEUTRAL_TEXTS

SETS = {
    "sweep": {
        "title": "Which one has the machine in it?",
        "ask": ("Same model, same sentence, two different voices. Ignore which voice you "
                "would rather listen to and ignore the reading. Listen for the ROBOTIC "
                "quality you have described before — the buzz or hum under the voice, the "
                "sense of something mechanical trying to sound human. Which side has more "
                "of it? Many of these pairs genuinely have no difference, and saying so is "
                "the right answer when it is true."),
        "labels": {"A": "A has more of the hum", "same": "No difference",
                   "B": "B has more of the hum"},
    },
}


def choose_pairs(f0, rng, n_pairs, min_rows):
    """Speaker pairs spanning the whole range of F0 gaps, including near-zero.

    Sampled by GAP rather than by speaker so the independent variable is spread evenly:
    drawing speakers at random would pile most pairs near the median gap and leave the
    extremes — where the hypothesis makes its sharpest prediction — to chance.
    """
    spk = sorted(s for s, d in f0.items() if d["rows"] >= min_rows)
    if len(spk) < 4:
        raise SystemExit("REFUSING: only %d speakers have >= %d rows; a sweep needs a "
                         "population to draw a range of gaps from." % (len(spk), min_rows))
    lo = min(f0[s]["f0"] for s in spk)
    hi = max(f0[s]["f0"] for s in spk)
    span = hi - lo
    if span < 40:
        raise SystemExit("REFUSING: the eligible speakers span only %.1f Hz (%.1f-%.1f), "
                         "which is not a sweep." % (span, lo, hi))

    # Target gaps evenly across 0 .. span, so the null (gap ~ 0) is as well represented as
    # the extreme. Each target takes the closest available pair not already used.
    if n_pairs < 4:
        raise SystemExit("REFUSING: --pairs %d cannot carry both a control band and a "
                         "sweep." % n_pairs)
    targets = [span * k / (n_pairs - 1) for k in range(n_pairs)]
    used, pairs = set(), []
    for t in targets:
        best = None
        for _ in range(400):
            a, b = rng.sample(spk, 2)
            if (a, b) in used or (b, a) in used:
                continue
            gap = abs(f0[a]["f0"] - f0[b]["f0"])
            d = abs(gap - t)
            if best is None or d < best[0]:
                best = (d, a, b, gap)
        if best is None:
            continue
        _, a, b, gap = best
        used.add((a, b))
        # lower-pitched speaker first; the side it is served on is flipped below
        if f0[a]["f0"] > f0[b]["f0"]:
            a, b = b, a
        pairs.append((a, b, gap))
    # ⚠ THE CONTROL BAND IS ASSERTED, NOT HOPED FOR. `best is None` used to `continue`
    # with no count check, so a thin pool could return fewer pairs than asked and — worse —
    # no near-zero-gap pair at all, leaving the null the docstring promises unpopulated
    # while everything downstream still reported a result.
    if len(pairs) < n_pairs:
        raise SystemExit("REFUSING: asked for %d pairs and the pool yielded %d. The gap "
                         "targets could not be met from %d speakers."
                         % (n_pairs, len(pairs), len(spk)))
    narrow = [p for p in pairs if p[2] < 25]
    if len(narrow) < 2:
        raise SystemExit("REFUSING: only %d pair(s) under a 25 Hz gap. Those are the "
                         "control — two voices the hypothesis says should sound alike — "
                         "and a sweep without them cannot distinguish a pitch effect from "
                         "a voice-identity effect." % len(narrow))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--f0-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--salt", default="f0-sweep-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    f0 = {int(k): v for k, v in json.loads(Path(args.f0_json).read_text())["speakers"].items()}
    rng = random.Random(args.seed)
    pairs = choose_pairs(f0, rng, args.pairs, args.min_rows)

    arms = {"fixed": args.ckpt}
    lane_kind, n_spks, vat_dim = ear_bench.require_comparable(arms)
    for a, b, _ in pairs:
        for s in (a, b):
            if s >= n_spks:
                raise SystemExit("REFUSING: speaker %d is outside this checkpoint's table "
                                 "of %d — it would render a different voice." % (s, n_spks))

    # ⚠⚠ SHUFFLED BEFORE SERVING, AND THIS IS THE WHOLE POINT OF THE LINE.
    # The first run of this bench appended one pair per ascending gap target and served
    # them in that order: Spearman(serving position, gap) = +0.996. The headline result —
    # +0.918 against gap size — was therefore numerically identical to +0.918 against ITEM
    # POSITION, so fatigue, learning and expectation were not separable from pitch. The
    # three control pairs were items 1-3 and the widest gaps were items 12-15. The
    # docstring above claimed "the gaps live only in the key"; they lived in the order.
    rng.shuffle(pairs)

    bench = ear_bench.Bench(args.out, args.salt, args.seed, lane_kind)
    served, truth = [], {}
    # ⚠ EXACTLY HALF THE PAIRS PUT THE LOW VOICE ON SIDE A. The tally is by PITCH, so a
    # listener's side preference would otherwise land entirely on the low side.
    flip = [True] * (len(pairs) // 2) + [False] * (len(pairs) - len(pairs) // 2)
    rng.shuffle(flip)

    for i, ((low, high, gap), swap) in enumerate(zip(pairs, flip)):
        text = TEXTS[i % len(TEXTS)]
        pair_key = "pair_%02d" % i
        sides = [("A", high), ("B", low)] if swap else [("A", low), ("B", high)]
        item = {"id": pair_key, "set": "sweep", "text": text,
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, spk in sides:
            # ⚠ THE SIDE KEY IS THE SPEAKER, NOT THE LETTER. `ear_bench.opaque` hashes
            # salt|pair_key|side_key, and the re-point guard in `Bench.write` compares
            # (item["A"], item["B"]). With "A"/"B" as the side key those ids are a constant
            # function of the pair index, so re-staging the same --out with different
            # speakers produced identical ids and the guard could not fire. Keying on the
            # speaker makes a changed assignment change the id, which is what the guard
            # watches for. The sibling benches were already safe: they key on the arm and
            # on the lane respectively.
            item[side_key] = bench.render(
                pair_key, "spk%d" % spk, args.ckpt, text, spk, (0.0, 0.0, 0.0),
                delivery.DELIVERY_UNKNOWN, "spk%d" % spk)
        served.append(item)
        truth[pair_key] = {"low": low, "high": high, "gap_hz": round(gap, 1),
                           "f0_low": f0[low]["f0"], "f0_high": f0[high]["f0"],
                           "low_side": "A" if not swap else "B"}
        print("  %s  %4.0f Hz vs %4.0f Hz   gap %5.1f" % (pair_key, f0[low]["f0"],
                                                          f0[high]["f0"], gap))

    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    bench.write(Path(args.out).name, SETS, served,
                {"arm": args.ckpt, "pairs": len(served), "n_spks": n_spks,
                 "vat_dim": vat_dim, "hypothesis": "hum severity rises as F0 falls"},
                key_out=key_out)
    # The gaps go in the key beside the clip map, never in the manifest.
    k = json.loads(Path(key_out).read_text())
    k["pairs"] = truth
    Path(key_out).write_text(json.dumps(k, indent=2))
    print("\nRENDERED %d pairs, %d clips. Key: %s" % (len(served), bench.written, key_out))


if __name__ == "__main__":
    main()

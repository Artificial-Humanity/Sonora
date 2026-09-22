"""Does the hum track the SPEAKER'S NATURAL PERIODICITY? One checkpoint, two voices.

WHY THIS BENCH EXISTS, AND WHAT THE LAST ONE GOT WRONG
-------------------------------------------------------
`render_ear_hnr_ckpt.py` served a rough speaker and a clean speaker in SEPARATE items, so
the listener was never asked to compare them. It could only test whether one checkpoint
beat another, which it answered: no — detection 6/8 on both halves, direction 6/12, a
coin. The contrast the measurement is actually about was never put in front of an ear.

This bench puts it there. Both sides are ONE checkpoint. A is a speaker with low natural
HNR, B is a speaker at the same pitch with high natural HNR, and the question is which
render carries more of the hum. The measurement says the rough one should — the model
pulls every voice toward one middling periodicity, which strips the breath and
irregularity out of a rough voice and leaves something closer to a pure oscillation.

⚠⚠ THE CONTROL IS PAIRS WITH NO PERIODICITY GAP, NOT CATCH TRIALS. Both sides here are
genuinely different voices, so "can you hear a difference" is trivially yes and a
same-versus-different catch trial would measure nothing. The null that matters is
DIRECTION: two speakers matched on pitch AND on HNR, where nothing predicts which side
should sound hummier. A listener who splits those as hard as they split a 2 dB gap is
reporting voice identity, not periodicity, and the result means nothing. Same instrument
check that validated the F0 sweep.

⚠ THE CONTROL IS BUILT BY THE SAME CODE AS THE CONTRAST (`lib/hnr_pairs.py`). Built
separately the two could drift on the pitch tolerance, the partition rule or the row-count
rule, and the control would then differ from the contrast in more than the thing under
test — which is what a control exists to rule out.

⚠ RECORDING CONDITIONS RIDE WITH HNR AND ARE MATCHED, NOT IGNORED. HNR measured off a
recording moves with the room as well as with the voice, and train-other-500 is the
noisier half holding 58% of eligible speakers. Both members of every pair come from the
same partition, so "rough" cannot quietly mean "recorded worse" — though it can still mean
"recorded in a worse room", which is a question for the ear and not for this tool.

Usage:
    python scripts/tools/render_ear_hnr_speaker.py \
        --ckpt /path/checkpoint_epoch=005_step=0063107.ckpt \
        --hnr-json /data/model-training/sonora/pitch_error/speaker_hnr_all.json \
        --out /data/model-training/sonora/eartest/hnr_speaker
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import ear_bench, hnr_pairs                          # noqa: E402
from matcha import delivery                                   # noqa: E402

SETS = {
    "hum": {
        "title": "Which voice has more of the machine in it?",
        "ask": ("Same model, same sentence, two different speakers. Ignore which voice "
                "you would rather listen to and ignore the reading. Rate EACH clip on "
                "its own for the ROBOTIC quality you have described — the buzz or hum "
                "under the voice, the sense of something mechanical trying to sound "
                "human. 0 means you cannot hear it at all. 5 means it sounds like a "
                "Freak-a-Zoid robot. Giving both clips the same rating is a real "
                "answer."),
        "scale": {"max": 5, "anchors": {
            "0": "cannot hear the hum at all",
            "5": "a Freak-a-Zoid robot"}},
    },
}


def spread(pairs, n, rng):
    """`n` pairs spaced over the F0 range, so pitch stays readable as a second axis."""
    if len(pairs) < n:
        raise SystemExit("REFUSING: %d pair(s) available and %d asked for."
                         % (len(pairs), n))
    lo = min(p[3] for p in pairs)
    hi = max(p[3] for p in pairs)
    targets = [lo + (hi - lo) * k / (n - 1) for k in range(n)] if n > 1 else [(lo + hi) / 2]
    taken, out, running = set(), [], 0.0
    for t in targets:
        best = None
        for i, p in enumerate(pairs):
            if i in taken:
                continue
            k = (abs(p[3] - t), abs(running + p[4]))
            if best is None or k < best[0]:
                best = (k, i)
        taken.add(best[1])
        running += pairs[best[1]][4]
        out.append(pairs[best[1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hnr-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--contrast", type=int, default=12)
    ap.add_argument("--control", type=int, default=8)
    ap.add_argument("--max-f0-gap", type=float, default=5.0)
    ap.add_argument("--min-hnr-gap", type=float, default=2.0)
    ap.add_argument("--max-control-hnr-gap", type=float, default=0.3)
    ap.add_argument("--row-ratio", type=float, default=2.0)
    ap.add_argument("--no-match-partition", dest="match_partition",
                    action="store_false")
    ap.add_argument("--salt", default="hnr-speaker-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.control < 4:
        raise SystemExit("REFUSING: --control %d. Without enough zero-gap pairs there is "
                         "no way to tell a periodicity effect from a voice-identity "
                         "effect, and this bench's whole claim rests on that "
                         "distinction." % args.control)
    if args.max_control_hnr_gap >= args.min_hnr_gap:
        raise SystemExit("REFUSING: --max-control-hnr-gap %.2f is not below "
                         "--min-hnr-gap %.2f, so the control and the contrast are the "
                         "same population."
                         % (args.max_control_hnr_gap, args.min_hnr_gap))

    spk = {int(k): v for k, v in
           json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    rows_of = lambda s: spk[s]["rows"]                                   # noqa: E731
    rng = random.Random(args.seed)

    contrast = spread(hnr_pairs.matched_pairs(
        spk, rows_of, args.max_f0_gap, args.min_hnr_gap, float("inf"),
        args.row_ratio, args.match_partition), args.contrast, rng)
    control = spread(hnr_pairs.matched_pairs(
        spk, rows_of, args.max_f0_gap, 0.0, args.max_control_hnr_gap,
        args.row_ratio, args.match_partition), args.control, rng)

    arms = {"fixed": args.ckpt}
    lane_kind, n_spks, vat_dim = ear_bench.require_comparable(arms)
    items = []
    for kind, ps in (("contrast", contrast), ("control", control)):
        for rough, clean, gap, f0, d_f0 in ps:
            for s in (rough, clean):
                if s >= n_spks:
                    raise SystemExit("REFUSING: speaker %d is outside this checkpoint's "
                                     "table of %d — it would render a different voice."
                                     % (s, n_spks))
            items.append({"kind": kind, "rough": rough, "clean": clean,
                          "hnr_gap": gap, "f0": f0, "d_f0": d_f0})

    # ⚠ SHUFFLED BEFORE SERVING, and the sides flipped half the time. The first F0 bench
    # served in ascending order of its own independent variable and scored +0.996 between
    # serving position and that variable, so fatigue and learning were inseparable from
    # the effect. The flip matters just as much: the tally is by ROUGHNESS, so a listener
    # with a side preference would otherwise pile it entirely onto the rough side.
    rng.shuffle(items)
    flip = [True] * (len(items) // 2) + [False] * (len(items) - len(items) // 2)
    rng.shuffle(flip)

    bench = ear_bench.Bench(args.out, args.salt, args.seed, lane_kind)
    served, truth = [], {}
    for i, (it, swap) in enumerate(zip(items, flip)):
        pair_key = "item_%02d" % i
        text = ear_bench.NEUTRAL_TEXTS[i % len(ear_bench.NEUTRAL_TEXTS)]
        sides = ([("A", it["clean"]), ("B", it["rough"])] if swap
                 else [("A", it["rough"]), ("B", it["clean"])])
        item = {"id": pair_key, "set": "hum", "text": text,
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, s in sides:
            # ⚠ THE SIDE KEY IS THE SPEAKER, NOT THE LETTER, so re-staging with different
            # speakers changes the opaque id and `Bench.write`'s re-point guard can fire.
            # Keyed on "A"/"B" those ids are a constant function of the item index and the
            # guard is blind.
            item[side_key] = bench.render(
                pair_key, "spk%d" % s, args.ckpt, text, s, (0.0, 0.0, 0.0),
                delivery.DELIVERY_UNKNOWN, "spk%d" % s)
        served.append(item)
        truth[pair_key] = {
            "kind": it["kind"], "rough": it["rough"], "clean": it["clean"],
            "rough_side": "B" if swap else "A",
            "hnr_rough": spk[it["rough"]]["hnr"], "hnr_clean": spk[it["clean"]]["hnr"],
            "hnr_gap": round(it["hnr_gap"], 3), "f0": round(it["f0"], 1),
            "d_f0": round(it["d_f0"], 1),
            "partition": spk[it["rough"]]["partition"], "text": text}

    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    bench.write(Path(args.out).name, SETS, served,
                {"ckpt": args.ckpt, "contrast": len(contrast), "control": len(control),
                 "n_spks": n_spks, "vat_dim": vat_dim,
                 "prediction": "the LOWER-HNR speaker of a pitch-matched pair carries "
                               "more hum; zero-gap control pairs carry none"},
                key_out=key_out)
    k = json.loads(Path(key_out).read_text())
    k["items"] = truth
    Path(key_out).write_text(json.dumps(k, indent=2))
    print("\nRENDERED %d items (%s), %d clips."
          % (len(served), " ".join("%s=%d" % kv for kv in
                                   sorted(Counter(v["kind"] for v in truth.values())
                                          .items())), bench.written))
    print("  contrast pairs: mean HNR gap %.2f dB, mean |pitch gap| %.1f Hz"
          % (sum(p[2] for p in contrast) / len(contrast),
             sum(abs(p[4]) for p in contrast) / len(contrast)))
    print("  control  pairs: mean HNR gap %.2f dB, mean |pitch gap| %.1f Hz"
          % (sum(p[2] for p in control) / len(control),
             sum(abs(p[4]) for p in control) / len(control)))
    print("Key: %s" % key_out)


if __name__ == "__main__":
    main()

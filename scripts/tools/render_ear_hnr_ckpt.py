"""Render a BLIND test of the prediction that the LATER checkpoint has less of the hum.

THE PREDICTION, AND WHAT WOULD FALSIFY IT
------------------------------------------
The hum is HNR range compression: the model pulls every voice toward one middling amount
of periodicity, which strips the breath and irregularity out of a rough voice and leaves
something closer to a pure oscillation. Measured on a pitch-matched sample of 48 speakers,
the compression slope climbs monotonically across the training run — 0.537, 0.792, 0.825,
0.880, 0.920 against a vocoder floor of 0.927 — and the within-pair effect decays from
+0.799 dB to +0.251 dB, where it stops being significant.

`ep000_s007011` is the checkpoint holdout diff loss SELECTED, and the one whose renders
have been listened to. The final checkpoint of the same run carries much less of the
defect. So the measurement predicts the final checkpoint sounds cleaner.

⚠⚠ "LATER SOUNDS BETTER" WOULD NOT CONFIRM ANYTHING. A later checkpoint is better at many
things at once, and a listener who prefers it everywhere has told us nothing about
periodicity. The prediction that can fail is DIFFERENTIAL: the improvement should be
concentrated on speakers the model renders TOO PERIODICALLY — the rough, low-HNR ones —
and be small or absent on their clean partners, whom the measurement says are already
rendered close to faithfully.

So every rough speaker is served alongside the CLEAN SPEAKER THEY WERE PITCH-MATCHED TO in
`build_hnr_error_filelist.py`. The two differ in periodicity and not in pitch, which means
a result concentrated on the rough half cannot be a pitch preference, and a result spread
evenly across both halves is a general checkpoint preference rather than this mechanism.

⚠ CATCH TRIALS ARE THE SAME CHECKPOINT TWICE, WITH A DIFFERENT NOISE DRAW — not the same
file twice. `ear_bench.seed_for` keys the noise on the PAIR, so two sides of one pair that
share a checkpoint and a text would be bit-identical, and "can you tell these apart" would
be a question about whether the listener is awake. Giving the two sides different pair keys
gives them different draws, so a catch trial asks the real null: can you tell two samples
of ONE model apart? The false-alarm rate that comes out of it is what makes the hit rate
mean something — the source-audio test scored 0/8 false alarms, and that is why its 7/7
was worth believing.

⚠ ONE UNDIFFERENTIATED SET. Splitting rough, clean and catch into labelled tabs would tell
the listener which is which, and whether they can hear it without being told is the whole
question. The classes live only in the key.

Usage:
    python scripts/tools/render_ear_hnr_ckpt.py \
        --arm selected=/path/checkpoint_epoch=000_step=0007011.ckpt \
        --arm final=/path/checkpoint_epoch=005_step=0063107.ckpt \
        --key /data/model-training/sonora/pitch_error/hnr_clips.key.json \
        --out /data/model-training/sonora/eartest/hnr_ckpt
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import ear_bench                                     # noqa: E402
from matcha import delivery                                   # noqa: E402

SETS = {
    "hum": {
        "title": "Which one has more of the machine in it?",
        "ask": ("Same sentence, same voice, two renders. Ignore which you would rather "
                "listen to and ignore the reading. Listen for the ROBOTIC quality you "
                "have described before — the buzz or hum under the voice, the sense of "
                "something mechanical trying to sound human. Which side has more of it? "
                "Many of these pairs genuinely have no difference, and saying so is the "
                "right answer when it is true."),
        "labels": {"A": "A has more of the hum", "same": "No difference",
                   "B": "B has more of the hum"},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT",
                    help="exactly two: the selected checkpoint and the later one")
    ap.add_argument("--key", required=True,
                    help="hnr_clips.key.json — supplies the pitch-matched pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--pairs", type=int, default=8,
                    help="rough speakers to serve; as many clean partners are added")
    ap.add_argument("--catch", type=int, default=8)
    ap.add_argument("--salt", default="hnr-ckpt-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    arms = dict(a.split("=", 1) for a in args.arm)
    if len(arms) != 2:
        raise SystemExit("REFUSING: --arm given %d times with %d distinct names. This "
                         "bench contrasts exactly two checkpoints."
                         % (len(args.arm), len(arms)))
    if args.pairs < 4:
        raise SystemExit("REFUSING: --pairs %d. The result is a DIFFERENCE between the "
                         "rough and clean halves, and four of each is already thin."
                         % args.pairs)
    # ⚠ THE CATCH FLOOR IS THE INSTRUMENT, NOT POLITENESS. Without enough same-checkpoint
    # pairs there is no false-alarm rate, and a hit rate with no false-alarm rate cannot
    # be told apart from a listener who always picks a side.
    total = 2 * args.pairs + args.catch
    if args.catch < 0.25 * total:
        raise SystemExit(
            "REFUSING: %d catch trial(s) out of %d items is %.0f%%, under the 25%% that "
            "measures a false-alarm rate. Raise --catch or lower --pairs."
            % (args.catch, total, 100.0 * args.catch / total))

    clips = json.loads(Path(args.key).read_text())["clips"]
    pairs = {}
    for c in clips.values():
        pairs.setdefault(c["pair"], {})[c["side"]] = c
    usable = [p for p in pairs.values() if {"rough", "clean"} <= set(p)]
    if len(usable) < args.pairs:
        raise SystemExit("REFUSING: %s holds %d complete rough/clean pair(s) and --pairs "
                         "is %d." % (args.key, len(usable), args.pairs))
    # widest periodicity contrast first: those are the speakers the measurement makes its
    # sharpest claim about, so they are where a real effect is most likely to be audible
    usable.sort(key=lambda p: -p["rough"]["hnr_gap"])
    chosen = usable[:args.pairs]

    lane_kind, n_spks, vat_dim = ear_bench.require_comparable(arms)
    for p in chosen:
        for side in ("rough", "clean"):
            if p[side]["spk"] >= n_spks:
                raise SystemExit("REFUSING: speaker %d is outside this checkpoint's table "
                                 "of %d — it would render a different voice."
                                 % (p[side]["spk"], n_spks))

    rng = random.Random(args.seed)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, lane_kind)
    a_name, b_name = sorted(arms)

    items = []
    for i, p in enumerate(chosen):
        for side in ("rough", "clean"):
            items.append({"kind": side, "spk": p[side]["spk"], "f0": p[side]["f0"],
                          "hnr": p[side]["hnr"], "hnr_gap": p[side]["hnr_gap"],
                          "pair_of": i, "arms": (a_name, b_name)})
    pool = [p[s] for p in chosen for s in ("rough", "clean")]
    for j in range(args.catch):
        c = pool[j % len(pool)]
        items.append({"kind": "catch", "spk": c["spk"], "f0": c["f0"], "hnr": c["hnr"],
                      "hnr_gap": c["hnr_gap"], "pair_of": None,
                      "arms": (b_name, b_name)})

    # ⚠ SHUFFLED BEFORE SERVING. The first F0 bench appended items in ascending order of
    # its independent variable and scored +0.996 between serving position and that
    # variable, so fatigue and learning were inseparable from the effect. Built in here
    # rather than remembered.
    rng.shuffle(items)
    flip = [True] * (len(items) // 2) + [False] * (len(items) - len(items) // 2)
    rng.shuffle(flip)

    served, truth = [], {}
    for i, (it, swap) in enumerate(zip(items, flip)):
        pair_key = "item_%02d" % i
        text = ear_bench.NEUTRAL_TEXTS[i % len(ear_bench.NEUTRAL_TEXTS)]
        left, right = it["arms"]
        if swap:
            left, right = right, left
        item = {"id": pair_key, "set": "hum", "text": text,
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, arm in (("A", left), ("B", right)):
            # ⚠ A CATCH TRIAL MUST NOT SHARE ITS NOISE DRAW. Both sides of a normal pair
            # pass the same `pair_key`, so `seed_for` gives them identical noise and the
            # only difference heard is the checkpoint. A catch trial holds the checkpoint
            # fixed, so that same sharing would make the two files bit-identical.
            rk = pair_key if it["kind"] != "catch" else "%s_%s" % (pair_key, side_key)
            item[side_key] = bench.render(
                rk, side_key, arms[arm], text, it["spk"], (0.0, 0.0, 0.0),
                delivery.DELIVERY_UNKNOWN, arm)
        served.append(item)
        truth[pair_key] = {"kind": it["kind"], "spk": it["spk"], "f0": it["f0"],
                           "hnr": it["hnr"], "hnr_gap": round(it["hnr_gap"], 3),
                           "A_arm": left, "B_arm": right, "text": text}

    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    bench.write(Path(args.out).name, SETS, served,
                {"arms": arms, "pairs": args.pairs, "catch": args.catch,
                 "n_spks": n_spks, "vat_dim": vat_dim,
                 "prediction": "the later checkpoint has less hum, and the gain is "
                               "concentrated on the ROUGH (low-HNR) half"},
                key_out=key_out)
    k = json.loads(Path(key_out).read_text())
    k["items"] = truth
    Path(key_out).write_text(json.dumps(k, indent=2))

    kinds = {}
    for v in truth.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    print("\nRENDERED %d items (%s), %d clips."
          % (len(served), " ".join("%s=%d" % kv for kv in sorted(kinds.items())),
             bench.written))
    print("Key: %s" % key_out)
    print("\n⚠ The prediction is DIFFERENTIAL: rough should beat clean. A result spread "
          "evenly\n  across both halves is a checkpoint preference, not this mechanism.")


if __name__ == "__main__":
    main()

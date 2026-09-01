"""Lane control: does REQUESTING a delivery lane damage the audio?

The question the vat7r ear test could not answer, and could not answer because of how I
built it: its Set A and Set B shared no text and Set B had no delivery-blank control, so
"lane-conditioned renders sound worse" (12/12 notes naming a robotic hum, against 4/8 on
the same speaker in Set A) is confounded with the text having changed.

This holds EVERYTHING fixed except the delivery vector. One checkpoint, one speaker, one
text, V/A/T neutral, one seed per pair — the only difference between the two sides is
whether the delivery one-hot is zero or names a lane.

⚠ IT ASKS A DIFFERENT QUESTION FROM THE A/B BENCH, AND THE LABELS SAY SO. Requesting a
lane is SUPPOSED to change the delivery, so "which is better" would conflate "I prefer
this reading" with "this one is damaged". The listener is asked which clip carries more
of the robotic hum, and `choice_means` records that the tally counts a DEFECT rather
than a preference — otherwise a later reader sees "lane 8, blank 2" and reads it as a
win for the lane.

⚠ WHY THIS MATTERS FOR THE CORPUS: 7,101 of the 7,146 lane-labeled rows in v7r use an
expressive_registers speaker, and Newscaster and Speech never co-occur with a LibriTTS
voice at all. Asking a LibriTTS voice for a lane is asking for a combination the corpus
does not contain. If the answer is "the lane damages the audio", the fix is lane labels
on ordinary voices, not a bigger multiplier on 799 synthetic ones.

Usage:
    python scripts/tools/render_ear_lane_control.py \
        --arm donor=/path/to/v7_best.ckpt \
        --arm v7r_late=/path/to/v7r_s0010517.ckpt \
        --serve donor \
        --out /data/model-training/sonora/eartest/vat7r_lane_control

⚠ A FOLLOW-UP ROUND GOES TO A NEW --out, NOT THE SAME ONE. Serving the second arm into a
directory whose first arm has been judged would drop those pairs from items.json, and the
unblinder resolves a verdict to its clips through that file alone — the result becomes
unreadable. `ear_bench.Bench.write` refuses this now; before 2026-09-01 it did not, and
this docstring recommended it. Clip ids are salt-derived, so copy the already-rendered
wavs into the new directory and nothing re-renders:

    cp <old>/clips/<the other arm's clips>.wav <new>/clips/

The container recipe for running either bench is in `scripts/lib/ear_bench.py`.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import ear_bench                                     # noqa: E402
from matcha import delivery                                   # noqa: E402

SPK = 4896          # LibriTTS 5181, 678 clips — the voice the owner called "good"
VAT = (0.0, 0.0, 0.0)
BLANK = "blank"

# The same three lines Set B used, so the two tests are readable against each other.
TEXT = [
    "The results arrived this morning, and they are not what anyone expected.",
    "We will begin at the top of the hour, once everyone has taken their seats.",
    "There is a great deal more to say about this, but not tonight.",
]

ASK = ("Both clips are the SAME model, the same voice and the same words. One had a "
       "delivery lane requested and one did not, so the READING is supposed to differ "
       "— ignore that. Judge only the robotic or electronic hum: which clip has MORE "
       "of it? In the note, say whether you could hear any delivery difference at all.")
LABELS = {"A": "A has more hum", "same": "Same amount", "B": "B has more hum"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--serve", required=True,
                    help="comma-separated arm names to put IN the test. Others are "
                         "rendered and left on disk, so a follow-up round costs no GPU "
                         "— but it needs a NEW --out; see the module docstring.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--salt", default="vat7r-lane-control-v1")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_file():
            raise SystemExit(f"--arm wants NAME=CKPT with a real checkpoint; got {spec!r}")
        arms[name] = path
    serve = args.serve.split(",")
    if any(s not in arms for s in serve):
        raise SystemExit(f"--serve names an undeclared arm: {serve} vs {list(arms)}")

    lane_kind, n_spks, vat_dim = ear_bench.require_comparable(arms)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, lane_kind)

    # ⚠ BALANCED, NOT INDEPENDENTLY RANDOM. An unbiased coin gave 10 blank-first out of
    # 12 on the first draw (p = 0.039), and on THIS test that is worse than on an A/B:
    # requesting a lane is meant to change the delivery audibly, so a listener who
    # notices "the plainer read is usually A" can start answering from expectation
    # instead of from the hum. A shuffled balanced list removes the pattern to learn.
    rng = random.Random(args.seed)
    sets, served = {}, []
    n_pairs = len(TEXT) * len(delivery.ACTIVE_DELIVERY_LANES)
    flips = {}
    for arm in arms:
        f = [True] * (n_pairs // 2) + [False] * (n_pairs - n_pairs // 2)
        rng.shuffle(f)
        flips[arm] = f
    for arm, ckpt in arms.items():
        set_name = f"C_{arm}"
        for ti, text in enumerate(TEXT):
            for lane in delivery.ACTIVE_DELIVERY_LANES:
                # ⚠ ONE PAIR KEY PER (arm, text, lane) => ONE SEED SHARED BY BOTH SIDES.
                # The blank side is re-rendered per lane rather than reused across the
                # four, so each pair is its own controlled comparison instead of one
                # blank render being compared against four differently-seeded lanes.
                pk = f"{set_name}_T{ti:02d}_{lane}"
                blank_id = bench.render(pk, BLANK, ckpt, text, SPK, VAT,
                                        delivery.DELIVERY_UNKNOWN, BLANK)
                lane_id = bench.render(pk, lane, ckpt, text, SPK, VAT, lane, "lane")
                if arm not in serve:
                    continue
                a, b = ((blank_id, lane_id) if flips[arm][len(served)]
                        else (lane_id, blank_id))
                served.append({"id": pk, "set": set_name, "text": text, "spk": SPK,
                               "vat": list(VAT), "delivery_ui": f"{lane} vs blank",
                               "A": a, "B": b})
        if arm in serve:
            sets[set_name] = {
                "title": f"Set C — does requesting a lane damage the audio? ({arm})",
                "ask": ASK, "labels": LABELS,
                # ⚠ The tally counts a DEFECT. Without this the report reads as a
                # preference and "lane 8, blank 2" looks like the lane winning.
                "choice_means": "was judged to carry MORE robotic hum",
            }

    if not served:
        raise SystemExit("--serve selected no arm that produced pairs")
    bench.write("vat7r_lane_control", sets, served,
                {"arms": arms, "served": serve, "spk": SPK, "vat": list(VAT),
                 "n_spks": n_spks, "vat_dim": vat_dim}, args.key_out)
    print(f"  served sets: {list(sets)}   (rendered but not served: "
          f"{[a for a in arms if a not in serve]})")


if __name__ == "__main__":
    main()

"""Render a BLIND pairwise ear test between two or more acoustic CHECKPOINTS.

The verdict instrument for a corpus or model change. `score_holdout.py` prices the COST
of a change on dev-clean LibriTTS; nothing else we own measures the PURCHASE, which is
expressive delivery. This renders the same prompts under the same conditions from each
checkpoint, hides which is which, and hands the pairs to the ear-test app.

Sibling bench: `render_ear_lane_control.py` holds the checkpoint fixed and contrasts
CONDITIONING instead. Shared machinery — seeds, refusals, opaque naming — is in
`scripts/lib/ear_bench.py`, in one copy on purpose.

Usage:
    python scripts/tools/render_ear_ab.py \
        --arm donor=/path/to/v7_best.ckpt \
        --arm v7r_late=/path/to/v7r_s0010517.ckpt \
        --pair donor,v7r_late \
        --out /data/model-training/sonora/eartest/vat7r_vs_v7
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import ear_bench                                     # noqa: E402
from matcha import delivery                                   # noqa: E402

SPK_A, SPK_B = 4896, 4899
SPK_LANE = 4896

SET_A_TEXT = [
    "I told you this would happen, and now here we are, standing in the rain.",
    "The bridge opens at six, so we have exactly forty minutes to make the crossing.",
    "She set the letter down, read it again, and finally understood what he meant.",
    "Nothing about this arrangement is temporary, whatever the notice on the door says.",
]
SET_B_TEXT = [
    "The results arrived this morning, and they are not what anyone expected.",
    "We will begin at the top of the hour, once everyone has taken their seats.",
    "There is a great deal more to say about this, but not tonight.",
]
VAT_SETTINGS = [("neutral", (0.0, 0.0, 0.0)), ("bright", (1.0, 1.0, 0.0))]

SETS = {
    "A_domain_vat": {
        "title": "Set A — domain and V/A/T",
        "ask": ("Delivery lane is BLANK for both. This is the part of the corpus change "
                "that was actually large. Listen for naturalness, prosody and life — "
                "does one read sound more like a person and less like a reader?"),
    },
    "B_delivery_lane": {
        "title": "Set B — delivery lane",
        "ask": ("Same text, one delivery lane requested. First ask whether the lane is "
                "audible AT ALL, then which arm does it better. This lane is taught by "
                "very few clips, so 'no difference' is a real and expected answer."),
    },
}


def build_items():
    """The item matrix. Set A and Set B ask DIFFERENT questions — never pool them.

    Set A (delivery BLANK, V/A/T varied) tests the 22% of the rebalanced corpus that was
    really oversampled: 95,877 Emilia rows, every one of them delivery-blank.

    Set B (V/A/T neutral, delivery lane varied) tests the 1.7% that carries lane labels
    at all — 799 distinct clips repeated 9x.
    ⚠ Set B asks for a combination the corpus almost never contains: 7,101 of the 7,146
    lane-labeled rows use an expressive_registers speaker, and Newscaster and Speech
    NEVER co-occur with a LibriTTS voice. A null result does not prove the delivery
    channel is dead. It is also the product question, which is why it is asked.
    """
    items = []
    for ti, text in enumerate(SET_A_TEXT):
        for spk in (SPK_A, SPK_B):
            for vname, vat in VAT_SETTINGS:
                items.append({"id": f"A{ti:02d}_spk{spk}_{vname}", "set": "A_domain_vat",
                              "text": text, "spk": spk, "vat": list(vat),
                              "delivery": delivery.DELIVERY_UNKNOWN,
                              "delivery_ui": "unknown"})
    for ti, text in enumerate(SET_B_TEXT):
        for lane in delivery.ACTIVE_DELIVERY_LANES:
            items.append({"id": f"B{ti:02d}_{lane}", "set": "B_delivery_lane",
                          "text": text, "spk": SPK_LANE, "vat": [0.0, 0.0, 0.0],
                          "delivery": lane, "delivery_ui": lane})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--pair", required=True,
                    help="the two arm names that form the blind A/B, comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0,
                    help="render only the first N items (smoke test; leaves an "
                         "INCOMPLETE test directory — do not serve it)")
    ap.add_argument("--salt", default="vat7r-ear-v1",
                    help="changes every opaque filename; bump to re-blind a repeat test")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm wants NAME=CKPT, got {spec!r}")
        if not Path(path).is_file():
            raise SystemExit(f"arm {name}: no checkpoint at {path}")
        arms[name] = path
    pair = args.pair.split(",")
    if len(pair) != 2 or any(p not in arms for p in pair):
        raise SystemExit(f"--pair must name exactly two declared arms; got {pair}")

    lane_kind, n_spks, vat_dim = ear_bench.require_comparable(arms)
    items = build_items()
    if args.limit:
        print(f"⚠ --limit {args.limit}: SMOKE TEST, the output is not a servable test")
        items = items[:args.limit]

    bench = ear_bench.Bench(args.out, args.salt, args.seed, lane_kind)
    for arm_name, ckpt in arms.items():
        for it in items:
            bench.render(it["id"], arm_name, ckpt, it["text"], it["spk"], it["vat"],
                         it["delivery"], arm_name)

    # ⚠ A/B SIDE IS RANDOMIZED PER ITEM, from a seeded RNG so the test is reproducible.
    # A fixed side lets a listener learn "left is always the new one" after three items
    # and score the rest from memory rather than from the audio.
    rng = random.Random(args.seed)
    served = []
    for it in items:
        a, b = pair if rng.random() < 0.5 else pair[::-1]
        served.append({k: it[k] for k in ("id", "set", "text", "delivery_ui")} | {
            "spk": it["spk"], "vat": it["vat"],
            "A": ear_bench.opaque(it["id"], a, args.salt),
            "B": ear_bench.opaque(it["id"], b, args.salt)})
    bench.write("vat7r_vs_v7", SETS, served,
                {"arms": arms, "pair": pair, "n_spks": n_spks, "vat_dim": vat_dim},
                args.key_out)
    print(f"  blind pair: {pair[0]} vs {pair[1]}")


if __name__ == "__main__":
    main()

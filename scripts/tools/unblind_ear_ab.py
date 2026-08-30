"""Join blind ear-test verdicts to the arm key and report the result.

Run BY HAND after the listening session, never by the app: the whole point of the
sibling `_keys/` directory is that the serving container cannot reach this map.

⚠ SET A AND SET B ARE REPORTED SEPARATELY AND MUST STAY THAT WAY. They interrogate
levers of very different size in the same corpus change — Set A the 22% that was
really oversampled, Set B the 1.7% carrying delivery labels. A pooled number belongs
to neither and reads as a verdict on both.

⚠ "NO DIFFERENCE" IS DATA, NOT A MISSING ANSWER. It is reported as its own count and
excluded from the sign test, which asks a narrower question: GIVEN that a listener
heard a difference, did it go one way more often than a coin would? A test with 20
ties and 4 splits 3-1 has not found an effect, and the tie count is what says so.

Usage:
    python scripts/tools/unblind_ear_ab.py \
        --key  /data/model-training/sonora/eartest/_keys/vat7r_vs_v7.key.json \
        --test /data/model-training/sonora/eartest/vat7r_vs_v7
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path


def sign_test(a, b):
    """Two-sided exact binomial p for `a` successes in `a+b` trials at p=0.5."""
    n = a + b
    if n == 0:
        return None
    k = min(a, b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    manifest = json.loads((Path(args.test) / "items.json").read_text())
    items = {i["id"]: i for i in manifest["items"]}
    set_meta = manifest.get("sets", {})
    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit(f"no verdicts yet at {vpath}")
    with vpath.open(newline="", encoding="utf-8") as f:
        verdicts = list(csv.DictReader(f))

    # ⚠ `label`, not `arm`. A contrast bench pairs two CONDITIONING settings from one
    # checkpoint, so "which arm won" is not the question it asks. The A/B bench writes
    # the arm name as the label, so its report is unchanged. `arm` is the pre-2026-08-30
    # spelling and is accepted so an older key file still reads.
    arm_of = {k: v.get("label", v.get("arm")) for k, v in key["clips"].items()}
    per_set = defaultdict(lambda: {"wins": Counter(), "tie": 0, "notes": [],
                                   "unsure": 0})
    for v in verdicts:
        if not v["choice"]:
            continue
        it = items.get(v["item"])
        if it is None:
            print(f"  ⚠ verdict for unknown item {v['item']} — skipped")
            continue
        s = per_set[v["set"]]
        if v["choice"] == "same":
            s["tie"] += 1
        else:
            s["wins"][arm_of[it[v["choice"]]]] += 1
        if v.get("confidence") == "unsure":
            s["unsure"] += 1
        if v.get("note"):
            s["notes"].append((v["item"], v["choice"], v["note"]))

    # ⚠ THE TWO LABELS ARE DERIVED PER SET, NOT TAKEN FROM key["pair"]. One test can
    # hold sets that contrast different things — an A/B set contrasts checkpoints while
    # a control set contrasts conditioning — and a single global pair would mislabel one
    # of them while still printing plausible numbers.
    labels_in = defaultdict(set)
    for it in items.values():
        labels_in[it["set"]] |= {arm_of[it["A"]], arm_of[it["B"]]}

    print()
    for name in sorted(per_set):
        s = per_set[name]
        lab = sorted(labels_in[name])
        if len(lab) != 2:
            print(f"  ⚠ {name}: expected two labels, found {lab} — skipped")
            continue
        a1, a2 = lab
        w1, w2, tie = s["wins"][a1], s["wins"][a2], s["tie"]
        n = w1 + w2 + tie
        total = sum(1 for i in items.values() if i["set"] == name)
        p = sign_test(w1, w2)
        # ⚠ PRINT WHAT A COUNT MEANS. Not every bench tallies a preference: the lane
        # control and the ODE bench ask which clip carries MORE of a defect, so "10: 8"
        # means 10 steps LOST eight times. Without this line the two kinds of report are
        # visually identical and the reader supplies the wrong polarity for free.
        means = set_meta.get(name, {}).get("choice_means", "was preferred")
        print(f"  == {name}  ({n} of {total} judged) ==")
        print(f"     counts below = how often each {means}")
        print(f"     {a1:14s} {w1:3d}")
        print(f"     {a2:14s} {w2:3d}")
        print(f"     {'no difference':14s} {tie:3d}   ({s['unsure']} marked not sure)")
        if p is None:
            print("     every judgement was a tie — no direction to test")
        else:
            print(f"     sign test on the {w1 + w2} splits: p = {p:.3f}"
                  f"{'  (not distinguishable from a coin)' if p > 0.05 else ''}")
        for item, choice, note in s["notes"]:
            # ⚠ A TIE HAS NO ARM TO NAME. `items[item]["same"]` is a KeyError, and the
            # fixture that was supposed to cover this rigged every tie with an EMPTY
            # note, so the path never ran. A tie carrying a note is the most informative
            # row in the file — it is a listener saying what they could not hear.
            who = "tie" if choice == "same" else arm_of[items[item][choice]]
            print(f"       · {item} [{who}] {note}")
        print()
    print("  ⚠ Do not pool the sets. See the module docstring.\n")


if __name__ == "__main__":
    main()

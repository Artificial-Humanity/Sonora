"""Render a BLIND pairwise ear test between two or more acoustic checkpoints.

The verdict instrument for the vat7r diversity probe. `score_holdout.py` prices the
COST of a corpus change on dev-clean LibriTTS; nothing we own measures the PURCHASE,
which is expressive delivery. This renders the same prompts under the same conditions
from each checkpoint, hides which is which, and hands the pairs to the ear-test web
app for a forced choice.

⚠ THE SEED IS THE WHOLE EXPERIMENT. `synthesise` samples noise at temperature 0.667,
so two renders of one prompt differ even from ONE checkpoint. Every arm of an item is
rendered under the SAME torch seed, so the only thing that varies inside a pair is the
weights. Without this the test measures sampling noise and calls it a model difference.

⚠ FILENAMES ARE OPAQUE ON PURPOSE. The app serves `clips/<sha1>.wav`; the arm lives
only in `key.json`, which the app never reads. A listener who can see "v7r" in a URL
is not running a blind test.

CPU-only. See the container invocation in the experiment config.

Usage:
    python scripts/tools/render_ear_ab.py \
        --arm donor=/path/to/v7_best.ckpt \
        --arm v7r_late=/path/to/v7r_s0010517.ckpt \
        --arm v7r_early=/path/to/v7r_s0003505.ckpt \
        --pair donor,v7r_late \
        --out /data/model-training/sonora/eartest/vat7r_vs_v7
"""

import argparse
import hashlib
import json
import random
import zlib
from pathlib import Path

import soundfile as sf
import torch

from matcha import delivery
from matcha.cli import (detect_lane, load_matcha, load_vocoder_24k,
                        process_text_for_lane, to_waveform)

DEVICE = torch.device("cpu")

# --- Fixed render settings. IDENTICAL FOR EVERY ARM AND EVERY ITEM. ------------------
# Taken from the Vocalizer/derisk defaults so this bench and the interactive one agree
# about what "neutral settings" means. Changing any of these invalidates comparison
# against a previously rendered test, so they live here as constants, not flags.
N_TIMESTEPS = 10
TEMPERATURE = 0.667
LENGTH_SCALE = 1.0
GUIDANCE = 1.0

# --- Speakers -----------------------------------------------------------------------
# Embedding-table INDICES, not LibriTTS ids, and valid only for a checkpoint carrying
# v7's speaker table. Picked by distinct-clip count in the v7/v7r corpus, so the
# embedding is well trained and a difference between arms is about the trunk rather
# than about a barely-seen voice:
#   4896 -> LibriTTS 5181 (678 clips)   4899 -> LibriTTS 5198 (473)
#   3778 -> LibriTTS 3922 (464)
SPK_A, SPK_B = 4896, 4899
SPK_LANE = 4896

# --- Prompt text --------------------------------------------------------------------
# Written for this test rather than sampled from a corpus: every arm must render text
# NO checkpoint has trained on, and original lines carry no licence question. Each is a
# complete utterance, which the corpus rule requires and no score can detect.
SET_A_TEXT = [
    "I told you this would happen, and now here we are, standing in the rain.",
    "The bridge opens at six, so we have exactly forty minutes to make the crossing.",
    "She set the letter down, read it again, and finally understood what he meant.",
    "Nothing about this arrangement is temporary, whatever the notice on the door says.",
]
# ⚠ DELIBERATELY LANE-NEUTRAL CONTENT. If newscaster text only ever renders under the
# Newscaster lane, text and lane are confounded and a heard difference proves nothing.
# The same three lines run under all four lanes, so the lane is the only variable.
SET_B_TEXT = [
    "The results arrived this morning, and they are not what anyone expected.",
    "We will begin at the top of the hour, once everyone has taken their seats.",
    "There is a great deal more to say about this, but not tonight.",
]

# V/A/T are per-speaker z-scores clamped at 2 sigma during derivation, so +1 is already
# near the edge of the trained range. A larger request does not buy more expression, it
# buys a FiLM activation the trunk never saw.
VAT_SETTINGS = [("neutral", (0.0, 0.0, 0.0)), ("bright", (1.0, 1.0, 0.0))]


def build_items():
    """The item matrix. Set A and Set B answer DIFFERENT questions — never pool them.

    Set A (delivery BLANK, V/A/T varied) tests the 22% of the rebalanced corpus that was
    really oversampled: 95,877 Emilia rows, every one of them delivery-blank.

    Set B (V/A/T neutral, delivery lane varied) tests the 1.7% that carries lane labels
    at all — 799 distinct clips repeated 9x.
    ⚠ Set B is asking for a combination the corpus almost never contains: 7,101 of the
    7,146 lane-labeled rows use an expressive_registers speaker, and Newscaster and
    Speech NEVER co-occur with a LibriTTS voice. A null result here does not prove the
    delivery channel is dead. It is also the product question, which is why it is asked.
    """
    items = []
    for ti, text in enumerate(SET_A_TEXT):
        for spk in (SPK_A, SPK_B):
            for vname, vat in VAT_SETTINGS:
                items.append({
                    "id": f"A{ti:02d}_spk{spk}_{vname}", "set": "A_domain_vat",
                    "text": text, "spk": spk, "vat": list(vat),
                    "delivery": delivery.DELIVERY_UNKNOWN, "delivery_ui": "unknown",
                })
    for ti, text in enumerate(SET_B_TEXT):
        for lane in delivery.ACTIVE_DELIVERY_LANES:
            items.append({
                "id": f"B{ti:02d}_{lane}", "set": "B_delivery_lane",
                "text": text, "spk": SPK_LANE, "vat": [0.0, 0.0, 0.0],
                "delivery": lane, "delivery_ui": lane,
            })
    return items


def opaque(item_id, arm, salt):
    return hashlib.sha1(f"{salt}|{item_id}|{arm}".encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--pair", required=True,
                    help="the two arm names that form the blind A/B, comma separated")
    ap.add_argument("--out", required=True)
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

    # ⚠ REFUSE A COMPARISON THAT CANNOT BE ONE. Different speaker-table sizes mean
    # speaker id 4896 is a different voice in each arm, and different vat_dim means the
    # conditioning vector means different things. Both render happily and produce a
    # confident wrong verdict, which is the failure this check exists to prevent.
    shapes = {n: detect_lane(p) for n, p in arms.items()}
    if len({s for s in shapes.values()}) != 1:
        for n, s in shapes.items():
            print(f"  {n}: lane={s[0]} n_spks={s[1]} vat_dim={s[2]}")
        raise SystemExit("REFUSING: arms disagree on lane/n_spks/vat_dim — not comparable.")
    lane_kind, n_spks, vat_dim = next(iter(shapes.values()))
    if lane_kind != "vat" or vat_dim != delivery.VAT_DIM:
        raise SystemExit(
            f"REFUSING: this bench renders contract-v2 checkpoints (lane=vat, "
            f"vat_dim={delivery.VAT_DIM}); got lane={lane_kind} vat_dim={vat_dim}.")

    items = build_items()
    if args.limit:
        print(f"⚠ --limit {args.limit}: SMOKE TEST, the output is not a servable test")
        items = items[:args.limit]
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    vocoder, sr = load_vocoder_24k(DEVICE)
    key, done = {}, 0
    for arm_name, ckpt in arms.items():
        print(f"== arm {arm_name}: {ckpt}")
        model = load_matcha("custom", ckpt, DEVICE)
        for it in items:
            name = opaque(it["id"], arm_name, args.salt)
            wav_path = out / "clips" / f"{name}.wav"
            key[name] = {"arm": arm_name, "item": it["id"]}
            if wav_path.exists():
                continue
            enc = process_text_for_lane(1, it["text"], DEVICE, lane_kind)
            vec = delivery.vat_vector(*it["vat"], it["delivery"])
            # SAME seed for every arm of this item. See the module docstring.
            # ⚠ crc32, NOT hash(): str hashing is salted per PROCESS, so a render that
            # resumed in a second process would seed the remaining arms differently and
            # silently turn sampling noise into the measured difference.
            torch.manual_seed(args.seed + zlib.crc32(it["id"].encode()) % 65536)
            with torch.no_grad():
                o = model.synthesise(
                    enc["x"], enc["x_lengths"], n_timesteps=N_TIMESTEPS,
                    temperature=TEMPERATURE, length_scale=LENGTH_SCALE,
                    spks=torch.tensor([it["spk"]], dtype=torch.long),
                    vat=torch.tensor([vec]), guidance=GUIDANCE)
                wav = to_waveform(o["mel"], vocoder, None)
            sf.write(wav_path, wav.cpu().numpy(), sr, "PCM_24")
            done += 1
            if done % 10 == 0:
                print(f"   {done} clips")

    # ⚠ A/B SIDE IS RANDOMIZED PER ITEM, from a seeded RNG so the test is reproducible.
    # A fixed side would let a listener learn "left is always the new one" after three
    # items and score the rest from memory rather than from the audio.
    rng = random.Random(args.seed)
    served = []
    for it in items:
        a, b = pair if rng.random() < 0.5 else pair[::-1]
        served.append({k: it[k] for k in ("id", "set", "text", "delivery_ui")} | {
            "spk": it["spk"], "vat": it["vat"],
            "A": opaque(it["id"], a, args.salt), "B": opaque(it["id"], b, args.salt)})
    (out / "items.json").write_text(json.dumps(
        {"test": "vat7r_vs_v7", "sample_rate": sr, "items": served}, indent=2))
    # The unblinding map. NOT served: the app reads items.json only.
    (out / "key.json").write_text(json.dumps(
        {"pair": pair, "salt": args.salt, "clips": key}, indent=2))
    (out / "render_meta.json").write_text(json.dumps({
        "arms": arms, "pair": pair, "n_spks": n_spks, "vat_dim": vat_dim,
        "n_timesteps": N_TIMESTEPS, "temperature": TEMPERATURE,
        "length_scale": LENGTH_SCALE, "guidance": GUIDANCE, "seed": args.seed,
        "sample_rate": sr, "items": len(items), "clips_written": done}, indent=2))
    print(f"\n  {len(items)} items x {len(arms)} arms -> {out}")
    print(f"  blind pair: {pair[0]} vs {pair[1]}  ({done} clips rendered this run)")


if __name__ == "__main__":
    main()

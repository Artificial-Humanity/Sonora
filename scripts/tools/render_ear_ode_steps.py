"""Is 10 ODE steps too few? The cheapest thing that could move the quality ceiling.

The vat7r ear test found one complaint surviving every arm: a robotic or electronic hum
the listener named in nearly every note, present in BOTH checkpoints. Before anyone
spends a quarter on a decoder, rule out the sampler.

⚠ 10 IS THE HOUSE DEFAULT EVERYWHERE — `matcha.cli --steps`, the Vocalizer slider, and
both ear benches. So this does not ask whether one test was handicapped. It asks whether
EVERY quality judgement this lab has made was made at a step count nobody chose on
purpose. `vocalizer.py` already carries the suspicion in one line: the OpenAI-shaped
endpoint uses `25 if guidance > 1.0 else 10`.

⚠ WHY THE COMPARISON IS CLEAN. `CFM.forward` draws z ONCE (`torch.randn_like(mu)`) and
`solve_euler` is deterministic after it; `n_timesteps` builds `t_span` AFTER the draw.
Both sides of a pair share a seed, so they share z EXACTLY, and the duration predictor
is untouched — the two clips are the same length, the same words and the same noise,
integrated coarsely or finely. Nothing else differs.

⚠ BOTH SPEAKERS, AND THAT IS THE POINT. 4896 was called "both good" and 4899 "highly
robotic" in all eight of its notes, on 678 and 473 clips respectively. If more steps fix
anything, the room to show it is on 4899. Testing only the good voice would produce a
confident null.

Usage:
    python scripts/tools/render_ear_ode_steps.py \
        --ckpt /path/to/v7_best.ckpt \
        --out /data/model-training/sonora/eartest/ode_steps
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import ear_bench                                     # noqa: E402
from matcha import delivery                                   # noqa: E402

SPEAKERS = [4896, 4899]     # "both good" / "highly robotic" — see the docstring
VAT = (0.0, 0.0, 0.0)
BASELINE = 10               # the house default under test
# 10 vs 64 is the MAXIMUM-CONTRAST arm: if even that is inaudible, step count is ruled
# out and no follow-up is needed. 10 vs 32 exists to price the fix if 10 vs 64 lands.
CONTRASTS = [(10, 32), (10, 64)]

TEXT = [
    "I told you this would happen, and now here we are, standing in the rain.",
    "The bridge opens at six, so we have exactly forty minutes to make the crossing.",
    "She set the letter down, read it again, and finally understood what he meant.",
]

# ⚠ ADDED 2026-08-30, AFTER THE LANE CONTROL CAME BACK. Requesting a delivery lane makes
# the audio measurably worse on a LibriTTS voice: blank 0, lane 10, two ties, p = 0.002,
# and the blank render was never the worse one in twelve pairs. Speech and Dialogue drew
# "drastic" and "very severe"; Neutral and Newscaster were mild. That makes a new question
# worth the same sitting: is the lane damage an INTEGRATION artifact that more sampling
# repairs, or is it the conditioning pushing the trunk somewhere it was never trained?
# Cheap to ask, and the answers differ by a quarter of work.
LANE_SET = ("Speech", "Dialogue")      # the two the owner called drastic / very severe
LANE_CONTRAST = (10, 64)               # maximum contrast; this is a screen, not a curve
LANE_SPK = 4896                        # the voice the lane control measured the damage on
# The lane control's text, so the two tests read against each other line for line.
LANE_TEXT = [
    "The results arrived this morning, and they are not what anyone expected.",
    "We will begin at the top of the hour, once everyone has taken their seats.",
    "There is a great deal more to say about this, but not tonight.",
]
LANE_ASK = ("Both clips requested the SAME delivery lane from the same model, voice and "
            "words, with the same random noise. They differ ONLY in how finely the "
            "sampler integrated. The lane is known to add a hum — the question here is "
            "whether more sampling removes it. Which clip carries MORE hum? 'Same "
            "amount' means more steps do not fix it, which is a real answer.")

ASK = ("Both clips are the SAME model, voice, words and random noise. They differ ONLY "
       "in how finely the sampler integrated — one used more steps than the other. They "
       "are the same length on purpose. Which clip carries MORE of the robotic or "
       "electronic hum? If they are the same, say so: that is the result that rules the "
       "sampler out.")
LABELS = {"A": "A has more hum", "same": "Same amount", "B": "B has more hum"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--seed", type=int, default=8642)
    ap.add_argument("--salt", default="ode-steps-v1")
    args = ap.parse_args()
    if not Path(args.ckpt).is_file():
        raise SystemExit(f"no checkpoint at {args.ckpt}")

    lane_kind, n_spks, vat_dim = ear_bench.require_comparable({"ckpt": args.ckpt})
    bench = ear_bench.Bench(args.out, args.salt, args.seed, lane_kind)

    rng = random.Random(args.seed)
    sets, served = {}, []
    for lo, hi in CONTRASTS:
        set_name = f"D_steps_{lo}v{hi}"
        n = len(SPEAKERS) * len(TEXT)
        # Balanced sides, shuffled — see the note in render_ear_lane_control.py. Here the
        # cue a listener could learn is subtler but the fix costs nothing.
        flips = [True] * (n // 2) + [False] * (n - n // 2)
        rng.shuffle(flips)
        k = 0
        for spk in SPEAKERS:
            for ti, text in enumerate(TEXT):
                pk = f"{set_name}_spk{spk}_T{ti:02d}"
                a_id = bench.render(pk, f"s{lo}", args.ckpt, text, spk, VAT,
                                    delivery.DELIVERY_UNKNOWN, str(lo), n_timesteps=lo)
                b_id = bench.render(pk, f"s{hi}", args.ckpt, text, spk, VAT,
                                    delivery.DELIVERY_UNKNOWN, str(hi), n_timesteps=hi)
                A, B = (a_id, b_id) if flips[k] else (b_id, a_id)
                k += 1
                served.append({"id": pk, "set": set_name, "text": text, "spk": spk,
                               "vat": list(VAT),
                               "delivery_ui": f"{lo} vs {hi} ODE steps",
                               "A": A, "B": B})
        sets[set_name] = {
            "title": f"Set D — {lo} vs {hi} ODE steps (same noise, same length)",
            "ask": ASK, "labels": LABELS,
            "choice_means": "was judged to carry MORE robotic hum",
        }

    # ⚠ APPENDED AFTER the blank-lane sets, so their RNG draws and therefore their A/B
    # assignment are untouched. Verified against the previous manifest.
    lo, hi = LANE_CONTRAST
    set_name = f"D_lane_steps_{lo}v{hi}"
    n = len(LANE_SET) * len(LANE_TEXT)
    flips = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flips)
    k = 0
    for lane in LANE_SET:
        for ti, text in enumerate(LANE_TEXT):
            pk = f"{set_name}_{lane}_T{ti:02d}"
            a_id = bench.render(pk, f"s{lo}", args.ckpt, text, LANE_SPK, VAT, lane,
                                str(lo), n_timesteps=lo)
            b_id = bench.render(pk, f"s{hi}", args.ckpt, text, LANE_SPK, VAT, lane,
                                str(hi), n_timesteps=hi)
            A, B = (a_id, b_id) if flips[k] else (b_id, a_id)
            k += 1
            served.append({"id": pk, "set": set_name, "text": text, "spk": LANE_SPK,
                           "vat": list(VAT),
                           "delivery_ui": f"{lane} · {lo} vs {hi} ODE steps",
                           "A": A, "B": B})
    sets[set_name] = {
        "title": f"Set E — does more sampling repair the LANE damage? ({lo} vs {hi})",
        "ask": LANE_ASK, "labels": LABELS,
        "choice_means": "was judged to carry MORE robotic hum",
    }

    bench.write("ode_steps", sets, served,
                {"ckpt": args.ckpt, "baseline_steps": BASELINE,
                 "contrasts": CONTRASTS, "speakers": SPEAKERS,
                 "n_spks": n_spks, "vat_dim": vat_dim}, args.key_out)
    print(f"  contrasts: {CONTRASTS}   speakers: {SPEAKERS}")


if __name__ == "__main__":
    main()

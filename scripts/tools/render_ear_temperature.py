"""DOES THE ODE TEMPERATURE CARRY THE HUM? One noise draw, three magnitudes.

WHERE THIS SITS (2026-09-24)
----------------------------
The hum is in the mel the acoustic model predicts. Ruled out, each blind: the training audio
(`source_audio`), the mel/vocoder round trip (`ceiling`), the vocoder's response to
predicted mels (`vocoder_ft`, fine-tuned on them: -0.08, p = 0.79), the stale mel buffers
(`mel_stats`, 11 of 12 tied), and the ODE step count (`ode_steps`, 10 v 64 all tied). Every
one of those benches rendered at temperature 0.667, and nothing has ever varied it.

`CFM.forward` draws z ~ N(0, I) once and scales it by the temperature before the solve, so
the temperature is the size of the noise the mel is integrated FROM. If the buzz is a
trajectory that starts too far out and does not come back — or one that starts too close
and collapses onto an average — the temperature moves it. If the buzz is in the field the
decoder learned, the temperature does not, and the lever is the decoder.

    t0667_vs_t0333   the current setting against half the noise
    t0667_vs_t1000   the current setting against the full unit noise the flow was trained on

That is `--design sweep`, judged 2026-09-24: 0333 +0.17 (p 0.62), 1000 -0.33 (p 0.12), and
pooled POST HOC as one direction -0.25, 7 better / 1 worse, p 0.070. A lead, so:

⚠⚠ `--design confirm` IS PRE-REGISTERED (owner asked for it, 2026-09-24). 24 pairs, all
t0667_vs_t1000, on speakers no earlier bench has served. Prediction: severity is LOWER at
1.0. Test: exact sign-flip over the 24 paired differences, two-sided, confirmed at p < 0.05.
Written here before a clip was rendered, so the test cannot be chosen after the result.

⚠⚠ ONE SEED PER PAIR, SO BOTH SIDES START FROM THE SAME NOISE DIRECTION. `Bench.synth`
seeds per pair and z is drawn once, so the only difference inside a pair is the magnitude.
The durations do not depend on the temperature, so both sides are also the same length.

⚠ THE SPEAKERS ARE NEW TO THE LISTENER. The ceiling and mel_stats speakers are excluded, so
no voice from the last three sittings is recognised. The picks spread over natural HNR and
each kind is cycled across that spread: the owner hears "buzz" on rough voices and
"layering" on clean ones, and the answer may differ between the two.

⚠ ONLY MODEL RENDERS ARE SERVED. Real audio has scored 0.00-0.06 in two benches; it prices
nothing here.

Usage:
    python scripts/tools/render_ear_temperature.py \\
        --ckpt /path/checkpoint_epoch=005_step=0063107.ckpt \\
        --corpus data/libritts_r_full_vat_v7 \\
        --hnr-json /data/model-training/sonora/pitch_error/speaker_hnr_all.json \\
        --exclude-key <ceiling speakers>.json --exclude-key <mel_stats speakers>.json \\
        --out /data/model-training/sonora/eartest/temperature
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import ear_bench                                     # noqa: E402
from lib.corpus_filelist import read_corpus                   # noqa: E402
from matcha import delivery                                   # noqa: E402
from matcha.cli import to_waveform                            # noqa: E402
from matcha.delivery import VAT_DIM                           # noqa: E402

BASE = 0.667
DESIGNS = {"sweep": {"t0667_vs_t0333": 12, "t0667_vs_t1000": 12},
           "confirm": {"t0667_vs_t1000": 24}}
LEVELS = {"t0667": BASE, "t0333": 0.333, "t1000": 1.0}

SETS = {
    "hum": {
        "title": "How much machine is in each one?",
        "ask": ("Two renders of the same sentence by the model. Ignore which you would "
                "rather listen to and ignore the reading. Rate EACH clip on its own for the "
                "ROBOTIC quality you have described — the buzz or hum under the voice, the "
                "layering, the sense of something mechanical trying to sound human. 0 means "
                "you cannot hear it at all. 5 means it sounds like a Freak-a-Zoid robot. "
                "Giving both clips the same rating is a real answer. A note saying 'buzz' "
                "or 'layering' helps, and so does one saying a side sounds dull or smeared."),
        "scale": {"max": 5, "anchors": {
            "0": "cannot hear the hum at all",
            "5": "a Freak-a-Zoid robot"}},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hnr-json", required=True)
    ap.add_argument("--design", choices=sorted(DESIGNS), default="sweep")
    # ⚠ REQUIRED: the picks are a deterministic spread over HNR rank, so without the earlier
    # benches' speakers excluded they would largely redraw the ceiling bench's voices.
    ap.add_argument("--exclude-key", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="temperature-v1")
    ap.add_argument("--seed", type=int, default=1618)
    args = ap.parse_args()
    KINDS = DESIGNS[args.design]

    if ear_bench.TEMPERATURE != BASE:
        raise SystemExit("REFUSING: ear_bench renders at %.3f and this bench's baseline is "
                         "%.3f. The baseline must be what every earlier bench heard."
                         % (ear_bench.TEMPERATURE, BASE))

    held_out = set()
    for k in args.exclude_key:
        items = json.loads(Path(k).read_text()).get("items") or {}
        if not items:
            raise SystemExit("REFUSING: %s carries no `items`, so the hold-out would be "
                             "silently empty." % k)
        held_out |= {int(v["spk"]) for v in items.values()}

    spk_hnr = {int(k): v for k, v in
               json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    by_spk = defaultdict(list)
    for line, wav, s, nphon in read_corpus(args.corpus, "train", VAT_DIM):
        if s in spk_hnr and s not in held_out:
            by_spk[s].append((wav, line.split("|")[2]))

    order = sorted(by_spk, key=lambda s: spk_hnr[s]["hnr"])
    need = sum(KINDS.values())
    picks = [order[round(i * (len(order) - 1) / (need - 1))] for i in range(need)]
    if len(set(picks)) != need:
        raise SystemExit("REFUSING: %d speakers cannot spread %d distinct picks."
                         % (len(order), need))
    # ⚠⚠ STRATIFIED OVER HNR, THEN SHUFFLED — a random assignment can leave one kind
    # without the roughest voices, and serving in HNR order is the confound that sank the
    # 2026-09-19 pitch dose response. See render_ear_mel_stats.py.
    cycle = list(KINDS)
    kinds = [cycle[i % len(cycle)] for i in range(need)]
    if Counter(kinds) != Counter(KINDS):
        raise SystemExit("REFUSING: the cycle yields %s, not %s." % (dict(Counter(kinds)),
                                                                    KINDS))

    rng = random.Random(args.seed)
    plan = list(zip(picks, kinds))
    rng.shuffle(plan)
    # Side placement balanced WITHIN each kind, not only overall (review, 2026-09-24).
    flips = {}
    for kd, n in KINDS.items():
        f = [True] * (n // 2) + [False] * (n - n // 2)
        rng.shuffle(f)
        flips[kd] = f
    flip = [flips[kd].pop() for _, kd in plan]

    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    vocoder, sr = bench.vocoder, bench.sample_rate
    model = bench._model_for(args.ckpt)
    hp = model.hparams.get("data_statistics") or {}
    if not hp:
        raise SystemExit("REFUSING: %s records no data_statistics, so there is no way to "
                         "tell whether its mel buffers are stale." % args.ckpt)
    if (abs(float(model.mel_mean) - float(hp["mel_mean"])) > 1e-4
            or abs(float(model.mel_std) - float(hp["mel_std"])) > 1e-4):
        raise SystemExit("REFUSING: the model still denormalises with stale buffers.")

    served, truth, limited = [], {}, 0
    clips = Path(args.out) / "clips"
    for i, ((s, kind), swap) in enumerate(zip(plan, flip)):
        cand = [(w, p) for w, p in by_spk[s]
                if Path(w).is_file() and sf.info(w).duration >= args.min_seconds]
        if not cand:
            raise SystemExit("REFUSING: speaker %d has no clip of at least %.1fs."
                             % (s, args.min_seconds))
        src, phon = rng.choice(cand)
        pair_key = "item_%02d" % i
        made, frames = {}, set()
        for lbl in kind.split("_vs_"):
            o = bench.synth(pair_key, args.ckpt, "(phonemes)", s, (0.0, 0.0, 0.0),
                            delivery.DELIVERY_UNKNOWN, phonemes=phon,
                            temperature=LEVELS[lbl])
            frames.add(int(o["mel"].shape[-1]))
            with torch.no_grad():
                made[lbl] = to_waveform(o["mel"], vocoder, None).numpy()
        if len(frames) != 1:
            raise SystemExit("REFUSING: %s rendered %s frames across temperatures; the "
                             "durations were meant to be identical." % (pair_key, frames))
        left_lbl, right_lbl = kind.split("_vs_")
        sides = [("A", left_lbl), ("B", right_lbl)]
        if swap:
            sides = [("A", right_lbl), ("B", left_lbl)]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            audio, bound = ear_bench.match_loudness(made[lbl], sr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), audio, sr, "PCM_24")
            bench.key[nm] = {"label": lbl, "pair": pair_key, "side": side_key}
            item[side_key] = nm
        served.append(item)
        truth[pair_key] = {"kind": kind, "spk": s, "hnr": spk_hnr[s]["hnr"],
                           "f0": spk_hnr[s]["f0"], "A_label": sides[0][1],
                           "B_label": sides[1][1], "source": src}
        print("  %s  spk%-6d HNR %5.2f  %s" % (pair_key, s, spk_hnr[s]["hnr"], kind))

    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    bench.write(Path(args.out).name, SETS, served,
                {"ckpt": args.ckpt, "design": args.design, "kinds": KINDS, "levels": LEVELS,
                 "lufs": args.lufs, "peak_limited_sides": limited,
                 # Overrides Bench.write's defaults, which describe `render`: this bench
                 # sets the temperature per side and writes its clips itself.
                 "temperature": "per side, see levels", "clips_written": 2 * len(served),
                 "prediction": ("PRE-REGISTERED: severity is lower at 1.0; exact two-sided "
                                "sign-flip over 24 pairs, confirmed at p < 0.05"
                                if args.design == "confirm" else
                                "if either pairing moves severity, the buzz is in the "
                                "sampling and the temperature is a free lever; if both are "
                                "null, the buzz is in the decoder's learned field")},
                key_out=key_out)
    k = json.loads(Path(key_out).read_text())
    k["items"] = truth
    Path(key_out).write_text(json.dumps(k, indent=2))
    if limited:
        print("\n⚠ %d side(s) hit the peak ceiling before reaching %.1f LUFS." % (limited,
                                                                               args.lufs))
    print("\nRENDERED %d items (%s)." % (len(served), " ".join(
        "%s=%d" % kv for kv in sorted(Counter(v["kind"] for v in truth.values()).items()))))
    print("Key: %s" % key_out)


if __name__ == "__main__":
    main()

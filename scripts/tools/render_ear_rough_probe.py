"""DID TRAINING ON ROUGH VOICES TEACH THE DECODER TO RENDER THEM? One render, two models.

`vat7_rough_probe` fine-tuned vat7 ep005 for 6,000 steps on only the voices with natural
HNR under 4.5 dB — the voices the owner hears "buzz" on. The 16 rough and 8 clean speakers
served here were held out of that fine-tune, train and val alike
(build_rough_voice_corpus.py), so this measures GENERALISATION to rough voices, not
memorisation of the ones trained on.

    base_vs_probe   the same phonemes, speaker, conditioning and noise draw, rendered by
                    ep005 and by the probe. Both through the SAME vocoder.

⚠⚠ PRE-REGISTERED (2026-09-25, before any clip was rendered): on the 16 ROUGH speakers,
severity is lower for the probe. Test: exact two-sided sign-flip over those 16 paired
differences, confirmed at p < 0.05.
    * confirmed -> the decoder can render rough voices and they are under-trained; the lever
      is rebalancing the data.
    * not confirmed -> the decoder cannot, on this evidence; the DiT spike moves up.
The 8 CLEAN speakers are the regression check, reported, not tested: a rough-only fine-tune
may cost clean voices, which would rule the probe out as a candidate but not change the
answer to its question.

⚠ THE SEED IS PER PAIR, so both models start from the same noise, and the renders are made
model by model (all base sides, then all probe sides) so each checkpoint loads once.

Usage:
    python scripts/tools/render_ear_rough_probe.py \\
        --base <vat7_finetune ep005_s063107> --probe <vat7_rough_probe step=0005999> \\
        --corpus data/libritts_r_full_vat_v7 \\
        --speakers /data/model-training/sonora/pitch_error/_rough_probe_bench_speakers.json \\
        --out /data/model-training/sonora/eartest/rough_probe
"""

import argparse
import json
import random
import sys
from collections import defaultdict
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

KIND = "base_vs_probe"

SETS = {
    "hum": {
        "title": "How much machine is in each one?",
        "ask": ("Two renders of the same sentence by the model. Ignore which you would "
                "rather listen to and ignore the reading. Rate EACH clip on its own for the "
                "ROBOTIC quality you have described — the buzz or hum under the voice, the "
                "layering, the sense of something mechanical trying to sound human. 0 means "
                "you cannot hear it at all. 5 means it sounds like a Freak-a-Zoid robot. "
                "Giving both clips the same rating is a real answer. A note saying 'buzz' "
                "or 'layering' helps, and so does one saying a side sounds unlike the "
                "speaker."),
        "scale": {"max": 5, "anchors": {
            "0": "cannot hear the hum at all",
            "5": "a Freak-a-Zoid robot"}},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--speakers", required=True,
                    help="JSON {items: {..: {spk, hnr, group: rough|clean}}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="rough-probe-v1")
    ap.add_argument("--seed", type=int, default=3141)
    args = ap.parse_args()

    spk = list(json.loads(Path(args.speakers).read_text())["items"].values())
    groups = defaultdict(int)
    for v in spk:
        groups[v["group"]] += 1
    if dict(groups) != {"rough": 16, "clean": 8}:
        raise SystemExit("REFUSING: the pre-registered design is 16 rough + 8 clean, got %s."
                         % dict(groups))
    by_spk = defaultdict(list)
    for line, wav, s, nphon in read_corpus(args.corpus, "train", VAT_DIM):
        by_spk[s].append((wav, line.split("|")[2]))

    rng = random.Random(args.seed)
    plan = []
    for v in sorted(spk, key=lambda v: v["hnr"]):
        cand = [(w, p) for w, p in by_spk[int(v["spk"])]
                if Path(w).is_file() and sf.info(w).duration >= args.min_seconds]
        if not cand:
            raise SystemExit("REFUSING: speaker %s has no clip of %.1fs." % (v["spk"],
                                                                          args.min_seconds))
        src, phon = rng.choice(sorted(cand))
        plan.append({"spk": int(v["spk"]), "hnr": v["hnr"], "group": v["group"],
                     "source": src, "phon": phon})
    rng.shuffle(plan)
    # Side placement balanced within each group.
    flips = {}
    for g, n in groups.items():
        f = [True] * (n // 2) + [False] * (n - n // 2)
        rng.shuffle(f)
        flips[g] = f
    for p in plan:
        p["swap"] = flips[p["group"]].pop()

    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    vocoder, sr = bench.vocoder, bench.sample_rate

    made = defaultdict(dict)
    for lbl, ckpt in (("base", args.base), ("probe", args.probe)):
        model = bench._model_for(ckpt)
        hp = model.hparams.get("data_statistics") or {}
        if not hp or abs(float(model.mel_mean) - float(hp["mel_mean"])) > 1e-4:
            raise SystemExit("REFUSING: %s denormalises with stale or unknown statistics."
                             % ckpt)
        for i, p in enumerate(plan):
            o = bench.synth("item_%02d" % i, ckpt, "(phonemes)", p["spk"], (0.0, 0.0, 0.0),
                            delivery.DELIVERY_UNKNOWN, phonemes=p["phon"])
            with torch.no_grad():
                made[i][lbl] = to_waveform(o["mel"], vocoder, None).numpy()
    for i in made:
        if len(made[i]["base"]) != len(made[i]["probe"]):
            # Durations come from each model's own predictor, so a length difference is
            # legitimate here — recorded, not refused.
            plan[i]["len_differs"] = True

    served, truth, limited = [], {}, 0
    clips = Path(args.out) / "clips"
    for i, p in enumerate(plan):
        pair_key = "item_%02d" % i
        sides = [("A", "base"), ("B", "probe")]
        if p["swap"]:
            sides = [("A", "probe"), ("B", "base")]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            audio, bound = ear_bench.match_loudness(made[i][lbl], sr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), audio, sr, "PCM_24")
            bench.key[nm] = {"label": lbl, "pair": pair_key, "side": side_key}
            item[side_key] = nm
        served.append(item)
        truth[pair_key] = {"kind": KIND, "spk": p["spk"], "hnr": p["hnr"],
                           "group": p["group"], "A_label": sides[0][1],
                           "B_label": sides[1][1], "source": p["source"],
                           "len_differs": bool(p.get("len_differs"))}
        print("  %s  spk%-6d HNR %5.2f  %s" % (pair_key, p["spk"], p["hnr"], p["group"]))

    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    bench.write(Path(args.out).name, SETS, served,
                {"base": args.base, "probe": args.probe, "lufs": args.lufs,
                 "peak_limited_sides": limited, "clips_written": 2 * len(served),
                 "prediction": "PRE-REGISTERED: on the 16 rough speakers severity is lower "
                               "for the probe; exact two-sided sign-flip over 16 pairs, "
                               "confirmed at p < 0.05. Clean speakers: regression check, "
                               "reported not tested."},
                key_out=key_out)
    k = json.loads(Path(key_out).read_text())
    k["items"] = truth
    Path(key_out).write_text(json.dumps(k, indent=2))
    print("\nRENDERED %d items (%d rough, %d clean); %d pair(s) differ in length."
          % (len(served), groups["rough"], groups["clean"],
             sum(t["len_differs"] for t in truth.values())))
    print("Key: %s" % key_out)


if __name__ == "__main__":
    main()

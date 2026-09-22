"""IS THE HUM THE ACOUSTIC MODEL'S, OR IS IT THE CEILING? Three conditions, one clip.

THE QUESTION
------------
Everything measured so far says the acoustic model compresses periodicity, and the ear
says that compression is not what makes the hum: at the run's final checkpoint the model's
HNR distribution matches real audio in mean (5.20 vs 5.06 dB) and spread (1.78 vs 1.81),
and the hum is still loudly present. So what is left to blame — the model, or the
representation it predicts into?

`vocoder_sdt` established that the mel/vocoder round trip is AUDIBLE (d' 1.79, 0/8 false
alarms). Nobody has asked whether that artifact IS the hum. If a round trip of a REAL
recording carries it, the ceiling is the 80-bin mel and no amount of acoustic-model
retraining touches it. If it does not, the defect is squarely the model's.

⚠⚠ THE THREE PAIRINGS CONTROL EACH OTHER, WHICH IS WHY THERE ARE NO CATCH TRIALS. Every
item draws ONE source recording and serves two of {real, round trip, model}:

    real_vs_rt     does the vocoder alone carry the hum?
    rt_vs_model    does the model add hum BEYOND the vocoder?
    real_vs_model  the total, for scale

If the hum is entirely the representation's, `real_vs_rt` shows it and `rt_vs_model`
returns nothing. If it is entirely the model's, the reverse. A fabricated same/different
catch trial would add nothing either pairing does not already provide, and whichever
pairing comes back null prices the listener's criterion for the others.

⚠⚠ THE MODEL SPEAKS THE RECORDING'S OWN PHONEMES, NOT ITS TRANSCRIPT RE-PHONEMIZED.
Running the transcript back through G2P would put a second difference — front-end error —
on top of the one under test. The filelists are already IPA and `Bench.render(phonemes=)`
takes the row's field 3 through the same path training used.

⚠ LOUDNESS IS MATCHED BEFORE ANYTHING IS SERVED. The teacher-portfolio comparison had to
be redone because a level difference read as a quality difference, in the wrong direction.
Three differently-produced signals side by side would measure gain otherwise.

⚠ ONE SOURCE CLIP APPEARS IN AT MOST ONE ITEM. Serving the same recording under two
pairings lets a listener recognise it, and a recognised clip is judged against memory of
the last time rather than against the clip beside it.

Usage:
    python scripts/tools/render_ear_ceiling.py \
        --ckpt /path/checkpoint_epoch=005_step=0063107.ckpt \
        --corpus data/libritts_r_full_vat_v7 \
        --hnr-json /data/model-training/sonora/pitch_error/speaker_hnr_all.json \
        --out /data/model-training/sonora/eartest/ceiling
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import ear_bench                                     # noqa: E402
from lib.corpus_filelist import read_corpus                   # noqa: E402
from matcha.cli import to_waveform                            # noqa: E402
from matcha import delivery                                   # noqa: E402
from matcha.delivery import VAT_DIM                           # noqa: E402
from matcha.utils.audio import mel_spectrogram                # noqa: E402

KINDS = ("real_vs_rt", "rt_vs_model", "real_vs_model")

SETS = {
    "hum": {
        "title": "Which one has more of the machine in it?",
        "ask": ("Two versions of the same sentence. Some are real recordings, some are "
                "not, and some pairs genuinely have no difference. Ignore which you "
                "would rather listen to and ignore the reading. Listen for the ROBOTIC "
                "quality you have described before — the buzz or hum under the voice, "
                "the sense of something mechanical trying to sound human. Which side has "
                "more of it? 'No difference' is the right answer when it is true."),
        "labels": {"A": "A has more of the hum", "same": "No difference",
                   "B": "B has more of the hum"},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hnr-json", required=True)
    ap.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--per-kind", type=int, default=8)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="ceiling-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.per_kind < 4:
        raise SystemExit("REFUSING: --per-kind %d. Each pairing is one arm of a "
                         "three-way decomposition and four items cannot carry one."
                         % args.per_kind)

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    for k in ("n_fft", "n_feats", "sample_rate", "hop_length", "win_length", "f_min",
              "f_max"):
        if k not in cfg:
            raise SystemExit("REFUSING: %s has no `%s`. The mel here must be the one the "
                             "checkpoint was trained on, and guessing a parameter makes "
                             "the round trip a different transform."
                             % (args.data_config, k))

    spk_hnr = {int(k): v for k, v in
               json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    rows = read_corpus(args.corpus, "train", VAT_DIM)
    by_spk = defaultdict(list)
    for line, wav, s, nphon in rows:
        if s in spk_hnr:
            by_spk[s].append((wav, line.split("|")[2]))

    # ⚠ SPREAD OVER NATURAL HNR, because that is the axis the whole investigation is
    # about. Drawing speakers at random would pile the sample at the corpus median and the
    # result would say nothing about whether the ceiling differs for a rough voice.
    order = sorted(by_spk, key=lambda s: spk_hnr[s]["hnr"])
    need = args.per_kind * len(KINDS)
    if len(order) < need:
        raise SystemExit("REFUSING: %d speakers carry both rows and a measured HNR; %d "
                         "items need that many distinct ones." % (len(order), need))
    picks = [order[round(i * (len(order) - 1) / (need - 1))] for i in range(need)]

    rng = random.Random(args.seed)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    vocoder, sr = bench.vocoder, bench.sample_rate
    if sr != cfg["sample_rate"]:
        raise SystemExit("REFUSING: the vocoder is %d Hz and the mel config says %d."
                         % (sr, cfg["sample_rate"]))
    clips = Path(args.out) / "clips"

    def roundtrip(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32")).float().unsqueeze(0)
        mel = mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                              cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                              cfg["f_max"], center=False)
        with torch.no_grad():
            return to_waveform(mel, vocoder, None).numpy()

    plan = [(picks[i], KINDS[i % len(KINDS)]) for i in range(need)]
    rng.shuffle(plan)
    flip = [True] * (len(plan) // 2) + [False] * (len(plan) - len(plan) // 2)
    rng.shuffle(flip)

    served, truth, limited = [], {}, 0
    for i, ((s, kind), swap) in enumerate(zip(plan, flip)):
        cand = [(w, p) for w, p in by_spk[s]
                if Path(w).is_file() and sf.info(w).duration >= args.min_seconds]
        if not cand:
            raise SystemExit("REFUSING: speaker %d has no readable clip of at least "
                             "%.1fs." % (s, args.min_seconds))
        src, phon = rng.choice(cand)
        x, xsr = sf.read(src, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if xsr != cfg["sample_rate"]:
            raise SystemExit("REFUSING: %s is %d Hz and the mel config is %d."
                             % (src, xsr, cfg["sample_rate"]))

        pair_key = "item_%02d" % i
        made = {}
        if kind in ("real_vs_rt", "real_vs_model"):
            made["real"] = x
        if kind in ("real_vs_rt", "rt_vs_model"):
            made["rt"] = roundtrip(x)
        left_lbl, right_lbl = kind.split("_vs_")
        if "model" in (left_lbl, right_lbl):
            name = ear_bench.opaque(pair_key, "model", args.salt)
            # ⚠ THE LANE IS `DELIVERY_UNKNOWN`, NOT 0. The vocabulary is closed and
            # `delivery_index` refuses anything outside it — an integer looks like an
            # index and is not one. Caught by that refusal on the first run.
            bench.render(pair_key, "model", args.ckpt, "(phonemes)", s, (0.0, 0.0, 0.0),
                         delivery.DELIVERY_UNKNOWN, "model", phonemes=phon)
            made["model"], _ = sf.read(str(clips / ("%s.wav" % name)), dtype="float32")

        sides = [("A", left_lbl), ("B", right_lbl)]
        if swap:
            sides = [("A", right_lbl), ("B", left_lbl)]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            audio, bound = ear_bench.match_loudness(made[lbl], xsr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), audio, xsr, "PCM_24")
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
                {"ckpt": args.ckpt, "per_kind": args.per_kind, "lufs": args.lufs,
                 "peak_limited_sides": limited,
                 "prediction": "if real_vs_rt carries the hum the ceiling is the mel; "
                               "if rt_vs_model carries it the model is to blame"},
                key_out=key_out)
    k = json.loads(Path(key_out).read_text())
    k["items"] = truth
    Path(key_out).write_text(json.dumps(k, indent=2))
    if limited:
        print("\n⚠ %d side(s) hit the peak ceiling before reaching %.1f LUFS, so those "
              "are not fully level-matched." % (limited, args.lufs))
    print("\nRENDERED %d items (%s)."
          % (len(served), " ".join("%s=%d" % kv for kv in
                                   sorted(Counter(v["kind"] for v in truth.values())
                                          .items()))))
    print("Key: %s" % key_out)


if __name__ == "__main__":
    main()

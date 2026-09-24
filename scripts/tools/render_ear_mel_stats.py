"""DOES THE STALE NORMALISATION MAKE THE HUM? One model render, denormalised two ways.

THE FINDING THIS TESTS (2026-09-22)
-----------------------------------
A checkpoint carries two copies of the mel statistics, and in `vat7_finetune` they
disagree:

    hyper_parameters.data_statistics   mean -5.544  std 2.430   what the run TRAINED on
    state_dict mel_mean / mel_std      mean -6.631  std 2.483   what inference DENORMALISES with

The buffers are set in `__init__` from the config and then OVERWRITTEN by
`load_state_dict` — the warm start carried a donor's values forward, and nothing in
training ever reads or updates them. So every render through `load_matcha` (the Vocalizer,
every ear bench, the ceiling bench's model condition) hands the vocoder a mel about 1.09 log
units low: -9.4 dB, uniform across all 80 bins. Measured on 64 aligned renders against the
real mels of the same clips: -1.07 in loud frames, -1.17 in quiet ones, -0.8 to -1.3 in
every 10-bin band. The vocoder never saw a mel at that level in training — it trained on
peak-normalised audio.

⚠⚠ THIS CONFOUNDS THE CEILING BENCH. Its model condition carried the offset and its round
trip did not, so the +1.75 severity gap contains it. Three pairings pull it apart:

    stale_vs_fixed      the same render, the same noise, two denormalisations
    rt_vs_rt_shifted    a REAL recording's mel, vocoded as-is and with the stale transform
                        applied — the vocoder's response to the offset, with no model at all
    real_vs_fixed       what remains once the statistics are right

⚠ LOUDNESS IS MATCHED AFTER VOCODING, SO THE 9 dB ITSELF IS NOT WHAT IS HEARD. A level
difference would be heard and would read as a quality difference. What survives the match
is how the vocoder behaves on a mel at the wrong level, which is the question.

⚠ THE CEILING BENCH'S SPEAKERS ARE EXCLUDED. That bench spreads its picks over natural HNR
by rank, so the same count would draw the same speakers and a listener could recognise a
voice from two hours ago.

Usage:
    python scripts/tools/render_ear_mel_stats.py \\
        --ckpt /path/checkpoint_epoch=005_step=0063107.ckpt \\
        --corpus data/libritts_r_full_vat_v7 \\
        --hnr-json /data/model-training/sonora/pitch_error/speaker_hnr_all.json \\
        --exclude-key /data/model-training/sonora/eartest/_keys/ceiling.key.json \\
        --out /data/model-training/sonora/eartest/mel_stats
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
from matcha import delivery                                   # noqa: E402
from matcha.cli import to_waveform                            # noqa: E402
from matcha.delivery import VAT_DIM                           # noqa: E402
from matcha.utils.audio import mel_spectrogram                # noqa: E402
from matcha.utils.model import denormalize                    # noqa: E402

KINDS = {"stale_vs_fixed": 12, "rt_vs_rt_shifted": 6, "real_vs_fixed": 6}

SETS = {
    "hum": {
        "title": "How much machine is in each one?",
        "ask": ("Two versions of the same sentence. Some are real recordings and some "
                "are not. Ignore which you would rather listen to and ignore the "
                "reading. Rate EACH clip on its own for the ROBOTIC quality you have "
                "described — the buzz or hum under the voice, the layering, the sense of "
                "something mechanical trying to sound human. 0 means you cannot hear it "
                "at all. 5 means it sounds like a Freak-a-Zoid robot. Giving both clips "
                "the same rating is a real answer."),
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
    ap.add_argument("--exclude-key", action="append", default=[])
    ap.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="mel-stats-v1")
    ap.add_argument("--seed", type=int, default=4321)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data_config).read_text())
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
        raise SystemExit("REFUSING: %d speakers cannot spread %d distinct picks; one voice "
                         "would be served twice." % (len(order), need))

    rng = random.Random(args.seed)
    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    vocoder, sr = bench.vocoder, bench.sample_rate
    model = bench._model_for(args.ckpt)

    # ⚠ THE STALE PAIR IS READ FROM THE FILE, NOT FROM THE MODULE. Since the load hook
    # corrects the buffers (matcha/mel_stats.py), `model.mel_mean` is already the fixed
    # value, and reading it here would make every pair two copies of one clip.
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    buf_mean, buf_std = float(raw["mel_mean"]), float(raw["mel_std"])
    del raw
    hp = model.hparams.get("data_statistics") or {}
    hp_mean, hp_std = float(hp["mel_mean"]), float(hp["mel_std"])
    print("  buffers   mean %.4f std %.4f   (what inference uses)" % (buf_mean, buf_std))
    print("  hparams   mean %.4f std %.4f   (what training normalised with)"
          % (hp_mean, hp_std))
    if abs(buf_mean - hp_mean) < 1e-3 and abs(buf_std - hp_std) < 1e-3:
        raise SystemExit("REFUSING: this checkpoint's two sets of statistics agree, so "
                         "every pair here would be two copies of one clip.")

    def vocode(mel):
        with torch.no_grad():
            return to_waveform(mel, vocoder, None).numpy()

    def real_mel(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32")).unsqueeze(0)
        return mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                               cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                               cfg["f_max"], center=False)

    # ⚠⚠ STRATIFIED OVER HNR, THEN SHUFFLED — two separate faults, both in the first draft.
    # A random assignment of kinds to the HNR-sorted picks gave `stale_vs_fixed` none of the
    # roughest voices, which the ceiling bench says carry the most hum (rho -0.537). And
    # zipping onto `picks` served the items in ascending HNR, which is the serving-order
    # confound that sank the 2026-09-19 pitch dose response (+0.996). The cycle below
    # spreads every kind across the whole range; the shuffle breaks the order.
    cycle = ["stale_vs_fixed", "rt_vs_rt_shifted", "stale_vs_fixed", "real_vs_fixed"]
    kinds = [cycle[i % len(cycle)] for i in range(need)]
    if Counter(kinds) != Counter(KINDS):
        raise SystemExit("REFUSING: the cycle yields %s, not %s." % (dict(Counter(kinds)),
                                                                    KINDS))
    plan = list(zip(picks, kinds))
    rng.shuffle(plan)
    flip = [True] * (need // 2) + [False] * (need - need // 2)
    rng.shuffle(flip)

    served, truth, limited = [], {}, 0
    clips = Path(args.out) / "clips"
    for i, ((s, kind), swap) in enumerate(zip(plan, flip)):
        cand = [(w, p) for w, p in by_spk[s]
                if Path(w).is_file() and sf.info(w).duration >= args.min_seconds]
        if not cand:
            raise SystemExit("REFUSING: speaker %d has no readable clip of at least %.1fs."
                             % (s, args.min_seconds))
        src, phon = rng.choice(cand)
        x, xsr = sf.read(src, dtype="float32")
        if xsr != sr:
            raise SystemExit("REFUSING: %s is %d Hz and the vocoder is %d." % (src, xsr, sr))

        pair_key = "item_%02d" % i
        made = {}
        if kind == "rt_vs_rt_shifted":
            m = real_mel(x)
            made["rt"] = vocode(m)
            # Exactly what a stale render does to a correctly-normalised output.
            made["rt_shifted"] = vocode((m - hp_mean) / hp_std * buf_std + buf_mean)
        else:
            o = bench.synth(pair_key, args.ckpt, "(phonemes)", s, (0.0, 0.0, 0.0),
                            delivery.DELIVERY_UNKNOWN, phonemes=phon)
            made["fixed"] = vocode(denormalize(o["decoder_outputs"], hp_mean, hp_std))
            if kind == "stale_vs_fixed":
                made["stale"] = vocode(denormalize(o["decoder_outputs"], buf_mean, buf_std))
            else:
                made["real"] = x
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
                {"ckpt": args.ckpt, "kinds": KINDS, "lufs": args.lufs,
                 "peak_limited_sides": limited,
                 "stats": {"buffers": [buf_mean, buf_std], "hparams": [hp_mean, hp_std]},
                 "prediction": "if stale_vs_fixed carries the hum, the stale buffers are "
                               "a cause and the fix is free; rt_vs_rt_shifted says whether "
                               "the vocoder alone produces it from a level-shifted mel"},
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

"""DID FINE-TUNING THE VOCODER ON THE MODEL'S OWN MELS REMOVE THE HUM?

The vocoder learned on mels computed from real audio only. `render_aligned_mels.py` gave it
vat7 ep005's own mels, aligned to the real audio they came from, and hifi-gan's fine-tuning
mode trained it to turn the one into the other. This bench asks whether that moved the
hum, and what it cost on real mels.

    model_old_vs_model_new   ONE model render, ONE mel, two vocoders — the fine-tune's
                             effect on exactly the signal it was trained for
    rt_old_vs_rt_new         a REAL recording's mel through both — the regression check,
                             and a test of the hypothesis that the "layering" the owner
                             hears on clean voices is partly the vocoder's (mel_stats:
                             round trips of real clean voices drew layering at 1-2)
    real_vs_model_new        what remains

⚠⚠ THE SPEAKERS ARE THE CEILING BENCH'S 24, AND THAT IS WHY THEY WERE HELD OUT.
`build_vocoder_ft_filelist.py` kept every clip of theirs out of the fine-tune, so the new
vocoder has never seen these voices' mels — and each item reuses the ceiling bench's own
source recording, so a model render here can be read beside its severity there.

⚠ THE MEL IS DENORMALISED WITH THE TRAINING CONFIG'S STATISTICS, the only ones the
fine-tune ever saw. `load_matcha` corrects the stale buffers on load (matcha/mel_stats.py);
this refuses if it did not, rather than vocoding a mel 9.4 dB low into a vocoder that was
just taught the right level.

⚠ STRATIFIED OVER HNR, THEN SHUFFLED — see render_ear_mel_stats.py for the two faults this
prevents.

Usage:
    python scripts/tools/render_ear_vocoder_ft.py \\
        --ckpt /path/checkpoint_epoch=005_step=0063107.ckpt \\
        --new-vocoder /data/model-training/vocoder/vocoder_ft/vat7_ep005/cp/g_02520000 \\
        --corpus data/libritts_r_full_vat_v7 \\
        --sources /data/model-training/sonora/pitch_error/_ceiling_sources.json \\
        --out /data/model-training/sonora/eartest/vocoder_ft
"""

import argparse
import json
import random
import sys
from collections import Counter
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
from matcha.cli import load_vocoder_24k, to_waveform          # noqa: E402
from matcha.delivery import VAT_DIM                           # noqa: E402
from matcha.utils.audio import mel_spectrogram                # noqa: E402

KINDS = {"model_old_vs_model_new": 12, "rt_old_vs_rt_new": 6, "real_vs_model_new": 6}
CYCLE = ["model_old_vs_model_new", "rt_old_vs_rt_new", "model_old_vs_model_new",
         "real_vs_model_new"]

SETS = {
    "hum": {
        "title": "How much machine is in each one?",
        "ask": ("Two versions of the same sentence. Some are real recordings and some "
                "are not. Ignore which you would rather listen to and ignore the "
                "reading. Rate EACH clip on its own for the ROBOTIC quality you have "
                "described — the buzz or hum under the voice, the layering, the sense of "
                "something mechanical trying to sound human. 0 means you cannot hear it "
                "at all. 5 means it sounds like a Freak-a-Zoid robot. Giving both clips "
                "the same rating is a real answer. A note saying 'buzz' or 'layering' "
                "helps."),
        "scale": {"max": 5, "anchors": {
            "0": "cannot hear the hum at all",
            "5": "a Freak-a-Zoid robot"}},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--new-vocoder", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--sources", required=True,
                    help="JSON {items: {id: {spk, source, hnr, f0}}} — the ceiling "
                         "bench's speakers and recordings, with nothing that unblinds it")
    ap.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="vocoder-ft-v1")
    ap.add_argument("--seed", type=int, default=2718)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    src = json.loads(Path(args.sources).read_text())["items"]
    need = sum(KINDS.values())
    if len(src) != need:
        raise SystemExit("REFUSING: %d sources for %d items." % (len(src), need))
    phon_of = {wav: line.split("|")[2]
               for line, wav, s, n in read_corpus(args.corpus, "train", VAT_DIM)}
    missing = [v["source"] for v in src.values() if v["source"] not in phon_of]
    if missing:
        raise SystemExit("REFUSING: %d source(s) are not training rows, e.g. %s."
                         % (len(missing), missing[:2]))

    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    old_voc, sr = bench.vocoder, bench.sample_rate
    new_voc, new_sr = load_vocoder_24k(ear_bench.DEVICE, ckpt=args.new_vocoder)
    if new_sr != sr:
        raise SystemExit("REFUSING: the vocoders disagree on sample rate.")
    model = bench._model_for(args.ckpt)
    hp = model.hparams.get("data_statistics") or {}
    if abs(float(model.mel_mean) - float(hp["mel_mean"])) > 1e-4:
        raise SystemExit("REFUSING: the model denormalises with mean %.4f and trained on "
                         "%.4f — the load-time correction did not run."
                         % (float(model.mel_mean), float(hp["mel_mean"])))

    def vocode(voc, mel):
        with torch.no_grad():
            return to_waveform(mel, voc, None).numpy()

    def real_mel(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32")).unsqueeze(0)
        return mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                               cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                               cfg["f_max"], center=False)

    order = sorted(src.values(), key=lambda v: v["hnr"])
    kinds = [CYCLE[i % len(CYCLE)] for i in range(need)]
    if Counter(kinds) != Counter(KINDS):
        raise SystemExit("REFUSING: the cycle yields %s, not %s." % (dict(Counter(kinds)),
                                                                    KINDS))
    rng = random.Random(args.seed)
    plan = list(zip(order, kinds))
    rng.shuffle(plan)
    flip = [True] * (need // 2) + [False] * (need - need // 2)
    rng.shuffle(flip)

    served, truth, limited = [], {}, 0
    clips = Path(args.out) / "clips"
    for i, ((v, kind), swap) in enumerate(zip(plan, flip)):
        s, wav = int(v["spk"]), v["source"]
        x, xsr = sf.read(wav, dtype="float32")
        if xsr != sr:
            raise SystemExit("REFUSING: %s is %d Hz and the vocoder is %d." % (wav, xsr, sr))
        pair_key = "item_%02d" % i
        made = {}
        if kind == "rt_old_vs_rt_new":
            m = real_mel(x)
            made["rt_old"], made["rt_new"] = vocode(old_voc, m), vocode(new_voc, m)
        else:
            o = bench.synth(pair_key, args.ckpt, "(phonemes)", s, (0.0, 0.0, 0.0),
                            delivery.DELIVERY_UNKNOWN, phonemes=phon_of[wav])
            made["model_new"] = vocode(new_voc, o["mel"])
            if kind == "model_old_vs_model_new":
                made["model_old"] = vocode(old_voc, o["mel"])
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
        truth[pair_key] = {"kind": kind, "spk": s, "hnr": v["hnr"], "f0": v["f0"],
                           "A_label": sides[0][1], "B_label": sides[1][1], "source": wav}
        print("  %s  spk%-6d HNR %5.2f  %s" % (pair_key, s, v["hnr"], kind))

    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    bench.write(Path(args.out).name, SETS, served,
                {"ckpt": args.ckpt, "new_vocoder": args.new_vocoder, "kinds": KINDS,
                 "lufs": args.lufs, "peak_limited_sides": limited,
                 "prediction": "if model_old_vs_model_new carries the hum, the vocoder's "
                               "response to predicted mels was part of it; "
                               "rt_old_vs_rt_new prices what the fine-tune cost"},
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

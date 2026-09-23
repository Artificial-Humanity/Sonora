"""DID FINE-TUNING THE VOCODER ON THE MODEL'S OWN MELS REMOVE THE HUM?

The vocoder learned on mels computed from real audio only. `render_aligned_mels.py` gave it
vat7 ep005's own mels, aligned to the real audio they came from, and hifi-gan's fine-tuning
mode trained it to turn the one into the other. This bench asks whether that moved the
hum, WHEN in the run it moved, and what it cost on real mels.

Every fine-tune checkpoint is a condition (`--new-vocoder`, repeatable), and each pair
puts ONE of them beside the old vocoder:

    model_old_vs_model_<ck>   ONE model render, ONE mel, two vocoders — the fine-tune's
                              effect on exactly the signal it was trained for. 24 pairs,
                              6 per checkpoint for four.
    rt_old_vs_rt_<ck>         a REAL recording's mel through both — the regression check,
                              and a test of the hypothesis that the "layering" the owner
                              hears on clean voices is partly the vocoder's (mel_stats:
                              round trips of real clean voices drew layering at 1-2).
                              8 pairs, 2 per checkpoint.

⚠⚠ ONE BENCH ACROSS ALL CHECKPOINTS, NOT ONE BENCH EACH, and the cost is stated rather
than hidden: six pairs per checkpoint cannot rank neighbours. What the design CAN answer
is the pooled question (did the fine-tune help at all, 24 pairs) and a trend over steps.
The validation mel error went flat after the first 2,500 steps (0.872 -> 0.827, then
0.828, 0.826), and mel distance has been blind to this defect nine times, so the ear is
asked where the gain stopped rather than being told.

There is no real-vs-model pairing: two benches have already put real audio at 0.00 and
0.06, and the model's absolute severity here reads against those.

⚠⚠ THE SPEAKERS ARE THE CEILING BENCH'S 24, AND THAT IS WHY THEY WERE HELD OUT.
`build_vocoder_ft_filelist.py` kept every clip of theirs out of the fine-tune, so the new
vocoder has never seen these voices' mels — and each model item reuses the ceiling bench's
own source recording, so a render here can be read beside its severity there. The round
trips use a DIFFERENT recording of eight of those speakers, so no sentence is served
twice.

⚠ THE MEL IS DENORMALISED WITH THE TRAINING CONFIG'S STATISTICS, the only ones the
fine-tune ever saw. `load_matcha` corrects the stale buffers on load (matcha/mel_stats.py);
this refuses if it did not, rather than vocoding a mel 9.4 dB low into a vocoder that was
just taught the right level.

⚠ STRATIFIED OVER HNR, THEN SHUFFLED — see render_ear_mel_stats.py for the two faults this
prevents. Checkpoints are cycled over the HNR-sorted speakers, so each one spans the range.

Usage:
    python scripts/tools/render_ear_vocoder_ft.py \\
        --ckpt /path/checkpoint_epoch=005_step=0063107.ckpt \\
        --new-vocoder .../cp/g_02512500 --new-vocoder .../cp/g_02515000 \\
        --new-vocoder .../cp/g_02517500 --new-vocoder .../cp/g_02520000 \\
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

MODEL_PAIRS, RT_PAIRS = 24, 8


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
    ap.add_argument("--new-vocoder", action="append", required=True,
                    help="a fine-tune generator checkpoint; repeat for each one compared")
    ap.add_argument("--base-step", type=int, default=2510000,
                    help="the step the fine-tune resumed from, so a label reads as steps "
                         "INTO this run (g_02512500 -> ft02500), not hifi-gan's lifetime")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--sources", required=True,
                    help="JSON {items: {id: {spk, source, hnr, f0}}} — the ceiling "
                         "bench's speakers and recordings, with nothing that unblinds it")
    ap.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="vocoder-ft-v1")
    ap.add_argument("--seed", type=int, default=2718)
    args = ap.parse_args()

    cks = []
    for path in args.new_vocoder:
        step = int(Path(path).name.split("_")[-1]) - args.base_step
        if step <= 0:
            raise SystemExit("REFUSING: %s is not after --base-step %d." % (path,
                                                                         args.base_step))
        cks.append(("ft%05d" % step, path))
    if len({c[0] for c in cks}) != len(cks):
        raise SystemExit("REFUSING: two --new-vocoder arguments name the same step.")
    if MODEL_PAIRS % len(cks) or RT_PAIRS % len(cks):
        raise SystemExit("REFUSING: %d checkpoints do not divide %d model and %d round-"
                         "trip pairs evenly." % (len(cks), MODEL_PAIRS, RT_PAIRS))

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    src = json.loads(Path(args.sources).read_text())["items"]
    if len(src) != MODEL_PAIRS:
        raise SystemExit("REFUSING: %d sources for %d model pairs." % (len(src), MODEL_PAIRS))
    rows_of, phon_of = {}, {}
    for line, wav, s, n in read_corpus(args.corpus, "train", VAT_DIM):
        phon_of[wav] = line.split("|")[2]
        rows_of.setdefault(s, []).append(wav)
    missing = [v["source"] for v in src.values() if v["source"] not in phon_of]
    if missing:
        raise SystemExit("REFUSING: %d source(s) are not training rows, e.g. %s."
                         % (len(missing), missing[:2]))

    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    old_voc, sr = bench.vocoder, bench.sample_rate
    voc = {}
    for lbl, path in cks:
        voc[lbl], new_sr = load_vocoder_24k(ear_bench.DEVICE, ckpt=path)
        if new_sr != sr:
            raise SystemExit("REFUSING: %s disagrees with the old vocoder on sample rate."
                             % path)
    model = bench._model_for(args.ckpt)
    hp = model.hparams.get("data_statistics") or {}
    if abs(float(model.mel_mean) - float(hp["mel_mean"])) > 1e-4:
        raise SystemExit("REFUSING: the model denormalises with mean %.4f and trained on "
                         "%.4f — the load-time correction did not run."
                         % (float(model.mel_mean), float(hp["mel_mean"])))

    def vocode(v, mel):
        with torch.no_grad():
            return to_waveform(mel, v, None).numpy()

    def real_mel(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32")).unsqueeze(0)
        return mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                               cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                               cfg["f_max"], center=False)

    rng = random.Random(args.seed)
    order = sorted(src.values(), key=lambda v: v["hnr"])
    plan = [(v, "model", cks[i % len(cks)][0]) for i, v in enumerate(order)]
    # Round trips: every third speaker up the HNR order, on a recording the model items do
    # not use, so no sentence is served twice.
    for j, v in enumerate(order[::MODEL_PAIRS // RT_PAIRS][:RT_PAIRS]):
        other = [w for w in rows_of[int(v["spk"])] if w != v["source"] and
                 Path(w).is_file() and sf.info(w).duration >= args.min_seconds]
        if not other:
            raise SystemExit("REFUSING: speaker %s has no second clip of %.1fs."
                             % (v["spk"], args.min_seconds))
        plan.append((dict(v, source=rng.choice(sorted(other))), "rt", cks[j % len(cks)][0]))
    rng.shuffle(plan)
    need = len(plan)
    flip = [True] * (need // 2) + [False] * (need - need // 2)
    rng.shuffle(flip)

    served, truth, limited = [], {}, 0
    clips = Path(args.out) / "clips"
    for i, ((v, what, ck), swap) in enumerate(zip(plan, flip)):
        s, wav = int(v["spk"]), v["source"]
        x, xsr = sf.read(wav, dtype="float32")
        if xsr != sr:
            raise SystemExit("REFUSING: %s is %d Hz and the vocoder is %d." % (wav, xsr, sr))
        pair_key = "item_%02d" % i
        old_lbl, new_lbl = "%s_old" % what, "%s_%s" % (what, ck)
        if what == "rt":
            mel = real_mel(x)
        else:
            mel = bench.synth(pair_key, args.ckpt, "(phonemes)", s, (0.0, 0.0, 0.0),
                              delivery.DELIVERY_UNKNOWN, phonemes=phon_of[wav])["mel"]
        made = {old_lbl: vocode(old_voc, mel), new_lbl: vocode(voc[ck], mel)}
        kind = "%s_vs_%s" % (old_lbl, new_lbl)
        sides = [("A", old_lbl), ("B", new_lbl)]
        if swap:
            sides = [("A", new_lbl), ("B", old_lbl)]
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
                {"ckpt": args.ckpt, "new_vocoders": dict(cks), "base_step": args.base_step,
                 "lufs": args.lufs, "peak_limited_sides": limited,
                 "prediction": "if model_old_vs_model_ftN carries the hum, the vocoder's "
                               "response to predicted mels was part of it; "
                               "rt_old_vs_rt_ftN prices what the fine-tune cost"},
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

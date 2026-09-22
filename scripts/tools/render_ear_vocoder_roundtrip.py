"""Render a BLIND ear test of the VOCODER ALONE, across the speaker pitch range.

WHAT THIS ISOLATES, AND WHY IT IS THE NEXT QUESTION
---------------------------------------------------
Three results stand (`notes/STATE.md` § the hum): the robotic hum is NOT in the training
audio, the model produces it, and its severity rises smoothly as speaker pitch falls
(Spearman +0.918 against pitch gap, permutation p = 0.006).

Coverage does not explain that gradient. The 100-120 Hz band already holds 16% of the
corpus, and a 205 Hz voice was still judged hummier than a 253 Hz one — there is no
shortage of 205 Hz speakers. What DOES vary smoothly with pitch is physics: at 90 Hz the
harmonics sit 90 Hz apart, an 80-bin mel cannot resolve them individually down there, and
whatever reconstructs audio from that mel has to invent the structure it cannot see.
Invented periodic structure is what a buzz is.

So this puts **no acoustic model in the path at all**. Real recording -> mel -> vocoder ->
audio, against the untouched original. If the round trip hums at 90 Hz and not at 200 Hz,
the vocoder is the culprit and the corpus and the sampler are both exonerated. If it is
clean at every pitch, the vocoder is cleared and the defect is in the mel the acoustic
model PREDICTS, which is a different lever entirely.

⚠ THIS DIRECTLY RE-OPENS A CLOSED MEASUREMENT, ON PURPOSE. The vocoder was called
"perceptually transparent" on 2026-08-06 at mel L1 = 10.2% of mel_std, and that number has
been quoted since to rule the vocoder out. It was a GLOBAL average. A vocoder can be
transparent on average and poor at 90 Hz, and nothing we own would have caught it — the
measurement had no pitch axis.

⚠ THE MEL PATH IS THE TRAINING ONE OR THE TEST IS WORTHLESS. Parameters come from the
data config, not from here, and `matcha.utils.audio.mel_spectrogram` is the same function
`text_mel_datamodule.get_mel` calls, with `center=False` as it uses. A probe that built
its own mel would measure a pipeline that does not exist.

⚠⚠ THE CONTROL IS TWO-SIDED, BECAUSE ONE SIDE OF IT IS NOT A CONTROL. Identical-clip
pairs catch only a FALSE POSITIVE — a listener hearing a difference where there is none.
When the listener ties everything, an identical pair tying is ENTAILED, not evidence, and
the first run of this bench tied 14 of 14 with nothing in the design able to show the
instrument was working at all. That is the repo's "empty enumeration" class landing on the
control itself.

So `--positive` pairs serve the original against a clip put through the SAME round trip
`--positive-depth` times. It is the identical distortion, compounded, so it needs no new
artifact type and no synthetic buzz that might not resemble the real thing. The measured
mel distance at depth 1 and at depth N is PRINTED, so what the ear was asked to detect is
on the record rather than assumed. If the positive pairs tie too, the instrument is blind
to this distortion class and the single-pass result means nothing either way.

⚠ LOUDNESS IS MATCHED AFTER THE ROUND TRIP. Vocoding changes level, and the louder side of
a pair reads as the better one — the confound that already ran one verdict backwards here.

Usage (in the training image; the host venv has no torch):
    python scripts/tools/render_ear_vocoder_roundtrip.py \
        --corpus data/libritts_r_full_vat_v7 --f0-json speaker_f0.json \
        --data-config configs/data/libritts_r_full_vat_v7.yaml \
        --out /data/model-training/sonora/eartest/vocoder_roundtrip
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # noqa: E402
import pyloudnorm                                             # noqa: E402
import soundfile as sf                                        # noqa: E402
import torch                                                  # noqa: E402
import yaml                                                   # noqa: E402

from lib import ear_bench                                     # noqa: E402
from matcha.cli import load_vocoder_24k, to_waveform          # noqa: E402
from matcha.utils.audio import mel_spectrogram                # noqa: E402

# ⚠⚠ THE BRIEF IS DELIBERATELY UNPRIMED, AND THAT IS A CORRECTION.
# The previous brief said "listen for the robotic quality you have described", named the
# hum, and then counted the hums found. The owner raised it themselves, against a result
# that was going their way: "I couldn't be entirely sure of any robotic hum. I began to
# question if over-active listening out for it was influencing my ears." Naming the percept
# and then measuring it is how an expectation becomes a finding. This brief names no
# artifact, describes nothing to listen for, and does not say how many pairs differ.
SETS = {
    "roundtrip": {
        "title": "Do these two differ?",
        "ask": ("Two versions of the same recording. Some pairs are identical and some are "
                "not, and the proportion is not stated on purpose. Do not go looking for "
                "any particular flaw — if you find yourself straining to justify a "
                "difference, that is the answer to record as 'no difference'. Say which "
                "side sounds altered ONLY when it is plain to you."),
        "labels": {"A": "A sounds altered", "same": "No difference",
                   "B": "B sounds altered"},
    },
}


def refuse(msg):
    raise SystemExit("REFUSING: %s" % msg)


def match_loudness(x, sr, target):
    """Level-match, peak-limited. Returns the audio and whether the ceiling bound."""
    loud = pyloudnorm.Meter(sr).integrated_loudness(x.astype("float64"))
    if not np.isfinite(loud):
        refuse("a clip has no measurable loudness")
    gain = 10.0 ** ((target - loud) / 20.0)
    peak = float(np.max(np.abs(x))) or 1.0
    g = min(gain, 0.99 / peak)
    return (x * g).astype("float32"), g < gain


# ⚠⚠ prove_writable MOVED TO ear_bench ON 2026-09-22 and this file kept a local copy of
# it in the same diff — a second implementation of a rule this repo has already been
# bitten by twice. The local one also chmod'd unguarded, so it raised where the shared
# copy tolerates a directory someone else owns.
prove_writable = ear_bench.prove_writable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--f0-json", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--identical", type=int, default=2,
                    help="pairs serving the SAME audio on both sides (false-positive check)")
    ap.add_argument("--positive", type=int, default=2,
                    help="pairs serving a KNOWN-WORSE clip (discrimination check)")
    ap.add_argument("--positive-max-f0", type=float, default=130.0,
                    help="positive controls are placed at or below this pitch")
    ap.add_argument("--positive-depth", type=int, default=4,
                    help="how many times to re-vocode the known-worse side")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="vocoder-roundtrip-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.identical < 1 or args.positive < 1:
        refuse("--identical %d --positive %d: this bench needs BOTH controls. Identical "
               "pairs alone cannot tell a calibrated listener from one answering 'tie' by "
               "default, which is exactly how the first run of this test produced a "
               "vacuous pass." % (args.identical, args.positive))
    cfg = yaml.safe_load(Path(args.data_config).read_text())
    need = ("n_fft", "n_feats", "sample_rate", "hop_length", "win_length", "f_min", "f_max")
    missing = [k for k in need if k not in cfg]
    if missing:
        refuse("%s does not state %s — the mel must come from the training config, never "
               "from this script" % (args.data_config, ", ".join(missing)))
    print("MEL from %s: n_fft=%d n_mels=%d sr=%d hop=%d win=%d f=%s-%s"
          % (args.data_config, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
             cfg["hop_length"], cfg["win_length"], cfg["f_min"], cfg["f_max"]))

    f0 = {int(k): v for k, v in json.loads(Path(args.f0_json).read_text())["speakers"].items()}
    rows = {}
    with open(os.path.join(args.corpus, "train_op.txt"), encoding="utf-8") as fh:
        for line in fh:
            p = line.strip().split("|")
            if len(p) == 4:
                rows.setdefault(int(p[1]), []).append(p[0])

    pool = sorted(s for s in f0 if f0[s]["rows"] >= args.min_rows and s in rows)
    if len(pool) < args.pairs:
        refuse("only %d speakers qualify and %d pairs were asked for" % (len(pool), args.pairs))

    # Speakers spread evenly over the PITCH RANGE, not sampled at random: the prediction is
    # about the extremes and a random draw would crowd the middle.
    pool.sort(key=lambda s: f0[s]["f0"])
    idx = [round(i * (len(pool) - 1) / (args.pairs - 1)) for i in range(args.pairs)]
    chosen = [pool[i] for i in sorted(set(idx))]

    rng = random.Random(args.seed)
    device = torch.device("cpu")
    vocoder, sr = load_vocoder_24k(device)
    if sr != cfg["sample_rate"]:
        refuse("the vocoder is %d Hz and the config says %d" % (sr, cfg["sample_rate"]))
    meter_target = args.lufs

    def mel_of(wav):
        y = torch.from_numpy(np.asarray(wav, dtype="float32")).float().unsqueeze(0)
        return mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                               cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                               cfg["f_max"], center=False)

    def mel_distance(a, b):
        """Mean |mel| difference, in units of the reference's own std.

        The same normalisation the 2026-08-06 transparency figure used (mel L1 as a
        fraction of mel_std), so the number printed here is comparable to the 10.2% that
        has been quoted to rule the vocoder out.
        """
        ma, mb = mel_of(a), mel_of(b)
        n = min(ma.shape[-1], mb.shape[-1])
        ma, mb = ma[..., :n], mb[..., :n]
        return float((ma - mb).abs().mean() / ma.std())

    def roundtrip(wav):
        y = torch.from_numpy(wav).float().unsqueeze(0)
        mel = mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                              cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                              cfg["f_max"], center=False)
        with torch.no_grad():
            return to_waveform(mel, vocoder, None).numpy()

    prove_writable(args.out)
    clips = Path(args.out) / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    served, truth, key, limited = [], {}, {}, 0
    # ⚠ THE POSITIVE CONTROL MUST SIT WHERE THE ARTIFACT LIVES. The previous run let
    # rng.choice place it and all three landed at 166-212 Hz — so the control asking "can
    # you hear this distortion class?" was posed entirely in the register where the thing
    # under investigation does not occur, and its failure was uninterpretable rather than
    # informative.
    low_pool = [s for s in chosen if f0[s]["f0"] <= args.positive_max_f0]
    if len(low_pool) < 1:
        refuse("no chosen speaker is at or below --positive-max-f0 %.0f Hz, so the positive "
               "control would land in a register where the artifact is not expected — which "
               "is what made the previous run's control uninterpretable" % args.positive_max_f0)
    plan = ([(s, "vocoded") for s in chosen]
            + [(rng.choice(chosen), "identical") for _ in range(args.identical)]
            + [(rng.choice(low_pool), "positive") for _ in range(args.positive)])
    rng.shuffle(plan)
    swaps = [True] * (len(plan) // 2) + [False] * (len(plan) - len(plan) // 2)
    rng.shuffle(swaps)
    mel_l1 = {"vocoded": [], "positive": []}

    for i, (spk, kind) in enumerate(plan):
        same = kind == "identical"
        cand = [p for p in rows[spk] if sf.info(p).duration >= args.min_seconds]
        if not cand:
            refuse("speaker %d has no clip of at least %.1fs" % (spk, args.min_seconds))
        src = rng.choice(cand)
        x, xsr = sf.read(src, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if xsr != cfg["sample_rate"]:
            refuse("%s is %d Hz and the mel config is %d" % (src, xsr, cfg["sample_rate"]))

        orig, l1 = match_loudness(x, xsr, meter_target)
        if same:
            other, l2 = match_loudness(x, xsr, meter_target)
        else:
            depth = args.positive_depth if kind == "positive" else 1
            y = x
            for _ in range(depth):
                y = roundtrip(y)
            other, l2 = match_loudness(y, xsr, meter_target)
            mel_l1[kind].append(mel_distance(x, y))
        limited += int(l1) + int(l2)

        pair_key = "pair_%02d" % i
        # ⚠ EXACTLY HALF EACH WAY. Every non-identical pair is (original, altered), so an
        # unbalanced coin puts the altered condition on one side more often and a listener's
        # side preference lands on it. The sibling benches already balance; this one used
        # `rng.random() < 0.5` and with 12 pairs a 9-3 split had ~15% probability.
        swap = swaps[i]
        sides = [("A", other, "vocoded" if not same else "original"),
                 ("B", orig, "original")]
        if swap:
            sides = [("A", orig, "original"),
                     ("B", other, "vocoded" if not same else "original")]
        item = {"id": pair_key, "set": "roundtrip", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        # ⚠ THE CROP LENGTH IS COMPUTED ONCE, FOR THE PAIR. `min(len(audio), len(orig))`
        # inside the loop is per-SIDE, so when the round trip came out SHORTER than the
        # original only the original was truncated — to its own length, which is no
        # truncation at all — and the two served clips differed in duration. A length
        # difference is a cue a listener can use without knowing they are using it.
        n_pair = min(len(a) for _, a, _ in sides)
        for side_key, audio, label in sides:
            name = ear_bench.opaque(pair_key, side_key, args.salt)
            n = n_pair                              # the round trip can differ by a frame
            sf.write(str(clips / ("%s.wav" % name)), audio[:n], xsr, "PCM_24")
            key[name] = {"label": label, "pair": pair_key, "side": side_key}
            item[side_key] = name
        served.append(item)
        truth[pair_key] = {"spk": spk, "f0": f0[spk]["f0"], "kind": kind,
                           "identical": same, "source": src}
        print("  %s  spk%-5d  %5.1f Hz  %s" % (pair_key, spk, f0[spk]["f0"], kind))

    manifest = {"test": Path(args.out).name, "sample_rate": sr, "sets": SETS,
                "items": served}
    (Path(args.out) / "items.json").write_text(json.dumps(manifest, indent=2))
    key_out = args.key_out or str(Path(args.out).parent / "_keys" /
                                  ("%s.key.json" % Path(args.out).name))
    Path(key_out).parent.mkdir(parents=True, exist_ok=True)
    Path(key_out).write_text(json.dumps(
        {"test_dir": args.out, "salt": args.salt, "clips": key, "pairs": truth}, indent=2))

    # ⚠ The app runs as a different uid. Everything written above was written as root when
    # this runs in a container, so hand the tree to the shared group explicitly.
    for p in [Path(args.out), clips, Path(args.out) / "items.json"] + list(clips.iterdir()):
        try:
            os.chmod(p, 0o2775 if p.is_dir() else 0o664)
        except PermissionError:
            pass

    print("\n=== WHAT THE EAR IS BEING ASKED TO HEAR ===")
    print("   (mel L1 as a fraction of mel std — the same units as the 2026-08-06")
    print("    transparency figure of 10.2%, so these are directly comparable)")
    for kind, vals in mel_l1.items():
        if vals:
            print("   %-9s depth %d   mean %.1f%%   range %.1f-%.1f%%"
                  % (kind, args.positive_depth if kind == "positive" else 1,
                     100*np.mean(vals), 100*min(vals), 100*max(vals)))
    if mel_l1["vocoded"] and mel_l1["positive"]:
        ratio = np.mean(mel_l1["positive"]) / max(np.mean(mel_l1["vocoded"]), 1e-9)
        print("   the positive control is %.1fx the single-pass distortion" % ratio)
        if ratio < 1.5:
            print("   ⚠ THAT IS NOT MUCH LOUDER A SIGNAL. If the positive pairs tie, it may "
                  "mean the control was too subtle rather than that the listener is blind — "
                  "raise --positive-depth before concluding anything.")
    frac = args.identical / float(len(served))
    print("\nSTAGED %d pairs -> %s" % (len(served), args.out))
    print("   %d single-pass  %d positive (<= %.0f Hz)  %d CATCH TRIALS (%.0f%%)"
          % (len(chosen), args.positive, args.positive_max_f0, args.identical, 100*frac))
    if frac < 0.25:
        print("   ⚠ A catch-trial rate under 25%% cannot measure a false-alarm rate, and "
              "without one a subtle detection cannot be told from an expectation.")
    print("KEY %s" % key_out)
    if limited:
        print("⚠ %d sides hit the peak ceiling before reaching %.1f LUFS" % (limited, args.lufs))


if __name__ == "__main__":
    main()

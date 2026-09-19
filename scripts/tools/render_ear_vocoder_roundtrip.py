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

⚠ IDENTICAL-CLIP PAIRS ARE THE INSTRUMENT CHECK. Two of the pairs serve the same audio on
both sides. A listener who hears a difference there is not calibrated for the rest, and the
result should be thrown out rather than explained.

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

SETS = {
    "roundtrip": {
        "title": "Which one has the machine in it?",
        "ask": ("Two versions of the SAME recording of the same person. Ignore the voice, "
                "ignore the words. Listen for the robotic quality you have described — the "
                "buzz or hum under the voice, a sense of something reconstructed. Which "
                "side has more of it? Some of these pairs are genuinely identical and "
                "'no difference' is then the only right answer."),
        "labels": {"A": "A has more of the hum", "same": "No difference",
                   "B": "B has more of the hum"},
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


def prove_writable(out):
    """⚠ THE APP MUST BE ABLE TO WRITE BEFORE A LISTENER IS INVITED.

    A container render runs as root, so the test directory comes out root-owned; docker
    then creates a missing `verdicts/` as root:root and the app, running as another uid,
    cannot write to it. Every POST returns 500 — and the page repaints from its own memory,
    so it looks exactly like it is recording. That cost the owner ten minutes of listening
    on 2026-09-19 with nothing saved. Creating the directory is not enough; this writes a
    file and reads it back.
    """
    v = Path(out) / "verdicts"
    v.mkdir(parents=True, exist_ok=True)
    os.chmod(v, 0o2775)
    probe = v / ".writable"
    probe.write_text("probe")
    if probe.read_text() != "probe":
        refuse("%s did not read back what was written to it" % v)
    probe.unlink()
    # The app runs as a different uid than a container render. Group-writable plus setgid
    # is what makes the directory usable by both; it is asserted, not assumed.
    if not (os.stat(v).st_mode & 0o020):
        refuse("%s is not group-writable, so the ear-test app will not be able to save "
               "verdicts even though this process can" % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--f0-json", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--identical", type=int, default=2,
                    help="pairs serving the SAME audio on both sides (instrument check)")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="vocoder-roundtrip-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

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
    plan = [(s, False) for s in chosen] + [(rng.choice(chosen), True)
                                           for _ in range(args.identical)]
    rng.shuffle(plan)

    for i, (spk, same) in enumerate(plan):
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
        other, l2 = (match_loudness(x, xsr, meter_target) if same
                     else match_loudness(roundtrip(x), xsr, meter_target))
        limited += int(l1) + int(l2)

        pair_key = "pair_%02d" % i
        swap = rng.random() < 0.5
        sides = [("A", other, "vocoded" if not same else "original"),
                 ("B", orig, "original")]
        if swap:
            sides = [("A", orig, "original"),
                     ("B", other, "vocoded" if not same else "original")]
        item = {"id": pair_key, "set": "roundtrip", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, audio, label in sides:
            name = ear_bench.opaque(pair_key, side_key, args.salt)
            n = min(len(audio), len(orig))          # the round trip can differ by a frame
            sf.write(str(clips / ("%s.wav" % name)), audio[:n], xsr, "PCM_24")
            key[name] = {"label": label, "pair": pair_key, "side": side_key}
            item[side_key] = name
        served.append(item)
        truth[pair_key] = {"spk": spk, "f0": f0[spk]["f0"], "identical": same,
                           "source": src}
        print("  %s  spk%-5d  %5.1f Hz  %s" % (pair_key, spk, f0[spk]["f0"],
                                               "IDENTICAL (control)" if same else "vs vocoded"))

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

    print("\nSTAGED %d pairs (%d identical controls) -> %s" % (len(served), args.identical, args.out))
    print("KEY %s" % key_out)
    if limited:
        print("⚠ %d sides hit the peak ceiling before reaching %.1f LUFS" % (limited, args.lufs))


if __name__ == "__main__":
    main()

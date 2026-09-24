"""Choose the clips the vocoder is fine-tuned on, and the clips it is checked on.

THE EXPERIMENT
--------------
The ceiling bench (2026-09-22) rated the model's renders at 2.12 on the 0-5 hum scale, the
mel/vocoder round trip of a real recording at 0.38, and the recording itself at 0.06. The
vocoder was fine-tuned on mels taken from REAL audio only, so every mel it has seen is one
the acoustic model never makes. Fine-tuning it on the model's own mels, paired with the real
audio they were aligned to, asks: how much of the +1.75 is the vocoder's response to an
off-distribution mel?

⚠⚠ THE CEILING BENCH'S SPEAKERS ARE HELD OUT, WHOLE. The re-test serves the same items, and
a vocoder that has seen those voices' predicted mels could learn them rather than the
defect. Their clips appear in neither list. The key is read for the speaker ids only.

⚠⚠ ONLY REAL RECORDINGS ARE TRAINING TARGETS. The fine-tune teaches the vocoder to turn a
predicted mel into the audio beside it, so the audio beside it IS the definition of correct.
A synthetic clip carries the teacher engine's own artifacts — the DOUBLE/split defect is one
— and the vocoder would learn to put them back. The source roots are an ALLOWLIST plus a
DENYLIST, and a row whose root is on neither REFUSES: a new corpus source must be classified
by a person, not defaulted into the training target.

⚠ THE VALIDATION CLIPS COME FROM TRAINING SPEAKERS — one extra clip each from --val-clips
of them. hifi-gan's validation loss therefore measures new sentences in known voices, not
new voices; the held-out voices are the ear bench's job.

⚠ PCM_16 ONLY. hifi-gan's `load_wav` divides whatever scipy returns by 32768, so a 24-bit or
float file would train at the wrong scale with no error. The header is read anyway.

⚠ BALANCED PER SPEAKER. The corpus has 5385 speakers and very unequal row counts; a uniform
draw over rows would fit the vocoder to the few voices with the most clips. Up to
--per-speaker rows each, drawn with a fixed seed.

Writes, into --out:
    ft_train_op.txt, ft_val_op.txt    corpus rows, verbatim, for render_aligned_mels.py
    hifigan_train.txt, hifigan_val.txt  the same clips as hifi-gan reads them
    manifest.json                     what was chosen, excluded and why

Usage:
    python scripts/tools/build_vocoder_ft_filelist.py \\
        --corpus data/libritts_r_full_vat_v7/train_op.txt \\
        --exclude-key /data/model-training/sonora/eartest/_keys/ceiling.key.json \\
        --out /data/model-training/vocoder/vocoder_ft/vat7_ep005
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf

REAL_ROOTS = (
    "/data/model-training/datasets/LibriTTS_R/",
    "/data/model-training/datasets/emilia_kept_24k/",
)
SYNTHETIC_ROOTS = (
    # Teacher-engine renders (Dia, and the rest of the directed-teacher portfolio).
    "/data/model-training/datasets/expressive_registers_24k/",
)


def source_of(wav):
    for r in REAL_ROOTS:
        if wav.startswith(r):
            return "real"
    for r in SYNTHETIC_ROOTS:
        if wav.startswith(r):
            return "synthetic"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="the acoustic model's train_op.txt")
    ap.add_argument("--exclude-key", action="append", default=[],
                    help="an ear-test key whose items' speakers are held out; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-speaker", type=int, default=4)
    ap.add_argument("--val-clips", type=int, default=200)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--min-seconds", type=float, default=1.0,
                    help="hifi-gan cuts 8192-sample segments; anything shorter is padding")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    held_out = set()
    for k in args.exclude_key:
        items = json.loads(Path(k).read_text()).get("items") or {}
        if not items:
            raise SystemExit("REFUSING: %s carries no `items`, so no speaker can be read "
                             "from it and the hold-out would be silently empty." % k)
        held_out |= {int(v["spk"]) for v in items.values()}

    rows_by_spk, unknown, synthetic = defaultdict(list), set(), 0
    for line in Path(args.corpus).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wav, spk = line.split("|")[0], int(line.split("|")[1])
        src = source_of(wav)
        if src is None:
            unknown.add(wav.rsplit("/", 2)[0])
            continue
        if src == "synthetic":
            synthetic += 1
            continue
        if spk in held_out:
            continue
        rows_by_spk[spk].append(line)
    if unknown:
        raise SystemExit("REFUSING: %d source root(s) are neither real nor synthetic here, "
                         "e.g. %s. Classify them in REAL_ROOTS or SYNTHETIC_ROOTS."
                         % (len(unknown), sorted(unknown)[:3]))

    rng = random.Random(args.seed)
    train, val, short, missing = [], [], 0, 0
    spks = sorted(rows_by_spk)
    val_spks = set(rng.sample(spks, min(args.val_clips, len(spks))))
    for s in spks:
        cand = list(rows_by_spk[s])
        rng.shuffle(cand)
        want = args.per_speaker + (1 if s in val_spks else 0)
        kept = []
        for line in cand:
            wav = line.split("|")[0]
            try:
                info = sf.info(wav)
            except RuntimeError:
                missing += 1
                continue
            if info.samplerate != args.sample_rate:
                raise SystemExit("REFUSING: %s is %d Hz, not %d."
                                 % (wav, info.samplerate, args.sample_rate))
            if info.subtype != "PCM_16":
                raise SystemExit("REFUSING: %s is %s. hifi-gan scales every file by 1/32768, "
                                 "so only PCM_16 reads at the right level."
                                 % (wav, info.subtype))
            if info.frames < args.min_seconds * args.sample_rate:
                short += 1
                continue
            kept.append(line)
            if len(kept) == want:
                break
        if s in val_spks and kept:
            val.append(kept.pop())
        train.extend(kept)

    stems = [Path(line.split("|")[0]).stem for line in train + val]
    if len(set(stems)) != len(stems):
        raise SystemExit("REFUSING: two chosen clips share a file stem. hifi-gan finds a "
                         "clip's mel by stem alone, so one would train on the other's mel.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (out / ("ft_%s_op.txt" % name)).write_text("\n".join(rows) + "\n", encoding="utf-8")
        # ⚠ hifi-gan appends `.wav` and joins onto --input_wavs_dir. The paths are absolute,
        # so os.path.join returns them unchanged whatever that directory is.
        (out / ("hifigan_%s.txt" % name)).write_text(
            "\n".join(r.split("|")[0][:-len(".wav")] for r in rows) + "\n", encoding="utf-8")
    manifest = {
        "corpus": args.corpus, "seed": args.seed, "per_speaker": args.per_speaker,
        "train_clips": len(train), "val_clips": len(val),
        "train_speakers": len({r.split("|")[1] for r in train}),
        "held_out_speakers": sorted(held_out), "held_out_from": args.exclude_key,
        "synthetic_rows_excluded": synthetic, "short_skipped": short,
        "unreadable_skipped": missing, "real_roots": REAL_ROOTS,
        "synthetic_roots": SYNTHETIC_ROOTS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("train %d clips over %d speakers, val %d; held out %d speakers; %d synthetic rows "
          "excluded; %d short, %d unreadable skipped"
          % (len(train), manifest["train_speakers"], len(val), len(held_out), synthetic,
             short, missing))


if __name__ == "__main__":
    main()

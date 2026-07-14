"""VAT corpus derivation over LibriTTS-R (dataset-landscape.md §Strategy).

Walks a local LibriTTS-R subset, measures per-clip features, derives V/A/T
labels, phonemizes transcripts through the espeak-free op-G2P lane, remaps
speakers to contiguous ids, and emits `path|spk|ipa|v,a,t` filelists (the
`load_vat` datamodule format) plus a speaker map and a derivation report.

Label derivation v0 (de-risk configuration, north star §7):
    A (arousal slot) = per-speaker z-score of integrated loudness (LUFS),
        clamped to [-1, 1] at 2 sigma. Per-speaker normalization is load-
        bearing: global normalization would bake mic gain / speaker identity
        into the energy label — leakage by construction.
    V = T = 0 (the de-risk experiment trains exactly one channel).
The full V/A/T derivation (Parler annotations + aligned prosody measures)
extends this script; keep every formula documented here and versioned.

License wall: emitted audio paths carry the LibriTTS_R component and pass
the manifest as `libritts_r` (permissive). Run from the Sonora repo root:
    python scripts/derive_vat_corpus.py [--root DIR] [--out DIR] [--workers N]
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULT_ROOT = "/data/model-training/datasets/LibriTTS_R/train-clean-100"
MIN_SECONDS = 1.0
MAX_SECONDS = 16.0
SAMPLE_RATE = 24000
VAL_FRACTION = 0.03
SEED = 1234


def find_clips(root):
    """Yields (wav_path, normalized_text, speaker_id) for every utterance."""
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chapter in sorted(os.listdir(spk_dir)):
            ch_dir = os.path.join(spk_dir, chapter)
            if not os.path.isdir(ch_dir):
                continue
            for name in sorted(os.listdir(ch_dir)):
                if not name.endswith(".wav"):
                    continue
                txt = os.path.join(ch_dir, name[:-4] + ".normalized.txt")
                if not os.path.exists(txt):
                    continue
                with open(txt, encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    yield os.path.join(ch_dir, name), text, spk


def measure_clip(args):
    """Worker: (wav_path,) -> dict of acoustic measures, or None to skip."""
    wav_path = args
    import numpy as np
    import soundfile as sf

    try:
        wav, sr = sf.read(wav_path, dtype="float32")
    except Exception:
        return wav_path, None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    seconds = len(wav) / sr
    if sr != SAMPLE_RATE or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return wav_path, None
    try:
        import pyloudnorm

        lufs = float(pyloudnorm.Meter(sr).integrated_loudness(wav))
    except Exception:
        rms = float(np.sqrt(np.mean(wav**2)))
        lufs = 20 * float(np.log10(max(rms, 1e-9)))
    if not np.isfinite(lufs) or lufs < -60:
        return wav_path, None  # silence / broken clip
    return wav_path, {"seconds": seconds, "lufs": lufs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="data/libritts_r_vat")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    ap.add_argument("--no-neural-oov", action="store_true")
    args = ap.parse_args()

    clips = list(find_clips(args.root))
    print(f"found {len(clips)} utterances with transcripts under {args.root}")

    with mp.Pool(args.workers) as pool:
        measured = dict(pool.imap_unordered(measure_clip, [c[0] for c in clips], chunksize=64))
    kept = [(p, t, s) for p, t, s in clips if measured.get(p)]
    print(f"kept {len(kept)} after duration/rate/loudness filters "
          f"({len(clips) - len(kept)} dropped)")

    # Per-speaker loudness z-score -> arousal slot, clamped at 2 sigma.
    import numpy as np

    by_spk = {}
    for p, _, s in kept:
        by_spk.setdefault(s, []).append(measured[p]["lufs"])
    spk_stats = {s: (float(np.mean(v)), float(np.std(v) + 1e-6)) for s, v in by_spk.items()}
    speakers = sorted(by_spk.keys(), key=int)
    spk_index = {s: i for i, s in enumerate(speakers)}

    # Phonemize (espeak-free lane) and assemble rows.
    from matcha.text.op_g2p import OpenPhonemizerG2P
    from matcha.data.license_wall import enforce as license_check  # noqa: F401

    g2p = OpenPhonemizerG2P(use_neural_oov=not args.no_neural_oov)
    rows, bad_vocab = [], 0
    for i, (p, text, s) in enumerate(kept):
        ipa = g2p.phonemize(text)
        if g2p.validate(ipa):
            bad_vocab += 1
            continue
        mean, std = spk_stats[s]
        a = max(-1.0, min(1.0, (measured[p]["lufs"] - mean) / (2 * std)))
        rows.append(f"{p}|{spk_index[s]}|{ipa}|0.0,{a:.4f},0.0")
        if (i + 1) % 5000 == 0:
            print(f"  phonemized {i + 1}/{len(kept)}")
    print(f"phonemized {len(rows)} rows ({bad_vocab} dropped for vocab violations)")
    s_ = g2p.stats
    total = s_["dict_hits"] + s_["neural_hits"] + s_["oov_misses"]
    print(f"G2P: dict {100 * s_['dict_hits'] / max(total, 1):.2f}% | "
          f"neural {s_['neural_hits']} | unresolved {s_['oov_misses']}")

    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(int(len(rows) * VAL_FRACTION), 1)
    os.makedirs(args.out, exist_ok=True)
    for name, part in (("val_op.txt", rows[:n_val]), ("train_op.txt", rows[n_val:])):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        print(f"wrote {len(part)} rows -> {os.path.join(args.out, name)}")

    with open(os.path.join(args.out, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({"n_spks": len(speakers), "libritts_id_to_index": spk_index}, f, indent=2)
    report = {
        "root": args.root,
        "derivation": "v0-derisk: A=per-speaker LUFS z-score clamped [-1,1]@2sigma; V=T=0",
        "utterances_found": len(clips),
        "kept": len(rows),
        "n_spks": len(speakers),
        "seconds_total": float(sum(measured[p]["seconds"] for p, _, _ in kept)),
        "filters": {"min_s": MIN_SECONDS, "max_s": MAX_SECONDS, "sample_rate": SAMPLE_RATE},
        "license": "LibriTTS-R CC-BY-4.0 (see configs/data_licenses.yaml)",
    }
    with open(os.path.join(args.out, "derivation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"hours: {report['seconds_total'] / 3600:.1f} | speakers: {len(speakers)}")
    print("report ->", os.path.join(args.out, "derivation_report.json"))


if __name__ == "__main__":
    main()

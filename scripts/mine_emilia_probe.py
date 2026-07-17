"""Emilia-YODAS mining probe (emilia-mining-plan.md step 1).

Samples N utterances per extracted tar dir, applies the base filters
(duration, DNSMOS floor), computes the standing acoustic measures (LUFS +
phonation composite via derive_vat_corpus) at 24 kHz, and writes
probe_measures.jsonl plus a plain filelist of the sampled audio for
scripts/eiv_score.py. Analysis (tail richness vs the LibriTTS-R reference
distribution) happens downstream — this script only measures.

Usage:
    python scripts/mine_emilia_probe.py --dirs <extracted tar dirs> \
        --out <probe out dir> [--sample 5000] [--dnsmos-min 3.0]
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET_SR = 24000


def measure_one(args):
    path, meta = args
    import librosa
    import numpy as np
    from derive_vat_corpus import phonation_measures

    try:
        wav, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if len(wav) < TARGET_SR:
        return None
    try:
        import pyloudnorm
        lufs = float(pyloudnorm.Meter(TARGET_SR).integrated_loudness(wav))
    except Exception:
        lufs = 20 * float(np.log10(max(float(np.sqrt((wav**2).mean())), 1e-9)))
    if not np.isfinite(lufs) or lufs < -60:
        return None
    phon = phonation_measures(wav.astype(np.float32), TARGET_SR)
    if phon is None:
        return None
    return {"wav": path, "lufs": lufs, **phon,
            "duration": meta.get("duration"), "dnsmos": meta.get("dnsmos"),
            "speaker": meta.get("speaker"), "text": meta.get("text", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=5000, help="per dir")
    ap.add_argument("--dnsmos-min", type=float, default=3.0)
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=16.0)
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    jobs = []
    for d in args.dirs:
        cands = []
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            meta = json.load(open(os.path.join(d, name), encoding="utf-8"))
            if not (args.min_s <= (meta.get("duration") or 0) <= args.max_s):
                continue
            if (meta.get("dnsmos") or 0) < args.dnsmos_min:
                continue
            audio = os.path.join(d, name[:-5] + ".mp3")
            if os.path.exists(audio):
                cands.append((audio, meta))
        picked = random.sample(cands, min(args.sample, len(cands)))
        jobs += picked
        print(f"{d}: {len(cands)} pass filters, sampled {len(picked)}")

    with mp.Pool(args.workers) as pool:
        results = [r for r in pool.imap_unordered(measure_one, jobs, chunksize=16) if r]
    print(f"measured {len(results)}/{len(jobs)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probe_measures.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "probe_filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(r["wav"] for r in results) + "\n")
    print(f"-> {args.out}/probe_measures.jsonl + probe_filelist.txt")


if __name__ == "__main__":
    main()

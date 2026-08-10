"""Acoustic pass for the v6 append set — the A and T half of rung 2's labels.

The 12-head EIV pass (`eiv_score.py`) produces **V only**. Per
`anchor_emilia_labels.py:35-36`, the other two channels are acoustic:

    A = clamp2( zg(lufs) )
    T = clamp2( zg( zg(alpha_db) + zg(cpp) - zg(h1h2) - zg(soft) ) )

Only `soft` (Soft_vs._Harsh) comes from an EIV head; LUFS, alpha ratio, CPP and
H1-H2 come from an acoustic measurement pass. Two such passes exist in this repo
and NEITHER covers the expressive-registers bank:

  * `derive_vat_corpus.py` walks a LibriTTS-R subset via `--root` (speaker/chapter
    dirs + `*.normalized.txt`), which this bank is not shaped like; and
  * `mine_emilia_probe.py` walks extracted Emilia tars (`*.json` sidecars + `.mp3`),
    which this bank is also not shaped like.

So this script is the third producer, for the third corpus shape: a flat bank keyed
by `ratings.csv`. It writes the SAME row shape as `mine_emilia_probe.py`, because
`anchor_emilia_labels.emilia_rows()` is the consumer whose contract we have to meet.

`derive_vat_corpus.py:536-546` hard-aborts with "label sources do not cover every
kept clip" rather than writing zeros, so a gap here fails the v6 build at derive
time rather than quietly labelling a clip neutral. That is the behaviour we want;
this script exists to close the gap before it fires.

MEASUREMENT REUSE, NOT REIMPLEMENTATION. `phonation_measures` is imported from
`derive_vat_corpus` — the same function that measured every LibriTTS-R clip in the
v1-v5 lineage and (via `mine_emilia_probe`) every Emilia keep. A second
implementation of the tension composite would put this half of the corpus on a
silently different scale, which is the one failure that would not announce itself.

TWO THINGS THIS BANK DOES THAT THE OTHER TWO SOURCES DO NOT:

  1. **Mixed sample rates.** 148 of the 846 clips are 44.1 kHz, the rest 24 kHz.
     `derive_vat_corpus.measure_clip` REJECTS `sr != 24000` outright, so reusing it
     directly would have dropped all 148 without comment. We resample to 24 kHz the
     way `mine_emilia_probe` does (`librosa.load(sr=24000)`) and record `native_sr`
     on every row so the resampled ones stay identifiable. Emilia was measured after
     the same resample, so this puts the two appended halves on the same footing.
  2. **Clips longer than the corpus filter.** 14 clips exceed
     `derive_vat_corpus.MAX_SECONDS` (22.0 s), up to 64 s. This script MEASURES them
     and flags them (`over_max_seconds: true`) rather than dropping them, because
     whether to drop or to raise the filter is a corpus decision, not a measurement
     one, and a silent drop here would change the append count without a record.

Read-only on `ratings.csv` (the Auditions app is a live writer): its mtime_ns is
recorded in the manifest so a concurrent edit is visible after the fact.

Run:  .venv/bin/python scripts/measure_expressive_registers.py \
          --ids <v6_append_set.json> --out /data/model-training/sonora/expressive_registers_measures
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BANK = "/data/model-training/datasets/sonora-expressive-registers"
TARGET_SR = 24000


def measure_one(job):
    """(id, wav_path, meta) -> row dict, or (id, None, reason) when unmeasurable."""
    clip_id, path, meta = job
    import librosa
    import numpy as np

    from derive_vat_corpus import phonation_measures

    try:
        wav, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception as e:  # noqa: BLE001 — the reason is the useful part
        return clip_id, None, f"unreadable: {type(e).__name__}"
    seconds = len(wav) / TARGET_SR
    if len(wav) < TARGET_SR:
        return clip_id, None, f"under 1.0 s ({seconds:.2f})"

    try:
        import pyloudnorm

        lufs = float(pyloudnorm.Meter(TARGET_SR).integrated_loudness(wav))
    except Exception:
        rms = float(np.sqrt(np.mean(wav**2)))
        lufs = 20 * float(np.log10(max(rms, 1e-9)))
    if not np.isfinite(lufs) or lufs < -60:
        return clip_id, None, f"silent / broken (lufs {lufs:.1f})"

    phon = phonation_measures(wav.astype(np.float32), TARGET_SR)
    if phon is None:
        # Fewer than 10 usable voiced frames — T is not derivable, so a row here
        # would be a guess wearing a measurement's clothes.
        return clip_id, None, "too few voiced frames for the tension composite"

    return clip_id, {
        "wav": path,
        "id": clip_id,
        "lufs": lufs,
        **phon,
        "duration": seconds,
        "native_sr": meta["native_sr"],
        "resampled": meta["native_sr"] != TARGET_SR,
        "over_max_seconds": meta["over_max_seconds"],
        "engine": meta["engine"],
        "delivery": meta["delivery"],
        "register": meta["register"],
        "campaign": meta["campaign"],
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True,
                    help="JSON with {'v6_append_ids': [...]}, or a plain JSON list of ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default=BANK)
    ap.add_argument("--ratings", default=None, help="defaults to <bank>/ratings.csv")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    args = ap.parse_args()

    import soundfile as sf

    from derive_vat_corpus import MAX_SECONDS

    ratings = args.ratings or os.path.join(args.bank, "ratings.csv")
    ratings_mtime = os.stat(ratings).st_mtime_ns
    rows = {r["id"]: r for r in csv.DictReader(open(ratings, encoding="utf-8"))}

    raw = json.load(open(args.ids, encoding="utf-8"))
    ids = raw["v6_append_ids"] if isinstance(raw, dict) else raw
    print(f"append set: {len(ids)} ids · ratings.csv mtime_ns {ratings_mtime}")

    unknown = [i for i in ids if i not in rows]
    if unknown:
        sys.exit(f"FATAL: {len(unknown)} id(s) not in ratings.csv, e.g. {unknown[:3]}")

    jobs, absent = [], []
    for clip_id in ids:
        r = rows[clip_id]
        path = os.path.normpath(os.path.join(args.bank, r["link"]))
        if not os.path.exists(path):
            absent.append(clip_id)
            continue
        info = sf.info(path)
        seconds = info.frames / info.samplerate
        jobs.append((clip_id, path, {
            "native_sr": info.samplerate,
            "over_max_seconds": seconds > MAX_SECONDS,
            "engine": r["engine"],
            "delivery": r["delivery"],
            "register": r["register"],
            "campaign": r["campaign"],
        }))
    if absent:
        sys.exit(f"FATAL: {len(absent)} clip(s) missing on disk, e.g. {absent[:3]}")

    with mp.Pool(args.workers) as pool:
        results = list(pool.imap_unordered(measure_one, jobs, chunksize=8))

    measured = [row for _, row, _ in results if row]
    failed = [(cid, why) for cid, row, why in results if row is None]
    measured.sort(key=lambda r: r["id"])

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "probe_measures.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in measured:
            f.write(json.dumps(row) + "\n")

    resampled = sum(1 for r in measured if r["resampled"])
    over = [r["id"] for r in measured if r["over_max_seconds"]]
    manifest = {
        "source_bank": args.bank,
        "ratings_csv_mtime_ns": ratings_mtime,
        "ids_file": os.path.abspath(args.ids),
        "requested": len(ids),
        "measured": len(measured),
        "failed": [{"id": c, "reason": w} for c, w in failed],
        "resampled_to_24k": resampled,
        "over_max_seconds": {"max_seconds": MAX_SECONDS, "n": len(over), "ids": over},
        "measure_source": "derive_vat_corpus.phonation_measures + pyloudnorm (shared "
                          "with the LibriTTS-R lineage and mine_emilia_probe)",
    }
    with open(os.path.join(args.out, "measures_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nmeasured {len(measured)}/{len(ids)}")
    print(f"  resampled from a non-24k native rate: {resampled}")
    print(f"  over MAX_SECONDS={MAX_SECONDS}: {len(over)}  (measured and flagged, NOT dropped)")
    if failed:
        print(f"  UNMEASURABLE: {len(failed)} — these have no A/T and will fail the "
              f"derive-time coverage check:")
        for cid, why in failed:
            print(f"    {cid}: {why}")
    print(f"\n-> {out_path}")
    print(f"-> {os.path.join(args.out, 'measures_manifest.json')}")


if __name__ == "__main__":
    main()

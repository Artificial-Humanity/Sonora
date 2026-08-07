"""Build a REAL-SPEECH reference pool from LibriTTS-R, and merge the synthetic one in.

`ref_select`'s docstring has named this the upgrade path since it was written —
"Real-speech pools + measured age norms are the upgrade path; swap POOL_PATH when
they land." They have landed.

WHY, measured 2026-08-02. The v1 pool is 193 clips and every one is SYNTHETIC:
moss85 49, qwen 84, dia 6, longcat 54. Three of those four engines are retired —
dia set aside, longcat excluded pending affect-transfer, moss85 superseded by
moss_vg — so we clone voices for the current five engines from clips produced by
engines we no longer use. Of the 16 narration-register references, only 4 come
from an engine still in the portfolio. The direction-relay audit flagged the
100%-synthetic pool in July; the narration lanes running out of male references
mid-build on 2026-08-02 is the same defect arriving as a crash.

LibriTTS-R answers every requirement: 243 speakers with a clip in the 4-10 s
reference window `select_reference` enforces, **120 F / 123 M** against the v1
pool's 6 F / 10 M in narration, gender free from `speakers.tsv`, CC-BY-4.0, clean
24 kHz, already on disk and already labelled by our own VAT derivation.

SELECTION, per speaker, one clip:
  * duration inside the hard 4-10 s window (owner floor 2026-07-25)
  * VAT closest to neutral — a narration reference should be a steady read, and
    the derived per-speaker z-scores say which clips are
  * LOW pitch excursion. This is the measured Chatterbox split driver
    (`MAX_REF_EXCURSION=240`): swooping references break cloning engines, steady
    ones do not, and median F0 predicts nothing. Choosing on it up front means the
    pool is safe by construction rather than filtered at selection time.

The synthetic pool is MERGED, not replaced: its expressive registers (arrogance,
grief, threat…) have no real-audio equivalent yet, and dropping them would break
expressive casting to fix narration casting.

    .venv/bin/python scripts/synthesis/build_reference_pool.py            # report
    .venv/bin/python scripts/synthesis/build_reference_pool.py --apply
"""
import argparse
import collections
import csv
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_select import MAX_REF_EXCURSION as _REF_EXCURSION  # noqa: E402

CORPUS = pathlib.Path("data/libritts_r_vat_v3c")
LIBRITTS = pathlib.Path("/data/model-training/datasets/LibriTTS_R/train-clean-100")
V1_POOL = pathlib.Path(
    "/data/model-training/datasets/sonora-expressive-registers/v1")
OUT = pathlib.Path("/data/model-training/datasets/reference-pool-v2")

MIN_S, MAX_S = 4.0, 10.0          # select_reference's hard window
MAX_EXCURSION = _REF_EXCURSION                 # B-L5: one definition
CANDIDATES_PER_SPEAKER = 6        # F0 is the expensive step; rank cheaply first


def speaker_gender():
    path = LIBRITTS / "speakers.tsv"
    with open(path, encoding="utf-8") as f:
        return {r["READER"]: r["GENDER"]
                for r in csv.DictReader(f, delimiter="\t") if r.get("READER")}


def corpus_rows():
    """wav -> (speaker, v, a, t) from the derived corpus filelists."""
    rows = {}
    for part in ("train_op.txt", "val_op.txt"):
        for line in open(CORPUS / part, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            wav, _spk, _ipa, vat = line.split("|")
            v, a, t = (float(x) for x in vat.split(","))
            rows[wav] = (os.path.basename(wav).split("_")[0], v, a, t)
    return rows


def durations():
    return {json.loads(l)["wav"]: json.loads(l)["seconds"]
            for l in open(CORPUS / "measures.jsonl", encoding="utf-8")}


def clip_text(wav):
    p = pathlib.Path(wav).with_suffix("").as_posix() + ".normalized.txt"
    try:
        return open(p, encoding="utf-8").read().strip()
    except OSError:
        return ""


def measure_f0(path):
    """(median, p10, p90) F0 in Hz over voiced frames, or None."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=None, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=2048, hop_length=256)
    vals = f0[voiced & np.isfinite(f0)] if f0 is not None else None
    if vals is None or len(vals) < 20:
        return None
    return (float(np.median(vals)), float(np.percentile(vals, 10)),
            float(np.percentile(vals, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit-speakers", type=int)
    args = ap.parse_args()

    genders = speaker_gender()
    rows, durs = corpus_rows(), durations()

    # Rank cheaply (neutrality + duration), then measure F0 on the short list only.
    by_speaker = collections.defaultdict(list)
    for wav, (spk, v, a, t) in rows.items():
        d = durs.get(wav)
        if d is None or not (MIN_S <= d <= MAX_S):
            continue
        neutrality = abs(v) + abs(a) + abs(t)
        by_speaker[spk].append((neutrality, wav, d, (v, a, t)))
    for cands in by_speaker.values():
        cands.sort()

    speakers = sorted(by_speaker)
    if args.limit_speakers:
        speakers = speakers[:args.limit_speakers]
    print(f"{len(speakers)} speakers with a clip in the {MIN_S}-{MAX_S}s window; "
          f"measuring F0 on up to {CANDIDATES_PER_SPEAKER} candidates each", flush=True)

    chosen, acoustics, skipped = [], {}, collections.Counter()
    for i, spk in enumerate(speakers):
        g = genders.get(spk)
        if g not in ("M", "F"):
            skipped["no gender in speakers.tsv"] += 1
            continue
        pick = None
        for neutrality, wav, d, vat in by_speaker[spk][:CANDIDATES_PER_SPEAKER]:
            m = measure_f0(wav)
            if m is None:
                continue
            med, p10, p90 = m
            exc = p90 - p10
            # Steady beats neutral: a swooping reference breaks cloning engines,
            # and every candidate here is already among the speaker's calmest.
            if pick is None or exc < pick[0]:
                pick = (exc, wav, d, vat, med, p10, p90)
            if exc < 60:            # comfortably steady; stop paying for more
                break
        if pick is None:
            skipped["F0 unmeasurable on every candidate"] += 1
            continue
        exc, wav, d, vat, med, p10, p90 = pick
        if exc >= MAX_EXCURSION:
            skipped[f"excursion >= {MAX_EXCURSION}"] += 1
            continue
        cid = f"libritts_{spk}_{pathlib.Path(wav).stem}"
        rel = f"audio/{cid}.wav"
        chosen.append({
            "file": rel, "id": cid, "engine": "libritts-r",
            "register": "neutral_narration",
            "text": clip_text(wav),
            "intended_vat": {"V": round(vat[0], 4), "A": round(vat[1], 4),
                             "T": round(vat[2], 4)},
            "duration": round(d, 2),
            "gender": "Male" if g == "M" else "Female",
            "speaker": spk,
            "license": "CC-BY-4.0",
            "text_source": "LibriTTS-R (LibriVox, public domain text)",
            "source_wav": wav,
            "campaign": "reference-pool-v2",
            # Real speech: there is no intended-vs-measured gap to record, and no
            # engine that could have failed to follow direction.
            "owner_relabel": False,
        })
        acoustics[rel] = {"f0_median": med, "f0_p10": p10, "f0_p90": p90,
                          "f0_excursion": exc}
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(speakers)} speakers, {len(chosen)} kept", flush=True)

    g_count = collections.Counter(c["gender"] for c in chosen)
    excs = sorted(a["f0_excursion"] for a in acoustics.values())
    print(f"\nreal-speech references: {len(chosen)}  {dict(g_count)}")
    if excs:
        print(f"  excursion  min {excs[0]:.0f}  median {excs[len(excs)//2]:.0f}  "
              f"max {excs[-1]:.0f} Hz  (ceiling {MAX_EXCURSION:.0f})")
    for k, v in skipped.items():
        print(f"  !! {v} x {k}")

    # --- merge the synthetic pool ------------------------------------------
    v1 = [json.loads(l) for l in open(V1_POOL / "metadata.jsonl", encoding="utf-8")]
    v1_ac = json.loads((V1_POOL / "pool_acoustics.json").read_text())
    print(f"\nsynthetic pool merged in: {len(v1)} clips "
          f"({len(set(c['register'] for c in v1))} registers) — kept because its "
          f"expressive registers have no real-audio equivalent yet")
    total = len(chosen) + len(v1)
    narr = sum(1 for c in chosen + v1 if c.get("register") == "neutral_narration")
    print(f"merged pool: {total} clips, {narr} in neutral_narration "
          f"(was 16 — a {narr / 16:.0f}x increase)")

    if not args.apply:
        print("\nreport only — pass --apply to write")
        return 0

    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    # Symlinks: the real clips live in LibriTTS-R and the synthetic ones in a
    # PUBLISHED tier directory. Copying would duplicate ~100 MB and, worse, fork
    # the published audio — a pool is an index, not a new artifact.
    for c in chosen:
        link = OUT / c["file"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pathlib.Path(c["source_wav"]).resolve())
    for c in v1:
        link = OUT / c["file"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((V1_POOL / c["file"]).resolve())

    with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
        for c in chosen + v1:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    merged_ac = dict(v1_ac)
    merged_ac.update(acoustics)
    (OUT / "pool_acoustics.json").write_text(json.dumps(merged_ac, indent=1))
    (OUT / "PROVENANCE.md").write_text(
        "# reference-pool-v2\n\n"
        f"Built {len(chosen)} real-speech references from LibriTTS-R "
        "train-clean-100 (CC-BY-4.0), one per speaker, merged with the "
        f"{len(v1)} synthetic clips of the v1 expressive-registers tier.\n\n"
        "Audio is SYMLINKED, never copied — this directory is an index over "
        "audio that lives elsewhere. The v1 tier is published; forking its wavs "
        "here would create a second copy that could drift.\n\n"
        "Selection per speaker: duration in the 4-10 s window `select_reference` "
        "enforces, VAT nearest neutral among that speaker's clips, then the LOWEST "
        "pitch excursion among the calmest candidates — the measured Chatterbox "
        "split driver, so the pool is steady by construction rather than filtered "
        "at selection time.\n\n"
        "Gender is from LibriTTS-R `speakers.tsv`. Age is not labelled and is "
        "inferred by ref_select from within-gender F0 percentile, as before.\n")
    print(f"\nwrote {total} entries -> {OUT / 'metadata.jsonl'}")
    print("Switch ref_select.POOL_PATH to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

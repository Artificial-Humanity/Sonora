# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "pyloudnorm"]
# ///
"""normalize_loudness — bring every clip in a bank to one integrated loudness.

WHY THIS EXISTS (measured 2026-07-28). Across the 193 clips in v1 the per-engine
median RMS spread was **5.1 dB** — dia -17.0, moss85 -20.1, longcat -21.3,
qwen -22.1 dBFS. That is not a cosmetic problem:

  1. It biases the audition. Louder reads as more present and more confident, so
     the ear is being told which engine it is before it judges the performance.
  2. It makes the reference pool inconsistent. v1 is what `ref_select.py` draws
     cloning references from, and VibeVoice/LongCat/Chatterbox/Zonos all condition
     on a reference clip — so a 5 dB spread across references is a 5 dB spread in
     the conditioning signal.

CORRECTION (2026-07-28): an earlier version of this file claimed loudness also
contaminated the measured arousal label. **That was wrong, and it was asserted
without checking.** For the synth lane, `qc_verdict.raw_measures` takes A from the
EIV Arousal head, and EIV is EmoWhisper — a `WhisperFeatureExtractor` front end,
whose log-mel is normalised against each clip's OWN max (`max(log_spec,
log_spec.max() - 8.0)`). A constant gain cancels there, so EIV is level-invariant
and `measured_z` survives normalisation unchanged. The large per-engine A and T
offsets that `qc_verdict --per-engine` exists to correct are real, but they come
from how the engines actually deliver, not from their output gain.

⚠ DO NOT RUN THIS ON THE LibriTTS-R CORPUS LANE. `derive_vat_corpus.py` derives
A as the **per-speaker z-score of integrated loudness** — there, level IS the
arousal label. Normalising every clip to one LUFS would drive that variance to
zero and annihilate the channel. This script is for TEACHER SYNTHESIS banks only,
where A comes from EIV. The two lanes disagree about what loudness means, and
that is a real trap, not a stylistic difference.

So this runs at BUILD time: after render, before audition. Both the ear and the
reference pool then see one level.

Target is **-23 LUFS integrated (EBU R128)** with a **-1.0 dBFS peak ceiling**. If
the gain needed to hit -23 would push the peak above the ceiling, we take the
smaller gain and land quiet — preserving the waveform matters more than hitting a
number, and a clipped clip is unusable while a 2 dB-quiet one is merely imperfect.
Every such case is reported.

ORIGINALS ARE NEVER DESTROYED. Each source wav is copied to `_pre_loudnorm/`
beside it before being rewritten, and the applied gain is recorded in
`loudnorm.jsonl`. Idempotent: a clip already listed in the sidecar is skipped, so
re-running after adding one engine to a bank does not re-gain the rest.

C-L4 — WHAT "ALREADY LISTED" MEANS. The sidecar keyed on PATH alone, and a path is
not a clip: a reroll re-renders in place, keeping the filename. The stale record then
matched, the new take was skipped, and it shipped at whatever level its engine
produced — into a bank whose whole purpose is that every clip sits at one level, and
into the reference pool four engines condition on. Nothing reported it; the clip was
counted under `already-done`. Its pristine backup was skipped for the same reason, so
`_pre_loudnorm/` kept the ORIGINAL of a take that no longer exists.

The same keying produced the opposite defect once already — the quarantine move
(SKIP_DIRS below), where a clip normalized under its old path looked unprocessed and
was gained twice. One key, two failures in opposite directions, because the key does
not identify the audio.

A record now carries `sha_out`, the digest of the bytes this script left on disk, and
a clip is skipped only when the file still hashes to it. Records written before this
existed cannot be checked that way, so they fall back to the instrument: re-measure,
and if the file still sits at the loudness the record says it was left at, it is the
take we normalized. That upgrade is what stops the first run after this change from
re-gaining every clip in every existing bank.

Usage:
    .venv/bin/python scripts/stages/normalize_loudness.py --dir <bank_dir> [--target -23.0] [--dry-run]
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import synth_common  # noqa: E402

SIDECAR = "loudnorm.jsonl"
BACKUP_DIR = "_pre_loudnorm"
# Quarantine dirs live INSIDE the bank's audio dir, so rglob walks straight into
# them. Their clips were normalized once already, under their old top-level name —
# the sidecar keys on the path, so the move makes them look unprocessed and the
# next run gains them a second time (measured 2026-07-31: 45 clips re-gained by a
# routine 11-clip reroll). Skipping them here rather than at the call site keeps
# the guarantee where the recursion is.
SKIP_DIRS = {BACKUP_DIR, "_dropped", "_superseded"}
# pyloudnorm's BS.1770 meter uses a 400 ms block; anything shorter has no
# integrated loudness at all and must be passed through untouched.
MIN_SECONDS = 0.45


# How far the re-measured loudness of a legacy-record clip may sit from the level the
# record says it was left at and still be believed to be that clip.
#
# MEASURED 2026-08-07 across every loudnorm.jsonl on disk — 1,029 clips in 11 banks:
#   * 1,014 sit where their record says, and the worst of them is off by 0.185 dB
#     (p50 and p90 are 0.000 — a gain plus PCM_16 quantization is nearly exact);
#   * 15 are off by 0.90 to 7.09 dB.
# The gap between the two populations is 0.185 .. 0.904 dB and there is nothing in it, so
# the threshold is not a guess: 0.5 sits in the middle, ~2.7x above the worst honest drift
# and ~1.8x below the smallest reroll.
#
# Those 15 are the defect, not the instrument. Every one is a clip that WAS normalized and
# was then re-rendered in place — `the-return_nar_0059..0063_doc_QWE` were re-rendered
# twenty minutes after their gain was applied, and nine of the r2 rerolls likewise. Under
# the path key each matched a stale record, was counted under `already-done`, and shipped
# at its engine's native level: up to 7 dB off the level every other clip in the bank sits
# at. They went to the ear that way (loudness reads as confidence — reason 1 in the header)
# and into the reference pool four cloning engines condition on (reason 2).
LEGACY_LUFS_TOLERANCE = 0.5


def measure(data, rate):
    """Integrated loudness, or None when the clip is too short/silent to have one."""
    if len(data) < int(MIN_SECONDS * rate):
        return None
    lufs = pyln.Meter(rate).integrated_loudness(data)
    # Digital silence and near-silence return -inf; there is nothing to normalize to.
    return None if not np.isfinite(lufs) else lufs


def digest(path):
    """sha256 of the file's bytes — the clip's identity, which its path is not."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_lufs(rec):
    """Where a record says it left the clip, or None if it cannot say.

    `gain_db` is the gain ACTUALLY applied, which is below the requested one whenever the
    peak ceiling capped it — so this is `lufs_in + gain_db`, not the target. Reading the
    target instead would call every peak-capped clip a reroll.
    """
    if rec.get("skipped"):
        return None                       # never gained; nothing to check it against
    try:
        return float(rec["lufs_in"]) + float(rec["gain_db"])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="bank directory containing *.wav")
    ap.add_argument("--target", type=float, default=-23.0, help="target LUFS (default -23, EBU R128)")
    ap.add_argument("--peak-ceiling", type=float, default=-1.0, help="max peak dBFS after gain")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--backup-dir", default=None,
                    help="where originals go (default: <dir>/_pre_loudnorm). Point this "
                         "OUTSIDE a published dataset tree so the backups are not shipped.")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    wavs = sorted(p for p in root.rglob("*.wav") if not SKIP_DIRS & set(p.parts))
    if not wavs:
        sys.exit(f"no wavs under {root}")

    sidecar = root / SIDECAR
    # Last record per path wins: the sidecar is append-only, so a clip normalized, rerolled
    # and normalized again has several, and only the most recent describes the file on disk.
    done = {}
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("file"):
                    done[rec["file"]] = rec

    backup = pathlib.Path(args.backup_dir) if args.backup_dir else root / BACKUP_DIR
    ceiling_amp = 10 ** (args.peak_ceiling / 20.0)

    changed = skipped = untouched = 0
    ceiling_hits, records, rerolled, adopted = [], [], [], []

    for wav in wavs:
        rel = str(wav.relative_to(root))
        prev = done.get(rel)
        sha_now = digest(wav)

        if prev is not None and prev.get("sha_out") == sha_now:
            skipped += 1                        # the take we normalized, byte for byte
            continue

        data, rate = sf.read(str(wav), always_2d=False)
        if data.ndim > 1:                      # meter and gain both want mono
            data = data.mean(axis=1)

        lufs = measure(data, rate)
        if lufs is None:
            untouched += 1
            # Untouched, so in and out are the same bytes — recorded so a later run
            # skips it on the digest like everything else, and NOTICES if it is rerolled.
            records.append({"file": rel, "skipped": "too-short-or-silent",
                            "sha_in": sha_now, "sha_out": sha_now})
            continue

        if prev is not None:
            # A record for this path that does not match the file. Either it predates
            # `sha_out` (every record written before 2026-08-07) or the clip was
            # re-rendered in place. Ask the instrument which: if the file still sits at
            # the loudness the record says it was left at, it IS that take and the record
            # simply lacked a digest — adopt one and move on. Otherwise it is a reroll,
            # and the whole point of C-L4 is that it gets gained.
            want = expected_lufs(prev)
            if prev.get("sha_out") is None and want is not None \
                    and abs(lufs - want) <= LEGACY_LUFS_TOLERANCE:
                adopted.append(rel)
                skipped += 1
                records.append(dict(prev, sha_out=sha_now, sha_verified_by="lufs"))
                continue
            rerolled.append((rel, lufs))

        gain_db = args.target - lufs
        gain = 10 ** (gain_db / 20.0)
        peak = float(np.max(np.abs(data))) if len(data) else 0.0

        # Back the gain off rather than clip. Landing quiet is recoverable; a
        # squared-off waveform is not, and it would read as distortion in audit.
        capped = False
        if peak * gain > ceiling_amp and peak > 0:
            gain = ceiling_amp / peak
            gain_db = 20 * np.log10(gain)
            capped = True
            ceiling_hits.append((rel, lufs, args.target - lufs, gain_db))

        rec = {
            "file": rel,
            "lufs_in": round(float(lufs), 2),
            "lufs_target": args.target,
            "gain_db": round(float(gain_db), 2),
            "peak_capped": capped,
            "sha_in": sha_now,
        }
        changed += 1

        if args.dry_run:
            records.append(rec)
            continue

        # The backup is keyed on path too, so a reroll used to find the FIRST take's
        # original already sitting there and keep it — preserving the pristine copy of a
        # wav that no longer exists while the new take's original was lost on the spot.
        # Both are worth keeping; the digest is what tells them apart.
        dest = backup / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():                  # never overwrite a pristine original
            shutil.copy2(wav, dest)
        elif digest(dest) != rec["sha_in"]:
            dest = dest.with_name(f"{dest.stem}.{rec['sha_in'][:8]}{dest.suffix}")
            if not dest.exists():
                shutil.copy2(wav, dest)
        rec["backup"] = str(dest.relative_to(backup))
        synth_common.write_wav_atomic(str(wav), data * gain, rate, subtype="PCM_16")
        rec["sha_out"] = digest(wav)
        records.append(rec)

    if not args.dry_run:
        synth_common.append_jsonl(sidecar, records)

    verb = "would normalize" if args.dry_run else "normalized"
    print(f"{verb} {changed}  already-done {skipped}  too-short/silent {untouched}  (target {args.target} LUFS)")
    if adopted:
        print(f"  {len(adopted)} clip(s) carried a record with no content digest and "
              f"still measure at the level it recorded — same take, digest adopted "
              f"(one-time, for banks normalized before 2026-08-07)")
    if rerolled:
        # The C-L4 case, and it must never be silent again: these are clips that WERE
        # normalized, were then re-rendered in place, and under the old path key were
        # counted as `already-done` and shipped at their engine's native level.
        verb2 = "would be re-gained" if args.dry_run else "re-gained"
        print(f"  {len(rerolled)} clip(s) changed on disk since they were normalized "
              f"(reroll) and {verb2}; their originals are kept alongside the first take's:")
        for rel, lufs in rerolled[:10]:
            print(f"    {rel}  now {lufs:.1f} LUFS")
        if len(rerolled) > 10:
            print(f"    … and {len(rerolled) - 10} more")
    if ceiling_hits:
        print(f"  {len(ceiling_hits)} clip(s) hit the {args.peak_ceiling} dBFS ceiling and landed short of target:")
        for rel, lufs, wanted, got in ceiling_hits[:10]:
            print(f"    {rel}  in {lufs:.1f} LUFS  wanted {wanted:+.1f} dB  applied {got:+.1f} dB")
        if len(ceiling_hits) > 10:
            print(f"    … and {len(ceiling_hits) - 10} more")


if __name__ == "__main__":
    main()

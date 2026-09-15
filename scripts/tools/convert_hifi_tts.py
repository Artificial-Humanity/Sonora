# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["pyarrow", "soundfile", "librosa>=0.10", "soxr", "numpy"]
# # NOT torch and NOT the G2P stack. This converts audio and writes text; the VAT
# # labels, the phonemes and the filters are `scripts/lib/derive_vat_corpus.py`'s job
# # and it runs afterwards over this tree.
# ///
"""Hi-Fi TTS v1 (HF parquet, 44.1 kHz FLAC) -> a LibriTTS-R-shaped 24 kHz wav tree.

HOW TO RUN IT: the repo **.venv**, not `uv run` (AGENTS.md §3, same as
`derive_vat_corpus.py`). uv still manages that venv.

    .venv/bin/python scripts/tools/convert_hifi_tts.py --dry-run
    .venv/bin/python scripts/tools/convert_hifi_tts.py

This is rung 4's first half. Hi-Fi TTS is the largest corpus on this disk — 40 GB of
parquet WITH audio, 291.7 h, 10 speakers, measured 2026-09-14 — and the only cleared one
that needs no download. VCTK is the other half and is a different shape (a zip).

WHY THIS LAYOUT, AND IT IS NOT A CHOICE. `derive_vat_corpus.find_clips` walks exactly
`<root>/<speaker>/<chapter>/<stem>.wav` with a sibling `<stem>.normalized.txt`, and skips
any wav whose text file is missing — silently. So emitting that shape verbatim makes the
existing derivation consume this corpus with nothing but `--root`, and emitting anything
else would need an adapter nobody asked for.

WHY 24 kHz HERE RATHER THAN LATER. `derive_vat_corpus` REFUSES audio that is not already
`SAMPLE_RATE` (line 315: `if sr != SAMPLE_RATE or not MIN_SECONDS <= seconds <= MAX_SECONDS`)
— it rejects, it does not resample. Hi-Fi TTS ships 44.1 kHz, so a tree written at native
rate would derive an EMPTY corpus and the failure would look like a bad filter rather than
a missing conversion step.

WHAT THE DERIVATION WILL DROP, measured here so it is not a surprise there: 9,631 of
322,478 train rows (2.99%) are under `MIN_SECONDS = 1.0`. Nothing exceeds `MAX_SECONDS =
22.0` — the longest clip is 19.92 s. ⚠ The 4-second floor does NOT apply: that is
`qc_gate.SPEECH_MIN_SECONDS`, a hard gate on MEASURED VAD speech in the SYNTHESIS QC path,
not a corpus filter. Applying it here would discard 74.5% of this corpus for a rule that
was never about real audio.

⚠⚠ THE SPEAKER ID IS THE `speaker` COLUMN, NEVER THE DIRECTORY NAME. The source paths are
`audio/<speaker>_<subset>/<book>/<stem>.flac` — `6097_clean`, `6097_other` — and **speaker
6097 appears in both, inside `train` alone** (measured 2026-09-14). Keying on the directory
would split one reader into two identities, each with its own embedding row, which is the
exact confusion `make_warmstart._index_map` refuses a corpus for.

⚠⚠ FOUR OF THESE TEN READERS ARE ALREADY IN LibriTTS-R, AND AT LEAST TWO SHARE RECORDINGS.
Both corpora are LibriVox-derived and both use LibriVox reader ids, so the ids mean the same
person. Measured 2026-09-14 against `LibriTTS_R/train-clean-100/speakers.tsv`:

    92   Cori Samuel     6097  Phil Benson     6670  Mike Pelton     8051  Maria Kasper

and an exact-transcript probe over the same reader found **22 utterances present in both
corpora**, every one of them inside two reader/book pairs:

    reader 6670  LibriTTS-R chapter 296083  ==  Hi-Fi TTS book 8274
    reader 8051  LibriTTS-R chapter 118101  ==  Hi-Fi TTS book 7670

Those are not stock phrases — the longest is 131 characters of Florence Nightingale
biography. **The same LibriVox recording is in both corpora under two id schemes.**

This script does NOT deduplicate, deliberately. Dropping rows here would hide the overlap
from the merge that has to reason about it, and exact-transcript matching only catches the
utterances whose segmentation happens to align — Hi-Fi TTS cuts much shorter (median 2.74 s),
so the true overlap is larger than 22 and is a BOOK-level fact, not an utterance-level one.
What this script does instead is record `speaker`, `book` and `subset` per clip in
`manifest.jsonl`, which is what a book-level dedup needs. **The v8 merge owns that decision.**

⚠ THE HOLDOUT IS NOT AT RISK, and this was checked before anything was written. None of the
ten readers appears in `LibriTTS_R/dev-clean`, verified against both `holdout.txt` and
`holdout_8w.txt` (5,463 rows, 40 speakers each) with a positive control. Re-check it if the
speaker set ever changes: contaminating the holdout is silent and permanent, and there is no
second dev-clean to fall back on.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys

SRC = "/data/model-training/datasets/Hi-Fi-TTS"
OUT = "/data/model-training/datasets/hifi_tts_24k"
TARGET_SR = 24000

# The source's own splits. `train.*` is the corpus; `dev.*`/`test.*` are its 1.9 h of
# evaluation material and are EXCLUDED by default — not because they are tainted for us
# (our holdout is LibriTTS-R dev-clean, a different corpus) but because a training corpus
# that quietly absorbs another project's test split is the kind of provenance claim nobody
# can later verify. Pass them explicitly if they are ever wanted.
DEFAULT_SPLITS = ("train.clean", "train.other")


def shards(src, splits):
    """Every parquet file for the requested splits, derived from disk — never hand-listed.

    The shard count is in the filename (`-00000-of-00035-`) and is deliberately NOT parsed:
    a count read from a name is a claim about the directory, and the directory is right
    here. Refuses a split that matches nothing, because a typo'd split name silently
    converting zero clips is this repo's most-repeated failure.
    """
    data = os.path.join(src, "data")
    out = []
    for sp in splits:
        hit = sorted(f for f in os.listdir(data) if f.startswith(sp + "-") and f.endswith(".parquet"))
        if not hit:
            raise SystemExit(
                "no parquet matches split %r under %s — available: %s"
                % (sp, data, sorted({f.split("-")[0] for f in os.listdir(data) if f.endswith(".parquet")}))
            )
        out.extend(os.path.join(data, f) for f in hit)
    return out


def _dest(out, speaker, src_file, stem_suffix=".wav"):
    """`<out>/<speaker>/<book>/<stem><suffix>`, from the source's own path.

    `src_file` is `audio/<speaker>_<subset>/<book>/<stem>.flac`. The book component is kept
    because it is the unit the LibriTTS-R overlap lands on, and because a single directory
    holding 300,000 files is a filesystem problem rather than a layout.
    """
    parts = src_file.split("/")
    book = parts[2]
    stem = parts[-1].rsplit(".", 1)[0]
    return os.path.join(out, speaker, book, stem + stem_suffix)


def _convert_shard(job):
    """One parquet shard -> wavs + texts. Returns counts and manifest rows.

    Resumable the way `merge_expressive_registers._stage_24k` is: an existing destination whose
    samplerate and frame count already match is REUSED rather than re-decoded, so an
    interrupted run costs a stat per clip instead of a resample. The frame check is what
    makes that safe — a truncated file from a killed write has the wrong length and is
    redone. Writes go to `.tmp` and are renamed, so a kill never leaves a partial wav that
    a later run would trust.
    """
    path, out, dry, force = job
    import pyarrow.parquet as pq
    import soundfile as sf

    librosa = None
    n_written = n_reused = n_failed = n_notext = 0
    rows = []
    t = pq.read_table(path, columns=["speaker", "file", "duration", "text_normalized", "audio"])
    speakers = t["speaker"].to_pylist()
    files = t["file"].to_pylist()
    durs = t["duration"].to_pylist()
    texts = t["text_normalized"].to_pylist()
    audios = t["audio"]
    shard = os.path.basename(path)
    subset = "clean" if ".clean" in shard else "other"

    for i, (spk, src_file, dur, text) in enumerate(zip(speakers, files, durs, texts)):
        wav = _dest(out, spk, src_file)
        txt = _dest(out, spk, src_file, ".normalized.txt")
        rows.append({
            "id": os.path.basename(wav)[:-4], "wav": wav, "speaker": spk,
            "book": src_file.split("/")[2], "subset": subset,
            "source_duration": dur, "shard": shard, "source_file": src_file,
        })
        # ⚠⚠ A CLIP WITH NO TRANSCRIPT IS NOT CONVERTED AT ALL (#477), and this is the whole
        # reason the check is up here rather than beside the write. `find_clips` skips a wav
        # whose sibling text is missing WITHOUT SAYING SO, so writing the audio anyway would
        # produce a clip that counts as converted, occupies disk, and then vanishes from the
        # corpus with nothing in any log. An earlier version of this function did exactly
        # that — it guarded the TEXT write on a non-empty body and wrote the wav
        # unconditionally, four lines under a comment warning about this precise shape.
        # Measured: 0 of 323,978 real rows have an empty `text_normalized`, so this is latent
        # rather than live. Latent is why it needs to be loud.
        body = (text or "").strip()
        if not body:
            n_notext += 1
            rows[-1]["skipped"] = "empty text_normalized — no transcript, so no clip"
            continue
        if dry:
            continue
        # The text is rewritten even when the wav is reused: a half-written pair costs one
        # cheap write to repair and is expensive to diagnose.
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        with open(txt + ".tmp", "w", encoding="utf-8") as f:
            f.write(body + "\n")
        os.replace(txt + ".tmp", txt)

        expected = None
        if not force and os.path.exists(wav):
            try:
                got = sf.info(wav)
                expected = round(dur * TARGET_SR)
                if got.samplerate == TARGET_SR and abs(got.frames - expected) <= 2:
                    n_reused += 1
                    continue
            except Exception:
                pass
        try:
            if librosa is None:
                import librosa  # noqa: F811  — imported once per worker, not per clip
            raw = audios[i].as_py()["bytes"]
            y, _ = librosa.load(io.BytesIO(raw), sr=TARGET_SR, mono=True)
            # ⚠ `format="WAV"` IS LOAD-BEARING and its absence is not a style question.
            # The destination is `<stem>.wav.tmp` so the rename can be atomic, and
            # soundfile infers the container from the EXTENSION — which here is `.tmp`.
            # Without this it raises `TypeError: No format specified and unable to get
            # format from file extension`, on every clip, measured 2026-09-14: a smoke run
            # wrote 3,600 transcripts and 0 wavs.
            # ⚠ `scripts/tools/merge_expressive_registers._stage_24k` had the identical
            # defect and it is FIXED (#476) — this comment said otherwise until #479, having
            # gone stale in the same commit that repaired it. Kept as a pointer because the
            # two writes share a shape worth recognising, not as an open defect.
            sf.write(wav + ".tmp", y, TARGET_SR, subtype="PCM_16", format="WAV")
            os.replace(wav + ".tmp", wav)
            n_written += 1
        except Exception as e:                      # noqa: BLE001 — reported, never silent
            n_failed += 1
            rows[-1]["error"] = "%s: %s" % (type(e).__name__, e)
    return shard, n_written, n_reused, n_failed, n_notext, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                    help="comma-separated source splits (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="convert only the first N shards — for a smoke run, not for a corpus")
    ap.add_argument("--force", action="store_true", help="re-decode even where a good wav exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and report without writing a byte")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    files = shards(args.src, splits)
    if args.limit_shards:
        files = files[:args.limit_shards]
    print("%s %d shard(s) from %s -> %s"
          % ("would convert" if args.dry_run else "converting", len(files), args.src, args.out))

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    jobs = [(f, args.out, args.dry_run, args.force) for f in files]
    written = reused = failed = notext = 0
    manifest = []
    # Fork is safe here for the reason `derive_vat_corpus` documents: nothing on this path
    # touches the GPU, so the fork-after-GPU wedge on gfx1151 cannot apply.
    if args.workers > 1 and len(jobs) > 1:
        with mp.Pool(args.workers) as pool:
            for shard, w, r, fl, nt, rows in pool.imap_unordered(_convert_shard, jobs):
                written += w; reused += r; failed += fl; notext += nt; manifest.extend(rows)
                print("  %-52s +%-6d reused %-6d failed %-4d no-text %d"
                      % (shard, w, r, fl, nt), flush=True)
    else:
        for job in jobs:
            shard, w, r, fl, nt, rows = _convert_shard(job)
            written += w; reused += r; failed += fl; notext += nt; manifest.extend(rows)
            print("  %-52s +%-6d reused %-6d failed %-4d no-text %d"
                  % (shard, w, r, fl, nt), flush=True)

    # ⚠ FLOOR ON THE POPULATION THAT MATTERS. A run that converted nothing and a run whose
    # shards were all empty are the same zero, and an empty corpus is the failure this whole
    # lane keeps rediscovering. Refuse rather than report success over an empty list.
    if not manifest:
        raise SystemExit("no clips found in %d shard(s) — nothing was converted" % len(files))

    speakers = sorted({r["speaker"] for r in manifest}, key=int)
    hours = sum(r["source_duration"] for r in manifest) / 3600
    if not args.dry_run:
        mpath = os.path.join(args.out, "manifest.jsonl")
        with open(mpath + ".tmp", "w", encoding="utf-8") as f:
            for r in manifest:
                f.write(json.dumps(r) + "\n")
        os.replace(mpath + ".tmp", mpath)
        print("manifest -> %s (%d rows)" % (mpath, len(manifest)))

    verb = "would convert" if args.dry_run else "converted"
    print("%s %d clip(s) | %d written, %d reused, %d failed, %d skipped for no transcript "
          "| %.1f source-hours | %d speakers: %s"
          % (verb, len(manifest), written, reused, failed, notext, hours, len(speakers),
             ",".join(speakers)))
    if notext:
        print("⚠ %d clip(s) had an empty `text_normalized` and were NOT converted — they carry "
              "a `skipped` key in the manifest. `find_clips` would have dropped them silently."
              % notext, file=sys.stderr)
    if failed:
        print("⚠ %d clip(s) FAILED — they are in the manifest with an `error` key." % failed,
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

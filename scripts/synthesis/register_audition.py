"""register_audition — bridge book-prose synthesis output into the audition queue.

The Dataset Auditions app (audition.ai-lab-0:8095) is ratings.csv-driven: a clip
only appears in the review `todo` queue if a row exists in ratings.csv. The
book-prose lane (book_ingest.py -> synth_{dia,qwen,moss85}.py) stops at producing
wavs + per-engine manifests and never touches ratings.csv, so rendered books never
reach the vetting surface. This script closes that gap.

It walks a book's audio/ dir, reads the *_manifest.jsonl files, and appends one
`unaudited` row per rendered clip to ratings.csv (SSOT). Idempotent: clips whose id
is already in ratings.csv are skipped, so it's safe to re-run per book (or after a
partial render adds more clips).

Rows conform to the CSV's *actual on-disk header* (not an assumed schema), and the
`link` is written relative to RATINGS_DIR so the app's DATA_ROOT confinement passes.

Run:
  uv run Sonora/scripts/synthesis/register_audition.py \
      --book /data/model-training/datasets/book-prose/apple-cart [--dry-run]
  # every book under the book-prose root:
  uv run Sonora/scripts/synthesis/register_audition.py --all [--dry-run]
  # or point straight at a manifests dir (what synth_bank.sh passes as its out_dir):
  uv run Sonora/scripts/synthesis/register_audition.py --audio-dir <out_dir> [--dry-run]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Mirror the audition app's path resolution (same env overrides).
DATA_ROOT = Path(os.environ.get(
    "AUDITION_DATA_ROOT", "/data/model-training/datasets")).resolve()
RATINGS_DIR = Path(os.environ.get(
    "AUDITION_RATINGS_DIR", str(DATA_ROOT / "sonora-expressive-registers"))).resolve()
RATINGS_CSV = RATINGS_DIR / "ratings.csv"
BOOK_PROSE_ROOT = Path(os.environ.get(
    "BOOK_PROSE_ROOT", str(DATA_ROOT / "book-prose"))).resolve()

# Fallback header if ratings.csv doesn't exist yet (matches app's CSV_FIELDS).
DEFAULT_FIELDS = ["campaign", "id", "engine", "register",
                  "gender", "score", "note", "status", "link"]


def _read_header_and_ids():
    """Return (fieldnames, set-of-existing-ids). Empty defaults if no CSV yet."""
    if not RATINGS_CSV.is_file():
        return DEFAULT_FIELDS, set()
    with open(RATINGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or DEFAULT_FIELDS
        ids = {r.get("id", "") for r in reader}
    return list(fields), ids


def _manifest_rows(audio: Path):
    """Yield (record, wav_path) for every manifest line in an audio/manifests dir."""
    manifests = sorted(audio.glob("*_manifest.jsonl"))
    if not manifests:
        print(f"  ! no *_manifest.jsonl under {audio}", file=sys.stderr)
    for mf in manifests:
        with open(mf, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ! {mf.name}:{lineno} bad JSON ({e}); skipped",
                          file=sys.stderr)
                    continue
                yield rec, audio / rec.get("wav", "")


def _under_data_root(p: Path) -> bool:
    """True if p resolves inside DATA_ROOT (the app confines served audio to it)."""
    p = p.resolve()
    return DATA_ROOT == p or DATA_ROOT in p.parents


def _build_row(rec, wav: Path, book_slug: str, fields):
    """Map a manifest record to a ratings row dict keyed by the on-disk header."""
    link = os.path.relpath(wav.resolve(), RATINGS_DIR)  # e.g. ../book-prose/<b>/audio/x.wav
    full = {
        "campaign": rec.get("campaign") or f"book-{book_slug}",
        "id": rec.get("id", ""),
        "engine": rec.get("engine", ""),
        "register": rec.get("register", ""),
        "gender": "",           # derived on the fly by the app when blank
        "score": "",
        "note": "",
        "status": "unaudited",  # -> lands in the `todo` filter
        "link": link,
    }
    return {k: full.get(k, "") for k in fields}


def register_audio_dir(audio: Path, slug: str, existing_ids, fields, dry_run: bool):
    """Return list of new row-dicts for one manifests dir; reports skips/missing."""
    new_rows, seen = [], set(existing_ids)
    added = skipped = missing = outside = 0
    for rec, wav in _manifest_rows(audio):
        cid = rec.get("id", "")
        if not cid:
            print(f"  ! manifest record without id in {slug}; skipped",
                  file=sys.stderr)
            continue
        if cid in seen:
            skipped += 1
            continue
        if not wav.is_file():
            print(f"  ! wav missing for {cid}: {wav}; skipped", file=sys.stderr)
            missing += 1
            continue
        if not _under_data_root(wav):
            # The audition app only serves audio under DATA_ROOT; a row pointing
            # outside it would 404 in the UI, so don't queue an unplayable clip.
            print(f"  ! {cid} wav is outside DATA_ROOT ({wav}); not queued",
                  file=sys.stderr)
            outside += 1
            continue
        new_rows.append(_build_row(rec, wav, slug, fields))
        seen.add(cid)          # guard against dup ids across manifests
        added += 1
    verb = "would add" if dry_run else "queued"
    extra = "".join([f", {missing} missing-wav" if missing else "",
                     f", {outside} outside-data-root" if outside else ""])
    print(f"  {slug}: {verb} {added}, skipped {skipped} (already present){extra}")
    return new_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", action="append", default=[], metavar="DIR",
                    help="book-prose book dir (repeatable); scans DIR/audio. "
                         "e.g. .../book-prose/apple-cart")
    ap.add_argument("--audio-dir", action="append", default=[], metavar="DIR",
                    help="a manifests dir directly (repeatable); slug = its parent "
                         "name. This is what synth_bank.sh passes as its out_dir.")
    ap.add_argument("--all", action="store_true",
                    help=f"process every book under {BOOK_PROSE_ROOT}")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added without writing")
    args = ap.parse_args()

    # Build the list of (audio_dir, slug) targets from whichever flags were given.
    targets = []
    if args.all:
        for d in sorted(BOOK_PROSE_ROOT.iterdir()):
            if (d / "audio").is_dir():
                targets.append((d / "audio", d.name))
    for b in args.book:
        bd = Path(b).resolve()
        targets.append((bd / "audio", bd.name))
    for a in args.audio_dir:
        ad = Path(a).resolve()
        targets.append((ad, ad.parent.name))  # e.g. .../apple-cart/audio -> apple-cart
    if not targets:
        ap.error("give --book <dir>, --audio-dir <dir> (both repeatable), or --all")

    fields, existing_ids = _read_header_and_ids()
    print(f"ratings.csv: {RATINGS_CSV} ({len(existing_ids)} existing ids, "
          f"{len(fields)}-col header)")

    all_new = []
    for audio, slug in targets:
        if not audio.is_dir():
            print(f"  ! {audio} is not a dir; skipped", file=sys.stderr)
            continue
        all_new += register_audio_dir(audio, slug, existing_ids, fields, args.dry_run)

    if not all_new:
        print("nothing to add.")
        return
    if args.dry_run:
        print(f"[dry-run] {len(all_new)} rows would be appended to ratings.csv")
        return

    # Append conforming to the on-disk header; create with header if absent.
    new_file = not RATINGS_CSV.is_file()
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerows(all_new)
    print(f"appended {len(all_new)} unaudited rows to ratings.csv")


if __name__ == "__main__":
    main()

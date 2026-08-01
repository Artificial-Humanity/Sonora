# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""stage_pool — separate the segmented POOL from what is STAGED for training.

Owner design 2026-08-01: "We should separate out our dataset into the full pool and what
is staged for training. If we're already in here, we might as well break up the recording
into trainable segments and add them all to the pool. We should choose selections
linearly, from the start of the book reading up to our required number, record the range
and leave the rest in our pool."

WHY THE SPLIT EXISTS
--------------------
Segmenting is the expensive, one-time part (fetch, ASR anchor, CTC refine); selecting is
cheap and reversible. Doing all the segmenting once and staging a slice means the corpus
can grow later without re-running a single alignment.

It also fixes a real balance hazard. *Uneasy Money* alone is 6h41m across 25 sections. Fed
in whole it would dominate a corpus of ~800 keeps with ONE reader, ONE accent and ONE
delivery — and because `derive_vat_corpus` computes V/A/T as PER-SPEAKER z-scores, a
speaker holding most of the corpus effectively defines the population everyone else is
measured against. The pool keeps that material available without letting it set the
population.

SELECTION IS LINEAR AND RECORDED
--------------------------------
Clips are taken in READING ORDER (section, then sentence index) from where the last
staging run stopped, never sampled at random. Owner rationale 2026-08-01: **"there may be
value in training on continuous readings."**

That is a substantive bet, not a convenience. Contiguous clips preserve what isolated
sentences destroy:

  * **Declination and discourse arcs.** Pitch and energy drift across a paragraph and
    reset at its boundary. Randomly sampled sentences carry that drift as unexplained
    variance; contiguous ones carry it as structure. This is also the cleanest available
    test of the unexplained duration/arousal correlation (r ~ -0.42) — with continuous
    reading order we can finally ask whether it is declination or content.
  * **The north-star "hush".** A narrator lowering his voice as the mice creep past the
    Grand High Witch is a prosodic contour ACROSS utterances. A corpus of independent
    sentences contains no example of one.
  * **Verifiability.** "The range" is one pair of offsets. Anyone can check what was
    taken, resume from it, or undo it — which random sampling with a seed only
    approximates.

Every run appends to `staging_log.json`.

TAGS AND THE EAR
----------------
Staged clips inherit the ear-confirmed (reader, title) attributes
([[new-reader-ear-rule]]) and enter as `keep` WITHOUT a per-clip audition — the owner's
call for a known quantity: "we don't need to spend as much time on known quantities like
the chosen Libri recordings."

That relaxation applies ONLY to clips with no detected issue. Any clip carrying a QC
finding still enters as `unaudited` and reaches the ear, per the standing rule that every
QC failure is auditioned regardless of source or tier ([[qc-gate-mandatory]]). A book is
a known quantity; a flagged clip inside it is not.

Usage:
    uv run stage_pool.py --campaign librivox-v1 --status
    uv run stage_pool.py --campaign librivox-v1 --stage 250 --apply
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import shutil
import tempfile

DATASETS = pathlib.Path("/data/model-training/datasets")
RATINGS = DATASETS / "sonora-expressive-registers/ratings.csv"
PROFILES = DATASETS / "reader_profiles.json"
ATTRS = ("gender", "age", "accent")


def load_pool(campaign_dir: pathlib.Path) -> list[dict]:
    """Every segmented clip, in reading order."""
    pool: list[dict] = []
    for man in sorted(campaign_dir.rglob("librivox_manifest.jsonl")):
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    pool.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    seen, uniq = set(), []
    for r in pool:                       # last record per id wins, same as qc_gate
        if r.get("id"):
            seen.add(r["id"])
    by_id = {r["id"]: r for r in pool if r.get("id")}
    uniq = list(by_id.values())
    uniq.sort(key=lambda r: (r.get("book") or "", int(r.get("section") or 0),
                             int(r.get("sentence_index") or 0)))
    return uniq


def qc_flagged(campaign_dir: pathlib.Path) -> set[str]:
    """Ids with any failing gate — these keep their ear pass."""
    out: set[str] = set()
    qc = campaign_dir / "qc_measures.jsonl"
    if not qc.is_file():
        return out
    for line in qc.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") and not all((r.get("gates") or {}).values()):
            out.add(r["id"])
    return out


def confirmed_tags(rec: dict) -> dict:
    """Ear-confirmed attributes for this clip's (reader, title), or {}."""
    if not PROFILES.is_file():
        return {}
    prof = json.loads(PROFILES.read_text(encoding="utf-8"))
    entry = prof.get(rec.get("reader") or "") or {}
    t = (entry.get("titles") or {}).get(rec.get("book") or "") or {}
    return {a: t[a] for a in ATTRS if t.get(a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="dir under datasets/, e.g. librivox-v1")
    ap.add_argument("--stage", type=int, default=0, help="how many clips to stage")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    args = ap.parse_args()

    cdir = DATASETS / args.campaign
    log_path = cdir / "staging_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else \
        {"campaign": args.campaign, "runs": []}
    staged_ids = {i for run in log["runs"] for i in run["ids"]}

    pool = load_pool(cdir)
    flagged = qc_flagged(cdir)
    unstaged = [r for r in pool if r["id"] not in staged_ids]

    by_book = collections.Counter(r.get("book") or "?" for r in pool)
    print(f"POOL   {len(pool)} clips  "
          f"({sum(r.get('seconds', 0) for r in pool)/3600:.2f} h)")
    for b, n in by_book.most_common():
        print(f"         {b}: {n}")
    print(f"STAGED {len(staged_ids)} across {len(log['runs'])} run(s)")
    print(f"POOLED {len(unstaged)} still available")
    if args.status or not args.stage:
        for run in log["runs"]:
            print(f"  run {run['run']}: {run['count']} clips, "
                  f"reading-order range {run['range'][0]}..{run['range'][1]} ({run['date']})")
        return 0

    take = unstaged[: args.stage]
    if not take:
        print("nothing left to stage")
        return 0
    order = {r["id"]: i for i, r in enumerate(pool)}
    rng = [order[take[0]["id"]], order[take[-1]["id"]]]
    n_flag = sum(1 for r in take if r["id"] in flagged)
    tags = confirmed_tags(take[0])
    print(f"\nstaging {len(take)} clips, reading-order {rng[0]}..{rng[1]}")
    print(f"  tags from the ear-confirmed profile: {tags or '(NONE — refusing)'}")
    print(f"  {len(take)-n_flag} enter as keep; {n_flag} carry a QC finding -> unaudited")
    if not tags:
        print("  !! no confirmed (reader, title) profile — audition one clip first "
              "([[new-reader-ear-rule]])")
        return 1
    if not args.apply:
        print("\nDRY RUN — pass --apply to write")
        return 0

    before = RATINGS.stat().st_mtime_ns
    with RATINGS.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        hdr, rows = rd.fieldnames or [], list(rd)
    have = {r["id"] for r in rows}
    added = 0
    for rec in take:
        if rec["id"] in have:
            continue
        row = {k: "" for k in hdr}
        row.update({
            "campaign": f"book-{args.campaign}", "id": rec["id"],
            "engine": rec.get("engine", "librivox"),
            "status": "unaudited" if rec["id"] in flagged else "keep",
            "score": "" if rec["id"] in flagged else "5",
            "link": f"../{args.campaign}/audio/{rec['wav']}",
        })
        row.update({a: v for a, v in tags.items() if a in hdr})
        rows.append(row)
        added += 1
    if RATINGS.stat().st_mtime_ns != before:
        print("ABORT: ratings.csv changed under us (live writer)")
        return 1
    fd, tmp = tempfile.mkstemp(dir=str(RATINGS.parent), suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    if RATINGS.stat().st_mtime_ns != before:
        os.unlink(tmp)
        print("ABORT: ratings.csv changed during write")
        return 1
    shutil.copymode(str(RATINGS), tmp)
    os.replace(tmp, str(RATINGS))

    log["runs"].append({
        "run": len(log["runs"]) + 1, "date": "2026-08-01", "count": len(take),
        "range": rng, "qc_flagged": n_flag, "tags": tags,
        "ids": [r["id"] for r in take],
    })
    log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nstaged {added} new rows into ratings.csv; {len(pool)-len(staged_ids)-len(take)} "
          f"left in the pool")
    print(f"  recorded in {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

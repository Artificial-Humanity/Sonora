"""Phase 1S Step 2b: the text bank for Arm Q's synthetic corpus, 90 speakers x 400 lines.

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § Step 2 — Arm Q (owner, approved
2026-10-29), "Text bank and corpus size", and Option A (owner, 2026-10-29): every sentence of the
corpus is globally distinct, so no sentence is read by two speakers.

The selection rules and their constants are `scripts/lib/probe_text_bank.py`'s, called with Step 2b's
length bounds; the discovery is `build_1ci_text_bank.discover_books`. Step 1c-i's constants are not
changed here (`probe_text_bank.MIN_CLIP_SECONDS` etc. keep 1c-i's values).

    .venv/bin/python scripts/tools/build_step2_text_bank.py --check-pool     # progress, exit 0
    .venv/bin/python scripts/tools/build_step2_text_bank.py \\
        --out /data/model-training/datasets/probes/qwen-step2-text

OUTPUTS in --out:
  lines.jsonl           all 36,000 selected lines in canonical bank order (ids 2b-0001 ..).
  speakers/spk_NN.jsonl the 400 lines of speaker NN (00-89): 360 `primary` and 40 `reserve`, each
                        tagged with `speaker_index`, `role`, and `speaker_line_index` (1..400).
  meta.json             commit, tree state, seed, constants, sha256 of lines.jsonl and of every
                        speaker file, books used, holdout and ledger exclusions, pool statistics.

ASSIGNMENT. Line k of the canonical order goes to speaker k mod 90 at speaker line k div 90 + 1.
So every speaker takes lines spread across the whole bank, the first 360 of its 400 are primary,
and the last 40 are the reserve for top-up. No line appears twice.
"""
import argparse
import hashlib
import json
import math
import os as _os
import pathlib
import sys as _sys

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib", "tools"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import build_1ci_text_bank as b1  # noqa: E402
import probe_text_bank as ptb  # noqa: E402
from synth_common import write_json_atomic, write_text_atomic  # noqa: E402

TOOL = "scripts/tools/build_step2_text_bank.py"
# Option A (owner, 2026-10-29), with the registered bounds: 68 to 126 characters at 14 chars/s.
MIN_CLIP_SECONDS = 4.8
MAX_CLIP_SECONDS = 9.0
N_SPEAKERS = 90
LINES_PER_SPEAKER = 360
RESERVE_PER_SPEAKER = 40
SPEAKER_LINES = LINES_PER_SPEAKER + RESERVE_PER_SPEAKER
TARGET = N_SPEAKERS * SPEAKER_LINES          # 36,000 globally distinct lines
DEFAULT_SEED = 20261029
MAX_PER_BOOK = 800
ID_PREFIX = "2b-"                             # distinct from Step 1c-i's "1ci-"
DEFAULT_OUT = "/data/model-training/datasets/probes/qwen-step2-text"


def assign_speakers(rows):
    """-> [(speaker index, role, speaker line index, row)] for each row of `rows` in canonical
    order: line k to speaker k mod N_SPEAKERS, at speaker line k div N_SPEAKERS + 1. Refuses a row
    count other than TARGET."""
    if len(rows) != TARGET:
        raise ValueError(f"the corpus is {TARGET} lines; got {len(rows)}")
    out = []
    for k, row in enumerate(rows):
        s, j = k % N_SPEAKERS, k // N_SPEAKERS
        role = "primary" if j < LINES_PER_SPEAKER else "reserve"
        out.append((s, role, j + 1, row))
    return out


def speaker_files(rows):
    """-> {speaker index: [row, ...]} in speaker line order, each row tagged for its speaker."""
    spk = {s: [] for s in range(N_SPEAKERS)}
    for s, role, j, row in assign_speakers(rows):
        spk[s].append(dict(row, speaker_index=s, role=role, speaker_line_index=j))
    return spk


def build(args):
    """-> (canonical rows, speaker files, meta). Refuses (BankError) when the pool cannot make
    TARGET distinct lines under the per-book ceiling."""
    ledger = b1.load_ledger(args.ledger)
    sources = b1.load_holdout(args.holdout, args.chapters, args.books_txt)
    holdout_keys = ptb.holdout_title_keys(sources)
    holdout_pg_ids = sorted({s["book_id"] for s in sources}, key=int)
    found = b1.discover_books(args.books_root, holdout_keys, holdout_pg_ids, ledger,
                              min_clip_seconds=MIN_CLIP_SECONDS,
                              max_clip_seconds=MAX_CLIP_SECONDS)
    used, candidates = found["used"], found["candidates"]
    try:
        lines, stats = ptb.select_lines(candidates, n=args.n, seed=args.seed,
                                        max_per_book=args.max_per_book)
    except ptb.BankError as e:
        b1.refuse(f"REFUSING: {e}.")
    rows = ptb.bank_rows(lines, used, id_prefix=ID_PREFIX)
    spk = speaker_files(rows)
    for slug, b in used.items():
        b["selected"] = stats["per_book"].get(slug, 0)
    est = [r["est_seconds"] for r in rows]
    meta = {
        "tool": TOOL, **b1.tool_commit(),
        "spec": "Notes: Sonora/phase-1s-step1-preregistration.md § Step 2 — Arm Q, "
                "Text bank and corpus size; Option A",
        "seed": args.seed, "n_lines": len(rows), "n_speakers": N_SPEAKERS,
        "lines_per_speaker": LINES_PER_SPEAKER, "reserve_per_speaker": RESERVE_PER_SPEAKER,
        "sha256": ptb.canonical_sha256(rows), "canonical_form": ptb.CANONICAL_FORM,
        "speaker_sha256": {f"spk_{s:02d}": ptb.canonical_sha256(spk[s])
                           for s in range(N_SPEAKERS)},
        "constants": {"CHARS_PER_SEC": ptb.CHARS_PER_SEC,
                      "MIN_CLIP_SECONDS": MIN_CLIP_SECONDS, "MAX_CLIP_SECONDS": MAX_CLIP_SECONDS,
                      "TARGET": TARGET, "max_per_book": args.max_per_book,
                      "NEAR_DUP_SHINGLE": ptb.NEAR_DUP_SHINGLE,
                      "NEAR_DUP_JACCARD": ptb.NEAR_DUP_JACCARD},
        "parsers": b1._parser_versions(),
        "est_seconds": {"min": min(est), "max": max(est)},
        "pool": {"candidates": len(candidates), **{k: v for k, v in stats.items()
                                                   if k not in ("per_book", "per_author")}},
        "selection": {"rule": "round-robin across authors (name order), then across each "
                              "author's books (slug order); max_per_book is a ceiling",
                      "max_per_book": args.max_per_book, "n_authors": len(stats["per_author"]),
                      "per_author": stats["per_author"], "per_book": stats["per_book"]},
        "rejections": dict(sorted(found["rejections"].items())),
        "books_root": str(args.books_root),
        "books_used": used, "books_skipped": found["skipped"],
        "ledger": {"file": str(args.ledger),
                   "sha256": b1._sha256(pathlib.Path(args.ledger).read_bytes()),
                   "excluded": found["ledger_excluded"]},
        "holdout": {"file": str(args.holdout),
                    "sha256": b1._sha256(pathlib.Path(args.holdout).read_bytes()),
                    "source_books": [{k: s[k] for k in ("book_id", "title", "author",
                                                        "project_titles")} for s in sources],
                    "pg_ids": holdout_pg_ids},
        "exclusions": {"books": found["excluded_books"],
                       "documents": found["excluded_documents"]},
    }
    return rows, spk, meta


def write(out, rows, spk, meta):
    """Write lines.jsonl, speakers/ and meta.json. Refuses to replace a different bank. -> status."""
    out = pathlib.Path(out)
    text = ptb.lines_jsonl(rows)
    lines_path, meta_path = out / "lines.jsonl", out / "meta.json"
    if lines_path.exists():
        if lines_path.read_text(encoding="utf-8") != text:
            b1.refuse(f"REFUSING: {lines_path} exists and differs from this run's bank "
                      f"(sha256 {meta['sha256']}). Choose another --out.")
    elif meta_path.exists():
        b1.refuse(f"REFUSING: {meta_path} exists without a lines.jsonl beside it.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "speakers").mkdir(exist_ok=True)
    write_text_atomic(lines_path, text)
    for s in range(N_SPEAKERS):
        write_text_atomic(out / "speakers" / f"spk_{s:02d}.jsonl", ptb.lines_jsonl(spk[s]))
    write_json_atomic(meta_path, meta)
    return f"wrote {lines_path}, {N_SPEAKERS} speaker files and {meta_path}"


def check_pool(args):
    """Progress toward TARGET: usable books and deduplicated candidates on disk at the Step 2b
    bounds. Runs discovery and exact_dedupe only; exits 0."""
    ledger = b1.load_ledger(args.ledger)
    sources = b1.load_holdout(args.holdout, args.chapters, args.books_txt)
    holdout_keys = ptb.holdout_title_keys(sources)
    holdout_pg_ids = sorted({s["book_id"] for s in sources}, key=int)
    found = b1.discover_books(args.books_root, holdout_keys, holdout_pg_ids, ledger,
                              min_clip_seconds=MIN_CLIP_SECONDS,
                              max_clip_seconds=MAX_CLIP_SECONDS)
    kept, _dropped = ptb.exact_dedupe(found["candidates"], seed=args.seed)
    books = len(found["used"])
    need_books = math.ceil(TARGET / args.max_per_book)
    print(f"usable books {books} (need at least {need_books} at {args.max_per_book} lines "
          f"per book)")
    print(f"deduplicated candidate sentences {len(kept)} at {MIN_CLIP_SECONDS}-"
          f"{MAX_CLIP_SECONDS} s (target {TARGET}; shortfall {max(0, TARGET - len(kept))})")
    print(f"skipped books {len(found['skipped'])}; excluded by holdout "
          f"{len(found['excluded_books'])}; excluded by ledger {len(found['ledger_excluded'])}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, type=pathlib.Path)
    ap.add_argument("--check-pool", action="store_true",
                    help="print the usable books and deduplicated candidates against the target; "
                         "exit 0")
    ap.add_argument("--books-root", default=b1.BOOKS_ROOT, type=pathlib.Path)
    ap.add_argument("--holdout", default=b1.HOLDOUT, type=pathlib.Path)
    ap.add_argument("--chapters", default=_os.path.join(b1.LIBRITTS, "CHAPTERS.txt"),
                    type=pathlib.Path)
    ap.add_argument("--books-txt", default=_os.path.join(b1.LIBRITTS, "BOOKS.txt"),
                    type=pathlib.Path)
    ap.add_argument("--ledger", default=b1.LEDGER, type=pathlib.Path)
    ap.add_argument("--seed", default=DEFAULT_SEED, type=int)
    ap.add_argument("--n", default=TARGET, type=int)
    ap.add_argument("--max-per-book", default=MAX_PER_BOOK, type=int)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from a checkout with uncommitted changes (recorded in meta.json)")
    args = ap.parse_args(argv)
    if args.check_pool:
        return check_pool(args)
    commit = b1.tool_commit()
    if commit["tree_clean"] is not True and not args.allow_dirty:
        b1.refuse(f"REFUSING: {_SONORA_REPO} has uncommitted changes (or git cannot say), so "
                  "meta.json's commit would not name the code that ran. Commit first, or pass "
                  "--allow-dirty.")
    rows, spk, meta = build(args)
    print(write(args.out, rows, spk, meta))
    print(f"sha256 {meta['sha256']}  lines {len(rows)}  books {len(meta['books_used'])}  "
          f"authors {meta['selection']['n_authors']}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

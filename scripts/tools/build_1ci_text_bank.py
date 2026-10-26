"""Phase 1S Step 1c-i: build the 300-line text bank every cell reads.

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c — pre-registration,
Design, 1c-i "Text". The selection rules and their constants are `scripts/lib/probe_text_bank.py`'s;
this tool finds the books, reads the holdout's source books, and writes the bank.

    .venv/bin/python scripts/tools/build_1ci_text_bank.py \\
        --out /data/model-training/datasets/probes/qwen-1ci-text

SOURCES. Each `<books-root>/<slug>/book.epub` whose own OPF says Standard Ebooks (a
standardebooks.org identifier and publisher "Standard Ebooks"). The epub, not `<slug>_bank.json`:
a bank holds 40-90 director-windowed passages per book, the epub the whole text, and the epub
carries its provenance in the file that is read. Where a `<slug>_bank.json` exists, its
`source.url` must equal the epub's identifier, or the book is left out as a provenance conflict.
A directory without a `book.epub` is listed in `meta.json` as skipped, with the reason.

HOLDOUT. The dev-clean `<reader>/<chapter>` paths in the holdout filelist (paths only; no audio is
opened) are mapped to their source books through LibriTTS's `CHAPTERS.txt` and `BOOKS.txt`. A book
is excluded when its title or full title matches a holdout book's BOOKS.txt title or LibriVox
project title (`probe_text_bank.match_title`: accents folded, volume/part/version tags removed,
then equal, a whole-word prefix of at least two content words, or a word-set Jaccard >= 0.5 unless
the two authors are both known and differ), or when it shares a Project Gutenberg id with one. A
document inside a book (a story in a collection) whose title matches is excluded too. The match
over-excludes on purpose. Both lists, and every exclusion with the rule that fired, go in
`meta.json`.

LEDGER. `--ledger` (default `/data/model-training/datasets/books_ledger.json`, refused if absent)
records each book's lane. A book with any ledger entry in the real-audio lane (`lane` force-align
or `verdict` REAL-AUDIO) is excluded, so this synthetic bank stays disjoint from the real-audio
lane; so is a book the ledger does not record at all (its lane is unknown). Entries are matched by
the epub's Standard Ebooks identifier (`source_url`, `se_slug`, an `se:`/legacy `lv:` key) or a
Gutenberg id it lists (`probe_text_bank.ledger_entries`). Each used book's ledger keys and lanes,
and every ledger exclusion, go in `meta.json`.

TEXT. Sentences in `probe_text_bank.DENYLIST` (exact texts found defective in review) are dropped
before selection; each hit goes in `meta.json`. Selection balances AUTHORS, then books
(`probe_text_bank.select_lines`); the per-author and per-book counts go in `meta.json`.

OUTPUT, in `--out`:
* `lines.jsonl`, one row per line in bank (render) order: `id`, `text`, `book`, `title`,
  `source_url`, `est_seconds`, `chars`.
* `meta.json`: the tool commit and whether its tree was clean, the seed, every constant, the
  books used with their author, ledger keys and lanes, candidate and selected counts, the
  per-author and per-book selection, the books skipped, the holdout titles and the exclusions,
  the ledger and its exclusions, the denylist and its hits, the rejection counts, the pool size,
  and `sha256`, the hash of the canonical line list (`probe_text_bank.CANONICAL_FORM`).

Refuses with exit 2: fewer than 300 lines qualify; a missing or unreadable ledger; a `lines.jsonl`
already in `--out` that differs from this run's; a `meta.json` there without a `lines.jsonl`; a
checkout with uncommitted changes (the commit would not name the code), unless `--allow-dirty`. A
`lines.jsonl` identical to this run's is left as it is, with its `meta.json`.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, _os.path.join(_SONORA_REPO, "scripts", "lib")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import book_ingest  # noqa: E402
import probe_text_bank as ptb  # noqa: E402
from synth_common import write_json_atomic, write_text_atomic  # noqa: E402

TOOL = "scripts/tools/build_1ci_text_bank.py"
BOOKS_ROOT = "/data/model-training/datasets/book-prose"
# `data/libritts_r_holdout_devclean` in the shared checkout is a symlink to this directory.
HOLDOUT = "/data/model-training/sonora/data/libritts_r_holdout_devclean/holdout.txt"
LIBRITTS = "/data/model-training/datasets/LibriTTS"
LEDGER = "/data/model-training/datasets/books_ledger.json"


def tool_commit():
    """{"commit": this checkout's HEAD, "tree_clean": `git status --porcelain` empty}, each None
    when git cannot say."""
    def git(*argv):
        try:
            return subprocess.run(["git", "-C", _SONORA_REPO, *argv], capture_output=True,
                                  text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": head.strip() if head else None,
            "tree_clean": None if status is None else not status.strip()}


def refuse(message):
    """Print `message` to stderr and exit 2."""
    print(message, file=_sys.stderr)
    raise SystemExit(2)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_holdout(holdout, chapters_txt, books_txt):
    sources = ptb.holdout_sources(
        ptb.holdout_chapters(pathlib.Path(holdout).read_text(encoding="utf-8")),
        pathlib.Path(chapters_txt).read_text(encoding="utf-8", errors="replace"),
        pathlib.Path(books_txt).read_text(encoding="utf-8", errors="replace"))
    if not sources:
        refuse(f"REFUSING: {holdout} names no dev-clean chapter, so no holdout book "
                         "could be excluded.")
    return sources


def load_ledger(path):
    """The books ledger as a dict, refusing (exit 2) when it is missing or is not a JSON object:
    without it no book's lane can be checked."""
    path = pathlib.Path(path)
    if not path.is_file():
        refuse(f"REFUSING: books ledger {path} is missing, so no book's lane can be checked "
               "against the real-audio lane. Pass --ledger.")
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        refuse(f"REFUSING: books ledger {path} cannot be read as JSON: {e}")
    if not isinstance(ledger, dict):
        refuse(f"REFUSING: books ledger {path} is not a JSON object.")
    return ledger


def _same_url(a, b):
    return (a or "").rstrip("/").lower() == (b or "").rstrip("/").lower()


def discover_books(root, holdout_keys, holdout_pg_ids, ledger):
    """-> {"used", "skipped", "excluded_books", "excluded_documents", "ledger_excluded",
    "rejections", "denylist_hits", "candidates"}."""
    r = {"used": {}, "skipped": [], "excluded_books": [], "excluded_documents": [],
         "ledger_excluded": [], "rejections": {}, "denylist_hits": [], "candidates": []}
    for d in sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir()):
        slug, epub = d.name, d / "book.epub"
        if not epub.is_file():
            r["skipped"].append({"book": slug, "reason": "no local book.epub"})
            continue
        data = epub.read_bytes()
        book = ptb.read_epub(data)
        meta = book["meta"]
        if not ptb.is_standard_ebooks(meta):
            r["skipped"].append({"book": slug, "reason": "epub is not a Standard Ebooks edition",
                                 "identifier": meta["identifier"],
                                 "publisher": meta["publisher"]})
            continue
        bank = d / f"{slug}_bank.json"
        bank_url = None
        if bank.is_file():
            bank_url = (json.loads(bank.read_text(encoding="utf-8")).get("source") or {}).get("url")
            if not _same_url(bank_url, meta["identifier"]):
                r["skipped"].append({"book": slug,
                                     "reason": "bank source.url differs from the epub",
                                     "bank_url": bank_url, "identifier": meta["identifier"]})
                continue
        why = ptb.holdout_match(meta, holdout_keys, holdout_pg_ids)
        if why:
            r["excluded_books"].append({"book": slug, "title": meta["title"], "reason": why})
            continue
        use, ledger_record = ptb.ledger_decision(ptb.ledger_entries(ledger, meta))
        if not use:
            r["ledger_excluded"].append({"book": slug, "title": meta["title"],
                                         "source_url": meta["identifier"], **ledger_record})
            continue
        cands, rej, docs, deny = ptb.book_candidates(slug, book, holdout_keys)
        r["excluded_documents"] += docs
        r["denylist_hits"] += deny
        for k, v in rej.items():
            r["rejections"][k] = r["rejections"].get(k, 0) + v
        _src, licence, _note = book_ingest.text_provenance(meta["identifier"], None)
        r["used"][slug] = {"title": meta["title"], "fulltitle": meta["fulltitle"],
                           "author": meta["author"], "source_url": meta["identifier"],
                           "text_license": licence, "pg_ids": meta["pg_ids"],
                           "ledger": ledger_record["ledger"],
                           "epub_sha256": _sha256(data), "bank_url_checked": bank_url is not None,
                           "candidates": len(cands)}
        r["candidates"] += cands
    return r


def build(args):
    ledger = load_ledger(args.ledger)
    sources = load_holdout(args.holdout, args.chapters, args.books_txt)
    holdout_keys = ptb.holdout_title_keys(sources)
    holdout_pg_ids = sorted({s["book_id"] for s in sources}, key=int)
    found = discover_books(args.books_root, holdout_keys, holdout_pg_ids, ledger)
    used, candidates = found["used"], found["candidates"]
    excl_books, excl_docs = found["excluded_books"], found["excluded_documents"]
    try:
        lines, stats = ptb.select_lines(candidates, n=args.n, seed=args.seed,
                                        max_per_book=args.max_per_book)
    except ptb.BankError as e:
        refuse(f"REFUSING: {e}.")
    rows = ptb.bank_rows(lines, used)
    for slug, b in used.items():
        b["selected"] = stats["per_book"].get(slug, 0)
    lanes = {}
    for b in used.values():
        for lane in sorted({e["lane"] for e in b["ledger"]}, key=str):
            lanes[str(lane)] = lanes.get(str(lane), 0) + 1
    est = [r["est_seconds"] for r in rows]
    meta = {
        "tool": TOOL, **tool_commit(),
        "spec": "Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c — pre-registration, "
                "Design, 1c-i Text",
        "seed": args.seed, "n_lines": len(rows),
        "sha256": ptb.canonical_sha256(rows), "canonical_form": ptb.CANONICAL_FORM,
        "constants": {"CHARS_PER_SEC": ptb.CHARS_PER_SEC,
                      "MIN_CLIP_SECONDS": ptb.MIN_CLIP_SECONDS,
                      "MAX_CLIP_SECONDS": ptb.MAX_CLIP_SECONDS, "BANK_SIZE": ptb.BANK_SIZE,
                      "n": args.n, "max_per_book": args.max_per_book,
                      "NEAR_DUP_SHINGLE": ptb.NEAR_DUP_SHINGLE,
                      "NEAR_DUP_JACCARD": ptb.NEAR_DUP_JACCARD},
        "est_seconds": {"min": min(est), "max": max(est)},
        "pool": {"candidates": len(candidates), **{k: v for k, v in stats.items()
                                                   if k not in ("per_book", "per_author")}},
        "selection": {"rule": "round-robin across authors (name order), then across each "
                              "author's books (slug order); max_per_book is a ceiling",
                      "max_per_book": args.max_per_book, "n_authors": len(stats["per_author"]),
                      "per_author": stats["per_author"], "per_book": stats["per_book"]},
        "rejections": dict(sorted(found["rejections"].items())),
        "denylist": {"entries": [{"text": t, "reason": why} for t, why in ptb.DENYLIST.items()],
                     "hits": found["denylist_hits"]},
        "books_root": str(args.books_root),
        "books_used": used, "books_skipped": found["skipped"],
        "ledger": {
            "file": str(args.ledger),
            "sha256": _sha256(pathlib.Path(args.ledger).read_bytes()),
            "rule": "excluded: any matching entry with lane force-align or verdict REAL-AUDIO",
            "not_in_ledger": ptb.NOT_IN_LEDGER,
            "books_used_by_lane": dict(sorted(lanes.items())),
            "excluded": found["ledger_excluded"],
        },
        "holdout": {
            "file": str(args.holdout),
            "sha256": _sha256(pathlib.Path(args.holdout).read_bytes()),
            "chapters_txt": str(args.chapters),
            "chapters_txt_sha256": _sha256(pathlib.Path(args.chapters).read_bytes()),
            "books_txt": str(args.books_txt),
            "books_txt_sha256": _sha256(pathlib.Path(args.books_txt).read_bytes()),
            "n_chapters": sum(len(s["chapters"]) for s in sources),
            "source_books": [{k: s[k] for k in ("book_id", "title", "author", "project_titles")}
                             for s in sources],
            "pg_ids": holdout_pg_ids,
        },
        "exclusions": {"books": excl_books, "documents": excl_docs,
                       "any": bool(excl_books or excl_docs)},
    }
    return rows, meta


def write(out, rows, meta):
    """Write lines.jsonl and meta.json, refusing to replace a different bank. -> a status line."""
    out = pathlib.Path(out)
    text = ptb.lines_jsonl(rows)
    lines_path, meta_path = out / "lines.jsonl", out / "meta.json"
    if lines_path.exists():
        if lines_path.read_text(encoding="utf-8") != text:
            refuse(f"REFUSING: {lines_path} exists and differs from this run's bank "
                             f"(sha256 {meta['sha256']}). Choose another --out.")
        if meta_path.exists():
            return f"unchanged: {lines_path} is identical; left it and its meta.json as they are"
        write_json_atomic(meta_path, meta)
        return f"unchanged lines; wrote the missing {meta_path}"
    if meta_path.exists():
        refuse(f"REFUSING: {meta_path} exists without a lines.jsonl beside it.")
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(lines_path, text)
    write_json_atomic(meta_path, meta)
    return f"wrote {lines_path} ({len(rows)} lines) and {meta_path}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--books-root", default=BOOKS_ROOT, type=pathlib.Path)
    ap.add_argument("--holdout", default=HOLDOUT, type=pathlib.Path)
    ap.add_argument("--chapters", default=_os.path.join(LIBRITTS, "CHAPTERS.txt"),
                    type=pathlib.Path)
    ap.add_argument("--books-txt", default=_os.path.join(LIBRITTS, "BOOKS.txt"), type=pathlib.Path)
    ap.add_argument("--ledger", default=LEDGER, type=pathlib.Path,
                    help="books_ledger.json; a book in its real-audio lane, or absent from it, "
                         "is left out")
    ap.add_argument("--seed", default=ptb.DEFAULT_SEED, type=int)
    ap.add_argument("--n", default=ptb.BANK_SIZE, type=int)
    ap.add_argument("--max-per-book", default=ptb.MAX_PER_BOOK, type=int)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from a checkout with uncommitted changes (recorded in meta.json)")
    args = ap.parse_args(argv)
    commit = tool_commit()
    if commit["tree_clean"] is not True and not args.allow_dirty:
        refuse(f"REFUSING: {_SONORA_REPO} has uncommitted changes (or git cannot say), "
                         "so meta.json's commit would not name the code that ran. Commit first, "
                         "or pass --allow-dirty.")
    rows, meta = build(args)
    print(write(args.out, rows, meta))
    print(f"sha256 {meta['sha256']}  pool {meta['pool']['pool_size']}  "
          f"books {len(meta['books_used'])}  authors {meta['selection']['n_authors']}  "
          f"holdout exclusions: {len(meta['exclusions']['books'])} book(s), "
          f"{len(meta['exclusions']['documents'])} document(s)  "
          f"ledger exclusions: {len(meta['ledger']['excluded'])}  "
          f"denylist hits: {len(meta['denylist']['hits'])}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

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
is excluded when its normalised title or full title equals a holdout book's BOOKS.txt title or
LibriVox project title (`probe_text_bank.norm_title`), or when it shares a Project Gutenberg id
with one. A document inside a book (a story in a collection) whose title matches is excluded too.
Both lists, and every exclusion, go in `meta.json`.

OUTPUT, in `--out`:
* `lines.jsonl`, one row per line in bank (render) order: `id`, `text`, `book`, `title`,
  `source_url`, `est_seconds`, `chars`.
* `meta.json`: the tool commit and whether its tree was clean, the seed, every constant, the
  books used with candidate and selected counts, the books skipped, the holdout titles and the
  exclusions, the rejection counts, the pool size, and `sha256`, the hash of the canonical line
  list (`probe_text_bank.CANONICAL_FORM`).

Refuses with exit 2: fewer than 300 lines qualify; a `lines.jsonl` already in `--out` that differs
from this run's; a `meta.json` there without a `lines.jsonl`; a checkout with uncommitted changes
(the commit would not name the code), unless `--allow-dirty`. A `lines.jsonl` identical to this
run's is left as it is, with its `meta.json`.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

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
    print(message, file=sys.stderr)
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


def _same_url(a, b):
    return (a or "").rstrip("/").lower() == (b or "").rstrip("/").lower()


def discover_books(root, title_keys, holdout_pg_ids):
    """-> (used, skipped, excluded_books, excluded_documents, rejections, candidates)."""
    used, skipped, excl_books, excl_docs, rejections, candidates = {}, [], [], [], {}, []
    for d in sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir()):
        slug, epub = d.name, d / "book.epub"
        if not epub.is_file():
            skipped.append({"book": slug, "reason": "no local book.epub"})
            continue
        data = epub.read_bytes()
        book = ptb.read_epub(data)
        meta = book["meta"]
        if not ptb.is_standard_ebooks(meta):
            skipped.append({"book": slug, "reason": "epub is not a Standard Ebooks edition",
                            "identifier": meta["identifier"], "publisher": meta["publisher"]})
            continue
        bank = d / f"{slug}_bank.json"
        bank_url = None
        if bank.is_file():
            bank_url = (json.loads(bank.read_text(encoding="utf-8")).get("source") or {}).get("url")
            if not _same_url(bank_url, meta["identifier"]):
                skipped.append({"book": slug, "reason": "bank source.url differs from the epub",
                                "bank_url": bank_url, "identifier": meta["identifier"]})
                continue
        why = ptb.holdout_match(meta, title_keys, holdout_pg_ids)
        if why:
            excl_books.append({"book": slug, "title": meta["title"], "reason": why})
            continue
        cands, rej, docs = ptb.book_candidates(slug, book, title_keys)
        excl_docs += docs
        for k, v in rej.items():
            rejections[k] = rejections.get(k, 0) + v
        _src, licence, _note = book_ingest.text_provenance(meta["identifier"], None)
        used[slug] = {"title": meta["title"], "fulltitle": meta["fulltitle"],
                      "author": meta["author"], "source_url": meta["identifier"],
                      "text_license": licence, "pg_ids": meta["pg_ids"],
                      "epub_sha256": _sha256(data), "bank_url_checked": bank_url is not None,
                      "candidates": len(cands)}
        candidates += cands
    return used, skipped, excl_books, excl_docs, rejections, candidates


def build(args):
    sources = load_holdout(args.holdout, args.chapters, args.books_txt)
    title_keys = ptb.holdout_title_keys(sources)
    holdout_pg_ids = sorted({s["book_id"] for s in sources}, key=int)
    used, skipped, excl_books, excl_docs, rejections, candidates = discover_books(
        args.books_root, title_keys, holdout_pg_ids)
    try:
        lines, stats = ptb.select_lines(candidates, n=args.n, seed=args.seed,
                                        max_per_book=args.max_per_book)
    except ptb.BankError as e:
        refuse(f"REFUSING: {e}.")
    rows = ptb.bank_rows(lines, used)
    for slug, b in used.items():
        b["selected"] = stats["per_book"].get(slug, 0)
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
                                                   if k != "per_book"}},
        "rejections": dict(sorted(rejections.items())),
        "books_root": str(args.books_root),
        "books_used": used, "books_skipped": skipped,
        "holdout": {
            "file": str(args.holdout),
            "sha256": _sha256(pathlib.Path(args.holdout).read_bytes()),
            "chapters_txt": str(args.chapters),
            "chapters_txt_sha256": _sha256(pathlib.Path(args.chapters).read_bytes()),
            "books_txt": str(args.books_txt),
            "books_txt_sha256": _sha256(pathlib.Path(args.books_txt).read_bytes()),
            "n_chapters": sum(len(s["chapters"]) for s in sources),
            "source_books": [{k: s[k] for k in ("book_id", "title", "project_titles")}
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
    ap.add_argument("--chapters", default=os.path.join(LIBRITTS, "CHAPTERS.txt"),
                    type=pathlib.Path)
    ap.add_argument("--books-txt", default=os.path.join(LIBRITTS, "BOOKS.txt"), type=pathlib.Path)
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
          f"books {len(meta['books_used'])}  holdout exclusions: "
          f"{len(meta['exclusions']['books'])} book(s), "
          f"{len(meta['exclusions']['documents'])} document(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

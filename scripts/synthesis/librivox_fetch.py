# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""librivox_fetch — pull a REAL-AUDIO book's sections and canonical text to disk.

Stage B.0 of the force-align lane (notes/book-prose-lane.md, notes/librivox-quote-mining-plan.md).
`book_router.py` decides WHICH books enter the lane; this fetches the two things the
aligner needs for one of them:

  1. the LibriVox section audio (per-chapter MP3s, with their catalogued titles), and
  2. the CANONICAL text from the Gutenberg source the LibriVox project itself names.

Point 2 is the whole reason this is a separate stage. Per [[force-align-first-dataprep]]
the canonical text is the source of truth and ASR is fallback-only, so we must fetch the
text the readers actually read — the `url_text_source` recorded on the LibriVox project —
rather than guess at an edition. A mismatched edition produces alignments that look fine
and are quietly wrong.

Network only: no torch, no GPU, no ffmpeg. Runs on the host. The aligner runs in a
container afterwards (see librivox_align.sh).

Layout written per book:

    <root>/<ledger_key>/
        book.json          catalogue metadata + section list + text source
        audio/NNN.mp3      one file per LibriVox section, zero-padded in reading order
        text/source.txt    canonical text, Gutenberg header/footer stripped

Resumable: an existing, non-empty file of the expected size is left alone, so a killed
run costs only the section it was on.

Usage:
    .venv/bin/python scripts/synthesis/librivox_fetch.py --key lv:uneasy-money-by-p-g-wodehouse
    .venv/bin/python scripts/synthesis/librivox_fetch.py --url https://librivox.org/... --sections 1-3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASETS = pathlib.Path("/data/model-training/datasets")
LEDGER = DATASETS / "books_ledger.json"
DEFAULT_ROOT = DATASETS / "book-prose/real-audio"

UA = {"User-Agent": "Mozilla/5.0 (research; Sonora dataset force-align lane)"}
API = "https://librivox.org/api/feed/audiobooks/"
SLEEP = 1.0


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman(tok: str) -> str:
    """'viii' -> '8'. Anything that is not a clean roman numeral passes through.

    LibriVox slugs and titles disagree on numeral form within one series: the slug
    `...-volume-8-...` belongs to the title "World's Famous Orations, Vol. VIII". Volume
    number is the ONLY thing distinguishing volumes of an anthology by one author, so
    without this the series cannot be resolved at all.
    """
    if not tok or any(c not in _ROMAN for c in tok):
        return tok
    total = prev = 0
    for c in reversed(tok):
        v = _ROMAN[c]
        total += -v if v < prev else v
        prev = max(prev, v)
    return str(total) if total else tok


def _words(s: str) -> list[str]:
    """Normalized comparison tokens.

    Apostrophes are DELETED rather than spaced (so "World's" -> "worlds", matching the
    slug) — the same rule book_router already uses. A leading article is dropped because
    slugs carry one the titles often omit, and "vol"/"volume" and roman numerals are
    folded so an anthology's volumes compare.
    """
    w = re.sub(r"[^\w\s]", " ", (s or "").replace("'", "").replace("’", "")).lower().split()
    if w and w[0] in ("the", "a", "an"):
        w = w[1:]
    return [_roman("vol" if t == "volume" else t) for t in w]


def api_book(url_or_title: str) -> dict | None:
    """Resolve a LibriVox project from its URL slug.

    The API's `title=^...` is a LITERAL prefix match against the stored title, so a slug
    can never reproduce a title that carries punctuation. "Speeches: Literary and Social"
    slugs to `speeches-literary-and-social-by-charles-dickens`, and both
    `^speeches literary and social` and the two-word `^speeches literary` miss on the
    colon — which is why this returned "no match" for Dickens and for two of the three
    World's Famous Orations volumes on 2026-08-01.

    So the query is deliberately SHORT and the real matching happens client-side on
    punctuation-stripped words. Tiers narrow the candidate set; they do not decide.

    The old blind `return books[0]` is gone. It was the same defect already fixed in
    book_router: with a one-word query "^Speeches" the first result is *Speeches Against
    Catilina*, so asking for Dickens would have silently fetched Cicero. A wrong project
    is far worse than no project — the caller can warn, but it cannot detect this.
    """
    slug = urllib.parse.urlparse(url_or_title).path.strip("/").split("/")[-1] \
        if "librivox.org" in url_or_title else url_or_title
    title_part, _, author_part = slug.partition("-by-")
    want = _words(title_part.replace("-", " "))
    want_author = _words(author_part.replace("-", " "))
    if not want:
        return None

    # Query and comparison are deliberately different alphabets. `want` is normalized for
    # COMPARING; the query must use RAW slug text, because `title=^` is literal against the
    # stored title. And the slug has already lost punctuation the title keeps, so a whole
    # token can be unmatchable: "World's Famous Orations" slugs to `worlds-...`, and
    # `^worlds` matches nothing. Truncating the first token to a character prefix restores
    # it — `^world` does match. Leading articles are tried both ways for the same reason.
    raw = re.sub(r"[^\w\s]", " ", title_part.replace("-", " ")).lower().split()
    heads = [raw]
    if raw and raw[0] in ("the", "a", "an"):
        heads.append(raw[1:])
    queries: list[str] = []
    for toks in heads:
        for n in (len(toks), 3, 2, 1):
            if 0 < n <= len(toks):
                queries.append(" ".join(toks[:n]))
        if toks:
            queries += [toks[0][:k] for k in (6, 5, 4) if len(toks[0]) > k]

    tried: list[str] = []
    for q in queries:
        if not q or q in tried:
            continue
        tried.append(q)
        u = API + "?" + urllib.parse.urlencode(
            {"title": f"^{q}", "limit": 500, "extended": 1, "format": "json"})
        try:
            books = json.loads(fetch(u)).get("books") or []
        except (urllib.error.HTTPError, json.JSONDecodeError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code != 404:
                raise
            books = []
        time.sleep(SLEEP)
        cand = [b for b in books
                if (lambda t: t == want or t[: len(want)] == want)(_words(b.get("title")))]
        if len(cand) == 1:
            return cand[0]
        if len(cand) > 1 and want_author:
            # Author is a TIE-BREAKER, never a filter. For an anthology the slug names
            # the EDITOR while the API lists the individual orators — requiring agreement
            # rejected World's Famous Orations Vols VIII and X, whose titles already
            # disambiguate them by volume number. Only reach for the author when the
            # title genuinely leaves more than one candidate standing.
            narrowed = [b for b in cand if set(want_author) & set(_words(" ".join(
                f"{a.get('first_name', '')} {a.get('last_name', '')}"
                for a in (b.get("authors") or []))))]
            if len(narrowed) == 1:
                return narrowed[0]
    return None


# --------------------------------------------------------------- Gutenberg text
PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                      re.I | re.S)
PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                    re.I | re.S)


def gutenberg_plaintext(url_text_source: str) -> tuple[str, str] | None:
    """-> (text, resolved_url). Strips the PG header/footer per book-prose-lane rules."""
    m = re.search(r"(?:ebooks|etext|files)/(\d+)", url_text_source or "")
    if not m:
        return None
    eid = m.group(1)
    # Ordered by how reliably each yields plain UTF-8 text.
    candidates = [
        f"https://www.gutenberg.org/ebooks/{eid}.txt.utf-8",
        f"https://www.gutenberg.org/files/{eid}/{eid}-0.txt",
        f"https://www.gutenberg.org/files/{eid}/{eid}.txt",
    ]
    for cand in candidates:
        try:
            raw = fetch(cand).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - try the next mirror shape
            time.sleep(SLEEP)
            continue
        if len(raw) < 2000:
            continue
        s = PG_START.search(raw)
        if s:
            raw = raw[s.end():]
        e = PG_END.search(raw)
        if e:
            raw = raw[: e.start()]
        return raw.strip(), cand
    return None


def download(url: str, dest: pathlib.Path) -> tuple[bool, int]:
    """-> (downloaded_now, bytes). Skips a file that already looks complete."""
    if dest.exists() and dest.stat().st_size > 1024:
        return False, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = fetch(url, timeout=300)
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True, len(data)


def parse_sections(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--key", help="books_ledger key, e.g. lv:uneasy-money-by-p-g-wodehouse")
    g.add_argument("--url", help="librivox.org project URL")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--sections", help="1-based selection, e.g. '1-3' or '1,4,7' (default all)")
    ap.add_argument("--text-only", action="store_true", help="skip the audio download")
    args = ap.parse_args()

    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {}
    key, url = args.key, args.url
    if key:
        entry = ledger.get(key)
        if not isinstance(entry, dict):
            print(f"!! {key} not in ledger", file=sys.stderr)
            return 1
        if entry.get("lane") != "force-align":
            print(f"!! {key} is lane={entry.get('lane')}, not force-align", file=sys.stderr)
            return 1
        url = entry.get("url")
    if not key:
        key = "lv:" + urllib.parse.urlparse(url).path.strip("/").split("/")[-1]

    print(f"resolving {url}")
    meta = api_book(url)
    if not meta:
        print("!! LibriVox API returned no match", file=sys.stderr)
        return 1
    sections = meta.get("sections") or []
    print(f"  {meta.get('title')} — {len(sections)} sections, {meta.get('totaltime')}")
    print(f"  text source: {meta.get('url_text_source')}")

    out = pathlib.Path(args.root) / key.replace(":", "_")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "text").mkdir(parents=True, exist_ok=True)

    # ---- canonical text -------------------------------------------------
    got = gutenberg_plaintext(meta.get("url_text_source") or "")
    if got:
        text, src = got
        (out / "text/source.txt").write_text(text, encoding="utf-8")
        print(f"  text: {len(text):,} chars from {src}")
    else:
        text, src = "", None
        print("  !! no Gutenberg plaintext — the aligner cannot run without canonical "
              "text; supply it by hand at text/source.txt", file=sys.stderr)

    # ---- audio ----------------------------------------------------------
    want = parse_sections(args.sections, len(sections))
    fetched = skipped = total = 0
    if not args.text_only:
        for i in want:
            sec = sections[i]
            surl = sec.get("listen_url") or sec.get("url")
            if not surl:
                continue
            dest = out / "audio" / f"{i + 1:03d}.mp3"
            try:
                new, n = download(surl, dest)
            except Exception as exc:  # noqa: BLE001 - report and continue the batch
                print(f"  !! section {i + 1}: {type(exc).__name__}", file=sys.stderr)
                continue
            total += n
            if new:
                fetched += 1
                print(f"  [{i + 1:03d}] {n / 1e6:6.1f} MB  {(sec.get('title') or '')[:48]}")
                time.sleep(SLEEP)
            else:
                skipped += 1
    print(f"\n  audio: {fetched} fetched, {skipped} already present, {total / 1e6:.1f} MB")

    (out / "book.json").write_text(json.dumps({
        "ledger_key": key,
        "librivox_url": url,
        "title": meta.get("title"),
        "totaltime": meta.get("totaltime"),
        "num_sections": meta.get("num_sections"),
        "url_text_source": meta.get("url_text_source"),
        "text_fetched_from": src,
        "text_chars": len(text),
        "sections": [
            {"index": i + 1, "title": s.get("title"),
             "playtime": s.get("playtime"), "listen_url": s.get("listen_url"),
             "reader": (s.get("readers") or [{}])[0].get("display_name")
             if s.get("readers") else None}
            for i, s in enumerate(sections)
        ],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {out}/book.json")

    if args.key and args.key in ledger:
        ledger[args.key]["fetched_to"] = str(out)
        ledger[args.key]["status"] = "fetched; awaiting align"
        LEDGER.write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  ledger: {args.key} -> fetched; awaiting align")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

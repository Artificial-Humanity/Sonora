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
    .venv/bin/python scripts/tools/librivox_fetch.py --key lv:uneasy-money-by-p-g-wodehouse
    .venv/bin/python scripts/tools/librivox_fetch.py --url https://librivox.org/... --sections 1-3
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
# The markers and the cut moved to synth_common (A-M8) so the epub lane in book_ingest
# uses the SAME one — it had no PG stripping at all, only a Standard Ebooks filename SKIP
# list, so PG licence prose was reaching the renderers as though it were the novel.
# Re-exported under the old names: this module's own tests and callers already use them.
PG_START = synth_common.PG_START_RE
PG_END = synth_common.PG_END_RE


REPLACEMENT = "�"


def decode_gutenberg(raw: bytes, url: str) -> str:
    """Decode a PG plaintext file without baking U+FFFD into the corpus (A-H2).

    This was `raw.decode("utf-8", errors="replace")`. Project Gutenberg still serves a
    great many pre-2018 editions as **ISO-8859-1**, and `errors="replace"` turns every
    non-ASCII byte in them into U+FFFD: `café` becomes `caf�`, curly quotes and
    em-dashes become U+FFFD, and the file still looks fine at a glance because the ASCII
    is untouched. Nothing downstream detects it — the aligner matches on the ASCII, the
    clip ships, and the transcript carries a character that is not in any vocabulary.
    The `-0.txt` candidate in particular is *defined* as the Latin-1 edition.

    So: strict UTF-8 first (correct for modern editions and fails loudly rather than
    silently corrupting), then the two encodings PG actually used, then a final check
    that no replacement character survived from any source.
    """
    for encoding in ("utf-8", "cp1252", "iso-8859-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != "utf-8":
            print(f"  note: {url} is not UTF-8; decoded as {encoding}")
        break
    else:  # pragma: no cover - iso-8859-1 maps every byte, so this is unreachable
        raise ValueError(f"{url}: no candidate encoding decoded the file")

    # A source file can legitimately CONTAIN U+FFFD if whoever produced it made this same
    # mistake upstream. We cannot repair that, but shipping it silently is what the
    # finding is about, so say so and let the caller reject the edition.
    if REPLACEMENT in text:
        n = text.count(REPLACEMENT)
        raise ValueError(
            f"{url}: {n} U+FFFD replacement characters are present in the decoded text. "
            "The edition is already corrupt at source — pick another PG edition or supply "
            "text/source.txt by hand. (Decoding here is strict, so these did not come "
            "from us.)"
        )
    return text


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
            raw = decode_gutenberg(fetch(cand), cand)
        except ValueError as exc:  # a corrupt edition — say which, then try the next
            print(f"  !! {exc}", file=sys.stderr)
            time.sleep(SLEEP)
            continue
        except Exception:  # noqa: BLE001 - try the next mirror shape
            time.sleep(SLEEP)
            continue
        if len(raw) < 2000:
            continue
        raw = synth_common.strip_pg_boilerplate_text(raw)
        return raw.strip(), cand
    return None


def looks_like_mp3(data: bytes) -> bool:
    """Magic-byte check: an ID3 tag, or an MPEG frame sync (0xFF Ex/Fx)."""
    return data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


def download(url: str, dest: pathlib.Path) -> tuple[bool, int]:
    """-> (downloaded_now, bytes). Skips a file that already looks complete.

    A-M12: "complete" used to mean `size > 1024`, and an HTML error page — LibriVox's 404,
    an archive.org rate-limit notice, a captcha — clears 1024 bytes comfortably. It got
    written as `003.mp3`, and because the check is size-only, every RESUME of the batch
    then skipped it as already done. The bad file is permanent and the failure surfaces
    much later as an unalignable section. Both the resume check and the fresh download now
    look at what the bytes actually are.
    """
    if dest.exists() and dest.stat().st_size > 1024:
        with dest.open("rb") as fh:
            head = fh.read(3)
        if looks_like_mp3(head):
            return False, dest.stat().st_size
        print(f"  !! {dest.name} exists but is not MP3 (probably a saved error page) "
              "— refetching", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = fetch_with_retry(url)
    if not looks_like_mp3(data):
        raise ValueError(
            f"{url} returned {len(data):,} bytes that are not MP3 "
            f"(starts {data[:16]!r}) — not saving it as audio"
        )
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True, len(data)


# A-M13. One `fetch` and no retry, in a lane that pulls sixty-odd multi-megabyte files
# from archive.org over a rate-limited connection. The caller catches the exception and
# prints "!! section N" and moves on, so a single transient 503 or dropped socket left a
# permanent hole in the book — and because the resume check only looks at files that
# EXIST, a re-run does re-fetch it, but only if somebody notices the count is short. The
# 61-section Dickens is the shape that makes this bite.
RETRY_DELAYS = (2, 8, 30)


def fetch_with_retry(url: str, timeout: int = 300) -> bytes:
    """`fetch` with backoff on the failures that are worth retrying.

    A 404 is NOT retried: the URL is wrong and waiting will not fix it. Rate limits and
    5xx are, because they are the ones that are about timing rather than about the
    request — which is the whole reason a bulk transfer needs this and a single call does
    not.
    """
    last = None
    for attempt, delay in enumerate((0,) + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return fetch(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        if attempt < len(RETRY_DELAYS):
            print(f"  .. {type(last).__name__} on {url.rsplit('/', 1)[-1]}; "
                  f"retrying in {RETRY_DELAYS[attempt]}s", file=sys.stderr)
    raise last


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
    print(f"resolving {url}")
    meta = api_book(url)
    if not meta:
        print("!! LibriVox API returned no match", file=sys.stderr)
        return 1

    # A-H4. `api_book` matches on punctuation-stripped WORDS with a prefix fallback, so a
    # near-miss resolves to a real, wrong project — and every path below then writes that
    # project's audio into a directory named for the slug we ASKED for, because `key` is
    # derived from the request, never from the answer. Nothing downstream can notice: the
    # audio is valid, the sections are consistent, and the aligner will happily align the
    # wrong book against the wrong text. The API hands us the project's own URL; compare
    # it. One check, and it closes the only failure mode in this script that produces a
    # plausible corpus rather than an error.
    if url and "librivox.org" in url:
        want_slug = urllib.parse.urlparse(url).path.strip("/").split("/")[-1].lower()
        got_url = meta.get("url_librivox") or ""
        got_slug = urllib.parse.urlparse(got_url).path.strip("/").split("/")[-1].lower()
        if got_slug and got_slug != want_slug:
            print(
                f"!! RESOLVED THE WRONG PROJECT\n"
                f"   asked for: {want_slug}\n"
                f"   API gave:  {got_slug}  ({meta.get('title')!r})\n"
                f"   Refusing — this would have written {got_slug}'s audio into "
                f"{key}'s directory, and nothing downstream could tell.",
                file=sys.stderr,
            )
            return 1
        if not got_slug:
            print("  !! API returned no url_librivox; cannot verify the match",
                  file=sys.stderr)

    sections = meta.get("sections") or []
    print(f"  {meta.get('title')} — {len(sections)} sections, {meta.get('totaltime')}")
    print(f"  text source: {meta.get('url_text_source')}")

    # A-M10. `key` was `"lv:" + slug` whenever `--key` was not given, and the output
    # directory is `key.replace(":", "_")` — so `--key pg:6684` wrote `pg_6684` while
    # `--url <the same book>` wrote `lv_uneasy-money-by-p-g-wodehouse`. The same book,
    # downloaded twice into two trees, decided by invocation style. Resolving against the
    # ledger (which now knows both spellings, A-M9) gives one directory either way.
    #
    # It has to happen HERE and not at the top: the etext id comes from the API's
    # `url_text_source`, which is not known until the call above returns.
    if not key:
        etext = synth_common.etext_id_from(meta.get("url_text_source"))
        key, entry = synth_common.resolve_ledger_key(
            ledger, url=url, etext_id=etext)
        if key is None:
            print("!! cannot derive a ledger key from this URL", file=sys.stderr)
            return 1
        if entry is not None:
            print(f"  ledger: existing entry {key} (status: {entry.get('status')})")
        else:
            print(f"  ledger: no entry yet; this book is {key}")

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

    # A-M10. This was `if args.key and args.key in ledger`, so a fetch driven by `--url`
    # updated NOTHING — the audio landed, the text landed, and the ledger went on saying
    # "pending force-align ingest" indefinitely, which is the one field anyone reads to
    # find out what still needs doing. It also required the key to pre-exist, so the first
    # fetch of a book the router had not yet seen recorded nothing either.
    #
    # A-H3. `update_json` re-reads under an exclusive lock and touches only this key: the
    # `ledger` dict read at the top of main() is by now however many minutes of downloads
    # stale, so writing it back whole silently erased a concurrent router's entries.
    def _mark(led):
        entry = led.setdefault(key, {})
        entry["ledger_key"] = key
        entry["fetched_to"] = str(out)
        entry["librivox_url"] = url
        entry["status"] = "fetched; awaiting align"

    synth_common.update_json(LEDGER, _mark)
    print(f"  ledger: {key} -> fetched; awaiting align")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

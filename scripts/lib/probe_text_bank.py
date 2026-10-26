"""probe_text_bank — the fixed text bank a speaker-consistency probe reads in every cell.

Phase 1S Step 1c-i (Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c — pre-registration,
Design, "Text"): "the same 300 lines in every cell, so cells differ only in voice and engine:
complete prose sentences from Standard Ebooks with at least `MIN_CLIP_SECONDS` of estimated speech
and at most about 8 s; no title that is a source book of `data/libritts_r_holdout_devclean`."

The lines are read in one neutral-narration delivery by cloned voices, and speaker embeddings are
measured over them, so this module keeps only plain narration: one declarative sentence, no
quotation or dialogue, nothing a narrator cannot say. Every function here is pure (bytes and strings
in, values out); `scripts/tools/build_1ci_text_bank.py` does the file reading and writing.

WHY NOT `book_ingest.parse_epub`. It is the bank lane's parser and it is right for that lane, but it
reads through ebooklib, which rewrites `<body>` and drops its `epub:type`, and it keeps every `<p>`
of a prose document. This bank needs the structure parse_epub discards: Standard Ebooks marks the
reading text `bodymatter`, and puts poems, letters, epigraphs and inscriptions in `<blockquote>` or
under a `z3998:verse` / `z3998:letter` type. Those are not narration, so the walk below reads the
XHTML directly and skips them. The length model, the abbreviation expansion, the
illustration check and the completeness gate are book_ingest's and are imported, not copied.
"""
import hashlib
import io
import json
import posixpath
import re
import unicodedata
import zipfile

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, _os.path.join(_SONORA_REPO, "scripts", "lib")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from book_ingest import (  # noqa: E402
    CHARS_PER_SEC,
    MIN_CLIP_SECONDS,
    is_complete_utterance,
    normalize_speakable,
    references_the_invisible,
    split_sentences,
)

# The registered "at most about 8 s" of estimated speech. Inclusive, like the floor:
# MIN_CLIP_SECONDS <= chars / CHARS_PER_SEC <= MAX_CLIP_SECONDS.
MAX_CLIP_SECONDS = 8.0
BANK_SIZE = 300
# No book supplies more than this many lines (10% of the bank). Selection is round-robin across
# books, so the cap binds only when few books qualify or some have few candidates.
MAX_PER_BOOK = 30
DEFAULT_SEED = 20261026
# Near-duplicate: Jaccard similarity of word-trigram sets at or above this, against any line
# already selected. Exact duplicates (same text after `dedupe_key`) are removed from the pool first.
NEAR_DUP_SHINGLE = 3
NEAR_DUP_JACCARD = 0.5
ID_PREFIX = "1ci-"

CANONICAL_FORM = ("sha256 of the UTF-8 bytes of every line's `text` in bank order, each followed "
                  "by one LF; equivalently `jq -r .text lines.jsonl | sha256sum`")


def est_seconds(text):
    return len(text) / CHARS_PER_SEC


def within_length(text):
    """True when the estimated speech is inside the registered bounds, both edges included."""
    s = est_seconds(text)
    return MIN_CLIP_SECONDS <= s <= MAX_CLIP_SECONDS


# ---------------------------------------------------------------- text

# Standard Ebooks typesetting: U+2060 WORD JOINER beside dashes and ellipses, U+00A0 after
# honorifics, U+200A HAIR SPACE inside ellipses. None of it is speech; joiners go, spaces become
# a plain space.
_DROP_CHARS = dict.fromkeys(map(ord, "\u2060\ufeff\u200b\u00ad"), None)
_SPACE_CHARS = dict.fromkeys(map(ord, "\u00a0\u200a\u2009\u202f\u2002\u2003\u2007\u2008"), " ")


def clean_text(text):
    text = (text or "").translate(_DROP_CHARS).translate(_SPACE_CHARS)
    return re.sub(r"\s+", " ", text).strip()


# Quotation marks of every convention. U+2019 is left out: it is the apostrophe far more often
# than a closing quote, and a sentence whose opening mark is ‘ is caught by that mark.
QUOTE_CHARS = "\u201c\u201d\"\u2018\u00ab\u00bb\u201e\u201a\u2039\u203a"
_QUOTE_RE = re.compile("[" + re.escape(QUOTE_CHARS) + "]|(?<![A-Za-z])'|'(?![A-Za-z])")
_SYMBOL_RE = re.compile(r"[()\[\]{}<>*&/\\#@%$§†‡¶|_~^=+]")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
# A letter-dot inside the sentence: initials ("H. A. Wilcox") and dotted abbreviations ("p.m.").
_INITIAL_RE = re.compile(r"\b[A-Za-z]\.(?=\s*\S)")


def _speakable_char(c):
    """ASCII, a Latin letter (accents included), or the apostrophe and dashes of running prose."""
    return c.isascii() or c in "\u2019\u2013\u2014" or (
        c.isalpha() and "LATIN" in unicodedata.name(c, ""))


def has_quote(text):
    return bool(_QUOTE_RE.search(text))


def sentence_rejection(text):
    """None when `text` is a usable bank line, otherwise the first reason it is not.

    `text` is the cleaned, `normalize_speakable`-expanded sentence; the length is measured on it,
    because that is what an engine reads.
    """
    if not within_length(text):
        return "too_short" if est_seconds(text) < MIN_CLIP_SECONDS else "too_long"
    if has_quote(text):
        return "quotation"
    if not is_complete_utterance(text):
        return "incomplete"
    if not text[0].isalpha() or not text[0].isupper():
        return "not_capitalised"
    if not text.endswith("."):
        return "not_declarative"          # questions, exclamations and trailing ellipses
    if any(c in text for c in "?!\u2026") or "..." in text:
        return "not_declarative"
    if any(c.isdigit() for c in text):
        return "digits"
    if _SYMBOL_RE.search(text):
        return "symbol"
    if _ALLCAPS_RE.search(text):
        return "all_caps"
    if _INITIAL_RE.search(text[:-1]):
        return "initials"
    if references_the_invisible(text):
        return "references_the_invisible"
    if not all(_speakable_char(c) for c in text):
        return "symbol"
    return None


def sentences_from_paragraph(paragraph):
    """The paragraph's sentences, cleaned and normalised as they would be read.

    A paragraph that quotes anyone is skipped whole: the sentences around a quotation are its
    attribution and stage business ("He turned to the window."), which belong to a scene rather
    than to narration, and a splitter can leave a quotation's tail inside a neighbouring sentence.
    """
    paragraph = clean_text(paragraph)
    if not paragraph or has_quote(paragraph):
        return []
    return [normalize_speakable(clean_text(s)) for s in split_sentences(paragraph)]


# ---------------------------------------------------------------- titles

_ARTICLES = {"the", "a", "an"}


def norm_title(title):
    """Lowercase, `&` -> and, apostrophes deleted, other punctuation -> space, leading article and
    a trailing LibriVox `(version N)` dropped, spaces collapsed.

    The same rules as `scripts/tools/book_router.norm_title`, plus the version tag, which only
    LibriVox project titles carry. Lossy so that "The Count of Monte Cristo" and "Count of Monte
    Cristo" agree; deliberately not clever enough to equate different titles.
    """
    t = (title or "").lower().replace("&", " and ")
    t = re.sub(r"\(\s*version\s+\w+\s*\)", " ", t)
    t = re.sub(r"['‘’]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    words = t.split()
    while words and words[0] in _ARTICLES:
        words.pop(0)
    return " ".join(words)


def _index_rows(text):
    for line in text.splitlines():
        if not line.strip() or line.startswith(";") or "|" not in line:
            continue
        yield [f.strip() for f in line.split("|")]


def holdout_chapters(holdout_text):
    """{(reader_id, chapter_id)} from a holdout filelist whose first field is a
    `.../dev-clean/<reader>/<chapter>/<utt>.wav` path. Reads paths only."""
    out = set()
    for line in holdout_text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 1)[0].split("/")
        if "dev-clean" not in parts:
            raise ValueError(f"holdout row is not a dev-clean path: {line[:120]!r}")
        i = parts.index("dev-clean")
        out.add((parts[i + 1], parts[i + 2]))
    return out


def holdout_sources(chapters, chapters_txt, books_txt):
    """The LibriTTS source books of `chapters`: [{book_id, title, project_titles, chapters}],
    sorted by book_id. Refuses a chapter CHAPTERS.txt does not list, or lists under another reader
    or a subset other than dev-clean, rather than leave a holdout book unnamed."""
    by_chapter = {r[0]: r for r in _index_rows(chapters_txt) if r[0].isdigit()}
    titles = {r[0]: r[1] for r in _index_rows(books_txt) if r[0].isdigit() and len(r) > 1}
    books = {}
    for reader, chapter in sorted(chapters):
        row = by_chapter.get(chapter)
        if row is None or len(row) < 8:
            raise ValueError(f"chapter {chapter} (reader {reader}) is not in CHAPTERS.txt")
        if row[1] != reader or row[3] != "dev-clean":
            raise ValueError(f"CHAPTERS.txt row for chapter {chapter} is reader {row[1]}, "
                             f"subset {row[3]}; the holdout says reader {reader}, dev-clean")
        book_id = row[5]
        if book_id not in titles:
            raise ValueError(f"book {book_id} (chapter {chapter}) is not in BOOKS.txt")
        b = books.setdefault(book_id, {"book_id": book_id, "title": titles[book_id],
                                       "project_titles": set(), "chapters": []})
        b["project_titles"].add(row[7])
        b["chapters"].append(f"{reader}/{chapter}")
    return [{**b, "project_titles": sorted(b["project_titles"])}
            for _k, b in sorted(books.items(), key=lambda kv: int(kv[0]))]


def holdout_title_keys(sources):
    """{normalised title: the holdout book title it came from}, over BOOKS.txt titles and LibriVox
    project titles."""
    keys = {}
    for b in sources:
        for t in (b["title"], *b["project_titles"]):
            k = norm_title(t)
            if k:
                keys.setdefault(k, b["title"])
    return keys


# ---------------------------------------------------------------- epub

_SKIP_TAGS = {"blockquote", "header", "hgroup", "footer", "figure", "figcaption", "table",
              "aside", "nav", "address"}
_SKIP_TYPES = ("epigraph", "verse", "poem", "song", "hymn", "lyrics", "letter", "drama",
               "stage-direction", "inscription", "footnote", "endnote", "note", "z3998:sign")
_PG_ID_RE = re.compile(r"gutenberg\.org/(?:ebooks|etext)/(\d+)")


def _epub_type(tag):
    return (tag.get("epub:type") or tag.get("type") or "") if hasattr(tag, "get") else ""


def _skipped(p):
    """True when <p> sits inside something that is not the narrator's running prose."""
    for anc in (p, *p.parents):
        name = getattr(anc, "name", None)
        if name is None:
            continue
        if name in _SKIP_TAGS:
            return True
        et = _epub_type(anc)
        if any(t in et for t in _SKIP_TYPES):
            return True
    return False


def _foreign(p):
    """True when <p> contains text marked as another language (a narrator may not say it)."""
    for el in p.find_all(True):
        lang = el.get("xml:lang") or el.get("lang")
        if lang and not lang.lower().startswith("en"):
            return True
    return False


def read_epub(epub_bytes):
    """-> {"meta": {...}, "documents": [{"href", "titles", "paragraphs"}]} in spine order.

    `meta` holds the OPF's identifier, title, fulltitle, author, publisher, rights and the Project
    Gutenberg ids among its `dc:source`s. `documents` holds only bodymatter documents; each
    paragraph is a `<p>` outside the structures in `_SKIP_TAGS` / `_SKIP_TYPES` and free of
    foreign-language markup. `titles` are the document's <title> and headings.
    """
    from bs4 import BeautifulSoup

    zf = zipfile.ZipFile(io.BytesIO(epub_bytes))
    container = BeautifulSoup(zf.read("META-INF/container.xml"), "lxml-xml")
    opf_path = container.find("rootfile")["full-path"]
    opf = BeautifulSoup(zf.read(opf_path), "lxml-xml")
    base = posixpath.dirname(opf_path)

    def dc(name, id_=None):
        for el in opf.find_all(name):
            if el.prefix == "dc" and (id_ is None or el.get("id") == id_):
                return clean_text(el.get_text())
        return None

    sources = [clean_text(el.get_text()) for el in opf.find_all("source") if el.prefix == "dc"]
    meta = {
        "identifier": dc("identifier"),
        "title": dc("title", "title") or dc("title"),
        "fulltitle": dc("title", "fulltitle"),
        "author": dc("creator"),
        "publisher": dc("publisher"),
        "rights": dc("rights"),
        "pg_ids": sorted({m.group(1) for s in sources for m in [_PG_ID_RE.search(s)] if m},
                         key=int),
    }
    manifest = {it["id"]: it["href"] for it in opf.find_all("item") if it.get("href")}
    documents = []
    for ref in opf.find_all("itemref"):
        href = manifest.get(ref.get("idref"))
        if not href or not href.endswith((".xhtml", ".html", ".htm")):
            continue
        soup = BeautifulSoup(zf.read(posixpath.join(base, href)), "lxml-xml")
        body = soup.find("body")
        if body is None or "bodymatter" not in _epub_type(body):
            continue
        titles = []
        head_title = soup.find("title")
        if head_title:
            titles.append(clean_text(head_title.get_text(" ")))
        for h in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            titles.append(clean_text(h.get_text(" ")))
        for el in body.find_all(True):
            if "title" in _epub_type(el).split() and el.name not in {"h1", "h2", "h3", "h4",
                                                                       "h5", "h6"}:
                titles.append(clean_text(el.get_text(" ")))
        paragraphs = []
        for p in body.find_all("p"):
            if _skipped(p) or _foreign(p):
                continue
            t = clean_text(p.get_text(""))
            if t:
                paragraphs.append(t)
        documents.append({"href": href, "titles": [t for t in titles if t],
                          "paragraphs": paragraphs})
    return {"meta": meta, "documents": documents}


def is_standard_ebooks(meta):
    """Provenance from the book itself: a standardebooks.org identifier, published by Standard
    Ebooks."""
    ident = (meta.get("identifier") or "").lower()
    return (ident.startswith("https://standardebooks.org/ebooks/")
            and (meta.get("publisher") or "").strip() == "Standard Ebooks")


def holdout_match(book, title_keys, holdout_pg_ids):
    """Why `book` is a holdout source book, or None. `book` carries `title`, `fulltitle` and
    `pg_ids`: a normalised title equal to a holdout title, or a shared Project Gutenberg id."""
    for field in ("title", "fulltitle"):
        k = norm_title(book.get(field))
        if k and k in title_keys:
            return f"{field} {book.get(field)!r} matches holdout book {title_keys[k]!r}"
    shared = sorted(set(book.get("pg_ids") or ()) & set(holdout_pg_ids), key=int)
    if shared:
        return f"Project Gutenberg id(s) {shared} are holdout source books"
    return None


def book_candidates(slug, book, title_keys):
    """Candidate lines from one parsed book: (candidates, rejections, excluded_sections).

    A document one of whose titles is a holdout book's title (a story in a collection) is
    left out whole and reported. Candidates keep reading order.
    """
    cands, rejections, excluded = [], {}, []
    for doc in book["documents"]:
        hit = next((t for t in doc["titles"] if norm_title(t) in title_keys), None)
        if hit:
            excluded.append({"book": slug, "document": doc["href"], "title": hit,
                             "holdout_title": title_keys[norm_title(hit)]})
            continue
        for para in doc["paragraphs"]:
            if has_quote(clean_text(para)):
                rejections["quoted_paragraph"] = rejections.get("quoted_paragraph", 0) + 1
                continue
            for s in sentences_from_paragraph(para):
                why = sentence_rejection(s)
                if why:
                    rejections[why] = rejections.get(why, 0) + 1
                    continue
                cands.append({"book": slug, "text": s})
    return cands, rejections, excluded


# ---------------------------------------------------------------- selection


class BankError(ValueError):
    """The candidates cannot make the bank the spec asks for."""


def dedupe_key(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower().replace("\u2019", "")).split())


def _shingles(text):
    w = dedupe_key(text).split()
    n = NEAR_DUP_SHINGLE
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _rank(seed, *parts):
    return hashlib.sha256("\x1f".join(map(str, (seed, *parts))).encode("utf-8")).hexdigest()


def exact_dedupe(candidates):
    """Drop every candidate whose `dedupe_key` was already seen, keeping the first in input
    order. -> (kept, n_dropped)."""
    seen, kept = set(), []
    for c in candidates:
        k = dedupe_key(c["text"])
        if k in seen:
            continue
        seen.add(k)
        kept.append(c)
    return kept, len(candidates) - len(kept)


def select_lines(candidates, *, n=BANK_SIZE, seed=DEFAULT_SEED, max_per_book=MAX_PER_BOOK):
    """Pick `n` lines from `candidates` (dicts with `book` and `text`) deterministically.

    Each candidate's rank is sha256(seed, book, text), so the result depends on the candidate SET
    and the seed, not on input order. Books take turns in slug order; each turn takes the book's
    best-ranked line that is not a near-duplicate of one already taken, until the bank has `n` lines
    or the book has `max_per_book`. The chosen lines are then ordered by sha256(seed, "order",
    text): render order matters to the drift criterion, so no book is tied to a position.

    -> (lines in bank order, stats). Raises BankError when fewer than `n` lines qualify.
    """
    pool, n_exact = exact_dedupe(candidates)
    if len(pool) < n:
        raise BankError(f"only {len(pool)} candidate lines qualify (after removing {n_exact} "
                        f"exact duplicates); the bank needs {n}")
    by_book = {}
    for c in pool:
        by_book.setdefault(c["book"], []).append(c)
    queues = {b: sorted(cs, key=lambda c: _rank(seed, c["book"], c["text"]))
              for b, cs in by_book.items()}
    cursor = dict.fromkeys(queues, 0)
    taken, shingles, per_book, near_dups = [], [], dict.fromkeys(queues, 0), 0
    while len(taken) < n:
        progressed = False
        for book in sorted(queues):
            if len(taken) >= n:
                break
            if per_book[book] >= max_per_book:
                continue
            q = queues[book]
            while cursor[book] < len(q):
                c = q[cursor[book]]
                cursor[book] += 1
                sh = _shingles(c["text"])
                if any(_jaccard(sh, s) >= NEAR_DUP_JACCARD for s in shingles):
                    near_dups += 1
                    continue
                taken.append(c)
                shingles.append(sh)
                per_book[book] += 1
                progressed = True
                break
        if not progressed:
            raise BankError(
                f"only {len(taken)} lines can be chosen from {len(pool)} candidates in "
                f"{len(queues)} book(s) at {max_per_book} per book after near-duplicate removal; "
                f"the bank needs {n}")
    ordered = sorted(taken, key=lambda c: _rank(seed, "order", c["text"]))
    stats = {"pool_size": len(pool), "exact_duplicates_removed": n_exact,
             "near_duplicates_skipped": near_dups,
             "per_book": {b: k for b, k in sorted(per_book.items()) if k}}
    return ordered, stats


def bank_rows(lines, books):
    """lines.jsonl rows: id, text, book, title, source_url, est_seconds, chars. `books` maps a slug
    to its {title, source_url}."""
    rows = []
    for i, c in enumerate(lines, 1):
        b = books[c["book"]]
        rows.append({"id": f"{ID_PREFIX}{i:04d}", "text": c["text"], "book": c["book"],
                     "title": b["title"], "source_url": b["source_url"],
                     "est_seconds": round(est_seconds(c["text"]), 3), "chars": len(c["text"])})
    return rows


def canonical_sha256(rows):
    """See CANONICAL_FORM."""
    return hashlib.sha256("".join(r["text"] + "\n" for r in rows).encode("utf-8")).hexdigest()


def lines_jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in rows)

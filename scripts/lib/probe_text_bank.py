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

BALANCE. Selection takes turns across AUTHORS first, then across that author's books, so one
prolific author (three Lovecraft books) cannot fill a quarter of the bank; `MAX_PER_BOOK` stays a
ceiling. See `select_lines`.

HOLDOUT TITLES. A holdout book is matched over-inclusively (`title_key`, `match_title`): a false
exclusion costs one book of candidates, a missed one puts holdout text in the probe.

LEDGER. `books_ledger.json` records which lane each book is in; a book in the real-audio
(force-align) lane, or not in the ledger at all, is left out (`ledger_decision`), so this
synthetic bank stays disjoint from the real-audio lane.
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
from synth_common import ledger_key_candidates, norm_title  # noqa: E402

# The registered "at most about 8 s" of estimated speech. Inclusive, like the floor:
# MIN_CLIP_SECONDS <= chars / CHARS_PER_SEC <= MAX_CLIP_SECONDS.
MAX_CLIP_SECONDS = 8.0
BANK_SIZE = 300
# No book supplies more than this many lines (10% of the bank). A ceiling, not a target: selection
# is round-robin across authors and then across each author's books (`select_lines`), so the cap
# binds only when few authors qualify or some have few candidates.
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
    """ASCII, or the curly apostrophe and the en and em dashes of running prose."""
    return c.isascii() or c in "\u2019\u2013\u2014"


def has_non_ascii_letter(text):
    """True for any letter outside ASCII ("Angus Óg", "naïve", "Brontë"): an engine may read an
    accented name in another language's phonology, and the probe compares voices, not names."""
    return any(c.isalpha() and not c.isascii() for c in text)


# A question set with a full stop: a sentence that opens with an auxiliary and "not" ("Had not
# Cicely written…", "Did not the miller…") or a contracted negative auxiliary ("Wasn’t it…").
# Conservative on purpose: only these openings, which in prose are almost always questions. The
# rare declarative of this shape (a counterfactual "Had not the storm come, …") is lost too, which
# costs a candidate and nothing else.
_INVERSION_AUX = ("Had|Did|Was|Is|Were|Are|Could|Would|Should|Does|Do|Has|Have|Can|Will|Shall|"
                  "Might|Must")
_INVERSION_RE = re.compile(
    rf"^(?:(?:{_INVERSION_AUX})\s+not\b"
    r"|(?:Hadn|Didn|Wasn|Isn|Weren|Aren|Couldn|Wouldn|Shouldn|Doesn|Don|Hasn|Haven|Can|Won|Shan"
    r"|Mightn|Mustn|Ain)[\u2019']t\b)")


def is_interrogative_inversion(text):
    return bool(_INVERSION_RE.match(text))


# Exact source sentences (as the bank would carry them: cleaned and `normalize_speakable`-expanded)
# that pass every rule above and are still defective. Each is dropped before selection, and each
# hit is recorded in meta.json. Found in review of the first build (its ids in parentheses).
DENYLIST = {
    "At the first sight of Paula\u2019s face, she started erect and seem to forget her weakness "
    "in a moment.":
        "typo carried from the source: 'seem to forget' for 'seemed to forget' (1ci-0117)",
    "Screamingly sentient, dumbly delirious, only the gods that were can tell.":
        "verbless opening fragment; no reading of it is plain narration (1ci-0175)",
    "Had not Cicely written that he had gone West, soon after her own departure from New York.":
        "a question set with a full stop, so it would be read as a statement (1ci-0283)",
    "Indeed\u2014Danforth is closer mouthed than I: for he saw, or thinks he saw, one thing he "
    "will not tell even me.":
        "opens mid-argument ('Indeed\u2014') and splices its clauses with ': for'; out of context "
        "it does not read as a complete statement (1ci-0031)",
}


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
    if is_interrogative_inversion(text):
        return "interrogative_inversion"  # a question ending in a full stop
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
    if has_non_ascii_letter(text):
        return "non_ascii_letter"
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

# Words that carry no identity in a title: the overlap rules below need CONTENT words in common.
TITLE_STOPWORDS = frozenset("the a an of and or to in on at for by with from as".split())
# Two titles whose token sets overlap at least this much (and share a content word) are the same
# book, unless their authors are both known and differ. Catches "Center" / "Centre".
TITLE_JACCARD = 0.5
_TITLE_NUMERAL = (r"(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                  r"twelve|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)")
# "Volume 1", "Vol. IV", "Part II", "Book 01", "Edition 2": a tag naming one part of a work.
_VOLUME_TAG_RE = re.compile(r"\b(?:volume|vol|part|pt|book|bk|edition)\b\.?\s*" + _TITLE_NUMERAL
                            + r"\b")
_ORDINAL_EDITION_RE = re.compile(r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+"
                                 r"edition\b")
# LibriVox project titles only: "House of the Seven Gables (Version 2)".
_VERSION_TAG_RE = re.compile(r"\(\s*version\s+\w+\s*\)")
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def fold_accents(text):
    """NFKD, then drop the combining marks: "My Ántonia" -> "My Antonia"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text or "")
                   if not unicodedata.combining(c))


def title_key(title):
    """A title as the holdout match compares it: accents folded, a LibriVox `(version N)` and any
    volume / part / book / edition tag removed, then `synth_common.norm_title` (lowercase, `&` ->
    and, apostrophes deleted, other punctuation -> space, a leading article dropped).

    `norm_title` itself is shared with `scripts/tools/book_router.py` and is not changed here: the
    router must not equate different titles, while this key is deliberately over-inclusive, because
    a holdout book that slips into the probe costs more than a book wrongly left out.
    """
    t = fold_accents(title).lower()
    t = _VERSION_TAG_RE.sub(" ", t)
    t = _VOLUME_TAG_RE.sub(" ", t)
    t = _ORDINAL_EDITION_RE.sub(" ", t)
    return norm_title(t)


def _content_tokens(tokens):
    return [w for w in tokens if w not in TITLE_STOPWORDS]


def title_relation(a, b):
    """"equal", "prefix" or None for two `title_key`s.

    "prefix": one title's words are the leading words of the other's ("waverley" / "waverley or tis
    sixty years since"), and the shorter has at least 2 content words. A shorter title of a single
    content word must be equal: "erewhon" is not "erewhon revisited" by this rule.
    """
    if not a or not b:
        return None
    if a == b:
        return "equal"
    ta, tb = a.split(), b.split()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if long_[:len(short)] == short and len(_content_tokens(short)) >= 2:
        return "prefix"
    return None


def title_jaccard(a, b):
    """Jaccard similarity of two `title_key`s' word sets; 0 unless they share a content word."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb or not set(_content_tokens(sa & sb)):
        return 0.0
    return len(sa & sb) / len(sa | sb)


def author_surnames(author):
    """{surname} of each name in an author field: accents folded, lowercase, the last word of each
    comma-, `and`- or `&`-separated name, a Jr./Sr./II suffix ignored. "H. P. Lovecraft" ->
    {"lovecraft"}; BOOKS.txt's "Frederick Douglass, Frederick Augustus Washington Bailey" ->
    {"douglass", "bailey"}."""
    t = fold_accents(author).lower().replace("&", ",")
    t = re.sub(r"\band\b", ",", t)
    out = set()
    for name in t.split(","):
        words = [w for w in re.sub(r"[^\w\s]", " ", name).split() if w not in _NAME_SUFFIXES]
        if words:
            out.add(words[-1])
    return out


def authors_agree(a, b):
    """True when two author fields share a surname, False when both name someone and none is
    shared, None when either is empty (BOOKS.txt leaves the author blank for most of its rows)."""
    sa, sb = author_surnames(a), author_surnames(b)
    if not sa or not sb:
        return None
    return bool(sa & sb)


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
    """The LibriTTS source books of `chapters`: [{book_id, title, author, project_titles,
    chapters}], sorted by book_id; `author` is BOOKS.txt's third field, often empty. Refuses a
    chapter CHAPTERS.txt does not list, or lists under another reader or a subset other than
    dev-clean, rather than leave a holdout book unnamed."""
    by_chapter = {r[0]: r for r in _index_rows(chapters_txt) if r[0].isdigit()}
    titles = {r[0]: r[1] for r in _index_rows(books_txt) if r[0].isdigit() and len(r) > 1}
    authors = {r[0]: r[2] for r in _index_rows(books_txt) if r[0].isdigit() and len(r) > 2}
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
                                       "author": authors.get(book_id, ""),
                                       "project_titles": set(), "chapters": []})
        b["project_titles"].add(row[7])
        b["chapters"].append(f"{reader}/{chapter}")
    return [{**b, "project_titles": sorted(b["project_titles"])}
            for _k, b in sorted(books.items(), key=lambda kv: int(kv[0]))]


def holdout_title_keys(sources):
    """[{key, title, author}]: one entry per distinct `title_key` of each holdout book's BOOKS.txt
    title and LibriVox project titles; `title` is the BOOKS.txt title it names, `author` that
    book's BOOKS.txt author. In book order, so the first match reported is stable."""
    out, seen = [], set()
    for b in sources:
        for t in (b["title"], *b["project_titles"]):
            k = title_key(t)
            if k and (k, b["title"]) not in seen:
                seen.add((k, b["title"]))
                out.append({"key": k, "title": b["title"], "author": b.get("author") or ""})
    return out


def match_title(title, author, holdout_keys):
    """-> (holdout entry, rule) for the first holdout title `title` matches, else None.

    Rules, on `title_key`s: "equal"; "prefix" (`title_relation`); or "jaccard" — a word-set
    Jaccard of at least TITLE_JACCARD with a content word in common, unless `author` and the
    holdout book's author are both known and share no surname (`authors_agree` is False). An
    unknown author counts as agreeing: BOOKS.txt names no author for "A Journey to the Centre of
    the Earth", the case this rule exists for, nor for most of its rows.
    """
    k = title_key(title)
    if not k:
        return None
    for h in holdout_keys:
        rel = title_relation(k, h["key"])
        if rel:
            return h, rel
        j = title_jaccard(k, h["key"])
        if j >= TITLE_JACCARD and authors_agree(author, h["author"]) is not False:
            return h, f"jaccard {j:.2f}"
    return None


# ---------------------------------------------------------------- epub

_SKIP_TAGS = {"blockquote", "header", "hgroup", "footer", "figure", "figcaption", "table",
              "aside", "nav", "address"}
# Matched as whole `epub:type` tokens, with or without the `z3998:` vocabulary prefix.
_SKIP_TYPES = frozenset({"epigraph", "verse", "poem", "song", "hymn", "lyrics", "letter", "drama",
                         "stage-direction", "inscription", "footnote", "footnotes", "endnote",
                         "endnotes", "rearnote", "rearnotes", "note", "sign"})
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
        for token in _epub_type(anc).split():
            if token in _SKIP_TYPES or token.split(":", 1)[-1] in _SKIP_TYPES:
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


def holdout_match(book, holdout_keys, holdout_pg_ids):
    """Why `book` is a holdout source book, or None. `book` carries `title`, `fulltitle`, `author`
    and `pg_ids`: either title matches a holdout title (`match_title`), or it shares a Project
    Gutenberg id with a holdout book."""
    for field in ("title", "fulltitle"):
        hit = match_title(book.get(field), book.get("author"), holdout_keys)
        if hit:
            h, rule = hit
            return f"{field} {book.get(field)!r} matches holdout book {h['title']!r} ({rule})"
    shared = sorted(set(book.get("pg_ids") or ()) & set(holdout_pg_ids), key=int)
    if shared:
        return f"Project Gutenberg id(s) {shared} are holdout source books"
    return None


def book_candidates(slug, book, holdout_keys):
    """Candidate lines from one parsed book: (candidates, rejections, excluded_sections,
    denylist_hits).

    A document one of whose titles matches a holdout book's title (`match_title`; a story in a
    collection) is left out whole and reported. A sentence in DENYLIST is dropped and reported
    before any other rule looks at it. Candidates keep reading order and carry the book's author.
    """
    author = book["meta"].get("author") or ""
    cands, rejections, excluded, deny = [], {}, [], []
    for doc in book["documents"]:
        hit = next(((t, m) for t in doc["titles"]
                    for m in [match_title(t, author, holdout_keys)] if m), None)
        if hit:
            t, (h, rule) = hit
            excluded.append({"book": slug, "document": doc["href"], "title": t,
                             "holdout_title": h["title"], "rule": rule})
            continue
        for para in doc["paragraphs"]:
            if has_quote(clean_text(para)):
                rejections["quoted_paragraph"] = rejections.get("quoted_paragraph", 0) + 1
                continue
            for s in sentences_from_paragraph(para):
                if s in DENYLIST:
                    rejections["denylist"] = rejections.get("denylist", 0) + 1
                    deny.append({"book": slug, "text": s, "reason": DENYLIST[s]})
                    continue
                why = sentence_rejection(s)
                if why:
                    rejections[why] = rejections.get(why, 0) + 1
                    continue
                cands.append({"book": slug, "author": author, "text": s})
    return cands, rejections, excluded, deny


# ---------------------------------------------------------------- ledger

# What happens to a book `books_ledger.json` does not record: it is EXCLUDED. Its lane is unknown,
# so it might be (or become) a real-audio book, and the bank must stay disjoint from that lane.
NOT_IN_LEDGER = "excluded"
REAL_AUDIO_LANE = "force-align"
REAL_AUDIO_VERDICT = "REAL-AUDIO"


def _url_key(url):
    return (url or "").strip().rstrip("/").lower()


def ledger_entries(ledger, meta):
    """{key: entry} for every ledger entry that names this epub's book: one filed under a key
    `synth_common.ledger_key_candidates` derives from its Standard Ebooks identifier or any
    Gutenberg id its OPF lists, or one whose `source_url` is the identifier or whose `se_slug` is
    its author/title path. Over-inclusive on purpose; each match only ever excludes."""
    ident = meta.get("identifier") or ""
    pg_ids = list(meta.get("pg_ids") or ())
    keys = set(ledger_key_candidates(url=ident))
    for pg in pg_ids:
        keys.update(ledger_key_candidates(etext_id=pg))
    slug_path = _url_key(ident).split("standardebooks.org/ebooks/", 1)[-1] if ident else ""
    out = {}
    for k, e in (ledger or {}).items():
        if not isinstance(e, dict):
            continue
        gut = re.search(r"(?:ebooks|etext)/(\d+)", e.get("gutenberg_url") or "")
        if (k in keys or (ident and _url_key(e.get("source_url")) == _url_key(ident))
                or (slug_path and (e.get("se_slug") or "").strip("/").lower() == slug_path)
                or (gut and gut.group(1) in pg_ids)):
            out[k] = e
    return dict(sorted(out.items()))


def ledger_decision(entries):
    """-> (use, record). `use` is False for a book with no ledger entry (NOT_IN_LEDGER) or with
    any entry in the real-audio lane (`lane` force-align or `verdict` REAL-AUDIO). `record` lists
    each matched entry's key, lane and verdict, plus the reason when the book is left out."""
    record = {"ledger": [{"key": k, "lane": e.get("lane"), "verdict": e.get("verdict")}
                         for k, e in entries.items()]}
    if not entries:
        return False, {**record, "reason": f"not in the ledger ({NOT_IN_LEDGER})"}
    real = [k for k, e in entries.items()
            if e.get("lane") == REAL_AUDIO_LANE
            or (e.get("verdict") or "").upper() == REAL_AUDIO_VERDICT]
    if real:
        return False, {**record, "reason": f"real-audio lane in the ledger: {real}"}
    return True, record


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


def _author(c):
    """The candidate's author, or its book when none is recorded (each such book its own author)."""
    return c.get("author") or c["book"]


def exact_dedupe(candidates, *, seed=DEFAULT_SEED):
    """Keep one candidate per `dedupe_key`: the copy ranked first by sha256(seed, book, text), so
    which copy survives (and which book it credits) does not depend on input order.
    -> (kept, in rank order; n_dropped)."""
    best = {}
    for c in candidates:
        r = (_rank(seed, c["book"], c["text"]), c["book"], c["text"])
        k = dedupe_key(c["text"])
        if k not in best or r < best[k][0]:
            best[k] = (r, c)
    kept = [c for _r, c in sorted(best.values(), key=lambda rc: rc[0])]
    return kept, len(candidates) - len(kept)


def select_lines(candidates, *, n=BANK_SIZE, seed=DEFAULT_SEED, max_per_book=MAX_PER_BOOK):
    """Pick `n` lines from `candidates` (dicts with `book`, `text` and optionally `author`)
    deterministically.

    Each candidate's rank is sha256(seed, book, text), so the result depends on the candidate SET
    and the seed, not on input order (`exact_dedupe` keeps the best-ranked copy of a duplicate).
    Authors take turns in name order; on its turn an author offers its next book in slug order
    (round-robin over its books), skipping a book that is exhausted or holds `max_per_book`
    lines, and that book gives its best-ranked line that is not a near-duplicate of one already
    taken. So each author supplies about n / authors lines, spread evenly over its books, and
    `max_per_book` is a ceiling. An author that runs out drops out of the turns and the rest
    continue in the same order; when `n` does not divide evenly, the authors first in name order
    take the extra line. The chosen lines are then ordered by sha256(seed, "order", text): render
    order matters to the drift criterion, so no book or author is tied to a position.

    -> (lines in bank order, stats). Raises BankError when fewer than `n` lines qualify.
    """
    pool, n_exact = exact_dedupe(candidates, seed=seed)
    if len(pool) < n:
        raise BankError(f"only {len(pool)} candidate lines qualify (after removing {n_exact} "
                        f"exact duplicates); the bank needs {n}")
    queues, author_of = {}, {}
    for c in pool:
        queues.setdefault(c["book"], []).append(c)
        if author_of.setdefault(c["book"], _author(c)) != _author(c):
            raise ValueError(f"book {c['book']!r} has two authors: {author_of[c['book']]!r}, "
                             f"{_author(c)!r}")
    for q in queues.values():
        q.sort(key=lambda c: (_rank(seed, c["book"], c["text"]), c["text"]))
    books_of = {}
    for b in sorted(queues):
        books_of.setdefault(author_of[b], []).append(b)
    cursor, turn = dict.fromkeys(queues, 0), dict.fromkeys(books_of, 0)
    per_book, per_author = dict.fromkeys(queues, 0), dict.fromkeys(books_of, 0)
    taken, shingles, near_dups = [], [], 0

    def take_from(book):
        nonlocal near_dups
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
            per_author[author_of[book]] += 1
            return True
        return False

    while len(taken) < n:
        progressed = False
        for author in sorted(books_of):
            if len(taken) >= n:
                break
            books = books_of[author]
            for i in range(len(books)):
                book = books[(turn[author] + i) % len(books)]
                if per_book[book] >= max_per_book:
                    continue
                if take_from(book):
                    turn[author] = (turn[author] + i + 1) % len(books)
                    progressed = True
                    break
        if not progressed:
            raise BankError(
                f"only {len(taken)} lines can be chosen from {len(pool)} candidates in "
                f"{len(queues)} book(s) by {len(books_of)} author(s) at {max_per_book} per book "
                f"after near-duplicate removal; the bank needs {n}")
    ordered = sorted(taken, key=lambda c: _rank(seed, "order", c["text"]))
    stats = {"pool_size": len(pool), "exact_duplicates_removed": n_exact,
             "near_duplicates_skipped": near_dups,
             "per_author": {a: k for a, k in sorted(per_author.items()) if k},
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

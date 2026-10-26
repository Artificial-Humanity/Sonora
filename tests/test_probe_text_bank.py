"""Step 1c-i's text bank: `scripts/lib/probe_text_bank.py` and `scripts/tools/build_1ci_text_bank.py`.

Synthetic fixtures throughout (sentences, a minimal Standard Ebooks-shaped epub, LibriTTS index
files); the tests at the end that read the real `/data` sources skip when those are absent.
"""
import io
import json
import pathlib
import random
import zipfile

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import book_ingest  # noqa: E402
import build_1ci_text_bank as tool  # noqa: E402
import probe_text_bank as ptb  # noqa: E402

CPS = ptb.CHARS_PER_SEC
MIN_CHARS = round(ptb.MIN_CLIP_SECONDS * CPS)      # 56 at 14 chars/s and 4 s
MAX_CHARS = round(ptb.MAX_CLIP_SECONDS * CPS)      # 112 at 14 chars/s and 8 s

VOCAB = ("river mill stone bridge lantern meadow harbour orchard window candle garden letter "
         "morning evening winter summer valley forest village tower carriage traveller kitchen "
         "chimney market fiddle ribbon pocket shadow mirror ladder kettle blanket cottage "
         "steeple hedgerow pebble thistle sparrow heron willow cellar attic parlour doorway "
         "staircase hillside quarry ferry cart barrel saddle bucket lamp table cupboard "
         "fireside hearth woollen silver copper velvet linen marble timber granite meadowlark "
         "wandered rested waited lingered gathered carried followed noticed watched remembered "
         "climbed crossed opened closed folded mended painted polished swept counted").split()


def sentence(rng, lo=60, hi=100):
    """A plain declarative sentence of `lo`..`hi` characters from VOCAB."""
    while True:
        words = ["The"] + [rng.choice(VOCAB) for _ in range(30)]
        s = words[0]
        for w in words[1:]:
            if len(s) + 1 + len(w) + 1 > hi:
                break
            s += " " + w
        s += "."
        if lo <= len(s) <= hi:
            return s


def sized(n_chars):
    """A plain sentence of exactly `n_chars` characters."""
    s = "The quiet river ran past the old mill and the green fields below the stone bridge"
    s = " ".join(s.split())
    while len(s) + 1 < n_chars:
        s += " and the hills"
    s = s[: n_chars - 1].rstrip()
    while len(s) < n_chars - 1:
        s += "o"
    return s + "."


# ---------------------------------------------------------------- length


def test_the_floor_is_book_ingests_and_not_a_copy():
    assert ptb.MIN_CLIP_SECONDS is book_ingest.MIN_CLIP_SECONDS
    assert ptb.CHARS_PER_SEC is book_ingest.CHARS_PER_SEC
    src = SCRIPTS.src("probe_text_bank.py")
    assert "MIN_CLIP_SECONDS =" not in src and "CHARS_PER_SEC =" not in src
    assert ptb.MAX_CLIP_SECONDS == 8.0


@pytest.mark.parametrize("n_chars,why", [
    (MIN_CHARS - 1, "too_short"),
    (MIN_CHARS, None),
    (MAX_CHARS, None),
    (MAX_CHARS + 1, "too_long"),
])
def test_length_bounds_include_both_edges(n_chars, why):
    text = sized(n_chars)
    assert len(text) == n_chars
    assert ptb.sentence_rejection(text) == why
    if why is None:
        assert ptb.MIN_CLIP_SECONDS <= ptb.est_seconds(text) <= ptb.MAX_CLIP_SECONDS


def test_the_edges_are_exactly_the_registered_seconds():
    assert ptb.est_seconds(sized(MIN_CHARS)) == ptb.MIN_CLIP_SECONDS
    assert ptb.est_seconds(sized(MAX_CHARS)) == ptb.MAX_CLIP_SECONDS


# ---------------------------------------------------------------- what a narrator reads


@pytest.mark.parametrize("text,why", [
    ("“Come here at once and sit down by the fire,” she said to the boy.", "quotation"),
    ("He wrote the word ‘home’ across the top of the long white page.", "quotation"),
    ('The sign above the door said "Closed" in large and faded letters today.', "quotation"),
    ("Why had the old miller left the gate of the orchard open all night long?", "not_declarative"),
    ("The whole valley shook as the great tower came down at last upon them!", "not_declarative"),
    ("The carriage stopped, and the traveller looked out at the dark road…", "not_declarative"),
    ("The carriage arrived at the inn in 1842 with the post and three travellers.", "digits"),
    ("The figures are given in the table below, see Fig. 12 for the full detail.", None),
    ("The miller (who was asleep) never heard the carriage come into the yard.", "symbol"),
    ("The letter was signed by the secretary of the RSPCA and dated last week.", "all_caps"),
    ("The parcel came from H. A. Wilcox of Providence on the morning train.", "initials"),
    ("the miller never heard the carriage come into the yard that cold night.", "incomplete"),
])
def test_sentence_rejections(text, why):
    got = ptb.sentence_rejection(text)
    if why is None:      # the illustration case: whatever fires first, it is rejected
        assert got is not None
    else:
        assert got == why


def test_references_the_invisible_is_applied(monkeypatch):
    text = "The plan of the cellar is shown on the last page of this little volume."
    assert ptb.sentence_rejection(text) is None                       # control
    monkeypatch.setattr(ptb, "references_the_invisible", lambda t: True)
    assert ptb.sentence_rejection(text) == "references_the_invisible"


def test_a_paragraph_that_quotes_anyone_yields_nothing():
    para = ("The old man turned slowly towards the window of the cottage. "
            "“It will rain before the evening is out,” he said quietly. "
            "Then he sat down again beside the fire and folded his hands.")
    assert ptb.sentences_from_paragraph(para) == []
    control = para.replace("“It will rain before the evening is out,” he said quietly. ",
                           "")
    assert ptb.sentences_from_paragraph(control) == [
        "The old man turned slowly towards the window of the cottage.",
        "Then he sat down again beside the fire and folded his hands."]


def test_text_is_cleaned_and_made_speakable():
    raw = "The\u00a0letter came\u2060\u2014by the late post\u2060\u2014from Madison, Wis. today."
    [s] = ptb.sentences_from_paragraph(raw)
    assert s == "The letter came—by the late post—from Madison, Wisconsin today."


# ---------------------------------------------------------------- epub


def make_epub(*, title="The Quiet Valley", identifier="https://standardebooks.org/ebooks/a/b",
              publisher="Standard Ebooks", pg_ids=("11111",), documents=None, fulltitle=None):
    """A minimal Standard Ebooks-shaped epub: container, OPF with dc metadata and a spine, and one
    XHTML file per (head title, body epub:type, inner xhtml) in `documents`."""
    documents = documents or []
    ns = ('xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="epub/content.opf" media-type="application/oebps-package+xml"/>'
                   "</rootfiles></container>")
        items, refs = [], []
        for i, (head, btype, inner) in enumerate(documents):
            href = f"text/doc-{i}.xhtml"
            items.append(f'<item id="d{i}" href="{href}" media-type="application/xhtml+xml"/>')
            refs.append(f'<itemref idref="d{i}"/>')
            z.writestr(f"epub/{href}",
                       f'<?xml version="1.0" encoding="utf-8"?><html {ns}><head><title>{head}'
                       f'</title></head><body epub:type="{btype}">{inner}</body></html>')
        srcs = "".join(f"<dc:source>https://www.gutenberg.org/ebooks/{p}</dc:source>"
                       for p in pg_ids)
        full = f'<dc:title id="fulltitle">{fulltitle}</dc:title>' if fulltitle else ""
        z.writestr("epub/content.opf",
                   '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf"'
                   ' version="3.0" unique-identifier="uid"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:identifier id="uid">{identifier}</dc:identifier>'
                   f'<dc:title id="title">{title}</dc:title>{full}'
                   f'<dc:creator id="author">A. Writer</dc:creator>'
                   f'<dc:publisher id="publisher">{publisher}</dc:publisher>{srcs}</metadata>'
                   f'<manifest>{"".join(items)}</manifest><spine>{"".join(refs)}</spine></package>')
    return buf.getvalue()


def prose_doc(paras, head="Chapter I", btype="bodymatter z3998:fiction"):
    return (head, btype, '<section epub:type="chapter"><h2>' + head + "</h2>"
            + "".join(f"<p>{p}</p>" for p in paras) + "</section>")


def test_read_epub_keeps_only_running_bodymatter_prose():
    keep = "The road ran down past the mill and over the old stone bridge to the town."
    inner = (
        '<section epub:type="chapter"><h2>Chapter I</h2>'
        f"<p>{keep}</p>"
        '<blockquote epub:type="z3998:poem"><p><span>The wind went over the hill and far away '
        "into the evening light.</span></p></blockquote>"
        '<blockquote epub:type="z3998:letter"><p>The weather here has been fine for the whole of '
        "the week and we are all well.</p></blockquote>"
        '<p>The waiter brought a dish he called <i xml:lang="fr">pommes de terre</i> with the '
        "cold meat.</p>"
        '<div epub:type="z3998:verse"><p>Over the river and through the wood the carriage went '
        "along.</p></div>"
        "</section>")
    data = make_epub(documents=[
        ("Imprint", "frontmatter", "<p>This ebook is the product of many hours of work by "
                                   "volunteers for the good of all.</p>"),
        ("Chapter I", "bodymatter z3998:fiction", inner),
    ])
    book = ptb.read_epub(data)
    assert ptb.is_standard_ebooks(book["meta"])
    assert book["meta"]["pg_ids"] == ["11111"]
    assert [d["paragraphs"] for d in book["documents"]] == [[keep]]
    assert "Chapter I" in book["documents"][0]["titles"]


@pytest.mark.parametrize("identifier,publisher,ok", [
    ("https://standardebooks.org/ebooks/a/b", "Standard Ebooks", True),
    ("https://www.gutenberg.org/ebooks/768", "Standard Ebooks", False),
    ("https://standardebooks.org/ebooks/a/b", "Project Gutenberg", False),
])
def test_provenance_is_read_from_the_epub(identifier, publisher, ok):
    meta = ptb.read_epub(make_epub(identifier=identifier, publisher=publisher))["meta"]
    assert ptb.is_standard_ebooks(meta) is ok


# ---------------------------------------------------------------- holdout

CHAPTERS_TXT = """; header
;ID    |READER|MINUTES| SUBSET           | PROJ.|BOOK ID| CH. TITLE | PROJECT TITLE
100    | 7    | 10.00 | dev-clean        | 5    | 1184  | Chapter 1 | Count of Monte Cristo
101    | 7    | 10.00 | dev-clean        | 5    | 1184  | Chapter 2 | Count of Monte Cristo
200    | 8    | 10.00 | dev-clean        | 6    | 77    | Chapter 1 | House of the Seven Gables (Version 2)
300    | 9    | 10.00 | train-clean-100  | 7    | 99    | Chapter 1 | Unrelated Book
"""
BOOKS_TXT = """; header
1184   | The Count of Monte Cristo | Alexandre Dumas
77     | House of the Seven Gables  |
99     | Unrelated Book             |
"""


def holdout_txt(pairs):
    return "".join(f"/x/LibriTTS_R/dev-clean/{r}/{c}/{r}_{c}_000001_000000.wav|6|t|0,0,0\n"
                   for r, c in pairs)


def test_holdout_paths_map_to_their_source_books():
    chapters = ptb.holdout_chapters(holdout_txt([("7", "100"), ("7", "101"), ("8", "200")]))
    sources = ptb.holdout_sources(chapters, CHAPTERS_TXT, BOOKS_TXT)
    assert [(s["book_id"], s["title"]) for s in sources] == [
        ("77", "House of the Seven Gables"), ("1184", "The Count of Monte Cristo")]
    keys = ptb.holdout_title_keys(sources)
    assert {"count of monte cristo", "house of the seven gables"} <= set(keys)


@pytest.mark.parametrize("pairs,match", [
    ([("7", "999")], "not in CHAPTERS"),       # unknown chapter
    ([("8", "100")], "reader 7"),              # wrong reader
    ([("9", "300")], "subset train-clean-100"),
])
def test_holdout_mapping_refuses_what_it_cannot_name(pairs, match):
    with pytest.raises(ValueError, match=match):
        ptb.holdout_sources(ptb.holdout_chapters(holdout_txt(pairs)), CHAPTERS_TXT, BOOKS_TXT)


def test_a_non_dev_clean_holdout_row_is_refused():
    with pytest.raises(ValueError, match="dev-clean"):
        ptb.holdout_chapters("/x/LibriTTS_R/train-clean-100/1/2/a.wav|6|t\n")


@pytest.fixture
def holdout_keys():
    sources = ptb.holdout_sources(ptb.holdout_chapters(holdout_txt([("7", "100"), ("8", "200")])),
                                  CHAPTERS_TXT, BOOKS_TXT)
    return ptb.holdout_title_keys(sources), sorted({s["book_id"] for s in sources}, key=int)


@pytest.mark.parametrize("book,hit", [
    ({"title": "The Count of Monte Cristo", "pg_ids": []}, True),
    ({"title": "Count of Monte-Cristo", "pg_ids": []}, True),
    ({"title": "The House of the Seven Gables", "pg_ids": []}, True),
    ({"title": "Gables", "fulltitle": "The House of the Seven Gables: A Romance", "pg_ids": []},
     False),
    ({"title": "Some Other Book", "pg_ids": ["1184"]}, True),     # same Gutenberg edition
    ({"title": "Some Other Book", "pg_ids": ["1185"]}, False),
    ({"title": "The Count", "pg_ids": []}, False),
])
def test_holdout_title_match(holdout_keys, book, hit):
    keys, pg = holdout_keys
    assert (ptb.holdout_match(book, keys, pg) is not None) is hit


def test_a_holdout_story_inside_a_collection_is_excluded(holdout_keys):
    keys, _pg = holdout_keys
    rng = random.Random(1)
    good = [sentence(rng) for _ in range(3)]
    bad = [sentence(rng) for _ in range(3)]
    book = ptb.read_epub(make_epub(documents=[
        prose_doc(good, head="The Quiet Valley"),
        prose_doc(bad, head="The Count of Monte Cristo"),
    ]))
    cands, _rej, excluded = ptb.book_candidates("coll", book, keys)
    assert [c["text"] for c in cands] == good
    assert [e["holdout_title"] for e in excluded] == ["The Count of Monte Cristo"]


# ---------------------------------------------------------------- selection


def pool(n_books, per_book, seed=7):
    rng = random.Random(seed)
    return [{"book": f"book-{b:02d}", "text": sentence(rng)}
            for b in range(n_books) for _ in range(per_book)]


def test_selection_is_deterministic_and_order_independent():
    cands = pool(13, 40)
    a, _ = ptb.select_lines(cands, seed=5)
    b, _ = ptb.select_lines(list(reversed(cands)), seed=5)
    sha = lambda lines: ptb.canonical_sha256(ptb.bank_rows(  # noqa: E731
        lines, {c["book"]: {"title": "t", "source_url": "u"} for c in cands}))
    assert len(a) == ptb.BANK_SIZE
    assert sha(a) == sha(b)
    c, _ = ptb.select_lines(cands, seed=6)
    assert sha(c) != sha(a)


def test_the_canonical_hash_is_the_text_lines():
    import hashlib
    rows = [{"text": "One line."}, {"text": "Two lines."}]
    assert ptb.canonical_sha256(rows) == hashlib.sha256(b"One line.\nTwo lines.\n").hexdigest()


def test_books_take_turns_and_the_cap_binds():
    """Round-robin keeps books level; the cap is what stops a big book taking up the slack when the
    small ones run out. Two 400-line books beside ten 10-line books would supply 100 lines each
    uncapped; at the cap the bank cannot be made, and selection refuses rather than exceed it."""
    even, stats = ptb.select_lines(pool(13, 40))
    per = stats["per_book"]
    assert sum(per.values()) == ptb.BANK_SIZE
    assert max(per.values()) - min(per.values()) <= 1
    assert max(per.values()) <= ptb.MAX_PER_BOOK

    big = [dict(c, book=f"big-{i % 2}") for i, c in enumerate(pool(1, 800, seed=1))]
    small = [dict(c, book=f"small-{i % 10}") for i, c in enumerate(pool(1, 100, seed=2))]
    _lines, uncapped = ptb.select_lines(big + small, max_per_book=10**6)
    assert uncapped["per_book"]["big-0"] > ptb.MAX_PER_BOOK      # what the cap prevents
    with pytest.raises(ptb.BankError, match=f"at {ptb.MAX_PER_BOOK} per book"):
        ptb.select_lines(big + small)


def test_the_cap_can_make_the_bank_impossible():
    with pytest.raises(ptb.BankError, match="per book"):
        ptb.select_lines(pool(5, 100), max_per_book=30)       # 5 x 30 = 150 < 300


def test_refuses_fewer_than_the_bank_size():
    with pytest.raises(ptb.BankError, match="only 299 candidate lines qualify"):
        ptb.select_lines(pool(13, 23)[:299])


def test_exact_duplicates_are_removed_from_the_pool():
    cands = pool(13, 40)
    dup = dict(cands[0], book="book-07", text=cands[0]["text"].lower())   # same words
    kept, dropped = ptb.exact_dedupe(cands + [dup])
    assert dropped == 1 and kept == cands
    _lines, stats = ptb.select_lines(cands + [dup])
    assert stats["exact_duplicates_removed"] == 1
    assert stats["pool_size"] == len(cands)


def test_a_near_duplicate_is_never_selected_beside_its_twin():
    rng = random.Random(3)
    base = sentence(rng, 70, 90)
    near = base[:-1] + " again."
    cands = [{"book": "a", "text": base}, {"book": "b", "text": near}]
    cands += [{"book": "c", "text": sentence(rng)} for _ in range(5)]
    with pytest.raises(ptb.BankError):
        ptb.select_lines(cands, n=7, max_per_book=10)
    lines, stats = ptb.select_lines(cands, n=6, max_per_book=10)
    texts = {c["text"] for c in lines}
    assert not {base, near} <= texts
    assert stats["near_duplicates_skipped"] >= 1


# ---------------------------------------------------------------- the tool


def write_fixture(root, *, n_books=3, per_book=6, holdout_title=None):
    """A books root of synthetic Standard Ebooks epubs and LibriTTS-shaped holdout files."""
    rng = random.Random(11)
    books = root / "book-prose"
    for b in range(n_books):
        d = books / f"book-{b}"
        d.mkdir(parents=True)
        title = holdout_title if (holdout_title and b == 0) else f"Quiet Valley Volume {b}"
        (d / "book.epub").write_bytes(make_epub(
            title=title, identifier=f"https://standardebooks.org/ebooks/a/book-{b}",
            pg_ids=(str(50000 + b),),
            documents=[prose_doc([sentence(rng) for _ in range(per_book)])]))
    (books / "real-audio").mkdir()
    ht = root / "holdout.txt"
    ht.write_text(holdout_txt([("7", "100"), ("8", "200")]))
    (root / "CHAPTERS.txt").write_text(CHAPTERS_TXT)
    (root / "BOOKS.txt").write_text(BOOKS_TXT)
    return ["--books-root", str(books), "--holdout", str(ht),
            "--chapters", str(root / "CHAPTERS.txt"), "--books-txt", str(root / "BOOKS.txt"),
            "--allow-dirty"]


def test_tool_writes_the_bank_and_its_meta(tmp_path):
    argv = write_fixture(tmp_path, holdout_title="The Count of Monte Cristo")
    out = tmp_path / "out"
    assert tool.main(argv + ["--out", str(out), "--n", "10"]) == 0
    rows = [json.loads(ln) for ln in (out / "lines.jsonl").read_text().splitlines()]
    meta = json.loads((out / "meta.json").read_text())
    assert len(rows) == 10
    assert set(rows[0]) == {"id", "text", "book", "title", "source_url", "est_seconds", "chars"}
    assert meta["sha256"] == ptb.canonical_sha256(rows)
    assert {"book-1", "book-2"} == set(meta["books_used"])
    assert [e["book"] for e in meta["exclusions"]["books"]] == ["book-0"]
    assert meta["exclusions"]["any"] is True
    assert {"book": "real-audio", "reason": "no local book.epub"} in meta["books_skipped"]
    assert [s["title"] for s in meta["holdout"]["source_books"]] == [
        "House of the Seven Gables", "The Count of Monte Cristo"]
    assert meta["pool"]["pool_size"] == 12
    assert meta["constants"]["MIN_CLIP_SECONDS"] == book_ingest.MIN_CLIP_SECONDS
    assert all(ptb.MIN_CLIP_SECONDS <= r["est_seconds"] <= ptb.MAX_CLIP_SECONDS for r in rows)


def test_tool_refuses_fewer_than_300(tmp_path, capsys):
    argv = write_fixture(tmp_path)
    with pytest.raises(SystemExit) as e:
        tool.main(argv + ["--out", str(tmp_path / "out")])
    assert e.value.code == 2
    assert "the bank needs 300" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_tool_refuses_to_overwrite_a_different_bank(tmp_path, capsys):
    argv = write_fixture(tmp_path)
    out = tmp_path / "out"
    tool.main(argv + ["--out", str(out), "--n", "10"])
    before = (out / "lines.jsonl").read_bytes(), (out / "meta.json").read_bytes()
    assert tool.main(argv + ["--out", str(out), "--n", "10"]) == 0          # identical: no-op
    assert "unchanged" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        tool.main(argv + ["--out", str(out), "--n", "10", "--seed", "99"])
    assert e.value.code == 2
    assert "differs" in capsys.readouterr().err
    assert ((out / "lines.jsonl").read_bytes(), (out / "meta.json").read_bytes()) == before


def test_tool_refuses_a_meta_without_lines(tmp_path):
    argv = write_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "meta.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        tool.main(argv + ["--out", str(out), "--n", "10"])
    assert e.value.code == 2


def test_tool_skips_a_book_whose_bank_names_another_source(tmp_path):
    argv = write_fixture(tmp_path, n_books=3, per_book=6)
    d = tmp_path / "book-prose" / "book-1"
    (d / "book-1_bank.json").write_text(json.dumps({"source": {"url": "https://elsewhere/x"}}))
    out = tmp_path / "out"
    tool.main(argv + ["--out", str(out), "--n", "10"])
    meta = json.loads((out / "meta.json").read_text())
    assert "book-1" not in meta["books_used"]
    assert any(s["book"] == "book-1" and "differs" in s["reason"] for s in meta["books_skipped"])


# ---------------------------------------------------------------- real sources (skip when absent)

REAL_EPUB = pathlib.Path(tool.BOOKS_ROOT) / "mike" / "book.epub"
REAL_INDEX = pathlib.Path(tool.LIBRITTS)


@pytest.mark.skipif(not REAL_EPUB.is_file(), reason=f"{REAL_EPUB} absent")
def test_a_real_standard_ebooks_epub_yields_plain_narration():
    book = ptb.read_epub(REAL_EPUB.read_bytes())
    assert ptb.is_standard_ebooks(book["meta"])
    cands, rej, _ = ptb.book_candidates("mike", book, {})
    assert len(cands) > ptb.MAX_PER_BOOK
    assert rej.get("quoted_paragraph", 0) > 0          # positive control: the filter fired
    assert all(ptb.sentence_rejection(c["text"]) is None for c in cands)


@pytest.mark.skipif(not (pathlib.Path(tool.HOLDOUT).is_file()
                         and (REAL_INDEX / "CHAPTERS.txt").is_file()),
                    reason="holdout filelist or LibriTTS index absent")
def test_the_real_holdout_names_its_source_books():
    sources = tool.load_holdout(tool.HOLDOUT, REAL_INDEX / "CHAPTERS.txt",
                                REAL_INDEX / "BOOKS.txt")
    keys = ptb.holdout_title_keys(sources)
    assert len(sources) >= 40
    assert ptb.holdout_match({"title": "The Count of Monte Cristo"}, keys, []) is not None

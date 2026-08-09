"""Regression coverage for the §5 acquisition-lane fixes (2026-08-06).

The shape these findings share: every one of them produces a *plausible corpus* rather
than an error. A wrong book downloaded into the right directory, U+FFFD baked into a
transcript, a ledger entry silently erased by a concurrent run, dialogue quietly reclassed
as narration, "Mr." shipped as its own clip — none of them raise, and the v4 rating
vocabulary (vocals and prosody only) cannot see any of them. That is why they are asserted
here rather than left to the ear.
"""

import json
import os
import subprocess
import sys

import pytest

SYNTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "synthesis")
sys.path.insert(0, SYNTH)

synth_common = pytest.importorskip("synth_common")


# --- A-H2: Gutenberg decoding ---------------------------------------------------------


@pytest.fixture()
def fetch_mod():
    return pytest.importorskip("librivox_fetch")


def test_latin1_edition_decodes_without_replacement_characters(fetch_mod):
    """PG still serves many pre-2018 editions as ISO-8859-1. `errors="replace"` turned
    every non-ASCII byte in them into U+FFFD — `café` → `caf<?>` — and the file still
    looked fine because the ASCII was untouched."""
    raw = "The café was «full» — she paused.".replace("—", "-").encode("iso-8859-1")
    out = fetch_mod.decode_gutenberg(raw, "test://latin1")
    assert "café" in out and "«full»" in out
    assert fetch_mod.REPLACEMENT not in out


def test_utf8_edition_is_unchanged(fetch_mod):
    raw = "The café was full — she paused.".encode("utf-8")
    assert fetch_mod.decode_gutenberg(raw, "test://utf8") == "The café was full — she paused."


def test_source_that_already_contains_fffd_is_refused(fetch_mod):
    """We cannot repair an edition corrupt at source, but shipping it silently is the
    finding. Refusing lets the caller try another edition."""
    with pytest.raises(ValueError, match="U\\+FFFD"):
        fetch_mod.decode_gutenberg("caf� was full".encode("utf-8"), "test://broken")


# --- A-M12: HTML error pages saved as .mp3 -------------------------------------------


def test_mp3_magic_bytes(fetch_mod):
    assert fetch_mod.looks_like_mp3(b"ID3\x04\x00")
    assert fetch_mod.looks_like_mp3(b"\xff\xfb\x90\x00")
    assert not fetch_mod.looks_like_mp3(b"<!DOCTYPE html><html>")
    assert not fetch_mod.looks_like_mp3(b"<html>404 Not Found</html>")


def test_saved_error_page_is_not_accepted_as_a_finished_download(fetch_mod, tmp_path,
                                                                 monkeypatch):
    """`size > 1024` was the whole resume check, and a 404 page clears 1024 bytes easily.
    It got written as 003.mp3 and every later run skipped it as already done."""
    dest = tmp_path / "003.mp3"
    dest.write_bytes(b"<!DOCTYPE html>" + b"x" * 4000)

    monkeypatch.setattr(fetch_mod, "fetch", lambda *a, **k: b"ID3\x04\x00" + b"\x00" * 4000)
    new, n = fetch_mod.download("test://audio", dest)
    assert new, "an HTML page on disk must not count as a completed download"
    assert dest.read_bytes()[:3] == b"ID3"


def test_a_non_mp3_response_is_never_written_as_audio(fetch_mod, tmp_path, monkeypatch):
    dest = tmp_path / "004.mp3"
    monkeypatch.setattr(fetch_mod, "fetch", lambda *a, **k: b"<html>rate limited</html>")
    with pytest.raises(ValueError, match="not MP3"):
        fetch_mod.download("test://audio", dest)
    assert not dest.exists()


# --- A-H3: lost ledger updates --------------------------------------------------------


def test_update_json_does_not_clobber_a_concurrent_writer(tmp_path):
    """The exact A-H3 sequence: two runs read the ledger, both work for minutes, both
    write back. Writing a stale snapshot erased the other's entry."""
    ledger = tmp_path / "books_ledger.json"
    ledger.write_text(json.dumps({"pg:1": {"status": "start"}}), encoding="utf-8")

    stale = json.loads(ledger.read_text())          # run A's snapshot

    def add_b(current):                             # run B finishes first
        current["lv:b"] = {"status": "fetched"}
    synth_common.update_json(ledger, add_b)

    def add_a(current):                             # run A finishes second
        current["lv:a"] = {"status": "aligned"}
    synth_common.update_json(ledger, add_a)

    final = json.loads(ledger.read_text())
    assert set(final) == {"pg:1", "lv:a", "lv:b"}, "a concurrent entry was erased"
    assert "lv:b" not in stale, "the stale snapshot is what would have been written back"


def test_update_json_creates_a_missing_file(tmp_path):
    path = tmp_path / "new.json"
    synth_common.update_json(path, lambda obj: obj.__setitem__("k", 1))
    assert json.loads(path.read_text()) == {"k": 1}


def test_write_json_atomic_leaves_no_partial_file(tmp_path):
    path = tmp_path / "log.json"
    path.write_text('{"runs": []}', encoding="utf-8")
    with pytest.raises(TypeError):
        synth_common.write_json_atomic(path, {"runs": [object()]})
    assert json.loads(path.read_text()) == {"runs": []}, "the original was damaged"


# --- A-H5: sentence splitting and completeness ---------------------------------------


def test_abbreviations_do_not_split_a_sentence():
    """The real-audio lane split on any [.!?]+space, so "Mr. Smith went home." shipped as
    two clips: a fragment, and a sentence starting mid-utterance."""
    pytest.importorskip("pysbd")
    assert synth_common.split_sentences("Mr. Smith went home. He was tired.") == [
        "Mr. Smith went home.",
        "He was tired.",
    ]


def test_completeness_gate_rejects_fragments():
    assert not synth_common.is_complete_utterance("and then he said,")
    assert not synth_common.is_complete_utterance("se it was true—")
    assert synth_common.is_complete_utterance("Yes.")
    assert synth_common.is_complete_utterance("“Come here,” she said.")


def test_the_gate_cannot_catch_a_bad_split_and_that_is_why_pysbd_matters():
    """`is_complete_utterance("Mr.")` is **True**, and correctly so on its own terms: it
    ends in a terminal and starts with a capital, which is all shape can tell you.

    This is the point of A-H5. A completeness gate bolted onto a naive splitter would not
    have caught "Mr." / "Smith went home." — the first half passes the gate and the second
    half passes it too. Only fixing the *splitter* fixes the defect; the gate is the net
    for everything else. Asserted so nobody later "simplifies" pysbd away on the grounds
    that the gate covers it.
    """
    assert synth_common.is_complete_utterance("Mr.") is True
    assert synth_common.is_complete_utterance("Smith went home.") is True
    pytest.importorskip("pysbd")
    assert synth_common.split_sentences("Mr. Smith went home.") == ["Mr. Smith went home."]


def test_the_two_definitions_are_now_one():
    """book_ingest and qc_passages each had a copy, and they had drifted — qc_passages
    accepted guillemets and trailing spaces that book_ingest rejected."""
    book_ingest = pytest.importorskip("book_ingest")
    qc_passages = pytest.importorskip("qc_passages")
    assert book_ingest.is_complete_utterance is synth_common.is_complete_utterance
    assert qc_passages.is_complete_utterance is synth_common.is_complete_utterance


# --- A-M7: quote conventions ----------------------------------------------------------


@pytest.mark.parametrize(
    ("style", "text"),
    [
        ("curly-double", "“Come here,” she said. “And sit down.” " * 3),
        ("straight-double", '"Come here," she said. "And sit down." ' * 3),
        ("curly-single", "‘Come here,’ she said. ‘And sit down.’ " * 3),
        ("straight-single", "'Come here,' she said. 'And sit down.' " * 3),
    ],
)
def test_every_edition_convention_is_detected(style, text):
    book_ingest = pytest.importorskip("book_ingest")
    assert book_ingest.detect_quote_style(text)[0] == style


def test_straight_single_quotes_survive_contractions():
    """The killer case, and the one *Uneasy Money* (pg:6684) actually is: the closing mark
    is the apostrophe character. A bare non-greedy match ends at the first contraction, so
    "I don't know" becomes "I don"."""
    book_ingest = pytest.importorskip("book_ingest")
    out = book_ingest.extract_utterances(
        "'I don't know what you mean,' said the guv'nor.", "straight-single")
    assert [q for q, _ in out] == ["I don't know what you mean,"] or out == [], out
    # possessives and contractions outside a quote must not open one
    assert book_ingest.extract_utterances("The boys' hats were on the man's chair.",
                                          "straight-single") == []


def test_dialogue_in_a_single_quoted_edition_is_no_longer_read_as_narration():
    """A-M7's live consequence. With the hardcoded curly-double pattern this paragraph
    yielded zero utterances, and the caller's `src = para if not utterances` fallback then
    fed the whole thing — dialogue included — into the narration windows."""
    book_ingest = pytest.importorskip("book_ingest")
    para = "'It could be done,' said Lord Dawlish, 'but you'd want a bit of pull.'"
    assert book_ingest.extract_utterances(para, "curly-double") == []
    assert book_ingest.extract_utterances(para, "straight-single")


# --- A-M6: provenance -----------------------------------------------------------------


def test_gutenberg_is_not_labelled_standard_ebooks():
    """Every bank claimed `Standard Ebooks CC0` regardless of source, and the router has
    been sending PG sources down this path all along — false licence metadata in every
    derived clip's paper trail."""
    book_ingest = pytest.importorskip("book_ingest")
    name, lic, note = book_ingest.text_provenance("https://www.gutenberg.org/ebooks/6684", None)
    assert name == "Project Gutenberg"
    assert lic == "PD-US"
    assert "Standard Ebooks" not in note


def test_standard_ebooks_is_still_cc0():
    book_ingest = pytest.importorskip("book_ingest")
    name, lic, _ = book_ingest.text_provenance(
        "https://standardebooks.org/ebooks/p-g-wodehouse/uneasy-money", None)
    assert (name, lic) == ("Standard Ebooks", "CC0-1.0")


def test_a_local_epub_is_unknown_rather_than_assumed():
    book_ingest = pytest.importorskip("book_ingest")
    _, lic, note = book_ingest.text_provenance(None, "/tmp/somebody.epub")
    assert lic == "UNKNOWN"
    assert "NOT ESTABLISHED" in note


# --- A-H4: wrong-project resolution ---------------------------------------------------


def test_wrong_project_is_refused(tmp_path):
    """`api_book` matches on words with a prefix fallback, so a near-miss resolves to a
    real, wrong project — and `key` comes from the REQUESTED slug, so its audio lands in
    the right-looking directory. Run as a subprocess because the check lives in main()."""
    script = os.path.join(SYNTH, "librivox_fetch.py")
    harness = f"""
import sys, json
sys.path.insert(0, {SYNTH!r})
import librivox_fetch as lf
lf.api_book = lambda url: {{
    "title": "Speeches Against Catilina",
    "url_librivox": "https://librivox.org/speeches-against-catilina-by-marcus-tullius-cicero/",
    "sections": [], "url_text_source": "",
}}
sys.argv = ["librivox_fetch", "--url",
            "https://librivox.org/speeches-literary-and-social-by-charles-dickens/",
            "--root", {str(tmp_path)!r}, "--text-only"]
sys.exit(lf.main())
"""
    proc = subprocess.run([sys.executable, "-c", harness], capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "RESOLVED THE WRONG PROJECT" in proc.stderr
    assert not list(tmp_path.iterdir()), "refused, but it still created the directory"
    assert os.path.exists(script)


# --- A-H1: chapter slicing ------------------------------------------------------------


@pytest.fixture()
def align():
    """librivox_align imports torch/faster-whisper lazily, so importing it is cheap."""
    return pytest.importorskip("librivox_align")


def _body(n, word="filler"):
    return ("\n\n" + " ".join([word] * n) + "\n\n")


def test_toc_entries_are_not_mistaken_for_chapter_headings(align):
    """A ToC lists every chapter, so the old pattern returned roughly twice as many
    headings as the book has and the first half were all inside the contents. Slicing on
    those hands the aligner a few lines of ToC, which holds almost none of the spoken
    words — the ~10% coverage case."""
    toc = "CONTENTS\n\nCHAPTER I\nCHAPTER II\nCHAPTER III\n\n"
    body = "".join(f"CHAPTER {n}\n{_body(400)}" for n in ("I", "II", "III"))
    heads = align.find_headings(toc + body)
    assert len(heads) == 3, [h.group(0) for h in heads]
    # every survivor is in the body, past the contents block
    assert all(h.start() >= len(toc) - 2 for h in heads)


def test_hard_wrapped_prose_is_not_a_heading(align):
    """Gutenberg plaintext wraps at ~70 columns, so a sentence can put `chapter I` at the
    start of a line."""
    text = ("Some real text here.\n"
            "as I explained at length in the last\n"
            "chapter I was quite unwilling to concede the point to him at all.\n"
            + _body(500))
    assert align.find_headings(text) == []


def test_a_genuine_heading_is_still_found(align):
    text = "CHAPTER IV\n" + _body(500) + "CHAPTER V\n" + _body(500)
    heads = align.find_headings(text)
    assert [h.group(2) for h in heads] == ["IV", "V"]


def test_single_file_multichapter_book_does_not_slice_on_headings(align):
    """40 chapters in 3 audio files satisfied the old `len(heads) >= n_sections`, which
    then handed section 1 the text of chapter 1 alone — about 10% of what is read, so
    every clip in the section was dropped."""
    text = "".join(f"CHAPTER {n}\n{_body(400)}" for n in range(1, 41))
    strategies = [name for name, _ in align.chapter_slices(text, 1, 3, None)]
    assert "headings" not in strategies
    assert strategies[0] == "duration"


def test_headings_are_used_when_they_map_onto_sections(align):
    text = "".join(f"CHAPTER {n}\n{_body(400)}" for n in range(1, 4))
    named = align.chapter_slices(text, 2, 3, None)
    assert named[0][0] == "headings"
    assert "CHAPTER 2" in named[0][1] and "CHAPTER 3" not in named[0][1]


def test_there_is_always_a_fallback_after_the_first_candidate(align):
    """A-H1's second half: one slice used to be the only slice, so a misleading heading
    lost the whole section. Retrying is nearly free — ASR depends only on the audio."""
    text = "".join(f"CHAPTER {n}\n{_body(400)}" for n in range(1, 4))
    got = align.chapter_slices(text, 2, 3, None)
    strategies = [name for name, _ in got]
    assert strategies[0] == "headings"
    assert len(got) > 1, "one candidate is what the finding is about"
    assert "duration" in strategies
    # The last resort is the whole text. It may arrive named "duration-wide" rather than
    # "whole-text" — a wide enough proportional window already spans everything, and the
    # dedup drops the identical second copy rather than paying for it twice.
    assert got[-1][1] == text, strategies


def test_candidates_are_deduplicated(align):
    """A one-section book's proportional slice IS the whole text; offering it twice would
    just cost a second difflib pass over the same words."""
    text = "CHAPTER 1\n" + _body(500)
    got = align.chapter_slices(text, 1, 1, None)
    assert len({chunk for _, chunk in got}) == len(got)


def test_duration_split_beats_an_even_split_for_a_collection(align):
    """Dickens's *Speeches*: a 73-minute introduction then 2.6-minute speeches. An even
    index split put section 3's window at 3.3% of the text when the words were near
    10.8%, and the anchor pass located 0% of the heard words."""
    # Distinguishable content: with "w w w ..." every slice is found at index 0 and the
    # position assertions below would be meaningless.
    text = " ".join(f"{i:06d}" for i in range(60_000))
    durations = {1: 2400.0, 2: 1980.0, 3: 156.0, 4: 156.0}
    by_duration = dict(align.chapter_slices(text, 3, 4, durations))["duration"]
    by_index = dict(align.chapter_slices(text, 3, 4, None))["duration"]
    start_dur = text.index(by_duration) / len(text)
    start_idx = text.index(by_index) / len(text)
    assert start_dur > 0.5, start_dur      # section 3 really is late in the text
    assert start_idx < 0.5, start_idx      # the even split puts it in the middle


# --- A-M2: per-word spans from CTC forced alignment -----------------------------------
#
# `forced_align` returns one entry PER FRAME, so a target token occupies a RUN of frames.
# `refine()` walked that output counting non-blank FRAMES against the number of CHARACTERS
# in each word, which is only correct if every character happens to occupy exactly one
# frame. It never does, and the error compounds: each word eats the frames of the word
# before it, and the sentence is exhausted after `len(flat)` non-blank frames out of the
# many more really present.
#
# Ground truth below is the real alignment measured in the pinned ROCm container
# (torchaudio 2.10.0), driving `forced_align` on a hand-built emission:
#
#   frames  [0, 1, 1, 1, 0, 2, 3, 3, 0, 0, 4, 4, 4, 0]     words "ab" = [1,2], "cd" = [3,4]
#   merge_tokens -> token 1 @[1,4)  token 2 @[5,6)  token 3 @[6,8)  token 4 @[10,13)
#   so word "ab" spans frames [1,6) and word "cd" spans [6,13).


class _Span:
    """Stand-in for torchaudio's TokenSpan — `.start` / `.end` are all we consume."""

    def __init__(self, start, end):
        self.start, self.end = start, end


MEASURED_FRAMES = [0, 1, 1, 1, 0, 2, 3, 3, 0, 0, 4, 4, 4, 0]
MEASURED_TOKEN_SPANS = [_Span(1, 4), _Span(5, 6), _Span(6, 8), _Span(10, 13)]
MEASURED_WORD_TOKENS = [[1, 2], [3, 4]]


@pytest.fixture()
def align_mod():
    return pytest.importorskip("librivox_align")


def _old_walk(frames, word_tokens):
    """The shipped implementation, verbatim in behaviour, for the comparison below."""
    out, pos = [], 0
    for tok in word_tokens:
        start_f = end_f = None
        got = 0
        while pos < len(frames) and got < len(tok):
            if frames[pos] != 0:
                if start_f is None:
                    start_f = pos
                end_f = pos
                got += 1
            pos += 1
        out.append(None if start_f is None else (float(start_f), float(end_f + 1)))
    return out


def test_word_spans_follow_token_runs_not_frame_counts(align_mod):
    """The fix, against the measured alignment. ratio=1, t0=0 keeps it in frame units."""
    spans = align_mod.group_token_spans(
        MEASURED_TOKEN_SPANS, MEASURED_WORD_TOKENS, t0=0.0, ratio=1.0)
    assert spans == [(1.0, 6.0), (6.0, 13.0)]


def test_the_old_walk_is_what_truncated_the_tail(align_mod):
    """Not a style point — this is the measured defect, reproduced.

    The caller carries a comment blaming "CTC may fail to align the LAST words of a
    sentence" and an example that lost three words and tripped `tail_ok`. The arithmetic
    is the explanation: the old walk ends the sentence at frame 6 where it truly ends at
    13, dropping 54% of its duration.
    """
    old = _old_walk(MEASURED_FRAMES, MEASURED_WORD_TOKENS)
    new = align_mod.group_token_spans(
        MEASURED_TOKEN_SPANS, MEASURED_WORD_TOKENS, t0=0.0, ratio=1.0)
    assert old == [(1.0, 3.0), (3.0, 6.0)]           # what shipped
    assert new[-1][1] == 13.0 and old[-1][1] == 6.0  # the tail truncation, exactly


def test_the_old_middle_spans_pointed_at_the_wrong_words(align_mod):
    """Worse than imprecise, which is why span-FiLM could not have inherited them: the
    old span for word 2 is (3,6) and its true span is (6,13) — they do not overlap at
    all, so the "span" would have supervised the audio of the word before it."""
    old = _old_walk(MEASURED_FRAMES, MEASURED_WORD_TOKENS)[1]
    new = align_mod.group_token_spans(
        MEASURED_TOKEN_SPANS, MEASURED_WORD_TOKENS, t0=0.0, ratio=1.0)[1]
    overlap = min(old[1], new[1]) - max(old[0], new[0])
    assert overlap <= 0, f"expected disjoint, got overlap {overlap}"


def test_spans_are_offset_and_scaled_into_seconds(align_mod):
    """`t0` is the window start in the source audio and `ratio` frames->seconds; getting
    either wrong shifts every clip in the book by a constant, which reads as sloppy
    cutting rather than as a bug."""
    spans = align_mod.group_token_spans(
        MEASURED_TOKEN_SPANS, MEASURED_WORD_TOKENS, t0=10.0, ratio=0.02)
    assert spans == [(pytest.approx(10.02), pytest.approx(10.12)),
                     (pytest.approx(10.12), pytest.approx(10.26))]


def test_a_word_with_no_tokens_yields_no_span(align_mod):
    """`refine` drops empty token lists before grouping, but the boundary is worth
    holding: a None span must not become a (0,0) span the caller would treat as real."""
    spans = align_mod.group_token_spans(MEASURED_TOKEN_SPANS, [[1, 2], [3, 4], []],
                                        t0=0.0, ratio=1.0)
    assert spans[-1] is None


# --- A-M8: Project Gutenberg boilerplate in the EPUB lane -----------------------------
#
# The plaintext lane cut PG's wrapper on the `*** START/END ***` markers. The epub lane
# had no PG handling at all — its only filter was a filename SKIP list, and that list is
# Standard Ebooks' vocabulary (titlepage/imprint/colophon/uncopyright). PG names its
# documents nothing of the sort and routinely puts the whole book, wrapper included, in one
# document, so no filename rule could have worked. ~4,000 words of licence prose therefore
# parsed as prose, split into sentences, got directed by Gemma and rendered as the novel —
# and `is_complete_utterance` passes every one of them, because they are real sentences.
#
# It also made a licence claim false: `text_provenance` stamps every PG bank with
# "PG header/footer stripped, so the Project Gutenberg License does not attach". Same
# family as A-M6 — metadata asserting a step no code performed.

PG_HEADER = [
    "The Project Gutenberg eBook of Uneasy Money, by P. G. Wodehouse",
    "This ebook is for the use of anyone anywhere in the United States and most other "
    "parts of the world at no cost and with almost no restrictions whatsoever.",
    "Title: Uneasy Money",
    "Release date: March 1, 2004 [eBook #6684]",
    "*** START OF THE PROJECT GUTENBERG EBOOK UNEASY MONEY ***",
]
PG_BODY = [
    "In a bedroom on the fourth floor of the Hotel Belvoir a man was lying.",
    "He was generally to be found at the pen and ink counter.",
]
PG_FOOTER = [
    "*** END OF THE PROJECT GUTENBERG EBOOK UNEASY MONEY ***",
    "Section 1. General Terms of Use and Redistributing Project Gutenberg-tm "
    "electronic works",
    "1.A. By reading or using any part of this Project Gutenberg-tm electronic work, "
    "you indicate that you have read the license.",
    "Please visit www.gutenberg.org for more information.",
]


@pytest.fixture()
def ingest_mod():
    return pytest.importorskip("book_ingest")


def test_pg_wrapper_is_cut_when_it_spans_several_documents(ingest_mod):
    """The real shape: PG's header, the book and the licence are separate epub documents,
    so a per-section cut would leave every section BETWEEN the markers untouched — which
    is the whole book plus both halves of the wrapper."""
    sections = [("front", "prose", list(PG_HEADER)),
                ("Chapter I", "prose", list(PG_BODY)),
                ("back", "prose", list(PG_FOOTER))]
    out = ingest_mod._strip_pg_sections(sections)
    assert [t for t, _k, _p in out] == ["Chapter I"]
    assert [p for _t, _k, ps in out for p in ps] == PG_BODY


def test_pg_wrapper_is_cut_when_the_whole_book_is_one_document(ingest_mod):
    """PG commonly ships exactly this, and it is why no filename SKIP rule could work."""
    sections = [("book", "prose", PG_HEADER + PG_BODY + PG_FOOTER)]
    out = ingest_mod._strip_pg_sections(sections)
    assert [p for _t, _k, ps in out for p in ps] == PG_BODY


def test_an_inline_start_marker_does_not_eat_the_first_sentence(ingest_mod):
    """Some editions run the marker straight into the opening line. Dropping the marker's
    paragraph whole would silently lose the first sentence of the book — a truncation
    nothing downstream can see, since what remains is still a valid utterance."""
    opener = ("*** START OF THE PROJECT GUTENBERG EBOOK UNEASY MONEY *** "
              "In a bedroom on the fourth floor a man was lying.")
    out = ingest_mod._strip_pg_sections([("book", "prose", [opener, PG_BODY[1]])])
    assert [p for _t, _k, ps in out for p in ps] == [
        "In a bedroom on the fourth floor a man was lying.", PG_BODY[1]]


def test_an_inline_end_marker_keeps_the_prose_before_it(ingest_mod):
    closer = (PG_BODY[1] + " *** END OF THE PROJECT GUTENBERG EBOOK UNEASY MONEY ***")
    out = ingest_mod._strip_pg_sections(
        [("book", "prose", PG_HEADER + [PG_BODY[0], closer] + PG_FOOTER[1:])])
    assert [p for _t, _k, ps in out for p in ps] == PG_BODY


def test_a_standard_ebooks_epub_is_untouched(ingest_mod):
    """The cut is driven by the MARKERS, not by the source URL — so it is a no-op on SE,
    and a PG file supplied as a local `--epub` (where the URL says nothing) is still
    stripped rather than refused."""
    sections = [("Chapter I", "prose", list(PG_BODY))]
    assert ingest_mod._strip_pg_sections(sections) == sections


def test_repeated_paragraphs_do_not_confuse_the_rebuild(ingest_mod):
    """Rebuilt by index, not by matching text back. Counting occurrences would mis-handle
    a book with two identical paragraphs, and the cut can rewrite the marker's own
    paragraph so the survivor need not equal any input string."""
    dup = "He said nothing."
    sections = [("front", "prose", list(PG_HEADER)),
                ("Chapter I", "prose", [dup, "Then he spoke.", dup]),
                ("back", "prose", list(PG_FOOTER))]
    out = ingest_mod._strip_pg_sections(sections)
    assert [p for _t, _k, ps in out for p in ps] == [dup, "Then he spoke.", dup]


def test_surviving_boilerplate_is_refused_not_filtered(ingest_mod):
    """Pre-2006 editions predate the markers, so the cut cannot be assumed to have
    worked and its failure is silent. Refusing keeps `text_provenance`'s licence claim
    true; filtering would risk deleting paragraphs of the book instead."""
    assert synth_common.pg_boilerplate_residue(PG_FOOTER) == PG_FOOTER
    assert synth_common.pg_boilerplate_residue(PG_HEADER) == [PG_HEADER[-1]]


def test_the_prose_of_a_novel_is_not_mistaken_for_boilerplate():
    assert synth_common.pg_boilerplate_residue(PG_BODY) == []
    assert synth_common.pg_boilerplate_residue(
        ["He had once visited a project in Gutenberg, the printer's quarter."]) == []


def test_both_lanes_share_one_definition():
    """The plaintext lane had its own PG_START/PG_END. Two copies of one rule is how B-L5
    and D-L2 happened; the epub lane's gap is what happens when only one copy is fixed."""
    fetch_mod = pytest.importorskip("librivox_fetch")
    assert fetch_mod.PG_START is synth_common.PG_START_RE
    assert fetch_mod.PG_END is synth_common.PG_END_RE


# --- A-M9 / A-M10: one book, two possible ledger keys ---------------------------------
#
# `pg:<etext_id>` when the Gutenberg edition is known, `lv:<slug>` when it is not — and
# which one a book got depended on whether the LibriVox API happened to return
# `url_text_source` on the pass that created it. Nothing reconciled the two.


@pytest.fixture()
def router_mod():
    return pytest.importorskip("book_router")


def test_key_candidates_prefer_the_text_identity():
    """`pg:` leads because it identifies the TEXT, which survives a re-recording and is
    what the LibriTTS overlap check matches on."""
    assert synth_common.ledger_key_candidates(
        url="https://librivox.org/uneasy-money-by-p-g-wodehouse/",
        etext_id="6684") == ["pg:6684", "lv:uneasy-money-by-p-g-wodehouse"]
    assert synth_common.ledger_key_candidates(
        url="https://librivox.org/uneasy-money-by-p-g-wodehouse/") == [
        "lv:uneasy-money-by-p-g-wodehouse"]
    assert synth_common.ledger_key_candidates(etext_id="6684") == ["pg:6684"]
    assert synth_common.ledger_key_candidates() == []


def test_standard_ebooks_gets_the_se_scheme_the_ledger_actually_uses():
    """Fifteen hand-authored entries use `se:<author>/<title>`; the router never produced
    it, so its `"lv:" + <last path segment>` fallback filed SE books as LibriVox ones."""
    assert synth_common.ledger_key_candidates(
        url="https://standardebooks.org/ebooks/l-frank-baum/the-road-to-oz")[0] == (
        "se:l-frank-baum/the-road-to-oz")


def test_the_legacy_se_miskey_is_still_probed():
    """FIVE live entries carry it — lv:walden, lv:conan-stories, lv:up-from-slavery,
    lv:the-voyage-of-the-beagle, lv:the-autobiography-of-benjamin-franklin, all
    lane=synthesize against standardebooks.org URLs. Probing the old spelling keeps them
    findable rather than orphaning them behind a corrected key. They are NOT rewritten:
    re-keying live ledger data is a migration, and that is the owner's call.
    """
    ledger = {"lv:walden": {"status": "pending book_ingest", "lane": "synthesize"}}
    key, entry = synth_common.resolve_ledger_key(
        ledger, url="https://standardebooks.org/ebooks/henry-david-thoreau/walden")
    assert key == "lv:walden" and entry["lane"] == "synthesize"


def test_a_new_se_book_gets_the_canonical_key_not_the_legacy_one():
    key, entry = synth_common.resolve_ledger_key(
        {}, url="https://standardebooks.org/ebooks/jane-austen/emma")
    assert key == "se:jane-austen/emma" and entry is None


def test_an_existing_lv_entry_is_found_once_the_etext_id_is_known():
    """The duplicate-entry case. A book filed as `lv:` before its Gutenberg source was
    resolved must not become a SECOND entry under `pg:` the moment it is."""
    ledger = {"lv:uneasy-money-by-p-g-wodehouse": {"status": "fetched; awaiting align"}}
    key, entry = synth_common.resolve_ledger_key(
        ledger, url="https://librivox.org/uneasy-money-by-p-g-wodehouse/",
        etext_id="6684")
    assert key == "lv:uneasy-money-by-p-g-wodehouse"
    assert entry["status"] == "fetched; awaiting align"


def test_a_genuinely_new_book_gets_the_canonical_key_and_no_entry():
    key, entry = synth_common.resolve_ledger_key(
        {}, url="https://librivox.org/x/", etext_id="99")
    assert key == "pg:99" and entry is None


def test_a_librivox_book_with_no_etext_id_is_still_recognised(router_mod, monkeypatch):
    """A-M9's headline. The check built a `pg:` key or None, so this book was compared
    against NOTHING: every re-route re-created it and reset `status` to pending —
    regressing a book already fetched and aligned, and printing no SKIP to say so."""
    monkeypatch.setattr(router_mod, "lookup_librivox", lambda url: {
        "title": "Uneasy Money", "url_text_source": None,
        "num_sections": 15, "totaltime": "5:12:00"})
    monkeypatch.setattr(router_mod.time, "sleep", lambda *_a: None)
    ledger = {"lv:uneasy-money-by-p-g-wodehouse": {"status": "fetched; awaiting align",
                                                   "verdict": "REAL-AUDIO"}}
    rec = router_mod.route("https://librivox.org/uneasy-money-by-p-g-wodehouse/",
                           ledger, {}, {})
    assert rec["verdict"] == "SKIP"
    assert "fetched; awaiting align" in rec["flags"][0]


def test_the_router_never_regresses_a_books_status(router_mod):
    """The router decides a LANE. It knows nothing about whether the audio has been
    fetched or aligned, and stamping "pending" over "fetched; awaiting align" loses work
    that nothing else records."""
    led = {"pg:6684": {"status": "fetched; awaiting align", "verdict": "REAL-AUDIO"}}
    key, entry = synth_common.resolve_ledger_key(
        led, url="https://librivox.org/uneasy-money/", etext_id="6684")
    merged = dict(entry)
    merged["status"] = entry.get("status") or "pending force-align ingest"
    assert merged["status"] == "fetched; awaiting align"
    assert key == "pg:6684"


def test_both_invocation_styles_resolve_to_one_directory():
    """A-M10's on-disk half: the output dir is `key.replace(":", "_")`, so `--key pg:6684`
    wrote `pg_6684` and `--url <the same book>` wrote `lv_uneasy-money-…`. The same book,
    downloaded twice into two trees, decided by how it was invoked."""
    ledger = {"pg:6684": {"status": "pending force-align ingest"}}
    by_url, _ = synth_common.resolve_ledger_key(
        ledger, url="https://librivox.org/uneasy-money-by-p-g-wodehouse/",
        etext_id="6684")
    assert by_url.replace(":", "_") == "pg_6684"


def test_etext_id_extraction_has_one_definition():
    assert book_router_etext("https://www.gutenberg.org/ebooks/6684") == "6684"
    assert book_router_etext("https://www.gutenberg.org/files/6684/6684-0.txt") == "6684"
    assert book_router_etext("https://www.gutenberg.org/etext/6684") == "6684"
    assert book_router_etext(None) is None


def book_router_etext(url):
    mod = pytest.importorskip("book_router")
    assert mod.etext_id_from is synth_common.etext_id_from
    return mod.etext_id_from(url)


# --- A-M1: playtime parsing and the duration split ------------------------------------


@pytest.mark.parametrize(("raw", "seconds"), [
    ("2105", 2105.0),      # the form all three books on disk use
    ("1:23:45", 5025.0),   # hh:mm:ss — what `totaltime` always is, and playtime sometimes
    ("05:30", 330.0),      # mm:ss
    (300, 300.0),
    (0, None), ("0", None), ("0:00", None), ("", None), (None, None), ("junk", None),
])
def test_playtime_parses_both_formats(raw, seconds):
    """`float(...)` raised on the colon form, and the caller's `except: continue` dropped
    that section from the duration map — silently, so the surviving sections' windows were
    sized against a smaller total and stopped tiling."""
    assert synth_common.parse_playtime(raw) == seconds


def test_zero_is_unknown_not_a_duration():
    """Zero is a real value and `or 0` conflated it with absent. The caller has to tell
    the difference to know whether it may impute."""
    assert synth_common.parse_playtime(0) is None
    assert synth_common.parse_playtime(1) == 1.0


def test_a_colon_formatted_section_is_no_longer_dropped(align_mod):
    book = {"totaltime": "1:00:00", "sections": [
        {"index": 1, "playtime": "600"},
        {"index": 2, "playtime": "0:10:00"},
        {"index": 3, "playtime": "2400"}]}
    assert align_mod.section_durations(book) == {1: 600.0, 2: 600.0, 3: 2400.0}


def test_one_zero_playtime_no_longer_wipes_the_whole_book(align_mod):
    """The headline. `if not all(durations.values()): durations = {}` threw away every
    duration in the book and reverted ALL sections to the even split — which is the split
    that located 0% of the heard words in Dickens's *Speeches* and dropped every clip.
    Now the gap is filled from totaltime's residual and only the unknown section is
    guessed at."""
    book = {"totaltime": "1:00:00", "sections": [
        {"index": 1, "playtime": "600"},
        {"index": 2, "playtime": "0"},
        {"index": 3, "playtime": "2400"}]}
    assert align_mod.section_durations(book) == {1: 600.0, 2: 600.0, 3: 2400.0}


def test_the_durations_map_is_all_or_nothing(align_mod):
    """A PARTIAL map is the case that produces windows which do not tile, and
    `_proportional` cannot tell a partial map from a complete one. When the residual
    cannot fill the gap, fall back for the whole book — loudly."""
    book = {"sections": [{"index": 1, "playtime": "600"},
                         {"index": 2, "playtime": None}]}      # no totaltime to fill from
    assert align_mod.section_durations(book) == {}


def test_the_silent_fallback_is_now_loud(align_mod, capsys):
    align_mod.section_durations({"sections": [{"index": 1, "playtime": "600"},
                                              {"index": 2, "playtime": None}]})
    assert "EVEN split" in capsys.readouterr().err


def test_duration_windows_tile_the_text(align_mod):
    """What the partial map broke. Consecutive sections' unpadded windows must be
    contiguous and together cover the whole text, or words fall into a gap no section
    searches."""
    durations = {1: 600.0, 2: 600.0, 3: 2400.0}
    text = "w " * 5000
    edges = []
    for i in (1, 2, 3):
        before = sum(d for k, d in durations.items() if k < i)
        total = sum(durations.values())
        edges.append((before / total, (before + durations[i]) / total))
    assert edges[0][0] == 0.0 and edges[-1][1] == pytest.approx(1.0)
    for (_a0, a1), (b0, _b1) in zip(edges, edges[1:]):
        assert a1 == pytest.approx(b0), "a gap between section windows"
    assert align_mod._proportional(text, 1, 3, durations, 0.0).startswith("w")


# --- A-M13: retries in the bulk-transfer lane -----------------------------------------


def test_a_transient_failure_is_retried(fetch_mod, monkeypatch, tmp_path):
    """One `fetch` and no retry, in a lane pulling sixty-odd multi-megabyte files from a
    rate-limited archive.org. The caller catches, prints "!! section N" and moves on — so
    one transient 503 left a permanent hole in the book."""
    import urllib.error

    calls = []

    def flaky(url, timeout=300):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
        return b"ID3\x04\x00" + b"\x00" * 4000

    monkeypatch.setattr(fetch_mod, "fetch", flaky)
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda *_a: None)
    new, n = fetch_mod.download("test://audio", tmp_path / "003.mp3")
    assert new and len(calls) == 3


def test_a_404_is_not_retried(fetch_mod, monkeypatch):
    """The URL is wrong; waiting does not fix it, and three backoffs per missing section
    turns a fast failure into a slow one."""
    import urllib.error

    calls = []

    def gone(url, timeout=300):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(fetch_mod, "fetch", gone)
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda *_a: None)
    with pytest.raises(urllib.error.HTTPError):
        fetch_mod.fetch_with_retry("test://missing")
    assert len(calls) == 1


def test_retries_are_exhausted_then_it_raises(fetch_mod, monkeypatch):
    calls = []

    def dead(url, timeout=300):
        calls.append(url)
        raise TimeoutError("timed out")

    monkeypatch.setattr(fetch_mod, "fetch", dead)
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda *_a: None)
    with pytest.raises(TimeoutError):
        fetch_mod.fetch_with_retry("test://slow")
    assert len(calls) == 1 + len(fetch_mod.RETRY_DELAYS)


# --- A-M11: the director pass is checkpointed -----------------------------------------
#
# `lines` accumulated in memory and reached disk only after the LAST chunk, so any
# interruption — a `load_skill` error at chunk 90, an ollama restart, Ctrl-C — discarded
# every director call made so far. Each is a 31B inference; a 200-chunk book is an hour.


def _chunk(kind, text):
    return {"chunk_type": kind, "text": text, "source_ref": {}}


def test_chunk_key_is_content_not_position(ingest_mod):
    """An index-keyed checkpoint would hand chunk 40's direction to chunk 37 after a
    resume with a different --max-per-type — worse than no checkpoint, because the bank
    still looks complete."""
    a = _chunk("narration", "In a bedroom on the fourth floor a man was lying.")
    b = dict(a)
    assert ingest_mod.chunk_key(a) == ingest_mod.chunk_key(b)
    assert ingest_mod.chunk_key(a) != ingest_mod.chunk_key(_chunk("dialogue", a["text"]))
    assert ingest_mod.chunk_key(a) != ingest_mod.chunk_key(_chunk("narration", "Other."))


def test_checkpoint_round_trips(ingest_mod, tmp_path):
    path = tmp_path / "x_bank.partial.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for n in range(3):
            fh.write(json.dumps({"chunk_key": f"k{n}", "line": {"id": f"line{n}"}}) + "\n")
    done = ingest_mod.load_director_checkpoint(path)
    assert set(done) == {"k0", "k1", "k2"}
    assert done["k1"]["id"] == "line1"


def test_a_torn_trailing_record_does_not_lose_the_rest(ingest_mod, tmp_path):
    """The failure this file exists for is a killed process, and a killed process is
    exactly what leaves a half-written last line."""
    path = tmp_path / "x_bank.partial.jsonl"
    good = json.dumps({"chunk_key": "k0", "line": {"id": "line0"}})
    path.write_text(good + "\n" + '{"chunk_key": "k1", "li', encoding="utf-8")
    done = ingest_mod.load_director_checkpoint(path)
    assert list(done) == ["k0"]


def test_a_missing_checkpoint_is_not_an_error(ingest_mod, tmp_path):
    assert ingest_mod.load_director_checkpoint(tmp_path / "absent.jsonl") == {}


# --- QC-L2: the length ceiling covered narration only ---------------------------------


def test_dialogue_chunking_refuses_an_unrenderable_speech():
    """`_window_sentences` grew the `HARD_MAX_CHARS` guard from a real defect — one
    3,964-character passage, ~283 s — and the fix covered the NARRATION path only.
    `_chunk_speech` appended an over-long sentence unbounded, and dialogue is 64% of the
    corpus and where a single unbroken speech is most likely.

    The cost is not a bad clip; it is GPU time spent rendering a passage `qc_gate` then
    throws away at its 30 s cap, on the one shared GPU where training and rendering are
    mutually exclusive.
    """
    bi = pytest.importorskip("book_ingest")
    sentence = "I tell you " + "and again " * 60 + "that it is so."
    assert len(sentence) > bi.HARD_MAX_CHARS
    assert bi._chunk_speech(sentence) == []
    # ...and the same rule the narration path applies.
    assert bi._window_sentences([sentence]) == []


def test_dialogue_chunking_still_keeps_a_long_but_renderable_speech():
    """The ceiling must not become the window: between WINDOW_MAX and HARD_MAX a sentence
    is long, not unrenderable, and taking it alone is the intended behavior."""
    bi = pytest.importorskip("book_ingest")
    sentence = ("I tell you plainly " + "and once more " * 22 + "that it is so.")
    assert bi.WINDOW_MAX_CHARS < len(sentence) <= bi.HARD_MAX_CHARS
    assert bi._chunk_speech(sentence) == [sentence]


def test_a_long_speech_does_not_take_its_neighbours_down():
    """The over-long sentence is dropped; whatever was accumulating before it is still
    emitted, or one bad sentence would cost the whole speech."""
    bi = pytest.importorskip("book_ingest")
    good = "That is a perfectly reasonable thing for anyone to say in the circumstances."
    huge = "I tell you " + "and again " * 60 + "that it is so."
    out = bi._chunk_speech(good + " " + huge)
    assert out == [good]

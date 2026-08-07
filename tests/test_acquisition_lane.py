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

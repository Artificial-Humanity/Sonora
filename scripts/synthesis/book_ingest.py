"""book_ingest — prototype front-end for the book-prose synthesis lane.

Fetch a permissive ebook (Standard Ebooks / Project Gutenberg) -> parse -> chunk
(narration windows + dialogue-with-attribution) -> Gemma-4 director-pass (VAD +
register + per-engine direction, via the live ollama endpoint) -> emit the flat
bank the synth_{vibevoice,dia,qwen,moss_vg}.py renderers consume.

PROTOTYPE NOTES
- Parsing/segmentation are done in Python here to validate the LANE fast. Production
  should converge onto Prosodia's `folioparser` (EPUB->text) + `stage::segmenter`
  (sentence split + Paragraph{target_characters}) for on-device dogfooding — see
  notes/book-prose-lane.md (Part 1 — Operations; was book-prose-operations.md).
- Director = the `MODEL` constant below, served by ollama on :11434 (read `content`,
  give generous num_predict). Named once, there — a second copy in prose is a second
  thing to forget, which is how this line spent a day naming the wrong model.

Run:
  .venv/bin/python scripts/synthesis/book_ingest.py --url <SE url> --out <dir> [--dry-run] [--max-per-type N]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (book_ingest prototype; contact lmcfarlin)"
OLLAMA = "http://localhost:11434/api/chat"
# Measured 2026-08-02 on 24 real narration passages, casting for zonos with the live
# skill file, register and V/A/T supplied exactly as casting_pass supplies them. The
# 2026-07-29 malformed-JSON finding does NOT discriminate here — pass 2 is grammar-
# constrained by `format` and pass 1 is short, so all three variants parsed 24/24.
# What separates them is whether they OBEY the skill file:
#
#   model                    emotion omitted   rate 14-16   pitch 20-45   distinct casts
#   gemma-4-26b-a4b-qat        8/24             14/24        12/24         4/24
#   gemma-4-31b-qat-spec      24/24             24/24        24/24        16/24
#   gemma-4-e4b-qat-spec       5/24             24/24        24/24         5/24
#
# The MoE was the default, and on 16 of 24 narration lines it emitted an emotion
# vector — 15 of them neutral-dominant, several the literal [0,…,0,1.0]. zonos.md
# says "`emotion` is omit on every narration lane, without exception" because that
# exact conditioning destabilized 5 of 9 zonos narration groups on 2026-07-30. The
# renderer's unconditional path and _l1(None) both shipped; the model in the chair
# would have re-introduced the defect on two thirds of the lines anyway, and the
# director path would have looked broken when the fault was model selection.
#
# The reuse column matters too: zonos.md calls one casting call across different
# registers "a failure", and the two rejected variants emit 4-5 distinct castings
# across 24 lines. 4.0 s/call against 2.4 is not a real cost on a pass that runs a
# few dozen times per book. e4b stays the right pick for high-volume judging
# (judge_passages), where there is no skill file to obey.
MODEL = "gemma-4-31b-qat-spec"

CHARS_PER_SEC = 14.0            # mirrors synth_dia.py length model
# Owner floor (2026-07-25): nothing shorter than 4 s of speech enters a bank.
# Purpose, in the owner's words: "avoid content too short to gauge performance
# around". This gates INPUT TEXT, not render duration.
#
# ⚠ SUPERSEDED IN PART (QC-L5). This used to add "Do NOT add an output-duration gate at
# 4 s; the QC gate's duration-vs-text check already catches real failures" — and on
# 2026-08-07 the owner made exactly that gate, `speech_ok`, hard at 4 s of VAD speech
# (`qc_gate.py`). The instruction outlived the decision that overruled it. What remains
# true is the reason it was written: this constant gates TEXT, so it is not the place to
# express an output rule, and the two floors are independent even though they share a
# number.
# Audit evidence — moss85 keep rate by estimated length: <2 s 60%, 2-4 s 40%,
# 4-10 s 91%, 10 s+ 100%. Short lines are also where MOSS reads the prompt
# aloud (the instruction:text ratio hits 13x on ~1 s dialogue fragments), and
# nari-labs document that Dia input under ~5 s "will sound unnatural".
MIN_CLIP_SECONDS = 4.0
MIN_CLIP_CHARS = int(MIN_CLIP_SECONDS * CHARS_PER_SEC)   # 56
WINDOW_MIN_CHARS = 90          # ~6 s of speech
# Raised 240 -> 300 (owner, 2026-08-01), i.e. ~17 s -> ~21.4 s of speech.
#
# TWO reasons, and the production one is the point. Training: MAX_SECONDS in
# derive_vat_corpus went 16 -> 22 s, so the model is fitted on longer utterances and a
# longer render window is no longer out of distribution. Production: every window costs a
# Gemma director pass, so window size sets the Gemma -> instruct -> Sonora cycle count for
# a whole book. Measured on the delivery-v1 bank (mean window 200 ch, 71% packing at the
# 240 cap), 300 cuts chunks — and director passes — by ~20% across the 31 ingested books.
#
# The old comment called 240 the "engine-reliable ceiling". Our own audit data says
# otherwise: 120-240 ch fails 18% (n=146), 240-310 ch fails 11% (n=27), 310-400 ch fails
# 0% (n=6). Reliability does not degrade until HARD_MAX_CHARS; 240 was conservative.
#
# ⚠ 300 is not a round number, it is CHATTERBOX'S limit (synth_chatterbox.MAX_CHARS = 300,
# past which it warns and risks its 1000-token / ~40 s ceiling). Do not raise this to the
# 308 that 22 s x CHARS_PER_SEC would allow without changing that renderer first — the
# binding constraint here is the engine, not the training cap.
WINDOW_MAX_CHARS = 300
ATTRIB_VERBS = (
    "said|asked|replied|whispered|murmured|cried|shouted|snarled|muttered|"
    "answered|exclaimed|gasped|breathed|hissed|demanded|pleaded|sighed|"
    "laughed|sobbed|screamed|growled|stammered|repeated|added|continued|"
    "called|returned|observed|remarked|insisted|protested|urged|warned"
)

def _lexicon():
    """Controlled register lexicon (markup-schema-brief.md §Field semantics).

    The brief always specified `utterance.register` as a controlled vocabulary
    governed by the app's Recategorize flow; it was never enforced, and by
    2026-07-25 ratings.csv held 138 distinct labels across 554 keeps. Regenerate
    with build_register_lexicon.py.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "register_lexicon.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["lexicon"]
    except Exception:
        return []


REGISTER_LEXICON = _lexicon()

# The roster is DERIVED, never typed here. It used to be the literal string
# "vibevoice"|"qwen"|"moss_vg"|"dia", frozen at the 2026-07-25 portfolio, and the
# portfolio moved twice underneath it: vibevoice and dia were set aside on
# 2026-07-29, and zonos, chatterbox and orpheus joined. So this prompt offered two
# retired engines, withheld three live ones, and called vibevoice the "PREMIER
# default" — an ingest run today would have routed a whole book to engines nothing
# is allowed to render on. `ref_select` already owns that decision for every other
# bank builder; reading it here is the same rule the ENGINE_MIX comment states.
def _castable_engines():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ref_select
    return [e for e in ref_select.ENGINE_MIX if e not in ref_select.SET_ASIDE]


CASTABLE_ENGINES = _castable_engines()

# Engine choice only. V/A/T + register are a SEPARATE pass (they describe the
# line, not the target), and voice_design/instruct are a THIRD pass driven by the
# per-engine skill files in director_skills/. Splitting these was forced by the
# 2026-07-25 finding that a single combined call let the training labels drift
# with whichever engine the director happened to be writing for.
#
# Each engine is described by the CHANNEL it offers, because that is what the
# choice actually turns on — the direction-relay audit (2026-07-25) established
# that the slots are unevenly distributed, and a line whose delivery must be
# described cannot go to an engine with nowhere to put the description.
_ENGINE_GUIDE = {
    "qwen": "the richest instruction slot in the portfolio and the measured gold "
            "standard — casting, delivery and accent all reach it as plain text. "
            "The default for a line whose PERFORMANCE must be described. Renders "
            "younger and higher than you describe; account for it.",
    "chatterbox": "no text-instruction slot at all: two numbers and a reference "
                  "clip. Voice IDENTITY comes from the reference, so choose it "
                  "when who is speaking matters more than how.",
    "zonos": "the only engine with NUMERIC prosody dials (pitch_std, "
             "speaking_rate). No prose slot. Its measured weakness — a dry, "
             "level read — is the requirement for narration, so it is the "
             "natural first choice for narration windows.",
    "orpheus": "a voice from a closed set, prefixed onto the text, plus a few "
               "inline tags. Nothing else reaches it. Reasonable for plain "
               "dialogue; it cannot be told anything a voice name does not say.",
    "moss_vg": "instruct-driven like qwen, and strong on dark, menace, oratory, "
               "force and situational framing. It overlaps qwen and fails more "
               "often, so prefer qwen unless the line is squarely in that band.",
}

DIRECTOR_SYSTEM = (
    "You are the Emotional Director for an audiobook TTS pipeline. You read one passage "
    "(narration, or a character's spoken line with its attribution) and label it, then "
    "choose which speech engine should render it.\n"
    "Output ONLY compact minified JSON, no markdown, with EXACTLY these keys:\n"
    '{"valence": float in [-1,1], "arousal": float in [-1,1], "tension": float in [-1,1], '
    '"register": string, "engine": one of '
    + "|".join(f'"{e}"' for e in CASTABLE_ENGINES) + "}\n"
    "`register` MUST be copied EXACTLY from this controlled lexicon — never invent, "
    "alter or compose a label; pick the closest fit:\n"
    + ", ".join(REGISTER_LEXICON) + "\n"
    "Engine guide:\n"
    + "".join(f"  {e} = {_ENGINE_GUIDE[e]}\n"
              for e in CASTABLE_ENGINES if e in _ENGINE_GUIDE)
    + "Do NOT write a voice design or delivery instruction here; a later pass does that "
    "per engine. Valence = pleasant(+)/unpleasant(-); Arousal = energy; Tension = "
    "held/threat/unease. JSON only."
)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def text_provenance(url, local_epub):
    """-> (source_name, license_id, note). A-M6.

    Every bank written before 2026-08-06 claimed `Standard Ebooks CC0` and
    `text_license: CC0` regardless of where the text came from, and the router has been
    sending Project Gutenberg sources down this path all along. That is false licence
    metadata, and it is the kind that propagates: it lands in the bank, then in every
    derived clip's paper trail, and a later audit of "what may we redistribute" reads it
    as fact. SE and PG both yield freely redistributable text, so nothing shipped is
    *unlicensed* — but the record has to say which, or it is not a record.

    A locally supplied `--epub` is explicitly UNKNOWN rather than assumed: this script
    cannot see where such a file came from, and guessing is how the original defect
    started.
    """
    if local_epub and not url:
        return ("local file", "UNKNOWN",
                "Text: provenance NOT ESTABLISHED — supplied as a local epub. "
                "Establish the source and licence before any redistribution.")
    u = (url or "").lower()
    if "standardebooks.org" in u:
        return ("Standard Ebooks", "CC0-1.0", "Text: Standard Ebooks (CC0 1.0).")
    if "gutenberg.org" in u:
        return ("Project Gutenberg", "PD-US",
                "Text: Project Gutenberg (US public domain; PG header/footer stripped, "
                "so the Project Gutenberg License does not attach).")
    return (u or "unknown", "UNKNOWN",
            f"Text: provenance NOT ESTABLISHED for {url!r}. "
            "Establish the source and licence before any redistribution.")


def se_epub_url(page_url):
    """Find the compatible .epub download link on a Standard Ebooks page."""
    html = fetch(page_url).decode("utf-8", "replace")
    hrefs = re.findall(r'href="([^"]+\.epub)"', html)
    # prefer the plain compatible epub (not _advanced, not .kepub)
    cands = [h for h in hrefs if "advanced" not in h and "kepub" not in h] or hrefs
    if not cands:
        raise SystemExit("No .epub download link found on the SE page.")
    url = urllib.parse.urljoin(page_url, cands[0])
    # SE serves an interstitial for bare download URLs; ?source=download yields the actual epub+zip.
    return url + ("&" if "?" in url else "?") + "source=download"


def parse_epub(epub_bytes, source_url=None):
    """Return [(chapter_title, [paragraph_text, ...]), ...] for chapter documents only.

    `source_url` selects the boilerplate rule (A-M8). The filename `SKIP` list below is
    **Standard Ebooks' vocabulary** — `titlepage`, `imprint`, `colophon`, `uncopyright` —
    and Project Gutenberg names its documents nothing of the sort. PG also routinely puts
    an entire book, header and licence footer included, in ONE document, so no filename
    rule could have removed them. Every PG epub therefore parsed its wrapper as prose:
    ~4,000 words of terms of use split into sentences, directed by Gemma, and rendered by
    a teacher engine as though it were the novel. `is_complete_utterance` passes them
    happily — they are grammatical sentences.
    """
    import io
    from bs4 import BeautifulSoup
    import ebooklib
    from ebooklib import epub

    SKIP = ("titlepage", "imprint", "colophon", "uncopyright", "halftitle", "toc",
            "endnotes", "loi", "dramatis-personae", "copyright", "dedication", "epigraph")

    def clean(t):
        return re.sub(r"\s+", " ", t.replace("﻿", "").replace("​", "")).strip()

    def has_type(sub):
        return lambda t: sub in (t.get("epub:type") or "")

    book = epub.read_epub(io.BytesIO(epub_bytes))
    sections = []  # (title, kind, items)   kind ∈ {"drama","prose"}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = item.get_name().lower()
        if any(s in name for s in SKIP):
            continue
        soup = BeautifulSoup(item.get_content(), "lxml-xml")
        body = soup.find("body") or soup
        for note in body.find_all(has_type("endnote")):
            note.decompose()
        # ebooklib rewrites the <body> tag (drops its epub:type), so detect drama by the
        # presence of stage-direction spans (inner epub:type survives).
        is_drama = body.find(has_type("stage-direction")) is not None
        title_el = body.find(["h1", "h2", "h3"])
        title = title_el.get_text(" ", strip=True) if title_el else name
        if is_drama:                                           # a play: speeches, stage dirs stripped
            speeches = []
            for p in body.find_all("p"):
                if p.find_parent(has_type("verse")):
                    continue                                   # songs / recited poems (z3998:verse)
                for sd in p.find_all(has_type("stage-direction")):
                    sd.decompose()                             # inline stage directions
                # skip blocking stage directions: a <p> that OPENS with a bolded persona name
                # ("Sempronius, smart and young, shows his profile…") is action, not a spoken line.
                first = next((ch for ch in p.children
                              if getattr(ch, "name", None) or (isinstance(ch, str) and ch.strip())), None)
                if getattr(first, "name", None) == "b" and "persona" in (first.get("epub:type") or ""):
                    continue
                txt = clean(p.get_text(" ", strip=True))
                if len(txt) >= 15:                             # a real spoken line, not a bare cue
                    speeches.append(txt)
            if speeches:
                sections.append((title, "drama", speeches))
            # NOTE: bare scene-setting <p>s ("An office in the royal palace…") carry no markup
            # signal distinguishing them from speech, so a few leak; owner audit is the catch-net.
        else:                                                  # prose (novel, preface, essay)
            paras = [clean(p.get_text(" ", strip=True)) for p in body.find_all("p")]
            paras = [p for p in paras if p]
            if paras:
                sections.append((title, "prose", paras))

    # A-M8. Applied ACROSS sections, not within one: PG's START marker and its END marker
    # routinely land in different documents, so a per-section cut would leave every
    # section between them untouched — which is the whole book plus both halves of the
    # wrapper. Flatten, cut, redistribute.
    #
    # Driven by the MARKERS, not by `source_url`. A Standard Ebooks epub carries none, so
    # this is a no-op there; a PG file handed over as a local `--epub` (where the URL says
    # nothing) is still stripped rather than refused. `source_url` only sharpens the
    # message below.
    sections = _strip_pg_sections(sections)

    # Refuse rather than filter. PG has changed its wrapper several times and pre-2006
    # editions predate the `***` markers entirely, so the cut above cannot be assumed to
    # have worked, and its failure mode is silent. `text_provenance` stamps every PG bank
    # with "PG header/footer stripped, so the Project Gutenberg License does not attach"
    # — a claim about licensing that propagates into every derived clip's paper trail,
    # and until this landed no code made it true. It has to be true or it has to stop.
    residue = synth_common.pg_boilerplate_residue(
        [p for _t, _k, ps in sections for p in ps])
    if residue:
        where = f" ({source_url})" if source_url else ""
        raise SystemExit(
            f"Project Gutenberg boilerplate survived stripping{where} — refusing to build "
            f"a bank from it ({len(residue)} paragraph(s)). First:\n\n"
            f"  {residue[0][:300]}\n\n"
            "This edition's wrapper does not carry the standard `*** START/END OF THE "
            "PROJECT GUTENBERG EBOOK ***` markers (pre-2006 editions predate them). Pick "
            "another PG edition, or supply the text by hand. NOT filtered automatically "
            "on purpose: silently deleting paragraphs of the book is the worse failure."
        )
    return sections


def _strip_pg_sections(sections):
    """Apply the PG header/footer cut across a flattened section list, then rebuild it.

    By INDEX, not by matching text back: the cut can rewrite the marker's own paragraph
    (keeping the prose on the far side of the marker), so the surviving string need not
    equal any input string. Counting occurrences would also mis-handle a book with two
    identical paragraphs, which is exactly the kind of edition that turns up once.
    """
    flat = [(si, pi, p)
            for si, (_t, _k, paras) in enumerate(sections)
            for pi, p in enumerate(paras)]

    start = end = None
    head_text = tail_text = None
    for n, (_si, _pi, p) in enumerate(flat):
        if start is None:
            m = synth_common.PG_START_RE.search(p)
            if m:
                start, tail_text = n, p[m.end():].strip()
                continue
        m = synth_common.PG_END_RE.search(p)
        if m:
            end, head_text = n, p[: m.start()].strip()
            break

    if start is None and end is None:
        return sections            # not a PG-wrapped document; leave it alone

    lo = 0 if start is None else start
    hi = len(flat) - 1 if end is None else end

    kept = {}                      # (section_index) -> [paragraph, ...]
    for n in range(lo, hi + 1):
        si, _pi, p = flat[n]
        if n == start:
            p = tail_text          # prose after an inline START marker, if any
        elif n == end:
            p = head_text          # prose before an inline END marker, if any
        if p:
            kept.setdefault(si, []).append(p)

    # Sections that lose every paragraph — the wrapper's own documents — drop out.
    return [(sections[si][0], sections[si][1], kept[si]) for si in sorted(kept)]


# One definition each, in synth_common — re-exported here because three modules and the
# test suite already import them from book_ingest. See synth_common for why they moved.
# The path insert is explicit rather than relying on the one inside `_castable_engines`,
# which happens to run first today; that is an ordering accident, not an import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_common  # noqa: E402  (module handle: parse_epub uses the A-M8 PG helpers)
from synth_common import (  # noqa: E402,F401
    _CLOSERS,
    _OPENERS,
    _TERMINALS,
    is_complete_utterance,
    split_sentences,
)


# Abbreviations a TTS engine will not expand for you. ARCHITECTURE §1 is explicit
# that "digits/abbreviations are the CALLER's job (op_g2p does not expand them)" —
# there is an `_abbreviations` table in matcha/text/cleaners.py, but it is the
# LJSpeech list (mr/mrs/dr/st/co/jr/rev) and it runs on TRAINING text only, never
# on what we hand a teacher engine.
#
# Found by ear, 2026-08-03: "…a literary society in Madison, Wis." was spoken as
# the letters. No model would have fixed that; every engine read exactly what it
# was given. The common titles are omitted on purpose — Mr./Mrs./Dr. are read
# correctly by every engine in the portfolio precisely because they are ubiquitous,
# and expanding them would change 67 lines of existing text for no gain. What
# fails is the RARE abbreviation.
_SPEAKABLE = [
    # US state abbreviations as they appear in 19th-c prose
    ("Ala", "Alabama"), ("Ariz", "Arizona"), ("Ark", "Arkansas"),
    ("Cal", "California"), ("Calif", "California"), ("Col", "Colorado"),
    ("Conn", "Connecticut"), ("Del", "Delaware"), ("Fla", "Florida"),
    ("Ga", "Georgia"), ("Ill", "Illinois"), ("Ind", "Indiana"),
    ("Kan", "Kansas"), ("Kans", "Kansas"), ("Ky", "Kentucky"),
    ("La", "Louisiana"), ("Md", "Maryland"), ("Mass", "Massachusetts"),
    ("Mich", "Michigan"), ("Minn", "Minnesota"), ("Miss", "Mississippi"),
    ("Mo", "Missouri"), ("Mont", "Montana"), ("Neb", "Nebraska"),
    ("Nebr", "Nebraska"), ("Nev", "Nevada"), ("Okla", "Oklahoma"),
    ("Ore", "Oregon"), ("Oreg", "Oregon"), ("Pa", "Pennsylvania"),
    ("Penn", "Pennsylvania"), ("Tenn", "Tennessee"), ("Tex", "Texas"),
    ("Vt", "Vermont"), ("Va", "Virginia"), ("Wash", "Washington"),
    ("Wis", "Wisconsin"), ("Wisc", "Wisconsin"), ("Wyo", "Wyoming"),
    # reference-style abbreviations that read as letters
    ("Fig", "Figure"), ("Figs", "Figures"), ("No", "Number"),
    ("Vol", "Volume"), ("Chap", "Chapter"), ("pp", "pages"),
]
# The trailing period does DOUBLE DUTY and must not be swallowed. "…in Madison,
# Wis. An hour before…" is two sentences; a naive `Wis\.` -> `Wisconsin` produces
# "Madison, Wisconsin An hour", welding them together and deleting the pause the
# reader needs. Sentence-final is detected by what follows — whitespace then a
# capital — and keeps its period; every other position drops it, because the dot
# was only marking the abbreviation.
_SPEAKABLE_FINAL = [(re.compile(rf"\b{a}\.(?=\s+[A-Z])"), f"{b}.") for a, b in _SPEAKABLE]
_SPEAKABLE_MID = [(re.compile(rf"\b{a}\."), b) for a, b in _SPEAKABLE]


def normalize_speakable(text):
    """Expand abbreviations an engine would otherwise spell out.

    Applied at MINT time so the expansion lands in the canonical text as well as
    the render text — otherwise ASR/WER would score the render against a
    transcript that still says "Wis." and the mismatch would read as an engine
    defect. The same text feeds the training corpus, so fixing it here fixes both.

    Case-sensitive on purpose: `\\bLa\\.` lowercased would rewrite "la." inside
    ordinary prose, and `No.` -> `Number` must not fire on the word "no."
    """
    for pat, full in _SPEAKABLE_FINAL:
        text = pat.sub(full, text)
    for pat, full in _SPEAKABLE_MID:
        text = pat.sub(full, text)
    return text


# Passages whose sense depends on something the listener cannot see. Found by ear
# 2026-08-03 on a Darwin passage: "This passage is making references to
# illustrations not visible here. Mentions of 'Fig #' come out robotic, giving away
# that it's a TTS speaking." Expanding `Fig.` to `Figure` makes
# it pronounceable but not sensible — a narrator reading "as shown in Figure 12"
# with no figure is reading a defect aloud. These are dropped at mint, not fixed.
_UNSPEAKABLE_CONTEXT = re.compile(
    r"\b(fig|figs|figure|figures|plate|plates|table|tables)\b\.?\s*\d",
    re.IGNORECASE)


def references_the_invisible(text):
    """True when the passage points at an illustration, plate or table."""
    return bool(_UNSPEAKABLE_CONTEXT.search(text))




# ---------------------------------------------------------------- quote conventions
#
# A-M7/M8, and it was CONFIRMED LIVE rather than latent. Dialogue extraction matched only
# curly DOUBLE quotes, so an edition that quotes any other way produced zero utterances —
# and because the caller falls back to `src = para` when there are no utterances, the
# dialogue did not go missing, it silently entered the NARRATION windows. On *Uneasy
# Money* (`pg:6684`) that is not hypothetical: the edition quotes with single quotes, 818
# against 26 doubles, so 1,355 of its 1,366 clips read as narration when the true figure
# is 1,201. It surfaced only because the miscount would have justified a
# delivery-homogeneous mark on a novel.
#
# Note the two halves used to disagree with each other: `_QUOTED_SPAN` (which strips
# quotes out of narration) already handled straight quotes while the extractor did not.
# One resolver now serves both, so they cannot drift.
CURLY_OPEN, CURLY_CLOSE = "“", "”"
SQ_OPEN, SQ_CLOSE = "‘", "’"
DEFAULT_QUOTE_STYLE = "curly-double"

# The four conventions found in the sources this lane actually pulls from. Each entry is
# (utterance_pattern, span_pattern) — the first captures what was said, the second matches
# the whole quoted run so it can be stripped out of the narrator's prose.
#
# The single-quote forms need boundary rules because their closing mark IS the apostrophe:
# a bare non-greedy `'(.+?)'` ends at the first contraction, turning "I don't know" into
# "I don". Requiring a word boundary on both sides solves the common cases —
#   man's, don't  -> the mark sits between two word characters: neither open nor close
#   'I don't know' -> opens after a space, skips the contraction, closes before a space
# A plural possessive (`the boys' hats`) can still close a span early; that produces a
# fragment, which is exactly what is_complete_utterance() already rejects.
_STRAIGHT_SINGLE_OPEN = r"(?<![\w'])'(?=[\w“\"])"
QUOTE_PATTERNS = {
    "curly-double": (
        CURLY_OPEN + r"(.+?)" + CURLY_CLOSE,
        r"[" + CURLY_OPEN + r'"][^' + CURLY_CLOSE + r'"]{2,}[' + CURLY_CLOSE + r'"]',
    ),
    "straight-double": (r'"(.+?)"', r'"[^"]{2,}"'),
    "curly-single": (
        SQ_OPEN + r"(.+?)" + SQ_CLOSE + r"(?![A-Za-z])",
        SQ_OPEN + r"[^" + SQ_OPEN + r"]{2,}?" + SQ_CLOSE + r"(?![A-Za-z])",
    ),
    "straight-single": (
        _STRAIGHT_SINGLE_OPEN + r"(.+?)'(?![\w])",
        _STRAIGHT_SINGLE_OPEN + r"[^']{2,}?'(?![\w])",
    ),
}

# How to COUNT each convention when voting. Counting raw characters does not work for the
# single forms — U+2019 and U+0027 are apostrophes far more often than quotes, so a novel
# with 4,391 apostrophes would win every vote. Count plausible OPENINGS instead.
_QUOTE_COUNTERS = {
    "curly-double": lambda t: min(t.count(CURLY_OPEN), t.count(CURLY_CLOSE)),
    "straight-double": lambda t: t.count('"') // 2,
    "curly-single": lambda t: t.count(SQ_OPEN),
    "straight-single": lambda t: len(re.findall(_STRAIGHT_SINGLE_OPEN, t)),
}


def detect_quote_style(text):
    """Pick the edition's dominant quote convention from the text itself.

    Per BOOK, not per paragraph: a single paragraph carries too few quotes to vote, and
    editions are internally consistent about this.
    """
    counts = {name: fn(text) for name, fn in _QUOTE_COUNTERS.items()}
    style, n = max(counts.items(), key=lambda kv: kv[1])
    return (style if n >= 5 else DEFAULT_QUOTE_STYLE), counts


def quote_re(style=None):
    """Compiled `open(.+?)close` for one convention."""
    return re.compile(QUOTE_PATTERNS.get(style, QUOTE_PATTERNS[DEFAULT_QUOTE_STYLE])[0])


def quote_span_re(style=None):
    """Compiled matcher for a whole quoted span, used to strip dialogue from narration."""
    return re.compile(QUOTE_PATTERNS.get(style, QUOTE_PATTERNS[DEFAULT_QUOTE_STYLE])[1])


def extract_utterances(paragraph, style=None):
    """Every quoted utterance in the paragraph, as (utterance, attribution).

    Replaces extract_dialogue(), which took `re.search(r"“(.+?)”")` — non-greedy,
    FIRST MATCH ONLY — and so had three failure modes, all of which reached the
    audit surface (2026-07-28):

      1. A SPLIT QUOTATION yielded only its first half. "Come here," she said,
         "and sit down." became `Come here,` — the commonest dialogue form in
         prose, reduced to a fragment ending on a comma.
      2. Everything after the first quote in a paragraph was DISCARDED, so
         multi-exchange paragraphs lost all but one utterance.
      3. Any curly-quoted span matched, so scare-quotes and quoted phrases became
         "dialogue": `the railroad magnate`, `especially cool,`.

    Split quotations are rejoined here rather than dropped — the words are the
    speaker's, and the halves reconstruct the line they actually said. Everything
    else is left to is_complete_utterance(), which rejects (1) and (3) on shape
    without needing to know why they are malformed.
    """
    spans = list(quote_re(style).finditer(paragraph))
    if not spans:
        return []
    verb = re.compile(r"\b(?:" + ATTRIB_VERBS + r")\b")

    merged, i = [], 0
    while i < len(spans):
        text, start, end = spans[i].group(1).strip(), spans[i].start(), spans[i].end()
        # absorb continuations: this half does not finish the sentence, and the
        # gap to the next quote is a short attribution rather than new narration
        while i + 1 < len(spans):
            gap = paragraph[end:spans[i + 1].start()]
            if len(gap) > 60 or not verb.search(gap):
                break
            if text.rstrip(_CLOSERS).rstrip()[-1:] in _TERMINALS:
                break
            # An attribution that CLOSES its sentence ends the utterance too:
            # “No,” she said. “That is not…” is two sentences, not one split
            # quotation, and merging them yields "No, That is not…".
            if gap.strip()[-1:] in _TERMINALS:
                break
            text = text + " " + spans[i + 1].group(1).strip()
            end = spans[i + 1].end()
            i += 1
        merged.append((text, start, end))
        i += 1

    out = []
    for text, start, end in merged:
        if not is_complete_utterance(text):
            continue
        remainder = (paragraph[:start] + " " + paragraph[end:]).strip()
        attr = ""
        am = re.search(r"([A-Z][\w' ]{0,40}?\b(?:" + ATTRIB_VERBS + r")\b[\w' ]{0,30})", remainder)
        if am:
            attr = am.group(1).strip()
        elif verb.search(remainder):
            attr = re.search(r".{0,25}\b(?:" + ATTRIB_VERBS + r")\b.{0,20}", remainder).group(0).strip()
        out.append((text, attr))
    return out


def build_chunks(sections, style=None):
    """dialogue chunks (play speeches, or quoted prose dialogue) + narration windows.

    `style` is the edition's quote convention (see `detect_quote_style`). Passing None
    keeps the historical curly-double behaviour, which is wrong for a meaningful share of
    Project Gutenberg editions — callers should detect it from the text.
    """
    dialogue, narration = [], []
    for si, (title, kind, items) in enumerate(sections):
        if kind == "drama":                              # play: every speech is dialogue
            for pi, speech in enumerate(items):
                for w in _chunk_speech(speech):
                    dialogue.append({
                        "chunk_type": "dialogue", "text": w,
                        "source_ref": {"section": si, "idx": pi, "kind": "drama", "attribution": "play dialogue"},
                    })
        else:                                            # prose: quoted dialogue + narration windows
            for pi, para in enumerate(items):
                # every utterance, not just the first; unattributed speech is kept
                # rather than silently dropping the whole paragraph, which is what
                # the old `if dlg and dlg[1]` did — a quote with no attribution
                # produced neither a dialogue chunk nor narration windows.
                utterances = extract_utterances(para, style)
                for quote, attr in utterances:
                    # Through _chunk_speech, not appended raw. Raw appends applied NO
                    # length bound at all, so one-second fragments — "Mr. Heathcliff?",
                    # "Rough weather!" — entered banks as passages. They are complete
                    # utterances, which is why the completeness gate passes them, but
                    # they carry no prosodic arc to perform or judge. This is also the
                    # bulk of the measured 37% too-short rate. _chunk_speech applies the
                    # floor AND windows an over-long speech to the engine ceiling.
                    for w in _chunk_speech(quote):
                        dialogue.append({
                            "chunk_type": "dialogue", "text": w,
                            "source_ref": {"section": si, "para": pi, "kind": "prose",
                                           "attribution": attr or "unattributed dialogue"},
                        })
                # Narration comes from EVERY prose paragraph, not only the quote-free
                # ones. `if not utterances:` used to gate this, so in a dialogue-heavy
                # novel each mixed paragraph contributed its quote and threw away the
                # surrounding prose — "He said nothing for a while. <quote> and turned
                # back to the window." lost both halves of the narration. Measured over
                # the 13 books on hand that is ~1,500 good passages discarded.
                # Quoted spans are removed first so dialogue is not also emitted as
                # narration; what remains is the narrator's own voice.
                src = para if not utterances else _strip_quotes(para, style)
                for sent_group in _window_sentences(split_sentences(src)):
                    if not is_complete_utterance(sent_group):
                        continue
                    narration.append({
                        "chunk_type": "narration", "text": sent_group,
                        "source_ref": {"section": si, "para": pi, "kind": "prose"},
                    })
    return dialogue, narration


def _chunk_speech(text):
    """Window a spoken line to the engine-reliable ceiling, applying the length floor.

    The docstring here used to claim terse dialogue was deliberately KEPT, which the
    code has never done — it has always applied MIN_CLIP_CHARS. Corrected rather than
    made true: a 15-character clip has no arc to perform, and the owner's 4 s floor was
    set from a measured keep-rate cliff.
    """
    if len(text) <= WINDOW_MAX_CHARS:
        # The completeness gate applies here too. It did not, and that is how
        # 'You do not seem to understand me,' — a fragment from the Shaw play — reached
        # the CERTIFIED pool, where ref_select can still cast it. Drama speeches take
        # this short path, so the gate was skipped for every play we ingested.
        if len(text) < MIN_CLIP_CHARS or not is_complete_utterance(text):
            return []
        return [text]
    out, cur = [], ""
    for s in split_sentences(text):
        if len(s) > WINDOW_MAX_CHARS:               # a single long sentence: take it alone
            if cur:
                out.append(cur.strip()); cur = ""
            # ...unless no engine can render it (QC-L2). `_window_sentences` grew this
            # guard from the 3,964-character / ~283 s passage that motivated
            # HARD_MAX_CHARS; the fix covered the NARRATION path only, and dialogue —
            # which is 64% of the corpus and where a single unbroken speech is most
            # likely — kept appending unbounded. The cost is GPU time spent rendering a
            # passage that qc_gate then throws away at its 30 s gate.
            if len(s) <= HARD_MAX_CHARS:
                out.append(s.strip())
            continue
        if len(cur) + len(s) + 1 <= WINDOW_MAX_CHARS:
            cur = (cur + " " + s).strip()
        else:
            out.append(cur.strip()); cur = s
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) >= MIN_CLIP_CHARS and is_complete_utterance(c)]


# Past this a passage is unrenderable, not merely long: Zonos caps at 30 s and
# Chatterbox warns past 300 characters. The old code took any over-long sentence
# "alone", which emitted one 3,964-character passage (~283 s) among others.
HARD_MAX_CHARS = 400


def _strip_quotes(para, style=None):
    """Paragraph with quoted dialogue removed, leaving the narrator's own prose."""
    return re.sub(r'\s+', ' ', quote_span_re(style).sub(' ', para)).strip()


def _window_sentences(sentences):
    windows, cur = [], ""
    for s in sentences:
        if len(s) > WINDOW_MAX_CHARS:               # a single long sentence: take it alone
            if cur:
                windows.append(cur.strip()); cur = ""
            if len(s) <= HARD_MAX_CHARS:            # ...unless no engine can render it
                windows.append(s.strip())
            continue
        if len(cur) + len(s) + 1 <= WINDOW_MAX_CHARS:
            cur = (cur + " " + s).strip()
        else:
            if len(cur) >= WINDOW_MIN_CHARS:
                windows.append(cur.strip())
            cur = s
    if len(cur) >= WINDOW_MIN_CHARS:
        windows.append(cur.strip())
    return windows


def librivox_check(title):
    """Informational router probe: does a permissive audiobook already exist?

    Returns a list of matching titles, or None for "could not tell".

    The API answers a zero-result query with **HTTP 404**, not an empty list, so
    the no-match case arrived here as an exception and was returned as a string —
    indistinguishable from a real outage, and main() printed "no LibriVox match ->
    SYNTHESIZE lane (correct)" for both. Routing does not actually depend on this
    (an SE/PG URL routes straight to SYNTHESIZE by owner policy 2026-07-22; the
    submission IS the lane decision), so nothing was mis-routed. But a log line
    asserting a negative it never established is the shape of finding that keeps
    turning up in this pipeline, so the two cases are now separated: 404 means no
    match, anything else means unknown.
    """
    q = urllib.parse.urlencode({"title": title, "format": "json"})
    try:
        data = json.loads(fetch("https://librivox.org/api/feed/audiobooks/?" + q)
                          .decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []          # the API's way of saying "nothing matched"
        print(f"  librivox probe unavailable: HTTP {e.code}", flush=True)
        return None
    except Exception as e:
        print(f"  librivox probe unavailable: {e}", flush=True)
        return None
    return [b.get("title") for b in data.get("books", [])][:5]


def _extract_json(content):
    """Robustly pull the JSON object out of a model reply (fences, prose, trailing commas)."""
    if not content:
        return None
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = content[start:end + 1]
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def director_tag(chunk, retries=2):
    """Call the live Gemma director; return the parsed VAD/engine/direction dict (or None).
    think=False: this is a fast structured judgment, not a reasoning task — skipping the
    chain-of-thought stops it from eating the token budget / bleeding into `content`."""
    if chunk["chunk_type"] == "dialogue":
        attr = chunk["source_ref"].get("attribution", "")
        user = f'A character speaks (attribution: "{attr}"). Their line: “{chunk["text"]}”'
    else:
        user = f"Narration passage: {chunk['text']}"
    for _ in range(retries):
        body = json.dumps({
            "model": MODEL, "stream": False, "think": False,
            "options": {"num_predict": 400, "temperature": 0.2},
            "messages": [
                {"role": "system", "content": DIRECTOR_SYSTEM},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                content = json.loads(r.read())["message"]["content"]
        except Exception:
            continue
        tag = _extract_json(content)
        if not tag:
            continue
        if REGISTER_LEXICON and tag.get("register") not in REGISTER_LEXICON:
            print(f"    off-lexicon register {tag.get('register')!r} -> neutral")
            tag["register"] = "neutral"
        # This used to default to "vibevoice" and coerce any unrecognised name to
        # it — the same silent-fallback shape that build_direction now treats as
        # fatal, and for the same reason: a line ends up rendered by an engine
        # nobody chose and audited under a name nobody wrote. It aged into
        # something worse than a mis-attribution, because vibevoice was set aside
        # on 2026-07-29, so the fallback pointed at an engine that must not render
        # at all. An off-roster emission is now treated exactly like unparseable
        # JSON: retry, and if the retries are spent, drop the chunk loudly rather
        # than substitute a choice.
        engine = tag.get("engine")
        if engine not in CASTABLE_ENGINES:
            print(f"    off-roster engine {engine!r} (castable: "
                  f"{', '.join(CASTABLE_ENGINES)}) — retrying", flush=True)
            continue
        tag["engine"] = engine
        # Second pass: casting + delivery, written in the chosen engine's own
        # language. Dia has no direction channel, so it is skipped entirely.
        if engine != "dia":
            cast = casting_pass(chunk["text"], engine, labels={
                "V": tag.get("valence", 0.0), "A": tag.get("arousal", 0.0),
                "T": tag.get("tension", 0.0), "register": tag.get("register", "")})
            if cast is None:
                continue
            tag.update(cast)
        return tag
    return None


SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "director_skills")

# Orpheus's two closed sets. Both are soft-closed: the shipped code validates
# NEITHER (its voice-check function is dead code), so an unlisted voice name or tag
# is interpolated into the prompt and spoken aloud as text. These live here, next to
# the schema that publishes them, and orpheus.md is gated against them by
# scripts/test_skill_files.py.
ORPHEUS_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
ORPHEUS_TAGS = ("<laugh>", "<chuckle>", "<sigh>", "<cough>",
                "<sniffle>", "<groan>", "<yawn>", "<gasp>")

# Voices that measured defective and must never be cast. `tara` drew room tone
# plus white-noise hiss on 5 of 5 clips (2026-07-28) and orpheus.md says "Never
# cast" — but that ban lived only in markdown, while this module's fallback was
# literally `tara`, so every director hiccup cast the worst-measured voice into
# a 15%-of-mix engine. A rule a renderer cannot read is not a rule.
ORPHEUS_BANNED = ("tara",)
ORPHEUS_CASTABLE = tuple(v for v in ORPHEUS_VOICES if v not in ORPHEUS_BANNED)
# jess is the measured-reliable female voice (0/14 defects in the probe,
# ~3% pooled) — the right place to land when casting is unusable.
ORPHEUS_FALLBACK = "jess"

# Every engine build_direction knows how to assemble a payload for. dia and moss85
# take no casting pass, so they are here but not in CASTING_SCHEMA. Anything absent
# is a hard error rather than a silent rewrite — see build_direction().
KNOWN_ENGINES = ("vibevoice", "dia", "qwen", "moss_vg", "moss85",
                 "chatterbox", "zonos", "orpheus", "longcat")

# The delivery lanes that are a NARRATOR reading, as opposed to a character
# speaking. Kept here rather than imported from ref_select so build_direction has
# no import-time dependency on the allocation tables — this is a property of the
# lane vocabulary (ARCHITECTURE §1), which is contract, not of the render mix.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from matcha.delivery import NARRATION_LANES  # noqa: E402,F401

# Chatterbox reads narration too fast at the skill file's provisional 0.5. See the
# chatterbox branch of build_direction for the measurement and why this is a
# ceiling rather than a new default.
NARRATION_MAX_EXAGGERATION = 0.4

CASTING_SYSTEM = (
    "You are the Casting and Delivery Director for an audiobook TTS pipeline. You "
    "write direction for ONE NAMED TTS ENGINE. A skill file for that engine follows; "
    "it describes what that engine can and cannot actually be told. FOLLOW IT "
    "EXACTLY — writing direction the engine cannot act on is worse than writing "
    "none.\n"
    "Do not emit valence, arousal, tension or register — those are already fixed for "
    "this line.\nJSON only."
)

# Engine-shaped output schema. qwen and moss_vg read exactly ONE string, so asking
# for two fields made the director emit its whole description twice — wasting the
# token budget into truncated JSON and duplicating the instruction. vibevoice
# genuinely has two: design is casting input, instruct is an audit-only note.
#
# The revisit engines (2026-07-28) are not prose-directable at all, so their schemas
# are parameters, not sentences — the shape of each entry follows the shape of the
# engine, exactly as its skill file does. `voice_design` appears wherever RELAY says
# design == "casting": it never reaches the model, it selects the reference clip
# through ref_select, and without it those engines draw an arbitrary voice per seed
# (which is what produced Zonos's "gender-coverage gap" in the void 2026-07-17 run).
CASTING_SCHEMA = {
    "qwen":       {"instruct": "string"},
    "moss_vg":    {"instruct": "string"},
    "vibevoice":  {"voice_design": "string", "instruct": "string"},
    "chatterbox": {"voice_design": "string",
                   "exaggeration": "number 0.25-1.0",
                   "cfg_weight": "number 0.2-0.6"},
    "zonos":      {"voice_design": "string",
                   "emotion": "array of EXACTLY 8 numbers, proportions not magnitudes, "
                              "order [happiness,sadness,disgust,fear,surprise,anger,other,neutral]"
                              " — or null to turn emotion conditioning OFF, "
                              "which is what plain narration wants",
                   "pitch_std": "number 20-150",
                   "speaking_rate": "number 5-30"},
    "orpheus":    {"voice": "EXACTLY one of: " + "|".join(ORPHEUS_CASTABLE),
                   "render_text": "string — the line verbatim, optionally with at most "
                                  "2 tags from " + " ".join(ORPHEUS_TAGS)},
    "longcat":    {"voice_design": "string"},
}


def _schema_str(engine):
    return "{" + ", ".join(f'"{k}": {v}' for k, v in CASTING_SCHEMA[engine].items()) + "}"


# Machine-enforced shape for ollama's structured-output `format`. Prompt wording alone
# does NOT hold: chatterbox.md is long and table-rich, and the director answered it
# with a prose "Recommended Configuration" table, and later with ZONOS's schema —
# right engine named, right skill file loaded, wrong shape emitted (2026-07-28).
# Moving the contract after the skill file bought a working smoke test and nothing
# more; one more table in the file broke it again. Constrained decoding makes the
# shape structural instead of persuasive. Ranges and closed sets are still checked in
# _validate_casting — this guarantees the KEYS and TYPES, not the values.
_NUM = {"type": "number"}
_STR = {"type": "string"}
CASTING_JSON_SCHEMA = {
    "qwen":       {"instruct": _STR},
    "moss_vg":    {"instruct": _STR},
    "vibevoice":  {"voice_design": _STR, "instruct": _STR},
    "chatterbox": {"voice_design": _STR, "exaggeration": _NUM, "cfg_weight": _NUM},
    # emotion is nullable on purpose: null is the only way to say "emotion off",
    # which is what narration wants. A required 8-float array made the director
    # physically unable to emit the state the renderer implements.
    "zonos":      {"voice_design": _STR,
                   "emotion": {"type": ["array", "null"], "items": _NUM,
                               "minItems": 8, "maxItems": 8},
                   "pitch_std": _NUM, "speaking_rate": _NUM},
    "orpheus":    {"voice": {"type": "string", "enum": list(ORPHEUS_CASTABLE)},
                   "render_text": _STR},
    "longcat":    {"voice_design": _STR},
}


def _json_schema(engine):
    props = CASTING_JSON_SCHEMA[engine]
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def _validate_casting(engine, d):
    """Reject a director emission the engine would mis-render, so casting_pass retries.

    Every rule here is a failure some engine actually exhibits, not defensive
    programming: an invented Orpheus voice name is interpolated into the prompt and
    SPOKEN aloud (the validation function in the shipped code is dead), a short Zonos
    emotion vector silently reshapes the conditioning, and a Chatterbox exaggeration
    emitted without its cfg_weight reads rushed. Cheap to check, expensive to audit.
    """
    if engine == "orpheus":
        if d.get("voice") not in ORPHEUS_CASTABLE:
            return False
        bad = set(re.findall(r"<[a-z]+>", d.get("render_text", ""))) - set(ORPHEUS_TAGS)
        return not bad
    if engine == "zonos":
        e = d.get("emotion")
        # None is valid and load-bearing: it means emotion conditioning off.
        if e is not None:
            if not isinstance(e, list) or len(e) != 8:
                return False
            if not all(isinstance(x, (int, float)) and x >= 0 for x in e) or sum(e) <= 0:
                return False
        return _in_range(d.get("pitch_std"), 20, 150) and _in_range(d.get("speaking_rate"), 5, 30)
    if engine == "chatterbox":
        return (_in_range(d.get("exaggeration"), 0.25, 1.0)
                and _in_range(d.get("cfg_weight"), 0.2, 0.6))
    return True


def _in_range(v, lo, hi):
    return isinstance(v, (int, float)) and lo <= v <= hi


def _clamp(v, lo, hi, default):
    return min(max(float(v), lo), hi) if isinstance(v, (int, float)) else default


def _l1(vec):
    """Normalise Zonos's 8-float emotion vector to the shape the model receives.

    Zonos L1-normalises internally, so magnitudes never arrive; doing it here makes
    the manifest record what was actually conditioned on.

    Returns None when there is no usable vector, and None means "emotion truly
    off" — synth_zonos adds `emotion` to unconditional_keys for it. This used to
    fall back to the neutral-dominant vector [.05,…,.77], which is NOT off: it is
    the exact conditioning measured destabilizing 5 of 9 zonos narration groups
    on 2026-07-30 (clause-boundary pauses, rushed resumption, skipped words). The
    renderer's fix and the skill file's "omit emotion for narration" were both
    unreachable through the director because this function could not express it.
    """
    if not isinstance(vec, list) or len(vec) != 8:
        return None
    vals = [max(float(x), 0.0) if isinstance(x, (int, float)) else 0.0 for x in vec]
    total = sum(vals)
    if total <= 0:
        return None
    return [round(v / total, 4) for v in vals]


def load_skill(engine):
    """Load a per-engine director adapter.

    Refuses WIP files: `director_skills/sonora.md` is written ahead of the interface
    it describes and must never reach a live director pass. Encoding the refusal here
    rather than trusting nobody sets engine="sonora" is the same discipline the relay
    audit was written to enforce.
    """
    path = os.path.join(SKILL_DIR, f"{engine}.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "WIP — NOT LOADED" in text or "WIP - NOT LOADED" in text:
        raise RuntimeError(
            f"{engine}.md is marked WIP and must not be loaded into a director pass")
    return text


def casting_pass(text, engine, labels=None, retries=2):
    """Per-engine casting/delivery, governed by director_skills/<engine>.md.

    `labels` is the line's already-decided {V, A, T, register}, passed as READ-ONLY
    context. Step 5 of the onboarding pattern separates the line pass from the engine
    pass so the training labels stop drifting with whichever engine is being written
    for — but separating the passes was mistaken for withholding the labels, and for
    a parameter-only engine that is fatal. Chatterbox's whole output is `exaggeration`,
    an AROUSAL dial, and the director was being asked to choose it with the arousal
    withheld: across 20 registers it emitted (0.25, 0.3) on 18 of them, including
    victory and urgency (2026-07-28). Knowing the label is not the same as relabelling;
    emitting one is still forbidden below.
    """
    # The output contract is repeated AFTER the skill file, and that placement is
    # load-bearing. Stated only before it, the last thing the director reads is a
    # long markdown document full of tables — and it answers in kind: chatterbox.md
    # reliably produced a prose "Recommended Configuration" table and zero JSON
    # (2026-07-28 smoke test), while the shorter skill files happened to survive.
    # Recency wins, so the contract goes last.
    schema = _schema_str(engine)
    system = (CASTING_SYSTEM
              + "\nOutput ONLY compact minified JSON, no markdown, with EXACTLY "
              + "these keys:\n" + schema
              + "\n\n===== SKILL FILE: " + engine + " =====\n" + load_skill(engine)
              + "\n===== END SKILL FILE =====\n\n"
              + "Reply with ONE line: the minified JSON object and nothing else. No "
              + "prose, no markdown, no table, no code fence, no explanation.\n"
              + "Required keys, exactly these: " + schema)
    user = f"Target engine: {engine}\n\nThe line to be performed:\n“{text}”"
    if labels:
        vat = ", ".join(f"{k}={labels[k]:+.2f}" for k in ("V", "A", "T") if k in labels)
        user += ("\n\nAlready decided for this line — treat as FIXED CONTEXT you must "
                 "serve, never restate:\n"
                 f"  register: {labels.get('register', 'unspecified')}\n"
                 f"  valence/arousal/tension: {vat}\n"
                 "Your direction must FIT THIS LINE. Direction identical to what you "
                 "would write for a different register is a failure.")
    for _ in range(retries):
        body = json.dumps({
            "model": MODEL, "stream": False, "think": False,
            "format": _json_schema(engine),
            "options": {"num_predict": 900, "temperature": 0.2},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(OLLAMA, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                content = json.loads(r.read())["message"]["content"]
        except Exception:
            continue
        d = _extract_json(content)
        # Required keys ARE the schema keys. They were hard-coded to ("instruct",)
        # for everything but vibevoice, which silently accepted a chatterbox emission
        # carrying no numbers at all — the schema is the contract, so read it there.
        if d and all(k in d for k in CASTING_SCHEMA[engine]) and _validate_casting(engine, d):
            return d
    return None


def slug_from_url(url):
    path = urllib.parse.urlparse(url).path.rstrip("/")
    seg = (path.split("/")[-1] or "book").lower()
    return re.sub(r"[^a-z0-9-]", "", seg) or "book"


def _merge(vd, ins):
    """One instruction string: persona sentence, then the delivery imperative.

    Guards against duplication: a director following a single-string skill file
    (MOSS, Qwen) often returns the same sentence in both fields, and blindly
    concatenating would say it twice.
    """
    head, tail = (vd or "").strip(), (ins or "").strip()
    if not head:
        return tail
    if not tail:
        return head
    a, b = head.rstrip(".!?").casefold(), tail.rstrip(".!?").casefold()
    if a == b or a in b:
        return tail
    if b in a:
        return head
    if head[-1] not in ".!?":
        head += "."
    return head + " " + tail


def build_direction(tag, text, dia_guidance=3.0, lane=None):
    """Assemble the per-engine `direction` payload.

    `lane` is the delivery lane (Dialogue / Neutral / Documentary / Newscaster /
    Speech). Optional, so every existing caller is unaffected; passing it lets this
    function enforce the lane rules a skill file can only ask for — see the zonos
    branch, where narration is forced unconditional.

    Single source of truth for what each renderer is actually handed — the bank
    builders previously disagreed, and moss85 silently lost its whole voice
    design in the quote-pilot/director-bench banks because those builders wrote
    a `design` key that synth_moss85.py never reads (owner finding 2026-07-25).

    Renderer contracts, verified against each model's shipped API:
      vibevoice -> design feeds ref_select only (gender + age band); the model
                   itself gets text + a reference wav and has NO instruct slot.
      qwen      -> `generate_voice_design(text, instruct, language)`. There is
                   NO `voice_description` parameter; passing one lands it in
                   **kwargs where it is silently dropped. A single merged
                   instruct string is the only channel that reaches the model.
      moss85    -> instruct only, so design MUST be merged in here too.
      dia       -> render_text only; no instruct channel at all.
      chatterbox-> two numbers (exaggeration + cfg_weight); design casts the
                   reference clip. No prose slot exists.
      zonos     -> emotion SHAPE + pitch_std + speaking_rate; design casts the
                   speaker embedding. No prose slot exists.
      orpheus   -> a voice from a closed 8, prefixed onto the text by the renderer;
                   optional inline tags from a closed 8. Nothing else.
      longcat   -> nothing at all. Reference selection is the whole decision.
    """
    engine = tag.get("engine", "vibevoice")
    if engine not in KNOWN_ENGINES:
        # Previously this silently rewrote anything unrecognised to "vibevoice".
        # That is the exact failure this function was written to end: a bank line
        # tagged `zonos` would have been handed to VibeVoice and audited as Zonos.
        # It also caught `moss_vg` — a live portfolio engine that was never in the
        # whitelist — so its lines were rewritten too. Unknown engines are fatal.
        raise ValueError(
            f"build_direction: unknown engine {engine!r}. Known: {', '.join(KNOWN_ENGINES)}. "
            f"Add it here and to RELAY in audition/app/main.py before rendering a clip.")
    vd = tag.get("voice_design", "") or ""
    ins = tag.get("instruct", "") or ""
    if engine == "vibevoice":
        # design is casting metadata, not direction; keep it verbatim for
        # ref_select's gender/age parse. instruct is retained for audit only.
        return engine, {"design": vd, "instruct": ins}
    if engine in ("qwen", "moss_vg", "moss85"):
        # the casting pass already emitted ONE string for these; _merge only has
        # work to do on legacy tags that still carry a separate voice_design.
        direction = {"instruct": _merge(vd, ins)}
        # B-M7. `synth_moss85.py` reads `direction["quality"]` — a real renderer input,
        # the 2.1B recipe's recording-condition string — and this function had no slot for
        # it. So a moss85 line built HERE lost it, while `make_bulk_bank.py` kept it by
        # bypassing this function. That is the bypass explaining itself: the builders
        # diverged because the SSOT could not express what the renderer accepts. Closing
        # the gap here is what makes "never bypass build_direction" a rule rather than a
        # wish. moss_vg and qwen have no such parameter, so it is moss85-only.
        if engine == "moss85" and tag.get("quality"):
            direction["quality"] = tag["quality"]
        return engine, direction
    if engine == "chatterbox":
        # Two numbers and a casting call. `exaggeration` is a RATE PROFILE, not an
        # emotion selector, and raising it alone reads rushed — cfg_weight is the
        # other half of one control, so it is defaulted rather than left absent.
        #
        # NARRATION CEILING, measured 2026-08-03 — and it contradicts the skill
        # file, which is why it is a ceiling rather than a new default. chatterbox.md
        # says Neutral = 0.5 and "do NOT drift lower; low exag reads subdued/ironic",
        # but that guidance was marked "provisional, not yet auditioned as narration"
        # and rested on a 7/7 dialogue record. The first narration audition says the
        # opposite: 3 of 19 heard clips came back "too rapid of a rate", "a bit too
        # hurried of a pace", "a bit too rapid" — two of them at 0.5.
        #
        # The warning it overrides was established on DIALOGUE. "Subdued" may well
        # be the target in a narration lane rather than a defect, which is exactly
        # what the reroll at 0.35 is meant to settle. Until it does, cap rather than
        # move the default, so a director asking for 0.5 on narration gets the slowest
        # value the evidence supports instead of the fastest.
        exag = _clamp(tag.get("exaggeration"), 0.25, 1.0, 0.5)
        if lane in NARRATION_LANES:
            exag = min(exag, NARRATION_MAX_EXAGGERATION)
        return engine, {"design": vd,
                        "exaggeration": exag,
                        "cfg_weight": _clamp(tag.get("cfg_weight"), 0.2, 0.6, 0.5)}
    if engine == "zonos":
        # The emotion vector is silently L1-normalised downstream, so we normalise
        # here too: what the model receives is then exactly what the manifest
        # records, and an audit reading "happiness 0.85" can never again mean 0.616.
        emotion = _l1(tag.get("emotion"))
        # NARRATION IS UNCONDITIONAL, ENFORCED — not requested. zonos.md says
        # `emotion` is omitted on every narration lane "without exception", because
        # a neutral-dominant vector is the measured cause of the pause-and-rush
        # instability that broke 5 of 9 narration groups on 2026-07-30 (8% keep on
        # Neutral against 91% with it off). The director obeys that ~97% of the
        # time, which is not the same as always: one line of 38 in
        # delivery-v1-narration-r2 came back `[0,…,0.2,0.8]` — the exact shape.
        # Leaving it to the prompt makes the rule a wish, and this pipeline has
        # already learned that lesson once, from a banned voice that stayed the
        # director's fallback. `lane` is optional so existing callers are unchanged.
        if emotion is not None and lane in NARRATION_LANES:
            emotion = None
        return engine, {"design": vd,
                        "emotion": emotion,
                        "pitch_std": _clamp(tag.get("pitch_std"), 20, 150, 45.0),
                        "speaking_rate": _clamp(tag.get("speaking_rate"), 5, 30, 15.0)}
    if engine == "orpheus":
        # The only string that reaches the model is f"{voice}: {text}", assembled in
        # the renderer. An unlisted voice would be spoken aloud, so fall back to the
        # maintainers' default rather than trusting the emission.
        voice = (tag.get("voice") if tag.get("voice") in ORPHEUS_CASTABLE
                 else ORPHEUS_FALLBACK)
        rt = tag.get("render_text") or text
        if set(re.findall(r"<[a-z]+>", rt)) - set(ORPHEUS_TAGS):
            rt = text     # an invented tag is pronounced; drop the whole emission
        return engine, {"voice": voice, "render_text": rt}
    if engine == "longcat":
        # Nothing is directable. Casting the reference is the entire decision.
        return engine, {"design": vd}
    # dia: control is inline text. nari-labs generation guidelines say to repeat
    # the trailing speaker tag to improve end-of-audio quality (Dia improvises
    # tails otherwise) — see the docs' "Generation Guidelines".
    # Temperature 1.8 is CORRECT for Dia and was re-confirmed the hard way
    # (2026-07-26): dropping to 1.5 to curb over-generation instead pushed 7/20
    # clips into white-noise collapse — ASR word-error 0.79-1.00 with DNSMOS
    # 1.2-1.8, the same failure the 2026-07-17 audit saw at 1.3-1.4. Over-length
    # was never a temperature problem; it was synth_dia.token_budget(). Do not
    # lower this again without re-reading that experiment.
    #
    # B-M7. Inline tags ("(laughs) ") are a real Dia channel and `bulk_spec.json` uses
    # one, but this function had no slot for them — so `make_bulk_bank.py` built its own
    # render_text and, in doing so, dropped the TRAILING `[S1]`. That trailing tag is not
    # decoration: it is the end-of-audio guard nari-labs' guidelines prescribe, and
    # without it Dia improvises a tail. A bank built by that path renders long and wrong
    # while looking correctly directed.
    tags = (tag.get("dia_tags") or "").strip()
    body = f"{tags} {text}".strip() if tags else text
    return engine, {"render_text": f"[S1] {body} [S1]",
                    "temperature": 1.8, "guidance": dia_guidance}


def chunk_key(chunk):
    """Stable identity for a chunk, for the director pass's checkpoint (A-M11).

    Content, not position. `sample` is deterministic for identical arguments, but a resume
    with a different `--max-per-type` reshuffles every index — and an index-keyed
    checkpoint would then hand chunk 40's direction to chunk 37, which is worse than
    having no checkpoint at all because the bank still looks complete.
    """
    blob = f"{chunk['chunk_type']}\x00{chunk['text']}".encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def load_director_checkpoint(path):
    """-> {chunk_key: bank_line} from a partial run. Tolerates a torn final line."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                # The last line of a killed run can be half-written. Everything before it
                # is still good, and that is the entire point of the file.
                print("  .. checkpoint: ignoring one torn trailing record", flush=True)
                continue
            if rec.get("chunk_key") and rec.get("line"):
                done[rec["chunk_key"]] = rec["line"]
    return done


def to_bank_line(idx, chunk, tag, slug, seed=1234):
    text = chunk["text"]
    engine, direction = build_direction(tag, text)
    return {
        "id": f"{slug}_{chunk['chunk_type'][:3]}_{idx:04d}",
        "engine": engine,
        "register": tag.get("register", "unspecified"),
        "chunk_type": chunk["chunk_type"],
        "intended": {"V": tag.get("valence", 0.0), "A": tag.get("arousal", 0.0), "T": tag.get("tension", 0.0)},
        "seed": seed,
        "text": text,
        "direction": direction,
        "source_ref": chunk["source_ref"],
    }


def main():
    ap = argparse.ArgumentParser()
    # --epub: ingest an epub already on disk. Every book we have routed keeps its
    # book.epub beside its output, and re-downloading to re-ingest is both wasteful and
    # a live dependency on the source site staying up. Exactly one of --url/--epub.
    ap.add_argument("--url", default=None)
    ap.add_argument("--epub", default=None, help="local .epub path (instead of --url)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--slug", default=None, help="id/campaign slug; derived from the URL if omitted")
    ap.add_argument("--dry-run", action="store_true", help="parse + chunk + router only; no director calls")
    ap.add_argument("--max-per-type", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    slug = args.slug or slug_from_url(args.url)

    print(f"== book_ingest: {slug} ==", flush=True)
    print("== router probe (LibriVox) ==", flush=True)
    lv = librivox_check(args.title) if args.title else "(no title given; skipped)"
    print("  LibriVox matches:", lv, flush=True)
    if isinstance(lv, list) and lv:
        print("  ⚠️ router: a LibriVox recording exists — normally this routes REAL-AUDIO "
              "(force-align lane). Running SYNTHESIZE anyway (explicit).", flush=True)
    elif lv is None:
        print("  router: LibriVox could NOT be checked — this is not a negative result. "
              "Proceeding on the queue's routing decision, which is what governs.",
              flush=True)
    else:
        print("  router: no LibriVox match → SYNTHESIZE lane (correct).", flush=True)

    print("== fetch + parse ==", flush=True)
    if not args.url and not args.epub:
        sys.exit("need --url or --epub")
    epub_url = None if args.epub else (
        se_epub_url(args.url) if "standardebooks.org" in args.url else args.url)
    print("  epub:", epub_url, flush=True)
    # `args.url` and not `epub_url`: for Standard Ebooks the two differ (the epub link is
    # resolved off the page), and it is the SOURCE that decides whether PG's wrapper needs
    # cutting. A locally supplied --epub passes None, so the residue check still runs and
    # a PG file handed over by hand is refused rather than silently rendered.
    sections = parse_epub(open(args.epub, "rb").read() if args.epub else fetch(epub_url),
                          source_url=args.url)
    n_drama = sum(1 for _, k, _ in sections if k == "drama")
    total_items = sum(len(items) for _, _, items in sections)
    print(f"  sections: {len(sections)} ({n_drama} drama / {len(sections) - n_drama} prose)  units: {total_items}", flush=True)

    # A-M7: read the edition's quote convention off the whole book before chunking.
    # Printed because it decides whether dialogue is found at all, and a wrong answer is
    # otherwise invisible — it shows up as a suspiciously narration-heavy novel.
    style, quote_counts = detect_quote_style(
        "\n".join(p for _, _, items in sections for p in items))
    print(f"  quote style: {style}  {quote_counts}", flush=True)

    dialogue, narration = build_chunks(sections, style)
    print(f"== chunks ==  dialogue+attribution: {len(dialogue)}   narration windows: {len(narration)}", flush=True)

    # sample spread across the book
    def spread(lst, n):
        if len(lst) <= n:
            return lst
        step = len(lst) / n
        return [lst[int(i * step)] for i in range(n)]

    sample = spread(dialogue, args.max_per_type) + spread(narration, args.max_per_type)
    print(f"  sampled {len(sample)} chunks for tagging ({args.max_per_type}/type)", flush=True)

    if args.dry_run:
        with open(os.path.join(args.out, "chunks_preview.json"), "w") as f:
            json.dump({"dialogue": dialogue[:20], "narration": narration[:20]}, f, indent=2, ensure_ascii=False)
        print("  DRY RUN — wrote chunks_preview.json; no director calls.", flush=True)
        for c in sample[:6]:
            print(f"    [{c['chunk_type']}] {c['text'][:90]}", flush=True)
        return

    print(f"== director-pass ({MODEL} via ollama) ==", flush=True)

    # A-M11. `lines` used to accumulate in memory and reach disk only after the last
    # chunk, so ANY interruption — a `load_skill` error at chunk 90, an ollama restart,
    # Ctrl-C — discarded every director call made so far. Each one is a 31B inference; a
    # 200-chunk book is an hour of them. Now each result is appended as it is produced and
    # a re-run picks up where it stopped.
    #
    # FAILURES ARE NOT CHECKPOINTED, deliberately. Same reasoning as D-M6: a transiently
    # failed chunk must be retried on the next run rather than recorded as done and
    # dropped forever. The cost is re-attempting a permanently malformed chunk each time,
    # which is bounded and visible in the failure count.
    partial_path = os.path.join(args.out, f"{slug}_bank.partial.jsonl")
    done = load_director_checkpoint(partial_path)
    if done:
        print(f"  resuming: {len(done)} chunk(s) already directed in a previous run",
              flush=True)

    lines, failures, resumed = [], 0, 0
    with open(partial_path, "a", encoding="utf-8") as checkpoint:
        for i, chunk in enumerate(sample):
            key = chunk_key(chunk)
            if key in done:
                lines.append(done[key])
                resumed += 1
                continue
            tag = director_tag(chunk)
            if not tag:
                failures += 1
                print(f"  [{i}] FAILED to parse director JSON ({chunk['chunk_type']})", flush=True)
                continue
            line = to_bank_line(i, chunk, tag, slug)
            lines.append(line)
            checkpoint.write(json.dumps({"chunk_key": key, "line": line},
                                        ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
            print(f"  [{i}] {chunk['chunk_type']:9} eng={line['engine']:6} "
                  f"V={line['intended']['V']:+.1f} A={line['intended']['A']:+.1f} T={line['intended']['T']:+.1f} "
                  f"{line['register']:22} | {chunk['text'][:55]}", flush=True)
    if resumed:
        print(f"  reused {resumed} directed chunk(s) from the checkpoint", flush=True)

    src_name, src_license, src_note = text_provenance(args.url, args.epub)
    if src_license == "UNKNOWN":
        print(f"  !! {src_note}", flush=True)
    bank = {
        "version": f"book-{slug}-1",
        "campaign": f"book-{slug}",
        "license_note": f"{src_note[:-1]} ({args.title or slug}, {args.author}). "
                        "Synthetic audio from Apache/MIT engines. Director: Gemma 4 (Apache-2.0).",
        "text_provenance": f"book:{slug} ({src_name}, {src_license})",
        "source": {"url": args.url, "epub": epub_url, "title": args.title, "author": args.author,
                   "text_source": src_name, "text_license": src_license,
                   "router_librivox": lv},
        "lines": lines,
    }
    out = os.path.join(args.out, f"{slug}_bank.json")
    synth_common.write_json_atomic(out, bank, indent=2)
    # Only once the real bank is safely on disk. Until then the checkpoint is the only
    # record of an hour of 31B inference.
    try:
        os.remove(partial_path)
    except OSError:
        pass
    print(f"== DONE ==  wrote {len(lines)} bank lines ({failures} failures) -> {out}", flush=True)


if __name__ == "__main__":
    main()

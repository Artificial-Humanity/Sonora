"""book_ingest — prototype front-end for the book-prose synthesis lane.

Fetch a permissive ebook (Standard Ebooks / Project Gutenberg) -> parse -> chunk
(narration windows + dialogue-with-attribution) -> Gemma-4 director-pass (VAD +
register + per-engine direction, via the live ollama endpoint) -> emit the flat
bank the synth_{vibevoice,dia,qwen,moss_vg}.py renderers consume.

PROTOTYPE NOTES
- Parsing/segmentation are done in Python here to validate the LANE fast. Production
  should converge onto Prosodia's `folioparser` (EPUB->text) + `stage::segmenter`
  (sentence split + Paragraph{target_characters}) for on-device dogfooding — see
  notes/book-prose-operations.md.
- Director = gemma-4-26b-a4b-qat served by ollama on :11434 (reasoning model; read
  `content`, give generous num_predict).

Run:
  uv run --with ebooklib --with beautifulsoup4 --with lxml --with pysbd \
    Sonora/scripts/synthesis/book_ingest.py --url <SE url> --out <dir> [--dry-run] [--max-per-type N]
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (book_ingest prototype; contact lmcfarlin)"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma-4-26b-a4b-qat"

CHARS_PER_SEC = 14.0            # mirrors synth_dia.py length model
# Owner floor (2026-07-25): nothing shorter than 4 s of speech enters a bank.
# Purpose, in the owner's words: "avoid content too short to gauge performance
# around". This gates INPUT TEXT, not render duration — a clip that lands slightly
# under 4 s because the engine spoke fast is fine. Do NOT add an output-duration
# gate at 4 s; the QC gate's duration-vs-text check already catches real failures.
# Audit evidence — moss85 keep rate by estimated length: <2 s 60%, 2-4 s 40%,
# 4-10 s 91%, 10 s+ 100%. Short lines are also where MOSS reads the prompt
# aloud (the instruction:text ratio hits 13x on ~1 s dialogue fragments), and
# nari-labs document that Dia input under ~5 s "will sound unnatural".
MIN_CLIP_SECONDS = 4.0
MIN_CLIP_CHARS = int(MIN_CLIP_SECONDS * CHARS_PER_SEC)   # 56
WINDOW_MIN_CHARS = 90          # ~6 s of speech
WINDOW_MAX_CHARS = 240        # ~17 s — engine-reliable ceiling
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

# Engine choice only. V/A/T + register are a SEPARATE pass (they describe the
# line, not the target), and voice_design/instruct are a THIRD pass driven by the
# per-engine skill files in director_skills/. Splitting these was forced by the
# 2026-07-25 finding that a single combined call let the training labels drift
# with whichever engine the director happened to be writing for.
DIRECTOR_SYSTEM = (
    "You are the Emotional Director for an audiobook TTS pipeline. You read one passage "
    "(narration, or a character's spoken line with its attribution) and label it, then "
    "choose which speech engine should render it.\n"
    "Output ONLY compact minified JSON, no markdown, with EXACTLY these keys:\n"
    '{"valence": float in [-1,1], "arousal": float in [-1,1], "tension": float in [-1,1], '
    '"register": string, "engine": one of "vibevoice"|"qwen"|"moss_vg"|"dia"}\n'
    "`register` MUST be copied EXACTLY from this controlled lexicon — never invent, "
    "alter or compose a label; pick the closest fit:\n"
    + ", ".join(REGISTER_LEXICON) + "\n"
    "Engine guide: vibevoice = PREMIER default incl. neutral. It is REFERENCE-CLONED "
    "and takes NO spoken direction at all — casting is everything, so choose it when "
    "the voice matters more than the performance. qwen and moss_vg = the two engines "
    "that actually follow written direction; prefer them when strong DRAMATIC "
    "expression is expected (qwen renders younger and higher than described; moss_vg "
    "handles dark/menace/oratory/force and situational framing well). dia = takes NO "
    "written direction, but is NOT a flat reader: it is markedly expressive and its "
    "punctuation is performed, so write the punctuation you want. Its measured quality "
    "is the weakest of the four, so prefer another engine when the line matters.\n"
    "Do NOT write a voice design or delivery instruction here; a later pass does that "
    "per engine. Valence = pleasant(+)/unpleasant(-); Arousal = energy; Tension = "
    "held/threat/unease. JSON only."
)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


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


def parse_epub(epub_bytes):
    """Return [(chapter_title, [paragraph_text, ...]), ...] for chapter documents only."""
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
    return sections


def split_sentences(text):
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in seg.segment(text) if s.strip()]


_OPENERS = "“\"'([‘"
_CLOSERS = "”\"')]’"
_TERMINALS = ".!?…"


def is_complete_utterance(text):
    """Is this text a whole utterance a listener could judge a performance of?

    THE GATE THAT MATTERS FOR AUDITABILITY (owner, 2026-07-28). A clip that is
    merely short stays judgeable — the owner has knowingly let short-but-whole
    clips through, because the sentence is still there. A clip that is INCOMPLETE
    destroys the judgement itself: prosody lives in the arc of a finished
    utterance, so a fragment ending on a comma has no terminal contour to assess
    and no amount of vocal quality rescues it.

    This is why the gate is completeness and NOT a token-count floor.
    audit-markup-v0 read its ≤6-token failures as a LENGTH problem and prescribed
    a floor; length was the correlate, not the cause. A floor would reject good
    short-but-whole clips and pass every long fragment.

    Note the rating vocabulary cannot see this defect: under v4 the score means
    vocals and prosody only, so fragments were rated 4-5 for a fine-sounding voice
    and are statistically invisible in ratings.csv. 79 of them reached the audit
    surface across two independent lanes. The instrument will not catch this; the
    gate has to.
    """
    t = (text or "").strip()
    if not t:
        return False
    core = t.rstrip(_CLOSERS).rstrip()
    if not core or core[-1] not in _TERMINALS:
        return False        # ends on a comma, dash, conjunction — mid-utterance
    head = t.lstrip(_OPENERS).lstrip()
    if not head or not (head[0].isupper() or head[0].isdigit()):
        return False        # starts mid-word or mid-clause ("se it was true—")
    return True


def extract_utterances(paragraph):
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
    spans = list(re.finditer(r"“(.+?)”", paragraph))
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


def build_chunks(sections):
    """dialogue chunks (play speeches, or quoted prose dialogue) + narration windows."""
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
                utterances = extract_utterances(para)
                for quote, attr in utterances:
                    dialogue.append({
                        "chunk_type": "dialogue", "text": quote,
                        "source_ref": {"section": si, "para": pi, "kind": "prose",
                                       "attribution": attr or "unattributed dialogue"},
                    })
                if not utterances:
                    for sent_group in _window_sentences(split_sentences(para)):
                        if not is_complete_utterance(sent_group):
                            continue
                        narration.append({
                            "chunk_type": "narration", "text": sent_group,
                            "source_ref": {"section": si, "para": pi, "kind": "prose"},
                        })
    return dialogue, narration


def _chunk_speech(text):
    """Window a spoken line to the engine-reliable ceiling WITHOUT dropping short lines
    (unlike narration windows, terse dialogue like 'My dear man,' is kept)."""
    if len(text) <= WINDOW_MAX_CHARS:
        return [text] if len(text) >= MIN_CLIP_CHARS else []
    out, cur = [], ""
    for s in split_sentences(text):
        if len(s) > WINDOW_MAX_CHARS:
            if cur:
                out.append(cur.strip()); cur = ""
            out.append(s.strip()); continue
        if len(cur) + len(s) + 1 <= WINDOW_MAX_CHARS:
            cur = (cur + " " + s).strip()
        else:
            out.append(cur.strip()); cur = s
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) >= MIN_CLIP_CHARS and is_complete_utterance(c)]


def _window_sentences(sentences):
    windows, cur = [], ""
    for s in sentences:
        if len(s) > WINDOW_MAX_CHARS:               # a single long sentence: take it alone
            if cur:
                windows.append(cur.strip()); cur = ""
            windows.append(s.strip()); continue
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
    """Informational router probe: does a permissive audiobook already exist?"""
    try:
        q = urllib.parse.urlencode({"title": title, "format": "json"})
        data = json.loads(fetch("https://librivox.org/api/feed/audiobooks/?" + q).decode("utf-8", "replace"))
        books = data.get("books", [])
        return [b.get("title") for b in books][:5]
    except Exception as e:
        return f"(librivox probe failed: {e})"


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
        engine = tag.get("engine", "vibevoice")
        if engine not in ("vibevoice", "qwen", "moss_vg", "dia"):
            engine = "vibevoice"
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

# Every engine build_direction knows how to assemble a payload for. dia and moss85
# take no casting pass, so they are here but not in CASTING_SCHEMA. Anything absent
# is a hard error rather than a silent rewrite — see build_direction().
KNOWN_ENGINES = ("vibevoice", "dia", "qwen", "moss_vg", "moss85",
                 "chatterbox", "zonos", "orpheus", "longcat")

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
                              "order [happiness,sadness,disgust,fear,surprise,anger,other,neutral]",
                   "pitch_std": "number 20-150",
                   "speaking_rate": "number 5-30"},
    "orpheus":    {"voice": "EXACTLY one of: " + "|".join(ORPHEUS_VOICES),
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
    "zonos":      {"voice_design": _STR,
                   "emotion": {"type": "array", "items": _NUM,
                               "minItems": 8, "maxItems": 8},
                   "pitch_std": _NUM, "speaking_rate": _NUM},
    "orpheus":    {"voice": {"type": "string", "enum": list(ORPHEUS_VOICES)},
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
        if d.get("voice") not in ORPHEUS_VOICES:
            return False
        bad = set(re.findall(r"<[a-z]+>", d.get("render_text", ""))) - set(ORPHEUS_TAGS)
        return not bad
    if engine == "zonos":
        e = d.get("emotion")
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
    the manifest record what was actually conditioned on. A missing or malformed
    vector falls back to neutral-dominant rather than to an arbitrary draw.
    """
    if not isinstance(vec, list) or len(vec) != 8:
        return [0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.77]
    vals = [max(float(x), 0.0) if isinstance(x, (int, float)) else 0.0 for x in vec]
    total = sum(vals)
    if total <= 0:
        return [0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.77]
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


def build_direction(tag, text, dia_guidance=3.0):
    """Assemble the per-engine `direction` payload.

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
        return engine, {"instruct": _merge(vd, ins)}
    if engine == "chatterbox":
        # Two numbers and a casting call. `exaggeration` is a RATE PROFILE, not an
        # emotion selector, and raising it alone reads rushed — cfg_weight is the
        # other half of one control, so it is defaulted rather than left absent.
        return engine, {"design": vd,
                        "exaggeration": _clamp(tag.get("exaggeration"), 0.25, 1.0, 0.5),
                        "cfg_weight": _clamp(tag.get("cfg_weight"), 0.2, 0.6, 0.5)}
    if engine == "zonos":
        # The emotion vector is silently L1-normalised downstream, so we normalise
        # here too: what the model receives is then exactly what the manifest
        # records, and an audit reading "happiness 0.85" can never again mean 0.616.
        return engine, {"design": vd,
                        "emotion": _l1(tag.get("emotion")),
                        "pitch_std": _clamp(tag.get("pitch_std"), 20, 150, 45.0),
                        "speaking_rate": _clamp(tag.get("speaking_rate"), 5, 30, 15.0)}
    if engine == "orpheus":
        # The only string that reaches the model is f"{voice}: {text}", assembled in
        # the renderer. An unlisted voice would be spoken aloud, so fall back to the
        # maintainers' default rather than trusting the emission.
        voice = tag.get("voice") if tag.get("voice") in ORPHEUS_VOICES else "tara"
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
    return engine, {"render_text": f"[S1] {text} [S1]",
                    "temperature": 1.8, "guidance": dia_guidance}


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
    ap.add_argument("--url", required=True)
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
    else:
        print("  router: no LibriVox match → SYNTHESIZE lane (correct).", flush=True)

    print("== fetch + parse ==", flush=True)
    epub_url = se_epub_url(args.url) if "standardebooks.org" in args.url else args.url
    print("  epub:", epub_url, flush=True)
    sections = parse_epub(fetch(epub_url))
    n_drama = sum(1 for _, k, _ in sections if k == "drama")
    total_items = sum(len(items) for _, _, items in sections)
    print(f"  sections: {len(sections)} ({n_drama} drama / {len(sections) - n_drama} prose)  units: {total_items}", flush=True)

    dialogue, narration = build_chunks(sections)
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

    print("== director-pass (Gemma 4 26B via ollama) ==", flush=True)
    lines, failures = [], 0
    for i, chunk in enumerate(sample):
        tag = director_tag(chunk)
        if not tag:
            failures += 1
            print(f"  [{i}] FAILED to parse director JSON ({chunk['chunk_type']})", flush=True)
            continue
        line = to_bank_line(i, chunk, tag, slug)
        lines.append(line)
        print(f"  [{i}] {chunk['chunk_type']:9} eng={line['engine']:6} "
              f"V={line['intended']['V']:+.1f} A={line['intended']['A']:+.1f} T={line['intended']['T']:+.1f} "
              f"{line['register']:22} | {chunk['text'][:55]}", flush=True)

    bank = {
        "version": f"book-{slug}-1",
        "campaign": f"book-{slug}",
        "license_note": f"Text: Standard Ebooks CC0 ({args.title or slug}, {args.author}). "
                        "Synthetic audio from Apache/MIT engines. Director: Gemma 4 (Apache-2.0).",
        "text_provenance": f"book:{slug} (Standard Ebooks CC0)",
        "source": {"url": args.url, "epub": epub_url, "title": args.title, "author": args.author,
                   "text_license": "CC0", "router_librivox": lv},
        "lines": lines,
    }
    out = os.path.join(args.out, f"{slug}_bank.json")
    with open(out, "w") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    print(f"== DONE ==  wrote {len(lines)} bank lines ({failures} failures) -> {out}", flush=True)


if __name__ == "__main__":
    main()

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


def extract_dialogue(paragraph):
    """If the paragraph contains quoted speech, return (quote, attribution_phrase) else None.
    Standard Ebooks uses curly quotes U+201C/U+201D."""
    m = re.search(r"“(.+?)”", paragraph)
    if not m:
        return None
    quote = m.group(1).strip()
    if len(quote) < 12:
        return None
    # attribution = the non-quoted remainder, if it names a speech verb
    remainder = (paragraph[: m.start()] + " " + paragraph[m.end():]).strip()
    attr = ""
    am = re.search(r"([A-Z][\w' ]{0,40}?\b(?:" + ATTRIB_VERBS + r")\b[\w' ]{0,30})", remainder)
    if am:
        attr = am.group(1).strip()
    elif re.search(r"\b(?:" + ATTRIB_VERBS + r")\b", remainder):
        attr = re.search(r".{0,25}\b(?:" + ATTRIB_VERBS + r")\b.{0,20}", remainder).group(0).strip()
    return quote, attr


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
                dlg = extract_dialogue(para)
                if dlg and dlg[1]:
                    quote, attr = dlg
                    dialogue.append({
                        "chunk_type": "dialogue", "text": quote,
                        "source_ref": {"section": si, "para": pi, "kind": "prose", "attribution": attr},
                    })
                elif not dlg:
                    for sent_group in _window_sentences(split_sentences(para)):
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
    return [c for c in out if len(c) >= MIN_CLIP_CHARS]


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
            cast = casting_pass(chunk["text"], engine)
            if cast is None:
                continue
            tag.update(cast)
        return tag
    return None


SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "director_skills")

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
CASTING_SCHEMA = {
    "qwen":      '{"instruct": string}',
    "moss_vg":   '{"instruct": string}',
    "vibevoice": '{"voice_design": string, "instruct": string}',
}


def load_skill(engine):
    with open(os.path.join(SKILL_DIR, f"{engine}.md"), encoding="utf-8") as f:
        return f.read()


def casting_pass(text, engine, retries=2):
    """Per-engine casting/delivery, governed by director_skills/<engine>.md."""
    system = (CASTING_SYSTEM
              + "\nOutput ONLY compact minified JSON, no markdown, with EXACTLY "
              + "these keys:\n" + CASTING_SCHEMA[engine]
              + "\n\n===== SKILL FILE: " + engine + " =====\n" + load_skill(engine))
    user = f"Target engine: {engine}\n\nThe line to be performed:\n“{text}”"
    for _ in range(retries):
        body = json.dumps({
            "model": MODEL, "stream": False, "think": False,
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
        required = ("voice_design", "instruct") if engine == "vibevoice" else ("instruct",)
        if d and all(k in d for k in required):
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
    """
    engine = tag.get("engine", "vibevoice")
    if engine not in ("vibevoice", "dia", "qwen", "moss85"):
        engine = "vibevoice"
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

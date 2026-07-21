"""LibriVox quote mining, stage A: quote + attribution extraction (text side).

First stage of Notes/Sonora/librivox-quote-mining-plan.md: over a REAL-AUDIO
book's SE text, find quoted dialogue spans and their attribution clauses —
preceding ("then she muttered, “…”") and following/inverted
("…,” mused Jack" / "…,” Sarah said") — and tag each with
the attribution verb's prior label classes. Text-only, no audio, no GPU, no
Gemma (the frozen annotator joins later only for ambiguous cast mapping).

Output: <out>/quote_candidates.jsonl (one row per quote span: chapter,
paragraph index, quote text, clause position/verb/speaker token, verb
classes, estimated seconds) plus a stdout sizing report by verb class —
the stage-B go/no-go input.

Usage:
    uv run --with ebooklib --with beautifulsoup4 --with lxml \
        python scripts/mine_quote_candidates.py \
        --url https://standardebooks.org/ebooks/robert-hugh-benson/the-necromancers \
        --out /data/model-training/datasets/book-prose/necromancers
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthesis"))

CHARS_PER_SEC = 14.0  # mirrors book_ingest / synth_dia length model

# Attribution-verb lexicon -> prior label classes (librivox-quote-mining-plan.md;
# grows app-governed like the SCM register lexicon). A verb may carry several
# classes; "neutral" anchors the abundant baseline.
VERB_CLASSES = {
    "shouted": ["arousal_up"], "cried": ["arousal_up"], "exclaimed": ["arousal_up"],
    "yelled": ["arousal_up"], "bellowed": ["arousal_up"], "called": ["arousal_up"],
    "screamed": ["arousal_up", "tension_up"],
    "murmured": ["arousal_down"], "muttered": ["arousal_down"],
    "whispered": ["arousal_down", "lax_breathy"], "breathed": ["arousal_down", "lax_breathy"],
    "sighed": ["arousal_down", "lax_breathy", "valence_neg"],
    "snapped": ["tension_up"], "hissed": ["tension_up"], "growled": ["tension_up"],
    "snarled": ["tension_up"], "demanded": ["tension_up"], "barked": ["tension_up"],
    "gasped": ["tension_up", "arousal_up"], "stammered": ["tension_up"],
    "sobbed": ["valence_neg"], "groaned": ["valence_neg"], "moaned": ["valence_neg"],
    "wailed": ["valence_neg"], "grumbled": ["valence_neg"], "complained": ["valence_neg"],
    "pleaded": ["valence_neg", "tension_up"],
    "laughed": ["valence_pos"], "chuckled": ["valence_pos"], "beamed": ["valence_pos"],
    "cheered": ["valence_pos", "arousal_up"],
    "said": ["neutral"], "replied": ["neutral"], "answered": ["neutral"],
    "asked": ["neutral"], "continued": ["neutral"], "repeated": ["neutral"],
    "added": ["neutral"], "returned": ["neutral"], "observed": ["neutral"],
    "remarked": ["neutral"], "began": ["neutral"], "went_on": ["neutral"],
    "insisted": ["tension_up"], "protested": ["tension_up"], "urged": ["tension_up"],
    "warned": ["tension_up"], "mused": ["arousal_down"], "reflected": ["arousal_down"],
}
VERB_RE = "|".join(sorted((v.replace("_", " ") for v in VERB_CLASSES), key=len, reverse=True))

# Speaker token: a capitalized name (1-2 words, honorifics ok), a pronoun, or a
# determiner phrase ("the girl", "his mother", "a voice").
NAME = r"(?:(?:Mr|Mrs|Miss|Dr|Lady|Lord|Sir|Father)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
PRON = r"(?:he|she|they|I|we|it)"
DETP = r"(?:(?:the|a|an|his|her|their|my|that)\s+[a-z]+(?:\s+[a-z]+)?)"
SPEAKER = f"(?:{NAME}|{PRON}|{DETP})"
ADV = r"(?:\s+\w+ly)?"  # "said quietly," / "murmured gently the other"

# Attribution detection is OPEN-CLASS (owner rule 2026-07-21): the syntactic
# slot is the signal — in "…," <verb> <Speaker> almost any verb is an
# utterance/expression verb. VERB_RE (the lexicon) is tried first for known
# classes; ANYVERB catches the rest ("scolded", "whimpered", "retorted",
# "ejaculated"...), recorded with classes=None so the lexicon grows from
# real text instead of guesswork. Speaker-first shapes keep the closed
# lexicon ("Jack laughed" ambiguity: was it speech or action?).
ANYVERB = r"[a-z]+(?:ed|said|says|cried|told|began|sang|sung|spoke|went on)"
# following/inverted clause, right after the closing quote:
#   ", " mused Jack   |   ", " Sarah said   |   ", " said the girl softly
POST_VERB_FIRST = re.compile(rf"^\s*({VERB_RE}){ADV}\s+({SPEAKER})", re.IGNORECASE)
POST_ANYVERB_FIRST = re.compile(rf"^\s*({ANYVERB}){ADV}\s+({SPEAKER})")
POST_NAME_FIRST = re.compile(rf"^\s*({SPEAKER})\s+({VERB_RE})\b", re.IGNORECASE)
# preceding clause, right before the opening quote:  Laurie said, " | she muttered softly: "
PRE_CLAUSE = re.compile(rf"({SPEAKER})\s+({VERB_RE}){ADV}\s*[,:.]?\s*$", re.IGNORECASE)
PRE_ANYVERB = re.compile(rf"({SPEAKER})\s+({ANYVERB}){ADV}\s*[,:.]?\s*$")

QUOTE = re.compile("“(.*?)”", re.DOTALL)  # SE curly double quotes


def classify(verb):
    return VERB_CLASSES.get(verb.lower().replace(" ", "_"), None)


def mine_paragraph(text):
    """Yield candidate dicts for each quoted span in one paragraph."""
    for m in QUOTE.finditer(text):
        quote = m.group(1).strip()
        if not quote:
            continue
        before, after = text[: m.start()], text[m.end():]
        pos = verb = speaker = None
        pm = POST_VERB_FIRST.match(after) or POST_NAME_FIRST.match(after)
        if pm:
            a, b = pm.group(1), pm.group(2)
            verb, speaker = (a, b) if classify(a) else (b, a)
            pos = "post"
        elif (om := POST_ANYVERB_FIRST.match(after)):
            verb, speaker = om.group(1), om.group(2)
            pos = "post"
        else:
            bm = PRE_CLAUSE.search(before[-80:]) or PRE_ANYVERB.search(before[-80:])
            if bm:
                speaker, verb = bm.group(1), bm.group(2)
                pos = "pre"
        classes = classify(verb) if verb else None
        yield {
            "quote": quote,
            "chars": len(quote),
            "est_seconds": round(len(quote) / CHARS_PER_SEC, 1),
            "clause": pos,
            "verb": verb.lower() if verb else None,
            "classes": classes,
            "speaker": speaker,
            "context": text[max(0, m.start() - 60): m.end() + 60],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Standard Ebooks page URL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-secs", type=float, default=1.0)
    ap.add_argument("--max-secs", type=float, default=16.0)
    args = ap.parse_args()

    from book_ingest import fetch, se_epub_url, parse_epub

    os.makedirs(args.out, exist_ok=True)
    epub_cache = os.path.join(args.out, "book.epub")
    if os.path.exists(epub_cache):
        epub_bytes = open(epub_cache, "rb").read()
        print(f"using cached epub ({len(epub_bytes)} bytes)")
    else:
        epub_bytes = fetch(se_epub_url(args.url))
        open(epub_cache, "wb").write(epub_bytes)
        print(f"fetched epub ({len(epub_bytes)} bytes) -> {epub_cache}")

    rows, narration_paras = [], 0
    for title, kind, paras in parse_epub(epub_bytes):
        if kind != "prose":
            continue
        for i, para in enumerate(paras):
            found = list(mine_paragraph(para))
            if not found:
                narration_paras += 1
            for c in found:
                c.update({"chapter": title, "para": i})
                rows.append(c)

    out_path = os.path.join(args.out, "quote_candidates.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    in_len = [r for r in rows if args.min_secs <= r["est_seconds"] <= args.max_secs]
    attributed = [r for r in in_len if r["verb"]]
    by_class = collections.Counter(cl for r in attributed for cl in (r["classes"] or []))
    by_pos = collections.Counter(r["clause"] for r in attributed)
    by_verb = collections.Counter(r["verb"] for r in attributed)

    print(f"\nquotes: {len(rows)} total | {len(in_len)} within "
          f"{args.min_secs}-{args.max_secs}s | {len(attributed)} attributed "
          f"({100 * len(attributed) / max(len(in_len), 1):.0f}%) | "
          f"narration-only paragraphs: {narration_paras}")
    print(f"clause position: {dict(by_pos)}")
    unknown = collections.Counter(r["verb"] for r in attributed if r["classes"] is None)
    if unknown:
        print(f"\nopen-class verbs caught outside the lexicon "
              f"({sum(unknown.values())} quotes) — lexicon candidates:")
        for v, n in unknown.most_common(20):
            print(f"  {v:14s} {n}")
    print("\nby class (attributed, in-length):")
    for cl, n in by_class.most_common():
        print(f"  {cl:14s} {n}")
    print("\ntop verbs:")
    for v, n in by_verb.most_common(15):
        print(f"  {v:12s} {n}")
    print(f"\n{len(rows)} candidates -> {out_path}")


if __name__ == "__main__":
    main()

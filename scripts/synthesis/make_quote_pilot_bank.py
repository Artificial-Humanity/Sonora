"""Build the quote-pilot synth bank: mined quotes -> Gemma director -> bank JSON.

The pilot of the synth half of the v1.1 rescope (librivox-quote-mining-plan.md
§companion lane; owner go 2026-07-21): take stage-A quote candidates, give the
live Gemma 26B director each quote WITH ITS SCENE CONTEXT (the preceding two
sentences and the following sentence — owner spec), and emit a script bank in
the exact shape synth_{dia,qwen,moss85}.py consume. The director's V/A/T is
the training label by construction; the bank records the attribution
verb/speaker and context as provenance.

Selection (default --n 20, 3-16 s): every non-neutral verb-attributed quote
first, then the most content-expressive of the rest (exclamatory/questioning),
topped up with plain "said" neutrals as controls.

Usage:
    uv run --with ebooklib --with beautifulsoup4 --with lxml --with pysbd \
        python scripts/synthesis/make_quote_pilot_bank.py \
        --candidates /data/model-training/datasets/book-prose/necromancers/quote_candidates.jsonl \
        --epub /data/model-training/datasets/book-prose/necromancers/book.epub \
        --out /data/model-training/datasets/book-prose/necromancers/quote_pilot_bank.json
"""

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHARS_PER_SEC = 14.0


def sentences(text):
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in seg.segment(text) if s.strip()]


def build_context(paras, para_idx, quote):
    """Preceding 2 sentences + following 1 sentence around the quote,
    crossing paragraph boundaries when the quote sits at an edge."""
    para = paras[para_idx]
    at = para.find("“" + quote[:40])
    pre_text = para[:at] if at >= 0 else ""
    post_text = para[at + len(quote):] if at >= 0 else ""
    pre_sents = sentences(pre_text)
    i = para_idx - 1
    while len(pre_sents) < 2 and i >= 0:
        pre_sents = sentences(paras[i]) + pre_sents
        i -= 1
    post_sents = sentences(post_text)
    j = para_idx + 1
    while len(post_sents) < 1 and j < len(paras):
        post_sents = post_sents + sentences(paras[j])
        j += 1
    return " ".join(pre_sents[-2:]), (post_sents[0] if post_sents else "")


# Recording-style guardrails (owner audit of quote-pilot-v1, 2026-07-21: renders
# arrived with hallucinated production values — 1960s-broadcast voicing, sermon-
# in-a-hall reverb — that no direction asked for; the teachers fill the vacuum
# with their training data's room tone unless told not to).
DRY_CLAUSE = (" Recorded close-mic in a dry modern studio: no room reverb, no "
              "echo, no radio/broadcast processing, no background ambience.")
ENV_WORDS = re.compile(
    r"\b(radio|broadcast|announcer|vintage|1[89]\d0s|echo|reverb|cathedral|"
    r"church|hall|auditorium|stadium|megaphone|telephone|PA system|gramophone)\b",
    re.IGNORECASE)
STYLE_RULE = ("\nRecording style is FIXED: a dry, close-mic modern studio. Never "
              "describe era, broadcast media, microphones, or room acoustics in "
              "voice_design or instruct — only the speaker and the delivery.")


def call_director(user, model, url, retries=3):
    from book_ingest import DIRECTOR_SYSTEM

    for attempt in range(retries):
        body = json.dumps({
            "model": model, "stream": False, "think": False,
            "options": {"num_predict": 400, "temperature": 0.2},
            "messages": [{"role": "system", "content": DIRECTOR_SYSTEM + STYLE_RULE},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                content = json.load(r)["message"]["content"]
            m = re.search(r"\{.*\}", content, re.DOTALL)
            d = json.loads(m.group(0))
            for k in ("valence", "arousal", "tension", "register", "engine",
                      "voice_design", "instruct"):
                if k not in d:
                    raise KeyError(k)
            if d["engine"] not in ("dia", "qwen", "moss85"):
                raise ValueError(d["engine"])
            hits = ENV_WORDS.findall(d["voice_design"] + " " + d["instruct"])
            if hits:
                print(f"    direction lint hit {hits}; retrying")
                user += ("\n\nIMPORTANT: your previous answer described recording "
                         f"conditions ({', '.join(hits)}). Describe ONLY the speaker "
                         "and delivery; the studio is always dry and modern.")
                continue
            d["voice_design"] = d["voice_design"].rstrip(".") + "." + DRY_CLAUSE
            return d
        except Exception as e:
            print(f"    director retry ({e!r})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--epub", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-secs", type=float, default=3.0)
    ap.add_argument("--max-secs", type=float, default=16.0)
    ap.add_argument("--neutral-controls", type=int, default=4)
    ap.add_argument("--campaign", default="quote-pilot-v1")
    ap.add_argument("--id-prefix", default="qp")
    ap.add_argument("--exclude-banks", nargs="*", default=[],
                    help="prior bank JSONs whose quotes must not be re-picked")
    ap.add_argument("--ollama", default="http://localhost:11434/api/chat")
    ap.add_argument("--model", default="gemma-4-26b-a4b-qat")
    args = ap.parse_args()

    from book_ingest import parse_epub

    rows = [json.loads(l) for l in open(args.candidates, encoding="utf-8")]
    pool = [r for r in rows if args.min_secs <= r["est_seconds"] <= args.max_secs]
    used = set()
    for b in args.exclude_banks:
        for l in json.load(open(b, encoding="utf-8"))["lines"]:
            used.add(l["text"][:60])
    pool = [r for r in pool if r["quote"][:60] not in used]

    expressive = [r for r in pool if r.get("classes") and "neutral" not in r["classes"]]
    rest = [r for r in pool if r not in expressive]
    exclam = sorted((r for r in rest if "!" in r["quote"]),
                    key=lambda r: -r["quote"].count("!"))
    neutrals = [r for r in rest if "!" not in r["quote"]
                and r.get("verb") == "said"][: args.neutral_controls]
    # Controls are RESERVED slots (v1 lesson: exclamatory picks crowded them out).
    quota = args.n - min(len(neutrals), args.neutral_controls)
    picked, seen = [], set()
    for r in expressive + exclam:
        key = r["quote"][:60]
        if key in seen:
            continue
        seen.add(key)
        picked.append(r)
        if len(picked) == quota:
            break
    picked += neutrals[: args.n - len(picked)]
    print(f"picked {len(picked)}: {len([r for r in picked if r in expressive])} "
          f"verb-expressive, {len([r for r in picked if r in exclam])} exclamatory, "
          f"{len([r for r in picked if r in neutrals])} neutral controls")

    chapters = {t: paras for t, kind, paras in parse_epub(open(args.epub, "rb").read())
                if kind == "prose"}

    lines = []
    for i, r in enumerate(picked):
        pre, post = build_context(chapters.get(r["chapter"], []), r["para"], r["quote"])
        attr = (f'{r["verb"]} {r["speaker"]}' if r.get("verb") else "none in text")
        user = (
            f"Scene context (the two sentences before the line):\n{pre or '(chapter opening)'}\n\n"
            f'A character speaks (attribution: "{attr}"). Their line:\n“{r["quote"]}”\n\n'
            f"What follows the line:\n{post or '(paragraph ends)'}"
        )
        print(f"  [{i + 1}/{len(picked)}] {r['quote'][:50]!r} ({attr})")
        d = call_director(user, args.model, args.ollama)
        if d is None:
            print("    SKIPPED (director failed)")
            continue
        cls = (r.get("classes") or ["content"])[0]
        direction = {"design": d["voice_design"], "instruct": d["instruct"]}
        if d["engine"] == "dia":
            # Dia's renderer contract (synth_dia.py): [S1]-tagged render text
            # + sampling knobs live in direction (book_ingest sets the same).
            direction.update({"render_text": f"[S1] {r['quote']}",
                              "temperature": 1.8, "guidance": 4.0})
        lines.append({
            "id": f"{args.id_prefix}_{i:02d}_{cls}",
            "engine": d["engine"],
            "register": d["register"],
            "intended": {"V": round(float(d["valence"]), 2),
                         "A": round(float(d["arousal"]), 2),
                         "T": round(float(d["tension"]), 2)},
            "seed": 1234,
            "text": r["quote"],
            "direction": direction,
            "source_ref": {"book": "pg:14275", "chapter": r["chapter"],
                           "para": r["para"], "verb": r.get("verb"),
                           "speaker": r.get("speaker"), "clause": r.get("clause"),
                           "context_pre": pre, "context_post": post},
        })

    bank = {
        "version": 1,
        "campaign": args.campaign,
        "license_note": "Text: The Necromancers (Standard Ebooks, CC0/public domain). "
                        "Audio: synthetic, rendered by Apache-2.0 teacher models "
                        "(Dia / Qwen3-TTS / MOSS-TTSD); labels = director-intended VAT.",
        "lines": lines,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    import collections
    print(f"\n{len(lines)} lines -> {args.out}")
    print("engines:", dict(collections.Counter(l['engine'] for l in lines)))


if __name__ == "__main__":
    main()

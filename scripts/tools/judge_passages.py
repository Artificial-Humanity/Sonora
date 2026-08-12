# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Gemma judgement pass over extracted passages — the checks scripting CANNOT make.

DIVISION OF LABOUR, and it is deliberate. `qc_passages.py` keeps everything with a
determinate answer: completeness, length bounds, repetition, ALLCAPS, brackets. Those
are exact, free, and auditable — and a model asked "is this complete?" would be right
maybe 97% of the time, which is strictly worse than a regex that is right always, at
24,000x the cost. This file handles only what a regex provably cannot decide:

  dialect      What regional accent the SPELLING encodes, if any. Dialect spelling is
               accent NOTATION, not a defect — "shoo'll not oppen 't" is the author
               writing phonetics. Advisory: it routes casting, it does not reject.
  speakable    Mechanical hazards only (ambiguous punctuation, unexpanded abbreviations).
               Advisory. Does NOT gate — see verdict().
  unit         Is this one natural performance beat, or an arbitrary 240-character cut?
               _window_sentences cuts on a character count; a dramatic beat has real
               edges.
  speaker      Who is talking. Enables per-character casting consistency across a book,
               which we cannot do at all today.

COST ORDERING MATTERS. Script first, model second: the expensive pass never sees the
~44% the cheap checks already reject. 24,332 passages x 26b would be 13+ hours; the
gated subset we actually intend to render is a few hundred.

MODEL COMPARISON. --model accepts any ollama tag and --compare runs two models over the
same passages and reports their agreement, so "is 26b worth it over 4b" is measured
rather than assumed. The precedent was `make_director_bench.py`, which benchmarked
g2/g4/g26 on identical inputs; it was deleted 2026-08-12 as finished campaign tooling
(#26 step 2) and is in git history, not on disk.

⚠ VALIDATE BEFORE TRUSTING. This judge has NOT been checked against the owner's ear. The
lesson of the audio work is that a plausible instrument can be confidently wrong for days
— seven failed there. Run it, then have the owner audit a sample of what it REJECTED and
what it PASSED, blind, before it gates anything.

Usage:
    .venv/bin/python scripts/tools/judge_passages.py --bank clean.json --model gemma-4-26b-a4b-qat --out judged.jsonl
    .venv/bin/python scripts/tools/judge_passages.py --bank clean.json --compare gemma-4-e4b-qat,gemma-4-26b-a4b-qat
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"

SYSTEM = """You judge whether a passage extracted from a novel is usable as a
text-to-speech performance clip for a training corpus. You are NOT judging literary
quality, and you are NOT judging whether the writing is good.

Answer four questions:

1. dialect — Does the SPELLING encode a regional accent? Dialect spelling is the author
   writing phonetics: "There's nobbut t' missis; and shoo'll not oppen 't" is Yorkshire
   notation, not an error. Name the region if you can ("Yorkshire", "Scots", "Cockney",
   "US Southern"); use "none" for standard spelling. This is CASTING INFORMATION — such
   a passage needs a voice that can carry the accent, and read flat it sounds absurd.
   Never treat dialect as a reason to reject.

2. speakable — MECHANICAL hazards only: ambiguous punctuation, unexpanded abbreviations,
   bare numerals or symbols that could be voiced several ways. Archaic vocabulary and
   dialect spelling are NOT hazards — engines render both faithfully (measured).

3. unit — Is this ONE natural performance beat with a real beginning and end, rather
   than an arbitrary slice? A passage that starts cleanly and finishes its thought is a
   unit; one that stops because it ran out of characters is not.

4. speaker — If the passage is spoken by a character, give the character's name if it is
   determinable from the text alone; otherwise "narrator" for narration, or "unknown".

Reply with ONLY a JSON object, no prose, no code fence:
{"dialect": "...", "speakable": true/false, "unit": true/false, "speaker": "...",
 "reason": "<8 words max, only if unit is false>"}"""


def ask(model, text, timeout=120):
    body = json.dumps({
        "model": model, "system": SYSTEM, "prompt": f"PASSAGE:\n{text}",
        "stream": False, "format": "json",
        "options": {"temperature": 0.0, "num_predict": 160},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read())["response"]
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, f"transport: {e}"
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"unparseable: {raw[:60]}"
    if "unit" not in d:
        return None, f"missing keys: {sorted(d)[:4]}"
    return d, None


def verdict(d):
    """A passage is gated on `unit` ALONE. Everything else is casting metadata.

    There is deliberately NO `standalone` / comprehensibility check. One existed and was
    REMOVED (owner, 2026-07-29) because it was specified from the wrong frame: it asked
    whether a LISTENER following the story could make sense of the passage. These clips
    train prosody and register, not narrative — a line that opens mid-argument still
    carries a complete intonation contour, and its register label comes from the director
    pass, not from listener comprehension. It was the judge's LARGEST rejection category
    (14% on 31b, 4% on e4b), so it was discarding usable training data at scale. Do not
    reintroduce it without a reason that applies to TRAINING rather than to listening.

    `speakable` was ALSO demoted from a gate to advisory (2026-07-29) after a render test
    falsified it. It rejected Yorkshire dialect and archaic prose as unrenderable; Qwen and
    Orpheus both read those passages faithfully, verified by the owner reading along. The
    owner's diagnosis is the right one: dialect spelling is ACCENT NOTATION — the text is
    describing how it should sound, and it needs "the right coupling" to a voice that can
    carry it. Read in flat American it sounds absurd, which is a CASTING failure, not a
    text defect. So `dialect` is now captured as a routing signal instead of a rejection
    reason. Rejecting those passages threw away both the passage and the instruction.
    """
    return bool(d.get("unit"))


def load(bank):
    d = json.load(open(bank, encoding="utf-8"))
    lines = d.get("lines", d) if isinstance(d, dict) else d
    return [l for l in lines if isinstance(l, dict) and l.get("text")]


def run(model, rows, limit):
    out, errs = [], 0
    for i, l in enumerate(rows[:limit]):
        d, err = ask(model, l["text"])
        if d is None:
            errs += 1
            print(f"  [{i+1}/{min(limit,len(rows))}] ERROR {err}", file=sys.stderr)
            continue
        out.append({"id": l.get("id"), "text": l["text"], "model": model,
                    "pass": verdict(d), **d})
        if (i + 1) % 25 == 0:
            print(f"  {i+1} judged…", flush=True)
    return out, errs


def summarise(res, model):
    n = len(res)
    if not n:
        print(f"{model}: no results")
        return
    p = sum(1 for r in res if r["pass"])
    print(f"\n{model}: {n} judged, {p} pass ({p/n:.0%}), {n-p} rejected")
    for k in ("unit", "speakable"):
        bad = sum(1 for r in res if not r.get(k))
        print(f"    fails {k:11s} {bad:4d} ({bad/n:5.1%})")
    ex = [r for r in res if not r["pass"]][:4]
    for r in ex:
        print(f"      [{r.get('reason','')[:34]}] {r['text'][:64]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    # e4b, not the MoE. THIS SCRIPT IS THE INSTRUMENT that measured the difference
    # on 2026-07-29 — 100 identical calls per variant — and it kept defaulting to
    # the model its own measurement disqualified: `gemma-4-26b-a4b-qat` emitted
    # 13/100 malformed JSON (```json fences, blank keys, `speak-able`) against
    # 0/100 for both plain dense variants. e4b over 31b here because passage
    # judging is the volume job and e4b is ~3x faster at 91% pass; `-spec` is
    # speculative decoding, lossless and 1.29x on this size.
    ap.add_argument("--model", default="gemma-4-e4b-qat-spec")
    ap.add_argument("--compare", help="comma-separated models to run head-to-head")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", help="write judgements jsonl")
    args = ap.parse_args()

    rows = load(args.bank)
    print(f"{len(rows)} passages in {args.bank}; judging {min(args.limit, len(rows))}")

    models = [m.strip() for m in args.compare.split(",")] if args.compare else [args.model]
    allres = {}
    for m in models:
        print(f"\n== {m} ==", flush=True)
        res, errs = run(m, rows, args.limit)
        if errs:
            print(f"  ({errs} errors)")
        allres[m] = res
        summarise(res, m)

    if len(models) == 2:
        a, b = (allres[m] for m in models)
        by = {r["id"]: r for r in b if r.get("id")}
        both = [(x, by[x["id"]]) for x in a if x.get("id") in by]
        if both:
            agree = sum(1 for x, y in both if x["pass"] == y["pass"])
            print(f"\n== agreement over {len(both)} shared passages: {agree/len(both):.0%} ==")
            # Where they differ is the only place the bigger model can be earning its cost.
            diff = [(x, y) for x, y in both if x["pass"] != y["pass"]]
            print(f"   {len(diff)} disagreements — these are what an ear must adjudicate:")
            for x, y in diff[:6]:
                print(f"     {models[0]}={'PASS' if x['pass'] else 'rej '} "
                      f"{models[1]}={'PASS' if y['pass'] else 'rej '}  {x['text'][:56]!r}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for m, res in allres.items():
                for r in res:
                    f.write(json.dumps(r) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

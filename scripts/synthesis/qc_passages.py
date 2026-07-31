# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Passage QC — validate extracted book text BEFORE any GPU time is spent on it.

WHY THIS EXISTS. `is_complete_utterance` gates book_ingest at extraction, but nothing
re-checks a bank afterwards, so anything built before the gate landed (2026-07-28) or
hand-edited since ships unexamined. Measured consequence: 324 of 1040 lines in the
existing book banks are incomplete (31%), and of the 158 audited, 21 fragments were
KEPT — 10 at score 5 — because under the v4 vocabulary the score means vocals and prosody
only. The rating instrument structurally cannot see this defect. The gate has to.

Checks are evidence-driven: every one below was chosen after surveying all 23,258 gated
passages the 13 on-disk books actually yield, not guessed at. What that survey found:

  19.7%  over 240 chars (the engine-reliable ceiling), longest 3,964 chars ~ 283 s
   8.6%  open on a conjunction — NORMAL fiction prose, deliberately NOT flagged
   7.4%  contain Mr./Mrs./St. abbreviations — TTS handles these; informational only
   1.7%  contain digits — engines differ on number normalisation; worth an eye
   1.2%  embed a speech attribution ("...," said the lady) inside the clip
   0.0%  ALLCAPS runs, verse repetition, editorial brackets — essentially absent

So the dominant real defect is LENGTH, not hygiene, and the second is completeness.

SEVERITY. `fail` rejects the passage; `warn` flags it for a human. Nothing here silently
rewrites text — a QC pass that edits its input cannot be trusted to report on it.

Usage:
    uv run qc_passages.py --bank <bank.json> [--json report.jsonl] [--write-clean out.json]
    uv run qc_passages.py --chunks <chunks.jsonl>          # book_ingest intermediate
Exit status is non-zero when any passage fails, so this can gate a pipeline step.
"""
import argparse
import json
import os
import re
import sys

# Speech-rate constant used across the lane (book_ingest.CHARS_PER_SEC).
CHARS_PER_SEC = 14.0
MIN_CHARS = 56           # ~4 s — owner floor 2026-07-25
TARGET_MAX_CHARS = 240   # ~17 s — book_ingest.WINDOW_MAX_CHARS, engine-reliable ceiling
HARD_MAX_CHARS = 400     # ~29 s — past Zonos's 30 s cap and Chatterbox's 300-char warning

_TERMINALS = ".?!…"
_CLOSERS = "”\"')]»’ "
_OPENERS = "“\"'([«‘ "


def is_complete_utterance(text: str) -> bool:
    """Whole-utterance test. Kept in sync with book_ingest.is_complete_utterance.

    Duplicated deliberately: this script must be runnable against a bank without
    importing the ingest module and its epub/pysbd dependency chain.
    """
    t = (text or "").strip()
    if not t:
        return False
    core = t.rstrip(_CLOSERS).rstrip()
    if not core or core[-1] not in _TERMINALS:
        return False
    head = t.lstrip(_OPENERS).lstrip()
    return bool(head) and (head[0].isupper() or head[0].isdigit())


def check(text: str, chunk_type: str = "") -> list[tuple[str, str, str]]:
    """Returns [(severity, code, detail), ...]. Empty means the passage is clean."""
    out = []
    t = (text or "").strip()
    n = len(t)

    if not t:
        return [("fail", "empty", "no text")]
    if not is_complete_utterance(t):
        out.append(("fail", "incomplete", f"ends {t[-12:]!r}"))
    if n < MIN_CHARS:
        out.append(("fail", "too-short", f"{n} chars ~ {n / CHARS_PER_SEC:.1f}s < 4s"))
    if n > HARD_MAX_CHARS:
        out.append(("fail", "too-long", f"{n} chars ~ {n / CHARS_PER_SEC:.0f}s — unrenderable"))
    elif n > TARGET_MAX_CHARS:
        out.append(("warn", "long", f"{n} chars ~ {n / CHARS_PER_SEC:.0f}s > 17s"))

    # Degenerate repetition: the 'Mike Wryke Wryke Mike Wryke' case, and song refrains.
    # A 3+ word phrase repeated back-to-back is never natural prose.
    if re.search(r"(\b\w+\b(?:\s+\b\w+\b){2,})\s+\1", t, re.I):
        out.append(("fail", "repetition", "phrase repeats back-to-back (verse/degenerate)"))

    if re.search(r"\b[A-Z]{2,}\b(\s+\b[A-Z]{2,}\b){2,}", t):
        out.append(("fail", "allcaps", "3+ consecutive ALLCAPS words (heading/front matter)"))

    # Editorial interpolation is the author's aside, not the speaker's words.
    if "[" in t or "]" in t:
        out.append(("warn", "brackets", "editorial interpolation"))

    # A dialogue clip should be the utterance, not the utterance plus its attribution —
    # the engine will speak "said the lady" aloud in the character's voice.
    if chunk_type == "dialogue" and re.search(
            r"[”\"'],?\s+(said|asked|replied|cried|answered|murmured)\b", t, re.I):
        out.append(("warn", "attribution", "speech tag inside a dialogue clip"))

    if re.search(r"\d", t):
        out.append(("warn", "digits", "engines differ on number normalisation"))

    return out


def load(path, kind):
    if kind == "chunks":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    d = json.load(open(path, encoding="utf-8"))
    return d.get("lines", d) if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bank", help="a bank.json (uses .lines[].text)")
    g.add_argument("--chunks", help="book_ingest chunks jsonl (uses .text)")
    ap.add_argument("--json", help="write a per-passage report here")
    ap.add_argument("--write-clean", help="write a bank containing only passing passages")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    args = ap.parse_args()

    path = args.bank or args.chunks
    rows = load(path, "chunks" if args.chunks else "bank")
    if not rows:
        sys.exit(f"no passages in {path}")

    report, counts, failed, warned = [], {}, 0, 0
    for i, r in enumerate(rows):
        text = r.get("text", "") if isinstance(r, dict) else str(r)
        ctype = (r.get("chunk_type") or "") if isinstance(r, dict) else ""
        issues = check(text, ctype)
        f = [x for x in issues if x[0] == "fail"]
        w = [x for x in issues if x[0] == "warn"]
        failed += bool(f)
        warned += bool(w and not f)
        for sev, code, _ in issues:
            counts[(sev, code)] = counts.get((sev, code), 0) + 1
        report.append({"idx": i, "id": r.get("id") if isinstance(r, dict) else None,
                       "chars": len(text.strip()),
                       "fail": [c for _, c, _ in f], "warn": [c for _, c, _ in w],
                       "detail": [d for _, _, d in issues], "text": text})

    n = len(rows)
    print(f"{os.path.basename(path)}: {n} passages")
    print(f"  PASS {n - failed - warned}   WARN {warned}   FAIL {failed}"
          f"   ({failed / n:.1%} rejected)")
    if counts:
        print()
        for (sev, code), c in sorted(counts.items(), key=lambda kv: (-kv[1])):
            print(f"  {sev:4s} {code:12s} {c:6d}  ({c / n:5.1%})")
    if not args.quiet:
        worst = [r for r in report if r["fail"]][:8]
        if worst:
            print("\n  examples of rejected passages:")
            for r in worst:
                print(f"    [{','.join(r['fail'])}] {r['text'][:76]!r}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            for r in report:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {args.json}")

    if args.write_clean:
        keep = [rows[r["idx"]] for r in report if not r["fail"]]
        src = json.load(open(path, encoding="utf-8")) if args.bank else {}
        out = dict(src, lines=keep) if isinstance(src, dict) else keep
        with open(args.write_clean, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"wrote {args.write_clean} ({len(keep)}/{n} passages kept)")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

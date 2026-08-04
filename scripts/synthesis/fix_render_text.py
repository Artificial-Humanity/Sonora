"""Repair banks whose `render_text` drifted out of sync with `text`.

Companion to check_bank.py, which detects the divergence and refuses to render it.
This fixes the stored bank so the render can proceed.

`render_text` is not an independent field with its own meaning — it is `text` plus
engine markup. When they disagree on the WORDS, `text` is the authority: it is what
the director wrote, what the WER gate scores against, what the manifest publishes
and what the training corpus keeps. `render_text` is a derived copy that failed to
follow, so the repair is to re-derive it and preserve any markup it carries.

    .venv/bin/python scripts/synthesis/fix_render_text.py <bank.json>
    .venv/bin/python scripts/synthesis/fix_render_text.py <bank.json> --apply
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_bank  # noqa: E402

# Markup is positional, so it cannot be transplanted blindly onto a rewritten
# sentence. This handles the only case that occurs in practice and the only one
# it can handle safely: markup that sits at a word boundary is re-attached by
# following the word before it. Anything more tangled is reported, not guessed.
TOKEN = re.compile(r"(<[a-z]+>|\[S\d\]|\([a-z ]+\))", re.IGNORECASE)


def rederive(text, old_render):
    """`text`'s words, carrying `old_render`'s markup where it can be placed."""
    marks = []                      # (index into old word stream, markup token)
    words = 0
    for part in TOKEN.split(old_render):
        if not part.strip():
            continue
        if TOKEN.fullmatch(part):
            marks.append((words, part))
        else:
            words += len(part.split())
    if not marks:
        return text, True
    new = text.split()
    if words != len(new):
        return None, False          # word counts differ; placement is a guess
    out, cursor = [], 0
    for idx, tok in marks:
        out.extend(new[cursor:idx])
        out.append(tok)
        cursor = idx
    out.extend(new[cursor:])
    return " ".join(out), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    problems = check_bank.check(args.bank)
    if not problems:
        print("bank is already consistent — nothing to do")
        return 0
    bad = {cid for cid, _e, _d in problems}

    bank = json.load(open(args.bank, encoding="utf-8"))
    fixed, refused = 0, []
    for line in bank["lines"]:
        if line["id"] not in bad:
            continue
        d = dict(line["direction"])
        was = d["render_text"]
        new, ok = rederive(line["text"], was)
        if not ok:
            refused.append((line["id"], "markup cannot be replaced safely"))
            continue
        d["render_text"] = new
        line["direction"] = d
        line.setdefault("corrections", {})["render_text"] = {
            "was": was,
            "reason": "did not match line['text']; text is the authority"}
        fixed += 1
        print(f"  {line['id']} [{line['engine']}]")
        print(f"    was: {was[:90]!r}")
        print(f"    now: {new[:90]!r}")
    for cid, why in refused:
        print(f"  REFUSED {cid}: {why}")

    if not args.apply:
        print(f"\n{fixed} fixable, {len(refused)} refused — pass --apply")
        return 0

    p = pathlib.Path(args.bank)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(bank, fh, indent=2, ensure_ascii=False)
    shutil.copymode(str(p), tmp)
    os.replace(tmp, str(p))
    print(f"\nfixed {fixed} line(s) in {p.name}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())

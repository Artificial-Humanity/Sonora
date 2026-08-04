"""Pre-flight a bank before it reaches a GPU. Refuses to render an inconsistent one.

THE BUG THIS EXISTS FOR (2026-08-04). Two engines are fed `direction.render_text`,
not `line["text"]`: orpheus and dia. Everything else — the ASR/WER gate, the
manifest, the audition card, the training corpus — reads `line["text"]`. They are
two copies of the same sentence and nothing kept them in sync.

A reroll expanded "Madison, Wis." to "Madison, Wisconsin" in `text` and left
`render_text` alone. The model was handed the stale string, so the clip still said
"Wis." while every artifact around it claimed "Wisconsin".

QC could not catch it and never will: one wrong word in a 35-word line is WER 0.029,
far under any threshold worth having. The gate compares the audio to `text`, so a
`text`/`render_text` divergence is exactly the class of defect it is blind to — the
audio is measured against the wrong reference. Only the ear found it.

So the check has to happen BEFORE the render, on the bank itself, which is the one
place both strings are visible at once. It is wired into synth_bank.sh ahead of
every engine, so it covers hand-authored banks and future builders too — not just
the reroll script that happened to break it.

`render_text` is ALLOWED to differ by engine markup, because that is its whole
reason to exist: orpheus takes inline <laugh>/<sigh> tags and dia takes [S1]
speaker tags and parenthetical non-verbals. Those are stripped from both sides
before comparing. What must not differ is the words.

    .venv/bin/python scripts/synthesis/check_bank.py <bank.json>
"""
import json
import re
import sys

# Stripped from BOTH sides before comparing. Stripping only one side is its own
# trap: it makes every legitimately-tagged orpheus line look divergent, and 19
# such false positives are enough to make anyone stop reading the output.
MARKUP = re.compile(
    r"<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>"      # orpheus
    r"|\[S\d\]"                                                    # dia speakers
    r"|\((?:sighs?|laughs?|chuckles?|clears throat|gasps?|coughs?|"
    r"groans?|yawns?|sniffles?)\)",                                # dia non-verbals
    re.IGNORECASE)


def canon(text):
    """Words only — markup removed, whitespace collapsed."""
    return " ".join(MARKUP.sub("", text or "").split())


def check(bank_path):
    bank = json.load(open(bank_path, encoding="utf-8"))
    problems = []
    for line in bank.get("lines", []):
        rt = (line.get("direction") or {}).get("render_text")
        if rt is None:
            continue                      # engine has no render_text channel
        a, b = canon(line.get("text", "")), canon(rt)
        if a == b:
            continue
        first = next((f"text={x!r} render_text={y!r}"
                      for x, y in zip(a.split(), b.split()) if x != y),
                     f"length differs ({len(a.split())} vs {len(b.split())} words)")
        problems.append((line["id"], line.get("engine", "?"), first))
    return problems


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    problems = check(argv[1])
    if not problems:
        return 0
    print(f"!! {len(problems)} line(s) where the words sent to the model differ "
          f"from the words everything else records:", file=sys.stderr)
    for cid, engine, first in problems:
        print(f"     {cid} [{engine}]  {first}", file=sys.stderr)
    print("   The render would not say what the manifest, the WER gate and the "
          "corpus all claim it says.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

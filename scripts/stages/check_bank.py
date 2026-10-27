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

A line may also pin its reference by FILE (`ref_file`, Step 1c-i; see
`ref_select.pinned_reference_file`). Each one is checked here as well: its fields, an
absolute path under `ref_select.REF_FILE_ROOT`, a file that exists, and a sha256 equal to
the file's, so a wrong anchor stops the bank before a GPU is booked rather than at clip 1.
A line pinning both `ref_id` and `ref_file` is refused.

    .venv/bin/python scripts/stages/check_bank.py <bank.json>
"""
import json
import re
import sys

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import ref_select  # noqa: E402

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


def check_refs(bank_path):
    """(id, engine, problem) for every `ref_file` that would not render as written.

    The key's presence is what makes a line a file pin, as in the renderers: a falsy value
    (`{}`, null) is reported, never read as "no pin" and silently cast."""
    bank = json.load(open(bank_path, encoding="utf-8"))
    problems, hashed = [], {}          # a 300-line bank pins one file 300 times
    for line in bank.get("lines", []):
        if "ref_file" not in line:
            continue
        spec = line["ref_file"]
        found = ref_select.ref_file_problems(spec)
        if line.get("ref_id"):
            found.append(f"pins both ref_id={line['ref_id']!r} and ref_file")
        if not found and not _os.path.isfile(spec["path"]):
            found.append(f"ref_file {spec['path']} does not exist")
        elif not found:
            if spec["path"] not in hashed:
                try:
                    hashed[spec["path"]] = ref_select.file_sha256(spec["path"])
                except OSError as exc:
                    hashed[spec["path"]] = exc
            got = hashed[spec["path"]]
            if isinstance(got, OSError):
                found.append(f"ref_file {spec['path']} cannot be read: {got}")
            elif got != spec["sha256"]:
                found.append(f"ref_file {spec['path']} does not have sha256 {spec['sha256']}")
        problems += [(line.get("id", "?"), line.get("engine", "?"), p) for p in found]
    return problems


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    problems, refs = check(argv[1]), check_refs(argv[1])
    if refs:
        print(f"!! {len(refs)} reference problem(s) in pinned `ref_file`s:", file=sys.stderr)
        for cid, engine, problem in refs:
            print(f"     {cid} [{engine}]  {problem}", file=sys.stderr)
    if not problems:
        return 1 if refs else 0
    print(f"!! {len(problems)} line(s) where the words sent to the model differ "
          f"from the words everything else records:", file=sys.stderr)
    for cid, engine, first in problems:
        print(f"     {cid} [{engine}]  {first}", file=sys.stderr)
    print("   The render would not say what the manifest, the WER gate and the "
          "corpus all claim it says.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

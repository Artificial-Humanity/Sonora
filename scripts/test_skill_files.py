#!/usr/bin/env python3
"""Gate: the director skill files must agree with the code they describe.

The skill files in scripts/synthesis/director_skills/ are Gemma's adapters — they
tell the director what each engine can be told. They are specifications, and until
now nothing executed them, so they could drift from the code silently. They did:
`vibevoice.md` asserted that the word `feminine` selects a female reference, while
`ref_select.design_gender()` matches only female|woman|maternal|girl — so a design
reading "feminine, warm timbre" silently cast a MALE voice. Caught by a doc audit on
2026-07-26, not by anything automatic.

This turns the load-bearing claims into assertions. It is the red/green half of the
"write the skill file first" discipline (owner's framing: TDD for the director
interface) — the half a markdown file cannot provide on its own.

Run:  uv run scripts/test_skill_files.py
Exit: 0 all claims hold; 1 otherwise.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.join(HERE, "synthesis")
SKILLS = os.path.join(SYNTH, "director_skills")
sys.path.insert(0, SYNTH)

failures = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def skill(name):
    with open(os.path.join(SKILLS, f"{name}.md"), encoding="utf-8") as f:
        return f.read()


# --- vibevoice.md: the literal words it tells Gemma to use must actually parse ----
def test_vibevoice_casting_words():
    from ref_select import design_gender, design_age_target, AGE_BAND_NAMES

    text = skill("vibevoice")

    # every word the file lists as female-selecting must in fact select female
    listed = re.findall(r"`(female|woman|maternal|girl|feminine)`", text)
    for w in set(listed):
        got = design_gender(f"{w}, warm timbre")
        # the file may name a word specifically to warn it does NOT work
        warned = re.search(rf"`{w}`\*\* is NOT matched|`{w}` is NOT matched", text)
        if warned:
            check(got == "M",
                  f"vibevoice.md warns `{w}` is not matched, but design_gender says {got}")
        else:
            check(got == "F",
                  f"vibevoice.md lists `{w}` as female-selecting, but design_gender says {got}")

    # every age word it offers must resolve to the band it claims
    for word, claimed in re.findall(r"`(child|teen|young|middle-aged|elderly)` \((0\.\d+)\)", text):
        target = design_age_target(f"a {word} speaker")
        check(target is not None, f"vibevoice.md offers age word `{word}` but it matches no band")
        if target is not None:
            check(abs(target - float(claimed)) < 1e-9,
                  f"vibevoice.md says `{word}` -> {claimed}, code gives {target}")

    # the mature/middle-aged collision the file warns about must still be real
    check(design_age_target("mature voice") == design_age_target("middle-aged voice"),
          "vibevoice.md warns `mature` collides with `middle-aged`; that is no longer true")
    check("elderly" in AGE_BAND_NAMES.values(), "AGE_BAND_NAMES lost the elderly band")


# --- dia.md: the tag vocabulary it publishes must equal the one the code uses -----
def test_dia_tag_vocabulary():
    from make_teacher_ab_bank import DIA_TAGS

    text = skill("dia")
    in_doc = set(re.findall(r"`(\([a-z ]+\))`", text))
    in_code = set(DIA_TAGS)
    # the doc also names counter-examples ("Never invent a tag. `(whispers)`, …") —
    # every tag on such a line is a negative example, not part of the vocabulary
    for line in text.splitlines():
        if "never invent" in line.lower() or "not in the set" in line.lower():
            in_doc -= set(re.findall(r"`(\([a-z ]+\))`", line))

    missing = in_code - in_doc
    extra = in_doc - in_code
    check(not missing, f"dia.md omits tags the code accepts: {sorted(missing)}")
    check(not extra, f"dia.md lists tags absent from DIA_TAGS: {sorted(extra)}")
    check(len(in_code) == 21, f"DIA_TAGS should hold 21 tags, holds {len(in_code)}")


# --- WIP files must be unloadable ------------------------------------------------
def test_wip_guard():
    from book_ingest import load_skill

    for engine in ("qwen", "moss_vg", "vibevoice", "dia"):
        try:
            check(bool(load_skill(engine)), f"{engine}.md loaded empty")
        except Exception as e:
            check(False, f"{engine}.md failed to load: {e}")

    if os.path.exists(os.path.join(SKILLS, "sonora.md")):
        try:
            load_skill("sonora")
            check(False, "sonora.md is WIP but load_skill() accepted it")
        except RuntimeError:
            check(True, "")


# --- register lexicon must be a controlled set the director can actually copy -----
def test_register_lexicon():
    import json

    with open(os.path.join(SYNTH, "register_lexicon.json"), encoding="utf-8") as f:
        lex = json.load(f)
    labels = lex["lexicon"]
    check(len(labels) > 0, "register_lexicon.json is empty")
    check(labels == sorted(labels), "register lexicon is not sorted (unstable diffs)")
    check(len(labels) == len(set(labels)), "register lexicon contains duplicates")
    bad = [l for l in labels if not re.fullmatch(r"[a-z0-9_]+", l)]
    check(not bad, f"register labels must be snake_case: {bad[:5]}")


def main():
    for fn in (test_vibevoice_casting_words, test_dia_tag_vocabulary,
               test_wip_guard, test_register_lexicon):
        try:
            fn()
        except Exception as e:
            failures.append(f"{fn.__name__} raised: {e!r}")

    if failures:
        print(f"FAIL — {len(failures)} of {checks} claims broken:\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS — {checks} skill-file claims verified against code")
    return 0


if __name__ == "__main__":
    sys.exit(main())

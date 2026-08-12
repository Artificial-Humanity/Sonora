#!/usr/bin/env python3
"""Gate: the director skill files must agree with the code they describe.

The skill files in scripts/assets/director_skills/ are Gemma's adapters — they
tell the director what each engine can be told. They are specifications, and until
now nothing executed them, so they could drift from the code silently. They did:
`vibevoice.md` asserted that the word `feminine` selects a female reference, while
`ref_select.design_gender()` matches only female|woman|maternal|girl — so a design
reading "feminine, warm timbre" silently cast a MALE voice. Caught by a doc audit on
2026-07-26, not by anything automatic.

This turns the load-bearing claims into assertions. It is the red/green half of the
"write the skill file first" discipline (owner's framing: TDD for the director
interface) — the half a markdown file cannot provide on its own.

Run:  .venv/bin/python scripts/gates/test_skill_files.py
Exit: 0 all claims hold; 1 otherwise.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# #26 step 3: the skill files and the register lexicon are checked-in ASSETS and now
# have one home, rather than sitting beside whichever script happened to read them.
ASSETS = os.path.join(HERE, "..", "assets")
SKILLS = os.path.join(ASSETS, "director_skills")
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib", "stages"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

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


# --- the literal casting words every cloning engine's file publishes must parse ---
# vibevoice, chatterbox and zonos all route voice_design through the SAME function
# (ref_select.select_reference), so all three files make the same claims and all three
# are checked. chatterbox and zonos joined on 2026-07-28, when their casting was wired
# for the first time — until then neither engine had ever been passed a reference.
CASTING_ENGINES = ("vibevoice", "chatterbox", "zonos")


def test_casting_words():
    from ref_select import design_gender, design_age_target, AGE_BAND_NAMES

    for engine in CASTING_ENGINES:
        text = skill(engine)

        # every word the file lists as female-selecting must in fact select female
        listed = re.findall(r"`(female|woman|maternal|girl|feminine)`", text)
        check(bool(listed), f"{engine}.md names no casting gender words at all")
        for w in set(listed):
            got = design_gender(f"{w}, warm timbre")
            # the file may name a word specifically to warn it does NOT work
            warned = re.search(rf"`{w}`\*\* is NOT matched|`{w}` is NOT matched", text)
            if warned:
                check(got == "M",
                      f"{engine}.md warns `{w}` is not matched, but design_gender says {got}")
            else:
                check(got == "F",
                      f"{engine}.md lists `{w}` as female-selecting, but design_gender says {got}")

        # every age word offered must resolve to a band, and to the claimed one where
        # the file commits to a number (only vibevoice.md publishes the targets)
        offered = set(re.findall(r"`(child|teen|young|middle-aged|elderly)`", text))
        check(bool(offered), f"{engine}.md offers no age words")
        for word in offered:
            check(design_age_target(f"a {word} speaker") is not None,
                  f"{engine}.md offers age word `{word}` but it matches no band")
        for word, claimed in re.findall(
                r"`(child|teen|young|middle-aged|elderly)` \((0\.\d+)\)", text):
            target = design_age_target(f"a {word} speaker")
            if target is not None:
                check(abs(target - float(claimed)) < 1e-9,
                      f"{engine}.md says `{word}` -> {claimed}, code gives {target}")

        # the mature/middle-aged collision every one of these files warns about
        if "matronly" in text or "mature" in text:
            check(design_age_target("mature voice") == design_age_target("middle-aged voice"),
                  f"{engine}.md warns `mature` collides with `middle-aged`; no longer true")

    check("elderly" in AGE_BAND_NAMES.values(), "AGE_BAND_NAMES lost the elderly band")


# --- orpheus.md: two closed sets, neither validated by the shipped code -----------
def test_orpheus_closed_sets():
    """An unlisted Orpheus voice or tag is interpolated into the prompt and SPOKEN.

    canopylabs' own validation function is dead code, so this file and
    book_ingest.ORPHEUS_* are the only things standing between a typo and a clip of
    a narrator saying "julia". They must agree exactly.
    """
    from book_ingest import ORPHEUS_VOICES, ORPHEUS_TAGS

    text = skill("orpheus")

    # the voice enum, published as a single backticked run
    in_doc = set(re.findall(r"`(tara|leah|jess|leo|dan|mia|zac|zoe)`", text))
    check(in_doc == set(ORPHEUS_VOICES),
          f"orpheus.md voice set != ORPHEUS_VOICES: "
          f"doc-only {sorted(in_doc - set(ORPHEUS_VOICES))}, "
          f"code-only {sorted(set(ORPHEUS_VOICES) - in_doc)}")

    # the tag set, minus any line naming a counter-example
    tags_doc = set(re.findall(r"`(<[a-z]+>)`", text))
    for line in text.splitlines():
        if "never invent" in line.lower() or "not in the set" in line.lower():
            tags_doc -= set(re.findall(r"`(<[a-z]+>)`", line))
    check(tags_doc == set(ORPHEUS_TAGS),
          f"orpheus.md tag set != ORPHEUS_TAGS: "
          f"doc-only {sorted(tags_doc - set(ORPHEUS_TAGS))}, "
          f"code-only {sorted(set(ORPHEUS_TAGS) - tags_doc)}")

    # the file must keep forbidding emotions.txt — the highest-probability way to
    # poison a campaign, because it looks exactly like a controlled vocabulary
    check("emotions.txt" in text, "orpheus.md no longer warns about emotions.txt")
    # and it must keep naming the checkpoint, since -pretrained has no controls at all
    check("-ft" in text and "-pretrained" in text,
          "orpheus.md must state that control lives in -ft, not -pretrained")

    # The ban must be executable, not just written down. It was markdown-only
    # until 2026-08-02 while the code's fallback voice was literally `tara`.
    from book_ingest import ORPHEUS_BANNED, ORPHEUS_CASTABLE, ORPHEUS_FALLBACK

    check("Never cast" in text or "never cast" in text,
          "orpheus.md no longer states a never-cast voice")
    for voice in ORPHEUS_BANNED:
        check(voice not in ORPHEUS_CASTABLE,
              f"{voice} is banned but still castable")
        check(re.search(rf"`{voice}`[^\n]*[Nn]ever cast", text) is not None
              or re.search(rf"[Nn]ever cast[^\n]*`{voice}`", text) is not None,
              f"orpheus.md must say `{voice}` is never cast")
    check(ORPHEUS_FALLBACK in ORPHEUS_CASTABLE,
          f"orpheus fallback {ORPHEUS_FALLBACK} is not castable")
    check(ORPHEUS_FALLBACK not in ORPHEUS_BANNED,
          f"orpheus fallback {ORPHEUS_FALLBACK} is a banned voice")


# --- zonos.md: "omit emotion" must be reachable through the director --------------
def test_zonos_emotion_off_is_representable():
    """The skill file tells the director to emit `emotion: null` for narration.

    That instruction was unreachable until 2026-08-02: _l1() turned None into the
    neutral-dominant vector — the exact conditioning measured destabilizing 5 of 9
    zonos narration groups — and the JSON schema required an 8-float array, so the
    director could not emit null even when told to. The renderer implemented the
    fix correctly the whole time; nothing could reach it.
    """
    from book_ingest import CASTING_JSON_SCHEMA, _l1, build_direction

    text = skill("zonos")
    check("null" in text or "omit" in text,
          "zonos.md no longer documents turning emotion off")

    # None must survive normalisation as None — not become a vector.
    check(_l1(None) is None, "_l1(None) must stay None (emotion off)")
    check(_l1("nonsense") is None, "_l1 of a malformed vector must be None")
    check(_l1([1, 1, 0, 0, 0, 0, 0, 0]) == [0.5, 0.5, 0, 0, 0, 0, 0, 0],
          "_l1 must still L1-normalise a real vector")

    # and the schema must permit it
    emo = CASTING_JSON_SCHEMA["zonos"]["emotion"]["type"]
    check("null" in emo if isinstance(emo, list) else False,
          f"zonos emotion schema must allow null, got {emo!r}")

    # end to end: a director emission with emotion null reaches the renderer as None
    _eng, d = build_direction({"engine": "zonos", "voice_design": "a calm narrator",
                               "emotion": None, "pitch_std": 45.0,
                               "speaking_rate": 15.0}, "some text")
    check(d["emotion"] is None,
          f"build_direction must pass emotion=None through, got {d['emotion']!r}")


# --- the pause gate's two thresholds live in two files and must agree -------------
def test_pause_thresholds_agree():
    """qc_gate measures the pause; register_audition writes the advisory note.

    The advisory band is the range where the clip is queued for the ear rather
    than failed outright, so if the two constants drift the auditor either gets
    no note on a flagged clip or a note on a clip nothing flagged. Calibrated
    2026-08-02 against 369 audited clips; the numbers are load-bearing, not
    decorative.
    """
    import qc_gate
    import register_audition

    check(qc_gate.PAUSE_FLAG_MAX == register_audition.PAUSE_FLAG_SECONDS,
          f"pause advisory threshold drifted: qc_gate {qc_gate.PAUSE_FLAG_MAX} "
          f"vs register_audition {register_audition.PAUSE_FLAG_SECONDS}")
    check(qc_gate.PAUSE_FLAG_MAX < qc_gate.PAUSE_HARD_MAX,
          "the advisory band must sit BELOW the hard cap, else it can never fire")
    check(qc_gate.PAUSE_MIN_GAP < qc_gate.PAUSE_FLAG_MAX,
          "silences are measured at a finer grain than the flag threshold")


# --- the relay matrix must exist in BOTH places it is meant to live ---------------
def test_relay_matrix_agrees():
    """Step 2 of the onboarding pattern: encode the relay matrix twice so it cannot
    rot — once in build_direction (what renderers receive) and once in the audition
    app's RELAY (what the audit card strikes through). An engine present in one and
    absent from the other is how a model gets scored for direction it never got.
    """
    from book_ingest import KNOWN_ENGINES

    # ⚠ This pointed at `../../../AI-Lab-AMD/audition/app/main.py` until 2026-08-12 — where
    # the app lived BEFORE it came home in `2399aed`. The sibling path stopped existing that
    # day, so `os.path.exists` was False and the check `return`ed on every run: a gate whose
    # whole purpose is catching an engine that gets scored for direction it never received
    # has been reporting success while comparing nothing. The escape hatch was right when
    # the app was a separate repo and became a permanent skip when it stopped being one.
    # No fallback now, deliberately: the app is in this tree, so its absence is a failure.
    app = os.path.join(HERE, "..", "..", "audition", "app", "main.py")
    # `check()` RECORDS and returns — it does not short-circuit — so without this the
    # open below raises, `main()`'s `except Exception` logs the same fact a second time,
    # and the two checks after it never run: the claim count would vary with the failure.
    if not os.path.exists(app):
        check(False, f"the audition app is not where this gate expects it: {app}")
        return
    with open(app, encoding="utf-8") as f:
        src = f.read()
    block = re.search(r"^RELAY = \{(.*?)^\}", src, re.S | re.M)
    check(block is not None, "could not find the RELAY block in the audition app")
    if block:
        in_relay = set(re.findall(r'"([a-z0-9_]+)":\s*\{', block.group(1)))
        missing = set(KNOWN_ENGINES) - in_relay
        check(not missing,
              f"engines in build_direction but absent from audition RELAY: {sorted(missing)}")


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

    for engine in ("qwen", "moss_vg", "vibevoice", "dia",
                   "chatterbox", "zonos", "orpheus", "longcat"):
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

    with open(os.path.join(ASSETS, "register_lexicon.json"), encoding="utf-8") as f:
        lex = json.load(f)
    labels = lex["lexicon"]
    check(len(labels) > 0, "register_lexicon.json is empty")
    check(labels == sorted(labels), "register lexicon is not sorted (unstable diffs)")
    check(len(labels) == len(set(labels)), "register lexicon contains duplicates")
    bad = [l for l in labels if not re.fullmatch(r"[a-z0-9_]+", l)]
    check(not bad, f"register labels must be snake_case: {bad[:5]}")


# --- the director's emission must be rejected where the engine would mis-render ---
def test_casting_schema_shapes():
    """Each of these is a failure an engine actually exhibits, not a style rule.

    They are asserted here because the casting pass retries on a rejection, so a
    silently-loosened validator would not fail loudly — it would just start letting
    poisoned emissions through, which is precisely how the last four verdicts died.
    """
    from book_ingest import (CASTING_SCHEMA, KNOWN_ENGINES, ORPHEUS_BANNED,
                             ORPHEUS_FALLBACK, build_direction,
                             _validate_casting)

    for engine in CASTING_SCHEMA:
        check(engine in KNOWN_ENGINES,
              f"{engine} has a casting schema but build_direction does not know it")

    # an invented Orpheus voice is SPOKEN — reject, and scrub it if it survives
    check(not _validate_casting("orpheus", {"voice": "julia", "render_text": "hello there"}),
          "an unlisted Orpheus voice must be rejected — it gets spoken aloud")
    _, d = build_direction({"engine": "orpheus", "voice": "julia",
                            "render_text": "hi <whisper> there"}, "The line.")
    # The fallback must be the measured-reliable voice. This used to assert
    # "tara" — encoding the very defect orpheus.md bans, so the test agreed
    # with the bug instead of catching it.
    check(d["voice"] == ORPHEUS_FALLBACK and "<whisper>" not in d["render_text"],
          "build_direction must scrub an invented Orpheus voice/tag, not pass it through")
    check(d["voice"] not in ORPHEUS_BANNED,
          "build_direction must never fall back to a banned voice")

    # Zonos: 8 floats or nothing; a short vector silently reshapes the conditioning
    check(not _validate_casting("zonos", {"emotion": [1, 2, 3], "pitch_std": 50,
                                          "speaking_rate": 15}),
          "a Zonos emotion vector that is not exactly 8 floats must be rejected")
    _, d = build_direction({"engine": "zonos", "emotion": [4, 0, 0, 0, 0, 0, 0, 4],
                            "pitch_std": 90, "speaking_rate": 15}, "The line.")
    check(abs(sum(d["emotion"]) - 1.0) < 1e-3,
          "the emotion vector must be L1-normalised before it is recorded")

    # Chatterbox: exaggeration without cfg_weight reads rushed — both, always
    check(not _validate_casting("chatterbox", {"exaggeration": 0.8}),
          "Chatterbox exaggeration emitted without cfg_weight must be rejected")

    # an unknown engine must be fatal, never silently rewritten to vibevoice
    try:
        build_direction({"engine": "definitely-not-an-engine"}, "The line.")
        check(False, "build_direction accepted an unknown engine instead of raising")
    except ValueError:
        check(True, "")


def main():
    for fn in (test_casting_words, test_dia_tag_vocabulary,
               test_orpheus_closed_sets, test_zonos_emotion_off_is_representable,
               test_pause_thresholds_agree, test_relay_matrix_agrees,
               test_wip_guard, test_register_lexicon, test_casting_schema_shapes):
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

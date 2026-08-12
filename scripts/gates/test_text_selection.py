#!/usr/bin/env python3
"""Gate: no incomplete utterance may reach a bank.

WHY (owner, 2026-07-28). A clip that is merely SHORT stays judgeable — the owner
has knowingly let short-but-whole clips through, because the sentence is still
there. A clip that is INCOMPLETE destroys the judgement itself: prosody lives in
the arc of a finished utterance, so half a sentence ending on a comma has no
terminal contour to assess, and no amount of vocal quality rescues it.

The rating vocabulary cannot catch this. Under v4 the score means vocals and
prosody ONLY, so a fragment spoken in a fine voice was rated 4-5 and is
statistically invisible in ratings.csv — fragments there score 4.33 mean against
4.32 for complete lines. The instrument is blind to the defect by construction,
which is exactly why it needs a gate instead of an audit.

Every case below is a REAL text that reached the audit surface. 79 of them did,
across two independent lanes (book prose and quote-pilot mining).

Run:  .venv/bin/python scripts/gates/test_text_selection.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from book_ingest import is_complete_utterance, extract_utterances  # noqa: E402

failures = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


# --- real fragments that were rated 4-5 and should never have been offered -----
FRAGMENTS = [
    "You do not seem to understand me,",              # split quote, first half
    "Dear Laurie is so impetuous and sensitive,",      # split quote, first half
    "It would be a comfort to get at the garden again,",
    "When I first heard of the poor girl’s death,",
    "The ferry goes at six,",
    "se it was true—suppose really and truly Amy was here, and—",  # cut mid-word
    "n’t know what to do, I’m so sorry. But don’t spoil it all,",  # cut mid-word
    "only wondered whether there was any particular precaution—",  # starts mid-clause
    "the railroad magnate",                            # scare-quote phrase
    "especially cool,",                                # scare-quote phrase
    "Say one prayer—",                                 # trails off on a dash
]

# --- texts that MUST survive: short is fine, incomplete is not ----------------
COMPLETE = [
    "Yes.",                                            # short but whole
    "Get down. Stay quiet.",
    "You will want to think very carefully about the next thing you say to me.",
    "“Everything is quite in hand, I assure you.”",     # wrapped in quotes
    "Well?",
    "Have you decided?",
    "And the third door — oh, the third door! — was the one nobody sensible opened.",
]


def test_fragments_rejected():
    for t in FRAGMENTS:
        check(not is_complete_utterance(t),
              f"fragment accepted as complete: {t[:60]!r}")


def test_complete_survive():
    for t in COMPLETE:
        check(is_complete_utterance(t),
              f"complete utterance rejected: {t[:60]!r}")


def test_no_length_floor():
    """The gate is completeness, NOT length.

    audit-markup-v0 read its ≤6-token failures as a length problem and prescribed
    a token-count floor. Length was the correlate; a floor would reject good
    short-but-whole clips and pass every long fragment. Both halves asserted here.
    """
    check(is_complete_utterance("Yes."), "a 4-character whole sentence must pass")
    check(not is_complete_utterance(
        "I had meant to say something about the arrangements for the evening, and about "
        "the way the whole business had been handled from the very beginning, but"),
        "a long fragment must fail — length is not the gate")


# --- the producer: split quotations rejoin, scare quotes never become speech ---
def test_extract_utterances():
    cases = [
        # split quotation -> ONE rejoined utterance, not its first half
        ("“Come here,” she said, “and sit down before you fall over.”",
         ["Come here, and sit down before you fall over."]),
        ("“I told you before,” he muttered, “that the bridge would not hold.”",
         ["I told you before, that the bridge would not hold."]),
        # attribution that CLOSES its sentence is a boundary, not a split
        ("“No,” she said. “That is not what the report says, and it never was.”",
         ["That is not what the report says, and it never was."]),
        # every utterance in the paragraph, not just the first
        ("“Well?” he asked. “Have you decided?” She said nothing at all.",
         ["Well?", "Have you decided?"]),
        # scare-quotes and quoted phrases are not dialogue
        ("They called him “the railroad magnate” for years afterwards.", []),
        ("He was “especially cool,” the papers said, under fire.", []),
    ]
    for para, expected in cases:
        got = [t for t, _ in extract_utterances(para)]
        check(got == expected,
              f"extract_utterances({para[:40]!r}...) -> {got}, expected {expected}")


def main():
    for fn in (test_fragments_rejected, test_complete_survive,
               test_no_length_floor, test_extract_utterances):
        try:
            fn()
        except Exception as e:
            failures.append(f"{fn.__name__} raised: {e!r}")
    if failures:
        print(f"FAIL — {len(failures)} of {checks} checks broken:\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS — {checks} text-selection claims verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

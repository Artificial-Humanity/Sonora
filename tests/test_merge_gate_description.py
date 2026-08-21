"""Nothing may describe the merge gate as the gate it stopped being on 2026-08-20.

⚠ WHY A GUARD AND NOT A FOURTH SWEEP (#203). Three sweeps were run against this. The first
fixed 4 sites and the commit claimed it had found them all; the second found 6 and made the
same claim; the review then found 8. Each pass was done carefully and each left sites behind,
because prose describing a mechanism has no edge that a person can see the end of.

`merge_branch.sh` is the only thing in this repo that reaches `main` on its own, so a document
that misdescribes it is a document that tells an agent the wrong thing about the one
irreversible step in the lane. The scattering is the point: the gate is described in scripts,
personas, the workflow map, a manifest and a test docstring, and nothing compared them.

⚠ THIS IS A TEXT SCAN, WITH THE LIMITS TEXT SCANS HAVE. It catches the phrasings below and is
blind to any other way of saying the same wrong thing — a silent miss, not an error. It is
worth having anyway: a check that catches the spellings that actually occurred is strictly
better than the nothing that let them accumulate. When a green run says "no RECOGNISED
description disagrees", it never says "every description is right".

⚠ **NO COUNT IS GIVEN HERE ON PURPOSE.** The first version said "the three spellings" while
defining six patterns, and the commit message said six (#214) — a number in prose beside a
list is a number that goes stale the next time the list grows, which it did within one review.
Count `STALE` if you need the figure.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pre-floor gate, in the spellings that were actually found in this repo. Each is a claim
# that ANY unclosed issue blocks — which stopped being true when LOW began to ride.
STALE = [
    re.compile(r"`open`, `review`,? (?:or|and) `escalated`"),
    re.compile(r"open, review or escalated"),
    re.compile(r"'open', 'review'"),
    re.compile(r"every issue on the branch is closed"),
    re.compile(r"anything unclosed on it"),
    re.compile(r"no issue in `open`"),
    # ⚠ ADDED AFTER THE GUARD SHIPPED GREEN WHILE MISSING A SITE (#213). DEVELOPER.md said
    # the gate "counts `escalated` alongside `open` and `review`" — a description of the old
    # rule that none of the six patterns above matched. The guard had the same defect as the
    # three sweeps it replaced, one level up: a list assembled from what had been FOUND, not
    # from what could be SAID. Every pattern below came from a real site, and that is exactly
    # why the docstring refuses to call the list complete.
    re.compile(r"counts `escalated` alongside"),
    re.compile(r"counts `open`, `review`"),
    re.compile(r"while any of its issues is (?:open|unclosed)"),
    re.compile(r"any (?:open|unclosed) issue blocks"),
]

# ⚠ EXEMPTIONS ARE PER-LINE SUBSTRINGS AND ARE PRINTED, following the doc-claims gate's
# pattern, so an exemption cannot quietly become a hiding place. These are legitimate
# historical or state-machine statements, NOT descriptions of the merge condition.
EXEMPT = [
    "state` has four exclusive values",          # the state machine, not the gate
    "`open`, `review`, `escalated`, `closed`",   # ditto — the enumeration itself
    "then it became",                            # test_workflow_lane.py's history sentence
    "until 2026-08-17",                          # explicit historical narration
    # ⚠ A line that quotes the old rule AND dates its replacement in the same breath is
    # narrating history, which the repo does on purpose. Found by this guard's first run:
    # test_workflow_lane.py:11 ends "…open, review or escalated"; **since 2026-08-20 it is a'
    # and its correction wraps onto the next line, so a per-line scan cannot see the repair.
    "since 2026-08-20",
]


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z", "*.md", "*.py", "*.sh"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split("\0") if f]


def test_no_tracked_file_describes_the_pre_floor_merge_gate():
    hits, exempted = [], []
    for rel in _tracked():
        if rel.startswith("tests/test_merge_gate_description"):
            continue  # this file names the stale spellings on purpose
        path = os.path.join(REPO, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if "merge_branch.sh" not in text and "merge gate" not in text.lower():
            continue  # only files that speak about the gate can misdescribe it
        for n, line in enumerate(text.splitlines(), 1):
            if not any(p.search(line) for p in STALE):
                continue
            if any(e in line for e in EXEMPT):
                exempted.append(f"{rel}:{n}")
                continue
            hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    print(f"\n  exempted {len(exempted)} historical line(s): {', '.join(exempted) or 'none'}")
    assert not hits, (
        "these describe the merge gate as zero-open-issues; it is a SEVERITY FLOOR "
        "(MEDIUM+ blocks, LOW rides):\n  " + "\n  ".join(hits))


# The owner's second requirement (2026-08-20): "make this simple to reconfigure in a single
# place, should we ever pivot again." A document that names the threshold is a second place.
# ⚠ THESE ENUMERATE PHRASINGS; THE DEFECT IS SEMANTIC. A regex cannot tell "grade this LOW"
# (legitimate — a grading instruction) from "LOW rides" (a threshold claim). So the list is
# built from claim SHAPES that actually occurred, and the docstring refuses to call it
# coverage. The first four caught nothing that was not already exempt — the check had never
# been red — while seven live copies used spellings none of them matched (#219).
HARDCODED = [
    re.compile(r"(?:MEDIUM|LOW|HIGH|CRITICAL) (?:and|or) above"),
    re.compile(r"MEDIUM\+"),
    re.compile(r"\b(?:LOW|MEDIUM|HIGH|CRITICAL) (?:rides|issues RIDE|left open)\b"),
    re.compile(r"(?:medium|low|high|critical) (?:and|or) above", re.I),
    # "grading it LOW lets the branch land", "MEDIUM stops the branch"
    re.compile(r"\b(?:LOW|MEDIUM|HIGH|CRITICAL)\b[^.\n]{0,40}\b(?:stops|lets|blocks) the branch"),
    # "exactly as MEDIUM does", "a LOW may still be open", "a LOW that rides"
    re.compile(r"exactly as (?:LOW|MEDIUM|HIGH|CRITICAL) does"),
    re.compile(r"\bA (?:LOW|MEDIUM|HIGH|CRITICAL) (?:may|that|left)\b"),
]

# Where naming the value is the point rather than a copy: config.env DEFINES it, merge_floor
# and its tests EXERCISE the ladder, and this file lists the spellings it hunts.
VALUE_OWNERS = (
    "workflow/config.env",
    "workflow/scripts/merge_floor.py",
    "tests/test_merge_floor.py",
    "tests/test_merge_gate_description.py",
)


def test_no_document_hardcodes_the_configured_threshold():
    """⚠ THE OWNER'S RULING, AS A CHECK. The floor moved into `workflow/config.env` so a pivot
    is one edit. That only holds while no document repeats the value — and the history here is
    unambiguous: the same rule spelled out in prose survived three sweeps and a first guard.

    A pointer ("the floor set in workflow/config.env") never goes stale. "MEDIUM and above
    blocks" goes stale the moment someone changes the setting, and nothing would say so.
    """
    hits = []
    for rel in _tracked():
        if rel.startswith(VALUE_OWNERS):
            continue
        path = os.path.join(REPO, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if "merge_branch.sh" not in text and "merge gate" not in text.lower():
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if any(p.search(line) for p in HARDCODED):
                hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not hits, (
        "these name the merge floor's threshold instead of pointing at MERGE_SEVERITY_FLOOR "
        "in workflow/config.env, which is the single place it is set:\n  " + "\n  ".join(hits))


def test_the_guard_can_actually_fail():
    """⚠ A guard that cannot go red is the defect it exists to prevent — AGENTS.md §5b."""
    assert any(p.search("it refuses while an issue is open, review or escalated")
               for p in STALE)
    assert not any(p.search("it refuses anything at or above the severity floor")
                   for p in STALE)


def test_the_HARDCODED_guard_can_actually_fail():
    """⚠ #219: `STALE` had this sibling and `HARDCODED` did not, so the second check had never
    been shown to fire. It matched zero non-exempt lines — which reads identically to "the
    repo is clean" and is why seven live copies sat under a green run.

    Every string below is a real spelling that was in the tree at 2b160c9.
    """
    real_sites = [
        "these LOW issues RIDE past the floor and stay open after the merge:",
        "Grading a finding MEDIUM stops the branch; grading it LOW lets the branch land",
        "AN UNGRADED FINDING BLOCKS, exactly as MEDIUM does.",
        "⚠ **A LOW may still be open**: it rides to a follow-up",
        "A LOW that rides is still open",
        "a LOW left open does not stop a merge",
        "MEDIUM and above blocks",
    ]
    for text in real_sites:
        assert any(p.search(text) for p in HARDCODED), f"not caught: {text!r}"

    # …and it must not fire on the pointer form, or the fix would be unwritable.
    for ok in ["a finding at or above the floor set in workflow/config.env blocks",
               "grade this finding LOW unless it misdirects work",
               "anything below the floor rides to a follow-up branch"]:
        assert not any(p.search(ok) for p in HARDCODED), f"false positive: {ok!r}"

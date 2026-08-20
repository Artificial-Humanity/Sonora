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


def test_the_guard_can_actually_fail():
    """⚠ A guard that cannot go red is the defect it exists to prevent — AGENTS.md §5b."""
    assert any(p.search("it refuses while an issue is open, review or escalated")
               for p in STALE)
    assert not any(p.search("it refuses anything at or above the severity floor")
                   for p in STALE)

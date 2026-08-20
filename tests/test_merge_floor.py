"""The merge floor blocks what the owner said it should, and rides what they said it should.

⚠ THESE TESTS CALL `merge_floor.rides` — they do not restate the rule. A test that
reimplements the logic it checks passes when both copies are wrong the same way, which has
happened in this repo four times (#161, #172, #178 and one of #161's own fixes).

⚠ THE FIRST PROOF OF THIS RULE WAS LIVE FIXTURE ISSUES and it is not repeatable: proving
"escalated blocks" needs an escalated issue, `escalated` is a one-way door that only the
owner's `user_decision` clears, and the fixture had to be deleted server-side with its
comments. That is why the rule is a module. See merge_floor.py's docstring.
"""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "merge_floor", os.path.join(REPO, "workflow", "scripts", "merge_floor.py"))
merge_floor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_floor)


@pytest.mark.parametrize("severity", ["medium", "high", "critical"])
def test_medium_and_above_block(severity):
    # The ruling's own words: "Block on MEDIUM and above."
    assert not merge_floor.rides("open", severity)


def test_low_rides_which_is_the_whole_point_of_the_floor():
    # If this ever fails, the floor has silently become the zero-open-issues gate it
    # replaced — and nothing else would report that, because a stricter gate never goes red.
    assert merge_floor.rides("open", "low")


@pytest.mark.parametrize("severity", ["", None, "   "])
def test_ungraded_blocks(severity):
    # Every issue filed before 2026-08-20 has severity="" — 111 of them. A floor that waves
    # through what it cannot grade is not a floor.
    assert not merge_floor.rides("open", severity)


def test_an_unknown_severity_blocks_rather_than_rides():
    # The PocketBase field's values can be widened without anyone touching this repo. When
    # that happens the new value must block until someone decides what it means, exactly as
    # `state!="closed"` fails closed on an unknown state.
    assert not merge_floor.rides("open", "trivial")
    assert not merge_floor.rides("open", "cosmetic")


@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical", ""])
def test_escalated_blocks_at_every_severity(severity):
    # Severity says how bad the finding is; `escalated` says a human is waiting. The floor
    # does not get to overrule the second question with an answer to the first.
    assert not merge_floor.rides("escalated", severity)


def test_review_state_does_not_by_itself_block_a_low():
    # `review` means Ozzy fixed it and Janis has not verified yet. Under the ruling that is
    # still a LOW, and LOW rides — the floor is about severity, not about who looked last.
    # ⚠ This is the one case where the floor is genuinely more permissive than the gate it
    # replaced, so it is stated as a test rather than left implicit.
    assert merge_floor.rides("review", "low")


def test_case_and_whitespace_do_not_change_the_verdict():
    # PocketBase returns exactly what was written; a hand-set "LOW" must not become a block
    # merely for its case, and " low " must not ride on a value nobody meant.
    assert merge_floor.rides("open", "LOW")
    assert merge_floor.rides("open", " low ")
    assert not merge_floor.rides("ESCALATED", "low")

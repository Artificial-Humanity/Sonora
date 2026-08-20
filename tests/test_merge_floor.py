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


@pytest.mark.parametrize("state", ["frobnicated", "", "   ", None, "closed"])
def test_an_unknown_state_blocks_rather_than_rides(state):
    """⚠ #204: the first version blacklisted `escalated` and whitelisted severity, so
    `rides("frobnicated", "low")` was True.

    `merge_branch.sh` documents the opposite property for its own filter — a fifth state must
    block "until someone decides what it means". A guard with one blacklist in it has a hole
    shaped like the future, and this is the test that would have caught it.

    `closed` is included because it must never ride even though the query is supposed to
    exclude it — the floor does not get to assume its caller filtered correctly.

    ⚠ `"OPEN "` is deliberately NOT here. The first draft of this test asserted it should
    block, contradicting its own docstring: the strip-and-fold applies to state exactly as it
    does to severity, so `"OPEN "` normalises to a rideable state and rides. The test was
    wrong, not the code. It is covered below instead.
    """
    assert not merge_floor.rides(state, "low")


@pytest.mark.parametrize("state", ["open", "review", "OPEN ", " Review"])
def test_the_two_rideable_states_still_ride(state):
    """The other half of #204: whitelisting must not have narrowed the floor to nothing.

    An issue Ozzy has fixed sits in `review` until Janis verifies it. If that blocked, a LOW
    could never ride and the floor would silently be the zero-open-issues gate it replaced.

    The odd spellings check that the normalisation is symmetric with severity's — PocketBase
    returns whatever was written, and a hand-set `"OPEN "` means the same thing as `"open"`.
    """
    assert merge_floor.rides(state, "low")


# --------------------------------------------------------------------------- #
# the floor is CONFIGURED, not compiled in (owner, 2026-08-20)
# --------------------------------------------------------------------------- #
def test_the_threshold_comes_from_config_not_from_this_module():
    """⚠ The owner's ruling: settle the rule everywhere it surfaces, AND make it
    reconfigurable from a single place. `workflow/config.env` is that place."""
    assert merge_floor.floor_setting() == "medium"


def test_a_pivot_to_a_different_floor_needs_no_code_change():
    """The whole point of the ruling. Changing one config line must move the boundary, with
    no edit to this module, to merge_branch.sh, or to any document."""
    assert merge_floor.rides("open", "medium", floor="high")     # medium rides under a high floor
    assert not merge_floor.rides("open", "high", floor="high")
    assert not merge_floor.rides("open", "low", floor="low")     # nothing rides under a low floor
    assert merge_floor.rides("open", "low", floor="critical")
    assert merge_floor.rides("open", "high", floor="critical")


@pytest.mark.parametrize("bad", ["", "medum", "MEDIUM-ISH", "none", "off", "0", None])
def test_an_unusable_floor_setting_blocks_everything(bad):
    """⚠ THERE IS DELIBERATELY NO "OFF". A typo in config must not quietly open the gate, and
    "none"/"off"/"0" are the spellings someone reaching for one would try. All block."""
    for sev in ("low", "medium", "high", "critical"):
        assert not merge_floor.rides("open", sev, floor=bad)


def test_a_missing_config_file_blocks_rather_than_defaults(tmp_path):
    """An unreadable config is indistinguishable from a config that says nothing, and both
    must refuse. Defaulting here would mean a broken install merges as though configured."""
    assert merge_floor.floor_setting(str(tmp_path / "absent.env")) == ""
    assert not merge_floor.rides("open", "low", floor="")


def test_the_ladder_is_ordered_and_the_order_is_the_comparison():
    """Position in LADDER *is* the severity comparison, so a reorder silently changes every
    verdict. Pinned so that reordering fails here rather than at a merge."""
    assert merge_floor.LADDER == ("low", "medium", "high", "critical")

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
    "merge_floor", os.path.join(REPO, "FerroStep", "workflow", "scripts", "merge_floor.py"))
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
    reconfigurable from a single place. `FerroStep/workflow/config.env` is that place."""
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


def test_rideable_states_derive_from_the_lane_definition(tmp_path):
    """⚠ Phase 2: WHICH states ride is the lane definition's fact, not this module's.

    A definition marking `review` halted must flip the verdict with no edit here — and no
    readable definition must block EVERYTHING, because a gate that cannot read its own rules
    may not guess them. The last assertion pins that the live definition yields exactly the
    states every other test in this file assumes, so a definition edit that changes the
    rideable set goes red here rather than silently rewriting the floor's meaning.

    ⚠ IT WENT RED AS DESIGNED ON 2026-08-27, when `disputed` was added to the lane. Updated
    deliberately, and the direction was checked before it was: `disputed` is NOT halted, so it
    rides — but only at the severities `open` already rides at, because `rides()` still
    requires `severity < floor`. A disputed MEDIUM blocks exactly as an open medium does.
    That was the thing worth verifying, since a new state that quietly waved findings past
    the gate would have made "dispute it" the cheapest way to land a branch."""
    d = tmp_path / "lane.json"
    d.write_text('{"states": ["open", "review"], "halted": ["review"], "terminal": []}',
                 encoding="utf-8")
    assert merge_floor.rideable_states(str(d)) == frozenset({"open"})
    assert not merge_floor.rides("review", "low", states=merge_floor.rideable_states(str(d)))
    assert merge_floor.rideable_states(str(tmp_path / "absent.json")) == frozenset()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert merge_floor.rideable_states(str(bad)) == frozenset()
    assert merge_floor.rideable_states() == frozenset({"open", "review", "disputed"})
    # ⚠ AND THE VERDICT, NOT JUST THE MEMBERSHIP. The set above is what the floor READS; this
    # is what it DOES with it, and only the second one can catch a new state opening the gate.
    assert merge_floor.rides("disputed", "low") is True
    assert merge_floor.rides("disputed", "medium") is False


def test_the_ladder_is_ordered_and_the_order_is_the_comparison():
    """Position in LADDER *is* the severity comparison, so a reorder silently changes every
    verdict. Pinned so that reordering fails here rather than at a merge."""
    assert merge_floor.LADDER == ("low", "medium", "high", "critical")


# --------------------------------------------------------------------------- #
# #218 — the one tracker write that can clear the gate without closing anything
# --------------------------------------------------------------------------- #
def _issue_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "issue_mod", os.path.join(REPO, "FerroStep", "workflow", "scripts", "issue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_issue_py_does_not_keep_its_own_copy_of_the_severity_order():
    """⚠ Two copies of a severity ORDER disagree silently: the gate ranks one way and the
    grade refusal another, and the symptom is a grade accepted in one place and reclassified
    in the other. The first draft of #218's fix declared its own tuple with a comment claiming
    it was "shared with merge_floor.LADDER" — a relationship asserted and unenforced.
    """
    mod = _issue_module()
    # Equal in value — it must actually rank the same way…
    assert mod.SEVERITY_LADDER == merge_floor.LADDER
    # …and DERIVED, not re-typed. `is` cannot express this: issue.py loads its own instance of
    # merge_floor, so the tuples are equal without being identical. The property worth pinning
    # is that no second literal exists to drift, so the check is on the source.
    src = open(os.path.join(REPO, "FerroStep", "workflow", "scripts", "issue.py"), encoding="utf-8").read()
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert '"low", "medium", "high", "critical"' not in body, \
        "issue.py re-declares the severity ladder instead of importing merge_floor.LADDER"


def test_the_reviewer_name_resolves_from_the_roster_not_config_env():
    """#216 made the setting consumed; 2026-08-24 moved it. Identities live in FerroStep/config.yaml
    (the FerroStep roster), config.env dropped its identity keys in the same commit, and
    issue.py resolves the reviewer through `ferrostep agent-env`. This RUNS the resolution
    and checks both ends: the name is non-empty, and it is the roster's — not a literal
    surviving in issue.py. The config.env half needs no binary and always runs."""
    cfg = open(os.path.join(REPO, "FerroStep", "workflow", "config.env"), encoding="utf-8").read()
    assert "REVIEWER_NAME=" not in cfg, \
        "config.env still carries REVIEWER_NAME — two copies of the identity drift"
    import shutil
    if shutil.which("ferrostep") is None:
        pytest.skip("ferrostep not on PATH — the roster resolution itself was NOT exercised")
    mod = _issue_module()
    name = mod.reviewer_name()
    assert name, "the roster resolved an empty reviewer — the #218 guard would name nobody"
    roster = open(os.path.join(REPO, "FerroStep", "config.yaml"), encoding="utf-8").read()
    assert f"name: {name}" in roster, \
        "issue.py's reviewer does not match FerroStep/config.yaml — it is hardcoded somewhere"


def test_the_grade_guard_is_documented_as_a_convention_not_a_mechanism():
    """⚠ #225: the docstring claimed "no direction that can clear a gate is self-served".

    It does not follow. `--author` is SELF-DECLARED — anyone may pass `--author Janis` — so the
    check is a speed bump that leaves a self-declared name on the record, exactly like the git
    identity DEVELOPER.md §1 calls a convention the repo cannot enforce.

    Claiming a guarantee a guard does not provide is worse than the gap itself: a reader who
    believes a boundary is structural stops checking it. This pins the honest wording so the
    stronger claim cannot come back.
    """
    src = open(os.path.join(REPO, "FerroStep", "workflow", "scripts", "issue.py"), encoding="utf-8").read()
    assert "CONVENTION, NOT A MECHANISM" in src

    # ⚠ THE PHRASE MAY APPEAR, BUT ONLY AS A QUOTATION OF THE OLD CLAIM. This repo's comments
    # routinely explain a fix by quoting the defect, so a bare "not in src" fails on the
    # correction itself — the third guard on this branch to fire on its own explanatory prose.
    # What must not return is the phrase ASSERTED. A quoted line says so on the same line.
    claims = [l for l in src.splitlines()
              if "no direction that can clear a gate is self-served" in l
              and "It said" not in l]
    assert not claims, f"the stronger claim is asserted again, not quoted: {claims}"


BR = "fix/merge-gate-severity-floor"


@pytest.mark.parametrize("caller,author,rec_branch,here,sev,floor,blocks,why", [
    ("Ozzy",  "Janis", BR,      BR, "low",  "medium", True,
     "the case the guard exists for"),
    ("Janis", "Janis", BR,      BR, "low",  "medium", False,
     "#228: the reviewer is exempt — the first version tested only the ISSUE's author, so its "
     "own remedy 'ask Janis to grade it' sent Janis to Janis"),
    ("Ozzy",  "Janis", "old/x", BR, "low",  "medium", False,
     "#229: 102 of 102 ungraded records are the reviewer's, so an author-only test caught the "
     "whole legacy set it claimed to carve out. Branch scope makes the carve-out real"),
    ("Ozzy",  "Ozzy",  BR,      BR, "low",  "medium", False,
     "a worker's own filing is its own to grade"),
    ("Ozzy",  "Janis", BR,      BR, "high", "medium", False,
     "grading UP cannot clear a gate"),
    ("Ozzy",  "Janis", BR,      BR, "low",  "",       False,
     "an uncomparable floor is the caller's own checks to make, not this one's"),
])
def test_the_ungraded_guard_blocks_exactly_the_intended_case(
        caller, author, rec_branch, here, sev, floor, blocks, why):
    """⚠ #231: this guard's only test was a two-line source scan, which cannot tell a guard
    that WORKS from one that is merely PRESENT — #228 and #229 were both live underneath one.

    ⚠ And the alternative to a pure predicate was live fixtures: proving the earlier version by
    hand meant grading a real record on another branch, which Janis flagged as the worker
    setting severity on a reviewer's finding as a byproduct of testing. A pure function needs
    no record and leaves nothing to clean up server-side.
    """
    assert _issue_module().ungraded_guard_blocks(
        caller, author, rec_branch, here, sev, floor, reviewer="Janis") is blocks, why


def test_the_guard_is_inert_when_no_reviewer_is_configured():
    """A repo that has not set REVIEWER_NAME has no reviewer to protect, and the guard must not
    refuse everyone on the strength of an empty string matching an empty author."""
    assert _issue_module().ungraded_guard_blocks(
        "Ozzy", "", BR, BR, "low", "medium", reviewer="") is False


def test_grade_is_listed_in_the_usage_block_with_the_other_writes():
    """#224: `grade` was absent from issue.py's own usage block, so the one command that writes
    the field the merge gate reads was undiscoverable beside its siblings.

    ⚠ This REPLACED a source scan asserting two literal strings from the guard's body (#231).
    That scan could not tell a guard that WORKS from one that is merely PRESENT — #228 and
    #229 were both live underneath it — and it went red on the fix rather than on a defect.
    The guard's behaviour is covered by the parametrised cases above; this pins only a
    discoverability fact, which is the kind of thing a source check can honestly assert.
    """
    src = open(os.path.join(REPO, "FerroStep", "workflow", "scripts", "issue.py"), encoding="utf-8").read()
    usage = src.split('"""', 2)[1]
    for cmd in ("list", "show", "file", "grade", "take", "review", "escalate", "close",
                "reopen", "comment", "escalated"):
        assert f"issue.py {cmd}" in usage, f"{cmd} missing from the usage block"


def test_cmd_grade_actually_CALLS_the_guard_not_merely_defines_it():
    """⚠ #234: `13cfcbe` extracted `ungraded_guard_blocks` and, in the same commit, deleted the
    only assertion that `cmd_grade` is wired to it. The predicate was then well covered and the
    WIRING was covered by nothing — a guard that exists and is never called is the shape this
    whole branch has been about.

    ⚠ #233 is why this is a RUN and not a read. `13cfcbe`'s message claimed the branch-scope
    condition was "verified by regrading #181, a legacy record the previous version would have
    refused." #181 was ALREADY `low` when I ran that, and the guard only fires on UNGRADED
    records — so the command exercised nothing and I reported it twice. This test is the
    verification that claim should have been.

    Run against a fake tracker, following `test_workflow_lane.py`'s idiom: no live record, so
    nothing to clean up server-side afterwards.
    """
    import contextlib
    import importlib.util
    import io
    import types

    spec = importlib.util.spec_from_file_location(
        "_issue_grade_wiring", os.path.join(REPO, "FerroStep", "workflow", "scripts", "issue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Inject the cache rather than resolving: this test pins the guard's WIRING, not the
    # roster resolution (covered above), and must run where `ferrostep` is not on PATH.
    mod._REVIEWER_CACHE = "Janis-under-test"
    reviewer = mod.reviewer_name()
    here = mod.current_branch()
    patched = {}
    # UNGRADED, filed by the reviewer, stamped with the branch we are on — the exact case.
    rec = {"id": "x", "number": 1, "severity": "", "author": reviewer, "branch_name": here}
    fake = types.SimpleNamespace(
        find=lambda args, number: rec,
        call=lambda path, method=None, body=None: patched.update(body or {}) or (200, {}),
        add_comment=lambda args, r, text, author: None,
    )
    args = types.SimpleNamespace(number=1, repo="r", comment="", author="Ozzy",
                                 severity="low")

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(SystemExit):
            mod.cmd_grade(fake, args)
    assert not patched, "cmd_grade wrote the severity despite the guard refusing"

    # …and the same command succeeds once any single condition is false — here, the caller.
    patched.clear()
    args.author = reviewer
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        mod.cmd_grade(fake, args)
    assert patched.get("severity") == "low", (
        "the reviewer was refused too — that is #228, which this branch fixed")

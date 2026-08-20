"""`merge_branch.sh` classifies and refuses — run, not read.

⚠ WHY THIS FILE EXISTS (#208). Every other assertion about this script greps its SOURCE, and
the only execution was `bash -n`. That is a syntax check, not a behaviour check — and the
verdict this glue computes is the sole decision gating every merge to `main`.

**Two defects landed in exactly this glue on a green 1391-test suite.** #201: the rule module
was never committed, so the import died on any fresh checkout. #205: `BLOCKING` was built with
`grep '^BLOCK '`, which silently dropped any line matching neither prefix, in the same commit
that removed the catch-all `die` protecting it.

⚠ **Of those two, THIS FILE CATCHES ONLY #205** — and an earlier revision of this paragraph
claimed it caught both, fifteen lines above the passage explaining why it cannot (#212). #201
is a missing import target; the import lives inside the heredoc, and the fixture stubs the
heredoc away. **#201 belongs to #207's tracked-dependency check, not to anything here.**
Getting a coverage claim wrong in the file written to document a coverage limit is why that
limit is now stated in two places.

HERMETIC BY CONSTRUCTION, the same pattern as `test_request_review.py`: `python3` is stubbed
onto PATH so the gate reads canned tracker output instead of PocketBase, `--dry-run` stops
before any git write, and the stub is what makes the classification observable at all.

⚠⚠ WHAT THIS FILE DOES **NOT** COVER, STATED BECAUSE IT LOOKS LIKE IT DOES. The stub replaces
the `python3` heredoc, and `merge_floor.rides()` is called **inside** that heredoc — so these
tests never execute the floor rule. They exercise the SHELL half: how already-classified lines
are routed, which of the two refusals is printed, and the exit status.

Measured, not assumed: setting `RIDES = frozenset()` — which makes every LOW block and turns
the floor back into the zero-open-issues gate — leaves **all seven of these passing**. The rule
is covered by `tests/test_merge_floor.py`, which imports and calls it directly.

Naming that split matters more than it looks. `test_a_low_finding_rides_and_the_merge_proceeds`
reads like end-to-end proof that a below-floor finding rides. It is not. It proves the shell
does the right thing **when handed a line the rule already marked RIDE**. Believing otherwise would leave the
most important behaviour in the lane covered by a test that cannot see it — which is the
finding this file was written for.
"""
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "workflow", "scripts", "merge_branch.sh")


@pytest.fixture
def gate(tmp_path):
    """Run merge_branch.sh --dry-run with the tracker query stubbed. Returns (rc, output)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()

    def _run(tracker_output):
        # ⚠ The stub replaces `python3` ONLY for the gate's heredoc query. It echoes the lines
        # the real query would print, so everything downstream — the classification, the
        # routing between the two refusals, the exit status — is the script's own code.
        stub = bindir / "python3"
        stub.write_text(
            "#!/bin/sh\n"
            "cat >/dev/null\n"          # swallow the heredoc on stdin
            f"printf '%s' {shquote(tracker_output)}\n"
        )
        stub.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{bindir}:{env['PATH']}"
        p = subprocess.run(
            [SCRIPT, "--branch", "test/gate-behaviour", "--dry-run", "--no-push"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
        )
        return p.returncode, p.stdout + p.stderr

    return _run


def shquote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def test_a_medium_finding_refuses_the_merge(gate):
    rc, out = gate("BLOCK  #1     open       medium   a real finding\n")
    assert rc != 0
    assert "below the merge floor" in out


def test_a_line_marked_ride_does_not_block_the_merge(gate):
    """⚠ SCOPE: this is about the SHELL, not the rule. It proves a RIDE-marked line does not
    reach `$BLOCKING` and the dry run proceeds. Whether a LOW *becomes* a RIDE is
    `test_merge_floor.py`'s question — see this module's docstring for why the distinction is
    load-bearing rather than pedantic."""
    rc, out = gate("RIDE   #1     open       low      a nit\n")
    assert rc == 0, out
    assert "RIDE" in out
    assert "would: git checkout" in out


def test_an_ungraded_finding_refuses(gate):
    rc, out = gate("BLOCK  #1     open       UNGRADED filed before severity existed\n")
    assert rc != 0
    assert "UNGRADED must be graded" in out


def test_output_the_gate_cannot_classify_refuses_and_says_so(gate):
    """⚠ #205 + #209 together. An unrecognised line must refuse (#205) AND must not be
    described as a finding with an issue number to grade (#209) — there is no number."""
    rc, out = gate("surprise: the tracker changed shape\n")
    assert rc != 0
    assert "not recognise" in out
    # ⚠ Anchored on "no issue number", not on a whole sentence. The literal was
    # "nothing to grade" until #210 reworded this path, and the assertion went red on a
    # rewording rather than a behaviour change — which is the failure mode the sibling
    # guard in test_workflow_lane.py already carries a comment about.
    assert "no issue number" in out
    assert "below the merge floor" not in out


def test_a_tracker_outage_refuses_without_calling_it_a_finding(gate):
    rc, out = gate("TRACKER_UNREACHABLE: connection refused\n")
    assert rc != 0
    assert "cannot read the tracker" in out


def test_a_branch_with_no_issues_refuses_as_unreviewed(gate):
    """"Nobody reviewed it" and "reviewed clean" are the same reading from the tracker."""
    rc, out = gate("NEVER_REVIEWED\n")
    assert rc != 0
    assert "NO issues at all" in out


def test_a_missing_floor_module_names_itself_rather_than_the_tracker(gate):
    """It fails closed either way; the point is that the message sends the reader to the
    right place. Routing it through the tracker error cost a real diagnosis once."""
    rc, out = gate("FLOOR_MODULE_MISSING: No module named 'merge_floor'\n")
    assert rc != 0
    assert "merge_floor.py is missing" in out
    assert "NOT a tracker problem" in out


def test_mixed_output_does_not_tell_you_there_is_nothing_to_grade(gate):
    """⚠ #210: #209's defect with the sign flipped, inside #209's own fix.

    The branch was chosen on "ANY line is unrecognised" and then printed a message asserting
    "ALL of this is unrecognised" — so a list holding both a stray line and a real finding got
    "there is nothing to grade" printed directly beneath a gradeable issue number.

    Both causes must be named, separately. A reader told "nothing to grade" while findings sit
    in the same block will believe the half that lets them stop looking.
    """
    rc, out = gate("surprise: the tracker changed shape\n"
                   "BLOCK  #1     open       medium   a real finding\n")
    assert rc != 0
    assert "not recognise" in out          # the stray line is named
    assert "these ARE findings below the floor" in out   # and so are the real ones
    assert "#1" in out


def test_unrecognised_output_alone_does_not_invent_findings(gate):
    """The other side of #210: with no BLOCK lines, the findings paragraph must not appear —
    or the gate would report findings that do not exist, which is the same lie inverted."""
    rc, out = gate("surprise: the tracker changed shape\n")
    assert rc != 0
    assert "not recognise" in out
    assert "these ARE findings below the floor" not in out

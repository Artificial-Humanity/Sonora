"""The driver's safety properties, pinned because it runs unattended with write access.

`workflow/review_cycle.sh` alternates a review with a `claude -p` WORKER that edits files and
commits, without a human in the loop. That is a different risk class from anything else in
`scripts/`, and exactly one property makes it acceptable:

    ⚠ NOTHING IT RUNS CAN REACH `main`.

The repo has no branch protection and force-push is unblocked (AGENTS.md §1), so the deny on
`git push` is the only thing between an unattended loop and production. Every assertion here
exists because the fix that establishes it would otherwise be correct-but-unpinned — the shape
this cycle produced five times (#110, #118, #119, and twice more), and the one time it guarded
a *capability* rather than a sentence, reverting it would have restored unbounded nested
reviews (#115).
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "workflow" / "review_cycle.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def _array(name):
    start = SOURCE.index(f"{name}=(")
    end = SOURCE.index("\n)", start)
    out = []
    for line in SOURCE[start:end].splitlines():
        out.extend(re.findall(r'"([^"]+)"', line.split("#", 1)[0]))
    return out


def test_the_worker_is_denied_git_push():
    """The single property that makes an unattended write loop acceptable."""
    assert "Bash(git push:*)" in _array("WORKER_DENY")


def test_the_driver_itself_never_pushes():
    """Denying the worker is pointless if the driver pushes on its behalf."""
    # ⚠ STRING LITERALS ARE STRIPPED FIRST. A naive search for `git push` matches the deny
    # entry "Bash(git push:*)" and the closing message that tells the human to push by hand —
    # both are data. The first version of this test failed against a correct script for
    # exactly that reason, which is the same false-red that #112's parser produced.
    code = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
    code = re.sub(r'"[^"\n]*"', '""', code)
    code = re.sub(r"'[^'\n]*'", "''", code)
    assert not re.search(r"\bgit\s+push\b", code), "review_cycle.sh must not push"


def test_the_worker_cannot_re_enter_the_loop():
    """A worker that can run the driver or the launcher recurses, billed, unattended.

    ⚠ THE FULL PATH IS CHECKED, NOT THE BASENAME, AND THAT IS NOW THE POINT. This asserted
    `any("review_cycle.sh" in d for d in deny)` until 2026-08-17 — a substring test that stayed
    GREEN after the scripts moved from `scripts/` to `workflow/` while every deny entry still
    named the old path. The guard was disarmed and the test said nothing, because a permission
    entry naming a path that does not exist denies precisely nothing.
    """
    deny = _array("WORKER_DENY")
    for s in ("workflow/review_cycle.sh", "workflow/request_review.sh",
              "workflow/merge_branch.sh"):
        assert f"Bash({s}:*)" in deny, f"the worker must not be able to run {s}"
        assert f"Bash(./{s}:*)" in deny, f"the './' form of {s} is a second way in"


def test_every_denied_script_actually_exists():
    """⚠ A DENY ON A PATH THAT DOES NOT EXIST IS NOT A DENY.

    The failure above was invisible because nothing tied the deny list to the filesystem. This
    does: rename or move a script without updating the list in the same commit, and the entry
    silently stops matching while the worker regains the capability.
    """
    for entry in _array("WORKER_DENY"):
        m = re.fullmatch(r"Bash\(\.?/?([\w][\w./-]*\.(?:sh|py)):\*\)", entry)
        if m:
            assert (REPO / m.group(1)).is_file(), \
                f"WORKER_DENY names {m.group(1)}, which does not exist — it denies nothing"


def test_the_worker_keeps_the_tracker_script():
    """⚠ The inverse assertion, deliberate.

    `workflow/issue.py` is how the worker takes an issue, comments, escalates, and moves one to
    `review` — its actual job — and it is the single path that enforces the counter cap and the
    mandatory-comment rules. Denying it would push the worker back to raw tracker writes, where
    none of those refusals exist.
    """
    assert not any("issue.py" in d for d in _array("WORKER_DENY"))


def test_git_dash_c_is_NOT_denied_to_the_worker():
    """⚠ The inverse assertion, and it is deliberate.

    DEVELOPER.md §1 requires every commit to be authored
    `git -c user.name=Ozzy -c user.email=…`. Denying `git -c` — which the *reviewer* denies,
    correctly, for the opposite reason — would leave the worker unable to commit as itself,
    and the failure would look like an identity bug rather than a permission one.
    """
    assert not any(d.startswith("Bash(git -c") for d in _array("WORKER_DENY"))


def test_every_claude_call_carries_a_spend_ceiling():
    """It spends money unattended; an uncapped call is the whole hazard."""
    for m in re.finditer(r"^\s*claude -p .*?(?=\n\s*\w+=\$\?|\n\s*set -e)", SOURCE, re.S | re.M):
        assert "--max-budget-usd" in m.group(0), "an unattended claude call with no ceiling"


def test_the_review_ceiling_is_bounded_to_four():
    """Three fix passes need four reviews; five would mean the cap is not working."""
    assert '=~ ^[1-4]$' in SOURCE


def test_it_refuses_a_dirty_tree():
    """The worker commits, so uncommitted edits would be swept into a commit nobody wrote."""
    assert "working tree is dirty" in SOURCE


def test_the_abort_marker_is_checked_before_the_exit_code():
    """A review can exit 0 and still say the change must not land — a content signal, not a
    failure signal. Order matters: checking rc first would `exit` before reading the text."""
    marker = SOURCE.index("MUST-NOT-LAND")
    rc_check = SOURCE.index('if [[ "$RC" -ne 0 ]]')
    assert marker < rc_check


def test_the_reviewer_persona_is_told_to_emit_the_marker():
    """The driver greps for a token the reviewer must know to produce — a machine contract
    split across two files, which is how one half gets edited alone."""
    persona = (REPO / "workflow" / "REVIEWER.md").read_text(encoding="utf-8")
    assert "MUST-NOT-LAND" in persona


def _open_filter():
    """The convergence check itself, not merely a line that resembles it.

    ⚠ The predecessor of this helper is why it exists. That test asserted `'branch_name!=""'
    in SOURCE` — anywhere in the file — and stayed green after the convergence filter stopped
    containing it, because an unrelated guard further down still did. A substring search over
    a whole script does not pin the line you mean.
    """
    return next(ln for ln in SOURCE.splitlines() if ln.startswith("OPEN_FILTER="))


def test_convergence_is_scoped_to_the_branch_under_review():
    """Convergence must count THIS branch's issues and no others.

    ⚠ The old guard was `branch_name!=""`, which excluded the migrated GitHub backlog only
    while that backlog carried no branch at all. It now sits on `github-issues-fixes`, so the
    clause that once dropped it would sweep it in — and since nobody works it until that
    branch is rebased, the loop could never see itself converge.
    """
    flt = _open_filter()
    assert 'branch_name=\\"$BRANCH\\"' in flt, flt
    assert 'branch_name!=""' not in flt, flt
    # ...and $BRANCH is the checked-out branch, resolved exactly once.
    assigns = [ln for ln in SOURCE.splitlines() if ln.startswith("BRANCH=")]
    assert len(assigns) == 1, assigns
    assert "rev-parse --abbrev-ref HEAD" in assigns[0], assigns[0]


def _guard_readings():
    """The two `agent_passes` readings the stall guard compares."""
    return [ln.strip() for ln in SOURCE.splitlines()
            if ln.strip().startswith(("BEFORE_SUM=", "AFTER_SUM="))]


def test_the_stall_guard_compares_like_with_like():
    """⚠ THIS GUARD FAILED OPEN FOR ITS ENTIRE LIFE, AND SILENTLY.

    `BEFORE_SUM` was the COUNT of open issues on this branch; `AFTER_SUM` was the COUNT of
    `agent_passes=0` issues REPO-WIDE. Two populations, two scopes, compared for equality — so
    the stall it exists to catch could only be caught by coincidence, and a crashed worker
    would have looped forever spending no attempts. That is the exact unbounded loop the cap
    exists to prevent, reintroduced by the thing automating it.

    Nothing about the output said so: a guard that never fires looks identical to a guard that
    never needed to. So this pins the shape rather than the wording — BOTH readings must come
    from the same helper on the same argument.
    """
    before, after = _guard_readings()
    assert before.split("=", 1)[1] == after.split("=", 1)[1], (before, after)
    assert "pb_passes" in before, before


def test_the_stall_guard_sums_over_every_state():
    """A worker may ESCALATE, which takes an issue out of `state="open"`.

    If the sum were restricted to open issues it could FALL across a pass that did real work,
    and the guard would read that as a stall and stop the loop. `pb_passes` filters on
    `branch_name` alone, deliberately.
    """
    body = SOURCE[SOURCE.index("pb_passes() {"):]
    body = body[:body.index("\n}\n")]
    flt = next(ln for ln in body.splitlines() if "urllib.parse.quote" in ln)
    assert "branch_name" in flt, flt
    assert "state=" not in flt, flt
    assert "perPage=500" in body, "an unpaged read silently under-sums"


def test_a_falling_counter_is_not_treated_as_a_stall():
    """The owner resets `agent_passes` to re-arm an issue — their dial, never "corrected".

    A drop therefore means work happened AND the owner intervened, which is the one case where
    exiting would be most wrong.
    """
    assert "AFTER_SUM < BEFORE_SUM" in SOURCE
    stall = SOURCE.index("AFTER_SUM == BEFORE_SUM")
    fall = SOURCE.index("AFTER_SUM < BEFORE_SUM")
    exits = [m.start() for m in re.finditer(r"exit 5", SOURCE)]
    assert any(stall < e < fall for e in exits), "the stall branch must be the one that exits"


def test_no_query_names_the_removed_escalated_field():
    """`escalated` is a value of `state`, not a field beside it (owner, 2026-08-17).

    Not a style preference: the column is GONE, and PocketBase answers a filter naming it
    with HTTP 400. A leftover `escalated=false` would not merely fail to narrow a query — it
    makes the driver read the tracker as unreachable and stop.
    """
    for ln in SOURCE.splitlines():
        code = ln.split("#", 1)[0]
        if "state=" in code or code.startswith("OPEN_FILTER="):
            assert "escalated=" not in code, ln
    assert 'state=\\"open\\"' in _open_filter()


def test_it_is_classified_in_the_pipeline_manifest():
    manifest = (REPO / "scripts" / "pipeline_manifest.py").read_text(encoding="utf-8")
    assert "workflow/review_cycle.sh" in manifest


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0

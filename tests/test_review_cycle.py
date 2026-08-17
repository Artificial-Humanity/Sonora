"""The driver's safety properties, pinned because it runs unattended with write access.

`scripts/review_cycle.sh` alternates a review with a `claude -p` WORKER that edits files and
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
SCRIPT = REPO / "scripts" / "review_cycle.sh"
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
    """A worker that can run the driver or the launcher recurses, billed, unattended."""
    deny = _array("WORKER_DENY")
    for s in ("review_cycle.sh", "request_review.sh"):
        assert any(s in d for d in deny), f"the worker must not be able to run {s}"


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
    persona = (REPO / "Personas" / "REVIEWER.md").read_text(encoding="utf-8")
    assert "MUST-NOT-LAND" in persona


def test_convergence_excludes_the_migrated_backlog():
    """`branch_name!=""` drops the 9 migrated GitHub issues, which carry no branch_name and are
    worked by nobody. Without it the loop can never see itself converge."""
    assert 'branch_name!=""' in SOURCE


def test_it_is_classified_in_the_pipeline_manifest():
    manifest = (REPO / "scripts" / "pipeline_manifest.py").read_text(encoding="utf-8")
    assert "scripts/review_cycle.sh" in manifest


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0

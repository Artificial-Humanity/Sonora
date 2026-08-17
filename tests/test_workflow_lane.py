"""`workflow/` — the merge gate and the tracker script.

These two are where the workflow stopped being prose. `workflow/WORKFLOW.md` describes a state
machine and two mandatory-comment rules; `issue.py` refuses the violations and
`merge_branch.sh` refuses the merge. This repo's most expensive recurring lesson is that **a
rule in a file is not an enforcement mechanism**, so what is pinned here is the refusing, not
the wording.

⚠ `merge_branch.sh` is the only thing in the repo that reaches `main` on its own. Until
2026-08-17 the property that made the lane safe was "nothing here can push"; now it is "nothing
merges while an issue is open, review or escalated". A narrower guard, and the reason the
assertions below are specific about it.
"""

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "workflow" / "merge_branch.sh"
ISSUE = REPO / "workflow" / "issue.py"
MERGE_SRC = MERGE.read_text(encoding="utf-8")
ISSUE_SRC = ISSUE.read_text(encoding="utf-8")
WORKFLOW_MD = (REPO / "workflow" / "WORKFLOW.md").read_text(encoding="utf-8")

# ⚠ COMMENTS STRIPPED BEFORE ANY SEARCH FOR AN OPERATION.
#
# The first version of this file searched the raw text and produced four false failures at
# once: `merge_branch.sh`'s header explains why the worker used to be denied `git push`, and
# `git push` in that sentence is prose. Every one of those tests was reading documentation and
# reporting it as behaviour — the same substring-over-the-whole-file defect these tests were
# written to catch elsewhere. Search the code, not the file.
MERGE_CODE = "\n".join(ln.split("#", 1)[0] for ln in MERGE_SRC.splitlines())


# --- the merge gate ---------------------------------------------------------------------

def test_the_gate_is_fail_closed_on_unknown_states():
    """⚠ `state!="closed"`, NOT a list of the three open states.

    If a fifth state is ever added, `!="closed"` blocks the merge until someone decides what it
    means; `open||review||escalated` would silently ignore it and let the branch land. A merge
    gate must fail towards refusing.
    """
    assert 'state!="closed"' in MERGE_SRC
    assert '"escalated"' not in MERGE_SRC.split("UNSETTLED=")[1].split("PY\n")[0], \
        "the gate enumerates states instead of excluding closed — that is fail-open"


def test_an_unreachable_tracker_refuses_the_merge():
    """Unreachable is not clear. The opposite choice lands unreviewed work whenever
    PocketBase happens to be down — the failure being silent is what makes it serious."""
    assert "TRACKER_UNREACHABLE" in MERGE_SRC
    assert MERGE_SRC.index("TRACKER_UNREACHABLE:*") < MERGE_SRC.index("if [[ -n \"$UNSETTLED\""), \
        "the unreachable check must precede the emptiness check, or 'down' reads as 'clear'"


def test_the_gate_runs_before_any_git_write():
    """A stale local view must not be able to authorise a merge."""
    for op in ("git checkout", "git merge --no-ff", "git push"):
        assert MERGE_CODE.index("UNSETTLED=") < MERGE_CODE.index(op), op


def test_the_push_names_both_ends_of_the_refspec():
    """⚠ `push.default=upstream` is set in this repo, so a bare `git push` from a branch that
    inherited `origin/main` as its upstream sends it to main whatever the branch is called.
    Naming both ends means what lands is what was just merged and gated."""
    m = re.search(r"git push \S+ (\S+)", MERGE_CODE)
    assert m and ":" in m.group(1), "the push must use an explicit src:dst refspec"


def test_it_parses():
    assert subprocess.run(["bash", "-n", str(MERGE)]).returncode == 0
    ast.parse(ISSUE_SRC)


# --- the tracker script -----------------------------------------------------------------

def _transitions():
    """`TRANSITIONS` as the script actually defines it, read from the AST rather than imported
    — importing would authenticate to PocketBase at module scope."""
    tree = ast.parse(ISSUE_SRC)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TRANSITIONS":
            return ast.literal_eval(node.value)
    raise AssertionError("TRANSITIONS not found in issue.py")


def test_escalated_has_no_route_back_to_open():
    """⚠ Only the owner releases an escalation, and a server-side hook performs it.

    A subcommand that moved `escalated -> open` would let an agent quietly un-ask a question it
    raised, which is the one thing escalation exists to prevent.
    """
    for action, rule in _transitions().items():
        assert "escalated" not in rule["from"], \
            f"{action} claims to move an issue out of 'escalated'"


def test_reopen_and_escalate_require_a_comment():
    """The two mandatory-comment rules, as refusals rather than sentences.

    `reopen` spends one of Ozzy's three attempts; without a reason it is spent on a guess.
    `escalate` asks the owner a question; one with no question attached cannot be answered.
    """
    src = ISSUE_SRC
    assert 'require_comment(args, "reopen")' in src
    assert 'require_comment(args, "escalate")' in src
    fn = src[src.index("def require_comment"):src.index("def show_row")]
    assert "die(" in fn, "require_comment must refuse, not warn"


def test_the_counter_only_moves_up_by_one_and_stops_at_the_cap():
    """`take` is the only writer of `agent_passes`, and the cap is what bounds the loop."""
    fn = ISSUE_SRC[ISSUE_SRC.index("def cmd_take"):ISSUE_SRC.index("def transition")]
    assert "cur + 1" in fn, "the counter must advance by exactly one"
    assert ">= MAX_PASSES" in fn, "the cap must be enforced where the counter moves"
    # ⚠ The WRITE form, `"agent_passes":`. Counting the bare name also counts the READ
    # (`rec.get("agent_passes")`) and fails against correct code — which it did.
    assert fn.count('"agent_passes":') == 1, "take must write the counter exactly once"
    # ⚠ `file` legitimately writes `"agent_passes": 0` — a new issue starts at zero, and the
    # workflow names that as one of the three fields filing must set. Excluding it by hand
    # rather than loosening the assertion: everything OTHER than these two must not touch it.
    filing = ISSUE_SRC[ISSUE_SRC.index("def cmd_file"):ISSUE_SRC.index("def cmd_take")]
    others = ISSUE_SRC.replace(fn, "").replace(filing, "")
    assert '"agent_passes":' not in others, "a third place writes agent_passes"


def test_nothing_writes_user_decision():
    """It is the owner's field. An agent writing there forges the answer to its own question."""
    assert '"user_decision":' not in ISSUE_SRC


def test_there_is_no_delete():
    """⚠ The tracker is the sole record that a finding ever existed.

    Checked as CODE, not as text: the docstring says "there is no delete subcommand and there
    must never be one", and a naive search for the word finds that sentence and fails.
    """
    tree = ast.parse(ISSUE_SRC)
    names = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "delete" not in names
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value.strip().upper() != "DELETE", "an HTTP DELETE is constructed"
    assert 'add("delete")' not in ISSUE_SRC and 'add_parser("delete")' not in ISSUE_SRC


def test_filing_sets_the_three_fields_the_workflow_requires():
    """`state: open`, `agent_passes: 0`, and the branch — the workflow names all three, and
    each was previously something a reviewer had to remember."""
    fn = ISSUE_SRC[ISSUE_SRC.index("def cmd_file"):ISSUE_SRC.index("def cmd_take")]
    assert '"state": "open"' in fn
    assert '"agent_passes": 0' in fn
    assert '"branch_name": branch' in fn


def test_the_issue_number_collision_is_retried():
    """⚠ `number` does not auto-assign and is unique per repo, so two writers filing at once
    collide. Retried rather than pre-reserved: the unique index is the real arbiter, and a
    reserved-then-abandoned number leaves a permanent hole in the sequence."""
    fn = ISSUE_SRC[ISSUE_SRC.index("def cmd_file"):ISSUE_SRC.index("def cmd_take")]
    assert "for attempt in range(" in fn


# --- the map ----------------------------------------------------------------------------

def test_the_workflow_map_matches_the_states_the_code_enforces():
    """WORKFLOW.md is the map; drift between it and the scripts is the failure this catches."""
    for state in ("open", "review", "escalated", "closed"):
        assert f"`{state}`" in WORKFLOW_MD
    assert "workflow/merge_branch.sh" in WORKFLOW_MD

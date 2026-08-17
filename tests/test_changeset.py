"""`scripts/changeset.sh` — the local-only pull request, and the gate work must pass to land.

Untested until 2026-08-17, which is how the defect pinned below reached a working tree: it is
the script that decides whether a branch has converged, so a wrong answer here either strands
finished work or lands unfinished work, and neither is visible from its output.

Two classes of assertion:

  * the MERGE GATE reads the tracker correctly (convergence, and what escalation excludes)
  * the script SURVIVES ITS OWN QUOTING — three layers deep (bash -> heredoc -> python),
    which is where it actually broke
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "changeset.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def _heredoc_body():
    """The python handed to `python3 -` inside pbq(), between the redirection and its terminator.

    ⚠ Anchored on the REDIRECTION LINE, not on the token `<<PY`. Searching for the bare token
    matched the prose warning above pbq() — which discusses the heredoc and quite properly
    contains backticks, being a *shell* comment where they expand to nothing. The first run of
    this test failed on that comment rather than on any python. A test that locates its
    subject by substring finds the first thing that looks like it, not the thing.
    """
    m = re.search(r"^\s*python3 - .*<<PY$", SOURCE, re.M)
    assert m, "pbq()'s heredoc redirection not found — did the invocation change?"
    start = m.end()
    return SOURCE[start:SOURCE.index("\nPY\n", start)]


def test_the_python_heredoc_contains_no_backticks():
    """⚠ THE HEREDOC IS UNQUOTED, SO BACKTICKS IN IT ARE COMMAND SUBSTITUTION.

    It cannot be quoted — `${_body}` has to interpolate — so the shell expands the whole
    python body, comments included. MEASURED: a markdown-style `state` written in a `#`
    comment ran `state` as a command, deleted the line it was on, and produced

        changeset.sh: line 60: state: command not found
        IndentationError: expected an indented block ...

    Two errors, neither of them naming a backtick, on a line that held nothing but prose.
    `bash -n` does not catch it either — the script is syntactically valid; the damage
    happens at expansion time.

    ⚠ AND IT IS NOT RELIABLY FATAL, which is the real argument for a test. Re-introducing the
    exact defect and running `list` printed `line 67: state: command not found` to stderr and
    then **the correct output anyway** — the eaten line happened not to matter to that
    subcommand. Whether this corrupts a comment or a `def` is down to where the backticks
    land, so the visible symptom ranges from nothing at all to a merge gate that cannot
    answer. Do not rely on noticing it.
    """
    body = _heredoc_body()
    assert "`" not in body, [ln for ln in body.splitlines() if "`" in ln]


def test_it_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_the_merge_gate_counts_open_issues_on_the_branch():
    """Convergence is `state="open"` scoped to the changeset's own branch — nothing wider."""
    body = _heredoc_body()
    fn = body[body.index("def open_issue_count"):]
    fn = fn[:fn.index("\n${")] if "\n${" in fn else fn[:400]
    assert 'branch_name="%s"' in fn, fn
    assert 'state="open"' in fn, fn


def test_no_query_names_the_removed_escalated_field():
    """`escalated` is a value of `state`, not a field beside it (owner, 2026-08-17).

    The column is gone; PocketBase answers a filter naming it with HTTP 400. In this script
    that failure lands squarely on the merge gate, which is the worst place for it.
    """
    for ln in SOURCE.splitlines():
        code = ln.split("#", 1)[0]
        if "escalated=" in code:
            raise AssertionError(ln)


def test_the_owners_queue_reads_the_escalated_state():
    """`status` must still show what the owner owes a decision on — by state, now."""
    assert 'state=\\"escalated\\"' in SOURCE


def test_merge_is_checked_server_side_before_touching_git():
    """A stale local view must not be able to authorise a merge.

    The count comes back from the tracker and gates the `git merge` that follows; reversing
    that order would let a cached or hand-edited local state land unconverged work.
    """
    merge = SOURCE[SOURCE.index("\nmerge)"):]
    gate = merge.index('[[ "$OPENC" == "0" ]]')
    assert gate < merge.index("git merge --no-ff"), "merge runs before the convergence gate"
    assert gate < merge.index("git checkout"), "checkout runs before the convergence gate"


def test_it_never_pushes():
    """Same property as review_cycle.sh, for the same reason: no branch protection, force-push
    unblocked (AGENTS.md §1). `merge` merges LOCALLY and stops."""
    for ln in SOURCE.splitlines():
        code = ln.split("#", 1)[0]
        assert not re.search(r"\bgit\s+push\b", code), ln
    assert "NOT PUSHED" in SOURCE


def test_it_refuses_to_open_a_changeset_on_main():
    """The point of a local PR is that the work is somewhere other than where it lands."""
    assert '"$BRANCH" != "main"' in SOURCE

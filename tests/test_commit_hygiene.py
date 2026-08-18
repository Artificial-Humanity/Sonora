"""Commit authorship and trailers on the current branch — checked, not remembered.

WHY THIS FILE EXISTS (2026-08-18, issue #101)
---------------------------------------------
Two rules in `workflow/DEVELOPER.md` §1 had no enforcement:

  * commits are authored **Ozzy <ozzy@artificialhumanity.io>**, via the `-c` pair, because the
    repo's configured identity is deliberately the owner's — so a forgotten `-c` does not
    error, it silently commits an agent's work under a human's name;
  * **no `Co-Authored-By: Ziggy` trailer** — you are the author, and a co-author trailer
    naming a different agent misattributes the work.

⚠ **BOTH WERE BROKEN FOR EIGHT COMMITS AND NOTHING NOTICED.** The trailer came from a
`CLAUDE.md` at the workspace root that had already been DELETED; an agent went on obeying it
from a summary written before the deletion. A reviewer reading `DEVELOPER.md` caught it by
hand. That is the definition of a rule without a mechanism, and `AGENTS.md` §1 is explicit
that a rule is not one.

⚠ **HISTORY IS NOT REWRITTEN.** The owner's decision, 2026-08-18: leave the existing commits
alone, fix the hygiene going forward. So this checks commits **after** `GRANDFATHERED_THROUGH`
and says plainly which ones it is not looking at — an exemption that cannot quietly widen.

⚠⚠ **THE SCOPE IS THIS BRANCH'S OWN UNMERGED WORK, AND THE FIRST VERSION GOT THAT WRONG**
(issue #102). It checked every commit reachable from `HEAD` after the boundary, which is not
"the agent's commits" — it is *everyone's*. Two consequences, both measured on 2026-08-18:

  * `merge_branch.sh:162` runs `git merge --no-ff` with no `-c` pair, so the merge commit
    carries the configured identity. Performing that merge in a throwaway clone produced
    `fcff394 lmcfarlin <2363604+lmcfarlin@users.noreply.github.com>` and turned this file red
    on `main`. **The script that lands a branch was the thing that broke the suite.**
  * The owner is 388 of 396 commits on `main` and hand-commits there regularly. DEVELOPER.md
    §1 says the configured identity was left as theirs **deliberately**, so the guard
    forbade exactly what the repo permits on purpose.

And the failure message told whoever saw it to re-author the commit as Ozzy. The likely
reader was the owner, looking at their own work; a guard whose remedy misattributes a human's
commits to an agent is worse than no guard, since #101 was about attribution being silently
wrong and this made it loudly wrong in the other direction.

So the range is `BASE..HEAD`, non-merge, after the boundary — the commits this lane produced
and has not yet landed. On the base branch itself there is nothing to check, and this file
SKIPS with a message rather than passing quietly.
"""

import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(key, default):
    """⚠ FROM `workflow/config.env`, NOT TYPED OUT HERE. `workflow/` is a portable lane meant
    to be copied whole into another repo (AGENTS.md §1); an identity hardcoded in `tests/`
    does not travel with it and becomes a second definition that drifts."""
    try:
        src = open(os.path.join(REPO, "workflow", "config.env"), encoding="utf-8").read()
    except OSError:
        return default
    m = re.search(r"^%s=(.*)$" % re.escape(key), src, re.M)
    return m.group(1).split("#", 1)[0].strip() if m else default


DEVELOPER = (_cfg("DEVELOPER_NAME", "Ozzy"), _cfg("DEVELOPER_EMAIL", "ozzy@artificialhumanity.io"))
BASE_BRANCH = _cfg("BASE_BRANCH", "main")
BAD_TRAILER = "Co-Authored-By: Ziggy"

# ⚠ THE BOUNDARY, AND IT IS A COMMIT, NOT A DATE. Everything up to and including this SHA
# predates the owner's decision and is deliberately exempt. A date would drift with the
# clock; a SHA is a fact. `80be925` is the last commit made under the old habit.
GRANDFATHERED_THROUGH = "80be925"


def _base_ref():
    """The remote-tracking base if there is one, else the local branch, else None."""
    for ref in ("refs/remotes/origin/" + BASE_BRANCH, "refs/heads/" + BASE_BRANCH):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                          cwd=REPO, capture_output=True).returncode == 0:
            return ref
    return None


def _on_base_branch():
    r = subprocess.run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                       cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == BASE_BRANCH


def _commits():
    """(sha, author, email, body) for THIS BRANCH'S own unmerged commits after the boundary.

    ⚠ `--no-merges`, and scoped to `BASE..HEAD` (issue #102). A merge commit is made by
    `merge_branch.sh` without a `-c` pair and carries the configured identity, which is the
    owner's on purpose; and everything already on the base branch is history this guard was
    never given a mandate over. Empty when the boundary is not an ancestor — a branch cut
    from elsewhere is not evidence of anything."""
    base = _base_ref()
    if base is None or _on_base_branch():
        return []
    anc = subprocess.run(["git", "merge-base", "--is-ancestor",
                          GRANDFATHERED_THROUGH, "HEAD"], cwd=REPO, capture_output=True)
    if anc.returncode != 0:
        return []
    r = subprocess.run(
        ["git", "log", "--no-merges", f"{base}..HEAD", f"^{GRANDFATHERED_THROUGH}",
         "--format=%H%x1f%an%x1f%ae%x1f%B%x1e"],
        cwd=REPO, capture_output=True, text=True)
    out = []
    for rec in r.stdout.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, name, email, body = rec.split("\x1f", 3)
        out.append((sha[:9], name, email, body))
    return out


def test_no_commit_carries_the_ziggy_trailer():
    bad = [f"  {sha}  {name}" for sha, name, _e, body in _commits() if BAD_TRAILER in body]
    assert not bad, (
        "commits carry a `Co-Authored-By: Ziggy` trailer, which DEVELOPER.md §1 rules out "
        "inside Sonora — you are the author:\n" + "\n".join(bad))


def test_commits_are_authored_as_the_developer():
    bad = [f"  {sha}  {name} <{email}>" for sha, name, email, _b in _commits()
           if (name, email) != DEVELOPER]
    assert not bad, (
        "unmerged commits on this branch are not authored as the developer:\n"
        + "\n".join(bad)
        + "\n\n⚠ READ WHICH OF THESE TWO IT IS BEFORE ACTING.\n"
        "  * An AGENT wrote them and forgot the `-c` pair. The repo's configured identity is "
        f"the owner's on purpose, so nothing errors — the work just lands under their name. "
        f"Recommit as:\n"
        f"      git -c user.name={DEVELOPER[0]} -c user.email={DEVELOPER[1]} commit …\n"
        "  * The OWNER hand-committed on an agent's branch. Then the authorship is correct "
        "and MUST NOT be changed — DEVELOPER.md §1 leaves the configured identity theirs so "
        "their own commits stay theirs. Move the commit to its own branch, or add its SHA to "
        "an exemption here as a deliberate decision.\n"
        "Do not re-author a human's commit as an agent to make this pass.")


def test_the_grandfather_boundary_still_exists_and_is_named():
    """⚠ A guard whose exemption points at nothing is not exempting — it is disarmed.

    If the boundary SHA ever leaves this history (a rebase, a fresh clone with a shallow
    fetch), `_commits()` returns empty and BOTH tests above pass while checking nothing. This
    is the test that notices.
    """
    r = subprocess.run(["git", "cat-file", "-e", GRANDFATHERED_THROUGH + "^{commit}"],
                       cwd=REPO, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"{GRANDFATHERED_THROUGH} is not in this checkout's history — the "
                    f"hygiene checks above are inert here and are NOT reporting a pass")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor",
                          GRANDFATHERED_THROUGH, "HEAD"], cwd=REPO, capture_output=True)
    if anc.returncode != 0:
        pytest.skip(f"{GRANDFATHERED_THROUGH} is not an ancestor of HEAD — nothing to check "
                    f"on this branch, and that is not a pass either")
    if _base_ref() is None:
        pytest.skip(f"no {BASE_BRANCH} ref in this checkout — the range BASE..HEAD cannot be "
                    f"formed, so the checks above are inert and are NOT reporting a pass")
    if _on_base_branch():
        pytest.skip(f"HEAD is {BASE_BRANCH} — this guard covers a branch's own unmerged "
                    f"commits, so there is nothing here to check (issue #102)")
    assert True


def test_the_range_excludes_what_it_is_not_a_guard_over():
    """⚠ THE #102 REGRESSION TEST, as a property of the range rather than of one SHA.

    Two things must stay out of `_commits()`: anything already on the base branch, and merge
    commits. The first is history this guard has no mandate over; the second is made by
    `merge_branch.sh` with the configured identity and would fail the moment a branch lands.
    """
    base = _base_ref()
    if base is None or _on_base_branch():
        pytest.skip("no branch range to check here")
    shas = {sha for sha, _n, _e, _b in _commits()}
    if not shas:
        pytest.skip("no unmerged commits after the boundary on this branch")
    on_base = subprocess.run(["git", "log", base, "--format=%h"],
                             cwd=REPO, capture_output=True, text=True).stdout.split()
    overlap = shas & {s[:9] for s in on_base}
    assert not overlap, f"the range reaches commits already on {base}: {sorted(overlap)}"
    merges = subprocess.run(["git", "log", "--merges", f"{base}..HEAD", "--format=%h"],
                            cwd=REPO, capture_output=True, text=True).stdout.split()
    assert not (shas & {m[:9] for m in merges}), "the range includes a merge commit"

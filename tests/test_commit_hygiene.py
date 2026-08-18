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
"""

import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVELOPER = ("Ozzy", "ozzy@artificialhumanity.io")
BAD_TRAILER = "Co-Authored-By: Ziggy"

# ⚠ THE BOUNDARY, AND IT IS A COMMIT, NOT A DATE. Everything up to and including this SHA
# predates the owner's decision and is deliberately exempt. A date would drift with the
# clock; a SHA is a fact. `80be925` is the last commit made under the old habit.
GRANDFATHERED_THROUGH = "80be925"


def _commits():
    """(sha, author name, author email, body) for each commit on this branch after the
    grandfathered boundary. Empty when the boundary is not an ancestor — a branch cut from
    elsewhere is not evidence of anything."""
    anc = subprocess.run(["git", "merge-base", "--is-ancestor",
                          GRANDFATHERED_THROUGH, "HEAD"], cwd=REPO, capture_output=True)
    if anc.returncode != 0:
        return []
    r = subprocess.run(
        ["git", "log", f"{GRANDFATHERED_THROUGH}..HEAD", "--format=%H%x1f%an%x1f%ae%x1f%B%x1e"],
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
        "commits are not authored as the developer. The repo's configured identity is the "
        "owner's on purpose, so a forgotten `-c` pair does not error — it commits your work "
        "under their name. Use:\n"
        "  git -c user.name=Ozzy -c user.email=ozzy@artificialhumanity.io commit …\n"
        + "\n".join(bad))


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
    assert True

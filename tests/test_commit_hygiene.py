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


def _base_ref(repo=REPO):
    """The remote-tracking base if there is one, else the local branch, else None."""
    for ref in ("refs/remotes/origin/" + BASE_BRANCH, "refs/heads/" + BASE_BRANCH):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                          cwd=repo, capture_output=True).returncode == 0:
            return ref
    return None


def _on_base_branch(repo=REPO):
    r = subprocess.run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                       cwd=repo, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == BASE_BRANCH


def _commits(repo=REPO, boundary=GRANDFATHERED_THROUGH):
    """(sha, author, email, body) for THIS BRANCH'S own unmerged commits after the boundary.

    ⚠ `--no-merges`, and scoped to `BASE..HEAD` (issue #102). A merge commit is made by
    `merge_branch.sh` without a `-c` pair and carries the configured identity, which is the
    owner's on purpose; and everything already on the base branch is history this guard was
    never given a mandate over. Empty when the boundary is not an ancestor — a branch cut
    from elsewhere is not evidence of anything."""
    base = _base_ref(repo)
    if base is None or _on_base_branch(repo):
        return []
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", boundary, "HEAD"],
                         cwd=repo, capture_output=True)
    if anc.returncode != 0:
        return []
    r = subprocess.run(
        # ⚠ `%(trailers)` AS WELL AS `%B` — git's own parse, not a substring of the body.
        # See test_no_commit_carries_the_ziggy_trailer for why.
        ["git", "log", "--no-merges", f"{base}..HEAD", f"^{boundary}",
         "--format=%H%x1f%an%x1f%ae%x1f%(trailers)%x1f%B%x1e"],
        cwd=repo, capture_output=True, text=True)
    out = []
    for rec in r.stdout.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, name, email, trailers, body = rec.split("\x1f", 4)
        # ⚠ FULL SHA, TRUNCATED ONLY WHERE IT IS PRINTED (issue #108). This used to store
        # `sha[:9]`, and the regression test below then compared that set against `%h`
        # output — seven characters here, because `core.abbrev=auto` scales with repo size.
        # A nine-character string never equals a seven-character one, so BOTH of its
        # assertions were incapable of failing, whatever the range contained. Comparing
        # full object names removes the question rather than picking a matching width.
        out.append((sha, name, email, trailers, body))
    return out


def _short(sha):
    return sha[:9]


def carries_bad_trailer(trailers, body):
    """True if this commit carries the forbidden co-author line, by EITHER mechanism.

    ⚠ TWO CHECKS, BECAUSE EACH ALONE HAS A MEASURED HOLE (#242).

    `%(trailers)` is git's own parse and is the right primary: it excludes a mention of the
    name in ordinary prose, which the old substring check flagged as a violation — that is
    why the mechanism changed, and it was the correct change.

    But git only parses a trailer block in the LAST paragraph. A commit whose message is
    `…trailer…` + blank line + one closing sentence carries the line verbatim and parses as
    having no trailers at all. Measured over four shapes in a throwaway repo; that one is the
    gap, and the old substring check caught it.

    So: the parsed field, OR a LINE-ANCHORED match on the body. Line-anchored is what keeps
    prose out — `Suite: 1429 passed` mentioning the name mid-sentence does not start a line
    with `Co-Authored-By:`.
    ⚠ THE SECOND CHECK MUST MATCH THE TRAILER'S SHAPE, NOT ITS WORDS. A first version used
    `re.match(r"\s*Co-Authored-By:.*Ziggy", line)` and immediately flagged `6e89848` — whose
    message QUOTES the trailer while explaining it, and whose prose wraps so that the phrase
    begins a line. That is the defect `0afe042` removed, reintroduced by its own fix.

    A real trailer is `Key: Name <address>` alone on its line. Prose is not, however it wraps.
    """
    if BAD_TRAILER in trailers:
        return True
    return any(re.fullmatch(r"Co-Authored-By:\s*Ziggy\s*<[^>]+>\s*", line, re.I)
               for line in body.splitlines())


def test_no_commit_carries_the_ziggy_trailer():
    bad = [f"  {_short(sha)}  {name}" for sha, name, _e, trailers, body in _commits()
           if carries_bad_trailer(trailers, body)]
    assert not bad, (
        "commits carry a `Co-Authored-By: Ziggy` trailer, which DEVELOPER.md §1 rules out "
        "inside Sonora — you are the author:\n" + "\n".join(bad))


def test_commits_are_authored_as_the_developer():
    bad = [f"  {_short(sha)}  {name} <{email}>" for sha, name, email, _t, _b in _commits()
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


OWNER = ("lmcfarlin", "owner@example.invalid")


def _git(repo, *args, **kw):
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=env, **kw)
    assert r.returncode == 0, f"git {' '.join(args)} failed:\n{r.stderr}"
    return r.stdout.strip()


def _commit(repo, msg, who, n=[0]):
    n[0] += 1
    (repo / f"f{n[0]}.txt").write_text(msg, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", f"user.name={who[0]}", "-c", f"user.email={who[1]}",
         "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def landed_branch(tmp_path):
    """A branch that has merged the base INTO itself, which is where `--no-merges` bites.

    ⚠ BUILT RATHER THAN OBSERVED, and that is the whole point of issue #108. The two
    properties this guard holds — no merge commits, nothing already on base — are
    INDISTINGUISHABLE from the real repo's current shape: `origin/main..HEAD` contains no
    merge, and it happens to equal `GRANDFATHERED_THROUGH..HEAD` on this branch. Measured:
    with `--no-merges` deleted, or the range reverted to `{GRANDFATHERED_THROUGH}..HEAD`,
    the whole file stayed green. A guard for a condition the repo does not currently
    exhibit has to create the condition.

    ⚠ Note the shape that does NOT work here, because it is the obvious one: landing the
    branch onto base and checking from base. After `merge_branch.sh` merges, `base..HEAD`
    on the branch is EMPTY — base contains everything the branch had — and on `main` the
    guard skips by design. The first version of this fixture did that and its own control
    test caught it.
    """
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", BASE_BRANCH)
    boundary = _commit(repo, "boundary", OWNER)          # stands in for GRANDFATHERED_THROUGH
    _git(repo, "branch", "work")
    owner_on_base = _commit(repo, "owner hand-commit on base", OWNER)
    _git(repo, "checkout", "-q", "work")
    dev = _commit(repo, "agent work on the branch", DEVELOPER)
    # Catching the branch up on base. `git merge` takes the CONFIGURED identity — no `-c`
    # pair, exactly as merge_branch.sh:162 does — so this commit is owner-authored and sits
    # squarely inside `base..HEAD`.
    _git(repo, "-c", f"user.name={OWNER[0]}", "-c", f"user.email={OWNER[1]}",
         "merge", "--no-ff", BASE_BRANCH, "-q", "-m", f"merge {BASE_BRANCH} into work")
    merge = _git(repo, "rev-parse", "HEAD")
    return {"repo": repo, "boundary": boundary, "owner_on_base": owner_on_base,
            "dev": dev, "merge": merge}


def test_a_merge_commit_never_enters_the_range(landed_branch):
    """The failure #102 was filed about, reproduced in miniature."""
    shas = {sha for sha, _n, _e, _t, _b in _commits(landed_branch["repo"], landed_branch["boundary"])}
    assert landed_branch["merge"] not in shas, (
        "the merge commit is in the range, so the author check will fail on it — and it is "
        "made by merge_branch.sh with the owner's configured identity, on purpose")


def test_a_commit_already_on_the_base_branch_never_enters_the_range(landed_branch):
    shas = {sha for sha, _n, _e, _t, _b in _commits(landed_branch["repo"], landed_branch["boundary"])}
    assert landed_branch["owner_on_base"] not in shas, (
        "a commit already on the base branch is in the range. That is history this guard "
        "has no mandate over, and the owner is 388 of 396 commits there")


def test_the_branch_own_work_DOES_enter_the_range(landed_branch):
    """⚠ THE CONTROL. Both assertions above are 'X is not in a set', which an empty set
    satisfies for free — the exact way the first version of this test passed while checking
    nothing. This one fails if the range has gone inert."""
    shas = {sha for sha, _n, _e, _t, _b in _commits(landed_branch["repo"], landed_branch["boundary"])}
    assert landed_branch["dev"] in shas, (
        "the branch's own commit is NOT in the range, so the two exclusion tests above are "
        "passing on an empty set and prove nothing")


def test_the_range_excludes_what_it_is_not_a_guard_over():
    """⚠ THE #102 REGRESSION TEST, as a property of the range rather than of one SHA.

    Two things must stay out of `_commits()`: anything already on the base branch, and merge
    commits. The first is history this guard has no mandate over; the second is made by
    `merge_branch.sh` with the configured identity and would fail the moment a branch lands.
    """
    base = _base_ref()
    if base is None or _on_base_branch():
        pytest.skip("no branch range to check here")
    shas = {sha for sha, _n, _e, _t, _b in _commits()}
    if not shas:
        pytest.skip("no unmerged commits after the boundary on this branch")
    assert all(len(s) == 40 for s in shas), "expected full object names from --format=%H"

    def _log(*args):
        return set(subprocess.run(["git", "log", *args, "--format=%H"],
                                  cwd=REPO, capture_output=True, text=True).stdout.split())

    # ⚠ THE CONTROL. Both assertions below are "an intersection is empty", which is also
    # what a comparison of two incompatible formats produces — that is exactly how the
    # first version of this test passed while checking nothing (issue #108). So first
    # prove the two sides CAN meet: HEAD is on HEAD's own log.
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    assert head in _log("-1", "HEAD"), (
        "the comparison itself is broken — a commit is not matching its own log entry, so "
        "the two assertions below would pass no matter what the range held")

    overlap = shas & _log(base)
    assert not overlap, f"the range reaches commits already on {base}: {sorted(overlap)}"
    assert not (shas & _log("--merges", f"{base}..HEAD")), \
        "the range includes a merge commit"


@pytest.mark.parametrize("shape,message,should_fire", [
    ("A — trailer as the last paragraph",
     "fix: something\n\nCo-Authored-By: Ziggy <ziggy@artificialhumanity.io>\n", True),
    ("B — trailer, then a closing sentence",
     "fix: something\n\nCo-Authored-By: Ziggy <ziggy@artificialhumanity.io>\n\n"
     "And some closing prose here.\n", True),
    ("C — the name in prose only",
     "fix: something\n\nThis explains why a Ziggy trailer is forbidden here.\n\n"
     "Suite: 1429 passed.\n", False),
    ("D — alongside another trailer",
     "fix: something\n\nSigned-off-by: Someone <s@example.com>\n"
     "Co-Authored-By: Ziggy <ziggy@artificialhumanity.io>\n", True),
])
def test_the_trailer_detection_actually_fires(tmp_path, shape, message, should_fire):
    """⚠ THE CONTROL #241 SAID WAS MISSING, AND IT WAS RIGHT.

    `0afe042` moved this guard onto `%(trailers)` — an unvalidated git format field — and
    added no proof the field is ever non-empty. If it returned "" for every commit (an older
    git, a typo in the format, a field reorder, a change to BAD_TRAILER's spelling) the guard
    would pass green forever while checking nothing.

    That hazard is what this entire FILE is built around: three of its tests exist solely as
    controls, and its module docstring narrates issue #108, where both assertions of a guard
    were incapable of failing whatever the range contained. The one assertion with no control
    was the one that grew a new dependency.

    ⚠ The `landed_branch` fixture cannot serve: every commit it builds has a one-line message,
    so `%(trailers)` is empty for all of them. It exercises the RANGE, never the FIELD.

    Shapes A–D are the four measured in #242. B is the one `%(trailers)` alone misses.
    """
    repo = tmp_path / "t"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", BASE_BRANCH)
    (repo / "f.txt").write_text(shape, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", f"user.name={DEVELOPER[0]}", "-c", f"user.email={DEVELOPER[1]}",
         "commit", "-q", "-m", message)
    trailers = _git(repo, "log", "-1", "--format=%(trailers)")
    body = _git(repo, "log", "-1", "--format=%B")
    assert carries_bad_trailer(trailers, body) is should_fire, (
        f"shape {shape}: detection returned the wrong answer")

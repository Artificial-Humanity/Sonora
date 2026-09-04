"""`merge_branch.sh` must author its merge commit as the ROSTER's developer — run, not read.

⚠ WHY THIS FILE EXISTS. The script reached `main` on its own for the length of the lane's
life and never resolved the roster identity, so every merge it made was authored by whatever
`user.email` the repo happened to carry. That is the owner's, deliberately and permanently
(DEVELOPER.md §1), so the failure is SILENT — git does not error, the push succeeds, and the
misattribution is only visible to someone who thinks to run `git log --merges`. Merge commits
on `main` carry the owner's name from exactly this gap.

⚠ IT IS A BEHAVIOUR TEST ON PURPOSE. `test_merge_gate_behaviour.py` says it plainly: every
other assertion about this script greps its SOURCE, and a grep cannot tell a `-c` pair that is
present from one that is reached. So this drives the REAL merge path — past `--dry-run`, past
the dirty-tree check — inside a throwaway repository, and reads the author off the commit that
comes out.

HERMETIC. A temp git repo gets the lane's two config files, a `python3` stub standing in for
the tracker query, and a `ferrostep` stub standing in for the roster. The repo's own configured
identity is set to a stand-in for the owner, so a regression does not merely fail to set the
right name — it produces the wrong one, which is the actual defect.

⚠ `--no-push` throughout. Nothing here has a remote.
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "FerroStep", "workflow", "scripts", "merge_branch.sh")

REPO_IDENT = ("Repo Owner", "owner@example.invalid")     # what a missed -c pair would use
ROSTER_IDENT = ("Roster Dev", "dev@example.invalid")     # what the roster says


def _git(cwd, *args):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def lane(tmp_path):
    """A throwaway repo with a merge candidate, the lane's config, and stubbed helpers.

    Returns a runner taking the script path to execute, so a MUTATED copy can be driven
    through the identical harness — which is what makes the control meaningful.
    """
    work = tmp_path / "repo"
    (work / "FerroStep" / "workflow" / "scripts").mkdir(parents=True)
    _git(None if False else str(work.parent), "init", "-q", "-b", "main", str(work))
    _git(work, "config", "user.name", REPO_IDENT[0])
    _git(work, "config", "user.email", REPO_IDENT[1])

    # the lane's real settings, so the gate's own reads resolve
    shutil.copy(os.path.join(REPO, "FerroStep", "workflow", "config.env"),
                work / "FerroStep" / "workflow" / "config.env")
    shutil.copy(os.path.join(REPO, "FerroStep", "config.yaml"), work / "FerroStep" / "config.yaml")
    (work / "seed.txt").write_text("base\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "seed")

    _git(work, "checkout", "-q", "-b", "feature/x")
    (work / "seed.txt").write_text("changed\n")
    _git(work, "commit", "-qam", "a reviewed change")
    _git(work, "checkout", "-q", "main")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    # tracker query -> nothing blocking, so the gate proceeds to the merge
    (bindir / "python3").write_text("#!/bin/sh\ncat >/dev/null\nprintf 'RIDE   #1  open  low  a rideable finding\\n'\n")
    (bindir / "python3").chmod(0o755)
    # roster -> the shell assignments `ferrostep agent-env` emits
    (bindir / "ferrostep").write_text(
        "#!/bin/sh\n"
        f"echo \"AGENT_TITLE=developer\"\n"
        f"echo \"AGENT_NAME='{ROSTER_IDENT[0]}'\"\n"
        f"echo \"AGENT_EMAIL='{ROSTER_IDENT[1]}'\"\n")
    (bindir / "ferrostep").chmod(0o755)

    def _run(script=SCRIPT):
        env = dict(os.environ)
        env["PATH"] = f"{bindir}:{env['PATH']}"
        # ⚠ --no-review is not incidental. Without it the gate attempts a review, which
        # cannot run in a throwaway repo; the script tolerates that and continues, so the
        # test would still pass — while exercising an error path nobody chose. Pinning it
        # keeps what these tests drive stable under someone else's edit to that sub-step.
        p = subprocess.run([script, "--branch", "feature/x", "--no-push", "--no-review"],
                           cwd=str(work), env=env, capture_output=True, text=True, timeout=120)
        author = subprocess.run(["git", "log", "-1", "--format=%an <%ae>"], cwd=str(work),
                                capture_output=True, text=True).stdout.strip()
        return p.returncode, p.stdout + p.stderr, author

    _run.work = work
    return _run


def test_the_merge_commit_is_authored_by_the_roster_not_the_repo(lane):
    rc, out, author = lane()
    assert rc == 0, out
    assert author == f"{ROSTER_IDENT[0]} <{ROSTER_IDENT[1]}>", out
    assert author != f"{REPO_IDENT[0]} <{REPO_IDENT[1]}>"


def test_without_the_c_pair_the_owners_identity_is_what_lands(lane, tmp_path):
    """The control. Strip the `-c` pair and the same harness must produce the WRONG author.

    Without this the test above could pass because the repo happened to be configured that
    way, or because nothing reached the merge at all. It has to be able to come out wrong.
    """
    mutated = tmp_path / "merge_branch_mutated.sh"
    src = open(SCRIPT, encoding="utf-8").read()
    broken = src.replace(
        'git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" \\\n    merge --no-ff "$BRANCH"',
        'git merge --no-ff "$BRANCH"')
    assert broken != src, "the -c pair moved — this control is no longer mutating anything"
    mutated.write_text(broken)
    mutated.chmod(0o755)

    rc, out, author = lane(str(mutated))
    assert author == f"{REPO_IDENT[0]} <{REPO_IDENT[1]}>", (
        f"the mutation did not produce the owner's identity (got {author!r}) — "
        "so the passing test above is not evidence the -c pair is what does the work")
    assert rc != 0, "the post-merge author check should have refused the mutated merge"


def test_a_roster_refusal_stops_the_merge_rather_than_landing_it_wrong(lane, tmp_path):
    """A roster that refuses must leave `main` untouched, not fall back to the repo identity."""
    bad = tmp_path / "bin" / "ferrostep"
    bad.write_text("#!/bin/sh\necho 'roster: no such entry' >&2\nexit 1\n")
    bad.chmod(0o755)

    rc, out, author = lane()
    assert rc != 0, out
    assert "NOTHING WAS MERGED" in out, out
    assert author == f"{REPO_IDENT[0]} <{REPO_IDENT[1]}>" and "seed" in subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=str(lane.work), capture_output=True,
        text=True).stdout, "main moved despite the roster refusing"

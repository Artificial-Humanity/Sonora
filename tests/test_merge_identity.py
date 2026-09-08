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

⚠ HERMETIC INCLUDES THE CALLER'S SHELL (#395). `GIT_AUTHOR_*` / `GIT_COMMITTER_*` override every
`-c user.*` pair below — the seed commits' AND the script's — so an exported `GIT_AUTHOR_NAME`
in the developer's own shell (the environment #394 was filed from) made the seed commit the
environment's, the script refuse up front, and three tests fail with messages naming the wrong
cause ("main moved" when it had not). The fixture strips those variables from the environment
it hands to git and to the script; `env_extra` is the ONLY way a test puts them back.

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


# ⚠ Stripped from the environment the fixture hands to git and to the script (#395). Any of
# these in the caller's shell overrides the `-c user.*` pairs this file's premise rests on.
IDENTITY_VARS = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")


def _hermetic_env():
    return {k: v for k, v in os.environ.items() if k not in IDENTITY_VARS}


def _git(cwd, *args, env=None):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True, check=True,
                          env=env if env is not None else _hermetic_env())


def _ident(cwd, line="author"):
    """`Name <email>` of HEAD's author or committer line."""
    fmt = {"author": "%an <%ae>", "committer": "%cn <%ce>"}[line]
    return subprocess.run(["git", "log", "-1", f"--format={fmt}"], cwd=str(cwd),
                          capture_output=True, text=True).stdout.strip()


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
    # ⚠ The premise, asserted where it is made (#395): every test below reads "the owner's
    # identity" off REPO_IDENT. If the seed carries anything else, the failure belongs here,
    # under its own name — not in a test three assertions later saying "main moved".
    for line in ("author", "committer"):
        assert _ident(work, line) == f"{REPO_IDENT[0]} <{REPO_IDENT[1]}>", (
            f"the seed commit's {line} is {_ident(work, line)!r}, not the stand-in owner — "
            "the fixture is not hermetic against the calling environment")

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

    def _run(script=SCRIPT, *flags, env_extra=None):
        env = _hermetic_env()
        env["PATH"] = f"{bindir}:{env['PATH']}"
        env.update(env_extra or {})
        # ⚠ --no-review is not incidental. Without it the gate attempts a review, which
        # cannot run in a throwaway repo; the script tolerates that and continues, so the
        # test would still pass — while exercising an error path nobody chose. Pinning it
        # keeps what these tests drive stable under someone else's edit to that sub-step.
        p = subprocess.run([script, "--branch", "feature/x", "--no-push", "--no-review", *flags],
                           cwd=str(work), env=env, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr, _ident(work)

    _run.work = work
    return _run


def test_the_merge_commit_is_authored_by_the_roster_not_the_repo(lane):
    rc, out, author = lane()
    assert rc == 0, out
    assert author == f"{ROSTER_IDENT[0]} <{ROSTER_IDENT[1]}>", out
    assert author != f"{REPO_IDENT[0]} <{REPO_IDENT[1]}>"
    assert _ident(lane.work, "committer") == f"{ROSTER_IDENT[0]} <{ROSTER_IDENT[1]}>", out


@pytest.fixture
def exported_identity(monkeypatch):
    """The environment #395 named: the developer's shell with `GIT_AUTHOR_NAME` exported.

    Requested BEFORE `lane` in the test signature so it is in `os.environ` while the fixture
    seeds its repo — which is where the leak bit, not only in the script run."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Exported Person")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Exported Person")


def test_the_fixture_is_hermetic_against_an_exported_identity(exported_identity, lane):
    """#395. With `GIT_AUTHOR_NAME` in the calling process's environment the seed commit was
    authored by it, the script refused up front, and the reader was sent to "main moved".
    Neither may happen: the seed is the stand-in owner's (asserted inside the fixture) and the
    merge goes through authored by the roster, exactly as it does from a clean shell."""
    assert os.environ["GIT_AUTHOR_NAME"] == "Exported Person"     # the case is really set up
    rc, out, author = lane()
    assert rc == 0, out
    assert author == f"{ROSTER_IDENT[0]} <{ROSTER_IDENT[1]}>", out
    assert "Exported Person" not in out, out


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


def test_the_dry_run_previews_the_c_pair_the_real_merge_uses(lane):
    """#393. The preview printed the bare `git merge --no-ff` — the pre-9056233 command, the
    one that authored merges as the owner — under a comment arguing that a preview of a
    different command is worse than none. It must name the same `-c` pair the merge below
    it runs with, resolved from the roster, and say the author is checked afterwards."""
    rc, out, author = lane(SCRIPT, "--dry-run")
    assert rc == 0, out
    assert (f'would: git checkout main && git -c user.name="{ROSTER_IDENT[0]}" '
            f'-c user.email="{ROSTER_IDENT[1]}" merge --no-ff feature/x') in out, out
    assert "check the merge author" in out, out
    assert "git merge --no-ff" not in out, "the pre-fix command is still being previewed"
    assert "seed" in subprocess.run(["git", "log", "-1", "--format=%s"], cwd=str(lane.work),
                                    capture_output=True, text=True).stdout, "a dry run merged"


def test_the_dry_run_still_answers_without_a_roster_and_says_so(lane, tmp_path):
    """The property 9056233 chose — a dry run reports the gate's verdict on a box with no
    roster — survives #393. But it must say the roster did not resolve, not print a green
    preview of a merge that would refuse."""
    bad = tmp_path / "bin" / "ferrostep"
    bad.write_text("#!/bin/sh\necho 'roster: no such entry' >&2\nexit 1\n")
    bad.chmod(0o755)

    rc, out, author = lane(SCRIPT, "--dry-run")
    assert rc == 0, out
    assert "<roster developer>" in out and "did NOT resolve" in out, out


@pytest.mark.parametrize("var", IDENTITY_VARS)
def test_git_author_in_the_environment_is_refused_before_the_merge(lane, var):
    """#394. `GIT_AUTHOR_*` OVERRIDES a `-c user.*` pair (measured 2026-09-07), so with it set
    the merge would land under the invoker's name and the post-merge check would refuse a
    commit already on `main`. Refusing first leaves `main` where it was — and the message
    names the variable, so the reader is not sent to amend a commit that never happened.
    #396: `GIT_COMMITTER_*` does the same to the committer line, and was neither refused nor
    checked — a merge landed "Roster Dev authored, Committer Person committed" and pushed."""
    rc, out, author = lane(SCRIPT, env_extra={var: "Someone Else"})
    assert rc != 0, out
    assert var in out and "NOTHING WAS MERGED" in out, out
    assert "seed" in subprocess.run(["git", "log", "-1", "--format=%s"], cwd=str(lane.work),
                                    capture_output=True, text=True).stdout, "main moved"
    assert "--amend" not in out, "refused up front, so no amend instruction should print"


@pytest.mark.parametrize("line", ["author", "committer"])
def test_git_identity_in_the_environment_does_override_the_c_pair(lane, tmp_path, line):
    """The control for the test above: with the up-front refusal stripped, the same
    environment produces a merge whose author — or, #396, committer — line is the
    ENVIRONMENT's: the outcome the refusal exists to prevent, caught only afterwards by the
    post-merge check. Without this the refusal could be guarding against something git does
    not do, and the committer half of the check could be reading a line nothing can change."""
    mutated = tmp_path / "merge_branch_mutated.sh"
    src = open(SCRIPT, encoding="utf-8").read()
    broken = src.replace('if _OVERRIDE="$(env_identity_override)"; then\n  die',
                         'if false; then\n  die')
    assert broken != src, "the up-front refusal moved — this control mutates nothing"
    mutated.write_text(broken)
    mutated.chmod(0o755)

    prefix = {"author": "GIT_AUTHOR", "committer": "GIT_COMMITTER"}[line]
    rc, out, author = lane(str(mutated), env_extra={f"{prefix}_NAME": "Env Person",
                                                    f"{prefix}_EMAIL": "env@example.invalid"})
    got = _ident(lane.work, line)
    assert got == "Env Person <env@example.invalid>", (
        f"the environment did not override the -c pair on the {line} line (got {got!r}), so "
        "the up-front refusal is guarding against a mechanism that does not exist")
    assert rc != 0 and "--amend" in out, out

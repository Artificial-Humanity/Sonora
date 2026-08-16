"""The review launcher's guards, which have twice been defeated with the suite green (#110).

`scripts/request_review.sh` decides what a reviewer is told and what it is allowed to run.
Nothing tested it. That is not a general "add tests" gap — it is specific, and it has a
history:

* **#99** — `--range` accepted a bare ref. `git rev-list HEAD` is the whole history while
  `git diff --stat HEAD` is the working tree, so the brief announced 371 commits beside a
  two-file diffstat of uncommitted edits.
* **#106** — the guard written for #99 tested `*".."*`, and **`...` contains `..`**, so it
  admitted three-dot ranges and reproduced #99 through the guard built to stop it. On a
  divergent pair: 411 commits against a diffstat describing a different comparison.

Both landed with 1063 tests passing, because none of them ran this file. **A guard nobody
exercises is indistinguishable from a guard that was deleted**, which is the same lesson
`test_stage_coverage.py` records about unwired stages.

WHAT IS LOCKED HERE, and why each one rather than "the script works":

1. **Guard ORDER, not just presence.** #106's fix is that `...` is checked *before* `..`.
   A test that only asserted "three dots are refused" would pass against a reordered file
   that no longer refuses them, because the two-dot branch would catch it with the wrong
   message. So the three-dot message is asserted specifically.
2. **The permission flags**, as text. `--permission-mode auto` classified `python -c` as
   safe, so with it present no allowlist could make the reviewer read-only (#100, owner
   decision 2026-08-15). Its return would be silent and total.
3. **Both spellings of every value-taking git global option** (#101). `Bash(git --git-dir:*)`
   does NOT match the token `--git-dir=/path` — measured — so an entry can name an option and
   still miss it, which reads as covered.
4. **The absence of a `Bash(git:*)` wildcard** in the allowlist (#92). It pre-approved
   `git push`, `commit` and `reset` for the one role that must never write to `main`.

HERMETIC BY CONSTRUCTION: the behavioural tests stub `claude` onto PATH and point `HOME` at
an empty directory. The stub is never executed — every case exits at a guard or at
`--dry-run`, both of which precede the launch — but the script checks for the binary early
(fail fast, correctly), so its absence would otherwise skip the whole file on any machine
without Claude Code installed. The empty `HOME` makes the tracker lookup fail closed to
"unreachable", which the script warns about and continues past; that keeps these tests off
the network and independent of whether PocketBase is up.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "request_review.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# behavioural — the guards, run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Invoke the launcher with a stub `claude` and an empty HOME. Returns (rc, out)."""
    box = tmp_path_factory.mktemp("reqrev")
    bindir = box / "bin"
    bindir.mkdir()
    stub = bindir / "claude"
    # Exits non-zero: if a change ever lets execution reach the launch, the test that
    # expected a guard fails loudly instead of silently starting a real review.
    stub.write_text("#!/bin/sh\necho 'STUB CLAUDE WAS INVOKED' >&2\nexit 97\n")
    stub.chmod(0o755)
    home = box / "home"
    home.mkdir()

    tmp = box / "tmp"
    tmp.mkdir()

    def _run(*args):
        env = dict(os.environ)
        env["PATH"] = f"{bindir}:{env['PATH']}"
        env["HOME"] = str(home)
        # ⚠ TMPDIR IS PINNED, and the previous version's assertion anchored on a literal
        # "/tmp/" instead. `mktemp -t` follows $TMPDIR, which the fixture inherits — so on
        # macOS, or any CI that sets TMPDIR, the credential path would not contain "/tmp/"
        # and the #94 assertion would pass against the bug again. Controlling it here makes
        # the check independent of the host.
        env["TMPDIR"] = str(tmp)
        p = subprocess.run(
            [str(SCRIPT), *args],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
        )
        return p.returncode, p.stdout + p.stderr

    _run.tmpdir = tmp
    return _run


def test_a_bare_ref_is_refused(run):
    """#99: rev-list reads it as all history, diff --stat as the working tree."""
    rc, out = run("--range", "HEAD", "--dry-run")
    assert rc != 0, out
    assert "two-dot" in out


def test_a_three_dot_range_is_refused_by_the_three_dot_guard(run):
    """#106, and the message matters: `...` contains `..`, so ORDER is the fix.

    Asserting on the three-dot message rather than merely on a non-zero exit is what makes
    this test able to fail if the two checks are ever swapped back.
    """
    rc, out = run("--range", "origin/main...HEAD", "--dry-run")
    assert rc != 0, out
    assert "has three" in out, "caught by the wrong guard — check the ordering"


def test_the_three_dot_check_precedes_the_two_dot_check_in_source():
    """Belt-and-braces on the ordering, independent of any range resolving on this machine."""
    three = SOURCE.index('== *"..."*')
    two = SOURCE.index('== *".."*')
    assert three < two, "the two-dot test must not run first: `...` contains `..`"


def test_a_two_dot_range_is_accepted(run):
    rc, out = run("--range", "HEAD~1..HEAD", "--review-id", "test-range-ok", "--dry-run")
    assert rc == 0, out
    assert "STUB CLAUDE WAS INVOKED" not in out, "--dry-run must not launch a review"


def test_four_reviews_are_allowed_and_five_are_not(run):
    """The cap is THREE FIX PASSES, which need FOUR reviews (owner, 2026-08-15).

    An earlier version refused `--pass 4` — it had miscounted reviews for fix passes, and
    would have blocked the review that verifies the final fix, which is the one that decides
    whether anything gets escalated at all.
    """
    ok, out = run("--range", "HEAD~1..HEAD", "--review-id", "t4",
                  "--pass", "4", "--prior", "x", "--notes", "n", "--dry-run")
    assert ok == 0, out
    bad, out5 = run("--range", "HEAD~1..HEAD", "--review-id", "t5",
                    "--pass", "5", "--prior", "x", "--notes", "n", "--dry-run")
    assert bad != 0, out5


def test_a_later_pass_requires_the_prior_review_ids(run):
    """A one-shot reviewer that cannot find its own findings re-derives them as new issues."""
    rc, out = run("--range", "HEAD~1..HEAD", "--review-id", "t2", "--pass", "2", "--dry-run")
    assert rc != 0, out
    assert "--prior" in out


def test_dry_run_names_no_concrete_credential_path(run):
    """#94, asserted so that it can actually FAIL against #94.

    ⚠ The first version of this test globbed /tmp for `pb-mcp-*.json` before and after. That
    cannot detect the bug its own docstring described: #94 was *create the file, print its
    path, delete it on the EXIT trap*, and the trap fires before `subprocess.run` returns —
    so the before/after sets match either way. It passed against the bug and against the fix
    equally, which is the worst state a regression test can be in.

    What survives the trap is the OUTPUT. The pre-fix version printed a real
    `/tmp/pb-mcp-XXXXXX.json` path in the command it advertised as runnable — and with
    `--strict-mcp-config`, running that named a file that no longer existed, starting a
    reviewer with no tracker access at all. So: assert no concrete path is printed. The fixed
    script prints a placeholder describing when the file is written instead.
    """
    rc, out = run("--range", "HEAD~1..HEAD", "--review-id", "t-cred", "--dry-run")
    assert rc == 0, out
    # Not anchored on "/tmp/": mktemp -t follows $TMPDIR (#117). The fixture pins TMPDIR, and
    # the pattern matches the template wherever it lands.
    assert not re.search(r"pb-mcp-\S*\.json", out), (
        "--dry-run printed a concrete credential path; that file is deleted on exit, so the "
        "command it advertises is not runnable (#94)"
    )
    assert "--mcp-config" in out, "the dry run should still show that a config is passed"
    # ⚠ A glob for "pb-mcp-t-cred*" used to sit here. It could NEVER match — the template is
    # `pb-mcp-XXXXXX.json` and the review id never enters the filename — so it was a
    # tautology, added four lines under a docstring calling tautologies the worst state a
    # regression test can be in (#116). Replaced with the real property, against the TMPDIR
    # this fixture controls.
    assert not list(run.tmpdir.glob("pb-mcp-*")), "a dry run must leave no credential file"


# --------------------------------------------------------------------------- #
# static — the flags the reviewer is launched with
# --------------------------------------------------------------------------- #
def _array(name, source=None):
    """Every quoted entry of a bash array literal.

    ⚠ Per LINE, not per entry: these arrays pack several entries on one line, and a
    first version of this helper took only `line.split('"')[1]`. It reported 16 false
    failures against a correct script — a parser bug arriving as a wall of red about
    security flags, which is exactly the shape that gets a real finding dismissed.
    """
    src = SOURCE if source is None else source
    start = src.index(f"{name}=(")
    end = src.index("\n)", start)
    entries = []
    for line in src[start:end].splitlines():
        # ⚠ COMMENTS ARE DROPPED FIRST. `re.findall` over the raw slice takes every quoted
        # run in the block, including inside a `#` comment — and every assertion built on
        # this helper is a membership test, so an entry named only in a comment would
        # satisfy it. That is correct today purely because the comments here quote entry
        # names in backticks, which is a style convention holding up a security check.
        # It fails in the dangerous direction: delete a real entry, mention it in a comment,
        # and the suite stays green.
        code = line.split("#", 1)[0]
        entries.extend(re.findall(r'"([^"]+)"', code))
    return entries


def test_auto_permission_mode_is_not_used():
    """#100 (owner decision): the classifier judged `python -c` safe, so with it on no
    allowlist could make the reviewer read-only. Its return would be silent and total."""
    # ⚠ Slice from the REAL invocation, not the first "claude -p " in the file — the
    # --dry-run block prints a mock command earlier, and its explanatory text mentions
    # the flag by name. A first version matched that and failed against a correct script.
    launched = SOURCE[SOURCE.index('claude -p "$PROMPT"'):]
    assert "--permission-mode" not in launched


def test_the_allowlist_has_no_git_wildcard():
    """#92: `Bash(git:*)` pre-approved push, commit and reset for the one role that must
    never write to main."""
    allow = _array("REVIEWER_ALLOW")
    assert "Bash(git:*)" not in allow
    assert not any(a.startswith("Bash(git)") or a == "Bash(git *)" for a in allow)


def test_no_editing_tool_is_available_or_allowed():
    assert "Edit" not in SOURCE[SOURCE.index("REVIEWER_TOOLS="):].split("\n")[0]
    deny = _array("REVIEWER_DENY")
    for t in ("Edit", "Write", "NotebookEdit"):
        assert t in deny


@pytest.mark.parametrize(
    "opt", ["--git-dir", "--work-tree", "--namespace", "--config-env", "--exec-path"]
)
def test_value_taking_git_options_are_denied_in_both_spellings(opt):
    """#101, and the sharp half: `Bash(git --git-dir:*)` was ALREADY in the deny list and did
    NOT match `git --git-dir=/path`. The matcher tokenises on whitespace, so an entry can name
    an option and still miss it — which is worse than an absent entry, because it reads as
    covered."""
    deny = _array("REVIEWER_DENY")
    assert f"Bash(git {opt}:*)" in deny
    assert f"Bash(git {opt}=:*)" in deny


@pytest.mark.parametrize("opt", ["-c", "-C", "-p", "-P", "--no-pager", "--bare"])
def test_bare_git_global_options_are_denied(opt):
    """`git -c core.pager=cat tag -l` escaped `Bash(git tag:*)` — measured. `git -c … <verb>`
    is not exotic here: it is the spelling DEVELOPER.md mandates for every commit."""
    assert f"Bash(git {opt}:*)" in _array("REVIEWER_DENY")


@pytest.mark.parametrize("verb", ["push", "commit", "reset", "checkout", "rebase", "clean"])
def test_writing_git_verbs_are_denied(verb):
    assert f"Bash(git {verb}:*)" in _array("REVIEWER_DENY")


def test_the_script_is_classified_in_the_pipeline_manifest():
    """#90: it is a tracked shell under scripts/, so test_stage_coverage requires a decision
    about what it is. Asserted here too so this file fails for its own reason."""
    manifest = (REPO / "scripts" / "pipeline_manifest.py").read_text(encoding="utf-8")
    assert "scripts/request_review.sh" in manifest


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_the_array_parser_ignores_entries_that_appear_only_in_comments():
    """#118: the comment-stripping in `_array` was itself unpinned.

    Reverting it left all 29 tests green, because no comment inside either array happens to
    contain a double-quoted run — so the protection rested on a style convention, and a hand
    mutation is not a regression test. This pins the parser directly, against a synthetic
    block, so the guard survives someone tidying the helper.

    The failure it prevents is the dangerous direction: delete a real deny entry, mention it
    in a comment, and every membership assertion keeps passing.
    """
    synthetic = (
        'REVIEWER_DENY=(\n'
        '  "Bash(git push:*)"\n'
        '  # "Bash(git commit:*)" — named here only, must NOT count as an entry\n'
        '  "Bash(git reset:*)"  # trailing comment with "Bash(git checkout:*)" in it\n'
        ')\n'
    )
    got = _array("REVIEWER_DENY", source=synthetic)
    assert got == ["Bash(git push:*)", "Bash(git reset:*)"], got
    assert "Bash(git commit:*)" not in got, "a commented-out entry was counted as present"
    assert "Bash(git checkout:*)" not in got, "a trailing-comment entry was counted as present"

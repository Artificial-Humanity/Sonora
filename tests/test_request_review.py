"""The review launcher's guards, which have twice been defeated with the suite green (#110).

`FerroStep/workflow/scripts/request_review.sh` decides what a reviewer is told and what it is allowed to run.
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

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "FerroStep" / "workflow" / "scripts" / "request_review.sh"
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
    rc, out = run("--range", "HEAD~1..HEAD", "--dry-run")
    assert rc == 0, out
    assert "STUB CLAUDE WAS INVOKED" not in out, "--dry-run must not launch a review"


def test_four_reviews_are_allowed_and_five_are_not(run):
    """The cap counts FIX PASSES, which need ceiling-plus-one reviews (owner, 2026-08-15).
    Asserted at the SHIPPED ceiling (the definition's agent_passes.max = 3, so reviews cap
    at 4); the script derives that sum, and this test runs against the live definition.

    An earlier version refused `--pass 4` — it had miscounted reviews for fix passes, and
    would have blocked the review that verifies the final fix, which is the one that decides
    whether anything gets escalated at all.
    """
    ok, out = run("--range", "HEAD~1..HEAD",                   "--pass", "4", "--notes", "n", "--dry-run")
    assert ok == 0, out
    bad, out5 = run("--range", "HEAD~1..HEAD",                     "--pass", "5", "--notes", "n", "--dry-run")
    assert bad != 0, out5


def test_the_brief_tells_the_reviewer_which_branch_to_stamp(run):
    """Replaces the old --prior contract, which the branch made unnecessary.

    A one-shot reviewer used to be handed the previous passes' ids so it could find its own
    earlier findings. With the BRANCH as the unit of work that threading is gone: the reviewer
    queries `branch_name="<branch>" && state="open"` and gets exactly the right set, whichever
    pass filed each one. What must still hold is that the brief NAMES the branch — an
    unstamped issue belongs to no unit and appears in no convergence check.
    """
    rc, out = run("--range", "HEAD~1..HEAD", "--dry-run")
    assert rc == 0, out
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO),
                            capture_output=True, text=True).stdout.strip()
    assert branch in out, "the brief must name the branch the reviewer stamps issues with"
    assert "branch_name" in out


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
    rc, out = run("--range", "HEAD~1..HEAD", "--dry-run")
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
    # ⚠ RENDERED, NOT `_array` (#249). `_array` reads only the array literal and cannot see a
    # `+=` append — 35 entries of the 40 the matcher receives. A wildcard added by an append
    # was invisible to this guard, which is the one that keeps push/commit/reset away from the
    # role that must never write to main.
    allow = _rendered_allowlist()
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
    assert "FerroStep/workflow/scripts/request_review.sh" in manifest


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


def test_the_launcher_grant_is_scoped_to_dry_run():
    """#119: the #115 fix was correct and unpinned, which is this cycle's most repeated shape.

    `Bash(./FerroStep/workflow/scripts/request_review.sh:*)` — the whole script — was granted while the comment
    beside it justified only `--dry-run`. Without that flag the script launches a real nested
    `claude -p`, files issues under a branch_name nobody watches, and the nested reviewer holds
    the same entry: unbounded recursion, billed, with a credential file at every level.

    ⚠ The fix landed with nothing holding it: `_array("REVIEWER_ALLOW")` had exactly one
    caller, looking only for `Bash(git:*)`, so reverting to the whole-script entry left all
    1094 tests green. Four fixes in this cycle have now been correct-but-unpinned (#110, #118,
    #119, and the #80 shape) — and this is the first where the unpinned guard protected a
    CAPABILITY rather than the accuracy of a sentence.

    ⚠⚠ AND IT WAS STILL ON THE BLIND INSTRUMENT UNTIL #249. #245 built
    `_rendered_allowlist()` — which parses what the matcher actually receives — and used it for
    the two NEW tests while leaving this one, and the git-wildcard guard, on `_array`. So a
    whole-script launcher grant introduced as a `+=` append would have been invisible to the
    guard written to prevent exactly that, in the same file, thirty lines from the instrument
    that would have caught it.
    """
    launcher = [a for a in _rendered_allowlist() if "request_review.sh" in a]
    assert launcher, "the reviewer needs to be able to dry-run the launcher it reviews"
    for entry in launcher:
        assert "--dry-run" in entry, (
            f"{entry!r} grants the whole launcher. Without --dry-run it starts a real nested "
            "review; scope every launcher entry to the flag."
        )


def test_the_sibling_repos_are_offered_read_only_and_never_writable():
    """The reviewer can READ its sibling checkouts, because Sonora describes mechanisms
    implemented there — and since 2026-08-27 routes findings to a repo it must be able to check.

    A review had to file #114 as "verified, direction undetermined" — it could not tell whether
    the PocketBase hook behind `user_decision` exists, because it lives in the sibling repo and
    listing outside the working directory is refused. That is the correct behaviour from the
    reviewer and a worse outcome than letting it look.

    ⚠ Pinned per #119's lesson: `--add-dir` widens what the file tools reach, so if
    Edit/Write ever returned to `--tools`, this would become write access to a second repo.
    The two properties are asserted together deliberately.
    """
    assert "--add-dir" in SOURCE, "the reviewer should be able to read the sibling repos"
    # ⚠ PLURAL SINCE #347. This pinned the single-value `ADD_DIR_ARGS=(--add-dir "$SIBLING")`,
    # which was correct while the resolver granted exactly one — it `break`s on the first
    # candidate that exists, and the two configured entries were the same repo by two paths, so
    # nothing looked wrong. It became wrong when REVIEWER.md started routing findings to
    # FerroStep and telling the reviewer to VERIFY the boundary: the instruction named a repo
    # the launcher could not grant.
    #
    # ⚠ The security property this test exists for is UNCHANGED and is the reason it is
    # asserted per-argument rather than loosened to "--add-dir appears somewhere": every
    # granted path must arrive through `--add-dir`, which widens READS only. `--add-dir` on N
    # repos is still read-only on N repos; it is the Edit/Write pairing that would make it
    # write access, and that is asserted below.
    assert "ADD_DIR_ARGS+=(--add-dir " in SOURCE, (
        "the launcher no longer accumulates --add-dir per sibling; a single-value form grants "
        "only the first candidate and silently drops the rest (#347)")
    # ⚠ THE CANDIDATE PATHS MOVED TO config.env ON 2026-08-17, and this assertion moved with
    # them. It pinned the literal `$REPO_ROOT/../../AI-Lab-AMD` — a fact about this lab's disk
    # layout rather than about the launcher, and exactly the sort of thing that must not
    # survive `FerroStep/workflow/` being copied into another repo. What the script must still do is READ
    # the list from config; which paths are in it is the port's business.
    assert "SIBLING_REPO_CANDIDATES" in SOURCE, \
        "the sibling paths must come from config.env, not from this script"
    cfg = (REPO / "FerroStep" / "workflow" / "config.env").read_text(encoding="utf-8")
    assert "SIBLING_REPO_CANDIDATES=" in cfg, "config.env must declare the setting, even if empty"
    # And read-only: no editing tool exists, and all three are denied by name.
    tools = SOURCE[SOURCE.index("REVIEWER_TOOLS="):].split("\n")[0]
    for t in ("Edit", "Write", "NotebookEdit"):
        assert t not in tools, f"{t} in --tools would make --add-dir a write grant"
        assert t in _array("REVIEWER_DENY")


def _rendered_allowlist(script=None):
    """The allowlist AS THE MATCHER WILL SEE IT — parsed from `--dry-run`, not from source.

    ⚠⚠ THIS REPLACED A SOURCE SCAN THAT WAS BLIND TO EVERY GATE PATH (#245). The scan read the
    array literal plus `REVIEWER_ALLOW+=("...")` appends by regex. The same commit that fixed
    #239 moved the gate paths into a `for _g in ...` word list, so the only text the regex
    could reach was the append's literal `Bash($PYBIN $_g:*)` — an unexpanded shell variable.
    No gate path was visible to it at all, and reverting the loop list to the #239 defect left
    the test GREEN.

    ⚠ THE LESSON IS NOT "ADD THE LOOP TO THE REGEX". A scanner has to model shell expansion and
    will be wrong again at the next restructure — that is twice inside one commit already
    (`_array` missed appends; the regex then missed the loop). `--dry-run` renders the array
    through the same code path the real call uses, so there is nothing left to model. It is the
    precedent `tests/test_review_cycle.py` set for the same reason (#231).
    """
    # ⚠⚠ `--full`, NOT A RANGE, AND THAT IS THE WHOLE OF THIS LINE'S HISTORY. The first
    # version passed `--range origin/main..HEAD`, which is EMPTY on `main` — the launcher
    # refuses an empty range, so all five tests that call this helper went red the moment the
    # branch merged. They had only ever been run ON the branch, where the range is non-empty.
    #
    # ⚠ THAT IS THE SAME CLASS AS THE DEFECTS THEY EXIST TO CATCH: green for a reason
    # unrelated to what is being asserted — here, the repo happening to be in a particular
    # state. `--full` needs no commits ahead of main (the launcher says so in the very refusal
    # this produced), and `REVIEWER_ALLOW` is built from PYBIN and the gates directory with no
    # reference to RANGE, so the rendered list is identical either way.
    # `script` lets a COPIED lane be rendered through the same parser (the negative-branch
    # test below). One parser, two callers: a second ad-hoc scan is how #245 happened.
    r = subprocess.run([str(script or SCRIPT), "--full", "--dry-run", "--developer", "Ozzy"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, f"--dry-run failed, so this test proves nothing: {r.stderr}"
    for line in r.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("--allowedTools"):
            body = stripped.removeprefix("--allowedTools").rstrip()
            return shlex.split(body.removesuffix("\\"))
    raise AssertionError("--dry-run printed no --allowedTools line")


def test_no_allow_entry_ends_mid_token():
    """⚠ #239: `Bash($PYBIN scripts/gates/:*)` could never match, because the real token is
    `scripts/gates/test_doc_claims.py` and the matcher tokenises on whitespace. The gate was
    refused for the entire life of the branch that granted it.

    ⚠ THIS REPO ALREADY PINNED THE RULE FROM THE DENY SIDE — see
    `test_value_taking_git_options_are_denied_in_both_spellings`: "an entry can name an option
    and still miss it — which is worse than an absent entry, because it reads as covered."
    The allow side had no equivalent, so the same mistake was available and I made it.

    A trailing `/` is the checkable case: a path fragment is definitionally mid-token. This
    does not catch every mid-token prefix — `--outp` would pass — and the docstring says so
    rather than implying coverage it does not have.
    """
    bad = [e for e in _rendered_allowlist()
           if e.startswith("Bash(") and e.removesuffix(":*)").endswith("/")]
    assert not bad, (
        "these allow entries end mid-token and can never match a real command; name the file "
        f"rather than the directory: {bad}")


def test_every_gate_on_disk_is_granted():
    """⚠⚠ #247: the grant named FOUR gates while `scripts/gates/` held SIX.

    `test_film_export_gate.py` and `test_vat_identity.py` were refused — measured by the
    reviewer, mid-review — while `REVIEWER.md` §1 told that same reviewer the whole directory
    was reachable. A hand-kept list beside a directory goes stale the moment a gate is added,
    and this one was stale on the commit that wrote it.

    ⚠ DERIVED FROM DISK ON BOTH SIDES. Asserting a list of six names here would be the same
    defect one layer up: the next gate added would leave the grant and this test agreeing with
    each other and disagreeing with the directory.
    """
    on_disk = sorted(p.name for p in (REPO / "scripts" / "gates").glob("test_*.py"))
    assert on_disk, "no gates found — this test would pass vacuously"
    granted = _rendered_allowlist()
    missing = [g for g in on_disk if not any(g in e for e in granted)]
    assert not missing, (
        "these gates exist on disk but are not in the rendered allowlist, so the reviewer is "
        f"refused when it runs them: {missing}")


def test_the_gate_entries_reach_the_rendered_allowlist():
    """The control for the test above — prove the data it inspects is actually there.

    ⚠⚠ WITHOUT THIS, AN EMPTY PARSE PASSES FOREVER. The test above asserts a NEGATIVE, so a
    `_rendered_allowlist()` returning `[]` — a renamed flag, a changed print format, a moved
    line — satisfies it silently. That is precisely how #245 survived: a green negative
    evaluated over data that was not in the string being searched.
    """
    entries = _rendered_allowlist()
    gates = [e for e in entries if "scripts/gates/" in e]
    assert gates, f"no gate entry in the rendered allowlist at all: {entries}"
    for e in gates:
        assert e.endswith(".py:*)"), f"a gate entry does not name a file: {e}"


# --- the self-check gate (owner, 2026-08-27) ----------------------------------------------

def _self_review(at, pass_idx, max_reviews=4):
    """Drive the shipped evaluator itself — extracted from the launcher, not reimplemented.

    ⚠ A second copy of the resolution rules in this file would be the thing they exist to
    prevent: a test that agrees with a rule it restated, while the script does something else.
    """
    fn = SOURCE[SOURCE.index("self_review_scheduled() {"):SOURCE.index("\nif self_review_scheduled")]
    script = (f'die() {{ echo "$*" >&2; exit 9; }}\nMAX_REVIEWS={max_reviews}\n'
              f'SELF_REVIEW_AT={at!r}\n{fn}\n'
              f'if self_review_scheduled {pass_idx}; then echo DUE; else echo SKIP; fi\n')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("at,idx,want", [
    ("none", 1, "SKIP"), ("", 1, "SKIP"),
    ("first", 1, "DUE"), ("first", 2, "SKIP"),
    ("all", 1, "DUE"), ("all", 4, "DUE"),
    ("1,4", 1, "DUE"), ("1,4", 4, "DUE"), ("1,4", 2, "SKIP"),
])
def test_the_self_check_schedule_resolves_to_a_set_of_review_indices(at, idx, want):
    """`first` and `all` are spellings of sets, not separate code paths."""
    r = _self_review(at, idx)
    assert r.returncode == 0, r.stderr
    assert want in r.stdout, f"SELF_REVIEW_AT={at!r} at review {idx}: {r.stdout} {r.stderr}"


@pytest.mark.parametrize("bad", ["frist", "First ", "1;4", "yes", "true", "0", "1,0"])
def test_an_unrecognised_schedule_DIES_and_never_falls_back_to_none(bad):
    """⚠⚠ A TYPO RESOLVING SILENTLY TO "NEVER" gives this lane a self-check that is
    configured, documented and never runs — and **a check that never fires is
    indistinguishable from one that ran clean.** There is no conservative default worth having,
    so the value is quoted back and the run stops."""
    r = _self_review(bad, 1)
    assert r.returncode != 0, (
        f"SELF_REVIEW_AT={bad!r} was accepted; a bad value must refuse, not resolve to off: "
        f"{r.stdout}")


def test_an_index_above_the_derived_ceiling_DIES_naming_both_numbers():
    """⚠ THE ONE MOST LIKELY TO BE TYPED. `SELF_REVIEW_AT=5` under a 3-fix-pass definition
    reads as ON in the config file and is OFF in every run. The refusal names the index AND the
    ceiling so it teaches the arithmetic rather than just rejecting."""
    r = _self_review("1,5", 1, max_reviews=4)
    assert r.returncode != 0, "an unreachable index was accepted: " + r.stdout
    assert "5" in r.stderr and "4" in r.stderr, (
        "the refusal must name both the offending index and the derived ceiling: " + r.stderr)


def test_a_validation_error_fires_even_when_that_index_is_not_the_current_one():
    """⚠ A SETTING IS CHECKED WHEN IT IS READ, NOT WHEN IT HAPPENS TO FIRE. Validating only the
    matching token would let `1,99` read as valid for the whole of review 1 and die at review 2
    — a configuration error surfacing as a mid-cycle failure."""
    r = _self_review("1,99", 1, max_reviews=4)
    assert r.returncode != 0, "99 was not validated because review 1 matched first: " + r.stdout


def test_out_of_range_is_not_the_same_as_never_reached():
    """⚠ `1,4` stays LEGAL when a cycle converges at review 2 — index 4 simply does not come
    up. That is a lane finishing early, not a misconfiguration, and it must stay silent."""
    r = _self_review("1,4", 2, max_reviews=4)
    assert r.returncode == 0, "a legal schedule was refused at a review it does not name: " + r.stderr
    assert "SKIP" in r.stdout


def test_a_dry_run_reports_the_self_check_and_does_not_run_it():
    """⚠⚠ `--dry-run` is documented as spending nothing, and it is the ONE launcher form the
    REVIEWER may invoke — its allowlist entry is scoped to the flag precisely because the flag
    "files nothing, launches nothing, and writes no credential file". Executing an
    operator-supplied `SELF_REVIEW_CMD` there would make that false and hand the reviewer a way
    to run it."""
    code = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
    gate = code[code.index("if self_review_scheduled"):]
    gate = gate[:gate.index("\nfi\n")]
    assert 'DRY_RUN" -eq 1' in gate, (
        "the self-check gate does not special-case --dry-run, so a dry run executes "
        "SELF_REVIEW_CMD:\n" + gate)
    assert gate.index('DRY_RUN" -eq 1') < gate.index("eval"), (
        "the dry-run branch must come before the eval, or it cannot prevent it")


def _run_gate(cmd, at="all", pass_idx=1, dry=0):
    """Drive the SHIPPED gate block — the enforcing half, not the scheduler.

    ⚠ Extracted from the launcher rather than reimplemented, for the same reason
    `_self_review` is: a test that restates the rule agrees with itself while the script does
    something else.
    """
    fn = SOURCE[SOURCE.index("self_review_scheduled() {"):SOURCE.index("\nif self_review_scheduled")]
    gate = SOURCE[SOURCE.index("if self_review_scheduled"):]
    gate = gate[:gate.index("\nfi\n") + 4]
    script = (f'die() {{ echo "$*" >&2; exit 9; }}\nMAX_REVIEWS=4\nDRY_RUN={dry}\n'
              f'SELF_REVIEW_AT={at!r}\nSELF_REVIEW_CMD={cmd!r}\nPASS={pass_idx}\n'
              f'{fn}\n{gate}\necho REACHED_THE_REVIEW\n')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_a_failing_self_check_command_REFUSES_the_review():
    """⚠⚠ THE VERIFIED HALF WAS WATCHED BY NOTHING (#351). The scheduling half had 22 cases;
    the part the whole design rests on had none, and mutation-measured, replacing the `die`
    with `|| true` left every assertion green.

    `SELF_REVIEW_CMD` is the half that is *enforced* rather than *prompted* — the commit that
    added it says so as the design. A gate that cannot verify must not pretend to; one that
    verifies and then proceeds anyway is worse, because its output says it checked.
    """
    r = _run_gate("false")
    assert r.returncode != 0, (
        "a failing self-check command did not stop the run:\n" + r.stdout + r.stderr)
    assert "REACHED_THE_REVIEW" not in r.stdout, (
        "the review was requested anyway — the die() does not prevent the launch")
    assert "self-check command failed" in r.stderr, r.stderr


def test_a_passing_self_check_command_lets_the_review_proceed():
    """The other direction, without which the test above is satisfied by a gate that refuses
    everything."""
    r = _run_gate("true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REACHED_THE_REVIEW" in r.stdout, "a passing self-check must not block the review"


def test_a_self_check_command_that_calls_exit_does_not_silently_end_the_script():
    """⚠ THE SUBSHELL IS NOT DECORATION. The value is operator-supplied and `exit 1` is a
    plausible thing to type into a "make this fail" setting. Without `( … )` the eval would
    exit THE LAUNCHER at that point — no die, no message, and a zero exit status reading as a
    review that ran clean."""
    r = _run_gate("exit 1")
    assert r.returncode != 0, r.stdout + r.stderr
    assert "self-check command failed" in r.stderr, (
        "an `exit` inside SELF_REVIEW_CMD ended the script instead of failing the check — "
        "the eval is not in a subshell:\n" + r.stdout + r.stderr)
    assert "REACHED_THE_REVIEW" not in r.stdout


def test_no_self_check_command_means_the_checklist_only():
    """An empty SELF_REVIEW_CMD is the documented "no mechanical gate" case and must not be
    read as a command that failed."""
    r = _run_gate("")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REACHED_THE_REVIEW" in r.stdout
    assert "self-check list is DEVELOPER.md" in r.stderr


def test_a_dry_run_does_not_execute_the_command_even_when_it_would_fail():
    """⚠⚠ BEHAVIOURAL, not a source scan. `--dry-run` is the ONE launcher form the reviewer's
    allowlist grants; if it executed SELF_REVIEW_CMD, the reviewer would gain a way to run an
    operator-supplied command. A failing command must therefore NOT stop a dry run."""
    r = _run_gate("false", dry=1)
    assert r.returncode == 0, (
        "a dry run executed the self-check command and was stopped by it:\n"
        + r.stdout + r.stderr)
    assert "WOULD run" in r.stderr, r.stderr
    assert "REACHED_THE_REVIEW" in r.stdout


def test_the_reviewer_launches_with_its_roster_entry_s_model_and_effort(run):
    """Owner, 2026-09-02: the model and effort come from the reviewer's entry in FerroStep/config.yaml.
    The expected values are READ from the roster here, not typed — a literal in this test
    would be the second copy the arrangement exists to remove."""
    import yaml
    entry = yaml.safe_load(open(os.path.join(REPO, "FerroStep", "config.yaml")))["agents"]["reviewer"]
    rc, out = run("--range", "HEAD~1..HEAD", "--dry-run")
    assert rc == 0, out
    assert f"--model {entry['model']} --effort {entry['effort']}" in out, out


def test_a_flag_still_overrides_the_roster(run):
    rc, out = run("--range", "HEAD~1..HEAD", "--dry-run", "--model", "some-other", "--effort", "low")
    assert rc == 0, out
    assert "--model some-other --effort low" in out, out


# --- the promotion-step grant (owner, 2026-09-10) -----------------------------------------

def _config_env_value(key):
    """One `KEY=value` out of config.env, which is plain by contract (its own header says so)."""
    for line in (REPO / "FerroStep" / "workflow" / "config.env").read_text(
            encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _run_sh_default_interpreter():
    """The interpreter `scripts/litert_export/run.sh` composes, derived from that file.

    ⚠ COMPOSED FROM ITS TWO LINES, NEVER TYPED HERE. Writing the joined path in would add one
    more place for it to live, and the one that agrees with none of the others.

    ⚠ NO COUNT IS STATED HERE, DELIBERATELY, AND IT USED TO BE (#434). This said "the THIRD
    copy" while its own caller said "a fourth copy" seven lines below — two numbers disagreeing
    inside the docstrings of the pin that exists to keep these places in step, and the commit
    that corrected the count everywhere else left this one behind. Correcting the number would
    have re-armed the same trap: nothing can fail when a number in prose goes stale, and
    AGENTS.md §5b says to derive counts rather than state them. `_INTERPRETER_COPIES` below is
    the enumeration; read the length off it if you need one.
    """
    src = (REPO / "scripts" / "litert_export" / "run.sh").read_text(encoding="utf-8")
    work = re.search(r'SONORA_LITERT_WORK:-([^}"]+)', src)
    py = re.search(r'SONORA_LITERT_PY:-\$\{SONORA_LITERT_WORK\}([^}"]+)\}', src)
    assert work and py, "run.sh no longer composes its interpreter the way this test reads it"
    return work.group(1) + py.group(1)


def _check_publishable_docstring_interpreter():
    """The path spelled out in the promoter's pasteable command, read from its docstring."""
    src = (REPO / "scripts" / "tools" / "check_publishable.py").read_text(encoding="utf-8")
    m = re.search(r'SONORA_LITERT_PY:-([^}"]+)\}', src)
    assert m, "check_publishable.py no longer spells the interpreter the way this test reads it"
    return m.group(1)


# ⚠ THE ENUMERATION, not a sentence with a number in it (#434). Keyed by PATH so the
# completeness test below can compare it against what is actually on disk. `run.sh` is
# deliberately absent: it OWNS the value and composes it, so it is what the others are compared
# against rather than one of them — and it does not contain the joined literal at all.
_INTERPRETER_COPIES = {
    "FerroStep/workflow/config.env":
        ("REVIEWER_TORCH_PY", lambda: _config_env_value("REVIEWER_TORCH_PY")),
    "scripts/tools/check_publishable.py":
        ("the docstring command", _check_publishable_docstring_interpreter),
}


def _tracked_files_naming_the_interpreter():
    """Every tracked file that contains the composed interpreter path as a literal string."""
    r = subprocess.run(["git", "grep", "-l", "-F", _run_sh_default_interpreter()],
                       cwd=REPO, capture_output=True, text=True)
    # ⚠ `git grep` exits 1 for "no matches", which is not a failure here — but 2+ is, and
    # treating every non-zero as "nothing found" is the instrument-failure-read-as-a-negative
    # shape AGENTS.md §5b tabulates.
    assert r.returncode in (0, 1), f"git grep failed ({r.returncode}): {r.stderr}"
    return {line for line in r.stdout.split() if line}


def test_every_literal_copy_of_the_interpreter_is_enrolled_in_the_pin():
    """⚠⚠ THE PIN CHECKS WHAT IT IS TOLD ABOUT, SO THE ENUMERATION HAS TO BE COMPLETE (#434).

    Three passes of this issue were spent on prose that said how many places hold this path —
    "second copy", "THREE PLACES", "ALL THREE" — each correct when written and none of them able
    to fail afterwards. A number in a sentence is not a mechanism; this is. With the enumeration
    provably complete, no sentence needs to state a size, which is why they are all gone.

    Same shape as `test_doc_claims_registry`'s "a fact no document states is a fact nobody is
    checking", and the same remedy: assert on the registry rather than re-running a sweep by
    hand. The sweeps are what failed — three of them, each keyed on the previous wording rather
    than on the claim.

    ⚠ A HIT IN THIS TEST FILE IS NOT FIXED BY ENROLLING IT. The rule is that the path is never
    typed here; it is composed from `run.sh`. Delete it instead.
    """
    found = _tracked_files_naming_the_interpreter()
    assert found, (
        "no tracked file contains the interpreter literal, so this test would pass over "
        "nothing — either the scan broke or run.sh stopped composing what the copies spell")
    missing = sorted(found - set(_INTERPRETER_COPIES))
    assert not missing, (
        "these tracked files spell out the interpreter path but are not in "
        f"_INTERPRETER_COPIES, so nothing pins them to run.sh: {missing}. Add each one (with a "
        "reader), or delete the literal if the file should be composing it instead")


def test_the_promotion_interpreter_matches_the_export_lane_default():
    """⚠⚠ EVERY PLACE THE PATH IS SPELLED OUT, PINNED TO THE ONE THAT OWNS IT (#434).

    `scripts/litert_export/run.sh` owns the default and composes it. The places in
    `_INTERPRETER_COPIES` spell it out literally, each for a reason: `config.env` so the
    reviewer's allowlist can name it, `check_publishable.py`'s docstring so the promoter's
    command is pasteable. Each of them once described itself as one half of a pair with
    `run.sh` and neither mentioned the other, so "change both" reached some of them and left
    the rest naming an interpreter the export lane no longer used.

    ⚠ ADD A NEW PLACE TO `_INTERPRETER_COPIES`, NOT A SENTENCE SAYING HOW MANY THERE ARE. The
    count is derived from that dict wherever one is needed; §5b's rule is that a number in
    prose goes stale with nothing able to fail, and #434's residual was exactly that.

    ⚠ COMPOSED FROM run.sh's OWN TWO LINES, NEVER TYPED HERE — that would add one more place,
    and the one that agrees with none of the others.
    """
    want = _run_sh_default_interpreter()
    places = {f"{path} ({what})": read()
              for path, (what, read) in _INTERPRETER_COPIES.items()}
    assert places, "the enumeration is empty, so this test would pass over nothing"
    assert all(places.values()), f"a copy has gone missing, so nothing pins it: {places}"
    drifted = {k: v for k, v in places.items() if v != want}
    assert not drifted, (
        f"run.sh composes {want!r}; these disagree, so the reviewer or the promoter is pointed "
        f"at an interpreter the export lane no longer uses: {drifted}")


def test_the_promotion_step_grant_reaches_the_rendered_allowlist():
    """⚠ A GRANT NOBODY EXERCISED IS INDISTINGUISHABLE FROM ONE THAT NEVER MATCHES (#239), so
    this asserts on what the matcher actually receives rather than on the source line.

    Skipped with its reason printed when the interpreter is absent, because the entry is
    guarded on that and an unconditional assertion would fail on a host that legitimately has
    no LiteRT harness. `test_the_promotion_step_grant_is_absent_without_an_interpreter` below
    is the other half.
    """
    interp = _config_env_value("REVIEWER_TORCH_PY")
    if not interp or not os.access(interp, os.X_OK):
        pytest.skip(f"no executable interpreter at {interp!r} — the grant is guarded off here")
    if not (REPO / "scripts" / "tools" / "check_publishable.py").exists():
        pytest.skip("check_publishable.py is not in this tree — the grant is guarded off")
    want = f"Bash({interp} scripts/tools/check_publishable.py:*)"
    allow = _rendered_allowlist()
    assert allow, "the rendered allowlist is empty — this test would pass vacuously"
    assert want in allow, (
        f"the promotion step is not granted; the reviewer is refused when it runs it. "
        f"wanted {want!r}")


def test_the_promotion_step_grant_names_the_script_not_the_bare_interpreter():
    """The narrowing the owner made on 2026-08-20, applied to this entry: granting the
    interpreter alone is arbitrary code execution under a new spelling."""
    interp = _config_env_value("REVIEWER_TORCH_PY")
    bad = [e for e in _rendered_allowlist()
           if interp and interp in e and "check_publishable.py" not in e]
    assert not bad, f"the torch interpreter is granted without naming a command: {bad}"


def test_the_promotion_step_grant_is_absent_without_an_interpreter(tmp_path):
    """⚠ THE GUARD'S NEGATIVE BRANCH — named in the docstring above and, until #434's sibling
    #433, not written. The commit's own comment says the absent branch is the point ("a ported
    lane adds nothing rather than a stale entry"), and it was the half that had only been
    reasoned about.

    ⚠ IT NEEDS A COPIED LANE, not an environment variable. `request_review.sh` sources
    `config.env` AFTER the environment, so `REVIEWER_TORCH_PY=/nonexistent` on the command line
    is overwritten before the guard runs and the entry renders anyway — measured, by the
    reviewer, when it tried to check this branch. That is config.env's contract working as
    designed, and it means the only honest route is a lane whose config differs.

    ⚠ THE CONTROL IS THAT SOMETHING ELSE STILL RENDERS. Asserting only an absence would pass
    against a copied script that failed outright and printed no allowlist at all — the green
    negative over data that was never there (#245).
    """
    lane = tmp_path / "workflow"
    shutil.copytree(REPO / "FerroStep" / "workflow", lane)
    cfg = lane / "config.env"
    original = cfg.read_text(encoding="utf-8")

    def render(value):
        cfg.write_text(
            re.sub(r"^REVIEWER_TORCH_PY=.*$", f"REVIEWER_TORCH_PY={value}",
                   original, flags=re.M),
            encoding="utf-8")
        return _rendered_allowlist(lane / "scripts" / "request_review.sh")

    for value, label in (("", "empty"), ("/nonexistent/python", "a path that is not executable")):
        allow = render(value)
        assert any("-m pytest" in e for e in allow), (
            f"the copied lane rendered no allowlist at all with {label}, so the absence below "
            f"would prove nothing: {allow}")
        granted = [e for e in allow if "check_publishable.py" in e]
        assert not granted, (
            f"the grant was rendered with REVIEWER_TORCH_PY {label} — a ported lane would carry "
            f"an entry naming an interpreter it does not have, which reads as covered and can "
            f"never match: {granted}")

    real = _config_env_value("REVIEWER_TORCH_PY")
    if real and os.access(real, os.X_OK):
        assert any("check_publishable.py" in e for e in render(real)), (
            "the same copied lane does NOT render the entry with a valid interpreter, so the "
            "absences above are not attributable to the guard")


def test_every_granted_directory_is_a_physical_path():
    """⚠ A SYMLINKED GRANT READS AS GIVEN AND DELIVERS NOTHING (#451).

    `SIBLING_REPO_CANDIDATES` gained `notes`, which is a symlink out of the repo. The launcher
    resolved candidates with `cd X && pwd` — the path you arrived BY — so the grant named
    `…/github/notes` while the harness fenced Bash out of the directory it points at. The entry
    was present, the reviewer was still refused, and only the reviewer trying it found out.

    This asserts on what `--dry-run` renders, not on the source, because the defect was in a
    resolved VALUE rather than in a line of code.
    """
    rendered = []
    out = subprocess.run([str(SCRIPT), "--full", "--dry-run", "--developer", "Ozzy"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, f"--dry-run failed, so this proves nothing: {out.stderr[-400:]}"
    toks = shlex.split(out.stdout.replace("\\\n", " "))
    for i, t in enumerate(toks):
        if t == "--add-dir" and i + 1 < len(toks):
            rendered.append(toks[i + 1])
    assert rendered, "no --add-dir rendered, so this test would pass over nothing"
    unresolved = [d for d in rendered if os.path.realpath(d) != d]
    assert not unresolved, (
        "these granted directories are not physical paths, so the harness fences the reviewer "
        f"out of what they actually point at while the entry reads as granted: {unresolved}")


def test_no_granted_directory_is_labelled_by_basename_alone():
    """⚠ A ONE-WORD LABEL TOLD THE REVIEWER ITS OWN REPO WAS OUT OF RANGE (#453).

    Resolving candidates physically (#451) made one entry land on `…/Notes/Sonora`, whose
    basename is `Sonora`. The brief listed it under a heading saying these are NOT part of your
    review range — so it named the repo under review as something to skip. The classification
    was right and the label beside it was wrong, which is AGENTS.md §5's shape and the one this
    repo pays most for.

    ⚠ Asserted as "at least two components", not as "never the string Sonora". A name-specific
    check passes for every OTHER directory whose basename collides, and goes stale the moment
    the candidate list changes — the same hand-list defect as #247.
    """
    out = subprocess.run([str(SCRIPT), "--full", "--dry-run", "--developer", "Ozzy"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, f"--dry-run failed, so this proves nothing: {out.stderr[-400:]}"
    labels = re.findall(r"^\* `[^`]+` — \*\*([^*]+)\*\*$", out.stdout, flags=re.M)
    assert labels, "no granted directories were listed, so this test would pass over nothing"
    flat = [lab for lab in labels if "/" not in lab]
    assert not flat, (
        "these granted directories are labelled by basename alone, so one of them can silently "
        f"name the repo under review and read as excluded from it: {flat}")


def _command_prefix(entry):
    """`Bash(git config --get:*)` -> `git config --get`. None for anything else."""
    if not entry.startswith("Bash(") or not entry.endswith(":*)"):
        return None
    return entry[len("Bash("):-len(":*)")]


def _shadowed(allow, deny):
    """Allow entries a deny entry swallows, because deny beats allow in the matcher."""
    out = []
    for a in allow:
        pa = _command_prefix(a)
        if pa is None:
            continue
        for d in deny:
            pd = _command_prefix(d)
            if pd is None or pd == pa:
                continue
            # `pa == pd` cannot reach here — identical entries are skipped above — so the
            # test is the prefix alone. It read as covering the equal case and could not.
            if pa.startswith(pd + " "):
                out.append((a, d))
    return out


def _rendered_denylist():
    """The deny list AS THE MATCHER WILL SEE IT, parsed from `--dry-run`.

    ⚠ `_array("REVIEWER_DENY")` reads the array LITERAL and is blind to any `+=` append — which
    is #245 exactly, the defect that replaced a source scan with this rendering for the allow
    side. The shadow check below compared a rendered allow list against a source-scanned deny
    list; no append exists today, so it was latent, but a deny added by append would have been
    invisible to the one guard written to catch a deny swallowing a grant (#456).
    """
    r = subprocess.run([str(SCRIPT), "--full", "--dry-run", "--developer", "Ozzy"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, f"--dry-run failed, so this test proves nothing: {r.stderr}"
    for line in r.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("--disallowedTools"):
            body = stripped.removeprefix("--disallowedTools").rstrip()
            return shlex.split(body.removesuffix("\\"))
    raise AssertionError("--dry-run printed no --disallowedTools line")


def test_no_deny_entry_silently_swallows_an_allow_entry():
    """⚠⚠ DENY BEATS ALLOW, SO A BROAD DENY KILLS A NARROW GRANT WITH THE SUITE GREEN (#452).

    Granting the read-only `git config` verbs required removing `Bash(git config:*)` from
    `REVIEWER_DENY`, because a deny that prefixes an allow wins and the grant would have read as
    given and never matched — #239's shape. Restoring that deny would silently disarm all five
    grants, and nothing tested it: `rg "git config" tests/` was empty when the reviewer looked.

    ⚠ ASSERTED AS A GENERAL RELATION, NOT A LIST OF THE FIVE ENTRIES. Naming them here is the
    hand-list that goes stale when a sixth is added, and it would not catch the same mistake
    made against a different grant. This catches any deny/allow pair with that shape.
    """
    allow, deny = _rendered_allowlist(), _rendered_denylist()
    assert allow and deny, "one of the lists is empty, so this test would pass over nothing"
    bad = _shadowed(allow, deny)
    assert not bad, (
        "these allow entries are swallowed by a broader deny, so they read as granted and can "
        f"never match: {bad}")


def test_the_shadow_relation_can_fire():
    """The positive control. The test above asserts a NEGATIVE over two live lists, which an
    empty parse or a broken prefix rule satisfies in silence."""
    assert _shadowed(["Bash(git config --get:*)"], ["Bash(git config:*)"]), \
        "the relation cannot see a broad deny swallowing a narrow allow"
    assert not _shadowed(["Bash(git config --get:*)"], ["Bash(git push:*)"]), \
        "the relation reports unrelated entries as shadowed"
    assert not _shadowed(["Bash(git config --get:*)"], ["Bash(git config --get:*)"]), \
        "an identical pair is not a shadow; it is the same entry named twice"


def test_every_read_only_grant_the_source_intends_actually_renders():
    """The launcher builds the read-only grants from a `for` list. Derived from that list, so
    what it catches is the gap between INTENT and what the matcher receives.

    ⚠ IT DOES NOT FREEZE THE LIST, AND THAT IS DELIBERATE. Measured: dropping one entry from the
    loop leaves this green, because both sides shrink together. Removing a grant is a DECISION —
    the owner's entitlement ruling is a floor, not a fixed set — so a test that blocked it would
    be pinning a choice rather than catching a defect. The floor below only catches a broken
    parse, not a deliberate removal, and says so rather than implying otherwise.
    """
    m = re.search(r'for _ro in ((?:"[^"]+"\s*\\?\s*)+); do', SOURCE)
    assert m, "the read-only grant loop is no longer shaped the way this test reads it"
    intended = re.findall(r'"([^"]+)"', m.group(1))
    assert len(intended) >= 4, f"only {len(intended)} read-only grants parsed; the scan is broken"
    allow = _rendered_allowlist()
    missing = [f"Bash({i}:*)" for i in intended if f"Bash({i}:*)" not in allow]
    assert not missing, (
        f"the script builds these grants and the matcher never receives them: {missing}")


def test_a_symlinked_candidate_is_granted_by_its_physical_path(tmp_path):
    """⚠ THE GUARD ABOVE ONLY EXERCISES ITS CASE WHERE THE `notes` SYMLINK EXISTS (#454).

    `notes` is gitignored, so in a worktree or a fresh clone the candidate is skipped, the two
    real siblings pass trivially, and the `cd && pwd` mutation stays green. The guard for #451
    was therefore environment-dependent — the exact shape #451 itself was about, reappearing in
    its own fix.

    This builds the case instead of hoping the host supplies it: a copied lane whose config
    names a symlink, run against this repo. It needs no `notes`, no sibling checkouts, and
    nothing gitignored.
    """
    lane = tmp_path / "workflow"
    shutil.copytree(REPO / "FerroStep" / "workflow", lane)
    target = tmp_path / "real_target"
    target.mkdir()
    link = tmp_path / "linked_candidate"
    link.symlink_to(target, target_is_directory=True)

    cfg = lane / "config.env"
    cfg.write_text(re.sub(r"^SIBLING_REPO_CANDIDATES=.*$",
                          f"SIBLING_REPO_CANDIDATES={link}", cfg.read_text(encoding="utf-8"),
                          flags=re.M), encoding="utf-8")

    out = subprocess.run([str(lane / "scripts" / "request_review.sh"),
                          "--full", "--dry-run", "--developer", "Ozzy"],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"the copied lane failed: {out.stderr[-400:]}"
    toks = shlex.split(out.stdout.replace("\\\n", " "))
    granted = [toks[i + 1] for i, t in enumerate(toks) if t == "--add-dir" and i + 1 < len(toks)]

    assert granted, "the copied lane granted nothing, so the assertion below proves nothing"
    # ⚠ THE POSITIVE HALF COMPARES REALPATHS (#457). `target` comes from `tmp_path` while the
    # launcher renders the PHYSICAL path this test exists to demand, so comparing the two
    # spellings asserts that the temp base is physical as much as anything about the launcher.
    # Where the base is a link the two disagree for a reason that has nothing to do with the
    # code under test, and a security test false-reds on a machine whose only sin is a
    # symlinked `/tmp` (macOS: `/tmp` -> `/private/tmp`). Realpathing both sides makes the
    # assertion say what it means: the SAME DIRECTORY, however each side spells the way there.
    #
    # ⚠⚠ BUT #457's STATED TRIGGER DOES NOT REPRODUCE ON THE PINNED pytest, AND THE ISSUE SAID
    # IT WOULD. MEASURED 2026-09-14 against pytest 9.1.1: `_pytest/tmpdir.py::getbasetemp`
    # calls `.resolve()` on BOTH paths it can return — the `--basetemp` it was given and the
    # `tempfile.gettempdir()` root it derives — so `tmp_path` is already physical even when the
    # base is a link. Run under `--basetemp=<a symlink>`, the pre-fix assertion PASSED. So this
    # is not a live false-red being fixed; what is being removed is a dependence on a pytest
    # implementation detail that has moved before (pytest #4427 is that code choosing
    # `abspath` over `resolve` for a platform difference) and that nothing here would notice
    # moving again. Do not re-derive the macOS claim from this comment: it was not measured on
    # macOS, and on this pytest it is false.
    physical = [os.path.realpath(g) for g in granted]
    assert os.path.realpath(target) in physical, (
        f"the symlinked candidate was granted as something other than its physical target — "
        f"granted {granted}, expected {target}")
    # ⚠⚠ AND THE NEGATIVE HALF MUST NOT BE REALPATHED, WHICH IS WHY THEY ARE WRITTEN
    # DIFFERENTLY. `os.path.realpath(link)` IS `target` — that is what a symlink is — so
    # resolving this side would compare the physical path with itself and the assertion could
    # never fail, whatever the launcher printed. The defect is a grant SPELLED as the link
    # (`cd X && pwd`, the path you arrived by), which is a property of the literal string the
    # launcher rendered: `granted` is checked unresolved, on purpose.
    # That split is also why the assertion above no longer distinguishes a link-grant from a
    # target-grant on a symlinked temp base — under the defect both realpath to the same
    # directory. It is this line that catches it there, and it is the only one that can.
    assert str(link) not in granted, (
        "the candidate was granted by its LINK path, which is what left the reviewer fenced "
        "out of the directory it points at (#451)")


# --------------------------------------------------------------------------- #
# the reviewer's spend ceiling — OPTIONAL BY DESIGN (#464)
# --------------------------------------------------------------------------- #
# ⚠⚠ THE REVIEWER RAN WITH NO CEILING AT ALL UNTIL 2026-09-07, AND NOTHING HERE SAW IT.
# `review_cycle.sh` put `--max-budget-usd` on the WORKER's call; this launcher — the longer,
# whole-diff call — passed none, while the driver's own `--help` called its flag a ceiling
# "per claude call". The fix (`BUDGET_ARGS`, built from the roster's `budget_usd`, which
# `ferrostep agent-env` emits) landed untested at all three of its points: the build, the
# call site, and the `--dry-run` preview of it.
#
# ⚠ WHAT IS PINNED IS NOT "THERE IS A CEILING". The roster deliberately sets none today
# (`FerroStep/config.yaml`, owner 2026-09-07 — absent IS the setting), so a test demanding one
# would freeze a decision the owner has not made and would go red against the correct
# deployment. The property is the PLUMBING, in both directions: a ceiling the roster sets
# reaches the real argv, an absent one yields no flag and no refusal, and the preview agrees
# with the call either way — #393 is what a dry run that describes a different command costs,
# and a preview is believed precisely because nobody re-derives it.


def _budget_lane(tmp_path, budget=None):
    """A whole ported repo — `FerroStep/` inside a git repo — with the reviewer's `budget_usd`
    set or absent.

    ⚠ THE WHOLE FOLDER, NOT JUST `workflow/`, and that is forced. The ROSTER is the input these
    tests vary, and the launcher reads it from `$REPO_ROOT/FerroStep/config.yaml`, where
    `REPO_ROOT` is `git rev-parse --show-toplevel` — not a path relative to the script. The
    copied-lane trick the tests above use (copy `workflow/`, run it with `cwd=REPO`) therefore
    goes on reading THIS repo's roster no matter what is written into the copy, so it cannot
    reach the one value under test.
    """
    root = tmp_path / "ported"
    shutil.copytree(REPO / "FerroStep", root / "FerroStep")

    cfg = root / "FerroStep" / "workflow" / "config.env"
    # ⚠⚠ THE SELF-CHECK IS DISARMED, AND NOT FOR SPEED: a REAL launch `eval`s `SELF_REVIEW_CMD`
    # before it spends a review, and in this repo that command is the whole pytest suite. Left
    # armed, this fixture would run the suite from inside the suite (here it would merely die,
    # since the ported repo has neither `.venv` nor `tests/` — a failure whose cause reads as
    # nothing to do with budgets).
    cfg.write_text(re.sub(r"^SELF_REVIEW_AT=.*$", "SELF_REVIEW_AT=none",
                          cfg.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")

    if budget is not None:
        roster = root / "FerroStep" / "config.yaml"
        text, n = re.subn(r"^(  reviewer:\n)", r"\g<1>    budget_usd: %s\n" % budget,
                          roster.read_text(encoding="utf-8"), count=1, flags=re.M)
        # ⚠ THE EDIT IS VERIFIED. A roster restructure would otherwise leave this fixture
        # writing NOTHING, and the ceiling test would then be exercising the ABSENT case under
        # a name that says the opposite — green, and asserting the reverse of what it claims.
        assert n == 1, "the reviewer entry is not shaped the way this fixture edits it"
        roster.write_text(text, encoding="utf-8")

    # An `origin` remote rather than a REPO_SLUG line, for the reason
    # `tests/test_review_cycle.py::_ported_lane` gives: `config.env` ships the slug EMPTY and
    # the launcher derives it, so hardcoding one leaves the half that actually runs unexercised.
    subprocess.run(["git", "init", "-q", "."], cwd=root, check=True)
    subprocess.run(["git", "remote", "add", "origin",
                    "git@github.com:Example-Org/ported-lane.git"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "init"], cwd=root, check=True)
    return root


def _launch(root, *args):
    """Run the ported launcher and return `(rc, output, argv)` — argv being EXACTLY what it
    handed `claude`, recorded by a stub.

    ⚠⚠ WHAT KEEPS A PAID, UNATTENDED REVIEW OUT OF THIS TEST IS THE STUB ON `PATH`, AND
    NOTHING ELSE. `tests/test_review_cycle.py::_real_startup` had to write that down after its
    own docstring credited a `HOME` redirect (#253); this helper is the same hazard one script
    along, because unlike every other behavioural test in this file it is NOT stopped by
    `--dry-run` or by a guard — it runs the launcher to the end on purpose.
    ⚠ The stub exits 0 rather than the fixture's 97: this call is the thing under test, so a
    non-zero would make the launcher report a failed review and mask it.
    ⚠ Argv is recorded NUL-SEPARATED. The brief is one argument containing dozens of newlines,
    so a line-per-argument record cannot be split back into arguments at all.
    """
    bindir = root.parent / "bin"
    bindir.mkdir(exist_ok=True)
    sink = root.parent / "argv"
    stub = bindir / "claude"
    stub.write_text('#!/bin/sh\n: > "$ARGV_SINK"\n'
                    'for a in "$@"; do printf \'%s\\0\' "$a" >> "$ARGV_SINK"; done\nexit 0\n')
    stub.chmod(0o755)

    home = root.parent / "home"
    home.mkdir(exist_ok=True)
    # The launcher lifts the PocketBase credential out of `~/.claude.json` and refuses without
    # one, so an empty HOME (what the module fixture uses) never reaches the call. The URL is a
    # dead loopback port on purpose: the tracker is then UNREACHABLE, which the launcher warns
    # about and continues past, and this test files nothing and touches no network.
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"pocketbase": {
        "env": {"PB_URL": "http://127.0.0.1:1", "PB_EMAIL": "e", "PB_PASSWORD": "p"}}}}))

    if sink.exists():
        sink.unlink()
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["HOME"] = str(home)
    env["ARGV_SINK"] = str(sink)
    p = subprocess.run([str(root / "FerroStep" / "workflow" / "scripts" / "request_review.sh"),
                        *args],
                       cwd=str(root), env=env, capture_output=True, text=True, timeout=300)
    argv = sink.read_text(encoding="utf-8").split("\0")[:-1] if sink.exists() else []
    return p.returncode, p.stdout + p.stderr, argv


def _previewed_ceiling(text):
    """What `--dry-run` SAYS about the ceiling: the value, or None where it says there is none.

    ⚠ SILENCE RAISES RATHER THAN READING AS None. A preview that mentions the ceiling nowhere
    is the #393 defect itself — the reader is left to assume, and the natural assumption is
    that the command shown is the command run. Folding that into "no ceiling" would make this
    helper agree with the bug in the one case it exists to catch.
    """
    for line in text.splitlines():
        s = line.strip().rstrip("\\").strip()
        if s.startswith("--max-budget-usd"):
            return s.split(None, 1)[1]
        if "no --max-budget-usd" in s:
            return None
    raise AssertionError(
        "the dry run says nothing at all about the spend ceiling, so a reader cannot tell "
        "the capped case from the uncapped one and will assume the command shown is complete")


def test_a_roster_ceiling_reaches_the_real_claude_call_and_the_preview_of_it(tmp_path):
    """⚠ THE CALL SITE AND THE PREVIEW ARE TWO DIFFERENT BLOCKS, so both are checked here.

    The launcher builds `BUDGET_ARGS` once and spends it twice: `${BUDGET_ARGS[@]+…}` on the
    real `claude -p`, and a separate `printf` in the `--dry-run` branch. Nothing tied them
    together, which is exactly how `merge_branch.sh` came to advertise a command it did not
    run (#393).

    ⚠ ASSERTED ON THE ARGV THE LAUNCHER ACTUALLY PASSED, not on the source and not on the
    preview alone — the lesson `_rendered_allowlist` records for the allowlist, applied to the
    one argument that decides what a run can spend. Only the argv can see an entry lost to a
    quoting or `set -u` mistake in the array expansion.
    """
    # ⚠ AN ODD VALUE. A round one could be produced by something other than this roster key —
    # a default, a fallback, a coincidence — and the point is that THIS setting arrived.
    root = _budget_lane(tmp_path, budget="3.77")
    rc, out, argv = _launch(root, "--full", "--developer", "Ozzy")
    assert rc == 0, f"the ported lane did not complete: {out[-800:]}"
    # ⚠ POSITIVE CONTROL FIRST: every assertion below reads `argv`, and an empty capture — a
    # stub that never ran, a launcher that exited at a guard — satisfies "the value is right"
    # vacuously in the one direction that matters.
    assert "-p" in argv, f"no claude launch was captured at all, so this proves nothing: {argv}"
    assert "--max-budget-usd" in argv, (
        "the roster sets budget_usd and the reviewer was launched without a ceiling anyway — "
        f"the uncapped state the flag was added to end: {argv}")
    passed = argv[argv.index("--max-budget-usd") + 1]
    assert passed == "3.77", f"the ceiling passed is not the one the roster set: {passed!r}"

    drc, dout, _ = _launch(root, "--full", "--dry-run", "--developer", "Ozzy")
    assert drc == 0, f"the dry run refused: {dout[-800:]}"
    assert _previewed_ceiling(dout) == passed, (
        "the dry run advertises a different ceiling from the one the real call carries; a "
        "preview that disagrees with its command is worse than none, because it is believed")


def test_no_roster_budget_means_no_flag_and_no_refusal(tmp_path):
    """⚠⚠ ABSENT IS THE WHOLE MEANING OF NO CEILING (`FerroStep/config.yaml`, owner 2026-09-07),
    which makes this the half that is easy to break by "improving" the other one.

    A deployment that sets nothing must keep behaving exactly as it did: no flag, and above
    all no refusal — a launcher that died without a budget would impose a limit nobody asked
    for on every ported lane, and the failure would arrive as an unrelated-looking startup
    error. Running to the call site rather than reading it is what makes that checkable: an
    empty `BUDGET_ARGS` that reaches `claude` as an empty STRING, or aborts the shell, is
    invisible to any scan of the source and fatal at the one moment it happens.

    ⚠ IT IS NOT A TEST OF THE `${BUDGET_ARGS[@]+…}` GUARD, and an earlier draft of this
    docstring said it was. MEASURED on this host (bash 5.3.9): the bare `"${BUDGET_ARGS[@]}"`
    form does NOT abort on an empty array under `set -u` — bash stopped treating that as unset
    in 4.4 — so swapping the guarded expansion for the plain one leaves this test green. The
    guard still earns its place for a lane ported to an older bash; this is simply not the
    instrument that would catch its removal, and saying otherwise would be a claim about a
    mutation nobody ran.

    ⚠ AND THE PREVIEW MUST SAY SO RATHER THAN GO QUIET. Printing nothing here renders a command
    that looks complete and is not capped, which is how the uncapped reviewer survived reading.
    """
    root = _budget_lane(tmp_path)
    rc, out, argv = _launch(root, "--full", "--developer", "Ozzy")
    assert rc == 0, (
        "a roster with no budget_usd REFUSED the launch; absent is a setting, not an omission, "
        f"and every lane that sets nothing is now broken: {out[-800:]}")
    # ⚠ POSITIVE CONTROL ON A NEGATIVE. "No --max-budget-usd in argv" is satisfied by an empty
    # capture, which is precisely what a launcher that never reached the call would leave.
    assert "-p" in argv and "--system-prompt-file" in argv, (
        f"no claude launch was captured, so the absence asserted below proves nothing: {argv}")
    assert "--max-budget-usd" not in argv, (
        "a ceiling was passed although the roster sets none — the reviewer is now capped at a "
        f"number nobody chose: {argv}")

    drc, dout, _ = _launch(root, "--full", "--dry-run", "--developer", "Ozzy")
    assert drc == 0, f"the dry run refused: {dout[-800:]}"
    assert _previewed_ceiling(dout) is None, (
        "the dry run previews a ceiling the real call does not carry — the #393 shape, with "
        "the preview claiming the safer of the two states")

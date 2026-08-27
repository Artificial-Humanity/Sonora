"""The review launcher's guards, which have twice been defeated with the suite green (#110).

`workflow/scripts/request_review.sh` decides what a reviewer is told and what it is allowed to run.
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
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "workflow" / "scripts" / "request_review.sh"
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
    assert "workflow/scripts/request_review.sh" in manifest


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

    `Bash(./workflow/scripts/request_review.sh:*)` — the whole script — was granted while the comment
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
    # survive `workflow/` being copied into another repo. What the script must still do is READ
    # the list from config; which paths are in it is the port's business.
    assert "SIBLING_REPO_CANDIDATES" in SOURCE, \
        "the sibling paths must come from config.env, not from this script"
    cfg = (REPO / "workflow" / "config.env").read_text(encoding="utf-8")
    assert "SIBLING_REPO_CANDIDATES=" in cfg, "config.env must declare the setting, even if empty"
    # And read-only: no editing tool exists, and all three are denied by name.
    tools = SOURCE[SOURCE.index("REVIEWER_TOOLS="):].split("\n")[0]
    for t in ("Edit", "Write", "NotebookEdit"):
        assert t not in tools, f"{t} in --tools would make --add-dir a write grant"
        assert t in _array("REVIEWER_DENY")


def _rendered_allowlist():
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
    r = subprocess.run([str(SCRIPT), "--full", "--dry-run", "--developer", "Ozzy"],
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

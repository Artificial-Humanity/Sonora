"""The driver's safety properties, pinned because it runs unattended with write access.

`workflow/scripts/review_cycle.sh` alternates a review with a `claude -p` WORKER that edits files and
commits, without a human in the loop. That is a different risk class from anything else in
`scripts/`, and exactly one property makes it acceptable:

    ⚠ NOTHING IT RUNS CAN REACH `main`.

The repo has no branch protection and force-push is unblocked (AGENTS.md §1), so the deny on
`git push` is the only thing between an unattended loop and production. Every assertion here
exists because the fix that establishes it would otherwise be correct-but-unpinned — the shape
this cycle produced five times (#110, #118, #119, and twice more), and the one time it guarded
a *capability* rather than a sentence, reverting it would have restored unbounded nested
reviews (#115).
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "workflow" / "scripts" / "review_cycle.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def _array(name):
    start = SOURCE.index(f"{name}=(")
    end = SOURCE.index("\n)", start)
    out = []
    for line in SOURCE[start:end].splitlines():
        out.extend(re.findall(r'"([^"]+)"', line.split("#", 1)[0]))
    return out


def test_the_worker_is_denied_git_push():
    """The single property that makes an unattended write loop acceptable."""
    assert "Bash(git push:*)" in _array("WORKER_DENY")


def test_the_driver_itself_never_pushes():
    """Denying the worker is pointless if the driver pushes on its behalf."""
    # ⚠ STRING LITERALS ARE STRIPPED FIRST. A naive search for `git push` matches the deny
    # entry "Bash(git push:*)" and the closing message that tells the human to push by hand —
    # both are data. The first version of this test failed against a correct script for
    # exactly that reason, which is the same false-red that #112's parser produced.
    code = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
    code = re.sub(r'"[^"\n]*"', '""', code)
    code = re.sub(r"'[^'\n]*'", "''", code)
    assert not re.search(r"\bgit\s+push\b", code), "review_cycle.sh must not push"


def test_the_worker_cannot_re_enter_the_loop():
    """A worker that can run the driver or the launcher recurses, billed, unattended.

    ⚠ THE FULL PATH IS CHECKED, NOT THE BASENAME, AND THAT IS NOW THE POINT. This asserted
    `any("review_cycle.sh" in d for d in deny)` until 2026-08-17 — a substring test that stayed
    GREEN after the scripts moved from `scripts/` to `workflow/` while every deny entry still
    named the old path. The guard was disarmed and the test said nothing, because a permission
    entry naming a path that does not exist denies precisely nothing.
    """
    deny = _array("WORKER_DENY")
    for s in ("workflow/scripts/review_cycle.sh", "workflow/scripts/request_review.sh",
              "workflow/scripts/merge_branch.sh"):
        assert f"Bash({s}:*)" in deny, f"the worker must not be able to run {s}"
        assert f"Bash(./{s}:*)" in deny, f"the './' form of {s} is a second way in"


def test_every_denied_script_actually_exists():
    """⚠ A DENY ON A PATH THAT DOES NOT EXIST IS NOT A DENY.

    The failure above was invisible because nothing tied the deny list to the filesystem. This
    does: rename or move a script without updating the list in the same commit, and the entry
    silently stops matching while the worker regains the capability.
    """
    for entry in _array("WORKER_DENY"):
        m = re.fullmatch(r"Bash\(\.?/?([\w][\w./-]*\.(?:sh|py)):\*\)", entry)
        if m:
            assert (REPO / m.group(1)).is_file(), \
                f"WORKER_DENY names {m.group(1)}, which does not exist — it denies nothing"


def test_the_worker_keeps_the_tracker_script():
    """⚠ The inverse assertion, deliberate.

    `workflow/scripts/issue.py` is how the worker takes an issue, comments, escalates, and moves one to
    `review` — its actual job — and it is the single path that enforces the counter cap and the
    mandatory-comment rules. Denying it would push the worker back to raw tracker writes, where
    none of those refusals exist.
    """
    assert not any("issue.py" in d for d in _array("WORKER_DENY"))


def test_git_dash_c_is_NOT_denied_to_the_worker():
    """⚠ The inverse assertion, and it is deliberate.

    DEVELOPER.md §1 requires every commit to be authored
    `git -c user.name=Ozzy -c user.email=…`. Denying `git -c` — which the *reviewer* denies,
    correctly, for the opposite reason — would leave the worker unable to commit as itself,
    and the failure would look like an identity bug rather than a permission one.
    """
    assert not any(d.startswith("Bash(git -c") for d in _array("WORKER_DENY"))


def test_every_claude_call_carries_a_spend_ceiling():
    """It spends money unattended; an uncapped call is the whole hazard."""
    for m in re.finditer(r"^\s*claude -p .*?(?=\n\s*\w+=\$\?|\n\s*set -e)", SOURCE, re.S | re.M):
        assert "--max-budget-usd" in m.group(0), "an unattended claude call with no ceiling"


def test_the_review_ceiling_derives_from_the_definition():
    """⚠ This asserted `^[1-4]$` was IN the script — the cap's arithmetic shadow (max+1)
    wearing a refusal, which would silently disagree with any owner change to
    agent_passes.max. The ceiling now derives from the lane definition, and the literal
    range must not come back."""
    assert '=~ ^[1-4]$' not in SOURCE, "the hardcoded review-ceiling range returned"
    assert "sonora-lane.json" in SOURCE, "the ceiling no longer derives from the definition"
    assert "DEF_REVIEWS" in SOURCE


def test_it_refuses_a_dirty_tree():
    """The worker commits, so uncommitted edits would be swept into a commit nobody wrote."""
    assert "working tree is dirty" in SOURCE


def test_the_abort_marker_is_checked_before_the_exit_code():
    """A review can exit 0 and still say the change must not land — a content signal, not a
    failure signal. Order matters: checking rc first would `exit` before reading the text.

    ⚠ Pin the CHECK, not the first mention of the token. This asserted
    `SOURCE.index("MUST-NOT-LAND")`, which finds the `usage()` heredoc ~340 lines above the
    check it means — so it stayed green with the abort block moved after the rc check AND
    with the abort block deleted outright (both mutation-measured, #355). Same lesson as
    `_open_filter()` below: a substring search over a whole script does not pin the line
    you mean.
    """
    assert 'if grep -q "MUST-NOT-LAND"' in SOURCE, "the abort check is gone"
    marker = SOURCE.index('if grep -q "MUST-NOT-LAND"')
    rc_check = SOURCE.index('if [[ "$RC" -ne 0 ]]')
    assert marker < rc_check


def test_the_reviewer_persona_is_told_to_emit_the_marker():
    """The driver greps for a token the reviewer must know to produce — a machine contract
    split across two files, which is how one half gets edited alone."""
    persona = (REPO / "workflow" / "REVIEWER.md").read_text(encoding="utf-8")
    assert "MUST-NOT-LAND" in persona


def _open_filter():
    """The convergence check itself, not merely a line that resembles it.

    ⚠ The predecessor of this helper is why it exists. That test asserted `'branch_name!=""'
    in SOURCE` — anywhere in the file — and stayed green after the convergence filter stopped
    containing it, because an unrelated guard further down still did. A substring search over
    a whole script does not pin the line you mean.
    """
    return next(ln for ln in SOURCE.splitlines() if ln.startswith("OPEN_FILTER="))


def test_convergence_is_scoped_to_the_branch_under_review():
    """Convergence must count THIS branch's issues and no others.

    ⚠ The old guard was `branch_name!=""`, which excluded another branch's issues only
    while they carried no branch at all. Once they carry one, the clause that dropped them
    sweeps them in — and since nobody works another branch on this cycle, the loop could
    never see itself converge.

    ⚠ This docstring cited "the migrated GitHub backlog … on `github-issues-fixes`" until
    2026-08-19. Those records are not in the live tracker (it holds nothing below #90); they
    are in `notes/tracker-export-2026-08-17.json`. The property under test is unchanged —
    only the example was stale. ⚠ It was one of FOUR copies, not three: `workflow/REVIEWER.md`
    carried it twice — the retraction was added beside one and the other survived another two
    passes (issue #163). Counting the copies before claiming completeness is the lesson.
    """
    flt = _open_filter()
    assert 'branch_name=\\"$BRANCH\\"' in flt, flt
    assert 'branch_name!=""' not in flt, flt
    # ⚠⚠ AND BY REPO (2026-08-27), which branch alone does NOT give you. `rescopes` gained a
    # `repo` label on 2026-08-26; four findings moved to Artificial-Humanity/FerroStep the
    # next day and KEPT this branch's `branch_name`, because a rescope moves the label it is
    # given and nothing else. The driver then reported `currently: 4` on a branch whose own
    # queue was empty — and no fix pass could ever clear them, since they belong to another
    # repo and another agent. It would have burned every review to the ceiling at the
    # per-call spend and reported NOT CONVERGED.
    #
    # ⚠ `merge_branch.sh` and `issue.py` were ALREADY scoped this way. This was the one of the
    # three that was not, which is why the GATE correctly said `BLOCKED by [335]` while the
    # DRIVER saw four extra — a gate and a driver disagreeing about which issues belong to a
    # branch, the same class as two copies of the severity ladder.
    assert 'repo=' in flt, (
        "the convergence filter is not scoped by repo; it counts issues rescoped away from "
        "this repo and the loop can never converge: " + flt)
    # ...and $BRANCH is the checked-out branch, resolved exactly once.
    assigns = [ln for ln in SOURCE.splitlines() if ln.startswith("BRANCH=")]
    assert len(assigns) == 1, assigns
    assert "rev-parse --abbrev-ref HEAD" in assigns[0], assigns[0]


def _guard_readings():
    """The two `agent_passes` readings the stall guard compares."""
    return [ln.strip() for ln in SOURCE.splitlines()
            if ln.strip().startswith(("BEFORE_SUM=", "AFTER_SUM="))]


def test_the_stall_guard_compares_like_with_like():
    """⚠ THIS GUARD FAILED OPEN FOR ITS ENTIRE LIFE, AND SILENTLY.

    `BEFORE_SUM` was the COUNT of open issues on this branch; `AFTER_SUM` was the COUNT of
    `agent_passes=0` issues REPO-WIDE. Two populations, two scopes, compared for equality — so
    the stall it exists to catch could only be caught by coincidence, and a crashed worker
    would have looped forever spending no attempts. That is the exact unbounded loop the cap
    exists to prevent, reintroduced by the thing automating it.

    Nothing about the output said so: a guard that never fires looks identical to a guard that
    never needed to. So this pins the shape rather than the wording — BOTH readings must come
    from the same helper on the same argument.
    """
    before, after = _guard_readings()
    assert before.split("=", 1)[1] == after.split("=", 1)[1], (before, after)
    assert "pb_passes" in before, before


def test_the_stall_guard_sums_over_every_state():
    """A worker may ESCALATE, which takes an issue out of `state="open"`.

    If the sum were restricted to open issues it could FALL across a pass that did real work,
    and the guard would read that as a stall and stop the loop. `pb_passes` filters on
    `branch_name` alone, deliberately.
    """
    body = SOURCE[SOURCE.index("pb_passes() {"):]
    body = body[:body.index("\n}\n")]
    flt = next(ln for ln in body.splitlines() if "urllib.parse.quote" in ln)
    assert "branch_name" in flt, flt
    assert "state=" not in flt, flt
    assert "perPage=500" in body, "an unpaged read silently under-sums"


def test_a_falling_counter_is_not_treated_as_a_stall():
    """The owner resets `agent_passes` to re-arm an issue — their dial, never "corrected".

    A drop therefore means work happened AND the owner intervened, which is the one case where
    exiting would be most wrong.
    """
    assert "AFTER_SUM < BEFORE_SUM" in SOURCE
    stall = SOURCE.index("AFTER_SUM == BEFORE_SUM")
    fall = SOURCE.index("AFTER_SUM < BEFORE_SUM")
    exits = [m.start() for m in re.finditer(r"exit 5", SOURCE)]
    assert any(stall < e < fall for e in exits), "the stall branch must be the one that exits"


def test_no_query_names_the_removed_escalated_field():
    """`escalated` is a value of `state`, not a field beside it (owner, 2026-08-17).

    Not a style preference: the column is GONE, and PocketBase answers a filter naming it
    with HTTP 400. A leftover `escalated=false` would not merely fail to narrow a query — it
    makes the driver read the tracker as unreachable and stop.
    """
    for ln in SOURCE.splitlines():
        code = ln.split("#", 1)[0]
        if "state=" in code or code.startswith("OPEN_FILTER="):
            assert "escalated=" not in code, ln
    assert 'state=\\"open\\"' in _open_filter()


def test_it_is_classified_in_the_pipeline_manifest():
    manifest = (REPO / "scripts" / "pipeline_manifest.py").read_text(encoding="utf-8")
    assert "workflow/scripts/review_cycle.sh" in manifest


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def _ported_lane(tmp_path):
    """A minimal repo holding `workflow/` and NOTHING ELSE — the ported-lane shape.

    ⚠ NO `.gitignore`, DELIBERATELY. WORKFLOW.md's "Porting this lane" says to copy
    `workflow/` into the new repo, so a ported copy has no ignore entry for the driver's
    runtime files. A fixture that copied this repo's `.gitignore` would make the test pass
    for a reason the ported lane does not have, which is the failure mode `_array` had.
    """
    (tmp_path / "workflow" / "scripts").mkdir(parents=True)
    for name in ("request_review.sh",):
        p = tmp_path / "workflow" / "scripts" / name
        p.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)
    dst = tmp_path / "workflow" / "scripts" / "review_cycle.sh"
    dst.write_text(SOURCE, encoding="utf-8")
    dst.chmod(0o755)
    (tmp_path / "workflow" / "DEVELOPER.md").write_text("persona\n", encoding="utf-8")
    # The driver derives its review ceiling from the lane definition at startup (phase 2)
    # and refuses without one — correct for a real repo, so the fixture provides the file,
    # exactly as WORKFLOW.md's porting instructions say it must travel with the lane.
    (tmp_path / "workflow" / "sonora-lane.json").write_text(
        '{"counters": [{"name": "agent_passes", "max": 3}]}\n', encoding="utf-8")
    # ⚠ THE DRIVER NOW NEEDS A TRACKER SLUG, for the same reason it needs the lane definition
    # above: its convergence filter is scoped by `repo` as well as `branch_name` (2026-08-27),
    # because a branch-only filter counts issues RESCOPED to another repo and the loop can
    # then never converge. An underivable slug is a refusal, deliberately — a filter that
    # silently widens to other repos is how that was missed in the first place.
    #
    # An `origin` remote rather than a REPO_SLUG line, because that is the path Sonora itself
    # takes: `config.env` ships with REPO_SLUG EMPTY and the slug is derived, so hardcoding
    # one here would leave the derivation — the half that actually runs — unexercised.
    (tmp_path / "workflow" / "config.env").write_text("REPO_SLUG=\n", encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "remote", "add", "origin",
                    "git@github.com:Example-Org/ported-lane.git"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _commit_all(repo, msg):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", msg], cwd=repo, check=True)


def _real_startup(repo, *extra):
    """Run the driver for real — no `--dry-run` — and stop it right after startup.

    ⚠ `--dry-run` CAN NO LONGER BE THE VEHICLE (#248). The old test ran the dry run and
    asserted the files were gone, so the suite asserted the very deletion that broke
    `--help`'s "files nothing" promise. A test has to reach a REAL startup to see the clear.

    ⚠ HOME IS REDIRECTED SO THE TRACKER LOOKUP FAILS ON PURPOSE. `pb()` reads
    `~/.claude.json`; with it absent the helper prints "unreachable" and the driver stops with
    exit 4, which makes this deterministic and offline rather than dependent on a live tracker.

    ⚠⚠ BUT EXIT 4 ARRIVES *AFTER* `request_review.sh`, NOT BEFORE (#253). An earlier version of
    this docstring said "before it can spawn `claude`", which is false and expensively so: the
    `pb` call whose "unreachable" produces exit 4 sits INSIDE the review loop, one statement
    below the launcher invocation. **The only thing keeping a real, paid, unattended `claude
    -p` out of this test is the stub `request_review.sh` that `_ported_lane` writes** — never
    the HOME redirect. Anyone reusing this helper against a lane with a REAL launcher would
    spawn a live review, and the docstring told them it was safe.
    """
    env = dict(os.environ, HOME=str(repo / "fake-home"))
    (repo / "fake-home").mkdir(exist_ok=True)
    return subprocess.run(["bash", "workflow/scripts/review_cycle.sh", *extra],
                          cwd=repo, capture_output=True, text=True, env=env)


@pytest.mark.parametrize("runtime_file", [".review_cycle.notes", ".review_cycle.stop"])
def test_a_stale_runtime_file_does_not_block_the_next_run(tmp_path, runtime_file):
    """⚠⚠ THE DRIVER USED TO RUN EXACTLY ONCE PER MANUAL CLEANUP, and nothing said so.

    The worker's brief, step 5, tells it to write `.review_cycle.notes`; nothing removed the
    file; and the dirty-tree check refuses to start when the tree is not clean, which an
    untracked file made it. Run 1 succeeded and run 2 died at the door — blaming "your
    uncommitted edits", which is not what was there.

    ⚠ PARAMETRISED OVER BOTH RUNTIME FILES (#246). The first fix moved the notes `rm` above the
    refusal and left the stop file's below it, so the identical death survived one filename
    over, while the commit's own reasoning covered both.

    ⚠ THIS RUNS THE SCRIPT INSTEAD OF READING IT. A source scan for the `rm` passes with it
    placed AFTER the dirty-tree check, where it can never be reached — the ordering IS the fix,
    and only execution sees ordering (#231).
    """
    repo = _ported_lane(tmp_path)
    (repo / runtime_file).write_text("left over from a previous cycle\n", encoding="utf-8")

    r = _real_startup(repo)
    assert r.returncode == 4, (
        f"expected the tracker-unreachable stop (4) after startup, got {r.returncode}: "
        f"{r.stderr[-400:]}")
    assert not (repo / runtime_file).exists(), (
        f"{runtime_file} survived a real startup, so the next run starts from a poisoned tree")


@pytest.mark.parametrize("runtime_file", [".review_cycle.notes", ".review_cycle.stop"])
def test_a_dry_run_does_not_touch_the_runtime_files(tmp_path, runtime_file):
    """⚠⚠ #248: `--help` says a dry run "Spends nothing, files nothing" — it deleted both.

    Two real losses, and both silent. The notes file is the loop's ONLY memory across passes
    (DEVELOPER.md §3: without it "the next review cannot tell a fix from an omission"), so
    checking the plan between a worker pass and the next review destroyed the brief. And the
    stop file is the documented halt control, so a dry run in a second terminal DISARMED a
    halt an operator had deliberately armed, and the cycle kept spending.

    ⚠ THE PREVIOUS TEST HERE ASSERTED THE DELETION AS A REQUIREMENT. It used `--dry-run` as a
    cheap way to reach startup and then asserted the files were gone — so the suite positively
    agreed with the defect, and nobody weighed that assertion against `--help`. That is why
    this is written as a promise about the FLAG rather than a by-product of the fixture.
    """
    repo = _ported_lane(tmp_path)
    (repo / runtime_file).write_text("armed by the operator\n", encoding="utf-8")

    r = subprocess.run(["bash", "workflow/scripts/review_cycle.sh", "--dry-run"],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, f"the dry run refused: {r.stderr[-400:]}"
    assert (repo / runtime_file).exists(), (
        f"--dry-run deleted {runtime_file}, but --help promises it files nothing")


def test_a_dry_run_is_not_refused_by_its_own_leftovers(tmp_path):
    """The exclusion in the dirty-tree check, pinned as behaviour.

    A ported lane has no `.gitignore` entry for these files, so without the pathspec exclusion
    a dry run over a leftover notes file would die on the dirty-tree refusal — reporting a
    failure the real run does not hit, because the real run clears them first.
    """
    repo = _ported_lane(tmp_path)
    (repo / ".review_cycle.notes").write_text("stale\n", encoding="utf-8")
    (repo / ".review_cycle.stop").write_text("armed\n", encoding="utf-8")
    r = subprocess.run(["bash", "workflow/scripts/review_cycle.sh", "--dry-run"],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, f"leftovers alone refused the dry run: {r.stderr[-400:]}"


@pytest.mark.parametrize("stop_flag,dirt", [
    # ⚠⚠ THE DIRT IS CHOSEN PER CASE SO EACH ONE DISCRIMINATES. A first draft used the same
    # two dirty files everywhere; the glob and basename-collision rows then passed against the
    # VULNERABLE script too, because the exclusion hid one file and the other still reported
    # the tree as dirty. Two of five "controls" could not fail — the same green-negative defect
    # #245 and #248 were about, committed inside the fix for #252. Each row's dirt is now
    # exactly what that row's exclusion would have hidden.
    pytest.param([], ["a.md"], id="default"),
    pytest.param(["--stop-file", "/tmp/somedir/"], ["a.md"],
                 id="trailing-slash-empty-basename"),
    pytest.param(["--stop-file", "/tmp/*.md"], ["a.md"], id="glob-in-basename"),
    pytest.param(["--stop-file", "/tmp/halt.txt"], ["halt.txt"],
                 id="basename-collides-with-tracked-file"),
    pytest.param(["--stop-file", "sub/halt"], ["a.md"], id="stop-file-below-the-root"),
])
def test_real_dirt_still_refuses(tmp_path, stop_flag, dirt):
    """⚠ THE CONTROL ON THE EXCLUSION — it must hide the driver's two files and NOTHING else.

    The dirty-tree check is what stops the unattended worker absorbing edits nobody wrote a
    message for, and the driver spawns that worker with auto-mode and commit rights.

    ⚠⚠ PARAMETRISED BECAUSE THE FIRST VERSION OF THIS CONTROL ONLY EVER RAN THE DEFAULT FLAG,
    and stayed green through all three failures below (#252). The exclusion was built from
    `${STOPFILE##*/}` — the BASENAME of an operator-supplied path — dropped into a git
    pathspec unvalidated:

      * a TRAILING SLASH made the basename empty, and `git status --porcelain -- . ':(exclude)'`
        exits 0 with no output on ANY tree — the guard did not narrow, it VANISHED, silently;
      * a GLOB hid every matching file at any depth (fnmatch without FNM_PATHNAME);
      * a plain BASENAME COLLISION hid a tracked, modified file at the repo root.

    `sub/halt` is the opposite failure: it excluded `halt`, never matched `sub/halt`, and the
    driver died with the wrong-cause message #246 was filed about.
    """
    repo = _ported_lane(tmp_path)
    (repo / "sub").mkdir(exist_ok=True)
    (repo / "sub" / "keep.txt").write_text("k\n", encoding="utf-8")
    for name in ("a.md", "halt.txt"):
        (repo / name).write_text("original\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    for name in dirt:
        (repo / name).write_text("CHANGED\n", encoding="utf-8")

    r = _real_startup(repo, *stop_flag)
    assert r.returncode == 1, (
        f"dirt {dirt} was not refused with {stop_flag or '(default)'}: "
        f"rc={r.returncode}\n{r.stderr[-500:]}")
    assert ("working tree is dirty" in r.stderr
            or "must name a file" in r.stderr), r.stderr[-500:]


def test_the_drivers_own_files_are_not_read_as_dirt(tmp_path):
    """The other half: the exclusion must actually exclude, or #246 comes straight back.

    A stop file below the repo root is the case the basename version got wrong in this
    direction — it excluded `halt`, never matched `sub/halt`, and refused the run.
    """
    repo = _ported_lane(tmp_path)
    (repo / "sub").mkdir(exist_ok=True)
    (repo / "sub" / "keep.txt").write_text("k\n", encoding="utf-8")
    _commit_all(repo, "fixture")
    (repo / ".review_cycle.notes").write_text("stale\n", encoding="utf-8")
    (repo / "sub" / "halt").write_text("armed\n", encoding="utf-8")

    r = subprocess.run(["bash", "workflow/scripts/review_cycle.sh", "--dry-run",
                        "--stop-file", "sub/halt"],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"the driver's own runtime files were read as dirt: {r.stderr[-500:]}")


@pytest.mark.parametrize("var", ["$NOTES_FILE", "$STOPFILE"])
def test_each_runtime_file_is_cleared_before_the_dirty_tree_check(var):
    """Ordering, pinned separately — the behavioural test cannot name WHY it failed.

    ⚠⚠ SEARCHING FOR `rm -f "$STOPFILE"` IS THE WRONG INSTRUMENT, and the first draft of this
    test used it and failed against a correct script. That exact string DOES occur — inside
    `check_stop`, which removes the file on the halt path, 90 lines BELOW the refusal. The
    startup clear takes both files on one line, so a string search finds the wrong `rm`.

    Scanning LINES for an `rm -f` naming the variable, and requiring one before the refusal,
    is indifferent to how the two are grouped — one line or two, either order.
    """
    lines = [l.split("#", 1)[0] for l in SOURCE.splitlines()]
    dirty = next(i for i, l in enumerate(lines) if "working tree is dirty" in l)
    cleared = [i for i, l in enumerate(lines)
               if l.strip().startswith("rm -f") and var in l]
    assert cleared, f"nothing clears {var} at all"
    assert min(cleared) < dirty, (
        f"{var} is only cleared at line {min(cleared) + 1}, below the dirty-tree refusal at "
        f"line {dirty + 1} — where it cannot stop the refusal it exists to prevent")


def test_the_notes_path_has_one_definition():
    """It was written out three times; the `rm` was added against one of the copies."""
    code = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
    assert code.count(".review_cycle.notes") == 1, (
        "the literal path belongs in NOTES_FILE alone; every other use reads the variable")


def test_an_underivable_slug_is_a_REFUSAL_not_a_branch_only_filter(tmp_path):
    """⚠ THE OTHER DIRECTION, and the one that matters. Falling back to a branch-only filter
    when the slug cannot be derived would restore the defect above silently, in exactly the
    repos least placed to notice — a freshly ported lane with no `origin` yet. So it stops.

    The positive half is `_ported_lane` itself: every other test in this file now runs against
    a repo whose slug is DERIVED from `origin`, so the derivation is exercised continuously
    rather than by one assertion.
    """
    repo = _ported_lane(tmp_path)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo, check=True)
    out = _real_startup(repo, "--dry-run")
    assert out.returncode != 0, (
        "a repo with no origin and no REPO_SLUG ran anyway — the filter has silently widened "
        "to every repo's issues:\n" + out.stdout + out.stderr)
    assert "slug" in (out.stdout + out.stderr).lower(), out.stdout + out.stderr


def test_no_claude_call_passes_both_append_system_prompt_forms():
    """⚠⚠ THE CLI REFUSES THE PAIR, AND THE LOOP HAD NEVER COMPLETED A FIX PASS.

        Error: Cannot use both --append-system-prompt and --append-system-prompt-file.

    MEASURED 2026-08-27 on the first unattended run: the review completed and filed five
    findings, then the worker died instantly on argument parsing. Three files described this
    driver as the way to run the lane, and the owner had authorised it as automation.

    ⚠ THE FAILURE WAS LOUD AND STILL NEARLY INVISIBLE. The driver's stop message says "the
    tree may hold partial work; inspect before re-running" — which reads as a worker that ran
    and broke, not one that never started. Nothing in the output told the two apart, and the
    tree being clean afterwards is equally consistent with both.

    Pinned on the SHAPE rather than on today's error text, because the CLI's message is not
    this repo's to keep stable.
    """
    calls = re.findall(r"^\s*claude -p .*?(?=\n\s*\w+=\$\?|\n\s*set -e)", SOURCE, re.S | re.M)
    assert calls, "no claude invocation found — this guard is scanning nothing"
    for c in calls:
        both = "--append-system-prompt-file" in c and re.search(
            r"--append-system-prompt(?!-file)", c)
        assert not both, (
            "a claude call passes BOTH --append-system-prompt and its -file form; the CLI "
            "refuses that combination and the call dies before it starts:\n" + c[:400])


def test_the_slug_derivation_does_not_pass_an_unmatched_string_through(tmp_path):
    """⚠ #344 — THE REFUSAL COULD NOT FIRE. `sed -E 's#…#…#'` leaves a NON-MATCHING string
    unchanged, so `[[ -n "$slug" ]]` accepted whatever `git remote get-url` said — a bare
    `Sonora`, an unparsed URL, anything at all. The refusal was written specifically to avoid
    failing open and failed open. `-n … p` prints only on a match.

    ⚠ It is asserted on the SHAPE, because the regex alone cannot settle this: a local
    checkout path still yields a well-formed, wrong `repos/Sonora`. That case is caught by the
    data-level guard in the test below, which is the actual fix.
    """
    line = next(ln for ln in SOURCE.splitlines() if "REPO_SLUG_FILTER=\"$(printf" in ln)
    assert "sed -nE" in line and line.rstrip().endswith("p')\""), (
        "the slug sed must print only on a match, or a non-matching origin passes through "
        "unchanged and the emptiness refusal below it can never fire:\n  " + line)


def test_zero_issues_in_ANY_state_is_refused_not_reported_as_convergence():
    """⚠⚠ #344 — "NO OPEN ISSUES" AND "THE FILTER MATCHED NOTHING" ARE THE SAME READING.

    With a slug that names no repo in the tracker, `OPEN` is 0 and the driver announced
    CONVERGED having read nothing — the empty-enumeration vacuous pass, in the one line that
    decides the loop is finished.

    ⚠ `merge_branch.sh` already learned this and calls it NEVER_REVIEWED; it was measured
    there when a branch's findings moved away and the gate offered to push 21 unreviewed
    commits to main. The driver had no equivalent. Pinned as ORDER — the total must be
    consulted BEFORE `CONVERGED=1` — because a check that runs after the flag is set is
    decoration.
    """
    code = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
    assert "EVER=" in code, "the driver does not ask whether this branch has any issues at all"
    ever = code.index("EVER=")
    # ⚠ EVERY arm, not the first one found. This read `code.index("CONVERGED=1")`, and #357
    # inserted a SECOND one inside the --allow-empty-tracker bypass, ~16 lines above the
    # convergence this test was written for. `.index` silently moved to the bypass, the
    # refusal below fell outside the span, and `exit 5` became deletable green (#359).
    # Same first-occurrence shape as #355, in this file, one test along.
    flags = [m.start() for m in re.finditer(r"CONVERGED=1", code)]
    assert flags, "the driver never reports convergence at all"
    assert ever < min(flags), (
        "the zero-issues check runs after CONVERGED is set, so it cannot prevent the claim")
    # ⚠ And unreachable must not read as zero — the same trap one layer down.
    assert "unreachable" in code[ever:min(flags)], (
        "an unreachable tracker is not a count of zero; it must refuse separately")
    # ⚠ ORDER IS NOT ENOUGH: the arm must still REFUSE. Every assertion above stayed green
    # with the refusal deleted outright — measured both before and after #357 introduced the
    # bypass, so this half was pre-existing rather than new (#359).
    assert "exit 5" in code[ever:max(flags)], (
        "the zero-issues arm no longer refuses; ordering alone does not stop the claim")


def test_the_empty_tracker_bypass_stays_opt_in():
    """⚠⚠ #344's REFUSAL IS NOW KEPT BY A DEFAULT, AND NOTHING WAS WATCHING IT.

    `--allow-empty-tracker` (#357) turns the refusal above into a pass — correctly, for the
    one case that reaches the same zero legitimately: a correct slug on a repo that has
    simply never had an issue filed. But the whole safety content of that change is which way
    it defaults, and a one-character edit restores #344 exactly: the driver announces
    CONVERGED over a tracker holding no records, having read nothing.

    Measured (#359): with the default flipped to 1, every zero-argument test in this module
    stayed green. An escape hatch is only an escape hatch while somebody has to ask for it.
    """
    # ⚠ PRESENCE FIRST, so the assertion below cannot be satisfied by the arm's absence —
    # deleting the flag would otherwise leave "the default is 0" trivially true.
    assert "--allow-empty-tracker) ALLOW_EMPTY_TRACKER=1;" in SOURCE, \
        "the opt-in bypass arm is gone"
    d = re.search(r"^ALLOW_EMPTY_TRACKER=(\S+)", SOURCE, re.M)
    # ⚠ `set -u` is on, so an undeclared default aborts the driver rather than defaulting off.
    assert d, "the bypass default is not declared at all"
    assert d.group(1) == "0", \
        f"the empty-tracker bypass is no longer opt-in (default {d.group(1)!r}): #344 is bypassed"


def test_the_reviewer_has_a_sanctioned_way_to_say_there_is_no_blocker():
    """⚠⚠ A PROHIBITION WITHOUT AN AFFORDANCE HALTED A CLEAN CYCLE.

    §6 forbids the token "in any other context" because the driver greps for it anywhere — and
    gave no approved way to report the negative. On 2026-08-27 a review that found nothing
    blocking wrote *"there is no MUST-NOT-LAND finding"* and stopped the loop it was reporting
    as clean. One occurrence, in a sentence meaning the opposite.

    ⚠ The grep is NOT loosened, and that is the point: a false halt costs one cycle, a false
    pass can land work a reviewer said must not land, on a branch with no protection and a
    worker instructed to push. The fix is the missing phrase, plus a message that shows what
    matched so a §6 violation is diagnosable without opening the log.
    """
    persona = (REPO / "workflow" / "REVIEWER.md").read_text(encoding="utf-8")
    assert "Nothing here blocks this range." in persona, (
        "REVIEWER.md gives no sanctioned wording for 'no blocker', so the only vocabulary for "
        "it is the token that halts the cycle")
    code = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
    assert 'grep -n "MUST-NOT-LAND"' in code, (
        "the halt does not show which line matched, so a §6 violation cannot be told from a "
        "real refusal without reading the log")

"""`workflow/` — the merge gate and the tracker script.

These two are where the workflow stopped being prose. `workflow/WORKFLOW.md` describes a state
machine and two mandatory-comment rules; `issue.py` refuses the violations and
`merge_branch.sh` refuses the merge. This repo's most expensive recurring lesson is that **a
rule in a file is not an enforcement mechanism**, so what is pinned here is the refusing, not
the wording.

⚠ `merge_branch.sh` is the only thing in the repo that reaches `main` on its own. Until
2026-08-17 the property that made the lane safe was "nothing here can push"; now it is "nothing
merges while an issue is open, review or escalated". A narrower guard, and the reason the
assertions below are specific about it.
"""

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "workflow" / "scripts" / "merge_branch.sh"
ISSUE = REPO / "workflow" / "scripts" / "issue.py"
MERGE_SRC = MERGE.read_text(encoding="utf-8")
ISSUE_SRC = ISSUE.read_text(encoding="utf-8")
WORKFLOW_MD = (REPO / "workflow" / "WORKFLOW.md").read_text(encoding="utf-8")

# ⚠ COMMENTS STRIPPED BEFORE ANY SEARCH FOR AN OPERATION.
#
# The first version of this file searched the raw text and produced four false failures at
# once: `merge_branch.sh`'s header explains why the worker used to be denied `git push`, and
# `git push` in that sentence is prose. Every one of those tests was reading documentation and
# reporting it as behaviour — the same substring-over-the-whole-file defect these tests were
# written to catch elsewhere. Search the code, not the file.
MERGE_CODE = "\n".join(ln.split("#", 1)[0] for ln in MERGE_SRC.splitlines())


# --- the merge gate ---------------------------------------------------------------------

def test_the_gate_is_fail_closed_on_unknown_states():
    """⚠ `state!="closed"`, NOT a list of the three open states.

    If a fifth state is ever added, `!="closed"` blocks the merge until someone decides what it
    means; `open||review||escalated` would silently ignore it and let the branch land. A merge
    gate must fail towards refusing.
    """
    assert 'state!="closed"' in MERGE_SRC
    assert '"escalated"' not in MERGE_SRC.split("UNSETTLED=")[1].split("PY\n")[0], \
        "the gate enumerates states instead of excluding closed — that is fail-open"


def test_an_unreachable_tracker_refuses_the_merge():
    """Unreachable is not clear. The opposite choice lands unreviewed work whenever
    PocketBase happens to be down — the failure being silent is what makes it serious."""
    assert "TRACKER_UNREACHABLE" in MERGE_SRC
    # ⚠ Anchored on the `$UNSETTLED` emptiness TEST, however it is spelled. Pinning the literal
    # `if [[ -n "$UNSETTLED"` broke the moment the condition grew a whitespace strip — a test
    # that fails on a rewording rather than a behaviour change.
    empty_check = re.search(r'if \[\[ -n "\$\{?UNSETTLED', MERGE_CODE)
    assert empty_check, "the emptiness check on $UNSETTLED was not found"
    assert MERGE_CODE.index("TRACKER_UNREACHABLE:*") < empty_check.start(), \
        "the unreachable check must precede the emptiness check, or 'down' reads as 'clear'"


def test_a_branch_with_no_issues_at_all_is_refused():
    """⚠ "NO OPEN ISSUES" AND "NEVER REVIEWED" ARE THE SAME READING.

    A branch nobody reviewed has zero open issues, exactly like one reviewed clean — so the
    gate cannot tell them apart and must not guess the flattering one. MEASURED on the
    `workflow` branch: once its open findings moved to another branch, the gate declared 21
    unreviewed commits settled and offered to push them to main.

    An override exists (`--allow-unreviewed`), because a genuinely clean review does file
    nothing and that case has to stay reachable — but it must be typed.
    """
    assert "NEVER_REVIEWED" in MERGE_CODE
    assert "ALLOW_UNREVIEWED" in MERGE_CODE
    assert "--allow-unreviewed" in MERGE_SRC


def test_the_gate_states_what_it_does_not_prove():
    """Closed issues show a review ran at SOME point, not that one covered the tip being
    merged — nothing records which commit was reviewed. Saying so is the difference between a
    gate and a rubber stamp."""
    assert "NOT that a review covered" in MERGE_SRC


def test_the_gate_runs_before_any_git_write():
    """A stale local view must not be able to authorise a merge."""
    for op in ("git checkout", "git merge --no-ff", "git push"):
        assert MERGE_CODE.index("UNSETTLED=") < MERGE_CODE.index(op), op


def test_the_push_names_both_ends_of_the_refspec():
    """⚠ `push.default=upstream` is set in this repo, so a bare `git push` from a branch that
    inherited `origin/main` as its upstream sends it to main whatever the branch is called.
    Naming both ends means what lands is what was just merged and gated."""
    m = re.search(r"git push \S+ (\S+)", MERGE_CODE)
    assert m and ":" in m.group(1), "the push must use an explicit src:dst refspec"


def test_it_parses():
    assert subprocess.run(["bash", "-n", str(MERGE)]).returncode == 0
    ast.parse(ISSUE_SRC)


# --- the tracker script -----------------------------------------------------------------

def _transitions():
    """`TRANSITIONS` as the script actually defines it, read from the AST rather than imported
    — importing would authenticate to PocketBase at module scope."""
    tree = ast.parse(ISSUE_SRC)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TRANSITIONS":
            return ast.literal_eval(node.value)
    raise AssertionError("TRANSITIONS not found in issue.py")


def test_escalated_has_no_route_back_to_open():
    """⚠ Only the owner releases an escalation, and a server-side hook performs it.

    A subcommand that moved `escalated -> open` would let an agent quietly un-ask a question it
    raised, which is the one thing escalation exists to prevent.
    """
    for action, rule in _transitions().items():
        assert "escalated" not in rule["from"], \
            f"{action} claims to move an issue out of 'escalated'"


def test_reopen_and_escalate_require_a_comment():
    """The two mandatory-comment rules, as refusals rather than sentences.

    `reopen` spends one of Ozzy's three attempts; without a reason it is spent on a guess.
    `escalate` asks the owner a question; one with no question attached cannot be answered.
    """
    src = ISSUE_SRC
    assert 'require_comment(args, "reopen")' in src
    assert 'require_comment(args, "escalate")' in src
    fn = src[src.index("def require_comment"):src.index("def show_row")]
    assert "die(" in fn, "require_comment must refuse, not warn"


def test_the_counter_only_moves_up_by_one_and_stops_at_the_cap():
    """`take` is the only writer of `agent_passes`, and the cap is what bounds the loop."""
    fn = ISSUE_SRC[ISSUE_SRC.index("def cmd_take"):ISSUE_SRC.index("def transition")]
    assert "cur + 1" in fn, "the counter must advance by exactly one"
    assert ">= MAX_PASSES" in fn, "the cap must be enforced where the counter moves"
    # ⚠ The WRITE form, `"agent_passes":`. Counting the bare name also counts the READ
    # (`rec.get("agent_passes")`) and fails against correct code — which it did.
    assert fn.count('"agent_passes":') == 1, "take must write the counter exactly once"
    # ⚠ `file` legitimately writes `"agent_passes": 0` — a new issue starts at zero, and the
    # workflow names that as one of the three fields filing must set. Excluding it by hand
    # rather than loosening the assertion: everything OTHER than these two must not touch it.
    filing = ISSUE_SRC[ISSUE_SRC.index("def cmd_file"):ISSUE_SRC.index("def cmd_take")]
    others = ISSUE_SRC.replace(fn, "").replace(filing, "")
    assert '"agent_passes":' not in others, "a third place writes agent_passes"


def test_nothing_writes_user_decision():
    """It is the owner's field. An agent writing there forges the answer to its own question."""
    assert '"user_decision":' not in ISSUE_SRC


def test_there_is_no_delete():
    """⚠ The tracker is the sole record that a finding ever existed.

    Checked as CODE, not as text: the docstring says "there is no delete subcommand and there
    must never be one", and a naive search for the word finds that sentence and fails.
    """
    tree = ast.parse(ISSUE_SRC)
    names = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "delete" not in names
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value.strip().upper() != "DELETE", "an HTTP DELETE is constructed"
    assert 'add("delete")' not in ISSUE_SRC and 'add_parser("delete")' not in ISSUE_SRC


def test_filing_sets_the_three_fields_the_workflow_requires():
    """`state: open`, `agent_passes: 0`, and the branch — the workflow names all three, and
    each was previously something a reviewer had to remember."""
    fn = ISSUE_SRC[ISSUE_SRC.index("def cmd_file"):ISSUE_SRC.index("def cmd_take")]
    assert '"state": "open"' in fn
    assert '"agent_passes": 0' in fn
    assert '"branch_name": branch' in fn


def test_the_issue_number_collision_is_retried():
    """⚠ `number` does not auto-assign and is unique per repo, so two writers filing at once
    collide. Retried rather than pre-reserved: the unique index is the real arbiter, and a
    reserved-then-abandoned number leaves a permanent hole in the sequence."""
    fn = ISSUE_SRC[ISSUE_SRC.index("def cmd_file"):ISSUE_SRC.index("def cmd_take")]
    assert "for attempt in range(" in fn


# --- the map ----------------------------------------------------------------------------

def test_the_workflow_map_matches_the_states_the_code_enforces():
    """WORKFLOW.md is the map; drift between it and the scripts is the failure this catches."""
    for state in ("open", "review", "escalated", "closed"):
        assert f"`{state}`" in WORKFLOW_MD
    assert "workflow/scripts/merge_branch.sh" in WORKFLOW_MD


# --- portability ------------------------------------------------------------------------

CONFIG = (REPO / "workflow" / "config.env").read_text(encoding="utf-8")


def test_the_repo_slug_is_not_hardcoded_anywhere_in_the_lane():
    """⚠ THE ONE FAILURE A COPY-PASTE PORT PRODUCES SILENTLY.

    `workflow/` is meant to be copied into another repo wholesale. A hardcoded slug survives
    that copy and files the NEW repo's issues against the OLD one, where they look entirely
    normal — right title, right branch, right author — and nothing ever flags them. So the slug
    is derived from `git remote get-url origin` and no file in the lane may name one.
    """
    for path in sorted((REPO / "workflow").rglob("*")):
        if not (path.is_file() and path.suffix in (".sh", ".py")):
            continue
        code = "\n".join(l.split("#", 1)[0] for l in path.read_text(encoding="utf-8").splitlines())
        assert "Artificial-Humanity/" not in code, \
            f"{path.relative_to(REPO)} hardcodes a repo slug — it will not survive a port"
    # ⚠ config.env is EXEMPT, and only config.env. It is the file a port is expected to edit,
    # and `SIBLING_REPO_CANDIDATES` legitimately names this lab's paths — which is precisely
    # why those paths were moved OUT of request_review.sh and into it.


def test_config_leaves_the_slug_empty_so_it_derives():
    assert re.search(r"^REPO_SLUG=\s*$", CONFIG, re.M), \
        "REPO_SLUG must be empty in config.env so it derives from the origin remote"


def test_every_setting_has_a_fallback_in_the_scripts():
    """An unedited copy must run. Each setting is read with a default beside it."""
    issue = ISSUE_SRC
    for key, src in (("MAX_PASSES", issue), ("COMMENT_MAX", issue)):
        assert re.search(r'CFG\.get\("%s"\)\s*or\s*\d+' % key, src), key
    assert 'BASE_BRANCH="${BASE_BRANCH:-main}"' in MERGE_CODE


def test_porting_instructions_exist_and_name_the_untravelled_parts():
    """A port that silently lacks the hook turns escalation into a one-way door."""
    for needed in ("Porting this lane", "@workflow/DEVELOPER.md", "config.env",
                   "user_decision", "DOES NOT TRAVEL"):
        assert needed in WORKFLOW_MD, needed


def test_escalation_comments_are_required_to_be_ste():
    """⚠ An escalation comment is addressed to the OWNER (owner, 2026-08-17) — the one thing in
    the tracker written to them rather than near them. The check is length-only and says so;
    the mechanism for the rest is the `ste` skill."""
    assert "ste_warnings" in ISSUE_SRC
    assert "ASD-STE100" in ISSUE_SRC
    assert "NECESSARY, NOT SUFFICIENT" in ISSUE_SRC
    dev = (REPO / "workflow" / "DEVELOPER.md").read_text(encoding="utf-8")
    assert "ESCALATION COMMENT IS THE EXCEPTION" in dev


# --- the full code review ----------------------------------------------------------------

FULL = REPO / "workflow" / "scripts" / "full_review.sh"
FULL_SRC = FULL.read_text(encoding="utf-8")
FULL_CODE = "\n".join(l.split("#", 1)[0] for l in FULL_SRC.splitlines())
LAUNCHER = (REPO / "workflow" / "scripts" / "request_review.sh").read_text(encoding="utf-8")


def test_full_review_cuts_a_dated_branch_with_no_upstream():
    """⚠ `--no-track` is not optional here. `push.default=upstream` is set in this repo, so a
    branch inheriting `origin/main` sends a bare `git push` straight to main whatever it is
    called — measured. A review branch is the last thing that should have that property."""
    assert 'BRANCH="review-$DATE"' in FULL_CODE
    assert "--no-track" in FULL_CODE
    assert 'date +%F' in FULL_CODE


def test_full_review_resumes_an_existing_sweep_rather_than_refusing():
    """A second run on the same date is how a sweep CONTINUES: pass 1 filed issues, Ozzy fixed
    some, this is the next review. Refusing would make the obvious command wrong on its second
    use — which is the use that matters."""
    assert 'git show-ref --verify --quiet "refs/heads/$BRANCH"' in FULL_CODE
    assert "resuming" in FULL_SRC


def test_full_review_dry_run_answers_on_a_dirty_tree():
    """⚠ A dirty tree is a reason not to SWITCH BRANCHES, not a reason to refuse to say what
    would happen. `merge_branch.sh` had the identical bug; both now check after the dry run."""
    dirty = FULL_CODE.index("git status --porcelain")
    guard = FULL_CODE.index('if [[ "$DRY_RUN" -eq 0 ]]')
    assert guard < dirty, "the dirty-tree check must sit inside the not-a-dry-run branch"


def test_the_range_guards_are_skipped_not_loosened_under_full():
    """A fresh review branch has ZERO commits ahead — the exact state the range guards refuse.
    They stay strict for range reviews and are bypassed wholesale for a full one."""
    assert 'if [[ "$FULL" -eq 0 ]]; then' in LAUNCHER
    # ...and the guards themselves are unchanged.
    assert "--range must be a TWO-dot range" in LAUNCHER
    assert "A bare ref is refused on purpose" in LAUNCHER


def test_variables_the_full_path_skips_are_initialised():
    """⚠ `set -u` turns a read of an unassigned variable into a fatal error with a line number
    and no cause. `COMMITS` was assigned only inside the range branch and read outside it —
    the second time this exact shape bit this script, after `$BRANCH`."""
    for var in ("COMMITS=", "COMMIT_COUNT=0", "TIP="):
        i = LAUNCHER.index(var)
        assert i < LAUNCHER.index('if [[ "$FULL" -eq 0 ]]; then'), \
            f"{var} must be initialised before the block that may skip it"


def test_the_full_brief_carries_an_inventory_and_no_diffstat():
    """An empty diffstat would read as "nothing changed" rather than "this axis does not
    apply"."""
    assert "THIS IS A FULL CODE REVIEW. THERE IS NO COMMIT RANGE." in LAUNCHER
    assert "tracked files" in LAUNCHER
    assert "what you covered" in LAUNCHER.lower() or "what you did not" in LAUNCHER


def test_the_brief_is_built_without_nesting_heredocs_in_the_brief_string():
    """⚠ MEASURED: an unquoted heredoc inside `$( )` inside the double-quoted `BRIEF=` lost its
    backslashes — markdown fences rendered as `\\\\\\` and the inventory vanished entirely,
    while the script still exited 0. The scope blocks are plain variables now."""
    assert "${SCOPE_LINES}" in LAUNCHER and "${SCOPE_BODY}" in LAUNCHER
    brief = LAUNCHER[LAUNCHER.index('BRIEF="## This run'):]
    brief = brief[:brief.index('\n"\n')]
    assert "<<" not in brief, "no heredoc may be opened inside the BRIEF string"


def test_the_reviewer_persona_knows_about_full_reviews():
    """The launcher's brief and the persona are a machine contract split across two files."""
    rev = (REPO / "workflow" / "REVIEWER.md").read_text(encoding="utf-8")
    assert "FULL CODE REVIEW" in rev
    assert "Porting this lane" in WORKFLOW_MD


def test_full_review_is_declared_in_the_manifest():
    manifest = (REPO / "scripts" / "pipeline_manifest.py").read_text(encoding="utf-8")
    assert "workflow/scripts/full_review.sh" in manifest


# --- the crash handler's count (issue found 2026-08-18) ----------------------------------

def test_the_crash_handler_counts_everything_not_closed():
    """⚠ THE ONE QUERY THAT ONLY EVER RUNS WHEN A REVIEW HAS ALREADY FAILED.

    `request_review.sh` answers "did the crashed review leave findings behind?" by counting
    issues on the branch. Scoped to `state="open"` it returned **0 on any branch that had been
    through a fix pass** — because a fix pass moves every issue to `review`, which is exactly
    when a crashed review is most likely and most damaging.

    Measured: a 529 killed review pass 3 of `pydantic-boundaries` with FIVE findings sitting in
    `review`, and the handler printed *"Nothing is filed … treat the range as unreviewed …
    push anyway"*. Against the live tracker, `state="open"` answered 0 and `state!="closed"`
    answered 5.

    ⚠ The bug survived because this path is unreachable in a normal run. Nothing exercises a
    crash handler except a crash, so its query had never been asked in the state that matters —
    the guard-liveness class, in the guard that reports on other guards failing.
    """
    src = (REPO / "workflow" / "scripts" / "request_review.sh").read_text(encoding="utf-8")
    assert 'branch_name="%s" && state!="closed"' in src, (
        "the crash handler's count is not scoped to `state!=\"closed\"` — if it filters on "
        "`open`, it reports 0 on every branch that has had a fix pass and tells the worker to "
        "push unreviewed work")
    # ⚠ COMMENTS STRIPPED FIRST, and the first draft of this line did not — it matched the
    # ⚠-comment three lines above, which QUOTES `state="open"` to explain the bug. Prose that
    # quotes the old code is not the old code; this is the fourth time that shape has cost
    # something this week, and it is what M4 of notes/quality-mechanisms-plan.md is for.
    helper = src.split("pb_helper()")[1].split("\n}")[0]
    code = "\n".join(ln for ln in helper.splitlines() if not ln.lstrip().startswith("#"))
    assert 'state="open"' not in code, (
        "pb_helper still carries a `state=\"open\"` filter in executable code")


def test_the_three_lane_scripts_agree_on_what_unfinished_means():
    """⚠ ONE DEFINITION, THREE READERS. `merge_branch.sh` gates the merge, `issue.py` lists
    what is left, and `request_review.sh` reports after a crash. All three are answering the
    same question — *is there anything unresolved on this branch?* — and a disagreement
    between them is silent: each looks right on its own.

    `merge_branch.sh` states the reasoning (fail-closed: a state added later stays counted
    rather than dropping out of every check at once). The other two must not drift from it.
    """
    for rel in ("workflow/scripts/merge_branch.sh", "workflow/scripts/issue.py",
                "workflow/scripts/request_review.sh"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert 'state!="closed"' in src, f"{rel} no longer uses the fail-closed definition"

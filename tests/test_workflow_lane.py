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

import pytest
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


# --------------------------------------------------------------------------------------
# The tracker script must not degrade a refused query into an empty answer.
#
# WHY (2026-08-18). `cmd_show` queried `issue_comments` with `sort=created`, which that
# collection does not have, so PocketBase answered HTTP 400 for EVERY issue. The status was
# assigned to `st` and never read, so `show` printed zero comments from the day it was
# written -- while 28 comment records sat in the collection, four of them on #92 and two on
# #100. `reopen` and `escalate` REFUSE to run without a comment; the one channel the rules
# make mandatory was the one the worker could not read.
#
# `cmd_escalated` had the other half of the shape: `(r.get("items") or []) if st == 200
# else []`, which renders a broken query as "nothing is escalated" -- the most reassuring
# possible output for the queue the owner reads to find what is waiting on them.
#
# ⚠ AST, NOT A TEXT SCAN. This comment contains both offending idioms verbatim, and a text
# scan would match it -- the trailing-comment defect this repo keeps re-learning.

def _call_status_targets(tree):
    """-> [(status_name, lineno, enclosing_function)] for every `st, x = ....call(...)`."""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt, val = node.targets[0], node.value
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2):
                continue
            if not (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute)
                    and val.func.attr == "call"):
                continue
            if isinstance(tgt.elts[0], ast.Name):
                out.append((tgt.elts[0].id, node.lineno, fn))
    return out


def test_every_tracker_response_status_is_read():
    """A status captured and never read is a query whose failure has no consequence."""
    tree = ast.parse(ISSUE_SRC)
    targets = _call_status_targets(tree)
    assert len(targets) >= 6, f"only {len(targets)} .call() sites found — is the walk working?"
    unread = []
    for name, lineno, fn in targets:
        # the window ends at the next assignment to the same name in the same function
        later = [ln for nm, ln, f in targets if f is fn and nm == name and ln > lineno]
        end = min(later) if later else 10 ** 9
        reads = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Name) and n.id == name
                 and isinstance(n.ctx, ast.Load) and lineno < n.lineno < end]
        if not reads:
            unread.append(f"  {fn.name}:{lineno} captures {name!r} and never reads it")
    assert not unread, (
        "these tracker queries discard their HTTP status, so a refused query is "
        "indistinguishable from an empty answer:\n" + "\n".join(unread))


def test_no_reader_degrades_a_refused_query_to_an_empty_result():
    """The `... if st == 200 else []` shape, as an AST pattern rather than a string.

    A refused query and a genuine zero must not print the same thing. This is the same
    silent-negative that made a crash handler report "0 issues filed" with five present.
    """
    tree = ast.parse(ISSUE_SRC)
    statuses = {name for name, _l, _f in _call_status_targets(tree)}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        names = {n.id for n in ast.walk(node.test)
                 if isinstance(n, ast.Name) and n.id in statuses}
        if not names:
            continue
        for branch in (node.body, node.orelse):
            empty = ((isinstance(branch, (ast.List, ast.Tuple)) and not branch.elts)
                     or (isinstance(branch, ast.Dict) and not branch.keys))
            if empty:
                offenders.append(
                    f"  line {node.lineno}: a conditional on {sorted(names)} falls back to "
                    f"an empty literal")
    assert not offenders, (
        "a refused tracker query must fail loudly, not read as 'nothing matched':\n"
        + "\n".join(offenders))


def test_the_comment_query_does_not_sort_on_a_field_that_collection_lacks():
    """⚠ MEASURED, NOT STYLISTIC. `issue_comments` carries author/body/issue/posted_at/seq
    and no `created` — PocketBase returns 400 on a sort by a missing field, and that 400 is
    what hid every comment in the tracker."""
    tree = ast.parse(ISSUE_SRC)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if "issue_comments" in node.value and "sort=created" in node.value:
            bad.append(node.lineno)
        # the sort may be split across adjacent implicitly-concatenated literals
        if node.value.startswith("&sort=created") or node.value == "sort=created":
            bad.append(node.lineno)
    assert not bad, (
        f"issue_comments has no `created` field; sorting on it returns HTTP 400 "
        f"(lines {sorted(set(bad))}). Sort on `posted_at,seq`.")


def test_a_written_comment_carries_both_ordering_fields():
    """⚠ `seq` IS THE FIELD THE REVIEWER'S DOCUMENTED QUERY SORTS ON (issue #109).

    `issue_comments` has no `created`/`updated` system field — measured: its columns are
    exactly id, issue, author, body, posted_at, seq — so nothing stamps an order unless this
    script does. It stamped `posted_at` and not `seq`, which fixed `show` and left
    `REVIEWER.md` §4's `sort="seq"` returning every agent comment ahead of the findings it
    answers. AST, because the comment above names both fields and a text scan would match it.
    """
    tree = ast.parse(ISSUE_SRC)
    posts = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "call" and len(node.args) >= 2):
            continue
        path = node.args[0]
        if not (isinstance(path, ast.Constant) and isinstance(path.value, str)
                and "issue_comments" in path.value):
            continue
        method = node.args[1]
        if not (isinstance(method, ast.Constant) and method.value == "POST"):
            continue
        body = node.args[2] if len(node.args) > 2 else None
        keys = {k.value for k in body.keys
                if isinstance(k, ast.Constant)} if isinstance(body, ast.Dict) else set()
        posts.append(keys)
    assert posts, "no POST to issue_comments found — has the writer moved?"
    for keys in posts:
        missing = {"seq", "posted_at", "issue", "author", "body"} - keys
        assert not missing, (
            f"a comment is written without {sorted(missing)}. `seq` and `posted_at` are the "
            f"only ordering this collection has; a comment missing either is unplaceable in "
            f"its own thread.")


def test_issue_numbers_are_floored_against_the_export():
    """⚠ THE UNIQUE INDEX CANNOT SEE A FILE ON DISK (issue #168).

    `cmd_file`'s six-attempt retry is the whole of its collision safety, and it only fires on
    a LIVE record. `notes/tracker-export-2026-08-17.json` holds 79 records numbered 12–120;
    nothing on the allocation path opens it. So on an empty collection — `items: []`, HTTP
    200, no error, no warning — allocation started at 1 and marched cleanly up through the
    reserved band, reissuing numbers that name different findings in the export.

    Reachable rather than theoretical: this collection was wiped once, on 2026-08-17.
    """
    tree = ast.parse(ISSUE_SRC)
    floor = next((n.value.value for n in ast.walk(tree)
                  if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "NUMBER_FLOOR" for t in n.targets)
                  and isinstance(n.value, ast.Constant)), None)
    assert floor is not None, "NUMBER_FLOOR is gone — allocation is unfloored again"

    # ⚠ NOT `if export.exists():` (issue #171). The export is the ONLY thing that makes the
    # floor a measurement rather than a guess, so wrapping the comparison in a silent
    # conditional made the one real assertion optional — and it would vanish exactly on a
    # checkout where the file was lost, which is the case where a wrong floor does damage.
    # The file is tracked, so its absence is a fault to report, not a reason to check less.
    import json

    export = REPO / "notes" / "tracker-export-2026-08-17.json"
    assert export.exists(), (
        f"{export.name} is missing — it is tracked, and without it NUMBER_FLOOR cannot be "
        f"checked against anything. This test would otherwise pass while proving nothing.")
    data = json.loads(export.read_text(encoding="utf-8"))
    rows = data["issues"] if isinstance(data, dict) and "issues" in data else data
    highest = max(int(r["number"]) for r in rows)
    assert floor >= highest, (
        f"NUMBER_FLOOR is {floor} but the export reaches #{highest}; allocation could "
        f"reissue a number that already names a different finding")

    # the floor must actually be APPLIED, not merely defined
    assign = [n for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "n"]
    assert any("NUMBER_FLOOR" in ast.unparse(a.value) for a in assign), (
        "NUMBER_FLOOR is defined but the allocation no longer uses it")


def test_moving_to_review_warns_at_zero_and_is_silent_above_it():
    """⚠ `agent_passes` IS THE CAP'S MECHANISM AND NOTHING WATCHED IT (2026-08-19).

    A full fix pass went by with `take` skipped: four issues were fixed, commented and moved
    to `review`, one of them still reading 0. The REVIEWER noticed, not the tooling — and
    `review_cycle.sh` treats "the worker failed to advance agent_passes" as a stop condition,
    so an unattended run would have halted on it.

    ⚠⚠ THIS IS BEHAVIOURAL BECAUSE TWO STRUCTURAL VERSIONS BOTH FAILED, AND FAILED IN THE
    WAYS THIS REPO ALREADY HAD NAMES FOR.

      * v1 asserted `"die(" not in ast.unparse(fn)` over the whole function — constraining
        the implementation, not the behaviour, and blind to `sys.exit` or `raise` (#172).
      * v2 scoped that to the branch and scanned `ast.unparse(body)` for stop tokens — but
        the branch's one statement is a `print` whose argument is a five-line operator
        message, and `unparse` emits literals verbatim, **so the scan ran over the prose**
        (#174). A TEXT SCAN OVER CODE, inside a guard, in the repo whose doc-claims gate
        exists because of exactly that.
      * and neither constrained the PREDICATE: `== -1`, `> 3` and `and False` all satisfy
        "the test mentions agent_passes, the body prints, no stop token appears", while the
        warning never fires again (#172, returned).

    Running it settles all three at once and scans nothing. The property is one sentence:
    **warns at 0, silent at 1 and above.**
    """
    import contextlib
    import importlib.util
    import io
    import types

    spec = importlib.util.spec_from_file_location("_issue_under_test", ISSUE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run(passes):
        """-> what cmd_review wrote to stderr for an issue at this counter."""
        rec = {"id": "x", "number": 1, "state": "open", "agent_passes": passes}
        fake = types.SimpleNamespace(
            find=lambda args, number: rec,
            patch=lambda rec_id, body: None,
            add_comment=lambda args, r, text, author: None,
        )
        args = types.SimpleNamespace(number=1, repo="r", comment="", author="Ozzy")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            mod.cmd_review(fake, args)
        return err.getvalue()

    assert "agent_passes = 0" in run(0), (
        "moving an issue to `review` at agent_passes = 0 says nothing — a pass that spent no "
        "attempt is unrecorded, and review_cycle.sh halts on exactly that")
    for counted in (1, 2, 3):
        assert run(counted) == "", (
            f"the warning fires at agent_passes = {counted}; it must be silent once the pass "
            f"has been counted, or it is noise and gets ignored")


def test_the_zero_warning_does_not_refuse():
    """⚠ A WARNING, NOT A REFUSAL. Zero is also what the owner's dial reads after a
    deliberate re-arm, and refusing would make the tooling argue with them. Asserted by
    RUNNING it — the transition must still happen."""
    import contextlib
    import importlib.util
    import io
    import types

    spec = importlib.util.spec_from_file_location("_issue_under_test2", ISSUE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    patched = {}
    rec = {"id": "x", "number": 1, "state": "open", "agent_passes": 0}
    fake = types.SimpleNamespace(
        find=lambda args, number: rec,
        patch=lambda rec_id, body: patched.update(body),
        add_comment=lambda args, r, text, author: None,
    )
    args = types.SimpleNamespace(number=1, repo="r", comment="", author="Ozzy")
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        mod.cmd_review(fake, args)
    assert patched.get("state") == "review", (
        "cmd_review refused the transition at agent_passes = 0. It must only warn — 0 is "
        "also the owner's deliberate re-arm value")


# --- every path the lane names must exist ------------------------------------------------
#
# ⚠ WHY (2026-08-19, issue #177). `request_review.sh` ran
# `python3 "$REPO_ROOT/scripts/lib/find_changeset.py" 2>/dev/null || true` against a file
# deleted with the mechanism it probed. The redirect swallowed `No such file`, so the variable
# was empty on every run and the brief silently took its fallback arm — telling every reviewer
# the work "cannot be shown to be finished", on more than twenty consecutive passes.
#
# The closest existing guard says this finding almost verbatim:
# `test_review_cycle.py::test_every_denied_script_actually_exists` — "⚠ A DENY ON A PATH THAT
# DOES NOT EXIST IS NOT A DENY. The failure above was invisible because nothing tied the deny
# list to the filesystem." It is scoped to one array in one file. This is the same rule for
# every path the lane names.

LANE_FILES = sorted(list((REPO / "workflow").rglob("*.sh")) + list((REPO / "workflow").rglob("*.py")))

# A repo-relative path to a concrete file: a real top-level directory, then a known
# extension. Measured when written: 38 literals across the lane, 0 absent.
#
# ⚠ `ya?ml` IS IN THE LIST BECAUSE `configs` IS (issue #179). The directory list and the
# extension list are one rule, and `configs/` holds YAML almost exclusively — so naming the
# directory while omitting its extension made the guard claim a coverage its pattern did not
# deliver. Measured before widening: 38 hits either way, 0 absent, so this adds reach without
# adding noise today.
_LANE_PATH = re.compile(
    r'(?:\$REPO_ROOT/|\$\{REPO_ROOT\}/|"\s*)?'
    r'((?:scripts|workflow|tests|matcha|configs)/[A-Za-z0-9_./-]+\.(?:py|sh|json|md|env|ya?ml))')


def _lane_path_literals(root=None):
    """(file, line, path) for every repo-relative path named in LANE code, comments excluded.

    ⚠ `root` EXISTS SO THE EXTRACTOR CAN BE TESTED (issues #178, #161, #172). Three guards on
    this branch were written by copying the scan into the test, which asserts about a
    reimplementation: change the real one and the test stays green. Measured — both mutants
    of the first version of this pair were MISSED for exactly that reason.
    """
    out = []
    files = (LANE_FILES if root is None else
             sorted(list((root / "workflow").rglob("*.sh")) + list((root / "workflow").rglob("*.py"))))
    base = REPO if root is None else root
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            # ⚠ THE RULE IS "EXPLANATORY `#` COMMENTS ARE EXEMPT; EVERYTHING THE LANE
            # EMITS OR RUNS IS NOT" — and the first version of this comment described only
            # half of it (issue #178). `#` comments are stripped because three of them in
            # this file QUOTE the deleted `find_changeset.py` path to explain the defect; a
            # scan that read its own explanation as a violation is the trailing-comment trap.
            #
            # ⚠ PROSE INSIDE A STRING IS DELIBERATELY *NOT* EXEMPT, and that is a choice
            # rather than an oversight. The eight such sites in the lane today — brief text
            # telling the reviewer to read `workflow/REVIEWER.md`, `merge_branch.sh`'s help
            # naming `request_review.sh`, `issue.py`'s docstring naming `workflow/config.env`
            # — are all LIVE POINTERS A READER FOLLOWS. A stale one is the same defect as a
            # stale probe, and #175 on this very branch was exactly that: a comment citing
            # the wrong REVIEWER.md section. The commit that added this called the exclusion
            # "mutation-tested both directions" when both directions tested were `#`
            # comments; `test_prose_naming_a_missing_path_is_caught` below is the direction
            # that was missing.
            # ⚠ ONE MECHANISM, NOT TWO. A `startswith("#")` early-continue stood here and
            # was DEAD: `split("#", 1)[0]` already yields "" for a whole-line comment, so
            # removing the guard changed nothing — measured, the mutant was a no-op. Dead
            # code reads as load-bearing to the next editor, which is the argument that took
            # `EXEMPT_FILES` out of test_schemas.py (issue #147).
            code = line.split("#", 1)[0]
            for m in _LANE_PATH.finditer(code):
                out.append((f.relative_to(base).as_posix(), i, m.group(1)))
    return out


def test_the_path_scan_still_finds_the_lane():
    """The assertion below is "nothing is missing", which an empty scan satisfies for free."""
    found = _lane_path_literals()
    assert len(LANE_FILES) >= 4, f"only {len(LANE_FILES)} lane scripts — is the walk working?"
    assert len(found) >= 20, f"only {len(found)} path literals found — is the pattern working?"


def test_every_path_the_lane_names_exists():
    """⚠ A PATH THAT DOES NOT EXIST DOES NOTHING — silently if it is executed, misleadingly
    if it is read.

    Two halves, and the guard covers both: a probe, deny or call becomes a no-op —
    `2>/dev/null` hides a missing file exactly as well as it hides a failed query, which is
    why the original went unseen for two days and was reported for twenty — and a path named
    in prose sends a reader somewhere that is not there, which is #175 on this branch.
    """
    missing = [f"  {f}:{i}  {p}" for f, i, p in _lane_path_literals()
               if not (REPO / p).exists()]
    assert not missing, (
        "the lane names paths that do not exist:\n" + "\n".join(missing) +
        "\n\nEach is either a probe, deny or call that cannot do anything — and nothing at "
        "runtime will say so — OR a sentence sending a reader to a file that is not there. "
        "⚠ Check which before repairing: eight of these sites are prose (the brief, "
        "merge_branch.sh's help, issue.py's docstring), where nothing executes and the fix "
        "is the sentence.")


def test_prose_naming_a_missing_path_is_caught(tmp_path):
    """⚠ THE DIRECTION `6a9f7bc` CLAIMED TO HAVE TESTED AND HAD NOT (issue #178).

    It reported the comment exclusion "mutation-tested both directions"; both directions were
    `#` comments. Prose inside a shell string — the brief, help text, a docstring — was
    scanned as code and never exercised. It still is scanned, on purpose: those are pointers a
    reader follows, and this branch exists because one of them went stale. So the behaviour is
    asserted rather than assumed.
    """
    lane = tmp_path / "workflow" / "scripts"
    lane.mkdir(parents=True)
    (lane / "fake.sh").write_text(
        '#!/usr/bin/env bash\n'
        '# scripts/lib/gone_from_a_comment.py must NOT count\n'
        'BRIEF="read scripts/lib/gone_from_prose.py for the rules"\n',
        encoding="utf-8")

    found = [p for _f, _i, p in _lane_path_literals(tmp_path)]   # ⚠ THE REAL EXTRACTOR

    assert "scripts/lib/gone_from_prose.py" in found, (
        "a path named in brief prose is no longer scanned — the reviewer can be sent to a "
        "file that does not exist, which is the defect this branch was opened for")
    assert "scripts/lib/gone_from_a_comment.py" not in found, (
        "a path quoted in a `#` comment is being scanned — comments in this lane quote "
        "deleted paths to explain them, and flagging that is the trailing-comment trap")


# One sample per directory and one per extension the rule advertises. ⚠ `.sh` was ABSENT
# from the first version while the test was named for covering every extension (issue #180) —
# a coverage claim checked against a list that did not cover it. The completeness of THIS
# table is asserted below against the pattern itself, so widening the pattern without adding
# a sample now fails rather than quietly narrowing what is proved.
_COVERAGE_SAMPLES = [
    "scripts/lib/schemas.py",         # scripts · py
    "workflow/scripts/issue.py",      # workflow
    "tests/test_scm.py",              # tests
    "matcha/delivery.py",             # matcha
    "configs/data/libritts.yaml",     # configs · yaml
    "workflow/scripts/merge_branch.sh",  # sh   <- the one #180 caught missing
    "workflow/config.env",            # env
    "workflow/WORKFLOW.md",           # md
    "scripts/assets/bulk_bank.json",  # json
    "configs/model/matcha.yml",       # yml
]


def test_the_coverage_table_names_every_token_the_pattern_advertises():
    """⚠ THE TABLE IS ITSELF A CLAIM, AND #180 IS WHAT HAPPENS WHEN NOTHING CHECKS IT.

    The samples below assert "the pattern covers what it advertises". That is only true while
    the table is complete, and it was not: `.sh` was missing while the test's own name said
    every extension. So the tokens are read out of the pattern and every one must appear.
    """
    dirs = re.search(r"\(\?:([a-z|]+)\)/", _LANE_PATH.pattern).group(1).split("|")
    exts = re.search(r"\\.\(\?:([a-z?|]+)\)", _LANE_PATH.pattern).group(1).split("|")
    joined = " ".join(_COVERAGE_SAMPLES)
    for d in dirs:
        assert f"{d}/" in joined, f"no sample exercises the {d!r} directory"
    for e in exts:
        # `ya?ml` advertises two spellings; both must be sampled
        for spelling in (["yaml", "yml"] if e == "ya?ml" else [e]):
            assert f".{spelling}" in joined, f"no sample exercises the .{spelling} extension"


@pytest.mark.parametrize("sample", _COVERAGE_SAMPLES)
def test_the_pattern_covers_every_directory_and_extension_the_rule_claims(sample):
    """⚠ #179's DEFECT AS A PROPERTY. `configs` was in the directory list while `yaml` was
    absent from the extensions, so the guard claimed a coverage its pattern did not deliver —
    and nothing detected the omission, because the lane happens to name no YAML today. A
    coverage claim that only holds by accident of the current tree is not a guard.

    Every combination the rule advertises is asserted here, so narrowing either list fails.
    """
    assert _LANE_PATH.search(sample), (
        f"{sample!r} is inside the directories and extensions this guard claims to cover, and "
        f"the pattern does not match it")

#!/usr/bin/env bash
#
# merge_branch.sh — merge the current branch to main, but only once it clears the SEVERITY FLOOR.
#
#     FerroStep/workflow/scripts/merge_branch.sh [--branch B] [--base main] [--no-push] [--dry-run]
#                              [--allow-unreviewed] [--no-review]
#
# ⚠ THE GATE IS ON THE MERGE, NOT THE PUSH (owner, 2026-08-17). A branch that merged
# legitimately is one whose push is unremarkable, so this pushes by default. What it will not
# do is merge a branch that still carries a finding at or above the floor.
#
# ⚠ SINCE 2026-08-20 THAT IS A SEVERITY FLOOR, NOT ZERO-OPEN-ISSUES. The threshold is
# MERGE_SEVERITY_FLOOR in FerroStep/workflow/config.env and is deliberately NOT repeated here; at or
# above it blocks, below it rides, and UNGRADED or ESCALATED block at any severity. This
# header described the old gate for the length of the branch that replaced it (#202) —
# which is the same defect the branch was fixing, one file up.
#
# That is the whole safety model now, and it is narrower than the one it replaces. Until today
# the worker was denied `git push` outright and nothing it ran could reach `main`. Now it can —
# so this check is no longer one guard among several, it is the guard. The repo has no branch
# protection and force-push is unblocked (AGENTS.md §1).
#
# ⚠ THE MERGE COMMIT IS AUTHORED AS THE ROSTER'S DEVELOPER, WHOEVER RUNS THIS SCRIPT (#394).
# DEVELOPER.md §1 leaves the repo's configured identity as the owner's so their HAND commits
# stay theirs; a merge through this script is not a hand commit, it is the lane's act, and
# it is the one thing that reaches `main` on its own. There is deliberately no path through
# here that lands a merge under the invoker's name — on EITHER line. `GIT_AUTHOR_*` in the
# environment OVERRIDES a `-c` pair (measured 2026-09-07), and `GIT_COMMITTER_*` overrides
# the committer line the same way (measured 2026-09-08, #396: `-c user.name="Roster Dev"`
# merged, committer `Committer Person`). Both pairs are refused before the merge rather
# than caught after it, and both lines are checked before the push. The first version
# refused the author pair only and checked the author line only, while claiming "no path".
#
# ⚠⚠ THIS SCRIPT IS FOR AGENTS. THE OWNER DOES NOT RUN IT (owner, 2026-09-07, deciding #394).
# That is why there is no opt-in and no --as-invoker flag: there is no case to serve. ⚠ And
# do NOT reintroduce "merge by hand if the commit is meant to be yours" as the escape — an
# earlier wording said exactly that, and a bare `git merge` SKIPS the severity floor, the
# server-side tracker re-read and the explicit push refspec. This repo has no branch
# protection and force-push is unblocked, so that advice traded the only guard in front of
# `main` for a name in an author field.
#
# Replaces `changeset.sh merge`. The changeset record is retired: a branch already has an
# identity and its issues already carry its state.
#
set -euo pipefail

# --- Per-repo settings -----------------------------------------------------
# ⚠ ONE FILE TO EDIT WHEN PORTING THIS LANE. Sourced rather than hardcoded so that copying
# `FerroStep/workflow/` into another repo does not carry this repo's identity with it. REPO_SLUG is
# DERIVED from `origin` when config.env leaves it empty — a stale hardcoded slug would file
# the new repo's issues against the old one, where they look perfectly normal.
_WF_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.env"
# shellcheck disable=SC1090
[[ -r "$_WF_CFG" ]] && source "$_WF_CFG"
BASE_BRANCH="${BASE_BRANCH:-main}"
if [[ -z "${REPO_SLUG:-}" ]]; then
  _url="$(git remote get-url origin 2>/dev/null || echo '')"
  REPO_SLUG="$(printf '%s' "$_url" | sed -E 's#(\.git)?$##; s#^.*[:/]([^/:]+/[^/]+)$#\1#')"
fi
BASE="${BASE_BRANCH}"
BRANCH=""
PUSH=1
DRY_RUN=0
ALLOW_UNREVIEWED=0
NO_REVIEW=0

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)  BRANCH="${2:?--branch needs a value}"; shift 2 ;;
    --base)    BASE="${2:?--base needs a value}"; shift 2 ;;
    --repo)    REPO_SLUG="${2:?--repo needs a value}"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --allow-unreviewed) ALLOW_UNREVIEWED=1; shift ;;
    --no-review) NO_REVIEW=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "merge_branch.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "merge_branch.sh: $*" >&2; exit 1; }

# Names the GIT_AUTHOR_* / GIT_COMMITTER_* variable(s) set in the environment, or fails when
# none are. These OVERRIDE a `-c user.*` pair (measured 2026-09-07: `GIT_AUTHOR_NAME=Env git
# -c user.name=Ozzy commit` is authored Env; #396: the committer pair does the same to the
# committer line), so a merge made under them would land wrong and be caught only by the
# post-merge check, after `main` has moved. Both paths ask this BEFORE the merge instead.
env_identity_override() {
  local v set=""
  for v in GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL; do
    [[ -n "${!v:-}" ]] && set="${set:+$set, }$v"
  done
  [[ -n "$set" ]] && printf '%s' "$set"
}

command -v python3 >/dev/null 2>&1 || die "python3 is not on PATH."
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository."
cd "$REPO_ROOT"

[[ -n "$BRANCH" ]] || BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" != "HEAD" ]] || die "detached HEAD — the branch is the unit of work."
[[ "$BRANCH" != "$BASE" ]] || die "refusing to merge $BASE into itself."

# --- The gate --------------------------------------------------------------
# ⚠ CHECKED SERVER-SIDE, BEFORE ANYTHING TOUCHES GIT, so a stale local view cannot authorise a
# merge. Prints one line per unsettled issue, or nothing.
#
# ⚠ THE FILTER IS `state!="closed"`, NOT A LIST OF THE THREE OPEN STATES — and that is
# FAIL-CLOSED BY CHOICE. If a fifth state is ever added, `!="closed"` blocks the merge until
# someone decides what it means, whereas `open||review||escalated` would silently ignore it and
# let the branch land. A merge gate should fail towards refusing.
#
# ⚠ SINCE 2026-08-20 THIS IS A SEVERITY FLOOR, NOT ZERO-OPEN-ISSUES (owner, ratified
# 2026-08-19, built here). The threshold lives in config.env, not in this file. The
# ruling had lived as prose in the personas and in AI-Lab-AMD's blueprint for a day short of
# a fortnight while this script kept the old behaviour, and nothing failed — because a
# stricter gate never goes red. That is the whole reason the ruling needed a mechanism.
#
# ⚠ UNGRADED BLOCKS, exactly as a finding at the floor does. Every issue filed before 2026-08-20 has
# severity="" (measured: 111 records), and a floor that waves through what it cannot grade
# is not a floor. The three ways to clear an ungraded issue are: close it, grade it LOW, or
# decide it is not LOW.
#
# ⚠ ESCALATED BLOCKS AT ANY SEVERITY. It means the owner owes a decision; severity says how
# bad the finding is, not whether someone is waiting on a human.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ⚠⚠ CALL FOR A REVIEW IF THE TIP IS NOT COVERED (owner, 2026-08-22).
#
# This gate used to print "⚠ this proves nothing AT OR ABOVE THE FLOOR is outstanding — NOT
# that no finding is, and NOT that a review covered <sha>" and then merge anyway. It named
# its own hole and left it open: a clean tracker plus five new commits reads exactly like a
# reviewed branch, and the worker is the only thing standing between the two. DEVELOPER.md
# records that two commits reached `main` unreviewed that way.
#
# `request_review.sh` now records the tip it completed on. If that does not match HEAD, this
# RUNS the review rather than telling someone to — the instruction was already printed at the
# NEVER_REVIEWED branch below and printing it is what did not work.
#
# ⚠ `--dry-run` NEVER LAUNCHES ONE. A dry run that costs a full review is not a dry run, and
# nobody would use it twice. It reports what would happen.
# ⚠ `--no-review` opts out for the case the marker cannot cover: a review that genuinely ran
# on another machine or before this mechanism existed. It is the honest escape hatch, and it
# is separate from `--allow-unreviewed`, which answers a different question (no issues at all).
_REVIEW_MARK="$(git rev-parse --git-dir)/sonora-reviewed-tips"
_TIP="$(git rev-parse HEAD)"
_REVIEWED_TIP="$(grep "[[:space:]]${BRANCH}\$" "$_REVIEW_MARK" 2>/dev/null | tail -1 | cut -d' ' -f1 || true)"

if [[ "$_REVIEWED_TIP" != "$_TIP" && "$NO_REVIEW" -ne 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "merge_branch.sh: HEAD ($(git rev-parse --short HEAD)) is NOT recorded as reviewed"
    echo "  last reviewed tip for '$BRANCH': ${_REVIEWED_TIP:-<none recorded>}"
    echo "  would: FerroStep/workflow/scripts/request_review.sh --range $BASE..HEAD  (then re-check)"
  else
    echo "merge_branch.sh: HEAD is not recorded as reviewed — calling for a review first."
    echo "  last reviewed tip for '$BRANCH': ${_REVIEWED_TIP:-<none recorded>}"
    # ⚠ ONLY FLAGS request_review.sh ACTUALLY TAKES. The first version passed
    # `--branch-name`, which it does not accept — the call would have died "unknown
    # argument" and the `|| true` would have swallowed it, leaving the merge to proceed as
    # if a review had run. Caught by running the dry run, not by reading. It derives the
    # branch itself; `--repo` is passed so both agree on the tracker slug.
    if "$_SCRIPT_DIR/request_review.sh" --range "$BASE..HEAD" --repo "$REPO_SLUG"; then
      echo "merge_branch.sh: review completed — re-reading the tracker."
    else
      # ⚠ NOT `die`. A review that fails to COMPLETE may still have filed real findings, and
      # the tracker read below is what decides. Dying here would discard them.
      #
      # ⚠⚠ BUT IT MUST NOT SAY "review finished" (#285). The first version printed that
      # unconditionally, so a review that died on its first call announced success and left
      # the reader to infer a clean range from a tracker that is clean because nothing ran.
      # "Check the instrument RAN before believing a negative" is this lane's own rule.
      echo "merge_branch.sh: ⚠ THE REVIEW DID NOT COMPLETE (non-zero exit)." >&2
      echo "  It may still have filed findings — Janis writes them one at a time — so the" >&2
      echo "  tracker read below is what decides, and a clean result here does NOT mean the" >&2
      echo "  range was reviewed. No marker was recorded, so this will ask again." >&2
    fi
  fi
fi
# The configured floor, for the messages only — merge_floor.py is what ENFORCES it, and
# reads config itself. This is display, never a second source of truth.
_FLOOR="${MERGE_SEVERITY_FLOOR:-<unset — everything blocks>}"
UNSETTLED="$(python3 - "$REPO_SLUG" "$BRANCH" "$_SCRIPT_DIR" <<'PY'
import json, os, socket, sys, urllib.parse, urllib.request
socket.setdefaulttimeout(20)
# ⚠ IMPORTED OUTSIDE THE try, DELIBERATELY. Inside it, a missing merge_floor.py surfaced as
# "cannot read the tracker" — fail-closed, so the merge was still refused, but the message
# sent the reader to PocketBase for a missing file. An error that misnames its own cause
# costs the next person an hour in the wrong place.
sys.path.insert(0, sys.argv[3])
try:
    from merge_floor import rides as _rides
except ImportError as e:
    print("FLOOR_MODULE_MISSING: %s" % e); raise SystemExit(0)
try:
    env = json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"]["pocketbase"]["env"]
    base = env.get("PB_URL", "http://127.0.0.1:8090")
    def call(path, method="GET", body=None, token=None):
        r = urllib.request.Request(base + path, method=method)
        if token: r.add_header("Authorization", token)
        d = None
        if body is not None:
            d = json.dumps(body).encode(); r.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(r, d) as x: return json.loads(x.read() or b"{}")
    tok = call("/api/collections/_superusers/auth-with-password", "POST",
               {"identity": env.get("PB_EMAIL"), "password": env.get("PB_PASSWORD")})["token"]
    repo, branch = sys.argv[1].replace('"', ""), sys.argv[2].replace('"', "")
    # ⚠ "NO OPEN ISSUES" AND "NEVER REVIEWED" ARE THE SAME READING, and this gate cannot tell
    # them apart from the tracker alone: a branch nobody has reviewed has zero open issues,
    # exactly like one reviewed clean. Measured on this very branch — after its open findings
    # moved elsewhere, the gate reported 21 unreviewed commits as settled and offered to push
    # them to main. So a branch with NO issues at all is refused below.
    ever = call("/api/collections/issues/records?perPage=1&skipTotal=false&filter="
                + urllib.parse.quote('repo="%s" && branch_name="%s"' % (repo, branch)),
                token=tok).get("totalItems", 0)
    if ever == 0:
        print("NEVER_REVIEWED")
    flt = urllib.parse.quote('repo="%s" && branch_name="%s" && state!="closed"'
                             % (repo, branch))
    r = call("/api/collections/issues/records?perPage=200&skipTotal=false&sort=number"
             "&fields=number,state,title,severity&filter=" + flt, token=tok)
    items = r.get("items") or []
    if r.get("totalItems", 0) > len(items):
        raise RuntimeError("paged: %d of %d" % (len(items), r["totalItems"]))
    # The floor. Anything not positively known to be LOW-and-not-escalated blocks — an
    # unrecognised severity value lands here too, deliberately, so widening the field's
    # values in PocketBase cannot silently widen what this gate permits.
    # ⚠ IMPORTED, NOT RESTATED. merge_floor.py is the one copy of the rule and is unit-
    # tested there; a floor written out twice is one that disagrees with itself eventually.
    for i in items:
        sev = (i.get("severity") or "").strip().lower()
        rides = _rides(i.get("state"), sev)
        print("%s  #%-5s %-10s %-8s %s"
              % ("RIDE " if rides else "BLOCK", i["number"], i["state"],
                 sev or "UNGRADED", i["title"][:60]))
except Exception as e:
    # ⚠ UNREACHABLE IS NOT CLEAR. A tracker that cannot be read must refuse the merge, not
    # wave it through — the failure mode of the opposite choice is landing unreviewed work
    # whenever PocketBase happens to be down.
    print("TRACKER_UNREACHABLE: %s" % e)
PY
)"

if [[ "$UNSETTLED" == FLOOR_MODULE_MISSING:* ]]; then
  die "FerroStep/workflow/scripts/merge_floor.py is missing or unreadable, so the severity floor cannot
     be evaluated. This is a broken installation, NOT a tracker problem and NOT a clean
     branch. (${UNSETTLED#FLOOR_MODULE_MISSING: })"
fi

if [[ "$UNSETTLED" == TRACKER_UNREACHABLE:* ]]; then
  die "cannot read the tracker, so cannot tell a settled branch from an unsettled one.
     Refusing rather than guessing. (${UNSETTLED#TRACKER_UNREACHABLE: })"
fi

if [[ "$UNSETTLED" == *NEVER_REVIEWED* ]]; then
  [[ "$ALLOW_UNREVIEWED" -eq 1 ]] || die "'$BRANCH' has NO issues at all — not one was ever
     filed against it. That reads identically to 'reviewed and found clean', and this gate
     cannot tell the two apart, so it refuses rather than guessing the flattering one.
       Review it:  FerroStep/workflow/scripts/request_review.sh
       Or, if a review genuinely ran and found nothing:  --allow-unreviewed"
  UNSETTLED="${UNSETTLED/NEVER_REVIEWED/}"
fi

# ⚠ BLOCKING IS "EVERYTHING THAT IS NOT A RIDE", NOT "EVERYTHING TAGGED BLOCK" (#205).
# The first version grepped '^BLOCK ', which quietly DROPPED any line matching neither
# prefix — and the catch-all `die` on non-empty $UNSETTLED that used to make the imprecise
# checks above safe had been removed in the same commit. So an unrecognised line, a stray
# warning on stdout, or a future tag this classifier has not been taught would have let the
# branch land. Only a line the python explicitly marked RIDE may be discarded here.
RIDING="$(printf '%s\n' "$UNSETTLED" | grep '^RIDE ' || true)"
BLOCKING="$(printf '%s\n' "$UNSETTLED" | grep -v '^RIDE ' | grep -v '^[[:space:]]*$' || true)"

# ⚠ NAME WHAT RIDES, AND DO NOT NAME THEIR SEVERITY. The header said "these LOW issues
# RIDE" over a list the gate had just computed. Raise the floor and a mid-ladder finding
# rides too, so the header contradicted the severity column one line beneath it — and the
# header is the part that says what to do about them (#219). "below-floor findings ride" is only true if someone can
# see what rode;
# a gate that drops findings without printing them is how the follow-up never happens.
if [[ -n "${RIDING//[[:space:]]/}" ]]; then
  echo "merge_branch.sh: these findings are BELOW the floor and RIDE — they stay open after"
  echo "  the merge:"
  echo "$RIDING"
  echo "  ⚠ they do NOT disappear — move them to a follow-up branch, or they sit on a"
  echo "    branch_name that no longer has a branch."
fi

if [[ -n "${BLOCKING//[[:space:]]/}" ]]; then
  # ⚠ TWO DIFFERENT REFUSALS SHARE THIS BLOCK, AND THEY NEED DIFFERENT INSTRUCTIONS (#209).
  # A line the classifier did not recognise lands here by design — that is the fail-closed
  # property #205 restored. But it is NOT a finding, so "these block: … grade one" told the
  # reader to run `issue.py grade <N>` for an <N> that does not exist. Right refusal, wrong
  # remedy, and `:97-100` argues against exactly this shape eighty lines up.
  # ⚠ SEPARATE THE TWO CAUSES BEFORE SPEAKING ABOUT EITHER (#210). The first version branched
  # on "ANY line is unrecognised" and then printed a message asserting "ALL of this is
  # unrecognised" — so a mixed list got "there is nothing to grade" printed directly beneath
  # findings that were gradeable. #209's defect with the sign flipped, in #209's own fix.
  UNKNOWN="$(printf '%s\n' "$BLOCKING" | grep -v '^BLOCK ' || true)"
  FINDINGS="$(printf '%s\n' "$BLOCKING" | grep '^BLOCK ' || true)"
  if [[ -n "${UNKNOWN//[[:space:]]/}" ]]; then
    echo "merge_branch.sh: refusing '$BRANCH' — the tracker returned output this gate does" >&2
    echo "  not recognise, so it cannot tell a settled branch from an unsettled one:" >&2
    echo "$UNKNOWN" >&2
    echo "  ⚠ these lines are NOT findings and have no issue number to grade. Unrecognised" >&2
    echo "    output means the gate or the tracker changed shape — fix that, not a symptom." >&2
    if [[ -n "${FINDINGS//[[:space:]]/}" ]]; then
      # Both causes at once. Say so, and show them apart — a reader told "nothing to grade"
      # while real findings sit in the same block will believe the wrong half.
      echo "  AND, separately, these ARE findings below the floor:" >&2
      echo "$FINDINGS" >&2
      echo "    close them, or grade them if they are UNGRADED:" >&2
      echo "      FerroStep/workflow/scripts/issue.py grade <N> --severity <s>" >&2
    fi
    die "refusing is the designed behaviour here, not a failure."
  fi
  echo "merge_branch.sh: '$BRANCH' is below the merge floor — these block:" >&2
  echo "$BLOCKING" >&2
  die "resolve them first. Anything at or above the configured floor must be closed.
     UNGRADED must be graded (or closed) because a floor cannot pass what it cannot read;
     escalated means the owner owes a decision at any severity (FerroStep/workflow/WORKFLOW.md §4).
       An UNGRADED one:  FerroStep/workflow/scripts/issue.py grade <N> --severity <s>
     ⚠ That is for grading what has NO grade. It is not a way past a finding that HAS one:
       lowering an existing grade is refused for anyone but the reviewer, because it would
       make a blocking finding ride without closing it (#218). Disagree with a grade? Argue
       it in the comments — a review is a report, not an order."
fi

echo "merge_branch.sh: '$BRANCH' clears the merge floor (MERGE_SEVERITY_FLOOR=$_FLOOR) —
  nothing at or above it, nothing ungraded, nothing escalated."
# ⚠ SAY WHAT THIS DOES NOT PROVE. Closed issues show that a review ran at SOME point, not that
# one covered the commit about to land: nothing records which tip was reviewed. Ozzy is
# responsible for having requested a review of the range being merged; this gate only refuses
# to land KNOWN-open findings. Stating the limit here so the line above is not read as more.
# ⚠ THIS LINE SAID "no finding is outstanding" UNTIL THE FLOOR EXISTED, and the floor made
# it false: LOW findings ride, so they ARE outstanding and the branch lands anyway. A gate
# whose success message overstates what it checked is how the check gets trusted for more
# than it does — the exact defect this repo keeps paying for.
echo "  ⚠ this proves nothing AT OR ABOVE THE FLOOR is outstanding — NOT that no finding is,
    and NOT that a review covered $(git rev-parse --short HEAD)."

# What the reader DOES after "nothing to merge" (#399). The refusal below names a re-run after
# a `--no-push` merge as its usual trigger, and that reader came back FOR the push: the merge
# that cleared the gate is sitting on local $BASE, unpushed, possibly a session later. A
# message that reports the state and stops leaves two wrong readings open — "already landed"
# (it is local only) and "merge differently" (the hand merge the header forbids). So say
# whether $BASE is ahead of origin/$BASE and, if it is, name the push — the SAME refspec the
# push below uses. That is not a bypass: the gate is on the merge, not the push (header).
# Silent when origin/$BASE does not resolve (no remote — the test harness): a count against a
# ref that is not there is not a count. "As last fetched" because origin/$BASE is a local
# reading of the remote, and nothing here fetches. Defined HERE, below the gate, because it
# echoes the push refspec and `test_the_gate_runs_before_any_git_write` reads the source in
# order — a definition above the gate is indistinguishable from a push above it to a grep.
unpushed_hint() {
  local n
  git rev-parse --verify -q "origin/$BASE" >/dev/null 2>&1 || return 0
  n="$(git rev-list --count "origin/$BASE..$BASE")"
  if [[ "$n" -gt 0 ]]; then
    echo "$BASE is $n commit(s) ahead of origin/$BASE as last fetched — a merge that already"
    echo "     cleared the gate is local only. Push it:  git push origin $BASE:$BASE"
  else
    echo "$BASE is not ahead of origin/$BASE as last fetched; nothing is waiting to be pushed."
  fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  # ⚠ THE SAME COMMAND THE REAL MERGE RUNS, `-c` PAIR INCLUDED (#393). This printed a bare
  # `git merge --no-ff` for the whole of the commit that added the pair below — the command
  # the script ran BEFORE it, the one that authored merges on `main` as the owner — four
  # lines above a comment arguing that a preview of a different command is worse than none.
  # The roster is ASKED here, not required: a dry run still answers on a box with no roster
  # (the reason the real resolution sits after this block), and it says so, because that
  # refusal is exactly where the real merge would stop.
  if _DRY_ENV="$(ferrostep agent-env --agent developer --roster "$REPO_ROOT/FerroStep/config.yaml" 2>/dev/null)" \
     && eval "$_DRY_ENV" && [[ -n "${AGENT_NAME:-}" && -n "${AGENT_EMAIL:-}" ]]; then
    echo "  would: git checkout $BASE && git -c user.name=\"$AGENT_NAME\" -c user.email=\"$AGENT_EMAIL\" merge --no-ff $BRANCH"
  else
    echo "  would: git checkout $BASE && git -c user.name=<roster developer> -c user.email=<roster developer> merge --no-ff $BRANCH"
    echo "  ⚠ the roster did NOT resolve here (ferrostep agent-env --agent developer). The real"
    echo "    merge REFUSES at that point, with nothing merged."
  fi
  if _OVERRIDE="$(env_identity_override)"; then
    echo "  ⚠ $_OVERRIDE is set in the environment and would OVERRIDE that -c pair. The real"
    echo "    merge REFUSES before merging; unset it first."
  fi
  # ⚠ THE SAME ANSWER THE REAL MERGE GIVES FOR A NO-OP (#398), previewed rather than
  # refused: a dry run still reports the gate's verdict, and then says where the real run
  # would stop. rc 128 is a ref that does not resolve — named, because the merge below
  # would refuse there too, and a preview of a merge that cannot happen is the #393 defect.
  if git merge-base --is-ancestor "$BRANCH" "$BASE" 2>/dev/null; then
    echo "  ⚠ nothing to merge: '$BRANCH' is already contained in '$BASE'. The real merge"
    echo "    REFUSES at that point, with nothing merged and nothing to amend."
    unpushed_hint | sed 's/^/    /'
  elif [[ $? -ne 1 ]]; then
    echo "  ⚠ '$BRANCH' or '$BASE' does not resolve as a ref here. The real merge REFUSES at"
    echo "    that point, with nothing merged."
  fi
  echo "  then:  check the merge author AND committer are the roster developer, before any push"
  # ⚠ THE SAME REFSPEC THE REAL PUSH USES. This printed `git push origin $BASE` while the real
  # command is `origin "$BASE:$BASE"` — a dry run that describes a different command from the
  # one it previews is worse than no dry run at all, because it gets believed.
  [[ "$PUSH" -eq 1 ]] && echo "  would: git push origin $BASE:$BASE"
  exit 0
fi

# ⚠ CHECKED HERE, NOT AT THE TOP. A dirty tree is a reason not to MERGE, not a reason to
# refuse to ANSWER — and the first version put it first, so `--dry-run` could not report the
# gate's verdict without a clean tree, which is most of what a dry run is for.
[[ -z "$(git status --porcelain)" ]] \
  || die "working tree is dirty. Commit or stash first — a merge would carry edits that were
     never reviewed, which is precisely what the gate above exists to prevent."

# ⚠ NOTHING TO MERGE IS ITS OWN REFUSAL (#398), and it sits HERE for the reason the dirty-tree
# check does: a no-op is a reason not to MERGE, not a reason to refuse to ANSWER, so the dry
# run above still reports the gate's verdict and previews this refusal instead of making it.
# `merge --no-ff` of a branch already contained in $BASE prints "Already up to date." and
# creates NO commit, rc 0 — and the post-merge identity check below then read $BASE's
# EXISTING tip, said "the merge landed", and printed `--amend --reset-author` against a
# commit that is already on `main` and already pushed. Right refusal, wrong instruction, and
# the instruction rewrites `main`; this repo has no branch protection to stop it. A re-run
# after a `--no-push` merge, or a second run after a successful one, produced it every time.
# Explicit rc handling: `--is-ancestor` is 0/1 for the answer and 128 when a ref is missing,
# and a missing ref is not "not merged".
if git merge-base --is-ancestor "$BRANCH" "$BASE" 2>/dev/null; then
  die "nothing to merge: '$BRANCH' is already contained in '$BASE' — every commit on it is
     already reachable from $BASE's tip ($(git rev-parse --short "$BASE")). NOTHING WAS MERGED,
     and there is nothing to amend. $(unpushed_hint)"
elif [[ $? -ne 1 ]]; then
  die "cannot tell whether '$BRANCH' is already in '$BASE': \`git merge-base --is-ancestor\`
     failed. Do both refs exist? NOTHING WAS MERGED."
fi

# ⚠ THE MERGE COMMIT NEEDS THE ROSTER'S IDENTITY, AND NOTHING ELSE SUPPLIES IT.
# This repo's configured git identity is the OWNER's, deliberately, so that their own hand
# commits stay theirs — which means a merge made WITHOUT the `-c` pair below does not error.
# It lands under their name, silently. That is not hypothetical: merge commits on `main` are
# authored as the owner from exactly this gap, and this is the one script in the repo that
# reaches `main` on its own. DEVELOPER.md §1 states the rule for a hand commit; the script
# that automates the merge was not obeying it.
# ⚠ ASSIGNMENT THEN CHECK, never `eval "$(ferrostep agent-env)"` in one step — eval's status
# is the emitted text's status, a refusal emits nothing, and `eval ""` is 0 (measured
# 2026-08-24). request_review.sh resolves the reviewer the same way, three lines apart.
# ⚠ RESOLVED HERE rather than beside the gate, for the reason the dirty-tree check states
# above it: a roster refusal is a reason not to MERGE, not a reason to refuse to ANSWER what
# the gate found. `--dry-run` still reports its verdict on a box with no roster.
# ⚠ `--agent developer`, NOT THE ROSTER'S DEFAULT (#394). The header says "the developer" and
# the merge is the developer's act; `default_agent` names the same entry today, and a script
# that relied on that would go on saying "developer" the day the default changed.
# ⚠ REFUSED BEFORE THE MERGE, not caught after it. `GIT_AUTHOR_*` / `GIT_COMMITTER_*` override
# the `-c` pair, so with either set the merge would land under the invoker's name and the check
# below would refuse a commit already on `main` — an amend instruction where a refusal was
# available for free.
if _OVERRIDE="$(env_identity_override)"; then
  die "$_OVERRIDE is set in the environment, and it OVERRIDES the roster identity this merge
     is authored with. NOTHING WAS MERGED. A merge through this script is the developer's act
     and this script is for agents (header above) — so unset it and re-run. ⚠ Do not reach for
     a hand merge instead: it skips the severity floor and the tracker re-check this performs."
fi
AGENT_ENV="$(ferrostep agent-env --agent developer --roster "$REPO_ROOT/FerroStep/config.yaml")" \
  || die "cannot resolve the developer from the roster: \`ferrostep agent-env\` refused, and
     its stderr is above. NOTHING WAS MERGED."
eval "$AGENT_ENV"
[[ -n "${AGENT_NAME:-}" && -n "${AGENT_EMAIL:-}" ]] \
  || die "the roster emitted no AGENT_NAME/AGENT_EMAIL for the developer. NOTHING WAS
     MERGED — a merge without them would land under this repo's configured identity, which
     is the owner's."

git checkout "$BASE"
_BASE_BEFORE="$(git rev-parse HEAD)"
git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" \
    merge --no-ff "$BRANCH" -m "merge $BRANCH"

# ⚠ ONLY INSPECT A COMMIT THIS RUN CREATED (#398). The identity check below ends in an
# `--amend --reset-author` instruction, and that instruction is only true of a commit that
# is not yet on the remote. If HEAD did not move, the merge made nothing — the ancestor
# check above should have refused already, and this is the guard for whatever it did not
# foresee — so the commit at HEAD is $BASE's old tip, pushed, and not ours to rewrite.
[[ "$(git rev-parse HEAD)" != "$_BASE_BEFORE" ]] \
  || die "no merge commit was created: $BASE is still at $(git rev-parse --short HEAD). NOTHING
     WAS MERGED, and there is nothing to amend — that commit was on $BASE before this ran."

# ⚠ VERIFY, DO NOT ASSUME — and do it BEFORE the push, which is the only window where the
# fix is free (DEVELOPER.md §1). The `-c` pair is a convention until something checks it.
# ⚠ BOTH LINES (#396). GitHub renders "X authored and Y committed", so a committer line the
# refusal above missed is still a merge under the invoker's name; an author-only check
# passed exactly that and pushed it. `--amend --reset-author` under the `-c` pair resets both.
_MERGE_AUTHOR="$(git log -1 --format='%an <%ae>')"
_MERGE_COMMITTER="$(git log -1 --format='%cn <%ce>')"
[[ "$_MERGE_AUTHOR" == "$AGENT_NAME <$AGENT_EMAIL>" \
   && "$_MERGE_COMMITTER" == "$AGENT_NAME <$AGENT_EMAIL>" ]] \
  || die "the merge landed but is authored '$_MERGE_AUTHOR' and committed by
     '$_MERGE_COMMITTER', not '$AGENT_NAME <$AGENT_EMAIL>' on both lines.
     IT IS NOT PUSHED. Fix it while that is still true:
       git -c user.name=\"$AGENT_NAME\" -c user.email=\"$AGENT_EMAIL\" commit --amend --reset-author"

echo "merged $BRANCH into $BASE (authored and committed $_MERGE_AUTHOR)"

if [[ "$PUSH" -eq 1 ]]; then
  # ⚠ EXPLICIT REFSPEC. Naming both ends means what lands is what this script just merged
  # and gated, rather than whatever a bare push would resolve to.
  #
  # `push.default` is NOT set in this repo today (measured 2026-09-11), so git's default
  # `simple` applies and REFUSES a push from a branch whose name differs from its upstream. ⚠
  # Keep this guard anyway, and the reason is stronger than the one that used to be here:
  # `push.default=upstream` WAS set, it is local config, and local config does not travel —
  # the 2026-08-17 tracker export predicted exactly this on four issues ("a fresh clone gets
  # push.default=simple ... every one of these traps returns intact") and that is what
  # happened. A guard that depends on reading the config is one the config can revoke
  # silently; this one is correct under either setting.
  git push origin "$BASE:$BASE"
  echo "pushed $BASE"
else
  # ⚠ NAME THE PUSH (#399), here as well as in the "nothing to merge" refusal a re-run
  # produces: this is where the reader is first told, and the refusal is what they see later.
  echo "⚠ NOT PUSHED (--no-push). Nothing is on the remote until you push:  git push origin $BASE:$BASE"
fi

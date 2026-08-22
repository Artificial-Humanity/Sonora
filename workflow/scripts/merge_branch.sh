#!/usr/bin/env bash
#
# merge_branch.sh — merge the current branch to main, but only once it clears the SEVERITY FLOOR.
#
#     workflow/scripts/merge_branch.sh [--branch B] [--base main] [--no-push] [--dry-run]
#                              [--allow-unreviewed] [--no-review]
#
# ⚠ THE GATE IS ON THE MERGE, NOT THE PUSH (owner, 2026-08-17). A branch that merged
# legitimately is one whose push is unremarkable, so this pushes by default. What it will not
# do is merge a branch that still carries a finding at or above the floor.
#
# ⚠ SINCE 2026-08-20 THAT IS A SEVERITY FLOOR, NOT ZERO-OPEN-ISSUES. The threshold is
# MERGE_SEVERITY_FLOOR in workflow/config.env and is deliberately NOT repeated here; at or
# above it blocks, below it rides, and UNGRADED or ESCALATED block at any severity. This
# header described the old gate for the length of the branch that replaced it (#202) —
# which is the same defect the branch was fixing, one file up.
#
# That is the whole safety model now, and it is narrower than the one it replaces. Until today
# the worker was denied `git push` outright and nothing it ran could reach `main`. Now it can —
# so this check is no longer one guard among several, it is the guard. The repo has no branch
# protection and force-push is unblocked (AGENTS.md §1).
#
# Replaces `changeset.sh merge`. The changeset record is retired: a branch already has an
# identity and its issues already carry its state.
#
set -euo pipefail

# --- Per-repo settings -----------------------------------------------------
# ⚠ ONE FILE TO EDIT WHEN PORTING THIS LANE. Sourced rather than hardcoded so that copying
# `workflow/` into another repo does not carry this repo's identity with it. REPO_SLUG is
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
    echo "  would: workflow/scripts/request_review.sh --range $BASE..HEAD  (then re-check)"
  else
    echo "merge_branch.sh: HEAD is not recorded as reviewed — calling for a review first."
    echo "  last reviewed tip for '$BRANCH': ${_REVIEWED_TIP:-<none recorded>}"
    # ⚠ ONLY FLAGS request_review.sh ACTUALLY TAKES. The first version passed
    # `--branch-name`, which it does not accept — the call would have died "unknown
    # argument" and the `|| true` would have swallowed it, leaving the merge to proceed as
    # if a review had run. Caught by running the dry run, not by reading. It derives the
    # branch itself; `--repo` is passed so both agree on the tracker slug.
    "$_SCRIPT_DIR/request_review.sh" --range "$BASE..HEAD" --repo "$REPO_SLUG" || true
    # ⚠ NOT `|| die`. A review that fails to COMPLETE may still have filed real findings, and
    # the tracker read below is what decides. Dying here would discard them.
    echo "merge_branch.sh: review finished — re-reading the tracker."
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
  die "workflow/scripts/merge_floor.py is missing or unreadable, so the severity floor cannot
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
       Review it:  workflow/scripts/request_review.sh
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
      echo "      workflow/scripts/issue.py grade <N> --severity <s>" >&2
    fi
    die "refusing is the designed behaviour here, not a failure."
  fi
  echo "merge_branch.sh: '$BRANCH' is below the merge floor — these block:" >&2
  echo "$BLOCKING" >&2
  die "resolve them first. Anything at or above the configured floor must be closed.
     UNGRADED must be graded (or closed) because a floor cannot pass what it cannot read;
     escalated means the owner owes a decision at any severity (workflow/WORKFLOW.md §4).
       An UNGRADED one:  workflow/scripts/issue.py grade <N> --severity <s>
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

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  would: git checkout $BASE && git merge --no-ff $BRANCH"
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

git checkout "$BASE"
git merge --no-ff "$BRANCH" -m "merge $BRANCH"
echo "merged $BRANCH into $BASE"

if [[ "$PUSH" -eq 1 ]]; then
  # ⚠ EXPLICIT REFSPEC. `push.default=upstream` is set in this repo, so a bare `git push` from
  # a branch that inherited `origin/main` as its upstream sends it to main regardless of its
  # own name. Naming both ends means what lands is what this script just merged and gated.
  git push origin "$BASE:$BASE"
  echo "pushed $BASE"
else
  echo "⚠ NOT PUSHED (--no-push). Nothing is on the remote until you push."
fi

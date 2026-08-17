#!/usr/bin/env bash
#
# merge_branch.sh — merge the current branch to main, but only once its issues are settled.
#
#     workflow/scripts/merge_branch.sh [--branch B] [--base main] [--no-push] [--dry-run]
#                              [--allow-unreviewed]
#
# ⚠ THE GATE IS ON THE MERGE, NOT THE PUSH (owner, 2026-08-17). A branch that merged
# legitimately is one whose push is unremarkable, so this pushes by default. What it will not
# do is merge a branch that still has issues in `open`, `review` or `escalated`.
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

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)  BRANCH="${2:?--branch needs a value}"; shift 2 ;;
    --base)    BASE="${2:?--base needs a value}"; shift 2 ;;
    --repo)    REPO_SLUG="${2:?--repo needs a value}"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --allow-unreviewed) ALLOW_UNREVIEWED=1; shift ;;
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
UNSETTLED="$(python3 - "$REPO_SLUG" "$BRANCH" <<'PY'
import json, os, socket, sys, urllib.parse, urllib.request
socket.setdefaulttimeout(20)
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
             "&fields=number,state,title&filter=" + flt, token=tok)
    items = r.get("items") or []
    if r.get("totalItems", 0) > len(items):
        raise RuntimeError("paged: %d of %d" % (len(items), r["totalItems"]))
    for i in items:
        print("  #%-5s %-10s %s" % (i["number"], i["state"], i["title"][:70]))
except Exception as e:
    # ⚠ UNREACHABLE IS NOT CLEAR. A tracker that cannot be read must refuse the merge, not
    # wave it through — the failure mode of the opposite choice is landing unreviewed work
    # whenever PocketBase happens to be down.
    print("TRACKER_UNREACHABLE: %s" % e)
PY
)"

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

if [[ -n "${UNSETTLED//[[:space:]]/}" ]]; then
  echo "merge_branch.sh: '$BRANCH' is not settled — these issues are not closed:" >&2
  echo "$UNSETTLED" >&2
  die "resolve them first. open -> Ozzy fixes it; review -> Janis has not verified it yet;
     escalated -> the owner owes a decision (workflow/WORKFLOW.md §4)."
fi

echo "merge_branch.sh: '$BRANCH' is settled — every issue on it is closed."
# ⚠ SAY WHAT THIS DOES NOT PROVE. Closed issues show that a review ran at SOME point, not that
# one covered the commit about to land: nothing records which tip was reviewed. Ozzy is
# responsible for having requested a review of the range being merged; this gate only refuses
# to land KNOWN-open findings. Stating the limit here so the line above is not read as more.
echo "  ⚠ this proves no finding is outstanding — NOT that a review covered $(git rev-parse --short HEAD)."

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

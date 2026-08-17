#!/usr/bin/env bash
#
# full_review.sh — the periodic whole-codebase review the owner asks for by name.
#
#     workflow/scripts/full_review.sh [--date YYYY-MM-DD] [--dry-run] [-- <request_review args>]
#
# Cuts `review-YYYY-MM-DD` from the base branch and starts a FULL review on it: no commit
# range, no history, just the code as it stands (owner, 2026-08-17).
#
# ⚠ IT IS A DIFFERENT KIND OF READ, NOT A BIGGER ONE. Per-change review can only see what
# changed; it is structurally blind to what ACCUMULATED — the doc that stopped being true, the
# guard nobody has run since it was written, the ninety files nothing invokes. That is what
# this is for, and why it deliberately ignores commit history.
#
# Everything after the branch is ordinary: issues carry `branch_name=review-YYYY-MM-DD`, Ozzy
# takes and fixes them, Janis verifies, and `merge_branch.sh` gates the merge exactly as usual.
#
set -euo pipefail

DATE=""
DRY_RUN=0
PASSTHRU=()

usage() { sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)    DATE="${2:?--date needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; PASSTHRU=("$@"); break ;;
    *) echo "full_review.sh: unknown argument: $1 (pass reviewer flags after --)" >&2; exit 2 ;;
  esac
done

die() { echo "full_review.sh: $*" >&2; exit 1; }

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository."
cd "$REPO_ROOT"

# --- Per-repo settings, same file the rest of the lane reads ----------------
_WF_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.env"
# shellcheck disable=SC1090
[[ -r "$_WF_CFG" ]] && source "$_WF_CFG"
BASE_BRANCH="${BASE_BRANCH:-main}"

# ⚠ The DATE IS THE BRANCH'S IDENTITY, so it comes from the clock rather than from a guess,
# and `--date` exists only for re-running a sweep that was started earlier.
[[ -n "$DATE" ]] || DATE="$(date +%F)"
[[ "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--date must be YYYY-MM-DD, got '$DATE'"
BRANCH="review-$DATE"

# ⚠ NOT UNDER --dry-run. A dirty tree is a reason not to SWITCH BRANCHES, not a reason to
# refuse to say what would happen — and putting it first made `--dry-run` unusable exactly when
# it is most wanted, mid-work. `merge_branch.sh` had the identical bug and the identical fix;
# if a third script grows a dry run, put its refusals after it too.
if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -z "$(git status --porcelain)" ]] \
    || die "working tree is dirty. Commit or stash first — this switches branches, and a full
     review that carries uncommitted edits reviews something that is in no commit."
fi

# ⚠ AN EXISTING BRANCH IS RESUMED, NOT REFUSED. A second run on the same day is the normal
# way a sweep continues: the first pass filed issues, Ozzy fixed some, and this is the next
# review. Refusing would make the obvious command wrong on the second use.
BASE_REF="origin/$BASE_BRANCH"
git rev-parse --verify --quiet "$BASE_REF" >/dev/null || BASE_REF="$BASE_BRANCH"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "full_review.sh: '$BRANCH' exists — resuming that sweep."
  [[ "$DRY_RUN" -eq 1 ]] || git checkout "$BRANCH"
else
  echo "full_review.sh: cutting '$BRANCH' from $BASE_REF."
  # ⚠ --no-track, ALWAYS. `push.default=upstream` is set here, so a branch that inherits
  # `origin/main` as its upstream sends a bare `git push` straight to main whatever it is
  # called — measured. A review branch is the last thing that should have that property.
  [[ "$DRY_RUN" -eq 1 ]] || git checkout --no-track -b "$BRANCH" "$BASE_REF"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  would run: workflow/scripts/request_review.sh --full ${PASSTHRU[*]:-}"
  exit 0
fi

exec "$REPO_ROOT/workflow/scripts/request_review.sh" --full "${PASSTHRU[@]}"

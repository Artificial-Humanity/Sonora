#!/usr/bin/env bash
# fix_pr.sh — resolve a PR's review feedback IN THE PR, from this machine.
#
# Replaces `.github/workflows/claude-fix.yml` and its `claude-fix` label, both retired
# 2026-08-11. The CI fix lane failed for an environmental reason, not a logical one: the
# GitHub runner image never carried what a real repair needs (torch, the ROCm stack,
# `/data`, the repo venv), and shipping that into CI cost more than the lane was worth.
# So the fix pass moved to the machine that submitted the work.
#
# BILLING: `claude -p` here uses this machine's interactive login. On a Pro/Max plan that
# is subscription quota, NOT metered API credit — the same subscription the review lane
# already draws through `CLAUDE_OAUTH_TOKEN`. Moving the fix pass local changed WHERE it
# runs, not what it costs.
#
# ⚠ THE PROTOCOL LIVES IN `.claude/commands/fix-pr.md`, NOT HERE. This script strips that
# file's frontmatter and feeds the body to `claude -p`, so `/fix-pr N` in a session and
# `fix_pr.sh N` in a shell run the identical instructions and cannot drift. Editing the
# prompt means editing that file.
#
# Usage:  scripts/fix_pr.sh <pr-number> [--no-tests] [--dry-run]
#   --no-tests   proceed with no runnable test suite; the agent marks edits UNVERIFIED
#   --dry-run    assemble the brief and print the prompt; call no model, change nothing
#
# Env:  SONORA_PY     python interpreter to test with (default: .venv/bin/python)
#       FIX_PR_MODEL  model override (default: claude-opus-5)
set -euo pipefail

die() { printf '\nABORT: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }

PR=""
NO_TESTS=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-tests) NO_TESTS=1 ;;
    --dry-run)  DRY_RUN=1 ;;
    -h|--help)  sed -n '1,30p' "$0"; exit 0 ;;
    -*)         die "unknown flag: $arg" ;;
    *)          [[ -n "$PR" ]] && die "give exactly one PR number (got '$PR' and '$arg')"
                PR="$arg" ;;
  esac
done
[[ -n "$PR" ]] || die "usage: scripts/fix_pr.sh <pr-number> [--no-tests] [--dry-run]"
[[ "$PR" =~ ^[0-9]+$ ]] || die "PR must be a number, got '$PR'"

# ── Preflight ────────────────────────────────────────────────────────────────────
# Every check here is one the agent cannot recover from on its own. Failing before the
# model is invoked keeps a broken run free.
for cmd in git gh jq claude; do
  command -v "$cmd" >/dev/null || die "'$cmd' is not on PATH"
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$REPO_ROOT"

PROTOCOL="$REPO_ROOT/.claude/commands/fix-pr.md"
[[ -f "$PROTOCOL" ]] || die "protocol file missing: $PROTOCOL"

gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login"

# Check the model login BEFORE doing any work. A run that dies on auth after checking
# out a branch has already moved the tree out from under whoever was using it.
AUTH_JSON="$(claude auth status 2>/dev/null || true)"
if [[ "$(jq -r '.loggedIn // false' <<<"$AUTH_JSON" 2>/dev/null)" != "true" ]]; then
  die "claude is not logged in — run: claude auth login"
fi
say "auth: $(jq -r '"\(.authMethod) · \(.subscriptionType // "no subscription") · \(.email)"' <<<"$AUTH_JSON")"

# A dirty tree is a hard refusal: this pass commits, and `git commit -a`-shaped mistakes
# would sweep the owner's uncommitted work into a fix commit on a shared branch.
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  die "working tree is not clean. Commit, stash or discard the above first — this pass makes commits."
fi

NWO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)" || die "cannot resolve the GitHub repository"
OWNER="${NWO%%/*}"
NAME="${NWO##*/}"

PR_JSON="$(gh pr view "$PR" --json number,state,title,headRefName,isCrossRepository 2>/dev/null)" \
  || die "cannot read PR #$PR in $NWO"
PR_STATE="$(jq -r .state <<<"$PR_JSON")"
PR_BRANCH="$(jq -r .headRefName <<<"$PR_JSON")"
[[ "$PR_STATE" == "OPEN" ]] || die "PR #$PR is $PR_STATE, not OPEN — nothing to resolve"
if [[ "$(jq -r .isCrossRepository <<<"$PR_JSON")" == "true" ]]; then
  die "PR #$PR is from a fork. This pass pushes to the PR branch and cannot push to a fork you do not own."
fi

say "PR #$PR · $NWO · branch '$PR_BRANCH'"
say "      $(jq -r .title <<<"$PR_JSON")"

# ── Test runner ──────────────────────────────────────────────────────────────────
# The premise of this lane is that the local environment is real. If it cannot run the
# suite, that premise is false HERE TOO, and the lane must say so rather than quietly
# reproducing the CI defect it was built to escape.
TEST_CMD=""
PY="${SONORA_PY:-$REPO_ROOT/.venv/bin/python}"
if [[ -x "$PY" ]] && "$PY" -c 'import pytest' >/dev/null 2>&1; then
  TEST_CMD="$PY -m pytest tests/ -q"
  say "tests: $TEST_CMD"
elif (( NO_TESTS )); then
  say "tests: UNAVAILABLE (--no-tests) — the agent will mark every edit UNVERIFIED"
else
  cat >&2 <<EOF

No runnable test suite: '$PY' is missing or has no pytest.

AGENTS.md §3 runs host code through the repo venv. Create one with the dependency set CI
proved sufficient — 516 passed / 9 skipped in 3.2s, and NO torch (~2 GB for one file,
tests/test_gate_scripts.py, which will skip):

  uv venv
  uv pip install --python .venv/bin/python \\
    pytest pyyaml numpy scipy librosa soundfile unidecode inflect fastapi

Or point SONORA_PY at an interpreter that has pytest, or pass --no-tests to accept an
UNVERIFIED pass.
EOF
  die "refusing to run a fix pass that cannot check itself (override with --no-tests)"
fi

# ── Gather the unresolved threads ────────────────────────────────────────────────
# Done in the script, not by the agent: it is deterministic, it costs no model tokens,
# it leaves a reviewable artifact, and it lets the run exit for free when there is
# nothing to fix.
STAMP="$(date +%Y%m%d-%H%M%S)"
BRIEF_DIR="$REPO_ROOT/logs/fix_pr/pr-$PR-$STAMP"
mkdir -p "$BRIEF_DIR"
BRIEF="$BRIEF_DIR/brief.json"
LOG="$BRIEF_DIR/run.log"

read -r -d '' GQL <<'EOF' || true
query($owner:String!, $name:String!, $pr:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$pr) {
      reviewThreads(first:100) {
        nodes {
          id isResolved isOutdated
          comments(first:50) {
            nodes { id databaseId body path line author { login } authorAssociation }
          }
        }
      }
      comments(first:100) {
        nodes { body author { login } authorAssociation }
      }
    }
  }
}
EOF

gh api graphql -F owner="$OWNER" -F name="$NAME" -F pr="$PR" -f query="$GQL" \
  > "$BRIEF_DIR/raw.json" || die "GraphQL query failed — see $BRIEF_DIR/raw.json"

jq '{
  unresolved_threads: [
    .data.repository.pullRequest.reviewThreads.nodes[]
    | select(.isResolved == false)
  ],
  discussion: .data.repository.pullRequest.comments.nodes
}' "$BRIEF_DIR/raw.json" > "$BRIEF"

THREADS="$(jq '.unresolved_threads | length' "$BRIEF")"
say "unresolved review threads: $THREADS"

if (( THREADS == 0 )); then
  say ""
  say "Nothing to resolve on PR #$PR. No model call made; brief kept at $BRIEF"
  exit 0
fi

# Move onto the PR branch only now that there is work to do.
git fetch --quiet origin "$PR_BRANCH" 2>/dev/null || true
gh pr checkout "$PR" || die "could not check out PR #$PR"

# ── Build the prompt from the protocol file ──────────────────────────────────────
# Strip the YAML frontmatter (the leading `---` block) — it is slash-command metadata,
# not instructions — then substitute the slash-command placeholder.
PROMPT="$(awk 'NR==1 && $0=="---" {fm=1; next} fm && $0=="---" {fm=0; next} !fm' "$PROTOCOL" \
  | sed "s/\$ARGUMENTS/$PR/g")"
[[ -n "$PROMPT" ]] || die "protocol file produced an empty prompt: $PROTOCOL"

PROMPT="$PROMPT

---

## This run

* REPO: $NWO  (OWNER=$OWNER, REPO=$NAME)
* PR: #$PR — branch \`$PR_BRANCH\`, already checked out
* UNRESOLVED THREADS: $THREADS — the full conversations, with \`authorAssociation\` on
  every comment, are in \`$BRIEF\`. READ THAT FILE FIRST.
* TEST COMMAND: ${TEST_CMD:-NONE — tests are UNAVAILABLE, mark every edit UNVERIFIED}
* Working directory: $REPO_ROOT"

if (( DRY_RUN )); then
  say ""
  say "── DRY RUN — prompt that would be sent ──────────────────────────────"
  printf '%s\n' "$PROMPT"
  say "────────────────────────────────────────────────────────────────────"
  say "brief: $BRIEF"
  exit 0
fi

# ── Run the pass ─────────────────────────────────────────────────────────────────
# `acceptEdits` plus an explicit allowlist rather than --dangerously-skip-permissions:
# this runs unattended in the repo checkout with push rights, and the difference between
# the two is whether a prompt-injected `rm -rf` needs approval.
say ""
say "running the fix pass — output also at $LOG"
say ""
claude -p "$PROMPT" \
  --model "${FIX_PR_MODEL:-claude-opus-5}" \
  --permission-mode acceptEdits \
  --allowedTools "Read,Edit,Write,Glob,Grep,Bash(git:*),Bash(gh:*),Bash(${PY}:*),Bash(pytest:*),Bash(uv:*)" \
  2>&1 | tee "$LOG"

say ""
say "fix pass finished. Log: $LOG"
say "Answer any open threads on the PR, then re-run: scripts/fix_pr.sh $PR"

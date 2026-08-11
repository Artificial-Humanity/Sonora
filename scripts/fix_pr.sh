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
  if (( DRY_RUN )); then
    # Same reasoning as the test-runner check below: a dry run neither commits nor
    # switches branches, so refusing it here would only block inspection.
    say "⚠ working tree is dirty — a real run would refuse here (--dry-run continues anyway)"
  else
    die "working tree is not clean. Commit, stash or discard the above first — this pass makes commits."
  fi
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
elif (( DRY_RUN )); then
  # A dry run makes no edits and calls no model, so the refusal below would only stop you
  # from INSPECTING the lane — which is the one thing you want when the environment is
  # broken. Report it and continue; a real run still refuses.
  say "tests: UNAVAILABLE — a real run would refuse here (--dry-run continues anyway)"
else
  cat >&2 <<EOF

No runnable test suite: '$PY' is missing or has no pytest.

AGENTS.md §3 runs host code through the repo venv. Create one with the set MEASURED to
work on this host (2026-08-11: 510 passed / 6 failed / 11 skipped in 1.6s, the 6 being
missing data artifacts) — and NO torch, which is ~2 GB for one file
(tests/test_gate_scripts.py) that will simply skip:

  uv venv
  uv pip install --python .venv/bin/python \\
    pytest pyyaml "numpy<2.2" scipy librosa soundfile unidecode inflect fastapi pysbd

⚠ BOTH PINS ARE LOAD-BEARING, and the version of this list without them does not
install. Unconstrained \`numpy\` resolves to 2.2+, which no numba supports, so the
resolver walks numba back to 0.53.1 — a 2021 release that cannot build on Python 3.12.
The failure surfaces as a librosa/numba BUILD error and says nothing about numpy.
\`pysbd\` is a real test dependency (tests/test_acquisition_lane.py) that is absent from
the CI install list and from pyproject's dev extra.

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

# WHERE THE ARTIFACTS GO IS NOT OBVIOUS, because `logs/` is not reliably writable.
# On ai-lab-0 it is root-owned (`drwxr-sr-x root datashare`, created by a container on
# 2026-07-14), so no non-root user can write into it — including `ai-mgr`, which the
# containers run as. Probe instead of assuming, and always SAY which directory was
# chosen: an artifact you cannot find is an artifact that does not exist.
BRIEF_ROOT=""
for candidate in "${FIX_PR_LOG_DIR:-}" "$REPO_ROOT/logs/fix_pr" "${TMPDIR:-/tmp}/sonora-fix-pr"; do
  [[ -n "$candidate" ]] || continue
  if mkdir -p "$candidate/pr-$PR-$STAMP" 2>/dev/null; then
    BRIEF_ROOT="$candidate"
    break
  fi
done
[[ -n "$BRIEF_ROOT" ]] || die "no writable directory for the brief (tried \$FIX_PR_LOG_DIR, $REPO_ROOT/logs/fix_pr, ${TMPDIR:-/tmp}/sonora-fix-pr)"

BRIEF_DIR="$BRIEF_ROOT/pr-$PR-$STAMP"
BRIEF="$BRIEF_DIR/brief.json"
LOG="$BRIEF_DIR/run.log"
say "artifacts: $BRIEF_DIR"

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

# ── Build the prompt from the protocol file ──────────────────────────────────────
# Strip the YAML frontmatter (the leading `---` block) — it is slash-command metadata,
# not instructions — then substitute the slash-command placeholder.
PROMPT="$(awk 'NR==1 && $0=="---" {fm=1; next} fm && $0=="---" {fm=0; next} !fm' "$PROTOCOL" \
  | sed "s/\$ARGUMENTS/$PR/g")"
[[ -n "$PROMPT" ]] || die "protocol file produced an empty prompt: $PROTOCOL"

if (( DRY_RUN )); then
  say ""
  say "── DRY RUN — protocol resolves, PR is reachable, $THREADS thread(s) to work ──"
  say "brief:    $BRIEF"
  say "prompt:   $(printf '%s' "$PROMPT" | wc -l) lines from $PROTOCOL"
  say "NOT done: branch checkout, test baseline, model call"
  exit 0
fi

# Move onto the PR branch only now that there is work to do — and only on a real run.
# A dry run that reassigns your working branch is not dry.
git fetch --quiet origin "$PR_BRANCH" 2>/dev/null || true
gh pr checkout "$PR" || die "could not check out PR #$PR"

# ── Baseline the suite BEFORE the agent touches anything ─────────────────────────
# Measured on the PR branch, so it describes the code the agent is about to edit.
#
# ⚠ THIS IS NOT BOOKKEEPING. On ai-lab-0, 2026-08-11, a clean checkout of this repo runs
# 510 passed / 6 failed / 11 skipped, and every one of those 6 failures is a MISSING DATA
# ARTIFACT (`data/libritts_r_vat_v4/`, `data/libritts_r_holdout_devclean/`) rather than a
# defect in the code. An agent that runs the suite cold, finds red, and starts "fixing"
# goes hunting corpus files it cannot rebuild — or, worse, edits a doc-claims registry to
# silence the gate. Handing over the before-picture is what makes "did I break anything?"
# answerable by SUBTRACTION instead of by judgement.
BASELINE_FILE="$BRIEF_DIR/baseline.txt"
BASELINE_SUMMARY="not run"
if [[ -n "$TEST_CMD" ]]; then
  say "baselining the suite on '$PR_BRANCH' …"
  # `|| true`: a red baseline is the expected case here, not an error.
  eval "$TEST_CMD" > "$BASELINE_FILE" 2>&1 || true
  BASELINE_SUMMARY="$(grep -E '^[0-9]+ (passed|failed)|[0-9]+ (passed|failed).*(in [0-9.]+s)' "$BASELINE_FILE" | tail -1)"
  [[ -n "$BASELINE_SUMMARY" ]] || BASELINE_SUMMARY="$(tail -1 "$BASELINE_FILE")"
  say "baseline: $BASELINE_SUMMARY"
fi

PROMPT="$PROMPT

---

## This run

* REPO: $NWO  (OWNER=$OWNER, REPO=$NAME)
* PR: #$PR — branch \`$PR_BRANCH\`, already checked out
* UNRESOLVED THREADS: $THREADS — the full conversations, with \`authorAssociation\` on
  every comment, are in \`$BRIEF\`. READ THAT FILE FIRST.
* TEST COMMAND: ${TEST_CMD:-NONE — tests are UNAVAILABLE, mark every edit UNVERIFIED}
* Working directory: $REPO_ROOT

### Test baseline — measured for you, before you changed anything

\`\`\`
$BASELINE_SUMMARY
\`\`\`

Full output: \`$BASELINE_FILE\`

⚠ **THESE FAILURES ALREADY EXISTED. THEY ARE NOT YOURS AND THEY ARE NOT YOUR JOB.** On
this machine they are missing DATA artifacts, not code defects — the corpus directories
they read are not present in this checkout. Do not try to fix them, do not edit a registry
or a test to silence them, and do not report them as regressions. Your obligation is
narrower and checkable: **re-run the same command when you are done and confirm the
failure set has not GROWN.** Name any new failure in your summary; if you cannot make it
go away, say so plainly rather than leaving the owner to diff two test logs."

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

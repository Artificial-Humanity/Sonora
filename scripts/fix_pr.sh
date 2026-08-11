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
#   --dry-run    gather the brief, report what WOULD run, then stop. Calls no model, makes
#                no commits, and does NOT switch branches or run the suite. It DOES write
#                the brief and the raw GraphQL response to the artifact directory, so it is
#                not side-effect free — it is decision-free.
#
# Env:  SONORA_PY        python interpreter to test with (default: .venv/bin/python)
#       FIX_PR_MODEL     model override (default: claude-opus-5)
#       FIX_PR_LOG_DIR   artifact directory override (default: probe logs/fix_pr, then TMPDIR)
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
    # Print the header comment up to (not including) the first line of code. A fixed
    # `sed -n '1,30p'` window went stale the moment the header grew — it ended by printing
    # `set -euo pipefail` — and the previous review's finding was specifically that --help
    # describes behaviour the script does not have. This cannot drift again.
    -h|--help)  awk 'NR>1 && /^set /{exit} NR>1' "$0"; exit 0 ;;
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
# ⚠ REFUSE WHITESPACE IN THE INTERPRETER PATH rather than escaping it through three hops.
# Making TEST_ARGV an array fixed only the hop inside this script. The same path also
# reaches the AGENT — as the TEST COMMAND line in the prompt it is told to run — and
# `--allowedTools` as `Bash(<path>:*)`. A space word-splits into a different command in
# both, executed by a process holding `acceptEdits` and push rights. One up-front refusal
# closes all three; escaping would have to be correct in all three.
# ⚠ WARN AND DISABLE, DO NOT `die`. An abort here fired ABOVE the `--no-tests` and
# `--dry-run` branches below, killing the two escape hatches this script argues for
# elsewhere. Marking the path unusable lets those branches work while a real run still
# refuses, through the ordinary "no runnable test suite" message.
# Commas matter as much as spaces: `--allowedTools` is a COMMA-separated list, so a comma in
# the path silently splits it into bogus entries — which the previous comment claimed was
# closed when only whitespace was checked.
PY_USABLE=1
if [[ "$PY" == *[[:space:],]* ]]; then
  PY_USABLE=0
  say "⚠ interpreter path holds whitespace or a comma and cannot be passed safely to the agent or to --allowedTools; treating the suite as unavailable: '$PY'"
fi
TEST_ARGV=()
if (( PY_USABLE )) && [[ -x "$PY" ]] && "$PY" -c 'import pytest' >/dev/null 2>&1; then
  # ⚠ `--color=no` IS LOAD-BEARING, NOT TIDINESS. `pyproject.toml` sets
  # `addopts = ["--color=yes", …]` — FORCED, not `auto` — so pytest emits ANSI even when
  # stdout is a file, and every short-summary line begins with an escape sequence instead
  # of with `FAILED`. The anchored grep below therefore matched ZERO lines against real
  # failures on every host; `${BASELINE_FAILURES:+…}` elided the block, and the agent got
  # no test IDs at all. Silent, which is how it survived being written AND reviewed once.
  # A CLI flag overrides addopts, so this wins.
  TEST_ARGV=("$PY" -m pytest tests/ -q --color=no)
  TEST_CMD="${TEST_ARGV[*]}"   # display AND the prompt — safe only because of the check above
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
# ⚠ THIS IS NOT BOOKKEEPING. The suite is RED on a clean checkout, and how it is red depends
# on the host: on ai-lab-0 (2026-08-11) 510 passed / 6 failed / 11 skipped, all six missing
# corpus directories; on a CI runner the same checkout gave 443 passed / 8 failed / 5 errors,
# from absent `pysbd`/`fastapi` and no `.venv`. An agent that runs the suite cold, finds red,
# and starts "fixing" goes hunting artifacts it cannot rebuild — or edits a registry to
# silence a gate. Handing over the before-picture makes "did I break anything?" answerable by
# SUBTRACTION instead of by judgement.
#
# ⚠ But the baseline is measured HERE, on the PR branch, so it cannot distinguish a failure
# pre-existing on `main` from one this PR introduced. The prompt below must say so: the first
# version of this block asserted these failures "already existed" and were "not yours", which
# launders a regression the PR caused into someone else's problem.
BASELINE_FILE="$BRIEF_DIR/baseline.txt"
BASELINE_SUMMARY="not run"
BASELINE_FAILURES=""
if [[ -n "$TEST_CMD" ]]; then
  say "baselining the suite on '$PR_BRANCH' …"
  # `|| true`: a red baseline is the expected case here, not an error.
  "${TEST_ARGV[@]}" > "$BASELINE_FILE" 2>&1 || true

  # ⚠ `|| true` ON THE GREP IS LOAD-BEARING UNDER `set -euo pipefail`. A non-matching grep
  # exits 1; `pipefail` propagates that through `| tail`; and the exit status of a plain
  # assignment IS the status of its command substitution — so `set -e` killed the script
  # here, AFTER `gh pr checkout` had already reassigned the operator's working branch. The
  # fallback on the next line was unreachable in precisely the case it was written for.
  # Found by the review lane on PR #60 and reproduced before fixing.
  BASELINE_SUMMARY="$(grep -E '[0-9]+ (passed|failed|error)' "$BASELINE_FILE" | tail -1 || true)"
  [[ -n "$BASELINE_SUMMARY" ]] || BASELINE_SUMMARY="$(tail -1 "$BASELINE_FILE" || true)"
  [[ -n "$BASELINE_SUMMARY" ]] || BASELINE_SUMMARY="(the suite produced no parseable summary — read $BASELINE_FILE)"
  say "baseline: $BASELINE_SUMMARY"

  # Hand over the failing test IDS, not a prose theory about why they fail. The previous
  # version asserted "on this machine they are missing DATA artifacts" — true on ai-lab-0,
  # false on the CI runner, where the same block would have told the agent that missing
  # `pysbd`/`fastapi` and an absent `.venv` were corpus problems.
  # ⚠ THE ESCAPE BYTE IS BUILT WITH printf, NOT WRITTEN AS `\x1b`. `\x` is NOT an ERE escape:
  # GNU grep (both target hosts) matches the LITERAL characters `x1b`, so the previous version
  # advertised ANSI tolerance it did not have. It looked correct only because this dev image
  # ships ugrep, which DOES implement `\x` — a portability trap one host cannot reveal.
  ESC="$(printf '\033')"
  BASELINE_FAILURES="$(grep -E "^(${ESC}\[[0-9;]*m)*(FAILED|ERROR)" "$BASELINE_FILE" | sed 's/ - .*//' | head -40 || true)"
fi

PROMPT="$PROMPT

---

## This run

* REPO: $NWO  (OWNER=$OWNER, REPO=$NAME)
* PR: #$PR — branch \`$PR_BRANCH\`, already checked out
* UNRESOLVED THREADS: $THREADS — the full conversations, with \`authorAssociation\` on
  every comment, are in \`$BRIEF\`. READ THAT FILE FIRST.
* TEST COMMAND: ${TEST_CMD:-NONE — tests are UNAVAILABLE, mark every edit UNVERIFIED}
* Working directory: $REPO_ROOT"

if [[ -n "$TEST_CMD" ]]; then
PROMPT="$PROMPT

### Test baseline — measured on this PR's code, before you edited anything

\`\`\`
$BASELINE_SUMMARY
\`\`\`

${BASELINE_FAILURES:+Failing/erroring tests:
\`\`\`
$BASELINE_FAILURES
\`\`\`
}
Full output: \`$BASELINE_FILE\`

⚠ **THIS IS NOT A LIST OF THINGS THAT ARE SOMEONE ELSE'S PROBLEM.** It was measured on the
PR branch, which already contains this PR's changes — so a failure here may be
**pre-existing on \`main\` OR introduced by this PR**, and this file cannot tell you which.
If a finding in your threads concerns one of these, or you can see the PR caused it, it is
**in scope**.

To find out which side a failure came from, compare against the merge base in a SEPARATE
worktree — never by moving this one. Run these in order and STOP if a step fails:

\`\`\`
git fetch origin main                     # NOT already fetched — the script fetches only this PR's branch
BASE=\$(git merge-base origin/main HEAD)
test -n "\$BASE" || echo "NO MERGE BASE — stop here, do not guess"
WT=\$(mktemp -d)/base && git worktree add "\$WT" "\$BASE"
ln -s $REPO_ROOT/.venv "\$WT/.venv" && ln -s $REPO_ROOT/data "\$WT/data"
( cd "\$WT" && .venv/bin/python -m pytest tests/ -q --color=no )
git worktree remove --force "\$WT"
\`\`\`

⚠ **If \`\$BASE\` is empty, \`git worktree add\` creates a branch at HEAD instead** — you would
compare the PR against ITSELF and call every failure pre-existing. The \`test -n\` line is
what prevents that; \`mktemp -d\` is what stops a second run colliding with a leftover
directory.

⚠ **THE TWO SYMLINKS ARE NOT OPTIONAL.** \`.venv\` and \`data/\` are gitignored, so a fresh
worktree has neither, and WITHOUT THEM THE SUITE LOOKS GREEN — the data-gated failures become
skips (measured: 509 passed / 18 skipped / 0 failed, against 510 / 6 / 11). An unpopulated
worktree tells you the merge base is clean and the PR broke six things it never touched.

⚠ Do NOT use \`git stash\` for this. It shelves changes and leaves you on the same commit, so
it does not reach \`main\` at all, and stashing mid-pass in a tree this script refuses to run
dirty can discard your own fixes.

What the baseline is actually for is a **subtraction**: re-run the same command when you are
done and confirm the failure set has **not GROWN**. Name any new failure in your summary,
and if you cannot make it go away, say so plainly rather than leaving the owner to diff two
test logs.

⚠ Do not \"fix\" a failure by editing a test or a registry so it stops reporting. Do not
diagnose these from a cached assumption about why the suite is red on some other machine —
**read the output.** How the suite is red is host-specific and the counts differ per host,
so any sentence here naming a cause would be the very defect this block was rewritten to
remove. The failing test IDs above are measured; a diagnosis is not."
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
  --allowedTools "Read,Edit,Write,Glob,Grep,Bash(git:*),Bash(gh:*),Bash(${PY}:*),Bash(pytest:*),Bash(uv:*),Bash(ln:*),Bash(mktemp:*),Bash(test:*)" \
  2>&1 | tee "$LOG"

say ""
say "fix pass finished. Log: $LOG"
say "Answer any open threads on the PR, then re-run: scripts/fix_pr.sh $PR"

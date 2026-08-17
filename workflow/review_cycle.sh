#!/usr/bin/env bash
#
# review_cycle.sh — drive the review loop to convergence without a human in the middle.
#
# review → fix → review → … until every issue this cycle produced is closed, escalated, or
# out of attempts. Built because the loop had no driver: every step was a person invoking
# request_review.sh, reading it, fixing, re-invoking — so when that person stopped, the loop
# stopped wherever it happened to be, leaving issues open that were merely mid-flight.
#
# ⚠⚠ IT NEVER PUSHES. NOTHING HERE REACHES `main`.
# `git push` is denied to the worker it spawns, and this script does not push either. The
# repo has no branch protection and force-push is unblocked (AGENTS.md §1), so an unattended
# loop with push rights would be the only thing standing between a bad afternoon and
# production. Converge here; a human pushes.
#
# ⚠ IT SPENDS MONEY UNATTENDED. Every `claude -p` call carries --max-budget-usd, and the run
# has a hard review ceiling. Read --help before the first real run.
#
# Stop it at any time by creating the stop file (default `.review_cycle.stop` in the repo
# root) — checked before every phase, so it takes effect within one step rather than at the
# end.
#
set -euo pipefail

RANGE="origin/main..HEAD"
DEVELOPER="Ozzy"
MAX_REVIEWS=4
MAX_USD=5
MODEL="opus"
EFFORT="xhigh"
STOPFILE=""
DRY_RUN=0

usage() {
  cat <<'USAGE'
review_cycle.sh — run the review loop to convergence. NEVER PUSHES.

  --range <RANGE>     Two-dot range under review.       (default: origin/main..HEAD)
  --developer <ID>    Worker identity.                  (default: Ozzy)
  --max-reviews <N>   Hard ceiling on reviews, 1-4.     (default: 4)
                      Four reviews is what three fix passes require: the review that finds
                      an issue, then one after each fix pass.
  --max-usd <N>       Spend ceiling PER claude call.    (default: 5)
  --model / --effort  Passed to both roles.             (default: opus / xhigh)
  --stop-file <PATH>  Create this file to halt.  (default: <repo>/.review_cycle.stop)
  --dry-run           Print the plan and exit. Spends nothing, files nothing.
  -h, --help          This.

STOPS ON, in order of precedence:
  1. the stop file appearing            5. a review or worker exiting non-zero
  2. a review reporting MUST-NOT-LAND   6. the worker failing to advance agent_passes
  3. convergence (nothing open)         7. the review ceiling
  4. the spend ceiling

CONVERGENCE means: branch_name="<the current branch>" && state="open"  ->  empty.
⚠ There is no `escalated=false` clause. Escalation is a VALUE OF `state`, not a flag beside it
(owner, 2026-08-17) — an escalated issue is not `open`, so it leaves this count on its own.
⚠ Scoped to the CURRENT BRANCH, which is what keeps another branch's backlog out of this
cycle's gate: the 48 migrated GitHub issues sit on `github-issues-fixes` and nobody is working
them, and a check that counted them would never reach zero.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --range)       RANGE="${2:?}"; shift 2 ;;
    --developer)   DEVELOPER="${2:?}"; shift 2 ;;
    --max-reviews) MAX_REVIEWS="${2:?}"; shift 2 ;;
    --max-usd)     MAX_USD="${2:?}"; shift 2 ;;
    --model)       MODEL="${2:?}"; shift 2 ;;
    --effort)      EFFORT="${2:?}"; shift 2 ;;
    --stop-file)   STOPFILE="${2:?}"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "review_cycle.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "review_cycle.sh: $*" >&2; exit 1; }
say() { echo "── review_cycle: $*" >&2; }

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository."
cd "$REPO_ROOT"
[[ -z "$STOPFILE" ]] && STOPFILE="$REPO_ROOT/.review_cycle.stop"
[[ -x workflow/request_review.sh ]] || die "workflow/request_review.sh not found or not executable"
[[ -r workflow/DEVELOPER.md ]] || die "workflow/DEVELOPER.md not readable"
[[ "$MAX_REVIEWS" =~ ^[1-4]$ ]] || die "--max-reviews must be 1-4 (three fix passes need four reviews)"

# ⚠ Refuse a dirty tree. The worker commits, so uncommitted edits would be swept into a
# commit nobody wrote a message for — and the range under review would stop matching the
# range that was read.
if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree is dirty. Commit or stash first — the worker commits, and it would
     otherwise absorb your uncommitted edits into a review it never read."
fi

rm -f "$STOPFILE"

# --- tracker helper --------------------------------------------------------
pb() {  # $1 = pocketbase filter -> prints totalItems, or "unreachable"
  python3 - "$1" <<'PY'
import json, os, socket, sys, urllib.parse, urllib.request
socket.setdefaulttimeout(15)
try:
    pb = json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"]["pocketbase"]
    env = pb.get("env", {}); base = env.get("PB_URL", "http://127.0.0.1:8090")
    def call(path, method="GET", body=None, token=None):
        r = urllib.request.Request(base + path, method=method)
        if token: r.add_header("Authorization", token)
        d = None
        if body is not None:
            d = json.dumps(body).encode(); r.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(r, d) as x: return json.loads(x.read() or b"{}")
    tok = call("/api/collections/_superusers/auth-with-password", "POST",
               {"identity": env.get("PB_EMAIL"), "password": env.get("PB_PASSWORD")})["token"]
    q = urllib.parse.quote(sys.argv[1])
    print(call("/api/collections/issues/records?perPage=1&skipTotal=false&filter=" + q,
               token=tok).get("totalItems", 0))
except Exception:
    print("unreachable")
PY
}

pb_passes() {  # $1 = branch -> prints SUM of agent_passes over its issues, or "unreachable"
  # ⚠ A SUM, NOT A COUNT, AND OVER EVERY STATE. This is the instrument for "did the worker
  # spend an attempt?", and both properties are load-bearing:
  #
  #   * a SUM sees 1->2 on an issue that was never at zero; a count of `agent_passes=0` does
  #     not, so a worker that only ever re-attempts already-tried issues looks idle;
  #   * ALL STATES, because the worker may ESCALATE. An escalated issue leaves `state="open"`,
  #     so a sum restricted to open issues can FALL across a pass that did real work — and the
  #     guard below would read that as a stall and stop the loop.
  #
  # Nothing in this loop lowers `agent_passes` (only the owner resets, and the owner is not in
  # it), so across a single worker run this is monotonic: it rises iff the worker incremented.
  python3 - "$1" <<'PY'
import json, os, socket, sys, urllib.parse, urllib.request
socket.setdefaulttimeout(15)
try:
    pb = json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"]["pocketbase"]
    env = pb.get("env", {}); base = env.get("PB_URL", "http://127.0.0.1:8090")
    def call(path, method="GET", body=None, token=None):
        r = urllib.request.Request(base + path, method=method)
        if token: r.add_header("Authorization", token)
        d = None
        if body is not None:
            d = json.dumps(body).encode(); r.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(r, d) as x: return json.loads(x.read() or b"{}")
    tok = call("/api/collections/_superusers/auth-with-password", "POST",
               {"identity": env.get("PB_EMAIL"), "password": env.get("PB_PASSWORD")})["token"]
    q = urllib.parse.quote('branch_name="%s"' % sys.argv[1].replace('"', ""))
    # ⚠ perPage is 500, not the API default of 10 — a truncated page silently under-sums and
    # the guard then sees a stall that never happened.
    r = call("/api/collections/issues/records?perPage=500&skipTotal=false&fields=agent_passes"
             "&filter=" + q, token=tok)
    items = r.get("items") or []
    if r.get("totalItems", 0) > len(items):
        raise RuntimeError("paged: %d of %d" % (len(items), r["totalItems"]))
    print(sum(int(i.get("agent_passes") or 0) for i in items))
except Exception:
    print("unreachable")
PY
}

# ⚠ RESOLVED ONCE, HERE, and read by everything below — the convergence filter and the
# counter guard must be talking about the same branch, and two separate `rev-parse` calls are
# two things to keep in step. `set -u` is on, so a use before this line is a hard failure
# rather than an empty filter; that is the safer half of the trade and it has bitten before.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# ⚠ No `escalated=false` clause: escalation is a value of `state` (owner, 2026-08-17), so an
# escalated issue is already not `open`. The clause is not merely redundant — it would be a
# FILTER ERROR now, since the field it named no longer exists.
OPEN_FILTER="branch_name=\"$BRANCH\" && state=\"open\""

check_stop() {
  if [[ -e "$STOPFILE" ]]; then
    say "STOP FILE PRESENT ($STOPFILE) — halting before $1."
    rm -f "$STOPFILE"; exit 0
  fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "range        : $RANGE"
  echo "developer    : $DEVELOPER"
  echo "max reviews  : $MAX_REVIEWS   (three fix passes need four reviews)"
  echo "spend ceiling: \$$MAX_USD per claude call"
  echo "model/effort : $MODEL / $EFFORT"
  echo "stop file    : $STOPFILE"
  echo "converged when: $OPEN_FILTER  -> 0   (currently: $(pb "$OPEN_FILTER"))"
  echo "⚠ this script NEVER pushes, and denies git push to the worker it spawns"
  exit 0
fi

# --- worker tool policy ----------------------------------------------------
# ⚠ THE WORKER GETS auto-mode AND THE REVIEWER DOES NOT, deliberately. The worker is the role
# that is *supposed* to write — it edits, runs tests, and commits — and enumerating every
# command a developer legitimately needs is not achievable. The reviewer's restriction (#100)
# exists because it must NOT write; that reasoning does not transfer.
#
# ⚠ WHAT IS DENIED IS THE ONLY THING THAT MATTERS: reaching `main`, or re-entering this loop.
# `git -c` is deliberately NOT denied — it is how DEVELOPER.md §1 requires every commit to be
# authored, so denying it would forbid the worker from committing correctly.
WORKER_DENY=(
  "Bash(git push:*)"
  "Bash(git remote:*)"
  "Bash(workflow/review_cycle.sh:*)" "Bash(./workflow/review_cycle.sh:*)"
  "Bash(workflow/request_review.sh:*)" "Bash(./workflow/request_review.sh:*)"
  # ⚠ THE MERGE IS NOT THIS LOOP'S TO MAKE. `merge_branch.sh` merges to main AND PUSHES once
  # the tracker says the branch is settled — correct when Ozzy runs it deliberately, wrong for
  # an unattended fix loop, which would then be able to land its own work. The driver's job
  # ends at convergence; the merge is a separate, deliberate act.
  "Bash(workflow/merge_branch.sh:*)" "Bash(./workflow/merge_branch.sh:*)"
)
# ⚠ `workflow/issue.py` is deliberately NOT denied: taking, commenting, escalating and moving
# an issue to `review` IS the worker's job, and it is the one path that enforces the counter
# cap and the mandatory-comment rules.

REVIEW_TIPS=()
CONVERGED=0
LAST_SUMMARY=""

for (( review=1; review<=MAX_REVIEWS; review++ )); do
  check_stop "review $review"

  say "review $review of $MAX_REVIEWS over $RANGE"
  NOTES_ARG=()
  [[ -f "$REPO_ROOT/.review_cycle.notes" ]] && NOTES_ARG=(--notes-file "$REPO_ROOT/.review_cycle.notes")

  set +e
  OUT="$(workflow/request_review.sh --range "$RANGE" --developer "$DEVELOPER" \
          --pass "$review" \
          "${NOTES_ARG[@]+"${NOTES_ARG[@]}"}" 2>&1)"
  RC=$?
  set -e
  printf '%s\n' "$OUT"
  LAST_SUMMARY="$OUT"

  RID="$(sed -n 's/.*as branch_name \([0-9a-zA-Z._-]*\),.*/\1/p' <<< "$OUT" | head -1 || true)"
  [[ -n "$RID" ]] && REVIEW_TIPS+=("$RID")

  # ⚠ CHECKED BEFORE THE EXIT CODE. A review can complete cleanly (rc 0) and still be telling
  # you the change must not land; that is a content signal, not a failure signal.
  if grep -q "MUST-NOT-LAND" <<< "$OUT"; then
    say "REVIEW SAYS MUST-NOT-LAND — stopping. This needs the owner; do not push."
    exit 3
  fi
  if [[ "$RC" -ne 0 ]]; then
    say "review exited $RC — stopping. ⚠ Findings may still have been filed under $RID;
        query it before concluding the range is unreviewed."
    exit "$RC"
  fi

  OPEN="$(pb "$OPEN_FILTER")"
  say "open cycle issues after review $review: $OPEN"
  if [[ "$OPEN" == "unreachable" ]]; then
    say "tracker unreachable — cannot tell converged from stuck. Stopping rather than guessing."
    exit 4
  fi
  if [[ "$OPEN" == "0" ]]; then
    CONVERGED=1
    say "CONVERGED after review $review — everything closed, escalated, or out of attempts."
    break
  fi
  if (( review == MAX_REVIEWS )); then
    say "review ceiling reached with $OPEN still open. Not a failure: issues below three fix
        passes keep their remaining attempts, and a later cycle may take them."
    break
  fi

  # --- fix pass ------------------------------------------------------------
  check_stop "fix pass $review"
  say "fix pass $review — spawning worker (git push denied)"
  BEFORE_SUM="$(pb_passes "$BRANCH")"

  WORKER_BRIEF="## This fix pass

You are **$DEVELOPER**, working in \`$REPO_ROOT\` on range \`$RANGE\`. Run by
\`workflow/review_cycle.sh\`, unattended — **there is no one to ask**, so where you would
normally check, act on your best reading and say so in the issue comment.

Issues to address: those under branch_name(s) \`$(IFS=,; echo "${REVIEW_TIPS[*]}")\` whose
\`state\` is \`open\`. \`escalated\` is a third value of \`state\`, so those are already excluded —
you do not filter them out, and you do not work them.

1. **Increment \`agent_passes\` by one on each issue you take on, FIRST**, before any work.
   The driver verifies this happened and stops if it did not.
2. Fix what is genuinely wrong; rebut what is not, in the issue's comments (max 1500 chars).
   **Close nothing** — the reviewer resolves.
3. Escalate anything that needs an owner decision; it drops out of re-review.
4. **Commit** as \`git -c user.name=$DEVELOPER -c user.email=${DEVELOPER,,}@artificialhumanity.io\`.
   ⚠ **DO NOT PUSH.** It is denied, and this loop converges without it.
5. Write what you did to \`$REPO_ROOT/.review_cycle.notes\` — the next review is briefed from
   that file and cannot otherwise tell a fix from an omission.
"

  set +e
  claude -p "Address the open issues from this review, then commit." \
    --append-system-prompt-file "$REPO_ROOT/workflow/DEVELOPER.md" \
    --append-system-prompt "$WORKER_BRIEF" \
    --model "$MODEL" --effort "$EFFORT" \
    --permission-mode auto \
    --max-budget-usd "$MAX_USD" \
    --disallowedTools "${WORKER_DENY[@]}"
  WRC=$?
  set -e
  if [[ "$WRC" -ne 0 ]]; then
    say "worker exited $WRC — stopping. The tree may hold partial work; inspect before re-running."
    exit "$WRC"
  fi

  # ⚠ THE CAP IS ONLY REAL IF THE COUNTER MOVED. A worker that crashed or declined leaves the
  # counters untouched, and re-running would retry forever without ever spending an attempt —
  # the unbounded loop the cap exists to prevent, reintroduced by the thing automating it.
  #
  # ⚠⚠ THIS GUARD DID NOT WORK UNTIL 2026-08-17, AND IT FAILED OPEN. `BEFORE_SUM` held the
  # COUNT of open issues on this branch; `AFTER_SUM` held the COUNT of `agent_passes=0` issues
  # REPO-WIDE. Two different populations over two different scopes, compared for equality — so
  # `AFTER == BEFORE` held only by coincidence and the stall it exists to catch sailed through.
  # The variables were named `_SUM` throughout, which is the tell: the instrument was meant to
  # be a sum of the counter and had drifted into a count of records.
  #
  # Both readings now come from the SAME call on the SAME branch, so the comparison is between
  # like and like — and a stall is `unchanged`, not `equal to some unrelated number`.
  AFTER_SUM="$(pb_passes "$BRANCH")"
  if [[ "$AFTER_SUM" != "unreachable" && "$BEFORE_SUM" != "unreachable" ]]; then
    if (( AFTER_SUM == BEFORE_SUM )); then
      say "worker did not advance agent_passes on any issue (sum stayed at $BEFORE_SUM) —
          stopping rather than looping. Check the worker's output above."
      exit 5
    fi
    # ⚠ A FALL IS NOT A STALL, AND MUST NOT BE READ AS ONE. Nothing in this loop lowers the
    # counter, so a drop means the owner re-armed an issue mid-run — their dial, never to be
    # "corrected" (AGENTS.md §1). Say so and carry on; the worker plainly did something.
    if (( AFTER_SUM < BEFORE_SUM )); then
      say "agent_passes fell ($BEFORE_SUM -> $AFTER_SUM) — an issue was re-armed mid-cycle.
          That is the owner's dial, not a fault. Continuing."
    fi
  fi
done

echo
say "─────────────────────────────────────────────"
if [[ "$CONVERGED" -eq 1 ]]; then
  say "RESULT: converged. Nothing open from this cycle."
else
  say "RESULT: $(pb "$OPEN_FILTER") issue(s) still open from this cycle."
fi
say "reviews run: ${#REVIEW_TIPS[@]}  ($(IFS=' '; echo "${REVIEW_TIPS[*]:-none}"))"
say "⚠ NOTHING WAS PUSHED. Read the summary above, then push by hand if you agree with it."
exit 0

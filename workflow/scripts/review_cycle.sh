#!/usr/bin/env bash
#
# review_cycle.sh — drive the review loop to convergence without a human in the middle.
#
# review → fix → review → … until every issue this cycle produced is closed, escalated, or
# out of attempts. ⚠ THAT IS NOT THE MERGE GATE, and the two are easy to conflate: since
# 2026-08-20 merge_branch.sh applies a SEVERITY FLOOR, so a finding below it does not stop
# a merge. This loop still drives every issue to a resting state, which is a stricter goal
# than the floor and deliberately so — a driver that stopped at the floor would leave LOWs
# mid-flight, which is the condition it was built to end.
#
# Built because the loop had no driver: every step was a person invoking request_review.sh,
# reading it, fixing, re-invoking — so when that person stopped, the loop stopped wherever it
# happened to be, leaving issues open that were merely mid-flight.
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
MAX_REVIEWS=""   # default derives from the lane definition once the repo root is known
MAX_USD=5
MODEL="opus"
EFFORT="xhigh"
STOPFILE=""
DRY_RUN=0
ALLOW_EMPTY_TRACKER=0   # ⚠ set -u is on; an undeclared default aborts the driver (#357)

usage() {
  cat <<'USAGE'
review_cycle.sh — run the review loop to convergence. NEVER PUSHES.

  --range <RANGE>     Two-dot range under review.       (default: origin/main..HEAD)
  --developer <ID>    Worker identity.                  (default: Ozzy)
  --max-reviews <N>   Hard ceiling on reviews.   (default: the definition's fix-pass
                      ceiling + 1 — agent_passes.max in workflow/sonora-lane.json)
                      That sum is what the fix-pass cap requires: the review that finds
                      an issue, then one after each fix pass.
  --max-usd <N>       Spend ceiling PER claude call.    (default: 5)
  --model / --effort  The WORKER's only.                (default: opus / xhigh)
                      The reviewer's are its roster entry in config.yaml; this driver
                      does not forward them to request_review.sh and never did.
  --stop-file <PATH>  Create this file to halt.  (default: <repo>/.review_cycle.stop)
  --dry-run           Print the plan and exit. Spends nothing, files nothing.
  --allow-empty-tracker
                      Accept convergence when the tracker holds NO issues under this repo
                      slug at all. Default is to REFUSE, because the usual cause is a wrong
                      slug and every count taken with it was taken over nothing. Use this
                      only when the slug is verified and the repo genuinely has no issues
                      yet — a first cycle on a newly adopted lane. (#357)
  -h, --help          This.

STOPS ON, in order of precedence:
  1. the stop file appearing            5. a review or worker exiting non-zero
  2. a review reporting MUST-NOT-LAND   6. the worker failing to advance agent_passes
  3. convergence (nothing open)         7. the review ceiling
  4. the spend ceiling

CONVERGENCE means: branch_name="<the current branch>" && state="open"  ->  empty.
⚠ There is no `escalated=false` clause. Escalation is a VALUE OF `state`, not a flag beside it
(owner, 2026-08-17) — an escalated issue is not `open`, so it leaves this count on its own.
⚠ Scoped to the CURRENT BRANCH, which keeps another branch's open issues out of this
cycle's gate: they will not close on this cycle, so a check that counted them could never
reach zero.
⚠ This paragraph used to cite "the 48 migrated GitHub issues on `github-issues-fixes`" as
the example. Those records are NOT in the live tracker (2026-08-19) — it holds nothing below
#90; they are in `notes/tracker-export-2026-08-17.json`. The rule is unchanged; the example
was stale, and an operator reading `--help` was told a backlog exists that they cannot see.
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
    --allow-empty-tracker) ALLOW_EMPTY_TRACKER=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "review_cycle.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "review_cycle.sh: $*" >&2; exit 1; }
say() { echo "── review_cycle: $*" >&2; }

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository."
cd "$REPO_ROOT"
[[ -z "$STOPFILE" ]] && STOPFILE="$REPO_ROOT/.review_cycle.stop"
# ⚠ NORMALISED AND VALIDATED HERE, because everything downstream treats it as a file path
# (#252). `--stop-file` is the one operator-supplied path in this script.
#   * A DIRECTORY, or anything with a trailing slash, is refused: `rm -f` on it fails under
#     `set -e` and kills the run with a message about `rm`, and its basename is empty, which
#     is what silenced the dirty-tree guard entirely.
#   * A RELATIVE path is made absolute against the repo root — the script has already `cd`ed
#     there, so it already resolved that way; making it explicit is what lets the exclusion
#     below compare whole paths instead of guessing.
[[ "$STOPFILE" == */ || -d "$STOPFILE" ]] && die "--stop-file must name a file, not a
     directory: $STOPFILE"
[[ "$STOPFILE" != /* ]] && STOPFILE="$REPO_ROOT/$STOPFILE"
[[ -x workflow/scripts/request_review.sh ]] || die "workflow/scripts/request_review.sh not found or not executable"
[[ -r workflow/DEVELOPER.md ]] || die "workflow/DEVELOPER.md not readable"
# ⚠ DERIVED from the lane definition — this was a hardcoded `^[1-4]$`, the cap's arithmetic
# shadow (max+1) wearing a refusal, which would have silently disagreed with any owner
# change to agent_passes.max (found 2026-08-24, the same day the cap moved).
DEF_REVIEWS="$(python3 - "$REPO_ROOT/workflow/sonora-lane.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
(c,) = [c for c in d.get("counters", []) if c.get("name") == "agent_passes"]
print(int(c["max"]) + 1)
PY
)" || die "cannot derive the review ceiling from workflow/sonora-lane.json"
[[ -n "$MAX_REVIEWS" ]] || MAX_REVIEWS="$DEF_REVIEWS"
[[ "$MAX_REVIEWS" =~ ^[0-9]+$ && "$MAX_REVIEWS" -ge 1 && "$MAX_REVIEWS" -le "$DEF_REVIEWS" ]] \
  || die "--max-reviews must be 1..$DEF_REVIEWS (the definition's fix-pass ceiling + 1)"

NOTES_FILE="$REPO_ROOT/.review_cycle.notes"

# ⚠⚠ CLEARED BEFORE THE DIRTY-TREE CHECK, AND THAT ORDER IS THE WHOLE FIX. The worker is told
# to write `.review_cycle.notes` (step 5 of its brief); nothing used to remove it. So run 1
# left it behind, and run 2 died at the check below — reporting a dirty tree and pointing at
# "your uncommitted edits", which named the wrong cause and sent the reader to look for edits
# they had not made. The obvious remedy, committing it, is worse: it puts cycle notes inside
# the range under review. THE DRIVER RAN EXACTLY ONCE PER MANUAL CLEANUP.
#
# ⚠ Removing it also ends a second, quieter failure: the file is read before EVERY review,
# including review 1, so a surviving file briefed the first review of the NEXT cycle with the
# LAST cycle's fix notes — describing fixes to a different range as though they were this one's.
#
# ⚠ THE `rm` IS LOAD-BEARING EVEN THOUGH `.gitignore` NOW LISTS THE FILE, because porting this
# lane copies `workflow/` and nothing else (WORKFLOW.md, "Porting this lane"). A ported copy
# gets no `.gitignore` entry, so a driver that leaned on ignoring alone would arrive in the new
# repo with the once-only bug intact. Ignoring keeps it out of `git status`; this keeps it out
# of the next cycle.
#
# ⚠⚠ BOTH RUNTIME FILES, NOT JUST THE NOTES (#246). The first version of this fix moved the
# notes `rm` up here and left `rm -f "$STOPFILE"` below the refusal, where it cannot do its
# job — while arguing, three lines above, that the ported lane is exactly why the `rm` matters.
# The argument is equally true of the stop file and was applied to one of the two files the
# same commit ignored. A stop file survives whenever the cycle ends by any of the seven routes
# in `--help` OTHER than the stop file itself (`check_stop` removes it on that one path only),
# or when it is created while no cycle is running; in a ported lane it is then untracked, the
# tree is dirty, and the next run dies at the check below naming the same wrong cause.
#
# ⚠⚠ NOT ON A DRY RUN (#248). `--help` promises "Print the plan and exit. Spends nothing,
# files nothing", and an unconditional clear broke that promise twice over: it destroyed the
# loop's only cross-pass memory, and it DISARMED THE HALT CONTROL — an operator who arms the
# stop file to stop a running cycle and then opens a second terminal to check the plan would
# silently un-arm it and the cycle would keep spending. `--stop-file` also takes an arbitrary
# path, so the unguarded form was an `rm -f` on operator-supplied input in the mode advertised
# as inert. The first version of this fix was pinned by a test that asserted the deletion as a
# REQUIREMENT, so the suite agreed with the bug.
if [[ "$DRY_RUN" -eq 0 ]]; then
  rm -f "$NOTES_FILE" "$STOPFILE"
fi

# ⚠ Refuse a dirty tree. The worker commits, so uncommitted edits would be swept into a
# commit nobody wrote a message for — and the range under review would stop matching the
# range that was read.
#
# ⚠ THE DRIVER'S OWN RUNTIME FILES ARE EXCLUDED, and that is what lets the clear above be
# skipped on a dry run without changing what a dry run reports. They are never part of the
# range and never something the worker could commit, so they are not "dirt" in the sense this
# check means. Without the exclusion, `--dry-run` in a ported lane (no `.gitignore` entry, a
# notes file left by the last cycle) would die here and report a refusal THE REAL RUN WOULD
# NOT HIT — the real run clears them one line up. Measured: the pathspec hides these two and
# still reports every other change.
# ⚠⚠ EXACT REPO-RELATIVE PATHS, COMPARED AS STRINGS — NOT A GIT PATHSPEC (#252). The first
# version excluded `":(exclude)${STOPFILE##*/}"`, putting the BASENAME of an operator-supplied
# path into a pathspec unvalidated, and it failed three ways — all measured:
#
#   * `--stop-file /tmp/somedir/` made the basename EMPTY, and `git status --porcelain --
#     . ':(exclude)'` exits 0 with NO output on any tree. The guard did not narrow, it
#     VANISHED — silently, with nothing to notice it by. A trailing slash is an ordinary typo.
#   * `--stop-file /tmp/*.md` put a glob in the pathspec. Matching is fnmatch WITHOUT
#     FNM_PATHNAME, so it hid every `.md` in the tree at any depth.
#   * `--stop-file /tmp/halt.txt` hid a TRACKED, MODIFIED `halt.txt` at the repo root, because
#     a basename cannot tell those two files apart.
#
# And it was over-narrow in the other direction: `--stop-file sub/halt` excluded `halt` rather
# than `sub/halt`, so the file never matched and the run died with the wrong-cause message
# #246 was filed about.
#
# String equality on the whole repo-relative path has none of those degrees of freedom: no
# pattern syntax, no depth ambiguity, and a path outside the repo simply never matches — which
# is correct, because git will not report it either.
_repo_relative() {  # prints the path relative to REPO_ROOT, or nothing when it is outside
  [[ "$1" == "$REPO_ROOT/"* ]] || return 0
  printf '%s' "${1#"$REPO_ROOT"/}"
}

tree_has_real_dirt() {
  local rel_notes rel_stop line path
  rel_notes="$(_repo_relative "$NOTES_FILE")"
  rel_stop="$(_repo_relative "$STOPFILE")"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="${line:3}"          # porcelain v1 is `XY<space>path`
    [[ -n "$rel_notes" && "$path" == "$rel_notes" ]] && continue
    [[ -n "$rel_stop"  && "$path" == "$rel_stop"  ]] && continue
    return 0
  done < <(git status --porcelain)
  return 1
}

if tree_has_real_dirt; then
  die "working tree is dirty. Commit or stash first — the worker commits, and it would
     otherwise absorb your uncommitted edits into a review it never read."
fi

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

pb_passes() {  # $1 = branch, $2 = repo slug -> SUM of agent_passes, or "unreachable"
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
    # ⚠ SCOPED BY REPO TOO — see OPEN_FILTER. A branch-only sum here includes issues
    # rescoped to another repo, whose counters this worker cannot move; the guard would then
    # read a genuine stall as progress, or progress as a stall, depending on which way the
    # other repo's agent happened to be working.
    q = urllib.parse.quote('repo="%s" && branch_name="%s"'
                           % (sys.argv[2].replace('"', ""), sys.argv[1].replace('"', "")))
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
# ⚠⚠ `repo` IS PART OF THIS FILTER AND WAS MISSING (2026-08-27). A branch-only filter counts
# issues that have been RESCOPED TO ANOTHER REPO but still carry this branch's name — and
# `rescopes` gained a `repo` label the day before, so that population exists. Measured here:
# four findings moved to Artificial-Humanity/FerroStep left this driver reporting 4 open on a
# branch whose own queue was empty. It would have run every review to the ceiling, at the
# per-call spend, and then reported NOT CONVERGED — a loop that cannot finish, caused by
# records it is not allowed to act on.
#
# ⚠ `merge_branch.sh` and `issue.py` both already filter on `repo && branch_name`; this was
# the one of the three that did not. The gate and the driver disagreeing about which issues
# belong to a branch is the same class as two copies of the severity ladder.
# ⚠ SOURCED, NOT JUST READ FROM THE ENVIRONMENT. `merge_branch.sh` and `issue.py` both take
# REPO_SLUG from workflow/config.env and fall back to `origin`; reading only the environment
# here would ignore a slug a ported lane has deliberately SET, and derive a different one from
# origin — reintroducing the gate/driver disagreement this block exists to end, in the one
# case where the two are not the same string.
_WF_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.env"
# shellcheck disable=SC1090
[[ -r "$_WF_CFG" ]] && source "$_WF_CFG"
REPO_SLUG_FILTER="${REPO_SLUG:-}"
if [[ -z "$REPO_SLUG_FILTER" ]]; then
  _url="$(git remote get-url origin 2>/dev/null || echo '')"
  # ⚠ `-n … p` PRINTS ONLY ON A MATCH (#344). Plain `s###` leaves a non-matching string
  # UNCHANGED, so the emptiness test below accepted anything — a bare `Sonora`, an unparsed
  # URL, whatever `git remote get-url` happened to say. `issue.py` derives the same slug with
  # an anchored `re.search` and an explicit "" fallback; this now matches that behaviour.
  REPO_SLUG_FILTER="$(printf '%s' "${_url%.git}" | sed -nE 's#^.*[:/]([^/:]+/[^/:]+)$#\1#p')"
fi
[[ -n "$REPO_SLUG_FILTER" ]] || die "cannot determine the tracker repo slug: config.env leaves REPO_SLUG empty and origin did not resolve. Refusing to run with a branch-only filter, which counts other repos' issues."
OPEN_FILTER="repo=\"$REPO_SLUG_FILTER\" && branch_name=\"$BRANCH\" && state=\"open\""

check_stop() {
  if [[ -e "$STOPFILE" ]]; then
    say "STOP FILE PRESENT ($STOPFILE) — halting before $1."
    rm -f "$STOPFILE"; exit 0
  fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "range        : $RANGE"
  echo "developer    : $DEVELOPER"
  echo "max reviews  : $MAX_REVIEWS   (the definition's fix-pass ceiling + 1)"
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
  "Bash(workflow/scripts/review_cycle.sh:*)" "Bash(./workflow/scripts/review_cycle.sh:*)"
  "Bash(workflow/scripts/request_review.sh:*)" "Bash(./workflow/scripts/request_review.sh:*)"
  # ⚠ THE MERGE IS NOT THIS LOOP'S TO MAKE. `merge_branch.sh` merges to main AND PUSHES once
  # the tracker says the branch is settled — correct when Ozzy runs it deliberately, wrong for
  # an unattended fix loop, which would then be able to land its own work. The driver's job
  # ends at convergence; the merge is a separate, deliberate act.
  "Bash(workflow/scripts/merge_branch.sh:*)" "Bash(./workflow/scripts/merge_branch.sh:*)"
)
# ⚠ `workflow/scripts/issue.py` is deliberately NOT denied: taking, commenting, escalating and moving
# an issue to `review` IS the worker's job, and it is the one path that enforces the counter
# cap and the mandatory-comment rules.

REVIEW_TIPS=()
CONVERGED=0
LAST_SUMMARY=""

for (( review=1; review<=MAX_REVIEWS; review++ )); do
  check_stop "review $review"

  say "review $review of $MAX_REVIEWS over $RANGE"
  NOTES_ARG=()
  [[ -f "$NOTES_FILE" ]] && NOTES_ARG=(--notes-file "$NOTES_FILE")

  set +e
  OUT="$(workflow/scripts/request_review.sh --range "$RANGE" --developer "$DEVELOPER" \
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
    # ⚠⚠ THE MATCH IS DELIBERATELY ANYWHERE-IN-THE-OUTPUT AND MUST STAY THAT WAY. REVIEWER.md
    # §6 forbids the token in any other context precisely because of this grep, and the
    # asymmetry is the whole argument: a false HALT costs one cycle, a false PASS can land work
    # a reviewer said must not land, onto a branch with no protection and a worker instructed
    # to push. Line-anchoring this would trade a cheap recurring cost for a rare expensive one.
    #
    # ⚠ WHAT IS FIXED INSTEAD IS THE DIAGNOSIS. On 2026-08-27 a review that found nothing
    # blocking wrote "there is no MUST-NOT-LAND finding" and halted the cycle it was reporting
    # as clean — one occurrence, in a sentence meaning the opposite, and the message below said
    # only "REVIEW SAYS MUST-NOT-LAND". Reading the log was the only way to tell a real refusal
    # from a §6 violation. Now the matching lines are printed, so the reader decides in one
    # glance without opening anything.
    say "REVIEW SAYS MUST-NOT-LAND — stopping. This needs the owner; do not push."
    say "the line(s) that matched — check whether any is an actual refusal, or §6 prose:"
    grep -n "MUST-NOT-LAND" <<< "$OUT" | sed 's/^/      /' >&2
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
    # ⚠⚠ THE SLUG IS TESTED AGAINST THE REPO, NOT THE BRANCH — and the first version of this
    # guard got that wrong in the direction that punishes a good outcome (Sonora #350).
    #
    # The hazard is real: with a slug naming no repo the tracker knows, `OPEN` is 0 and the
    # driver announced CONVERGED having read nothing. But the first fix asked "does this
    # BRANCH have any issues in any state" — and a branch reviewed clean has none, so a
    # genuinely clean review exited 5 saying "a branch nobody has reviewed". ⚠ THIS DRIVER RAN
    # THE REVIEW ITSELF, so it is the one caller that KNOWS a review happened; borrowing
    # `merge_branch.sh`'s NEVER_REVIEWED reasoning was borrowing a premise that does not hold
    # here. That gate cannot see whether a review ran and must refuse; this one can.
    #
    # ⚠ A clean range is the normal, good outcome (REVIEWER.md §7.5) and the routing rule now
    # sends the larger share of workflow findings to another repo, so a Sonora-clean pass is
    # routine rather than exotic. A guard that fires on the good path gets switched off.
    EVER="$(pb "repo=\"$REPO_SLUG_FILTER\"")"
    if [[ "$EVER" == "unreachable" ]]; then
      say "tracker unreachable while confirming convergence — refusing to call this converged."
      exit 4
    fi
    if [[ "$EVER" == "0" ]]; then
      # ⚠ STATE THE OBSERVATION AND OFFER BOTH BRANCHES — DO NOT NAME ONE CAUSE AS FACT (#357).
      # This said "That is a slug naming a repo the tracker does not have, not a clean review",
      # which is a diagnosis rather than a derivation. A CORRECT slug naming a repo the tracker
      # has no records for yet reaches the identical zero: a repo adopting this deliberately
      # portable lane, on its first cycle, whose first review finds nothing. On that path every
      # clause after the comma was false and the operator was sent to re-check two settings
      # that were already right, with nowhere to go afterwards — the same shape as #350, one
      # case narrower.
      if [[ "$ALLOW_EMPTY_TRACKER" -eq 1 ]]; then
        say "tracker holds no issues at all under repo=\"$REPO_SLUG_FILTER\" — proceeding
            anyway on --allow-empty-tracker."
        CONVERGED=1
        say "CONVERGED after review $review — nothing open (empty tracker, allowed)."
        break
      fi
      say "REFUSING to report convergence: the tracker holds NO issues at all under
          repo=\"$REPO_SLUG_FILTER\", in any state and on any branch — so every count taken
          with it, including the zero above, was taken over nothing. Two things reach this,
          and this script cannot tell them apart:
            1. the slug names a repo the tracker does not have (the common one) — check it
               against \`git remote get-url origin\` and workflow/config.env;
            2. the slug is RIGHT and this repo has simply never had an issue filed — a first
               cycle on a newly adopted lane whose review found nothing. It self-heals as
               soon as any issue is ever filed.
          If you have checked the slug and (2) is your case:  --allow-empty-tracker"
      exit 5
    fi
    CONVERGED=1
    say "CONVERGED after review $review — everything closed, escalated, or out of attempts."
    break
  fi
  if (( review == MAX_REVIEWS )); then
    say "review ceiling reached with $OPEN still open. Not a failure: issues below the
        fix-pass ceiling keep their remaining attempts, and a later cycle may take them."
    break
  fi

  # --- fix pass ------------------------------------------------------------
  check_stop "fix pass $review"
  say "fix pass $review — spawning worker (git push denied)"
  BEFORE_SUM="$(pb_passes "$BRANCH" "$REPO_SLUG_FILTER")"

  WORKER_BRIEF="## This fix pass

You are **$DEVELOPER**, working in \`$REPO_ROOT\` on range \`$RANGE\`. Run by
\`workflow/scripts/review_cycle.sh\`, unattended — **there is no one to ask**, so where you would
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
5. Write what you did to \`$NOTES_FILE\` — the next review is briefed from
   that file and cannot otherwise tell a fix from an omission.
"

  # ⚠⚠ ONE `--append-system-prompt`, NOT TWO FLAGS. The CLI refuses
  # `--append-system-prompt` together with `--append-system-prompt-file`:
  #     Error: Cannot use both --append-system-prompt and --append-system-prompt-file.
  # MEASURED 2026-08-27, on the first fix pass this driver was ever asked to run
  # unattended — the review completed, filed five findings, and the worker died instantly
  # on argument parsing. So the loop has never completed a fix pass, while three files
  # described it as the way to run the lane and the owner authorised it as automation.
  #
  # ⚠ The failure was LOUD and still nearly invisible: the driver's own stop message says
  # "the tree may hold partial work; inspect", which reads as a worker that ran and broke
  # rather than one that never started. Nothing distinguished them.
  #
  # ⚠ The persona goes FIRST and the brief SECOND, preserving the original order — the brief
  # is the run-specific override and must be able to contradict the persona. (DEVELOPER.md
  # also arrives via CLAUDE.md's `@import`, so this is belt-and-braces rather than the only
  # copy; it is kept because dropping it would be a behaviour change smuggled into a
  # syntax fix.)
  WORKER_PROMPT="$(cat "$REPO_ROOT/workflow/DEVELOPER.md")

$WORKER_BRIEF"

  set +e
  claude -p "Address the open issues from this review, then commit." \
    --append-system-prompt "$WORKER_PROMPT" \
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
  AFTER_SUM="$(pb_passes "$BRANCH" "$REPO_SLUG_FILTER")"
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

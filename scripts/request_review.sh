#!/usr/bin/env bash
#
# request_review.sh — run a one-shot code review (Janis) over a commit range.
#
# THIS BLOCKS. The review runs to completion and Janis's summary is printed on stdout.
# There is no tap-back and nothing to wait for afterwards: the review has arrived when
# this script returns. That is the whole reason the persistent reviewer session was
# retired — the old transport could fail silently in both directions, and a worker whose
# reviewer never answered saw a review that simply never arrived.
#
# The persona is static and lives in Personas/REVIEWER.md, passed with --system-prompt-file
# so its several kilobytes never go through shell quoting. Everything that changes per run
# — the range, the review_id, who to address, which pass, what the worker already did —
# is assembled here and passed inline with --append-system-prompt. No temp prompt file is
# written for it (owner, 2026-08-14).
#
# Usage: scripts/request_review.sh --help
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RANGE="origin/main..HEAD"
REVIEW_ID=""
DEVELOPER="Ozzy"
REPO_SLUG="Artificial-Humanity/Sonora"
PASS=1
NOTES=""
NOTES_FILE=""
PRIOR=""
MODEL="opus"
EFFORT="xhigh"
DRY_RUN=0

usage() {
  cat <<'USAGE'
request_review.sh — one-shot code review by Janis. Blocks; prints the review on stdout.

  --range <RANGE>       Commit range to review.        (default: origin/main..HEAD)
                        REVIEW THE WHOLE RANGE YOU WILL PUSH, not the last commit — a push
                        carries every unpushed commit, and this loop has measured the range
                        growing after the request on every cycle it has run.
  --review-id <ID>      Ties the filed issues to this review.
                        (default: the tip SHA of --range)
                        Pass one explicitly when reviewing something with no commit behind
                        it — a working tree, or a directory with no git. Generate it
                        (uuidgen, or timestamp-plus-noun); never invent a SHA-shaped string
                        for something that was never committed.
  --developer <ID>      Agent id the reviewer addresses in issues.   (default: Ozzy)
  --repo <SLUG>         Tracker `repo` field.      (default: Artificial-Humanity/Sonora)
  --pass <N>            Which pass of the cycle this is, 1-3.        (default: 1)
  --prior <ID[,ID...]>  review_ids from earlier passes of THIS cycle. Required from pass 2:
                        Janis is a fresh process and cannot otherwise find its own findings.
  --notes <TEXT>        What you fixed and how, what you rebutted and why.
  --notes-file <PATH>   Same, read from a file. Mutually exclusive with --notes.
  --model <M>           (default: opus)
  --effort <E>          low|medium|high|xhigh|max                    (default: xhigh)
  --dry-run             Print the brief and the exact command, then exit. Costs nothing.
  -h, --help            This.

Exit status is Claude's. NON-ZERO MEANS THE REVIEW DID NOT HAPPEN — say so and push anyway
(AGENTS.md §1); do not treat it as a clean review.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --range)       RANGE="${2:?--range needs a value}"; shift 2 ;;
    --review-id)   REVIEW_ID="${2:?--review-id needs a value}"; shift 2 ;;
    --developer)   DEVELOPER="${2:?--developer needs a value}"; shift 2 ;;
    --repo)        REPO_SLUG="${2:?--repo needs a value}"; shift 2 ;;
    --pass)        PASS="${2:?--pass needs a value}"; shift 2 ;;
    --prior)       PRIOR="${2:?--prior needs a value}"; shift 2 ;;
    --notes)       NOTES="${2:?--notes needs a value}"; shift 2 ;;
    --notes-file)  NOTES_FILE="${2:?--notes-file needs a value}"; shift 2 ;;
    --model)       MODEL="${2:?--model needs a value}"; shift 2 ;;
    --effort)      EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "request_review.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "request_review.sh: $*" >&2; exit 1; }

command -v claude >/dev/null 2>&1 || die "the 'claude' CLI is not on PATH."

# Run from the repo root regardless of where we were invoked, so --range, the persona path
# and the reviewer's own cwd all agree. AGENTS.md §6: code executes from the repo checkout.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die "not inside a git repository."
cd "$REPO_ROOT"

PERSONA="$REPO_ROOT/Personas/REVIEWER.md"
[[ -r "$PERSONA" ]] || die "reviewer persona not readable at $PERSONA"

# --- Resolve the range -----------------------------------------------------
# Refuse an empty range rather than paying for a review of nothing. `git rev-list` also
# fails loudly on a range naming a ref that does not exist, which is the common typo.
COMMITS="$(git rev-list "$RANGE" 2>/dev/null)" \
  || die "cannot resolve range '$RANGE'. Fetch first, or check the ref names."
[[ -n "$COMMITS" ]] && COMMIT_COUNT="$(printf '%s\n' "$COMMITS" | wc -l | tr -d ' ')" \
  || COMMIT_COUNT=0

if [[ "$COMMIT_COUNT" -eq 0 && -z "$REVIEW_ID" ]]; then
  die "range '$RANGE' is empty — nothing to review, and no --review-id to review under.
     If you meant to review an uncommitted working tree, pass --review-id explicitly."
fi

# The tip is the newest commit in the range: for A..B that is B, without parsing the
# range syntax ourselves (which breaks on '...', on a bare ref, and on A..B^).
TIP="$(printf '%s\n' "$COMMITS" | head -1)"
[[ -n "$REVIEW_ID" ]] || REVIEW_ID="$TIP"

[[ "$PASS" =~ ^[0-9]+$ ]] || die "--pass must be a number, got '$PASS'"
if [[ "$PASS" -gt 3 ]]; then
  die "--pass $PASS exceeds the three-pass cap (AGENTS.md §1). Whatever is still open after
     the third review is escalated, not reviewed a fourth time. If the owner has re-armed an
     issue by resetting agent_passes, that is a NEW cycle: start again at --pass 1 with a new
     range."
fi
if [[ "$PASS" -gt 1 && -z "$PRIOR" ]]; then
  die "--pass $PASS needs --prior <earlier review_id[,...]>. Janis is a one-shot process with
     no memory of the previous pass; without the earlier review_id it cannot find the
     findings it is supposed to be resolving, and will re-derive them as new issues."
fi

if [[ -n "$NOTES" && -n "$NOTES_FILE" ]]; then
  die "--notes and --notes-file are mutually exclusive."
fi
if [[ -n "$NOTES_FILE" ]]; then
  [[ -r "$NOTES_FILE" ]] || die "--notes-file not readable: $NOTES_FILE"
  NOTES="$(cat "$NOTES_FILE")"
fi
if [[ "$PASS" -gt 1 && -z "$NOTES" ]]; then
  echo "request_review.sh: WARNING — pass $PASS with no --notes/--notes-file." >&2
  echo "  Janis cannot tell a fix from an omission without them and will re-derive settled" >&2
  echo "  findings, burning a pass that exists to catch regressions in the fix pass." >&2
fi

# --- MCP: give the reviewer PocketBase and nothing else --------------------
# The user-scope config also carries a dozen remote claude.ai servers that a code reviewer
# has no use for; --strict-mcp-config drops all of them. The credential is lifted out of
# ~/.claude.json at run time into a 0600 temp file rather than duplicated into a second
# file on disk, and rather than passed as a --mcp-config STRING, which would put the
# PocketBase password in the process table for every user on the box to read in `ps`.
MCP_CONF="$(mktemp -t pb-mcp-XXXXXX.json)"
chmod 600 "$MCP_CONF"
cleanup() { rm -f "$MCP_CONF"; }
trap cleanup EXIT INT TERM

python3 - "$MCP_CONF" <<'PY' || die "could not extract the pocketbase MCP config from ~/.claude.json"
import json, os, sys
src = os.path.expanduser("~/.claude.json")
with open(src) as fh:
    cfg = json.load(fh)
pb = cfg.get("mcpServers", {}).get("pocketbase")
if not pb:
    sys.exit("no 'pocketbase' server in %s" % src)
with open(sys.argv[1], "w") as fh:
    json.dump({"mcpServers": {"pocketbase": pb}}, fh)
PY

# --- The brief: everything that changes per run ----------------------------
RANGE_LOG="$(git log --oneline --no-decorate "$RANGE" 2>/dev/null || true)"
DIFFSTAT="$(git diff --stat "$RANGE" 2>/dev/null || true)"

BRIEF="## This run

You are reviewing the repository at \`$REPO_ROOT\` (host: $(hostname)). That is your working
directory. The tracker \`repo\` field for everything you file is \`$REPO_SLUG\`.

- **Range to review:** \`$RANGE\` — $COMMIT_COUNT commit(s), tip \`$TIP\`.
- **review_id for THIS pass:** \`$REVIEW_ID\` — set it on every issue you file.
- **Developer to address:** $DEVELOPER. Name them in issue comments. They are blocked on this
  process and will read your stdout when it exits; there is no other channel back.
- **Pass $PASS of at most 3.**

### Commits in range

\`\`\`
${RANGE_LOG:-(none)}
\`\`\`

### Diffstat

\`\`\`
${DIFFSTAT:-(none)}
\`\`\`
"

if [[ -n "$PRIOR" ]]; then
  BRIEF+="
### Earlier passes of this cycle

Prior review_id(s): \`$PRIOR\`

Read those issues before you read the code — you filed them, but you do not remember them.
Resolve what is genuinely cleared, leave open what is not, and skip anything already
\`escalated\`. File new findings under \`$REVIEW_ID\`, not under the prior id.
"
fi

if [[ -n "$NOTES" ]]; then
  BRIEF+="
### What $DEVELOPER says it did since the last pass

$NOTES

Treat this as a claim to check, not a report to accept. Verify the fixes; a finding is
cleared when you have checked it, not when the worker says so.
"
else
  BRIEF+="
### Worker notes

None supplied.
"
fi

if [[ "$PASS" -eq 3 ]]; then
  BRIEF+="
### This is the final pass

The cycle ends here. Run the escalation query and flag what it returns; anything still open
after this review is escalated rather than reviewed again, including findings nobody has
attempted (\`agent_passes = 0\`). The worker pushes after you return.
"
fi

# --- Tools -----------------------------------------------------------------
# --tools sets what EXISTS. Edit, Write and NotebookEdit are absent, so the reviewer is
# read-only against the code structurally rather than by instruction — the one half of the
# worker/reviewer split that can be enforced instead of merely written down.
REVIEWER_TOOLS="Bash,Read,Grep,Glob"

# --allowedTools pre-approves; anything else falls to the auto-mode classifier. Deliberately
# NOT a blanket Bash: a reviewer needs to read history and run tests, not to write.
# pb_record_mutate is here because filing, commenting, closing and escalating all go
# through it — including, unavoidably, delete. REVIEWER.md forbids delete in words because
# the permission layer cannot express "this tool but not that operation".
REVIEWER_ALLOW=(
  "Read" "Grep" "Glob"
  "Bash(git:*)" "Bash(pytest:*)" "Bash(uv:*)" "Bash(.venv/bin/python:*)"
  "Bash(ls:*)" "Bash(rg:*)" "Bash(wc:*)" "Bash(find:*)"
  "mcp__pocketbase__pb_record_list"
  "mcp__pocketbase__pb_record_get"
  "mcp__pocketbase__pb_record_mutate"
  "mcp__pocketbase__pb_schema"
  "mcp__pocketbase__pb_health"
)

# Schema and instance administration are not a reviewer's business. pb_collection_delete in
# particular would drop the tracker itself.
REVIEWER_DENY=(
  "mcp__pocketbase__pb_collection_create"
  "mcp__pocketbase__pb_collection_patch"
  "mcp__pocketbase__pb_collection_delete"
  "mcp__pocketbase__pb_settings"
  "mcp__pocketbase__pb_backup"
)

PROMPT="Review the range named in your brief. File your findings, resolve what is cleared, and print your summary."

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "───── brief (--append-system-prompt) ─────"
  printf '%s\n' "$BRIEF"
  echo "───── command ─────"
  printf 'claude -p %q \\\n' "$PROMPT"
  printf '  --system-prompt-file %q \\\n' "$PERSONA"
  printf '  --append-system-prompt <the brief above> \\\n'
  printf '  --model %q --effort %q --permission-mode auto \\\n' "$MODEL" "$EFFORT"
  printf '  --tools %q \\\n' "$REVIEWER_TOOLS"
  printf '  --allowedTools %s \\\n' "${REVIEWER_ALLOW[*]}"
  printf '  --disallowedTools %s \\\n' "${REVIEWER_DENY[*]}"
  printf '  --strict-mcp-config --mcp-config %s\n' "$MCP_CONF"
  echo "───── (dry run: nothing was called, no review was filed) ─────"
  exit 0
fi

echo "request_review.sh: reviewing $RANGE ($COMMIT_COUNT commit(s)) as review_id $REVIEW_ID, pass $PASS." >&2
echo "request_review.sh: this blocks until the review completes." >&2

set +e
claude -p "$PROMPT" \
  --system-prompt-file "$PERSONA" \
  --append-system-prompt "$BRIEF" \
  --model "$MODEL" \
  --effort "$EFFORT" \
  --permission-mode auto \
  --tools "$REVIEWER_TOOLS" \
  --allowedTools "${REVIEWER_ALLOW[@]}" \
  --disallowedTools "${REVIEWER_DENY[@]}" \
  --strict-mcp-config \
  --mcp-config "$MCP_CONF"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 ]]; then
  echo "request_review.sh: THE REVIEW DID NOT HAPPEN (claude exited $STATUS)." >&2
  echo "  Do not treat this as a clean review. Say so in the commit trail or to the owner," >&2
  echo "  then push anyway — AGENTS.md §1." >&2
fi
exit "$STATUS"

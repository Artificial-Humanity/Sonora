#!/usr/bin/env bash
#
# changeset.sh — a local-only pull request.
#
# A branch gives you a set of commits. A pull request gives you five things a branch does
# not: an IDENTITY, a STATE, a BASE and HEAD, an accumulated REVIEW HISTORY, and a MERGE
# EVENT. Only the review history existed here, tied to individual `branch_name`s — so "is this
# piece of work done?" was a question the tracker could not answer, and the answer lived in
# whoever happened to be narrating the cycle.
#
# This gives you the other four, in PocketBase, with no GitHub and no PR ceremony:
#
#     changeset.sh open --title "…" [--branch B]   cut the record for a branch
#     changeset.sh status [N]           state, open issues, reviews, converged?
#     changeset.sh merge N              refuses unless converged, merges LOCALLY
#     changeset.sh abandon N --why "…"  a dropped set must be distinguishable from a live one
#     changeset.sh list                 open changesets for this repo
#
# ⚠ IT NEVER PUSHES. `merge` merges into the base branch in your local checkout and stops.
# Nothing here reaches the remote — same rule as review_cycle.sh, for the same reason: this
# repo has no branch protection and force-push is unblocked (AGENTS.md §1).
#
set -euo pipefail

REPO_SLUG="Artificial-Humanity/Sonora"
BASE="origin/main"
TITLE=""; DESC=""; WHY=""; NUM=""; BRANCH_ARG=""

usage() { sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; }

CMD="${1:-}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --title) TITLE="${2:?}"; shift 2 ;;
    --description) DESC="${2:?}"; shift 2 ;;
    --base) BASE="${2:?}"; shift 2 ;;
    --repo) REPO_SLUG="${2:?}"; shift 2 ;;
    --why) WHY="${2:?}"; shift 2 ;;
    # ⚠ For opening a record on a branch you are NOT standing on — a placeholder branch whose
    # work starts later. Without it the only way to register one is to reach past this script
    # and write the record by hand, which is how a tool stops being the way a thing is done.
    --branch) BRANCH_ARG="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    [0-9]*) NUM="$1"; shift ;;
    *) echo "changeset.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "changeset.sh: $*" >&2; exit 1; }
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository."
cd "$REPO_ROOT"

pbq() {  # $1 = python body operating on `call`/`TOK`; remaining args become ARGS[]
  # ⚠ THE BODY IS SHIFTED OFF BEFORE THE ARGS ARE PASSED. Without the shift it arrives as
  # ARGS[0] and every positional lands one place late — which surfaced as PocketBase
  # rejecting the *title* for exceeding 300 characters, because the title it was handed was
  # the python source. An off-by-one in argument order that reports as a validation error on
  # an unrelated field.
  local _body="$1"; shift
  python3 - "$REPO_SLUG" "$BASE" "$@" <<PY
import json, os, socket, sys, urllib.parse, urllib.request, urllib.error, datetime
socket.setdefaulttimeout(20)
REPO, BASE = sys.argv[1], sys.argv[2]
ARGS = sys.argv[3:]
pb = json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"]["pocketbase"]["env"]
BURL = pb.get("PB_URL", "http://127.0.0.1:8090")
def call(path, method="GET", body=None, token=None):
    r = urllib.request.Request(BURL + path, method=method)
    if token: r.add_header("Authorization", token)
    d = None
    if body is not None:
        d = json.dumps(body).encode(); r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, d) as x: return x.status, json.loads(x.read() or b"{}")
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")
_, a = call("/api/collections/_superusers/auth-with-password", "POST",
            {"identity": pb.get("PB_EMAIL"), "password": pb.get("PB_PASSWORD")})
TOK = a.get("token")
if not TOK: sys.exit("changeset.sh: cannot authenticate to the tracker")
q = urllib.parse.quote
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000Z")
def cs_by_number(n):
    _, r = call('/api/collections/changesets/records?perPage=1&filter=' +
                q('repo="%s" && number=%s' % (REPO, n)), token=TOK)
    return (r.get("items") or [None])[0]
def open_issue_count(branch):
    _, r = call('/api/collections/issues/records?perPage=1&skipTotal=false&filter=' +
                q('branch_name="%s" && state="open" && escalated=false' % branch), token=TOK)
    return r.get("totalItems", 0)
${_body}
PY
}

case "$CMD" in
open)
  [[ -n "$TITLE" ]] || die "open needs --title"
  BRANCH="${BRANCH_ARG:-$(git rev-parse --abbrev-ref HEAD)}"
  git show-ref --verify --quiet "refs/heads/$BRANCH" \
    || die "no such branch: $BRANCH"
  [[ "$BRANCH" != "HEAD" ]] || die "detached HEAD — cut a branch first; the changeset is the branch."
  [[ "$BRANCH" != "main" ]] || die "refusing to open a changeset on main. Cut a branch: the point
     of a local PR is that the work is somewhere other than where it lands."
  HEAD_SHA="$(git rev-parse HEAD)"
  pbq '
_, r = call("/api/collections/changesets/records?perPage=1&sort=-number&fields=number&filter=" +
            q("repo=\"%s\"" % REPO), token=TOK)
items = r.get("items") or []
n = (items[0]["number"] if items else 0) + 1
st, rec = call("/api/collections/changesets/records", "POST", {
    "repo": REPO, "number": n, "title": ARGS[0], "description": ARGS[1],
    "branch": ARGS[2], "base": BASE, "head_sha": ARGS[3],
    "state": "open", "opened_at": NOW}, TOK)
if st >= 300: sys.exit("failed to open changeset: %s" % json.dumps(rec)[:400])
print("changeset #%d opened  %s  (%s -> %s)" % (n, ARGS[0], ARGS[2], BASE))
print("  request reviews with: scripts/request_review.sh --changeset %d" % n)
' "$TITLE" "$DESC" "$BRANCH" "$HEAD_SHA"
  ;;

status)
  pbq '
n = ARGS[0]
if not n:
    _, r = call("/api/collections/changesets/records?perPage=50&sort=-number&filter=" +
                q("repo=\"%s\" && state=\"open\"" % REPO), token=TOK)
    rows = r.get("items") or []
    if not rows: sys.exit("no open changeset for %s" % REPO)
    cs = rows[0]
else:
    cs = cs_by_number(n)
    if not cs: sys.exit("no changeset #%s" % n)
openc = open_issue_count(cs["branch"])
_, esc = call("/api/collections/issues/records?perPage=100&skipTotal=false&sort=number&fields=number,title&filter=" +
              q("branch_name=\"%s\" && state=\"open\" && escalated=true" % cs["branch"]), token=TOK)
_, allr = call("/api/collections/issues/records?perPage=200&skipTotal=false&fields=number&filter=" +
               q("branch_name=\"%s\"" % cs["branch"]), token=TOK)
print("changeset #%s  [%s]  %s" % (cs["number"], cs["state"], cs["title"]))
print("  %s -> %s   head %s" % (cs["branch"], cs["base"], (cs.get("head_sha") or "")[:7]))
print("  issues filed : %s" % allr.get("totalItems", 0))
print("  open, unescalated : %s   %s" % (openc, "CONVERGED" if openc == 0 else "<- must reach 0 to merge"))
print("  escalated, awaiting the owner : %s %s" % (esc.get("totalItems",0),
      [i["number"] for i in (esc.get("items") or [])]))
' "${NUM:-}"
  ;;

merge)
  [[ -n "$NUM" ]] || die "merge needs a changeset number"
  [[ -z "$(git status --porcelain)" ]] || die "working tree is dirty — commit or stash first."
  # ⚠ Convergence is checked SERVER-SIDE before anything touches git, so a stale local view
  # cannot authorise a merge. Escalated issues do NOT block: escalation means "this can live
  # on main but a human must choose" (AGENTS.md §1) — blocking on them would turn a flag into
  # a gate and strand work waiting on an answer that was never about landing.
  READY="$(pbq '
n = ARGS[0]
cs = cs_by_number(n)
if not cs: sys.exit("no changeset #%s" % n)
if cs["state"] != "open": sys.exit("changeset #%s is already %s" % (n, cs["state"]))
c = open_issue_count(cs["branch"])
print("%s|%s|%s" % (c, cs["branch"], cs["base"]))
' "$NUM")" || die "$READY"
  IFS='|' read -r OPENC BRANCH CSBASE <<< "$READY"
  [[ "$OPENC" == "0" ]] || die "changeset #$NUM has $OPENC open unescalated issue(s). Converge first:
     scripts/review_cycle.sh   (or fix and re-review by hand)"
  LOCAL_BASE="${CSBASE#origin/}"
  echo "merging $BRANCH into $LOCAL_BASE, locally"
  git checkout "$LOCAL_BASE"
  git merge --no-ff "$BRANCH" -m "merge changeset #$NUM: $BRANCH"
  MSHA="$(git rev-parse HEAD)"
  pbq '
cs = cs_by_number(ARGS[0])
call("/api/collections/changesets/records/" + cs["id"], "PATCH",
     {"state": "merged", "merged_sha": ARGS[1], "merged_at": NOW}, TOK)
print("changeset #%s recorded as merged at %s" % (ARGS[0], ARGS[1][:7]))
' "$NUM" "$MSHA"
  echo "⚠ NOT PUSHED. Review \`git log\` and push by hand when you are satisfied."
  ;;

abandon)
  [[ -n "$NUM" ]] || die "abandon needs a changeset number"
  [[ -n "$WHY" ]] || die "abandon needs --why: a dropped set with no reason is indistinguishable
     from one someone forgot about, which is the state this record exists to remove."
  pbq '
cs = cs_by_number(ARGS[0])
if not cs: sys.exit("no changeset #%s" % ARGS[0])
d = (cs.get("description") or "") + "\n\nABANDONED: " + ARGS[1]
call("/api/collections/changesets/records/" + cs["id"], "PATCH",
     {"state": "abandoned", "description": d}, TOK)
print("changeset #%s abandoned" % ARGS[0])
' "$NUM" "$WHY"
  ;;

list)
  pbq '
_, r = call("/api/collections/changesets/records?perPage=100&sort=-number&filter=" +
            q("repo=\"%s\" && state=\"open\"" % REPO), token=TOK)
rows = r.get("items") or []
if not rows: print("no open changesets for %s" % REPO)
for cs in rows:
    print("#%-4s %-28s %-22s open-issues=%s" % (
        cs["number"], cs["title"][:28], cs["branch"][:22], open_issue_count(cs["branch"])))
'
  ;;

""|-h|--help) usage ;;
*) die "unknown command: $CMD" ;;
esac

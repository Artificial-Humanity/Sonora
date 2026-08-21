#!/usr/bin/env python3
"""issue.py — every standard tracker operation in the review loop, as one command.

    workflow/scripts/issue.py list      [--branch B] [--state S]
    workflow/scripts/issue.py show      N
    workflow/scripts/issue.py file      --title T (--body B | --body-file F) [--label L] [--branch B]
    workflow/scripts/issue.py take      N [N ...]          Ozzy: agent_passes += 1, BEFORE any work
    workflow/scripts/issue.py review    N --comment C      Ozzy: addressed, awaiting Janis
    workflow/scripts/issue.py escalate  N --comment C      Ozzy: the owner must decide
    workflow/scripts/issue.py close     N [--comment C]    Janis: verified resolved
    workflow/scripts/issue.py reopen    N --comment C      Janis: not resolved
    workflow/scripts/issue.py comment   N --text T
    workflow/scripts/issue.py escalated                    what the owner owes a decision on

WHY THIS EXISTS. The workflow's rules were prose in three files, and this repo's most
expensive recurring lesson is that **a rule in a file is not an enforcement mechanism**. Every
rule below is one an agent could previously break by forgetting:

  * a comment is MANDATORY on `reopen` and `escalate`, optional elsewhere -- refused here,
    not merely requested;
  * `number` does not auto-assign and must be unique per repo -- allocated here, with a
    retry when two writers race for the same one;
  * a new issue must carry `branch_name`, `state=open`, `agent_passes=0` -- set here;
  * `agent_passes` moves UP, by ONE, and never past 3 -- enforced here;
  * only legal state transitions are accepted (see TRANSITIONS).

⚠ IT DOES NOT DELETE. There is no delete subcommand and there must never be one: the tracker
is the sole record that a finding ever existed.

⚠ IT NEVER WRITES `user_decision`. That is the owner's field; a server-side hook in the
AI-Lab-AMD repo returns a decided issue to `open` and resets the counter.

Reads PocketBase credentials from ~/.claude.json (mcpServers.pocketbase.env). Stdlib only.
"""

import argparse
import datetime
import importlib.util
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

socket.setdefaulttimeout(20)

# ⚠ THE HIGHEST NUMBER ANY RECORD HAS EVER USED, live or exported. Allocation never goes at
# or below it (issue #168). `notes/tracker-export-2026-08-17.json` holds 79 records numbered
# 12–120 and the live collection cannot see them, so the unique index alone does not protect
# the range — and the collection has been wiped once already.
NUMBER_FLOOR = 120

def _config():
    """workflow/config.env, plus a derived repo slug. See that file for why it is derived.

    ⚠ THE SLUG IS DERIVED FROM `origin` BY DEFAULT, and that is a portability decision with a
    real failure behind it: a hardcoded slug that survives a copy of `workflow/` into another
    repo files that repo's issues against the ORIGINAL, where they look entirely normal and
    nothing ever flags them.
    """
    cfg = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.env")
    try:
        for line in open(path, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    if not cfg.get("REPO_SLUG"):
        import subprocess
        try:
            url = subprocess.run(["git", "remote", "get-url", "origin"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
            cfg["REPO_SLUG"] = m.group(1) if m else ""
        except Exception:
            cfg["REPO_SLUG"] = ""
    return cfg


CFG = _config()
REPO_SLUG = CFG.get("REPO_SLUG") or ""
MAX_PASSES = int(CFG.get("MAX_PASSES") or 3)
# ⚠ READ, NOT DECORATIVE. Until 2026-08-20 this setting existed and nothing consumed it
# (#216). `cmd_grade` now uses it to decide who may LOWER a severity, which is the one
# tracker write that can clear the merge gate without closing anything (#218).
REVIEWER_NAME = (CFG.get("REVIEWER_NAME") or "").strip()
# ⚠ IMPORTED, NOT REDECLARED. The first draft of this defined its own ("low", "medium",
# "high", "critical") tuple "shared with merge_floor.LADDER" — a comment asserting a
# relationship that nothing enforced. Two copies of a severity ORDER disagree silently: the
# gate would rank one way and this refusal another, and the symptom would be a grade that is
# accepted here and re-classified there. One copy, in the module that owns the comparison.
_MF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_floor.py")
_SPEC = importlib.util.spec_from_file_location("merge_floor", _MF)
_MERGE_FLOOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MERGE_FLOOR)
SEVERITY_LADDER = _MERGE_FLOOR.LADDER
COMMENT_MAX = int(CFG.get("COMMENT_MAX") or 1500)
OPEN_STATES = ("open", "review", "escalated")

# The state machine from workflow/WORKFLOW.md. `escalated -> open` is absent on purpose: only
# the owner's decision releases an escalation, and a server-side hook performs it.
TRANSITIONS = {
    "review": {"from": ("open",), "who": "Ozzy"},
    "escalate": {"from": ("open",), "who": "Ozzy"},
    "close": {"from": ("review", "open"), "who": "Janis"},
    "reopen": {"from": ("review",), "who": "Janis"},
}


def die(msg):
    sys.exit("issue.py: %s" % msg)


class PB:
    def __init__(self):
        try:
            env = json.load(open(os.path.expanduser("~/.claude.json")))[
                "mcpServers"]["pocketbase"]["env"]
        except Exception as e:
            die("cannot read PocketBase credentials from ~/.claude.json (%s)" % e)
        self.base = env.get("PB_URL", "http://127.0.0.1:8090")
        try:
            self.tok = self.call(
                "/api/collections/_superusers/auth-with-password", "POST",
                {"identity": env.get("PB_EMAIL"), "password": env.get("PB_PASSWORD")},
            )[1]["token"]
        except Exception as e:
            die("cannot authenticate to the tracker at %s (%s)" % (self.base, e))

    def call(self, path, method="GET", body=None):
        req = urllib.request.Request(self.base + path, method=method)
        tok = getattr(self, "tok", None)
        if tok:
            req.add_header("Authorization", tok)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
        except Exception as e:
            die("tracker unreachable at %s (%s)" % (self.base, e))

    def find(self, args, number):
        st, r = self.call(
            "/api/collections/issues/records?perPage=1&filter="
            + urllib.parse.quote('repo="%s" && number=%d' % (args.repo, number)))
        # ⚠ A REFUSED QUERY IS NOT A MISSING ISSUE. `if st == 200 else []` collapsed the two
        # into one message, and "no issue #100 in ..." sends the reader to look for a record
        # that is sitting right there (2026-08-18).
        if st != 200:
            die("lookup refused by the tracker: %s" % json.dumps(r)[:400])
        items = r.get("items") or []
        if not items:
            die("no issue #%d in %s" % (number, args.repo))
        return items[0]

    def patch(self, rec_id, body):
        st, r = self.call("/api/collections/issues/records/" + rec_id, "PATCH", body)
        if st >= 300:
            die("write refused by the tracker: %s" % json.dumps(r)[:400])
        return r

    def add_comment(self, args, rec, text, author):
        # ⚠ Comments live in their own collection; `issues.comments` is frozen legacy and must
        # not be written. Kept short by policy -- the owner capped comment length because a
        # reviewer averaged 1839 characters and detail belongs in the issue body.
        text = text.strip()
        if len(text) > COMMENT_MAX:
            die("comment is %d characters; the cap is %d. Put detail in the issue body, which "
                "allows 200,000." % (len(text), COMMENT_MAX))
        # ⚠ `posted_at` IS STAMPED HERE BECAUSE NOTHING ELSE STAMPS IT. This collection has
        # no `created` system field -- it was built to mirror the imported GitHub comments,
        # which carried their own timestamps. A comment written through this script had
        # `posted_at=""` and `seq=0`, so a reader sorting by time got every agent comment in
        # one undifferentiated block ahead of the imported ones (2026-08-18).
        # ⚠ `seq` IS STAMPED TOO, AND IT IS THE ORDERING FIELD THE READERS USE (issue #109).
        # Stamping only `posted_at` fixed `show` and left the documented query wrong:
        # `REVIEWER.md` §4 gives `sort="seq"` verbatim, and every comment this script had
        # ever written carried the schema default of `0`. Measured on #92 — four comments —
        # Ozzy's replies came back ahead of the Janis findings they answer, so the thread
        # read backwards for the one role that is told to run that query.
        st, prev = self.call(
            "/api/collections/issue_comments/records?perPage=200&fields=seq&filter="
            + urllib.parse.quote('issue="%s"' % rec["id"]))
        if st != 200:
            die("could not read the comment sequence: %s" % json.dumps(prev)[:400])
        nxt = max([int(c.get("seq") or 0) for c in (prev.get("items") or [])], default=0) + 1
        st, r = self.call("/api/collections/issue_comments/records", "POST",
                          {"issue": rec["id"], "author": author, "body": text, "seq": nxt,
                           "posted_at": datetime.datetime.now(datetime.timezone.utc)
                           .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"})
        if st >= 300:
            die("comment refused: %s" % json.dumps(r)[:400])


def require_comment(args, action):
    """⚠ THE MANDATORY-COMMENT RULE, AS A MECHANISM RATHER THAN A SENTENCE.

    `reopen` spends one of Ozzy's three attempts, and doing that without saying what is still
    wrong spends it on a guess. `escalate` asks the owner for a decision, and one arriving with
    no question attached cannot be answered. Both were prose rules before this script existed.
    """
    if not (args.comment or "").strip():
        die("%s requires --comment. %s" % (action, {
            "reopen": "Sending an issue back consumes one of Ozzy's three attempts; without a "
                      "reason it is spent on a guess.",
            "escalate": "The owner cannot answer a question that was not asked.",
        }[action]))


def show_row(i):
    # `--` rather than blank for an ungraded issue: blank reads as "nothing to say here",
    # and this column is the one the merge gate blocks on.
    return "  #%-5s %-10s %-8s passes=%-2s %s" % (
        i.get("number"), i.get("state"), i.get("severity") or "--",
        i.get("agent_passes") or 0, (i.get("title") or "")[:52])


def cmd_list(pb, args):
    clauses = ['repo="%s"' % args.repo]
    if args.branch:
        clauses.append('branch_name="%s"' % args.branch)
    if args.state:
        clauses.append('state="%s"' % args.state)
    elif not args.all:
        clauses.append('state!="closed"')
    st, r = pb.call("/api/collections/issues/records?perPage=200&skipTotal=false&sort=number"
                    "&fields=number,state,title,agent_passes,branch_name,severity&filter="
                    + urllib.parse.quote(" && ".join(clauses)))
    if st != 200:
        die("query refused: %s" % json.dumps(r)[:300])
    items = r.get("items") or []
    if r.get("totalItems", 0) > len(items):
        die("result was paged (%d of %d) -- refusing to report a partial list"
            % (len(items), r["totalItems"]))
    if not items:
        print("no issues match")
        return
    for i in items:
        print(show_row(i))
    print("  -- %d issue(s)" % len(items))


def cmd_show(pb, args):
    rec = pb.find(args, args.number)
    for k in ("number", "state", "severity", "agent_passes", "branch_name", "author",
              "labels", "title"):
        print("%-13s %s" % (k, rec.get(k)))
    if (rec.get("user_decision") or "").strip():
        print("\nUSER DECISION (the owner's; outranks any finding)\n%s" % rec["user_decision"])
    print("\n%s" % (rec.get("body") or "")[:4000])
    # ⚠⚠ THIS QUERY RETURNED HTTP 400 FOR EVERY ISSUE, AND THE STATUS WAS DISCARDED, SO
    # `show` PRINTED ZERO COMMENTS FROM THE DAY IT WAS WRITTEN. `issue_comments` has no
    # `created` system field, and PocketBase refuses a sort on a field that does not exist.
    # Measured 2026-08-18: 28 comment records in the collection, #92 had four and #100 had
    # two, and `show` displayed none of them for any issue. That is the whole reopen->fix
    # channel -- `reopen` and `escalate` REFUSE to run without a comment, and the worker
    # could not read the one thing the reviewer was required to write.
    #
    # ⚠ The status is checked now. An empty comment list and a refused query must not print
    # the same thing; that is the same silent-negative this script exists to remove.
    # ⚠ `seq` FIRST — IT IS THE DOCUMENTED ORDER AND THE ONLY RELIABLE ONE (issue #152).
    # Sorting `posted_at,seq` put a reply ABOVE the comment it answers: reproduced live on
    # #149, where `posted_at` order reads seq 1, 3, 2, 4. `REVIEWER.md` §4 gives
    # `sort="seq"` as the canonical query, and `seq` has been stamped per issue since #109
    # precisely so that this is answerable.
    #
    # ⚠ THE FIRST VERSION OF THIS PARAGRAPH EXPLAINED IT WRONGLY, TWICE (issue #156). It
    # asserted a mechanism — "the two writers stamp posted_at from different clocks" —
    # which is a guess about how the other writer works, not something measured from here.
    # v2 then blamed "the imported GitHub comments", a population with ZERO records in this
    # instance: the collection holds nothing numbered below 90. ⚠ An exact count stood here
    # and was stale within a day — the loop that writes this file also files issues into the
    # collection it was counting (issue #165). A number that its own subject changes does not
    # belong in a comment.
    #
    # ⚠ MEASURED, AND `posted_at` DOES NOT RESCUE THE ZERO BLOCK. 20 comments carry
    # `seq = 0` — the agent-written ones from before #109 stamped it — and **14 of those 20
    # also carry `posted_at = ""`**, so the second key is a constant for them; two issues
    # hold two rows tied on BOTH keys. Their order is unrecoverable, which is why a `seq`
    # backfill stays on the table rather than being closed off by this sort. `seq` is per
    # issue, so values repeat across issues by design and that is not a migration artifact.
    st, r = pb.call("/api/collections/issue_comments/records?perPage=100&sort=seq,posted_at"
                    "&filter=" + urllib.parse.quote('issue="%s"' % rec["id"]))
    if st != 200:
        die("comment query refused: %s" % json.dumps(r)[:400])
    items = r.get("items") or []
    if not items:
        print("\n(no comments)")
    for c in items:
        print("\n--- %s (%s)\n%s"
              % (c.get("author"), (c.get("posted_at") or "")[:19], c.get("body")))


def cmd_file(pb, args):
    body = args.body
    if args.body_file:
        body = open(args.body_file, encoding="utf-8").read()
    if not (body or "").strip():
        die("an issue with no body is a title someone has to guess at. Use --body or --body-file.")
    branch = args.branch or current_branch()
    # ⚠ `number` DOES NOT AUTO-ASSIGN and is unique per repo, so two reviewers filing at once
    # collide. Retried rather than pre-reserved: the unique index is the real arbiter, and a
    # reserved-then-abandoned number leaves a permanent hole in the sequence.
    for attempt in range(6):
        st, r = pb.call("/api/collections/issues/records?perPage=1&sort=-number&fields=number"
                        "&filter=" + urllib.parse.quote('repo="%s"' % args.repo))
        # ⚠ Checked, because the consequence of not checking is not an error message. A
        # refused query leaves `rows` empty, `n` becomes 1, and the POST below collides with
        # issue #1 six times before dying with "could not allocate a free issue number" --
        # a message about numbering, for a fault that has nothing to do with numbering.
        if st != 200:
            die("number lookup refused: %s" % json.dumps(r)[:400])
        rows = r.get("items") or []
        # ⚠ FLOORED, BECAUSE THE UNIQUE INDEX CANNOT SEE THE EXPORT (issue #168). The retry
        # below is the whole of the collision safety, and it only fires on a live record —
        # `notes/tracker-export-2026-08-17.json` is a file on disk and nothing on this path
        # opens it. So on an EMPTY collection (`items: []`, HTTP 200, no error) this used to
        # start at 1 and march cleanly up through #12–#120, reissuing numbers that name
        # different findings in the export. That is not hypothetical: this collection HAS
        # been wiped once, on 2026-08-17.
        #
        # 120 is the export's maximum, not 90: 90–120 is double-booked between the two
        # records already (issue #164), so it is the first number that is unambiguous.
        n = max((rows[0]["number"] if rows else 0), NUMBER_FLOOR) + 1 + attempt
        st, rec = pb.call("/api/collections/issues/records", "POST", {
            "repo": args.repo, "number": n, "title": args.title, "body": body,
            "state": "open", "agent_passes": 0, "branch_name": branch,
            "author": args.author, "labels": [args.label] if args.label else [],
            # ⚠ THE MERGE GATE READS THIS. An issue filed without it is UNGRADED, and
            # merge_branch.sh blocks on ungraded exactly as it blocks on MEDIUM — a floor
            # that lets an unknown through is not a floor. See REVIEWER.md § severity.
            "severity": args.severity or "",
        })
        if st < 300:
            # ⚠ SAY THE SEVERITY, INCLUDING WHEN THERE IS NONE. The line reported the two
            # fields that are always the same on a new issue and stayed silent about the one
            # that decides whether a branch can merge — so filing ungraded looked identical
            # to filing graded, and the consequence surfaced later at the gate (#206).
            print("filed #%d on %s (state=open, agent_passes=0, severity=%s)"
                  % (n, branch, args.severity or "UNGRADED — this BLOCKS the merge"))
            return
        if "number" not in json.dumps(rec):
            die("filing refused: %s" % json.dumps(rec)[:400])
    die("could not allocate a free issue number after 6 attempts")


def cmd_take(pb, args):
    """⚠ INCREMENT FIRST, BEFORE ANY WORK (owner, 2026-08-17).

    A pass that dies halfway has still spent its attempt. Counting afterwards makes a failed
    pass free, and "retry until it works" is the unbounded loop the cap exists to prevent.
    """
    for number in args.numbers:
        rec = pb.find(args, number)
        cur = int(rec.get("agent_passes") or 0)
        if rec["state"] != "open":
            die("#%d is '%s', not 'open' -- only open issues are taken. %s"
                % (number, rec["state"], {
                    "review": "It is already awaiting Janis.",
                    "escalated": "It is waiting on the owner's decision.",
                    "closed": "It is resolved.",
                }.get(rec["state"], "")))
        if cur >= MAX_PASSES:
            die("#%d is at agent_passes=%d, which is the cap. It cannot be attempted again -- "
                "escalate it instead:\n    workflow/scripts/issue.py escalate %d --comment '...'"
                % (number, cur, number))
        pb.patch(rec["id"], {"agent_passes": cur + 1})
        print("#%d taken: agent_passes %d -> %d" % (number, cur, cur + 1))


def transition(pb, args, action, new_state, author):
    rec = pb.find(args, args.number)
    rule = TRANSITIONS[action]
    if rec["state"] not in rule["from"]:
        die("#%d is '%s'; %s moves an issue from %s. See workflow/WORKFLOW.md."
            % (args.number, rec["state"], action, " or ".join(rule["from"])))
    # ⚠ Closing straight from `open` is legal but abnormal -- a duplicate, or a finding
    # withdrawn -- so it must be explained. The normal path, review -> closed, need not be.
    if action == "close" and rec["state"] == "open" and not (args.comment or "").strip():
        die("closing #%d directly from 'open' skips verification, so it needs --comment saying "
            "why (a duplicate, or a finding withdrawn). The normal path is Ozzy setting "
            "'review' first." % args.number)
    if (args.comment or "").strip():
        pb.add_comment(args, rec, args.comment, author)
    pb.patch(rec["id"], {"state": new_state})
    print("#%d: %s -> %s" % (args.number, rec["state"], new_state))


def cmd_review(pb, args):
    # ⚠ WARN IF THE PASS WAS NEVER TAKEN. `agent_passes` is the cap's whole mechanism and it
    # is incremented by the WORKER, first thing — but nothing checked, and on 2026-08-19 a
    # full fix pass went by with `take` skipped entirely: four issues were fixed, commented
    # and moved to `review` with one of them still reading 0. The reviewer noticed, not the
    # tooling, and `review_cycle.sh` treats "the worker failed to advance agent_passes" as a
    # stop condition — so an unattended run would have halted on it.
    #
    # ⚠ A WARNING, NOT A REFUSAL, and that is deliberate. Zero is also what the OWNER's dial
    # reads after a deliberate re-arm, and refusing here would make the tooling argue with
    # them. It is also too late to fix by then: the honest response is to say so in the pass
    # notes, not to backfill a counter, which would be writing a number to make the record
    # look right.
    rec = pb.find(args, args.number)
    if int(rec.get("agent_passes") or 0) == 0:
        print("issue.py: ⚠ #%d is moving to `review` with agent_passes = 0.\n"
              "  Either `take` was skipped this pass — the counter is the cap's mechanism and\n"
              "  a pass that spent no attempt is not recorded — or the owner re-armed it\n"
              "  deliberately, in which case ignore this. Do NOT backfill it to look right;\n"
              "  say which it was in the pass notes." % args.number, file=sys.stderr)
    transition(pb, args, "review", "review", args.author)


STE_SENTENCE_WORDS = 25


def ste_warnings(text):
    """⚠ AN ESCALATION COMMENT IS ADDRESSED TO THE OWNER, SO IT IS WRITTEN IN ASD-STE100
    (owner, 2026-08-17). It is the one thing in the tracker written *to* the owner rather than
    *near* them — a decision request, not tracker prose.

    ⚠⚠ NECESSARY, NOT SUFFICIENT, AND NOT AN STE PASS. This counts sentence length. It cannot
    see approved vocabulary, active voice, or one-instruction-per-sentence — the parts of the
    standard carrying most of the meaning. Silence here means "no long sentences", never "this
    is STE". The mechanism for the rest is the `ste` skill, which the persona is
    standing-instructed to use.
    """
    out = []
    for raw in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        s = raw.strip()
        if s and len(s.split()) > STE_SENTENCE_WORDS:
            out.append("%d words: %s..." % (len(s.split()), s[:58]))
    return out


def cmd_escalate(pb, args):
    require_comment(args, "escalate")
    warn = ste_warnings(args.comment)
    for w in warn:
        print("⚠ STE — %s" % w)
    if warn:
        print("  (escalation comments are written in ASD-STE100; use the `ste` skill. This "
              "check sees sentence length ONLY — passing it is not an STE pass.)")
    transition(pb, args, "escalate", "escalated", args.author)
    print("⚠ the owner owes a decision here. Tell them; they write `user_decision`, and a hook "
          "returns it to 'open' with a fresh counter.")


def cmd_close(pb, args):
    transition(pb, args, "close", "closed", args.author)


def cmd_reopen(pb, args):
    require_comment(args, "reopen")
    transition(pb, args, "reopen", "open", args.author)


def cmd_grade(pb, args):
    """Set `severity` on an issue that has none, or correct one that is wrong.

    ⚠ THIS EXISTS BECAUSE THE FIELD ARRIVED AFTER THE ISSUES DID. 111 records predate it and
    are ungraded, and the merge floor blocks on ungraded — so without this the gate could
    refuse a branch and offer no way to clear it but closing a real finding. A gate whose
    remedy is unreachable teaches people to bypass the gate.

    ⚠ IT DOES NOT TOUCH `state`. Grading is a judgement about how bad a finding is; it is not
    a way to move an issue through the loop, and conflating the two would let a grade close
    something.

    ⚠⚠ AND LOWERING A SEVERITY IS THE WORKER MARKING ITS OWN HOMEWORK (#218). It does not
    close the finding, but it does the merge-relevant half: the issue stays open, the gate
    reclassifies it as RIDE, and the branch lands. DEVELOPER.md forbids the equivalent through
    the other door — "CLOSE NOTHING … a worker closing its own findings … defeats the one
    thing the split buys" — and there was no such sentence for `grade` in any file, while the
    gate's own refusal text told the blocked party to run it with `low` listed first.

    So: raising is open to anyone, grading an UNGRADED issue is open to anyone (that is the
    remedy the gate must keep reachable), and LOWERING an existing grade is the reviewer's
    alone. The asymmetry is the point — no direction that can clear a gate is self-served.
    """
    rec = pb.find(args, args.number)
    was = (rec.get("severity") or "").strip()
    new_sev = args.severity.strip().lower()

    if was:
        try:
            lowering = SEVERITY_LADDER.index(new_sev) < SEVERITY_LADDER.index(was.lower())
        except ValueError:
            # An unknown value on either side. Refuse rather than guess a direction — the
            # gate blocks on unrecognised severities anyway, so guessing helps nobody.
            die("cannot compare severity %r with the stored %r; both must be one of %s"
                % (new_sev, was, ", ".join(SEVERITY_LADDER)))
        if lowering and args.author != REVIEWER_NAME:
            die("refusing: lowering %s -> %s makes a blocking finding RIDE past the merge\n"
                "     gate, which is the worker marking its own homework. Only %s may lower a\n"
                "     grade (workflow/config.env: REVIEWER_NAME).\n"
                "       Disagree with the grade? Argue it in the issue's comments and let the\n"
                "       reviewer regrade or refuse — a review is a report, not an order."
                % (was, new_sev, REVIEWER_NAME or "the reviewer"))

    st, r = pb.call("/api/collections/issues/records/" + rec["id"], "PATCH",
                    {"severity": new_sev})
    if st >= 300:
        die("grade refused by the tracker: %s" % json.dumps(r)[:400])
    # ⚠ COMMENT FIRST WOULD BE BETTER but the comment needs the record; if this write fails
    # the grade stands with no record of who or why (#222, LOW — filed, riding to a follow-up).
    if args.comment:
        pb.add_comment(args, rec, args.comment, args.author)
    print("#%d graded: %s -> %s" % (args.number, was or "UNGRADED", new_sev))


def cmd_comment(pb, args):
    rec = pb.find(args, args.number)
    pb.add_comment(args, rec, args.text, args.author)
    print("commented on #%d" % args.number)


def cmd_escalated(pb, args):
    st, r = pb.call("/api/collections/issues/records?perPage=200&skipTotal=false&sort=number"
                    "&fields=number,state,title,agent_passes,branch_name,user_decision"
                    "&filter=" + urllib.parse.quote(
                        'repo="%s" && state="escalated"' % args.repo))
    # ⚠ NOT `if st == 200 else []`. This is the queue the OWNER reads to find what is
    # waiting on a decision, and degrading a refused query to an empty list prints
    # "nothing is escalated" -- the most reassuring possible rendering of a broken query.
    if st != 200:
        die("escalation query refused: %s" % json.dumps(r)[:400])
    items = r.get("items") or []
    if not items:
        print("nothing is escalated")
        return
    print("awaiting a decision from the owner:")
    for i in items:
        print(show_row(i))
        if (i.get("user_decision") or "").strip():
            print("     ⚠ already carries a user_decision but is still 'escalated' -- the hook "
                  "should have released it. Do not act; report it.")


def current_branch():
    import subprocess
    try:
        b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        die("cannot determine the branch; pass --branch")
    if not b or b == "HEAD":
        die("detached HEAD -- the branch is the unit of work. Pass --branch.")
    return b


def main():
    # ⚠ `--repo` AND `--author` ARE ON BOTH THE PARENT AND EVERY SUBCOMMAND, VIA `parents=`.
    # With them on the parent alone, argparse accepts them only BEFORE the subcommand, so the
    # natural `issue.py reopen 97 --author Janis` died with a bare usage dump that named no
    # cause. A tool the loop is meant to reach for cannot have a wrong-looking word order.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=REPO_SLUG,
                        help="tracker repo slug; defaults to config.env or the origin remote")
    common.add_argument("--author", default=os.environ.get("ISSUE_AUTHOR", ""),
                        help="Janis or Ozzy; also read from $ISSUE_AUTHOR")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name):
        return sub.add_parser(name, parents=[common])

    s = add("list"); s.add_argument("--branch"); s.add_argument("--state")
    s.add_argument("--all", action="store_true", help="include closed"); s.set_defaults(fn=cmd_list)

    s = add("show"); s.add_argument("number", type=int); s.set_defaults(fn=cmd_show)

    s = add("file")
    s.add_argument("--title", required=True); s.add_argument("--body")
    s.add_argument("--body-file"); s.add_argument("--branch")
    s.add_argument("--label", choices=["bug", "documentation", "enhancement"])
    # Not `required=True`: a refusal here would push a reviewer mid-review into filing
    # nothing rather than filing ungraded, and an ungraded finding on the tracker beats a
    # finding that only ever existed in a summary. The MERGE GATE is where it bites.
    # ⚠ DERIVED. These were a third copy of the ladder; a value added to merge_floor.LADDER
    # and not here would be rankable by the gate and rejected at the command line.
    s.add_argument("--severity", choices=list(SEVERITY_LADDER))
    s.set_defaults(fn=cmd_file)

    s = add("grade"); s.add_argument("number", type=int)
    s.add_argument("--severity", required=True, choices=list(SEVERITY_LADDER))
    s.add_argument("--comment")
    s.set_defaults(fn=cmd_grade)

    s = add("take"); s.add_argument("numbers", type=int, nargs="+")
    s.set_defaults(fn=cmd_take)

    for name, fn in (("review", cmd_review), ("escalate", cmd_escalate),
                     ("close", cmd_close), ("reopen", cmd_reopen)):
        s = add(name)
        s.add_argument("number", type=int)
        s.add_argument("--comment", default="")
        s.set_defaults(fn=fn)

    s = add("comment"); s.add_argument("number", type=int)
    s.add_argument("--text", required=True); s.set_defaults(fn=cmd_comment)

    add("escalated").set_defaults(fn=cmd_escalated)

    args = p.parse_args()
    # ⚠ EVERY WRITING SUBCOMMAND BELONGS IN THIS TUPLE. `grade` was added 2026-08-20 and the
    # tuple was not extended (#211), so a grade could land unattributed — and severity is the
    # field the merge gate reads, which makes "who decided this blocks?" a question someone
    # will ask. A new write subcommand must be added here in the same commit that adds it.
    if not args.author and args.cmd in ("file", "review", "escalate", "close", "reopen",
                                        "comment", "grade"):
        die("--author is required for writes (Janis or Ozzy), or set $ISSUE_AUTHOR. "
            "An unattributed comment cannot be answered.")
    args.fn(PB(), args)


if __name__ == "__main__":
    main()

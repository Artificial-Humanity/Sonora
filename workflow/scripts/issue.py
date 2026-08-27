#!/usr/bin/env python3
"""issue.py — every standard tracker operation in the review loop, as one command.

    workflow/scripts/issue.py list      [--branch B] [--state S]
    workflow/scripts/issue.py show      N
    workflow/scripts/issue.py file      --title T (--body B | --body-file F) [--label L] [--branch B]
    workflow/scripts/issue.py grade     N --severity low|medium|high|critical
                                        writes the field the MERGE GATE reads. Raising is
                                        open; lowering an existing grade is the reviewer's
    workflow/scripts/issue.py take      N [N ...] [--note D]  Ozzy: agent_passes += 1, BEFORE any
                                        work; at the ceiling, --note routes it to escalated
    workflow/scripts/issue.py review    N --comment C      Ozzy: addressed, awaiting Janis
    workflow/scripts/issue.py dispute   N --kind finding|severity|scope --comment C
                                        Ozzy: I disagree and have NOT fixed it. Spends
                                        `disputes`, never `agent_passes` — a rebuttal must
                                        not cost a fix pass or nobody rebuts
    workflow/scripts/issue.py escalate  N --comment C      Ozzy: the owner must decide
    workflow/scripts/issue.py close     N [--comment C]    Janis: verified resolved
    workflow/scripts/issue.py reopen    N --comment C      Janis: not resolved
    workflow/scripts/issue.py comment   N --text T
    workflow/scripts/issue.py escalated                    what the owner owes a decision on

WHY THIS EXISTS. The workflow's rules were prose in three files, and this repo's most
expensive recurring lesson is that **a rule in a file is not an enforcement mechanism**. Every
rule below is one an agent could previously break by forgetting:

  * `number` does not auto-assign and must be unique per repo -- allocated here, with a
    retry when two writers race for the same one;
  * a new issue must carry `branch_name`, `state=open`, `agent_passes=0` -- set here;
  * ⚠ STATE MOVES AND THE PASS CAP ARE THE REFEREE'S SINCE PHASE 2 (owner directive,
    2026-08-24): take/review/escalate/close/reopen shell out to `ferrostep move`, and the
    rules they obey -- who may move what, the counter ceiling, where exhaustion routes,
    which moves need a note -- are DATA in workflow/sonora-lane.json, enforced by the
    engine, not code here. This module keeps the jobs the referee deliberately does not
    do: numbering, filing, comments, severity, and the queries.

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
# ⚠ MAX_PASSES IS GONE FROM config.env (phase 2): the ceiling is `agent_passes.max` in
# workflow/sonora-lane.json, and the engine enforces it — "out of attempts" and "escalate
# it" are one fact there (`on_exhausted`), not two that can disagree.
# ⚠ READ, NOT DECORATIVE. Until 2026-08-20 this setting existed and nothing consumed it
# (#216). `cmd_grade` uses it to decide who may LOWER a severity, which is the one
# tracker write that can clear the merge gate without closing anything (#218).
# ⚠ RESOLVED FROM THE ROSTER (config.yaml at the repo root) SINCE 2026-08-24, LAZILY:
# config.env dropped its identity keys when the FerroStep roster became the one place
# identities live, and only the grade guards need the name — the suite must import this
# module on machines where the `ferrostep` binary is not installed.
_REVIEWER_CACHE = None


def reviewer_name():
    global _REVIEWER_CACHE
    if _REVIEWER_CACHE is None:
        import subprocess
        roster = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
        try:
            out = subprocess.run(
                ["ferrostep", "agent-env", "--agent", "reviewer",
                 "--roster", roster, "--format", "json"],
                capture_output=True, text=True, check=True).stdout
            name = (json.loads(out).get("name") or "").strip()
        except Exception as e:
            die("cannot resolve the reviewer from the roster %s:\n       %s\n"
                "     `ferrostep agent-env` installs with FerroStep (~/.cargo/bin); the\n"
                "     roster is config.yaml at the repo root." % (roster, e))
        if not name:
            # The reader refuses an empty entry, so this is belt-and-braces — but an
            # empty reviewer would FAIL OPEN in ungraded_guard_blocks, which is the one
            # direction that guard must never fail.
            die("the roster %s resolved an empty reviewer name" % roster)
        _REVIEWER_CACHE = name
    return _REVIEWER_CACHE
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

# ⚠ THE STATE MACHINE LIVES IN workflow/sonora-lane.json AND THE REFEREE ENFORCES IT
# (phase 2). This table only maps a subcommand to the (role, target state) it REQUESTS —
# routing, not rules: legality, notes and the counter are the engine's refusals now.
# `escalated -> open` has no subcommand on purpose: only the owner's decision releases an
# escalation, and a server-side hook performs it.
ROLE_FOR = {
    "review": ("developer", "review"),
    "dispute": ("developer", "disputed"),
    "escalate": ("developer", "escalated"),
    "close": ("reviewer", "closed"),
    "reopen": ("reviewer", "open"),
}

# ⚠ THE KINDS A DISPUTE CAN BE, AS A FIELD ON THE NOTE RATHER THAN A STATE PER KIND. One
# lifecycle, so one state: three states would triple the transition table to record something
# a word already carries. Required at the command line because "what are you disputing" is
# exactly the ambiguity this state exists to remove -- a bare rebuttal reads as all three.
DISPUTE_KINDS = ("finding", "severity", "scope")


def die(msg):
    sys.exit("issue.py: %s" % msg)


# ⚠⚠ ASSEMBLED FROM TWO FRAGMENTS, AND NEVER PRINTED (#361). `review_cycle.sh` greps the
# REVIEW OUTPUT for this token and halts the cycle on any occurrence, anywhere, deliberately:
# a false halt costs one cycle, a false pass can land work a reviewer refused. Nothing
# guarded the other end. Issue 355 was filed with the token in its TITLE, so a later summary
# reporting closures by title -- a natural shape, and §7 asks only for numbers rather than
# forbidding titles -- halts a clean cycle and routes it to the owner.
#
# Spelling it here would put a live copy in a file the reviewer runs and whose output can
# reach that summary, which is the same trap one layer along. So it is built at runtime and
# the refusal below describes it instead of quoting it.
ABORT_TOKEN = "MUST-NOT" + "-LAND"


def refuse_abort_token(text, where):
    """The cycle-abort token must not enter the tracker.

    ⚠ MATCHED EXACTLY AS THE DRIVER MATCHES IT -- literal and case-sensitive, because
    `grep -q` is. A looser test here would refuse prose the driver would never halt on,
    which is a false refusal bought for nothing.

    ⚠ AND THERE IS DELIBERATELY NO BYPASS FLAG. Unlike `--allow-empty-tracker`, no legitimate
    case needs the literal in a record: a finding ABOUT the token can always describe it and
    cite the issue number, which is how #361 itself was written. A flag here would be a
    supported way to re-arm the trap -- and an escape hatch is only ever as safe as a default
    nobody is watching (#359).
    """
    if ABORT_TOKEN in (text or ""):
        die("this %s carries the cycle-abort token that review_cycle.sh greps for.\n"
            "  A tracker record holding it is a trap: any later review summary quoting this\n"
            "  text halts a CLEAN cycle and routes it to the owner (#361, #355).\n"
            "  Write it without the literal -- say \"the cycle-abort token\" and cite the\n"
            "  issue number. REVIEWER.md §6 carries the sanctioned phrasings." % where)


def redact(text):
    """Blunt the cycle-abort token on the way OUT of the tracker, wherever it is printed.

    ⚠⚠ THE WRITE GUARD CANNOT REACH WHAT IS ALREADY STORED. `refuse_abort_token` stops the
    NEXT record carrying the literal; it does nothing about the ones filed before it existed,
    and those are live: issue 355 carries it in its TITLE, 201/246/355 in their bodies (#361).
    Rewriting closed records would settle it too, and is deliberately NOT what this does --
    see #361 for that argument. This side needs no ruling: printing a record is not the same
    act as storing one, and the trap only springs when the text is PRINTED into a summary.

    ⚠ THE MEASURED ROUTE IS THE DOCUMENTED ONE. `issue.py list --branch <b> --all` -- the
    reroute REVIEWER.md §4 tells a reviewer to run when the MCP transport is stale -- prints
    titles, so following the instructions as written put the literal into the reviewer's
    context. That is not an unusual keystroke; it is the sanctioned read command.

    ⚠ REPLACED, NOT DROPPED. The reader still needs to see that something was there, or a
    title reads as though it were written that way. The marker names the issue that explains
    it, and cannot itself match what the driver greps for.
    """
    return (text or "").replace(ABORT_TOKEN, "[cycle-abort token, redacted — #361]")


def refuse_unpostable_comment(text, where="comment"):
    """Everything that makes a comment unpostable, in ONE place that can be called EARLY.

    ⚠ THIS EXISTS TO BE CALLED BEFORE THE STATE MOVE, NOT WHERE THE COMMENT IS WRITTEN.
    Both checks used to live inside `add_comment`, which `transition` reaches only AFTER
    `ferrostep_move` has already persisted the same text as the event note. So a refusal
    arrived with the record already moved, the note already stored, and the comment
    dropped -- and for `reopen`, whose comment the definition makes mandatory, that left
    the developer an unexplained reopen and one of three fix passes spent guessing (#362).

    ⚠ THE LENGTH CAP IS HOISTED WITH THE TOKEN, NOT JUST THE TOKEN. It is the same
    act-then-die shape and it drops the same mandatory comment; splitting them would fix
    one instance of the defect and leave its twin two lines away.
    """
    text = (text or "").strip()
    refuse_abort_token(text, where)
    if len(text) > COMMENT_MAX:
        die("%s is %d characters; the cap is %d. Put detail in the issue body, which "
            "allows 200,000." % (where, len(text), COMMENT_MAX))


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
        # ⚠ RE-CHECKED HERE ON PURPOSE, THOUGH `transition` ALREADY CHECKED. `cmd_comment`
        # reaches this method WITHOUT a state move, so this is the only gate on that path.
        # Two calls, one function -- not two copies of the rule that can disagree (#362).
        refuse_unpostable_comment(text)
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


# ⚠ The mandatory-comment rule moved twice: prose -> a refusal here (`require_comment`,
# 2026-08-13..24) -> the ENGINE (`requires_note` on the definition's transitions, phase 2).
# The middle step is deleted, not kept as a second gate: two refusals for one rule disagree
# the day one of them is edited.


def show_row(i):
    # `--` rather than blank for an ungraded issue: blank reads as "nothing to say here",
    # and this column is the one the merge gate blocks on.
    return "  #%-5s %-10s %-8s passes=%-2s %s" % (
        i.get("number"), i.get("state"), i.get("severity") or "--",
        i.get("agent_passes") or 0, redact(i.get("title"))[:52])


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
        print("%-13s %s" % (k, redact(str(rec.get(k)))))
    if (rec.get("user_decision") or "").strip():
        print("\nUSER DECISION (the owner's; outranks any finding)\n%s"
              % redact(rec["user_decision"]))
    print("\n%s" % redact(rec.get("body"))[:4000])
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
              % (c.get("author"), (c.get("posted_at") or "")[:19], redact(c.get("body"))))


def cmd_file(pb, args):
    body = args.body
    if args.body_file:
        body = open(args.body_file, encoding="utf-8").read()
    if not (body or "").strip():
        die("an issue with no body is a title someone has to guess at. Use --body or --body-file.")
    # ⚠ TITLE FIRST: it is the surface a summary quotes when reporting closures (#361).
    refuse_abort_token(args.title, "title")
    refuse_abort_token(body, "body")
    branch = args.branch or current_branch()
    # ⚠ `number` DOES NOT AUTO-ASSIGN and is unique per repo, so two reviewers filing at once
    # collide. Retried rather than pre-reserved: the unique index is the real arbiter, and a
    # reserved-then-abandoned number leaves a permanent hole in the sequence.
    for attempt in range(6):
        # ⚠⚠ GLOBAL MAXIMUM, NOT THIS REPO'S (Sonora #345, high). The filter here used to be
        # `repo="<this repo>"`, and a RESCOPE OUT OF THE REPO LOWERS THAT MAXIMUM — so the
        # next filing reissues a number that is already in use, for a different finding, under
        # a different repo. The unique index is `(repo, number)`, so nothing refuses it and
        # nothing warns.
        #
        # REPRODUCED LIVE 2026-08-27, by the reviewer's own first filing of the pass: four
        # findings had been rescoped Sonora -> FerroStep as 340-343, dropping Sonora's maximum
        # to 339, and the next four issues filed took 340, 341, 342, 343. **All four now have
        # a twin with the same number on the same branch.** It had already happened once
        # unnoticed: two records are numbered #255 and two commits on `main` disagree about
        # which one they mean.
        #
        # ⚠ THE SEAM WAS ALREADY VISIBLE IN THIS FUNCTION AND NOBODY READ IT. `NUMBER_FLOOR`
        # is documented three lines below as "the highest number ANY record has ever used,
        # live or exported" — a GLOBAL floor, applied to a PER-REPO maximum. Two scopes in one
        # expression, and the one that was wrong is the one that moves.
        #
        # A number in this tracker is cited bare — "#255" in a commit message, with no repo —
        # so it has to mean one thing across the whole collection. Per-repo numbering would be
        # fine in a tracker whose issues never move between repos; `rescopes` gained a `repo`
        # label on 2026-08-26 and this one's do.
        st, r = pb.call("/api/collections/issues/records?perPage=1&sort=-number&fields=number")
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


FERROSTEP_TIMEOUT = 60


def ferrostep_move(pb, rec, role, to_state, note, actor):
    """One state move, refereed by the engine against workflow/sonora-lane.json.

    The token is the session this module already authenticated -- no second auth. A refusal
    is the PRODUCT here, not an error to soften: the engine says which rule refused and what
    would satisfy it, so its stderr is printed verbatim.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmd = ["ferrostep", "move",
           "--workflow", os.path.join(root, "workflow", "sonora-lane.json"),
           "--store", "pocketbase:" + pb.base,
           "--map", os.path.join(root, "workflow", "issues.map.json"),
           "--record", rec["id"], "--role", role, "--to", to_state,
           "--actor", actor or role]
    if (note or "").strip():
        cmd += ["--note", note.strip()]
    env = dict(os.environ, FERROSTEP_POCKETBASE_TOKEN=pb.tok)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=FERROSTEP_TIMEOUT)
    except FileNotFoundError:
        die("the `ferrostep` binary is not on PATH. Since phase 2 the state machine is the\n"
            "     referee's, and issue.py no longer moves state itself. Install it:\n"
            "       cargo install --path <FerroStep checkout>/ferrostep-cli --locked")
    except subprocess.TimeoutExpired:
        die("ferrostep move timed out after %ds. ⚠ UNREACHABLE IS NOT REFUSED: the record\n"
            "     may or may not have moved -- read it before retrying." % FERROSTEP_TIMEOUT)
    if p.returncode != 0:
        die("the referee refused #%s (%s -> %s):\n%s"
            % (rec.get("number"), rec.get("state"), to_state,
               (p.stderr or p.stdout or "(no output)").strip()))
    out = (p.stdout or "").strip()
    if out:
        print(out)


def cmd_take(pb, args):
    """⚠ INCREMENT FIRST, BEFORE ANY WORK (owner, 2026-08-17).

    A pass that dies halfway has still spent its attempt. Counting afterwards makes a failed
    pass free, and "retry until it works" is the unbounded loop the cap exists to prevent.

    Since phase 2 the spend is the engine's open -> open developer move, and the ceiling is
    `agent_passes.max` in workflow/sonora-lane.json. ⚠ The attempt that finds the ceiling
    already spent is REFUSED unless --note says what decision the owner is being asked for;
    WITH the note it ROUTES the issue to `escalated`, the note riding the event.
    """
    note = getattr(args, "note", None)
    # ⚠ CHECKED ONCE, ABOVE THE LOOP, AND BEFORE ANY MOVE. This note rides the event as a
    # tracker write exactly as a `--comment` does, and it was the ONE write surface with no
    # guard at all -- the escalation route's note, which is the note the owner is meant to
    # read (#362). Hoisting it out of the loop matters for the same reason it is hoisted
    # above `ferrostep_move`: one note serves every number, so a per-iteration check would
    # move the first issue before refusing the second.
    refuse_abort_token(note, "note")
    for number in args.numbers:
        rec = pb.find(args, number)
        ferrostep_move(pb, rec, "developer", "open", note, args.author)


def transition(pb, args, action, author):
    role, to_state = ROLE_FOR[action]
    # ⚠ VALIDATE, THEN MOVE, THEN COMMENT -- and the validation is hoisted while the WRITE
    # is not. Move-before-write still holds for the reason below. What did not hold was
    # checking the text where it is written: `add_comment` is reached only after
    # `ferrostep_move` has persisted the identical text as the event note, so every refusal
    # fired too late to prevent the write it existed to prevent (#362).
    #
    # ⚠ ABOVE `pb.find` TOO, NOT MERELY ABOVE THE MOVE. The comment is a pure-text argument
    # and needs no record to judge, so refusing it before ANY tracker contact makes the
    # invariant one a test can state without qualification: a refusable comment touches the
    # tracker zero times. Ordered after the lookup it is still correct and no longer simple
    # -- "no writes, but one read" is the kind of caveat that later grows an exception.
    refuse_unpostable_comment(args.comment)
    rec = pb.find(args, args.number)
    # ⚠ MOVE FIRST, COMMENT SECOND. The engine's refusal must not leave a comment describing
    # a move that never happened. The comment is the human-readable trail (issue_comments);
    # the same text rides the move as its --note, which is the event's copy -- one utterance
    # recorded at two surfaces with two readers, not a maintained copy that can drift.
    ferrostep_move(pb, rec, role, to_state, args.comment, author)
    if (args.comment or "").strip():
        pb.add_comment(args, rec, args.comment, author)
    print("#%d: %s -> %s" % (args.number, rec["state"], to_state))


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
    transition(pb, args, "review", args.author)


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
    # ⚠ The note requirement is the ENGINE's now (open -> escalated carries
    # `requires_note` in the definition), so a bare escalate is refused there, not here.
    warn = ste_warnings(args.comment)
    for w in warn:
        print("⚠ STE — %s" % w)
    if warn:
        print("  (escalation comments are written in ASD-STE100; use the `ste` skill. This "
              "check sees sentence length ONLY — passing it is not an STE pass.)")
    transition(pb, args, "escalate", args.author)
    print("⚠ the owner owes a decision here. Tell them; they write `user_decision`, and a hook "
          "returns it to 'open' with a fresh counter.")


def cmd_dispute(pb, args):
    """Disagree with a finding on the record, at a price the loop can afford.

    ⚠⚠ THIS EXISTS BECAUSE `review` MEANT TWO THINGS. The developer's only exits from `open`
    were take-it, move-to-`review`, or escalate -- and `review` is the SAME STATE whether the
    finding was fixed or rebutted. So a disagreement, once made, was indistinguishable from a
    capitulation the moment it landed, and the lane's dispute rate read as zero across 256
    records. It was never zero; it was UNRECORDED (owner, 2026-08-27).

    ⚠ IT SPENDS `disputes`, NOT `agent_passes`, AND THAT IS THE POINT. Pricing disagreement at
    parity with compliance is the defect: if arguing costs a fix pass, arguing is what you stop
    doing. A dispute the reviewer rejects lands back at `open`, where proceeding costs one pass
    as it always did -- so being WRONG is priced, and disagreeing is not.

    ⚠ ONE RE-DISPUTE, THEN THE OWNER (`disputes.max` in workflow/sonora-lane.json -- their
    dial, like `agent_passes.max`, and not this module's to correct). A state nothing spends
    would let open -> disputed -> open cycle free, and `agent_passes` is otherwise the only
    thing guaranteeing this lane terminates at all.

    ⚠ AN OWNER RULING DOES NOT REFUND IT. `release.reset_counters` clears `agent_passes` and
    deliberately leaves `disputes` spent: after the owner settles an argument the developer
    gets a fresh budget to do the WORK and no further budget to RE-ARGUE. Widening that later
    is cheap; discovering a worker re-litigated a ruling is not.

    The note requirement is the ENGINE's (`requires_note` on the transition), not restated here.
    """
    args.comment = "[dispute:%s] %s" % (args.kind, (args.comment or "").strip())
    transition(pb, args, "dispute", args.author)
    print("⚠ #%d is now DISPUTED and is %s's to answer. You have not fixed it and must not: "
          "the branch still carries it, and the merge gate still weighs it at its severity."
          % (args.number, reviewer_name()))


def cmd_close(pb, args):
    # ⚠ review -> closed needs no note; open -> closed (withdraw/dedupe) REQUIRES one.
    # Both rules are the definition's, and the engine refuses -- not this module.
    transition(pb, args, "close", args.author)


def cmd_reopen(pb, args):
    transition(pb, args, "reopen", args.author)



def ungraded_guard_blocks(caller, rec_author, rec_branch, here, new_sev, floor,
                          reviewer=None):
    """Should `grade` refuse this write? Pure, so it can be tested without a tracker.

    ⚠ EXTRACTED BECAUSE THE INLINE VERSION WAS WRONG TWICE AND ITS ONLY TEST WAS A SOURCE
    SCAN (#231). A two-line `assert "SOME STRING" in src` cannot distinguish a guard that
    works from a guard that is merely present — #228 and #229 were both live under one.

    ⚠ AND BECAUSE THE ALTERNATIVE WAS LIVE FIXTURES. Proving the earlier version by hand meant
    grading a real record on another branch, which Janis correctly flagged as a worker setting
    severity on a reviewer's finding as a byproduct of a test. A pure function needs no record
    to exist, and leaves nothing to clean up server-side.

    All four conditions are necessary. Any one of them false and the write is allowed:

      * the CALLER is not the reviewer   — the reviewer is exempt (#228: the first version
                                           tested only the issue's author, so its own advice,
                                           "ask Janis", sent Janis to Janis);
      * the FINDING is the reviewer's    — a worker's own filing is its own to grade;
      * it is stamped with THIS branch   — makes the legacy carve-out real rather than
                                           rhetorical (#229: 102 of 102 ungraded records are
                                           the reviewer's, so an author-only test caught the
                                           whole legacy set it claimed to protect);
      * the grade is BELOW the floor     — grading up cannot clear a gate.
    """
    reviewer = reviewer_name() if reviewer is None else reviewer
    if not reviewer:
        return False                       # nobody is the reviewer; nothing to protect
    if caller == reviewer:
        return False
    if (rec_author or "") != reviewer:
        return False
    if (rec_branch or "") != (here or ""):
        return False
    f, n = (floor or "").strip().lower(), (new_sev or "").strip().lower()
    if f not in SEVERITY_LADDER or n not in SEVERITY_LADDER:
        return False                       # not comparable; the caller's own checks apply
    return SEVERITY_LADDER.index(n) < SEVERITY_LADDER.index(f)


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

    So, as implemented (see `ungraded_guard_blocks` for the predicate itself):

      * RAISING is open to anyone — grading up cannot clear a gate;
      * grading an UNGRADED issue is open to anyone EXCEPT all four of: the caller is not the
        reviewer, the finding is the reviewer's, it is stamped with the branch being merged,
        and the new grade is below the floor;
      * LOWERING an existing grade is the reviewer's alone.

    ⚠ The middle clause said "except where the reviewer filed it" until #235 — the rule from
    before the branch scope existed, and broader than the code by two conditions. A docstring
    stating a rule the function does not implement is what #212 and #225 were, in this file.

    ⚠ THIS IS A CONVENTION, NOT A MECHANISM, AND AN EARLIER VERSION OF THIS DOCSTRING CLAIMED
    OTHERWISE (#225). It said "no direction that can clear a gate is self-served", which does
    not follow: `--author` is SELF-DECLARED, so anyone can pass `--author Janis` and the check
    waves them through. There is no authentication here and none is available — `issue.py`
    runs as whoever runs it, exactly as the git identity in DEVELOPER.md §1 is a convention
    the repo cannot enforce.

    What the check actually buys is that the bypass must be TYPED ON PURPOSE and leaves a
    self-declared author on the record. That is worth having and it is not a guarantee; the
    difference is the whole of #218's lesson applied to #218's own fix.
    """
    rec = pb.find(args, args.number)
    was = (rec.get("severity") or "").strip()
    new_sev = args.severity.strip().lower()

    # ⚠ THE UNGRADED DOOR, NARROWED TO WHAT IT WAS ACTUALLY FOR (#225, corrected by #228/#229).
    #
    # The first version tested only `rec["author"]`, and it was wrong twice over:
    #
    #   * it never consulted the CALLER, so the refusal caught the reviewer too — and its own
    #     remedy said "ask Janis to grade it", which sent Janis to Janis (#228);
    #   * it described a narrow new class while MEASURING as the whole legacy set: 102 of 102
    #     ungraded records are authored by the reviewer, so the carve-out it claimed to protect
    #     was empty (#229). Together those made the below-floor remedy reachable by NOBODY, for
    #     NO record — inside the file that argues an unreachable remedy teaches bypassing.
    #
    # Three conditions now, all necessary: the CALLER is not the reviewer, the FINDING is the
    # reviewer's, and it is stamped with the branch being merged. That last one is what makes
    # the legacy carve-out real instead of rhetorical — an old record carries an old
    # branch_name and stays gradeable.
    _here = current_branch()
    floor = (_MERGE_FLOOR.floor_setting() or "").strip().lower()
    if not was:
        if ungraded_guard_blocks(args.author, rec.get("author"), rec.get("branch_name"),
                                 _here, new_sev, floor):
            die("refusing: #%d is an UNGRADED finding %s filed against '%s', the branch you\n"
                "     are on. Grading it below the floor (%s) would make it RIDE past the merge\n"
                "     gate without anyone having verified it.\n"
                "       Grade it AT or ABOVE the floor, or have %s grade it — %s is exempt from\n"
                "       this refusal, which the first version of this guard was not, so its\n"
                "       advice looped (#228).\n"
                "       Findings on OTHER branches, including every legacy record, are\n"
                "       unaffected: this is scoped to the branch being merged."
                % (args.number, reviewer_name(), _here, floor, reviewer_name(),
                   reviewer_name()))

    if was:
        try:
            lowering = SEVERITY_LADDER.index(new_sev) < SEVERITY_LADDER.index(was.lower())
        except ValueError:
            # An unknown value on either side. Refuse rather than guess a direction — the
            # gate blocks on unrecognised severities anyway, so guessing helps nobody.
            die("cannot compare severity %r with the stored %r; both must be one of %s"
                % (new_sev, was, ", ".join(SEVERITY_LADDER)))
        if lowering and args.author != reviewer_name():
            die("refusing: lowering %s -> %s makes a blocking finding RIDE past the merge\n"
                "     gate, which is the worker marking its own homework. Only %s may lower a\n"
                "     grade (config.yaml: the roster's reviewer entry).\n"
                "       Disagree with the grade? Argue it in the issue's comments and let the\n"
                "       reviewer regrade or refuse — a review is a report, not an order."
                % (was, new_sev, reviewer_name()))

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
    s.add_argument("--note", help="required only when the take finds the ceiling spent: "
                   "the decision being asked for; the engine then routes to escalated")
    s.set_defaults(fn=cmd_take)

    for name, fn in (("review", cmd_review), ("escalate", cmd_escalate),
                     ("close", cmd_close), ("reopen", cmd_reopen)):
        s = add(name)
        s.add_argument("number", type=int)
        s.add_argument("--comment", default="")
        s.set_defaults(fn=fn)

    s = add("dispute"); s.add_argument("number", type=int)
    s.add_argument("--kind", required=True, choices=list(DISPUTE_KINDS),
                   help="what you are disputing: the finding itself, its severity, or "
                        "whether it is in scope for this branch")
    s.add_argument("--comment", default="", help="why. The engine refuses a dispute without "
                   "one -- `requires_note` on the transition, not a check in this module")
    s.set_defaults(fn=cmd_dispute)

    s = add("comment"); s.add_argument("number", type=int)
    s.add_argument("--text", required=True); s.set_defaults(fn=cmd_comment)

    add("escalated").set_defaults(fn=cmd_escalated)

    args = p.parse_args()
    # ⚠ EVERY WRITING SUBCOMMAND BELONGS IN THIS SET. `grade` was added 2026-08-20 and the
    # tuple was not extended (#211), so a grade could land unattributed — and severity is the
    # field the merge gate reads, which makes "who decided this blocks?" a question someone
    # will ask. A new write subcommand must be added here in the same commit that adds it.
    #
    # ⚠⚠ HALF OF IT IS NOW DERIVED, because the warning above did not work. `dispute` was
    # added 2026-08-27 and this tuple was the SECOND copy of "which subcommands move an
    # issue" — the first being `ROLE_FOR`, six lines of which had to agree with seven lines
    # here by hand. Every transition subcommand is a write by construction, so it is read out
    # of `ROLE_FOR` and a new state can no longer arrive unattributed. The three that remain
    # written out are the ones that write something OTHER than a state.
    if not args.author and args.cmd in (set(ROLE_FOR) | {"file", "comment", "grade"}):
        die("--author is required for writes (Janis or Ozzy), or set $ISSUE_AUTHOR. "
            "An unattributed comment cannot be answered.")
    args.fn(PB(), args)


if __name__ == "__main__":
    main()

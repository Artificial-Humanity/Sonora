# The review workflow

Specified by the owner, 2026-08-17. **This file is the map of the loop.** `DEVELOPER.md` and
`REVIEWER.md` are each one role's half of it; where either disagrees with this file, this file
is right and the persona is stale.

The unit of work is a **branch**. Every issue carries its `branch_name`, and that is the only
thing tying findings to work — there is no pull request, no review document, and (since
2026-08-17) no changeset record.

---

## The state machine — the definition IS the map now

⚠⚠ **[sonora-lane.json](sonora-lane.json) is the one copy of the state machine, and the
FerroStep engine ENFORCES it** (owner directive, 2026-08-24 — phase 2 of the cutover).
States, roles, who may move what, which moves need a note, the `agent_passes` ceiling and
where exhaustion routes are all DATA there. `issue.py`'s subcommands only REQUEST moves
(through `ferrostep move`, against [issues.map.json](issues.map.json)); the engine's
refusals replaced both this section's prose rules and issue.py's coded ones. A restated
table here would be the second copy that drifts — this file carried exactly that table
until phase 2, and `issues.state` still has the same four exclusive values it declares.

Two facts stay in words, because they are WHY the data says what it says:

* ⚠ **`closed` is `terminal`, and only Janis writes it. `escalated` is `halted`** — a human
  owes a decision, `ferrostep awaiting` lists it, and no non-human role has a route out.
  The release is the owner's `user_decision`, performed server-side.
* ⚠ **The take is the `open → open` developer move, spending `agent_passes`** — "increment
  FIRST, before any work" as a mechanism. The attempt that finds the ceiling already spent
  is refused without a note saying what decision the owner is being asked for; WITH the
  note it ROUTES to `escalated`, so "out of attempts" and "escalate it" are one fact
  instead of two that can disagree.

---

## 1. Ozzy's initial work

1. The owner asks for work.
2. **Ozzy creates a branch.** All work happens on a branch; the branch is the reviewable unit.
3. Ozzy commits to it as needed.

## 2. The review cycle

1. **Ozzy runs `workflow/scripts/request_review.sh`** to call Janis.
2. **Janis wakes in a `-p` session and asks one question first: are there already issues under
   this `branch_name`?** That answer, and nothing else, decides which review this is.
   * **none → INITIAL review**
   * **any → FOLLOW-UP review**
3. **Initial review.** Janis reviews **every commit on the branch that is ahead of `main`**.
   Each finding becomes an issue: `state: open`, `agent_passes: 0`, `branch_name` set. Janis
   reports **the number of issues found** and is done.
4. **Follow-up review.** Janis takes every issue in `state: review` and decides whether Ozzy
   actually resolved it.
   * **Resolved → `closed`.** A comment is optional.
   * **Not resolved → back to `open`.** ⚠ **A comment is MANDATORY**, saying what is still
     wrong. Sending an issue back with no explanation spends one of Ozzy's three attempts on
     a guess.
   * ⚠ **Janis ALSO reads the new commits as code and files anything new** (owner's ruling,
     2026-08-17). This does not follow from the steps above and is deliberate: this lab has
     measured that the dominant failure of later rounds is **defects introduced by the fixes
     themselves** — issue #62 ran 7 → 5 → 9 → 7 findings, the later rounds mostly its own
     repairs. Verification alone would structurally miss exactly that. New findings are filed
     exactly like an initial review's: `open`, counter `0`.
5. Janis reports **how many issues remain open** and is done.

⚠ **Janis never escalates, and never touches `agent_passes`.**

## 2b. The full code review — a different read, not a bigger one

The owner periodically asks for a **"full code review"**. It is the same loop with one
difference: **there is no commit range.**

```bash
workflow/scripts/full_review.sh          # cuts review-YYYY-MM-DD from main, then reviews
```

1. A branch **`review-YYYY-MM-DD`** is cut from `main` (today's date). It starts with **zero
   commits ahead**, which is exactly the state an ordinary review refuses — so
   `request_review.sh --full` takes a different path through the range guards rather than a
   relaxed one.
2. **Janis reviews the code as it stands.** Commit history is neither the subject nor a scope.
3. **Everything after that is ordinary.** Issues carry `branch_name=review-YYYY-MM-DD`, Ozzy
   takes and fixes them, Janis verifies, escalation and the merge gate behave exactly as in
   §2–§4.

⚠ **WHY IT EXISTS: per-change review is STRUCTURALLY BLIND to accumulation.** It can only see
what changed. It cannot see the doc that quietly stopped being true, the guard nobody has run
since it was written, or the file nothing invokes any more — because none of those are *a
change*. This lab has measured all three. The sweep is the only thing that looks at the whole.

⚠ **A full review cannot read everything well in one pass**, and the brief says so. Janis is
told to report what it covered **and what it did not**. A sweep that quietly skipped a
subsystem is worse than one that names the gap, because the next sweep assumes it was read.

⚠ **Running it again on the same date RESUMES that sweep** — it checks out the existing branch
rather than refusing. A second run is the normal way a sweep continues: pass 1 filed issues,
Ozzy fixed some, and this is the next review.

## 3. Ozzy, after the review

1. Ozzy reads the issues for this branch.
2. **Clears the SEVERITY FLOOR → merge to `main`, push, and the workflow ends.** Use
   `workflow/scripts/merge_branch.sh`, which re-checks the tracker server-side and refuses
   otherwise. ⚠ **The floor is not zero-open-issues** (owner, ratified 2026-08-19, built 2026-08-20).
   **A finding at or above the severity floor set in `workflow/config.env` blocks; anything below it
   rides to a follow-up branch**; ungraded and escalated block at any severity.
   ⚠ **The threshold is deliberately NOT repeated here** — that file is the only place it is
   set, and a second copy is one that goes stale the moment someone reconfigures it. A finding that rides is still open — move it to a
   follow-up branch, or it sits on a `branch_name` that no longer has a branch. The move is
   `ferrostep rescope --set branch_name=…` with a note: a refereed, evented operation that
   replaced the raw tracker writes this lane used to perform (owner reversal, 2026-08-24;
   first proven live on record #292 the same day). ⚠ It refuses a TERMINAL record by
   design — a closed finding's scope is provenance — so move a riding finding while it is
   still open, never after it closes.
   ⚠ **The gate is on the MERGE, not the push** (owner, 2026-08-17): a branch that
   merged legitimately is one whose push is unremarkable.
3. For each issue, in order:
   * **Needs an owner decision → `escalated`.** ⚠ **The note is MANDATORY** — it says what
     decision is being asked for, and the engine refuses without it. An issue OUT OF
     ATTEMPTS escalates through the same door: the `take` that finds the ceiling spent
     refuses until given that note, then routes the issue itself.
   * **Otherwise: `take` FIRST — the engine spends the counter — then fix, commit, and set
     `review`.** A comment is optional.
     * ⚠ **The counter moves BEFORE the work, not after** (owner's ruling, 2026-08-17,
       amending their own step order). A pass that dies halfway — the session dies, the
       context runs out, Ozzy gives up — has still spent its attempt. Counting afterwards
       makes a failed pass free, and "retry until it works" is the unbounded loop the cap
       exists to prevent. **A crashed pass costs a pass.**
4. Ozzy taps Janis, and the cycle repeats from §2.

## 4. Escalation to the owner

1. **Ozzy notifies the owner** of the issues needing a decision. The owner reviews and writes
   guidance into `user_decision`.
2. **A decision returns the issue to `open` and resets `agent_passes` to `0`** — one save, done
   by a server-side hook, not by anyone remembering three edits.
   * The mechanism lives OUTSIDE this repo. Since the 2026-08-24 cutover it is the
     release-hook block inside the generated FerroStep hooks installed on the tracker
     (`ferrostep.issues.pb.js`, which replaced the old `issues_user_decision.pb.js`). A
     review of *this* repo cannot see it — this paragraph is a reviewer's only evidence.
   * ⚠ It releases **only** `escalated` → `open`. A decision written on a `closed` issue is a
     note, and must not reopen resolved work.
   * ⚠ **Both the return to `open` and the counter reset are needed.** Escalation happens at
     `agent_passes = 3`, which has no attempts left, so releasing the state alone would get the
     issue picked up, found out of attempts, and escalated again.
3. Ozzy acts on the guidance and rejoins §3.

---

## Porting this lane to another repo

The goal is a copy plus two lines. `workflow/` is meant to be self-contained: personas, map,
scripts and settings.

1. **Copy `workflow/` into the new repo.**
2. **Add the import to that repo's `CLAUDE.md`.** This is what makes the developer role need no
   flag — and `CLAUDE.md` is the only file Claude Code auto-discovers, while `AGENTS.md` is
   **not**, so this line is what makes any of the rest reachable at all:

   ```markdown
   ## By default you are the developer
   @workflow/DEVELOPER.md
   ```

   (Identities come from the FerroStep roster — copy `config.yaml` to the new repo root
   and edit its names, emails and persona paths there; nothing else restates them.)

3. **Check `workflow/config.env`** — the only file with per-repo settings, and every value has
   a working default. ⚠ **Leave `REPO_SLUG` empty** so it derives from
   `git remote get-url origin`: a hardcoded slug that survives the copy files the *new* repo's
   issues against the *old* one, where they look entirely normal and nothing ever flags them.
4. **Point that repo's `AGENTS.md` here**, in one line, so its rules-of-record names the lane
   rather than restating it.

⚠ **WHAT DOES NOT TRAVEL, AND MUST BE CHECKED:**

* **The tracker's schema.** `issues` and `issue_comments` must exist, with `state` carrying all
  four values — plus, since the referee cutover, `ferrostep_version` on `issues` and the
  `ferrostep_events` collection, created by FerroStep's provisioning, not by these scripts.
* **The referee itself.** The `ferrostep` binary (installed via cargo), and this lane's two
  data files — `workflow/sonora-lane.json` (rename its `name` for the new lane) and
  `workflow/issues.map.json`. Without the binary NO state moves at all; `issue.py` refuses
  by name rather than falling back to writing state itself.
* **The `user_decision` release hook** — since 2026-08-24 a block inside the generated
  FerroStep hooks installed on the tracker. It is what returns a decided issue to `open`
  with a fresh counter. Without it deployed, escalation is a one-way door and every
  decision strands out of attempts.
* **The personas' expertise.** `REVIEWER.md` §2 knows PyTorch, Matcha-TTS and this pipeline's
  failure modes; `DEVELOPER.md` names this repo's git traps. Both still *run* elsewhere, but a
  review is only as good as what the reviewer knows to look for. **Edit §2 for the new repo
  rather than assuming it transfers.**
* **The `ste` skill**, installed machine-wide from AI-Lab-AMD rather than carried here.

---

## What this replaced, and why it is written down

Four things in this map contradict what the repo enforced the day before, and each was a
deliberate owner ruling rather than a drift. They are listed so no one "restores" them:

* **Escalation moved from Janis to Ozzy.** It used to be the reviewer's normal path, with the
  worker allowed it as an exception. It is now the worker's alone. Ozzy holds the change and
  is first to know a question cannot be settled by more reviewing.
* **Escalation now BLOCKS the merge.** It used to be explicitly "a parking space, not a
  blocker — the work still ships". It is now one of the three states that hold a branch back.
* **The changeset record is retired.** A branch already has an identity, and its issues already
  carry its state; the record was a second place for the same truth.
* **`state: review` exists at all.** Ozzy previously had no way to say "addressed" except in a
  comment, so every re-review re-read everything to find out.

And one later than the four, same rule — listed so no one "restores" it either:

* **The state machine left prose and code for data** (owner directive, 2026-08-24, phase 2):
  `workflow/sonora-lane.json`, enforced by the FerroStep engine. issue.py's `TRANSITIONS`
  dict, its cap check and its mandatory-comment refusals are DELETED, not disabled — the
  engine's refusals are the one copy, and re-adding a coded guard beside them would be two
  refusals for one rule, disagreeing the day one is edited.

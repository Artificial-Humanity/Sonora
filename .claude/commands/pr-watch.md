---
description: Check my own open PRs for a finalized review and act on it. Safe to run on a loop.
argument-hint: (no arguments)
allowed-tools: Read, Bash(gh:*), Bash(git:*)
---

# PR watch

Check the pull requests **you** own for a review that has finished, and act on it. This is
the developer's half of the review lane: nothing notifies you, so you look.

Designed to be run on a loop — `/loop 10m /pr-watch` — so **the quiet path must be quiet.**
If there is nothing to do, say one line and stop. A watcher that narrates every lap trains
the reader to ignore it.

## 0. Which PRs are yours

⚠ **You do not own every open PR in this repository.** A repo has one assigned developer,
but floating sessions (the lab manager, a reviewer) open PRs here too, and acting on one of
theirs means pushing commits into work you know nothing about.

Your label is `$CLAUDE_AGENT_LABEL`. Read it from the environment:

```
echo "${CLAUDE_AGENT_LABEL:?no CLAUDE_AGENT_LABEL set — refusing to guess which PRs are mine}"
```

**If it is unset, STOP and say so.** Do not fall back to "all open PRs", do not infer from
branch names, and do not guess from `--author` — every session currently authenticates as the
same GitHub user, so `--author @me` returns everyone's work. Guessing here is how you push to
another developer's branch.

Then:

```
gh pr list --state open --label "$CLAUDE_AGENT_LABEL" \
  --json number,headRefOid,isDraft,title,headRefName
```

## 1. Has the review finished for the CURRENT head?

For each PR, the reviewer stamps every summary it posts with a marker naming the commit it
reviewed. The review is finished **for what is pushed now** only when that marker matches the
PR's head:

```
gh pr view <N> --json comments --jq '.comments[].body' \
  | grep -o 'janis-reviewed: [0-9a-f]\{40\}' | tail -1
```

* **No marker** → the PR has never been reviewed. Waiting. Report nothing.
* **Marker ≠ head** → you have pushed since the last review; a new one is queued or running.
  Waiting. Report nothing.
* **Marker == head** → the review is current. Continue.

⚠ This comparison is what makes the loop idempotent. A fix pass pushes, so head moves, so the
marker no longer matches and you wait again — you cannot fix the same head twice, and you
cannot act on a stale review. Do not replace it with "are there unresolved threads", which is
true continuously and would re-fire every lap.

## 2. What state is the PR in?

Count unresolved threads:

```
gh api graphql -f query='{repository(owner:"OWNER",name:"REPO"){pullRequest(number:N){reviewThreads(first:100){nodes{id isResolved comments(first:50){nodes{author{login} authorAssociation body}}}}}}}'
```

**No unresolved threads** → the review is current and clean. Say so once, name the PR, and
stop. **Do not merge.** Merging is the owner's, and on a draft it is not even possible.

**Unresolved threads exist** → work through ALL THREE of these BEFORE fixing, in order:

1. **Is the owner mid-conversation?** If the LAST comment in any unresolved thread comes from
   an `authorAssociation` of **OWNER**, **MEMBER** or **COLLABORATOR** and reads as a question
   to you or a decision in progress, that thread is a conversation, not a work item.
   Answer it if you can answer it; otherwise leave it and report that the owner has the ball.
   **Never resolve a thread the owner is still talking in.**
2. **How many passes have you already had?** Count your own fix-pass summaries — a badge you
   control, so the count is reliable:

   ```
   gh pr view <N> --json comments --jq '[.comments[] | select(.body | contains("Quincy") and contains("fix pass"))] | length'
   ```

   **At 2 or more, do the escalation in step 3.** ⚠ **This threshold is ADVISORY and it is
   arbitrary.** Do not defend it, and do not override it with a prediction — see below.

   ⚠ **DO NOT TRY TO MEASURE "IS IT CONVERGING?" FROM THE REVIEW TEXT.** An earlier version of
   this file did, with `grep -oE '[0-9]+ finding'`, and it was wrong three ways: the summary
   format never mandates that token so the number often *precedes* it and is missed; the
   pipeline flattens every match from every comment into one list, so it counts matches rather
   than rounds; and the first review of any PR has no predecessor, leaving the comparison
   undefined in the state every PR passes through.

   ⚠ **And a strict-decrease rule is no less strict than this count.** Measured on 2026-08-11:
   #62's rounds were `7 → 5 → 9 → 7`, so "must be smaller than the last" halts it at round 3 —
   exactly where a two-pass cap halts it. The version of this file that replaced the count with
   convergence claimed the count "would have blocked real work" on #62; **its own rule would
   have blocked it in the same place.** No automatic threshold distinguished #62's legitimate
   repair from #65's five flat rounds; the owner's judgement at round 4 did.

   **So report the numbers and let the human weigh them. Do not compute a verdict from them.**
   What *is* worth stating, because it is a mechanism rather than a heuristic: each fix lands in
   code the next review then reads, so a large or prose-heavy diff does not converge by being
   corrected — only by getting smaller.

3. **ESCALATION IS AN ACTION YOU TAKE, NOT A MESSAGE YOU SEND** — but it is a *reporting* action.
   Do these, then report:

   * **Leave every unresolved thread OPEN.** An open thread means outstanding work; resolving one
     to look tidy records it as fixed.
   * **Report the round sizes and what is still open**, so the owner can decide between merging
     and deferring — that call is theirs, and on #62 it was the thing that ended the loop.
   * **PROPOSE how to shrink the diff. Do not perform it.** Say which surface you would delete,
     or what you would carry into a follow-up off `main` — then stop.

   ⚠ **YOU MAY NOT DELETE CONTENT TO STOP THE REVIEWER FINDING THINGS.** An earlier version of
   this file authorised exactly that, on a trigger that same version made unreliable. A false
   positive on a reporting rule costs a wasted lap; a false positive on an autonomous delete
   destroys work, and "the findings stopped" is indistinguishable from "the evidence was
   removed". Deletion may be the right remedy — it was on #65 — but the owner authorises it.

   Only a genuine disagreement or a decision the owner has to make needs to wait for them,
   which is the same bar as filing an issue.

## 3. Report

One line per PR, and nothing at all for PRs that are simply waiting:

```
#61 review current (ffbde01) · 6 unresolved → running fix pass
#63 review current · clean · yours to mark ready
#62 not mine (no agent label match) — skipped
```

If every PR is waiting, say exactly one line: `nothing to act on — N PR(s) awaiting review`.

⚠ **Report an escalation loudly, every lap, until it is resolved.** A blocked PR that goes
quiet after one mention is a PR that gets forgotten — and the whole point of this command is
that nothing else is going to tell anyone.

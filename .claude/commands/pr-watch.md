---
description: Check my own open PRs for a finalized review and act on it. Safe to run on a loop.
argument-hint: (no arguments)
allowed-tools: Read, Bash(gh:*), Bash(git:*)
---

# PR watch

> ⚠ **THE LANE THIS WATCHES IS DISABLED (owner, 2026-08-13).** `Janis — PR Review` is
> `disabled_manually`, so **no review will ever appear on a PR**, and this command's quiet
> path is now its only path — it will report "nothing to do" forever, correctly and
> uselessly. It is kept because the workflow file is kept: `gh workflow enable 330746260`
> brings both back. **Review now happens BEFORE the push**, requested from the `Reviewer`
> session — see AGENTS.md §1. Do not put this on a loop expecting it to catch anything.

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

**Unresolved threads exist** → check these two things BEFORE fixing, in this order:

1. **Is the owner mid-conversation?** If the LAST comment in any unresolved thread comes from
   an `authorAssociation` of **OWNER**, **MEMBER** or **COLLABORATOR** and reads as a question
   to you or a decision in progress, that thread is a conversation, not a work item.
   Answer it if you can answer it; otherwise leave it and report that the owner has the ball.
   **Never resolve a thread the owner is still talking in.**
2. **Have you already had two passes at this?** Count your own fix-pass summaries:

   ```
   gh pr view <N> --json comments --jq '[.comments[] | select(.body | contains("Quincy") and contains("fix pass"))] | length'
   ```

   **At 2 or more, STOP and escalate.** Report the PR, what is still open, and what you would
   do — then leave it for the owner. Two rounds that did not converge is a disagreement or a
   missing decision, and a third lap is the loop this whole design exists to prevent.

Otherwise: run the fix pass, in this session, so the context that wrote the code is the
context that answers for it:

```
/fix-pr <N>
```

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

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

**Unresolved threads exist** → check these two things BEFORE fixing, in this order:

1. **Is the owner mid-conversation?** If the LAST comment in any unresolved thread comes from
   an `authorAssociation` of **OWNER**, **MEMBER** or **COLLABORATOR** and reads as a question
   to you or a decision in progress, that thread is a conversation, not a work item.
   Answer it if you can answer it; otherwise leave it and report that the owner has the ball.
   **Never resolve a thread the owner is still talking in.**
2. **Is the review CONVERGING?** Count the findings each round produced, oldest to newest:

   ```
   gh pr view <N> --json comments --jq '.comments[] | select(.body|contains("janis-reviewed")) | .body' | grep -oE '[0-9]+ finding'
   ```

   **If the newest round is not SMALLER than the one before it, stop — the diff is not
   converging, and another fix pass will not change that.**

   ⚠ **A COUNT OF PASSES IS THE WRONG TRIGGER, measured 2026-08-11.** This rule used to say
   "at 2 or more passes, escalate", and both of that day's long PRs show why it fails in both
   directions:

   * **#62** needed four rounds of legitimate repair across 25 files. A hard cap at two would
     have blocked real work.
   * **#65** ran `4 → 3 → 3 → 4 → 3` — flat over five rounds, every finding in the *prose*
     rather than the substance, which had been correct since the first commit. A cap at two
     was overridden twice with reasoning, once on the explicit claim that "this remedy is
     structural and will end the loop". It did not: the next round showed the fix had removed
     the explanation and kept the one token that was actually drifting.

   **Flatness has a mechanism, which is why it is the better signal: each fix lands in code the
   next review then reads.** So a large or prose-heavy diff never converges by being corrected —
   only by getting smaller. "My next fix is different" is a prediction; the round sizes are a
   measurement.

3. **ESCALATION IS AN ACTION YOU TAKE, NOT A MESSAGE YOU SEND.** Do not stop and wait. Do all
   of this yourself, then report it:

   * **Leave every unresolved thread OPEN.** An open thread means outstanding work; resolving
     one to look tidy records it as fixed.
   * **Shrink the diff.** Either delete the surface generating the findings — if they are all in
     prose or comments, the fix is deletion, not another correction — or carry the remainder
     into a small follow-up PR off `main`, never a stacked branch showing the whole diff.
   * **Say the rate in your report**, not just the lap count: `4 → 3 → 3 → 4 → 3, flat`. That
     is what tells the owner whether to merge and defer.

   Only a genuine disagreement or a decision the owner has to make needs to wait for them —
   which is the same bar as filing an issue.

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

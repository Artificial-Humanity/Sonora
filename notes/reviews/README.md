# notes/reviews — transient review handoffs

The `Reviewer` session drops its report here as **`review-<first 7 of the commit SHA>.md`**.

**These files are deliberately not tracked.** `.gitignore` carries the pattern, for one
reason: the review cycle ends with the file being **deleted**, and an untracked-but-visible
file at the top of `git status` is one `git add -A` away from being committed to `main` by
the very worker the review was for. Ignoring it removes the class.

## The lifecycle

1. Worker commits.
2. Worker asks `Reviewer` for a review of **the whole range it is about to push**
   (`@{push}..HEAD`), naming every commit in it. `Reviewer` writes `review-<sha7>.md` here,
   where `<sha7>` is the **tip of that range at the time of the request**.
3. Worker gets **one** pass at fixing what the review raised.
4. `Reviewer` re-reviews the same range, re-briefed — it now includes the fix commits, so its
   tip has moved and it gets its own `review-<sha7>.md`.
5. Anything still unresolved becomes a **GitHub issue**, filed by the worker.
6. Worker pushes to `main`, then **deletes every `review-*.md` from the cycle** — whether it
   ended in issues or ended clean. Either ending closes it.

⚠ **A cycle normally leaves TWO files here** (step 2's and step 4's), because step 3 commits
and moves the tip. Step 6 deletes both. "The review file", singular, was the first version of
this sentence and it is why a file survived the first cycle.

⚠ **The file is a handoff, not a record.** Nothing here survives the cycle that produced it,
so do not cite one from anywhere else in the repo, and do not treat this directory as review
history — that is what the issues and the git log are for. If a finding is worth keeping, it
is worth filing.

⚠ **An old `review-*.md` lying about means a cycle did not finish**, not that a review is
pending. Check whether its SHA is already on `main` before acting on anything inside it.
⚠ **That check gives the wrong answer after a rebase**, which is why AGENTS.md §1 no longer
pulls before pushing and says to **merge, not rebase**, when integration is actually needed.
A rebase rewrites local commits, so a reviewed SHA can cease to exist and will never appear on
`main` — making a finished cycle look abandoned forever. A merge leaves the reviewed SHAs
intact, so the cycle survives. If something does rebase the range, the cycle is over: delete
these files and start again.

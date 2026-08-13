# notes/reviews — transient review handoffs

The `Reviewer` session drops its report here as **`review-<first 7 of the commit SHA>.md`**.

**These files are deliberately not tracked.** `.gitignore` carries the pattern, for one
reason: the review cycle ends with the file being **deleted**, and an untracked-but-visible
file at the top of `git status` is one `git add -A` away from being committed to `main` by
the very worker the review was for. Ignoring it removes the class.

## The lifecycle

1. Worker commits.
2. Worker asks `Reviewer` for a review, naming the commit. `Reviewer` writes
   `review-<sha7>.md` here.
3. Worker gets **one** pass at fixing what the review raised.
4. `Reviewer` reviews again, after that pass.
5. Anything still unresolved becomes a **GitHub issue**.
6. Worker pushes to `main`, then **deletes the review file** — whether the cycle ended in
   issues or ended clean. Either ending closes it.

⚠ **The file is a handoff, not a record.** Nothing here survives the cycle that produced it,
so do not cite one from anywhere else in the repo, and do not treat this directory as review
history — that is what the issues and the git log are for. If a finding is worth keeping, it
is worth filing.

⚠ **An old `review-*.md` lying about means a cycle did not finish**, not that a review is
pending. Check whether its SHA is already on `main` before acting on anything inside it.

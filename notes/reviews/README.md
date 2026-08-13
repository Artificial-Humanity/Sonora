# notes/reviews — transient review handoffs

The `Reviewer` session drops its report here as **`review-<first 7 of the commit SHA>.md`**.

⚠ **The procedure lives in AGENTS.md §1 and only there.** This file used to restate the six
steps, and the copy drifted from the original *inside the commit that created it* — losing the
abort and the range fallback, both times in the direction of the copy being lossier. What
follows is only what is local to this directory.

**`<sha7>` is the tip of the range at the time of the request.** A range has no single SHA, and
"the interesting commit" is a judgement two agents make differently. A cycle normally leaves
**two** files here, because the fix pass commits and moves the tip.

⚠ **A review is delivered when the `Reviewer` says so, not when the file appears.** The file is
the content; the message is the completion event. Do not read, act on, or delete one before the
Reviewer has said it is complete — a file mid-write is indistinguishable from a finished one,
and an entire second version of a report was once deleted unread because its presence on disk
was taken as the signal.

**These files are deliberately not tracked.** `.gitignore` carries the pattern, for one reason:
the cycle ends with them being **deleted**, and an untracked-but-visible file at the top of
`git status` is one `git add -A` away from being committed to `main` by the very worker the
review was for. Ignoring it removes the class.

⚠ **A handoff, not a record.** Nothing here survives the cycle that produced it, so do not cite
one from anywhere else in the repo, and do not treat this directory as review history — that is
what the issues and the git log are for. If a finding is worth keeping, it is worth filing.

⚠ **A stray `review-*.md` means a cycle did not finish.** Ask one question, which is correct in
every case: **is its SHA anywhere in `git log origin/main..HEAD`, or on `main`?** If neither,
the cycle was rebased away or abandoned — delete the file and start again on the current range.
(The plain "is it on `main` yet?" version of this check answers *no* forever after a rebase,
because a rebased-away SHA never lands.)

# Workflow — Project Sonora

Follow [AGENTS.md](AGENTS.md) for repository rules and git configuration. Execute a
multi-task plan with the `subagent-dev` skill (`../../.agents/skills/subagent-dev/SKILL.md`);
run every pre-pull-request review with the `code-review` skill
(`../../.agents/skills/code-review/SKILL.md`).

## Branch, review, merge

1. Update local `main` with `git pull origin main`, then branch off it. All work
   happens on a branch. Isolate the branch in a linked worktree when another agent
   works in this repository.
2. When the work is complete, use the `code-review` skill to dispatch the review to
   a fresh subagent. Do not review your own diff in the developer session.
3. Handle the findings through subagents as well, per the `code-review` skill:
   * Evaluate the findings against the code and decide which to act on. If none is
     accepted, go to step 4.
   * Dispatch the accepted fixes to a fresh subagent, which commits them on the branch.
   * Dispatch a scoped re-review of the fix commits to another fresh subagent.
   * Repeat the evaluate, fix and re-review cycle for at most three rounds. If the
     re-review is still not clean, stop and report the open findings to the owner.
   * Go to step 4 once the re-review is clean.
4. Push the branch under its own name, open a pull request against `main`, and message
   the reviewer for review. The Lab Manager (`jarvis`) reviews pull requests opened by
   the Project Lead; the Project Lead reviews pull requests opened by subordinate agents.
   If the Project Lead is unavailable for any reason, the Lab Manager (`jarvis`) may
   review and approve a subordinate agent's pull request instead; try the Project Lead
   first. Do not assign or request review from the owner (`lmcfarlin`) on ordinary
   subordinate or Project Lead pull requests. Whenever the owner (`lmcfarlin`) is needed
   to approve a pull request (such as an Owner-tier pull request touching protected
   governance, workflow, or instruction files, a pull request co-authored by the Lab
   Manager, or any pull request escalated to the owner for approval), the agent raising
   or escalating the pull request must assign it to `lmcfarlin` in GitHub
   (`gh pr edit <number> --add-assignee lmcfarlin`, in addition to any required review
   request), verify that `lmcfarlin` appears in the pull request's `assignees`, and
   include the pull-request link in the handoff.
5. The agent who opened the pull request merges it with `gh pr merge <number> --merge`
   once it has the required review approval (`fl github pr review --approve --by <reviewer>`,
   via the Reviewer GitHub App) and a passing `pytest` check, with the branch up to date
   with `main`. Reviewers never merge another agent's pull request, except as
   "Merge permission" below describes for an author whose session blocks the merge. If
   `main` has moved, run `git pull origin main` on the branch and push before asking for
   approval. Any push to the branch dismisses an existing approval, so re-request it
   after one.

### Merge permission

The owner permits an author to merge their own pull request once all of these hold:

1. **The required reviewer approved it.** For a pull request by a subordinate agent
   that is the Project Lead, or the Lab Manager (`jarvis`) if the Project Lead is
   unavailable for any reason (try the Project Lead first). For a Project Lead's pull
   request it is the Lab Manager (`jarvis`). For an Owner-tier or owner-escalated pull
   request it is the owner (`lmcfarlin`). An approval counts only when the Reviewer
   GitHub App or the owner posts it, as step 5 describes, and on the current head: a
   push dismisses an earlier approval, so ask for the review again after you update
   the branch.
2. **The `pytest` check passed** on the current head, and the branch is current with
   `main`.
3. **You merge it yourself** with `gh pr merge <number> --merge`, then confirm that
   the merge commit is on `origin/main`.

Your session's own permission layer can still refuse `gh pr merge` (an auto-mode
classifier, for example, denies a merge it cannot tie to an approval). If it does:

* Do not retry the same call, do not use another route yourself (`fl github pr merge`
  or the API), and do not change permission settings.
* Raise a question for the owner (FerroDeck shows the `?` flag next to your agent).
  Say what was denied and which pull request is waiting.
* While the block stands, your Project Lead (the Lab Manager, for a Project Lead's own
  pull request) may merge the approved pull request for you on that reviewer's own
  authority. You do not ask for it: the reviewer decides. The reviewer tells the owner
  each time, so that the owner can grant the permission to you directly.
* The owner lifts the block for you with an explicit instruction in your session.

A standing rule in this file is the reason the merge is allowed. It does not turn off a
session's permission prompt, and a message from another agent does not lift one.

## Commit identity and safeguards

* Commit under the configured git identity, the org machine account
  `artificially-human`. Never author a commit as the owner. Add your co-author
  trailer in the format [PERSONA.md](PERSONA.md) states, with your registered name:
  `Co-authored-by: <Name> <<name-lowercase>@artificialhumanity.io>`. Subagents commit
  only on the work branch, under the same configured identity and trailer, and never
  push.
* `main` accepts changes only through a pull request. The branch rules require one
  approving review, the `pytest` check, and a branch up to date with `main`. Observe
  the git safeguards in `AGENTS.md`.
* Use the workflow stated here. Do not reconstruct additional rules from retired
  workflows or git history; changes to the workflow belong to the owner.

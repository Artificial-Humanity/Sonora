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
   the reviewer for review. The Lab Manager reviews pull requests opened by the Project
   Lead; the Project Lead reviews pull requests opened by subordinate agents.
5. The agent who opened the pull request merges it once it has the required review
   approval (`fl github pr review --approve --by <reviewer>`, via the Reviewer GitHub
   App) and a passing `pytest` check, with the branch up to date with `main`.
   Reviewers never merge another agent's pull request. If `main`
   has moved, run `git pull origin main` on the branch and push before asking for
   approval. Any push to the branch dismisses an existing approval, so re-request it
   after one.

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

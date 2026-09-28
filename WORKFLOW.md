# Workflow — Project Sonora

Follow [AGENTS.md](AGENTS.md) for repository rules and git configuration.

## Branch, review, merge

1. Update local `main` with `git pull origin main`, then branch off it. All work
   happens on a branch.
2. When the work is complete, use `superpowers:requesting-code-review` to dispatch
   a review.
3. Use `superpowers:receiving-code-review` to evaluate the findings. Address them,
   then commit the fixes.
4. Push the branch under its own name and open a pull request against `main`.
5. Merge the pull request once it has the owner's approval and a passing `pytest`
   check, with the branch up to date with `main`. If `main` has moved, run
   `git pull origin main` on the branch and push before asking for approval. Any push
   to the branch dismisses an existing approval, so re-request it after one.

## Commit identity and safeguards

* Commit under the configured git identity, the org machine account
  `artificially-human`. Never author a commit as the owner. Use
  [PERSONA.md](PERSONA.md) for the agent's co-author identity.
* `main` accepts changes only through a pull request. The branch rules require one
  approving review, the `pytest` check, and a branch up to date with `main`. The
  machine account cannot approve its own pull request; the owner approves. Observe
  the git safeguards in `AGENTS.md`.
* Use the workflow stated here. Do not reconstruct additional rules from retired
  workflows or git history; changes to the workflow belong to the owner.

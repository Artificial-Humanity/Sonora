# Workflow — Project Sonora

Follow [AGENTS.md](AGENTS.md) for repository rules and git configuration.

## Branch, review, merge

1. Branch off local `main`. All work happens on a branch.
2. When the work is complete, use `superpowers:requesting-code-review` to dispatch
   a review.
3. Use `superpowers:receiving-code-review` to evaluate the findings. Address them,
   then commit the fixes.
4. Push the branch under its own name and open a pull request against `main`.
5. Merge the pull request once it has the owner's approval and a passing `pytest`
   check, with the branch up to date with `main`.

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

# Workflow — Project Sonora

**This file holds how work gets done in this repo: branching, committing, landing, and
whatever review or checking sits around them.** [AGENTS.md](AGENTS.md) points here and tells
you to follow it.

---

## Branch, review, merge

**Standing across every project repo in this workspace** (owner, 2026-09-16):

1. **Branch off `main`.** All work happens on a branch.
2. **When the work is complete, call for a review.** Use the `superpowers:requesting-code-review`
   skill to dispatch it.
3. **Receive the review with the `superpowers:receiving-code-review` skill.** Address what it
   finds, then commit the fixes.
4. **Merge to `main`.** Once this review cycle has been followed, a direct push to `main` is
   allowed — a pull request is not required.

⚠ **THIS IS THE FIRST THING WRITTEN HERE SINCE THE 2026-09-15 REMOVAL, AND IT IS DELIBERATELY
SIMPLE.** The previous workflow — an agent review cycle with a reviewer process, an issue
tracker, a state machine, a severity floor and a merge gate — was removed on 2026-09-15, and
none of it was carried back in. This section does not restore it: it is the owner's new,
separate general policy, shared verbatim across the workspace's other repos, not a repaired
version of what was removed. If a removed rule turns out to be needed here, the owner
re-establishes it with its reasoning intact — do not reconstruct it from git history or memory.

## What is true right now, stated so nobody has to infer it

* **Nothing mechanically gates `main`.** There is no branch protection, force-push is
  unblocked, and the merge gate that used to stand there was removed — the review cycle above
  is what the owner asks you to follow, not something tooling enforces. The
  `push.default=simple` refusal described in [AGENTS.md](AGENTS.md) is the only mechanical
  thing left anywhere near it.
* **Commit identity is not automatic.** The repo's configured git identity is the owner's, on
  purpose. `roster.yaml` and `scripts/agent_env.py` resolve the agent's, and
  [docs/personas/DEVELOPER.md](docs/personas/DEVELOPER.md) §1 carries the command and the check.

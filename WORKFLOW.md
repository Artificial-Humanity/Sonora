# Workflow — Project Sonora

**This file holds how work gets done in this repo: branching, committing, landing, and
whatever review or checking sits around them.** [AGENTS.md](AGENTS.md) points here and tells
you to follow it.

---

## ⚠⚠ IT IS EMPTY ON PURPOSE, AND THAT IS NOT THE SAME AS "ANYTHING GOES"

Created 2026-09-16, by the owner, deliberately blank. The previous workflow — an agent review
cycle with a reviewer process, an issue tracker, a state machine, a severity floor and a merge
gate — was removed on 2026-09-15, and the prose describing it was stripped out of `AGENTS.md`
and `CLAUDE.md` on 2026-09-16. **The owner is establishing the replacement next and asked that
none of the old rules be carried over.**

So the absence of a rule here is a decision that has not been made yet, not permission.

⚠ **DO NOT POPULATE THIS FILE FROM THE OLD RULES.** They are in git history and they are
recoverable, which is exactly why reconstructing them from memory is the wrong move: a rule
recalled without its measurement becomes a rule nobody can date or defend, and this repo has
watched a deleted file go on being obeyed from a summary for eight commits. If a removed rule
turns out to be needed, the owner re-establishes it with its reasoning intact.

⚠ **THE OWNER WRITES WHAT GOES HERE.** An agent proposing a rule is fine; an agent installing
one into this file is the same failure as an agent writing its own permissions.

---

## What is true right now, stated so nobody has to infer it

These are facts about the current state, not rules, and they are here because their absence is
the thing most likely to be misread:

* **Nothing reviews your work.** There is no reviewer, no tracker, no findings, no severity
  floor.
* **Nothing gates `main`.** There is no branch protection, force-push is unblocked, and the
  merge gate that used to stand there was removed. The `push.default=simple` refusal described
  in [AGENTS.md](AGENTS.md) is the only thing left anywhere near it.
* **Commit identity is not automatic.** The repo's configured git identity is the owner's, on
  purpose. `roster.yaml` and `scripts/agent_env.py` resolve the agent's, and
  [docs/personas/DEVELOPER.md](docs/personas/DEVELOPER.md) §1 carries the command and the check.

---

## Until this file says otherwise

Use judgement, work on a branch, and **ask the owner when a decision is theirs** rather than
inventing a rule and writing it down as though it were settled.

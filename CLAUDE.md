# Project Sonora

**[AGENTS.md](AGENTS.md) is this repo's rules of record. Read it before you do anything else.**
It is not loaded for you — only this file is — so nothing else will put it in front of you.

**[WORKFLOW.md](WORKFLOW.md) is how work gets done here. Read it and follow it.**

@PERSONA.md

---

⚠ **THIS FILE IS A PASSTHROUGH AND NOTHING ELSE** (owner, 2026-09-16). It used to carry role
precedence, the commit-identity procedure, and an argument about how to infer your role from
the invocation. All of that moved or went: the rules are in `AGENTS.md`, the workflow is in
`WORKFLOW.md`, and the procedure and the identity are both in the persona above.
**Anything restated here becomes a second copy to drift.**

⚠ **The `@import` is the one line here that is not a pointer, and it has to be.** Claude Code
auto-discovers `CLAUDE.md` and does **not** auto-discover `AGENTS.md`, so an import anywhere
else loads nothing. It is the mechanism by which a session acquires its role, not a rule about
roles — remove it and every session silently has no persona, with no error and no warning.
`tests/test_persona_wiring.py` asserts it resolves.

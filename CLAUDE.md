# Project Sonora

**[AGENTS.md](AGENTS.md) is this repo's rules of record. Read it before you do anything else.**
It is not loaded for you — only this file is — so nothing else will put it in front of you.

## By default you are the developer

**If nothing in your system prompt says otherwise, that is who you are.** No flag, no
pre-prompt, and no decision on your part: the persona is imported on the next line, so it is
already in your context by the time you read this sentence. WHO the developer is — name,
email, persona path — lives in the FerroStep roster, [config.yaml](config.yaml)
(`default_agent: developer`); resolve it, never type it. ⚠ The import path below is the one
deliberate second copy of that entry's `persona` value: an `@import` cannot read YAML, so the
path is stated here and configured there. Change one, change both.

@workflow/DEVELOPER.md

⚠ **The one exception is the reviewer** — its persona resolves from the roster
(`ferrostep agent-env --agent reviewer`) and is handed to `claude -p` only by
`workflow/scripts/request_review.sh` as its `--system-prompt-file`. **If that file is
your system prompt it outranks everything here**, and the import above is not addressed to you.

MEASURED 2026-08-17: `--system-prompt-file` replaces the *default assistant prompt*; it does
**not** suppress this file, and imports come with it. So Janis is handed Ozzy's persona whether
or not that is wanted, and **the precedence rule is the only thing separating the two roles.**
It is therefore stated at three points on purpose — here, in REVIEWER.md §0, and again in the
brief the launcher appends at the call site.

⚠ **Do not try to infer your role from how you were invoked.** `-p` is not the test: the
developer runs under `-p` too, unattended, via `review_cycle.sh`. *(A signal does exist —
`CLAUDE_CODE_ENTRYPOINT` is `cli` interactively and `sdk-cli` under `-p`. Keep it as a
falsifier for a session that has become confused, never as the mechanism: it lives in the
environment rather than in your context, so using it means choosing to go and look, which is
the very step this import exists to remove.)*

⚠ **If you are committing, you are the developer**, and the author line is not automatic:

```bash
AGENT_ENV="$(ferrostep agent-env)"   # non-zero rc = the roster refused; stop, read stderr
eval "$AGENT_ENV"
git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit -m "…"
```

The repo's configured identity is the owner's, deliberately, so skipping the resolution
does not error at `git commit` — it silently commits your work under their name. ⚠ **The
assignment-then-eval split is load-bearing**: `eval "$(ferrostep agent-env)"` in one step
DISCARDS the reader's refusal, because eval's status is the emitted text's status and a
refusal emits nothing (measured 2026-08-24). `workflow/DEVELOPER.md` §1 has the check to
run after every commit.

**Keep this file short.** It exists to route and to import; the rules live in `AGENTS.md`,
the procedures in the personas, and the identities in `config.yaml`. Anything restated here
becomes a second copy to drift.
⚠ **An `@import` is not a restatement** — it is one copy, loaded from its own file.

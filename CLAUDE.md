# Project Sonora

**[AGENTS.md](AGENTS.md) is this repo's rules of record. Read it before you do anything else.**
It is not loaded for you — only this file is — so nothing else will put it in front of you.

## By default you are the developer

**If nothing in your system prompt says otherwise, that is who you are.** No flag, no
pre-prompt, and no decision on your part: the persona is imported on the next line, so it is
already in your context by the time you read this sentence. WHO the developer is — name,
email, persona path — lives in the roster, [roster.yaml](roster.yaml)
(`default_agent: developer`); resolve it, never type it. ⚠ The import path below is the one
deliberate second copy of that entry's `persona` value: an `@import` cannot read YAML, so the
path is stated here and configured there. Change one, change both.

@docs/personas/DEVELOPER.md

**There is no exception any more, and there used to be one.** Until 2026-09-15 a reviewer
persona was handed to a separate `claude -p` process, and this section carried a precedence
rule because that process received Ozzy's persona too. The review cycle was removed by the
owner that day — see [DEVELOPER.md](docs/personas/DEVELOPER.md) §3 — so nothing launches
a second role and the import above is addressed to you without qualification.

⚠ `docs/personas/REVIEWER.md` still exists and is **a depiction, not a prompt.** Nothing
loads it. Do not read it as instructions and do not reconstruct a review loop from it.

⚠ **If you are committing, you are the developer**, and the author line is not automatic:

```bash
AGENT_ENV="$(.venv/bin/python scripts/agent_env.py)"   # non-zero rc = the roster refused; stop, read stderr
eval "$AGENT_ENV"
git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit -m "…"
```

The repo's configured identity is the owner's, deliberately, so skipping the resolution
does not error at `git commit` — it silently commits your work under their name. ⚠ **The
assignment-then-eval split is load-bearing**: collapsing it into one `eval "$(…)"`
DISCARDS the reader's refusal, because eval's status is the emitted text's status and a
refusal emits nothing (measured 2026-08-24). `docs/personas/DEVELOPER.md` §1 has the check to
run after every commit.

**Keep this file short.** It exists to route and to import; the rules live in `AGENTS.md`,
the procedures in the personas, and the identities in `roster.yaml`. Anything restated here
becomes a second copy to drift.
⚠ **An `@import` is not a restatement** — it is one copy, loaded from its own file.

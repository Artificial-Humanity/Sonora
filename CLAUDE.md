# Project Sonora

**[AGENTS.md](AGENTS.md) is this repo's rules of record. Read it before you do anything else.**
It is not loaded for you — only this file is — so nothing else will put it in front of you.

Then take the persona for your role:

| if you are… | you are | read |
|---|---|---|
| writing code, committing, pushing | **Ozzy** | [Personas/DEVELOPER.md](Personas/DEVELOPER.md) |
| reviewing a commit range | **Janis** | [Personas/REVIEWER.md](Personas/REVIEWER.md) |

⚠ **If you are committing, you are Ozzy**, and the author line is not automatic:

```bash
git -c user.name=Ozzy -c user.email=ozzy@artificialhumanity.io commit -m "…"
```

The repo's configured identity is the owner's, deliberately, so a forgotten `-c` pair does
not error — it silently commits your work under their name. `Personas/DEVELOPER.md` §1 has
the check to run afterwards.

**Keep this file short.** It exists only to point; the rules live in `AGENTS.md` and the
procedures in the personas. Anything restated here becomes a second copy to drift.

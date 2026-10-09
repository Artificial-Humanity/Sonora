# Docs — canon

**Policy and canon. What is ratified, settled, and standing.** Other work conforms to what is
here; when a document elsewhere disagrees with one of these, this directory wins.

This directory follows the pattern established in **Prosodia**. The distinction from `notes/`
is not filing — it is which documents are *binding*.

**The test for admission:** does the file state a rule other work must conform to, or does it
record state, progress, research, or a plan? A runbook listing runs-to-date is a record and
belongs in `notes/`. A ratified contract is a rule and belongs here. **Being important is not
the test** — `notes/STATE.md` and `notes/training-sources.md` are load-bearing and are still
records.

⚠ **Deliberately small.** Prosodia's canon directory holds four files. A `docs/` that grows to
absorb everything authoritative-feeling is the flat `notes/` directory again with a new name.

⚠ **Nothing agent-facing lives under `docs/`.** The developer persona is `PERSONA.md` at the
repo root.

⚠ **`PERSONA.md` is not policy and this file does not describe it.** It is an agent-direction
file: UPPERCASE against the lowercase-kebab rule below, not canon, not in the table, and
excluded from this repo's doc gates (`_SCAN_EXCLUDE` in
`scripts/gates/test_doc_links.py`). **Nothing in `docs/` should be read as applying to it.**

## Rules for this directory

These apply to both `docs/` and `notes/`:

* **Each file owns its subject.** When two disagree, the one named as SSOT wins — and for the
  subjects below, that is the file in this directory.
* **Superseded narrative is deleted, not archived.** Git history is the archive.
* **Filenames are `lowercase-kebab-case.md`**, except `ARCHITECTURE.md` — the only uppercase
  file in `docs/`. `notes/` keeps `STATE.md`, and the repo root keeps `AGENTS.md`, `CLAUDE.md`,
  `WORKFLOW.md` and `PERSONA.md`.
  * ⚠ **AN UPPERCASE NAME IS A SIGNAL: MUST READ.** It is not decoration
    and it is not seniority — the owner navigates by it. A new file earns capitals only if
    reading it is genuinely required.
* **`[[double-bracket]]` names are memory slugs, not repo files.** They point at the agent's
  persistent memory and will not resolve as links. `scripts/gates/test_doc_links.py` knows
  this and skips them; it is the only place the rule is enforced rather than stated.

## The canon

| file | SSOT for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | the tier-independent architecture — the Director↔Actor contract (v2), corpus rules, gates, promotion |
| [vat-channels.md](vat-channels.md) | the three conditioning channels — §1 FiLM mechanism, §2 label recipes, §3 tension semantics |
| [direction-interface-brief.md](direction-interface-brief.md) | how Sonora receives direction — the contract-v2 decision record |
| [markup-schema-brief.md](markup-schema-brief.md) | SCM v0.1 — the ratified conveyance markup (sidecar-canonical, six tags) |
| [model-decisions.md](model-decisions.md) | model **shape** — size ladder, 150M ceiling, 24 kHz, the DiT decoder-v2 design, base-model choice |
| [tts-engine-onboarding.md](tts-engine-onboarding.md) | the engine onboarding pattern, revisit list, and the gotcha compendium |
| [audiobook-corpus-policy.md](audiobook-corpus-policy.md) | the owner's-audiobooks boundary and the private-lineage firewall |

_These are **design and policy records, not build status.** ⚠ The delivery channel that
`vat-channels.md` and `direction-interface-brief.md` describe **is in the model core** — a
five-wide one-hot block, `vat_dim` 8, `matcha/delivery.py`
(`notes/STATE.md` (private)). **The EXPORT half is what is still open.**_

## Where the rest is

`notes/` holds what is in flight, starting with `notes/STATE.md` (private) — what is true
now, the snapshot every arriving agent is told to read. It records state rather than stating
a rule, so it fails this directory's own test for admission. Alongside it:
`notes/quality-gap-plan.md` (private) (what happens next),
`notes/todo.md` (private), the campaign records, the research, the runbook
(`notes/training-operations.md` (private)), and the whole `high-ambition-*`
series. ⚠ `notes/` has no index: a hand-maintained copy of a directory listing drifts.

⚠ **The `high-ambition-N` series stays in `notes/` and must not move.** It is a **cross-repo**
series — goals 3 and 4 live in `Prosodia/notes` — and Prosodia links back to these files **by
name**. Moving them breaks links that no checker in this repo would ever see.

Inbound links are **reported, never failed**. They are Prosodia's lines to fix,
and a check no commit here can turn green is one everybody learns to ignore.

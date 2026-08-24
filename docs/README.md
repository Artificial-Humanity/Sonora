# Docs — canon

**Policy and canon. What is ratified, settled, and standing.** Other work conforms to what is
here; when a document elsewhere disagrees with one of these, this directory wins.

Split out of `notes/` on 2026-08-17 (owner) to match the pattern already established in
**Prosodia**. The distinction is not filing — it is which documents are *binding*. Before the
split, `direction-contract-v3-proposal.md` ("nothing here is ratified and nothing here is
built") and `markup-schema-brief.md` (ratified v0.1) were the same kind of file in the same
directory, and only their prose told them apart. A reviewer citing the first as authority was
making an error the structure invited.

**The test for admission:** does the file state a rule other work must conform to, or does it
record state, progress, research, or a plan? A runbook listing runs-to-date is a record and
belongs in `notes/`. A ratified contract is a rule and belongs here. **Being important is not
the test** — `notes/STATE.md` and `notes/training-sources.md` are load-bearing and are still
records.

⚠ **Deliberately small.** Prosodia's canon directory holds four files. A `docs/` that grows to
absorb everything authoritative-feeling is the flat `notes/` directory again with a new name.

## Rules for this directory

Inherited from `notes/README.md`, and they apply to both:

* **Each file owns its subject.** When two disagree, the one named as SSOT wins — and for the
  subjects below, that is the file in this directory.
* **Superseded narrative is deleted, not archived.** Git history is the archive. (The
  `notes/archive/` directory was removed 2026-08-02 — see the warning below.)
* **Filenames are `lowercase-kebab-case.md`**, except `ARCHITECTURE.md`, the one uppercase
  anchor here. `notes/` keeps the other, `STATE.md`.
* **`[[double-bracket]]` names are memory slugs, not repo files.** They point at the agent's
  persistent memory and will not resolve as links. `scripts/gates/test_doc_links.py` knows
  this and skips them; it is the only place the rule is enforced rather than stated.

## The canon

| file | SSOT for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | the tier-independent architecture — the Director↔Actor contract (v2), corpus rules, gates, promotion |
| [vat-channels.md](vat-channels.md) | the three conditioning channels — §1 FiLM mechanism, §2 label recipes, §3 tension semantics |
| [direction-interface-brief.md](direction-interface-brief.md) | how Sonora receives direction — the contract-v2 decision record (DECIDED 2026-07-30) |
| [markup-schema-brief.md](markup-schema-brief.md) | SCM v0.1 — the ratified conveyance markup (sidecar-canonical, six tags) |
| [model-decisions.md](model-decisions.md) | model **shape** — size ladder, 150M ceiling, 24 kHz, the DiT decoder-v2 design, base-model choice |
| [tts-engine-onboarding.md](tts-engine-onboarding.md) | the engine onboarding pattern, revisit list, and the gotcha compendium (ratified 2026-07-25) |
| [audiobook-corpus-policy.md](audiobook-corpus-policy.md) | the owner's-audiobooks boundary and the private-lineage firewall |

_These are **design and policy records, not build status.** ⚠ The delivery channel that
`vat-channels.md` and `direction-interface-brief.md` describe **SHIPPED in the model core on
2026-08-07** — a five-wide one-hot block, `vat_dim` 8, `matcha/delivery.py`
([STATE.md](../notes/STATE.md)). **The EXPORT half is what is still open.** This paragraph said
"not implemented" until 2026-08-22 (#283), in a file created after the channel shipped._

## Where the rest is

`notes/` holds everything in flight: [STATE.md](../notes/STATE.md) (what is true now),
[quality-gap-plan.md](../notes/quality-gap-plan.md) (what happens next),
[todo.md](../notes/todo.md), the campaign records, the research, the runbook
([training-operations.md](../notes/training-operations.md)), and the whole `high-ambition-*`
series. Start at [notes/README.md](../notes/README.md).

⚠ **The `high-ambition-N` series stays in `notes/` and must not move.** It is a **cross-repo**
series — goals 3 and 4 live in `Prosodia/notes` — and Prosodia links back to these files **by
name**. Moving them breaks links that no checker in this repo would ever see. Measured
2026-08-17: Prosodia holds 37 links into Sonora's prose, and moving the series would have
broken 21 of them.

⚠ **10 of those 37 were dead when the split was made** — `notes/archive/` was removed on
2026-08-02 and neither repo noticed for a fortnight. ⚠ **They have since been REPAIRED on
Prosodia's side, and the gate reports 0 today** (re-measured 2026-08-21: 31 relative links in,
all resolving; the archive references are now prose naming the deletion, not links). This
sentence said the gate "reports them on every run" until then — a claim that stopped being
true when someone else fixed their own files (#264).

The mechanism still stands and is the part worth keeping: inbound links are **reported, never
failed**. They are Prosodia's lines to fix, and a check no commit here can turn green is one
everybody learns to ignore.

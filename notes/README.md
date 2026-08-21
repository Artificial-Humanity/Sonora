# Notes — map

**In flight and transient.** Plans, proposals, campaign records, current state, research —
what is *being worked out*. Ratified policy and canon live in [`docs/`](../docs/README.md),
split out on 2026-08-17 to match Prosodia; **when a file here disagrees with one there,
`docs/` wins.**

The test: does the file state a rule other work must conform to (`docs/`), or does it record
state, progress, research or a plan (here)? Being load-bearing is not the test — `STATE.md`
and `training-sources.md` are both, and both are records.

One line per file, grouped by what question it answers. Rules for this directory:
each file owns its subject (when two disagree, the SSOT named here wins); superseded
narrative is **deleted, not archived** — git history is the archive (the `archive/`
directory and several closed records were removed 2026-08-02). Filenames are
`lowercase-kebab-case.md` except `STATE.md`, the uppercase anchor here — the other,
`ARCHITECTURE.md`, went to `docs/` with the split.

Two conventions that surprise readers:

- **A consolidated file keeps its sources' sections, and cross-references point at the
  *section*, not a filename.** `vat-channels.md`, `dataset-landscape.md`,
  `model-decisions.md` and `book-prose-lane.md` each absorbed 2–4 retired briefs; their
  headers name what folded in, so a citation to a vanished filename is findable.
- **`[[double-bracket]]` names are memory slugs, not repo files.** They point at the
  agent's persistent memory and are deliberate; they will not resolve as links.

## Start here — in this order

| # | file | is |
|---|---|---|
| 1 | [north-star.md](north-star.md) | **why any of this exists** — the vision, the seven-ambition arc, the size/hardware envelope and the load-bearing constraints. Owns nothing but its §1; every figure names the file that does |
| 2 | [STATE.md](STATE.md) | **what is true now**, per front |
| 3 | [quality-gap-plan.md](quality-gap-plan.md) | **what happens next**, and in what order — the SSOT for sequencing. **Its § the pathway is the whole route in one table**, Phase 0 → the five corpus rungs → decoder spike → the deferred conditioning chain and multilingual |
| 4 | [todo.md](todo.md) | open *residue* — code-review findings and the ears queue. Not the plan |
| 5 | [ARCHITECTURE.md](../docs/ARCHITECTURE.md) | **canon** — the Director↔Actor contract (v2), corpus rules, gates, promotion |

_**Where the front is today**: deliberately not stated here._ This paragraph used to carry a
third copy of a status owned by [quality-gap-plan.md](quality-gap-plan.md) **§ the pathway** and
summarised by [STATE.md](STATE.md) — rows 2 and 3 above — and it is the copy that drifted: it
reported rung 2 as built-but-unverified for ten days after `ep008` was selected, under the
repo's first *"Start here"* heading, where it changed what a reader thought the next action was
(#200). ⚠ **Do not restore it.** A status needs one owner, and the two rows already point at
both of this one's.

_Answering a specific question? **"What are we training on"** → `training-sources.md`.
**"May we use this dataset"** → `dataset-landscape.md`. **"How do I launch a run"** →
`training-operations.md`. **"Why is the model shaped like this"** → `model-decisions.md`._

_Bookkeeping: none to maintain here. The changelog and the `code-review-*.md` cycle were both
retired 2026-08-11 — a change is recorded by its commit message, and a review finding that its
cycle could not settle becomes a **PocketBase issue**, filed by the reviewer under
[`workflow/WORKFLOW.md`](../workflow/WORKFLOW.md). ⚠ This paragraph was stale in two ways until
2026-08-17: it said "GitHub issue (AGENTS.md §1)" — GitHub issues were retired and AGENTS.md no
longer mentions them — and it described a `notes/reviews/` directory that has not existed since
the review lane was rebuilt. Both are the drift the `docs/` split exists to make visible._

## Data — what trains, and under what license

| file | SSOT for |
|---|---|
| [training-sources.md](training-sources.md) | a source's **state** (on disk? trained on? blocked?) + the VAT corpus lineage. **This is the one that answers "what is actually feeding a run today"** |
| [dataset-landscape.md](dataset-landscape.md) | a source's **license** and role — the English survey, the NC two-fence ruling, and the multilingual survey, all folded in |
| [audiobook-corpus-policy.md](../docs/audiobook-corpus-policy.md) | the owner's-audiobooks boundary + private-lineage firewall |
| [book-prose-lane.md](book-prose-lane.md) | the book text→synthesis lane + the LibriVox real-audio/quote lane (operations first, rationale second) |
| [delivery-mix-campaign.md](delivery-mix-campaign.md) | the 50/30/8/6/6 delivery rebalance — **complete 2026-08-04**; kept for the findings and the traps it exposed |

_When these two disagree: `training-sources.md` wins on a source's **state**,
`dataset-landscape.md` wins on its **licence**._

## Labels & direction

| file | SSOT for |
|---|---|
| [vat-channels.md](../docs/vat-channels.md) | the three conditioning channels — §1 FiLM mechanism, §2 label recipes, §3 tension semantics |
| [direction-interface-brief.md](../docs/direction-interface-brief.md) | how Sonora receives direction (contract-v2 decision record; the reverse-conveyance design) |
| [markup-schema-brief.md](../docs/markup-schema-brief.md) | SCM v0.1 — the ratified conveyance markup (sidecar-canonical, six tags) |
| [casting-attribute-norms-brief.md](casting-attribute-norms-brief.md) | measured casting norms, cast sheet, identity-vs-portrayal |

_All four are **design records**, not build status. The delivery channel they describe is
not implemented in the model core — [todo.md §1](todo.md)._

_⚠ The first three are **canon** and live in [`docs/`](../docs/README.md); they are listed here
because this is where you look for them by topic.
`casting-attribute-norms-brief.md` stays in `notes/` — it is explicitly "design capture, **not
yet scheduled**", so it fails the ratified test its three siblings pass._

## Teacher synthesis

| file | SSOT for |
|---|---|
| [synthesis-pipeline.md](synthesis-pipeline.md) | the render→QC→audition pipeline mechanics + rating vocabulary v4 |
| [teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md) | engine license verdicts + measured standing |
| [tts-engine-onboarding.md](../docs/tts-engine-onboarding.md) | the onboarding pattern, revisit list, and the gotcha compendium |
| [teacher-training-data.md](teacher-training-data.md) | **what the teachers trained on** — and the multilingual plan. Reference, not a licence ruling: neither Qwen nor Chatterbox discloses a corpus, so what is copyable is method (staged quality curriculum, mixture balance by mode of address), not sources. Holds the scale gap that scopes every data decision |

## Model & training

| file | SSOT for |
|---|---|
| [quality-gap-plan.md](quality-gap-plan.md) | **sequencing** — the ordered plan to close the synthetic-vs-real gap (measurement repair → data → DiT spike), the gates between phases, and the 2026-08-06 diagnostics that scoped it |
| [model-decisions.md](../docs/model-decisions.md) | model **shape**: size ladder, 150M ceiling, 24 kHz, the DiT decoder-v2 design, base-model choice |
| [matcha-siblings-study.md](matcha-siblings-study.md) | the standing comparison bench (StableTTS, RapFlow, CosyVoice, Baichuan…) — check it before designing any component blind |
| [training-operations.md](training-operations.md) | the **runbook** — runs to date, launch/stop/resume, checkpoint settings, watchers, gates, footguns, registry conventions |
| [local-vs-runpod-decision.md](local-vs-runpod-decision.md) | the measure-before-renting instrumentation |

_The DiT spike is described in **three** places by design: `model-decisions.md` says what
it is, `matcha-siblings-study.md` says what public code already solved, and
`quality-gap-plan.md` says when it runs._

## Vision / roadmap

Parked unless marked live. The `high-ambition-N` numbering is a **cross-repo series** —
goals 3 and 4 live in `Prosodia/notes`, and Prosodia links back to these files by name,
so do not renumber or rename them.

| file | is |
|---|---|
| [north-star.md](north-star.md) | the **arc** the seven goals decompose from, and the envelope they must fit inside. Orientation — the index below is the SSOT for any goal's standing |
| [high-ambition-index.md](high-ambition-index.md) | the seven-goal map, and which repo each lives in |
| [high-ambition-1-matcha-actor.md](high-ambition-1-matcha-actor.md) | goal 1 — the directable production actor (**live**; this is the project) |
| [high-ambition-2-dramatic-reader.md](high-ambition-2-dramatic-reader.md) | goal 2 — full-cast audiobooks (design, needs 1 first) |
| [high-ambition-6-audience-conveyance-stt.md](high-ambition-6-audience-conveyance-stt.md) | goal 6 — conveyance-aware STT (vision, parked) |
| [high-ambition-7-singing.md](high-ambition-7-singing.md) | goal 7 — singing (**down the road**, gated behind the quality ladder). ⚠ *unrequested* singing is a defect and stays one |

_(Goal 5, the StyleTTS2-Lite re-platform, was **retired 2026-07-29** and its note deleted
2026-08-02. The decision record is [model-decisions.md § Why not Kokoro, StyleTTS2 or a GAN stack](../docs/model-decisions.md).)_

# Quality mechanisms plan

Commissioned by the owner 2026-08-17, after the Pydantic boundary work, in answer to: *what
else can we do to ensure safety and quality going forward?*

**The ranking below is derived from this repo's measured defect history, not from general
practice.** The tracker held 79 issues before the 2026-08-17 wipe; roughly 40% of them were a
document disagreeing with the code, which is the largest single class by a wide margin. Nearly
every defect found during the Pydantic branch itself was a **guard that had stopped guarding** —
not code that was wrong. Those two facts decide everything that follows.

## What this plan deliberately excludes

* **Static type checking.** Recorded judgement (issue #70): neither lint nor typing would have
  caught that week's defects. Nothing since has changed it. The defects are about *values that
  arrive at runtime from files and from other processes*, and about *checks that no longer
  bind* — neither is visible to a type checker.
* **Pydantic beyond genuine cross-process contracts.** Measured on the `pydantic-boundaries`
  branch: only 8 load sites sit inside substituting exception handlers, and most of those are
  legitimate network handling. Breadth is not justified. The criterion that *is* justified is
  cross-process coupling — `ratings.csv` is read by 14 modules, `qc_measures.jsonl` by 10 — and
  that is where the work already went.

---

## M0 — Split `notes/` into `notes/` and `docs/`

Owner, 2026-08-17: the established pattern in **Prosodia** is two directories, and this repo
should match it.

| | |
|---|---|
| **`docs/`** | **policy and canon.** Ratified, settled, standing. Other work must conform to it. |
| **`notes/`** | **in-flight and transient.** Plans, proposals, campaigns, current state, research. |

### Why this belongs in a quality plan rather than beside it

It is not filing. The dominant defect class here is a document disagreeing with the code, and
a large part of what makes that hard to spot is that **a reader cannot tell which documents are
binding.** `direction-contract-v3-proposal.md` opens "Nothing here is ratified and nothing here
is built"; `markup-schema-brief.md` is ratified v0.1. Both are `notes/*.md`, both are scanned by
the same gate, and only the prose inside distinguishes them. A reviewer who cites the first as
authority is making an error the directory structure invited.

It also sharpens M1: a claim in `docs/` is a claim the repo stands behind, and that is a
stronger reason to check it than "someone wrote a number down".

### The classification

⚠ **A first draft of this section moved sixteen files, and reading the sibling repo cut it to
seven.** Prosodia's `docs/` holds **four** files — `ARCHITECTURE.md`, `CONTRIBUTING.md`,
`ROADMAP.md` and a defensive-publication note — and everything else, *including
`high-ambition-3` and `high-ambition-4`*, is in `notes/`. The pattern the owner pointed at is a
small, tight canon directory, not a general reshelving. The draft was reasoning from the words
"policy and canon" instead of from the repo that already does this.

The discriminator that survives: **does the file state a rule other work must conform to, or
does it record state, progress, research or a plan?** A runbook listing runs-to-date is a
record. A ratified contract is a rule.

**`docs/` — the governing documents (7)**

| file | why |
|---|---|
| `ARCHITECTURE.md` | says "_Canon._" in its own subtitle |
| `direction-interface-brief.md` | "DECIDED 2026-07-30" — the Director↔Actor contract |
| `markup-schema-brief.md` | "**RATIFIED v0.1**" |
| `model-decisions.md` | "the settled answers … and only what they still direct" |
| `tts-engine-onboarding.md` | "ratified pattern, owner 2026-07-25. Applies to every speech engine" |
| `vat-channels.md` | the conditioning contract the model and the corpus both obey |
| `audiobook-corpus-policy.md` | the owner's-audiobooks boundary and private-lineage firewall |

Everything else stays in `notes/` — twenty files, including `STATE.md`, `todo.md`, both plans,
the proposal, the campaign records, the research, the runbook, and the whole `high-ambition-*`
series.

Deliberated and left in `notes/`, so the reasoning is on the record rather than rediscovered:
`training-sources.md` and `training-operations.md` (state and runbook — records, not rules),
`teacher-tts-audition-shortlist.md` (a running record of verdicts; the licence *rule* is in
`dataset-landscape.md`), and `casting-attribute-norms-brief.md`, which is explicitly "design
capture, **not yet scheduled**" and so fails the ratified test its sibling briefs pass.

### ⚠ Why the high-ambition series must not move

`notes/README.md` states the constraint and it is correct: *"the `high-ambition-N` numbering is
a **cross-repo series** — goals 3 and 4 live in `Prosodia/notes`, and Prosodia links back to
these files by name, so do not renumber or rename them."*

Measured: **Prosodia holds 37 links into `Sonora/…/notes/`, and they resolve.** The rejected
16-file draft would have broken **21** of them, none visible to any checker in this repo. The
7-file split breaks **3** — `vat-channels.md` (2) and `model-decisions.md` (1) — which are
repaired on Prosodia's side as part of this work rather than left to rot.

**And 10 of the 37 dangled when this was written** (since repaired — 0 as of 2026-08-21,
#264): `notes/archive/high-ambition-5-styletts2-lite.md` (5),
`notes/archive/exploit-before-train-measurement.md` (3), `notes/actor-model-and-training.md`
(2). The `archive/` directory was removed 2026-08-02 and neither repo noticed. That is the
strongest single argument for the checker: the cross-repo surface has been rotting silently for
two weeks, and the split is merely the event that made anyone look.

### ⚠ The breakage surface, measured

| | |
|---|---|
| absolute `notes/<file>.md` references across the tree | **190** |
| relative intra-`notes/` links (`[STATE.md](STATE.md)`) | **224** |

The second number is the dangerous one. Those links are bare filenames that resolve **because
every file sits in one directory**, and the split is exactly what stops that being true. They
do not error — they render, and go nowhere. This is the silent-degrade shape the whole plan is
about, so the move must not be done by hand.

### The work

1. Move the seven files, rewriting both link forms in the same commit.
2. **A link checker, and it is the durable part** — the split is the occasion, not the product.
   Every relative link in `docs/` and `notes/` must resolve to a file on disk.
   * ⚠ **Cross-repo aware.** Checking only Sonora's outbound links leaves the larger and
     already-rotting surface untouched — the 10 dead links found there **when this was
     written** proved that surface is the one that actually fails. (They were repaired; the
     count is 0 today. The argument rests on the rot having happened, not on it persisting.) Sibling checkouts are already configured for the reviewer in
     `workflow/config.env` (`SIBLING_REPO_CANDIDATES`); the checker uses the same idea and
     **reports rather than fails when a sibling is absent**, because a laptop without Prosodia
     checked out has not found a defect.
   * ⚠ **Skips `[[double-bracket]]` names.** Those are agent memory slugs, deliberately not
     repo files. `notes/README.md` is the only place that rule is written down, which makes it
     part of this checker's specification and not a footnote.
3. `docs()` in the doc-claims gate scans both directories. It hardcodes `notes/` today, so
   after the split a canon file drops out of the gate's view — the `.gitignore` failure of
   2026-08-17 with a different mechanism.
4. `docs/README.md` as the canon index; `notes/README.md` keeps the in-flight map.

### ⚠ The rules in `notes/README.md` are the split's real content

That file is not only an index. It carries the directory's standing rules, each of which the
split forces a decision about:

* **"when two disagree, the SSOT named here wins"** — an arbitration rule whose *here* is
  `notes/README.md`. Several SSOTs it arbitrates move to `docs/`, which would leave the
  in-flight index as tie-breaker over canon. The arbitration moves with the canon.
* **"except the two uppercase anchors"** — `STATE.md` and `ARCHITECTURE.md`, which the split
  puts in different directories. The sentence then describes neither directory correctly, and
  it is prose, so no gate catches it.
* **"`[[double-bracket]]` names are memory slugs"** — the checker's specification, above.
* **the high-ambition cross-repo constraint** — the reason the draft shrank.

Two paragraphs there are already stale and get fixed in the same commit: it describes a
`reviews/` directory that no longer exists, and says a finding "becomes a GitHub issue (AGENTS.md
§1)" when the PocketBase tracker replaced that and `AGENTS.md` no longer mentions GitHub issues
at all.

### Acceptance

Moving a file without fixing its inbound links must turn the checker red — proven by doing it
and watching it fail, the same way M1 was.

⚠ **THIS CRITERION USED TO SAY "the 10 pre-existing cross-repo dead links must be reported on
the first run; if they are not, the checker is not looking where the rot is." DO NOT RUN IT —
it now returns 0 and hands you the wrong diagnosis** (#264). Prosodia repaired those links, so
0 is the correct answer and the conclusion attached to it is false. ⚠ There is even a
plausible-looking culprit in reach for anyone who believes it: `INBOUND` requires a path
segment between `/Sonora/` and `notes/`, so a sibling writing `../../Sonora/notes/x.md` would
not match — nothing uses that shape today, so "fixing" it would be work done on a false alarm.

**An acceptance criterion written against a number someone else can repair expires without
notice.** Assert the MECHANISM instead: move a file, confirm red; restore, confirm green.

---

## M1 — Doc claims about the repo's own constants

### The measurement that reshaped this item

The original proposal was "widen `test_doc_claims.py`'s `docs()` to cover `AGENTS.md`,
`CLAUDE.md` and `workflow/*.md`". **Measured 2026-08-17, that change on its own enrols nothing.**
Running the existing registry over those five files:

| | |
|---|---|
| candidate files | 5 |
| numeric tokens in them | 472 |
| statements any existing FACT recognises | **0** |

The reason is structural, not incidental. Every fact in the registry today reads its truth from
a **corpus artifact** — `derivation_report.json`, a filelist's row count, a checkpoint on disk —
and the numbers in `AGENTS.md` and `workflow/` are not corpus numbers at all. They are
**protocol constants**: how many passes a review gets, how long a comment may be, what the state
vocabulary is. A registry keyed on corpus artifacts cannot see them however wide its file list
grows.

So M1 is two pieces, and the second is the real work.

### M1a — the file set

Add the five files to `docs()`. Six lines. It enrols zero statements **today**, and that is the
correct reason to do it rather than a reason to skip it: the moment someone writes a corpus
number into `AGENTS.md` in a recognised phrasing, it is checked without anybody remembering to
come back. This is exactly the argument that added `configs/data/*.yaml` on 2026-08-11, which
moved the counts from 32 files / 36 statements to 42 / 37 — one statement, and the point was
never the one statement.

⚠ **`workflow/` is portable by design.** Its numbers get *copied into other repos*, where they
drift independently and where nothing at all is checking them. That raises the value of M1b and
lowers the value of treating `workflow/` as ordinary prose.

### M1b — a second registry: protocol facts

Same design, different truth source. A fact reads a **constant from the file that owns it** and
checks the documents that restate it.

The forks-in-waiting, measured 2026-08-17. All of them **agree today** — this is prophylactic
work, and saying so plainly is part of the plan:

| constant | owner | independent restatements |
|---|---|---|
| review pass cap (`3`) | `workflow/config.env` `MAX_PASSES` | `WORKFLOW.md` ×2; `DEVELOPER.md` ×3; `issue.py` docstring; `config.env`'s **own comment** — **7 restatements in 4 files** |
| comment cap (`1500`) | `workflow/config.env` `COMMENT_MAX` | `REVIEWER.md`; separately enforced by the PocketBase `issue_comments.body.max` |
| issue body cap (`200,000`) | the PocketBase `issues.body.max` | `issue.py` error message; `config.env` comment; `DEVELOPER.md`; `REVIEWER.md` |
| state vocabulary | the PocketBase `issues.state` select | `WORKFLOW.md`; `DEVELOPER.md`; `REVIEWER.md`; `issue.py` transition table |

The pass-cap row was counted by hand as six and **measured as seven** once the check existed —
`config.env`'s own comment restates its own constant one line above the assignment, which is
the tightest fork in the table and the one hand-counting missed.

Two of these were checked by hand while writing this plan and both were **true**, including one
I expected to be false: `REVIEWER.md`'s "hard maximum 1500 characters, **enforced by the
schema**" is accurate — `issue_comments.body.max` really is 1500, so the cap survives a writer
that bypasses `issue.py`. Checking settled it; reading did not.

⚠ **`agent_passes` has no schema maximum, and that is correct.** The field is the owner's
dial — `0` means deliberately re-armed — so a schema ceiling would fight the owner.

⚠⚠ **SUPERSEDED IN PART (phase 2 of the FerroStep cutover, 2026-08-24).** The pass cap no
longer lives in `issue.py` or `config.env`: it is `agent_passes.max` in
`workflow/sonora-lane.json`, ENFORCED by the engine, and the state vocabulary's transition
table left `issue.py` for the same file. Two of the table's four forks are therefore
closed by consolidation rather than by the gate this plan proposed; the table stands as
the 2026-08-17 measurement that motivated exactly that move.

### Design constraints inherited from the existing gate

These are not style preferences. Each one is a hole that was found the hard way in
`test_doc_claims.py` and is documented at length there:

* **Never scope a fact on the value it checks.** A line held in scope only by the number under
  test drops out of scope the instant that number is corrupted — the one moment the check had a
  job. Found by testing the *red* direction, which is the only way this class is ever found.
* **Anchor the capture on the number actually being claimed**, and guard against a match
  starting inside a longer number.
* **Exemptions are explicit substrings, printed every run**, so an exemption cannot quietly
  become a hiding place.
* **A fact whose source is unreadable is a failure, not a pass.** A fact whose source is *not on
  this machine* is a printed skip.
* **Patterns stay narrow.** A loose pattern produces false failures, and a noisy gate is a gate
  someone turns off. The first run of the corpus registry produced 20 false failures from one
  loose pattern.

### What M1 will not cover

Only the phrasings the patterns match, and only constants that have a single owning file. A
protocol claim written in prose no pattern recognises stays invisible — a silent miss, not an
error. **Green means "nothing recognised disagrees", never "the docs are correct."**

### Acceptance

* `docs()` includes the five files; a test asserts each is present by full path, so a rename
  cannot silently drop one.
* At least the four facts above, each with a red-direction test: corrupt the constant, the gate
  must fail; corrupt a *document*, the gate must fail.
* The gate's own summary line reports the protocol facts separately from the corpus facts.

---

## M2 — Guard liveness

### The evidence

**Five guards were found silently disarmed in a single day (2026-08-17)**, every one of them
green and enforcing nothing:

1. `WORKER_DENY` named script paths that had moved into `workflow/scripts/`.
2. The test guarding that list used a substring match, so it stayed green after the move.
3. `.gitignore`'s unanchored `lib/` swallowed every new file in `scripts/lib/`.
4. `review_cycle.sh`'s stall guard compared two populations that were not comparable.
5. A convergence test matched an unrelated line and would have passed on a broken subject.

This is the dominant *runtime* failure mode, as doc drift is the dominant *content* one.

### The work

* **Any config naming a path is checked against the filesystem.** One instance of this was
  written on 2026-08-17 (`test_every_denied_script_actually_exists`); generalise it to every
  path-bearing constant in `workflow/` and `scripts/`.
* **Any guard test must be able to fail.** Generalise
  `test_the_guard_actually_asserts_something` — a test whose assertion is a substring search
  over a whole file is presumed disarmed until it demonstrates a red direction.
* **Anchored-ignore check.** Assert no `.gitignore` line that names a common source directory
  is unanchored. `lib/` cost this repo real files; `bin/`, `build/`, `docs/` are the same shape.

### Acceptance

Each check ships with a demonstration of its red direction in the same commit. A liveness check
that has never been seen to fail is the thing it was written to prevent.

---

## M3 — Declared-dependency check

### The evidence

On 2026-08-17 `scripts/lib/schemas.py` imported `pydantic` at module scope and `book_ingest`
imports that module, putting it on the teacher-synthesis path every container takes. `pydantic`
was reachable **only** through `fastapi`, in the `vocalizer` optional extra. A dev box had it by
accident; a training or eval image would not have had it at all, and the failure would have been
an `ImportError` inside a container — nowhere a test could see it. Fixed by declaring it core,
but nothing prevents the next one.

Precedent: `pyproject.toml` already records that `requirements.txt` had drifted **both ways** —
seven declared packages nothing imports, and three imported at module scope that were resolving
by transitive luck through `librosa`.

### The work

Walk every module-scope import on the core runtime path; resolve each to a distribution; assert
it appears in `[project.dependencies]` and not only in an optional extra or dependency group.
Lazy imports inside functions are out of scope by construction — that is the mechanism
`cleaners.py` uses on purpose to keep `phonemizer` (GPL-3.0) off the runtime path, and the check
must not punish it.

### Acceptance

Removing `pydantic>=2` from `[project.dependencies]` must turn the check red.

---

## M4 — Promote the AST test helper

### The evidence

Four false-greens on 2026-08-17 came from substring searches over whole files — **two of them
within an hour of each other, and one of those was inside a test that warns about this exact
mistake.** The specific traps, both measured:

* Docstrings quoting the old code matched a search for the old code.
* f-strings tokenize as `FSTRING_START`/`MIDDLE`/`END` on Python 3.12+, **not** as `STRING`, so
  a comment-and-string stripper written before 3.12 silently stops stripping them.

`code_only()` in `tests/test_schemas.py` handles both. It should not live in one test file.

### The work

Move it to a shared test helper alongside the AST predicates written with it (*is this call
inside a `try`?*, *does this module wrap a validated loader?*). Make the structural assertion the
short path so the substring search stops being the convenient one.

### Acceptance

`tests/test_schemas.py` imports the helper rather than defining it, and at least one other
existing source-string test is converted, proving the helper is general and not a rename.

---

## Sequencing

**M1 is done** (`b0dafdd`). It attacked the largest measured class with machinery that already
existed and was already trusted.

**M0 next, and it has to precede M2–M4** rather than sit beside them: it moves the files the
doc-claims gate reads, so running it after M2 would mean re-deriving M2's path checks against a
tree that had just changed underneath them. It is also the only item here with a large
mechanical surface (414 references), which is a second reason not to have it in flight
alongside anything else.

M2 after that: the largest *runtime* class, patterns proven but scattered. M3 and M4 are small,
independent, and can land in either order.

Each is its own branch and its own review under `workflow/WORKFLOW.md`.

## What M1 changed about the rest of this plan

Two things, both worth carrying forward:

* **The measurement went first and reordered the work.** The proposal was a file-set widening;
  the measurement said that buys zero and the registry was the real item. Do this for M0 and M2
  as well — M0's 224 relative links were found the same way and are the reason it needs a
  checker rather than a careful afternoon.
* **A mutation battery found a test of mine that passed for the wrong reason** — a scope that
  matched on letter case, so the assertion under it never ran. That is the third false-green of
  exactly this shape in two days. **Every mechanism in this plan ships with its mutations run
  one at a time**, which is the only step that has reliably found this class.

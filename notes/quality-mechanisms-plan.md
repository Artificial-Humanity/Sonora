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

⚠ **`agent_passes` has no schema maximum, and that is correct.** The cap lives only in
`issue.py`. The field is the owner's dial — `0` means deliberately re-armed — so a schema
ceiling would fight the owner. The fact to encode is the *documented* cap against
`MAX_PASSES`, never a claim that the tracker enforces it.

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

M1 first: it attacks the largest measured class with machinery that already exists and is
already trusted. M2 second: it is the largest *runtime* class and the patterns are proven but
scattered. M3 and M4 are small, independent, and can land in either order.

Each is its own branch and its own review under `workflow/WORKFLOW.md`. None of them is
blocked by another.

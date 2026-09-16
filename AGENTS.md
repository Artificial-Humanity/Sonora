# AGENTS — Project Sonora (Training Repo)

This is the entry point for any agent or developer working on Project Sonora's training
codebase. This is an independent GitHub repo (the PyTorch training pipeline that produces the
actor model artifacts published to the `Sonora/huggingface` sibling checkout). Internal engineering notes —
architecture, current state, open decisions — live in `notes/` (**private**, 2026-09-08: the
umbrella `Notes` repo, reachable here as a gitignored symlink). ⚠ **Named, not linked** — a public
clone has no `notes/`, so a link would render as a path a reader cannot follow. `docs/` is the
prose this repo carries. ⚠ **That directory also had a hand-maintained index, `notes/README.md`,
retired the same day**: it was a second copy of
the directory listing, it had drifted (three files were missing from it), and every convention it
carried is stated in `docs/README.md`, in § File Naming Conventions below, or in
`scripts/gates/test_doc_links.py` itself. Before starting work, read
`notes/STATE.md` (private) for the current state of the project and
`notes/todo.md` (private) for the open work.

⚠ **THE SYMLINK DOES NOT SURVIVE A CHECKOUT ACROSS THE MIGRATION, AND NOTHING TELLS YOU.**
`notes/` was tracked until 2026-09-08 and is gitignored after it, so any `checkout`, `merge`
or `bisect` that crosses that boundary rewrites the path. Measured in a throwaway repo
2026-09-09, both directions:

* Going **back** (to a commit where `notes/` was tracked), git **replaces the symlink with a
  real directory** and restores the tracked files.
* Coming **forward** again, it deletes those files and the directory with them — and **does
  not restore the symlink**. `notes/` is then simply absent.
* ⚠ `git status` is **clean** at every step, because `/notes` is ignored. There is no warning
  and no diff. It happened for real on the merge that landed the migration.

✅ **Git never writes THROUGH the symlink** — verified in the same test: the private repo's
files were untouched in both directions. So this costs you the link, never the notes.
**Recreate it with `ln -s ../../Notes/Sonora notes` and carry on**; if a suite suddenly
reports the private-notes skips on a machine that has the Notes repo, this is why.

---

## Core Stack Matrix

* **Language Ecosystem:** Python-based ML training pipeline.
* **Second lane — teacher synthesis (`scripts/stages/`):** five audited TTS engines
  (`chatterbox`, `qwen`, `zonos`, `orpheus`, `moss_vg`; `vibevoice` + `dia` are set aside,
  reversible — `ref_select.SET_ASIDE`) driven by a two-pass Gemma director — per-line
  V/A/T + a register copied from the 47-label controlled lexicon
  (`scripts/assets/register_lexicon.json`), then per-engine casting/delivery from
  `scripts/assets/director_skills/<engine>.md`. Engine allocation is the three-layer
  `ref_select.ENGINE_MIX_BY_LANE` (capability veto · measured per-lane weights ·
  diversity floor). **`build_direction()` in `book_ingest.py` is the single source of
  truth for what each engine actually receives — never bypass it; unknown engines are fatal.**
  The 2026-07-25 relay audit found rich direction had been silently dropped by every engine.
  **Standing rule:** no TTS model enters the portfolio without a studied interface and a
  Gemma skill-file adapter — see [docs/tts-engine-onboarding.md](docs/tts-engine-onboarding.md),
  which also carries the known-gotchas reference.
  Minimum clip length is 4 s of estimated speech (`MIN_CLIP_SECONDS`), gating input text and
  not render duration.
  ⚠ **That line is NOT gate-enforced**: `docs()` in `scripts/gates/test_doc_claims.py` scans
  `notes/`, `docs/`, `README.md` and `configs/data/` — not this file, and widening it is a
  scope decision nobody has taken. ⚠⚠ **A WRAPPED SENTENCE IS NOT ENROLLED EITHER**, whatever
  else is true: `scope` and `patterns` are matched **per line**, so the number and the phrase
  that selects the fact must share one. Until 2026-08-21 they did not — the number sat on one
  line and `MIN_CLIP_SECONDS` on the next, so each half matched and the pair never did, and
  the paragraph then claimed widening `docs()` would enrol it. It would not have. Keeping them
  on one line above is what makes that claim true.
* **Core Framework:** PyTorch & Matcha-TTS (conditional flow-matching).
* **Environment:** AMD ROCm PyTorch Docker container (optimized for Ryzen AI Max+).

---

## Integration Dependencies

* **Deployed code is a copy; the repo is the source** — see §6 and `notes/data-mirrors.md` (private).
* This repo is a standalone PyTorch training repository for building, fine-tuning, and
  exporting voice actor models (Matcha-TTS). It is consumed by Project Prosodia
  (`ProsodiaActor`) via exported artifacts promoted into the sibling **`Sonora/huggingface`**
  checkout — a direct clone of `artificial-humanity/Sonora` alongside this repo's
  `Sonora/github`, under the flat workspace layout adopted 2026-07-22 (the umbrella workspace
  is gone). Lineage and `current` flags live in its `registry.json`.

---

## File Naming Conventions

Names must be predictable so links resolve on case-sensitive systems (Linux/CI) as well as
case-insensitive macOS/Windows.

* **Canonical root marker files → `UPPERCASE`** (`SCREAMING_SNAKE_CASE` if multi-word): `README.md`, `LICENSE`, `CONTRIBUTING.md`, `ROADMAP.md`, `AGENTS.md`. Keep this set small and curated.
* **Top-level anchor docs → `UPPERCASE`, single word preferred:** `ARCHITECTURE.md`,
  `STATE.md`, `WORKFLOW.md`.
  * ⚠ **THE CAPITALS ARE A SIGNAL, NOT A STYLE** (owner, 2026-09-16): an UPPERCASE name here
    means **must read**. The owner uses it that way for their own navigation as much as for an
    agent's, so a file that is merely important-looking does not earn it — and a file that IS
    required reading should not be lowercase just because it is new.
* **All other docs & notes → `lowercase-kebab-case.md`:** e.g. `open-decisions.md`, `code-review-findings.md`. This is the rule for everything in `notes/`.
* **Source code → the language's own convention:** Rust `snake_case.rs`, Swift `PascalCase.swift`, Kotlin `PascalCase.kt`.
* **Never** let case be the only difference between two paths, and always reference files with their exact case.

⚠ **A CONSOLIDATED FILE KEEPS ITS SOURCES' SECTIONS, SO CITE THE SECTION AND NOT THE
FILENAME.** `vat-channels.md`, `dataset-landscape.md`, `model-decisions.md` and
`book-prose-lane.md` each absorbed two to four retired briefs, and their headers name what
folded in. A citation to a vanished filename is then still findable; one to a section
survives the consolidation that deleted the file. Carried here 2026-09-08 from
`notes/README.md`, which was retired — it was the only convention in that file not already
stated in `docs/README.md`, in this section, or in `scripts/gates/test_doc_links.py` itself.

---

## System Operational Mandates

### 1. Role loading and the git environment — facts, not workflow

⚠ **[WORKFLOW.md](WORKFLOW.md) IS HOW WORK GETS DONE HERE. READ IT AND FOLLOW IT.** 
This file holds the repo's FACTS — the stack, the environment, the mandates, the measured traps. `WORKFLOW.md` holds the PROCESS.

⚠ **What follows in this section is FACTS, not procedure** — how a session acquires its role,
and what this repo's git configuration actually does when you run a command. They are here
because they are measurements about this repo, and they hold whatever workflow the owner
writes. If you find a rule about *how to work* below, it escaped the sweep and belongs in
`WORKFLOW.md`.

⚠ **REMOVED FROM THIS SECTION, so nobody hunts for it:** the branch-and-land procedure, the
land-when-green convention, merge-never-rebase, the two review-range bullets, and the "a rule
here is not a mechanism" warning. They are in git history at `1215cb7` and earlier. **Do not
reconstruct them from memory** — a rule recalled without its measurement is one nobody can
date or defend.

**If you are the developer session for this repo, read
[PERSONA.md](PERSONA.md) now and work as Sonya.** It is your standing
brief and carries the steps below in the detail your role actually needs. This file keeps the
*contract between* the roles — the loop, the cap, the abort, and the repo facts both sides
depend on. It does not duplicate either role's procedure; that duplication is what drifted
three times in one pull request.

⚠ **NOTHING LOADS THIS FILE OR EITHER PERSONA FOR YOU.** Measured 2026-08-14, not assumed:
**Claude Code auto-discovers `CLAUDE.md` and does *not* auto-discover `AGENTS.md`.** A session
started with a bare `claude` here begins with neither this file nor a persona in context.
That is why [CLAUDE.md](CLAUDE.md) exists and is deliberately a few lines long: it is the one
file that *is* loaded, and all it does is send you here and to your role's persona.

* ⚠ **A SYMLINK DOES NOT WORK.** `CLAUDE.md -> AGENTS.md` is **not** followed — measured, with
  a canary. So the pointer has to be a real file, and keeping it to pointers is the only thing
  stopping it becoming a second copy of these rules that drifts from them.
* ⚠ **THE DEVELOPER PERSONA NEEDS NO FLAG — `CLAUDE.md` `@import`s IT** (owner, 2026-08-17:
  *"I'm still not very keen on having to start claude code with a pre-prompt"*). A bare
  `claude` in this repo comes up as Sonya, because the import is inlined into the auto-loaded
  file rather than linked from it. Measured the same day: `@`-imports resolve, including from
  a subdirectory, and **a plain `claude -p` receives the imported content with no action of
  its own.** The old route — `--append-system-prompt-file PERSONA.md` — still
  works and still survives `/clear`, but it is now redundant, and a persona that depends on
  someone remembering a flag is the failure this repo keeps re-learning.
⚠ **The abort below is the one part of the old loop that survives**, because it was never
about the loop — it is the owner's rule about what must not land.
⚠ **THERE IS AN EXIT THAT IS NOT A PUSH, and it is a human handoff.** If a finding is
that the change **should not land at all** — it corrupts data, it ships a known-broken
training path, it cannot be safely reverted — your own judgement does not settle it. **Do not push; take it
to the owner.** Filing an issue and pushing anyway is right for a defect that can live on
`main` and be fixed later; it is wrong for one that cannot. Deliberately a judgement call and
not a severity threshold: no automatic rule has ever separated legitimate repair from churn.

⚠ **THIS IS INTERIM.** The owner is settling a more complete workflow architecture; this is
the simple version that holds until then. Do not build tooling on its shape.

#### The git environment — measured, and none of it is optional

⚠ These are facts about what commands DO in this checkout. They constrain any workflow rather
than being one.

* ⚠ **NOTHING STANDS BETWEEN A SESSION AND `main`.** Measured 2026-08-13, not inferred:
  `gh api repos/:owner/:repo/branches/main/protection` returns **404 Branch not protected**.
  There is no branch protection, **force-push to `main` is unblocked**, there is no pre-push
  hook, and CI runs *after* a push rather than gating one. The abort above is the only thing
  in front of `main`, which is why it is a rule and not a preference.
* **One session commits to this repo** (owner, 2026-08-13), so `main` does not move under you
  and divergence is not an ordinary event.
* **`pull.rebase=false` is set** — `--local`, re-set 2026-09-11. ⚠ Local config does **not**
  travel with a clone, so a fresh checkout starts without it. Set it yourself.
* **`push.default` is deliberately LEFT UNSET** (owner, 2026-09-11, declining to re-set it), so
  git's `simple` applies and **refuses** to push a branch whose name differs from its upstream.
  ⚠⚠ **That refusal is currently the ONLY thing between a scratch branch and `main`** — the
  severity gate that used to sit behind it was removed on 2026-09-15. Setting
  `push.default=upstream` would spend the guard for a convenience, on a repo with no branch
  protection.
* ⚠⚠ **A CONFIG CLAIM IN PROSE IS NOT A CONFIG.** This file asserted that both settings above
  were configured when **neither was set at any level** — measured 2026-09-11 with a positive
  control proving the reader worked. Nothing compares this file to `git config`, so the
  sentences were read as descriptions of a configured repo for as long as they stood. Check the
  setting before relying on it, exactly as §5b says of any documented number.
* ⚠⚠ **NEVER `git push origin <branch>:main`.** It names both ends and so bypasses the refusal
  above. Reproduced here: an unreviewed commit on a scratch branch went
  `093cf40..a23f28a  sonora/scratch -> main` in one step. To put a branch on `origin` under its
  own name, `git push origin <branch>:<branch>` (or `-u` once).
* **Cut scratch branches from a LOCAL ref, not from `origin/main`** — measured to fail safely
  with `no upstream branch`, which is the refusal you want.
* **`origin/main..HEAD` is the range.** ⚠ **`@{push}..HEAD` does NOT resolve** under `simple`:
  `fatal: cannot resolve 'simple' push to a single destination` (measured 2026-09-11).
  `origin/main..HEAD` is correct under any config.
* ⚠ **`git config --local` WRITES THE SHARED CONFIG, WHICH EVERY WORKTREE READS.** This repo has
  **one worktree today** — `/data/repos/Sonora` stopped being a git checkout on 2026-08-29
  (`AI-Lab-AMD/scripts/deploy.sh`, and the **workspace** `AGENTS.md` §4 — not this file's) —
  so the hazard is latent rather than live, and it returns the moment a second worktree exists.
  "One committer, therefore harmless" is not the test: `commit.template` was set `--local`
  pointing at a `.gitmessage` absent from the other worktree, which made an interactive
  `git commit` **fatal** there — it refused and created nothing.
  * ⚠⚠ **`--worktree` IS NOT THE FIX, BECAUSE `extensions.worktreeConfig` IS NOT ENABLED.**
    Measured 2026-09-11 with the extension off: `git config --worktree commit.template
    .gitmessage` **exits 0**, writes to **`.git/config`** — the shared config — and creates no
    `config.worktree`. It does not error and it does not warn, so it silently produces the
    `--local` outcome it exists to avoid. **Enable the extension first
    (`git config extensions.worktreeConfig true`) or the instruction is worse than useless.**
### 2. Training & Troubleshooting Mandates

* **Training Workspace**: Training runs inside the ROCm Docker container (named `sonora_training`), whose compose definition lives in the `AI-Lab-AMD` sibling repo.
* **Active Log Review**:
  * To inspect the training progress, check the container logs directly: `docker logs sonora_training` or `docker logs -f sonora_training`.
  * Local execution metrics, hydra configurations, checkpoints, and tensorboard events are output directly to `logs/`.
  * ⚠ **`logs/` is NOT WRITABLE by a normal user on ai-lab-0** (`drwxr-sr-x root datashare`,
    created by a container on 2026-07-14). Group `datashare` has `r-x` only, so **no non-root
    process can write there — including `ai-mgr`, which the containers run as**. The directory
    is empty, so nothing has ever used it and the real training artifacts land under `/data`.
    Treat the line above as the intent, not the state: **probe before writing, and never
    assume a repo-relative log path is writable** — probe an env override, then a repo-relative
    dir, then `$TMPDIR`, and print which one you chose.
    Fixing the ownership needs root and has not been done.
* **Common Troubleshooting and Fixes**:
  * *Audio Decoding Failures (ROCm CUDA Mismatch)*: `torchaudio.load()` defaults to `torchcodec`, which fails inside the ROCm container due to missing CUDA dynamic libraries. Always use `soundfile.read(..., dtype='float32')` and convert to PyTorch tensors manually (see implementation in [matcha/data/text_mel_datamodule.py](matcha/data/text_mel_datamodule.py)).
  * *Matplotlib AttributeError in Validation*: Matplotlib 3.9+ removes `tostring_rgb()`. Use `np.asarray(fig.canvas.buffer_rgba())[:, :, :3]` instead (see implementation in [matcha/utils/utils.py](matcha/utils/utils.py)).
  * *Isolated build failures (NumPy 1.24.3 source compilation on Python 3.12)*: Pre-install `Cython` and run `uv pip install --no-build-isolation -e .` to reuse container-native compiled libraries.
  * *Local Gitignore Masking*: This repo's root `.gitignore` must contain `/data` (not recursive `data`) to avoid ignoring source code folders like `matcha/data/`.
* **Continuing Forward**: If a container run fails, apply the fix, commit to this repository, then **relaunch explicitly** (committing deploys nothing — GitOps was retired 2026-07-22; service deploys go through `AI-Lab-AMD/scripts/deploy.sh`) — `docker compose --profile training up -d sonora_training` from the `AI-Lab-AMD` repo (the service is profile-gated so ordinary syncs never start it). Container state is ephemeral: recreation wipes `/tmp` and pip installs, so copy artifacts out immediately (to `/data/model-training/…`, or promote blessed ones into the `Sonora/huggingface` registry checkout).

### 3. Python Tooling Mandate — uv

* **`uv` is the standard for all Python tooling in this organization**: interpreter/version
  management, virtual environments, dependency resolution, and tool execution. Prefer `uv pip`,
  `uv venv`, `uv sync` (with `pyproject.toml` + `uv.lock`), and `uv tool run` over bare
  `pip` / `python -m venv` / `pipx` / conda / poetry.
* **`uv run` is the one exception, and only for HOST scripts** (owner, 2026-08-01). Invoke them
  as **`.venv/bin/python scripts/…`**. `uv run` resolves dependencies against the repo
  `pyproject.toml` and *ignores* the inline PEP 723 block these scripts carry, which is what made
  four launch attempts fail that day, each on a different missing module. uv still owns the venv —
  create it with `uv venv`, populate it with `uv pip install --python .venv/bin/python …` — so the
  standard above is intact; only the invocation changes. Container scripts are unaffected.
  `tests/test_gate_scripts.py` enforces this for both shell scripts and Python docs.
* **To run the test suite: `uv pip install --python .venv/bin/python --group test`.** The set
  and the reasons for every pin live in `pyproject.toml` beside the group. ⚠ Deliberately not
  restated here — the same claim in two files drifted three times in one pull request, and
  after the third correction the two copies contradicted each other.
* **New work must use uv from the start** — new scripts, containers, CI steps, and docs. Do not
  introduce new `pip install` invocations.
* **Existing pip usage is legacy** and is being migrated; the catalog of migration points and their
  done-criteria lives in [AI-Lab-AMD/notes/cleanup-chores.md](../../AI-Lab-AMD/notes/cleanup-chores.md)
  in the `AI-Lab-AMD` sibling repo. When touching a file that contains legacy pip usage, migrate it
  as part of the change when practical (small diffs, e.g. `pip install X` → `uv pip install
  --system X` in containers), rather than leaving new debt.
* **Containers**: the `rocm/pytorch`-based service commands should bootstrap uv (single static
  binary — `pip install uv` once or `COPY --from=ghcr.io/astral-sh/uv`) and then install runtime
  deps with **`uv pip install --python /opt/venv/bin/python`** — these images ship their stack in
  an `/opt/venv` activated via PATH, and `--system` bypasses it into Debian's externally-managed
  Python, which refuses installs (PEP 668; learned 2026-07-13 when the first `--system` deploy
  crash-looped two services). Use `--system` only in images whose Python truly is the system one.
  uv's resolver speed materially shortens the recreate-reinstall cycle documented in this
  project's STATE ops notes.

### 4. The record of change — moved to [WORKFLOW.md](WORKFLOW.md)

⚠⚠ **WHAT A COMMIT MESSAGE MUST CONTAIN IS WORKFLOW, AND IT LEFT THIS FILE ON 2026-09-16.**
This section ran to 66 lines: what a message owes a reader, why the previous state was wrong,
the trailer rule, and how the history is the record. **None of it was migrated** — the owner
is establishing the replacement in `WORKFLOW.md` and asked that no old rules be carried over.

⚠ It is in git history at `1215cb7` and earlier. **Do not restore it from memory.**

**One thing here is a FACT rather than a rule, so it stays**: this repo's git history is the
only durable record of why a change was made. There is no tracker, no findings and no review
artifact any more, so a commit message is the last place a reason can live. That is a property
of the current state, not an instruction about length or form — `WORKFLOW.md` will say what
form the owner wants.

### 5. Measured defect patterns in this repo

⚠ **THIS WAS "CODE REVIEW STANDARDS" AND THE PROCESS PART IS GONE** (2026-09-16). When to
request a review, what was in scope for one, and what a review owed as a deliverable are
workflow — they belong in [WORKFLOW.md](WORKFLOW.md) if the owner re-establishes them, and
they were NOT migrated.

**What is left is not procedure. These are things that have actually gone wrong here, stated
so the next person does not rediscover them**, and they apply to anyone reading or writing
code in this repo whether or not anybody is reviewing it.

* ⚠ **A CLASSIFICATION AND THE INSTRUCTION BESIDE IT FAIL INDEPENDENTLY, AND THE SECOND IS
  WHERE THE DEFECTS HIDE.** Six instances across four rounds on 2026-08-11: in every one the
  code decided *correctly* and the instruction attached to it was wrong or impossible. A
  remedy naming a fix that could not address the cause; a bucket telling the reader to "score
  them first" about clips already scored; a comment claiming an override the tool never had.
  * *"Is this line true?"* is easy to read for. *"What would someone DO on reading this
    line?"* is a different question and almost never asked. Ask it of every message, comment,
    docstring and suggested remedy.
* ⚠⚠ **PROSE CLAIMING MORE THAN THE EXECUTABLE STATEMENT BESIDE IT IS THE MOST COMMON DEFECT
  THIS REPO HAS EVER PRODUCED** — measured across the review lane's whole life, and it
  outlived the lane. **Read the code, not the comment next to it.**
* ⚠ **AN EMPTY RESULT AND A BROKEN INSTRUMENT ARE INDISTINGUISHABLE.** Positive-control every
  negative and check that the check ran. A guard that passes over an empty population reports
  the same green as one that passed over a clean one.
* ⚠ **A RULE IS NOT A MECHANISM.** A sentence in a document cannot refuse anything, and a
  reader who believes the document over the code is misled by the more authoritative-looking
  one. This repo has paid for that repeatedly, in both directions.

⚠ **`.claude/**`, `AGENTS.md`, `CLAUDE.md` and `WORKFLOW.md` ARE CODE.** `.claude/commands/*.md`
is an executable prompt — it tells an agent holding push rights what to run — so it is closer
to a shell script than to a README. That began as a review-scope rule; with no review to
scope, it survives as a statement about what those files are.

### 5b. The doc-claims gate can stop enforcing WITHOUT going red

`scripts/gates/test_doc_claims.py` compares documented numbers against the artifacts on disk. It
has **five silent-disarm modes** — **four observed on 2026-08-11, and one reasoned from the
mechanism** (mode 2, see its own correction) — none of which was written down
anywhere until now — which is the same reason the workflow-validation trap in §1 cost the
same hour twice.

* ⚠ **A registry fact may only guard a STATIC artifact.** `audit-*/ratings.csv` is written
  **live** by the Dataset Listening app, so a count taken from it (the 1,279 keeps, say) moves as the
  audit continues, and a fact guarding it would go **red on correct work**. A gate that fails
  when nothing is wrong gets switched off, which costs every other fact in the registry.
* ⚠ **A fact enforced only in a file some convention DELETES is one deletion from being
  disarmed** — because the checker reports only a *disagreement*, so a fact matching no
  document simply passes.
  * ⚠ **CORRECTED: `v5 speakers` is NOT an example of this, and an earlier version of this
    bullet said it was.** Measured against the merge base: it had **two** enforced lines, in a
    `notes/code-review-*.md` and in `configs/data/…_v6.yaml:8` — the latter in scope since
    `configs/data/*.yaml` was added to the scan. Deleting the review document took it **2 → 1**,
    never to zero, so it was never disarmed. The hazard is real as a class; this was not an
    instance of it, and the claim was made by measuring on a tree that predated the config
    scan.
  * ⚠ **The same-line rule — real, and the reason mode 2 looked instantiated:**
    `notes/STATE.md` already stated v5's 2,500 speakers, but `scope` is
    matched **per line** and the scope token sat on the line above, so the fact never matched
    there. **Check that a fact's scope and its number are on the SAME LINE.**
  * ✅ **This class is now ENFORCED IN CODE, not by vigilance.** #62 added
    `tests/test_doc_claims_registry.py::test_every_fact_recognises_at_least_one_live_statement`
    — *"a fact no document states is a fact nobody is checking."* An earlier version of this
    bullet said nobody had swept for other facts in this position; that was true when written
    and #62 made it false. Do not re-add a manual sweep: assert on the registry instead.
* The general form of both: **a fact that matches zero lines is indistinguishable from a fact
  that passes.** When you add or move an enforced sentence, count the lines it matches — do
  not infer enforcement from a green gate.

⚠ **AND THE SAME SHAPE OUTSIDE THIS GATE: a tool's FAILURE is easily mistaken for its
NEGATIVE RESULT.** Four instances on 2026-08-11, three of them within an hour, each costing a
wrong answer stated confidently:

| Instrument | Failed because | Read as |
|---|---|---|
| `test_doc_claims.py` | fact matched no document | fact passes |
| `gh pr diff N -- path` | takes one arg; errored to **stderr** | empty diff, file untouched |
| `git merge-tree` | prints conflict to **stdout**, grepped stderr | no conflict |
| `git merge <ref>` | branch name misspelled, ref missing | merge conflict |

**Before believing a negative result, check that the instrument ran.** Non-zero exit, an empty
list where a real answer prints something, output on the stream you are not reading — all
produce a confident nothing. Two of the four above were caught only because another agent
contradicted the claim, and in both of those cases the person had *already run* the command
that would have falsified it and stopped reading once the answer looked settled.

**A cheap falsifier that is not run is not evidence.**

⚠ **A THIRD DISARM MODE, AND THE ONLY ONE READING CANNOT FIND: a test whose PREMISE and whose
SUBJECT are wrong in the same direction passes.** Found 2026-08-11 in
`test_one_bad_axis_is_enough_to_hold_a_clip_out_of_keeps`: the fixture set an intended `V: 0.9`
against a stub measuring `−0.79`, so V actually FAILED and the clip was a direction failure
too — "held out on the label alone" was false of it. **The assertion passed anyway, because the
count it asserted on was the buggy one.** Two defects agreeing, a green test between them, and
neither visible from the other.

* **The tell is that it cannot be found by reading either the test or the code** — only by
  running the case and looking at what the subject actually did, rather than at whether the
  number came out as expected.
* So when a guard's own test is the evidence that the guard works, **check the fixture states
  what you think it states.** A passing assertion proves the two sides agree; it does not prove
  either is right.
* Same family as the other two: the green result is indistinguishable from the correct one.

⚠ **A FOURTH MODE, AND THE ONLY ONE WITH NO MISSING MEASUREMENT: a claim can be internally
incoherent and still survive two readers, when both like the story it tells.** Found
2026-08-11, in this repo's own review lane. The claim was *"numba 0.53.1 cannot build on 3.12,
**but** the same resolution silently downgrades librosa by a major version"* — offered as the
stronger argument for a version pin, agreed with by a second agent in the same terms, and
**impossible**: if the build fails nothing is installed, so nothing is lived with. The
downgrade is planned in the resolution and never installed.

* **The resolution had been measured.** So "run the falsifier" would not have caught it — the
  numbers were right and the conclusion drawn from them was not.
* **The check that does catch it: state the END STATE, singular.** What does the machine
  actually end up in? One install has one outcome, and naming it forces the incompatible
  halves into the same sentence where they cannot both stand.
* ⚠ **A second reader agreeing is not verification** — it is likelier to be two people liking
  the same story. The agreement arrived one round after a review had already corrected the
  previous version of the same sentence.

⚠ **MODE 5, the inverse of mode 4: "the number is unchanged" is itself a claim, and it has to
be re-derived rather than inferred from the fix looking conservative.** Mode 4 is a measured
number with an unmeasured conclusion; this is an unmeasured number that a narrow-looking change
invites you to assume still holds. Found 2026-08-11 when a `ge90` change left "13 of the 20
campaigns" unregenerated — it was re-run and is still 13, so the note was right, which is
exactly why the habit is dangerous: being right this time costs nothing and teaches the wrong
lesson. **If a fix touches an input to a stated number, re-derive the number.**

⚠⚠ **A DOCS-VS-CODE FINDING IS A SWEEP, NOT A LINE EDIT** (owner, 2026-08-19: *"we should
always check the overall docs to ensure that the code's state is not recorded in a
contradicting document somewhere. We've run into cases of that before."*).

When a comment, docstring or commit message turns out to disagree with the code, **the fix is
not finished when that sentence is corrected.** Search the whole tracked tree — an
unrestricted `git grep`, never one scoped to `-- '*.md'` — for the same claim before closing
it: the number, the behaviour, the function's contract. The trigger sentence names comments
and docstrings, which no `.md` sweep can reach, and the mis-scoping was measured costing a
round: #279's sweep, correct against the rule as then written, missed the identical triple
in two source files (#288). (#274's earlier miss of the same claim was a sweep not run at
all, not one mis-scoped — #290.) This repo has
paid for the other half repeatedly: a deleted `CLAUDE.md` went on being obeyed from memory for
eight commits, and three doc-vs-artifact drifts turned up in a single day.

**Measured the day the rule was written.** Fixing `scm.validate` to agree with
`schemas.coerce_axis` looked complete and consistent at the call site. It was wrong:
`docs/markup-schema-brief.md` — the RATIFIED SCM v0.1 contract, three directories away —
says the sidecar stores VAT **continuous**, so the "fix" made the validator certify a sidecar
the contract forbids. Nothing failed; the comment beside the code was accurate. **The same
sweep found the brief contradicting ITSELF**: its field-semantics table gave the verifier
tolerance as `±0.25` while §5 item 6 of the same file recorded the owner's same-day amendment
to `±0.35`, which is what `scm.VAT_TOL` implements. That stood for a month.

* ⚠ **Prefer a POINTER to a restatement.** The table cell now names `scm.VAT_TOL` instead of
  repeating a number. A value in two places drifts; that is §5b's whole argument, applied to
  documents rather than to code.
* ⚠ **Do not add a third copy while fixing the second.** The first attempt at the amendment
  note above restated the history that §5 item 6 already carried. Point at the existing
  record.
* ⚠ **The registry can hold a CODE constant, not only a corpus number** —
  `scripts/gates/test_doc_claims.py`'s `const()` reads one by AST. A number that lives in
  code and is quoted in prose is exactly as driftable as a row count, and until 2026-08-19
  the gate could not see that class at all. **Register it rather than trusting the sweep to
  happen again.**

### 5c. A pipeline stage is WIRED, or it is merely WRITTEN ABOUT

`scripts/` holds **on the order of a hundred non-test `.py` files** (per-directory counts:
[scripts/README.md](scripts/README.md)'s table) and
most of them are *correctly* uninvoked — operator tools, finished campaign tooling. So "nothing calls this" carries no signal there, and a
**stage** that stopped being called is indistinguishable from a tool that never was. That is
how `qc_verdict.py` was named in a `synth_bank.sh` comment for a month and never ran, while
695 directed clips reached the ear with no direction check (issue #24).

* **The buckets say what a file IS; the manifest says whether it RUNS. Read
  [scripts/README.md](scripts/README.md) before adding anything under `scripts/`.**
  `stages/ lib/ tools/ gates/ assets/ teacher_audition/ litert_export/` (#26 step 3,
  2026-08-12). Every file under `scripts/<bucket>/` is **exactly two levels down** on
  purpose — one repo-root expression is then correct everywhere, which is what replaced 87
  scattered `sys.path.insert(0, dirname(__file__))` calls and what makes the path guard
  below possible.
* ⚠ **Checked-in data a script reads goes in `scripts/assets/`, resolved from the repo root.**
  `tests/test_asset_paths.py` asserts every in-repo path built from `__file__` points at the
  thing it names. That class of bug does not fail at import — it fails when something READS
  the path, which for the synthesis lane is hours into a GPU render, and for
  `book_ingest`'s register lexicon does not fail at all: the load sits inside
  `except Exception: return []`, so a wrong depth silently yields an empty controlled
  vocabulary and every bank after it is built without one.

* **`scripts/pipeline_manifest.py` declares every stage and which shell wires it**, and
  `tests/test_stage_coverage.py` iterates it in both directions: a stage declared and not
  invoked fails, a script invoked and not declared fails, and a `.sh` appearing under
  `scripts/` in none of the three categories fails. **Wire a stage → declare it in the same
  commit.** How to do that is in the manifest's docstring, which is the only copy.
* ⚠ **Two things look like a call and are not.** A **comment** (the #24 failure verbatim) and
  an **`echo`** — every stage in `synth_bank.sh` prints its own re-run command on failure, so
  the recovery hints name the very scripts under test. `tests/test_audit_sampling.py::_invocations`
  is this repo's one definition of "a call"; import it, never re-derive it. The guard that
  predated it asserted a substring over the whole file, and commenting out the real invocation
  left it **green**.
* ⚠ **A ratchet never observed going red is not known to work.** This one was built by mutating
  the tree eight ways — unwire a stage, demote it to an `echo`, add an undeclared stage, add a
  new shell, reverse a recorded non-invocation, unwire the nested EIV lane, hardcode a target
  into the dynamic-dispatch wrapper, empty the manifest — and recording which test caught each.
  The last is the point: **every test there iterates the manifest, so an emptied manifest would
  collect zero cases and report green.** That is §5b's mode 1 in a different file, and it is why
  `test_the_manifest_declares_the_lanes_it_is_supposed_to_cover` exists. Do the same for the
  next ratchet: a coverage test's own coverage is not self-evident.

### 6. Execute From The Repo — `/data` Holds Data

**Owner principle (2026-08-06): code executes from the repo checkout; `/data` holds what
its name implies** — datasets, checkpoints, model artifacts, venvs, training logs, service
runtime state, vendor checkouts. A byte-copy of our source under `/data` is something to
*remove*, not to manage. The full inventory, what is legitimately untracked, and the audit
behind this rule are in `notes/data-mirrors.md` (private).

* **When a tool writes its outputs next to its own source** — the usual reason a `/data`
  copy exists at all — give it an artifact-root variable defaulting to the script's own
  directory, point that at `/data`, and delete the copy. That is the shape that satisfies
  both halves. Worked example: `SONORA_LITERT_WORK` and
  [scripts/litert_export/run.sh](scripts/litert_export/run.sh).
* **For containers, bind-mount the repo path**, not a `/data` copy of it.

Where a copy still exists, the rest of this section governs it. **The repo is
authoritative in every case.**

* **Never edit code under `/data`.** Change it in the repo, commit, then deploy. An edit made
  on `/data` has no history, no diff and no review, and no way for anyone
  else to discover it happened.
* **A `/data` file that is *newer* than its tracked original is not authoritative — it is
  unreviewed.** If that edit is the one you want, commit it in the repo and redeploy; do not
  let the copy become the record.
* **Deploy explicitly.** Committing deploys nothing (GitOps retired 2026-07-22). Service
  stacks go through `AI-Lab-AMD/scripts/deploy.sh`; the training **deployment** at
  `/data/repos/Sonora` through `deploy.sh training-code`. ⚠ This said "the training clone"; it
  has not been a clone since 2026-08-29 (§7's table, and `deploy.sh`'s own header).
* **Adding a tool that will run from `/data`? Add it to `MIRRORS` in
  [tests/test_data_mirrors.py](tests/test_data_mirrors.py) in the same commit.** That gate
  compares every tracked file against its deployed copy and fails on any difference. It is the
  only thing standing between us and this failure mode, and it only covers what it is told
  about.
* **What legitimately lives only on `/data`, and must stay untracked:** venvs, `artifacts*/`,
  checkpoints, training logs, datasets, vendor checkouts (`toolchain/Zonos`,
  `services/comfyui`, `services/unsloth`, …) and service runtime state.

**Why this is a mandate and not a preference.** A copy that stops updating looks exactly like
a copy that is up to date — both directories are healthy, the scripts run, and the only
symptom is that a fix you believe shipped did not. It has already happened twice:

* `convert_vat.py` on `/data` ran **three weeks stale**, missing the `detect_vat_dim` seam
  guard that this repo recorded as landed. The guard was written, reviewed, committed and
  never installed.
* The training deploy clone went **dead for two weeks** in July, because `deploy.sh
  training-code`'s fast-forward pull cannot cross a history rewrite and failed silently.

Neither was found by noticing something break. Both were found by going to look.

---

### 7. The Deploy Cycle — repo → `/data`, and the check that comes FIRST

§6 says *why* the repo is authoritative. This is the *procedure*, and it is mandatory for
every target with a `/data` copy. Canonised 2026-08-08, when `audition/` moved into this repo
and the cycle stopped being another repo's internal business.

**Targets, and where each is sourced from.** `scripts/deploy.sh` lives in `AI-Lab-AMD`
because it is box tooling, but it deploys from whichever repo owns the code:

| target | source | command |
|---|---|---|
| `audition` → `/data/services/audition/app` | **this repo**, `audition/` | `deploy.sh audition` |
| `training-code` → `/data/repos/Sonora` | **this repo** (rsync; ⚠ **NOT a checkout since 2026-08-29** — it carries no `.git`, so the "ff-pull" this cell claimed cannot happen) | `deploy.sh training-code` |
| `dashboard` → `/data/services/dashboard` | `AI-Lab-AMD/dashboard` | `deploy.sh dashboard` |
| `stack` → the compose services | `AI-Lab-AMD` | `deploy.sh stack` |

⚠⚠ **AN AGENT WORKED IN THE DEPLOY CLONE FOR A WHOLE SESSION (2026-08-18), AND EVERY FACT
IT NEEDED WAS IN THIS SECTION.** It made thirteen commits in `/data/repos/Sonora`, including
two feature branches and a push to `origin/main`. Nothing stopped it: the clone has a `.git`,
the right remote, a clean tree, and passing tests. It *looks* exactly like a checkout.

How it got there is the part worth keeping. It reasoned from `git remote -v` (same URL), the
first reflog entry (`clone: from github.com/...`, not from a local path) and
`objects/info/alternates` (absent), and concluded the two trees were peer clones. Every one
of those observations was true. **None of them is a statement about what the system DOES** —
that lives in `AI-Lab-AMD/scripts/deploy.sh`, which names this path in a one-line comment at
the top and took ten seconds to read once anyone looked. The owner asked *"is that not a
downstream deployment?"* and was told no, twice, with evidence.

Nothing was lost, by luck rather than design: the work had been pushed to `origin` and copied
into the source repo for unrelated reasons. Had `deploy.sh training-code` run first, the
ff-pull would have taken the tree with it.

**The check that costs nothing: before editing any tree, confirm no `deploy.sh` target names
it.** The table above is the list. §6 states the principle; this is what ignoring it looks
like from the inside, which is: entirely normal.

⚠ **`deploy.sh` reads `SONORA_REPO` for the source checkout and defaults it to
`Sonora/github` — the caretaker's tree, which is usually on a feature branch.** Scoped to
`deploy.sh` on purpose: the same name has three *other* readers in this repo with different
defaults (`scripts/litert_export/run.sh`, `convert_vat.py`, `scripts/stages/score_holdout.py`
— the last falling back to the container path `/sonora`), so an unqualified "it defaults to
`Sonora/github`" is false of every in-repo consumer.

An agent working in a linked worktree passes it **per command**:

```bash
SONORA_REPO="$(git rev-parse --show-toplevel)" deploy.sh audition
```

* `--show-toplevel`, not `$PWD`: it is exact from any depth and returns the *linked
  worktree's* root. `$PWD` means "wherever I happen to be standing", and an agent that
  `cd`'d into `audition/` to make the change would hand `deploy.sh` a source root of
  `<worktree>/audition`. ⚠ **An earlier version of this bullet said that root "inherits a
  git dir from the worktree and passes `require_source`". It does not** —
  `<worktree>/audition/.git` does not exist at all, so `require_source`'s `-e` test *refuses*
  it. The advice is right and the reason was wrong, which contradicted the `require_source`
  note below — and the errata itself then pointed at that note as "eight lines down" when it
  is nearer thirty, a stale pointer inside a correction about stale claims. Its replacement
  said "at the end of this subsection", which was *also* positional and *also* wrong (the
  note sits about two-thirds through §7). **Positional cross-references rot — name the thing
  and let the name carry it.** Use
  `--show-toplevel` because it is correct from any depth, not because anything downstream
  fails to catch `$PWD`.
* ⚠ **Prefix it; do not `export` it.** `scripts/litert_export/run.sh` honours an inherited
  `SONORA_REPO` through `${SONORA_REPO:-…}`, so an export run later in the same session
  would resolve `import matcha` out of the worktree instead of the caretaker tree.
  `convert_vat.py`'s guard cannot see that — it only asserts `matcha/models/matcha_tts.py`
  exists, which is true of any worktree — and its own comment names the stakes: *a wrong
  export converts cleanly and its graphs run*.

⚠ **THE CHECK THAT ANSWERS "will this deploy revert someone's work" IS AN ANCESTRY TEST, NOT
A LOG RANGE.** `rsync --delete` replaces the target with *your tree*, so the question is
whether your tree already contains everything the deployed copy has:

```bash
git fetch origin                                          # remote-tracking refs are SHARED
                                                          # across worktrees and may be days old
git merge-base --is-ancestor <deployed-sha> HEAD           # exit 0 ⇒ your tree is a superset
git log --oneline HEAD..<deployed-sha> -- audition/        # non-empty ⇒ this deploy reverts work
```

An earlier version of this bullet said to run `git log <deployed-sha>..origin/main -- audition/`.
**That cannot detect the failure it was written for**: it compares two refs, neither of which
is the tree being rsynced, and whenever the deployed sha *is* main's tip the range is empty by
construction — so it prints nothing and reads as "safe" while your branch, forked before a
commit that touched `audition/`, is about to delete it. The 2026-08-12 deploy that prompted
this was in fact safe (`20713f1` is an ancestor of that branch's HEAD), which is the danger:
the wrong check agreed with the right one that once.

`require_source` used to refuse a worktree outright — `.git` is a file there, not a directory
— fixed in AI-Lab-AMD `882f620` (2026-08-12).

⚠ **A SQUASH MERGE ORPHANS A DEPLOY STAMP MADE FROM THE BRANCH, AND `status` THEN LIES.**
Observed 2026-08-12 minutes after #74 merged: the audition target read
**`STALE — source changed since (7557aea)`** while its bytes were **byte-identical to
`main`**. The stamp named the branch commit `c444f2c`, which the squash had replaced, so
`stamp_is_current` compared against a commit no longer reachable. Step 0 tells you to treat
STALE as drift and resolve it before editing — here there was nothing to resolve, and the
fix is one idempotent re-run: `deploy.sh audition` diffs first, finds nothing to copy, and
refreshes the stamp **without restarting**. So: **after a branch-sourced deploy is
squash-merged, re-run the deploy once to re-stamp.**

⚠ **Settle "orphaned stamp vs. real drift" with the rsync dry-run, NOT with
`test_data_mirrors`.** An earlier version of this note nominated that test, and it cannot
make the distinction: `MIRRORS` registers `audition/app` with a NON-RECURSIVE `*.py` glob,
which is `main.py` alone — one of the four tracked files under `audition/`. A green run
therefore says nothing about `static/`, and reading it as "the bytes match" is the
instrument-failure-mistaken-for-a-negative-result shape from §5b.

What answers it is the dry-run **with `deploy.sh`'s own three excludes**, which is byte-exact
over every file — an empty plan means the stamp drifted and **the content** did not (content
only; see the limit stated below the command):

```bash
sudo rsync -rl --delete --checksum --dry-run --itemize-changes \
  --exclude='__pycache__' --exclude='.deployed.json' --exclude='_contract/' \
  audition/app/ /data/services/audition/app/
```

⚠ **`-rl` ANSWERS ONE QUESTION: does the CONTENT match.** It cannot see **mode or ownership
drift** — dropping `-p`, `-o`, `-g` retires those comparisons along with the time comparison,
so a file deployed `0600` that should be `0644`, or owned by `root` where it should be the
service user, is byte-identical under `-rl` and prints nothing. That is the intended trade:
content drift is what this section exists to settle, and permission drift has other symptoms.
But it means **an empty plan here is not a full clean bill of health**, and the empty-plan rule
below is scoped to content only. (`-D` goes too, so a non-regular file is skipped with a
warning on **stderr** rather than itemised on stdout — irrelevant for this tree today.)
Stating the limit rather than implying its absence is the point: the two previous spellings
were both replaced for inviting a reader to treat silence as clean.

⚠ **`-rl`, NOT `-a`, AND NO `grep`.** This is the third spelling of this command; the
reasoning matters more than the flags.

`-a` is `-rlptgoD`, so it itemises differences in **time, perms, owner and group** as well as
content, and `--checksum` changes only how rsync *decides to transfer*, not what it reports.
A file with identical bytes and a skewed mtime prints `.f..t......`, and the empty-plan rule
above reads that as drift. The previous fix for that was to keep `-a` and pipe through
`grep -E '^[<>]fc'`, on the grounds that `c` in column 3 is the content verdict. **That
filter was worse than the noise it removed**, because `c` only ever appears for an item that
exists on BOTH sides. Measured on a synthetic pair (rsync 3.4.1):

| situation | itemised as | survived `^[<>]fc`? |
|---|---|---|
| content differs | `>fc........` | ✅ |
| identical bytes, skewed mtime | `.f..t......` | ✅ correctly dropped |
| **file present in the repo, MISSING from the deployment** | `>f+++++++++` | ❌ **dropped** |
| **file in the deployment, absent from the repo** | `*deleting   path` | ❌ **dropped** |
| symlink whose target changed | `cLc........` | ❌ dropped |
| directory not yet created | `cd+++++++++` | ❌ dropped |

For a newly-created item rsync replaces **every** attribute letter with `+`, so the content
column is `+` and never `c`; a deletion carries no update flags at all; and a changed symlink
puts `c` in column 1, so `[<>]` misses it even though column 3 matches. Net effect: a
deployment missing half of `static/` printed **nothing**, and the operator was told in bold
to read nothing as *no drift*. That is the §5b shape again — and one rung worse than the
version it replaced, which at least erred loudly.

Dropping `-ptgoD` fixes it at the source instead of filtering the symptom: rsync never
compares time, perms, owner or group, so **every row above reports** and there is nothing left
to filter — at the cost of the mode/ownership comparison noted above, which is a narrowing of
the question, not free. Verified both directions — mtime-only skew across five files prints
nothing; content drift, a missing file, an extra file and a changed symlink all print.
⚠ It also removes an exit-code trap: `rsync … | grep` exits **1** when there is no drift, so
the clean path was the failing path, which bites the first person to lift this into a
`set -e -o pipefail` script.

⚠ **`-rl` IS FOR THIS DIAGNOSTIC ONLY — never deploy with it.** `deploy.sh` uses `-a`
deliberately; a copy without `-ptgoD` would strip modes and ownership off a live service
directory. `--dry-run` is spelled long here so that deleting it is a visible edit.

⚠ **The excludes are not optional either**, and dropping them is the same mistake one rung down:
`.deployed.json` and `_contract/` are written by `deploy.sh` AFTER the copy and exist in no
commit, so a bare dry-run reports them as deletions and reads as drift. Measured just now:
five `*deleting` lines, not the "three" an earlier version of this note claimed — `_contract/`
contributes its own contents as well as itself, and the `__pycache__` count moves with
whatever interpreters have run, so **the number is not worth stating**, only the shape.
⚠ And `deploy.sh
audition` is the one-step version of this only on a CLEAN tree — `require_clean` refuses a
dirty one, correctly, so it is not available as a diagnostic mid-edit.

**Deploying is idempotent and free to over-run** (2026-08-08). `deploy.sh audition` /
`dashboard` diff the target first and do **nothing** when it already matches — no copy, no
stamp rewrite, no container restart. So running a deploy at the outset of any work costs
nothing and requires no judgement about whether it is needed; that is the point, because a
deploy that always restarted a live rating app made the safe habit the expensive one.
`training-code` (an **rsync** whitelist copy since 2026-08-29 — this said `git pull --ff-only`,
which that target has had no `.git` to do since) and `stack` (`compose up -d`) were already
idempotent by construction.

⚠ **`stack` REFUSES during a training run**, because `up -d` restarts manually-stopped
services and would put every inference engine back on the GPU under a live run — the
spin-down rule broken by a command that never mentions inference. Override with
`ALLOW_STACK_DURING_TRAINING=1`, then re-run `inference-engines.sh stop`.

#### 0 · BEFORE touching related code — confirm the deployed copy is current

**This step comes first and is the point of the section.**

```bash
../../AI-Lab-AMD/scripts/deploy.sh status                  # which revision is running
.venv/bin/python -m pytest tests/test_data_mirrors.py -q   # do the bytes still match
```

A target reading **STALE**, **DIRTY**, or *"deployed from a commit not in this repo"* means
the running copy is not the code you are about to edit. Editing on top of that and then
deploying **silently clobbers** whatever was actually live — and since `rsync --delete` is
not recoverable, there is nothing left to diff. Resolve drift *before* the first edit.

Not hypothetical: `convert_vat.py` ran **three weeks stale** on `/data` — the repo gained a
`detect_vat_dim` seam guard, the notes recorded it as landed, and the harness that actually
executed never received it. *A guard that is not installed is not a guard*, and the only
symptom was that a fix believed shipped had not.

#### 1 · Edit in the repo · 2 · Commit · 3 · Deploy · 4 · Verify

```bash
git commit …                                    # a dirty tree is REFUSED by deploy.sh
../../AI-Lab-AMD/scripts/deploy.sh <target>
../../AI-Lab-AMD/scripts/deploy.sh status       # the target should now read `current`
.venv/bin/python -m pytest tests/test_data_mirrors.py -q
```

#### THE RULE: edit at the SOURCE, deploy to the target. Never the reverse.

**Code is edited in the repo and deployed. It is never edited at the deploy target, and
never copied back from one.** This is not a style preference and it has no exceptions:

* An edit under `/data` has **no history, no diff and no review**. A
  `/data` file that is *newer* than its tracked original is not authoritative — it is
  unrecorded, and it will be destroyed without ceremony by the next deploy, because
  `rsync --delete` leaves nothing to recover or diff.
* It defeats every guard we have. `test_data_mirrors.py` reports drift but cannot tell an
  intentional in-place fix from an accident; `.deployed.json` will name a commit that does
  not contain what is running; and `deploy.sh status` will read `current` while the target
  holds code that exists nowhere in git.
* **Enforced, not merely stated.** `deploy.sh` diffs the target before copying and, when it
  finds content the repo does not match, prints the exact files and says they are about to
  be overwritten *before* doing it — so an in-place edit is caught at the moment it costs
  nothing rather than discovered later as a mystery.

If an edit was made at the target and is the one you want: **port it into the repo, commit
it, and deploy** — do not rsync it backwards.

#### Rules that travel with the cycle

* **Register a new target in the same commit that first deploys it.** `MIRRORS` in
  `tests/test_data_mirrors.py` covers only what it is told about, and
  `test_there_is_something_to_check` exists so a layout change cannot quietly empty it.
* **A contract the deployed copy needs must be SHIPPED, never transcribed.** The audition app
  cannot import `matcha` — its container has fastapi and uvicorn and nothing else — so
  `deploy.sh` copies `matcha/delivery.py` into `app/_contract/` **after** the `rsync --delete`,
  and the app **refuses to start** without it. Same pattern as the device G2P's exported
  `g2p_contractions.json`. A fallback to a literal would re-create the fork the asset exists
  to delete; D-C1's lesson is that a hand-synced table is a defect of omission on a delay.
* **Staleness is judged per source PATH, by ancestry** — against the last commit touching that
  target's own path, never against repo HEAD. A HEAD comparison called `dashboard` BEHIND for
  a week while it was perfectly current, and a check that cries wolf gets ignored, which is
  worse than no check.
* **A dirty tree is refused.** `ALLOW_DIRTY=1` overrides and the stamp records that it was
  used, so the exception leaves a trace.

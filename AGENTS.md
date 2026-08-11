# AGENTS — Project Sonora (Training Repo)

This is the entry point for any agent or developer working on Project Sonora's training
codebase. This is an independent GitHub repo (the PyTorch training pipeline that produces the
actor model artifacts published to the `Sonora/huggingface` sibling checkout). Internal engineering notes —
architecture, current state, open decisions — live in [notes/](notes/), mapped one-line-per-file
in [notes/README.md](notes/README.md). Before starting work, read
[notes/STATE.md](notes/STATE.md) for the current state of the project and
[notes/todo.md](notes/todo.md) for the open work.

---

## Core Stack Matrix

* **Language Ecosystem:** Python-based ML training pipeline.
* **Second lane — teacher synthesis (`scripts/synthesis/`):** five audited TTS engines
  (`chatterbox`, `qwen`, `zonos`, `orpheus`, `moss_vg`; `vibevoice` + `dia` are set aside,
  reversible — `ref_select.SET_ASIDE`) driven by a two-pass Gemma director — per-line
  V/A/T + a register copied from the 47-label controlled lexicon
  (`scripts/synthesis/register_lexicon.json`), then per-engine casting/delivery from
  `scripts/synthesis/director_skills/<engine>.md`. Engine allocation is the three-layer
  `ref_select.ENGINE_MIX_BY_LANE` (capability veto · measured per-lane weights ·
  diversity floor). **`build_direction()` in `book_ingest.py` is the single source of
  truth for what each engine actually receives — never bypass it; unknown engines are fatal.**
  The 2026-07-25 relay audit found rich direction had been silently dropped by every engine.
  **Standing rule:** no TTS model enters the portfolio without a studied interface and a
  Gemma skill-file adapter — see [notes/tts-engine-onboarding.md](notes/tts-engine-onboarding.md),
  which also carries the known-gotchas reference. Minimum clip length is **4 s of estimated
  speech**, gating input text (not render duration).
* **Core Framework:** PyTorch & Matcha-TTS (conditional flow-matching).
* **Environment:** AMD ROCm PyTorch Docker container (optimized for Ryzen AI Max+).

---

## Integration Dependencies

* **Deployed code is a copy; the repo is the source** — see §6 and [notes/data-mirrors.md](notes/data-mirrors.md).
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

* **Canonical root marker files → `UPPERCASE`** (`SCREAMING_SNAKE_CASE` if multi-word): `README.md`, `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`, `ROADMAP.md`, `AGENTS.md`. Keep this set small and curated.
* **Top-level anchor docs → `UPPERCASE`, single word preferred:** `ARCHITECTURE.md`, `STATE.md`.
* **All other docs & notes → `lowercase-kebab-case.md`:** e.g. `open-decisions.md`, `code-review-findings.md`. This is the rule for everything in `notes/`.
* **Source code → the language's own convention:** Rust `snake_case.rs`, Swift `PascalCase.swift`, Kotlin `PascalCase.kt`.
* **Never** let case be the only difference between two paths, and always reference files with their exact case.

---

## System Operational Mandates

### 1. Commit Hygiene

* **`main` is PR-only. Do not push to it directly** (owner, 2026-08-10). Branch, push the
  branch, open a PR, and let it merge. This applies to agent sessions exactly as it applies
  to the owner — an agent that "just needs one small fix on main" is the case the rule exists
  for. The two reasons it is a rule and not a preference:
  * **The Mac and `ai-lab-0` (and their agent sessions) commit concurrently.** Direct pushes
    to a shared `main` are how two sessions silently interleave half-finished work; a branch
    is a place for work to be incomplete without being everyone's problem.
  * **Nothing reviews a direct push.** `.github/workflows/claude-review.yml` triggers on
    `pull_request`, so work that skips the PR skips the review entirely — the automation
    cannot see a commit that was never proposed.
* **Branch naming**: `<type>/<short-slug>` matching the commit type (`fix/`, `feat/`,
  `docs/`, `chore/`), e.g. `fix/holdout-filelist-width`.
* **Work on the branch, commit and push liberally, open the PR only when the work is done**
  (owner, 2026-08-10). Pushing to your own branch is free and is the entire point of having
  one: commit early, commit often, push whenever, and let the branch hold work that is not
  yet finished. What is deliberate is the *timing of the PR*, not the timing of the commits.
  * **When completion is defined, completion opens the PR.** If a `/goal` has been set,
    achieving that goal IS the completion point — open the PR then, without being asked again.
  * **Otherwise the owner calls it.** With no goal set, work, push, and wait: the owner
    acknowledges the completion point and the PR follows from that.
  * **This is also what makes it cheap.** `.github/workflows/claude-review.yml` fires when a
    PR is opened AND on every push to an open one, so a PR opened at the *start* of the work
    bills a full model-rate review of half-finished code on every intermediate push. Opening
    at completion buys exactly one review, of work that is actually ready to be read.
* **Pull before push, every time.** Run `git pull --rebase` as the first step of any
  commit-and-push sequence on your branch, and rebase on `main` before opening the PR. If the
  tree holds the owner's uncommitted local edits, fetch and check ahead/behind instead of
  forcing a rebase.
* **The exception is the owner's, not yours.** If the owner explicitly directs a direct push
  to `main`, that is their call to make and it does not need re-litigating — say the rule
  exists once, then do as asked. An agent never grants itself the exception.
* ⚠ **A rule in this file is not an enforcement mechanism.** This project has already learned
  that the expensive way — `deploy.sh`'s "deploy the stack only when a service change is
  intended" was a header comment for weeks and then got ignored eleven hours into a live
  training run, which is why it is now a hard refusal in code. Treat the same way here: the
  authority is the branch protection on `main`, and this section only explains it. If a direct
  push to `main` ever *succeeds*, the protection is missing or was bypassed — report that
  rather than taking it as permission.
* **Review findings are resolved IN THE PULL REQUEST, by the machine that submitted it**
  (owner, 2026-08-11). The review workflow only comments; the repair runs here, via
  `scripts/fix_pr.sh <pr>` — or `/fix-pr <pr>` in a Claude Code session, which runs the same
  protocol from the same file (`.claude/commands/fix-pr.md`). It reads the unresolved threads,
  fixes what is genuinely wrong, replies in each thread, and pushes to the PR branch. A
  review comment is an argument, not an order: the fix pass is expected to push back in a
  reply where a finding is wrong, rather than making a change it believes is wrong.
  * **Two practices were retired to get here, and neither should be reintroduced casually.**
    * **The `claude-fix` label and `.github/workflows/claude-fix.yml` are gone.** The lane
      failed for an environmental reason, not a logical one: the runner image never carried
      what a real repair needs (torch, the ROCm stack, `/data`, the repo venv), and shipping
      that into CI cost more than the lane was worth. The submitting machine already has all
      of it. ⚠ `scripts/fix_pr.sh` refuses to run when it cannot execute the test suite —
      that refusal is the guard against rebuilding the same defect at a new address.
    * **Findings are no longer filed as issues.** On 2026-08-11 that practice turned a
      handful of findings into 25 issues in three bursts inside half an hour: each review
      produced a backlog, the backlog produced the next PR, and the PR produced the next
      review. The reviewer now holds neither `issues: write` nor `Bash(gh issue:*)`, because
      §1's own lesson is that a rule in a prompt is not a mechanism. Issues opened *before*
      that date are live work and stay open; `fix_pr.sh` checks them read-only so a fix pass
      does not collide with whoever owns one.
  * **What bounds the loop now is the commit range, not anyone's identity.** The reviewer
    ends each summary with `<!-- janis-reviewed: <sha> -->` and reviews only `<sha>..HEAD` on
    the next run, posting nothing when that range is empty or docs-only. So a fix push buys a
    small incremental read of the fix itself, and each lap is smaller than the last. This is
    deliberately identity-agnostic: it keeps working when agents commit under their own
    GitHub identity rather than a human's.
  * ⚠ **A PR THAT EDITS `.github/workflows/claude-review.yml` IS NOT REVIEWED, AND THE CHECK
    STILL GOES GREEN.** `claude-code-action` validates that the workflow file is
    byte-identical to the version on the default branch and, when it is not, logs
    `Skipping action due to workflow validation` and exits `conclusion: success`. The run
    takes ~37 seconds instead of the usual 11–14 minutes and posts nothing. **The green check
    means "skipped", not "clean"** — PR #59 rewrote that workflow and was merged unreviewed
    on the strength of it. This is a deliberate vendor security control, not a bug.
    * **This was already known, and that is the actual lesson.** It was identified during
      the lane's setup on 2026-08-10 — instrumentation has to land on `main` first — but the
      knowledge lived only in an agent's session memory and was never written into this
      repo, so it protected nobody and cost the same hour twice. **A trap that is not in the
      repo is not known.** That is why it is here now.
    * When a PR changes that workflow: review it by hand or in a session, say in the PR body
      that the automated review was structurally skipped, and keep unrelated changes out of
      it so only the workflow edit goes unreviewed. A fix to this lane cannot be validated by
      the PR that introduces it — only by the next PR that leaves the workflow alone.
  * **The human invocation is the turn token.** Nothing polls and there is no label. One
    invocation, one pass, then it hands back: whatever the pass could not settle alone is
    left as an OPEN thread with a question in it, the owner answers there, and re-running
    `fix_pr.sh` starts the next pass by reading those answers. Replies therefore go into the
    threads, not only into the summary — the summary is for the human, the threads are what
    the next pass reads.

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
    assume a repo-relative log path is writable** (`scripts/fix_pr.sh` is the worked example —
    it probes `$FIX_PR_LOG_DIR`, then `logs/fix_pr`, then `$TMPDIR`, and prints its choice).
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
* **The test set is a dependency GROUP, and it is the one answer to "what do I install to
  run the suite"** (owner-approved 2026-08-11):
  `uv pip install --python .venv/bin/python --group test`, declared once in `pyproject.toml`
  under `[dependency-groups]`. ⚠ Known-good uv: uv 0.11.29 (x86_64-unknown-linux-gnu); `--group` needs PEP 735 support.
  * ⚠ **It is not yet the ONLY copy, and the other copy has already drifted.**
    `.github/workflows/claude-review.yml` carries its own hand-written `pip install` line and
    it lacks `pysbd` and `fastapi`, so **tests fail inside the review lane itself** and every
    finding from a run needing those modules was reasoned rather than executed. `ci.yml`
    (arriving with #62) carries a third copy. Both should point at this group; until they do,
    "declared once" describes the intent, not the state.
    * ⚠ **The durable statement is qualitative, so state it that way:** *the review lane cannot
      run its own suite cleanly, so any finding from a run that needed the failing modules was
      reasoned rather than executed.* ⚠ There are **two independent causes**, and fixing the
      dependency list addresses only one: the missing `pysbd`/`fastapi` (which #66 fixes), and
      the absent `.venv/bin/python` that `test_campaign_tooling.py` and `test_gate_scripts.py`
      shell out to — **5 errors that survive #66**, because a CI checkout has no repo venv.
      Do not read a green install step as a clean suite.* That survives every test anyone adds. A count does not —
      measured 2026-08-11 it was 8 on `main` and 11 on #62's tree (which adds
      fastapi-dependent tests), both correct, and it will be wrong again next week. If you do
      quote a number, name the tree; better, quote the property instead.
    * ⚠ **The fix cannot be validated by its own pull request.** `claude-review.yml` is the
      file §1's workflow-validation trap applies to: the action refuses to run when the PR's
      copy differs from the default branch and reports the job **GREEN**. So that fix lands as
      its own PR, its own review is skipped **by design**, and the proof is the *next* PR that
      leaves the workflow alone. The failure mode is a green check, not a red one.
  * ⚠ **Not an extra.** `[project.optional-dependencies]` is *additive* to
    `[project.dependencies]`, so `.[dev]` drags **torch** in — ~2 GB for one test file
    (`tests/test_gate_scripts.py`) that skips without it. A PEP 735 group installs on its own.
    Measured: `--group test` resolves 43 packages and **zero** torch; `.[dev]` resolves torch.
  * ⚠ **`numpy<2.5` in that group is load-bearing** and the list does not install without it.
    Unconstrained, uv holds numpy at the newest (**2.5.2**), which numba 0.66 excludes
    (`numpy<2.5`), and walks **two** other packages back instead of walking numpy back: numba
    to **0.53.1** (2021, no numpy ceiling in its metadata, cannot build on 3.12) and — because
    librosa 1.0.0 declares `numba>=0.61.0` and would otherwise block that — **librosa from
    1.0.0 down to 0.11.0**. Verified: unconstrained resolves
    `numpy==2.5.2 numba==0.53.1 librosa==0.11.0`. ⚠ So the pin prevents a **silent
    major-version downgrade of librosa**, not merely a build failure. The failure surfaces as a **librosa** build error that mentions
    neither numpy nor numba. ⚠ It is a CEILING tracking numba's, not a claim that numba caps
    at 2.2; raise it as numba does, and do not drop it on the grounds that numba supports
    2.2, because the install still breaks.
  * The `dev` extra stays for what it is: `pre-commit`, plus test-time imports for anyone who
    genuinely wants the project installed too. It is not the answer to the question above.
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

### 4. Changelog Maintenance Requirement

* The project changelog lives at [notes/CHANGELOG.md](notes/CHANGELOG.md). Append a detailed chronological entry describing all technical modifications, refactoring milestones, and build-system changes **after committing** the corresponding work.
* **Scope: code work only.** Changelog entries are required for source, config, and dependency-manifest changes (`matcha/`, `scripts/`, `configs/`, `tests/`, `sky/`, `vocalizer.py`, `audition/`, `Makefile`, `pyproject.toml`/`setup.py`/`environments/`). They are **not** required for docs-only commits (`notes/`, `notebooks/`, `*.md`, comments-only changes).
* Every entry must be accompanied by the short 7-character commit SHA associated with the work.
* **The changelog is append-only across a release cycle.** Do not prune, rewrite, or remove historical entries. Entries are pruned/rolled over **only** when we tag and release a new version of the overall project — at which point the released entries are collected under that version's heading and the working section is reset for the next cycle.
* New entries go at the top under the current date, following the existing `Added` / `Changed` / `Fixed` / `Removed` structure.

### 5. Code Review Execution Standards

* **Scope: code work only.** Code reviews cover the same code changes that warrant changelog entries (see §4) — source, configs, and dependency manifests. Docs-only commits are out of scope and need no review.
* **A review is a report, not a fix pass.** Assume the deliverable is the findings document alone: the reviewing agent takes on fixes only when the owner explicitly asks it to, never as a rider on the review itself.
* **The review itself never warrants a changelog entry.** Review documents live in `notes/`, and writing, replacing, or deleting one is docs-only work under §4; the changelog material is the code commits that later close the findings.
* When performing a code review, cross-reference the changelog and corresponding commits.
* Create a review document matching the format `notes/code-review-[year][month][day]-[hhmmss].md`. Begin the document with the first evaluated short commit SHA, and end with the last evaluated commit SHA.
* Determine the range of commits to review by starting with the commit immediately following the end SHA of the *previous* code review. If no prior review exists, use all commits from the previous and current day.
* Once the new code review document has been written, delete the previous one to keep only the latest review active.
* Repoint the **Latest code review** pointer in [notes/STATE.md](notes/STATE.md) to the new document (only the link target changes; the surrounding line is phrased generically) so a session can find the current review without globbing the folder.

### 6. Execute From The Repo — `/data` Holds Data

**Owner principle (2026-08-06): code executes from the repo checkout; `/data` holds what
its name implies** — datasets, checkpoints, model artifacts, venvs, training logs, service
runtime state, vendor checkouts. A byte-copy of our source under `/data` is something to
*remove*, not to manage. The full inventory, what is legitimately untracked, and the audit
behind this rule are in [notes/data-mirrors.md](notes/data-mirrors.md).

* **When a tool writes its outputs next to its own source** — the usual reason a `/data`
  copy exists at all — give it an artifact-root variable defaulting to the script's own
  directory, point that at `/data`, and delete the copy. That is the shape that satisfies
  both halves. Worked example: `SONORA_LITERT_WORK` and
  [scripts/litert_export/run.sh](scripts/litert_export/run.sh).
* **For containers, bind-mount the repo path**, not a `/data` copy of it.

Where a copy still exists, the rest of this section governs it. **The repo is
authoritative in every case.**

* **Never edit code under `/data`.** Change it in the repo, commit, then deploy. An edit made
  on `/data` has no history, no diff, no review and no changelog entry, and no way for anyone
  else to discover it happened.
* **A `/data` file that is *newer* than its tracked original is not authoritative — it is
  unreviewed.** If that edit is the one you want, commit it in the repo and redeploy; do not
  let the copy become the record.
* **Deploy explicitly.** Committing deploys nothing (GitOps retired 2026-07-22). Service
  stacks go through `AI-Lab-AMD/scripts/deploy.sh`; the training clone at `/data/repos/Sonora`
  through `deploy.sh training-code`.
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
| `training-code` → `/data/repos/Sonora` | **this repo** (ff-pull; a real checkout) | `deploy.sh training-code` |
| `dashboard` → `/data/services/dashboard` | `AI-Lab-AMD/dashboard` | `deploy.sh dashboard` |
| `stack` → the compose services | `AI-Lab-AMD` | `deploy.sh stack` |

**Deploying is idempotent and free to over-run** (2026-08-08). `deploy.sh audition` /
`dashboard` diff the target first and do **nothing** when it already matches — no copy, no
stamp rewrite, no container restart. So running a deploy at the outset of any work costs
nothing and requires no judgement about whether it is needed; that is the point, because a
deploy that always restarted a live rating app made the safe habit the expensive one.
`training-code` (`git pull --ff-only`) and `stack` (`compose up -d`) were already idempotent
by construction.

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

* An edit under `/data` has **no history, no diff, no review and no changelog entry**. A
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

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

* **Canonical root marker files → `UPPERCASE`** (`SCREAMING_SNAKE_CASE` if multi-word): `README.md`, `LICENSE`, `CONTRIBUTING.md`, `ROADMAP.md`, `AGENTS.md`. Keep this set small and curated.
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

### 4. The Record of Change — git history and the pull request

* **There is no changelog** (owner, 2026-08-11). `notes/CHANGELOG.md` was retired, along with
  the review-document cycle that cross-referenced it, because both **predate this repo having
  pull requests or review at all** and had become a third place for the same facts to drift.
  The record of a change is now, in order of authority:
  1. **the commit message** — WHY the previous state was wrong, not merely what moved;
  2. **the pull request** — title, body, and the review threads, which hold the argument and
     its resolution;
  3. **`git log`** — which needs no maintenance to stay accurate.
* ⚠ **Do not reintroduce a changelog, and do not resurrect it under another name** — a
  `notes/changes-*.md`, a "release notes" file, a running summary in `STATE.md`. The failure
  was structural, not cosmetic: a hand-maintained narrative of what changed is a copy of
  information that already exists in two authoritative places, and the copy is the one that
  goes stale. This repo has already paid for doc-vs-artifact drift repeatedly.
* **What genuinely does not fit in a commit or a PR belongs in `notes/`** as a durable
  document about the *current* state of something (`notes/STATE.md`, a design note), never as
  a dated log of past events. If you catch yourself writing "on 2026-08-11 we changed X",
  that belongs in the commit that changed X.

### 5. Code Review Standards

* **Review happens in the pull request.** `.github/workflows/claude-review.yml` reviews each
  PR and posts findings inline; §1 covers how they are resolved. There is **no review
  document and no pointer to maintain** — the reviewer bounds its own range with the
  `<!-- janis-reviewed: <sha> -->` marker it writes into each summary, so the thing §5 used to
  ask a human to track is now automatic and per-PR.
  * ⚠ **The marker bounds a range only while the branch's history is APPEND-ONLY. A rebase or
    force-push invalidates it and costs a full re-review.** Measured on this PR: **one**
    force-push invalidated **two** markers at once. The previous marker stopped being an
    ancestor of HEAD (`compare` reports *diverged*),
    and the reviewer correctly fell back to reading everything. The prompt handles that and says
    so — but the cost is real, **tidying a branch's history is not free**, and an earlier version
    of this bullet claimed there was no SHA bookkeeping at all.
* **A review is a report, not a fix pass.** This survives the retirement and applies to any
  agent asked to review anything: the deliverable is the findings. Take on fixes only when
  the owner explicitly asks, never as a rider on the review itself.
* ⚠ **REVIEW THE INSTRUCTION, NOT ONLY THE CLASSIFICATION — they fail independently, and the
  second is where the defects hide.** Six instances across four rounds on 2026-08-11: in every
  one the code decided *correctly* and the instruction attached to it was wrong or impossible.
  A remedy naming a fix that cannot address the cause; a bucket telling the reader to "score
  them first" about clips already scored; a comment claiming an override the tool never had.
  * *"Is this line true?"* is easy to read for. *"What would someone DO on reading this line?"*
    is a different question and almost never asked. Ask it of every message, comment, docstring
    and suggested remedy — including the ones a review itself writes, since a `suggestion`
    block is committed in one click and gets far less scrutiny than the finding it hangs off.
* **Scope: code work only.** Source, configs and dependency manifests. Docs-only changes need
  no review, and the review workflow posts nothing when a PR's new range touches only `*.md`
  or `notes/`. ⚠ **With exceptions that are ALWAYS in scope regardless of extension:
  `.claude/**`, `AGENTS.md`, `CLAUDE.md`.** `.claude/commands/*.md` is an executable prompt —
  it tells an agent holding push rights what to run — so it is closer to a shell script than
  to a README. The unqualified version of this sentence let #64 merge with **zero review**;
  narrowed in #66.
* **Periodic wholesale review is a different altitude, and it is NOT this.** The owner keeps a
  floating reviewer session for reading the codebase and the product direction as a whole,
  on its own cadence. Per-PR review answers *"is this diff correct?"*; that one answers
  *"is this still coherent?"* — and it must not be collapsed into the PR lane, because per-diff
  volume will always crowd out the wider read. ⚠ **Where its output lands is an OPEN
  QUESTION** (2026-08-11): the retired convention was a timestamped document in `notes/`, and
  nothing has replaced it yet. Do not invent one silently — ask.

### 5b. The doc-claims gate can stop enforcing WITHOUT going red

`scripts/gates/test_doc_claims.py` compares documented numbers against the artifacts on disk. It
has **five silent-disarm modes** — **four observed on 2026-08-11, and one reasoned from the
mechanism** (mode 2, see its own correction) — none of which was written down
anywhere until now — which is the same reason the workflow-validation trap in §1 cost the
same hour twice.

* ⚠ **A registry fact may only guard a STATIC artifact.** `audit-*/ratings.csv` is written
  **live** by the Auditions app, so a count taken from it (the 1,279 keeps, say) moves as the
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

### 5c. A pipeline stage is WIRED, or it is merely WRITTEN ABOUT

`scripts/` holds **100 non-test `.py` files** (106 tracked, less the 6 gate scripts) and
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
  on `/data` has no history, no diff, no review and no pull request, and no way for anyone
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
over every file — an empty plan means the stamp drifted and the content did not:

```bash
sudo rsync -a --delete --checksum --dry-run --itemize-changes \
  --exclude='__pycache__' --exclude='.deployed.json' --exclude='_contract/' \
  audition/app/ /data/services/audition/app/ | grep -E '^[<>]fc'
```

⚠ **THE `grep` IS PART OF THE DIAGNOSTIC, not tidying.** `-a` is `-rlptgoD`, so it itemises
differences in **time, perms, owner and group** as well as content — and `--checksum` changes
only how rsync *decides to transfer*, not what it reports. So a bare run prints
`.f..tp..... main.py` for a file whose bytes are identical and whose mtime differs, and the
empty-plan rule above reads that as drift. Measured on a synthetic pair: identical content
with a different mtime and mode → `.f..tp.....`; filtered on `^[<>]fc` → nothing. Real content
drift → `>fc.tp.....`, which the filter keeps. **`c` in the third column is the content
verdict; the rest is metadata.** An earlier version of this note omitted the filter and was
therefore wrong in the direction an operator actually hits — non-empty output.

⚠ **The excludes are not optional either**, and dropping them is the same mistake one rung down:
`.deployed.json` and `_contract/` are written by `deploy.sh` AFTER the copy and exist in no
commit, so a bare dry-run reports three deletions and reads as drift. ⚠ And `deploy.sh
audition` is the one-step version of this only on a CLEAN tree — `require_clean` refuses a
dirty one, correctly, so it is not available as a diagnostic mid-edit.

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

* An edit under `/data` has **no history, no diff, no review and no pull request**. A
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

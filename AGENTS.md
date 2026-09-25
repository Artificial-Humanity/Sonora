# AGENTS — Project Sonora (Training Repo)

Sonora trains voice actor models with Python, PyTorch and Matcha-TTS in an AMD ROCm
container. Prosodia consumes exported artifacts from the sibling `Sonora/huggingface`
checkout; its `registry.json` records lineage and `current` flags.

Before starting work, read [WORKFLOW.md](WORKFLOW.md), [PERSONA.md](PERSONA.md),
`notes/STATE.md` and `notes/todo.md`. Follow `WORKFLOW.md` for branching, review,
committing and landing. Do not restore retired workflow rules from memory or history.

`docs/` contains this repo's documentation. `notes/` is a gitignored symlink into the
private `Notes` repo. Name private notes as code paths, not Markdown links: public
clones do not have them. A checkout across the notes migration can replace or remove
the symlink without affecting `git status`. If it is absent and `Notes/Sonora` exists,
restore it from this repo's root:

```bash
ln -s ../../Notes/Sonora notes
```

## Core Stack Matrix

* **Training:** Python, PyTorch and Matcha-TTS conditional flow-matching, inside the
  AMD ROCm PyTorch Docker environment.
* **Teacher synthesis:** `scripts/stages/` uses a two-pass Gemma director for per-line
  V/A/T and register, then per-engine casting and delivery. Registers come from
  `scripts/assets/register_lexicon.json`; adapters live in
  `scripts/assets/director_skills/<engine>.md`.
* **Engine selection:** use `ref_select.ENGINE_MIX_BY_LANE` for capability vetoes,
  measured lane weights and the diversity floor. Respect `ref_select.SET_ASIDE`.
* **Direction:** `build_direction()` in `book_ingest.py` is the single source of truth
  for engine input. Never bypass it. Unknown engines are fatal.
* **Engine onboarding:** study the interface and supply a Gemma skill-file adapter
  before adding a TTS engine. Follow [docs/tts-engine-onboarding.md](docs/tts-engine-onboarding.md).
* **Clip length:** `MIN_CLIP_SECONDS` gates estimated speech in the input text, not
  render duration. Read the constant for its value; the doc-claims gate does not scan
  `AGENTS.md`.

## File Naming Conventions

* Keep canonical root markers uppercase: `README.md`, `LICENSE`, `CONTRIBUTING.md`,
  `ROADMAP.md`, `AGENTS.md`. Use `SCREAMING_SNAKE_CASE` for multi-word markers.
* Use uppercase for required-reading anchor docs, preferably one word:
  `ARCHITECTURE.md`, `STATE.md`, `WORKFLOW.md`. Uppercase means **must read**.
* Use `lowercase-kebab-case.md` for other docs and notes.
* Follow each language's source naming convention: Rust `snake_case.rs`, Swift
  `PascalCase.swift`, Kotlin `PascalCase.kt`.
* Never distinguish paths by case alone. Use exact case in references.
* Cite the surviving section when a document absorbs other briefs. Use named sections
  rather than positional references or retired filenames.

## System Operational Mandates

### 1. Role loading and the git environment

* Work as Sonya under [PERSONA.md](PERSONA.md). Keep the owner's git author identity;
  use the persona's co-author identity for your contribution.
* Keep `CLAUDE.md` as a real file containing pointers/imports to these instructions
  and the persona. Claude Code loads `CLAUDE.md`; do not rely on automatic discovery
  of `AGENTS.md`, a `CLAUDE.md` symlink, or a startup flag to load the persona.
* If a change must not land because it corrupts data, ships a known-broken training
  path or cannot be safely reverted, **do not push; take it to the owner**.
* Do not build tooling around the interim workflow without the owner's replacement
  workflow being settled.

Treat the following git settings and protections as things to check, not guarantees
made by this document:

* This repo uses one committing session. Coordinate concurrent work.
* `main` has no branch protection or pre-push hook; CI runs after a push.
* Set `pull.rebase=false` locally in a fresh checkout. Local config does not travel
  with a clone.
* Leave `push.default` unset so Git's `simple` behavior refuses a branch whose name
  differs from its upstream. Do not set it to `upstream`.
* Never run `git push origin <branch>:main`. To publish a branch under its own name,
  use `git push origin <branch>:<branch>` or set its matching upstream with `-u`.
* Cut scratch branches from a local ref, not `origin/main`.
* Use `origin/main..HEAD` for the change range. `@{push}..HEAD` does not resolve under
  this checkout's `simple` configuration.
* `git config --local` changes config shared by all worktrees. Before relying on
  `git config --worktree` for isolation, enable `extensions.worktreeConfig`; without
  it, `--worktree` also writes shared config.

### 2. Training & Troubleshooting Mandates

* Training runs in `sonora_training`; its compose definition lives in `AI-Lab-AMD`.
  Read progress with `docker logs sonora_training` or `docker logs -f sonora_training`.
* Do not assume repo-relative `logs/` is writable. On ai-lab-0 it is owned by
  `root:datashare` without group write permission; training artifacts land under `/data`.
  Probe an environment override, then a repo-relative directory, then `$TMPDIR`.
  Print the selected path. Changing ownership requires root.
* For ROCm audio decoding, use `soundfile.read(..., dtype='float32')` and convert to
  tensors manually. `torchaudio.load()` can select `torchcodec` and fail on CUDA
  libraries. See [matcha/data/text_mel_datamodule.py](matcha/data/text_mel_datamodule.py).
* For Matplotlib 3.9+, use `np.asarray(fig.canvas.buffer_rgba())[:, :, :3]`, not
  `tostring_rgb()`. See [matcha/utils/utils.py](matcha/utils/utils.py).
* For isolated NumPy build failures on Python 3.12, pre-install `Cython` and use
  `uv pip install --no-build-isolation -e .` to reuse container-native libraries.
* Keep `/data`, not recursive `data`, in the root `.gitignore` so source directories
  such as `matcha/data/` remain visible.
* After a failed container run, fix and commit in this repo, deploy explicitly through
  `AI-Lab-AMD/scripts/deploy.sh`, then relaunch explicitly from `AI-Lab-AMD`:
  `docker compose --profile training up -d sonora_training`.
* Container recreation removes `/tmp` and pip installs. Copy artifacts out promptly
  to `/data/model-training/…`, or promote approved artifacts into the sibling
  `Sonora/huggingface` registry checkout.

### 3. Python Tooling Mandate — uv

* Use `uv` for Python versions, virtual environments, dependencies and tools. New
  scripts, containers, CI steps and docs must use it from the start.
* Run **host scripts** with `.venv/bin/python scripts/…`, not `uv run`. Create the
  environment with `uv venv` and populate it with
  `uv pip install --python .venv/bin/python …`. Container scripts are unaffected.
* Install test dependencies with `uv pip install --python .venv/bin/python --group test`.
  `pyproject.toml` owns the dependency list and pin rationale.
* When touching legacy pip usage, migrate it where practical. The migration list is
  [AI-Lab-AMD/notes/cleanup-chores.md](../../AI-Lab-AMD/notes/cleanup-chores.md).
* In `rocm/pytorch` containers, bootstrap uv with its static binary, a one-time
  `pip install uv`, or `COPY --from=ghcr.io/astral-sh/uv`. Install runtime dependencies
  with `uv pip install --python /opt/venv/bin/python`.
  Use `--system` only in images whose Python is actually the system interpreter.
  Do not add other new `pip install` invocations.

### 4. The record of change

Follow [WORKFLOW.md](WORKFLOW.md) for commit requirements. Git history is the record
of change; this repo has no changelog or review tracker.

### 5. Correctness checks

* Check both a classification and the instruction attached to it. Ask what someone
  will do after reading a message, comment, docstring or suggested remedy.
* Check prose against the executable statement it describes. Distinguish a written
  rule from a mechanism that enforces it.
* Before trusting a negative result, check the command's exit status, output streams
  and input population. Use positive controls to show that the check ran.
* Treat `.claude/**`, `AGENTS.md`, `CLAUDE.md` and `WORKFLOW.md` as code: their
  instructions control actions an agent can take.

### 5b. Doc claims and verification

* `scripts/gates/test_doc_claims.py` compares documented values with artifacts.
  Register facts only for static artifacts, not live files such as `ratings.csv`.
* Scope and value patterns match **per line**. Keep the scope token and number on
  the same line. When adding or moving an enforced statement, count its matches;
  a green gate alone does not prove coverage.
* Keep registry coverage enforced by
  `tests/test_doc_claims_registry.py::test_every_fact_recognises_at_least_one_live_statement`.
  Do not replace that assertion with a manual sweep.
* When a guard's test is its evidence, run the case and inspect the fixture and
  actual behavior. A passing assertion can agree with a faulty implementation.
* State the machine's final state when interpreting evidence. Do not combine
  incompatible outcomes, such as an install failing and also installing a downgrade.
  Another reader's agreement is not verification.
* If a fix touches an input to a documented number, re-derive the number.
* When documentation, comments, docstrings or commit messages disagree with code,
  search the **whole tracked tree** for the same claim. Do not restrict the search
  to Markdown. Resolve contradictory contracts as well as the original sentence.
* Prefer pointers to constants and existing records over repeated values or history.
  If prose must quote a code constant, register it through the gate's AST-based
  `const()` reader.

### 5c. Pipeline stages and script layout

* Read [scripts/README.md](scripts/README.md) before adding files under `scripts/`.
  Use its buckets and keep bucket files exactly two levels below the repo root.
* Put checked-in script data in `scripts/assets/` and resolve paths from the repo
  root. `tests/test_asset_paths.py` checks paths derived from `__file__`.
* Declare a stage in `scripts/pipeline_manifest.py` in the same commit that wires
  it into a shell. Follow the manifest's docstring for the procedure.
  `tests/test_stage_coverage.py` checks declarations and invocations in both directions,
  including shell category coverage.
* Use `tests/test_audit_sampling.py::_invocations` to identify real calls. Do not
  count comments or `echo` recovery hints as invocations.
* Show that a new guard fails on broken cases. Include an empty-population case;
  retain `test_the_manifest_declares_the_lanes_it_is_supposed_to_cover` so an empty
  manifest cannot silently pass.

### 6. Execute From The Repo — `/data` Holds Data

* The repo is authoritative. Execute source from the checkout;
  bind-mount the repo path into containers rather than adding source copies under `/data`.
  Existing deployment targets follow §7.
* `/data` holds datasets, checkpoints, model artifacts, venvs, training logs,
  vendor checkouts and service runtime state. Keep these untracked.
* If a tool writes outputs beside its source, give it an artifact-root variable
  defaulting to its own directory. Point that variable at `/data` and remove the
  source copy. See `SONORA_LITERT_WORK` in
  [scripts/litert_export/run.sh](scripts/litert_export/run.sh).
* Never edit project source under `/data` or at another deployment target.
  Never rsync source back from a deployment target into the repo.
  If a target contains a wanted change, port it into the repo, commit and redeploy.
* Committing deploys nothing. Use `AI-Lab-AMD/scripts/deploy.sh` explicitly.
* Register a new deployed tool or target in `MIRRORS` in
  [tests/test_data_mirrors.py](tests/test_data_mirrors.py) in the same commit that
  first deploys it. Keep its nonempty-coverage assertion.

The source/deployment inventory is in `notes/data-mirrors.md` (private).

### 7. The Deploy Cycle

Before editing a tree, check whether `AI-Lab-AMD/scripts/deploy.sh` names it as a
deployment target. Git metadata does not establish that a tree is a source checkout.

| target | source | command |
|---|---|---|
| `audition` → `/data/services/audition/app` | this repo, `audition/` | `deploy.sh audition` |
| `training-code` → `/data/repos/Sonora` | this repo, rsync copy without `.git` | `deploy.sh training-code` |
| `dashboard` → `/data/services/dashboard` | `AI-Lab-AMD/dashboard` | `deploy.sh dashboard` |
| `stack` → compose services | `AI-Lab-AMD` | `deploy.sh stack` |

#### Before editing related code

From this repo's root:

```bash
../../AI-Lab-AMD/scripts/deploy.sh status
.venv/bin/python -m pytest tests/test_data_mirrors.py -q
```

Resolve `STALE`, `DIRTY`, or an unknown deployed commit before editing. Inspect
content as well as the stamp: `rsync --delete` overwrites target changes.
`test_data_mirrors.py` covers only registered paths; its audition check uses a
nonrecursive `*.py` glob and does not cover `static/`.

For a full audition content comparison, use this diagnostic dry-run:

```bash
sudo rsync -rl --delete --checksum --dry-run --itemize-changes \
  --exclude='__pycache__' --exclude='.deployed.json' --exclude='_contract/' \
  audition/app/ /data/services/audition/app/
```

Keep all three excludes and inspect unfiltered output, stderr and exit status.
An empty successful plan means content matches; it does **not** check permissions
or ownership. Non-regular files can be skipped with a stderr warning.
Do not substitute `-a` for this content diagnostic or filter its itemized output:
metadata differences add noise, while filters can hide missing files and deletions.
**Never deploy with `-rl`**; use `deploy.sh`, which preserves metadata with `-a`.

#### Deploying from a branch or linked worktree

`deploy.sh` reads `SONORA_REPO`, defaulting to `Sonora/github`. For a linked worktree,
prefix each deploy command with its root:

```bash
SONORA_REPO="$(git rev-parse --show-toplevel)" deploy.sh audition
```

Do not use `$PWD` or export `SONORA_REPO` for the session. Other tools read that
variable with different defaults; an inherited value can select the wrong export source.

Before deploying, fetch current refs and check that your tree contains the deployed
commit. Remote-tracking refs are shared across worktrees.

```bash
git fetch origin
git merge-base --is-ancestor <deployed-sha> HEAD
git log --oneline HEAD..<deployed-sha> -- audition/
```

The ancestry check must exit 0. Investigate missing history before deploying;
adapt the log's source path to the target. Comparing the deployed commit only with
`origin/main` does not establish that your working branch contains it.

After squash-merging a branch-sourced deploy, rerun the deploy to refresh its stamp.
Use the content comparison to distinguish an orphaned stamp from actual drift.

#### Edit, commit, deploy, verify

After the checks and the review/commit cycle in `WORKFLOW.md`:

```bash
../../AI-Lab-AMD/scripts/deploy.sh <target>
../../AI-Lab-AMD/scripts/deploy.sh status
.venv/bin/python -m pytest tests/test_data_mirrors.py -q
```

* Confirm the target reads `current`, and check content beyond the mirror test's scope.
* Deploys refuse a dirty tree. `ALLOW_DIRTY=1` overrides this and is recorded in the stamp.
* `audition` and `dashboard` compare content before copying; matching content needs
  no copy or restart. An orphaned stamp can still need a refresh.
* `stack` refuses during training (`ALLOW_STACK_DURING_TRAINING=1` overrides). The guard
  dates from the spin-down rule, which the owner retired on 2026-09-25; whether it stays is
  AI-Lab-AMD's decision. Do not stop the inference engines as part of a launch.
* Ship shared contracts rather than transcribing them. `deploy.sh` copies
  `matcha/delivery.py` into audition's `app/_contract/` after `rsync --delete`;
  the app refuses to start without it. Do not add a literal fallback.
* Judge staleness by ancestry of the last commit touching the target's source path,
  not by repo HEAD.

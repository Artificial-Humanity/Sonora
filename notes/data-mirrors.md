# Code on `/data` — what is backed, and the failure mode that actually bit

_Written 2026-08-06, after auditing every tool directory on ai-lab-0's `/data`._

## The short answer

**Nothing of ours on `/data` is unbacked.** Every body of code we wrote is in a
GitHub-backed repo. The concern that prompted this audit — that the LiteRT export
toolchain lived only on `/data`, one disk failure from gone — turned out to be false: it
was migrated into `Sonora/github/scripts/litert_export/` on 2026-07-22.

**The real problem was the opposite shape, and worse for being invisible.** `/data` holds
working *copies*, and copies drift. Two had:

| file | drift |
|---|---|
| `convert_vat.py` | **three weeks stale.** The repo gained a `detect_vat_dim` seam guard on 2026-08-04 — recorded in `todo.md` as landed — and the harness that actually runs never received it |
| `README.md` | still documented a bare-`pip` install, predating the uv standard (AGENTS.md §3) |

Backup was never the risk. Divergence was, and nothing detected it: both directories look
healthy, the scripts run, and the only symptom is that **a fix you believe shipped did
not**. A guard nobody has watched fail is a guess; a guard that is not installed is not a
guard at all.

## The inventory

Ours, and where the source of truth lives:

| `/data` path | tracked in | status |
|---|---|---|
| `toolchain/litert-conversion/*.py`, `README.md` | `Sonora/github/scripts/litert_export/` | **copy RETIRED 2026-08-06** — executes from the repo now, see below |
| `toolchain/teacher-audition/render_*.{py,sh}`, `coach_dia_threat.py` | `Sonora/github/scripts/synthesis/teacher_audition/` | in sync |
| `services/audition/app/main.py` | `AI-Lab-AMD/audition/app/` | in sync |
| `services/dashboard/index.html`, `scripts/update_status.sh` | `AI-Lab-AMD/dashboard/` | in sync |

Not ours — vendor checkouts and service state, correctly untracked:

- `toolchain/Zonos` — upstream clone, has its own remote
- `services/{comfyui,unsloth,llama-factory,oobabooga,open-webui,openhands,ollama,mlflow,portainer,skypilot,lemonade}` — vendor code and runtime state
- `toolchain/{dnsmos,silero}`, `libtensorflowlite_c.so` — downloaded model/library artifacts
- `repos/Sonora` — the training **deploy clone**, which tracks the real remote

Also correctly on `/data` and correctly untracked: the litert working directory itself —
`.venv` (the bulk of its 7.2 GB), `artifacts*/`, checkpoints and logs. Those belong on the
data drive, not in git.

## The rule

Ratified as **[AGENTS.md](../AGENTS.md) §6**. In short:

**The repo is authoritative. `/data` is a working copy, always.**

A `/data` file that is *newer* than its tracked original is not authoritative — it is
unreviewed. It has no history, no diff, no pull request, and no way for anyone else to
discover it changed. If an edit made on `/data` is the right one, commit it in the repo
and redeploy; do not let the copy become the record.

`tests/test_data_mirrors.py` enforces it. It compares every tracked file against its
`/data` counterpart and fails on any difference, skipping cleanly on machines without
`/data`. It also asserts that it found at least 15 pairs to check, because a layout change
that emptied the list would otherwise turn the whole gate into a silent pass.

§6 additionally requires that a new `/data`-deployed tool be added to that test's `MIRRORS`
table **in the same commit that deploys it** — the gate only covers what it is told about,
so an unlisted tool is exactly as exposed as everything was before this note existed.

## The litert copy is gone, not policed

Owner principle, 2026-08-06: *code executes from the repo; `/data` is used for what its
name implies.* The harness was the case that made this awkward, and it is worth recording
why, because the shape will recur.

It does not run in a container at all — it runs on the host against a 6.6 GB venv sitting
beside its outputs, so "bind-mount the repo" was never even the right mechanism. The copy
existed for one reason: every script derived its paths from
`HERE = os.path.dirname(os.path.abspath(__file__))`, so the checkpoints it reads and the
~400 MB of `.tflite` / `.wav` / `artifacts*/` it writes all landed next to the source.
Running from the checkout would have dumped every bit of that into the working tree.

Splitting the two retired the copy. `SONORA_LITERT_WORK` (defaulting to `HERE`, so no
existing invocation changed) names the data root, and
[`scripts/litert_export/run.sh`](../scripts/litert_export/run.sh) runs a script from the
repo with the work dir, the model repo and the harness interpreter all wired up. `/data`
keeps the venv, checkpoints, graphs and artifacts — exactly what it is for.

`tests/test_data_mirrors.py::test_litert_harness_has_no_code_copy` guards this directly,
and is inverted from the others: the remaining mirrors are checked for *agreement*, this
one for *absence*.

## What is still worth doing

Three copies remain, each for a reason worth weighing rather than overriding:

- **`audition`** bind-mounts `/data/services/audition/app:/app:ro`, and the compose comment
  is explicit that this is the deploy copy and not the dev tree — the same reasoning that
  gives `sonora_training` a deploy clone. Pointing the mount at the repo would satisfy §6
  and would also serve half-edited working-tree state to a live app.
- **`dashboard`** is Caddy's `root`, not a mount, and `/etc/caddy/Caddyfile` is itself a
  copy of the one in `AI-Lab-AMD`.
- **`/data/repos/Sonora`** is the training deploy clone and the **standing exception**:
  `restart: unless-stopped` re-reads the mount, so a crash mid-run would resume a long
  unattended GPU job from whatever the tree happened to hold. Being a real git checkout, it
  also always knows its own revision — which a byte copy never does.

For those three the honest upgrade is not removal but *provenance*: have `deploy.sh` stamp
each target with the commit SHA it deployed and whether the tree was clean, so the question
becomes "which revision is running?" rather than "do these bytes still match?".

Related: the training deploy clone at `/data/repos/Sonora` has its own version of this
failure — a `deploy.sh training-code` fast-forward pull cannot cross a history rewrite, and
went dead for two weeks in July without anyone noticing. Same lesson, different mechanism:
**a copy that stops updating looks exactly like a copy that is up to date.**

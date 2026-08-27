# Training Operations — Runbook (ai-lab-0)

_The operational "how" of model training: what runs where, how to launch/observe/stop/resume,
and the standing verification gates. Model **selection** rationale lives in
[model-decisions.md § The base model — Matcha, reaffirmed](../docs/model-decisions.md); machine provisioning in
[AI-Lab-AMD/notes/machine-setup.md](../../../AI-Lab-AMD/notes/machine-setup.md); stack config is
[AI-Lab-AMD/docker-compose.yml](../../../AI-Lab-AMD/docker-compose.yml). What to run **next** and
why is [quality-gap-plan.md](quality-gap-plan.md). Created 2026-07-14 during the de-risk phase
start; keep the "Runs to date" table honest._

_Last updated: 2026-08-06._

## Runs to date

**Checkpoint standing lives in [STATE.md § Checkpoint lineage](STATE.md)** — one table, not
restated here. What belongs in this file is how to run one, and what has bitten.

### Reading a Matcha training curve (learned on vat3c)

**Settled at ~0.62 s/step** (measured steps 2099–2349), against the vat3 baseline's 2.11
s/step whole-run average — roughly 3.4× faster, on ~950 steps/epoch. Early steps look
catastrophic and are not: the MIOpen conv-tuning ramp took the baseline 37 → 22 → 16 →
11 → 3.4 s/step over its first ~450 steps. **Never compare an early window against a
whole-run average** — doing that produced a false alarm on this launch. Compare the same
step window on the previous run via MLflow `metrics/get-history`.

**The vat3c val curve sits far above train, and that was an artefact — fixed 2026-08-06.**
Epoch 1: train 2.012, val 3.425, with `diff_loss` 0.646 train vs 2.069 val while
`dur_loss` and `prior_loss` matched to three decimals. That was E-M5, not overfitting:
the logged diff tensor was summed unmasked, so it carried a floor proportional to each
batch's padding, and only train batches are length-bucketed. `compute_loss` masks now
(`matcha/models/components/flow_matching.py`, regression test
`tests/test_training_seams.py`), which removes the floor from both curves and the gap
with it. **Gradients never changed — but the logged numbers did, so no curve from before
that commit can be compared against one after it.** Read vat3c's recorded values as
history, not as a baseline.

**Progress never reaches stdout** — Lightning's rich progress bar writes `\r`, so
`docker logs` shows startup then silence. Read MLflow: `mlflow-local` on :5000, and the
API needs `-H 'Host: mlflow.ai-lab-0.mcfarlin.family'` or it refuses as DNS rebinding.

### Checkpoint and resume settings — CHANGED 2026-08-06, read before quoting an old note

The `configs/callbacks/default.yaml` shipping default is still `every_n_epochs: 100`, which
means **nothing lands on disk until epoch 99**, `save_last` included — a crash at epoch 80
loses the run. That is what the vat3 and the first vat3c launches inherited, and it is what
made the NIC fault cost 4.5 h instead of 40 minutes.

The compose command now **overrides it on the CLI**, so a run launched through
`docker compose --profile training up -d sonora_training` gets:

| override | value | why |
|---|---|---|
| `callbacks.model_checkpoint.every_n_epochs` | **10** | ~635 s/epoch, so worst-case loss is ~1.8 h, not the whole run |
| `callbacks.model_checkpoint.save_top_k` | **-1** | keep every checkpoint; the default 10 evicts by epoch, and we select by ear, not by `monitor` |
| `ckpt_path` | `/data/model-training/sonora/resume.ckpt` | a symlink the command re-points at the newest `checkpoint_epoch=*.ckpt` **of `$SONORA_EXPERIMENT`**, falling back to `$SONORA_WARMSTART` — so a restart resumes instead of starting over, and no run inherits another's weights |

Two consequences worth holding on to. **A run launched any other way still gets 100** — the
override lives in the compose `command:`, not in the repo config. And **the prep chain is now
non-fatal**: it was `apt-get update && … && python3 -m matcha.train`, so an unreachable network
aborted the chain before training started while `restart: unless-stopped` retried forever. It is
braced and `|| echo`'d now. *A hardware fault is the event that finds every `&&` in a container
command.*

### How to launch a run (changed 2026-08-06 — there is no longer a queued run)

```
SONORA_EXPERIMENT=<name> docker compose -f AI-Lab-AMD/docker-compose.yml \
    --profile training up -d sonora_training
```

`SONORA_EXPERIMENT` has **no default**, and unset the container prints why and idles
rather than training. Auto-resume is **scoped to the named experiment's own run dirs**, so
a run can only ever resume itself.

⚠ **`SONORA_WARMSTART` has NO DEFAULT** (2026-08-09, `AI-Lab-AMD a29b982`) — a fresh run
with it unset prints why and idles. It used to default to `vat3_ep099`, described here as
"the best checkpoint in the lineage", which stopped being true when ep019 was selected and
could not know it. **A default may encode a fact, never a judgement.** The current base is
in [STATE.md § Checkpoint lineage](STATE.md); the launcher points at that record rather than
copying it, so it cannot go stale the same way twice.

**A concluded run cannot be silently continued.** The launcher refuses any experiment whose
log dir holds a `RETIRED.md` or `SELECTED.md` verdict file — because auto-resume takes the
NEWEST checkpoint, which is not necessarily the one the verdict selected (for
`vat5_finetune` the newest is ep039, inside the degraded region, not the ep019 that was
picked). It also refuses `vat3c_finetune` by name. To build on a concluded run, choose a
**new** `SONORA_EXPERIMENT` and pass the selected checkpoint as `SONORA_WARMSTART`.

**Per-epoch checkpointing is forced at launch**, not by the experiment config: the compose
command appends `callbacks.model_checkpoint.every_n_epochs=1` and `save_top_k=-1`. The
repo config default was 100 until 2026-08-09 — longer than this lineage converges in — so a
run launched any other way retained only `last.ckpt` and the per-epoch checkpoints the
holdout methodology reads did not exist. Both now say 1.

This replaced a `command:` that hardcoded `experiment=vat3c_finetune` *and*, separately,
globbed that experiment's checkpoints for a resume target. Both halves were traps the
moment vat3c was retired: starting the container resumed a checkpoint we had just decided
to throw away, and — the quiet one — because the glob and the experiment name were
independent literals, launching **any successor run would silently have warm-started it
from the retired ep099**. The container also refuses `SONORA_EXPERIMENT=vat3c_finetune`
by name. Retirement rationale: [quality-gap-plan.md § Phase 0](quality-gap-plan.md), gate 0a.

The idle-instead-of-exit is deliberate: `restart: unless-stopped` turns a fast exit into
the 2,077-restart crash loop of 2026-08-04.

### One more thing bit on the vat3c launch

1. **The compose `command` was the queued run** — and `compose up -d sonora_training`
   started training immediately. Bringing the container up to inspect it launched the
   stale `vat3_finetune`/v2 run; it was caught during `apt-get` and stopped before a step
   ran. *Closed by the change above:* there is nothing queued to start by accident now.
2. **The corpus was never bound into the container.** Data configs name filelists
   relative to the repo root (`data/<corpus>/train_op.txt`), which resolves in the dev
   tree through a `data ->` symlink. The deploy clone has no symlink — it is untracked,
   so it does not survive a clone — and nothing bound the directory in its place. First
   launch died at datamodule setup with `FileNotFoundError`. Fixed by binding
   `/data/model-training/sonora/data:/workspace/data`, the same treatment `logs` already
   had for the same reason (`AI-Lab-AMD` `5209a20`).

Related: the **deploy clone had been stranded since 2026-07-22**. `/data/repos/Sonora`
sat at `3893885` — pre-flat-repo-migration history, 57 commits "ahead" (all
patch-equivalent to upstream, `git cherry` confirms) and 152 behind — so
`deploy.sh training-code` had been failing its fast-forward for two weeks and silently
deploying nothing. Reset to `origin/main` 2026-08-04; the old head is tagged
`pre-reset-20260804` in that clone. **A clone that cannot fast-forward is a clone that
is not deploying** — check `deploy.sh` output rather than assuming it ran.

**No run is queued.** There is no preset "next container to launch," and what *should* run next
is a sequencing question rather than a mechanical one — it is decided in
[quality-gap-plan.md](quality-gap-plan.md), not here. Two things this runbook still owes:

1. **Promotion is an open owner call, not a backlog item.** All promoted checkpoints
   (`Sonora/huggingface/vocoder-24k-hifigan/`, `derisk-energy-24k/`, `vat3-24k/`) are staged
   locally and not pushed to the public HF repo. Same question for each: they are validated
   *components* (a standalone vocoder; one VAT channel of three), not the shippable directable
   actor, so publishing sets a registry precedent worth a deliberate call.
2. **The formal §7 write-up of the de-risk verdict is still open.** Numbers are in
   `Sonora/huggingface/derisk-energy-24k/eval_report.json`; the table above plus the promotion
   commit are the only record.

The export lane is adapted but **not re-run against vat3c**: `convert_vat.py` converts the
derisk checkpoint + 24k vocoder to the split-graph lane (graph inputs `spk(1,64)` host-lookup
vector + `vat(1,3,T)`; decoder `t_emb` is 224-dim, not 160 — multi-speaker widens the U-Net) at
per-graph corr 1.000000, e2e waveform corr ≥ 0.9993, energy monotonic through TFLite, all graphs
GPU-clean. Shapes stay 256/512 pending the
[model-decisions.md § The size target](../docs/model-decisions.md) ceiling options. **Re-run it after any
fine-tune**, and note the lane's gates currently *report* rather than *refuse*
([todo.md §2](todo.md)).

**Build note (2026-07-15, kept for future watchers):** the derisk_energy watcher
(`render_vat_sweep.py` + `derisk_gate_watch.sh` + postprocess + timer) was built and armed before
the first checkpoint landed (`every_n_epochs: 100` gave the runway) — details in §Stop signal
below. Its plumbing test against the zero-init warmstart caught (and fixed, Sonora `16c1183`) a
tie-handling bug in the eval harness's Spearman that made an *inert* control channel score a
spurious ρ=1.0 PASS — worth remembering: **test every gate against a checkpoint that SHOULD fail
it.**

## Topology

Three training-related services in the compose stack, all **profile-gated** so an explicit
`deploy.sh` run or a plain `up -d` never starts a GPU run. (GitOps was **retired 2026-07-22** —
pushes deploy nothing; deploys are explicit via `AI-Lab-AMD/scripts/deploy.sh`.)

| Service | Profile | Launch |
|---|---|---|
| `sonora_training` (Matcha acoustic) | `training` | `SONORA_EXPERIMENT=<name> docker compose -f AI-Lab-AMD/docker-compose.yml --profile training up -d sonora_training` — idles with a message if the variable is unset |
| `vocoder_training` (HiFi-GAN) | `vocoder-training` | `... --profile vocoder-training up -d vocoder_training` |
| `sonora_vocalizer` (inference/dev, CPU) | — (auto) | part of the normal stack |

All use `rocm/pytorch:latest`, `/dev/kfd` + `/dev/dri`, and bootstrap deps via `uv pip install
--python /opt/venv/bin/python` in the container command ([AGENTS.md](../AGENTS.md) §3 — uv, and
`--python`, never `--system`). **Stop** a run with
`docker stop <container>` (or `compose stop`); both trainers checkpoint and resume cleanly.

## Locations

| What | Where |
|---|---|
| Datasets (license classes: `Sonora/configs/data_licenses.yaml`) | `/data/model-training/datasets/` (`LJSpeech-1.1`, `LibriTTS_R/train-clean-100`, `expresso` — NC, de-risk only). Full inventory + state: [training-sources.md](training-sources.md) |
| Matcha training logs/checkpoints | `/data/model-training/sonora/logs/train/<experiment>/runs/<stamp>/checkpoints/` (bound over `Sonora/logs`) |
| Warm-start checkpoints | `/data/model-training/sonora/warmstart/` (`matcha_vctk.ckpt` donor — 22.05 kHz VCTK, so warm-starting from it is a **retrain**, not a fine-tune; `derisk_energy_init.ckpt` / `vat3c_init.ckpt` built by `scripts/lib/make_warmstart.py`) |
| Auto-resume symlink | `/data/model-training/sonora/resume.ckpt` — re-pointed by the compose command at every start |
| Vocoder workspace | `/data/model-training/vocoder/` (`hifi-gan/` clone with local patches, `filelist_{train,val}.txt`, `cp_hifigan_24k/` checkpoints, `copysynth*/` gate reports, `pretrained/` originals) |
| Derived VAT filelists | `Sonora/data/libritts_r_vat_v3c/` (`{train,val}_op.txt`, `speakers.json`, `derivation_report.json`, `mel_statistics.json`) — and the `_v1`/`_v2`/`_v3`/`_v3b` siblings. ⚠ `Sonora/data` is an untracked symlink; a deploy clone needs the directory bound explicitly |
| Measurement harnesses | `/data/model-training/sonora/eval_phoneme_fix/` (paired per-clip scoring, blind A/B, copy-synthesis — the 2026-08-06 diagnostics) |

## Observability

* **MLflow** — `https://mlflow.ai-lab-0.mcfarlin.family`. Matcha runs log natively
  (`logger=mlflow` in the compose command). The vocoder logs via a local patch to the hifi-gan
  clone's `train.py` (experiment **`hifigan_24k`**: `gen_loss_total`, `mel_spec_error` every
  500 steps, `val_mel_spec_error` per validation) — active when `MLFLOW_TRACKING_URI` is set in
  the service env.
* **Tensorboard events** (vocoder) — `/data/model-training/vocoder/cp_hifigan_24k/logs/`.
* **Convergence watcher** (vocoder — retired with the run) — MLflow experiment
  **`hifigan_24k_gate`**: one run per checkpoint with `gate_mel_l1`, `gate_wer`,
  `gate_mel_delta_pct`, `gate_converged` (see §Stop signal). History:
  `/data/model-training/vocoder/gate_watch/history.jsonl`; journal:
  `journalctl -u vocoder-gate-watch`.
* **Convergence watcher** (derisk_energy — retired with the run) — MLflow experiment **`derisk_energy_gate`**: one run
  per epoch checkpoint with `gate_rho_min_abs`, `gate_leakage_max`, `gate_wer_delta_max`,
  `gate_converged` (see §Stop signal). History:
  `/data/model-training/sonora/derisk_gate_watch/history.jsonl`; journal:
  `journalctl -u derisk-gate-watch`; sweep WAVs (for listening) under
  `/data/model-training/sonora/derisk_eval/`.
* **Stdout** — `docker logs <container>`. `vocoder_training` sets `PYTHONUNBUFFERED=1`
  (upstream hifi-gan prints were invisible for hours behind Python's block buffer before this);
  matcha logs through Lightning and flushes fine without it.

## Resume semantics

* **HiFi-GAN:** auto-resumes from the newest `g_*`/`do_*` pair in `--checkpoint_path`. Warm
  start = drop donor files in that dir (that's how UNIVERSAL_V1 @ step 2,500,000 was seeded).
  Container recreation is therefore cheap: it re-installs deps and resumes the latest
  checkpoint; only steps since the last save are lost.
* **Matcha:** Lightning resume via `ckpt_path=`. For **shape-changing** warm starts (new
  speaker table, new FiLM tensors) build the init checkpoint first with
  `Sonora/scripts/lib/make_warmstart.py` (loads donor `strict=False`, drops shape mismatches,
  asserts only expected-fresh tensors, saves a resumable ckpt).

## Standing gates (run these, don't trust vibes)

| Gate | Script (Sonora repo) | Verdict criteria |
|---|---|---|
| Vocoder copy-synthesis | `scripts/tools/vocoder_copysynthesis.py --checkpoint <g_XXXXXXX>` | mel-L1 + WER vs the ASR floor on the same clips. **2026-07-14:** untuned baseline 0.659 / 0.323; ckpt 2,505,000 → 0.246 / 0.106; floor 0.064. Converged ≈ WER at floor + mel-L1 plateau (<5% between checkpoints) |
| Directability / §7 verdict | `scripts/tools/eval_harness.py` (manifest-driven) | pre-registered: Spearman ρ ≥ 0.9, ECAPA leakage ≤ 0.2 (vs real inter-speaker gap), WER Δ ≤ +0.10. **3-channel since 2026-07-16:** produced measures tension→phonation composite, valence→EIV head; plus cross-channel independence (X-sweep moves Y's measure ≤ 0.5× Y's own sweep; groups `clip::channel` via `render_vat_sweep.py --channels`). Should-fail tested against the derisk ckpt: energy PASS, inert V/T + independence correctly FAIL (`/data/model-training/sonora/gate3_shouldfail/report.json`) |
| Warm-start identity | `scripts/gates/test_vat_identity.py` | bit-identical synthesise at init for vat = 0/None/hot |
| Export (FiLM ops) | `scripts/gates/test_film_export_gate.py` | litert-torch conversion GPU-clean, corr ≈ 1.0 |
| Export (checkpoints) | litert-conversion harness (`/data/toolchain/litert-conversion/`; `convert_final.py` = 22.05k baseline-ljspeech-22k, **`convert_vat.py` = 24k/multi-speaker/VAT**) | per-graph corr ≈ 1.0 — re-run after ANY fine-tune. ~~FiLM graphs need the `vat` wrapper input (open)~~ done 2026-07-16: `spk` + `vat` inputs wired, energy-monotonicity check included |

## Stop signal — automated convergence watcher (STANDARD for long fine-tunes)

**Policy (2026-07-14): no unbounded training run starts without a stop signal in place.** A
training loop that never terminates on its own (HiFi-GAN runs to `training_epochs=3100` ≈
forever) gets a watcher that evaluates each new checkpoint against the run's pre-registered
convergence criterion and *surfaces* the verdict. The watcher never stops the trainer itself —
convergence flips a flag; a human listens/inspects before stopping and promoting.

**Notifier (added 2026-07-16, owner call — part of the standard):** on the FIRST flip to
CONVERGED (marker file did not previously exist), the gate postprocess sends a one-shot
**ntfy.sh push** to the owner's phone (`notify_ntfy()` in both `*_gate_postprocess.py`).
Topic lives in `/etc/ai-lab/ntfy.env` (root-only dir; the service units inject it via
`EnvironmentFile=` since the watchers run as `lmcfarlin`). Push failure never breaks the gate
verdict (logged warning only). Tested live 2026-07-16. Future watchers copy this wiring —
"watcher wired before launch" includes the notifier.

**Implementation (vocoder fine-tune, the template — retired 2026-07-15 after CONVERGED):**

| Piece | What |
|---|---|
| `AI-Lab-AMD/scripts/vocoder_gate_watch.sh` | Timer-driven: newest `g_*` newer than `gate_watch/last_step` → runs the copy-synthesis gate in a **throwaway CPU-only container** (no `/dev/kfd`; training keeps the GPU) → postprocess. All paths/knobs env-overridable. |
| `AI-Lab-AMD/scripts/vocoder_gate_postprocess.py` | Applies the rule **WER ≤ ASR floor + margin AND \|Δ mel-L1\| < plateau%** (defaults 0.064 + 0.02, 5% — `GATE_ASR_FLOOR`, `GATE_WER_MARGIN`, `GATE_MEL_PLATEAU_PCT`); appends `history.jsonl`; logs to MLflow `hifigan_24k_gate`; on convergence writes `gate_watch/CONVERGED` + a loud journal line. |
| `AI-Lab-AMD/scripts/vocoder-gate-watch.{service,timer}` | systemd oneshot + half-hourly timer (`Persistent=true`), `User=lmcfarlin` (docker + datashare groups). Checkpoints land every ~7 h, so worst-case detection lag ≈ 7%. |

**Implementation (derisk_energy acoustic run — built 2026-07-15 from the template):**

| Piece | What |
|---|---|
| `Sonora/scripts/tools/render_vat_sweep.py` | The generation half (lives in the training repo — it knows the model): CPU-loads the acoustic ckpt + the promoted 24k HiFi-GAN (`g_02510000`), renders val clips at energy ∈ {−1, −0.5, 0, +0.5, +1} (fixed seed; sweep rows differ only in vat), writes WAVs + `manifest.jsonl` + `speaker_refs.txt` + `render_meta.json`. |
| `AI-Lab-AMD/scripts/derisk_gate_watch.sh` | Timer-driven: newest `checkpoint_epoch=*.ckpt` under `logs/train/derisk_energy` newer than `derisk_gate_watch/last_ckpt` → render sweep + `eval_harness.py` in a **throwaway CPU-only container** → postprocess. Repo mounts **rw** (unlike the vocoder gate): the render imports matcha, so the container does the `Cython + -e .` install dance. Sweep WAVs land in `/data/model-training/sonora/derisk_eval/<runstamp>_epochNNN/`. |
| `AI-Lab-AMD/scripts/derisk_gate_postprocess.py` | Aggregates the harness's per-group verdicts: **CONVERGED = every sweep group passes \|ρ\| ≥ 0.9 AND leakage ≤ 0.2 AND WER Δ ≤ +0.10** (the pre-registered §7 thresholds, applied by the harness itself); appends `history.jsonl`; logs to MLflow **`derisk_energy_gate`** (`gate_rho_min_abs`, `gate_leakage_max`, `gate_wer_delta_max`, `gate_converged`); on convergence writes `derisk_gate_watch/CONVERGED`. |
| `AI-Lab-AMD/scripts/derisk-gate-watch.{service,timer}` | systemd oneshot + half-hourly timer (`*:11/30`, offset from the vocoder watcher's `:04/30`), `User=lmcfarlin`, `TimeoutStartSec=120min` (render + whisper + ECAPA + install dance). Checkpoints land every 100 epochs (`every_n_epochs: 100`), so a 30 min poll is generous. |

State/watch dir: `/data/model-training/sonora/derisk_gate_watch/` (`last_ckpt`, `history.jsonl`,
`CONVERGED`). ECAPA model cache persists at `/data/model-training/sonora/ecapa_cache`. Same
checking order and human-in-the-loop rule as the vocoder watcher: on CONVERGED, **listen to the
sweep renders** (does more energy actually sound more energetic, same voice?), then stop
`sonora_training` and run the §7 write-up.

**The signal, in checking order:** ① `gate_watch/CONVERGED` marker file (also the loud line in
`journalctl -u vocoder-gate-watch`) → ② `gate_converged` metric in MLflow `hifigan_24k_gate`
(the remote/iPad-visible view) → ③ trend in `gate_watch/history.jsonl`.

**On CONVERGED:** listen to the A/B pairs in `copysynth_<step>/` (the criterion is necessary,
not sufficient — artifacts hide from mel-L1), then `docker stop vocoder_training`, promote per
chore #7 flow, run the export gate, and launch the queued acoustic run.

**Measurement-noise note (2026-07-14):** re-running the gate on the *same* checkpoint
(g_02505000) gave WER 0.106 then 0.078 — whisper int8-CPU on n=8 clips has ~±0.03 noise, and
mel-L1 reproduced exactly (0.2459). So WER is already effectively at the floor and the stop
decision will hinge on the **mel plateau**, which is the reproducible metric. If WER ever needs
to carry decision weight, raise `--n` in the gate invocation rather than tightening the margin.

**Install/refresh (deploy = copy, like the Caddyfile; swap `vocoder`↔`derisk` as needed):**
```
sudo cp AI-Lab-AMD/scripts/derisk_gate_watch.sh AI-Lab-AMD/scripts/derisk_gate_postprocess.py /usr/local/bin/
sudo cp AI-Lab-AMD/scripts/derisk-gate-watch.service AI-Lab-AMD/scripts/derisk-gate-watch.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now derisk-gate-watch.timer
```
Disable with `sudo systemctl disable --now <name>-gate-watch.timer` when the run ends (the
vocoder one was disabled 2026-07-15 when its run stopped).

**Implementation (vat7 / rung 3 acoustic run — built 2026-08-26, IN PLACE AND ENABLED):**

| Piece | What |
|---|---|
| `AI-Lab-AMD/scripts/vat7_gate_watch.sh` | Timer-driven: newest `checkpoint_epoch=*.ckpt` under `logs/train/vat7_finetune` newer than `vat7_gate_watch/last_ckpt` → scores it against the FROZEN dev-clean holdout with `scripts/stages/score_holdout.py` in a **throwaway CPU-only container** → postprocess. ⚠ Carries its own `docker run` rather than reusing `score_holdout.sh`, which passes `/dev/kfd`: that wrapper is for interactive scoring **between** runs and its own header warns against a busy card. Repo mounted rw for the Cython + `-e .` dance. |
| `AI-Lab-AMD/scripts/vat7_gate_postprocess.py` | **CONVERGED = holdout `total` fails to beat the running best by >`VAT7_PLATEAU_PCT` for `VAT7_PATIENCE` consecutive checkpoints** (defaults **0.25%** and **3**); appends `history.jsonl`; logs to MLflow **`vat7_gate`**; writes `vat7_gate_watch/CONVERGED` + one-shot ntfy on the first flip. |
| `AI-Lab-AMD/scripts/vat7-gate-watch.{service,timer}` | oneshot + half-hourly at `*:18/30`, offset from the vocoder (`:04/30`) and derisk (`:11/30`) watchers. `User=lmcfarlin`, `TimeoutStartSec=180min`. |

⚠ **Both thresholds are DERIVED FROM MEASUREMENTS, not chosen.** `0.25%`: vat6's holdout
`total` spanned **1.7823–1.7921 across its entire run** — a 0.5% band — so a looser threshold
fires on the first checkpoint of any run. `patience 3`: v5 gave **100% of its gain in epochs
0–9 and epochs 10–39 were net worse**, so when the tail actively hurts, three flat checkpoints
is enough to act on.

⚠⚠ **IT IS A STOP SIGNAL AND NOT A SELECTION, and the code and the push both say so.** On vat6
four instruments gave four answers and diff/mel loss may be *anti-correlated* with naturalness
— so a plateau in this number is a statement about the **loss**. Choosing a checkpoint stays an
ear job. ⚠ It uses the holdout rather than `loss/val_epoch` because val is contaminated and is
not a generalization measure; the holdout is the SSOT and is valid **relatively**, which is all
a plateau detector asks. The objection *"never diagnose weaknesses on the holdout"* is answered
in the module docstring rather than ignored: nothing here feeds back into training, so there is
no gradient to overfit with — **and that answer stops holding the day this number steers what
data gets collected.**

**Verified before enabling** (2026-08-26): the real vat6 holdout report replayed through the
verdict converges at the third flat checkpoint, matching that run's recorded flat basin; a run
improving 1%/checkpoint never converges; improving-then-flat converges after exactly three flat
checkpoints; a history shorter than `patience` cannot read as a plateau; the service runs and
exits 0 with *"watcher is in place and idle"* before the run exists; and the ntfy push was sent
live **through the unit's own `EnvironmentFile` injection**. ⚠ `/etc/ai-lab/ntfy.env` is mode
`0640 root:lmcfarlin`, which **reads as group-readable and is not** — `/etc/ai-lab` is
`drwx------ root root`, so traversal is refused and a plain `open()` as `lmcfarlin` fails. The
unit's `EnvironmentFile=` is the only reason the push works.

**Adapting to a future run:** pre-register the criterion first (as the copy-synthesis gate and
the §7 eval-harness verdicts were), point the watch script at the run's checkpoint dir + gate
script (env overrides or a sibling script), pick thresholds, seed the watch dir with any
already-evaluated checkpoint, enable the timer. Two implementations now exist to crib from: the
vocoder one (self-contained gate script, ro mount) and the derisk one (render + harness split,
rw mount for the matcha install).

## Footguns (learned the hard way)

1. **Container recreation wipes the pip layer** — deps reinstall via the service command, but
   anything installed ad hoc inside a running container is gone (STATE 2026-07-12 note).
2. **ROCm first minutes look hung** — MIOpen solver search on a fresh cache can run 30+ min
   with only stderr warnings; check `docker top` CPU accumulation, not stdout. File evidence
   (checkpoints, events size) beats buffered logs.
3. **License wall** — training refuses undeclared/NC data at datamodule setup. NC de-risk runs
   need `SONORA_LICENSE_WALL=derisk` and are TAINTED (never promote artifacts).
4. **One GPU** — never run both trainers at once; the queued run waits.
5. **Filelists are derived artifacts** — regenerate with `scripts/tools/phonemize_filelist.py` /
   `scripts/lib/derive_vat_corpus.py`; don't hand-edit.
6. **The `/data/repos/Sonora` deploy clone is a bare `git clone`** — it has none of the
   machine-local derived data products that only exist in the dev tree's untracked `data/`
   subdirs (e.g. `data/libritts_r_vat/`, chore #11's motivating bug's sibling). Hit this
   2026-07-15 launching `derisk_energy`: copied `Sonora/data/libritts_r_vat/` from the dev tree
   into the deploy clone rather than regenerating (same commit, same source data, cheaper). Check
   for this class of gap whenever a training command references `data/` and the deploy clone is
   freshly (re)cloned.

7. **Spin down ALL inference before any training run** — the Gemma director, every `synth_*`
   renderer, and the Vocalizer. Standing ai-lab-0 rule, not a courtesy: they share the one GPU.
8. **`qc_gate.py` needs Python ≤3.12** — newer hosts resolve a numba that refuses to build.
   The repo `.venv` is 3.11: run it as `.venv/bin/python scripts/stages/qc_gate.py …`
   (never `uv run` on the host — AGENTS §3).
9. **Renders run as ai-mgr (105:109), not root**, in throwaway `rocm/pytorch` containers via
   `scripts/container_as_ai_mgr.sh` + `umask 002`. MIOpen's find-db must be **owned**
   by that user — chmod checks ownership, not write bits — hence the separate `miopen-ai-mgr`
   cache. See the gotchas section of [tts-engine-onboarding.md](../docs/tts-engine-onboarding.md).

## Cross-references

[quality-gap-plan.md](quality-gap-plan.md) (what to run next) ·
[training-sources.md](training-sources.md) (what it trains on) ·
[model-decisions.md § Sample rate](../docs/model-decisions.md) ·
[vat-channels.md](../docs/vat-channels.md) · [STATE.md](STATE.md) ·
[next-steps §B](../../../Prosodia/notes/next-steps.md) ·
[AI-Lab-AMD/notes/machine-setup.md](../../../AI-Lab-AMD/notes/machine-setup.md)

## Registry promotion conventions

*(Folded in from the retired `registry-housekeeping.md`, 2026-07-26 — the chores it tracked are
all done; these four conventions are still policy.)*

1. The **model card lives on the registry**, not in the training repo.
2. Every promotion records **provenance**: the training commit SHA and the MLflow run id.
3. `config.json` ships **in the registry** alongside the weights.
4. **Consumers pin a revision** — never track a moving branch.

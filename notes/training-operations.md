# Training Operations — Runbook (ai-lab-0)

_The operational "how" of model training: what runs where, how to launch/observe/stop/resume,
and the standing verification gates. Model **selection** rationale lives in
[actor-model-and-training.md](actor-model-and-training.md); machine provisioning in
[../Ai-Lab-0/machine-setup.md](../Ai-Lab-0/machine-setup.md); stack config is
[AI-Lab-0/docker-compose.yml](../../AI-Lab-0/docker-compose.yml). Created 2026-07-14 during the
de-risk phase start; keep the "Current runs" table honest._

## Current runs (update when this changes)

| Run | Container | State (2026-07-21) | Purpose |
|---|---|---|---|
| HiFi-GAN 24 kHz/80-band fine-tune | `vocoder_training` | **STOPPED 2026-07-15 @ g_02510000** — convergence watcher fired CONVERGED, A/B pairs human-audited (indistinguishable), checkpoint promoted (see below). Watcher timer disabled. | The 24 kHz vocoder ([decision](sample-rate-24khz-decision.md)) |
| §7 de-risk energy fine-tune | `sonora_training` | **STOPPED 2026-07-16 @ checkpoint_epoch=099** — convergence watcher fired CONVERGED (all 4 sweep groups, real margin: energy ρ≈1.000, WER Δ≤0.042, leakage≤0.091), sweep renders human-audited (energy/loudness discernible, natural). Training had continued to epoch 106 before being stopped; epoch 099 is the promoted checkpoint. Watcher timer disabled. | One trained FiLM channel (energy) |
| vat3 full 3-channel fine-tune | `sonora_training` | **STOPPED 2026-07-21 @ checkpoint_epoch=099** (owner call after the corrected audition; run had reached ~epoch 101 on a val-loss curve flat since epoch ~5). Verdict: energy PASS (ρ=1.0 ×4), tension near-pass (ρ=1.0 ×2, 0.90 ×2), **valence FAIL** (ρ scattered — labels, not steps, are the binding constraint). Eval-harness valence measure was found degenerate (raw EIV head, 81% dead zone) and switched to the valence_combo_v1 9-head combo mid-audit; both reports kept. Staged to `Registry/Sonora/vat3-24k/`; full resumable ckpt retained on host as the **v1.1 Emilia-continuation warm start**. MLflow run `wistful-dolphin-948` marked FINISHED. | First full-VAT (V/A/T) checkpoint; seed for v1.1 |

**No run queued behind this one.** Unlike vocoder→derisk_energy, there's no preset "next
container to launch." Both promoted checkpoints (`Registry/Sonora/vocoder-24k-hifigan/`,
`Registry/Sonora/derisk-energy-24k/`) are staged locally, not pushed to the public HF repo — same
open question for both: they're validated *components* (a standalone vocoder; one VAT channel of
three), not the shippable directable-actor deliverable, so publishing them sets a registry
precedent worth a deliberate call rather than a default push. The real next steps are scope
decisions, not mechanical continuations:

1. ~~Export gate does not apply yet.~~ **Adapted 2026-07-16:**
   `/data/toolchain/litert-conversion/convert_vat.py` converts the derisk checkpoint + 24k
   vocoder to the split-graph lane (new graph inputs: `spk(1,64)` host-lookup vector +
   `vat(1,3,T)`; decoder `t_emb` is 224-dim, not 160 — multi-speaker widens the U-Net). All
   gates pass: per-graph corr 1.000000, e2e waveform corr ≥ 0.9993, energy monotonic through
   the TFLite pipeline, all graphs GPU-clean. Artifacts staged in
   `Registry/Sonora/derisk-energy-24k/litert-split/`. Shapes stay 256/512 pending the
   [model-size-target-decision.md](model-size-target-decision.md) ceiling options.
2. **Full 3-channel VAT is gated on a corpus decision**, not just more training: valence/tension
   need labeled data the current corpus doesn't have, and Expresso (the expressive dataset on
   hand) is NC-tainted and can't ship. de-risk validated the FiLM/VAT *plumbing* on one clean
   channel — it didn't create a path to the other two.
3. **Formal §7 write-up** of the de-risk verdict is still open (numbers are in
   `Registry/Sonora/derisk-energy-24k/eval_report.json`; this table + the promotion commit are the
   only record so far).

**Build note (2026-07-15, kept for future watchers):** the derisk_energy watcher
(`render_vat_sweep.py` + `derisk_gate_watch.sh` + postprocess + timer) was built and armed before
the first checkpoint landed (`every_n_epochs: 100` gave the runway) — details in §Stop signal
below. Its plumbing test against the zero-init warmstart caught (and fixed, Sonora `16c1183`) a
tie-handling bug in the eval harness's Spearman that made an *inert* control channel score a
spurious ρ=1.0 PASS — worth remembering: **test every gate against a checkpoint that SHOULD fail
it.**

## Topology

Three training-related services in the compose stack, all **profile-gated** so GitOps syncs and
plain `up -d` never start a GPU run:

| Service | Profile | Launch |
|---|---|---|
| `sonora_training` (Matcha acoustic) | `training` | `docker compose -f AI-Lab-0/docker-compose.yml --profile training up -d sonora_training` |
| `vocoder_training` (HiFi-GAN) | `vocoder-training` | `... --profile vocoder-training up -d vocoder_training` |
| `sonora_vocalizer` (inference/dev, CPU) | — (auto) | part of the normal stack |

All use `rocm/pytorch:latest`, `/dev/kfd` + `/dev/dri`, and bootstrap deps via `uv pip install
--python /opt/venv/bin/python` in the container command (AGENTS §7). **Stop** a run with
`docker stop <container>` (or `compose stop`); both trainers checkpoint and resume cleanly.

## Locations

| What | Where |
|---|---|
| Datasets (license classes: `Sonora/configs/data_licenses.yaml`) | `/data/model-training/datasets/` (`LJSpeech-1.1`, `LibriTTS_R/train-clean-100`, `expresso` — NC, de-risk only) |
| Matcha training logs/checkpoints | `/data/model-training/sonora/logs/train/<experiment>/runs/<stamp>/checkpoints/` (bound over `Sonora/logs`) |
| Warm-start checkpoints | `/data/model-training/sonora/warmstart/` (`matcha_vctk.ckpt` donor, `derisk_energy_init.ckpt` built by `scripts/make_warmstart.py`) |
| Vocoder workspace | `/data/model-training/vocoder/` (`hifi-gan/` clone with local patches, `filelist_{train,val}.txt`, `cp_hifigan_24k/` checkpoints, `copysynth*/` gate reports, `pretrained/` originals) |
| Derived VAT filelists | `Sonora/data/libritts_r_vat/` (`{train,val}_op.txt`, `speakers.json`, `derivation_report.json`, `mel_statistics.json`) |

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
  `Sonora/scripts/make_warmstart.py` (loads donor `strict=False`, drops shape mismatches,
  asserts only expected-fresh tensors, saves a resumable ckpt).

## Standing gates (run these, don't trust vibes)

| Gate | Script (Sonora repo) | Verdict criteria |
|---|---|---|
| Vocoder copy-synthesis | `scripts/vocoder_copysynthesis.py --checkpoint <g_XXXXXXX>` | mel-L1 + WER vs the ASR floor on the same clips. **2026-07-14:** untuned baseline 0.659 / 0.323; ckpt 2,505,000 → 0.246 / 0.106; floor 0.064. Converged ≈ WER at floor + mel-L1 plateau (<5% between checkpoints) |
| Directability / §7 verdict | `scripts/eval_harness.py` (manifest-driven) | pre-registered: Spearman ρ ≥ 0.9, ECAPA leakage ≤ 0.2 (vs real inter-speaker gap), WER Δ ≤ +0.10. **3-channel since 2026-07-16:** produced measures tension→phonation composite, valence→EIV head; plus cross-channel independence (X-sweep moves Y's measure ≤ 0.5× Y's own sweep; groups `clip::channel` via `render_vat_sweep.py --channels`). Should-fail tested against the derisk ckpt: energy PASS, inert V/T + independence correctly FAIL (`/data/model-training/sonora/gate3_shouldfail/report.json`) |
| Warm-start identity | `scripts/test_vat_identity.py` | bit-identical synthesise at init for vat = 0/None/hot |
| Export (FiLM ops) | `scripts/test_film_export_gate.py` | litert-torch conversion GPU-clean, corr ≈ 1.0 |
| Export (checkpoints) | litert-conversion harness (`/data/toolchain/litert-conversion/`; `convert_final.py` = 22.05k v1-ljspeech, **`convert_vat.py` = 24k/multi-speaker/VAT**) | per-graph corr ≈ 1.0 — re-run after ANY fine-tune. ~~FiLM graphs need the `vat` wrapper input (open)~~ done 2026-07-16: `spk` + `vat` inputs wired, energy-monotonicity check included |

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
| `AI-Lab-0/scripts/vocoder_gate_watch.sh` | Timer-driven: newest `g_*` newer than `gate_watch/last_step` → runs the copy-synthesis gate in a **throwaway CPU-only container** (no `/dev/kfd`; training keeps the GPU) → postprocess. All paths/knobs env-overridable. |
| `AI-Lab-0/scripts/vocoder_gate_postprocess.py` | Applies the rule **WER ≤ ASR floor + margin AND \|Δ mel-L1\| < plateau%** (defaults 0.064 + 0.02, 5% — `GATE_ASR_FLOOR`, `GATE_WER_MARGIN`, `GATE_MEL_PLATEAU_PCT`); appends `history.jsonl`; logs to MLflow `hifigan_24k_gate`; on convergence writes `gate_watch/CONVERGED` + a loud journal line. |
| `AI-Lab-0/scripts/vocoder-gate-watch.{service,timer}` | systemd oneshot + half-hourly timer (`Persistent=true`), `User=lmcfarlin` (docker + datashare groups). Checkpoints land every ~7 h, so worst-case detection lag ≈ 7%. |

**Implementation (derisk_energy acoustic run — built 2026-07-15 from the template):**

| Piece | What |
|---|---|
| `Sonora/scripts/render_vat_sweep.py` | The generation half (lives in the training repo — it knows the model): CPU-loads the acoustic ckpt + the promoted 24k HiFi-GAN (`g_02510000`), renders val clips at energy ∈ {−1, −0.5, 0, +0.5, +1} (fixed seed; sweep rows differ only in vat), writes WAVs + `manifest.jsonl` + `speaker_refs.txt` + `render_meta.json`. |
| `AI-Lab-0/scripts/derisk_gate_watch.sh` | Timer-driven: newest `checkpoint_epoch=*.ckpt` under `logs/train/derisk_energy` newer than `derisk_gate_watch/last_ckpt` → render sweep + `eval_harness.py` in a **throwaway CPU-only container** → postprocess. Repo mounts **rw** (unlike the vocoder gate): the render imports matcha, so the container does the `Cython + -e .` install dance. Sweep WAVs land in `/data/model-training/sonora/derisk_eval/<runstamp>_epochNNN/`. |
| `AI-Lab-0/scripts/derisk_gate_postprocess.py` | Aggregates the harness's per-group verdicts: **CONVERGED = every sweep group passes \|ρ\| ≥ 0.9 AND leakage ≤ 0.2 AND WER Δ ≤ +0.10** (the pre-registered §7 thresholds, applied by the harness itself); appends `history.jsonl`; logs to MLflow **`derisk_energy_gate`** (`gate_rho_min_abs`, `gate_leakage_max`, `gate_wer_delta_max`, `gate_converged`); on convergence writes `derisk_gate_watch/CONVERGED`. |
| `AI-Lab-0/scripts/derisk-gate-watch.{service,timer}` | systemd oneshot + half-hourly timer (`*:11/30`, offset from the vocoder watcher's `:04/30`), `User=lmcfarlin`, `TimeoutStartSec=120min` (render + whisper + ECAPA + install dance). Checkpoints land every 100 epochs (`every_n_epochs: 100`), so a 30 min poll is generous. |

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
sudo cp AI-Lab-0/scripts/derisk_gate_watch.sh AI-Lab-0/scripts/derisk_gate_postprocess.py /usr/local/bin/
sudo cp AI-Lab-0/scripts/derisk-gate-watch.service AI-Lab-0/scripts/derisk-gate-watch.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now derisk-gate-watch.timer
```
Disable with `sudo systemctl disable --now <name>-gate-watch.timer` when the run ends (the
vocoder one was disabled 2026-07-15 when its run stopped).

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
5. **Filelists are derived artifacts** — regenerate with `scripts/phonemize_filelist.py` /
   `scripts/derive_vat_corpus.py`; don't hand-edit.
6. **The `/data/repos/Sonora` deploy clone is a bare `git clone`** — it has none of the
   machine-local derived data products that only exist in the dev tree's untracked `data/`
   subdirs (e.g. `data/libritts_r_vat/`, chore #11's motivating bug's sibling). Hit this
   2026-07-15 launching `derisk_energy`: copied `Sonora/data/libritts_r_vat/` from the dev tree
   into the deploy clone rather than regenerating (same commit, same source data, cheaper). Check
   for this class of gap whenever a training command references `data/` and the deploy clone is
   freshly (re)cloned.

## Cross-references

[sample-rate-24khz-decision.md](sample-rate-24khz-decision.md) ·
[vat-conditioning-design.md](vat-conditioning-design.md) ·
[dataset-landscape.md](dataset-landscape.md) · [STATE.md](STATE.md) ·
[next-steps §B](../Prosodia/next-steps.md) ·
[../Ai-Lab-0/machine-setup.md](../Ai-Lab-0/machine-setup.md)

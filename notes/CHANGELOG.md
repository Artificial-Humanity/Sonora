# Changelog

This document tracks technical changes, refactoring milestones, and build-system adjustments
for Project Sonora (the training pipeline and the teacher-synthesis lane).

> **Maintenance:** Required by [AGENTS.md](../AGENTS.md) §4, which is the rule of record —
> append an entry **after committing** code work (source, configs, dependency manifests;
> docs-only commits are exempt), always carrying the short 7-character commit SHA. The log is
> append-only within a release cycle: entries are pruned **only** when a new version of the
> overall project is tagged, at which point they collect under that version's heading. New
> entries go at the top under the current date, using the `Added` / `Changed` / `Fixed` /
> `Removed` structure.
>
> Maintained since **2026-08-06**, when the changelog + code-review cycle (AGENTS.md §4–§5)
> was adopted here to match the convention Prosodia already runs. History before that date
> lives in `git log` and [STATE.md](STATE.md).

---

## 2026-08-06

### Fixed — acquisition lane (§5)

- **`cf6429d` — the book/real-audio lane's silent-corruption bugs.** None of these raised;
  each produced a plausible corpus instead of an error.
  - **A-H4** — the resolved LibriVox project was never compared to the requested URL.
    Matching is word-based with a prefix fallback, and `key` comes from the *request*, so
    a near-miss wrote the wrong book's audio into the right-looking directory. Compared
    and refused now.
  - **A-H2** — `decode("utf-8", errors="replace")` on Gutenberg text turned every
    non-ASCII byte of an ISO-8859-1 edition into U+FFFD while the ASCII stayed clean.
    Strict UTF-8 → cp1252 → latin-1, then a check that no replacement character survived.
  - **A-H3** — `books_ledger.json` / `staging_log.json` were read at startup and written
    back whole after minutes of network work, so overlapping runs erased each other. New
    `synth_common.update_json` re-reads under an flock and renames into place.
  - **A-H5** — the real-audio lane split on any `[.!?]`+space, shipping "Mr." and "Smith
    went home." as two clips. pysbd now, shared with `book_ingest`, plus an
    `is_complete_utterance` gate. **The gate alone could never have fixed this** —
    `is_complete_utterance("Mr.")` is `True` by shape — and there is a test asserting that
    so pysbd isn't later dropped as redundant.
  - **A-M7** — dialogue extraction hardcoded curly double quotes; since the caller falls
    back to the raw paragraph when no utterances are found, a differently-quoted edition
    filed its dialogue as **narration**. Convention is detected per edition now.
    *Correction to the review's record:* *Uneasy Money* uses **straight** singles (U+0027,
    the apostrophe character), not curly. Measured on the real text: **0 → 1,165**
    utterances.
  - **A-M6** — banks claimed `Standard Ebooks CC0` regardless of source while the router
    has been feeding Gutenberg down this path all along. False licence metadata in every
    derived clip's paper trail; provenance follows the source now, and a local `--epub` is
    `UNKNOWN` rather than assumed.
  - **A-M12** — "complete download" meant `size > 1024`, which a 404 page clears; it was
    saved as `.mp3` and every resume skipped it. Magic-byte check on both paths.
- **`cf6429d` — two duplications collapsed**, both the kind the review keeps finding:
  `qc_passages` carried its own `is_complete_utterance` marked "kept in sync" (already
  drifted — it accepted guillemets and trailing spaces `book_ingest` rejected), and
  `_QUOTED_SPAN` handled straight quotes while the extractor beside it did not.

### Changed

- **`428cb55` — `pyproject.toml` is the single source of dependency truth; `requirements.txt`
  deleted.** It was the upstream list and had drifted **both ways**: seven declared packages
  nothing imports (`torchvision`, `torchmetrics`, `tensorboard`, `pandas`, `notebook`,
  `ipywidgets`, `seaborn`) and three imported at module scope but never declared
  (`soundfile`, `tqdm`, `PyYAML`), resolving by transitive luck through librosa. The
  replacement was derived by walking every import, not by editing the old list. Distribution
  renamed **`matcha-tts` → `sonora`** with an Apache-2.0 licence field; the import package
  stays `matcha`. `make create-package` deleted — it ran `twine upload` against metadata
  still claiming `name="matcha-tts"`, so one invocation would have published this fork to
  PyPI under upstream's name.
- **`428cb55` — GPL `phonemizer` is opt-in (`[espeak]`), and verified absent from a fresh
  training container.** README § 3 promises espeak is banned from the runtime path and the
  licence wall enforces that for *data*; nothing enforced it for *dependencies*.
  `cleaners.py` already imported it lazily, so only legacy LJSpeech checkpoints ever needed
  it. `gdown`/`wget` got the same treatment (`[download]`) — two network packages were
  mandatory in every container for a code path none of them take.
- **`428cb55` — `matcha/cli.py` works for Sonora checkpoints (E-H2).** It sent every
  checkpoint through espeak cleaners and wrote every file at 22050 Hz — so a Sonora
  checkpoint got phonemes it never trained on, in a 24 kHz waveform tagged 22.05 kHz
  (~9% slow, which sounds like a sluggish model rather than a header bug). The download
  guard read `not hasattr(args, "checkpoint_path") and args.checkpoint_path is None`, which
  argparse makes permanently False, so `--checkpoint_path` still fetched the upstream
  checkpoint over the network before discarding it — a hard failure offline. Lane is now
  read off the checkpoint; `--vat`/`--guidance` added and bounded; `--spk` range-checked.
  **Lane detection, op_g2p encoding and the 24 kHz vocoder loader now live once, in
  `matcha.cli`**, with `vocalizer.py` importing them — it had its own copy of all three, and
  its `SONORA_VOC24K_CONFIG` default pointed at a different (byte-identical, checked) file.
- **`428cb55` — the Vocalizer HTTP API enforces the control contract (E-M2).** It passed
  `valence`/`energy`/`tension`/`guidance` straight into the model unbounded while the UI
  could not. V/A/T are per-speaker z-scores clamped at 2σ in derivation, so `valence=50`
  does not make more emotion — it drives the FiLM trunk off the manifold and still renders
  fluent audio. Now one `CONTROL_BOUNDS` table, and **400 rather than 500**.
- **`428cb55` — `matcha/app.py` is no longer an entry point and no longer shares publicly
  (E-L4).** Its `launch(share=True)` opened a public tunnel from the machine holding
  unreleased checkpoints; now opt-in via `MATCHA_APP_SHARE=1`.

### Added

- **`428cb55` — `environments/`** replaces the 3-line `uv.lock` stub (T4/G-1): real
  `uv pip freeze` records of the two lanes that do the work, the container one produced by
  running the compose prep chain verbatim. A resolver lockfile would have been *actively
  misleading* — torch ships in the ROCm base image and is not installable from PyPI, so
  locking `torch>=2.0.0` pins a CUDA wheel and describes an environment we have never
  trained in. Named `environments/` because `.gitignore` line 41 ignores `env/` as a
  virtualenv, which would have committed the files in name only.
- **`428cb55` — `tests/test_cli_lanes.py`** (15 cases) covering every guard above, plus
  partial **D-M3**: the tokenizer *deletes* digits rather than expanding them ("I have 3
  cats" → `ˈaɪ hˈæv kˈæts`, verified live) and `g2p.validate()` cannot catch it because
  nothing illegal is present — a word is simply gone. Refused at synthesis input; D-M3
  stays open for the corpus lane.

- **`f015c8e` — `scripts/score_holdout.py` + `score_holdout.sh`, the never-trained holdout
  (Phase 0a).** Scores a checkpoint teacher-forced per clip with **paired** noise draws, so
  two checkpoints see identical timesteps and noise on identical clips; clip-to-clip
  variance dwarfs the effect being measured, and an unpaired comparison of two 5,463-clip
  means reads as noise. Runs in a throwaway ROCm container as `ai-mgr`, building
  `monotonic_align` in a `/tmp` copy so the owner's checkout is neither written to nor
  littered with build artifacts. **No `phonemizer` in the eval deps** — `matcha.text.cleaners`
  imports it lazily, the filelists are already IPA, so pulling GPL in to satisfy an import
  that never fires would be the G-3/G-4 mistake voluntarily.
- **`f015c8e` — `configs/data_licenses.yaml` declares `libritts_r_holdout_devclean`.**
  Without it `enforce()` refuses to load the filelist — the same trap that made v3/v3b/v3c
  structurally unrunnable. Declaring it is **not** permission to train on it, and the wall
  will not stop such a run; the guards are `--assert-disjoint-from` (any shared clip
  basename stops the run) and the deletion of the derive's `train_op.txt`/`val_op.txt` after
  concatenation into `holdout.txt`, so no file in the directory carries a name a training
  config would accept.

### Changed

- **`f015c8e` — `vat3c` ep099 is retired; `vat3-24k` ep099 is the base.** The holdout says
  the v3c fine-tune was a **regression, not a no-op**: +0.0164 against its own warm start
  over 5,463 unseen clips, all three loss terms worse, better on only 39.1% of clips, and
  +0.0443 worse on v3c's *own* val split. Not a normalisation artifact (+0.0172 under the
  v2 constants it trained with). The ear had already said "no audible change"; this
  sharpens the direction to *down*. **Phase 0b — the clean-lineage retrain from
  `matcha_vctk` — is not indicated**, because the v2 fine-tune shows a real gain on unseen
  audio (`diff` −0.0241, better on 78.7% of clips) and a compromised lineage could not
  produce that. Owner's call to ratify. Recorded in `quality-gap-plan.md` § 0a,
  `STATE.md` and `training-sources.md`.

- **`03fed75` — `vat3c_finetune` retired at every place it could be picked up.** A
  `RETIRED` banner on the experiment config (kept, not deleted — Phase 1 derives from it
  and the measurement must stay reproducible), `RETIRED.md` in the experiment and both
  checkpoint-bearing run dirs, `warmstart/vat3_ep099.ckpt` naming the correct base, and
  `resume.ckpt` re-pointed off the retired checkpoint it had been left aimed at. The
  launcher rewrite is **AI-Lab-AMD `a155eb6`**: it carried *two* independent hardcodings
  of vat3c — the experiment name and the resume glob — so queueing a successor run by
  editing the obvious one would silently have warm-started it from the retired ep099.
  `SONORA_EXPERIMENT` is now required (unset ⇒ the container idles with a message rather
  than exiting, since `restart: unless-stopped` turns a fast exit into a crash loop),
  auto-resume is scoped to that experiment's own run dirs, and `vat3c_finetune` is
  refused by name. New launch contract: `notes/training-operations.md`.

### Fixed

- **`e5e5ed1` — E-M5: the logged diffusion loss is masked.** `BASECFM.compute_loss` summed
  its residual over the full padded tensor. The decoder masks its own output, but the
  target residual `u = x1 - (1-σ)z` does not (`z` is drawn `randn_like` across the padding),
  so each batch's logged `diff_loss` carried a floor proportional to its padding fraction.
  Train batches have been length-bucketed since 2026-08-01 and val batches are not, so the
  floor fell on val and presented as a 3.2× train/val gap — vat3c epoch 1 logged diff 0.646
  train vs 2.069 val while `dur_loss` and `prior_loss` matched to three decimals.
  Gradients are unchanged (the padded positions carried none); **only the logged value
  moves, and it moves down.** `diff_loss` and `loss/train` / `loss/val` therefore do not
  compare across this commit — a second scale break in the metric after the bucketing
  change. Documented at every site that teaches someone to read the curve.
- **`e5e5ed1` — E-M6: the RoPE cache is built outside inference mode.**
  `on_validation_end` synthesises under `@torch.inference_mode()`, so a
  `RotaryPositionalEmbeddings` cache first built or grown during validation was made of
  inference tensors; the next training step with text no longer than the cached length
  reused them and killed the run with "Inference tensors cannot be saved for backward".
  The build now nests `torch.inference_mode(False)`, so the cache is an ordinary detached
  tensor regardless of which mode first demanded it.

### Added

- **`e5e5ed1` — `tests/test_training_seams.py`**, regression coverage for both of the
  above. Verified to fail against pre-fix source (4.643 vs 1.643 logged loss at 75%
  padding; the RoPE case reaching the real backward crash). The tests need the model's
  dependency stack, so they `importorskip` on the host venv and run in the ROCm training
  container:
  `docker compose run --rm --entrypoint pytest sonora_training tests/test_training_seams.py`

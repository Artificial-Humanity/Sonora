# Sonora — Expressive & Directable TTS Actor Training

Sonora is a directable, castable, and mobile-friendly text-to-speech (TTS) actor training codebase. 

The model is built on top of the **Matcha-TTS** architecture (conditional flow-matching mel decoder solved with a few-step ODE + vocoder integration), augmented with custom emotion conditioning and continuous voice casting features.

---

## 🏛️ Storage Layout Recommendations

When setting up for model training, it is recommended to use a high-speed storage volume for dataset caches and output checkpoints to optimize disk speed and minimize write wear on the primary system drive.

### Recommended Workspace Structure
* **Code Workspace:** The cloned repository folder.
* **Large Asset Directory:** A dedicated folder or mount on a high-throughput SSD.

It is common practice to symlink the following folders in your repository to your fast storage drive:
* `data/` -> (symlink to your dataset/preprocessing cache folder)
* `outputs/` -> (symlink to your training runs/checkpoints folder)

---

## 🐳 Containerized Training Environment

To ensure stable package dependencies and leverage GPU/hardware acceleration (such as ROCm for AMD hardware or CUDA for NVIDIA hardware), training should be run inside a Docker container.

### Example Container Launch (AMD ROCm)
If training on an AMD GPU with ROCm support, you can launch a container with device mappings:

```bash
docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  --shm-size 8G \
  --security-opt seccomp=unconfined \
  -v /path/to/projects:/projects \
  -v /path/to/data:/data \
  rocm/pytorch:latest
```

### Installation
Once inside the container:
1. Navigate to the project folder. Since 2026-07-22 the workspace is flat: `Sonora/github` (this repo) sits alongside `Sonora/huggingface` (the model-registry checkout of `artificial-humanity/Sonora`). The GitHub repo itself is still named `Sonora`; there is no umbrella repo.
2. Install project requirements with [uv](https://github.com/astral-sh/uv), this organization's standard for Python tooling:
   ```bash
   uv pip install --no-build-isolation -e .
   ```
   *(Note: `--no-build-isolation` is recommended when using pre-installed container PyTorch/NumPy dependencies to build Cython extensions.)*
3. **No `espeak-ng` required.** Sonora phonemizes through the permissive `op_g2p` lane (OpenPhonemizer dictionary → DeepPhonemizer TFLite OOV fallback) against a locked 178-symbol IPA vocab, shared by training and runtime. espeak-ng (GPL-3.0) was removed from the training path on 2026-07-14 and is banned from the runtime path by the licence wall.

---

## 🔄 Phased Training Plan

Training is structured in sequential phases to isolate complexity:

### Phase 1: Plain Fine-Tune
* **Goal:** Fine-tune a base checkpoint on a single-speaker dataset (e.g., LJSpeech) to verify the toolchain and environment.
* **Process:** Configure data configs, execute the training loop via `python matcha/train.py`, and inspect Tensorboard outputs under `outputs/`.

### Phase 2: Export & On-Device Validation
* **Goal:** Confirm the split-graph `litert-torch` export (Plan A since 2026-07-12) produces GPU-clean `.tflite` graphs — three graphs (text encoder / decoder / vocoder) at fixed shapes with the ODE loop host-side. `torch → ONNX` + `onnx2tf` is the Plan B monolith. See [`scripts/litert_export/`](scripts/litert_export/).
* **Process:**
  1. Convert with the split-graph recipe (`convert_final.py` for the 22.05 kHz baseline, `convert_vat.py` for 24 kHz/multi-speaker/VAT). `python -m matcha.onnx.export` is the Plan B path, not the default.
  2. Validate per-graph parity (corr ≈ 1.0) and end-to-end waveform parity against torch, then that the graphs load and synthesize in the target runtime.
* **Status:** verified at parity on Phase 0 and on the de-risk checkpoint. The gate suite **refuses** on failure (since F-C1, 2026-08-06 — it *reported* rather than refused until then, which is a gate that cannot fail). ⚠ Valence/tension have still never been driven nonzero through a converted graph — see [`notes/todo.md`](notes/todo.md) §2 before trusting it.

### Phase 3: Directability (VAT Conditioning)
* **Goal:** condition on `(valence, energy, tension)` — energy occupies the arousal slot — via zero-init FiLM in the text encoder and flow decoder.
* **Status: shipped.** The de-risk run validated the architecture on energy (rho ~ 1.000, 2026-07-16); the full 3-channel run (`vat3-24k`, 2026-07-22) landed energy PASS, tension near-pass, **valence FAIL** — a corpus-label limit, not an architectural one. `vat3c` (2026-08-06) re-ran the three channels on a phoneme-corrected corpus and changed nothing audible.
* **Process:** Implement FiLM/AdaLN modulation on the text encoder + flow decoder, preprocess training data with VAT labels, and retrain.

### Phase 4: Delivery — the 4th conditioning channel
* **Goal:** one of `{Dialogue, Neutral, Documentary, Newscaster, Speech}` + `unknown`, embedded host-side onto the same zero-init FiLM path (Director↔Actor contract v2).
* **Status:** **shipped on the training side** — `vat_dim` is **8** (three V/A/T channels plus a five-wide one-hot delivery block), and `vat5_finetune` trained 8-wide to `ep019` (2026-08-08/09). The expressive corpus that gives the delivery channel something to learn from is **merged into v6 and BUILT** (2026-08-10): **826 appended rows**, 816 of them delivery-labelled across four lanes, out of 1,004 eligible keeps (158 already in v5, 14 over-length, 6 dropped on digits). `Documentary` is retired into Neutral; `vat_dim` stays 8. No v6 run is queued yet. The export lane is deliberately still 3-wide. Seam assertions proven to fire (`scripts/gates/test_vat_dim_seams.py`).

### Phase 5: Casting & Speaker Embedding Blends
* **Goal:** Re-derive continuous voice casting spaces (e.g., age, masculinity, strain) in speaker-embedding space.
* **Process:** Train speaker-embedding Look-Up Tables (LUTs) for anchor voices and map the casting grid to this space.

---

## 🧭 Where the engineering notes live

This README is the setup guide. The internal record — architecture canon, current state,
what runs next — is in [`notes/`](notes/), mapped one line per file in
[`notes/README.md`](notes/README.md). Start at [`notes/STATE.md`](notes/STATE.md).
Agent/developer entry point: [`AGENTS.md`](AGENTS.md).

---

## 📄 License & Credits

Sonora is licensed under the **Apache License, Version 2.0**. 

The project is built on the [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) architecture (originally licensed under the MIT License). Attribution to the original creators can be found in the [NOTICE](NOTICE) file.

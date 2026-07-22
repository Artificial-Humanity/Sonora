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
1. Navigate to the `Sonora` project folder (this workspace's local checkout is named `Sonora-GH`, to disambiguate it from the `Sonora-HF` model registry checkout alongside it — the GitHub repo itself is still `Sonora`).
2. Install project requirements with [uv](https://github.com/astral-sh/uv), this organization's standard for Python tooling:
   ```bash
   uv pip install --no-build-isolation -e .
   ```
   *(Note: `--no-build-isolation` is recommended when using pre-installed container PyTorch/NumPy dependencies to build Cython extensions.)*
3. Install additional system packages (such as `espeak-ng`) if utilizing phonemization.

---

## 🔄 Phased Training Plan

Training is structured in sequential phases to isolate complexity:

### Phase 1: Plain Fine-Tune
* **Goal:** Fine-tune a base checkpoint on a single-speaker dataset (e.g., LJSpeech) to verify the toolchain and environment.
* **Process:** Configure data configs, execute the training loop via `python matcha/train.py`, and inspect Tensorboard outputs under `outputs/`.

### Phase 2: Export & On-Device Validation
* **Goal:** Confirm the `torch → ONNX` and downstream compilation targets (such as `.tflite` or ONNX Runtime) operate correctly.
* **Process:**
  1. Export checkpoint to ONNX: `python -m matcha.onnx.export`
  2. Validate that the exported model loads and synthesizes speech accurately inside your target runtime.

### Phase 3: Directability (VAT Conditioning)
* **Goal:** Introduce Valence, Arousal, and Tension (VAT) conditioning to the model.
* **Process:** Implement FiLM/AdaLN modulation on the text encoder + flow decoder, preprocess training data with VAT labels, and retrain.

### Phase 4: Casting & Speaker Embedding Blends
* **Goal:** Re-derive continuous voice casting spaces (e.g., age, masculinity, strain) in speaker-embedding space.
* **Process:** Train speaker-embedding Look-Up Tables (LUTs) for anchor voices and map the casting grid to this space.

---

## 📄 License & Credits

Sonora is licensed under the **Apache License, Version 2.0**. 

The project is built on the [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) architecture (originally licensed under the MIT License). Attribution to the original creators can be found in the [NOTICE](NOTICE) file.

# Sonora — Expressive & Directable TTS Actor Training

Sonora is a directable, castable, and mobile-friendly text-to-speech (TTS) actor model developed for the [Artificial-Humanity](file:///home/lmcfarlin/Projects/Artificial-Humanity) project. 

The model is built on top of the **Matcha-TTS** architecture (conditional flow-matching mel decoder solved with a few-step ODE + HiFi-GAN vocoder), augmented with custom emotion conditioning and continuous voice casting features.

---

## 🏛️ Storage Layout (Hybrid Design)

To optimize disk speed and prevent wear on the OS system drive, this repository uses a **hybrid layout** that symlinks all heavy training inputs and outputs to a dedicated, high-speed 4TB NVME drive mounted at `/data`.

* **Code Workspace (System SSD):** `~/Projects/Artificial-Humanity/Sonora/`
* **Data & Output Storage (NVME SSD):** `/data/model-training/`

### Symlinked Folders
* `data/` -> `/data/model-training/sonora/data/` (pre-processed datasets, wav paths, and phone files)
* `outputs/` -> `/data/model-training/sonora/outputs/` (saved checkpoints, logs, and exported models)

Shared raw datasets (e.g., LJSpeech, LibriTTS) should be stored in `/data/model-training/datasets/` to avoid duplicating space across different training runs or architectures.

---

## 🐳 Local AMD ROCm Training Environment

Since the host runs **Ubuntu 26.04 LTS** and has a **Ryzen AI Max+ 395 (Strix Halo iGPU, Radeon 8060S)** with **112 GB of allocatable unified memory**, but uses **Python 3.14**, local training must run inside a pre-built ROCm PyTorch Docker container to ensure stable package dependencies.

### Start the Training Container
From the host terminal, run:

```bash
docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  --shm-size 8G \
  --security-opt seccomp=unconfined \
  -v /home/lmcfarlin/Projects:/projects \
  -v /data:/data \
  rocm/pytorch:latest
```

Once inside the container:
1. Navigate to `/projects/Artificial-Humanity/Sonora`.
2. Install project requirements (`pip install -r requirements.txt`).
3. Compile custom training extensions (like `monotonic_align`).

---

## 🔄 Phased Training Plan

Training is structured in sequential phases to isolate complexity and de-risk the pipeline.

### Phase 1: Plain Fine-Tune (Learn the Loop)
* **Goal:** Fine-tune a stock checkpoint on a small, clean dataset (e.g., LJSpeech or a single LibriTTS speaker) to verify the pipeline.
* **Process:** Configure Hydra/YAML data configs, run the training loop, and inspect Tensorboard outputs under `outputs/`.

### Phase 2: Export & On-Device Validation
* **Goal:** Confirm the `torch → ONNX → TFLite` compilation target is compatible with the Prosodia FFI runtime.
* **Process:**
  1. Export checkpoint to ONNX: `python -m matcha.onnx.export`
  2. Lower ONNX to TFLite via `onnx2tf`.
  3. Validate that the `.tflite` model successfully loads and plays audio in the `ProsodiaTuner` harness.

### Phase 3: Directability (VAT Conditioning)
* **Goal:** Add Valence, Arousal, and Tension (VAT) conditioning to the model.
* **Process:** Implement FiLM/AdaLN modulation on the text encoder + flow decoder, preprocess training data with VAT labels, and retrain.

### Phase 4: Casting & Speaker Embedding Blends
* **Goal:** Re-derive the continuous voice spaces (age, masculinity, strain) in speaker-embedding space.
* **Process:** Train speaker-embedding Look-Up Tables (LUTs) for anchor voices and map the platform-level casting grid to this space.

---

## 📄 License & Credits
Sonora is licensed under the **Apache License, Version 2.0**. 

The project is built on the [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) architecture (originally licensed under the MIT License). Attribution to the original creators can be found in the [NOTICE](NOTICE) file.

---

## 📖 Key Project Pointers
* Main Project Workspace: [Artificial-Humanity](file:///home/lmcfarlin/Projects/Artificial-Humanity)
* Behavioral Rules & Guidelines: [AGENTS.md](file:///home/lmcfarlin/Projects/Artificial-Humanity/AGENTS.md)
* Comprehensive Actor Notes: [actor-model-and-training.md](file:///home/lmcfarlin/Projects/Artificial-Humanity/Notes/Prosodia/actor-model-and-training.md)
* Production Design Document: [high-ambition-1-matcha-actor.md](file:///home/lmcfarlin/Projects/Artificial-Humanity/Notes/Prosodia/high-ambition-1-matcha-actor.md)

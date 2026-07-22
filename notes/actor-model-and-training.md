# Actor Model & Training — Selection, Hardware, First-Run Plan

Consolidated guide for the Prosodia actor (TTS): **which model to train, on what hardware, and how**
(for a first-time trainer). The chosen first model is **Matcha-TTS** — production design in
[high-ambition-1-matcha-actor.md](high-ambition-1-matcha-actor.md); the higher-ceiling
[high-ambition-5-styletts2-lite.md](high-ambition-5-styletts2-lite.md) is the later re-platform.
See also the export-toolchain spike in [next-steps.md](../Prosodia/next-steps.md).

> **Status (decision open, 2026-06-14):** recommendation leans **Matcha-TTS** for the first,
> cost-conscious training effort, with **StyleTTS2-Lite** as the higher-ceiling Phase 2.
> Bias throughout: **minimize first-run risk and wasted spend.** The expensive mistake is not a
> cloud GPU bill (TTS models are small and cheap to train) — it's sinking days into a toolchain
> that fights you.

---
---

# PART 1 — MODEL SELECTION (StyleTTS2-Lite vs Matcha-TTS)

## What the actor has to do (the real requirements)

1. **On-device, mobile, cross-platform** (iOS / Android / desktop), real-time.
2. **Directable** — take Valence/Arousal/Tension (VAT) emotion codes from the Gemma Director and
   render them as audible prosody. This is *our* addition, not inherited from any base model.
3. **Castable / blendable voices** — a continuous voice space (age / masculinity / strain), blended
   per the existing `voice_loader` + casting grid.
4. **Small (~50–80 MB) and consumer-trainable** (single GPU).
5. **Permissive license (MIT / Apache-2.0)** end to end.
6. **Exportable** to the on-device runtime (we proved `torch → ONNX` is the robust backbone for
   *stock dynamic-shape* modules; the official `ai-edge-torch`/`litert_torch` path is *not* reliable
   for those — see the export spike in [next-steps.md](../Prosodia/next-steps.md). *Update
   2026-07-12:* with fixed-shape re-authoring, `litert-torch` split-graph export is proven on our
   own checkpoint and is now **Plan A**; the ONNX→`onnx2tf` monolith is the fallback).

## The two candidates

### StyleTTS2 (lineage we're already on) — MIT

- **Fit:** Its disentangled **continuous style-vector space** (128 acoustic + 128 prosodic) is the
  reason our whole actor stack exists — voice packs *are* style vectors, the casting grid
  interpolates them. Best-in-class naturalness.
- **Problem (proven by spike):** vanilla architecture is **export- and mobile-hostile** —
  bidirectional LSTMs (no clean export; CPU-only on mobile GPU/NPU delegates) and a diffusion style
  sampler. Training is **multi-stage and adversarial** (GAN discriminators + a WavLM SLM loss),
  which is notoriously finicky and unstable — a hard first training endeavor.
- **"Lite" means re-architecture, not fine-tuning:** to ship it we'd replace the LSTMs
  (transformer/conv), strip WavLM + discriminators, prune PL-BERT, and swap the diffusion sampler
  for a deterministic VAT-conditioned style predictor. That is real model surgery *plus* research.
  Full design: [high-ambition-5-styletts2-lite.md](high-ambition-5-styletts2-lite.md).

### Matcha-TTS — MIT

- **What it is:** a 2024 (ICASSP) non-autoregressive TTS using **conditional flow matching (OT-CFM)**.
  Pipeline: **transformer text encoder → duration predictor (monotonic alignment) → flow-matching
  mel decoder solved with a few-step ODE (default 5 steps) → HiFi-GAN vocoder**. ~18 M-param
  acoustic model; very fast (low RTF); trains on a single consumer GPU in hours.
- **Why it fits us better on the hard axes:**
  - **Export-friendly by design** — transformer + conv + a *fixed-step* ODE (no dynamic loop), and
    it ships an **official ONNX exporter** (`python -m matcha.onnx.export …`, can embed the
    vocoder). That's exactly the `torch → ONNX` backbone we validated, officially supported.
  - **Beginner-friendly training** — **single-stage**, no GAN/adversarial instability (the vocoder
    is a separate pretrained HiFi-GAN), Hydra-config + PyTorch-Lightning driven, pretrained
    checkpoints to fine-tune from.
  - **Mobile-friendly** — transformer/conv accelerate on GPU/NPU delegates; no LSTM CPU fallback.
- **What it lacks (the cost of switching):** Matcha conditions on a **speaker embedding**, not a
  rich disentangled style space. To get our **directable VAT + casting/blend** behaviour we'd add a
  conditioning mechanism (e.g. FiLM / AdaLN conditioning the encoder + flow decoder on a VAT/style
  vector) and re-derive the voice-pack/blend machinery. Tractable, but it's the real work.

## Decision matrix

| Criterion | StyleTTS2-Lite (re-arch) | Matcha-TTS |
|---|---|---|
| License | ✅ MIT | ✅ MIT |
| Quality ceiling | ✅ highest | ✅ very good (near-parity) |
| Directability/casting fit **out of the box** | ✅ native style space | ⚠️ must add VAT/style conditioning |
| Export to ONNX→TFLite | ⚠️ after replacing LSTMs/diffusion | ✅ official ONNX exporter, export-clean |
| Mobile GPU/NPU acceleration | ⚠️ only after LSTM removal | ✅ transformer/conv |
| **Training difficulty (first-timer)** | ❌ multi-stage + adversarial (hard) | ✅ single-stage, no GAN (approachable) |
| **Training cost/time** | ❌ higher (stages, instability, retries) | ✅ lower (fast CFM, stable) |
| Reuse of existing actor code | ✅ keeps style machinery | ⚠️ re-derive style/blend layer |
| Maturity of *our* export wrapper | ✅ already built + eager-validated | none yet (but official exporter exists) |

## On quality — what "higher ceiling" means (and doesn't)

Calling StyleTTS2 the "higher-ceiling" option is a statement about **peak achievable quality**, not
a pass/fail judgment on Matcha.

- **StyleTTS2** is among the highest-naturalness open models (human-level MOS in its paper; rich,
  varied prosody from its style diffusion). Its *best case* is top-tier.
- **Matcha-TTS** is **very good** — competitive naturalness, far faster — but rates a notch below
  StyleTTS2 on raw naturalness/expressiveness in head-to-head listening. A **real but narrow** gap.

Three reasons the gap matters less than the word "ceiling" suggests:

1. **Ceiling ≠ what you'll get on attempt one.** StyleTTS2's quality comes from adversarial,
   multi-stage training that is hard to land; a first-timer's first run can sound *worse* than a
   well-trained Matcha. In practice, **Matcha likely yields higher *actual* quality for your first
   model.**
2. **Your differentiator is the layer you build** (VAT directability + casting), not the stock
   model — and it ports to either base.
3. **The work transfers.** Data prep, the directability layer, and the whole `torch→ONNX→TFLite`
   pipeline are reused even if you later graduate to StyleTTS2 — so Matcha-first is not a throwaway.

**Where the gap could actually bite:** the [dramatic-reader](high-ambition-2-dramatic-reader.md)
ambition. Extreme theatrical expressiveness is exactly the axis where StyleTTS2's higher ceiling
shows most. If jaw-dropping dramatic delivery is a *hard requirement* (not a nice-to-have), weight
StyleTTS2-Lite higher despite the harder training. For natural, pleasant, directable long-form
narration, Matcha is very likely sufficient.

**De-risk with your ears before spending anything:** Matcha ships pretrained checkpoints + a demo
page; StyleTTS2/Kokoro have public samples. Best of all, **run pretrained Matcha on your own sample
passages locally on the M1 Max** (no training, no RunPod) and A/B it against a StyleTTS2/Kokoro
sample. If it clears your bar there, the ceiling difference is academic for v1; if it disappoints on
expressiveness, that's a real signal to favor StyleTTS2-Lite.

## Recommendation

**For your first training endeavor: start with Matcha-TTS.** Given you're new to training,
cost-conscious, and have said re-coding ProsodiaActor is acceptable, Matcha wins on the axes that
actually carry first-run risk: **stable single-stage training, an official export path that matches
our proven `torch→ONNX` backbone, mobile-friendly ops, and MIT licensing.** StyleTTS2's main
advantage — its native style space — is offset by the fact that our **VAT directability is
something we build regardless**, so it ports to Matcha; and its training (adversarial, multi-stage)
is a genuinely hard place to *learn* model training.

**Keep StyleTTS2-Lite as the higher-ceiling Phase 2.** If, after shipping a Matcha-based actor,
naturalness or style-control quality falls short, revisit the StyleTTS2 re-architecture with the
training experience you'll have gained. Don't make the hardest, most adversarial model your *first*
training project.

**Independent of model choice — de-GPL the G2P.** Both StyleTTS2 and Matcha default to
`phonemizer` → **espeak-ng (GPL-3.0)**. That, not the acoustic model, is the live risk to a
permissive core. The Rust G2P (`crates/actor/src/g2p.rs`, `lexicon.rs`) on a permissive lexicon
(CMUdict is BSD-style) is now in place so espeak-ng is out of the shipping path. Training data is
fine to ship from (LibriTTS = CC-BY-4.0, LJSpeech = public domain).

## License verification (GitHub API, 2026-06-14)

| Project | License | Verdict |
|---|---|---|
| StyleTTS2, PL-BERT, HiFi-GAN, **Matcha-TTS**, VITS, Piper, MeloTTS, F5-TTS | **MIT** | ✅ |
| Kokoro | **Apache-2.0** | ✅ (but LSTM architecture) |
| espeak-ng | **GPL-3.0** | ⚠️ G2P only — replace before shipping |
| Coqui **XTTS** | repo MPL-2.0 / **weights CPML non-commercial** | ❌ avoid |
| fish-speech | custom / NOASSERTION | ❌ avoid for permissive |

> Repo license ≠ model-weights license (XTTS is the classic trap). Always check the *weights* and
> the *training-data* terms separately, not just the code repo's SPDX tag.

---
---

# PART 2 — TRAINING GUIDE (first-time trainer)

Covers **both model paths** (Matcha-TTS · StyleTTS2-Lite) and **both hardware paths** (RunPod
NVIDIA · Strix Halo 395 AMD).

## 0. Training 101 (the mental model)

Neural TTS training = show the model thousands of *(text, audio)* pairs and adjust its weights until
its predicted audio matches the real audio. Concretely you need:

- **A dataset:** audio clips + exact transcripts. Quality/consistency matters more than raw size.
  Single-speaker fine-tunes work from ~1–10 hours; multi-speaker from tens–hundreds of hours
  (LibriTTS ≈ 585 h). Audio must be cleaned, trimmed, resampled to the model's rate (24 kHz here),
  and the text **phonemized** (G2P).
- **A base checkpoint (usually):** you rarely train from zero. **Fine-tuning** a pretrained model on
  your data is faster, cheaper, and far more forgiving than **from-scratch** — do this first.
- **A GPU:** training is thousands of forward/backward passes; throughput (compute) is the usual
  bottleneck, not memory, for these small models.
- **A loss going down + your ears:** you watch loss curves (TensorBoard) and periodically synthesize
  samples to judge quality. TTS is judged by listening, not just metrics.
- **Iterations:** expect several runs (a bad LR, a data bug, over/under-training). Cheap, fast,
  reproducible iteration is the single most important property of your setup — which is why
  toolchain reliability dominates the hardware decision below.

Key vocabulary: **epoch** (one pass over the data), **batch size** (clips per step; limited by VRAM),
**learning rate** (step size; the #1 thing newbies get wrong — too high = divergence), **checkpoint**
(saved weights), **vocoder** (mel-spectrogram → waveform; often a separate pretrained model).

## 1. Model-path training impact

### Path A — Matcha-TTS (recommended first endeavor)

- **Pipeline:** one model (acoustic) + a **separate pretrained HiFi-GAN vocoder** (you usually don't
  train the vocoder — use a universal one). Training is **single-stage**: minimize flow-matching loss
  + duration loss + prior loss. **No GAN, no adversarial instability.**
- **Tooling:** PyTorch-Lightning + Hydra configs. You edit a YAML (data paths, batch size, speaker
  count), run one `train.py`, watch TensorBoard. Pretrained checkpoints + an **official ONNX
  exporter** (`python -m matcha.onnx.export`) are provided.
- **Data prep:** filelists of `wav|text|speaker`, 22.05/24 kHz mono, `phonemizer` for G2P (espeak
  backend — note the GPL caveat; swap to our permissive G2P later). Monotonic-alignment is a CPU
  Cython extension (builds easily).
- **VRAM / time (rough, single GPU):** fine-tune fits comfortably in **≤ 12–16 GB**; a single-speaker
  fine-tune converges in **hours**; from-scratch multi-speaker in **1–3 days**.
- **Difficulty for a newbie:** **Low–moderate.** Stable, few moving parts, good docs, hard to blow up.
- **Our extra work (not part of vanilla training):** add VAT/style conditioning for directability and
  re-derive the casting/blend layer. Do this *after* a plain Matcha fine-tune sings — learn the
  pipeline first, then extend.

### Path B — StyleTTS2-Lite (higher ceiling, harder)

- **Pipeline:** **two stages.** Stage 1 = acoustic/alignment (text encoder, duration). Stage 2 =
  adversarial waveform + style training with **GAN discriminators** (+ a heavy WavLM "SLM" loss in
  vanilla, which the Lite plan strips). Adversarial training is **unstable** — sensitive to LR
  balance, prone to divergence/mode-collapse; debugging it is an art.
- **Our Lite re-architecture adds research risk:** replace LSTMs (transformer/conv), strip
  WavLM/discriminators, prune PL-BERT, swap diffusion → deterministic VAT style predictor. You are
  modifying a complex training codebase *and* changing the architecture — many failure modes.
- **VRAM / time:** designed for a **24 GB** card; the WavLM cut (Lite) drops that ~75%. A real
  from-scratch run is **multi-day** with retries.
- **Difficulty for a newbie:** **High.** Adversarial + multi-stage + custom architecture is a brutal
  first project. Best attempted once you've shipped one model end-to-end. Full design:
  [high-ambition-5-styletts2-lite.md](high-ambition-5-styletts2-lite.md).

**Takeaway:** Matcha is the gentle on-ramp; StyleTTS2-Lite is the advanced course. Learn on Matcha.

## 2. Hardware-path: RunPod (NVIDIA) vs Strix Halo 395 (AMD)

Short version: **for a first run, use RunPod NVIDIA. Use the Strix Halo for inference (it's superb
at that), and revisit it for training only once you're experienced and ROCm support for it has
matured.**

### Path 1 — RunPod, single NVIDIA GPU (recommended for first training)

- **Why:** every TTS repo targets **CUDA**; tutorials, Docker images, and custom ops "just work."
  Zero driver/toolchain yak-shaving — you rent a ready PyTorch+CUDA box and run.
- **What to rent:** a single **RTX 4090 (24 GB)** is plenty for both Matcha and StyleTTS2-Lite (the
  StyleTTS2 plan was explicitly designed around 24 GB). No need for A100/H100 for these small models.
- **Cost (ballpark, verify live):** community 4090 ≈ **$0.34–0.69/hr**. So:
  - Matcha single-speaker fine-tune (a few hours): **~$2–10**.
  - Matcha from-scratch multi-speaker (1–3 days): **~$10–50**.
  - StyleTTS2-Lite from-scratch + retries: **~$30–150**.
  - **Your "expensive route" fear is largely unfounded for small TTS models on RunPod** — a first
    experiment is tens of dollars, and you only pay while it runs. Use **persistent volumes** so you
    don't re-download data each session, and **spot/community** instances for non-critical runs.
- **Workflow:** rent → attach a network volume → upload dataset → `pip install -r requirements` →
  edit config → `train.py` → watch TensorBoard (port-forwarded) → download checkpoint → stop the pod
  (stop paying). Snapshot the volume so the next session is instant.
- **Newbie friction:** Low. The main discipline is **remembering to stop the pod**.

### Path 2 — Strix Halo 395 Max+ (Ryzen AI Max+ 395), 128 GB, Ubuntu 26.04 LTS [Plan B]

> 💡 **Plan B Status (Updated 2026-07-09):** The local server hosting this session has been evaluated. It runs **Ubuntu 26.04 LTS** with **ROCm 7.2.4** pre-installed and active, detecting the `gfx1151` (Radeon 8060S iGPU) with **112 GB** of allocatable VRAM pool. **Docker** is active and runs without password prompts, making local container-based training highly feasible.

- **The chip:** Zen 5 (16 cores) + **Radeon 8060S iGPU (RDNA 3.5, 40 CUs)** + XDNA2 NPU (~50 TOPS), with **128 GB unified LPDDR5X** (~256 GB/s) shared CPU/iGPU.
- **The good:**
  - **Zero cloud cost:** Free, local, unlimited run time.
  - **Vast memory overhead:** ROCm exposes up to **112 GB** of memory as allocatable VRAM pool. We can scale batch sizes (e.g., batch size 64+) to maximize hardware saturation.
  - **No distro friction:** The host is Ubuntu 26.04, which is AMD's tier-1 target OS (unlike the previously assumed Fedora).
- **The caveats (why it remains Plan B):**
  - **Python 3.14 system host:** The host's system Python is version 3.14. PyTorch does not officially distribute stable ROCm wheel binaries for Python 3.14 yet.
  - **Docker is mandatory:** To train locally, we *must* run inside the official `rocm/pytorch` container (e.g., `rocm/pytorch:latest` or `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release`) mapping `/dev/kfd` and `/dev/dri`. This bypasses the Python 3.14 host limitation, providing a preconfigured Python 3.12 + PyTorch + ROCm environment where Matcha's Cython-based components (like `monotonic_align`) compile cleanly.
  - **Compute Speed:** The RDNA 3.5 iGPU lacks dedicated tensor cores of NVIDIA discrete GPUs and shares its 256 GB/s bandwidth with the CPU. Per-step training will be several times slower than a rented RTX 4090 (Plan A). However, since Matcha-TTS is very small (~18M params) and CFM training is single-stage/fast, a full fine-tune is estimated to complete in 12–24 hours, making local training perfectly practical.
- **Local Container Launch Command:**
  ```bash
  docker run -it --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=video --ipc=host \
    --shm-size 8G \
    --security-opt seccomp=unconfined \
    -v $(pwd):/workspace \
    rocm/pytorch:latest
  ```

### Hardware decision table

| | RunPod 4090 (NVIDIA/CUDA) [Plan A] | Strix Halo 395 (AMD/ROCm) [Plan B] |
|---|---|---|
| Toolchain reliability | ✅ Everything targets CUDA out-of-the-box | ⚠️ Requires Docker mapping for ROCm due to Python 3.14 host |
| First-run friction | ✅ Low | ⚠️ Moderate (Docker run command, mounting workspace) |
| Training compute speed | ✅ Fast (dedicated tensor cores, 1008 GB/s) | ⚠️ Slower (integrated GPU, 256 GB/s) |
| Memory for training | 24 GB (ample for TTS) | ✅✅ 112 GB allocatable (huge capacity, zero OOM) |
| Cost | 💲 ~$2–150 per experiment, pay-per-use | 🆓 $0 (local hardware, power only) |
| **Inference** (incl. Director) | Rented, costs while running | ✅✅ **Outstanding** — 128 GB local |
| Recommended for first training?| ✅ **yes (Plan A)** | ⚠️ **As Plan B (local alternative)** |

## 2b. Inference vs training — and which machine does what

### Two kinds of "inference" (don't conflate them)

| | **Dev / testing** (host & experiment) | **Production on-device** (ships in the app) |
|---|---|---|
| Director (Gemma 4 E2B) | GGUF via **Ollama / llama.cpp** | **`.litertlm` via LiteRT-LM C-API**, embedded |
| Actor (TTS) | n/a — Ollama is LLM-only | **TFLite / ONNX** via LiteRT / ONNX Runtime, embedded |
| Purpose | pick the model, test prompts/behavior, eval | validate the *actual shipping artifact + runtime* |

> **Gotcha:** "it works in Ollama" ≠ "it works on-device." Ollama runs a GGUF model through a
> different runtime + quantization than the `.litertlm`/TFLite that ships. Ollama validates **model
> choice and behavior**; only the embedded LiteRT path validates the **on-device artifact**. Use both;
> don't let the first stand in for the second. (The TTS actor isn't an Ollama thing at all.)

### M1 Max (32 GB) vs Strix Halo 395 (128 GB) — for *this* project

The M1 Max you're already testing on is the better machine for the core work; the Strix Halo is a
**capacity** box, not a speed upgrade.

- **M1 Max 32 GB — your primary dev/test machine.** It *builds and runs the iOS/macOS apps + the
  Tuner* (the Strix Halo can't). Mature Apple ML tooling (Metal, MLX, Core ML, llama.cpp Metal) and
  ~400 GB/s memory bandwidth (higher than the Strix Halo's ~256 GB/s), so for models that fit it is
  often **faster**. And everything Prosodia *ships* fits comfortably: Gemma E2B ≈ 1.5–3 GB quantized,
  the TTS actor ≈ 0.1–0.3 GB. **For the core models, 32 GB is sufficient and preferable.**
- **Strix Halo 128 GB — earns its keep only for things that don't fit in 32 GB:**
  - hosting **big local LLMs (70B-class)** for **synthetic-data generation / labeling / evaluation**
    (e.g. generating or annotating training transcripts) — far cheaper than cloud APIs;
  - **evaluating larger Director candidates** before committing to the small on-device E2B;
  - running **several models at once** or **large contexts**;
  - a **non-Apple x86/Linux validation target** for the cross-platform (Linux/Windows/Android) builds.
- It is **not faster** (modest iGPU compute) and **can't build the Apple apps**. Its single advantage
  is memory *capacity*.

**Bottom line:** keep developing/testing on the **M1 Max** — it covers all of Prosodia's small
on-device models and is the Apple build target. Reach for the **Strix Halo** specifically when you
want big-model local hosting (eval / synthetic data) or to validate the non-Apple path. Neither is
your *training* machine — that's RunPod for the first run.

## 3. Recommended first-run plan

1. **Decide the model** (Part 1) → default **Matcha-TTS**.
2. **Setup the training platform**:
   - **Plan A (Recommended):** Rent a RunPod 4090, attach a persistent volume.
   - **Plan B (Alternative):** Setup local Docker environment mapping `/dev/kfd` and `/dev/dri` with `rocm/pytorch`.
3. **Fine-tune** Matcha on a small clean dataset (or a single LibriTTS speaker) to *learn the loop* — watch loss, synthesize samples, get a feel for over/under-training. (Cost: a few dollars on Plan A, free on Plan B).
4. **Export** with the official ONNX exporter → confirm `torch→ONNX` (already validated as our backbone) → then exercise the **ONNX→TFLite (`onnx2tf`)** leg (the one remaining unproven step) to land a real on-device `.tflite`.
5. **Only then** add the hard, custom parts: VAT/style directability conditioning, the casting/blend layer, and (if quality demands it) graduate to the StyleTTS2-Lite re-architecture.
6. Use the **Strix Halo for local inference/labeling** (Plan A) or **local training + inference** (Plan B) — including hosting the Gemma 4 Director.

**Guiding principle:** make your *first* success as boring and reliable as possible (Matcha + CUDA +
fine-tune). Spend novelty budget on the directability research and the on-device integration, not on
fighting an adversarial GAN or an immature GPU stack.

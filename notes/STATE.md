# Project State — Project Sonora

_Last updated: 2026-07-14._

The committed, curated snapshot of where Project Sonora (TTS training, model, and dataset prep) stands and what to do next. Behavioral rules and the stack/layout manifest live in [AGENTS.md](../../AGENTS.md).

---

## Current State

* **Workspace layout: `Models/` → `Reference/models/`; Sonora registry clone → `Sonora/model/` (2026-07-13):**
  the shared models dir moved under `Reference/` at the Mac workspace root to make clear these are
  reference assets, not workspaces. The artificial-humanity/Sonora HF clone — our own artifacts, not
  a reference model — moved out of it, from `Models/Sonora/` to `Sonora/model/` inside the Sonora
  project dir (untracked by the Sonora GitHub repo). Dated bullets below predate the move: read
  `Models/...` as `Reference/models/...` and `Models/Sonora/` as `Sonora/model/`. `ai-lab-0`
  mirrored the full pattern the same day: `/data/Models` → `/data/models` (symlinked as
  `Reference/models` in its workspace checkout), registry clone moved into its Sonora checkout's
  `model/`. Note the lowercase `models` — the box set the convention and the Mac followed.
  **Layout finalized later the same day (end-state C):** the registry clone moved once more,
  `Sonora/model/` → workspace-root **`Registry/Sonora/`** (sibling of `Reference/`; kills the
  repo-in-repo nesting; see [registry-housekeeping.md](registry-housekeeping.md)) — done on
  both machines the same day.
  Prosodia updated in lockstep: `prosodia_models.json` `modelsBase`, Swift fallback paths, `engine.rs` test paths,
  tuner README canonical listing. Same day, the staged actor file was **renamed
  `styletts2_lite.tflite` → `sonora.tflite`** — it is a Matcha-architecture Sonora model; the old
  name was a fossil from when StyleTTS2-Lite was the planned actor, and it would collide with the
  real StyleTTS2-Lite re-platform (high-ambition-5) later. Older bullets keep the old filename.
* **Phase 0 Baseline Training Completed (2026-07-11):** 
  * The baseline training run (`lyrical-goat-689`) completed successfully. The model was trained continuously for 12 hours inside the `sonora_training` container, reaching **Epoch 260** (step 101,599). 
  * Learning curves and loss metrics logged to the local MLflow server confirmed optimal convergence around **step 77,410** (approx. Epoch 188). 
  * Stopped the active GPU training container to conserve system resources.
* **Vocoder-Embedded ONNX & TFLite Models Exported (2026-07-11):**
  * Exported the Epoch 199 checkpoint (`checkpoint_epoch=199.ckpt`, matching the optimal convergence point) to ONNX with the `hifigan_T2_v1` vocoder embedded.
  * Compiled the ONNX graph to Float32 and Float16 TFLite formats using `onnx2tf` inside the `sonora_playpen` container.
  * Models generated:
    * `checkpoint_epoch=199_e2e_float32.tflite` (178 MB) — High-precision float model.
    * `checkpoint_epoch=199_e2e_float16.tflite` (89.8 MB) — Quantized, NPU/mobile-optimized model.
* **Hugging Face Model Registry Configured (2026-07-11):**
  * Created the central model storage repository on Hugging Face.
  * Successfully uploaded all Phase 0 TFLite models, full checkpoints, and the tensor mapping manifest to **[artificial-humanity/Sonora](https://huggingface.co/artificial-humanity/Sonora)** under the `v1-ljspeech` directory.
  * **Local model layout reorganized (2026-07-11):** in the workspace-root `Models/` folder (gitignored; canonical listing in `Prosodia/apps/tuner/README.md`), the vendored Matcha-TTS source + export artifacts were renamed `Matcha-TTS` → **`shivammehta25/Matcha-TTS/` (spike workspace — see its ARCHIVE.md)**, and two HF repos were cloned alongside: **`Models/Sonora/`** (our artificial-humanity/Sonora registry, LFS hydrated) and **`Models/litert-community/Matcha-TTS/`** (litert-community/Matcha-TTS — the split-graph TFLite + espeak-free G2P assets adopted in roadmap item 2).
  * **Hugging Face Pro Resources Deployed:** Configured the high-capacity storage bucket **[artificial-humanity/Prosodia-Storage](https://huggingface.co/buckets/artificial-humanity/Prosodia-Storage)** and the hosted Gradio Space **[artificial-humanity/Prosodia](https://huggingface.co/spaces/artificial-humanity/Prosodia)** for remote sample verification.
* **✅ Export-fidelity bug fixed & artifacts re-exported (2026-07-12):** the 2026-07-11 "TFLite
  garble" was root-caused to an encoder-LayerNorm axis mislowering in `onnx2tf` (details in Next
  Steps §1) and fixed at the source (Sonora commit `a537e03`). Re-exported float32/float16 TFLite +
  the ONNX oracle now pass a deterministic fidelity gate (cosine = 1.0000). New artifacts committed
  to the local HF clone (`Models/Sonora`, commit `cb18bcf`) — **push to `artificial-humanity/Sonora` still
  pending an HF write token** (none available in the remote session). *(Resolved by 2026-07-13:
  the registry is pushed through the split-graph lane commit; model card + provenance convention +
  engine `config.json` added the same day — see
  [registry-housekeeping.md](registry-housekeeping.md).)*
* **✅ Fixed artifacts verified end-to-end, server-side (2026-07-12, same session as the fix):**
  * **Multi-sentence ASR (stochastic, temp 0.667):** 8 short sentences rendered through the fixed
    float32 TFLite with matcha-CLI input parity (`english_cleaners2` + `intersperse` blanks) →
    **WER 0.000, every transcript verbatim** (faster-whisper base.en). The ONNX itself scored 0.125
    (one whisper quirk on "Hello world"). Key input detail rediscovered: the e2e graph expects
    **blank-interspersed ids** (`intersperse(ids, 0)`), so the usable text budget is ~24 raw phonemes
    against the 50-slot graph.
  * **fp16 caveat closed — weights-only-fp16 / f32-I/O artifact produced:** `onnx2tf -fdosm` (needs
    `tf_keras`) emits a SavedModel from the corrected IR; TF post-training float16 quantization then
    yields **59.4 MB** (vs 94 MB onnx2tf-fp16, 187 MB float32) with **all I/O tensors f32** — but TF
    mangles tensor names (`serving_default_x:0`), which breaks the engine's exact-name matching
    (`engine.rs`: `name == "x"` / `name == "scales"`). Fixed via flatbuffer round-trip rename
    (`Sonora/scripts/rename_tflite_tensors.py`). Renamed artifact: referee **PASS (deterministic
    cosine 0.9991)**, ASR parity with the ONNX (mean WER 0.125, identical miss).
  * **Espeak-free G2P validated against the locked vocab (litert-community dict):** the roadmap's
    "mapping/validation pass" is done — **99.997% of the 274,927-entry OpenPhonemizer dictionary
    phonemizes entirely into our locked 178-symbol vocab**; the only uncovered symbol is a combining
    nasal tilde (U+0303) in 7 French loanwords (map/strip rule needed, nothing else). `ᵻ` (41,851
    uses) is in our vocab; the dict never emits `ᵊ`/`'`/`ʼ`. **Also confirmed: Sonora's training
    `symbols.py` IS the locked vocab** (178 unique, `'` deduped, `ᵊ` present) — Phase 0 trained on
    the locked inventory, so no train/runtime vocab divergence exists. The litert-community config
    is stock keithito (177 unique + duplicate `'`, no `ᵊ`) — only delta.
  * Cloned **litert-community/Matcha-TTS** into `Models/litert-community/Matcha-TTS/` (split graphs
    textenc/decoder/vocoder fp16, G2P dict + DeepPhonemizer TFLite + meta, samples).
  * Staged for the Rust engine test: `Models/styletts2_lite.tflite` (= fixed float32) and
    `Models/config.json` (locked 178-symbol vocab, `sample_rate` 22050, generated from
    `matcha/text/symbols.py`).
  * **Rust engine end-to-end verified on Linux (`ai-lab-0`):** built `libtensorflowlite_c.so` from
    TF v2.21.0 source (now at `/data/toolchain/`; cmake needs `-DTENSORFLOW_SOURCE_DIR=<checkout>`
    or it silently fetches a second, older TF — plus `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` and
    XNNPACK off), linked the actor crate via
    `RUSTFLAGS="-L native=/data/toolchain -l dylib=tensorflowlite_c"`, and ran
    `test_reference_ids_render_tmp` against the staged fixed model: **test passed, and ASR on the
    engine-rendered WAV = `'the morning light.'` verbatim** — the shipping Rust path
    (`engine.rs::forward_impl` → TFLite C API → resample → sink format) is confirmed against the
    fixed artifact. *Open observation:* the Rust render is 1.18 s vs 0.57 s for the same ids from
    Python (nothing spurious is audible per ASR; possibly engine-side duration scaling or an
    untrimmed low-energy tail) — check pace by ear at the desktop audition. *(✅ Resolved
    2026-07-13: desktop audition verdict — pacing is fine by ear. Most likely explanation: the
    engine sets `length_scale = 1/speed` from the preset's speed multiplier, so differing speed
    values, not an engine bug, account for the duration delta.)*
  * **Ops footgun (2026-07-12):** `sonora_playpen` was *recreated* mid-session (fresh rw layer:
    `/tmp` artifacts and all pip installs lost). Copy artifacts out to `/data/Models/...`
    immediately after producing them; expect to reinstall the export toolchain
    (`onnx2tf[tensorflow]`, `tf_keras`, `onnxruntime`, `ai-edge-litert`, `faster-whisper`) after any
    recreation.
  * **Toolchain pin — `onnx2tf==2.5.2` for SavedModel export:** onnx2tf **2.6.1 regresses
    `-fdosm`** (rank error in `tflite_builder/saved_model_exporter.py::_kernel_transpose`,
    "Dimension must be 5 but is 4"). The fp16w-f32io artifact was regenerated post-recreation with
    2.5.2 (referee PASS 0.9991 again) and **pushed to HF as
    `checkpoint_epoch=199_e2e_fp16weights_f32io.tflite`** (commit `2b9afc6`) — now the
    mobile-recommended artifact on the model card.
* **✅ LiteRT side path MATERIALIZED — Epoch-199 split graphs at parity (2026-07-12 evening):** the
  reserve litert-torch path is no longer theoretical. Using the conversion harness from
  [google-ai-edge/litert-samples](https://github.com/google-ai-edge/litert-samples)
  (`compiled_model_api/text_to_speech/conversion/`, cloned + adapted at
  `/data/toolchain/litert-conversion/` with a uv venv), **our own `checkpoint_epoch=199.ckpt`
  converted first-try** once the environment was tamed:
  * **Per-graph fidelity: corr 1.000000** (textenc mu+logw, decoder, vocoder) — litert-community's
    exact bar. **All three graphs GPU-clean.**
  * **End-to-end fp16 waveform corr vs torch: 0.99963–0.99994** on four real sentences (host-side
    Euler ODE ×10, our fine-tune's own mel stats −5.5366/2.1161); ASR on the rendered samples is
    verbatim except whisper mishearing the word "Matcha". **Human-ear audition passed (2026-07-12):**
    the user listened to the rendered samples remotely and confirmed they sound good — the first
    by-ear validation of the trained voice through any export path. (The desktop Tuner audition
    remains for the full app path, but voice quality itself is now confirmed.)
  * **Artifacts** (✅ pushed to HF 2026-07-12, commit `b1a60b7` — [v1-ljspeech/litert-split](https://huggingface.co/artificial-humanity/Sonora/tree/main/v1-ljspeech/litert-split); `*.wav` added to the repo's LFS tracking en route):
    `matcha_textenc_fp16.tflite` (14.8 MB), `matcha_decoder_fp16.tflite` (22.5 MB),
    `matcha_vocoder_fp16.tflite` (29.0 MB), `emb.bin` (178×192 f32 host lookup), `config.json`.
  * **Parity vs the ONNX path is transitive, not bitwise:** both artifacts verify ≥0.9996 against
    the *same* torch model; direct waveform diff is meaningless (10 vs 5 baked ODE steps,
    length_scale 0.95 vs 1.0).
  * **Environment recipe** (for reproduction; full log `/data/toolchain/litert-conversion/uv_setup.log`):
    py3.12 uv venv; `litert-torch ai-edge-litert ai-edge-quantizer`; pure-python deps
    (`einops conformer diffusers phonemizer deep-phonemizer unidecode inflect omegaconf matplotlib`);
    **`scipy==1.14.1`** (1.18 propack breaks past `_stub.py`'s guard); matcha imported from the
    **Sonora checkout via PYTHONPATH** (pip `matcha-tts` sdist needs distutils, gone in py3.12); our
    hydra-era checkpoint needs `omegaconf` to unpickle; host `espeak-ng` for the parity harness's
    reference G2P only.
  * **What this unlocks:** `logw` is now a host-visible output → the exploit-before-train
    measurement can run on these graphs; per-graph delegate placement (their Pixel 8a recipe applies
    directly); fp16 split total 66 MB. **Remaining gap for on-device use:** the Prosodia Rust
    multi-graph runtime path (three interpreters + host ODE/length-regulator) — same prerequisite
    the onnx2tf split would need.
* **Models/ restructured to org/repo layout (2026-07-12 late):** the shared `Models/` dir now
  follows GitHub/HF `org/repo` naming — `Google/` (Gemma Director models — ⚠️ raises Debt F: the
  Swift harness hard-codes them at root), `shivammehta25/Matcha-TTS/` (the June export-spike
  workspace, labeled by an `ARCHIVE.md` — not a clean clone), `litert-community/Matcha-TTS/`,
  `Sonora/` (our HF registry clone), plus new academic pulls `IIEleven11/StyleTTS2FineTune/`,
  `semidark/StyleTTS2/`, `semidark/kikiri-tts/` (StyleTTS2/Kokoro side discussions — reference
  only for now). `config.json` + `styletts2_lite.tflite` stay at root (engine contract).
  `matcha_stock.tflite` no longer exists (optional Rust test skips it). Canonical listing:
  `Prosodia/apps/tuner/README.md`.
* **✅ Export-path priority REVERSED (2026-07-12, after the split-graph parity results):** the
  `litert-torch` **split-graph path is now Plan A** — the primary export path for all future
  checkpoints (VAT fine-tunes included) — and the monolithic `torch → ONNX → onnx2tf` e2e graph is
  **demoted to Plan B**: the maintained fallback and the desktop-Tuner audition artifact until the
  Prosodia Rust multi-graph runtime lands. Rationale: the split's advantages are structural
  (host-visible `logw` → `duration_scales`/`f0_bias` hooks + exploit-before-train measurement, no
  50-token static limit, per-graph delegate placement, tunable ODE steps, 66 MB fp16), whereas the
  monolith's parity win only proved it *correct*, not *equal*. The ONNX stage remains the numerical
  oracle for any onnx2tf export; the LiteRT/TFLite runtime is unchanged either way. Docs updated in
  lockstep: [Prosodia next-steps.md](../Prosodia/next-steps.md) (📌 callout + §B),
  [architecture-north-star.md](../Prosodia/architecture-north-star.md), roadmap §§1–2 below.
* **Local `Models/` layout on `ai-lab-0` (2026-07-12):** created `/data/Models` (owned by lmcfarlin)
  and symlinked it as `Models/` in the repo root (gitignored), mirroring the Mac layout. Cloned the
  HF registry into `Models/Sonora/` (git-lfs installed + hydrated). The vocoder checkpoint for
  re-export lives in the `sonora_playpen` container at `/root/.local/share/matcha_tts/hifigan_T2_v1`.
* **Audio-loading hardware bypass complete:** Replaced `torchaudio.load()` calls in the dataloader (`matcha/data/text_mel_datamodule.py`) with hardware-agnostic `soundfile.read()` to bypass the ROCm/CUDA library mismatch (`torchcodec` loading error) inside the container.
* **Matplotlib 3.9+ compatibility fix deployed:** Updated `save_figure_to_numpy` in `matcha/utils/utils.py` to use `np.asarray(canvas.buffer_rgba())[:, :, :3]`, avoiding the deprecated and removed `tostring_rgb()` method.
* **Lightweight checkpoint playpen web UI deployed:** Deployed a custom Gradio web application `playpen.py` inside the `sonora_playpen` container mapping port `7862`. It automatically discovers training checkpoints in `/workspace/logs/`, runs fast CPU inference, and streams synthetic audio to the browser via `https://sonora-playpen.ai-lab-0.mcfarlin.family`.
* **OpenAI-Compatible TTS API endpoints added:** Exposed standard REST routes (`/v1/models`, `/v1/voices`, `/v1/audio/speech`) from the playpen FastAPI server, allowing direct integration of Sonora checkpoints as custom voice backends inside Open WebUI.

---

## Next Steps & Roadmap

> **Immediate objectives (as of 2026-07-13):**
> 1. ✅ **Desktop pair DONE (2026-07-13):** Debt F build-checked (both app targets, several times
>    over, during the desktop session) and the **Tuner audition passed** — intelligible speech,
>    **pacing fine by ear** (closing the 1.18 s-vs-0.57 s question: the engine's
>    `length_scale = 1/speed` mapping explains it; not an engine bug). Voice identity is the
>    LJSpeech narrator, as expected for the Phase 0 single-speaker fine-tune. Caveat observed and
>    documented: Play→first-audio was ~a minute+ by construction of the monolithic e2e path
>    (whole-span render before any audio; fixed-shape cost ≈10–15 s/chunk on the M1 Max).
>    *Interim relief landed same day (Prosodia `104a9c8`): XNNPACK delegate ≈5× per chunk
>    (~2.9 s) + actor warm-up at app launch; the structural fix (streaming, proportional
>    compute) stays with the multi-graph runtime — validation item in Prosodia next-steps §B.*
>    Also fixed en route (`846a22c`): preset edits (speed dial, VAD, volume, casting) made after
>    the first Speak were silently discarded by a stale director cache — ear-verified fixed.
> 2. ✅ **Exploit-before-train measurement DONE (2026-07-14):** run on `ai-lab-0` against the
>    Epoch-199 litert split graphs (`exploit_measure.py` in the conversion harness; full results
>    in [exploit-before-train-measurement.md](exploit-before-train-measurement.md)). Headline:
>    **pace and loudness are free at inference** — per-token `duration_scales` through host
>    `logw` is surgical (phrase-local ×1.5–2.0, context drift ≤1.4%, WER 0, Spearman ρ = 1.0),
>    and a per-frame dB bias on the log-mel before the vocoder is dB-exact (−6 req → −6.04 meas,
>    zero context bleed, WER 0 at −12 dB). **Pitch and phonation are not**: no F0 input exists
>    (mel-bin roll buys ~−1.5 st before artifacts), no breathiness lever at all. This scopes
>    milestone 3: VAT training must own pitch + voice quality; it need not carry pace/energy.
> 3. **VAT directability (milestone 3) — conditioning code SHIPPED, de-risk phase RUNNING
>    (2026-07-14):** FiLM conditioning landed (Sonora `ad2baea`, design in
>    [vat-conditioning-design.md](../Sonora/vat-conditioning-design.md), identity-at-init +
>    export gates green); de-risk corpus derived (30,351 clips / 45.7 h / 247 spk @ native
>    24 kHz, energy labels — `derive_vat_corpus.py`); **HiFi-GAN 24 kHz/80-band vocoder
>    fine-tune TRAINING NOW** (`vocoder_training` container, warm from UNIVERSAL_V1; see
>    [sample-rate-24khz-decision.md](sample-rate-24khz-decision.md)); §7 de-risk acoustic run
>    queued behind it (`sonora_training` compose command preset; warm-start ckpt at
>    `/data/model-training/sonora/warmstart/derisk_energy_init.ckpt`). Verdict via
>    `scripts/eval_harness.py` (ρ ≥ 0.9, leakage ≤ 0.2, WER Δ ≤ +0.10). Corpus survey:
>    [dataset-landscape.md](dataset-landscape.md).
> 4. Small: bucket hygiene (replace the Jul-11 broken TFLites in Prosodia-Storage), uv items
>    3.5–3.8, re-run the litert conversion harness against any future checkpoint.
> 5. Deferred housekeeping (info gathered, nothing decided): registry/artifact-flow tidy-up —
>    model card, per-promotion provenance, `config.json` into the registry, revision pinning,
>    and the layout question — **settled same day: end-state C, workspace-root
>    `Registry/Sonora/`**, implemented on both machines. See
>    [registry-housekeeping.md](registry-housekeeping.md).


### 1. Integration & Audition Verification (Project Prosodia)
* **Goal:** Pull the exported TFLite models into your local laptop environment (where the Apple/Android reader apps run).
* **✅ EXPORT FIDELITY FIXED (2026-07-12) — the audition blocker is resolved.** The 2026-07-11
  garble was root-caused, not worked around: a per-op `onnx2tf -cotof` report localized **all 95
  diverging ops to the text encoder's LayerNorm** (decoder + vocoder converted perfectly, 0
  diverging). Matcha's encoder `LayerNorm` normalizes over the **channel axis** (`dim=1`) of a
  `[B, C, T]` tensor via a manual mean/rsqrt; `onnx2tf`'s NHWC layout pass mislowered that `dim=1`
  reduction (classic 3D-tensor axis ambiguity), corrupting the encoder output. **Fix:** rewrote that
  LayerNorm as a mathematically-identical **channels-last** norm (`transpose → last-axis
  `F.layer_norm` → transpose`) in `matcha/models/components/text_encoder.py` (Sonora commit
  `a537e03`), then re-exported. Post-fix: `onnx2tf` op report **559/559 pass** (was 361/457) and
  deterministic (temperature=0, RNG-free) **ONNX-vs-TFLite waveform cosine = 1.0000** (was 0.045).
  The `torch → ONNX → onnx2tf` backbone stands — **no split-graph or litert-torch rewrite was needed
  for the unblock.** *(Later the same day the split-graph path was nonetheless promoted to Plan A on
  its structural merits — see §2; the onnx2tf monolith remains the fallback and current audition
  artifact.)*
  * **Standing fidelity gate:** `Sonora/scripts/export_fidelity_referee.py` — a real-input
    ONNX-vs-TFLite correlation check (do **not** trust `onnx2tf`'s own `-cotof` self-report; it skips
    nondeterministic ops and emitted a false `cosine=1 pass=True` after its validator crashed on an
    fd limit). Note the graph's `RandomNormalLike` (CFM decoder noise) makes end-to-end cosine
    meaningless at temperature > 0 — use `--temperature 0` for a bit-comparable fidelity number,
    `--asr` for an intelligibility spot-check.
  * Runtime fixes landed en route and remain valid: audio-sink drain before `.finished`,
    static-limit chunking in `process_and_synthesize`, and Matcha blank interspersion in `tokenize`
    (`add_blank=True` parity — required for any Matcha checkpoint).
* **Remaining before we can hear it:** re-exported artifacts pushed to HF (see below), then a real
  **listen through the desktop Tuner** at temperature > 0 (deferred — needs the Mac/desktop app, not
  available in the current remote/iPad session). Fidelity is already proven deterministically; the
  Tuner pass confirms intelligibility of a stochastic render.
* **Staged 2026-07-11 (contract pre-checked against `engine.rs::forward_impl`):**
  * `Models/styletts2_lite.tflite` = copy of **`checkpoint_epoch=199_e2e_float32.tflite`**. Signature verified: `x` i64[1,50] / `x_lengths` i64[1] / `scales` f32[2] → `wav` f32[1,262144] + `wav_lengths` i32[1] — matches the engine's `is_matcha` detection and all dtype assumptions.
  * **Do NOT audition the float16 export on the current engine:** its `scales` input and `wav` output are f16, and the engine reads/writes both as f32 — silent garbage (mangled temperature/length-scale in, static out), no error raised. Needs engine f16 I/O support or a re-export with f32 I/O (weights-only fp16); folded into the split-graph re-export task.
  * **`Models/config.json` `sample_rate` corrected 24000 → 22050.** The 24000 value anticipated the Gate 2 24 kHz fine-tune, but Phase 0 is LJSpeech + `hifigan_T2_v1` at 22.05 kHz; the engine trusts the adjacent config, so 24000 would skip resampling and play ~9% fast / +1.4 semitones. Revert to 24000 only when a genuine 24 kHz model ships.
  * **Known constraint:** the e2e export bakes a **50-token static limit** (same as `matcha_stock.tflite`); the engine errors cleanly past it. Fine for audition; watch it during real book playback.
* **Remaining action:** Launch the Tuner and verify you can hear the trained voice speak.

### 2. Export Hardening & espeak-free Training G2P (from litert-community/Matcha-TTS, assessed 2026-07-11) — **split-graph path promoted to Plan A (2026-07-12)**
* **Status update (2026-07-12, evening — priority REVERSED):** the split-graph `litert-torch` path
  is now **Plan A**, the primary export path going forward; the monolithic `onnx2tf` route is
  **Plan B** (fallback + current desktop audition artifact). History within the day: the split was
  first promoted 2026-07-11 on the belief `onnx2tf` was fundamentally broken; the 2026-07-12
  root-cause (§1) disproved that (single encoder-LayerNorm axis bug, fixed) and briefly de-scoped
  the split to an optimization; then the litert-torch conversion of our own Epoch-199 checkpoint
  **materialized at parity the same evening** (per-graph corr 1.000000, human-ear validated), which
  settled the priority on merit: it (a) surfaces the duration predictor's `logw` so the Prosodia
  `duration_scales`/`f0_bias` hooks can reach it (unblocks the exploit-before-train measurement),
  (b) removes the 50-token static limit and per-forward latency of the monolithic graph, and
  (c) enables per-graph delegate placement on mobile. **The split export is done (see Current
  State); the open Plan A work is the Rust multi-graph runtime (action 1b).**
* **Original rationale (retained for record):** the split-graph re-export with per-graph correlation gates also removes the 50-token static limit and most of the ~14 s/forward latency of the monolithic graph.
* **Goal:** Adopt the export/runtime learnings from [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) (an fp16-TFLite conversion of the same upstream checkpoints we fine-tuned — no training code, so no pivot; full assessment in [next-steps.md](../Prosodia/next-steps.md)).
* **Action:**
  1. ~~**Split-graph re-export** of `checkpoint_epoch=199`~~ **✅ DONE (2026-07-12)** — textenc /
     CFM decoder / vocoder as three graphs, host-side Euler ODE loop + length regulator. The
     toolchain fork is **resolved in favor of `litert-torch`** (fixed-shape re-authoring,
     litert-community's route): it converted our checkpoint first-try and passed the per-graph
     correlation gate (corr 1.000000 per graph; artifacts on HF at `v1-ljspeech/litert-split`).
     Per-module `onnx2tf` was never needed; the fixed monolithic `onnx2tf` conversion is the
     Plan B fallback. The split surfaces the duration predictor's `logw`, so the Prosodia
     pipeline's `duration_scales`/`f0_bias` hooks can finally reach it — unlocking the
     exploit-before-train measurement.
  1b. **Rust runtime counterpart (Prosodia, `crates/actor`) — the open Plan A gap, critical-path:**
     the engine today loads a single e2e graph; the split needs a new multi-graph path — load three
     interpreters, host-side embedding lookup (`emb.bin`)/intersperse/pad, compute durations from
     `logw` on the host (where `duration_scales` finally applies), run the length regulator and
     N-step Euler loop, then the vocoder. Until this lands, the desktop Tuner auditions the
     monolithic Plan B artifact.
  2. ~~**Replace espeak-ng in the training containers**~~ **✅ DONE (2026-07-14, Sonora
     `d5dd4fc`):** `op_g2p.py` + `phonemize_filelist.py` + `ljspeech_op` configs; LJSpeech
     train+val pre-phonemized (13,100 lines, 0 vocab violations, 95.8% dict / 4.2% neural OOV);
     espeak-ng dropped from the `sonora_training` compose command (playpen keeps it — dev-tool
     inference on arbitrary text). The **license wall** landed in the same commit
     (`configs/data_licenses.yaml` + `license_wall.py` in the datamodule; NC blocked,
     `SONORA_LICENSE_WALL=derisk` taints §7 runs). GPL is out of the commercial data path;
     train-time/runtime G2P share one source of truth.
  3. **Add per-graph correlation checks** (tflite vs. torch, target ≥0.99 end-to-end waveform) to the export scripts as a standing fidelity gate.

### 3. Emotional Conditioning Layer (Valence, Arousal, Tension - VAT)
* **Goal:** Add emotional directability to the model so the Gemma 4 Director can control voice dynamics.
* **✅ §7 de-risk verdict: PASS (2026-07-16).** One channel (energy, in the arousal slot; V=T=0
  in the v0 corpus labels) trained end-to-end through the FiLM/VAT plumbing on multi-speaker
  LibriTTS-R @ 24 kHz — proves the conditioning architecture works before committing to the full
  3-channel training investment. All 4 sweep groups cleared the pre-registered thresholds with
  real margin: controllability ρ ≈ 1.000 (perfectly monotonic), WER Δ ≤ 0.042 (guardrail 0.10),
  ECAPA leakage ≤ 0.091 (ceiling 0.2). Sweep renders human-audited: energy/loudness change judged
  clearly discernible and natural. Checkpoint promoted (not yet public) to
  `Registry/Sonora/derisk-energy-24k/`. **Formal verdict:
  [derisk-energy-verdict.md](derisk-energy-verdict.md)** (criteria, setup, full numbers, what
  is/isn't proven, consequences).
* **Action (original, items 1–2 done; item 3 blocked on a corpus decision):**
  1. ~~Modify the Matcha-TTS network architecture... to condition on `[V, A, T]`.~~ Done —
     FiLM layers in both text encoder and flow-matching decoder, validated above.
  2. ~~Inject style-conditioning mechanisms (FiLM/AdaLN) into the transformer blocks.~~ Done.
  3. **Open:** full 3-channel VAT needs valence/tension-labeled data the current corpus doesn't
     have. Expresso (the expressive dataset on hand) is NC-tainted and can't ship — a clean
     labeled alternative doesn't yet exist. This blocks going past the single validated channel,
     not the architecture. Two decision briefs now cover the path: corpus approach
     ([vat-corpus-decision-brief.md](vat-corpus-decision-brief.md)) and tension semantics
     ([tension-definition-brief.md](tension-definition-brief.md) — recommends phonation
     tension, pressed↔breathy, labeled with the validated energy-channel recipe). Both await
     owner calls.
* **Long-passage check (2026-07-16, chunk-size sweep):** the derisk checkpoint does NOT
  degrade with passage length — WER 0.02–0.035 at ~2 min of continuous speech (vs 0.00–0.04
  short), speaking rate and loudness stable throughout (`scripts/chunk_size_sweep.py`; report
  + 18 WAVs in `/data/model-training/sonora/chunk_size_sweep/derisk_epoch099`, human audit
  pending). Kokoro's long-passage shortcoming does not carry over — the parallel
  duration-predictor architecture has no drift mechanism. Consequence: the Director's chunk
  size is bound by the litert export ceiling (256 tokens ≈ 5.46 s, or a two-tier export),
  not by model quality.
* **Size target (informal, 2026-07-15):** "Kokoro+" quality/capability at a **150M-param
  ceiling** for acoustic model + vocoder combined — not size-driven (Kokoro was never too big;
  see [model-size-target-decision.md](model-size-target-decision.md) for the actual shortcomings
  being targeted, including a newly-flagged one: long-passage quality degradation).
* **Model family + canon architecture (2026-07-16):** owner wants a size ladder (sonora-mini =
  the current commitment; mid/heavy per target hardware) without training into a corner.
  **[ARCHITECTURE.md](ARCHITECTURE.md)** is now the canon file: the tier-independent
  architecture, the pinned Director↔Actor contract v1 (text lane, V/A/T semantics,
  speaker-as-vector — never an id, mel interchange, chunking), gates, and promotion rules, with
  a maintenance covenant (changes land in the same commit as the behavior they pin). Rationale:
  [model-family-strategy.md](model-family-strategy.md). Also adopted: Emilia tail mining moved
  in-scope for the first 3-channel run (conveyance depth is corpus-bounded), and a
  classifier-free-guidance amplification lever noted (conditioning dropout 0.15 keeps the
  unconditional mode alive — testable on the current checkpoint).
* **3-channel corpus stage STARTED (2026-07-16, both owner approvals in hand):** tension =
  phonation (brief APPROVED) and corpus = EIV pseudo-labeling (brief APPROVED). Done same day:
  `derive_vat_corpus.py` **v1** (T = z(alpha)+z(CPP)−z(H1−H2) per speaker, re-z@2σ; V via
  `--valence-json` when the EIV pass lands) run over train-clean-100 → `data/libritts_r_vat_v1/`
  (30,351 rows, 45.7 h, 247 spks, 0 vocab violations) with the pre-registered **independence
  gate PASS: corr(T,A) = −0.092** (|r|<0.3); raw measures kept in `measures.jsonl`. Human-audit
  set built (`scripts/make_tension_audit_set.py` → `/data/model-training/sonora/tension_audit`,
  50 clips, pressed/breathy/neutral at |A|≤0.3 — **owner audit pending**). EIV assets staged in
  `/data/models/laion/`: 4 EIV-Large heads (Valence/Arousal/Distress/Soft-vs-Harsh;
  each head = MLP 128→…→1 over flattened 1500×768 BUD-E-Whisper encoder states — encoder also
  downloaded). Next: EIV labeling harness → V labels → 3-channel training. Also 2026-07-16:
  convergence-gate **ntfy notifier** wired (see training-operations.md §Stop signal); public HF
  ckpt replaced with safetensors after a picklescan flag (bit-exact, 338 tensors); Prosodia
  Space refreshed to the derisk demo (energy sweep, CFG, long-form).
* **3-channel corpus COMPLETE (recorded 2026-07-19; the work landed 2026-07-17 but this file
  went stale):** `data/libritts_r_vat_v2/` — 29,441 rows, **V/A/T all 100% labeled** (V = EIV
  `valence_combo_v1` per-speaker z; T = phonation trio + soft-json harshness repair),
  **independence gate PASS on all three pairs** (corr T,A = −0.053 · T,V = −0.066 · V,A =
  +0.023; |r| < 0.3). Corpus-scale EIV artifacts in `/data/model-training/sonora/eiv_scores/`.
  Remaining pre-training gates (2026-07-19): **owner audits now mobile** — `audit-tension-v2`
  + `audit-valence-v1` (50 clips each, drawn from v2) registered in the Dataset Auditions app
  (`audit-*` campaign convention, see AI-Lab-0/audition/README.md); Emilia tail integration
  (13,143 raw keeps → ASR/24k/labels/filelists) — **owner call 2026-07-19: HINGES ON THE
  AUDITIONS.** The calibration audits (tension/valence) + markup cards decide it: if the
  audits validate the label semantics at audiobook range, Emilia mining/processing through
  the framework proceeds (in the first vat3 run or a v1.1 continuation per what the audits
  show); until then it is deliberately not started. **Config + warm-start DONE (2026-07-19):**
  `configs/experiment/vat3_finetune.yaml` + `configs/data/libritts_r_vat_v2.yaml` (mel stats
  carry over — same 29,441-clip audio split) and
  `/data/model-training/sonora/warmstart/vat3_init.ckpt` built via `make_warmstart.py` from the
  promoted derisk epoch-099 checkpoint — **338/338 tensors warm, 0 fresh** (identical
  architecture), fresh optimizer. Launch when gates clear:
  `python -m matcha.train experiment=vat3_finetune ckpt_path=…/warmstart/vat3_init.ckpt`.
  (Env note: `sonora:latest` is the bare ROCm-torch image — project deps are pip-installed at
  container start with `PYTHONPATH=/workspace`; the in-tree prebuilt `monotonic_align` .so
  makes `pip install -e .` unnecessary, and its numpy==1.24.3 build pin fails on py3.12 anyway.) Spin-down is now scripted: `AI-Lab-0/scripts/inference-engines.sh
  stop|start|status` (stops Portainer first so GitOps can't relaunch engines mid-run; start
  re-enables it last; cycle-tested 2026-07-19).
* **AUDITS COMPLETE + vat3 LAUNCHED (2026-07-20).** All 196 audit cards rated in the app;
  per-class readouts archived (`audit-sets/*/results.json` + `markup_prep/spike_v1/audit_results.json`):
  * **`audit-markup-v0` — PASS (93%):** 89/96 kept, 64 at 4–5, only 4 register relabels +
    3 drops. Failures concentrate in ≤6-token LibriTTS fragments (9/14 fails vs 3/64 keeps) —
    the owner's phrase-selection call, quantified. **SCM/Gemma annotator validated as the
    labeling pass**; future rounds get a token-count floor on corpus picks.
  * **`audit-valence-v1` — moderate pass (62%):** V-neg 70% · neutral 60% · V-pos 55%;
    disagreements systematically relabeled to narrative/embodiment (acted character voices).
    Direction of the axis confirmed; acceptable clip-level noise for a continuous channel.
  * **`audit-tension-v2` — extremes fail as named:** breathy 0/20 literal (7/20 ratified as
    soft/lax passes — owner call), pressed 1/20, neutral band holds 6/10. Instrument values are
    real (raw-measure percentiles check out) but per-speaker-z extremes on restoration-processed
    LibriTTS-R are perceptually subtle. **RULING: T-axis semantics rescoped to LAX ↔ TIGHT**;
    "breathy"/"strained" vocabulary reserved for data with real aspiration (own synth renders,
    Emilia). T channel stays in training — instrument-truthful, audible range on synth data.
  * **Emilia — GO as a v1.1 continuation (owner call 2026-07-20):** vat3 trains now on the
    staged corpus; the 13,143-clip tail goes through ASR/24k/labels/filelists after training
    frees the GPU, then training continues as v1.1. The tension result *strengthens* the case:
    Emilia's unrestored audio has exactly the phonation texture LibriTTS-R restoration strips.
  * **vat3 run launched 2026-07-20** via the compose `training` profile (command updated to
    `experiment=vat3_finetune` + `vat3_init.ckpt`); spin-down script ran first (Portainer +
    all inference engines down); container `restart` set to `no` post-launch so a finished run
    can't relaunch into retraining. First launch tripped the **license wall** on the undeclared
    `libritts_r_vat_v2` dir (guard working as designed) — declared v1/v2 dirs under the
    `libritts_r` CC-BY-4.0 entry (Sonora c804ecd) and relaunched. Convergence/failure monitor
    armed (session watcher → push notification). MLflow logging verified end-to-end
    (experiment 1; hparams land at fit start, metrics per-step every 50).
    Two launch-blocking gfx1151 lessons hit and fixed same night:
    1. **Forked dataloader workers wedge** — forking the GPU-initialized trainer (40+ HIP
       threads, KFD SVM ranges) took minutes per fork with `amdgpu` restore-workqueue churn;
       20 workers = run never reached step 1, recurring every epoch. Fix (Sonora 20e1d4d):
       `multiprocessing_context="spawn"` + `persistent_workers` in the datamodule (workers
       never inherit the GPU address space; spawn cost paid once per fit), `num_workers`
       20 → 8 (spawned workers don't share COW pages).
    2. **First-run MIOpen tuning ~an hour** — matcha's variable-length batches make every new
       conv shape pay a kernel find on a cold db; looks like a hang (99% GPU, no step/metric
       output). Verified progressing via py-spy + find-db growth; now persisted at
       `/data/model-training/sonora/miopen:/root/.config/miopen` (compose bind, seeded via
       `docker cp` from the live run) so recreated containers start warm.
  * **vat3 COMPLETE (2026-07-21):** deliberately stopped at epoch 100 — the config inherits
    `trainer.max_epochs=-1` (**the run never self-stops**; set max_epochs for future rounds),
    so the stop criteria were the derisk-precedent round length + val plateau (`loss/val_epoch`
    flat at 3.178–3.185 over the final six epochs; train 2.91 from 3.28). Checkpoint:
    `logs/train/vat3_finetune/runs/2026-07-20_01-45-32/checkpoints/checkpoint_epoch=099.ckpt`
    (+ `last.ckpt`, saved 13:14 — checkpoints only land every 100 epochs). Graceful SIGTERM
    stop (exit 0); MLflow run `wistful-dolphin-948` re-marked FINISHED (Lightning's teardown
    stamps FAILED on SIGTERM — cosmetic). Warm MIOpen db (7.7 M) copied to the persistent
    bind before stop. Engines + Portainer GitOps restarted via `inference-engines.sh start`.
    **Next:** eval harness verdict across the three channels (energy→LUFS, tension→phonation,
    valence→EIV) on the 099 checkpoint → round-trip auditions (SCM acceptance test) →
    Emilia tail processing → v1.1 continuation.
* **CFG guidance validated by ear (2026-07-16, `scripts/render_guidance_demo.py`):** on
  exclamatory OOD text ("We won! We actually won the championship!"), guidance s=2–3 on the
  decoder field with **25 ODE steps** was audited clearly *more human* than the neutral render —
  much of perceived "robotic-ness" is prosodic mismatch (calm delivery of excited text), which
  guidance corrects. Caveats, all confirmed by the same audits: 10-step renders bury the win
  under solver artifacts (guidance needs ~25 steps ⇒ ~5× decoder compute); raw out-of-range VAT
  input (+2/+3) saturates — field extrapolation is the right lever, input extrapolation a dead
  end; low-energy renders are the weakest (thin quiet tail); and a robotic edge remains at s=3 —
  the corpus ceiling. Net: guidance is a real inference-time conveyance lever (host-side only,
  the exported decoder graph is unchanged — run it twice per step), and the main fix stays
  Emilia tail mining. Renders: `/data/model-training/sonora/guidance_demo{,_nt25}`.

### 4. Multi-Speaker & Casting Grid (Phase 1)
* **Goal:** Train on multi-speaker datasets (LibriTTS-R and Expresso) to establish the continuous voice casting and blending space.
* **Action:**
  1. Configure speaker-embedding sizes in `configs/model/matcha.yaml` (e.g., matching the multi-speaker splits).
  2. fine-tune the model on the prepared multi-speaker filelists to enable voice blending.

---

## Environment & Command Reference

> Operational runbook (launch/stop/resume, locations, MLflow, gates, footguns):
> [training-operations.md](training-operations.md). The commands below are kept for
> continuity; the runbook is canonical.

### Docker Compose Unified Stack
All containers are managed under the unified [docker-compose.yml](../../AI-Lab-0/docker-compose.yml) stack config. Recreate or restart containers using:
```bash
docker compose up -d
```

### Launch Vocalizer Container
The Vocalizer (`sonora_vocalizer`, renamed from `sonora_playpen` 2026-07-16; the standing
human-audit surface — see ARCHITECTURE.md §5) is part of the GitOps compose stack — it deploys
automatically on push and needs no manual launch. To bounce it explicitly:
```bash
docker compose -f AI-Lab-0/docker-compose.yml up -d sonora_vocalizer   # from the umbrella root
```
Training is profile-gated so syncs never start it; launch deliberately with:
```bash
docker compose -f AI-Lab-0/docker-compose.yml --profile training up -d sonora_training
```
Container commands bootstrap `uv` and install via `uv pip install --python /opt/venv/bin/python`
(AGENTS.md §7; the rocm/pytorch images ship their stack in `/opt/venv`).

### Accessing GPU Stats (Host)
```bash
amd-smi monitor
```

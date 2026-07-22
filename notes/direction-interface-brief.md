# Direction Interface — Design Brief (2026-07-19, PROPOSAL — no owner decisions yet)

_Owner prompt 2026-07-19: "We never did fully plan out exactly how Sonora should receive
direction on VAD, prosody, gender-depiction, age-depiction and so on" + the proposal to have
Gemma 4 26B mark up text for conveyance to match paired audio ("reverse-conveyance"), then train
Sonora on markup → sound. This brief pins the design space and weighs feasibility/value of the
reverse-conveyance pipeline. Decisions belong to a planning session._

## 1 · Current state (what direction-plumbing exists today)

| Layer | Exists | Gap |
|---|---|---|
| Actor conditioning contract | 3 continuous channels (V/A/T) via zero-init FiLM + speaker embedding; de-risked (energy ρ≈1.000) | Utterance-level only; no span/word granularity |
| Labels for the contract | `derive_vat_corpus.py` v1/v2: instrument-derived V/A/T over LibriTTS-R (LUFS · phonation trio+EIV · EIV valence combo), JL-calibrated | Descriptive ("how it *was* said"), not directive; no register/casting semantics |
| Director-side schema | Book-prose director-pass emits `{V,A,T, register, direction}` per chunk (Gemma 26B, text-only) | Not pinned as *the* production schema; no span markup; not consumed by training |
| Casting (gender/age) | Implicit via speaker embedding (speaker choice) | Not directable — no attribute axes ("older male, gravelly") |
| Prosody specifics (pauses, emphasis, pace) | Measurable (cdminix per-token pitch/energy/duration; alignment gaps) | No representation in the contract at all |

## 2 · Design principles (proposed)

1. **Contract-lock:** the Actor keeps a lean continuous contract (channels + casting vector).
   Markup is the **Director-facing** representation; a deterministic **compiler** maps markup →
   contract. Never feed markup tokens into the TTS text encoder (that's the Parler/prompt-TTS
   path — abandons the contract that just passed de-risk, and re-couples Actor size to language
   understanding).
2. **Instruments are the only ground truth about audio.** LLMs verbalize and interpret
   measurements; they never assert acoustic facts the instruments didn't measure.
3. **One schema, train = serve:** the markup schema used to build training pairs is the same
   schema the production Prosodia Director emits at runtime (dogfooding, same as the
   book-prose director-pass thesis).

## 3 · The reverse-conveyance pipeline (owner proposal, refined)

For **vetted/trusted audio+text pairs only** (owner constraint — LibriTTS-R/Hi-Fi TTS grade, or
our owner-certified clips):

```
audio ──► instruments ──► measurement notation ──► [quantizer] ──► symbolic notation ─┐
                                                                                      ├─► Gemma 26B ─► conveyance markup
text ─────────────────────────────────────────────────────────────────────────────────┘                    │
                                                                            [compiler] ◄──────────────────┘
                                                                                │
                                                              contract channels (+ span extension)
                                                                                │
                                                            Sonora trains on (markup-compiled → audio)
```

- **Measurement notation — utterance level CONSOLIDATED (2026-07-19):**
  `scripts/derive_markup_measures.py` →
  `/data/model-training/sonora/markup_prep/utterance_notation.jsonl` — **30,541 rows**, one
  per trusted text-adjoined clip: 30,351 LibriTTS-R v2 rows (transcript + V/A/T labels +
  LUFS/alpha/CPP/H1-H2 + all EIV heads + valence combo; zero gaps) + 190 owner-certified
  expressive-registers clips (text, register, gender, intended VAT, measured_z, qc, owner
  audit; EIV for 184 — the 6 book-prose clips have no EIV pass yet). Still missing for span
  markup: per-token pitch/energy/duration + pause structure — the cdminix layer (cleared
  CC-BY-4.0, script-based dataset, not on disk) or our own forced-alignment pass; see §6.
- **Quantizer (deterministic, pre-LLM):** bin raw floats into symbols (`pitch:+2`, `pause:long`,
  `tilt:pressed`) before prompting. LLMs are unreliable at interpreting raw float arrays;
  give Gemma symbols, not arithmetic. This step is code, not model.
- **Gemma's actual job — semantic fusion:** align the acoustic symbols with the *text's*
  meaning: which words carry the emphasis peaks; is the long pause rhetorical or a breath; what
  register word fits "pressed + fast + rising"; produce span markup + utterance direction. This
  is a bounded structured-output task — squarely within the 26B's demonstrated competence
  (director-pass emits clean VAD JSON; ~58 tok/s; 8k ctx is ample for utterance-scale input).
- **Round-trip verifier (code, post-LLM):** every markup claim must re-derive from the
  measurements (markup says `emphasis:"never"` → token measures for "never" must show the
  peak). Reject/flag mismatches. Gemma adds interpretation, never facts — hallucination is
  filtered structurally, not by trust.

## 4 · Feasibility — HIGH, with three engineered mitigations

| Risk | Mitigation |
|---|---|
| LLM numeric illiteracy | Quantizer: symbols in, no raw floats (§3) |
| Hallucinated markup | Round-trip verifier vs. measurements; reject rate is itself a health metric |
| Schema drift at bulk | Pin the schema first (§6); constrained JSON output; malformed-rate gate (same discipline as the director-pass spike criteria) |

**Calibration beds already on disk** — this is unusually well-provisioned:
1. **expressive-registers v1 (193 owner-certified clips)**: reverse-conveyance markup should
   recover the *known* register + intended-VAT from measurements+text alone. Blind, scored,
   zero new labeling cost.
2. **JL-Corpus (CC0, 2,400 acted clips)**: known emotion categories → quadrant agreement, the
   same harness that calibrated EIV.

**Cost:** batch inference on the box. Rough order: ~30k LibriTTS-R utterances × (input ~400 tok
+ output ~300 tok) at ~58 tok/s ≈ 40–50 GPU-hours — chunkable, resumable, and subject to the
spin-down rule. Spike first: ~100 clips (50 certified + 50 LibriTTS-R), owner-audit the markup.

## 5 · Value — HIGH

- **(markup → audio) training pairs at zero human labeling**, over the entire permissive stack —
  this is the direction-interface training set, not just labels.
- **Span-level conditioning targets for free** — per-token measures supply the training signal
  for the contract extension (emphasis/pause/pace), the main genuinely new Actor work.
- **Dogfoods the production Director schema** end-to-end before Prosodia ships it.
- **Dual-use** (high-ambition-6): the same pairs read backwards train the Audience listener.
- **Human-auditable**: markup is readable — the owner can audit direction quality by eye in the
  Auditions app long before a model consumes it.

## 6 · What this brief does NOT solve (needs its own design)

1. **Markup schema** — the pin-first task: tag vocabulary, span syntax, register lexicon,
   casting descriptor grammar. Must match the production Director's annotation schema.
2. **Casting axes** — making gender/age-depiction directable needs a designed attribute space
   (speaker-attribute embedding trained from metadata + descriptions), Phase-1 casting-grid
   territory. Today: speaker choice only.
3. **Span-level FiLM extension** — per-token (or per-span) conditioning in Matcha; architecture
   work with its own de-risk (the §7 playbook applies).

## 7 · Open questions for the planning session

- ~~Labeler model?~~ **Weighed 2026-07-19 (owner asked: on-disk 26B-A4B vs
  [DiffusionGemma-26B-A4B-it](https://hf.co/google/diffusiongemma-26B-A4B-it) vs dense
  [Gemma 4 31B-it](https://hf.co/google/gemma-4-31B-it); primary lens = the markup task,
  secondary = post-check/judge).** All three Apache-2.0. Facts from the cards 2026-07-19:
  DiffusionGemma = block-diffusion (256-token canvas, ≤48 denoise steps, ~15–20 tok/**forward
  pass** → >1100 tok/s on H100 FP8), 256K ctx, GGUF/ollama quantizations exist, but benchmarks
  trail the AR line (MMLU-Pro 77.6 vs 82.6; reasoning much weaker, AIME 69.1 vs 88.3) and the
  card documents **no constrained-decoding / JSON-schema story**. 31B dense = best quality,
  ~10–15 tok/s on this box (q4 ≈ 17 GB, bandwidth-bound).

  | | AR 26B-A4B (on disk) | DiffusionGemma 26B-A4B | Dense 31B |
  |---|---|---|---|
  | Bulk markup throughput (this box) | ~58 tok/s, proven | potentially **several-fold faster** (multi-token/pass; same ~4B-active bandwidth cost) — the one lever that matters at Emilia scale | ~4–6× slower than MoE |
  | Schema discipline | proven (director-pass, clean JSON w/ template) | **unproven**; no documented constrained decoding — malformed-rate is the make-or-break metric | proven family behavior |
  | Task quality (bounded fusion, verifier-backed) | sufficient (spike-validated) | likely sufficient — reasoning gap matters little here; verifier bounds the risk | best, likely unneeded margin |
  | Serving | live today (ollama ROCm) | GGUF/ollama exists upstream; needs ROCm/gfx1151 validation (mmap-gotcha territory) + ollama version check | same stack as 26B, just slower |
  | Judge/post-check fit | fallback | **worst fit** (weakest reasoning; judging is the one reasoning-ish step) | **best fit** — quality per short output, throughput irrelevant |

  **Recommendation:** spike all three on the ~100-clip calibration bed (cheap). Bulk pass:
  AR 26B-A4B stays default; **DiffusionGemma replaces it only if** its malformed-JSON rate ≤
  the AR model's and label agreement is within noise — in which case it meaningfully cuts the
  bulk/Emilia-scale cost. **Judge/hard-case escalation: dense 31B** regardless (note: the
  instrument verifier remains the only truly independent check — an all-Gemma
  generator+judge pair shares family blind spots). Infilling-style inline markup (frozen text,
  denoised tags) is a tantalizing DiffusionGemma fit but undocumented — treat as a stretch
  experiment, not a plan. Production-Director dogfooding: all three are Gemma-family; the E2B
  sibling ships AR, which mildly favors AR labelers for behavioral transfer.
  **Owner concurrence (2026-07-19): stay all-Gemma** — valuable for the Gemma end-state
  (dogfooding/behavioral transfer), *explicitly not* valuable as "peer review." Family
  self-review is not independent review; the instrument round-trip verifier is the only
  independent check in the loop, and the design should keep treating it as load-bearing.
  All three candidates now in the reference library (`/data/reference/models/{Google,unsloth}/…`,
  pulled 2026-07-19).

  **Serving status (2026-07-19):**
  - `gemma-4-26b-a4b-qat` — live in ollama (~58 tok/s, proven).
  - `gemma-4-31b-qat` — **registered + smoke-tested: 11.6 tok/s measured** (predicted 10–15).
    Source GGUF hardlinked at `/data/ollama/gemma4-31b/` (same Modelfile template, num_ctx 8192).
  - `diffusiongemma-26b-a4b` — **GGUF on disk but NOT servable by ollama**: the
    `diffusion_gemma` arch is an open ollama feature request (ollama/ollama#16664, no support
    as of 0.32.1 — not fixable by a version bump). Serving landscape (checked 2026-07-19):
    DiffusionGemma lives in **llama.cpp PR #24423** as a dedicated `llama-diffusion-cli`
    binary — NOT merged into mainline `llama-server`. That single fact rules out every
    llama.cpp-wrapping server at once: ollama (#16664 open) **and Lemonade** (installed on
    the box as the `lemonade-server` container; its LLM path is stock `llama-server` on
    Vulkan/ROCm — no mainline support, nothing to wrap; a CLI is not a server either way).
    Actual options when the spike needs it:
    (a) **build PR #24423 with ROCm and drive `llama-diffusion-cli` as a batch tool** —
    cheapest path and a good fit: the reverse-conveyance pass is a batch job (like the
    `synth_*.py` renderers), it does not need an OpenAI endpoint;
    (b) **vLLM-ROCm with the original safetensors** (~52 GB pull) — the only true *server*
    support today (natively integrated; community ~800 tok/s on NVIDIA), at the cost of a
    gfx1151 vLLM bring-up;
    (c) wait for the PR to merge → ollama/Lemonade inherit it. Prepared Modelfile kept at
    `/data/ollama/diffusiongemma-26b/` for that day.
- Schema shape: inline tags vs sidecar JSON? (Compiler prefers sidecar; humans prefer inline.)
- Which corpus first — LibriTTS-R expressive subset, or the certified synthetic set (smaller,
  intended-direction known, closes the loop fastest)?
- Does the Audience listener train from the same pairs immediately (dual-use now) or later?
- Where does reverse-conveyance run in the pipeline — a `derive_markup_corpus.py` sibling of
  `derive_vat_corpus.py`?

Cross-refs: [vat-corpus-decision-brief.md](vat-corpus-decision-brief.md) ·
[tension-definition-brief.md](tension-definition-brief.md) ·
[book-prose-operations.md](book-prose-operations.md) (director-pass) ·
[high-ambition-1-matcha-actor.md](high-ambition-1-matcha-actor.md) (contract-lock) ·
[high-ambition-6-audience-conveyance-stt.md](high-ambition-6-audience-conveyance-stt.md) ·
[dataset-landscape.md](dataset-landscape.md) §Strategy.

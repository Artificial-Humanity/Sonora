# Direction Interface — Design Brief (2026-07-19; DECIDED 2026-07-30)

> [!IMPORTANT]
> **Owner decisions, 2026-07-30** (the planning session this brief was waiting for):
> 1. **Delivery becomes a learned conditioning input in the next training run** — a small
>    embedded vector (5 lanes + unknown) through the same zero-init FiLM path as V/A/T.
>    Contract bumped to **v2** in [ARCHITECTURE.md](ARCHITECTURE.md) §1.
> 2. **Register stays Director-side.** The 47-label lexicon compiles to (V/A/T + delivery);
>    the Actor never receives a register id. (The teacher campaign ran this pattern all
>    month: Gemma compiles registers into engine-native parameters.)
> 3. **Tempo & loudness stay host-side** (duration_scales / pre-vocoder dB bias — surgical
>    per the exploit-before-train measurement). Training owns pitch + phonation.
> 4. **Casting** unchanged: 64-d speaker vector, casting-grid milestone; accent is casting.
> 5. **Span-level direction: SPIKE NOW, TRAIN LATER** — run the ~100-clip reverse-conveyance
>    calibration spike in parallel with corpus assembly; span-FiLM architecture waits until
>    the utterance-level certified run passes its gates.
> 6. **Labeler update (supersedes §7's bulk default):** measured 2026-07-29, the MoE
>    26B-A4B emits malformed JSON 13/100 vs 0/100 for the dense models — bulk labeler is
>    now **e4b** (0/100, fast), judge stays **dense 31b**. DiffusionGemma still unservable.
> Contract-relevant evidence from the teacher campaign: the most directable engine interface
> was low-dimensional continuous numeric dials (Zonos); the categorical 8-dim emotion vector
> was that engine's documented instability source; prose slots drifted most. Lean continuous
> contract confirmed.

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

## 7 · What was open here, and where it landed

All five questions this section carried are now answered elsewhere. Kept as a
one-line-each index rather than the original deliberation, which ran to five pages of
DiffusionGemma serving analysis for a model that never became servable.

| question | answer | lives in |
|---|---|---|
| Labeler model? | **e4b for volume, 31b for judgement.** Measured twice — the MoE emits 13/100 malformed JSON where dense emits 0/100 (2026-07-29), and on skill-file *obedience* the 31b scored 24/24 where the MoE managed 8/24 and e4b 5/24 (2026-08-02). **e4b was tested and is NOT the director.** DiffusionGemma was never servable (`diffusion_gemma` unsupported by ollama; upstream lives in an unmerged llama.cpp PR as a CLI) and is not pursued. | [book-prose-lane.md § Director model](book-prose-lane.md) |
| Inline tags vs sidecar JSON? | **Sidecar is canonical; inline is a rendered projection**, never authored, never parsed back. Ratified v0.1. | [markup-schema-brief.md §1](markup-schema-brief.md) |
| Which corpus first? | The certified synthetic set — smaller, intended direction known, closes the loop fastest. Ran as the 100-clip spike: **PASS at 93%**, 89/96 kept. | [markup-schema-brief.md §5](markup-schema-brief.md) |
| Does the Audience listener train from the same pairs now or later? | **Later.** Goal 6 is a parked vision note; nothing is scheduled. | [high-ambition-6](high-ambition-6-audience-conveyance-stt.md) |
| Where does reverse-conveyance run? | `scripts/derive_markup_measures.py` is the notation half and exists (30,541 rows). ⚠ It is **frozen at v2 paths** and mis-keys the contiguous v2 index under `"speaker"` — a trap for the span-markup spike (D-L5). | [todo.md §8](todo.md) |

**Still genuinely open**, and it is the one thing this brief did not solve: the **span
decode layer**. Per-token pitch/energy/duration and pause structure do not exist in the
notation store, so every span-typed field in SCM v0.1 is `checked: []` — present in the
schema, unverifiable in practice. That is §6 item 3, and Contract v2 puts the span-markup
spike ahead of span-FiLM.

Cross-refs: [vat-channels.md](vat-channels.md) ·
[book-prose-lane.md § Part 1 — Operations](book-prose-lane.md) (director-pass) ·
[high-ambition-1-matcha-actor.md](high-ambition-1-matcha-actor.md) (contract-lock) ·
[high-ambition-6-audience-conveyance-stt.md](high-ambition-6-audience-conveyance-stt.md) ·
[dataset-landscape.md](dataset-landscape.md) §Strategy.

# ARCHITECTURE — Project Sonora

_Canon. The tier-independent architecture of the Sonora model family — mini, mid, heavy — the
pinned contract every tier implements, and the process that produces one. Weights are per-tier
and disposable; **this architecture and the corpus are the product.** Maintenance covenant: any
commit that changes architectural behavior (corpus rules, contract, gates, promotion) updates
this file in the same commit. Detail lives in the linked docs; what is written HERE is what's
pinned._

_Established by the Phase 0 (v1-ljspeech) and §7 de-risk (derisk-energy-24k) lineages; every
pinned item below has shipped at least once. Last updated: 2026-07-16._

---

## 1. The Director↔Actor contract (v1)

Every tier is an implementation of this contract. A consumer (Director, Prosodia, the Vocalizer) built
against it must work against any tier unchanged.

| Interface | Pinned as | Notes |
|---|---|---|
| **Text** | 178-symbol locked IPA vocab; op_g2p lane (OpenPhonemizer dict → DeepPhonemizer TFLite OOV fallback, U+0303 rule); `cleaners=[no_cleaners]`; intersperse-0 | Digits/abbreviations are the **caller's** job (op_g2p does not expand them). espeak is banned from the runtime path. |
| **Control** | `vat = (valence, energy, tension)`, each float ∈ [−1, 1], 0 = speaker-neutral | Semantics are per-speaker z-scores of the corpus measures (§2). Slot map `{valence:0, energy:1, tension:2}`. Per-utterance today; per-token is the plumbing's native shape (expansion via the duration alignment). |
| **Speaker** | 64-dim float vector | **Never an id.** The mini tier resolves ids→vectors via a host-side table; a future speaker encoder produces the same vector. Nothing downstream may assume a roster. |
| **Rate** | `length_scale` (1.0 default) | Deliberate: speaking rate is NOT a learned channel. |
| **Amplification (optional)** | CFG guidance scale `s` on the decoder field; unconditional = vat 0; default 1 (off) | Pure host orchestration — the decoder graph is unchanged, run twice per ODE step and extrapolated. Validated by ear at s = 2–3 **with ≥ 25 ODE steps** (2026-07-16); at 10 steps solver artifacts dominate. Raw out-of-range VAT input saturates — never use input extrapolation. |
| **Audio interchange** | 80-band mel, 24 kHz, n_fft 1024, hop 256, win 1024, fmin 0, fmax 12000 | Acoustic↔vocoder boundary ([sample-rate-24khz-decision.md](sample-rate-24khz-decision.md)). Mel normalization stats are corpus properties (shipped in each export's `config.json`), not contract. |
| **Chunking** | Director chunks on sentence/clause boundaries within the tier's export budget (mini: 256 interspersed tokens / 512 mel frames ≈ 5.46 s) | Model quality does not bind before ~2 min (chunk-size sweep, 2026-07-16) — the budget is an export-shape property. Cross-chunk seams are a Director concern (pause-based joins). |

Contract changes bump the version and require an owner call.

## 2. Corpus & labels

* **License wall (absolute):** every training input CC-BY-4.0 or freer. No NC anywhere in the
  lineage — including as a pipeline tool (excludes Expresso, audeering models). Verify licenses
  in writing per source; record in the promotion README. Personally-acquired media never enters
  the public lineage; analysis/benchmark use and the private-lineage firewall are governed by
  [audiobook-corpus-policy.md](audiobook-corpus-policy.md) (2026-07-17).
* **Filelist format:** `path|spk|phonemes|v,a,t`, pre-phonemized (`scripts/phonemize_filelist.py`)
  so training images need no G2P stack. Filelists are derived data: regenerable by script, never
  the source of truth.
* **Label recipe (validated by the energy channel, ρ ≈ 1.000):** weak labels suffice. Continuous
  measures per utterance → **per-speaker z-score** → clamp/scale to [−1, 1]. Energy = LUFS.
  Valence = EIV pseudo-labels; tension = phonation composite (pressed↔breathy)
  ([vat-corpus-decision-brief.md](vat-corpus-decision-brief.md),
  [tension-definition-brief.md](tension-definition-brief.md) — both approved 2026-07-16).
* **Channel independence gate:** per-speaker |corr| < 0.3 between any two channels' labels,
  else residualize and re-check.
* **Calibration before training:** anchor new labels against a known-ground-truth set
  (JL-Corpus, CC0) + ~50-clip human audit.
* **Standing activity:** corpus accumulation is continuous, not per-run — tail mining
  (Emilia-YODAS CC-BY subset) targets the label extremes audiobook data undersamples. Deeper
  emotional conveyance is corpus-bounded; channels only interpolate what the data exhibits.

## 3. Model pattern

* Flow-matching acoustic model (backbone is the per-tier choice) + separate vocoder.
* **Conditioning:** FiLM, **zero-init** (new channels start as exact no-ops — enables warm-start
  and the plumbing-test gate), applied per encoder/decoder block; VAT trunk in-graph
  ([vat-conditioning-design.md](vat-conditioning-design.md)).
* **Conditioning dropout `vat_cond_dropout=0.15`** — non-negotiable: it tolerates label noise AND
  keeps the unconditional mode alive for classifier-free-guidance-style amplification at
  inference (the "amp it up" lever — validated 2026-07-16, see contract §1 Amplification row and
  `scripts/render_guidance_demo.py`).
* **Warm-start** from the nearest prior checkpoint (`scripts/make_warmstart.py` pattern); never
  cold-start when a lineage checkpoint exists.
* Vocoder per tier: fine-tune from a universal checkpoint at the contract mel config; gate by
  copy-synthesis (WER ≤ ASR floor + 0.02 AND |Δ mel-L1| < 5% between consecutive checkpoints).

## 4. Training operations

Full runbook: [training-operations.md](training-operations.md). Pinned:

* One trainer on the GPU at a time; profile-gated compose services; deploy = push to main.
* MLflow logging, unbuffered stdout.
* **Convergence watcher wired BEFORE launch** (systemd timer → CPU-only throwaway container →
  postprocess → `history.jsonl` + CONVERGED marker). The watcher never auto-stops the trainer;
  stopping is a human act after audit. The first flip to CONVERGED also fires a **one-shot
  owner push** (ntfy; topic in `/etc/ai-lab/ntfy.env`) — a gate nobody hears about is not a
  stop signal.
* **Every gate is tested against a checkpoint that SHOULD fail it** before it's trusted
  (zero-init plumbing test).

## 5. Gates (§7 regime — pre-registered, per channel)

Thresholds are written down before the run; overriding one requires a written reason.

| Gate | Threshold |
|---|---|
| Controllability | Spearman \|ρ\| ≥ 0.9 (requested vs produced measure) |
| Identity leakage | ECAPA drift / inter-speaker gap ≤ 0.2 |
| Intelligibility guardrail | WER Δ ≤ +0.10 (worst sweep row vs baseline) |
| Cross-channel independence | sweeping channel X at others=0 must not move the other channels' produced measures beyond their per-step effect |
| Human audit | mandatory before any promotion — the harness qualifies, ears decide |

Produced measures live in `scripts/eval_harness.py` `MEASURES` (energy→LUFS, duration→seconds,
f0→pyin median; each new channel adds its measure — tension→phonation composite).

The **Vocalizer** (`vocalizer.py`, `sonora-vocalizer.ai-lab-0.mcfarlin.family`) is the standing
human-audit surface (owner call, 2026-07-16): every new model capability or control ships with a
dial there in the same phase, so outputs stay vettable by ear at the current feature set.

## 6. Export lane

* Split graphs (textenc / decoder / vocoder), fixed shapes per tier budget, fp16 PTQ.
* GPU-clean op discipline (no BROADCAST_TO/GATHER_ND/…; additive attention masks; reshape-slice
  decimation; broadcast-mul instead of repeat) — enforced by static opcheck scan.
* Host-side: embedding/speaker tables, duration alignment, per-token control expansion, ODE loop.
* **Parity gates:** wrapper-vs-model ≤ 1e-3; per-graph fp16 corr ≈ 1.0; e2e waveform corr vs
  torch ≥ 0.999; control monotonicity verified THROUGH the exported pipeline.
* Reference implementations: `/data/toolchain/litert-conversion/convert_vat.py` (current),
  `build_matcha.py` (Phase 0).

## 7. Promotion & registry

* Promote the **audited, gated** checkpoint (not the latest) to `Registry/Sonora/<name>/` with:
  README (provenance: training run, stop signal, criterion + numbers, human-audit date, license
  statement), `gate_history.jsonl`, eval report, audited samples, render/export metadata.
* Publish to HF (`artificial-humanity/Sonora`) on owner call only.
* Tier naming: the current lineage is **sonora-mini** (150M ceiling, on-device;
  [model-size-target-decision.md](model-size-target-decision.md)); mid/heavy tiers reuse this
  architecture end-to-end with per-tier backbone, budget, and export target
  ([model-family-strategy.md](model-family-strategy.md)).

## 8. Staged capability (any tier)

Base acoustic → conditioning channels (§§2–5) → **expressive fine-tune** (on radar, not
scheduled: nonverbal vocal events, micro-prosody beyond the 3-dim control space) — each stage a
separate gated run on the same contract, so stages compose across tiers.

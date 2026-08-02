# Project State — Sonora

The current-state snapshot: what is true now, what is running, and what comes next.
Behavioral rules and the stack/layout manifest live in [AGENTS.md](../AGENTS.md); the
architecture canon is [ARCHITECTURE.md](ARCHITECTURE.md); open work is
[todo.md](todo.md). One entry per front, newest facts first — superseded narrative is
deleted, not banner'd (git history is the archive; the pre-2026-08-02 roadmap
narrative was removed in the consolidation pass).

_Last updated: 2026-08-02._

---

## VAT corpus — train on v3c

> **Train on `data/libritts_r_vat_v3c`.** Same 31,445 clips / 51.3 h / 247 speakers as
> v3 (`MAX_SECONDS=22`), twice corrected: **v3b** fixed the phonemes (v1–v3 all carry
> contraction-poisoned IPA on ~6.4% of rows — `I'll` phonemized as *ill*; the espeak-free
> G2P had no apostrophe representation), and **v3c** replaced the shuffled train/val
> split with a per-clip blake2b hash of the wav basename so growing the corpus stops
> re-rolling membership. **30,485 train / 960 val** (210 of 247 speakers in val).
> Verified: adding 2,000 synthetic rows moves 0 of 960 val clips.
>
> **Nothing has trained on v3, v3b or v3c.** The existing `vat3_finetune` ep099
> checkpoint trained on **v2** — poisoned phonemes included — but the labels barely
> moved between v2 and v3 (corr ≥ 0.9993 per channel on shared clips), so **vat3 is a
> viable fine-tune base**; a from-scratch retrain is not forced.
>
> Still open before a run: no experiment config points at v3c, and `data_statistics`
> in the v2 config were measured on the 29,441-clip split — re-run
> `generate_data_statistics.py` inside the training container. Lineage table and every
> source: **[training-sources.md](training-sources.md)** (SSOT). Residual review debt
> that touches training: [todo.md §1](todo.md).

Independence gate (v3): corr(T,A) = −0.059 · corr(T,V) = −0.066 · corr(V,A) = +0.027 —
all PASS. Channel standing (vat3-24k, 2026-07-22): energy **PASS**, tension
**near-pass**, valence **FAIL** — a corpus-label limit, not architectural; that failure
is what opened the directed teacher-synthesis lane
([vat-channels.md](vat-channels.md)).

## Checkpoint lineage

| checkpoint | trained on | status |
|---|---|---|
| Phase 0 `ep199` (baseline-ljspeech-22k) | LJSpeech @ 22.05 kHz | shipped to HF; the v1 voice; both export lanes verified on it |
| `derisk-energy-24k` ep099 | libritts_r_vat v1 (energy channel only) | §7 verdict PASS (ρ≈1.000, leakage ≤0.091, WER Δ ≤0.042); staged in `Sonora/huggingface/`, not pushed |
| `vat3-24k` ep099 | libritts_r_vat **v2** | energy PASS / tension near-pass / valence FAIL; staged; **the warm start for the next run** |
| 24 kHz HiFi-GAN vocoder `g_02510000` | LibriTTS-R, warm from UNIVERSAL_V1 | copy-synthesis converged, human-audited; staged |

Publishing staged checkpoints to the public HF repo remains a deliberate owner call —
they are validated components, not the shippable directable actor.

## Teacher synthesis & the expressive dataset

**Portfolio (five):** chatterbox · qwen · zonos · orpheus · moss_vg. VibeVoice and Dia
are **set aside** (2026-07-29, reversible, `ref_select.SET_ASIDE`) — VV stages scenes
on dialogue, and the survivors already narrate at 94%. Engine standing, license
verdicts, and the shortlist record: [teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md);
onboarding pattern + gotchas: [tts-engine-onboarding.md](tts-engine-onboarding.md).

**Delivery campaign (SSOT: [delivery-mix-campaign.md](delivery-mix-campaign.md)):**
1,071 fold-eligible keeps; Dialogue 578 is the 50% anchor, corpus completes at 1,156.
Newscaster and Speech are CLOSED; remaining **+116 = Neutral +81 / Documentary +35**,
riding on five books ingested 2026-08-02. Text shelf refilled (it had silently emptied
to 20 lines).

**Engine allocation is now three-layer** (`ref_select.py`, 2026-08-02): capability veto
(`ENGINE_CHANNELS` — the relay audit made executable), measured per-lane weights
(`ENGINE_MIX_BY_LANE`, heard production verdicts only), and a diversity floor
(`MIN_SHARE`/`MAX_SHARE_NARRATION` — max keep-rate is *not* the objective; a
one-timbre teacher corpus has failed at its job). This replaced the flat ratified mix
and resolved the "director picks engines per line" contradiction: the director's
roster now derives from `ENGINE_MIX − SET_ASIDE`.

**Audit tiers** (tag on an engine, separate from render share): trusted (qwen,
chatterbox-provisional) · normal (orpheus — explicit, its record was dragged by the
now-banned `tara` fallback) · scrutinized (zonos, moss_vg — 100% heard). Unknown
engines now default to **scrutinized**, not normal.

**Zonos tier decision is one bank away.** newscaster-v1 (78 clips): qwen 27/27 · zonos
29/31 · moss_vg 20/20. Three campaigns agree zonos's instability tracked the
neutral-dominant emotion conditioning, not the engine (93.5% keep with emotion truly
off vs 27% with the vector). The director path now produces true-unconditional zonos
narration end to end (verified on conan-stories, 39/40) — one audited director-driven
bank flips scrutinized → normal. Full ears queue: [todo.md §4](todo.md).

**Director models:** `gemma-4-31b-qat-spec` directs (skill-file obedience 24/24 where
the MoE managed 8/24 and e4b 5/24); `gemma-4-e4b-qat-spec` handles volume jobs
(labeler, judge_passages). **e4b for volume, 31b for judgement — e4b is NOT the
director.** Record: [book-prose-lane.md](book-prose-lane.md) § Director model.

## Pipeline integrity (hardening rounds, 2026-07-31 → 08-02)

- **QC gate is mandatory and enforced**: `synth_bank.sh` exits without registering on
  failure, refuses an empty measures file, and `qc_gate` exits nonzero on zero
  manifests/records/passes (the empty-glob "0/0 hard-pass" hole is closed). Every QC
  failure is auditioned at every tier; findings attach as direction-aware notes, never
  verdicts.
- **Dead-air gate shipped and verified** (Silero): hard 2.5 s + 1.4 s advisory that
  queues for the ear. Calibrated on 369 audited clips — catches 20 of 21 known
  pause-drops at a 3.3% flag rate; on newscaster-v1 the audit queue contains both of
  the ear's drops (zero missed, vs 2 before).
- **Ear-evidence provenance is clean again**: folded clips are keeps with a blank score
  + provenance note (the 60 fabricated `score=5` rows were blanked);
  `--mark-delivery`, `gate_calibration`, `reader_profile --learn` and novelty sampling
  all distinguish machine-written cells from ear-written ones. `ratings.csv` writers
  are guarded (audit_sampler included); `qc_flags.txt` writers append-dedup.
- **The book lane cuts clips correctly**: pad-before-clamp fixed (librivox-speech-v1
  re-cut deterministically, 69/79 windows byte-identical, zero heard verdicts
  invalidated); manifests dedup on clean exit; zero-clip books exit nonzero.
  `librivox-v2` (Uneasy Money re-cut, 1,366 clips / 3.13 h, all 25 sections) is the
  pool to stage going forward; `librivox-v1` stays only for its 12 valid ear verdicts.
- **Atomic writes** across all 8 renderers (`synth_common.py`); orpheus voice ban and
  zonos `emotion: null` are code, not markdown; loudnorm failure is fatal.
- teacher-ab-v1 keeps normalized to −23 LUFS (the 5.99 dB engine spread confounded the
  Qwen-vs-VV ranking; re-test is now possible — [todo.md §4](todo.md)).

## Model architecture fronts

- **Base reaffirmed:** Matcha stays; StyleTTS2-Lite retired as contingency (2026-07-29);
  the escape hatch is a scaled flow-matching backbone. Kokoro stays out. Records:
  [model-decisions.md](model-decisions.md).
- **Decoder v2 (DiT) is a staged spike NOW** — adopt iff it passes the parity gate vs
  the U-Net baseline. The spike's one named risk (tiny-DiT spectral texture) is already
  retired in public MIT code: StableTTS's 31M `DiTConVBlock` (conv FFN in-block + long
  skips + adaLN-Zero) with MAS retained. Start from that block shape; carry the CFG
  direction-strength lever into the spike's scope.
  [matcha-siblings-study.md](matcha-siblings-study.md) is the standing comparison
  bench — check it before designing any component blind.
- **Contract v2** (2026-07-30): delivery is the 4th FiLM channel (5 lanes + unknown ≡
  zero); register compiles away Director-side; tempo/loudness stay host-side. Pinned in
  [ARCHITECTURE.md](ARCHITECTURE.md) §1. The model core does not implement it yet —
  seam assertions are pre-work ([todo.md §1](todo.md)).
- **RapFlow-TTS (consistency-FM, ~2 NFE)** logged as its own later spike — throughput
  lever, separate de-risk cycle.

## Export lane

Plan A is the `litert-torch` split-graph export (textenc / decoder / vocoder, host-side
ODE), verified at parity on Phase 0 and adapted to the derisk checkpoint (`spk` + `vat`
inputs, all graphs GPU-clean, e2e corr ≥ 0.9993). The `torch → ONNX → onnx2tf`
monolith is Plan B. Harness: `/data/toolchain/litert-conversion/`.

⚠ The lane's gate suite currently *reports* rather than *refuses*, and
valence/tension have never been driven nonzero through a converted graph — that plus
the delivery export story is the blocking work before any vat3/delivery export
([todo.md §2](todo.md)).

## Next actions (short list)

1. delivery-v1-narration round 2 (+116 keeps; carries the zonos tier bank).
2. Ears queue in priority order — [todo.md §4](todo.md).
3. v3c experiment config + container `data_statistics` → next training run (delivery
   channel decisions ride it).
4. Export-lane gate hardening before that run's export.

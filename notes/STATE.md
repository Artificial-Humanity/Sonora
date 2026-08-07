# Project State — Sonora

The current-state snapshot: what is true now, what is running, and what comes next.
Behavioral rules and the stack/layout manifest live in [AGENTS.md](../AGENTS.md); the
architecture canon is [ARCHITECTURE.md](ARCHITECTURE.md); open work is
[todo.md](todo.md). One entry per front, newest facts first — superseded narrative is
deleted, not banner'd (git history is the archive; the pre-2026-08-02 roadmap
narrative was removed in the consolidation pass).

_Last updated: 2026-08-06._

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
> The `vat3_finetune` ep099 checkpoint trained on **v2** — poisoned phonemes included —
> but the labels barely moved between v2 and v3 (corr ≥ 0.9993 per channel on shared
> clips), so **vat3 is a viable fine-tune base**; a from-scratch retrain is not forced.
>
> **COMPLETE 2026-08-06, AND THE RUN IS RETIRED** — `vat3c_finetune` ran 100 epochs (run
> `2026-08-05_15-02-57`, exit 0), warm-started 338/338 (0 fresh) from vat3-24k ep099,
> after the 08-04 launch was killed by the E610 NIC fault. The ear said it bought nothing
> audible; the never-trained holdout, scored the same day, says it was **slightly worse**
> — +0.0164 against its own warm start, every loss term worse, and worse on v3c's own val
> split too. **The `op_g2p` fix was the whole fix; the fine-tune was a regression.** Use
> **vat3-24k ep099** as the base. Note this is a verdict on the *run*, not the corpus —
> v3c is still the corpus to build Phase 1 on. Paired per-clip scoring, blind A/B,
> copy-synthesis and the holdout table: **[quality-gap-plan.md](quality-gap-plan.md)**.
> Launch traps: [training-operations.md](training-operations.md).
>
> `configs/experiment/vat3c_finetune.yaml` points at
> it, `data_statistics` are **measured** on the 30,485-clip train split
> (`mel_mean −5.525067 / mel_std 2.388571`, vs v2's −5.504811 / 2.386137), and the v3
> line is finally declared in the license wall — it never had been, so `enforce()` would
> have refused any run that reached for it. Composition, a real batch, and a conditioned
> `synthesise` are all verified in-container. Lineage and every source:
> **[training-sources.md](training-sources.md)** (SSOT). Residual review debt that
> touches training: [todo.md §1](todo.md).
>
> Pre-flight for the run itself is unchanged and non-negotiable: stop **all** inference
> engines first ([spin-down rule](training-operations.md)), and run
> `scripts/test_vat_dim_seams.py`.

## The delivery channel — seams guarded, migration not started

`vat_dim` is still **3** everywhere; nothing about the 4th channel has been built. What
changed 2026-08-04 is that the four places which silently assumed 3 now refuse to: the
FiLM trunk, the filelist parse, the collate, and the export converter. Each is driven
with the wrong width and proven to fail in `scripts/test_vat_dim_seams.py` (13 checks) —
a guard nobody has watched fail is a guess. The data configs interpolate
`vat_dim: ${model.vat_dim}`, so the model config is the single source of truth and
bumping it to 4 now fails loudly at the filelist instead of quietly at the trunk.

The collate was the one worth the trouble. It was all-or-none: a single item without a
VAT dropped conditioning for the **whole batch**, silently, because `vat=None` is
legitimate everywhere downstream (it means neutral, and the encoder substitutes zeros).
A partially-labelled filelist would not have crashed or warned — it would have trained a
conditioned model on unconditioned batches, and the only symptom would be a channel that
never learned. Delivery is exactly the channel to hit it: unknown ≡ zero is a real,
intended value, and 17 clips are delivery-blank by design.

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
| `vat3-24k` ep099 | libritts_r_vat **v2** | energy PASS / tension near-pass / valence FAIL; staged; the warm start for `vat3c`. **Best checkpoint in the lineage on never-trained audio (1.8512)** — beats its own warm start by −0.0111, the gain concentrated in `diff` (−0.0241, 78.7% of clips). This is the Phase 1 base |
| `vat3c` ep099 | libritts_r_vat **v3c** | 2026-08-06, warm from vat3-24k 338/338 (verified bit-identical by scoring, not just by the log line). **RETIRED — a regression, not a no-op.** +0.0164 worse than its own init on the holdout, all three loss terms worse, better on 39.1% of clips; +0.0443 worse on v3c's own val too. Do not stage or export |
| 24 kHz HiFi-GAN vocoder `g_02510000` | LibriTTS-R, warm from UNIVERSAL_V1 | copy-synthesis converged, human-audited; staged. **Confirmed perceptually transparent 2026-08-06** — mel L1 = 10.2% of mel_std |

Publishing staged checkpoints to the public HF repo remains a deliberate owner call —
they are validated components, not the shippable directable actor.

**Every VAT checkpoint in this table now has a never-trained score (2026-08-06).**
`data/libritts_r_holdout_devclean` — LibriTTS-R dev-clean, 5,463 clips / 8.7 h / 40
speakers disjoint from the corpus's 247 — is the standing instrument; score new
checkpoints on it with `scripts/score_holdout.sh` (~100 min for eight on an idle card).
Full results and what they closed: [quality-gap-plan.md § 0a](quality-gap-plan.md).

⚠ `loss/val_epoch` remains unusable for cross-run comparison and always will be — splits
were re-drawn per corpus version while runs warm-started from the previous one, so 93–97%
of v3c's val clips were trained on earlier. That contamination is historical and does not
heal. **The holdout is the number; MLflow val is not.**

## Teacher synthesis & the expressive dataset

**Portfolio (five):** chatterbox · qwen · zonos · orpheus · moss_vg. VibeVoice and Dia
are **set aside** (2026-07-29, reversible, `ref_select.SET_ASIDE`) — VV stages scenes
on dialogue, and the survivors already narrate at 94%. Engine standing, license
verdicts, and the shortlist record: [teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md);
onboarding pattern + gotchas: [tts-engine-onboarding.md](tts-engine-onboarding.md).

**Delivery campaign — COMPLETE 2026-08-04 (SSOT:
[delivery-mix-campaign.md](delivery-mix-campaign.md)).** **1,189 fold-eligible keeps**
against the 1,156 it was sized for, at a measured 49.3 / 30.2 / 7.4 / 7.2 / 5.9 versus
the ratified book-actor mix of 50 / 30 / 8 / 6 / 6. Dialogue 578 is the anchor; Neutral
closed at 354/347 when 30 real-audio clips from `librivox-v2` came back 30/30 Neutral;
Newscaster 84, Speech 69. Documentary stays at 87/92 by owner call — no documentary
real audio is segmented anywhere and −5 does not pay for a fresh ingest.

The corpus is no longer the constraint. **The 4th FiLM channel it was built for still
does not exist in the model core** — see [todo.md §1](todo.md).

**Engine allocation is now three-layer** (`ref_select.py`, 2026-08-02): capability veto
(`ENGINE_CHANNELS` — the relay audit made executable), measured per-lane weights
(`ENGINE_MIX_BY_LANE`, heard production verdicts only), and a diversity floor
(`MIN_SHARE`/`MAX_SHARE_NARRATION` — max keep-rate is *not* the objective; a
one-timbre teacher corpus has failed at its job). This replaced the flat ratified mix
and resolved the "director picks engines per line" contradiction: the director's
roster now derives from `ENGINE_MIX − SET_ASIDE`.

**Audit tiers** (tag on an engine, separate from render share): trusted (qwen,
chatterbox-provisional) · normal (orpheus — explicit, its record was dragged by the
now-banned `tara` fallback; **zonos** as of 2026-08-04) · scrutinized (moss_vg —
100% heard). Unknown engines now default to **scrutinized**, not normal.

**Zonos promoted scrutinized → normal (2026-08-04).** The instability was never the
engine — it tracked the neutral-dominant emotion vector (93.5% keep with emotion truly
off vs 27% with it). What was missing was proof the DIRECTOR path could produce
true-unconditional zonos in production, since every earlier narration bank had
`emotion: null` hand-patched in. delivery-v1-narration-r2 closed that: 44/44 lines
directed with emotion off, 37 keeps from 38 heard, and the one non-keep is a
bookkeeping retirement rather than an ear verdict.

**moss_vg held at scrutinized.** 12/12 on narration is clean but thin, and the prior
20/20 was confounded — Newscaster flatters its radio-timbre failure mode, and its 24%
rate was measured on expressive material. Re-test on dialogue or an expressive
register before moving it. Full ears queue: [todo.md §4](todo.md).

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

The ordered plan, with the gate between each phase, is
**[quality-gap-plan.md](quality-gap-plan.md)** — it is the SSOT for sequencing and this list
is only its headline.

1. ~~**Phase 0a — the never-trained holdout.**~~ **DONE 2026-08-06.** Its gate passed
   (checkpoints do separate on unseen audio), it retired `vat3c` ep099 as a regression,
   and it closed **0b** — the clean-lineage retrain from `matcha_vctk` is **not
   indicated**, because the lineage demonstrably generalizes. Owner's call to ratify.
2. **Phase 1 — data, cheapest first**, warm-starting from `vat3-24k` ep099.
   Emilia-YODAS keeps (+43%) → expressive-registers (+116) → LibriTTS-R 10× → Hi-Fi TTS
   v1. #1 is the fastest test of whether volume moves quality at all, and the 10× is
   gated on its answer. 0a supplies the evidence this phase was assuming: 100 epochs
   against the same ~30k clips made the model *worse*, so the lever is corpus, not epochs.
3. **Phase 2 — the DiT decoder spike**, after Phase 1 lands, against a same-corpus U-Net
   baseline frozen as the last act of Phase 1.
4. Ears queue in priority order — [todo.md §4](todo.md).
5. Export-lane gate hardening before any vat3c/delivery export — [todo.md §2](todo.md).

## Pointers

- Change history — [CHANGELOG.md](CHANGELOG.md) (maintained per AGENTS.md §4 since 2026-08-06)
- Code on `/data` — [data-mirrors.md](data-mirrors.md). Nothing of ours is unbacked; the
  risk is **drift**, and it had already bitten: the running export harness was three weeks
  stale and missing a seam guard the repo recorded as landed. `tests/test_data_mirrors.py`
  fails on any divergence. **The repo is authoritative; `/data` is a working copy.**
- Latest code review — none yet (the first review follows AGENTS.md §5: with no prior review,
  it covers the previous and current day's commits)

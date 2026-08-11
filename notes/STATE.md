# Project State — Sonora

The current-state snapshot: what is true now, what is running, and what comes next.
Behavioral rules and the stack/layout manifest live in [AGENTS.md](../AGENTS.md); the
architecture canon is [ARCHITECTURE.md](ARCHITECTURE.md); open work is
[todo.md](todo.md). One entry per front, newest facts first — superseded narrative is
deleted, not banner'd (git history is the archive; the pre-2026-08-02 roadmap
narrative was removed in the consolidation pass).

_Last updated: 2026-08-10._

---

## VAT corpus — train on v5

> **Train on `data/libritts_r_emilia_vat_v5`.** Built 2026-08-08 — the first corpus here
> that is not one dataset. **41,138 train / 1,304 val · 78.5 h · 2,500 speakers**, against
> v4's 30,485 / 960 / 51.3 h / 247. It is v4's LibriTTS-R rows **byte-identical** plus
> **10,997 Emilia-YODAS keeps**. Warm start `warmstart/vat5_init.ckpt` at **338 warm
> (1 widened) / 0 fresh**, seam guards **30/30** in-container, smoke run trains to
> `val_epoch` 1.696. `configs/experiment/vat5_finetune.yaml`.
>
> **TRAINED 2026-08-08/09 — 48 epochs, holdout-scored, `ep019` SELECTED** (owner, on the
> ear; converged by epoch 9, so epochs 10–39 were net worse and the run is closed —
> `logs/train/vat5_finetune/SELECTED.md` refuses a relaunch). Rung 1's gate **PASSED**:
> `ep019 − vat5_init` = **−0.0606** on the clean holdout, improved on 82.1% of 5,463 clips,
> an order of magnitude past the −0.0111 that cleared gate 0a. **Data-limited, not
> capacity-limited — the 10× proceeds.** ep019 is the warm-start donor for v6.
>
> **It is a merge, not a derivation** (`scripts/merge_emilia_corpus.py`), and the two halves
> carry labels on deliberately different scales. LibriTTS keeps its per-speaker z; Emilia
> gets a **global anchor** against v4's distribution, because 2,408 speakers at a median of
> 3 clips have no per-speaker statistic and re-centring would hand 756 one-clip speakers a
> label of exactly 0.0 — clips selected FOR being extreme, trained as neutral. Owner's
> option 1. The semantic cost (a global anchor leaves some speaker identity in the affect
> channels) is stated in the data config rather than hidden.
>
> ⚠ **T saturates at 53.6%** on the Emilia half against LibriTTS's 4.7%, owner-accepted for
> this run. **That is not a labelling bug — it is the mining criteria arriving at the
> label**: keeps were tail-selected at a T threshold of 5.75 sigma, so the Emilia half is
> 27 hours of extremes against a LibriTTS centre at roughly 3:1. The failure signature is
> written down in [quality-gap-plan.md](quality-gap-plan.md) **before** the result exists
> and must be read before the holdout number, not after.
>
> ⚠ **`loss/val_epoch` is worse than useless here** — v5's val set is 1,304 clips of which
> 344 are Emilia, so it mixes two domains and a move in it could be either. Score on
> `data/libritts_r_holdout_devclean`; one epoch on it destroys it permanently.
>
> **Three of the plan's own numbers did not survive contact**, all recorded rather than
> quietly absorbed: "all 13,141 rows" is **10,997** (1,676 carry digits — D-M3 — and 468
> exceed the shared `ASR_MAX_WER`); `n_spks` 2,655 is **2,500**, because 155 speakers lost
> every clip; and `data_statistics` moved **ten times the precedent** that put "re-measure
> in-container" on the checklist — mel mean **0.1587**, std **0.3208**, against v2→v3c's
> 0.0203 / 0.0024. Measured values: `mel_mean −5.683762 / mel_std 2.709323`.
>
> **v4 is superseded and never ran past a smoke pass.** It remains the base v5 merges into
> and the donor its warm start widens from. The owner's call was to smoke rather than spend
> 100 epochs, because v4 is the same 31,445 clips `vat3c_finetune` already spent 100 epochs
> on and came back a measured regression — a full run would re-buy a conclusion we own.
> The v1–v4 lineage and every source: **[training-sources.md](training-sources.md)** (SSOT).
>
> ⚠ **What v5 does NOT close: the ear gap.** Its new material was mined on acoustics and
> nobody has heard a clip of it; every Emilia row is delivery-`unknown`, so the corpus still
> carries 45 delivery labels in 41,138 train rows. **Phase 1 rung 2** —
> `sonora-expressive-registers`, 832 rows to append — is what closes it. The order is
> deliberate: rung 1 asks "does volume move quality at all?" for no ear time.
>
> Pre-flight is unchanged and non-negotiable: stop **all** inference engines first
> ([spin-down rule](training-operations.md)), and run `scripts/test_vat_dim_seams.py`
> (30 checks). Residual review debt that touches training: [todo.md §1](todo.md).
> The quality ladder, one table: **[quality-gap-plan.md § the pathway](quality-gap-plan.md)**
> — the route to a model worth casting. Casting itself is **parked** with an
> end-condition (same file, § Parked), not scheduled; it is half of goal 1 and no rung
> ladders to it.

## The delivery channel — SHIPPED on the training side (2026-08-07)

`vat_dim` is **8**: three V/A/T channels plus a five-wide one-hot delivery block.
`matcha/delivery.py` is the only definition of that encoding, and the corpus derivation,
the CLI, the Vocalizer and the export converter all read it.

**Eight, not the four the review proposed.** A single ordered channel asserts the five
lanes lie on one continuum, and `seed_delivery.py` records that they do not — Dialogue vs
Neutral is a property of the TEXT, Newscaster vs Documentary a property of the RENDER. One
channel also cannot carry `unknown`, which the contract pins as the zero vector: on one
channel zero is the MIDDLE of the range. One-hot makes `unknown` all five channels at
zero — the absence of a value rather than a value on the axis — so a corpus with no
delivery labels reproduces v1 conditioning exactly. Owner call; ~1k extra parameters.

The 2026-08-04 seam work is what made this safe to do: the four places that silently
assumed 3 — FiLM trunk, filelist parse, collate, export converter — each refuse a
mismatch, and the data configs interpolate `vat_dim: ${model.vat_dim}` so the model config
is the single source of width. `scripts/test_vat_dim_seams.py` is now **35 checks** and
passes; it is the thing that makes a width disagreement fail loudly at the filelist
instead of quietly at the trunk.

⚠ Two things do NOT follow automatically. **Every existing filelist is 3-wide** and must
be re-derived (`--delivery-from`), and **the export lane is deliberately not migrated** —
`convert_vat.py` still refuses a wider checkpoint, because F-H2 is open and a mobile host
told nothing about the last five channels being categorical will interpolate them.

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

Every VAT checkpoint here has a never-trained holdout score. Score new ones with
`scripts/score_holdout.sh` before quoting any curve; stratify with
`scripts/stratify_holdout_sweep.py`.

| checkpoint | trained on | standing |
|---|---|---|
| Phase 0 `ep199` (baseline-ljspeech-22k) | LJSpeech @ 22.05 kHz | shipped to HF; the v1 voice; both export lanes verified on it |
| `derisk-energy-24k` ep099 | libritts_r_vat v1 (energy only) | §7 verdict PASS; staged, not pushed |
| `vat3-24k` ep099 | libritts_r_vat **v2** | energy PASS / tension near-pass / **valence FAIL**; staged. Was the Phase 1 base until v5 |
| `vat3c` ep099 | libritts_r_vat **v3c** | **RETIRED — a measured regression, not a no-op. Do not stage or export.** The launcher refuses the experiment by name |
| **`vat5_finetune` ep019** | **libritts_r_emilia_vat_v5** | **THE PICK** (owner, on the ear). 48 epochs, converged by epoch 9; **−0.0606 on the holdout** against its own init. **The warm-start donor for v6.** The run is closed — `SELECTED.md` refuses a relaunch |
| 24 kHz HiFi-GAN vocoder `g_02510000` | LibriTTS-R, warm from UNIVERSAL_V1 | **perceptually transparent** (mel L1 = 10.2% of mel_std) — not a quality lever |

Publishing staged checkpoints to the public HF repo remains a deliberate owner call — they
are validated components, not the shippable directable actor.

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

The corpus is no longer the constraint, and as of 2026-08-07 neither is the model core:
**the 4th FiLM channel it was built for exists**, as a five-wide one-hot block
(`vat_dim` 8). What is left is re-deriving the corpus at that width — see
[todo.md §2](todo.md).

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

**moss_vg held at scrutinized — and the hold no longer has evidence behind it (checked
2026-08-08).** The dialogue re-test it was waiting for already exists: **19 heard at 89.5%**
against a 73.5% lane mean, above `chatterbox`, which is trusted. The "**24% rate on
expressive material**" cited here is not reproducible from anything on disk and appears to
be a corrupted restatement of `4abfd3f`'s measurement — `moss_vg 19/54 (35%)` **rejected on
`delivery-v1-narration`**, a narration campaign. The one real weakness is **Documentary
(60.0% vs an 83.3% lane mean)**, and `ENGINE_MIX_BY_LANE` already holds moss_vg to a 0.05
floor share there for that reason. Promotion to `normal` is the owner's call; the
measurement supports it. Full ears queue: [todo.md §5](todo.md).

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
  Qwen-vs-VV ranking; re-test is now possible — [todo.md §6](todo.md)).

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
  [ARCHITECTURE.md](ARCHITECTURE.md) §1. **Implemented in the model core 2026-08-07** as a
  five-wide one-hot block (`vat_dim` 8, `matcha/delivery.py`); the EXPORT half is still
  open ([todo.md §1](todo.md)).
- **RapFlow-TTS (consistency-FM, ~2 NFE)** logged as its own later spike — throughput
  lever, separate de-risk cycle.

## Export lane

Plan A is the `litert-torch` split-graph export (textenc / decoder / vocoder, host-side
ODE), verified at parity on Phase 0 and adapted to the derisk checkpoint (`spk` + `vat`
inputs, all graphs GPU-clean, e2e corr ≥ 0.9993). The `torch → ONNX → onnx2tf`
monolith is Plan B. Harness: `/data/toolchain/litert-conversion/`.

**The export lane closed 2026-08-07.** The gate suite refuses rather than reports (F-C1,
08-06) and `config.json` now carries a machine-readable **control contract**: the
continuous range with a reject-don't-clamp rule, the one-hot delivery vocabulary with its
explicit unknown vector, and CFG's method plus its ≥25-ODE-step floor. A host can validate
BEFORE it renders — three of five demonstrated bad vectors were previously accepted
silently. Gates gained G3b/G3c (gain and normalised-RMSE: correlation is scale-invariant,
so `0.5 × reference` scored **1.000000**) and G6 (the delivery lanes must be mutually
distinguishable). The referee, the tensor renamer and `kotlin_replica` are all fixed —
see `git log` for the commits.

**The text front end joined the gates on 2026-08-07 (G7).** Everything above certifies the
GRAPH; nothing asked whether the text reaching it matches the text the model trained on,
and that fails the same way — fluent audio, wrong phonemes, nothing to see at the shell.
`kotlin_replica` had no contraction table, which is **D-C1 still live on the device side**
five days after the host was fixed, and it diverged from the training front end on **69 of
86 probe sentences**. The front end now lives in `scripts/litert_export/device_g2p.py`, the
apostrophe tables ship as `g2p_contractions.json` **exported from `matcha.text.op_g2p`**
(hand-syncing them is the same defect on a delay), a device that cannot find the asset
refuses rather than falling back, and `config.json` carries a `g2p` block declaring which
front end the graphs expect. **There is no mobile app** — front-end development has not
started in earnest — so the replica is not validating something nobody runs: it is the
spec, fixed before anyone ports it.

⚠ G7 **refuses a homograph-enabled export**, so if D-M4 is turned on for the next
derivation, `matcha/text/homographs.py` has to be ported to the device before the model
can ship. That cost belongs to the D-M4 decision ([todo.md §3](todo.md)) and was invisible
when it was filed.

⚠ What remains is a **re-export**: the artifacts on `/data` are the 3-channel era, and
`convert_vat.py` refuses a mismatched checkpoint by design. Re-derive the corpus at
`vat_dim` 8, retrain, then convert.

## Next actions (short list)

The ordered plan and the gate between each phase is
**[quality-gap-plan.md](quality-gap-plan.md)** — the SSOT for sequencing. This is only its
headline; open items are in [todo.md](todo.md).

1. **Phase 1 rung 2 — build v6** (+expressive-registers): decided, NOT built, warm start
   from **`ep019`**. The append set is settled at **832 rows** (822 delivery-labelled + 10
   blank) = 1,004 eligible − 158 whose audio is already in v5 — **three disjoint duplicate
   classes needing opposite treatment** (91 dropped, 45 excluded because v5 holds their
   labels, 22 `mk_` twins resolved to one row), not the 91 originally recorded. Those 846
   are **scored for valence** on a uniform 12-head EIV pass:
   `eiv_scores/expressive_registers_v6.jsonl`, derived to `corpus_soft_v6.json` /
   `corpus_valence_combo_v6.json` (31,197 entries = the v4 lineage + these 846, *not* the
   merged corpus).
   ✅ **All three prerequisites CLOSED 2026-08-10 — the 846 now carry V, A and T.** The
   acoustic pass had genuinely never run (`scripts/measure_expressive_registers.py` is the
   third measure producer, for the third corpus shape); **846/846 measured, 846/846
   labelled, 0 unmeasurable**. The lane is the **global anchor**, same as Emilia
   (`scripts/label_expressive_registers.py`) — per-speaker z would hand every appended row
   exactly 0.000, since one id per clip means n = 1.
   ⚠ **A needed a loudness correction that V and T did not.** This bank is loudnorm'd
   (median exactly −23.00 LUFS), LibriTTS-R is not (−18.16), and a 2.8-sd gap pinned **A at
   −1 on 94.4% of rows** — our encoder's target recorded as a property of the voice. It is
   centred **per campaign, not per bank**, because the bank holds at least three loudness
   targets and one offset still left the 103 librivox rows at −0.835. Final: A mean +0.024,
   sd 0.217, **2.4% clamped**. ⚠ A stays near-constant for the 638 clips loudnorm flattened
   and that is CORRECT — the model reproduces the normalised audio.
   **14 clips exceeded `MAX_SECONDS` (22 s) and were dropped** (owner, 2026-08-10) — all 14
   delivery-labelled, so the append is **832 = 822 labelled + 10 blank** and the blank half
   is untouched. The distribution did not move: V +0.005 / A +0.024 / T +0.261.
   ✅ **BUILT 2026-08-10 — `data/libritts_r_emilia_expressive_vat_v6`.**
   **41,937 train / 1,331 val · n_spks 3,326 · +826 rows · +2.08 h**, from
   `scripts/merge_expressive_registers.py`. v5's rows pass through **byte-identical** on
   both files (verified, not assumed) and v5's split reproduces under the shared hash, so
   rung 2's holdout stays comparable to rung 1's. Merged delivery: Dialogue 344 ·
   Neutral 328 · Newscaster 76 · Speech 68 · blank 10.
   **6 of the 832 dropped on digits (D-M3)** in the v6 append — five correctly (the audio
   speaks £5,000, 43° 10′, 1801, 20–30 fathoms); the sixth is a Gutenberg footnote marker
   `{53}` the reader does not say, costing a row from the 69-row **Speech** lane. Flagged,
   not stripped.
   <!-- "in the v6 append" is load-bearing: it is how scripts/test_doc_claims.py knows
   which corpus the 6 and the 832 above belong to. The registry used to reach this line by
   the rule id D-M3, which every v5 statement of the same rule carries too, so the v6
   entries claimed v5 sentences (#48). Keep a corpus marker on this line. -->
   ⚠ **The licence wall REFUSED the first build** — the staged 24 kHz tree was undeclared.
   `sonora_expressive_registers` is now in `configs/data_licenses.yaml`, verified by path:
   **0 of the 832 clips resolve into `LibriTTS_R` or any `emilia*` tree.**
   ⚠ **TWO IN-CONTAINER STEPS REMAIN, neither optional**: `data_statistics` must be
   **re-measured** (they cannot be inherited from v5 — this changes the audio set *and*
   the split), and `scripts/test_vat_dim_seams.py` must pass. `vat_dim` is unchanged at 8,
   so **`ep019` warm-starts with no widening.**
   Full derivation, tables and the rejected alternatives:
   [quality-gap-plan.md § Rung 2 build decisions](quality-gap-plan.md#rung-2-build-decisions--recorded-2026-08-09-corpus-not-built-no-run-queued).
2. **Rung 3 — the 10×** (LibriTTS-R full, ~615 h). Ungated: rung 1 passed. ~2.25 h/epoch,
   about a day to convergence — it does not need the local-vs-cloud decision first.
3. **Phase 2 — the DiT decoder spike**, after Phase 1 lands, against a same-corpus U-Net
   baseline frozen as the last act of Phase 1.
4. **A forced-ranking pass over the 46 ceiling-tied groups** — the scale cannot separate
   six engines at 5, and the corpus trains on keeps ([todo.md](todo.md) §4).

## Pointers

- Change history — `git log`, the commit messages, and the pull requests. **There is no
  changelog**: `CHANGELOG.md` was retired 2026-08-11 (AGENTS.md §4), together with the
  review-document cycle that cross-referenced it.
- Review sweep — 47 open items down to **9**, with **all three §1/§3 corpus decisions
  taken** on 2026-08-07 and only their execution left. §2's QC/audit/staging block closed
  on 2026-08-07 (`1ebd14a`, `0a505d3`, `a4b6ec5`, `97c14b4`): C-M5, C-M8 and C-L4 done,
  C-M4 down to one ear-blocked threshold. Three more findings were **live**: 15 clips
  across two campaigns had already shipped unnormalized (up to 7 dB off, one bank spanning
  12.7 dB), 8 clips passed every gate while missing three or more opening words, and the
  device G2P was still running the front end D-C1 condemned — 69 of 86 probe sentences
  wrong, five days after the host was fixed. The 2026-08-06 round closed both §1
  blockers (E-M5/E-M6), §5's Highs, §3's packaging and GPL/cli lane, §8's label-derivation
  bugs, **F-C1** (the only Critical) and F-H1/F-M4, plus C-M10 and D-M5. The **2026-08-07**
  round (`87c65f8`..`72786ac`, twelve commits) closed the whole of §1, §3, §5 and §6 and
  most of §7 — the host suite went 119 → **264** tests, the vat_dim seam checks 13 → **22**. The export lane (§1) closed the next day, `905e91b`..`e18877a`.
  Every fix carries a regression test; the running tally is [todo.md](todo.md).
  Six findings turned out to be **live or wider than filed**: the export harness ran three
  weeks stale on `/data`; `eiv_merge_corpus` fabricated a full-scale valence contribution
  from floating-point dust for 224 clips; the ONNX "Plan B" exporter silently dropped every
  conditioning channel; A-M2 was the tail truncation the aligner already blamed on CTC
  (**54% of a sentence dropped**, measured); an unpinned render lane resolves
  `transformers` **5.14.1** today against the 4.x the corpus was built on; and
  `make_bulk_bank.py` had been sending all 87 qwen lines to the model with the voice design
  stripped — verbatim the 2026-07-25 finding `build_direction` exists to prevent.
- Code on `/data` — [data-mirrors.md](data-mirrors.md). Nothing of ours is unbacked; the
  risk is **drift**, and it had already bitten: the running export harness was three weeks
  stale and missing a seam guard the repo recorded as landed. `tests/test_data_mirrors.py`
  fails on any divergence. **The repo is authoritative; `/data` is a working copy.**
- Review findings live in the pull request that raised them (AGENTS.md §1/§5); the
  timestamped review documents were retired 2026-08-11. ⚠ The pattern the final sweep
  found is worth carrying regardless of where findings get recorded: **both High findings
  were enforcement code that existed and was never wired in** — written, reviewed,
  committed, never called. Check that a guard is INVOKED, not merely present.

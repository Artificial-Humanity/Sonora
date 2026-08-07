# Open items — post-review residue

Distilled 2026-08-02 from the full code review (`codereview.md`, HEAD `47a405d`) and its
remediation log (`cleanup-20260802.md`). Both source documents are deleted; their full text
is in git history. **Every Critical and High finding was fixed** (`b9c7a7e`..`1faac46`,
verified in source) — this file is only what remains open, letter-coded as the review
coded it so the git-history record stays joinable.

Items are grouped by when they bite, not by subsystem. **Landed items are deleted, not
struck** (git history is the archive, per [README.md](README.md)); delete sections when
empty.

_**Pruned 2026-08-07** after a sweep that closed §1, §3, §5, §6 and most of §7 —
twelve commits, `87c65f8`..`72786ac`, each with a regression test; the suite went from 119
to 264 host tests plus 22 container seam checks. Sections 1, 3, 5 and 6 are now empty and
deleted. Four items turned out to be **live or wider than filed** and are recorded in
[CHANGELOG.md](CHANGELOG.md) rather than here: the ONNX "Plan B" exporter silently dropped
every conditioning channel (F-H1's shape); A-M2 was the tail truncation the aligner already
blamed on CTC, measured at **54% of a sentence dropped**; an unpinned render lane resolves
`transformers` **5.14.1** today against the 4.x the corpus was built on; and
`make_bulk_bank.py`'s bypass of `build_direction()` had been sending all 87 qwen lines to
the model with the voice design stripped._

_Earlier pruning notes for the 2026-08-06 round are in git history._

This file is **residue, not the plan.** What to do next, and in what order, is
[quality-gap-plan.md](quality-gap-plan.md).

---

## 1 · Export lane (before any vat3/delivery export)

_**F-C1 and F-H1 closed 2026-08-06.** The gate suite is a ledger and `gates_or_die()`
refuses to write artifacts unless every gate passes; **G5 drives valence / energy / tension
independently and requires each to move the waveform** (measured 7.0e-2 / 9.2e-2 / 8.6e-2
mean |delta| vs neutral — the first time V and T have been nonzero through a converted
graph). Also fixed: G4's monotonicity used `all()` over a possibly empty sequence, and
**F-M4**'s `config.json` wrote `time_embed_dim=1024` where the graphs consume
`in_channels` (224). 10/10 gates pass._

**This section is now the blocker it was always going to be.** The training side of the
delivery channel landed 2026-08-07 (`vat_dim` 3 → 8, one-hot; see
[ARCHITECTURE.md](ARCHITECTURE.md) §1 and `matcha/delivery.py`), and `convert_vat.py`
correctly refuses a wider checkpoint. Its refusal message names what is missing.

- [ ] **F-H2:** no delivery export story. Two halves now, and the second is new:
      nothing records or enforces the 2σ clamp contract for mobile hosts; and nothing
      tells the host that the last five channels are **categorical**. A host that
      interpolates them — as it reasonably would, having been handed eight floats and told
      three of them are continuous — produces a vector `delivery.lane_of_vector` refuses.
      `config.json` needs the lane vocabulary and the channel semantics, not just a width.
- [ ] **F-M1..M3, M5..M7:** referee binds inputs by dtype heuristic and can't score conditioned
      graphs; `rename_tflite_tensors` maps outputs by emission order with no shape check
      ("renamed 0 tensors" is success); `kotlin_replica` is unrunnable (depends on
      artifacts nothing produces) and has no conditioning coverage; waveform
      parity is Pearson-only (a systematic fp16 gain error scores 1.0 — no RMSE check
      anywhere, for a model whose flagship axis is energy); CFG guidance is inexportable
      and undocumented for mobile.

## 2 · Before the next training run

- [ ] **Score it on the holdout, not on `loss/val_epoch`.** Standing rule now that
      Phase 0a has landed, listed here because it is the step most easily skipped:
      `scripts/score_holdout.sh` against `data/libritts_r_holdout_devclean`, ~100 min for
      eight checkpoints on an idle card. MLflow val is permanently unusable for cross-run
      comparison (historical split contamination, 93–97% of v3c's val trained on earlier)
      and there are now **two** scale breaks in logged `diff_loss` — 2026-08-01 bucketing
      and 2026-08-06 masking. Deciding a run by its curve is deciding it by an artefact.
- [ ] **Re-derive the corpus at the new width.** `vat_dim` is 8 now, and every existing
      filelist is 3-wide — which the seam guards will refuse loudly at the filelist rather
      than quietly at the trunk, exactly as designed. `derive_vat_corpus.py --delivery-from
      <ratings.csv>` joins the 1,189 delivery labels; without it every clip is `unknown`,
      which is all-zero and reproduces v1 conditioning byte for byte. **The zero-init FiLM
      path means a warm start from `vat3-24k` ep099 is still valid** — the four new
      channels contribute nothing until they are trained — but the trunk's input conv
      changes shape, so the warm start needs the same treatment `make_warmstart.py`
      already applies to a widened tensor. Not yet done.

## 3 · QC / audit / staging

_**Closed 2026-08-07** (`cc4f1a5`, `9f53095`): **C-M6** — all six remaining writers are on
`synth_common.ratings_transaction`, which gained a `dry_run` mode because every one of them
has a report-only path. `pick_audit_subset`'s five-attempt retry loop went with them, on
purpose: with an flock serialising our own scripts the only writer left to race is a human
in a browser, and retrying is a way to eventually win a race we should not be running.
**C-M7** — the cross-title hint was gated on the truthiness of the title entry, but a title
whose ear pass disagreed with ITSELF still produces one (`{attr}_CONFLICT` in place of
`{attr}`), so a guess was written into the one title we have positive evidence is
inconsistent, and then marked machine-written so it looked settled. **C-M1** — deferral only
flipped one way. **C-M9** — both halves. **C-L5**, **C-L6**. Guards in
`tests/test_ratings_transaction.py` and `tests/test_audit_sampling.py`._

- [ ] **C-M4 (half-open, and part of it needs the owner's ear):** the dead-air gate
      shipped and is wired, but `qc_gate`'s speech-duration measure is still
      `librosa.effects.split(top_db=35)` (Silero was written to replace it) and the 4 s
      owner floor has no hard `speech_ok` gate. **head_ok** is also unbuilt — no gate sees
      head truncation, and `tail_lost()` already computes `blocks[0].a` (the head-loss word
      count) and discards it. **Both thresholds need ear calibration like `tail_ok`**,
      which is why this was not closed in the 2026-08-07 sweep: shipping a gate on a
      guessed threshold either passes truncated clips or rejects good ones, and neither
      failure announces itself. Sequenced behind § 6's ears queue.
- [ ] **C-M5:** remaining non-atomic writes — `stage_pool`'s ledger + staging_log,
      `publish_tier` (`.bak` only taken on the first run ever). tmp + `os.replace`
      everywhere. *(`reader_profile`'s ratings write went onto the transaction with C-M6;
      what is left here is its profiles.json.)*
- [ ] **C-M8:** two definitions of "ear-confirmed (reader, title)" — `--seed-ear` skips
      pairs the auditor still force-queues. *(C-M7, filed with it, is closed.)*
- [ ] **C-L4:** loudnorm sidecar keyed on path, so a rerolled-in-place wav ships
      unnormalized. *(C-L5 and C-L6, filed with it, are closed.)*

## 4 · Label derivation

_Mostly closed 2026-08-06 (`tests/test_label_derivation.py`, 12 cases). **D-M1** refuses a
short EIV head set instead of imputing `0.0` (which z-scores ~3σ NONZERO); only WEIGHTED
heads are required, since three carry weight 0.0 and two are legitimately absent. **D-M3**
the corpus lane refuses transcripts containing digits (`--allow-digits` to override) —
verified inert for existing data: 0 digits in 8,000 sampled LibriTTS transcripts and 0 in
all 5,736 dev-clean. **D-M6** error rows no longer count as done, so a transiently failed
clip is retried instead of silently dropped forever. **D-L5** `derive_markup_measures.py`
follows `--corpus` (default v3c, was pinned to v2) and records `speaker_index` separately
from the real LibriTTS `speaker`, verified against the wav paths._

- [ ] **D-L2 — the guard is fixed in code; the LABELS still carry the defect.** This was
      filed as "the two z-implementations disagree", which undersells it: `std or 1.0`
      only guards a std of *exactly* zero, and that is not the case that occurs. A head can
      be **constant** for a speaker — the EIV scorer returns one identical value for all 106
      of speaker `6531`'s clips on `Amusement`, likewise `8095`/Valence and `909`/Valence,
      **224 clips between them** — and then `v - mean` and `std` are both float dust ~1e-21.
      Dividing dust by dust gives **max|z| = 1.0**: a full-scale weighted contribution to
      that speaker's valence, manufactured from rounding error and indistinguishable from a
      measurement. `+ 1e-6` returns ~1e-15, i.e. zero, which is what a constant head
      actually says. `eiv_merge_corpus` is corrected, but **the shipped `_v2` files and
      therefore v3c were built with the broken guard**. Re-deriving moves the raw combo by
      up to **0.228** and changes 31,443 of 31,445 clips (the floor perturbs every std
      slightly), so this is a corpus version bump — **owner call**, not a silent fix.
      Separately: 17 speakers have <10 clips and 5 have ≤2, whose z is fixed by arithmetic
      (±1 exactly, landing on the V rail at ±0.500); 7.25% of train V sits at |V| ≥ 0.99.
      `derive_vat_corpus` reports both now rather than repairing them.
- [ ] **D-M4:** homograph flattening — one pronunciation per key (`read`, `wind`, `live`,
      `dove`, `bass`); past-tense "read" trains against the wrong vowel. **Measured
      2026-08-06: 3.3% of LibriTTS transcripts contain a known homograph** (most common:
      present, close, live, wind, read), so roughly half of those carry a wrong vowel on one
      word. A regression vs the espeak lane, which disambiguated by POS. Needs context-aware
      G2P — deliberately not attempted, since a partial heuristic would silently change
      pronunciations across the whole corpus.

## 5 · Embodiment — approved; the clip-level channel it waits on has SHIPPED

Owner approved the position 2026-08-02; the reasoning is canon in
[ARCHITECTURE.md § 1](ARCHITECTURE.md) and is not repeated here.

- [ ] **Embodiment bank across the current five survivor engines**, folded into a
      narration round rather than run as its own campaign. It settles whether the
      mean of 3.71 is the MODE or was just the old engines — of the four engines in
      those 28 rows (qwen 8, longcat 8, libritts-r 10, scm-spike 2) only qwen is
      still in the portfolio, so we have almost no result rather than a bad one.
- [ ] **Keep embodiment clips delivery-blank** and outside the 50/30/8/6/6
      percentages. This is a standing rule, not a task — listed so a future
      balance pass does not "fix" the blanks. It is now also ENFORCED at the
      encoding: `matcha.delivery` refuses any label outside the closed five, so an
      "Embodiment" lane cannot be added by accident, only by a contract change.
      Blank reaches the model as the all-zero block, which is `unknown`.
- [ ] **Span-FiLM phase:** delivery varying across an utterance on the same zero-init
      path, expanded through the duration alignment. The 17 keeps are the pilot set.
      **Its prerequisite is met** — the clip-level channel shipped 2026-08-07 — so this
      is now sequenced behind evidence (the bank above) rather than behind plumbing.
      A-M2's fix is the other half of the groundwork: per-word CTC spans were garbage
      until 2026-08-07, and span supervision would have inherited them.

## 6 · Ears queue (priority order)

1. [ ] **Qwen vs VibeVoice at equal loudness** — the one teacher-portfolio ranking the
       loudness confound (5.99 dB spread, Qwen +3.81 dB over VV) could have produced.
       The 56 keeps are normalized; a keep-*rate* re-test additionally needs the 24
       quarantined drops normalized (un-quarantine is an owner call).
2. [ ] **Orpheus tier** — `normal` is explicit in `ENGINE_TIER`; `trusted` is arguable on
       105 non-`tara` renders at 80.0% keep / mean 4.49. Policy, not measurement.
3. [ ] **moss_vg** — newscaster 20/20 is confounded (the register flatters its
       radio-timbre/IVR failure mode; its 24% rate was measured on expressive material).
       Re-test on dialogue or an expressive register before any tier movement.
4. [ ] **Ear-calibrate `tail_ok`, `speech_ok` and `head_ok`** (thresholds never
       calibrated) **and the 1.4 s pause advisory band**. § 3's C-M4 is blocked on this.

## 7 · Parked dataset decisions

SSOT is `training-sources.md` — not duplicated here. Headlines: the Expresso two-ruling
conflict (owner call), the other 90% of LibriTTS-R, Hi-Fi TTS parquet→wav conversion,
VCTK unzip, the Emilia-keeps merge (v1.1 lane), HiFiTTS-2's 2.8 TB fetch decision, and
retiring `librivox-v1` once `librivox-v2` is staged and its 12 ear verdicts re-earned.

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

_**2026-08-07, third pass** (`33d2a4f`): § 3's **D-M4** shipped its resolver and its
measurement and is now an owner decision, not engineering — and the measurement corrected
the finding's own premise, which had the damage spread across 3.3% of transcripts when it
is concentrated in a handful of words (`live` alone is 87 of 281). Net **11 open items ->
12**: measuring it surfaced a live one, that the device G2P still carries D-C1, filed in
§ 1._

_**2026-08-07, fourth pass**: § 1's device-front-end item closed the day after it was
filed (see § 1's note). Net **12 open items -> 11**. It also left one thing for the D-M4
decision to carry: gate G7 **refuses a homograph-enabled export**, because resolution needs
context evidence rather than a table and there is nothing to ship to a device. So turning
D-M4 on adds "port `matcha/text/homographs.py`" to the mobile lane — not a reason to decide
either way, but a cost that was invisible before._

_**2026-08-08, fifth pass — the ears queue was AUDITED rather than worked, and three of
its four items turned out to rest on numbers that do not survive re-measurement.** § 3 is
empty and deleted (D-L2 and D-M4 both landed in v4); § 1's re-derivation is done and what
remains there is the launch. §4 lost three checkboxes that were never tasks — a standing
rule already enforced in code, and two notes describing a spike that is gated shut.
**Net 11 open items -> 10.** The three re-measurements, each of which changes what the
owner should decide:_
- _**Orpheus** reads 90.8% ex-`tara` (filed: 80.0%) — but `jess` is 72% of that population
  at 97.4%, and without it the rest is 74.2% with `mia` at 55.6%. **Hold at `normal`.**_
- _**moss_vg**'s "24% on expressive material" is not reproducible; the source measured 35%
  rejected on a NARRATION campaign. The dialogue re-test it waits for already exists and
  passes at 89.5%. **Promote to `normal`.**_
- _**Qwen vs VibeVoice** — the 56 keeps already measure 0.01 dB apart, so the score re-test
  is unblocked and purely an ear job; only the keep-RATE half is still owner-blocked._
- _And the `uneasy-money` head-loss cluster was a size effect, not an alignment defect —
  but chasing it found a real one: `edge_loss` STRIPPED hyphens instead of splitting on
  them (`b6d29b2`)._

_Earlier pruning notes for the 2026-08-06 round are in git history._

This file is **residue, not the plan.** What to do next, and in what order, is
[quality-gap-plan.md](quality-gap-plan.md).

---

## 1 · Before the next training run

_**The export lane closed 2026-08-07** (`905e91b`..`e18877a`). It had been "the blocker it
was always going to be" for one day. What landed: **F-H2** — `config.json` carries a
machine-readable `control` block (continuous range + reject-don't-clamp, the one-hot
delivery vocabulary with its explicit unknown vector, and CFG's method and >=25-step
floor), so a host can validate BEFORE it renders instead of after someone listens; three
of five demonstrated bad vectors were previously accepted silently. **F-M1** — the referee
bound inputs by a dtype heuristic that fed the TOKEN COUNT into `spks` (same dtype, same
shape, no error), so it could not score the conditioned lane and quietly compared the
wrong render when it tried; bound by name now. **F-M5** — cosine and Pearson are
scale-invariant, so `0.5 * reference` scored **1.000000**; gain and normalised-RMSE gates
added to both the referee and the converter (G3b/G3c). **F-M2** — outputs were renamed by
emission order, so a swapped pair would have labelled the LENGTHS tensor `wav`; checked
against shape and dtype, and `renamed 0 tensors` is a failure rather than a success.
**F-M3/M6** — `kotlin_replica` could not run (three of its seven prerequisites are vendored
assets it never looked for) and validated the unconditioned Phase 0 graphs; it runs on
both lanes now and drives spk/vat/delivery through the same manifest the host reads.
**F-M7** folded into F-H2. New gate **G6**: the delivery lanes must be mutually
distinguishable, since five inputs on one summing junction pass the per-channel probe five
times._

_**The device front end closed 2026-08-07**, the day after it was filed. It was **D-C1
still live on the device side** — `kotlin_replica.phonemize` was a flat dictionary lookup
with no contraction table, five days after the host got one. Half the finding is answered
rather than fixed: **there is no shipped Kotlin app**, and mobile front ends have not
started in earnest, so the replica was not validating a front end nobody runs — it IS the
front end, as spec, and this was prophylactic. What landed: the text front end moved to
`scripts/litert_export/device_g2p.py`, the apostrophe tables ship as
`g2p_contractions.json` **exported from `matcha.text.op_g2p`** rather than transcribed
(D-C1 is a defect of omission; a hand-synced table is the same defect on a delay), a
device that cannot find the asset **refuses** instead of falling back, and new gate **G7**
holds the two front ends to identical phoneme strings over 86 probe sentences. The old
front end fails 69 of those 86, and worse than filed: `they've` → `θˈeɪv`, `she'd` →
`ʃˈɛd` (the word *shed*), `james's` → `dʒˈːmˈɛs`. Guards in
`tests/test_device_g2p_parity.py`._


- [ ] **Score it on the holdout, not on `loss/val_epoch`.** Standing rule now that
      Phase 0a has landed, listed here because it is the step most easily skipped:
      `scripts/score_holdout.sh` against `data/libritts_r_holdout_devclean`, ~100 min for
      eight checkpoints on an idle card. MLflow val is permanently unusable for cross-run
      comparison (historical split contamination, 93–97% of v3c's val trained on earlier)
      and there are now **two** scale breaks in logged `diff_loss` — 2026-08-01 bucketing
      and 2026-08-06 masking. Deciding a run by its curve is deciding it by an artefact.
_**Both smoke runs passed 2026-08-08** (`3b6b738`, `76b6aea`), and the owner's call was to
smoke rather than run v4 to 100 epochs — v4 is the SAME 31,445 clips `vat3c_finetune`
already spent 100 epochs on and came back a measured regression, so a full run would
re-buy a conclusion we own. **(1)** 8-wide on v4: 25/25 batches at 0.93 it/s, validation
ran, all four loss terms logged, restored cleanly from `vat4_init.ckpt`. **(2)** merged
stub (v4 + 200 real Emilia rows, `n_spks` 247 → 280): speaker map **PREFIX OK**, `spk_emb`
widened 247 → 280, **338 warm / 0 fresh**, trains to val_epoch 1.572. Seam guards **28/28**
in-container. Two latent bugs fell out, which is what smoke runs are for: `test: True` (no
model here defines `test_step`, unreachable only because `max_epochs: -1` means `fit()`
never returns — the first run to finish normally would crash after writing its checkpoints
and then crash-loop under `restart: unless-stopped`), and `make_warmstart` composing the
model from the experiment config alone, so a run with data overrides got a warm start built
for a different model._

- [ ] **Phase 1 #1 — the Emilia merge.** The scale question is settled (owner took the
      global anchor, `scripts/anchor_emilia_labels.py`, all 13,141 keeps label and none
      collapse to zero) and the machinery is proven end-to-end on a 200-row stub. What is
      left is the real derivation: all 13,141 rows, `n_spks` 247 → 2,655, phonemes on the
      fixed `op_g2p`, **`data_statistics` re-measured in-container** (they cannot be
      inherited here — v4 could only inherit v3c's because the audio and split were
      identical), and an explicit compose bind.
      ⚠ **T saturates at 54%** and the owner accepted it for this run — the prediction that
      makes that a test rather than a story is written down in
      [quality-gap-plan.md](quality-gap-plan.md), and it must be read BEFORE the holdout
      result, not after.

## 2 · QC / audit / staging

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

_**Closed 2026-08-07, second pass** (`1ebd14a`, `0a505d3`, `a4b6ec5`): **C-M5** — the last
three read-modify-writes across a gap (`staging_log.json` was atomic but STALE, which loses
a concurrent run's entry just as thoroughly; `reader_profiles.json`; `metadata.jsonl`), and
the two atomic writers that leaked their tmp on a failed rename. **C-M8** — one predicate,
in `reader_profile`. **C-L4** — the sidecar keys on content now, and the sweep that proved
the point found **15 clips across two live campaigns already shipped unnormalized**, up to
7 dB off, r2's ten spanning 12.7 dB inside one bank. Not repaired: re-gaining clips the ear
has already rated is a corpus change and the owner's call. Guards in
`tests/test_atomic_state_writes.py`, `test_ear_confirmation.py`, `test_loudness_identity.py`._

_**C-M4's first threshold closed 2026-08-07** (`97c14b4`): **`speech_ok` is a HARD GATE**
at the owner's 4 s, on the VAD figure. `librivox_align` keeps its energy gate as an ingest
pre-filter and the 0.7% disagreement is written down rather than assumed. ~4% of clips that
passed QC the day before do not now. Queueing the head clips then found the reason they
were never heard: **`qc_flagged` tested `all(gates.values())`, so an ADVISORY could not
queue a clip** — and `head_ok` is an advisory precisely because it has no threshold, which
it cannot get until the ear describes the failure once. The one finding that most needed an
auditor was the one that could not reach one. Fixed: `synth_common.head_flagged` (recomputes
from `text`/`asr_hyp`, since every `qc_measures.jsonl` on disk predates the measure),
honoured by `stage_pool.qc_flagged`, with `queue_head_audit.py` for the backlog. **4 of the
8 are queued; the other 4 are `uneasy-money` clips still in the POOL** — never staged, so
there was nothing to re-audition, and they are exactly the silent fold `qc_flagged` now
prevents. No verdict was discarded: status moves, score and the auditor's note stay._

- [ ] **C-M4 — the measurement shipped (`a4b6ec5`); one threshold is still yours.**
      `head_lost_frac`/`head_words_lost` is recorded beside the existing measures, in an
      `advisories` dict that gates NOTHING, and `gate_calibration.py --sweep head_lost_frac`
      reproduces the procedure that set `PAUSE_HARD_MAX`.
      **`head_ok` needs the ear, and the queue now exists** — that was the missing half.
      There is still nothing to calibrate against: no drop note in `ratings.csv` names a
      late start, because no auditor has been asked to listen for one.
      **4 clips are in the todo queue now** with a note saying the threshold comes from
      what they write — `wuthering-heights_nar_0036_neu_CHA` (4 of 19 words, 21% of the
      passage, WER 0.211 against a 0.35 gate), `_0040` (5 words), `castle-of-otranto_nar_
      0033_neu_ZON` (4 words, previously kept at 5) and `victory_whimsical_00_brightF_s5150`
      (3 words, previously kept). **This is the whole of what is left: hear them, then run
      `gate_calibration.py --sweep head_lost_frac`.** Blocked on ears, nothing else.
      **The `uneasy-money` cluster was investigated 2026-08-08 and is NOT an alignment
      defect — the flag was wrong in the direction it was stated.** That title holds 2,030
      of the 3,189 measured clips (64%), so 4 of 8 is a size effect; its head-loss rate is
      **0.20%, the LOWEST of any title with occurrences** and a third of the corpus-wide
      0.60%. What the investigation actually found is more useful:
      - **Head loss is ~5× a SYNTHESIS problem, not an alignment one**: 1.30% of rendered
        clips (13/1,001) against 0.27% of force-aligned real audio (6/2,188). Engines
        ramping in dominates; the aligner cutting late barely registers.
      - **All four real-audio cases are FALSE POSITIVES** — the words are spoken and ASR
        misheard the opening ("Two cars farther" → "To Carr's father", "He groped round" →
        "The grope ground"). This is exactly the asymmetry `edge_loss`'s own docstring
        warns about: no left context at the first word.
      - **One was a pure hyphenation artifact and is now fixed.** `edge_loss` STRIPPED
        hyphens instead of splitting on them, so `By-and-by` scored one token against a
        three-token `By and by` and reported a 3-word late start on a perfect opening. The
        calibration population is **8 → 7**. Measured before changing it: 8 head counts and
        46 tail counts move corpus-wide, and **0 clips had ever failed `tail_ok` because of
        it**, so no shipped corpus is affected.
      Net: the 4 queued clips are all SYNTHETIC, which is the right population to calibrate
      from, and the 3 remaining pooled real-audio cases are ASR noise rather than evidence.

## 4 · Conditioning axes — one approved, one OPEN

Owner approved the position 2026-08-02; the reasoning is canon in
[ARCHITECTURE.md § 1](ARCHITECTURE.md) and is not repeated here.

- [ ] **Embodiment bank across the current five survivor engines**, folded into a
      narration round rather than run as its own campaign. It settles whether the
      mean of 3.71 is the MODE or was just the old engines — of the four engines in
      those 28 rows (qwen 8, longcat 8, libritts-r 10, scm-spike 2) only qwen is
      still in the portfolio, so we have almost no result rather than a bad one.
_**Standing rule, not an open item** (de-checkboxed 2026-08-08): embodiment clips stay
delivery-blank and outside the 50/30/8/6/6 percentages, so a future balance pass does not
"fix" the blanks. It is ENFORCED at the encoding — `matcha.delivery` refuses any label
outside the closed five, so an "Embodiment" lane cannot arrive by accident, only by a
contract change. Blank reaches the model as the all-zero block, which is `unknown`._
- [ ] **Span-FiLM phase:** delivery varying across an utterance on the same zero-init
      path, expanded through the duration alignment. The 17 keeps are the pilot set.
      **Its prerequisite is met** — the clip-level channel shipped 2026-08-07 — so this
      is now sequenced behind evidence (the bank above) rather than behind plumbing.
      A-M2's fix is the other half of the groundwork: per-word CTC spans were garbage
      until 2026-08-07, and span supervision would have inherited them.

### The categorical emotion block — OPEN DECISION, not scheduled (raised 2026-08-07)

Owner question while the delivery width landed: *isn't 3 + 8 the standard?* It is — the
dimensional-plus-categorical hybrid (3 continuous VAD + 8 Plutchik primaries, or 6 Ekman)
is a common shape in emotional TTS. We do not have one, and whether we should is genuinely
undecided. Recorded here so the answer is a decision rather than an omission.

**Why we do not have one today.** Affect is carried entirely by the three continuous
channels. The categorical vocabulary that exists — the 47-label register lexicon, 554
certified keeps, 138 distinct labels seen in the wild — is **Director-side by contract**
(v2, ARCHITECTURE §1: "the Actor never sees a register id") and compiles down to V/A/T +
delivery + text before the model sees anything. Delivery's five lanes are *modes of
address*, not emotions, and are orthogonal to affect.

**The case for it, and it is not weak.** That compile-down is lossy in a specific,
documented way. `vat-channels.md` § Options A: *"Valence is the one dimension
acoustics-only genuinely can't reach — a sob and a laugh have similar energy."* Categorical
affects can occupy nearly the same point in a low-dimensional continuous space; anger and
fear sit close in V/A, and the discrete label is the only thing that separates them. And
valence is exactly the channel that **FAILED its standing test** on vat3-24k (energy PASS /
tension near-pass / valence FAIL).

**Why it is NOT scheduled, and what would change that.** That failure is currently
diagnosed as *a corpus-label limit, not architectural* — which is what opened the directed
teacher-synthesis lane in the first place. Adding channels would be an architectural answer
to a problem we believe is a data problem, and we are about to run the experiment that
tells us which: **Phase 1 is the data lever.** So:

- [ ] **Gate on Phase 1.** If valence is still failing after the corpus grows, the
      representation is the suspect and this becomes a real spike. If volume moves it, the
      question is answered and this stays closed. Do not spike it before that read — a
      categorical block trained on a corpus that cannot support the continuous channel
      would confound the two.
Two notes that are NOT open items — they are what the spike would look like if the gate
above ever opens, recorded so the answer is a decision rather than an improvisation:

- **If it is spiked, the shape is settled in advance** and cheap: it APPENDS as
      channels 8+ on the same zero-init FiLM path, so `unknown` stays all-zero and every
      existing checkpoint and filelist keeps its meaning. Reordering is the one edit that
      must never happen — position is the wire format now. Vocabulary and width are a
      **contract change requiring an owner call** (ARCHITECTURE §1); `matcha/delivery.py`
      is where it would live and refuses anything outside the closed set, so it cannot
      arrive by accident.
- **The labels do not exist yet in that form.** Delivery shipped with 1,189 labelled
      keeps; an 8-way emotion block starts at zero. The register lexicon is the obvious
      source (a 47 → 8 fold), but it is Director-authored intent rather than an ear
      verdict on the render, and the instrument-vs-intent distinction is the one this
      corpus keeps having to relearn. Costing that fold is part of the spike, not a
      preliminary.

## 5 · Ears queue (priority order)

1. [ ] **Qwen vs VibeVoice at equal loudness** — the one teacher-portfolio ranking the
       loudness confound (5.99 dB spread, Qwen +3.81 dB over VV) could have produced.
       **Verified 2026-08-08 by measuring the files, and the two halves are in very
       different states:**
       - **The SCORE re-test is UNBLOCKED — it is purely an ear job now.** The 56 keeps
         measure **qwen −22.99 LUFS / vibevoice −23.00**, a **0.01 dB** difference (qwen
         sd 0.04, vibevoice sd 0.00) against the filed +3.81 dB. The audio is already
         equal; what is not is the VERDICTS, because qwen 4.94 vs vibevoice 4.20 was
         scored on the original unequal renders. Re-hearing the 33 qwen+VV keeps at the
         loudness they now sit at is the whole test.
       - **The keep-RATE re-test is genuinely blocked, and on two things.** The 24
         quarantined drops are **not** normalized — measured in `_dropped/` at 10.5 dB and
         29.1 dB spreads (one clip at −42.2 LUFS). So it needs (a) those 24 re-gained,
         which is **re-gaining clips the ear has already rated** and therefore the owner's
         call by the same rule that left the [[loudnorm-reroll-defect]] clips alone, and
         (b) un-quarantining so they can be re-heard. Cost if both are taken: **80 clips**
         (56 keeps + 24 drops) — `dia` 15, `vibevoice` 5, `qwen` 2, `moss_vg` 2.
       Doing only the first is cheap and answers the ranking question that was actually
       filed; the keep-rate question can stay open without blocking it.
2. [ ] **Orpheus tier — RE-MEASURED 2026-08-08, and the case for `trusted` does not
       survive it.** The numbers here were stale ("105 non-`tara` renders at 80.0% keep /
       mean 4.49"); the current join of `ratings.csv` to the orpheus manifests says
       **109 heard ex-`tara`, 90.8% keep, mean 4.57** — which looks *better* than filed and
       above qwen. It is still the wrong number to decide on:

       | voice | heard | keeps | rate |
       |---|---|---|---|
       | `jess` | 78 | 76 | **97.4%** |
       | `dan` | 9 | 8 | 88.9% |
       | `mia` | 9 | 5 | **55.6%** |
       | `leah` | 8 | 5 | **62.5%** |
       | `zoe` | 5 | 5 | 100% |
       | `tara` | 21 | 0 | **0.0%** |

       **`jess` is 78 of the 109 (72%).** Strip it and the rest is 31 heard / 23 keeps =
       **74.2%**. So "orpheus at 90.8%" is very nearly "jess at 97.4%", and the tier is a
       tag on an ENGINE, not on a voice — `trusted` (1 per group + 3%) would fold `mia`
       and `leah` clips unheard at a ~4-in-10 rejection rate.
       This is the third instance of one shape: zonos's 31% was two populations averaged,
       moss_vg's 20/20 was a register that flatters it, and `ref_risk.py` exists **because
       of this exact case** — "Orpheus reads 36% rejected, but that is `tara` at 90%
       dragging up a `jess` that is 0 for 24."
       Against the bar: chatterbox is `trusted` at **77.8%**, qwen at **89.4%**. Orpheus as
       shipped, all voices, is **76.2% — below chatterbox.**
       **Recommendation: HOLD at `normal`.** The audit saving is real but it would be
       bought by not listening to the two voices that most need listening to. If the saving
       is wanted, the honest form is a restricted roster (`jess`/`dan`/`zoe`), and that is
       a code change — `ENGINE_TIER` is keyed on engine name and has no per-voice concept.
3. [ ] **moss_vg — the re-test this asks for ALREADY EXISTS, it passes, and the defect it
       guards is now INSTRUMENTED.** Checked 2026-08-08.

       ⚠ **Correction to the rates first published in this entry.** 11 of moss_vg's 20
       non-keeps are **bookkeeping retirements, not ear rejections** — 6 "retired unheard:
       group failed certification twice" and 5 "superseded by `_QWE`/`_CHA` (kept, score
       5)". Counting them as rejections is the exact error the zonos promotion note warns
       about. **On actual ear verdicts moss_vg is 95 heard / 90.5% keep**, not 81.1%, and
       **Documentary is 16 heard / 75.0%**, not 60.0%. By lane: Neutral 94.7% (38),
       Newscaster 95.2% (21), Dialogue 89.5% (19), Documentary 75.0% (16).
       `ENGINE_MIX_BY_LANE["Documentary"]`'s "56% (18)" comment is contaminated the same
       way and should be re-derived when that table is next touched.

       **The 9 real ear rejections split into two classes, and today both are caught:**
       - **Structural (5) — every one fails a gate that exists now.** Re-measured from the
         audio, because these rows predate the measures: `the-return_nar_0049` 2.82 s and
         `old-english-baron_nar_0045` 3.04 s internal pause (cap 2.5 → `pause_ok`);
         `tab_15_mystic` 0.67 s and `tab_18_accent_scots` 3.68 s of speech (floor 4.0 →
         `speech_ok`, hard since 2026-08-07); `nar_0050` 9 tail words (`tail_ok`).
       - **Timbre (4) — `radio_score` catches them.** Three "mid-1900s radio transmission"
         clips measure out/in-band 0.014, 0.072, 0.094 against a 0.10 threshold: **3 of 3
         recall.** Over all 116 moss_vg clips on disk it flags **23 (19.8%)** for the ear.
         The fourth is `windfairies_nar_0034`, "highly robotic… like an old-fashioned phone
         IVR" — 28.8 s, 26.2 s speech, 0.38 s worst pause, and it passes everything. That
         one is genuinely ear-only.

       **This is the bargain `qc_engine_defects.py` was written to offer**, in its own
       words: *"what the tier system needs in order to ever hand those engines a ride-along
       back — coverage traded for instrumentation, the same bargain qwen made."* **zonos
       took that bargain and was promoted 2026-08-04. moss_vg has the same instrument and
       was not.** Scrutinized costs 100% listening; instrumented `normal` costs ~20% (the
       radio flags) plus normal sampling — a 5× reduction for full recall on the
       characterized defect.

       ⚠ **The "24% rate on expressive material" is not reproducible and appears to be a
       corrupted restatement.** The source is `4abfd3f` (2026-07-31), which measured
       **`moss_vg 19/54 (35%)` REJECTED on `delivery-v1-narration`** — a narration campaign,
       not expressive material, and 35% not 24%. Today that same campaign reads 54 heard /
       66.7% keep, i.e. the same number. Nothing on disk produces 24%. A tier that costs
       **100% listening** should not rest on a figure that lost its provenance; the same
       line is in [STATE.md](STATE.md) and should be corrected there too.

       **Documentary is still the weak lane**, at 75.0% against an 83.3% lane mean, and
       `ref_select.ENGINE_MIX_BY_LANE["Documentary"]` already holds moss_vg to a **0.05
       floor share** for that reason. So the scrutinized tier is doing duplicate work: it
       buys 100% listening across every lane — including the two where moss_vg is the
       strongest engine we have — to guard a lane risk the allocation layer already prices
       and a timbre defect the detector already flags. That is precisely the split
       [[audit-trust-tiers]] insists on ("a tier is a TAG ON AN ENGINE; render share lives
       in `ENGINE_MIX_BY_LANE`").

       **Recommendation: PROMOTE moss_vg scrutinized → normal**, with
       `qc_engine_defects.py --campaign <c> --append-flags` run on every moss_vg bank so
       the radio flags keep reaching the ear. Owner call, policy not measurement — but the
       measurement no longer supports the hold, and the instrument the hold was waiting for
       has existed since 2026-08-02.
4. [ ] **The 4 head-truncation clips, already in the todo queue** (`97c14b4`) — the only
       thing `head_ok` is waiting on. Each carries a note saying the threshold comes from
       what the auditor writes. Then `gate_calibration.py --sweep head_lost_frac`.
       `speech_ok` left this list on 2026-08-07: it is a hard gate now, by owner's call.
       Still open alongside: ear-calibrating `tail_ok` and the 1.4 s pause advisory band.

## 6 · Parked dataset decisions

SSOT is `training-sources.md` — not duplicated here. Headlines: the Expresso two-ruling
conflict (owner call), the other 90% of LibriTTS-R, Hi-Fi TTS parquet→wav conversion,
VCTK unzip, the Emilia-keeps merge (v1.1 lane), HiFiTTS-2's 2.8 TB fetch decision, and
retiring `librivox-v1` once `librivox-v2` is staged and its 12 ear verdicts re-earned.

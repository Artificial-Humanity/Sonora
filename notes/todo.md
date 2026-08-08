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
**Net 11 open items -> 7** once the owner's verdicts on the four
audition clips landed and the two tier calls closed. The three re-measurements, each of
which changed what the owner decided:_
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

_**C-M4 CLOSED 2026-08-08** (`683c43f`). The owner heard the four queued clips and the
answer is **`head_ok` gets NO GATE — on evidence, not for want of it.** Ranked by
`head_lost_frac`: **0.214** DROP (shrill, nothing to do with the head) · **0.211** KEEP 5
(*"I hear all of the words in the printed text"*) · **0.132** KEEP 5 · **0.129** DROP
(*"the first two words are jumbled together… 'gojawayisabella'"*). The measure is
**anti-correlated with the ear** — the highest score is a perfect keep, the one real head
defect sits below two 5s, and any cut catching 0.129 catches both. Across all 8 flagged
clips, including the four real-audio ones, **not one was a late start**:
`head_lost_frac` measures ASR head DISAGREEMENT, not truncation. It stays advisory,
because slurred onsets are real, and the note was reworded — it told auditors "first N
words never spoken", which was false in 3 of the 4 cases they checked._

_Acting on it found the reason the sweep had been useless: **`gate_calibration.parse_score`
searched for `[1-5]`, so the reject marker `x` returned `None` and every caller skipped the
row.** 253 of 286 dropped rows carry `x` — **88% of every ear rejection was invisible to
the tool that calibrates gates against ear verdicts**, and a campaign full of drops swept
as "no rated clip matches the defect". Fixed. Two consequences: **`TAIL_LOST_MAX` 0.05 is
CONFIRMED** (3/3 catch, 1% false flag, on rejections that could not be seen before), and
one thing for the owner below._

_**`PAUSE_HARD_MAX` re-swept 2026-08-08 (`35bbce6`) — it stays at 2.5, against expectation.**
`--campaign-dir` is repeatable now (sweeping one campaign at a time is what made every
defect set thin), so this pooled 161 keeps across the three campaigns holding both the
measure and ear verdicts. The drops turn out to be **interleaved with keeps** — 2.112 keep,
1.984 keep, **1.600 DROP**, 1.568 keep, 1.536 keep, **1.408 DROP** — and the longest pause
in the set is a keep. No threshold separates them: catching the 1.600 drop hard-rejects two
keeps, sparing the 2.112 keep misses both drops. **Duration is the wrong axis.** Both drop
notes object to WHERE the pause falls ("after 'complete'", "between 'report' and 'the
driest'") — mid-phrase, where no reader breathes. The advisory band at 1.4 already catches
both at a 3% false-flag rate, which is what an advisory is for._

- [ ] **A pause-POSITION measure.** The one thing the re-sweep says is missing: `worst_pause`
      knows how long a silence is and nothing about whether it falls where a reader would
      breathe. A 2.1 s pause at a paragraph break is fine; a 1.4 s one mid-clause is not.
      This is a new measure, not a new constant — the phrase boundary is already available
      from the CTC alignment A-M2 fixed. Not urgent: the advisory band catches these today
      at the cost of an audition. ⚠ n = 2, and both from one voice in one campaign.

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

_**1 · Qwen vs VibeVoice at equal loudness — CLOSED 2026-08-08. The confound was real and
it ran BACKWARDS.** 33 keeps re-listened at 0.01 dB apart (they had been 3.81 dB apart when
first scored). Paired, restricted to clips still kept in both states:_

   | engine | n | before | after | Δ |
   |---|---|---|---|---|
   | qwen | 17 | 4.94 | **4.88** | −0.06 |
   | vibevoice | 15 | 4.20 | **3.33** | **−0.87** |
   | **gap** | | **+0.74** | **+1.55** | |

_The hypothesis was that Qwen's +3.81 dB flattered it, so normalising should have SHRUNK
the gap. It more than doubled. Bringing VibeVoice UP in level made its defects audible:
**loudness was masking VibeVoice, not manufacturing Qwen.** Qwen moving −0.06 is the
control that makes the attribution safe — it says re-listening on its own costs almost
nothing, so VibeVoice's −0.87 is the level change rather than fresher ears.
The teacher-portfolio ranking is confirmed and was UNDERSTATED, and this independently
supports the standing decision to set VibeVoice aside ([[narration-only-engines]])._

_What the re-listen exposed, in the owner's words: **"Vocal effect sounding like an older
television broadcast… it evokes a picture of black and white tv"** (two clips, both 5 → 1),
"Reverb effect to the point of sounding boxy", "Echo effect, like in a tiny space". That is
the ROOM-reverb family, not bandwidth: **`radio_score` does NOT catch it** — the worst
offender measures 0.3299 against a 0.10 bar and only 1 of 20 VV clips flags. Consistent
with [[reverb-artifact-diagnosis]], where reverb defeated four detector attempts and stayed
ear-only. One qwen reroll (`tab_05_whimsy_QWN`, a pitch note, otherwise positive)._

_⚠ Method note for the next re-test: the prior score was parked in the `note` column and the
audition app OVERWRITES that column when the owner types. 17 of 33 lost it. The comparison
survived only because `ratings_transaction` had taken `ratings.csv.bak-20260808-ab-loudness`.
Park before/after state in a sidecar, not in a field the app owns._

_**2 · Orpheus — CLOSED 2026-08-08: held at `normal`** (owner). The filed 80.0% was stale
   in the favourable direction — it is 90.8% ex-`tara` — but `jess` is 78 of those 109
   clips at 97.4%, and the remainder is 74.2% with `mia` at 55.6% and `leah` at 62.5%.
   All-voices it is 76.2%, BELOW `chatterbox` at 77.8%, which is already trusted. A tier
   tags an ENGINE, so `trusted` would fold the two voices that most need hearing. If the
   audit saving is ever wanted the honest form is a restricted roster (`jess`/`dan`/`zoe`),
   and that is a code change — `ENGINE_TIER` has no per-voice concept._

_**3 · moss_vg — CLOSED 2026-08-08: promoted to `normal`** (`4150dfa`), on the bargain
   `qc_engine_defects.py` was written to offer. Neither leg of the hold survived: the "24%
   on expressive material" is not reproducible (the source measured **35% rejected on a
   NARRATION campaign**), and 11 of its 20 non-keeps are bookkeeping retirements, so on
   real ear verdicts it is **95 heard / 90.5%**. The 5 structural rejections all fail
   `pause_ok`/`tail_ok`/`speech_ok` today; 3 of the 4 timbre ones are caught by
   `radio_timbre` at 19.8% batch flag. **The detector is now wired into `synth_bank.sh`** —
   it was a manual step, so the trade had been notional. Residual named:
   `windfairies_nar_0034`, IVR-flat prosody, passes every instrument._

4. ~~**The 4 head-truncation clips**~~ **HEARD 2026-08-08 — see § 2.** `head_ok` gets no
       gate (the measure is anti-correlated with the ear), `TAIL_LOST_MAX` 0.05 is
       confirmed at 3/3 catch / 1% false flag, and the pause cap is now the open question
       rather than the head. `speech_ok` left this list on 2026-08-07.

## 6 · The score scale has a ceiling, and it is pinned at human audio

**Owner, 2026-08-08, after the equal-loudness listen:** *"Qwen relays human-like prosody in
those cases where it scored a 5 that makes me rethink 5's given out to others. Cases like
'It's fine, it really fine…' are simply magnificent. Vibe Voice paled in comparison."*

That is a finding about the INSTRUMENT, and it measures. The exemplar line — *"No, it's
fine. It's completely fine. I only rearranged my entire year around it, so it's fine."* — is
rendered by seven engines, and **six of them hold a 5**: chatterbox, dia, moss_vg, orpheus,
qwen, zonos. Only VibeVoice was dropped. Across the corpus:

- **62 texts are rendered by ≥3 engines with ear verdicts** (478 clips) — controlled
  comparisons, identical text.
- **46 of those 62 groups have ≥3 DIFFERENT engines all scoring 5.** 229 clips tied at the
  ceiling. On three-quarters of the controlled comparisons the scale cannot separate them.
- **`librivox` — real human recordings — means 5.00, with 100% of its keeps at 5.** The top
  of the scale is "indistinguishable from a human read", and six synthetic engines are
  sitting on it.

**What this invalidates, specifically.** Any RANKING BY MEAN SCORE across engines. The
engines cluster in a 0.14-point band (chatterbox 4.72 · qwen 4.67 · zonos 4.65 · moss_vg
4.58) and **the scale places chatterbox above the engine the owner calls the gold
standard** — an inversion produced by compression, not by quality. Mean-score tables in
this file and in the changelog should be read as "all of these clear the bar", never as an
order. *(The tier calls of 2026-08-08 rest on keep RATE and defect characterisation, which
are binary and survive this; it is the means quoted alongside them that do not.)*

**And it is a corpus problem, not only a bookkeeping one.** Sonora trains on keeps. If
"magnificent" and "acceptable" both label as 5, the corpus's quality ceiling is set by the
weakest 5 in it — which is exactly the "clinging to a teacher's style" risk the portfolio
breadth exists to prevent, arriving through the labels instead of the engine mix.

- [ ] **A forced-ranking pass over the 46 tied groups** (229 clips). Same text, ≥3 engines,
      all at 5 — rank within the group instead of scoring absolutely. It is the protocol
      the owner already used informally ("directly compared"), it needs no scale change and
      invalidates no existing verdict, and it yields the comparative signal the absolute
      scale structurally cannot. Bounded: 46 sittings of 3–5 clips.
- [ ] **Anchor exemplars in the audition app.** The owner has named one; a "this is what a 5
      sounds like" reference beside the rating control is standard MOS practice and costs a
      link. Without it, 5 drifts per session and per engine, which is what happened here.

## 6 · Parked dataset decisions

SSOT is `training-sources.md` — not duplicated here. Headlines: the Expresso two-ruling
conflict (owner call), the other 90% of LibriTTS-R, Hi-Fi TTS parquet→wav conversion,
VCTK unzip, the Emilia-keeps merge (v1.1 lane), HiFiTTS-2's 2.8 TB fetch decision, and
retiring `librivox-v1` once `librivox-v2` is staged and its 12 ear verdicts re-earned.

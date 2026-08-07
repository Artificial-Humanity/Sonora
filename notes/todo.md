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
decision to carry, recorded in § 3: gate G7 **refuses a homograph-enabled export**, because
resolution needs context evidence rather than a table and there is nothing to ship to a
device. So turning D-M4 on adds "port `matcha/text/homographs.py`" to the mobile lane —
not a reason to decide either way, but a cost that was invisible before._

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
- [ ] **Re-derive the corpus at the new width.** `vat_dim` is 8 now, and every existing
      filelist is 3-wide — which the seam guards will refuse loudly at the filelist rather
      than quietly at the trunk, exactly as designed. `derive_vat_corpus.py --delivery-from
      <ratings.csv>` joins the 1,189 delivery labels; without it every clip is `unknown`,
      which is all-zero and reproduces v1 conditioning byte for byte. **The zero-init FiLM
      path means a warm start from `vat3-24k` ep099 is still valid** — the five new
      channels contribute nothing until they are trained — but the trunk's input conv
      changes shape, so the warm start needs the same treatment `make_warmstart.py`
      already applies to a widened tensor. Not yet done.
      **Three findings ride this one pass, and each would be a corpus version bump on its
      own**: the width itself, **D-L2**'s corrected z guard (§ 3 — the code is fixed, the
      shipped labels are not), and **D-M4**'s homograph resolution (§ 3). Doing them
      separately costs three bumps and three lineages for one re-derivation's worth of work.
      **DECIDED 2026-08-07 — the owner took all three.** So the next derivation is: 8-wide,
      `--delivery-from <ratings.csv>`, the corrected z guard, and `op_g2p(homographs=True)`.
      That is a new corpus version, not an amendment of v3c, and the checkpoint lineage
      restarts from `vat3-24k` ep099 via `make_warmstart.py` (zero-init FiLM keeps the warm
      start valid; the trunk's input conv changes shape and needs the widened-tensor
      treatment). **Not yet run** — this is the next command, and it is the gate before any
      training.

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
      ⚠ One pattern the finding did not name: **4 of the original 8 are `uneasy-money`**,
      across both librivox v1 and v2. One title supplying half the population is worth a
      look at that alignment before the threshold is chosen from it.

## 3 · Label derivation

_Mostly closed 2026-08-06 (`tests/test_label_derivation.py`, 12 cases). **D-M1** refuses a
short EIV head set instead of imputing `0.0` (which z-scores ~3σ NONZERO); only WEIGHTED
heads are required, since three carry weight 0.0 and two are legitimately absent. **D-M3**
the corpus lane refuses transcripts containing digits (`--allow-digits` to override) —
verified inert for existing data: 0 digits in 8,000 sampled LibriTTS transcripts and 0 in
all 5,736 dev-clean. **D-M6** error rows no longer count as done, so a transiently failed
clip is retried instead of silently dropped forever. **D-L5** `derive_markup_measures.py`
follows `--corpus` (default v3c, was pinned to v2) and records `speaker_index` separately
from the real LibriTTS `speaker`, verified against the wav paths._

- [x] **D-L2 — DECIDED 2026-08-07: the next derivation takes the corrected guard.**
      Owner's call. Nothing more to settle; it executes as part of § 1's single pass.
      The finding, kept because the re-derivation has not run yet:

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
- [ ] **D-M4 — the resolver shipped (`33d2a4f`) and is OFF; turning it on is your call.**
      Context-aware G2P exists now (`matcha/text/homographs.py`), with
      `op_g2p(homographs=False)` as the default so nothing has changed yet.
      **The filed estimate was wrong in shape, not just in size.** "3.3% of transcripts
      contain a homograph, so roughly half carry a wrong vowel" implied the damage was
      spread; measured over v3c it is **concentrated**. For `house` (438 occurrences),
      `does` (227), `number` (116), `wind` (108), `mouth` (81) and `perfect` (67) the
      dictionary's one pronunciation is right essentially every time — `wind` is not a
      defect in this corpus at all. **`live` alone is 87 of it**, because the dictionary
      ships `lˈaɪv`, the *adjective*, and narrative prose is almost all verb. Then `use`
      37, `read` 19, `content` 16, `suspect` 15, `lives` 9.
      **281 tokens in 277 rows would change — 0.88% of the corpus.** 85% of homograph
      tokens abstain. Precision was established by reading **all 288 flips of the first
      pass by hand**: 6 were wrong, each became a guard, and the re-audit is 281 flips at
      0 known errors. No IPA is invented anywhere — every alternate is the dictionary's
      own inflected form minus its suffix, or a rime-mate it already holds.
      The decision is the same one D-L2 poses, and **it is the same re-derivation**: this
      is a corpus change, so it wants to ride the § 1 pass rather than force a third bump.

      **One cost surfaced 2026-08-07 that was invisible when this was filed:** the mobile
      front end would have to resolve homographs too, and it cannot do it from a table —
      the decision needs the two tokens to the left and one to the right, with punctuation
      as a barrier. So `matcha/text/homographs.py` has to be ported alongside the
      dictionary. Gate **G7 refuses a homograph-enabled export** rather than letting the
      device quietly render `live` as the adjective the corpus was trained away from, so
      this is a known blocker rather than a future surprise. It is not an argument either
      way — mobile has not started — but a yes means the port carries a resolver.

      **THE CALL, ANSWERED 2026-08-07: YES.** The next derivation runs with
      `op_g2p(homographs=True)`, on the same pass as D-L2 and the width. The three notes
      below are kept because they are why no ear test is owed — the decision was an
      admission one, not a measurement one, and re-litigating that is how a settled call
      turns back into an open one.

      - **There is nothing to audition, and the audition app is not involved.**
        `measure_homographs.py` writes nothing except the file named by `--json`; it
        imports no synthesis, touches no `ratings.csv`, and stages no clips.
        `--apply-sample N` prints N rows as three lines of *text* — the sentence, its IPA
        with the flag off, its IPA with the flag on. It is a phoneme diff on stdout. An
        earlier note in this file said "to hear what moves", which was wrong.
      - **Auditioning the affected corpus clips would tell you nothing.** These are
        LibriTTS *real audio* rows: the reader already says `lˈɪv` in "they live on buds
        and blossoms". The wrong vowel is in the label we train against, not in the
        recording, so the clip sounds correct either way and the ear has no purchase on it.
      - **A meaningful ear test would have to be synthetic, and is not obviously worth
        running.** Render both phoneme strings through `vat3-24k` ep099 and A/B them —
        with per-pair randomised assignment, or position gets scored instead of the
        phonemes ([quality-gap-plan.md](quality-gap-plan.md), "two traps"). Expect
        confirmation rather than discrimination: all 281 flips are mechanical vowel and
        stress corrections sourced from the dictionary's own inflected forms, so the
        question an A/B answers ("is `lˈɪv` right for *live*?") is not the question in
        doubt. Worth doing only if the yes/no wants an independent check.

      Out of reach and recorded as data in `NOT_RESOLVABLE`, not as an absence: `bow`
      `row` `bass` `sow` `lead` (two senses, one part of speech — no POS rule can
      separate "took the lead" from the metal); `polish` and `august` (carried by CASE,
      and `cleaners.lowercase()` runs before the G2P sees the token — the evidence is
      destroyed upstream, which is a *cleaner* problem, not a G2P one); `excuse` (the
      dictionary gives /s/ for every form, so the verb's /z/ would have to be invented).
      Past-simple `read` and finite `wound` deliberately never fire: both are
      indistinguishable from the present without tense agreement across the clause.

## 4 · Conditioning axes — one approved, one OPEN

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
- [ ] **If it is spiked, the shape is settled in advance** and cheap: it APPENDS as
      channels 8+ on the same zero-init FiLM path, so `unknown` stays all-zero and every
      existing checkpoint and filelist keeps its meaning. Reordering is the one edit that
      must never happen — position is the wire format now. Vocabulary and width are a
      **contract change requiring an owner call** (ARCHITECTURE §1); `matcha/delivery.py`
      is where it would live and refuses anything outside the closed set, so it cannot
      arrive by accident.
- [ ] **The labels do not exist yet in that form.** Delivery shipped with 1,189 labelled
      keeps; an 8-way emotion block starts at zero. The register lexicon is the obvious
      source (a 47 → 8 fold), but it is Director-authored intent rather than an ear
      verdict on the render, and the instrument-vs-intent distinction is the one this
      corpus keeps having to relearn. Costing that fold is part of the spike, not a
      preliminary.

## 5 · Ears queue (priority order)

1. [ ] **Qwen vs VibeVoice at equal loudness** — the one teacher-portfolio ranking the
       loudness confound (5.99 dB spread, Qwen +3.81 dB over VV) could have produced.
       The 56 keeps are normalized; a keep-*rate* re-test additionally needs the 24
       quarantined drops normalized (un-quarantine is an owner call).
2. [ ] **Orpheus tier** — `normal` is explicit in `ENGINE_TIER`; `trusted` is arguable on
       105 non-`tara` renders at 80.0% keep / mean 4.49. Policy, not measurement.
3. [ ] **moss_vg** — newscaster 20/20 is confounded (the register flatters its
       radio-timbre/IVR failure mode; its 24% rate was measured on expressive material).
       Re-test on dialogue or an expressive register before any tier movement.
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

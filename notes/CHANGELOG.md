# Changelog

This document tracks technical changes, refactoring milestones, and build-system adjustments
for Project Sonora (the training pipeline and the teacher-synthesis lane).

> **Maintenance:** Required by [AGENTS.md](../AGENTS.md) §4, which is the rule of record —
> append an entry **after committing** code work (source, configs, dependency manifests;
> docs-only commits are exempt), always carrying the short 7-character commit SHA. The log is
> append-only within a release cycle: entries are pruned **only** when a new version of the
> overall project is tagged, at which point they collect under that version's heading. New
> entries go at the top under the current date, using the `Added` / `Changed` / `Fixed` /
> `Removed` structure.
>
> Maintained since **2026-08-06**, when the changelog + code-review cycle (AGENTS.md §4–§5)
> was adopted here to match the convention Prosodia already runs. History before that date
> lives in `git log` and [STATE.md](STATE.md).

---

## 2026-08-08

### Added — Phase 1 #1, the Emilia merge (§1)

- **`31207bf` — `data/libritts_r_emilia_vat_v5`, the first corpus here that is not one
  dataset.**
  41,138 train / 1,304 val, **78.5 h, 2,500 speakers**, against v4's 30,485 / 960 / 51.3 h /
  247. `scripts/merge_emilia_corpus.py`, `configs/data/libritts_r_emilia_vat_v5.yaml`,
  `configs/experiment/vat5_finetune.yaml`. Warm start `vat5_init.ckpt` at **338 warm
  (1 widened) / 0 fresh**; seam guards **30/30**; smoke run trains.
  - **v4's rows pass through BYTE-IDENTICAL** and its speaker indices are untouched —
    verified as prefix equality on the artifact, not asserted about the script. That is
    what makes the warm start legal: `spk_emb` row *i* is a person, and a renumbering that
    produced identical clips would pass a set comparison while silently reassigning every
    voice the model has learned. `spk_emb.weight` widened 247 → 2,500, new rows keeping the
    model's own init (a zero channel contributes nothing; a zero embedding is one specific
    point, not a neutral speaker).
  - **The merge is a separate script from `derive_vat_corpus` because of the labels.** The
    derive lane's per-speaker z is right for LibriTTS and destroys Emilia — 2,408 speakers
    at a median of 3 clips, tail-selected, so re-centring hands 756 one-clip speakers a
    label of exactly 0.0. Emilia is labelled on the **global anchor** instead (owner option
    1). **0 of 10,997 rows label all-zero.**
  - **2,144 of 13,141 keeps dropped, and the todo's "all 13,141" was never going to hold.**
    **1,676 carry digits** — D-M3, the tokenizer deletes rather than expands them, so "320
    members" would train against audio that says the number. NOT normalised to words: "1
    Chronicles" is *First* Chronicles, "Ezr. 232" is "two thirty-two", 1,262 distinct digit
    tokens. Guessing produces a transcript that looks right and matches nowhere, which is
    D-M3 arriving through the fix instead of the bug. The drop is unbiased across the mining
    criteria (T+ 13.9%, V+ 14.9%, V− 10.8%, A+ 12.5%) over 809 speakers, so it costs volume
    and not tail-selection. **468 exceed `ASR_MAX_WER`** against their YODAS caption —
    `process_emilia_tail` recorded that cross-check in July and deferred the threshold to
    merge time on purpose. Zero clips failed on audio, vocabulary or unresolved words.
  - **`n_spks` is 2,500, not the 2,655 the plan predicted** — 155 speakers lost every clip
    to those filters. Derived from the corpus, never assumed.
  - **`0fe4b3d` — the threshold is shared, not copied.** `ASR_MAX_WER` moved to `synth_common`; the QC
    gate re-exports it. Two lanes asking one question ("does this transcript disagree with
    this audio too much?") with two constants is B-L5 and D-L2 for the third time.
  - ⚠ **T saturates at 53.6% on the Emilia half** (LibriTTS: 4.7%), accepted by the owner
    for this run. The failure signature is in `quality-gap-plan.md`, written **before** the
    result, and must be read before the holdout number rather than after.

### Fixed — three things the merge walked into (§1)

- **`31207bf` — `data_statistics` inheritance would have been a real error this time,
  not a pedantic one.** v4 → v5 moves the mel mean by **0.1587** and the std by **0.3208**; the v2 → v3c
  move that put "re-measure in-container" on the checklist was 0.0203 / 0.0024. Adding
  27.2 h of podcast/YouTube audio to 51.3 h of studio reading is a different distribution by
  construction. The config shipped with the key **absent** until the container had measured
  it — `data_statistics` is positional, so a run before then dies naming it, where a
  plausible placeholder would not.
- **`92c04f5` — `make_warmstart`'s prefix proof read only the LibriTTS namespace.** A merged corpus
  keeps its sources separate (`libritts_id_to_index` beside `emilia_id_to_index`), so the
  check passed and reported *"247 donor speakers keep their index; 0 appended"* on a corpus
  appending 2,253. It was not wrong about safety — it was wrong in its summary, which is the
  line people quote. It now reads every `*_id_to_index` block, refuses a cross-namespace id
  collision, and refuses two speakers on one embedding row. Two new seam checks (28 → 30).
- **`b5abf90` — `on_validation_end` crashed with no logger configured.** It runs a full synthesis pass
  purely to log spectrogram images, then dereferenced `self.logger.experiment` —
  `AttributeError: 'NoneType'` — *after* validation had already succeeded, which reads as a
  data or checkpoint failure and is neither. `logger=[]` is exactly what a smoke run uses,
  so the one configuration used to prove a new corpus loads was the one that crashed. Same
  family as `test: True`: a supported setting failing late, in a place that blames something
  else.

### Measured — the ears queue emptied (§5, §2)

- **`683c43f` — `head_ok` gets NO GATE, on evidence.** The owner heard the four queued
  clips. Ranked by `head_lost_frac`: **0.214** DROP (shrill — nothing to do with the head),
  **0.211 KEEP 5** (*"I hear all of the words in the printed text"*), **0.132 KEEP 5**,
  **0.129** DROP (*"the first two words are jumbled together… 'gojawayisabella'"*). The
  measure is **anti-correlated with the ear**; any cut catching 0.129 catches both 5s.
  Across all 8 flagged clips not one was a late start — `head_lost_frac` measures ASR head
  DISAGREEMENT, not truncation. Stays advisory (slurred onsets are real), note reworded:
  it had told auditors "first N words never spoken", false in 3 of the 4 cases they checked.
- **`683c43f` — the calibration tool was discarding the ear's rejections.**
  `parse_score` searched for `[1-5]`, so the reject marker `x` returned `None` and every
  caller skipped the row: **253 of 286 dropped rows, 88% of every rejection, invisible to
  the tool whose job is calibrating gates against ear verdicts.** A campaign full of drops
  swept as "no rated clip matches the defect". **`TAIL_LOST_MAX` 0.05 confirmed** at 3/3
  catch / 1% false flag once they were visible.
- **`35bbce6` — `PAUSE_HARD_MAX` re-swept and DELIBERATELY unchanged.** `--campaign-dir` is
  repeatable now, since one-campaign-at-a-time is what made every defect set thin. Pooled
  over 161 keeps the drops are **interleaved with keeps** (2.112 keep, 1.984 keep, 1.600
  DROP, 1.568 keep, 1.536 keep, 1.408 DROP) and the longest pause in the set is a keep. No
  threshold separates them. **Duration is the wrong axis** — both drop notes object to
  WHERE the pause falls, mid-phrase. Filed as a pause-POSITION measure, not a new constant.
- **Qwen vs VibeVoice at equal loudness — the confound was real and ran BACKWARDS.** 33
  keeps re-listened at 0.01 dB apart (were 3.81 dB apart). Paired: qwen **4.94 → 4.88**
  (−0.06), vibevoice **4.20 → 3.33** (−0.87), **gap +0.74 → +1.55**. Normalising should
  have shrunk the gap if loudness flattered Qwen; it more than doubled. **Loudness was
  masking VibeVoice, not manufacturing Qwen** — and qwen's −0.06 is the control that makes
  the attribution safe. The ranking is confirmed and was understated. VibeVoice's exposed
  defect is ROOM reverb (*"an older television broadcast… black and white tv"*, 5 → 1
  twice), which `radio_score` does NOT catch (0.3299 against a 0.10 bar) — consistent with
  reverb having defeated four detector attempts.

### Changed — audit tiers, and the trade that lets them move (§5)

- **`4150dfa` — the defect detectors are wired, and moss_vg leaves `scrutinized`.**
  `qc_engine_defects.py` had existed since 2026-08-02 as, in its own words, *"what the tier
  system needs in order to ever hand those engines a ride-along back: coverage traded for
  instrumentation"* — but as a manual `--append-flags` step, so the trade was notional. A
  detector nobody runs trades nothing. `synth_bank.sh` runs it on every bank now.
  The hold rested on two things and **neither survived re-measurement**: the "24% rate on
  expressive material" is not reproducible from anything on disk (the source, `4abfd3f`,
  measured **19/54 = 35% REJECTED on `delivery-v1-narration`** — a *narration* campaign),
  and **11 of moss_vg's 20 non-keeps are bookkeeping retirements**, not ear rejections, so
  on real ear verdicts it is **95 heard / 90.5%**, not the 81.1% first reported here. The
  dialogue re-test the hold was waiting for already existed: 19 heard at **89.5%**, against
  a 73.5% lane mean and above `chatterbox`, which is trusted.
  What justified 100% listening was the DEFECT, and both halves are instrumented now: 5 of
  the 9 genuine rejections were structural (early EOS, 2.82 s and 3.04 s dead air, 0.67 s
  and 3.68 s of speech) and every one fails `pause_ok`, `tail_ok` or `speech_ok` today; 3
  of the 4 timbre failures are caught by `radio_timbre` (out/in-band 0.014 / 0.072 / 0.094
  against a 0.10 bar), flagging 19.8% of the batch. **The residual is named:**
  `windfairies_nar_0034`, "highly robotic… like an old-fashioned phone IVR", passes every
  instrument. **Orpheus held at `normal`** — 90.8% ex-`tara` is really `jess` at 97.4%
  carrying 72% of the population; the remainder is 74.2%.
- **`b6d29b2` — a hyphen is a separator, not a deletion.** `edge_loss` STRIPPED `-` rather
  than splitting on it, so `By-and-by` collapsed to one token against a three-token
  reference and reported a 3-word late start on a perfectly spoken opening. Every other
  text path already splits on hyphens. Measured before changing it: 8 head counts and 46
  tail counts move corpus-wide and **0 clips had ever failed `tail_ok` because of it**.
  Found while checking why 4 of the 8 head cases were one LibriVox title — which turned out
  to be a **size effect, not an alignment defect** (uneasy-money is 2,030 of 3,189 clips at
  a 0.20% head-loss rate, the lowest of any title). The real pattern: head loss is **~5× a
  synthesis problem** (1.30% rendered vs 0.27% force-aligned).

### Added — Phase 1 #1 prep (§1)

- **`e6a2d14` — the Emilia keeps were undeclared, and #1 is not a merge.** The licence wall
  would have refused every keep: they live in `emilia_kept/` and `emilia_kept_24k/` while
  the manifest declared only `emilia_yodas`. Provenance verified before declaring, since
  `emilia_original` next door is NC. And **per-speaker z would have destroyed the corpus
  silently** — the keeps are tail-selected (`T_full > p90` etc.) across 2,408 speakers at a
  median of 3 clips each, so re-centring takes V from mean +0.387 to **+0.000** and hands
  756 one-clip speakers a label of exactly 0.0.
- **`94d0184` — global anchoring (owner's option 1).** All 13,141 keeps label, **0 all-zero**.
  ⚠ T saturates at **54.0%** on the ±1 rail against LibriTTS's 4.69%; accepted for this run,
  with the failure signature recorded in `quality-gap-plan.md` **before** the result.
- **`2e7d908`, `76b6aea` — the speaker table widens, against proof.** `n_spks` 247 → 2,655
  needs `spk_emb.weight` widened, which is safe here (indices preserved, new ones appended)
  and wrong for the vctk donor (row *i* is a different person). `--donor-speakers` proves
  the prefix property; absent it the table stays fresh. New speaker rows keep the model's
  init — a zero channel means "contributes nothing", a zero embedding does not.

### Fixed — the smoke runs, and what they caught (§1)

- **`3b6b738` — `test: True` was a trap with teeth.** No model here defines `test_step`, so
  `trainer.test()` raises — unreachable only because `max_epochs: -1` means `fit()` never
  returns. Every production run has been stopped by hand or by the E610 fault, never by
  finishing. The first run to terminate normally would have crashed *after* writing its
  checkpoints and then crash-looped under `restart: unless-stopped`, reading as a training
  failure it is not. Found by the vat4 smoke run — the first thing here to set a finite
  `max_epochs`.
- **Both smoke runs pass.** 8-wide on v4: 25/25 batches, all four loss terms, clean restore.
  Merged stub (v4 + 200 Emilia rows, `n_spks` 280): speaker map PREFIX OK, 338 warm / 0
  fresh, val_epoch 1.572. Seam guards **28/28** in-container.

## 2026-08-07

### Added — the v4 corpus (§1)

- **`c6b33ee`, `b5b1f34`, `be1cce3` — `data/libritts_r_vat_v4`, all three owner decisions
  on one derivation pass.** 31,445 clips / 51.3 h / 247 speakers, 30,485 train / 960 val,
  independence gate PASS (T·A −0.059, T·V −0.066, V·A +0.027). The blake2b hash split held
  exactly as designed: **960/960 val clips identical to v3c**, so nothing held out became
  trainable. Licence wall accepts it; seam guards 22/22 in-container; warm start
  `vat4_init.ckpt` at **338/338 warm, 0 fresh**. Configs:
  `configs/data/libritts_r_vat_v4.yaml`, `configs/experiment/vat4_finetune.yaml`.
  - **D-M4 shipped** — `--homographs`, which REFUSES to combine with `--reuse-from` (the
    phonemes would come from the old corpus and the flag would silently do nothing).
    **269 train rows carry different IPA**: *"where are you going to live?"* is `lˈɪv` now,
    not the dictionary's adjective `lˈaɪv`.
  - **D-L2 shipped, and OVERSTATED itself.** Measured on the finished corpus rather than
    on the intermediate file: **V moved on 247 of 30,485 rows by at most 0.0008; A and T
    are bit-identical to v3c.** The "31,443 clips change, up to 0.228" headline is true of
    `corpus_valence_combo.json` and false of the labels, because `derive_vat_corpus`
    applies its OWN per-speaker z on top and a constant head's per-speaker-CONSTANT offset
    is exactly what the second z subtracts away — speaker `909`'s worst clip moves 0.2280
    in the combo and **0.000036** in the label. The corrected guard is right and stays; it
    was not, on its own, a reason to bump the corpus. The width was.
- **`b5b1f34` — `--delivery-from` applied nothing and said nothing about it.**
  `_load_delivery` keys on `normpath(link)` AND `basename(link)`, but `label()` looked up
  only the absolute corpus path — and `ratings.csv` records `link` relative to the ratings
  directory, so the normpath key can never match and the basename half was dead code. The
  first v4 run printed "delivery: 1635 labelled clip(s)" and applied **zero**: correct
  width, every delivery block all-zero, exit 0. Basename is now a fallback, an **ambiguous**
  basename (>1 lane across campaigns) is DROPPED rather than resolved last-write-wins, and
  the run reports what was APPLIED rather than what was loaded — they are different numbers
  and nothing said so.
  ⚠ **Delivery is nearly empty on this corpus and that is correct**: 48 of 31,445 clips
  carry a lane (39 Dialogue, 9 Neutral), the `audit-markup-v0` rows. The other ~1,590
  labels belong to the synthetic and LibriVox campaigns — a different corpus, merging in
  Phase 1. v4 buys the WIDTH, not the delivery signal.
- **`be1cce3` — `make_warmstart.py` discarded the conditioning pathway it was supposed to
  carry over.** It dropped shape-mismatched tensors and let random init stand, so
  `vat_trunk.net.0.weight` going (256,3,1) → (256,8,1) threw away everything `vat3-24k`
  ep099 learned about mapping V/A/T into the FiLM trunk — on a run whose entire purpose is
  to change the width and nothing else. `todo.md §1` asserted this treatment "already"
  existed; it did not, and the first `vat4_init.ckpt` was built with 2 fresh tensors.
  Widening is an explicit **allowlist**, not a rule: correct for the VAT trunk because
  contract v2 makes channel POSITION the wire format (channels 0–2 mean the same in both
  widths, the appended block is zero for `unknown`), and wrong for `spk_emb.weight`, which
  also grows (109 → 247) but whose row *i* is a different person. Verified: channels 0–2
  bit-identical to the donor, channels 3–7 all zero.

### Changed — QC admission and the ear queue (§2)

- **`97c14b4` — `speech_ok` is a HARD GATE at 4.0 s, on the VAD figure.** Owner's call.
  The threshold was never the open question — it has been theirs since 2026-07-25, and the
  keep-rate cliff is exactly there. What blocked it was the INSTRUMENT: `librivox_align`
  enforces the same floor with `librosa.effects.split` and says in as many words that it
  does so to match qc_gate, "because two different measures of one owner rule is one
  measure too many". The sweep settled that — mean VAD/energy ratio **1.008** over 150
  clips, median delta −0.06 s, and exactly **one clip in 150** changing side at the floor.
  `librivox_align` keeps the energy gate on purpose as an INGEST pre-filter, with the 0.7%
  disagreement now written down instead of assumed. Cost, stated plainly: **~4% of clips
  that passed QC the day before do not pass it now.** Guarded so a future `None` cannot
  silently become a gate that fails every clip, since `None` is falsy and
  `all(gates.values())` cannot tell it from a failure.
- **`97c14b4` — an advisory could not queue a clip, which is why the eight were never
  heard.** Found while acting on the owner's "stage them": `stage_pool.qc_flagged` tested
  `all(gates.values())`, and `head_ok` is an advisory *precisely because* it has no
  calibrated threshold — which it cannot get until the ear describes the failure once. So
  **the one finding that most needed an auditor was the one finding that could not reach
  one**, and a head-truncated clip kept the "known quantity" relaxation and folded into a
  bank as a silent `keep`. A threshold is only needed to REJECT a clip; queueing one costs
  a listen, and the listen is where the threshold comes from.
  - `synth_common.head_flagged` recomputes from `text`/`asr_hyp` when the row predates the
    head measure — which every `qc_measures.jsonl` on disk does, so old campaigns are
    covered without a re-run.
  - `register_audition` imports `HEAD_WORDS_FLAG` rather than re-declaring it: the number
    that QUEUES a clip and the number that TELLS the auditor what to listen for have to be
    one number, or the queue fills with clips whose note says nothing is wrong.
  - `queue_head_audit.py` — report-only by default, through the shared mtime-guarded
    `ratings_transaction`. Queues only clips that **passed every gate**, since a clip that
    already failed one is going to the ear anyway with a note that says why.
  - **4 of the 8 queued; the other 4 are `uneasy-money` clips still in the POOL** — never
    staged, so there was nothing to re-audition, and they are exactly the silent fold
    `qc_flagged` now prevents. **No verdict was discarded**: status moves to `unaudited`,
    the score and the auditor's own note stay in place, and the generated line goes after
    theirs. Two of the four had been kept, one at 5.
  - ⚠ A pattern the finding did not name: **4 of the 8 are one title**, `uneasy-money`,
    across both librivox v1 and v2. Worth looking at that alignment before a threshold is
    chosen from a population it supplies half of.
- Guards in `tests/test_edge_truncation.py` (20 cases, from 9).

### Fixed — the device text front end (§1)

- **`8a07655` — D-C1 was still live on the device side, five days after the host was
  fixed.** `kotlin_replica.phonemize` was `DICT.get(token) or phon_word(token)`: a flat
  dictionary lookup with a neural fallback and **no apostrophe handling of any kind**. The
  host got the contraction table on 2026-08-02, when D-C1 was found to have poisoned the
  IPA of corpus versions v1 through v3; the replica never got it. Not a near-miss —
  verified against the shipped assets, **0 of 274,927 dictionary keys contain an
  apostrophe** and `'` is absent from the neural charset, where unknown characters are
  silently skipped, so a contraction could only ever resolve to its apostrophe-stripped
  letters *and came back looking like a successful lookup*.
  **Measured on the probe corpus: the old front end diverges from the training front end
  on 69 of 86 probe sentences**, and worse in shape than the finding described —
  `they've` → `θˈeɪv` (a th-fricative on a pronoun), `she'd` → `ʃˈɛd` (the word *shed*),
  `i'm` → `ˈɪm`, `james's` → `dʒˈːmˈɛs`. The filed exemplars (`don't` → `dˈɔnt`, `we'll` →
  `wˈɛl`) were the mild cases.
  **Half the finding is answered rather than fixed: there is no shipped Kotlin app** and
  mobile front ends have not started in earnest, so the open question — whether a real app
  carried a table the replica lacked — has no subject. The replica is not validating a
  front end nobody runs; it *is* the front end, as spec, which is why this is worth
  landing before anyone ports it rather than after.
  - **`scripts/litert_export/device_g2p.py`** — the text front end as its own module, so
    it can be compared against `matcha.text.op_g2p` without loading the replica's graphs.
    Deliberately imports nothing from `matcha`: a replica that called the host's
    phonemizer would prove only that a function equals itself.
  - **`g2p_contractions.json`** — the tables ship as an asset **exported from
    `op_g2p.contraction_tables()`**, not transcribed into the port. D-C1 is a defect of
    omission and a hand-synced table is the same defect on a delay. A device that cannot
    find the asset **raises**; it does not fall back to the plain dictionary, because a
    silent fallback is exactly how this survived two corpus versions.
  - **Gate `G7`** — phoneme-string parity between the two front ends over 86 probe
    sentences, one per contraction-table entry plus 17 hand-built cases (clitics on
    arbitrary bases, the three possessive allomorphs, archaic forms, apostrophe names,
    abbreviation expansion, brackets and dashes, non-ASCII folding, punctuation, OOV).
    Every other gate certifies the graph; none of them asked whether the text reaching it
    matches the text the model trained on.
  - **`config.json` gains a `g2p` block** — the manifest is the only thing a host can
    read, and the front end is as capable of silently disagreeing with the corpus as the
    control surface was (F-H2's argument, applied one layer earlier).
  - **The normalization chain moved with it**: ASCII fold, lowercase, abbreviation
    expansion, bracket removal, hyphen-to-space, whitespace collapse — in that order,
    before tokenization. The replica ran `text.lower()` alone, so `Mr.` and `well-known`
    tokenized differently from the corpus. A Kotlin port needs a `unidecode` equivalent;
    that is a real porting dependency, since curly quotes and accented borrowings are
    ordinary in the book lane.
  - **`tests/test_device_g2p_parity.py`** (11 cases) — including a guard that the check is
    **not vacuous**: the flat lookup this replaced still fails it.
- **`8a07655` — D-C1 itself had no regression test, and G7 cannot supply one.** Found
  while checking G7 for vacuity: both front ends read one table, so an entry deleted from
  `_CONTRACTIONS` disappears from host and device together and **parity still passes while
  both sides are wrong**. Parity is not correctness. Anchored with absolute phoneme
  assertions (`don't` → `dˈoʊnt`, `we're` → `wɪɹ`, `'tis` → `tˈɪz`) plus table-size and
  clitic-set floors. The 2026-08-02 fix had shipped unguarded for five days.
- **`8a07655` — a D-M4 cost that was invisible when it was filed.** G7 **refuses a
  homograph-enabled export**: resolution needs the two tokens to the left and one to the
  right with punctuation as a barrier, so there is no table to ship and the device cannot
  reproduce it. Turning D-M4 on therefore adds "port `matcha/text/homographs.py`" to the
  mobile lane. Recorded in `todo.md §3` against the decision, and enforced at construction
  so the two front ends cannot drift the moment the flag flips.

### Added — G2P / label derivation (§3)

- **`33d2a4f` — D-M4: the dictionary has one pronunciation per key, and the other sense
  trains on it.** `matcha/text/homographs.py` resolves a heterophonic homograph from its
  left context; `op_g2p` takes `homographs=False` by default, so nothing changes until a
  caller asks. Filed as "3.3% of transcripts contain a known homograph, so roughly half
  of those carry a wrong vowel" — the half is wrong and so is the shape. Measured over
  v3c (31,445 rows, 556,454 tokens), the defect is **skewed, not spread**: for `house`
  (438 occurrences), `does` (227), `number` (116), `wind` (108), `mouth` (81) and
  `perfect` (67) the single shipped pronunciation is right essentially every time, while
  `live` alone accounts for 87 wrong tokens because the dictionary ships **lˈaɪv, the
  adjective**, and narrative prose is almost entirely the verb. Then `use` 37, `read` 19,
  `content` 16, `suspect` 15, `lives` 9. **281 tokens in 277 rows — 0.88% of the corpus —
  would change; 85% of homograph tokens abstain.**
  No IPA is invented: every alternate is the dictionary's own inflected form minus its
  suffix (`conducted kəndˈʌktᵻd` → `kəndˈʌkt`, exactly the verb the noun entry
  `kˈɑːndʌkt` is missing) or a rime-mate that is in the dictionary (`red ɹˈɛd`,
  `unwound ʌnwˈaʊnd`), and each entry carries its source.
  Nothing fires without positive evidence for a sense the dictionary does not already
  give, so an unrecognised context is a no-op. **All 288 flips of the first pass were read
  by hand; 6 were wrong, and each produced a guard** — a reduplication (`from mouth to
  mouth`, where the `to` is a preposition), a small-clause predicative (`pronounced it
  perfect`), a prepositional object (`growing on it close to the fence`), and a
  hyphen-split compound (`dove-like`, which reaches the tokenizer as two words). `wound`
  then lost its finite readings altogether: `to wound` is the transitive verb meaning
  injure, which IS `wˈuːnd`, and *"I wound and I heal"* is its present — four correct
  flips given up to avoid two wrong ones. It joins past-simple `read` as perfect/passive
  only. Re-audited: **281 flips, 0 known errors.**
  Eleven words are known and out of reach, recorded in `NOT_RESOLVABLE` as data rather
  than left as an absence: `bow` `row` `bass` `sow` `lead` have two senses sharing a part
  of speech; `polish` and `august` are carried by CASE and `cleaners.lowercase()` runs
  first; `excuse` would need its /z/ invented, since the dictionary gives /s/ for every
  form including `excused ɛkskjˈuːst`.
  `scripts/measure_homographs.py` is the dry run — the decision is always computed and
  counted, only its application is gated — and `--apply-sample` renders affected lines
  both ways. 43 tests in `tests/test_homograph_disambiguation.py`, including one per
  guard naming the sentence that produced it and one asserting every table default still
  matches the shipped dictionary, so a dictionary revision cannot quietly change what a
  flip means. Host suite 354 → 397.

### Fixed — QC / audit / staging (§2)

- **`1ebd14a` — C-M5 + C-M8: state that could be lost, and a rule with two answers.**
  The three remaining read-modify-writes each take a snapshot at startup, work for
  minutes, and write the snapshot back; what touched the file in between is gone, and
  gone silently, because the write itself succeeds. `staging_log.json` was the instructive
  one: it already went through `write_json_atomic`, so the write was never *torn* — it was
  *stale*, which loses a concurrent run's entry just as completely and would stage its
  clips a second time. `reader_profiles.json` was a plain `write_text` of a whole-file
  snapshot, and learning is per-reader and incremental, so one `--learn` could delete
  another's readers — losing that file does not cost a re-run, it un-confirms every
  (reader, title) and re-queues the whole ear backlog. `metadata.jsonl` was rewritten with
  `open(..., "w")`, which truncates first: an interrupted backfill leaves a *shorter valid*
  JSONL, a corpus that lost its tail and says nothing. Its `.bak`-on-first-run-only turns
  out to be correct (`--apply` only fills blanks; there is nothing to undo) — what was
  wrong is that every run printed "original kept at …", so run two named a file that is
  not its own pre-state as though it were an undo point. `write_json_atomic` and
  `write_text_atomic` also leaked their tmp on a failed rename, unlike the wav writers; a
  half-written twin of the ledger in a dataset directory is what a later `rglob("*.json")`
  reads as real.
  **C-M8** — "ear-confirmed (reader, title)" had two definitions at opposite thresholds.
  `pick_audit_subset` required all three of gender/age/accent before it stopped
  force-queuing a pair; `stage_pool` tested the *truthiness* of the attribute dict, so ANY
  ONE was enough to fold a whole title into the corpus as machine-written keeps, unheard.
  A pair on two of three was therefore both "confirmed enough to fold three hundred clips"
  and "still owed an ear pass" — and `learn()` produces exactly that shape **on purpose**:
  on disagreement WITHIN a title it writes `{attr}_CONFLICT` INSTEAD of `{attr}`. So the
  one title we have positive evidence is internally inconsistent is the one that arrives
  partial, and the looser reader folded it. C-M7 closed this on the hint path; the fold
  path went around it. One predicate now, in `reader_profile`, which owns the file it
  reads out of. **Preventive, not a repair** — both live pairs are fully confirmed, so
  unlike the four findings that had already fired this one had not.
- **`0a505d3` — C-L4: the loudnorm sidecar keyed on path, and a path is not a clip.**
  A reroll re-renders in place, keeping the filename; the stale record matched, the new
  take was skipped, and it shipped at its engine's native level — into a bank whose whole
  purpose is one level, and into the reference pool four cloning engines condition on.
  Counted under `already-done`. **Swept every `loudnorm.jsonl` on disk (1,029 clips, 11
  banks): 1,014 sit where their record says, worst case 0.185 dB; 15 are 0.90 to 7.09 dB
  away.** Those 15 are rerolls that shipped: `the-return_nar_0059..0063_doc_QWE` were
  re-rendered *twenty minutes* after their gain was applied, and ten more in
  `delivery-v1-narration-r2` now span **−30.1 to −17.4 LUFS — 12.7 dB inside one bank**,
  against the 5.1 dB per-engine spread that motivated the script. They went to the ear that
  way, where louder reads as more present. The same path key produced the *opposite*
  failure once already (the quarantine move, where a clip normalized under its old name
  was gained twice) — one key, two failures in opposite directions, because the key does
  not identify the audio. Records carry `sha_in`/`sha_out` now; legacy records fall back to
  the instrument (a clip still measuring at the level its record recorded IS that take),
  with the 0.5 dB threshold taken from the gap between the two measured populations rather
  than chosen. **Not applied to the live banks** — re-gaining clips the ear has already
  rated is a corpus change and the owner's call.

### Added — head truncation, and two thresholds left to the ear (§2)

- **`a4b6ec5` — C-M4.** `tail_lost()` computed the head-loss word count — `blocks[0].a` is
  exactly that — and returned only the tail, so nothing in the pipeline has ever seen a
  clip that *starts* late. Same defect as a truncated tail, invisible for the same reason:
  a global WER cannot separate "mangled throughout" from "the opening is missing".
  **Measured over every `qc_measures.jsonl` on disk (3,189 clips, 13 campaigns): 19 clips
  drop ≥3 opening words and EIGHT passed every gate** —
  `wuthering-heights_nar_0036_neu_CHA` lost *"There was scarcely time"*, 4 of 19 words and
  21% of the passage, at WER 0.211 against a 0.35 gate. Head loss runs about half as often
  as tail (0.6% vs 1.1% at ≥3 words); an earlier draft of the change asserted otherwise,
  before the sweep, and was wrong. `edge_loss` lives in `synth_common` so calibration can
  run over measures already on disk — every existing row carries `text` and `asr_hyp`, so
  eleven campaigns of evidence are available without re-running the gate.
  **`head_ok` and `speech_ok` gate nothing, deliberately**, and live in an `advisories`
  dict rather than `gates`: `hard_pass` is `all(gates.values())` and `None` is falsy, so an
  uncalibrated entry there would fail every clip. The run now names what it does not cover.
  The two are blocked for different reasons and **one of them turned out not to be**:
  `speech_ok`'s worry was that Silero counts less than the energy gate the owner's 4 s
  floor was set against, and that `librivox_align` uses the energy gate *on purpose*
  ("two different measures of one owner rule is one measure too many") — but over 150
  clips the mean VAD/energy ratio is **1.008**, the energy gate slightly *under*-counts
  here, and exactly **one clip in 150** changes side at 4 s. What is left is admission
  policy, not measurement. `head_ok` is blocked harder than filed: no drop note in
  `ratings.csv` names a late start, because no auditor has been asked to listen for one.
  `gate_calibration --sweep` makes the `PAUSE_HARD_MAX` procedure reusable — catch vs
  false-flag per candidate, defects selected by the owner's own note text, the matched
  clips printed so the regex can be checked rather than trusted, and a warning when the
  label count is too thin to conclude from.

### Added — the delivery channel (§1)

- **`72786ac` — contract v2's 4th conditioning channel ships. `vat_dim` 3 → 8.**
  Delivery had been specified since 2026-07-30 and delivery-v1 closed at 1,189 labelled
  keeps, with no model code implementing it. **The width is 8, not the 4 todo § 1
  proposed** (owner call): a single ordered channel asserts the five lanes lie on one
  continuum, and `seed_delivery.py` records that they do not — Dialogue vs Neutral is a
  property of the TEXT, Newscaster vs Documentary a property of the RENDER. A single
  channel also cannot carry `unknown`, which the contract pins as the zero vector: on one
  channel zero is the MIDDLE of the range, so the five lanes would sit asymmetrically
  around a hole and "no label" would be numerically adjacent to whichever lanes flank it.
  One-hot makes `unknown` all five channels at zero — the absence of a value rather than
  a value. Cost: ~1k parameters on a 1×1 conv.
  `matcha/delivery.py` is the only definition; the corpus derivation, the CLI, the
  Vocalizer and the export converter all read it. `--delivery LANE` on the CLI and a
  dropdown (not a slider — interpolating between lanes is meaningless) on the Vocalizer,
  per the standing rule that a capability ships with its vetting surface. The Vocalizer
  sizes the vector to the CHECKPOINT so a pre-v2 one still renders, and says
  "delivery IGNORED" rather than letting a dial silently no-op. An unrecognised lane is
  refused everywhere rather than read as unknown — it would otherwise be a silently
  unconditioned clip that still trains and is counted as unlabelled.
  **A second vocabulary turned up while guarding this**: `stage_pool.LANES` held the same
  five lanes in a DIFFERENT ORDER, which is harmless for reports and not harmless at all
  once position is the wire format. `NARRATION_LANES` was duplicated between `book_ingest`
  and `ref_select`, which import each other, so it now lives with the vocabulary.
  The EXPORT lane is deliberately not migrated — `convert_vat.py` still refuses, and now
  names why (F-H2, § 2). Verified in the pinned container: **22/22 seam checks** (13
  existing + 9 new) plus an end-to-end filelist round-trip.

### Fixed — the export lane, closed (§1)

- **`905e91b` — F-H2 + F-M7: the control contract.** `config.json` is the ONLY thing a
  mobile host can read, and it carried `vat_dim` plus a name-to-slot map — where each
  control is, and nothing about what may be sent there. **No bound was ever recorded, at
  any width** (verified against the shipped artifacts: no `control`, no min/max, no clamp
  key), so a request for valence 5 moved the FiLM activation off the manifold and still
  produced fluent speech. And **nothing said which channels are categorical**, so a host
  handed eight floats and three continuous names would reasonably crossfade all eight and
  blend Newscaster into Dialogue. `control` now carries both, plus CFG's method and its
  >=25-ODE-step floor (F-M7 — guidance is host orchestration and exports NOTHING, so a
  host cannot discover it from the artifacts). Demonstrated on five vectors; three were
  previously accepted silently. **G5** now probes a lane ONE-HOT rather than at ±1 (−1 on
  a lane is a vector the host is forbidden to send), and new **G6** requires the lanes to
  be mutually distinguishable — five inputs on one summing junction pass the per-channel
  probe five times and give the host five names for one behaviour.
- **`28756b4` — F-M1 + F-M5: the referee.** It bound graph inputs by a dtype heuristic, and
  on a conditioned graph `spks` is int64 shape (1,) — IDENTICAL to `x_lengths` — so it
  received the TOKEN COUNT as a speaker id. No error: it compared a render of the wrong
  speaker and reported a fidelity number for it. Bound by name now, with a shape fallback
  and a refusal on ambiguity, plus `--spk/--vat/--delivery`. Separately it gated on a
  scale-INVARIANT metric: `0.5 * reference` scores **cosine 1.000000** and passed a 0.99
  threshold, while the RMSE that would have caught it was computed, printed and thrown
  away. Worst possible blind spot for a model whose flagship axis is a loudness dial.
  Gain (dB) and normalised-RMSE gates added here and in the converter (**G3b**, **G3c**).
- **`1c8dc25` — F-M2: `rename_tflite_tensors`.** Outputs were mapped by EMISSION ORDER, so
  a swapped pair would rename the LENGTHS tensor `wav` — and Prosodia matches by name, so
  it would read a one-element int tensor as audio. Checked against shape and dtype now.
  `renamed 0 tensors` exited 0, so a graph whose names the table did not recognise was
  copied through untouched and called a success; it is a hard failure now. Also: the table
  predated contract v2, so a conditioned export's `spks`/`vat` kept their mangled names.
- **`e18877a` — F-M3 + F-M6: `kotlin_replica`.** It could not run: three of its seven
  prerequisites are VENDORED assets, not conversion outputs, and it only ever looked in
  `artifacts/` — dying at module scope on a bare FileNotFoundError naming one file and no
  directory. A preflight resolves both roots and lists every missing one with its
  producer. It also validated the unconditioned Phase 0 graphs, so the Kotlin port — which
  exists for the conditioned actor — was checked against a pipeline whose conditioning it
  does not have. Both lanes run now, driving spk/vat/delivery through the same manifest
  the host reads, with the mel stats taken from that manifest rather than hardcoded.

### Fixed — lanes, environments, acquisition, campaigns, QC

- **`87c65f8` — `matcha/app.py` deleted, and the two espeak leaks the item did not know
  about.** It phonemized through `english_cleaners2` unconditionally. The item claimed it
  was the LAST such place; auditing that claim found two more in the ONNX "Plan B" lane
  the README still documents. `matcha/onnx/infer.py` had no lane switch at all (an ONNX
  graph does not record its phoneme vocabulary, so `--lane` is now required with no
  default) and carried E-M2 in the lane the E-M2 fix never reached — hardcoded 22050, so
  24 kHz output played back ~9% slow. `matcha/onnx/export.py` builds no VAT input node,
  so exporting a conditioned checkpoint SUCCEEDED and yielded a permanently neutral graph:
  F-H1's shape again, an export whose logs say Sonora and whose graph is not. It refuses
  now, keyed on `use_vat` rather than `vat_dim` (which defaults to 3 on unconditioned
  checkpoints too).
- **`c48922b` — D-M2: the three container lanes pinned by digest, and what ran recorded.**
  All three ran `rocm/pytorch:latest` with unversioned `pip install`, so the environment
  that rendered a campaign — or scored the EIV heads the corpus labels derive from — was
  not merely unpinned but unrecoverable. Measured by dry-run resolution in the pinned
  image: an unpinned `pip install transformers` resolves to **5.14.1** today against the
  4.x the corpus was built on, and qwen was the only engine that pinned it. Also ruled
  out, because it is the case that matters most: no dependency set pulls a torch or nvidia
  wheel, so the image's ROCm build survives. One pin table (`scripts/container_env.sh`),
  installs migrated to `uv` per AGENTS § 3, and every run now freezes itself into
  `<campaign>/_env/<lane>.txt`. Verified live: a real install + capture produced a
  65-package freeze carrying `transformers==4.57.3`, owned 105:109.
- **`e5f27a7` — A-M2 was the tail truncation, not a latent span bug.** `forced_align`
  returns one entry per FRAME, so a token occupies a RUN of frames; `refine()` counted
  non-blank FRAMES against the number of CHARACTERS per word. Measured on a real alignment
  in the pinned container: the sentence ended at frame 6 where it truly ends at 13 —
  **54% of its duration dropped** — and word 2's span did not overlap its true span at
  all, so the middle spans pointed at the word before. The caller already carried a
  comment blaming CTC for exactly this. `torchaudio.functional.merge_tokens` is the fix.
- **`51231dc` — A-M8: Gutenberg boilerplate in the EPUB lane.** The plaintext lane cut
  PG's wrapper; the epub lane's only filter was a filename SKIP list in Standard Ebooks'
  vocabulary. PG names its documents nothing of the sort and often puts the whole book
  plus wrapper in ONE document, so ~4,000 words of licence prose parsed as prose, split
  into sentences, got directed and rendered as the novel. It also made a licence claim
  false: `text_provenance` stamps every PG bank "PG header/footer stripped", which no code
  performed — the A-M6 family. Residue that survives is REFUSED, not filtered.
- **`b0825c6` — A-M9/A-M10: one ledger identity per book.** Verified against the live
  ledger (56 entries: 27 `lv:`, 15 `se:`, 13 `pg:`). The SKIP check built a `pg:` key or
  None, so a LibriVox book with no etext id was compared against nothing and every
  re-route reset its status — regressing books already fetched and aligned, silently. The
  router never produced the `se:` scheme at all, which is why five Standard Ebooks titles
  sit under `lv:` keys today. `librivox_fetch` wrote `pg_6684` or
  `lv_uneasy-money-…` for one book depending on invocation style, and updated `status`
  only under an exact `--key`, so a `--url` fetch left the ledger saying "pending" forever.
- **`42cbf33` — A-M1/A-M13.** `float(playtime or 0)` inside `except: continue` dropped
  colon-formatted sections from the duration map, so the cumulative fraction was computed
  over a subset and the windows stopped tiling; and one zero playtime threw away every
  duration in the book, reverting to the even split that located **0%** of the heard words
  in Dickens's *Speeches*. Now all-or-nothing with residual imputation, and loud. Plus
  retries in a lane pulling sixty-odd files from a rate-limited archive.org.
- **`421d919` — A-M11: the director pass is checkpointed.** Results reached disk only
  after the last chunk, so a fault at chunk 90 discarded 90 calls to a 31B model. Keyed on
  CONTENT, not position — an index-keyed checkpoint would hand chunk 40's direction to
  chunk 37 after a resume with different arguments, which is worse than none because the
  bank still looks complete.
- **`09dac4d`, `98734f9` — § 6.** **B-M9**: `BRIGHT_REF_POLICY` was stamped into every
  chatterbox manifest and read by nothing, and the value it recorded did not describe what
  the code did (a third, undocumented hybrid). **B-M7**: `make_bulk_bank.py` bypassed
  `build_direction()`, which AGENTS.md forbids in as many words, and paid three times —
  Dia lines lost the trailing `[S1]` end-of-audio guard, all 87 qwen lines wrote the voice
  design to a `design` key `synth_qwen.py` never reads (verbatim the 2026-07-25 owner
  finding that `build_direction` exists to prevent), and ids omitted the engine. The
  bypass explained itself: the SSOT had no slot for `quality` or Dia inline tags, both of
  which the renderers genuinely take. Plus B-M6, B-L2/L3/L4/L6/L7/L8/L10.
- **`cc4f1a5` — C-M6/C-M7: one `ratings.csv` transaction.** Six writers, four private
  mtime flavours. `pick_audit_subset`'s five-attempt retry loop is gone on purpose: with
  an flock serialising our own scripts the only writer left to race is a human in a
  browser. **C-M7** came out of the same file — the cross-title hint was gated on the
  truthiness of the title entry, but a title whose ear pass disagreed with ITSELF still
  produces one, so a guess was written into the one title we have evidence is inconsistent
  and then marked machine-written so it looked settled.
- **`9f53095` — § 7.** **C-L6**: `round(rest * 0.03)` is zero below 17 clips, so the
  trusted tier's tail sample — the entire mechanism for noticing a trusted engine has
  begun drifting — silently did not exist on small batches. **C-M1**: deferral only
  flipped one way, so a clip QC-flagged by a LATER pass stayed deferred forever, breaking
  the tool's one non-negotiable rule. **C-M9**: the cohort glob swept `_dropped/` and
  `_superseded/` into statistics scored as a robust z against the cohort's own median, and
  the manifest glob was non-recursive while the wav glob was — so a standard campaign
  layout found no manifests and collapsed into one `"?"` cohort. **C-L5**: no publication
  policy for `librivox`, which would hard-error the day the first staged real-audio clip
  reached a metadata.jsonl.

---

## 2026-08-06

### Fixed — export gates (§2) and campaign tooling (§6)

- **`445d01e` — F-C1, the review's only open Critical, was two defects sharing a line.**
  The gates **printed** pass/fail and nothing acted on it: `main()` ran to completion, wrote
  `artifacts/` and `config.json`, and exited 0 whether the graphs matched the model or not —
  so a failed export was indistinguishable from a good one at the shell *and left a
  complete, shippable artifact set behind*. There is a ledger now and `gates_or_die()`
  refuses to write unless every gate passed. The second half is the one no gate covered:
  **valence and tension had never been driven nonzero through a converted graph** (G2 runs
  `vat = zeros`, G3/G4 drive `[0, a, 0]`), so a dropped or mis-sliced input would have
  passed every gate and shipped — reading on-device as a *training* failure. New **G5**
  drives each channel independently and requires the waveform to move. **Verified by a real
  conversion: 10/10 gates pass**, with V/E/T mean |delta| vs neutral of 7.0e-2 / 9.2e-2 /
  8.6e-2 — the first measured proof all three channels survive export. Also: G4's
  monotonicity used `all()` over a possibly *empty* sequence (PASS on zero evidence), and
  **F-M4** — `config.json` wrote `time_embed_dim=1024` while the graphs consume
  `in_channels` (224), a number the mobile host trusts to size its buffer.
- **`445d01e` — §6.** **B-M3** chatterbox/zonos/vibevoice now call the existing
  `rebuild_used_set`; the variety-bias `used` set started empty on every resume, so a
  resumed campaign silently lost its diversity guarantee. **B-M5** `attempt_seed` perturbs
  the seed per attempt and records the one actually used — "re-run under skip-if-exists" *is*
  the retry, and re-seeding identically meant a deterministic failure could never converge.
  **B-L5** `MAX_REF_EXCURSION` hoisted to `ref_select` (it was written out three times, and
  it encodes a measured finding).

### Fixed — QC / audit / staging (§7)

- **`dc38b21` — C-M10: a coverage floor for title-level delivery, and the evidence recorded.**
  `--mark-delivery` propagates a delivery to **every clip in a book**, and unanimity was the
  whole test — any sample, any size, any distribution. Both real samples are degenerate:
  librivox-v1's 12 clips are one contiguous run in **section 2 of 15**, librivox-v2's 30 one
  run in **section 1 of 25**. Safe only because the one marked title is homogeneous by
  construction. Floor is 8 clips / ≥3 sections / ≥25% spread (verified: 7% and 4% refused, an
  honest spread passes). **The coverage is now written into the ledger beside the mark** —
  clips heard, sections heard, spread, override flag — which is the first thing an audit of
  that decision needs and was recorded nowhere. `--thin-coverage` allows a genuine collection
  and stamps `thin_override`.
- **`dc38b21` — C-M6/D-M5: one `ratings_transaction`.** Six scripts each grew an mtime stamp
  after an owner-set accent value was lost on 2026-07-26, in four flavours, and `tag_spike`
  grew **none** — appending to the live file behind a copy-backup, which is not safe just
  because it is an append (the app rewrites the whole file to commit an edit). The shared
  transaction takes **both** guards because they cover different writers: an **flock**
  serialises our own scripts, which mtime cannot do, and an **mtime re-check inside the
  lock** catches the app, which takes no lock. `tag_spike` converted (D-M5 closed); the six
  stampers still carry their own flavour and are listed in todo.

### Fixed — label derivation (§8)

- **`90f2c1b` — imputed heads, unretried failures, and a z-guard that fabricated labels.**
  - **D-M1** — `mine_emilia_keeps` imputed a missing EIV head as raw `0.0`, which z-scores
    to ~**3σ of pure fiction** weighted into every clip's valence. Safe only because the
    Emilia and LibriTTS passes happen to share 12 heads; the scorer's *default* set would
    have poisoned the corpus silently. Refused now, at the anchor and per tar. Only
    **weighted** heads are required — three carry weight 0.0 and two are legitimately
    absent, so demanding all twelve would fail on correct data.
  - **D-M6** — `process_emilia_tail` counted `error` rows as done, so a transiently failed
    clip was skipped by every later run forever. Error rows stay as the record; they no
    longer count as complete.
  - **D-M3** — the corpus lane refuses transcripts containing digits (the tokenizer deletes
    rather than expands them, and `validate()` can't see it). **Verified inert** for
    existing data: 0 digits in 8,000 sampled LibriTTS transcripts and 0 in all 5,736
    dev-clean. Fires on Emilia YODAS captions — the corpus Phase 1 merges.
  - **D-L5** — `derive_markup_measures` was pinned to v2 and stored the derive's
    **contiguous index** under `"speaker"`; the index is re-derived per corpus version, so
    a join against `reader_profiles.json` would match the wrong voice without erroring. Now
    `--corpus` (default v3c), with `speaker_index` and the real LibriTTS `speaker` recorded
    separately and spot-checked against wav paths.
  - **D-L2 — filed as "the two implementations disagree", and that undersells it.**
    `std or 1.0` guards only a std of *exactly* zero, which is not the case that occurs. A
    head can be **constant** for a speaker — one identical value across all 106 of speaker
    `6531`'s clips on `Amusement`, likewise `8095`/Valence and `909`/Valence, **224 clips**
    — leaving `v − mean` and `std` both float dust ~1e-21. Dust ÷ dust = **max|z| = 1.0**,
    a full-scale label manufactured from rounding error. `+ 1e-6` gives ~0, the truth.
    `eiv_merge_corpus` was the broken half and is corrected. **The shipped `_v2` files and
    therefore v3c were built with the broken guard** — re-deriving moves the raw combo by
    up to **0.228** across 31,443 of 31,445 clips, so that is a corpus bump and an **owner
    call**; nothing generated was touched.

### Changed — execution layout

- **`13b5872` — the LiteRT harness executes from the repo; `/data` holds the data.** Owner
  principle 2026-08-06: *code executes from the repo checkout, `/data` is used for what its
  name implies.* The harness ran on the **host** (no container — so "bind-mount the repo"
  was never the mechanism) and every script derived its paths from `HERE = dirname(
  abspath(__file__))`, so checkpoints read and ~400 MB of `.tflite`/`.wav`/`artifacts*/`
  written all landed beside the source. That, and only that, is why a second copy existed
  on `/data` — and why it drifted. **`SONORA_LITERT_WORK`** now names the data root
  (defaulting to `HERE`, so no existing invocation changes), and
  `scripts/litert_export/run.sh` runs a script from the repo with the work dir, model repo
  and harness interpreter wired up. The 11 `.py` copies and README on `/data` are retired to
  `_retired_code_copies/`; the venv, checkpoints and graphs stay. `test_data_mirrors.py` is
  **inverted** for this one — checked for *absence*, not agreement — and verified by copying
  a file back and watching it fail. AGENTS.md §6 restated from "deploy from the repo copy" to
  the principle, naming the artifact-root pattern as the way to satisfy both halves. The
  three remaining copies (audition's read-only mount, the dashboard's Caddy root, the
  training deploy clone) are left deliberately, with reasons, in `notes/data-mirrors.md`.

### Fixed — export lane and `/data` drift

- **`fde7a09` — `tests/test_data_mirrors.py`, and the audit behind it.** The premise that
  the LiteRT toolchain lived only on `/data` was **wrong** — it was migrated into
  `scripts/litert_export/` on 2026-07-22, and all four bodies of our code on `/data` (export
  harness, teacher-audition renderers, audition app, dashboard) are in GitHub-backed repos.
  The real defect is the opposite shape: `/data` holds working **copies**, and two had
  drifted. `convert_vat.py` there was three weeks stale, **missing the `detect_vat_dim` seam
  guard the repo recorded as landed**, and the README still documented a bare-`pip` install.
  Backup was never the risk; divergence was, and nothing detected it — both directories look
  healthy and the only symptom is that a fix you believe shipped did not. The gate compares
  every tracked file against its `/data` counterpart, asserts a minimum pair count so a
  layout change can't turn it into a silent pass, and was proven by introducing a one-line
  drift. Inventory and the standing rule (**repo authoritative, `/data` a working copy**):
  `notes/data-mirrors.md`.
- **`fde7a09` — F-H1: `convert_vat.py` resolved the wrong `matcha`.** `SONORA_REPO`
  defaulted to `…/Artificial-Humanity/Sonora`, which stopped being a repo when the layout
  was flattened on 2026-07-22 — verified to contain no `matcha/` today. So `sys.path.insert`
  added an empty directory, the import fell through to whatever `matcha` the harness venv
  held (the stock PyPI package, per its own docs), and the converter would export
  **upstream's architecture while every log line said Sonora** — a wrong export that
  converts cleanly and whose graphs run. Now derived from `__file__` and **refuses to start**
  if `matcha/models/matcha_tts.py` is absent. Also repairs `VAL_FILELIST`, which resolved
  against the same dead path.

### Fixed — acquisition lane (§5)

- **`cf6429d` — the book/real-audio lane's silent-corruption bugs.** None of these raised;
  each produced a plausible corpus instead of an error.
  - **A-H4** — the resolved LibriVox project was never compared to the requested URL.
    Matching is word-based with a prefix fallback, and `key` comes from the *request*, so
    a near-miss wrote the wrong book's audio into the right-looking directory. Compared
    and refused now.
  - **A-H2** — `decode("utf-8", errors="replace")` on Gutenberg text turned every
    non-ASCII byte of an ISO-8859-1 edition into U+FFFD while the ASCII stayed clean.
    Strict UTF-8 → cp1252 → latin-1, then a check that no replacement character survived.
  - **A-H3** — `books_ledger.json` / `staging_log.json` were read at startup and written
    back whole after minutes of network work, so overlapping runs erased each other. New
    `synth_common.update_json` re-reads under an flock and renames into place.
  - **A-H5** — the real-audio lane split on any `[.!?]`+space, shipping "Mr." and "Smith
    went home." as two clips. pysbd now, shared with `book_ingest`, plus an
    `is_complete_utterance` gate. **The gate alone could never have fixed this** —
    `is_complete_utterance("Mr.")` is `True` by shape — and there is a test asserting that
    so pysbd isn't later dropped as redundant.
  - **A-M7** — dialogue extraction hardcoded curly double quotes; since the caller falls
    back to the raw paragraph when no utterances are found, a differently-quoted edition
    filed its dialogue as **narration**. Convention is detected per edition now.
    *Correction to the review's record:* *Uneasy Money* uses **straight** singles (U+0027,
    the apostrophe character), not curly. Measured on the real text: **0 → 1,165**
    utterances.
  - **A-M6** — banks claimed `Standard Ebooks CC0` regardless of source while the router
    has been feeding Gutenberg down this path all along. False licence metadata in every
    derived clip's paper trail; provenance follows the source now, and a local `--epub` is
    `UNKNOWN` rather than assumed.
  - **A-M12** — "complete download" meant `size > 1024`, which a 404 page clears; it was
    saved as `.mp3` and every resume skipped it. Magic-byte check on both paths.
- **`33dfb33` — A-H1: `chapter_slice` → `chapter_slices`, ordered candidates.** The
  heading fast path matched three non-headings, and there was no second attempt when it
  was wrong, so a misled slicer lost **every** clip in a book while each decision looked
  defensible in the log. Reproduced against the original: a ToC yields 6 headings for a
  3-chapter book and slice 1 is literally `'\nCHAPTER I\n'`; Gutenberg's ~70-column wrap
  makes "…in the last\nchapter I was quite unwilling…" match mid-paragraph; and 40 chapters
  in 3 audio files satisfies `len(heads) >= n_sections`, handing section 1 **2.5%** of what
  it reads. Filters now require a short line and a substantial body, and headings are used
  only when they map ~1:1 onto sections. The caller walks candidates (headings → duration →
  duration-wide → whole text) until coverage clears 60%, and names the winner — **nearly
  free, because ASR depends only on the audio**, so a retry is one `difflib` pass and no
  GPU. Recall deliberately not widened: *Uneasy Money* heads chapters with a bare numeral,
  unmatchable without also matching page numbers.
- **`cf6429d` — two duplications collapsed**, both the kind the review keeps finding:
  `qc_passages` carried its own `is_complete_utterance` marked "kept in sync" (already
  drifted — it accepted guillemets and trailing spaces `book_ingest` rejected), and
  `_QUOTED_SPAN` handled straight quotes while the extractor beside it did not.

### Changed

- **`428cb55` — `pyproject.toml` is the single source of dependency truth; `requirements.txt`
  deleted.** It was the upstream list and had drifted **both ways**: seven declared packages
  nothing imports (`torchvision`, `torchmetrics`, `tensorboard`, `pandas`, `notebook`,
  `ipywidgets`, `seaborn`) and three imported at module scope but never declared
  (`soundfile`, `tqdm`, `PyYAML`), resolving by transitive luck through librosa. The
  replacement was derived by walking every import, not by editing the old list. Distribution
  renamed **`matcha-tts` → `sonora`** with an Apache-2.0 licence field; the import package
  stays `matcha`. `make create-package` deleted — it ran `twine upload` against metadata
  still claiming `name="matcha-tts"`, so one invocation would have published this fork to
  PyPI under upstream's name.
- **`428cb55` — GPL `phonemizer` is opt-in (`[espeak]`), and verified absent from a fresh
  training container.** README § 3 promises espeak is banned from the runtime path and the
  licence wall enforces that for *data*; nothing enforced it for *dependencies*.
  `cleaners.py` already imported it lazily, so only legacy LJSpeech checkpoints ever needed
  it. `gdown`/`wget` got the same treatment (`[download]`) — two network packages were
  mandatory in every container for a code path none of them take.
- **`428cb55` — `matcha/cli.py` works for Sonora checkpoints (E-H2).** It sent every
  checkpoint through espeak cleaners and wrote every file at 22050 Hz — so a Sonora
  checkpoint got phonemes it never trained on, in a 24 kHz waveform tagged 22.05 kHz
  (~9% slow, which sounds like a sluggish model rather than a header bug). The download
  guard read `not hasattr(args, "checkpoint_path") and args.checkpoint_path is None`, which
  argparse makes permanently False, so `--checkpoint_path` still fetched the upstream
  checkpoint over the network before discarding it — a hard failure offline. Lane is now
  read off the checkpoint; `--vat`/`--guidance` added and bounded; `--spk` range-checked.
  **Lane detection, op_g2p encoding and the 24 kHz vocoder loader now live once, in
  `matcha.cli`**, with `vocalizer.py` importing them — it had its own copy of all three, and
  its `SONORA_VOC24K_CONFIG` default pointed at a different (byte-identical, checked) file.
- **`428cb55` — the Vocalizer HTTP API enforces the control contract (E-M2).** It passed
  `valence`/`energy`/`tension`/`guidance` straight into the model unbounded while the UI
  could not. V/A/T are per-speaker z-scores clamped at 2σ in derivation, so `valence=50`
  does not make more emotion — it drives the FiLM trunk off the manifold and still renders
  fluent audio. Now one `CONTROL_BOUNDS` table, and **400 rather than 500**.
- **`428cb55` — `matcha/app.py` is no longer an entry point and no longer shares publicly
  (E-L4).** Its `launch(share=True)` opened a public tunnel from the machine holding
  unreleased checkpoints; now opt-in via `MATCHA_APP_SHARE=1`.

### Added

- **`428cb55` — `environments/`** replaces the 3-line `uv.lock` stub (T4/G-1): real
  `uv pip freeze` records of the two lanes that do the work, the container one produced by
  running the compose prep chain verbatim. A resolver lockfile would have been *actively
  misleading* — torch ships in the ROCm base image and is not installable from PyPI, so
  locking `torch>=2.0.0` pins a CUDA wheel and describes an environment we have never
  trained in. Named `environments/` because `.gitignore` line 41 ignores `env/` as a
  virtualenv, which would have committed the files in name only.
- **`428cb55` — `tests/test_cli_lanes.py`** (15 cases) covering every guard above, plus
  partial **D-M3**: the tokenizer *deletes* digits rather than expanding them ("I have 3
  cats" → `ˈaɪ hˈæv kˈæts`, verified live) and `g2p.validate()` cannot catch it because
  nothing illegal is present — a word is simply gone. Refused at synthesis input; D-M3
  stays open for the corpus lane.

- **`f015c8e` — `scripts/score_holdout.py` + `score_holdout.sh`, the never-trained holdout
  (Phase 0a).** Scores a checkpoint teacher-forced per clip with **paired** noise draws, so
  two checkpoints see identical timesteps and noise on identical clips; clip-to-clip
  variance dwarfs the effect being measured, and an unpaired comparison of two 5,463-clip
  means reads as noise. Runs in a throwaway ROCm container as `ai-mgr`, building
  `monotonic_align` in a `/tmp` copy so the owner's checkout is neither written to nor
  littered with build artifacts. **No `phonemizer` in the eval deps** — `matcha.text.cleaners`
  imports it lazily, the filelists are already IPA, so pulling GPL in to satisfy an import
  that never fires would be the G-3/G-4 mistake voluntarily.
- **`f015c8e` — `configs/data_licenses.yaml` declares `libritts_r_holdout_devclean`.**
  Without it `enforce()` refuses to load the filelist — the same trap that made v3/v3b/v3c
  structurally unrunnable. Declaring it is **not** permission to train on it, and the wall
  will not stop such a run; the guards are `--assert-disjoint-from` (any shared clip
  basename stops the run) and the deletion of the derive's `train_op.txt`/`val_op.txt` after
  concatenation into `holdout.txt`, so no file in the directory carries a name a training
  config would accept.

### Changed

- **`f015c8e` — `vat3c` ep099 is retired; `vat3-24k` ep099 is the base.** The holdout says
  the v3c fine-tune was a **regression, not a no-op**: +0.0164 against its own warm start
  over 5,463 unseen clips, all three loss terms worse, better on only 39.1% of clips, and
  +0.0443 worse on v3c's *own* val split. Not a normalisation artifact (+0.0172 under the
  v2 constants it trained with). The ear had already said "no audible change"; this
  sharpens the direction to *down*. **Phase 0b — the clean-lineage retrain from
  `matcha_vctk` — is not indicated**, because the v2 fine-tune shows a real gain on unseen
  audio (`diff` −0.0241, better on 78.7% of clips) and a compromised lineage could not
  produce that. Owner's call to ratify. Recorded in `quality-gap-plan.md` § 0a,
  `STATE.md` and `training-sources.md`.

- **`03fed75` — `vat3c_finetune` retired at every place it could be picked up.** A
  `RETIRED` banner on the experiment config (kept, not deleted — Phase 1 derives from it
  and the measurement must stay reproducible), `RETIRED.md` in the experiment and both
  checkpoint-bearing run dirs, `warmstart/vat3_ep099.ckpt` naming the correct base, and
  `resume.ckpt` re-pointed off the retired checkpoint it had been left aimed at. The
  launcher rewrite is **AI-Lab-AMD `a155eb6`**: it carried *two* independent hardcodings
  of vat3c — the experiment name and the resume glob — so queueing a successor run by
  editing the obvious one would silently have warm-started it from the retired ep099.
  `SONORA_EXPERIMENT` is now required (unset ⇒ the container idles with a message rather
  than exiting, since `restart: unless-stopped` turns a fast exit into a crash loop),
  auto-resume is scoped to that experiment's own run dirs, and `vat3c_finetune` is
  refused by name. New launch contract: `notes/training-operations.md`.

### Fixed

- **`e5e5ed1` — E-M5: the logged diffusion loss is masked.** `BASECFM.compute_loss` summed
  its residual over the full padded tensor. The decoder masks its own output, but the
  target residual `u = x1 - (1-σ)z` does not (`z` is drawn `randn_like` across the padding),
  so each batch's logged `diff_loss` carried a floor proportional to its padding fraction.
  Train batches have been length-bucketed since 2026-08-01 and val batches are not, so the
  floor fell on val and presented as a 3.2× train/val gap — vat3c epoch 1 logged diff 0.646
  train vs 2.069 val while `dur_loss` and `prior_loss` matched to three decimals.
  Gradients are unchanged (the padded positions carried none); **only the logged value
  moves, and it moves down.** `diff_loss` and `loss/train` / `loss/val` therefore do not
  compare across this commit — a second scale break in the metric after the bucketing
  change. Documented at every site that teaches someone to read the curve.
- **`e5e5ed1` — E-M6: the RoPE cache is built outside inference mode.**
  `on_validation_end` synthesises under `@torch.inference_mode()`, so a
  `RotaryPositionalEmbeddings` cache first built or grown during validation was made of
  inference tensors; the next training step with text no longer than the cached length
  reused them and killed the run with "Inference tensors cannot be saved for backward".
  The build now nests `torch.inference_mode(False)`, so the cache is an ordinary detached
  tensor regardless of which mode first demanded it.

### Added

- **`e5e5ed1` — `tests/test_training_seams.py`**, regression coverage for both of the
  above. Verified to fail against pre-fix source (4.643 vs 1.643 logged loss at 75%
  padding; the RoPE case reaching the real backward crash). The tests need the model's
  dependency stack, so they `importorskip` on the host venv and run in the ROCm training
  container:
  `docker compose run --rm --entrypoint pytest sonora_training tests/test_training_seams.py`

# Open items — post-review residue

Distilled 2026-08-02 from the full code review (`codereview.md`, HEAD `47a405d`) and its
remediation log (`cleanup-20260802.md`). Both source documents are deleted; their full text
is in git history. **Every Critical and High finding was fixed** (`b9c7a7e`..`1faac46`,
verified in source) — this file is only what remains open, letter-coded as the review
coded it so the git-history record stays joinable.

Items are grouped by when they bite, not by subsystem. **Landed items are deleted, not
struck** (git history is the archive, per [README.md](README.md)); delete sections when
empty. _Last pruned 2026-08-06 — the v3c config, the license-wall v3 entry, the
delivery-channel seam assertions (E-M3, E-M1/F-H2/T8) and the zonos tier bank all landed
and their records now live in [STATE.md](STATE.md) and
[training-operations.md](training-operations.md). Later that day the two §1 blockers
landed too: **E-M5** (the logged `diff_loss` is masked now, so the 3.2× train/val gap is
gone and curves are again readable — but not across the fix) and **E-M6** (the RoPE cache
is built outside inference mode). Both carry regression tests in
[tests/test_training_seams.py](../tests/test_training_seams.py). **Pruned again
2026-08-06 evening**: the split-contamination item is closed — Phase 0a shipped the
never-trained holdout, scored the whole lineage on it, and retired `vat3c` ep099 as a
regression; what survives of that bullet is the standing rule below. The loss-curve
warning moved to its permanent home in
[training-operations.md](training-operations.md)._

This file is **residue, not the plan.** What to do next, and in what order, is
[quality-gap-plan.md](quality-gap-plan.md).

---

## 1 · Before the next training run

- [ ] **The delivery channel is designed, corpus-complete, and unbuilt.** `vat_dim` is
      still **3** everywhere; Contract v2 ([ARCHITECTURE.md](ARCHITECTURE.md) §1) makes
      delivery the 4th FiLM channel and delivery-v1 closed at 1,189 labelled keeps, but
      no model code implements it. The pre-work is done and this is the migration
      itself: bump `model.vat_dim` to 4 (the data configs interpolate
      `vat_dim: ${model.vat_dim}`, so the model config is the single source of truth),
      embed the 5 lanes + `unknown ≡ zero` host-side onto the same zero-init FiLM path,
      and re-run `scripts/test_vat_dim_seams.py` (13 checks) at every step — it is
      pre-placed precisely so a wrong width fails loudly at the filelist instead of
      quietly at the trunk. There is still **no delivery export story** (F-H2, §2).
- [ ] **Score it on the holdout, not on `loss/val_epoch`.** Standing rule now that
      Phase 0a has landed, listed here because it is the step most easily skipped:
      `scripts/score_holdout.sh` against `data/libritts_r_holdout_devclean`, ~100 min for
      eight checkpoints on an idle card. MLflow val is permanently unusable for cross-run
      comparison (historical split contamination, 93–97% of v3c's val trained on earlier)
      and there are now **two** scale breaks in logged `diff_loss` — 2026-08-01 bucketing
      and 2026-08-06 masking. Deciding a run by its curve is deciding it by an artefact.

## 2 · Export lane (before any vat3/delivery export)

_**F-C1 and F-H1 closed 2026-08-06.** The gate suite is a ledger now and
`gates_or_die()` refuses to write artifacts unless every gate passes — it printed PASS/FAIL
and exited 0 either way, leaving a complete shippable artifact set behind a failed export.
**G5 drives valence / energy / tension independently and requires each to move the
waveform**; a real conversion run measured 7.0e-2 / 9.2e-2 / 8.6e-2 mean |delta| vs neutral,
**the first time V and T have been nonzero through a converted graph**. Also fixed: G4's
monotonicity check used `all()` over a possibly empty sequence (PASS on zero evidence), and
**F-M4** — `config.json` wrote `time_embed_dim=1024` while the graphs consume `in_channels`
(224), a number the mobile host trusts to size its buffer. 10/10 gates pass.

**F-H1**, and it was live: the default `SONORA_REPO` pointed at
`…/Artificial-Humanity/Sonora`, which stopped being a repo when the layout was flattened
on 2026-07-22 — verified to contain no `matcha/`, so the import fell through to whatever
`matcha` the harness venv held and the converter would export UPSTREAM's architecture
while every log line said Sonora. It now derives the root from the file's own location and
**refuses to start** if `matcha/models/matcha_tts.py` is not there. The bundled correction:
these scripts are NOT outside source control — they live in `scripts/litert_export/` and
have since 2026-07-22. `/data/toolchain/litert-conversion/` is a working copy, it had
**drifted**, and `tests/test_data_mirrors.py` now fails on any divergence.
See [data-mirrors.md](data-mirrors.md)._

- [ ] **F-H2:** no delivery export story; nothing records or enforces the 2σ clamp
      contract for mobile hosts.
- [ ] **F-M1..M3, M5..M7:** referee binds inputs by dtype heuristic and can't score conditioned
      graphs; `rename_tflite_tensors` maps outputs by emission order with no shape check
      ("renamed 0 tensors" is success); `kotlin_replica` is unrunnable (depends on
      artifacts nothing produces) and has no conditioning coverage; waveform
      parity is Pearson-only (a systematic fp16 gain error scores 1.0 — no RMSE check
      anywhere, for a model whose flagship axis is energy); CFG guidance is inexportable
      and undocumented for mobile.

## 3 · Environment reproducibility (T4 / G-1)

_Mostly closed 2026-08-06. `pyproject.toml` is the single source of dependency truth
(`requirements.txt` deleted — it declared 7 packages nothing imports and omitted 3 that
are imported at module scope); GPL `phonemizer` is an opt-in `espeak` extra and is absent
from the training container; the `uv.lock` stub was replaced by real environment records
in [`environments/`](../environments/README.md); `create-package` (which would have
twine-uploaded this fork as `matcha-tts`) is gone, as are the bare `python` invocations
and the `matcha-tts-app` entry point. E-H2 and E-M2 are fixed with regression coverage in
[tests/test_cli_lanes.py](../tests/test_cli_lanes.py). What is left:_

- [ ] Pin `rocm/pytorch` image digests + per-engine wheels in `synth_bank.sh`,
      `librivox_align.sh`, `eiv_score.sh` (only qwen pins today); drop a `pip freeze`
      into every campaign dir. Raw scores are "immutable" but their environment isn't
      recorded — version-stamp EIV scores (D-M2). *(The repo-level half of this landed
      2026-08-06 — see [`environments/`](../environments/README.md) — but the render lane
      still pins nothing per campaign, which is where the reproducibility question
      actually bites.)*
- [ ] **`matcha/app.py` still phonemizes through `english_cleaners2` unconditionally.**
      Harmless in practice — it is the upstream demo, no longer an entry point, has no
      `make` target and needs the `espeak` extra to run at all — but it is the last place
      a Sonora checkpoint can be fed espeak phonemes. Point it at
      `cli.process_text_for_lane` or delete the file; deleting is probably right, since
      `vocalizer.py` supersedes it and the legacy-LJSpeech lane is auditable there.

## 4 · Ears queue (priority order, from the cleanup round)

1. [ ] **Qwen vs VibeVoice at equal loudness** — the one teacher-portfolio ranking the
       loudness confound (5.99 dB spread, Qwen +3.81 dB over VV) could have produced.
       The 56 keeps are normalized; a keep-*rate* re-test additionally needs the 24
       quarantined drops normalized (un-quarantine is an owner call).
2. [ ] **Orpheus tier** — `normal` is explicit in `ENGINE_TIER` now; `trusted` is
       arguable on 105 non-`tara` renders at 80.0% keep / mean 4.49. Policy, not
       measurement.
3. [ ] **moss_vg** — newscaster 20/20 is confounded (the register flatters its
       radio-timbre/IVR failure mode; its 24% rate was measured on expressive material).
       Re-test on dialogue or an expressive register before any tier movement.
4. [ ] **Ear-calibrate `tail_ok`** (threshold never calibrated; several catches could be
       ASR dropping a quiet final phrase) **and the 1.4 s pause advisory band**.

## 5 · Acquisition / alignment lane

_Partly closed 2026-08-06 (`tests/test_acquisition_lane.py`, 31 cases). **A-H2** decodes
Latin-1 PG editions properly and refuses text that already carries U+FFFD; **A-H3** ledger
and staging-log writes go through `synth_common.update_json` — re-read under an flock,
renamed into place — so a stale snapshot can no longer erase a concurrent run; **A-H4**
compares the resolved LibriVox project against the requested URL and refuses a mismatch;
**A-H5** the real-audio lane now splits with pysbd and gates on `is_complete_utterance`;
**A-M6** provenance follows the actual source instead of claiming Standard Ebooks CC0 for
Gutenberg text; **A-M7** the quote convention is detected per edition; **A-M12** a
non-MP3 response is never written as audio, nor accepted on resume; **A-H1**
`chapter_slice` became `chapter_slices`, returning ORDERED CANDIDATES that the caller
tries until one anchors._

_A-H1 detail worth keeping: the retry is nearly free because ASR depends only on the
audio, so a second candidate costs one `difflib` pass and no GPU. The heading filters are
a short-line test and a minimum body length, which between them kill ToC entries,
hard-wrapped prose ("…the last\nchapter I was unwilling…") and stray front matter;
headings are used only when they map ~1:1 onto audio sections, since 40 chapters in 3
files satisfied the old `>=` test and handed section 1 about 2.5% of what it reads.
**Recall is deliberately not widened**: *Uneasy Money* heads chapters with a bare numeral
("1"), which no safe pattern can match without also matching page numbers — the
proportional split already carries that book at 94%._

_Worth knowing from the A-M7 fix: *Uneasy Money* quotes with **straight** single quotes
(U+0027, the apostrophe character), not the curly singles the review recorded. Under the
old hardcoded curly-double pattern the whole novel yielded **0** utterances; it now yields
1,165. And on A-H5: `is_complete_utterance("Mr.")` is **True** by shape, so the gate
provably cannot catch a bad split — fixing the splitter was the only fix, which is why
pysbd is now a hard dependency of the align container._

- [ ] **A-M2:** `refine()` counts non-blank *frames* as tokens — masked today (only
      first/last used), but span-FiLM would inherit garbage per-word spans. Use
      `torchaudio.functional.merge_tokens`.
- [ ] **A-M8:** PG epub boilerplate parses as prose and becomes render text. (The quote
      half of A-M7/M8 is done; this half is not.)
- [ ] **A-M9/M10:** the `lv:`/`pg:` ledger-key split — SKIP detection checks only `pg:`,
      re-routes and duplicates `lv:` entries, regresses status; fetch updates status only
      under exact `--key`; two on-disk naming schemes by invocation style.
- [ ] **A-M11:** no checkpointing across the director pass — a `load_skill` error at
      chunk 90 discards 90 Gemma calls. Write the bank incrementally.
- [ ] **A-M1/M13:** one zero playtime silently reverts ALL sections to even split
      (and `hh:mm:ss` playtimes make windows that don't tile); no retries in a bulk-transfer
      lane. *(The `.mp3`-that-is-really-HTML half, A-M12, landed.)*

## 6 · Synthesis campaign tooling

_Partly closed 2026-08-06. **B-M3** chatterbox/zonos/vibevoice now call
`synth_common.rebuild_used_set` — the variety-bias `used` set started empty on every resume
because an already-rendered job skips before `select_reference()` runs, so a resumed
campaign silently lost its diversity guarantee. **B-M5** `synth_common.attempt_seed`
perturbs the seed per attempt and records the one actually used; "re-run under
skip-if-exists" IS the retry, and re-seeding identically meant a deterministic failure could
never converge. **B-L5** `MAX_REF_EXCURSION` hoisted to `ref_select` beside `REF_BLACKLIST`
(it was written out three times)._

- [ ] **B-M2:** `$BANK`/`$OUT` interpolated unquoted through two quoting layers; no
      `/data/*`-prefix or no-space guard (the single-level-OUT refusal landed; this is
      the rest).
- [ ] **B-M9:** chatterbox `BRIGHT_REF_POLICY` is recorded in manifests but never
      consulted — delete the knob or wire it.
- [ ] **B-M6/M7:** `synth_dia`/`synth_moss85` still use the explicit-keys manifest style
      whose field-dropping was fixed in the other six; `make_bulk_bank.py` emits Dia
      lines with leading tags and engine-less ids. Both are SET_ASIDE-reinstatement
      traps — fix before any VV/Dia revival.
- [ ] **B-L2/L3/L4/L6/L7/L8/L10:** uncaught `select_reference` LookupError kills an
      engine's remaining bank (moss_vg's per-clip try/except is the pattern); qwen loads
      the model before the zero-jobs check; vibevoice forks the CLI contract and skips
      the blacklist/excursion guards; moss multi-message decode clobbers filenames;
      `make_v3d_bank.py` is a guardless fork of ref_select scoring; newscaster
      `used.add(rid)` is a no-op; `container_as_ai_mgr.sh` masks its own failures.

## 7 · QC / audit / staging

- [ ] **C-M1:** a clip QC-flagged *after* it was deferred is never promoted to the queue
      — needs a defer→unaudited flip for flagged ids.
- [ ] **C-M4 (half-open):** the dead-air gate shipped and is wired, but `qc_gate`'s
      speech-duration measure is still `librosa.effects.split(top_db=35)` (Silero was
      written to replace it) and the 4 s owner floor has no hard `speech_ok` gate.
      **head_ok** is also unbuilt — no gate sees head truncation; `tail_lost()` already
      computes `blocks[0].a` (head-loss word count) and discards it. Both thresholds
      need ear calibration like tail_ok.
- [ ] **C-M10 (found 2026-08-04):** `stage_pool --mark-delivery` has **no coverage
      requirement**. It accepts unanimity from any sample — any size, any distribution —
      so one clip, or 30 contiguous clips from section 1 of a 25-section novel, certifies
      a title-level delivery that then propagates to every clip in the book. Both real
      samples to date are exactly that shape: librivox-v1's 12 audited clips are one
      contiguous run in section 2, and librivox-v2's 30 are one run in section 1. Safe so
      far only because the one title actually marked (`pg:824`, *Speeches*) is homogeneous
      by construction. Needs a floor on section spread and clip count before the mark is
      offered on anything that is not a collection.
- [ ] **C-M5:** remaining non-atomic writes — `stage_pool` (ledger + staging_log),
      `reader_profile`, `publish_tier` (`.bak` only taken on the first run ever).
      tmp + `os.replace` everywhere.
- [ ] **C-M6:** the ratings.csv mtime guard exists in four flavors (widest window:
      `seed_delivery` checks before serializing ~1,500 rows and never re-checks). One
      shared implementation, ideally `flock` on a sidecar (the app takes no lock either).
- [ ] **D-M5:** `tag_spike.py` appends to the live ratings.csv with copy-backup but no
      mtime guard *(verified still open)*.
- [ ] **C-M7/M8:** reader_profile cross-title hint fills into a title whose own evidence
      is `_CONFLICT`; two definitions of "ear-confirmed (reader,title)" — `--seed-ear`
      skips pairs the auditor still force-queues.
- [ ] **C-M9:** `qc_artifacts` cohort stats include `_dropped`/`_superseded` wavs and
      the non-recursive glob mis-attributes subdir wavs to a `"?"` cohort.
- [ ] **C-L4/L5/L6:** loudnorm sidecar keyed on path (a rerolled-in-place wav ships
      unnormalized); `publish_tier` has no `librivox` ENGINE_POLICY entry (hard-errors
      the day staged real-audio clips reach a metadata.jsonl); the trusted-tier 3%
      sample rounds to zero under ~17 clips.

## 8 · Label derivation

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

## 9 · Embodiment — approved, sequenced after the clip-level channel

Owner approved the position 2026-08-02; the reasoning is canon in
[ARCHITECTURE.md § 1](ARCHITECTURE.md) and is not repeated here.

- [ ] **Embodiment bank across the current five survivor engines**, folded into a
      narration round rather than run as its own campaign. It settles whether the
      mean of 3.71 is the MODE or was just the old engines — of the four engines in
      those 28 rows (qwen 8, longcat 8, libritts-r 10, scm-spike 2) only qwen is
      still in the portfolio, so we have almost no result rather than a bad one.
- [ ] **Keep embodiment clips delivery-blank** and outside the 50/30/8/6/6
      percentages. This is a standing rule, not a task — listed so a future
      balance pass does not "fix" the blanks.
- [ ] **Span-FiLM phase (later):** delivery varying across an utterance on the same
      zero-init path, expanded through the duration alignment. The 17 keeps are the
      pilot set. Prerequisite: the clip-level delivery channel ships first.

## 10 · Parked dataset decisions

SSOT is `training-sources.md` — not duplicated here. Headlines: the Expresso two-ruling
conflict (owner call), the other 90% of LibriTTS-R, Hi-Fi TTS parquet→wav conversion,
VCTK unzip, the Emilia-keeps merge (v1.1 lane), HiFiTTS-2's 2.8 TB fetch decision, and
retiring `librivox-v1` once `librivox-v2` is staged and its 12 ear verdicts re-earned.

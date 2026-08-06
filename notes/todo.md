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
[tests/test_training_seams.py](../tests/test_training_seams.py)._

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
- [ ] **The loss curves before 2026-08-06 are a different measurement.** E-M5 landed:
      `compute_loss` masks the residual, so `diff_loss` no longer carries the
      padding-fraction floor that made val read 3.2× train. Both terms drop, train more
      than val, and **nothing logged before the fix is comparable to anything after it** —
      the 2026-08-01 bucketing boundary is now the second-oldest of two discontinuities in
      the same metric. Listed here, not in the changelog only, because the next person to
      open MLflow will otherwise read the step change as a training effect.
- [ ] **Split contamination is historical and does not fix itself.** The hash split
      (`derive_vat_corpus._in_val`, 2026-08-02) is sound going forward, but every
      retained checkpoint warm-started through corpora whose splits were re-drawn, so
      93–97% of v3c's val clips were trained on earlier. `loss/val_epoch` is not a
      generalization measure for any of them. Fix = the never-trained LibriTTS-R
      `dev-clean` holdout, no training required —
      [quality-gap-plan.md § Phase 0a](quality-gap-plan.md).

## 2 · Export lane (before any vat3/delivery export)

- [ ] **F-C1 (the big one):** the whole LiteRT gate suite prints PASS/FAIL and exits 0 —
      and **valence/tension have never once been driven nonzero through a converted
      graph** (G2 runs `vat = zeros`; G3/G4 drive `[0, a, 0]`). Needs enforced thresholds
      + exit codes + a per-channel differential probe (drive each channel independently,
      assert differential output).
- [ ] **F-H1:** `convert_vat.py` `SONORA_REPO` default is the pre-flat `…/Sonora` — the
      sys.path insert falls through to whatever `matcha` the harness venv holds
      (documented as the stock pip package). Default to `…/Sonora/github`. Lives in
      `/data/toolchain/litert-conversion/`, not this repo.
- [ ] **F-H2:** no delivery export story; nothing records or enforces the 2σ clamp
      contract for mobile hosts.
- [ ] **F-M1..M7:** referee binds inputs by dtype heuristic and can't score conditioned
      graphs; `rename_tflite_tensors` maps outputs by emission order with no shape check
      ("renamed 0 tensors" is success); `kotlin_replica` is unrunnable (depends on
      artifacts nothing produces) and has no conditioning coverage; `config.json` writes
      `time_embed_dim=1024` while the masked graphs consume `in_channels`; waveform
      parity is Pearson-only (a systematic fp16 gain error scores 1.0 — no RMSE check
      anywhere, for a model whose flagship axis is energy); CFG guidance is inexportable
      and undocumented for mobile.

## 3 · Environment reproducibility (T4 / G-1)

- [ ] `uv.lock` is a 3-line stub; the venv that produced v3 and every render is recorded
      nowhere. Commit a real lock/constraints file; prune `requirements.txt` (still the
      stale upstream list naming GPL `phonemizer`).
- [ ] Pin `rocm/pytorch` image digests + per-engine wheels in `synth_bank.sh`,
      `librivox_align.sh`, `eiv_score.sh` (only qwen pins today); drop a `pip freeze`
      into every campaign dir. Raw scores are "immutable" but their environment isn't
      recorded — version-stamp EIV scores (D-M2).
- [ ] **G-3/G-4/E-M7:** Makefile + setup.py are upstream residue (`create-package` would
      twine-upload as `matcha-tts`; bare `python` invocations; upstream metadata, no
      license field). `phonemizer` (GPL) is an unconditional requirements dep;
      `english_cleaners2` is still the default cli/app lane against the README's
      "espeak banned from the runtime path".
- [ ] **E-H2:** `matcha/cli.py` is broken for every Sonora checkpoint (espeak cleaners,
      no `--vat`/`--guidance`, forced download even with `--checkpoint_path`) — teach it
      the op-G2P lane or retire it in favor of `vocalizer.py`. **E-M2:** the Vocalizer
      HTTP API accepts unclamped VAT/guidance. **E-L4:** `matcha/app.py` (upstream demo,
      `share=True` public tunnel) is still a setup.py entry point.

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

- [ ] **A-H1:** `chapter_slice` heading fast path fires on ToC pages, hard-wrapped prose
      ("…the last\nchapter…"), and single-file multi-chapter books (~10% coverage → every
      clip dropped, exit 0). No duration-split fallback when the coverage gate fails.
- [ ] **A-H2:** Gutenberg fallback decodes `errors="replace"` — ISO-8859-1 editions bake
      U+FFFD into transcripts ("caf�") with no detector.
- [ ] **A-H3:** `books_ledger.json` read-modify-written across minutes-long network runs
      by fetch + router (lost updates), bare `write_text` (torn file). Same class:
      `staging_log.json` (truncation forgets staged ranges → double-staging).
- [ ] **A-H4:** resolved LibriVox project URL never compared to the requested `--url` —
      a fuzzy-title near-miss downloads the wrong book/reader undetectably. One-line fix.
- [ ] **A-H5:** naive regex sentence splitter ships incomplete utterances ("Mr. / Smith
      went home.") in violation of [[completeness-over-length]]; `book_ingest` has
      `is_complete_utterance()` (pysbd), the real-audio lane uses neither.
- [ ] **A-M2:** `refine()` counts non-blank *frames* as tokens — masked today (only
      first/last used), but span-FiLM would inherit garbage per-word spans. Use
      `torchaudio.functional.merge_tokens`.
- [ ] **A-M6:** `book_ingest` hardcodes provenance "Standard Ebooks CC0" even for PG
      sources — false license metadata in every derived clip's paper trail.
- [ ] **A-M7/M8:** dialogue extraction matches only curly double quotes (straight/single
      quoted editions → zero utterances, dialogue enters narration windows); PG epub
      boilerplate parses as prose and becomes render text.
      **CONFIRMED LIVE 2026-08-04, not latent:** the *Uneasy Money* edition (`pg:6684`)
      quotes with single quotes — 818 against 26 doubles — so a double-quote test reads
      1,355 of its 1,366 clips as narration when the true figure is 1,201. It was caught
      only because the miscount would have justified a delivery-homogeneous mark on a
      novel. Anything deciding a lane, a window or a mark from quote characters is
      unsafe until this is fixed.
- [ ] **A-M9/M10:** the `lv:`/`pg:` ledger-key split — SKIP detection checks only `pg:`,
      re-routes and duplicates `lv:` entries, regresses status; fetch updates status only
      under exact `--key`; two on-disk naming schemes by invocation style.
- [ ] **A-M11:** no checkpointing across the director pass — a `load_skill` error at
      chunk 90 discards 90 Gemma calls. Write the bank incrementally.
- [ ] **A-M1/M12/M13:** one zero playtime silently reverts ALL sections to even split
      (and `hh:mm:ss` playtimes make windows that don't tile); resume enshrines HTML
      error pages saved as `.mp3` (magic-byte check); no retries in a bulk-transfer lane.

## 6 · Synthesis campaign tooling

- [ ] **B-M2:** `$BANK`/`$OUT` interpolated unquoted through two quoting layers; no
      `/data/*`-prefix or no-space guard (the single-level-OUT refusal landed; this is
      the rest).
- [ ] **B-M3:** the `used` variety-bias set starts empty on every resume
      (chatterbox/zonos/vibevoice) — rerolled clips cast as if the pool were untouched.
      Rebuild `used` from manifest rows at startup.
- [ ] **B-M5:** moss_vg/orpheus retry loops re-seed identically before each attempt —
      deterministic failures can never converge. Perturb `seed + attempt`, record it.
- [ ] **B-M9:** chatterbox `BRIGHT_REF_POLICY` is recorded in manifests but never
      consulted — delete the knob or wire it.
- [ ] **B-L5:** `MAX_REF_EXCURSION = 240.0` still duplicated in chatterbox + zonos;
      hoist to `ref_select` beside `REF_BLACKLIST` *(verified still open 2026-08-02)*.
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

- [ ] **D-M1:** `mine_emilia_keeps` imputes missing EIV heads as raw `0.0`, which
      z-scores *nonzero*. Safe today (verified identical coverage); trips corpus-wide on
      a default 4-head Emilia pass. `eiv_merge_corpus` (SystemExit) is the pattern.
- [ ] **D-M3:** digits and symbols are silently deleted by the tokenizer ("I have 3
      cats" → audio says "three", text drops it). Latent for LibriTTS (0/5,000 sampled),
      live for Emilia YODAS captions. Needs a digit check or normalization pass.
- [ ] **D-M4:** homograph flattening — one pronunciation per key (`read`, `wind`,
      `live`, `dove`, `bass`); past-tense "read" trains against the wrong vowel. A
      regression vs the espeak lane; same class as the fixed contraction poison but
      needs context-aware G2P, a larger fix.
- [ ] **D-M6:** `process_emilia_tail` resume treats error rows as done — transiently
      failed clips are permanently, silently skipped.
- [ ] **D-L2:** small-speaker z degeneracy (17 speakers <10 clips; 2-clip speakers
      pinned at ±0.5; 7% of train V saturated) + the two z-implementations disagree on
      the guard (`std + 1e-6` vs `std or 1.0`).
- [ ] **D-L5:** `derive_markup_measures.py` frozen at v2 paths; stores the contiguous v2
      index under `"speaker"` — mis-keying trap for the span-markup spike.

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

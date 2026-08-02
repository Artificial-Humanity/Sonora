# Sonora full code review — 2026-08-02

**Scope:** every Python and shell script in the repo at HEAD `47a405d`, reviewed in six
subsystem passes (acquisition/alignment, synthesis campaign tooling, QC/audit/staging,
label derivation, model/training core, LiteRT export) plus a cross-cutting repo-hygiene
pass. Reviewed against the project's actual bar: **reliable, repeatable dataset
generation at growing scale**, with the standing rules (QC-gate-mandatory, ear-only
scores, trust tiers, completeness, 4 s floor, mtime-guarded `ratings.csv`) treated as
contracts the code must enforce, not conventions it may assume.

**Method note:** findings marked **[verified]** were reproduced directly against the
source or live data during this review (code read, command run, or corpus grepped).
The rest were reported with file:line evidence by a subsystem pass and spot-checked
for plausibility but not independently re-executed.

---

## Executive summary

The design layer of this codebase is unusually strong: constants carry their
measurement history, postmortems live next to the code they fixed, the VAT
conditioning core is carefully built (the zero-init FiLM identity was explicitly
defended against the kaiming re-init trap), and the trust-tier policy is correctly
encoded on the happy path. Nothing found suggests the trained checkpoints are wrong.

The operational layer has not caught up. The failures cluster into a small number of
systemic patterns rather than scattered one-off bugs, and several of them are already
in shipped data:

1. **Wrong phoneme labels are in v3 right now.** The espeak-free G2P lane mangles
   every contraction (no apostrophes in the dict, no `'` in the neural charset):
   `don't` → `dˈɔnt`, `we'll` → `wˈɛl` (identical to *well*). 545 `dˈɔnt` in
   `train_op.txt` alone, ~1.3 % of all tokens, concentrated in dialogue — the register
   the delivery campaign anchors on. Homographs are flattened too (`read`, `wind`,
   `live` each get one pronunciation). **[verified]** → D-C1, D-M4.
2. **Every clip from the book lane carries ~100 ms of the next sentence.** The
   next-sentence clamp in `librivox_align.py` is applied *before* the pad, which adds
   the overshoot back; the start edge has no clamp at all. Systematic audio/transcript
   mismatch, invisible to WER. **[verified]** → A-C1.
3. **The mandatory QC gate can pass while measuring nothing.** `qc_gate.py` exits 0
   with `0/0 hard-pass` when its manifest glob matches nothing, and `synth_bank.sh`
   derives the campaign dir with `dirname "$OUT"` — one trailing slash produces an
   empty "passing" gate and an ungated batch reaches the ear. **[verified]** → B-C1.
4. **Two measured engine defects are re-armed by the director pipeline.** The SSOT
   direction builder converts `emotion: null` into the exact neutral-dominant vector
   measured destabilizing 5/9 zonos narration groups, making the renderer's
   `unconditional_keys` fix unreachable; and the Orpheus fallback voice is `tara`,
   the one voice the skill file bans for room reverb. Mitigations that exist only in
   markdown are not mitigations. **[verified]** → B-H1, B-H2.
5. **The ear's SSOT is writable without a guard, and ear evidence is being
   fabricated.** `audit_sampler.py` appends to the live `ratings.csv` with no mtime
   guard (the highest-value clips can silently vanish), and `stage_pool.py` writes
   folded, unheard clips as `status=keep, score=5` — synthetic ear verdicts that
   three downstream consumers (gate calibration, novelty sampling, `--mark-delivery`)
   then trust. **[verified]** → C-C1, C-H3.
6. **Most of the project's gates are prints, not gates.** The entire LiteRT parity
   suite exits 0 regardless of outcome (and never drives the valence/tension channels
   through a converted graph at all); `test_film_export_gate.py` prints PASS/FAIL and
   exits 0; `derive_vat_corpus.py`'s independence gate and coverage checks are
   advisory; the eval harness's controllability gate passes **sign-inverted**
   channels via `abs(rho)`. **[verified]** → F-C1, E-H1, D-H2/M7.
7. **The environment is not reproducible.** `uv.lock` is a 3-line stub, the actual
   venv is recorded nowhere, `requirements.txt` is the stale upstream list (still
   naming GPL `phonemizer`), and all three container lanes run `rocm/pytorch:latest`
   plus unpinned pip installs — a mid-campaign upstream release changes numerics
   between halves of one bank, invisibly. **[verified]** → G-1, B-M4, A-M4, D-M2.

---

## Cross-cutting themes

These recur across subsystems; the per-subsystem sections give the instances.

### T1. Gates that report instead of refuse
A "gate" whose worst outcome is a printed FAIL is observability, not protection.
Instances: the whole LiteRT export suite (F-C1), `test_film_export_gate.py` (E-M4),
`test_vat_identity.py` (E-M4), `derive_vat_corpus.py` coverage + independence gates
(D-H2, D-M7), `qc_gate.py` zero-clip and zero-pass batches (B-C1, B-M8),
`librivox_align.py` zero-clip books exiting 0 (A-L2), `phonemize_filelist.py`
writing the output file even on FAIL (D-L7), eval harness `abs(rho)` (E-H1),
`rename_tflite_tensors.py` "renamed 0 tensors" success (F-M6). The fix is uniform:
thresholds + `sys.exit(1)`, and empty-input treated as failure, everywhere a
downstream step consumes the result.

### T2. Fabricated or laundered ear evidence
The v4 vocabulary says score = ear only. Violations: `stage_pool.py` fabricated
`score=5` on unheard folds (C-H3); `--mark-delivery`'s "heard" test counting
deferred/dropped/machine-filled rows (C-H4); `seed_delivery.py` defaulting unknown
registers to Dialogue (C-H5); `reader_profile.py --learn` counting its own previous
propagations as agreeing votes (C-H6); `gate_calibration.py` counting dropped-with-
score and fabricated rows as ear keeps (C-M10). These compound: a propagated value
can re-certify the mark that produced it. The shared fix is provenance — machine-
written cells must be distinguishable from ear-written cells.

### T3. Non-atomic writes and races on live state
`books_ledger.json` is read-modify-written across minutes-long network windows by
`librivox_fetch` and `book_router` (lost updates), and written with bare
`write_text` by both plus `stage_pool` (torn file on crash) (A-H3, C-M5).
`librivox_align`'s manifest opens in append mode with no dedup — every re-run
duplicates rows **[verified]** (A-C2). All eight renderers write wavs non-atomically
with existence-keyed skip: a truncated wav from a crash is permanently skipped and
silently vanishes from the campaign (B-H4). `audit_sampler` appends to `ratings.csv`
with no guard at all (C-C1); the four guarded writers implement the guard four
slightly different ways (C-M6). Standard fixes: tmp + `os.replace` everywhere,
rebuild resume state from manifests, one shared guard implementation.

### T4. Unpinned environments
`rocm/pytorch:latest` + unpinned pip in `synth_bank.sh`, `librivox_align.sh`, and
`eiv_score.sh`; no lockfile for the host venv; per-engine deps unpinned except qwen
(the one place pins were added proves the team knows). For a lane whose raw scores
are "immutable" and whose renders are compared across days, environment identity is
part of the data's provenance. Pin image digests, pin wheels, drop a `pip freeze`
into every campaign dir.

### T5. Copy-paste drift
Eight renderers share a hand-copied harness; the explicit-keys manifest bug was
found and fixed in six of them and still lives in `synth_dia.py`/`synth_moss85.py`
(B-M6). `MAX_REF_EXCURSION` is duplicated in two renderers instead of living beside
the shared blacklist in `ref_select` (B-L5). Three Ollama chat helpers, three
validation regimes. `make_v3d_bank.py` is a frozen fork of `ref_select`'s scoring
with none of its guards (B-L7). A small `synth_common.py` (harness loop, atomic wav
write, manifest transaction, used-set rebuild) removes roughly half the synthesis
findings in one move.

### T6. Silent fallbacks that mislabel
When a join or lookup fails, the code picks a *plausible* value instead of failing:
unknown ids → `("?","?")` → the folding `normal` tier (C-H2); unknown register →
`Dialogue` (C-H5); missing emotion → neutral-dominant (B-H1); invalid voice → `tara`
(B-H2); uncovered valence → `0.0`, which reads as "at speaker mean" (D-H2); missing
EIV heads → raw `0.0`, which z-scores nonzero (D-M1); unknown eval channel →
"seconds" (E-L8). In a training-data pipeline the plausible-wrong default is the
worst choice; unknown must be loud, or must fall to the *conservative* side
(scrutinized tier, None, abort).

### T7. Run-mode and layout inconsistency
`uv run` still invoked on the host by `synth_bank.sh` (loudnorm, qc_gate, register)
and `librivox_align.sh` (qc_gate), and prescribed by several docstrings, against the
2026-08-01 owner rule (`.venv/bin/python`). Stale pre-flat-layout default in
`convert_vat.py` (`SONORA_REPO` → `…/Sonora`, which no longer contains the repo)
**[verified]** (F-H2). The lv:/pg: ledger-key split's infection surface is mapped in
A-M9/M10.

### T8. The delivery channel exists in scripts but nowhere else
Bank builders and notes assert delivery as the 4th FiLM channel; the model core is
`vat_dim=3` everywhere, the datamodule parses exactly `v,a,t`, the export lane
hardcodes `VAT_DIM=3`, and the export gate tests a synthetic 3-channel stand-in
(E-M1, F-H3). None of this fails loudly when delivery lands — the collate's
all-or-none `torch.stack` will instead silently drop conditioning for mixed batches.
Before the delivery migration, these seams need assertions.

---

## A. Acquisition & alignment
(`librivox_fetch.py`, `librivox_align.py`, `librivox_align.sh`, `book_ingest.py`,
`book_router.py`)

### Critical
- **A-C1 [verified]** `librivox_align.py:412-418` — clip-boundary clamp is undone by
  the pad applied after it. `t1 = min(t1, nxt_start - 0.02)` then
  `t1 = min(dur, t1 + PAD_SECONDS)` with `PAD_SECONDS = 0.12` puts the end 0.10 s
  *inside* the next sentence; `t0 = max(0.0, t0 - PAD_SECONDS)` has no clamp against
  the previous sentence at all. Every clip with an adjacent sentence (i.e. nearly
  all) carries audio its transcript lacks, at both edges, invisible to WER. The
  comment above the clamp shows it was added to fix exactly this; the fix is
  order-of-operations broken. Affects the 79 shipped Dickens clips and all future
  book-lane output. Fix: apply the pad first, clamp last (and add a previous-sentence
  clamp for `t0`).
- **A-C2 [verified]** `librivox_align.py:288` — manifest opened `"a"` with no
  dedup/resume logic. Wav names are deterministic, so a re-run after a mid-book crash
  overwrites wavs but *appends* duplicate manifest rows — potentially with different
  `t0/t1/text` if args changed; only the last matches the wav on disk. `qc_gate`
  already had to build last-record-wins dedup to survive this (its comment records
  416 records over 301 wavs); every other jsonl consumer is still exposed. Fix:
  last-record-wins on read is a band-aid; write a per-section tmp manifest and merge,
  or dedup-by-id before append.

### High
- **A-H1** `librivox_align.py:108-116` — `chapter_slice`'s heading fast path fires on
  ToC lines (PG contents pages match the regex per-line), hard-wrapped prose
  ("…in the last\nchapter I described…" — `[IVXLC]+` matches the pronoun *I*), and
  single-file recordings of multi-chapter books (`len(heads) >= n_sections` is
  trivially true at `n_sections=1`, slicing chapter 1 only → ~10 % coverage → every
  clip dropped, exit 0). No fallback retry with the duration split when the coverage
  gate fails. Silent yield loss at scale.
- **A-H2** `librivox_fetch.py:196` — Gutenberg fallback candidates decoded
  `utf-8, errors="replace"`; ISO-8859-1 editions get U+FFFD baked into `source.txt`
  and thence into clip transcripts verbatim. Alignment still succeeds (`norm_word`
  strips the damage), so "café" ships as "caf�" in training text with no detector.
- **A-H3** `librivox_fetch.py:247→331`, `book_router.py:288→337` —
  `books_ledger.json` read at start, written whole at end of a minutes-long network
  run: lost-update race against every other ledger writer, and bare `write_text`
  torn-file risk (fetch takes no `.bak`; router's `.bak` only covers its own lane).
- **A-H4** `librivox_fetch.py:99-172` — resolved project's URL never compared to the
  requested `--url`; a fuzzy-title near-miss downloads the wrong book/reader
  undetectably. The docstring claims the caller "cannot detect this" — it can: the
  API response carries the project URL. One-line check.
- **A-H5** `librivox_align.py:67` — naive regex sentence splitter (`(?<=[.!?])…`)
  splits on abbreviations ("Mr. / Smith went home.") and ships incomplete utterances
  in direct violation of the completeness rule; `book_ingest` built
  `is_complete_utterance()` (pysbd) for exactly this defect class, but the real-audio
  lane uses neither. Score cannot see it; it flows straight to audition.

### Medium
- **A-M1** `librivox_align.py:319-320` — one zero/missing playtime silently discards
  *all* durations (`if durations and not all(durations.values())`), reverting to the
  even split that the duration rewrite exists to fix — no message. And an
  `hh:mm:ss`-format playtime raises in the int parse and is *skipped*, leaving a
  durations dict that passes the `all()` check but sends one section down the
  even-split branch while neighbors use cumulative fractions — windows that don't
  tile. Degrade per-key, loudly.
- **A-M2** `librivox_align.py:247-260` — `refine()` counts non-blank *frames* as
  tokens (one token spans many frames), desynchronizing per-word spans. Currently
  masked because only first/last are used and min/max'd against anchors — but
  `t0 = min(t0, real[0][0])` can still pull the start up to 0.5 s into the previous
  sentence, and span-FiLM (roadmap) would inherit garbage. Use
  `torchaudio.functional.merge_tokens`.
- **A-M3** `librivox_align.sh:39` — qc_gate run via `uv run` on the host (run-mode
  rule violation; same file prints the correct `.venv/bin/python` idiom three lines
  later). Same stale guidance in fetch/ingest docstrings.
- **A-M4** `librivox_align.sh:19,27` — `rocm/pytorch:latest` + unpinned
  `pip install -q … >/dev/null 2>&1`: unrepeatable environment and fully swallowed
  install errors (surface later as a generic align failure).
- **A-M5** `librivox_fetch.py:296-332` — failed section downloads and missing
  canonical text still end in exit 0 and ledger status "fetched; awaiting align";
  partial books look complete.
- **A-M6** `book_ingest.py:920-924` — provenance hardcoded "Standard Ebooks CC0" even
  for PG/epub sources; false license metadata propagates into every derived clip's
  paper trail.
- **A-M7** `book_ingest.py:261` vs `:391` — dialogue extraction matches only curly
  double quotes; straight-quoted or single-quoted editions yield zero utterances and
  whole dialogue paragraphs enter *narration* windows, quotes included — wrong
  chunk_type, wrong director framing, wrong delivery labels.
- **A-M8** `book_ingest.py:145-146` — epub SKIP list is Standard-Ebooks-specific; PG
  epub boilerplate ("This eBook is for the use of anyone anywhere…") parses as prose
  and becomes render text. Fluent, well-formed, and not from the book.
- **A-M9** `book_router.py:252-256, 326-336` — SKIP detection checks only the `pg:`
  key: `lv:`-keyed entries are re-routed, duplicated under both keys once the lookup
  later succeeds, and their status regressed to pending. SE books are mis-namespaced
  `lv:<se-slug>`; PG `files/` URLs escape `etext_id_from`'s narrower regex twin.
- **A-M10** `librivox_fetch.py:328` — ledger status updated only under exact `--key`;
  the normal `--url` invocation leaves the router's `pg:NNN` entry stale forever, and
  the book dir gets the `lv_<slug>` naming scheme instead of `pg_NNN` — two on-disk
  naming schemes depending on invocation style.
- **A-M11** `book_ingest.py:903-930` — no checkpointing across the director pass; a
  `load_skill` error at chunk 90 discards 90 completed Gemma calls. Write the bank
  incrementally (jsonl) or guard `to_bank_line`.
- **A-M12** `librivox_fetch.py:212-215` — resume check is "exists and >1 KB"; an
  HTML error page saved as `NNN.mp3` is enshrined forever and later kills the whole
  align run (uncaught decode error). Cheap magic-byte check.
- **A-M13** `librivox_fetch.py:151-155` — only `HTTPError`/`JSONDecodeError` caught;
  a timeout/DNS blip aborts the whole fetch. No retries anywhere in a bulk-transfer
  lane.

### Low
- **A-L1** `librivox_align.py:143` — WhisperModel constructed per *section*; a
  25-section book loads the ASR model 25 times (the CTC bundle is cached two
  functions away, same lesson).
- **A-L2** exit 0 even when every section was gate-skipped; zero-clip books are
  indistinguishable from successes without reading logs.
- **A-L3** `book.json`/`source.txt` bare `write_text` (torn on crash; a truncated
  `source.txt` from an interrupted run is silently reused if the next text fetch
  fails).
- **A-L4** `-version-N` LibriVox URLs cannot resolve (fetch doesn't strip
  version tokens; router does).
- **A-L5** `book_ingest.py:859` — `--epub` without `--slug` crashes before the
  argument guard.
- **A-L6** `book_router.py:221` — `"play" in blob` matches "display"/"screenplay"
  (advisory only).
- **A-L7** router crashes with a stack trace when the ledger doesn't exist yet.
- **A-L8** `librivox_align.sh` `$*` flattening breaks any quoted/spaced extra arg.
- **A-L9** `_roman` folds ordinary words ("did"→999) — symmetric today, a collision
  trap tomorrow.
- **A-L10** duplicate slice computation, per-clip `import librosa`, unused vars in
  the hot loop.

---

## B. Synthesis campaign tooling
(`synth_bank.sh`, `container_as_ai_mgr.sh`, 8 × `synth_*.py`, `ref_select.py`,
`make_*_bank.py`, director skills, `test_skill_files.py`)

### Critical
- **B-C1 [verified]** `synth_bank.sh:173` + `qc_gate.py:197,261-271` — the mandatory
  QC gate is enforced by a `dirname` and a glob. `CAMPAIGN_DIR="$(dirname "$OUT")"`:
  a trailing slash on `OUT` makes the glob match nothing; qc_gate then prints
  `0/0 hard-pass`, writes an empty measures file, prints `QC-GATE-DONE`, exits 0, and
  `register_audition` queues every clip ungated — the precise failure the gate was
  built to end. Passing the campaign dir itself as `OUT` makes `CAMPAIGN_DIR` the
  datasets *root* (qc_gate ASR-transcribes the world). Fix: qc_gate exits nonzero on
  zero manifests/clips; synth_bank normalizes `OUT="${OUT%/}"` and asserts
  `qc_measures.jsonl` non-empty before registering.

### High
- **B-H1 [verified]** `book_ingest.py:628-634, 799-802` — the zonos "emotion truly
  off" state is unrepresentable through the director pipeline. `_l1(None)` returns
  the neutral-dominant vector `[0.05,…,0.77]`; `build_direction` always emits it; the
  casting schema *requires* an 8-float array so Gemma can't emit null either.
  Meanwhile `synth_zonos.py:186-192` correctly implements
  `emo is None → unconditional_keys | {"emotion"}` and `director_skills/zonos.md`
  instructs emitting `emotion: null`. Every director-driven zonos narration bank
  re-renders the defect measured 2026-07-30 (5/9 groups destabilized). Only
  hand-authored banks (e.g. newscaster) get the fix. Fix: `_l1(None) → None`, pass
  through, allow null in the schema, and point `test_skill_files` at the claim.
- **B-H2 [verified]** `book_ingest.py:807` — Orpheus fallback voice is `"tara"`, the
  voice `director_skills/orpheus.md` marks "REVERB + hiss. **Never cast.**" The
  jess-only room-tone rule exists solely in markdown; `synth_orpheus.py` has no voice
  guard, and the casting enum allows all 8 voices. Every director hiccup casts the
  worst-measured voice into a 15 %-of-mix engine. Fix: banned-voice set in code,
  fallback `jess`, asserted by `test_skill_files`.
- **B-H3 [verified]** `synth_bank.sh:155-160` — `normalize_loudness` failure prints
  "DO NOT audition until fixed" and then the script proceeds to QC and *registers the
  batch for audition* (also on the uv-missing branch). The message and the control
  flow contradict; should `exit 1` exactly like the qc-gate branch.
- **B-H4** all eight renderers — skip-if-exists keys on wav *existence*, writes are
  non-atomic, and the manifest row lands after the wav: a crash mid-`sf.write`
  leaves a truncated wav that every resume skips and no manifest row ever describes —
  the clip silently vanishes from the campaign while file counts look right (and a
  corrupt header can crash the loudnorm pass into B-H3). Fix: `{id}.wav.tmp` +
  `os.replace`, and/or skip only when a matching manifest row exists.

### Medium
- **B-M1** `synth_bank.sh` — every engine block ends `|| echo "(X failed —
  continuing)"`; no failure counter, final exit 0 even if all engines crashed at clip
  1. Echo a failed-engine count in the `== done ==` banner and reflect it in the exit
  code after registration.
- **B-M2** `synth_bank.sh:47-50` — `$BANK`/`$OUT` interpolated unquoted through two
  quoting layers; must be absolute `/data` paths with no spaces/quotes, enforced
  nowhere (a quote in a path breaks the `runuser` quoting). Add a
  `case "$BANK" in /data/*)` guard and quote the interpolations.
- **B-M3** chatterbox/zonos/vibevoice — the `used` variety-bias set starts empty on
  every resume (already-rendered jobs skip before `select_reference`), so rerolled
  clips cast as if the pool were untouched and casting isn't reproducible from the
  bank. Rebuild `used` from existing manifest rows at startup.
- **B-M4** `synth_bank.sh:35` + per-engine installs — `rocm/pytorch:latest`, unpinned
  transformers/snac/chatterbox-tts, a moving git HEAD for VibeVoice; only qwen pins.
  Manifests record seeds but not library versions. Pin digests + wheels; drop
  `pip freeze` into the out dir.
- **B-M5** moss_vg/orpheus retry loops re-seed identically (`torch.manual_seed(seed)`
  before each attempt) while the comments assume stochastic retry — deterministic
  failures can never converge; perturb `seed + attempt` and record the applied seed.
- **B-M6** `synth_dia.py:85-95`, `synth_moss85.py:56-66` — still the explicit-keys
  manifest style whose field-dropping was diagnosed and fixed in the other six
  engines (qwen's docstring memorializes it). SET_ASIDE is reversible; reinstatement
  resurrects a bug already paid for once.
- **B-M7** `make_bulk_bank.py:38-42` — Dia lines with leading tags (the "extreme,
  non-human" defect `_place_tags` exists to prevent), no trailing `[S1]`, and
  engine-less bank ids (collision → silent cross-engine skip). Legacy, but it is the
  on-disk spec a re-render would consume.
- **B-M8** `qc_gate.py` — no exit-code path reflects gate outcomes; combined with
  B-C1 the only enforcement is "the process didn't crash". Add one hard floor: exit
  nonzero when `hard_pass == 0` on a non-empty batch.
- **B-M9** `synth_chatterbox.py:164-165` — `BRIGHT_REF_POLICY` is never consulted
  (both exclusion and damping run unconditionally); the manifest records a policy
  string that didn't govern behavior. Delete the knob or wire it.
- **B-M10** `synth_orpheus.py:141` — `intended_gender` written `"F"/"M"` while the
  audition prefill contract expects `"Female"/"Male"`; Orpheus gender prefill is
  silently dropped and the auditor hand-enters it.

### Low
- **B-L1** `synth_bank.sh:43,47` — `-e HF_TOKEN=$HF_TOKEN` leaks the token into
  `ps`/`docker inspect` (same class as the tracked librivox_align.sh issue); use
  `-e HF_TOKEN` value-less form.
- **B-L2** `select_reference` LookupError uncaught in chatterbox/zonos/vibevoice —
  an uncastable line kills the engine's remaining bank; moss_vg's per-clip
  try/except is the pattern the others lack.
- **B-L3** `synth_qwen.py` loads the model before the zero-jobs check; and the
  `generate_voice_design or generate_custom_voice` getattr fallback silently switches
  API semantics.
- **B-L4** `synth_vibevoice.py` forks the CLI contract (multi-bank, optional `--out`)
  and calls `select_reference` without blacklist or excursion guard — reinstatement
  trap.
- **B-L5** `MAX_REF_EXCURSION = 240.0` duplicated in chatterbox + zonos; belongs in
  `ref_select` beside `REF_BLACKLIST` (which was hoisted for exactly this reason).
- **B-L6** moss_vg/moss85 multi-message decode writes one filename per N messages
  (clobber + duplicate rows if a job ever returns >1).
- **B-L7** `make_v3d_bank.py` — frozen fork of ref_select scoring, no blacklist/
  excursion/age logic, emits a bare list (not the bank envelope) — not
  synth_bank-consumable; dangerous if cargo-culted.
- **B-L8** `make_newscaster_bank.py:239` — `used.add(rid)` adds an *id* to a set
  compared against *file* keys (no-op; the `exclude` set is what actually works);
  engine blocks zip against thematically-ordered lines, correlating engine with
  topic.
- **B-L9** `test_skill_files.py` doesn't test the two markdown claims that turned out
  broken (zonos null-emotion, orpheus jess-only) nor that skill files cite the real
  `MAX_REF_EXCURSION`.
- **B-L10** `container_as_ai_mgr.sh` masks its own failures (`2>/dev/null` on
  groupadd/useradd/chown; no `set -e`); identity failures surface only as downstream
  engine errors already masked by B-M1.

### QC-contract verdict
The letter of the rule holds for gate *execution* (uv-missing and nonzero-exit paths
both stop before registering, with recovery instructions). It does not hold against
the empty-glob layout trap (B-C1), batch-level failure (B-M8), or the loudnorm step
(B-H3). `set -uo pipefail` without `-e` is the right choice given per-engine
continuations; no pipe launders a load-bearing exit code.

### Known-defect encoding map
| Defect | In code? |
|---|---|
| Chatterbox split-reverb | yes — excursion cap + damp + shared blacklist |
| Zonos rate ceiling | yes — clamp 16 |
| Zonos neutral-emotion conditioning | renderer yes; director pipeline **no** (B-H1) |
| Orpheus room tone (jess-only) | **no** — markdown only (B-H2) |
| Dia over-generation = token budget | yes |
| MOSS early-EOS | per-clip try/except (can't orphan bank) |
| Chatterbox train-only | manifest tag; enforcement in publish_tier |
| Cross-engine loudness | normalize_loudness in synth_bank — right home, wrong failure path (B-H3); standalone renderer invocations bypass it |

---

## C. QC / audit / staging
(`qc_gate.py`, `qc_artifacts.py`, `qc_engine_defects.py`, `qc_passages.py`,
`qc_verdict.py`, `gate_calibration.py`, `silero_vad.py`, `pick_audit_subset.py`,
`audit_sampler.py`, `stage_pool.py`, `reader_profile.py`, `seed_delivery.py`,
`register_audition.py`, `normalize_loudness.py`, `publish_tier.py`)

### Critical
- **C-C1 [verified]** `audit_sampler.py:256-262` — bare unguarded append to the live
  `ratings.csv`. It imports `register_audition` for conventions but not
  `_append_guarded`. If the audition app's read-modify-`os.replace` save lands around
  the append, the rows are silently erased *after* "appended N rows" was printed and
  `audit_sample.jsonl` was written — the disagreement/near-miss/novelty clips, chosen
  precisely because the instruments distrust them, ship un-auditioned with no error.
  The race window spans the whole slow F0-measurement run.

### High
- **C-H1** `register_audition.py:346-348` — clobbers `qc_flags.txt` wholesale while
  `qc_engine_defects.py:147-154` carefully appends-dedups the same file; run order
  decides whether engine-defect flags survive to `pick_audit_subset --flags`. Breaks
  "every QC failure is auditioned" for scrutinized-engine defects. Make both writers
  append-dedup.
- **C-H2 [verified]** `pick_audit_subset.py:255,351` — unknown ids fall back to
  `("?", "?")` → `DEFAULT_TIER = "normal"` → folding privileges. A ratings row whose
  id fails the bank/manifest join (MF/FM renames propagated to ratings but not the
  bank; a manifest line that failed `json.loads` and was silently continued) demotes
  a scrutinized zonos/moss_vg clip into a tier that folds its siblings unheard — the
  one thing the scrutinized tier exists to prevent. All unmatched clips also pool
  into one cross-engine group. Fail loud, or fail to scrutinized.
- **C-H3 [verified]** `stage_pool.py:344-346` — folded, unheard clips written as
  `status=keep, score=5`. Breaks the ear-only score invariant with verified
  downstream contamination: `gate_calibration` counts them as ear verdicts,
  `audit_sampler` counts them as scored history (prematurely retiring novelty
  sampling), and they satisfy `--mark-delivery`'s "heard" test. 12 such rows already
  on disk in `book-librivox-v1`. Folded clips need a distinguishable status/score
  (blank score + provenance note), like the certify path's `deferred` artifact.
- **C-H4** `stage_pool.py:218-221` — `--mark-delivery`'s homogeneity gate counts any
  row with status ≠ unaudited as "the ear said": deferred, dropped, reroll, and the
  fabricated keeps (C-H3) all vote. A staged run that propagated delivery can
  re-certify the mark that produced it (circular). Filter must be "human-audited AND
  delivery set by the ear".
- **C-H5** `seed_delivery.py:74-92` — unrecognized register falls through to
  `Dialogue`; the librivox real-audio campaigns (blank register, no `_nar_`/`_dia_`
  ids, not in the frozen SKIP list) would be mass-mislabeled on the documented
  "safe to repeat" re-run — audiobook narration and Dickens's *Speeches* stamped
  Dialogue. Default branch must return None (leave for the ear).
- **C-H6** `reader_profile.py:84-89, 176-191` — `learn()` excludes only
  `unaudited`, so machine-filled cells from a previous `--apply` (which fills
  deferred rows too) re-enter as agreeing votes; at `MIN_AGREE = 0.80`, four
  propagated copies out-vote one genuine later ear disagreement and `_CONFLICT` never
  raises — the profile echoes itself past the signal the (reader, title) design
  exists to surface. Count only ear-set cells, or record which cells `--apply`
  filled.

### Medium
- **C-M1** `pick_audit_subset.py:251-258` — QC flags intersect only currently-
  `unaudited` rows; a clip flagged after it was deferred is never promoted, no error.
  Needs a defer→unaudited flip for flagged ids.
- **C-M2** `qc_gate.py:210-217` — quarantined `_dropped`/`_superseded` wavs are
  substituted back in and processed identically: a deliberately dropped clip can
  re-enter keeps via `qc_filelist.txt`. Tag the row `quarantined: true` and exclude
  from hard_pass/filelist.
- **C-M3** `pick_audit_subset.py:369-386` — a group whose heard clips are all
  `reroll` certifies on zero heard evidence and folds its deferred siblings. Require
  keeps > 0 to fold.
- **C-M4** the 4 s speech floor has **no hard gate in qc_gate** (advisory note in
  register_audition only; librivox_align enforces its own), and qc_gate still
  measures speech with `librosa.effects.split(top_db=35)` while `silero_vad.py` —
  written to replace it, naming qc_gate as customer — is wired into nothing. Two
  instruments for one owner rule, and a short-text synthesis clip can pass every
  gate. Add `speech_ok` on the Silero measure.
- **C-M5** non-atomic writes: `stage_pool.py:254` (`books_ledger.json` — also raced
  with book_ingest), `stage_pool.py:381` (`staging_log.json` — truncation forgets
  staged ranges → double-staging), `reader_profile.py:167`, `publish_tier.py:111-116`
  (the `.bak` is only taken on the first run ever). tmp + `os.replace` everywhere.
- **C-M6** the mtime guard exists in four flavors: `seed_delivery` checks *before*
  serializing ~1,500 rows and never re-checks (widest window); `pick_audit_subset`
  compares float `st_mtime` while others use `st_mtime_ns`; `_append_guarded` is
  honest that verification only narrows its window and corrupts a row if the CSV
  lacks a trailing newline. One shared implementation (ideally `flock` on a sidecar,
  since the app takes no lock either).
- **C-M7** `reader_profile.py:186-191` — cross-title hint fills into a title whose
  own evidence is a `_CONFLICT` (any-audited-clip truthiness, not agreement).
- **C-M8** two definitions of "ear-confirmed pair": `stage_pool.confirmed_tags` (any
  one of gender/age/accent) vs `pick_audit_subset._confirmed_reader_titles` (all
  three) — `--seed-ear` skips pairs the auditor would still force-queue; partial
  new-reader-rule satisfaction.
- **C-M9** `qc_artifacts.py:207-214` — cohort stats include `_dropped`/`_superseded`
  wavs (biasing per-engine C50 medians toward rejected clips) and non-recursive
  manifest glob mis-attributes subdirectory wavs to a `"?"` engine cohort.
- **C-M10** `gate_calibration.py:65-69` — status ignored: dropped-with-score rows
  (38 verified in delivery-v1-narration) and fabricated 5s count as ear keeps; the
  FN-rate that gates pipeline 1.0 is computed over polluted cells.

### Low
- **C-L1** `pick_audit_subset` docstring says scrutinized tail "25 %"; code is
  `1.00` (code is right; doc is stale on the load-bearing constant).
- **C-L2** `register_audition --qc` help text documents the overruled draft behavior
  (reroll-instead-of-ear) — the code correctly does the opposite.
- **C-L3** `stage_pool.load_pool` dead code (`seen`, first `uniq`).
- **C-L4** `normalize_loudness` sidecar keyed on path: a rerolled-in-place wav is
  skipped as done and ships unnormalized; also silently downmixes stereo and rewrites
  PCM_16.
- **C-L5** `publish_tier` has no `librivox` ENGINE_POLICY entry — fails closed, but
  will hard-error the day staged real-audio clips reach a metadata.jsonl.
- **C-L6** trusted-tier 3 % sample rounds to zero under ~17 clips — the tail
  silently vanishes for small campaigns.
- **C-L7** `DEFAULT_TIER = "normal"` gives any brand-new engine folding privileges by
  default — tension with the onboarding rule; unknown engines should be loud or
  scrutinized.
- **C-L8** `qc_verdict` — missing eiv row fails every axis silently,
  indistinguishable from a genuine label failure.
- **C-L9** `qc_gate` DNSMOS divides by `len(wav)` — a zero-length wav kills the
  campaign run.
- **C-L10** `--mark-delivery` marks every title in the campaign — a mixed campaign
  can never mark just its speeches (refusal-safe but coarser than the key it
  writes).

### head_ok gate — confirmed absent, and where it slots in
No gate sees head truncation anywhere (`qc_engine_defects.leading_gap_score` sees
leading *silence* only, by its own docstring). `tail_lost()` (`qc_gate.py:138-150`)
already computes the difflib matching blocks; `blocks[0].a` is the head-loss word
count, currently discarded. Return it, add `HEAD_LOST_MAX`/`HEAD_WORDS_MIN` beside
the tail constants, gate beside `tail_ok`, add a "LISTEN TO THE START" triage branch
in `register_audition`. Threshold needs ear calibration like tail_ok.

---

## D. Label derivation (VAT / EIV / G2P)
(`derive_vat_corpus.py`, `eiv_merge_corpus.py`, `eiv_score.py/.sh`,
`derive_markup_measures.py`, `phonemize_filelist.py`, `matcha/text/op_g2p.py`,
Emilia miners, `tag_spike.py`)

### Critical
- **D-C1 [verified]** the G2P lane mispronounces every contraction/possessive, and
  it's in shipped v3. The 274,927-entry dictionary has zero apostrophe keys and the
  neural model's charset has no `'` (chars not in `c2i` are silently skipped), so
  contractions are phonemized from their apostrophe-stripped letters. Reproduced
  live: `don't → dˈɔnt` (should be /doʊnt/), `we'll → wˈɛl` (identical to *well*),
  and counted in the corpus: **545 `dˈɔnt` in `train_op.txt`**, ~1.3 % of all tokens
  across contraction forms, concentrated in dialogue. These count as `neural_hits`
  "successes", so "dict 95.39 % / 0 unresolved" masks it. The model trains audio
  /doʊ/ against phoneme /ɔ/ thousands of times — phoneme-label poison invisible to
  every gate. Fix: apostrophe strategy (dict augmentation for the closed contraction
  class + apostrophe-aware neural input), then re-phonemize v3 **before training**.

### High
- **D-H1 [verified]** the closed-vocab gate can't detect its primary failure mode.
  `op_g2p.py:185-186` falls back to `ipa = word` with a comment claiming it
  "surfaces as a vocab violation in validate()" — false: `a-z` and ASCII `'` are all
  in the 178-symbol vocab (espeak IPA reuses ASCII letters), so
  `validate('zyzzyqx') == []` and the FAIL/exit paths in `phonemize_filelist` and the
  `bad_vocab` drop in `derive_vat_corpus` never fire for pure-letter OOV. Latent
  today (0 unresolved in v3) but the advertised fail-loud invariant is a no-op.
- **D-H2** `derive_vat_corpus.py:357-378, 415-417` — coverage failures degrade
  silently to fabricated labels: uncovered clips get `V=0.0` (a plausible
  at-speaker-mean value, not a sentinel), 3- and 4-component tension raws are blended
  within one speaker population then re-z-scored, and the "uncovered" count is a
  print. The docstring records this exact failure shipping 1,094 mislabeled clips —
  the rerun was manual and no `missing > 0 → abort` gate was added.
- **D-H3** `derive_vat_corpus.py:439-441` — `--reuse-from` silently reassigns the
  train/val split: same seed, different input order (reuse builds train-then-val,
  fresh uses `find_clips` order) → different permutation. Any val-loss comparison
  across label versions — the whole point of a relabel — is contaminated. Shuffle a
  canonically sorted row list.
- **D-H4** one NaN in a raw score → speaker mean/std NaN → `clamp2(min(1.0, nan))`
  returns **1.0** — the speaker's whole channel saturates at the extreme, formatted
  as a clean `1.0000`. The independence gate goes NaN and prints FAIL, but the script
  still writes all filelists and exits 0. LUFS is isfinite-guarded; the EIV JSONs are
  not.

### Medium
- **D-M1** `mine_emilia_keeps.py:72,93` — missing EIV heads imputed as raw `0.0`,
  which z-scores to a *nonzero* pseudo-score (anchor means are nonzero). Currently
  safe (verified: both corpus files cover identical 30,351 wavs); an Emilia pass with
  the default 4-head set would trip it corpus-wide. Contrast `eiv_merge_corpus`,
  which correctly SystemExits.
- **D-M2** `eiv_score.sh` — unpinned image + pip for a scorer whose outputs must
  "stay comparable to the existing corpus"; a resumable pass spanning an upgrade
  produces two-inconsistent-halves undetectably (raw scores were never
  version-stamped). Also unquoted `${INPUTS[*]}` interpolation.
- **D-M3** digits and symbols silently deleted by the tokenizer
  (`"I have 3 cats & 50% of it" → "i hˈæv kˈæts ʌv ɪt"`, validate-clean). Audio says
  "three"; text drops it — MAS must absorb the mismatch. Latent for LibriTTS
  (0/5,000 sampled transcripts contain digits), live for Emilia YODAS captions and
  any future source. Violates the canonical-text premise; needs a digit check or
  normalization pass.
- **D-M4 [verified]** homograph flattening: one pronunciation per key, no context
  (`read`→ɹˈiːd, `wind`→wˈɪnd, `live`→lˈaɪv, `dove`, `bass`). Past-tense "read"
  trains against the wrong vowel. A regression vs the espeak lane this replaced;
  same class as D-C1, broader-known.
- **D-M5** `tag_spike.py:234-236` — appends to the live `ratings.csv` with
  copy-backup but no mtime guard (standing-rule violation; same class as C-C1).
- **D-M6** `process_emilia_tail.py:103-106` — resume treats error rows as done;
  transiently-failed clips are permanently skipped, silently absent, despite
  re-run-to-resume being the documented recovery.
- **D-M7** the independence gate prints "FAIL — residualize before training" and
  proceeds to write train/val filelists, exit 0. Only `derivation_report.json`
  records it; nothing downstream must read it. The label-derivation pass is the one
  generation pass without an enforcing gate.

### Low
- **D-L1** inline-dep drift in the one script that has a block
  (`derive_vat_corpus.py` lists librosa/numba, neither imported); several docstrings
  still prescribe `uv run`.
- **D-L2** small-speaker degeneracy is real in v3 (one 1-clip speaker → forced
  0,0,0; 17 speakers <10 clips; 2-clip speakers pinned at ±0.5; 7 % of train V
  saturated at ±1.0) and the two z-implementations disagree on the guard
  (`std + 1e-6` vs `std or 1.0` — the latter lets a 1e-300 std explode unclamped).
- **D-L3** `eiv_merge_corpus` ragged check compares head *count* not *set*; `--add`
  can silently overwrite raw scores for overlapping wavs ("raw is immutable" is
  convention, not code); `speaker()` mis-groups non-LibriTTS-shaped keys.
- **D-L4** `mine_emilia_keeps` cwd-dependent `LIB_MEASURES`; keeps copied by
  basename with no collision check.
- **D-L5** `derive_markup_measures.py` frozen at v2 paths/files; stores the
  contiguous v2 index under `"speaker"` — mis-keying trap.
- **D-L6** report inaccuracies (`seconds_total` includes dropped clips; speaker
  counts include fully-dropped speakers; TypeError when `corr_ta is None` on small
  runs).
- **D-L7** `phonemize_filelist.py` writes the `_op` output even on validation FAIL —
  a poisoned filelist persists for someone to train on later.
- **D-L8** `eiv_score.py` details: path-string resume key; `Head` reconstruction
  guesses ReLU placement by `i % 3` (unverifiable by load_state_dict);
  `make_tension_audit_set` "matched loudness" is per-speaker-relative, not absolute;
  `tag_spike` `.get("score", 0)` None-crash + `--no-register-audit` vs docstring.

---

## E. Model & training core
(`matcha/`, `configs/`, `vocalizer.py`, eval/test scripts)

### High
- **E-H1 [verified]** `eval_harness.py:339` — the controllability gate takes
  `abs(rho)`: a sign-inverted channel (the exact silent-wrong failure a channel-order
  swap or composite-sign bug produces) scores ρ = −0.95 and **passes**. Every
  measure mapping is positively directed; `abs()` only masks bugs. This is the
  objective gate for promoting directability checkpoints.
- **E-H2** `matcha/cli.py` is a broken entry point for every Sonora checkpoint:
  hardcoded `english_cleaners2` (espeak lane) against op-G2P/`no_cleaners`
  checkpoints — valid-but-distribution-mismatched phonemes, silently degraded
  output — and no `--vat`/`--guidance` args at all. `vocalizer.py` got both right;
  the CLI never did. Also contradicts README's "espeak banned from the runtime
  path".

### Medium
- **E-M1** the delivery 4th FiLM channel does not exist in the model core
  (`vat_dim=3` everywhere; datamodule parses exactly `v,a,t`; no configs) while
  bank builders already assert it. The collate's all-or-none
  `torch.stack(vats) if all(...) else None` will silently drop conditioning for a
  whole mixed batch during the migration — pre-place an assertion.
- **E-M2** Vocalizer HTTP API accepts unclamped VAT (`/v1/audio/speech` passes raw
  floats to `render()`; sliders enforce [-1,1], the API doesn't) — out-of-
  distribution conditioning renders silently degraded audio; guidance/ode_steps also
  unvalidated.
- **E-M3** documented bucketing off-switch is unreachable: `bucket_multiplier` isn't
  an `__init__` param, so `save_hyperparameters` never stores it; the getattr always
  returns 20 and passing it from yaml raises TypeError.
- **E-M4** `test_vat_identity.py` and `test_film_export_gate.py` print PASS/FAIL and
  always exit 0 (contrast `test_text_selection.py` and `eval_harness.py`, which set
  codes properly). The FiLM identity guarantee and export parity gate are decorative
  in any scripted pipeline.
- **E-M5** flow-matching loss logs unmasked padding noise (`u` isn't masked;
  gradients are fine, the *logged* diff_loss carries a padding-fraction floor) — the
  2026-08-01 bucketing change lowered logged loss independent of model quality;
  curves across that boundary are not comparable.
- **E-M6** RoPE cache built under `inference_mode` during `on_validation_end`
  synthesis can crash the next training step ("Inference tensors cannot be saved for
  backward") when the next batch's text is ≤ the cached length —
  `test_vat_identity.py` already dodges this in-process with a fresh model and a
  comment. Nondeterministic startup crash; rebuild the cache outside inference mode
  or drop it on train re-entry.
- **E-M7** GPL/stale-packaging remnants: `phonemizer` (GPL-3.0) is an unconditional
  requirements.txt dependency and `english_cleaners2` the default cli/app lane while
  README claims a runtime licence wall (the wall guards *training filelists* only);
  setup.py ships upstream name/author, no license field; Makefile `create-package`
  would twine-upload as `matcha-tts`. (No StyleTTS2 or LICENSE-COMMERCIAL/PATENT
  remnants — clean.)
- **E-M8** DiT-spike friction inventory (for the planned decoder swap): CFM
  hard-instantiates `Decoder` (no config selection); the U-Net downsampling factor
  is encoded in `fix_len_compatibility` and the `out_size` divisibility comment —
  two harness-side contracts outside the decoder; the FiLM trunk + per-level heads +
  re-zero-after-kaiming ordering live *inside* `Decoder` and must be replicated;
  `make_warmstart.py`'s allowlist will need a bump; `test_vat_identity` reaches into
  `decoder.estimator.mid_films[0]`.
- **E-M9** `cli.py:91-101` — inverted `hasattr` guard forces a pretrained-model
  download even with `--checkpoint_path`; custom-checkpoint runs fail offline.
  (Upstream bug, but it gates the project's main offline lane.)

### Low
- **E-L1** RTF hardcodes 22 050/256 — ~9 % flattering at 24 kHz.
- **E-L2** legacy ONNX export never passes `vat` — exporting a VAT checkpoint bakes
  neutral conditioning silently (litert lane handles it correctly).
- **E-L3** vocalizer lane-sniffing (`has_vat or n_spks > 1`) misroutes legacy
  multi-speaker checkpoints; module-level globals race under concurrent API calls
  selecting different checkpoints.
- **E-L4** `matcha/app.py` — upstream demo: downloads four checkpoints at import,
  `launch(share=True)` opens a public tunnel; superseded by Vocalizer but still a
  setup.py entry point.
- **E-L5** config drift: `configs/eval.yaml` references nonexistent mnist configs;
  `libritts_r_vat.yaml` still `num_workers: 20` while v2 documents why 8 under
  spawn.
- **E-L6** `matcha/train.py` global `weights_only=False` monkeypatch disarms
  torch.load safety process-wide.
- **E-L7** dataset `__init__` reseeds the *global* RNG (twice, same seed),
  overriding `seed_everything` and driving the out_size crop offsets — hidden
  coupling.
- **E-L8** `MEASURES.get(ch, "seconds")` — a typo'd channel silently measures clip
  duration.
- **E-L9** duration-dump layout mismatch vs `get_durations` lookup (dormant —
  `load_durations: false`).
- **E-L10** `on_validation_end` resets Lightning's persistent val iterator, assumes
  ≥2 samples.

### Verified good (core questions of this review)
- **unknown-VAT ≡ zero genuinely holds**: explicit zero substitution at all three
  entry points; FiLM heads zero-init and re-zeroed *after* the kaiming pass; CFG's
  unconditional branch uses `zeros_like(cond)`, matching the dropout-trained
  neutral.
- **Channel order/scale consistent everywhere they exist**: `V,A,T` from derivation
  → filelist → datamodule → render_vat_sweep → vocalizer; clamp z/2σ→[-1,1] at
  label time; only the HTTP API can violate range (E-M2).
- **Spawn-safety right** (forced spawn + persistent workers for the gfx1151 fork
  wedge; batch sampler in main process; per-epoch reshuffle works under Lightning
  2.x).
- **Masking through the FiLM path correct**, including cond/mask stride alignment
  through the U-Net.
- **hifigan/** vendored unmodified with its own MIT LICENSE.

---

## F. LiteRT export lane
(`scripts/litert_export/*`, `rename_tflite_tensors.py`, `export_fidelity_referee.py`,
`vocoder_copysynthesis.py`, `sky/`)

### Critical
- **F-C1 [verified]** the export "gate" suite cannot fail, and two of three
  conditioning channels are never exercised. `convert_vat.py`: G1 prints PASS/FAIL
  without exiting nonzero; G2 prints correlations with no threshold; G3 likewise; G4
  monotonicity is `all(...)` over a filtered dict — vacuously PASS when every row is
  skipped. Channel coverage: G2 runs decoder parity with `vat = zeros`; G3/G4 drive
  `[0.0, a, 0.0]` — **valence and tension have never once been driven nonzero
  through a converted graph**. A conversion that swapped or zeroed those channels
  would pass every automated signal in the lane. Same print-and-exit-0 theater in
  `build_matcha.py`, `e2e_matcha.py`, `e2e_masked.py`, `convert_final.py`; dirty
  artifacts still land under shippable names. Fix: enforced thresholds + exit codes,
  plus a per-channel differential probe (drive each channel independently, assert
  differential output response).

### High
- **F-H1 [verified]** `convert_vat.py:80-82` — `SONORA_REPO` defaults to the
  pre-flat `…/Sonora` (verified: contains only `github/`/`huggingface/`); the
  `sys.path` insert silently falls through to whatever `matcha` is importable in the
  harness venv — documented as the **stock** pip package (no FiLM/VAT). Loud today
  (TypeError on `use_vat`), a silent fork-vs-stock substitution the day signatures
  converge. `VAL_FILELIST` broken by the same root. Default to `…/Sonora/github`.
- **F-H2** `VAT_DIM = 3` hardcoded; config written with a 3-channel map; no delivery
  export story; and `test_film_export_gate.py` builds a synthetic random-weight
  3-channel `VATTrunk` (not the checkpoint, not the real graph) — when the model
  grows the 4th channel, the gate keeps green-lighting a 3-channel chain. Nothing
  records or enforces the 2σ clamp contract for mobile hosts either.
- **F-H3 [verified]** `test_film_export_gate.py:64` — the one thresholded gate ends
  in a print; no `sys.exit`. Unusable as a pipeline gate.

### Medium
- **F-M1** `export_fidelity_referee.py` — binds inputs by dtype heuristic (a
  vat/delivery float input either crashes or receives `scales`), picks output by max
  byte-size, ignores `wav_lengths`, truncates to min length before cosine (wrong-
  duration renders still score). It is also the *only* script in the lane with a real
  threshold and exit code — the referee needs to learn conditioned graphs.
- **F-M2** `rename_tflite_tensors.py` — maps `StatefulPartitionedCall:0/:1` →
  `wav`/`wav_lengths` by converter emission order with no shape sanity check
  (swapped order feeds the engine a scalar as the waveform); "renamed 0 tensors" is
  success.
- **F-M3** `kotlin_replica.py` asserts nothing numerically and has no
  spk/vat/delivery inputs — the mobile-side conditioning binding order, the
  highest-stakes surface, has no replica coverage; hardcodes MEL_MEAN/STD, TIME_DIM,
  LENGTH_SCALE while loading config.json in the same breath.
- **F-M4** kotlin_replica depends on artifacts nothing produces
  (`dp_g2p_matcha_fp16.tflite`, `g2p_dict.txt`; converter writes only fp32 and no
  dict; artifacts dir verified empty of g2p files) — unrunnable, and the G2P fp16
  step exists nowhere in-repo despite the README claiming it.
- **F-M5** `config.json` writes `time_embed_dim=1024` in both lanes while the masked
  graphs actually consume `in_channels` (160/224) — `exploit_measure.py` already has
  to know to ignore the field; any other consumer builds a wrong-size tensor.
- **F-M6** waveform parity is Pearson-only — scale/offset invariant, so a systematic
  fp16 gain error scores corr 1.0. For a model whose flagship axis is *energy*,
  there is no RMSE/absolute check anywhere, and min-length truncation hides length
  mismatches.
- **F-M7** CFG guidance (`flow_matching.py:84-87`) is inexportable and undocumented:
  no host pipeline implements it and config.json doesn't mention it — mobile can't
  match any desktop render with `guidance != 1`.

### Low
- **F-L1** no PEP 723 blocks anywhere in the lane (16 files) — runs only inside the
  undocumented `/data/toolchain/litert-conversion` venv.
- **F-L2** the only copy of the derisk checkpoint is pinned inside a root-owned
  training-log dir; `build_matcha.py` expects a ckpt beside itself that isn't in the
  repo.
- **F-L3** Google-attribution copyright headers on Sonora-authored files (only
  convert_vat.py carries the Artificial Humanity header) — Apache-2.0 hygiene.
- **F-L4** global monkeypatches (`torch.load` replaced permanently in
  convert_g2p_matcha; `_stub.py` fakes scipy badly enough that two siblings undo
  it).
- **F-L5** `vocoder_copysynthesis.py` hardcoded /data paths, bare-name imports via
  sys.path injection.

### Stale/dead inventory
Superseded: build_matcha's drop-mask lane, `e2e_matcha.py` (masked variants ship).
One-shot probes done: `probe_tx_standalone.py`, `probe_decoder_taps.py`. Unrunnable:
`kotlin_replica.py` (F-M4), `convert_vat.py` from the repo checkout (F-H1). Aging:
convert_vat pinned to the epoch-99 derisk checkpoint; nothing delivery-aware.
Different lane: `rename_tflite_tensors.py` (onnx2tf/engine path — easy to mistake).

### sky/ — clean
Region pinned (`runpod/US`, AU-4090 trap documented), no in-file credentials,
autostop backstops, smoke/train setup blocks identical as claimed. Self-documented
gap: file_mounts/license-wall/filelists never exercised.

---

## G. Repo hygiene & packaging (cross-cutting pass)

- **G-1 [verified]** no real lockfile: `uv.lock` is a 3-line stub (version headers,
  zero packages); `requirements.txt` is the stale upstream Matcha list (GPL
  `phonemizer`, `gradio==3.43.2`, optuna sweeper…); the actual venv contents are
  recorded nowhere. The environment that produced v3 and every render is not
  reproducible from the repo. Fix: `uv pip freeze` → a committed lock/constraints
  file, and prune requirements.txt to reality (dropping `phonemizer` per E-M7).
- **G-2 [verified]** test harness disconnected: `pyproject.toml` points pytest at
  `tests/` (doesn't exist), `--doctest-modules` on; real tests live as
  `scripts/test_*.py` and are never collected — `make test` tests nothing. Combined
  with T1 (gates that can't fail), the repo currently has **no** enforceable test
  entry point. Fix: a `tests/` shim or `testpaths = ["scripts"]` + naming, and give
  every gate script a real exit code (E-M4, F-H3).
- **G-3 [verified]** Makefile is upstream residue: `create-package` would
  twine-upload as matcha-tts, `train-*`/`start_app` use bare `python` (run-mode
  rule), `sync` does blind git pulls.
- **G-4** `setup.py` build pins (`cython==0.29.35`, `numpy==1.24.3`) predate the
  py3.11 venv; setup metadata still upstream's (E-M7 overlap).
- **G-5** `data/` holds `libritts_r_vat`, `_v1`, `_v2`, `_v3` side by side — fine as
  lineage, but nothing marks which is live; `derive_markup_measures.py` (D-L5) shows
  the cost of stale-version defaults.
- **G-6 [verified]** working tree clean, no committed build artifacts/pycache, LICENSE
  (Apache-2.0) + NOTICE + LICENSE-Matcha present and correct.

---

## Recommended fix order

**Before v3 trains (label correctness):**
1. D-C1 apostrophe/contraction G2P fix + re-phonemize v3 (the labels are wrong now).
2. D-H2/D-H4/D-M7: make derive_vat_corpus's coverage, NaN, and independence gates
   abort instead of print.
3. E-H1 drop `abs()` in the controllability gate; E-M4/F-H3 give the identity and
   export gates exit codes.
4. D-H3 stabilize `--reuse-from` split before any relabel comparison.

**Before the next campaign renders (workflow reliability):**
5. B-C1 + B-M8: qc_gate fails on empty/zero-pass batches; synth_bank normalizes
   `OUT` and asserts non-empty measures.
6. B-H1 zonos `emotion: null` pass-through; B-H2 orpheus banned-voice guard —
   plus test_skill_files assertions for both (B-L9).
7. B-H3 make loudnorm failure fatal; B-H4 atomic wav writes + manifest-keyed skip.
8. A-C1 pad-before-clamp fix and A-C2 manifest dedup — then re-cut the Dickens
   clips (cheap: alignment JSON is already on disk, only the slice/write step
   re-runs).
9. C-C1 guard audit_sampler's append; C-H1 make the qc_flags writers compose.

**Before the next fold/certify cycle (ear-evidence integrity):**
10. C-H3 provenance-bearing status for folded clips (defuses C-H4, C-H6, C-M10);
    C-H2 fail-loud unknown ids; C-H5 seed_delivery unknown → None.

**Background (debt that compounds):**
11. T3 atomic-write sweep (tmp+os.replace everywhere; shared mtime-guard).
12. T4 pin images/wheels; commit a real lockfile (G-1); pip-freeze into campaign
    dirs.
13. T5 `synth_common.py` consolidation; hoist `MAX_REF_EXCURSION`.
14. C-M4 wire silero_vad into qc_gate + add `speech_ok`; head_ok gate (blocks[0].a
    is already computed).
15. F-C1/F-H2 export-lane real gates + per-channel differential probe — required
    before the vat3/delivery export anyway.
16. E-M1/T8 delivery-migration assertions (collate all-or-none trap, VAT_DIM
    seams).
17. E-H2 teach cli.py the op-G2P lane + VAT args, or retire it in favor of
    vocalizer.
18. G-2 reconnect pytest; G-3/G-4 purge upstream packaging residue.

---

*Review conducted 2026-08-02 against HEAD `47a405d` while the newscaster-v1 render
was in flight. No source files were modified. Findings marked [verified] were
reproduced directly; all others carry file:line evidence from the subsystem passes.*

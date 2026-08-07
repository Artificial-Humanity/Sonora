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

## 2026-08-07

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

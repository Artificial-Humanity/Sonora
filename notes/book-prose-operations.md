# Book-Prose Synthesis — Operations Plan (2026-07-18)

_Operationalizes the [book-prose-synthesis-spike](book-prose-synthesis-spike.md): permissive book
prose → chunk + tag → the on-tap teacher TTS models → paired (text, audio) into the corpus. This
doc pins the runnable design; the spike doc holds the rationale. All new Python is `uv`-managed._

## Sources (settled)

**Standard Ebooks (CC0, lead) + Project Gutenberg (strip PG header/footer/trademark; filter
PD-translation)** for synthesis text; **librivox.org URLs** are also valid queue entries for the
force-align/real-audio lane (owner, 2026-07-19 — see Stage A). arXiv + Books3 dropped. See spike
§ Target text sources.

**LibriVox stance (owner, 2026-07-19):** LibriVox is a standing data source we want maximal use
of. But the division-of-labor rule ("in LibriVox ⇒ steer synthesis away") is now an **owner
choice, not an auto-skip**, whenever the owner explicitly queues an SE/PG book that LibriVox
covers — see the OWNER-CHOICE verdict below.

_Terminology: **"VAD" = Valence/Arousal/Tension** — the project's acronym (Prosodia README) for the
same triple the Sonora notes call V/A/T. The Director emits it; `book_ingest` emits it per chunk._

## Stage 0 — book-list intake (owner → pipeline)

**Mechanism: a queue file is the router's inbox** — **`/data/model-training/datasets/book_queue.txt`**
(central + easy to reach; owner call 2026-07-18, moved out of the repo), one URL per line, `#` for
comments, optional `| note` after a pipe (priority, why). The owner adds books two interchangeable
ways: **edit the file directly** (batches), or **paste URLs in chat** and they get appended (ad-hoc).
Either way the same automation consumes it — no separate path.

**Mark-in-place, not removed** (owner call): processed entries stay as a visible checklist/history.
Markers: `#` = comment; a plain line = **PENDING**; a line starting `*` = **PROCESSED** — the router
prepends `* ` and appends `  → <verdict> (<date>)`. Ambiguous matches stay PENDING with a `?? <reason>`
note for owner confirm; likewise an SE/PG book found in LibriVox stays PENDING with
`?? in LibriVox — synthesize instead?` for the owner's call. Verdicts: SKIP · REAL-AUDIO ·
OWNER-CHOICE · SYNTHESIZE (Stage A).

The router reads the queue, processes PENDING URLs, writes the full record to
**`/data/model-training/datasets/books_ledger.json`** (keyed on Gutenberg etext-id), marks the queue
line, and flags ambiguities. Queue = visible inbox+checklist; ledger = record of record. Scales from
one book to a batch; owner only in the loop for flagged ambiguities.

## Stage A — the source router (net-new; the real design work)

Given a book URL, **route it across three lanes** — the [force-align-first / division-of-labor
rule](dataset-landscape.md#-standing-data-prep-rule-force-align-text-to-audio-asr-is-fallback-only-owner-2026-07-18)
made executable. **Fully automated except ambiguous matches, which surface for a one-click owner
confirm** (owner, 2026-07-18).

| Verdict | Condition | Action |
|---|---|---|
| **SKIP** | Already in our `books_ledger` (any lane) **or** already in LibriTTS-R / MLS | Nothing — clean aligned audio+text already exists |
| **REAL-AUDIO** | The owner queued a **librivox.org URL** | Route to the **force-align lane**, **pinned**: use *this* book — and, if the link is a specific recording, *this* recording. A LibriVox link is an explicit owner pick, never an incidental crawl candidate. Align to the SE/PG text (never Whisper as primary) |
| ~~OWNER-CHOICE~~ | **RETIRED 2026-07-22 (owner rule change).** An SE/PG URL submitted via the book-submission page routes **straight to SYNTHESIZE — no overlap ask, no overlap check.** The owner pre-checks LibriVox before submitting and will *intentionally* submit an overlapping book after sampling LibriVox and finding it lacking. A submission is itself the lane decision. (LibriVox URLs still route REAL-AUDIO as before.) |
| **SYNTHESIZE** | The owner queued an **SE/PG URL** (via the book-submission page or queue) — regardless of LibriVox overlap | Route to `book_ingest` (Stage B) → teachers |

Checks:
- **LibriVox** JSON API (`/api/feed/audiobooks?title=&author=`); its projects reference their
  **Gutenberg source text**, so the join is strongest on etext-id.
- **LibriTTS-R / MLS membership** — index their book/reader lists once locally; check by
  title/author.
- **`books_ledger.json`** (net-new) — one row per work, **canonical key = Gutenberg etext-id**
  (secondary: SE slug, normalized author/title/translation). Records lane, status, campaign, chunk
  coverage. Rolls up the per-clip synthesis manifests to book level. This ledger is the connective
  tissue that makes "pieces now" safe (below).

**Ambiguity guard:** a fuzzy title/author hit, or a translation/edition mismatch (a *different*
translation in LibriVox than the SE/Gutenberg text), is NOT auto-resolved — it queues for owner
confirm so we never skip a book on a false match or align to the wrong edition.

## Stage B — `book_ingest` (the prep automation)

The analog of `make_bulk_bank.py`: emits the **same flat bank schema** the `synth_*` renderers
consume (`{id, engine, register, intended{V,A,T}, seed, text, direction{…}}`), just sourced from
books instead of a hand-authored `bulk_spec.json`. Four steps:

1. **Fetch + clean.** SE: download the CC0 EPUB → **reuse Prosodia `folioparser`**
   (`parse_epub` → chapter-ordered clean plain text; EPUB-only). Gutenberg: **net-new** `.txt`
   reader + strip PG header/footer/trademark + filter PD-translation (FolioParser doesn't cover
   plain-text, and it's intentionally not an on-device concern). Normalize unicode/quotes, drop
   front/back matter + footnotes.
2. **Segment** — **reuse Prosodia `stage::segmenter`** (`SentenceSegmenter` +
   `NarrationGrouping::{Sentence, Paragraph{target_characters}}`): quote-aware sentence split +
   bounded-length grouping = the same code the on-device reader uses, so training data is segmented
   exactly as books will be at runtime (train/serve consistency + dogfood). Produce the two chunk
   types (spike value thesis), bounded to the engine-reliable window (~LibriTTS utterance lengths,
   ~3–12 s; `target_characters` is the length knob, aligning with Dia's `token_budget` char→dur
   model):
   * **contiguous narration windows** — consecutive sentences, **order preserved** (continuity);
   * **dialogue line + its attribution** — `"…," she whispered` → the attribution verb/adverb is a
     director's note (self-labeled register, near-free). *(Attribution/diarization may be the
     Director's job rather than the segmenter's — the `director` crate already does casting/
     "who's speaking"; decide split at prototype.)*
3. **Tag → direction (the Gemma 4 director-pass).** **Reuse the Prosodia `director` VAD + casting
   schema/prompt** (not its LiteRT-LM engine binding — offline we serve the bigger Gemma 4 via
   Ollama/vLLM, not on-device LiteRT-LM). Reads each chunk + local context, emits
   `intended{V,A,T}` + the per-engine `direction` string + engine assignment (portfolio register
   affinities: soft→Qwen, force/dark/oratory→MOSS, narration/V+→Dia). **This IS the production
   Director's job** (character attribution/diarization + span→profile+VAT, per
   [high-ambition-2](high-ambition-2-dramatic-reader.md)) run offline at larger size — so it
   **dogfoods the Director** and its output schema should match the production Director's annotation
   schema. Dialogue attribution seeds the read cheaply; narration defaults neutral unless cued.
4. **Emit bank** → unchanged renderers.

### Director model — DECIDED: Gemma 4 26B-A4B (QAT)

**Owner call 2026-07-18: Gemma 4 26B-A4B QAT.** MoE — 26B total, ~4B active — so throughput +
energy stay near a 4B dense model while the 26B total gives literary/register breadth; QAT (≈4-bit,
quality near bf16 at much lower memory) makes it cheap on the box. Same family as the shipped
Gemma 4 E2B Director → the big offline sibling, maximal dogfooding. **Served via the ollama (ROCm)
container's OpenAI-compatible API on `:11434`** — already running on the box and proven on this exact
model (~58 tok/s GPU); `book_ingest` talks HTTP to that local endpoint, not the on-device LiteRT-LM
binding. No 12B bake-off; the MoE is the pick. (Serving stack = ollama, decided 2026-07-18 — an
earlier "LM Studio" mention was an over-read of the model's LM-Studio catalog link, not a decision.)

- **HF repo:** `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (Apache-2.0, QAT q4_0 GGUF, not gated).
- **Local path:** `/data/models/Google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (the reference
  library, alongside the on-device `gemma-4-E2B/E4B` Directors).
- **Live serving 2026-07-18:** registered **persistently in the ollama container as
  `gemma-4-26b-a4b-qat:latest`** (14 GB blob) with an explicit Gemma multi-turn Modelfile `TEMPLATE`
  + `num_ctx 8192`; runs on ROCm/GPU at ~58 tok/s. **Exposed on the LAN via Open WebUI** (`:3000`,
  `OLLAMA_BASE_URL=http://ollama:11434`) for personal testing. Endpoint for `book_ingest`:
  `http://localhost:11434` (OpenAI-compatible `/v1` or native `/api/chat`).
- **Reasoning-mode: RESOLVED.** Gemma 4 is a reasoning model — ollama returns its reasoning in a
  separate **`thinking`** field and the answer in **`content`**. The earlier empty `response` was a
  too-small `num_predict` (thinking ate the budget) on a bare-`FROM` import. With the proper template
  + adequate budget (~1024), a director prompt returns clean VAD JSON in `content`, e.g.
  `{valence:-0.7, arousal:0.6, tension:0.9, register:"hushed, trembling", direction:"escalating"}`
  for a trembling-threat line. `book_ingest` reads `content` (optionally sets `think:false` for the
  structured pass) and gives enough `num_predict` for thinking + JSON.
- **⚠️ Spin-down:** this is now a **standing inference service** — stop `gemma-4-26b-a4b-qat` in
  ollama (and the other inference containers) before any training run. See
  [[spin-down-inference-before-training]].

The ~30-chunk spike still **validates** it (not as a comparison) — score against ground truth
(owner audit + instrument verdicts): (1) VAD agreement, (2) attribution/diarization accuracy,
(3) structured-output reliability (malformed-JSON rate at bulk), (4) throughput + energy. Director
noise is absorbed downstream anyway (principle #1: "never trust intent, we measure").

**License: clean (Apache-2.0).** Gemma 4 ships under **Apache-2.0** (verified 2026-07-18:
ai.google.dev/gemma/apache_2 + the HF repo's `license:apache-2.0` tag — Gemma 1–3's "Gemma Terms of
Use" no longer applies). So the Director's generated VAD + direction text carries **no
redistribution restriction** — it can ship in the CC-BY-4.0 public manifest freely, and the Director
even satisfies the teacher-portfolio wall's "engines Apache/MIT only" by construction. No mitigation
needed.

## Stage C — synthesis + vetting (already built, unchanged)

`book_ingest` bank → `synth_{dia,qwen,moss85}.py` → `qc_gate.py` (ASR-fidelity + DNSMOS-tier +
duration sanity — ASR-fidelity weighs more here since chunks are longer) → `eiv_score.py` /
phonation instrument label check → `qc_verdict.py` → **Dataset Auditions app** (`audition.ai-lab-0
:8095`) + `ratings.csv` (SSOT), owner blind-audit at the standing bar ("affect obvious without the
keyword"). LongCat transfer multiplies certified keeps. Book-prose stays a **bounded minority** of
the corpus.

## Operational guard — no inference during training

Hard precondition on any training run: **spin down every inference engine on the box first** — the
Gemma director-pass, the `synth_*` renderers, the Vocalizer (unified-memory + thermal contention on
the Strix Halo). See [[spin-down-inference-before-training]]. The prep pipeline runs Gemma + TTS
inference, so a prep campaign and a training run never overlap.

## Whole books vs pieces (resolved)

**Pieces now — order-preserving contiguous windows, not isolated lines and not whole books.**
Isolated lines can't teach cross-sentence continuity; whole-book synthesis floods the corpus with
neutral narration (the imbalance we're fighting) and burns compute on low-value calm prose. Take
curated expressive chunks (dialogue+attribution) + a few contiguous narration runs per book, order
retained.

**Whole books become useful later — but the whole-book artifact is rarely synthetic:**
- Whole-book **text** is free on demand from SE/Gutenberg → the input for the Director learning
  book-level pacing, and for **eval scripts**. No need to pre-synthesize it.
- Whole-book **real audio** with continuity is the LibriVox **force-align lane's** job (the router's
  REAL-AUDIO verdict).
- Whole-book **synthetic renders** = an on-demand **integration/eval artifact** at the "does
  Prosodia hold delivery across a whole work" phase — not a pre-built training corpus.

The **`books_ledger` preserves whole-book optionality**: even taking only pieces, we register the
whole book + source, so we can return for more, fetch the full text/audio when the whole-book phase
arrives, and never reprocess.

## Minimal spike to run first (validates before any campaign)

1. One **SE title (CC0)** + one **Gutenberg title (stripped)**; prove the fetch/clean +
   PD-translation filter on real files.
2. Run the **router** on both (LibriVox/LibriTTS-R/MLS/ledger checks) — confirm the SKIP/REAL-AUDIO/
   SYNTHESIZE verdicts and the ambiguity-confirm path.
3. `book_ingest` prototype on ~30 chunks: ~15 contiguous-narration + ~15 dialogue-with-attribution.
4. **Director validation**: run stage-3 tagging with **Gemma 4 26B-A4B QAT** (via the ollama
   endpoint), score on the four criteria above — validating the chosen model, not comparing sizes.
5. Render across Dia/Qwen/MOSS-8.5B → unchanged `qc_gate`. Read out hard-pass rate on longer chunks;
   whether attribution-derived register survives instrument verification; whether contiguous
   passages hold prosody across sentences; LongCat transfer on a couple of passing anchors.
6. **Owner blind-audit** the keeps → the release gate.

Decisions the spike answers: (a) does book prose earn a standing lane; (b) does **Gemma 4 26B-A4B**
tag/attribute reliably enough at bulk; (c) is the value the continuity + self-labeled-dialogue
reframe as argued.

Cross-refs: [book-prose-synthesis-spike.md](book-prose-synthesis-spike.md) (rationale) ·
[synthesis-pipeline.md](synthesis-pipeline.md) (renderers, control interface, QC) ·
[dataset-landscape.md](dataset-landscape.md) (force-align rule, sources, division of labor) ·
[high-ambition-2-dramatic-reader.md](high-ambition-2-dramatic-reader.md) (the Director's
attribution/diarization job this dogfoods) · [teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md)
(ratified portfolio + license wall).

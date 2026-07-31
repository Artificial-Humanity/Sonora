# Book-Prose Lane — rationale and operations

> **Consolidated 2026-07-31** from `book-prose-synthesis-spike.md` (the rationale) +
> `book-prose-operations.md` (the runnable design). They were split spike-then-plan while
> the lane was still a proposal; it has been live since 2026-07-22 (`books_ledger.json`,
> `book_ingest.py`, 17 ingested books), so the split had become two hops to one answer.
> Operations lead, because that is what a reader needs now. The rationale follows as the
> record of why the lane is shaped this way.

---

# Part 1 — Operations

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
3. **Tag → direction (the Gemma 4 director-pass — TWO PASSES as of 2026-07-26).** **Reuse the
   Prosodia `director` VAD + casting schema/prompt** (not its LiteRT-LM engine binding — offline we
   serve the bigger Gemma 4 via Ollama/vLLM, not on-device LiteRT-LM).
   * **Pass 1 — label the LINE, engine-agnostically:** `intended{V,A,T}` + `register`. The
     `register` value MUST be copied verbatim from the controlled lexicon at
     `scripts/synthesis/register_lexicon.json` (47 labels, regenerated by
     `build_register_lexicon.py`). Free-text registers are a defect — they fragment the training
     labels, and before this was enforced ratings.csv had drifted to 138 distinct labels.
   * **Pass 2 — write casting/delivery PER ENGINE**, governed by
     `scripts/synthesis/director_skills/<engine>.md`. The JSON schema is **engine-shaped**:
     single-string engines (`qwen`, `moss_vg`) are asked for ONE field; VibeVoice is asked for
     casting metadata that `ref_select.py` parses; Dia is asked for inline text control only.
   `build_direction()` in `book_ingest.py` is the single source of truth for what each renderer is
   actually handed. Engine assignment stays with pass 1 (portfolio affinities: soft→Qwen,
   force/dark/oratory→`moss_vg`, casting-led→VibeVoice; Dia is markedly expressive but the weakest
   measured quality, so it is no longer the default narration lane). **This IS the production
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

`book_ingest` bank → `synth_{vibevoice,qwen,moss_vg,dia}.py` → `qc_gate.py` (ASR-fidelity + DNSMOS-tier +
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
5. Render across VibeVoice/Qwen/MOSS-VoiceGenerator/Dia → unchanged `qc_gate`. Read out hard-pass rate on longer chunks;
   whether attribution-derived register survives instrument verification; whether contiguous
   passages hold prosody across sentences; LongCat transfer on a couple of passing anchors.
6. **Owner blind-audit** the keeps → the release gate.

Decisions the spike answers: (a) does book prose earn a standing lane; (b) does **Gemma 4 26B-A4B**
tag/attribute reliably enough at bulk; (c) is the value the continuity + self-labeled-dialogue
reframe as argued.

Cross-refs: **§ Part 2 — Rationale** (below) (rationale) ·
[synthesis-pipeline.md](synthesis-pipeline.md) (renderers, control interface, QC) ·
[dataset-landscape.md](dataset-landscape.md) (force-align rule, sources, division of labor) ·
[high-ambition-2-dramatic-reader.md](high-ambition-2-dramatic-reader.md) (the Director's
attribution/diarization job this dogfoods) · [teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md)
(ratified portfolio + license wall).

---

# Part 2 — Rationale (2026-07-18, owner-initiated)

_Why the lane exists and what it is worth. Kept because the strategic reframe in here — that
the value is paired (text, audio) with KNOWN delivery, not the prose itself — is the thing
that keeps the lane pointed in the right direction._

## The one-sentence finding

**This is not a new pipeline — it is a new *text front-end* on the already-built teacher-synthesis
pipeline** ([synthesis-pipeline.md](synthesis-pipeline.md)). Every downstream stage exists:
`synth_{vibevoice,qwen,moss_vg,dia,longcat}.py` renderers, `qc_gate.py` (ASR-fidelity + DNSMOS-tier +
duration sanity), instrument label verification (EIV + phonation/LUFS), the bounded-minority
corpus rule, the CC-BY-4.0 publication plan. The **only new component** is a `book_ingest` stage
that emits the existing `script_bank.json` schema (`id, engine, register, intended V/A/T, text,
direction, seed`) from book chunks instead of hand-authored lines. That is what makes this a cheap,
high-leverage spike rather than a program.

## Terminology check first (needs a one-word owner confirm)

The concept said "VAD." In our stack that collides three ways — pinning it changes the design:

1. **V/A/T** (Valence / Arousal / Tension) — our actual conditioning axes (ARCHITECTURE §, VAT
   docs). **This is what I'm assuming "delivery tagging" means**, and everything below maps to it.
2. **VAD = Valence/Arousal/Dominance** — the classic psych affect model. If that's the intent,
   note we deliberately use *Tension* not *Dominance*; a VAD tagger's output would be re-projected
   onto V/A/T, not ingested raw.
3. **VAD = Voice Activity Detection** — audio-side, not text-side. It has a real role here but on
   the **output**: trim leading/trailing silence and catch Dia's improvised tails / silence-padding
   (the pilot's 12 s-file-with-4 s-speech pathology). If that's what you meant by adding VAD to
   chunks, it belongs after render, and `qc_gate.py`'s effective-speech-duration measure already
   does most of it.

I proceed on reading (1)+(3): tag chunks with intended **V/A/T + directorial note**, and use
**voice-activity detection on the rendered audio** as a QC/trim step.

## 📚 Target text sources — owner-maintained list (set 2026-07-18)

**DECIDED 2026-07-18 (owner): the books lane is Standard Ebooks + Project Gutenberg only.** arXiv
and Books3 are dropped — 1+2 give "more content than we can digest anyway." Verdicts apply the same
wall as everywhere else (CC-BY-4.0-or-freer, clean provenance, no NC/ND/pirated lineage). **Status
legend: 🟢 cleared · ⚪ dropped.**

| # | Source | Status | Verdict |
|---|---|---|---|
| 1 | **Standard Ebooks** | 🟢 **ACTIVE** | **Cleanest choice.** Editions released **CC0**; underlying texts PD; **translation copyright vetted by SE** (offloads the modern-translation-is-copyrighted hazard). Lead source. |
| 2 | **Project Gutenberg** | 🟢 **ACTIVE** | Usable **with a strip step** — bare PD text is free, but the **PG Trademark License** rides on the produced files. Strip header/footer/trademark, don't redistribute under the PG name; filter to PD-original **and** PD-translation ourselves. Scale-out source once the strip step is proven. |
| 3 | **arXiv Bulk Dataset** | ⚪ dropped | Not needed — 1+2 exceed digestible volume. Was never clean whole (default arXiv license ≠ third-party redistribution; only a CC-BY/CC0 subset qualifies) and carried heavy LaTeX-normalization + flat-affect costs. Parked here for recall if a *technical-narration* lane is ever wanted; would re-enter only as its filtered CC-BY/CC0 slice. |
| 4 | **The Pile / Books3** | ⚪ dropped | **Removed for cause, not just capacity.** No permissive Books3 set exists — ~196k books scraped from the **Bibliotik pirate tracker**, **DMCA'd down 2023**, central to active author suits (Kadrey/Silverman v. Meta); fails the wall outright. The Pile's clean books component is **PG-19 = Project Gutenberg** (source #2), so nothing of value is lost. Do not revisit. |

Lineage fit for the two active sources is strong: **Gutenberg text is literally what built
LibriSpeech/MLS** (Gutenberg audio ↔ text alignment) — using it as a synthesis *text source* is the
same clean provenance one step earlier. Engines are already Apache/MIT-only and the voice is
synthetic (no consent surface), so renders off sources 1–2 are **publishable under the same
CC-BY-4.0 plan** as the authored-line campaigns.

## ⭐ The strategic reframe — where the value actually is (and the trap)

Naïvely synthesizing book chunks **reproduces the exact imbalance the expressive-registers effort
exists to fix**: prose is dominated by neutral narration, so a raw book campaign would flood the
corpus with calm narration — the same calm-dominates tail problem
[dataset-landscape.md](dataset-landscape.md) flags for real audiobooks. So book-prose is **not** a
cheaper substitute for the balanced authored bank on the rare V/A/T tails. Its real, distinct
value is three things the authored single-line bank *cannot* give:

1. **Connected-narration continuity (the big one).** Authored lines are isolated; book passages are
   *connected multi-sentence prose*. Synthesizing contiguous passages produces **synthetic
   cross-sentence continuity** — the thing dataset-landscape calls out as LibriTTS-R's unique worth
   (chapter/utterance ordering) and the thing the Director needs the Actor to *sustain*. This is the
   feature that authored lines structurally lack.
2. **Self-annotated dialogue registers — nearly free labels.** Novels are full of **dialogue
   attribution that is literally a director's note**: *"she whispered," "he snarled," "he said,
   voice cracking."* The prose tells you the register of the quote. A `book_ingest` that pulls
   **(quoted line + its attribution verb/adverb)** yields **register-labeled expressive lines at
   near-zero authoring cost** — the attribution maps to intended V/A/T and to the per-engine
   direction string. This is the genuine synergy and probably the highest-value slice of the whole
   idea.
3. **Embodiment in the wild.** Narrated dialogue *is* narrator-within-narrator (the Beartown
   example, an owner requirement): a narration voice shifting to portray a character. Passages with
   quoted speech are natural embodiment training material with the shift boundary marked by the
   quote marks.

**Reframed thesis:** book-prose synthesis is a **volume + continuity + self-labeled-dialogue lane**
— it deepens narration and dialogue coverage — while the **balanced authored bank stays the
instrument for the scarce emotional tails**. Two complementary text sources, not a replacement.

## The new component: `book_ingest` (the only real work)

Produces `script_bank.json` rows from books. Stages:

1. **Fetch + normalize** — SE (CC0) or Gutenberg (strip header/footer/trademark). Filter to
   PD-original **and** PD-translation.
2. **Chunk to engine-reliable spans.** Not arbitrary length — chunk to the **render-reliability
   window** the pilot already mapped (Dia token budgeting in `synth_dia.py`; duration-vs-text
   sanity). Sentence / short-paragraph granularity; segment narration vs quoted dialogue at quote
   boundaries so #2 and #3 above fall out naturally.
3. **Tag delivery (the "prosody/VAD" step).** Per chunk emit intended **V/A/T + directorial note +
   engine/register**. Three signal sources, cheapest-first: (a) **dialogue attribution** parsed
   straight from the prose (free, high-precision where present); (b) a lightweight **LLM director
   pass** — read the chunk, emit intended V/A/T + a per-engine direction string (this is exactly the
   authored-bank's "intended direction," now generated); (c) lexical/prosodic priors as fallback.
   Narration defaults to neutral unless the passage cues otherwise.
4. **Emit bank rows** → hand to the **unchanged** renderers + QC gate.

Everything after step 4 is the pipeline as-built. **Labels are still generation-conditioned and
instrument-verified** (principle #1): we tag intent, render, then require the EIV/phonation
instruments to CONFIRM the intended direction or drop/relabel — identical to the authored-line
discipline, so a mis-tagged book chunk fails the same gate.

## ⚠️ Watch-items (each is already a known lesson, re-applied)

- **Dedup / parallel-data with LibriVox lineage.** Gutenberg text ⊂ the text behind
  LibriTTS-R/MLS/LibriVox. Synthesizing it blind = synthetic renders of text we may already have
  **real** audio for. Two takes: (bad) redundancy; (interesting) **paired real-vs-synthetic on
  identical text** = a distillation/contrast signal. To maximize *new* coverage, prefer SE titles
  and Gutenberg works **not** in the LibriVox set — or use the overlap deliberately as parallel
  data, never by accident.
- **Bounded minority holds.** Synthetic stays a bounded minority of any training corpus
  (portfolio principle #5); book-prose volume is cheap and could silently blow past that — cap it
  explicitly and `log` the cap, don't let easy volume rebalance the corpus toward synthetic.
- **Long-chunk engine stress.** Longer spans stress duration control and Dia improvisation harder
  than authored one-liners did; ASR-fidelity gate matters *more* here, and chunk length is the
  control (same finding as the Dia token-budget work).
- **Neutral-flood cap.** Rate-limit neutral-narration keeps so the continuity lane doesn't drown
  the expressive slices; sample dialogue/attribution-tagged chunks preferentially.
- **Voice identity is scrambled on ingest** (as today) — book-prose renders inherit the same
  identity-free posture; no real person, synthetic voices only.

## Minimal spike to actually run (validate before any campaign)

Small, decisive, uses only the existing renderers + one throwaway `book_ingest` prototype:

1. Pull **one SE title (CC0)** + **one Gutenberg title (stripped)**; confirm the strip + PD-
   translation filter on real files.
2. `book_ingest` prototype on ~30 chunks: ~15 **contiguous narration** spans (continuity test) +
   ~15 **dialogue-with-attribution** spans (self-labeled-register test). Emit the bank schema.
3. Render across the ratified four (VibeVoice / Qwen / MOSS-VoiceGenerator / Dia), run the
   **unchanged** `qc_gate.py`.
4. Read out: (a) hard-pass rate on longer book chunks vs authored one-liners; (b) does
   **attribution-derived register survive instrument verification** (the core bet of slice #2);
   (c) does contiguous-passage rendering hold prosody across sentences; (d) LongCat transfer on a
   couple of passing narration anchors.
5. **Owner blind-audit** the keeps against the standing bar ("affect obvious without the keyword")
   — the release gate, unchanged.

Decision the spike answers: *does book prose earn a standing lane in the synthesis pipeline, and is
its value the continuity+dialogue reframe (as argued) or something else the ears find?*

## Settled questions (owner, 2026-07-18)

1. **"VAD"/tagging = generate the bank's existing label + direction fields — RESOLVED.** There is
   no separate tagging layer to invent: the mechanism that already "pushes text through the models"
   is the per-line **`direction`** object + **`intended{V,A,T}`** label, per-engine
   (Qwen/MOSS = a natural-language *instruction*; Dia = *inline text tags* + sampling — see
   [synthesis-pipeline.md § Control interface](synthesis-pipeline.md)). So the tagging step **emits
   `intended{V,A,T}` + the per-engine `direction`** from book text — that IS "VAD tagging" here, and
   it's V/A/T (what the schema carries), not Dominance. Output-side **voice-activity trim** is a
   separate QC concern (post-render silence/tail trim), not this step.
2. **Standard Ebooks first — CONFIRMED.** SE as spike source (CC0 + vetted translations); Gutenberg
   as scale-out once the strip + PD-translation filter is proven.
3. **Steer AWAY from LibriVox-overlapping text — RESOLVED (division of labor).** LibriVox ships
   audio + an *unaligned* source-text pointer (usually the Gutenberg edition), not a transcript.
   Its **aligned derivatives — LibriTTS-R, MLS (already cleared) — DO carry ground-truth aligned
   text** ("LibriVox audio force-aligned to Gutenberg"). So: **real audio wherever it exists gets
   real/aligned text; synthesis is reserved for texts with no real audio.** Consequences:
   * Book-prose synthesis **avoids** LibriVox/LibriTTS-R/MLS-overlapping titles (real audio beats a
     synthetic render of the same text).
   * For **raw LibriVox** (the dramatic-narrator curation, dataset-landscape standout #3): pair the
     audio with the **official SE/Gutenberg text and force-align** (the LibriSpeech method) — **do
     not rely on Whisper**; Whisper is fallback-only, and here the official text always exists.
   * **Amendment (owner, 2026-07-19):** when the owner *explicitly queues* an SE/PG book that
     LibriVox covers, the steer-away is not applied automatically — the router asks whether to
     **synthesize instead of using LibriVox** (the owner may have judged that LibriVox recording
     below the quality bar). An owner-provided **librivox.org link** pins that book/recording for
     the force-align lane. See book-prose-operations.md Stage A (OWNER-CHOICE verdict).

_Sources = Standard Ebooks + Project Gutenberg only (arXiv + Books3 dropped, 2026-07-18);
librivox.org links accepted in the queue for the force-align lane (2026-07-19)._

Cross-refs: ****§ Part 1 — Operations** (below) (the runnable operations plan —
router, book_ingest, director bake-off, spike steps)** ·
[synthesis-pipeline.md](synthesis-pipeline.md) (the pipeline this front-ends) ·
[teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md) (ratified portfolio +
license wall) · [dataset-landscape.md](dataset-landscape.md) (continuity-vs-range framing;
LibriVox/MLS lineage) · [audiobook-corpus-policy.md](audiobook-corpus-policy.md) (public-vs-private
lineage) · [expressive-registers-dataset] standing directive (living dataset, CC-BY-4.0 publish).

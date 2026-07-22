# Spike: Book-Prose → Teacher-Synthesis Training Data (2026-07-18, owner-initiated)

_Owner concept: take permissive/public-domain ebooks → chunk them → tag each chunk with delivery
(affect + prosody) → feed the chunks through the on-tap synthesis teachers → keep the paired
(chunk text, rendered audio) and add both to the dataset. Owner text-source instinct: **Standard
Ebooks and Project Gutenberg.** This brief spikes feasibility, not a build._

## The one-sentence finding

**This is not a new pipeline — it is a new *text front-end* on the already-built teacher-synthesis
pipeline** ([synthesis-pipeline.md](synthesis-pipeline.md)). Every downstream stage exists:
`synth_{dia,qwen,moss85,longcat}.py` renderers, `qc_gate.py` (ASR-fidelity + DNSMOS-tier +
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
3. Render across the ratified three (Dia/Qwen/MOSS-8.5B), run the **unchanged** `qc_gate.py`.
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

Cross-refs: **[book-prose-operations.md](book-prose-operations.md) (the runnable operations plan —
router, book_ingest, director bake-off, spike steps)** ·
[synthesis-pipeline.md](synthesis-pipeline.md) (the pipeline this front-ends) ·
[teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md) (ratified portfolio +
license wall) · [dataset-landscape.md](dataset-landscape.md) (continuity-vs-range framing;
LibriVox/MLS lineage) · [audiobook-corpus-policy.md](audiobook-corpus-policy.md) (public-vs-private
lineage) · [expressive-registers-dataset] standing directive (living dataset, CC-BY-4.0 publish).

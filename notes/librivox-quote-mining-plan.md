# LibriVox Quote Mining — the story-driven valence/register lane (spec, 2026-07-21)

_The lead workstream of the v1.1 rescope
([emilia-keeps-audit-verdict.md](emilia-keeps-audit-verdict.md)): harvest **quoted character
dialogue** — with its attribution clause — from LibriVox audiobooks aligned to their SE/PG texts.
Owner direction (2026-07-21): clip the quoted segments, and capture the attribution statements
around them — preceding ("then she muttered, …") and following ("… mused Jack", "shouted
Frank"). This spec turns that into a pipeline on standing infrastructure._

## Why this beats instrument mining (the Emilia lesson, inverted)

The attribution verb is a **textual label of vocal delivery, written by the author and performed
by the narrator**: "whispered" tells you the register of the following quote before any
instrument hears it. Three independent signals then have to agree before a label ships:

1. **Text**: attribution verb (+ quote content sentiment) → expected register/V/A/T.
2. **Acoustics**: the standing instruments (LUFS, phonation composite, EIV combo) measured on
   the clipped audio.
3. **Ears**: the standing per-campaign human audition.

Text-first selection cannot be gamed by synthetic-speech artifacts or recording conditions (the
Emilia failure mode), and disagreement between (1) and (2) is itself signal — it flags either a
flat narrator (drop) or an instrument miss (calibration data).

## Pipeline (stages; A–B are text-only and GPU-free)

### A. Quote + attribution extraction (text side)

Over each REAL-AUDIO book's SE/PG chapter text (the canonical text per
[[force-align-first-dataprep]]):

* **Quote spans** — typographic first (curly quotes in SE texts are reliable), SCM quote/cast-map
  pass for speaker identity (the Gemma director already does cast mapping; frozen-annotator rule
  applies).
* **Attribution clause**, both shapes the owner named:
  - *preceding*: `<Name> <verb>[,:] "…"` — "then she muttered, '…'"
  - *following/inverted*: `"…," <verb> <Name>` / `"…," <Name> <verb>` — "… ," mused Jack /
    shouted Frank / Sarah said
  Regex over a **controlled attribution-verb lexicon** (seed list below), Gemma pass only for
  ambiguous cases. Record: book, chapter, quote char-span, clause char-span, verb, cast name,
  clause position (pre/post).
* **Verb lexicon → prior labels** (seed; grows app-governed like the SCM register lexicon):
  - energy/arousal ↑: shouted, cried, exclaimed, yelled, bellowed, called
  - energy/arousal ↓: murmured, muttered, whispered, breathed, sighed
  - tension ↑ (TIGHT): snapped, hissed, growled, snarled, demanded, barked
  - tension ↓ (LAX) + real aspiration: whispered, breathed, sighed
  - valence −: sobbed, groaned, moaned, wailed, grumbled, complained
  - valence +: laughed, chuckled, beamed, cheered
  - neutral anchor: said, replied, answered, asked, continued (the abundant baseline class)
* Also emit **neutral narration spans** from the same chapters — matched-narrator baselines the
  per-speaker z-scores and audits need.

### B. Candidate table + sizing

One JSONL of candidates per book; report verb-class counts before any audio work. Go/no-go per
book: enough non-neutral candidates to be worth aligning (say ≥ 200). This is where we learn
whether one novel yields hundreds or thousands of usable quotes.

### C. Alignment + clipping (audio side)

* Chapter-level: LibriVox mp3 ↔ SE/PG chapter text, **forced alignment of the canonical text**
  (never ASR-primary). Tooling decision at build time — candidates: MFA, torchaudio
  wav2vec2 CTC-segmentation, aeneas; long chapters likely need coarse anchoring (ASR timestamps
  as *anchors only*) before fine alignment. CPU-friendly; GPU optional.
* Clip the **quote span** (the acted voice) at word boundaries with small padding; keep the
  **attribution clause** timing as metadata (usually narrator-neutral — excluded from the
  emotional clip but a per-speaker baseline candidate).
* Per-clip QC: alignment confidence floor, duration 1–16 s, resample 24 kHz mono (same
  contract as `process_emilia_tail.py`).

### D. Labels + gates (the Emilia lessons, applied)

* Label = verb-lexicon prior, **cross-checked** by the instruments; agreement filter decides
  keep/flag. Verb class is the primary label; instruments never promote a clip on their own.
* *Candidate fourth vote (logged 2026-07-23):* **Kimi-Audio-7B-Instruct** (MIT weights) —
  speech-emotion recognition from an architecturally unrelated audio-LM as an independent
  cross-check on verb-prior/instrument agreement. Frozen-annotator rules apply (like Gemma).
  Not an actor candidate: no casting interface (stop-me rule case, 2026-07-23).
* **Human audition per campaign batch** (Auditions app, `audit-librivox-<book>` convention)
  *before* anything merges — same gate that just saved us.
* Speaker rows: one per narrator (LibriVox narrator = the voice; cast names are within-narrator
  acting, recorded as metadata — one narrator performing many characters is exactly the
  expressive-registers behavior Sonora wants to learn).
* License: LibriVox is public domain — clean lane; declare dataset dirs in
  `data_licenses.yaml` before training (the license wall is not optional).

### E. Corpus integration

Merge survivors into the v1.1 corpus (`libritts_r_vat_v3`): filelists, speaker-table extension,
mel-stats check, then continue training from the vat3 epoch-099 checkpoint
(`Sonora/huggingface/vat3-24k/`).

## Start point

`pg:14275` (The Necromancers) — already in `books_ledger.json` with a LibriVox recording
pinned (REAL-AUDIO route, owner-queued). Stage A over its SE text is the first concrete task
and costs nothing but CPU.

## Companion lane (parallel, small)

Directed teacher-synthesis valence renders (labels exact by construction) remain the second
depth source per the verdict note — and the two lanes share the audition workflow, so
their batches can interleave in the same campaign rhythm.

Linked from: [emilia-keeps-audit-verdict.md](emilia-keeps-audit-verdict.md) ·
[book-prose-operations.md](book-prose-operations.md) · [STATE.md](STATE.md)

## Full-cast / dramatic readings (owner submissions 2026-07-22)

The 12 owner-pinned LibriVox titles include several **full-cast dramatic readings** (Oliver
Twist v5, The Story Girl, plus a block of stage plays: Earnest, The Rivals, She Stoops to
Conquer, Six Characters, a one-act collection…). Owner pre-sampled; conveyance at standard.
These upgrade the lane in two ways and complicate it in one:

* **Upgrade 1 — real acted dialogue per role:** each character is a REAL distinct voice
  (identity layer = actual actor, not narrator portrayal). Emotional dialogue arrives
  pre-cast; the identity/portrayal split (casting-attribute-norms-brief.md) is trivially
  clean here.
* **Upgrade 2 — casting-norms corpus:** many distinct real voices with per-role text =
  ideal calibration material for the measured gender/age norms.
* **Complication — alignment must be role-aware:** multiple speakers per file means the
  align step needs the LibriVox per-section reader metadata (its API lists readers per
  section) and/or light diarization to attribute spans to voices before clipping. Plays
  align text-side via speaker headings (parse_epub's drama branch); dramatic-reading novels
  align to the novel text with cast-map propagation.

Single-narrator titles (Monte Cristo, Jane Eyre v3, The Awakening, Lesley Castle) proceed
on the original stage-C design unchanged.

## Publication boundary (owner rule 2026-07-22)

Clips from this lane (LibriVox-derived) TRAIN Sonora but are **never included in the
published expressive-registers dataset** — that artifact is own-synthesis-only. Not a
license constraint (LibriVox is public domain); policy: the published set is our original
contribution and its provenance stays trivial. Corpus integration (§E) is unaffected.

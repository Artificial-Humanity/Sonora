# Casting-Attribute Norms — owner directive (2026-07-22)

_Design capture, not yet scheduled. Owner direction, verbatim intent: train Sonora so it does
NOT inherit the gender conflation observed in other models (Qwen VoiceDesign flips gender
against explicit designs ~10% of the time; Higgs' un-instructed defaults skew young; both
attractors pull toward youthful/bright). Strategy: **define our own vocal norms per casting
attribute — measurably — and assign tags from those norms**, so casting is grounded in
acoustics we control rather than in whatever a teacher model absorbed._

## The attributes (owner-scoped 2026-07-22)

| Attribute | Values | Note |
|---|---|---|
| **Gender** | `male` · `female` · `neutral` | Deliberately simplified to three. |
| **Age** | `child` · `teen` · `adult` · `middle-aged` · `elderly` | Five bands. |
| **Accent** | open list — e.g. `British`, `American`, `African American`, `Indian`, … | Owner framing: not truly "race" but general accent, which stereotypically can include one — "we'll have to accept stereotypes here in some regard." Widest category; no exhaustive list attempted yet. |

Plus "other potentially useful characteristics" as they prove out (timbre classes, vocal
weight, etc.).

## Norms = measurements, not vibes

The owner's instinct: "things like pitch frequency ranges and such." Candidate instruments,
to be CALIBRATED ON OUR OWN AUDITED DATA (define the ranges ourselves — textbook constants
are priors, not norms):

* **Gender**: median F0 range per class + formant positions/spacing (vocal-tract length
  proxy) — F0 alone misclassifies (low female / high male voices); F0 × VTL jointly
  separates much better. `neutral` = the deliberate overlap band.
* **Age**: F0 (children high, elderly drift), formant scaling (VTL grows to adulthood),
  plus texture measures we already compute — jitter/shimmer-adjacent (aperiodicity rises
  with age), speech rate, H1-H2/CPP (breathiness patterns differ young vs old). The
  existing phonation composite inputs (alpha/CPP/H1H2, `derive_vat_corpus`) are reusable.
* **Accent**: not reducible to scalar acoustics — classifier/embedding territory (e.g.
  fine-tuned speech-embedding classifier over our audited accent tags). Hardest, widest,
  last.

## Where it plugs in (nearest first)

1. **Post-render casting verification (immediate use):** the synth pipeline's missing gate.
   Measure rendered clips against the norms → auto-flag gender/age mismatches instead of
   burning owner ears on them (Task #4's flip-rate sweep becomes the calibration set).
   Audited tag stays the SSOT ([[rename-on-tag-mismatch]]).
2. **Corpus casting tags:** run the instruments over LibriTTS-R speakers + synth voices +
   future LibriVox narrators → per-speaker gender/age (accent later) tags under OUR norms,
   owner-audited where the instruments are uncertain (the Auditions app already has the
   gender-correction affordance).
3. **Sonora conditioning (the actual goal):** casting attributes as explicit conditioning
   channels — the same FiLM pattern as VAT — so the shipped actor takes `gender/age/accent`
   tags grounded in measured norms. This is what prevents inherited conflation: Sonora's
   casting channels are trained against instrument-verified labels, never against another
   TTS's habits. Relates to the speaker-encoder (sonora-mid) item; sequencing vs v1.1 VAT
   work TBD — logged, not scheduled.

## How the director knows — the cast sheet (owner question, 2026-07-22)

Per-segment context (±2 sentences) cannot resolve casting: knowing "Sam" is a girl, twelve,
and Welsh takes book-wide coverage. Two designs, one per side of the lab/device split:

* **Lab (training pipeline): a whole-book CAST-SHEET PASS before direction.** The 26B-A4B
  director has a 256K context — an entire novel fits in one call (The Necromancers ≈ 110K
  tokens). Pass 0 reads the book and emits a cast sheet: characters + aliases ("Laurie" =
  "Mr. Baxter" = "her son"), gender/age/accent per OUR attribute scopes, with **evidence
  quotes** per attribute (auditability — the claim "female" cites "…his sister Maggie…").
  Stage-A mining already extracts speaker tokens per quote; the cast sheet joins token →
  attributes, and each per-segment direction call gets the relevant cast entries injected.
  Chapter-batched fallback if a book overflows context. Narrators (LibriVox lane) get sheet
  entries too.
* **Device (live reading): the story graph** (Prosodia/notes/director-narrative-memory.md)
  is the designed home — Character nodes carry exactly these attributes, stamped with
  `reveal_offset` + `source_span`. The wrinkle the graph already handles and the lab pass
  can ignore: **spoiler-gated casting** (a narrator whose gender is a late reveal must be
  cast neutrally until the reveal offset). Corpus prep may use whole-book knowledge freely;
  a live director must respect the reveal frontier.

The cast sheet also upgrades the quote-mining lane directly: unattributed quotes in
dialogue runs (65% of Necromancers quotes) become resolvable by cast-map propagation, and
the SCM cast-map tag gets its attribute payload from the same sheet.

## Identity vs. portrayal — the two-layer model (owner, 2026-07-22)

Most books are single-narrator, and the narrator's **voice identity persists through
cross-gender portrayal**: a female narrator deepens her pitch and "sounds bigger" for a
male character's quoted lines, but the voice remains female. So a clip carries TWO casting
layers, never conflated:

* **Identity** (the norms above): the narrator/speaker's own gender/age/accent. This is
  what the measured norms tag, what the speaker table encodes, and what stays constant
  across a narrator's whole book.
* **Portrayal**: the character being emulated within that voice — a *delivery
  modification* (pitch shift, size, register), not an identity change. Cast-sheet
  character attributes drive the portrayal target; the audible result is
  "identity-female performing male," which is its own acoustic signature. This is exactly
  the phenomenon the valence audit flagged as "narrative/embodiment (acted character
  voices)" — it confused a one-layer instrument, and it stops being noise once the two
  layers are separate metadata.

**Narrator identity resolution belongs in the cast sheet:** third-person narration → the
narrator is a voice choice (owner/casting pick, any identity). **First-person narration →
the narrator IS a character, and the voice must match that character's attributes** —
owner's example: The Hunger Games is narrated by its 16-year-old female protagonist; a
husky male narration voice would be wrong. The cast-sheet pass therefore also emits:
narration person (1st/3rd), and if 1st, which character narrates → their attributes become
the default casting for ALL narration in that book.

## Why now (evidence trail)

quote-pilot-v3/v3b/v3c (2026-07-22): Qwen rendered "middle-aged male, warm baritone" as a
young woman and "elderly female" as a male; its voices skew late-teens-to-mid-20s in both
genders; Higgs' un-instructed defaults share the young-skew. Casting via teacher-model
prompt adherence is unreliable — casting via measured, owner-defined norms is the
Sonora-native alternative.

Linked: [[teacher-synthesis-portfolio]] (engine traits) · [[vat-audit-verdicts]] ·
[[rename-on-tag-mismatch]] · notes/vat-conditioning-design.md (FiLM pattern).

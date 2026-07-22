# Sonora Conveyance Markup (SCM) — Schema Design Brief (2026-07-19, **RATIFIED v0.1** same day)

_The pin-first task from [direction-interface-brief.md](direction-interface-brief.md) §6: the
markup scheme must exist before the tagging phase. This drafts **SCM v0.1** for owner
ratification. Guiding constraints: (1) markup is the **Director-facing** layer — it compiles
deterministically to the Actor's continuous contract, never enters the text encoder; (2) every
claim is **measurement-anchored** so the round-trip verifier can check it; (3) the production
Prosodia Director emits this same schema at runtime (train = serve)._

## 1 · Core decision: sidecar is canonical, inline is a projection

**The canonical form is a sidecar JSON object per utterance.** The text stays byte-identical to
the training transcript; spans reference it by token index. An **inline projection** (tags woven
into the text) is *rendered from* the sidecar for human reading/audit — never authored, never
parsed back. This resolves the inline-vs-sidecar question by refusing the fragile half: authored
inline markup corrupts text round-trips; a rendered projection costs nothing.

```
sidecar (SoT) ──deterministic render──▶ inline view (for owner audit / prompts)
     │
     └──deterministic compile──▶ Actor contract (V/A/T channels [+ span deltas later], casting)
```

## 2 · SCM v0.1 draft

```json
{
  "scm": "0.1",
  "id": "tr_grief_herenow_00_brokenMF_s1234",
  "text": "She was just here. She was just... I was holding her hand a minute ago.",
  "utterance": {
    "vat": {"V": -0.85, "A": -0.30, "T": 0.40},
    "register": "grief_herenow",
    "style": ["hollow", "stunned"]
  },
  "spans": [
    {"tok": [3, 3], "type": "emphasis", "level": 2},
    {"tok": [7, 7], "type": "pause_after", "bin": "long"},
    {"tok": [8, 15], "type": "pace", "value": "slow"}
  ],
  "casting": {"gender": "Female", "age_band": "adult", "voice": ["broken", "soft"]},
  "direction": "Stunned disbelief collapsing into tenderness; let the ellipsis hang.",
  "provenance": {"source": "instruments+gemma-26b-a4b", "verified": true,
                 "verifier": {"pass": true, "checked": ["vat", "pause_after"], "flags": []}}
}
```

Inline projection of the same object (rendering rules fixed in the spec):

```
[V−0.85 A−0.30 T+0.40 · grief_herenow · F/adult]
She was just <em2>here</em2>. She was just...<pause:long/> <pace:slow>I was holding her hand a minute ago.</pace>
```

### Field semantics

| Field | Range / vocab | Compiles to | Verified against |
|---|---|---|---|
| `utterance.vat` | continuous [−1,1] each (the contract's units: per-speaker z clamped @2σ) | the three FiLM channels, directly | instrument V/A/T within tolerance (±0.25 default) |
| `utterance.register` | **controlled lexicon** = the expressive-registers vocabulary (grown only via the Recategorize/relabel flow — the app is the lexicon's governance surface) | nothing yet (metadata; later a register embedding if ever wanted) | EIV family profile consistency (heuristic) + owner audit |
| `utterance.style` | open adjectives, ≤3 | nothing (Director-facing color) | unverified (interpretive) |
| `spans[].type=emphasis` | `level` 1–3 | *(future)* per-token channel delta | token F0/energy peak percentile — **needs the span decode layer** |
| `spans[].type=pause_after` | bins: `micro` <120 ms · `short` <300 · `med` <600 · `long` ≥600 | *(future)* duration-model pause target | alignment gap in the bin — needs span layer |
| `spans[].type=pace` | `slow` / `fast` (local rate ≥1σ from utterance mean) | *(future)* per-span duration scale | phone-rate deviation — needs span layer |
| `spans[].type=pitch_move` | `rise` / `fall` (utterance-final or span-final F0 slope) | *(future)* | F0 slope sign/magnitude — needs span layer |
| `spans[].type=nonverbal` | `laugh` / `whisper` / `breath` / `cry` | *(future; engine-dependent)* | phonation measures (whisper: breathy-extreme) / classifier |
| `spans[].type=quote` | references a key in the utterance-level **`cast` map** (v0.1 per owner call — pulled forward from v0.2); optional per-quote `vat`/`register` override | cast entry → speaker selection (attribute axes later); overrides → the channels within the span | attribution/casting not acoustically verifiable pre-axes → owner audit + Director-consistency checks |
| `cast` (utterance/chunk level) | map of `narrator` + character keys → casting blocks (below); quote spans point into it — stable identity per character, matching the Director's casting job | see `casting` | see `casting` |
| `casting` / cast entries | `gender` Male/Female/Undefined (matches ratings vocab) · `age_band` child/young/adult/senior · `voice` open adjectives | speaker selection today; the Phase-1 attribute axes later | speaker metadata / owner gender tags |
| `direction` | one free sentence | nothing (the human/Director note) | **explicitly unverifiable** — the only interpretive-licensed field |
| `provenance` | source, verified, verifier report | — | — |

**Token indexing:** 0-based over whitespace tokens of `text`, inclusive `[start, end]`.
Whitespace tokens (not phones, not chars) because they're stable under the segmenter, readable
in review, and map 1-N onto phone alignments when the span layer lands.

**Numbers vs symbols:** the sidecar keeps VAT **continuous** (it IS the contract); binning to
symbols (`A:+2` etc.) happens only in the quantizer's Gemma prompt and the inline header —
presentation, not storage.

## 3 · The three producers, one schema

1. **Reverse-conveyance labeling (offline, now):** instruments → quantizer → Gemma fills
   `register`/`style`/`direction` (+ spans once the span decode exists) → verifier stamps
   `provenance.verified`. Utterance-only objects are **valid v0.1** — spans are optional, so
   the pipeline ships before the span layer does.
2. **Book-prose director-pass (offline, existing):** migrate its output
   (`{V,A,T, register, direction}`) to emit SCM objects — it is already SCM-shaped; this is a
   prompt/format change, not a redesign.
3. **Prosodia Director (on-device, later):** emits the same objects at runtime; the compiler
   in the reader is the same compiler used to build training pairs. Dogfooding is the point.

### Scope note (owner, 2026-07-19)

**Gemma is a frozen annotator in this lane — we are not fine-tuning it.** Clip-level tagging
with a local context window (±few sentences) is sufficient for the labeling pass's purpose:
producing verified (markup → audio) pairs for **Sonora's** training. Feeding larger text
blocks through Gemma and slicing afterwards is a **Director-conditioning** concern (how the
production Director should consume book-scale structure) and belongs to that workstream, not
this one.

## 4 · Verifier contract (what "verified: true" means)

Every measurable claim re-derives from the notation store within tolerance; interpretive fields
(`style`, `direction`) are exempt **and must never be load-bearing** (nothing compiles from
them). A claim whose measure is missing (e.g. spans before the span layer) is `checked: []` —
absent, not assumed. Failed rows keep their markup with `verified: false` + flags: they are
Gemma-debugging material, excluded from training pairs.

## 5 · Execution plan (after ratification)

1. **Freeze v0.1**: JSON Schema file + validator + inline renderer + compiler stub in
   `Sonora/scripts/markup/` (schema is a repo artifact, versioned like `bulk_spec`).
2. **Quantizer** (`utterance_notation.jsonl` → symbolized Gemma prompts) per the brief §3.
3. **100-clip spike** (50 certified + 50 LibriTTS-R): Gemma 26B fills the interpretive fields;
   registers recovery scored against known labels; malformed/verifier-fail rates read out;
   inline projections into an `audit-markup-v0` campaign for owner spot-audit in the app.
   **DONE + owner-audited 2026-07-20 — PASS (93%):** 89/96 kept, 64 at 4–5, 4 register
   relabels, 3 drops. Failures concentrate in ≤6-token LibriTTS fragments (9/14 fails vs
   3/64 keeps) → future rounds get a token-count floor on corpus picks. Full readout:
   `markup_prep/spike_v1/audit_results.json`. The SCM/Gemma annotator is validated as the
   labeling pass.
4. **Migrate the director-pass** prompt to SCM output (book lane inherits the schema).
5. **Span layer** (cdminix bring-up or forced-alignment pass) → span types activate, verifier
   grows their checks — no schema change, the fields are already there.
6. **Round-trip audition (owner idea 2026-07-19) — the acceptance test:** each audit card
   pairs the ORIGINAL clip with a **round-trip render** (SCM → compiled contract → Sonora) so
   the owner hears whether the markup captured the delivery. Gated on the **vat3 round**: the
   derisk checkpoint expresses only the energy channel, so a render today would test the
   model, not the markup. Post-vat3: render `rt_<id>` takes for the audited SCM rows, register
   them as paired rows (or add a small A/B-pair feature to the Auditions app), compare.
   Voice is NOT part of the comparison (Sonora speakers ≠ teacher voices) — the question is
   conveyance, not timbre.

## 6 · Owner decisions — RATIFIED (owner, 2026-07-19)

1. ✅ **Sidecar-canonical + rendered inline** (§1).
2. ✅ **Span reference = whitespace-token indices.**
3. ✅ **Register lexicon = controlled, app-governed** (grown via the relabel flow only).
4. ✅ **v0.1 tag inventory = SIX**: emphasis / pause_after / pace / pitch_move / nonverbal
   **+ `quote`** — owner pulled quote/character-voice forward from v0.2 into v0.1.
5. ✅ **Quote binding = utterance-level `cast` map** (stable character identity; per-quote
   VAT/register overrides allowed) — not inline-per-quote casting, not opaque tags.
6. ✅ **VAT verifier tolerance = ±0.35** on the clamped scale — **owner amendment later
   2026-07-19** (originally ±0.25): the spike showed ±0.25 is geometrically tighter than the
   0.4-wide quantizer bins, flagging bin-obedient claims; ±0.35 matches the bin. Copy-through
   (quantizer-stamped VAT) was considered on the same evidence and **declined** — Gemma keeps
   stating VAT so the model must attend to the measurements.
7. ✅ **Name = `scm`** ("Sonora Conveyance Markup"), version field `"scm": "0.1"`.

Example with cast + quote (book-dialogue shape):

```json
{
  "scm": "0.1",
  "text": "\"Hold the line,\" Magnus said quietly, and the room went still.",
  "utterance": {"vat": {"V": -0.1, "A": 0.2, "T": 0.6}, "register": "threat_warning"},
  "cast": {
    "narrator": {"gender": "Female", "age_band": "adult"},
    "magnus":   {"gender": "Male", "age_band": "senior", "voice": ["dry", "level"]}
  },
  "spans": [
    {"tok": [0, 3], "type": "quote", "cast": "magnus",
     "vat": {"V": -0.2, "A": 0.1, "T": 0.8}, "register": "threat_warning"},
    {"tok": [6, 6], "type": "emphasis", "level": 1}
  ],
  "direction": "The quiet kind of command — authority that never raises its voice."
}
```

Cross-refs: [direction-interface-brief.md](direction-interface-brief.md) (pipeline, model
choices, notation store) · [synthesis-pipeline.md](synthesis-pipeline.md) (register vocabulary)
· [book-prose-operations.md](book-prose-operations.md) (director-pass) ·
[high-ambition-1-matcha-actor.md](high-ambition-1-matcha-actor.md) (contract-lock).

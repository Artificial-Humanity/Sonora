# High-Ambition 2 — 🚀 "Dramatic Reader" & Full-Cast Audiobooks

> **Sequence:** 2 of 6 ([index](high-ambition-index.md)). Builds directly on the
> [1 — Matcha actor](high-ambition-1-matcha-actor.md) (needs a trained, directable,
> castable actor first). Then: [3 — Child Voices](../../../Prosodia/notes/high-ambition-3-child-voices.md) ·
> [4 — Multilingual G2P](../../../Prosodia/notes/high-ambition-4-multilingual-g2p.md) ·
> [6 — Audience STT](high-ambition-6-audience-conveyance-stt.md).
>
> _Rewritten 2026-08-02: the original was designed against StyleTTS2's style-vector
> space (LUT into style space, ECAPA verification loss on a style encoder, speaker
> classifier over the style diffusion). That mechanism is retired with the re-platform
> (git history has the full design); the durable content — the challenges and the
> runtime seam design — is kept below, mapped to Matcha's mechanisms._

## 🎯 Objective

Evolve the single-narrator actor into a **dramatic reader**: multi-voice,
multi-character performances with consistent identities across chapters — the narrator
and each character audibly distinct, emotionally directable, and stable.

## Core challenges (base-model-independent)

1. **Timbre identity vs emotional expression.** Emotion shifts must not read as
   speaker changes. Matcha mapping: identity = the 64-d speaker vector (contract:
   never an id), emotion = V/A/T through zero-init FiLM — the disentanglement is
   architectural rather than learned-and-hoped, which is precisely why the contract
   separates them. The residual risk is training data where the two correlate;
   the identity-leakage gate (ECAPA drift ≤ 0.2, ARCHITECTURE §5) is the instrument.
2. **Portrayal within identity.** A narrator *doing* a character's voice is a
   delivery excursion inside one identity, not an ID swap
   ([casting-attribute-norms-brief.md](casting-attribute-norms-brief.md) § Identity vs
   portrayal — the two-layer model). Full-cast means deliberately crossing what
   single-narrator reading must never cross: a large-enough excursion in speaker space
   *is* a cast change. The blending engine needs an explicit threshold policy —
   bounded excursions for portrayal, unbounded only on a cast switch.
3. **Character attribution / diarization upstream.** The Director must map text spans
   to characters (`{"speaker": "narrator", …}` / `{"speaker": "john_hushed", …}`).
   This is exactly the cast-sheet pass + SCM `cast` map already designed
   ([casting-attribute-norms-brief.md](casting-attribute-norms-brief.md) § cast sheet;
   [markup-schema-brief.md](../docs/markup-schema-brief.md) `quote`/`cast`), and the book-lane
   director-pass dogfoods it offline today.
4. **Acoustic boundary transitions.** Voice switches at quote boundaries risk pops,
   gain/F0 jumps, and unnatural timing. Runtime seam design (still the plan): short
   (5–10 ms) PCM cross-fades at profile switches, and programmatic breathing silence
   (150–300 ms) on narrator↔character transitions driven by the Director's payload —
   the same pause-based join pattern the chunking contract uses (ARCHITECTURE §1).
5. **Capacity.** Many high-fidelity voices in a ~150M-ceiling family without mutual
   degradation — the reason multi-speaker range is trained early (LibriTTS-R's 247
   speakers) and the reason a mid tier exists ([model-decisions.md](../docs/model-decisions.md)).

## Training-data implication

Full-cast dramatic material with real per-role voices exists license-clean: the
owner-pinned LibriVox full-cast readings and stage plays
([book-prose-lane.md](book-prose-lane.md) § Quote mining & full-cast readings), which
also calibrate the measured casting norms. Role-aware alignment (per-section reader
metadata + light diarization) is the prerequisite that lane already records.

## Status

Design note — sequenced after goal 1 ships. Nothing here is scheduled; the
prerequisites being built now (cast sheet, SCM quote spans, casting norms, full-cast
corpus lane) are all owned by their own briefs.

# Singing Roadmap (opened 2026-07-17, owner-commissioned)

Where singing fits Sonora's arc: not a singing model — a **dynamic-voicing model whose
narrator can cross the speech-song border when the text calls for it** (a lullaby to a
child, a keening grief, a chanted vow), with true score-conditioned singing held as a
Sonora-heavy option that must not contort the small-model contract today.

## The two walls, one of them new

1. **Performance license wall** (standing): recordings must be CC-BY-4.0 or freer. The SVS
   dataset world is NC-saturated (Opencpop, M4Singer, GTSinger); the clean island is
   VocalSet (CC-BY-4.0, ~10 h of vocal technique — sustained vowels, vibrato, belt,
   breathy; almost no lyrics).
2. **Composition wall** (NEW — singing only): a sung performance embeds the *musical work*
   (melody + lyrics), which carries its own copyright independent of the recording. Even a
   CC-BY performance of a protected song is encumbered. Clean paths: public-domain songs
   (traditional folk, hymns, pre-1929), or **melodies + lyrics we author in-repo** — the
   script-bank pattern already solves this for words; it extends to tunes.

## Phases

### S0 — Border probe (immediate, ~zero cost)
Add to the current pilot bank: a Dia `(sings)` lullaby line and a Qwen sing-song instruct
line (authored lyrics). Owner ear-audit answers: can our ratified teachers produce usable
melodic speech at all, and does affect survive it? No new tooling; VAT labels as usual.

### S1 — Melodic-speech register family (corpus phase, near-term)
If S0 passes: a register family in the bank — *lullaby* (V+ A− T−, the tenderness corner
with pitch), *keening/lament* (V− A mid, the grief-song border), *chanted vow* (the
fierce-devotion corner with meter), *sing-song whimsy* (already brushed by
victory_whimsical). All diegetic — a narrator singing inside narration — which is what
dynamic voicing meets in real books. New QC instrument: f0 contour statistics (librosa
pyin — pitch range, note-stability, melodicity index); cheap, and doubles as a general
prosody measure. Bounded-minority rules apply as ever.

### S2 — Clean singing data survey (parallel, low effort)
Inventory wall-clean sources before wanting them: VocalSet ingestion test (technique
vocalizations as T/A-extreme material even without lyrics); public-domain-song + CC-BY
performance hunts (expect slim pickings); synthetic melodic-speech from S1 as the
realistic volume source. Deliverable: a sourcing brief with hour counts, not downloads.

### S3 — Score-conditioned singing lane (Sonora-heavy, GATED)
True SVS needs pitch-as-content: an f0/note-target conditioning lane added to the
Director-Actor contract (contract v2 material — new lane beside op_g2p, VAT still applies
on top; a sung phrase has V/A/T too). Explicitly OUT of scope for Sonora-mini; opens only
after the 3-channel VAT training proves the base contract, per model-family-strategy
corner-risk discipline. Prior art to study then: DiffSinger/openvpi, VISinger2, NNSVS;
the commercial bar is Synthesizer V / ACE Studio.

## Gates
- S0 → S1: owner audit says teachers produce melodic speech with surviving affect.
- S1 → S2 scale-up: melodic-speech keeps pass QC + blind audit like any register.
- S3 opens: 3-channel VAT training converged AND owner elects the -heavy tier.

# High ambition 7 — Singing

_Opened 2026-07-17 as `singing-roadmap.md` (owner-commissioned). **Promoted to the
high-ambition series 2026-08-09 (owner)** — it is a goal to be tackled down the road, not a
near-term roadmap phase, and filing it as a roadmap made its first step look due. Home:
Sonora, because it changes the MODEL and the CORPUS rather than the app
([high-ambition-index.md](high-ambition-index.md))._

Where singing fits Sonora's arc: not a singing model — a **dynamic-voicing model whose
narrator can cross the speech-song border when the text calls for it** (a lullaby to a
child, a keening grief, a chanted vow), with true score-conditioned singing held as a
Sonora-heavy option that must not contort the small-model contract today.

---

## ⚠ First, the distinction this whole note rests on

**Unrequested singing is a DEFECT. Requested singing is this GOAL. They are opposite
problems and must never be conflated.**

| | unrequested singing | requested singing |
|---|---|---|
| what it is | the model sings, adds melody or adds music when nothing asked it to | the model sings *because the text and the direction called for it* |
| standing | **a measured defect class, today** | **high ambition 7 — down the road** |
| consequence | benched VibeVoice (uncontrollably sings / adds music) and Dia (same failure, to a smaller extent) | opens only under the gates below |
| what it costs | a fluent, confident, wrong render that the score cannot detect | nothing yet — it is not scheduled |

**The trap, stated plainly so nobody walks into it later:** when this goal eventually goes
live it will be tempting to read VibeVoice's and Dia's singing as an early asset — "they can
already do it." **They cannot do it *on request*, and that is the entire difference.** They
were benched for **uncontrollability**, not for inability, and a capability you cannot
withhold is not a capability. Un-benching either engine requires an answer to *"can it be
told NOT to"* — nothing in this goal supplies that, and progress here is not evidence about
them. (Their bench records, with end-conditions:
[teacher-tts-audition-shortlist.md § Benched engines](teacher-tts-audition-shortlist.md#benched-engines)
— the one record, do not restate a status here.)

The corollary matters as much: **pursuing goal 7 must not weaken the defect detector.** The
day a register family for melodic speech exists, "did the model sing?" stops being a
sufficient defect test and has to become "did the model sing *when it was not asked*" —
which is a harder measurement, and one nothing currently makes. That is a prerequisite of
S1, listed there.

---

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

⚠ The composition wall is why this goal cannot be opportunistic. Most of what sounds like
free singing data is not, and the check is one most licence surveys do not make — they
clear the recording and stop.

## Phases

**None of these is scheduled.** They are the shape the goal would take, kept so that when it
opens it does not start from a blank page.

### S0 — Border probe (cheap, whenever this opens)

Can our *live* teachers produce melodic speech **on purpose**, with affect surviving? Use
the live portfolio only (qwen · chatterbox · zonos · orpheus · moss_vg) on authored lyrics —
a Qwen sing-song instruct line and a chatterbox/zonos equivalent. Owner ear-audit. No new
tooling; VAT labels as usual.

⚠ **Do not probe a benched engine here.** The question is not "can they sing" — VibeVoice
will, unprompted, all day — but **"can they be *asked* to, and asked *not* to."** A probe
run against the engines benched *for* singing unbidden reads the defect as a pass.

### S1 — Melodic-speech register family (corpus phase)

A register family in the bank — *lullaby* (V+ A− T−, the tenderness corner with pitch),
*keening/lament* (V− A mid, the grief-song border), *chanted vow* (the fierce-devotion
corner with meter), *sing-song whimsy* (already brushed by `victory_whimsical`). All
diegetic — a narrator singing inside narration — which is what dynamic voicing meets in real
books. New QC instrument: f0 contour statistics (librosa pyin — pitch range, note-stability,
melodicity index); cheap, and doubles as a general prosody measure. Bounded-minority rules
apply as ever.

⚠ **Prerequisite, added 2026-08-09: the defect detector has to survive this phase.** Once
melodic speech is a legitimate register, "the model sang" no longer indicts a clip, and the
detector must become "the model sang **when nothing asked it to**" — i.e. it needs the
*directed intent* alongside the measurement. That is the same shape as the direction-adherence
gate already specified for Phase 1S (intended V/A/T vs measured V/A/T), and it should reuse
it rather than invent a second one. **Do not open S1 before that exists**, or the corpus
loses the ability to tell the goal from the defect.

### S2 — Clean singing data survey (parallel, low effort)

Inventory wall-clean sources before wanting them: VocalSet ingestion test (technique
vocalizations as T/A-extreme material even without lyrics); public-domain-song + CC-BY
performance hunts (expect slim pickings); synthetic melodic-speech from S1 as the realistic
volume source. Deliverable: a sourcing brief with hour counts, not downloads.

### S3 — Score-conditioned singing lane (Sonora-heavy, GATED)

True SVS needs pitch-as-content: an f0/note-target conditioning lane added to the
Director↔Actor contract (a new lane beside op_g2p; VAT still applies on top, since a sung
phrase has V/A/T too). Explicitly OUT of scope for Sonora-mini; opens only after the base
contract is proven, per the model-family-strategy corner-risk discipline. Prior art to study
then: DiffSinger/openvpi, VISinger2, NNSVS; the commercial bar is Synthesizer V / ACE Studio.

## Gates

- **Goal 7 opens at all:** owner elects it. It is behind the quality ladder — a model that
  cannot yet reliably deliver five *spoken* delivery lanes is not ready to be asked for a
  sixth mode of production.
- **S0 → S1:** owner audit says live teachers produce melodic speech on request, with
  affect surviving — *and* can be directed away from it.
- **S1 opens:** the unrequested-singing detector has been rebuilt around directed intent
  (see S1's prerequisite). Without it, this phase destroys the defect class.
- **S1 → S2 scale-up:** melodic-speech keeps pass QC + blind audit like any register.
- **S3 opens:** the base VAT contract is converged AND the owner elects the -heavy tier.

---

Linked from: [high-ambition-index.md](high-ambition-index.md) ·
[README.md](README.md). Engine bench records live in
[teacher-tts-audition-shortlist.md](teacher-tts-audition-shortlist.md) and are not restated
here.

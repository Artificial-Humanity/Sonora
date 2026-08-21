# Teacher-Synthesis Pipeline

Turns the teacher portfolio into labeled training material for the expressive corpus.
Everything here is public-lineage: engines Apache-2.0/MIT only, line texts authored
in-repo or sourced from the CC0/PG book lane. Campaign history and per-round findings
live in git history (the 2026-07 pilot/bulk daybook was cut 2026-08-02); this file
keeps the mechanics that stay true.

## Flow

```
bank (authored lines or book_ingest) ──> synth_*.py renderers ──> normalize_loudness
      │                                                                  │
      │  per-line: id, engine, register,                                 ▼
      │  intended{V,A,T}, seed, text, direction{…}                    qc_gate ── FAIL ⇒ EXIT, nothing registered
      │                                                                  │
      └──> engine roster = ref_select.ENGINE_MIX_BY_LANE − SET_ASIDE     ▼
                                                        register_audition --qc ──> audition queue
                                                                         │
                                                          keeps ──> corpus slice (bounded minority)
```

**Live engines: chatterbox · qwen · zonos · orpheus · moss_vg.** `synth_vibevoice.py`
and `synth_dia.py` still ship but are **set aside** (2026-07-29, reversible,
`ref_select.SET_ASIDE`). The interface table below keeps VibeVoice and Dia — it
documents what those interfaces DO, which stays true whether or not they are in use.

### The QC gate is MANDATORY and automatic (owner rule, 2026-07-31)

`render → normalize_loudness → qc_gate → register_audition --qc → queue`, and a
failing gate makes `synth_bank.sh` **exit without registering** — clips stay on disk,
out of the queue, with the two recovery commands printed. The gate exits nonzero on
zero manifests, zero records, or zero passes (an empty glob can no longer "pass"), and
it globs manifests at the campaign root as well as engine subdirs (hand-authored banks
are flat).

**Every QC failure is auditioned — at every trust tier, on every engine.** A QC
finding attaches as a **note** and never changes a clip's status: the instrument tells
the ear what to check, it does not decide. Findings are direction-aware, since
truncation and over-run send the auditor to opposite ends of the clip:

> `QC: ASR WER 0.66, transcribed only 20 words of 47 — CHECK THE END, text may be MISSING`
> `QC: ASR WER 0.42, transcribed 52 words but the passage has 37 — CHECK FOR AN IMPROVISED OR REPEATED TAIL`

### Gate composition, and why WER is not enough

- **`asr_ok`** (WER ≤ 0.35) is a global error rate and structurally cannot see a tail
  truncation: a clip that lost its final 19 of 139 words scored WER 0.24.
- **`tail_ok`** asks WHERE the render stopped: difflib-align hypothesis to reference,
  fail if >5% of the passage and ≥3 real words remain unspoken. ⚠ Threshold not yet
  ear-calibrated ([todo.md §4](todo.md)).
- **Dead-air** (Silero, 2026-08-02): hard gate at 2.5 s internal silence (no good clip
  has ever landed there) + an **advisory band from 1.4 s** that queues the clip for the
  ear with a where-to-listen note rather than condemning it — a long pause is sometimes
  a real dramatic choice. Calibrated on 369 audited clips: catches 20/21 known
  pause-drops at a 3.3% keep-flag rate. Reading `hard_pass` alone understates the
  gate — the advisory band is what catches real drops under the 2.5 s cap.
- **DNSMOS** is a quality tier, not the primary gate — it is register-biased
  (whisper/laughing/crying score low across all engines) and cannot separate
  "expressive" from "broken" in the 2.0–2.6 band. ASR fidelity is the collapse
  detector.
- A manifest APPENDS a record per render, so a rerolled clip has several — **last
  record wins**; it is the take that exists on disk.

Not yet built: a `head_ok` gate — nothing sees head truncation (`HEAD_LOST_MAX = None`
in `scripts/stages/qc_gate.py`, left unset on evidence rather than by omission).
⚠ **The 4 s speech floor IS hard-gated on the Silero measure**, and this sentence said it was
not until 2026-08-21: `speech_dur_vad = silero_vad.speech_duration(wav16)` feeds
`gates["speech_ok"] = speech_dur_vad >= SPEECH_MIN_SECONDS`. The constant is now registered
with the doc-claims gate (#189), so its VALUE can no longer drift from these notes — but a
sentence denying the gate exists is a class that gate still cannot see.

## Control interface — how text becomes *directed* audio (per engine)

There is no separate "tagging" layer: the mechanism is a per-line **`direction`**
object in the bank, and it is a **different control interface per engine**. Every line
also carries `intended: {V,A,T}` (numeric target) + `register` (from the controlled
lexicon) — generation-conditioned intent, verified downstream by the instruments. A
bank line = `{id, engine, register, intended{V,A,T}, seed, text, direction{…}}`.

| Engine | Control channels (in `direction`) | How it's passed |
|---|---|---|
| **Qwen** (`synth_qwen.py`) | ONE merged `instruct` string carrying voice **and** delivery; `text` = the words | `generate_voice_design(text=, instruct=, language=)` — there is **no** `voice_description` parameter; a separate `design` key lands in `**kwargs` and is silently dropped |
| **MOSS-VoiceGenerator** (`synth_moss_vg.py`) | ONE merged `instruct` string. **THE MOSS for all directed work** | `processor.build_user_message(text=, instruction=)` |
| **Zonos** (`synth_zonos.py`) | Numeric dials: `pitch_std`, `speaking_rate` (≥14 prose, ≤18), 8-float emotion vector **or `emotion: null`** → compiled to true unconditional (`unconditional_keys`) | `make_cond_dict(...)` — omitting emotion still conditions on the default neutral; null is the real off switch |
| **Chatterbox** (`synth_chatterbox.py`) | `exaggeration` + `cfg_weight` (coarse dial, not prose); casting via reference clip through `ref_select` (blacklist + `MAX_REF_EXCURSION` guard) | classic Chatterbox, NOT Turbo (Turbo discards the dial) |
| **Orpheus** (`synth_orpheus.py`) | `-ft` checkpoint: named voices (banned set in code — `tara` never) + emotion tags; prompt must append `128261` + `128257` | |
| VibeVoice *(set aside)* | **No natural-language slot.** `design` is casting metadata only — `ref_select` picks a reference clip; `instruct` never reaches the model | `"Speaker 0: {text}"` + reference wav |
| Dia *(set aside)* | **No instruct slot; only `render_text`**: `[S1]` tags, the closed 21-tag non-verbal set, punctuation. `temperature` **1.8 — validated, do not lower** (1.5 collapsed 7/20 to noise); token budget `1.35x + 1.0 s + 1.2 s/tag` (real speech rate ~11.6 chars/s) | |

**`build_direction()` in `scripts/lib/book_ingest.py` is the single source of
truth for each engine's payload** — bank builders must not invent keys. Unknown engines
are fatal (the silent rewrite-to-vibevoice fallback is gone). Each renderer writes
`<engine>_manifest.jsonl` echoing the full direction, seed, and license per clip.

Interface traps and per-engine gotchas: [tts-engine-onboarding.md](../docs/tts-engine-onboarding.md).

## Director architecture (two passes)

Labels describe the LINE; direction describes the ENGINE. They are separate calls —
when they were one, the training labels drifted with whichever engine Gemma was
writing for (54 distinct registers for 20 lines; identical V/A/T on only 4/20). Split,
both are 20/20.

1. **Line pass** (engine-agnostic) — `{V, A, T, register}`. `register` MUST be copied
   verbatim from `scripts/assets/register_lexicon.json` (47 labels, regenerated by
   `build_register_lexicon.py` from certified keeps); off-lexicon picks are coerced.
2. **Casting pass** (per engine) — `voice_design`/`instruct` governed by
   `scripts/assets/director_skills/<engine>.md`. The JSON schema is
   **engine-shaped**: single-string engines are asked for one field only. The engine
   roster offered to the director derives from `ref_select.ENGINE_MIX − SET_ASIDE` —
   never a hardcoded list.

Director model: **`gemma-4-31b-qat-spec`** (obedience-tested; e4b is for volume jobs,
not direction — see [book-prose-lane.md](book-prose-lane.md) § Director model).

**Minimum clip length is 4 s of estimated speech** (`MIN_CLIP_SECONDS`,
`book_ingest.py`) — it gates the INPUT TEXT, not rendered duration. Completeness
gates every lane (`is_complete_utterance()`): short-but-whole is ratable, incomplete
is not, and the score cannot detect the defect.

## Rating vocabulary v4 (owner 2026-07-26)

The audition score is **vocal and prosodic quality only** — a single 1–5 axis, not a
composite verdict. Categorisation (register, gender, accent, delivery) is corrected
via per-attribute dropdowns in the audition app and never folded into the number.
This is what makes scores comparable — but only for v4-era rows (`teacher-ab-v1`
onward); any analysis treating `score` as quality must scope itself to those
campaigns. Scale history for older rows: pre-v3 batches in `ratings_history.csv` used
1–10 (6+ = keep), mapped 6→1 … 10→5 on 2026-07-18; the Recategorize action was retired
with v4.

## Principles (owner-validated)

1. **Generation-conditioned labels, verified by instrument.** Intent is never trusted;
   EIV + phonation/LUFS measures must confirm the intended direction or the clip is
   dropped/relabeled. Instrument z-scales are normalized WITHIN engine
   (neutral-anchored) — per-engine channel offsets otherwise swamp the signal.
2. **QC gate on every clip** — synthetic clips are guilty until proven.
3. **Register diversity by design** — the controlled lexicon, sampled across register
   families; "not every depiction should be grief at the breaking point."
4. **Stage-2 transfer multiplies, never originates** (LongCat: anchors are QC-passed
   stage-1 clips, labels inherited then re-verified; synthetic anchors only).
5. **Bounded minority** — synthetic material stays a bounded minority of any training
   corpus; LibriTTS-R + real-audio keeps remain the base.

## Layout

- `scripts/assets/script_bank.json` — versioned authored line bank
- `scripts/stages/synth_<engine>.py` — renderers (`synth_common.py` = shared atomic
  wav/manifest plumbing)
- `scripts/assets/director_skills/<engine>.md` — the Gemma adapter per engine
- `scripts/assets/register_lexicon.json` + `build_register_lexicon.py`
- `scripts/lib/ref_select.py` — reference casting + `ENGINE_MIX_BY_LANE` +
  `SET_ASIDE` + blacklist (three-layer allocation, 2026-08-02)
- `scripts/stages/qc_gate.py` · `silero_vad.py` · `register_audition.py` ·
  `pick_audit_subset.py` · `audit_sampler.py` — the gate + audit chain
- Output: `/data/model-training/datasets/synthetic_v1/<campaign>/<engine>/`

## Standing directive: the dataset is a first-class project (owner, 2026-07-18)

"Since we put time and effort into these, let's continue to grow and refine this
dataset. It maximizes the return on our efforts and our return contributions to the
open source community that we ourselves gain from."

- The expressive-registers dataset is a LIVING deliverable — every campaign grows and
  refines it.
- **Publication**: certified keeps only, HF under **CC-BY-4.0**, full provenance
  manifests (engine, direction, seed), versioned per campaign close. The owner bar
  ("affect obvious without the keyword") is the release gate, not just the instrument
  bar. LibriVox-derived clips train Sonora but are **never published in this set** —
  the published artifact is own-synthesis-only (owner rule 2026-07-22).
- Nothing encumbered ever enters: engines Apache/MIT only, texts authored in-repo or
  CC0/PD, synthetic voices only.

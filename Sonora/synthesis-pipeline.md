# Teacher-Synthesis Pipeline (phase opened 2026-07-17)

Turns the ratified teacher portfolio (teacher-tts-audition-shortlist.md, FINAL section) into
labeled training material for the V slice (plus T support and embodiment). Everything here is
public-lineage: engines Apache-2.0/MIT only, line texts authored in-repo (original prose).

## Flow

```
script_bank.json ──> stage-1 renderers ──> QC gate ──> keeps + verified labels
 (authored lines,     synth_dia.py          qc_gate.py       │
  intended V/A/T,     synth_qwen.py         + eiv_score.py   ├──> corpus slice (bounded minority)
  per-engine          synth_moss85.py                        │
  direction)                                                 └──> anchors ──> stage-2 LongCat
                                                                   transfer (synth_longcat.py)
                                                                   ──> QC gate again
```

## Control interface — how text becomes *directed* audio (per engine)

There is no separate "tagging" layer: the mechanism is a per-line **`direction`** object in the
bank, and it is a **different control interface per engine**. Every line also carries `intended:
{V,A,T}` (numeric target) + `register` (human label) — these are *generation-conditioned* (the
intent, not a measurement) and are verified/relabeled downstream by the instruments (principle #1).
A bank line = `{id, engine, register, intended{V,A,T}, seed, text, direction{…}}`.

| Engine | Control channels (in `direction`) | How it's passed (`synth_*.py`) |
|---|---|---|
| **Qwen** (`synth_qwen.py`) | `design` = natural-language **voice description**; `instruct` = natural-language **delivery instruction**; `text` = the words | `generate_voice_design(voice_description=design, text=text, instruct=instruct, language="English")` |
| **MOSS-8.5B** (`synth_moss85.py`) | `instruct` = one bundled instruction anchoring **voice + delivery** together; optional `quality` | `processor.build_user_message(text=text, instruction=instruct, quality=…)` |
| **Dia** (`synth_dia.py`) | **No instruct channel.** `render_text` embeds control **inline**: `[S1]` speaker tag + nonverbal tags (`(sings)`, `[laughs]`) + words; plus `temperature` (≥1.5 floor), `guidance` (3.0) | `processor(text=[render_text])` → `generate(temperature=, guidance_scale=, max_new_tokens=token_budget(...))` |

Takeaways that matter for any new text front-end (e.g. book-prose ingest): **the "direction" is
what carries delivery**, it is engine-specific (Qwen/MOSS = a natural-language *instruction*; Dia =
*inline text tags* + sampling), and **`intended{V,A,T}` is the label to emit** — producing those
fields from source text is exactly what a `book_ingest` stage must do. Each renderer writes a
`<engine>_manifest.jsonl` echoing the full `direction`, seed, and license per clip (provenance).

## Principles (all owner-validated during the audition)

1. **Generation-conditioned labels, verified by instrument.** Each line carries intended
   V/A/T. Instruments (EIV heads via scripts/eiv_score.py, phonation composite + LUFS via
   derive_vat_corpus measures, LibriTTS-anchored z like the Emilia mining) must CONFIRM the
   intended direction or the clip is dropped/relabeled. This neutralizes MOSS-8.5B's
   instruct-adherence drift and Dia's improvisation: we never trust intent, we measure.
2. **QC gate on every clip** (synthetic clips are guilty until proven): DNSMOS floor
   (computed locally — Microsoft P.835 ONNX), duration-vs-text sanity (catches Dia
   improvised tails and collapse-to-noise), engine floors (Dia temp >= 1.3), instrument
   label check. Owner blind-audits a stratified sample per campaign, Emilia-style —
   acceptance standard: affect obvious without the keyword.
3. **Register diversity by design.** The bank samples each emotion across the owner's
   register taxonomy (grief: processed / here-and-now / vengeful; threat: direct / noir /
   warning; victory: peak / everyday / whimsical; plus protective urgency, neutral
   narration, embodiment). "Not every depiction should be grief at the breaking point."
4. **Stage-2 transfer multiplies, never originates.** LongCat anchors are stage-1 clips
   that PASSED QC. Anchor labels are inherited, then re-verified (transfer re-renders
   character; affect must survive). Synthetic anchors only — never a real person's voice.
5. **Bounded minority.** Synthetic material stays a bounded minority slice of any training
   corpus (audiobook-corpus-policy.md precedent); LibriTTS-R + Emilia keeps remain the base.
6. **Dia staging note**: two-speaker scene staging (the mugging trick) both coaches register
   and disciplines length, but produces mixed-affect dialogue clips; until a segmentation
   step exists, production Dia jobs are single-speaker (beats + tags), staging reserved.

## Layout

- `Sonora/scripts/synthesis/script_bank.json` — versioned line bank (id, engine, register,
  intended V/A/T, text, per-engine direction, seed).
- `Sonora/scripts/synthesis/synth_{dia,qwen,moss85,longcat}.py` — per-engine renderers
  (throwaway-container pattern, recipes from the audition); each writes
  `<out>/<engine>_manifest.jsonl` alongside wavs (full provenance per clip).
- `Sonora/scripts/synthesis/qc_gate.py` — DNSMOS + duration sanity + phonation/LUFS
  measures; emits measures.jsonl + filelist for eiv_score.py; verdict merge produces
  keeps + audit sample.
- Output: `/data/model-training/datasets/synthetic_v1/<campaign>/<engine>/`.

## Standing directive: the dataset is a first-class project (owner, 2026-07-18)

"Since we put time and effort into these, let's continue to grow and refine this dataset.
It maximizes the return on our efforts and our return contributions to the open source
community that we ourselves gain from."

Consequences:
* The synthetic expressive-registers dataset is a LIVING deliverable, not a byproduct —
  every campaign grows it (new registers, deeper pools, better takes) and refines it
  (re-rolls of weak takes, relabels from owner audits, improved QC certification).
* **Publication plan**: certified keeps only (hard-pass + instrument-confirmed labels +
  surviving owner audit), released on HF under **CC-BY-4.0** with full provenance
  manifests (engine, direction text, seed → reproducible/extensible by others), dataset
  card documenting the register taxonomy, V/A/T label semantics, and QC pipeline.
  Versioned releases per campaign close (v1 = bulk-1 keeps). Repo name = owner's call
  (working suggestion: artificial-humanity/sonora-expressive-registers).
* Quality bar for the public set is the OWNER bar, not just the instrument bar — the
  blind-audit standard ("affect obvious without the keyword") is the release gate.
* Nothing encumbered ever enters: engines Apache/MIT only, texts/lyrics authored
  in-repo, synthetic voices only (no consent surface). This is the wall, paid forward.

## Status
- 2026-07-18: **rating system v3 (owner redesign, supersedes v2).** Score vocabulary for
  ratings.csv (SSOT at dataset root; xlsx retired):
  * **x** — not a keeper. Drop.
  * **0** — possible keeper with a fixable production defect (e.g., voice pitch):
    re-create with the noted change and re-test.
  * **x1–x5** — wrong register, real quality: conveyance rates 1–5 but the emotion is
    not the labeled one (e.g., a grief_herenow take that is actually a very good
    "tired" → x4). Rename/recategorize; the original direction text becomes evidence
    of what the TRUE register's directions sound like.
  * **1–5** — keeper at that quality level.
  v2's "5 = actionable, never silent" is superseded: its two actions are now explicit
  codes (0 = re-roll, x# = relabel). A relabel that ALSO needs a re-roll is just a
  **0** — the re-roll gets evaluated on its own merits.
  **Quality = training weight (owner intent, confirmed feasible):** score drives
  sampling exposure — 5's are seen most, 1's stay in the mix. Implemented at
  filelist-build time (per-clip repeat factor / weighted sampler); score also ships in
  the public manifest as a `quality` field so downstream users can weight too.
  **Migration of pre-v3 scores (1–10 scale, 6+ = keep) — owner-confirmed, APPLIED
  2026-07-18** to ratings.{csv,xlsx} (74 rows): 6→1, 7→2, 8→3, 9→4, 10→5; reroll
  status → 0; dropped → x. The 3 already-relabeled rows (narration_low_western, tired,
  neutral_narration) had only the v2 action-flag "5" (no real quality) — scores blanked
  and flagged RATE ME for re-scoring under v3. How-To legend rewritten for v3.
  ratings_history.csv / History sheet untouched (append-only; pre-v3 batches remain on
  the 1–10 scale, interpret via this mapping).
  **Go-forward: the scale is strictly 1–5 (owner, 2026-07-18).** 6–10 are retired from
  the vocabulary entirely — a former 10 is a 5, former 9 a 4, etc. (the mapping above is
  now just history). **Re-rating is unrestricted:** any already-rated clip can be
  re-scored at any time (the Auditions "Re-rate" view lists every rated clip, paginated,
  each showing its current score) — there is no special former-9 recalibration queue.
- 2026-07-18: **round-3 audit (46 clips: transfer-1 + b1 pending) + rating system v2.**
  Owner protocol change: **5 = actionable, never silent** — either re-roll with the noted
  tweak OR relabel to the register it actually fits ("it adjusts how we look at the
  dials"). Running ledger now lives at sonora-expressive-registers/ratings.{csv,xlsx}
  (+ append-only ratings_history.csv); xlsx has relative links for click-to-listen;
  owner may re-rate on later listenings.
  Results: 36/46 at 6+, **dataset v1 rebuilt: 131 clips / 16.1 min** (2x growth from
  transfer + pendings). TWO first-ever 10s: moss narration_03 ("classic... useful for
  almost any type of book") and — notably — a TRANSFER (oratory_02): stage 2 produces
  top-tier material. Chant transferred BETTER than its source (7 -> 9).
  5-protocol discoveries: **tired** (from a dismissive take), **narration_low_western**
  ("Cold Mountain" narrator, from a failed vengeful), **sermon/preacher** concept (from
  fierce_devotion — "replace the words with a psalm... perfect preacher"; needs
  distortion-free re-roll). Dia's voice lottery = **accent diversity feature**
  (midwestern/UK/southern noted approvingly at 8-9).
  Systematics for bulk-2: bright Qwen voices sit a HALF-STEP too high (tweeter edge
  distortion on AirPods — physical, not aesthetic); MOSS-anchored LOUD transfers carry
  edge distortion (excited/fierce family) — watch in stage-2 QC; melodic family currently
  lands as "beautiful soft poem-reading," not singing — true song remains S1-open.
- 2026-07-18: **transfer-1 (stage 2, first production run): 63 anchors -> 63 renders,
  62/63 hard-pass (98%; zero WER across the board — transfers speak their NEW text
  cleanly), 44/63 instrument keeps (70%).** Survival map: performance-forward registers
  transfer perfectly (embodiment 8/8, keening 6/6, arrogance 6/6, lullaby 3/3,
  narration 4/4); the soft family shows the SAME instrument blindness as stage 1
  (tenderness/dismissive/processed-grief transfers are structurally clean HPs awaiting
  ears). Stage-2 economics confirmed: ~1.7x dataset growth per pass at zero authoring
  cost, pending owner audit of the contested slice. Review batch staged in
  audio-review/: 34 transfer clips (all non-keeps + per-register keep spot-checks) +
  bulk-1's 12 pending clips; blank RATINGS.txt ready.
- 2026-07-18: **bulk-1 owner audit complete (numeric protocol v1: 1-10, 6+ keeps).**
  25/28 audited keepers; **dataset v1 = 63 clips (~8 min), 12 pending re-audit
  (unaudited-family edge cases, fold into b2 audit), 3 dropped.** Patterns:
  * **Qwen swept the soft family** the instruments couldn't hear: dismissive 9/9,
    processed grief 8/7, tenderness 8, intimate warmth 9, lullaby 9 — HP tags were
    instrument blindness, confirmed. MOSS's soft/contained takes failed by ear
    (tenderness 4 "forced, disinterested"; vengeful 5 "monotone and lazy"; lazy chant 5)
    — **soft registers reassign to Qwen; MOSS keeps force/dark/oratory** (oratory 9,
    noir 9, arrogance 8, fierce 8).
  * **Instruments produce false-POSITIVES on lazy contained takes** (vengeful was a
    certified KEEP; ears heard monotone) — cold+flat measures as correct V−/T+ but
    performance quality is invisible to instruments. Ears stay the release gate both ways.
  * Direction tweaks for bulk-2: mellow the Qwen bright-register voice a notch (owner:
    conveyance very good, voice "a tad extreme"); push chant instruct toward song
    (current reads as recited poem); fierce_devotion_01 note "intent there, not fierce."
- 2026-07-18: **bulk-1 rendered + gated + verdicted.** 159 jobs (Qwen 87 / MOSS 56 /
  Dia 16), 148/159 hard-pass (93%, vs pilot 63%), 69 full instrument keeps
  (neutral-anchored per-engine recentering). ~19 min rendered, ~9-12 min keep-grade.
  Findings:
  * ASR-gate + resume + fixes all held at volume; Dia's shout register failed wholesale
    in production (all 6 victory takes) — Dia narrows to narration/urgency/lottery duty.
  * Per-engine recentering v2 = NEUTRAL-ANCHORED zero (full-pool mean is biased by
    register composition). Extremes verify superbly (keening 6/6, embodiment 8/8,
    arrogance 6/6); SOFT registers are below instrument resolution (tenderness,
    dismissive, processed grief barely differ from calm narration acoustically) —
    ears are the release gate for the soft family, as the standing directive says.
  * 28-clip stratified owner audit delivered (KEEP vs HP tags; contested registers
    doubled). Owner verdicts will set dataset v1 membership.
- 2026-07-17: pilot campaign `pilot` — 15 lines (5/engine) across 10 registers, including
  two embodiment lines (Qwen + MOSS). Purpose: validate renderers, QC gate, and label
  verification end-to-end on owner-auditable volume before any bulk run.
- 2026-07-17: pilot rendered + gated + verdicted end-to-end (8/15 hard-pass, 3/15 full
  keeps). **Calibration findings (the pilot's actual product):**
  1. **Dia token budgeting works** (durations text-proportional, tails bounded) and is now
     in synth_dia.py; Dia still fills whatever room it gets, so the budget is the control.
  2. **DNSMOS is register-biased**: laughing/crying/whispered clips score low across ALL
     engines (whimsical 2.25, whisper 2.05, breathless 3.17) — the 3.3 mining floor was
     calibrated on clean read speech. Pending owner's blind listen: move to register-class
     floors (narration 3.3 / expressive ~2.8 / whisper own tier) with an absolute collapse
     floor. Dia threat_direct at ~1.2 on two independent renders = likely true collapse at
     temp 1.3 (the cliff the coaching experiment mapped).
  3. **Instrument z-scales need per-engine normalization**: measured T z-values of ±5-11
     against raw LibriTTS anchors = channel offsets per engine (codec/sr/style). Same
     lesson as per-speaker T re-z and within-cluster V mining — normalize WITHIN engine
     before direction checks. Needs bulk-campaign volume (5 clips/engine is too few for
     stable engine stats); pilot-2 fix.
  4. **V+ detection is weak** (all three victories measured V<=0 while A measured hugely
     positive) — consistent with the Emilia V+ mining failure. Options for pilot-2: direct
     Elation/Amusement-head check for positive registers, or accept
     generation-conditioning + owner audit as the V+ label authority.
  5. Verdict boundary bug fixed (|intended| = 0.3 now counts as weak intent, inclusive).
  Owner blind-audit of all 15 clips in flight — ears calibrate the floors.
- 2026-07-17: **owner blind-audit of the pilot (full verdicts logged here; the floors are
  now calibrated by ear).** Score: 12/15 clips pass on conveyance (Qwen 5/5, MOSS 5/5,
  Dia 2/5).
  * **DNSMOS register bias CONFIRMED** — every ears-passed expressive clip the gate had
    flagged (whimsical 2.25, crying 3.13, breathless 3.17) is good. But the whisper-bias
    guess was WRONG: dia_protective_urgency (2.05) is true white-noise collapse and
    dia_threat_direct (1.26) is wordless non-speech. DNSMOS can't separate "expressive"
    from "broken" in the 2.0-2.6 band → **replace DNSMOS-as-primary-gate with an ASR
    fidelity check** (transcribe, compare to script; catches collapse, wordless output,
    half-empty files, improvised tails, all in one instrument). DNSMOS floor drops to a
    quality-tier role (narration 3.3 / expressive 2.5 advisory).
  * **Dia production reliability**: 2/5 collapses at temp 1.3-1.4 (audition used 1.8 —
    the lowered temps were the mistake; the 1.3 "floor" from the coaching run is
    cliff-adjacent, not safe). Temp floor raised to 1.5, default 1.8; register control
    belongs to text/staging, not temperature. grief_processed_01: 12 s file, 4 s speech
    (silence padding) → QC now measures effective speech duration. victory_peak: emotion
    perfect, but background inhale artifact + overlong pause — QC-passable, register
    documented.
  * **Owner relabels (labels are register-corrected, not discarded)**:
    grief_herenow_01 → *fear_impending_loss* ("fear and concern over impending loss" —
    a new register; 01 and 02 play as consecutive scenes); victory_everyday →
    *excited_good_news* ("the longshot job called... victory is too strong a word");
    victory_whimsical → conveyance fantastic, VOICING niche-extreme ("too much for all
    but a tiny few niche scenarios") — re-design the voice, keep the register.
  * **Embodiment: Qwen is the lead** ("Wow... phenomenal achievement of a voice emulating
    a voice"); MOSS passes but shift is barely audible. Qwen owns embodiment synthesis.
  * MOSS-8.5B: all five passable-to-excellent, incl. a narration the owner would accept
    for a full audiobook. Dia neutral narration "among the best of our female voicings."
- 2026-07-18: **pilot round 2** — full 24-line bank (attitude registers, fierce_devotion,
  lullaby probes) rendered, ASR-gated (19/24 hard-pass), EIV'd (+Affection head),
  verdicted (9/24 full instrument keeps). Owner audit of 14 unheard clips in flight.
  Headlines:
  * **Qwen lullaby_probe_02 is a FULL instrument-certified keep** (WER 0.00 on sung
    lyrics; V+/A−/T− all confirmed) — singing-roadmap S0 structurally passed on the Qwen
    path. Dia's `(sings)` take collapsed (wordless, DNSMOS 1.47): the tag is
    production-unreliable, like its cold-menace register (threat_direct collapsed again
    at temp 1.5 — retired from Dia, menace stays with MOSS).
  * **passionate_vow_01 measured V=+2.09** — first time the V-combo confirmed a
    fierce-positive (shouted love). fierce_devotion_01 (MOSS) showed the expected V+
    false-negative (−1.91) with A/T confirmed — owner audit is label authority there,
    as pre-declared.
  * Temp fix validated: protective_urgency whisper now real speech (WER 0.06).
  * ASR-gate register bias found at the extreme: ecstatic shouting over Dia's crowd
    artifact defeats Whisper (victory_peak WER 0.67 on an owner-praised take) — owner
    override stands above every instrument, by design.
  * Recurring systematic across all verdicts: per-engine channel offsets inflate/skew
    z-scales (Qwen A always positive, MOSS T always high, T magnitudes ±5-20) —
    per-engine normalization is THE pilot-2 calibration, needs bulk volume.
- 2026-07-18: **owner audit round 2 — PILOT CLOSED. 12/14 usable; the two deaths (Dia
  cold-menace, Dia sings) were already retired.** Verdicts and consequences:
  * **SINGING S0 OWNER-PASSED (Qwen path)**: lullaby_probe_02 "very very impressive...
    a good start to build from for proper singing. Might also be the most tender of
    tender that we've come up with" — melodic speech with surviving (peak!) affect.
    S1 melodic-speech register family is UNLOCKED. Dia's take: "eerily melodic...
    not intelligible" — confirms instrument verdict.
  * **Register champions crowned**: intimate_warmth (Qwen) "fantastic and romantic...
    Juliette lying with Romeo"; arrogance (MOSS) "Fantastic. Arrogance to the tee";
    tenderness_01 (Qwen) very high quality (borders intimate — the V+/A−/T− corner is
    a continuum, fine).
  * **Owner relabels round 2**: MOSS fierce_devotion → *impassioned_oratory* (NEW
    register — "presidential campaign" speech; very good, but lacks warm edges; the
    warm-edged male fierce-devotion specimen is still open). Dia grief_processed take →
    *simmering_resentment* candidate ("jilted wife... simmering anger" — register drift
    that discovered a register; has noise artifact + long gaps, re-roll before keeping).
    Qwen dismissive → irritated-adjacent (acceptable; opening grunt noted).
  * **Fairy-voice systematic (Qwen)**: passionate_vow "top notch" conveyance but the
    cartoon voice returned on a high-A register — apply the MOSS anchoring lesson to
    Qwen voice DESIGNS in bulk (concrete anchored descriptions for all high-A lines).
  * **MOSS murmur lo-fi artifact**: tenderness_02 sounds tape-recorded (conveyance very
    good — "apology on his estranged wife's answering machine") — try the flagship's
    `quality=` dial on soft registers next round.
  * Dia pacing pathology recurring (long inter-phrase gaps on victory/grief takes) —
    known, QC-visible via speech_dur/duration ratio; acceptable per-take, watch in bulk.

**PILOT COMPLETE.** Next: bulk campaign (bank expansion to ~40-60 lines/register-family,
per-engine z-normalization from bulk volume, Qwen anchored designs, MOSS quality dial,
S1 melodic family) → LongCat transfer expansion of certified keeps → bounded slice into
the v3 corpus merge.

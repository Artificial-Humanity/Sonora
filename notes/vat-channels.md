# VAT channels — mechanism, labels, semantics, outcomes

> **Consolidated 2026-07-26** from `vat-conditioning-design.md` (§1) +
> `vat-corpus-and-labels.md` (§2) + `tension-semantics.md` (§3) — git history has them under
> those names. Three closed decision briefs about ONE object, the three conditioning channels,
> that cross-referenced each other in every footer: a reader chasing "what does T mean and
> where do its labels come from" needed all three open.
>
> **The sections below are §1/§2/§3 of this file.** Where the original briefs pointed at each
> other by filename, they now point at the section. (A rename pass in July rewrote those
> cross-references into links to this file itself; fixed 2026-08-06.)

**Current standing (2026-07-22, `vat3-24k`):** energy **PASS**, tension **near-pass**, valence
**FAIL** — attributable to corpus labels, not training. That failure is what opened the directed
teacher-synthesis lane. The T axis was rescoped **LAX ↔ TIGHT** on 2026-07-20; "breathy" and
"strained" vocabulary is reserved for data with genuine aspiration.

**Resolved (2026-07-28):** for own-synthesis rows the normalisation unit is the
engine/voice, not "speaker" — loudness correlated with **engine** across a ~5 dB RMS
spread, which would have injected a spurious arousal signal. `normalize_loudness.py`
(−23 LUFS) is wired into `synth_bank.sh` before QC and registration, and its failure is
fatal; the teacher-ab-v1 keeps were retro-normalised 2026-08-02. Nothing was clipping.

---

## §1 Mechanism — FiLM conditioning

*(Normative for any future training run. This is the section people grep for mid-implementation;
`casting-attribute-norms-brief.md` points here for "the FiLM pattern".)*

_Owner call, made in the Track-1 design conversation on 2026-07-14. This is the spec for the
milestone-3 conditioning code (the "one real model-code prerequisite" in next-steps §B)._

## Decisions

1. **Mechanism: full FiLM blocks** (scale + shift), not concat-only and not AdaLN.
   A dedicated FiLM layer after **each conformer block in the text encoder** and **each block in
   the CFM decoder**: `x = x * (1 + γ) + β` with `γ, β = head_i(trunk(vat))`. Chosen over the
   minimal follow-the-grain option for control authority (scaling, not just shifting), and over
   AdaLN-Zero for toolchain safety — AdaLN rewires LayerNorm, the exact territory of the
   2026-07-12 onnx2tf export bug.
2. **Per-token conditioning shape from day one.** The VAT input is a sequence `[B, 3, T]`
   (T = token axis in the encoder, mel-frame axis in the decoder), even though training labels
   are per-utterance (broadcast at train time). This is what makes the mid-sentence hush — the
   Prosodia per-span contract — an inference-time capability instead of a future architecture
   change. Known risk, accepted: within-utterance variation may need a later fine-tune pass to
   be fully obeyed; utterance-level obedience is what the corpus can verify first.
3. **Conditioning dropout, p ≈ 0.15, neutral = zeros.** VAT is replaced by the zero vector on
   ~15% of training steps, so (a) `VAT = 0` remains a well-trained neutral voice, (b) inference
   gains a CFG-style direction-strength knob by extrapolating conditioned vs neutral outputs.

## Implementation notes (bind the spec, keep the details flexible)

- **Zero-init identity:** the last linear of every FiLM head initializes to zero → γ = 0, β = 0
  → the network is exactly the Phase-0 checkpoint at step 0 of the warm start. Non-negotiable;
  it protects the already-validated voice.
- **Shared trunk:** one `MLP(3 → D_cond)` trunk, per-block heads. Raw `[1, 3, T]` stays the
  graph input (trunk lives in-graph) so the runtime contract is model-independent.
- **Decoder time axis:** token-level VAT is broadcast to mel frames on the host via the same
  duration/length-regulator alignment that expands `mu` — no new alignment machinery.
- **Runtime contract:** `engine.rs` already wires a `vat`-named `f32[3]` input; the host
  broadcasts span values to `[3, T]`. The split graphs each gain one input; per-span dictation
  needs no re-export later.
- **Export gate:** after the code lands, re-run the litert conversion harness + per-graph parity
  on the modified architecture *at init* (VAT = 0 must reproduce Phase-0 behavior) before any
  training — catches conversion surprises while the diff is pure plumbing.
- **§7 de-risk rides this plumbing:** the single-channel energy experiment = same code with
  V = T = 0 and A carrying the free-to-measure loudness label. No separate scaffold.

## Sequencing note

The VAT fine-tune trains on LibriTTS-R (multi-speaker), so `n_spks > 1` speaker-embedding
configuration comes along in the same run — a deliberate slice of milestone 4 pulled forward.

Linked from: [next-steps §B](../../../Prosodia/notes/next-steps.md) (VAT-conditioning item),
[STATE roadmap §3](STATE.md), [high-ambition-1](high-ambition-1-matcha-actor.md).


---

## §2 Label sourcing and the corpus decision

_The §7 de-risk verdict (PASS 2026-07-16 — ρ≈1.000, leakage ≤0.091, WER Δ ≤0.042; full
record in git history under `archive/derisk-energy-verdict.md`) closed the
architecture question; the blocker for full 3-channel VAT training is **labels**: valence and
tension need per-utterance values the current corpus doesn't have, and every input must stay
CC-BY-4.0-or-freer (open-decision tightening #3 — no NC anywhere in the lineage, which excludes
Expresso). This brief turns the dataset-landscape "derivation-pipeline thesis" into concrete
options. Licenses re-verified 2026-07-16 (web)._

## What the de-risk taught us about label quality

Weak labels are enough. The energy channel hit ρ ≈ 1.000 from crude per-speaker LUFS z-scores —
no human labeling, no fancy annotator. Conditioning dropout (p=0.15) + zero-init FiLM tolerate
label noise gracefully. The bar for V/T labels is therefore "consistent and directionally right,"
not "gold standard."

## The labeler landscape (licenses re-verified 2026-07-16)

| Resource | License | Verdict |
|---|---|---|
| **LAION Empathic-Insight-Voice** (Small/Large, 2025) | **CC-BY-4.0** (model + BUD-E-Whisper base) | ✅ The only permissive **continuous valence + arousal** labeler found. 54 regression heads incl. dedicated V/A + 40 fine-grained emotion intensities (EmoNet-Voice). Caveats: trained largely on synthetic voice-acting data; model card states research *intent* (non-binding, not a license term). |
| audeering w2v2 dim-SER / wav2small | CC-BY-**NC-SA**-4.0 | ❌ NC — excluded, including as a pipeline tool. |
| emotion2vec / emotion2vec+ | FunASR custom license | ⚠️ Commercial use allowed but nonstandard terms (unilateral revision, termination clauses); **categorical only** anyway — weak fit. |
| parler libritts-r speaker descriptions | CC-BY-4.0 | ✅ but **no valence signal** — attributes are pitch/rate/monotony/quality (arousal-adjacent only). Useful for A, not V. |
| cdminix libritts-r-aligned prosody measures | CC-BY-4.0 | ✅ per-token pitch/energy/duration — substrate for A and T derivation. |
| CREMA-D | ODbL | ⚠️ share-alike copyleft on derived databases — ambiguous for model weights; treat as excluded absent legal review. Categorical only. |
| JL-Corpus | CC0 | ✅ tiny (2,400 utt, 4 NZ speakers), categorical — useful as a **calibration/sanity anchor**, not training scale. |

## Options

**A. Pure acoustic derivation (no external model).** Extend the v0 recipe: A from LUFS/pitch
dynamics (validated), T from effort proxies (spectral tilt, rate, jitter — cdminix measures),
V from… nothing good. Valence is the one dimension acoustics-only genuinely can't reach
(a sob and a laugh have similar energy). *Low risk, but ships a weak V channel — undermines the
point of full VAT.*

**B. LAION Empathic-Insight-Voice pseudo-labeling (RECOMMENDED).** Run EIV over LibriTTS-R
(and later Emilia-YODAS for tail mining) → continuous V/A per utterance; keep the validated
LUFS-based A (or blend), derive T from acoustic proxies and/or EIV's fine-grained heads
(e.g. stress/nervousness intensities — check which of the 40 heads map to the tension
semantics). License-clean end to end. Validation before trusting it: label the JL-Corpus (CC0,
known categorical emotions) with EIV and check V/A quadrant agreement; audit ~50 clips by ear.
*This is the same "derivation pipeline" shape the landscape doc already blessed — EIV just fills
the valence hole it left open.*

**C. Categorical bootstrapping (emotion2vec + JL + CREMA-D → map to V/A).** More moving parts,
two license-review burdens (FunASR custom, ODbL), coarse quadrant-level labels. *Dominated by B.*

**D. Human labeling at scale.** Cost/time not justified when the de-risk proved weak labels
suffice. Keep human ears for the small calibration set and the final audition gates only.

## Recommendation

**B**, phased:
1. **Tension semantics first** (small design task): pin down what T means acoustically before
   deriving it — the design docs define the plumbing but not the perceptual target. Done as its
   own brief, now **§3 below** (recommends phonation tension, pressed↔breathy — awaiting the
   same owner call as this brief).
2. **Label LibriTTS-R with EIV-Large** (CPU/GPU batch job, same class as the derive_vat_corpus
   run) → `derive_vat_corpus.py v1`: V from EIV, A = LUFS z-score (validated) optionally blended
   with EIV-A, T per (1).
3. **Calibrate** on JL-Corpus + ~50-clip human audit before training on it.
4. **Train 3-channel VAT** (same warm-start recipe off the derisk checkpoint), watcher wired
   before launch per the standing policy — the eval harness already measures all three channels
   (energy→LUFS, duration→seconds, f0→pyin median; T will need a measure added).
5. **Tails (OWNER CALL 2026-07-16: in-scope for the FIRST 3-channel run, not deferred):** mine
   Emilia-YODAS (CC-BY subset) with the same labeler for high-|V|/|A|/|T| segments audiobooks
   undersample. Rationale: conveyance depth is corpus-bounded — Kokoro-era testing showed
   "excited" rendering as merely eager; channels only interpolate what the data exhibits, so the
   label extremes must correspond to genuinely extreme speech.

**DECIDED (owner, 2026-07-16): Option B approved**, including the EIV dependency with its
caveats. All questions in this brief are now settled: step 5 (Emilia mining) in-scope; tension =
phonation tension per **§3 below** (approved same day). Execution begins with
`derive_vat_corpus.py v1`.

**Calibration results (2026-07-17, JL-Corpus, 2,400 acted clips, leave-one-speaker-out):** the
pre-registered calibration step earned its keep. EIV's dedicated **Valence head FAILED** (d'=0.23
happy+excited vs sad+angry — can't even call "excited" positive); the encoder-pairing hypothesis
was eliminated (laion/BUD-E-Whisper and mkrausio/EmoWhisper-AnS-Small are the **same checkpoint**,
identical md5). The **Arousal head is strong** (d'=1.60) and **Soft_vs._Harsh moderate** (d'=0.65,
angry/assertive=harsh — adopted into tension v2). Valence was **rescued via the fine-grained
heads**: a sign-constrained 9-head weighted combo fit on JL (CC0), LOSO CV **d'=0.88**, no
arousal/harshness features (independence hygiene) — frozen as
`/data/model-training/sonora/eiv_scores/valence_combo_v1.json`. Amendment to B: **corpus V labels
come from the combo, not the Valence head.** Corpus relabeled as `data/libritts_r_vat_v2`
(30,351 rows; independence gates corr(T,A)=-0.053, corr(T,V)=-0.066, corr(V,A)=+0.023 — all PASS).
Caveat recorded: d'≈0.9 = noisy-but-directionally-right labels, the regime the energy channel
already proved workable; Emilia tails remain the depth fix.

Linked from: [dataset-landscape.md](dataset-landscape.md) (the thesis this executes) ·
[STATE.md](STATE.md).


---

## §3 Tension semantics (LAX ↔ TIGHT)

_Step 1 of the corpus plan (**§2 above**, Recommendation B) is "tension semantics first": pin
down what T **means** before deriving labels for it. This brief proposes definitions and a
recommendation. V and A are settled (V = affect polarity from EIV; A = energy/arousal, validated
at ρ ≈ 1.000 from per-speaker LUFS z-scores)._

## Requirements — what any T definition must satisfy

1. **Director-meaningful.** A director must be able to say "more tension" and mean something an
   actor would understand. T is a *direction*, not a taxonomy.
2. **Labelable at corpus scale, license-clean.** Same bar as the energy channel: weak labels
   suffice, but they must be derivable over LibriTTS-R (and later Emilia-YODAS) with
   CC-BY-4.0-or-freer inputs and tools whose terms permit it.
3. **Harness-measurable.** `eval_harness.py` needs a `produced` measure computed from rendered
   audio (as LUFS is for energy), or the §7-style controllability gate can't run.
4. **Independent of V and A.** The whole point of a third channel. If T's labels correlate
   strongly with A's in the corpus, FiLM will learn a redundant loudness knob. Requirement:
   per-speaker |corr(T, A)| and |corr(T, V)| below ~0.3 on the labeled corpus, or residualize.
5. **Human-auditable.** A ±1 sweep must produce a difference a listener can name in one word.

## Options

**A. Phonation tension — pressed ↔ lax/breathy voice quality (RECOMMENDED).**
The laryngeal-strain axis: high T = tight, pressed, hard glottal closure (think gritted teeth,
suspense, held anger); low T = relaxed, breathy, easy (think reassurance, drowsiness, intimacy).
This is the classic voice-quality dimension, and crucially it is *not* loudness: whispered urgency
is high-T/low-A; a relaxed shout (cheering) is low-T/high-A. Acoustically well-established
correlates, all computable with numpy/librosa:

* **Spectral balance** — alpha ratio or Hammarberg index (energy above vs. below ~1–2 kHz rises
  with pressed phonation);
* **CPP** (cepstral peak prominence — drops with breathiness, the low-T end);
* **H1–H2** (first-harmonic dominance — high in breathy, low in pressed voice).

Label recipe = the *identical* recipe that already won for energy: composite z-score per speaker
over voiced frames → weak label. Self-supervised, no external model, no license surface at all.
Calibration: EIV's stress/anxiety-family heads + JL-Corpus (CC0) anxious-vs-neutral anchors +
the standing ~50-clip human audit. Harness measure: the same composite on rendered audio.

**B. Dominance (PAD's third axis).** Assertive/in-command ↔ submissive/timid. Semantically clean
(V-A-D is the textbook triple) and EIV may expose usable heads, but its acoustic correlates (pitch
floor, declination, rate) overlap the controls we already have (length_scale, and f0 is on the
harness roadmap), and "dominance" is a social-stance word, not an actor's body-state word. Weaker
harness measure. *Viable fallback if A's audit fails.*

**C. Urgency/stress composite (rate + pause compression + pitch-range).** Reads as "tension" in
the suspense sense, but rate is already directly controllable via `length_scale` — burning a
learned channel on it is redundant — and pitch-range manipulation collides with planned f0
control. *Not recommended.*

**D. Emotional intensity ("how strongly felt").** This is arousal by another name; guaranteed
collinear with A. *Rejected on requirement 4.*

## Recommendation

**A — phonation tension**, with the pressed↔breathy semantics written into the Director contract
**[RESCOPED 2026-07-20 to LAX ↔ TIGHT — see STATE.md; "breathy"/"strained" vocabulary is reserved for data with real aspiration]** ("tension: −1 = breathy/relaxed … +1 = pressed/strained"). Rationale: it is the only option that
is simultaneously (a) zero new license surface, (b) the exact validated energy-channel recipe with
a different feature, (c) acoustically orthogonal to loudness by construction, and (d) the axis
voice actors actually modulate when directed toward "tense."

Execution plan (slots into corpus-brief step 1→2):

1. Implement the composite (alpha ratio + CPP + H1–H2, voiced frames, per-speaker z-score) in
   `derive_vat_corpus.py v1`; measure corpus corr(T, A) and corr(T, V) — **gate: |r| < 0.3**,
   else residualize T against A per speaker and re-check.
2. Calibrate: score JL-Corpus with it (anxious/angry should rank high-T, sad/neutral low-T);
   cross-check against EIV stress-family heads on a LibriTTS-R sample; ~50-clip human audit
   ("does +T sound *strained* and −T sound *breathy*, at matched loudness?").
3. Add `tension → phonation composite` to `MEASURES` in `eval_harness.py`, pre-register the same
   §7 thresholds (ρ ≥ 0.9, leakage ≤ 0.2, WER Δ ≤ +0.10) **plus the cross-channel gate: sweeping
   T at A=0 must move LUFS by less than the energy channel's per-step effect** — this is the
   channel-independence check the derisk verdict listed as unproven.

Known risks, stated up front: audiobook speech (LibriTTS-R) has a narrow phonation range — the
label distribution may be thin at both tails (same tail problem as |V|, same eventual fix: Emilia
mining); recording-condition artifacts can leak into spectral measures (per-speaker z-scoring is
the mitigation, as it was for LUFS); and the pressed↔breathy audit could fail perceptually — in
which case fall back to B (dominance) rather than stretching the definition.

**DECIDED (owner, 2026-07-16): Option A approved** with the Director-contract wording above
(tension: −1 = breathy/relaxed … +1 = pressed/strained). Corpus-brief step 2 (EIV labeling run +
`derive_vat_corpus.py v1`) is unblocked and in progress.

**Calibration status (2026-07-16, within-speaker human audit):** the composite's independence
gate passed (corr(T,A) = −0.092, 30,351 clips), and the owner audited same-voice triplets
(breathy/neutral/pressed at matched loudness, 2 speakers): **the breathy (−T) end is validated**
("far better, matching that description"); **the pressed (+T) end FAILED** — judged between
breathy and neutral, not tenser than neutral. Interpretation: the composite detects breathiness
reliably but does not isolate strain at the top end (aspiration noise and bright/resonant
narration both confound alpha/CPP there). **Recalibration DONE (2026-07-17) — tension v2:**
`-z(EIV Soft_vs._Harsh)` added as a fourth composite component (JL calibration: d'=0.65,
angry/assertive rank harsh, polarity confirmed high=soft; Distress rejected — weak, d'=0.31).
Corpus relabeled (`data/libritts_r_vat_v2`), independence gates all PASS (corr(T,A)=-0.053,
corr(T,V)=-0.066). **Within-speaker re-audit (owner, 2026-07-17): breathy and neutral remain good; pressed is
"better" but takes on a nasal quality — improved, not yet earning the word "pressed."**
Interpretation: the harshness detector, hunting pressed voice in polite narration, lands on
nasal-adjacent brightness — the corpus lacks genuine strain to find. Owner call: **hold the
3-channel launch and mine Emilia-YODAS first** ("let's give Emilia her turn") — real strained
speech fixes the +T tail at the data level, which no label formula can. Dominance fallback
retired unless Emilia-enriched +T also fails audit.

✅ **The data-level fix this section calls for has reached the model** (2026-08-09). The
Emilia mining's keeps merged as
`libritts_r_emilia_vat_v5` — **10,997 of 13,141**, the rest lost to digits (D-M3) and the
shared `ASR_MAX_WER` — and `vat5_finetune` trained on it to `ep019`.

⚠ **It did not do what THIS section wanted, and that is the finding.** The mining was
tail-selected on T, so the Emilia half arrives **53.6% saturated on T** against LibriTTS's
4.7%. The pre-registered worry was that this hands the model a shortcut; **read 2026-08-09,
it did not** — no T-specific regression on the never-trained holdout, CI straddling zero
([quality-gap-plan.md](quality-gap-plan.md) § READ-OUT). So T is not poisoned, but neither
has this section's valence FAIL been retested: v5's new material is delivery-blank and
nobody has heard a clip of it. State in [training-sources.md](training-sources.md).

Linked from: **§2 above** · [STATE.md](STATE.md).

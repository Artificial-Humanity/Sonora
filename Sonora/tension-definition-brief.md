# Tension (T) Channel — Definition Brief (2026-07-16) — **APPROVED: Option A (phonation tension), owner call 2026-07-16**

_Step 1 of the corpus plan ([vat-corpus-decision-brief.md](vat-corpus-decision-brief.md)
Recommendation B) is "tension semantics first": pin down what T **means** before deriving labels
for it. This brief proposes definitions and a recommendation. V and A are settled (V = affect
polarity from EIV; A = energy/arousal, validated at ρ ≈ 1.000 from per-speaker LUFS z-scores)._

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
("tension: −1 = breathy/relaxed … +1 = pressed/strained"). Rationale: it is the only option that
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

Linked from: [vat-corpus-decision-brief.md](vat-corpus-decision-brief.md) ·
[vat-conditioning-design.md](vat-conditioning-design.md) · [derisk-energy-verdict.md](derisk-energy-verdict.md) ·
[STATE.md](STATE.md) §3.

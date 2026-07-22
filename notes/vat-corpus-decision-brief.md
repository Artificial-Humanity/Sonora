# Full-VAT Corpus — Decision Brief (2026-07-16) — **APPROVED: Option B (EIV pseudo-labeling), owner call 2026-07-16**

_The §7 de-risk verdict ([derisk-energy-verdict.md](derisk-energy-verdict.md)) closed the
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
   own brief: [tension-definition-brief.md](tension-definition-brief.md) (recommends phonation
   tension, pressed↔breathy — awaiting the same owner call as this brief).
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
phonation tension per [tension-definition-brief.md](tension-definition-brief.md) (approved same
day). Execution begins with `derive_vat_corpus.py v1`.

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
[derisk-energy-verdict.md](derisk-energy-verdict.md) · [vat-conditioning-design.md](vat-conditioning-design.md) ·
[STATE.md](STATE.md) §3.

# Delivery-lane remediation — the crossed bank

**Designed 2026-09-03. Not built.** This file owns the campaign that tests whether
*within-speaker lane contrast* repairs the delivery channel. The channel's standing — that
requesting a lane makes the audio worse — is [STATE.md](STATE.md), and the two arms that
killed the sampler and the row-duplication explanations are recorded in
[vat7r_rebalance.yaml § LANE CONTROL](../configs/experiment/vat7r_rebalance.yaml). Neither is
restated here.

---

## The measurement this campaign exists to act on

Three explanations for the damage are dead on measurement. None of them had looked at how the
labels are DISTRIBUTED over speakers. That turns out to be the whole story.

```bash
.venv/bin/python scripts/tools/measure_delivery_confound.py --corpus data/libritts_r_full_vat_v7
```

⚠ **The numbers are derived by that tool, not stated here.** It exists because the ep010
probe's design was lost the other way round — the outputs survived, the recipe did not — which
is what `probe_delivery_intercept.py` was written about. Its headline is `crossed_fraction`:
of the rows carrying a lane, the share whose speaker id carries at least two DIFFERENT lanes.
That is the only population from which a model can tell delivery apart from voice.

On v7 it is under one percent. Almost every labelled speaker id holds exactly one lane, no id
holds all four assignable lanes, and most labelled speakers hold exactly one row in the entire
corpus. **So for the great majority of the delivery signal, the lane and the speaker embedding
are the same fact.** The model was never shown one voice delivering two lanes.

The cause is recorded at `scripts/tools/merge_expressive_registers.py:27`, and it is a
bookkeeping loss rather than a design choice: *"One speaker id per clip (owner, 2026-08-09).
Identity is not recoverable"* — the bank was rendered without writing down which voice produced
each clip, so the merge had nothing to group on.

### What this explains, and what it does not

It explains the ear results without needing a new hypothesis. A lane request activates a
direction that in training only ever co-occurred with one-shot speaker embeddings, which is
why the free text says *"these do almost sound like different voices"*. It explains why
severity never tracked clip count, because volume was never the variable. And it explains the
9× null: repeating a confounded row cannot break a confound.

⚠⚠ **IT DOES NOT ESTABLISH THAT A CROSSED BANK REPAIRS THE CHANNEL.** It establishes that the
corpus cannot teach delivery as a factor, which is a statement about the data. Whether the FiLM
path *also* entangles delivery with the speaker embedding is a separate question, and the bank
below is what separates them.

⚠ **One earlier lead is answered by this and should not be re-run as its own test.** The
"speaker identity shifts when a lane is requested" lead was filed as a possible architectural
defect. The corpus taught it. **The speaker-dependent hum is NOT answered** — 4899 drew "very
noticeable" in 6 of 6 notes where 4896 did not, at every step count, and that still wants its
own designed test.

---

## The bank

**One voice, all four assignable lanes.** The lane vocabulary is
[matcha/delivery.py](../matcha/delivery.py) and is not restated here; Documentary is retired,
so four is the whole assignable set.

Three requirements, in order of how much they matter:

1. **Pin the voice at render time and write it into the manifest.** This is the single thing
   the old bank lost. A reference path, so the merge can give one speaker id to every render
   of one voice.
2. **Cross voice against lane.** Each pinned voice renders all four lanes.
3. **Give each voice many rows.** An embedding trained on one clip is memorised, not learned.

Text is held constant where the lane allows it. Neutral, Newscaster and Speech are properties
of the RENDER, so one text carries all three; Dialogue is a property of the TEXT and needs its
own lines against the same voice ([STATE.md](STATE.md) § the delivery channel).

⚠ **Real audio cannot supply this, and that is measured rather than assumed.** Run the tool
above: the LibriTTS-R contribution reaches Dialogue and Neutral only, and Newscaster and Speech
have no real-audio rows at all. An audiobook reader does not deliver a newscast or a rally
speech. This is the targeted-remediation case
[quality-gap-plan.md § Phase 1S](quality-gap-plan.md) was written for, and it does not need the
bulk synthetic lane that is deliberately sequenced behind real sources.

---

## The rulings — owner, 2026-09-03

Recorded with their date and source because this workspace has already been burned by a
ruling nobody could attribute. Each was chosen from a stated set of alternatives.

| # | decision | ruling |
|---|---|---|
| 1 | sequencing | rung 3's verdict is scored first; this recipe is drafted alongside it |
| 2 | speaker ids | a NEW id per pinned reference — a clone asserts nothing about who the voice is, and no real speaker's embedding takes on synthetic acoustics |
| 3 | size | about 700 rows on about 20 voices — the same order the existing bank cost, so it is a like-for-like replacement rather than an addition |
| 4 | engines | chatterbox and zonos — the two live engines that clone from a reference, so identity is pinned by construction |
| 5 | the existing labelled rows | delivery set to `unknown`. Rows, audio and speaker ids all stay, so `n_spks` does not move and the warm start still copies verbatim |
| 6 | lane label | the render request, ear-audited by sample |
| 7 | texts | reuse the existing bank's texts — same words, same lanes, only the voice structure differs |
| 8 | measurement | its own short probe run, the v7r shape: warm start, one variable, then the existing blind lane bench |
| 9 | the diversity floor | a bank-scoped exemption recorded in the campaign config. `ref_select` is NOT edited, so ordinary allocation is unchanged |
| 10 | identity drift | an engine that does not hold the voice across a lane change is gated out of that lane |
| 11 | ear audit | complete voice sets — all four lanes of one voice heard together |
| 12 | clone licence | train on it, publish nothing until the owner rules |

**Ruling 5 has a cost, stated rather than implied.** It edits prior rows, so the strictly
growing rule's guarantee — that rung *n*'s holdout number compares to rung *n−1*'s — has to be
re-established for the arm that uses it rather than inherited.

**Ruling 12 exists because the density changed the question.** LibriTTS-R is CC-BY-4.0 and
`configs/data_licenses.yaml`'s `sonora_expressive_registers` entry reasons about ENGINE WEIGHTS
— it does not reason about a cloned reference voice. Twenty voices at tens of clips each is a
far more identifiable clone than the one-offs that entry was written for.

### Decided by the developer, with the rule stated

* **The references come from the real-speech half of `reference-pool-v2`**, balanced by gender,
  drawn only from speakers in the training split. ⚠ **Never a dev-clean speaker** — that would
  contaminate the one instrument that compares runs across rungs.
* **The prediction goes in the campaign config before the run starts**, as every other arm on
  this front has done.
* **The QC gate runs after the render pass**, as it does after every generation pass.

---

## Recorded before the run

**A crossed bank is the first design that CAN separate lane from voice. That it WILL remove the
degradation is an open question, not an inference from the measurement above.** The bar is the
existing blind lane bench, unchanged, so the result is comparable with the donor and v7r arms
already on record.

⚠ **The pre-flight that can invalidate the whole bank:** a teacher engine may not hold identity
across a lane change itself. If asking chatterbox for Newscaster changes the voice, the bank
rebuilds the same confound one level down and reads as a clean design while doing it. Measure
speaker similarity across the four lanes within each pinned reference, on the bank, BEFORE the
merge — ruling 10 is what happens when it fails.

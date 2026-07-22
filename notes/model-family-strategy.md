# Sonora Model Family — Strategy (2026-07-16)

_Owner intent: a size ladder like other model families — **sonora-mini** (the current 150M-ceiling
on-device commitment), **sonora-mid**, **sonora-heavy** — chosen per target hardware, without the
current work training us into a corner where a bigger tier means starting over. The pinned answer
lives in [ARCHITECTURE.md](ARCHITECTURE.md) (canon); this note is the rationale._

## The framing

No model family carries weights across tiers — Gemma 12B is not a grown 2B; they are separate
trainings of a shared recipe. So the corner-risk is never "will we retrain for mid" (we will);
it's "would the *architecture* have to be rebuilt." The durable assets are deliberately
scale-free:

| Asset | Scale-free? | Why |
|---|---|---|
| Corpus + label pipeline (license wall, z-score weak labels, EIV/phonation derivation) | ✅ | Data outlives architectures; re-running the automated pipeline over more hours IS the data scaling story. |
| Eval regime (§7 pre-registered gates, watchers, human-audit workflow) | ✅ | Bigger models are held to the same or tighter thresholds. |
| Director↔Actor contract (V/A/T semantics, speaker-as-vector, text lane, chunking) | ✅ | Pinned in ARCHITECTURE.md §1; tiers are implementations of it. |
| Export/gate discipline (split graphs, GPU-clean rules, parity gates) | ✅ method | Fixed shapes/budgets are per-tier parameters of the same method. |
| Vocoder | per-tier, separable | Already a separate graph; swaps independently at the mel interchange. |
| Backbone + weights | ❌ per-tier | Expected. Matcha's encoder/decoder plausibly stretches to a ~100–300M mid (trainable on ai-lab-0 with patience); a heavy tier likely wants a DiT-style flow-matching backbone — still the same contract, mel interchange, and gates. |

## The two real corner-risks and their insurance (both cheap, both taken)

1. **Speaker representation.** The 247-slot embedding table is a roster, not a representation —
   a dead end at scale. Insurance: the contract pins **speaker = 64-dim vector, never an id**
   (the export graphs already take a vector; the table lookup is host-side). A future speaker
   encoder (zero-shot voices, mid-tier territory) produces the same vector and nothing
   downstream moves.
2. **Semantic drift between tiers.** If V/A/T mean subtly different things per tier's corpus,
   tiers stop being interchangeable and every consumer forks. Insurance: channel semantics and
   units are contract items (versioned, owner-gated changes), pinned before `derive_vat_corpus.py
   v1` gets written.

## Consequences already adopted

* Corpus accumulation is a **standing activity** (Emilia-YODAS tail mining moved in-scope for the
  first 3-channel run) — tier ambitions raise the stakes on hours, and the label extremes are
  what deeper emotional conveyance is bounded by.
* The **expressive fine-tune** stays on the radar as a *stage* any tier can receive (nonverbal
  vocal events, micro-prosody beyond the 3-dim control space), not a fork of the family.
* Compute reality: mini and plausibly mid train on ai-lab-0; heavy (≥1B-class) implies rented
  compute — a budget decision for its day, not an architectural one.
* Naming: keep publishing the current lineage unsuffixed; the `-mini` suffix gets applied when a
  second tier actually exists (owner: "we don't need to add -mini yet").

Linked from: [ARCHITECTURE.md](ARCHITECTURE.md) ·
[model-size-target-decision.md](model-size-target-decision.md) · [STATE.md](STATE.md) §3.

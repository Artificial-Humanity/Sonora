# VAT Conditioning — Design Decision (2026-07-14)

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

Linked from: [next-steps §B](../Prosodia/next-steps.md) (VAT-conditioning item),
[STATE roadmap §3](STATE.md), [high-ambition-1](high-ambition-1-matcha-actor.md).

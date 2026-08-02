# Matcha siblings — what the descendants learned that we haven't

_Owner request 2026-08-01: "if we're going to do it, let's do it to the best of our
abilities." Six repos studied for what transfers into Sonora, with the
[Decoder v2 DiT spike](model-decisions.md#decoder-v2--staged-dit-spike) as the
immediate consumer._

> **STANDING REFERENCE (owner 2026-08-01): "keep all of these sources on tap as case
> studies."** This is not a one-off memo — it is the comparison bench. When a design
> question comes up (block shape, sampling schedule, markup grammar, the ear), check
> whether a sibling already answered it before designing blind. Add engines as they
> arrive; keep the licence flags current, since those are the cheapest thing to get
> wrong and the most expensive to discover late.
>
> Scope note: the survey spans **both repos** by nature. Decoder findings serve Sonora;
> [§ Baichuan-Audio](#baichuan-audio-apache-20-code-and-weights--measure-it-against-prosodia-not-sonora)
> serves Prosodia's Solo Book Club layer, and is linked from
> [voice-interruption-and-discussion.md §6](../../../Prosodia/notes/voice-interruption-and-discussion.md).

**Headline: the spike's single named risk is already solved in public MIT code.**
[§ Decoder v2](model-decisions.md#decoder-v2--staged-dit-spike) records "tiny-scale DiT
is less proven than tiny-scale U-Nets on fine spectral texture" and proposes two
untested mitigations. StableTTS ships both, at 31M, with MAS retained.

---

## The lineage is not what the list implies

Only four of the six are Matcha descendants. Sorting this out first matters, because
two of them answer a different question than the one being asked.

| repo | actual relation to Matcha | family |
|---|---|---|
| **StableTTS** | borrows Matcha's flow-matching code; keeps MAS + duration predictor + length regulator | **true sibling** — same skeleton, different decoder |
| **RapFlow-TTS** | credits Matcha for "network architecture, dataset handling, text normalization" | **true sibling** — same architecture, different *objective* |
| **Match-TTSG** | same author (shivammehta25); Matcha extended to joint speech+gesture | **true sibling** — same decoder, extra output modality |
| **F5-TTS** | conceptual only — DiT + flow matching, but discards duration modelling | cousin |
| **CosyVoice** | not a descendant — LLM + supervised semantic tokens → CFM decoder | different family |
| **Baichuan-Audio** | not a descendant — audio LLM; its decoder is CosyVoice 2.0's | different family |

The two "different family" entries are still worth having read, but for the *direction*
and *perception* lanes rather than the decoder. See § Wrong question, right answer.

---

## 1. StableTTS — the spike, already built (MIT)

The most valuable find by a distance. `models/diffusion_transformer.py` implements a
`DiTConVBlock` that is close to a drop-in for what § Decoder v2 specifies:

| § Decoder v2 asks for | StableTTS has |
|---|---|
| DiT blocks with **adaLN-Zero** | `adaLN_modulation` → 6 params (`shift/scale/gate` × msa/mlp), gates zero-initialisable |
| conditioning = concat(V/A/T, delivery, speaker 64-d) | conditioning enters as a single vector `c` through `gin_channels`, **no cross-attention** |
| keep MAS + duration predictor + length regulator | keeps all three — MAS at training, duration predictor at inference, `generate_path()` as regulator |
| mini ≈ 20M | **31M**, trained on 600 h |

**It independently reached both of our proposed risk mitigations, and went further than
one of them.** The plan suggests "a shallow conv stem in the patch embedding"; StableTTS
instead makes the **entire FFN convolutional** (`Conv1d`, not Linear) inside every
transformer block — a design it takes from HierSpeech++, described as "a combination of
original DiT and FFT (FastSpeech's feed-forward transformer) for better prosody." It
also adds **U-Net-like long skip connections** across the DiT stack, which is our other
named mitigation, listed in its v1.1 notes as an audio-quality improvement.

Also uses RoPE, and LayerNorm with `elementwise_affine=False` (correct — the affine
comes from adaLN).

**Why this matters beyond convenience:** it is direct evidence against the risk that
motivated staging the spike behind a parity gate. A ~31M conv-FFN DiT holding prosody on
600 h is the experiment we were about to run blind.

**Independent confirmation of the CFG lever.** StableTTS applies classifier-free guidance
by randomly swapping the speaker embedding for a learnable `fake_speaker` during
training. Applied to *our* conditioning vector that yields an inference-time **guidance
scale on direction strength** — "how hard should the actor take the note?" — exactly the
knob [[vocalizer-vetting-surface]] obliges us to ship a dial for.

This is **not a new idea here**: [STATE.md](STATE.md) records it adopted 2026-07-16 as
"a classifier-free-guidance amplification lever … (conditioning dropout 0.15 keeps the
unconditional mode alive — testable on the current checkpoint)." What is new is (a) a
working precedent that it survives a **DiT** decoder rather than only the U-Net, and
(b) the observation that it never carried into § Decoder v2's scope. Carry it.

⚠️ Weights: license unstated in the README. Irrelevant if we port the block design
(MIT code) and train our own, which is what § Decoder v2 specifies anyway — "decoder
weights (no warm-start)."

## 2. RapFlow-TTS — the throughput answer (Apache-2.0, code *and* weights)

Built on Matcha's network architecture, so the delta is the **training objective**, not
the model. It enforces velocity consistency along an FM-straightened ODE trajectory,
in three stages: straight-flow → +consistency → +adversarial.

Result: **2 NFE** on LJSpeech at quality that Grad-TTS, VoiceFlow and Matcha-TTS all
degrade badly at — a claimed **5–10× NFE reduction**.

This lands directly on the reason the corpus cap was raised. The stated goal for 22 s was
long-form ebook reading and "reducing Gemma → instruct → Sonora cycles." Inference cost
per utterance is the other half of that equation, and this is a 5–10× lever on it that
requires no architecture change.

English-only (LJSpeech, VCTK) — but we are English-first, so unusually well matched.
Apache-2.0 on both code and weights makes it one of the cleanest licence positions in the
whole survey.

⚠️ Cost: stage 3 introduces discriminators, i.e. adversarial training complexity we do
not currently carry. And changing the objective invalidates prior de-risk measurements —
it would need its own §7 cycle, not a free ride on the decoder's.

## 3. Match-TTSG — the multi-channel precedent (MIT, same author)

Matcha's own author extending it to emit **speech and 3D gesture from one decoder**
rather than two pipelines — reporting better quality, smaller model, ~10× faster
inference, and better cross-modal sync than the separate-pipeline baseline.

The transferable idea is structural, not literal: a single flow-matching decoder can
carry an additional correlated output channel driven by shared conditioning, and doing so
*helped* rather than cost. That is the strongest available evidence for the
[Direction Contract v2](model-decisions.md) choice to make delivery a 4th FiLM channel
inside the existing decoder instead of bolting on a head.

## 4. F5-TTS — consult narrowly, as already decided (MIT code, ⚠️ CC-BY-NC weights)

DiT + ConvNeXt V2 text encoder, Vocos/BigVGAN vocoders, "Sway Sampling" as an
inference-time flow-step schedule, 16 NFE.

§ Decoder v2 already says: "consult for block shape, do not import its no-alignment
training style; we keep MAS." **That call is now independently confirmed** — StableTTS
demonstrates DiT and MAS coexisting happily, so F5's alignment-free approach is a choice
we can decline without giving up the backbone. Sway Sampling is worth a look as a
cheap inference-side win, orthogonal to everything else.

⚠️ **Weights are CC-BY-NC** (Emilia training data) — a WEIGHTS-tag licence wall per
[[nc-license-stance]]. Code is MIT. Read the code; do not touch the checkpoints.

---

## Wrong question, right answer

### CosyVoice (Apache-2.0) — a working precedent for the markup schema

Not a decoder sibling, but its **instruct mode is the closest public analogue to the
Director↔Actor contract**: a natural-language description plus a special end token
prepended to the input text, *and* fine-grained inline control — vocal-burst tags such as
`[laughter]` and `[breath]` placed **between text tokens**, with 100+ instruction types.

That is span-level conveyance markup working in a shipped Apache-2.0 model. It is direct
prior art for [[scm-markup-schema]] and for the span-markup spike that Contract v2 puts
ahead of span-FiLM. Worth studying the exact tokenisation before we finalise how spans
reach the actor.

### Baichuan-Audio (Apache-2.0, code and weights) — measure it against Prosodia, not Sonora

**Owner correction 2026-08-01, and it is the right one.** Benchmarked against Sonora this
model looks irrelevant: it is not a Matcha descendant, and for the decoder it offers
nothing StableTTS doesn't offer better. Benchmarked against **Prosodia** — the wider
product Sonora is a component of — it is the closest public reference for the one
subsystem the project has openly declared missing.

[voice-interruption-and-discussion.md §2](../../../Prosodia/notes/voice-interruption-and-discussion.md)
states the gap without hedging:

> The project today is **output-only** … there is **no speech *input*** anywhere in the
> stack (no ASR, no microphone capture, no voice-activity detection). … The Director
> (Gemma) and the Actor exist; the **ear** does not.

That is the Solo Book Club layer — barge in by voice, ask or discuss, resume — a
first-class long-intended goal. (An earlier draft called it the subject of a "pending
patent disclosure" — stale: there is no patent track, per the § below.)
Baichuan-Audio is an end-to-end **speech interaction** system: Whisper Large encoder →
8-layer RVQ tokenizer at 12.5 Hz → Qwen2.5-7B emitting interleaved text/audio tokens →
flow-matching decoder → HiFi-GAN (from CosyVoice 2.0). Bilingual ZH/EN, real-time,
multi-turn.

It speaks to §6's open questions almost line by line:

| §6 open question | what Baichuan-Audio offers |
|---|---|
| on-device ASR with a **permissive licence**, "same bar as the rest of the stack" | **Apache-2.0 on weights** — clears the [[apache-2-default-license]] bar outright |
| latency: mic → ASR → Gemma → TTS must feel conversational | built for real-time interaction; **12.5 Hz** token rate is the design choice that makes LLM-side latency tractable |
| multi-turn discussion, not just one-shot Q&A | its whole target is multi-turn speech conversation |
| language coverage (ties to high-ambition-4) | ⚠️ ZH/EN only — fine English-first, no help multilingual |

**And it is the mechanism that makes the ear symmetric with the contract.** The
interruption note already cross-references
[high-ambition-6](high-ambition-6-audience-conveyance-stt.md): "the voice-input side
should perceive prosodic conveyance (emphasis, V/A/T) through the same control contract
the Director dictates through." A plain Whisper transcript throws that away — the
listener's *tone* is lost, and "why would she **lie** to him?" arrives indistinguishable
whether asked in incredulity or idle curiosity. Baichuan's tokenizer is explicitly built
to keep both: mel-reconstruction loss supervising the **acoustic** side, pretrained-LLM
supervision the **semantic** side, over a tapered codebook `{8K, 4K, 2K, 1K, 1K, 1K, 1K,
1K}`. That is high-ambition-6's requirement, with shipped Apache-2.0 weights.

## Take the ear, not the brain

Two hard limits mean this is a component to harvest, not an architecture to adopt.

1. **On-device budget.** Qwen2.5-7B + Whisper Large will not sit on a phone beside
   Gemma 4 and Sonora. The **tokenizer is separable from the LLM backbone**, and the
   tokenizer is the part we need.
2. **It would dissolve the crown jewel.**
   [architecture-north-star.md §2](../../../Prosodia/notes/architecture-north-star.md)
   is explicit that the durable asset is "the Director and the typed contract — not the
   acoustic weights." An end-to-end audio LLM collapses Director → contract → Actor into
   one opaque model, trading the ownable asset for a commodity capability. Worse, it
   breaks spoiler-safety by construction: §5 requires filtering **at retrieval, before
   Gemma sees anything**, and an end-to-end model has no such seam. The reveal-frontier
   gate only exists because the reasoning is separable.

So: **Baichuan's tokenizer as the ear, feeding conveyance-aware transcript into the
existing Director.** Not Baichuan's LLM as a replacement for the Director, and not its
decoder as a replacement for Sonora.

**No IP question here** (owner 2026-08-01). There is no patent track — the posture is a
defensive statement in the repo plus fully Apache-2.0 open source, and the patent idea
belonged to an abandoned dual-licensing strategy. So adjacent prior art in end-to-end
speech interaction is simply context, not a threat to analyse: we are not filing, and
nothing here is being held back. The only live licence question is the ordinary one this
whole note tracks — **can we ship it Apache-2.0?** — and Baichuan-Audio can
([[apache-2-default-license]]).

---

## What should change in the plan

1. **Adopt StableTTS's `DiTConVBlock` shape as the spike's starting point** rather than
   building a plain DiT and discovering the texture problem ourselves. Conv FFN in-block
   + long skips, from the outset.
2. **Add CFG on the conditioning vector** to the spike's scope — one training-time change
   (probabilistic conditioning dropout) buying an inference-time direction-strength dial.
3. **Log consistency-FM (RapFlow) as a separate, later spike** with its own de-risk cycle.
   It is a big throughput lever but it changes the objective, and stacking it onto the
   decoder swap would confound both gates. One change at a time, as with the current
   two-changes-one-de-risk pairing.
4. **Route Baichuan-Audio to high-ambition-6**, not to decoder work.
5. **Read CosyVoice's instruct tokenisation before finalising span markup.**

Unchanged and now better-supported: keeping MAS, keeping the mel interchange, keeping the
parity gate, and refusing F5's alignment-free style.

## Sources

- StableTTS — https://github.com/KdaiP/StableTTS
- RapFlow-TTS — https://github.com/naver-ai/RapFlow-TTS · paper https://arxiv.org/abs/2506.16741
- Match-TTSG — https://github.com/shivammehta25/Match-TTSG
- F5-TTS — https://github.com/SWivid/F5-TTS · paper https://arxiv.org/abs/2410.06885
- CosyVoice — https://github.com/QwenAudio/CosyVoice · CosyVoice 3 https://arxiv.org/pdf/2505.17589
- Baichuan-Audio — https://github.com/baichuan-inc/Baichuan-Audio · paper https://arxiv.org/abs/2502.17239

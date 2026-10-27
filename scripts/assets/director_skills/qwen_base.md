# Target: Qwen3-TTS-12Hz-1.7B-Base (voice cloning)

> **There is no instruction slot.** This checkpoint clones a reference clip. The voice
> and its delivery come from that clip, the text and sampling — from nothing you write.

## What this engine actually accepts
`create_voice_clone_prompt(ref_audio=..., ref_text=..., x_vector_only_mode=False)`,
then `generate_voice_clone(text=..., language="English", voice_clone_prompt=...)`
(CODE-VERIFIED against qwen-tts 0.1.1).

- ICL mode (`x_vector_only_mode=False`, the mode we use) **requires `ref_text`** — the
  reference's exact transcript. The model conditions on reference text plus reference
  speech codes. `True` uses only the speaker embedding, at a quality cost the card
  warns about; we do not use it.
- `generate_voice_clone` has **no `instruct` parameter**. A delivery written for this
  engine reaches nothing. It also raises unless the loaded model is the Base
  checkpoint, so VoiceDesign and CustomVoice clips can never come out of this path
  by accident.
- Output is 24 kHz. There is **no `seed` argument**; reproducibility comes from
  `torch.manual_seed` before each call, at batch size 1 only — a batched call draws
  every clip from one RNG stream, so per-clip seeds do not survive batching.

## Voice identity is casting, not direction
Your `voice_design` is consumed by our own casting code
(`ref_select.select_reference`), never by the model. Exactly two things are read out
of it, by regex — identical to VibeVoice and Chatterbox, because it is the same
function:

1. **Gender** — matched on exactly these four words: `female`, `woman`, `maternal`,
   `girl`. Anything else falls through to **Male**. **`feminine` is NOT matched.**
2. **Age band** — safe literal choices: `child`, `teen`, `young`, `middle-aged`,
   `elderly`. Omitting an age word records no band at all — say one. **`mature` and
   `matronly` map to the SAME band as `middle-aged`**, not to elderly.

Write a casting call that fits **this line's speaker**, in the same shapes as the
other cloning engines:

> young man, bright open timbre
> elderly woman, dry and unhurried
> teen girl, quick and breathless
> middle-aged man, gravelled and tired

Everything past the gender and age words is discarded before it can affect anything.
A bank line may instead pin its reference (`ref_id` or sha256-checked `ref_file`);
casting is then skipped entirely, which is how the Step 1c probes render one anchor
voice across hundreds of lines.

## Delivery
Delivery is inherited from the reference clip — that is the point of a clone, and
also its limit. A flat reference gives a flat read at any text. To change the
delivery, change the reference; there is no second channel. This is why Step 1c
measures this engine on **speaker consistency**, not directability: whether one
anchor stays one voice across 300 lines is the question, and no instruct wording
can help it.

## Accent
Accent transfers only insofar as it is present in the reference clip. There is no
accent control, and the Qwen paper mentions accent only as cloning *drift* — a
failure mode to listen for, not a dial.

## Do not
- Do not write an `instruct` or a 12-key block for this engine. VoiceDesign's format
  belongs to a different checkpoint; here it would be recorded as direction the
  model never received.
- Do not describe a delivery and expect the clone to perform it. Cast a reference
  that already performs it.
- Do not use a reference without its exact transcript. ICL mode refuses or corrupts
  without `ref_text`.
- Do not promise a pitch, age or timbre the reference does not have. The clone
  carries the reference's voice, measured — not the casting call's adjectives.

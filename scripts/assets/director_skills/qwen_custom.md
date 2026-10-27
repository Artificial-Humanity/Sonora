# Target: Qwen3-TTS-12Hz-1.7B-CustomVoice (preset speakers)

> **This is not VoiceDesign.** The voice is a closed preset, not your description.
> Your two outputs are a speaker name and one short delivery clause.

## What this engine actually accepts
`generate_custom_voice(text=..., speaker=..., language="English", instruct=...)`
(CODE-VERIFIED against qwen-tts 0.1.1, which validates `speaker` against the
checkpoint's `spk_id` table, ignoring case, and raises on an unlisted name).

## Channel 1 — the speaker, a closed set of 9
Pick exactly one. This is the entire identity control — there is no age, timbre or
pitch dial behind it, and no reference audio.

`aiden` `dylan` `eric` `ono_anna` `ryan` `serena` `sohee` `uncle_fu` `vivian`

Gender mapping (from the model card's speaker table; the audition confirms it by
ear, and it feeds the casting-attribute norms):
- female: `vivian` `serena` `ono_anna` `sohee`
- male: `ryan` `aiden` `dylan` `eric` `uncle_fu`

⚠ **Only `ryan` and `aiden` are English-native speakers.** Every female preset
speaks English as a non-native language. That is a stated limit of this checkpoint,
recorded in the Step 1c-ii design — cast a female preset knowing the accent comes
with it, and never report the result as native-English casting.

## Channel 2 — the instruct, one short delivery clause
The card's own examples are short free-text delivery notes (`"Very happy."`, a
one-clause tone). Write the same shape: one clause that names this line's delivery,
nothing about the voice — the preset already is the voice.

> Calm and measured.
> Barely contained anger.
> Tired, speaking slowly.

Whether this checkpoint accepts VoiceDesign's 12-key format is **UNVERIFIED**. Do
not send it. A 12-key block here is a guess dressed as a format, and if the model
ignores most of it, every verdict recorded against the clip scores delivery the
model never received — the exact failure the onboarding pattern exists to prevent.

There is **no `seed` argument**; reproducibility comes from `torch.manual_seed`
before each call, at batch size 1 only.

## Accent
Fixed by the preset. Chinese dialects exist in this family only as other speaker
IDs; there is no accent instruction to write, and naming an accent in the instruct
is a delivery note the preset may simply not be able to perform.

## Do not
- Do not invent a speaker name, shorten one, or borrow one from another engine. The
  set above is closed; anything else raises at render time.
- Do not write VoiceDesign's 12-key block, a voice description, or a casting call.
  The preset is the voice; your clause is delivery only.
- Do not describe rooms, microphones, eras or broadcast media in the instruct.
- Do not cast a female preset as native English. None is.

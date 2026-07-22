# 24 kHz Move — Decision (2026-07-14)

_Owner call: the milestone-3 corpus and all subsequent training run at **native 24 kHz** — no
resampling of LibriTTS-R — honoring the original Gate 2 decision (2026-06-18) that Phase 0
pragmatically deviated from (LJSpeech @ 22.05 kHz + `hifigan_T2_v1`). Rationale: resampling has
its own risks/artifacts; LibriTTS-R is native 24 kHz; and the Rust engine's `StageAudioSink` is
native 24 kHz, so the actor's current 22.05→24 resample step disappears — model samples reach
the sink untouched._

## Vocoder: fine-tune HiFi-GAN to 24 kHz / 80-band (owner call, same conversation)

Chosen over Vocos-24k and BigVGAN-v2-24k (both pretrained, both 100-band): keeping **n_feats =
80** preserves the Phase-0 warm start (encoder/decoder shapes unchanged), the engine mel
contract, and the proven litert export lane — Vocos additionally has an ISTFT head with no
standard TFLite lowering (export-lane risk), and BigVGAN (~110M params) is outside the
on-device budget. Cost accepted: a HiFi-GAN fine-tune (from the universal checkpoint) on
LibriTTS-R, days of hands-off wall-clock on the Strix Halo.

## Consequences / task list

1. **Vocoder fine-tune is now on the critical path to the §7 de-risk verdict** — the eval
   harness needs rendered audio. Start it early; it's hands-off compute that runs while the
   corpus pipeline is built.
2. **Verify the vocoder standalone before the de-risk run** (copy-synthesis: ground-truth mel →
   vocoder → parity/ASR vs the recording). Two novelties are entering at once (24 kHz +
   conditioning) against the "make the first success boring" rule — this check keeps a bad
   vocoder from confounding the conditioning verdict.
3. **Mel spec to lock when writing the vocoder config:** sr 24000, n_fft 1024, hop 256,
   win 1024, n_mels 80, f_min 0 — and **f_max open**: 8000 maximizes transfer from the
   universal checkpoint; **12000 (lean)** actually uses the new bandwidth (above f_max the
   acoustic model can't specify content — staying at 8000 wastes part of the point of 24 kHz).
   Decide with a quick A/B at vocoder-config time. 80 bands over 0–12 kHz is coarser than the
   100-band configs Vocos/BigVGAN use at 24 kHz; accepted to keep warm-start compatibility.
4. **Frame arithmetic shifts:** hop 256 @ 24 kHz = 93.75 fps (was 86.13); the 512-frame fixed
   split graph now spans ~5.46 s (was ~5.94 s). Chunking budgets in the runtime notes stay
   valid; re-derive exact numbers at export time.
5. **Corpus stats:** mel mean/std recomputed for the 24 kHz multi-speaker corpus (Phase 0's
   −5.54/2.12 are LJSpeech@22.05).
6. **Engine `config.json` `sample_rate` flips 22050 → 24000 only when a genuine 24 kHz model
   ships** (the standing STATE warning) — and the actor's resample path becomes a no-op for it.
7. Warm start across the rate change is approximate for the decoder (mel content shifts) but
   full for the text encoder/durations; expected and acceptable for a fine-tune.

Linked from: [next-steps §A Gate 2](../Prosodia/next-steps.md),
[vat-conditioning-design.md](vat-conditioning-design.md) (sequencing note),
[dataset-landscape.md](dataset-landscape.md).

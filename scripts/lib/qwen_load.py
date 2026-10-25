"""Load Qwen3-TTS for rendering: the language model in bfloat16, the waveform decoder in float32.

⚠ qwen-tts 0.1.1's `Qwen3TTSModel.from_pretrained` forwards its keyword arguments, `dtype`
included, into `Qwen3TTSTokenizer.from_pretrained`. Loaded with `dtype=torch.bfloat16`, the speech
tokenizer (the SnakeBeta waveform decoder) therefore runs in bfloat16 although its
`speech_tokenizer/config.json` declares float32. Its error follows the voice: about -79 dBFS in
silence and -53 dBFS in loud speech, 29.6-34.6 dB SNR against a float32 decode of the same codes,
while band power moves by at most 0.13 dB (Notes: Sonora/phase-1s-step1-preregistration.md, the
bf16-decoder section). The owner ruled for float32 (2026-10-25), so `load_qwen` reloads the
tokenizer in float32 and attaches it, and refuses to return a model whose decoder is anything else.

Imports `torch` and `qwen_tts` only when called, so the qwen container is the only place that needs
them.
"""
import os

TOKENIZER_DTYPE = "float32"


def decoder_dtype(model):
    """The dtype of a loaded `Qwen3TTSModel`'s waveform decoder, e.g. "float32"."""
    param = next(model.model.speech_tokenizer.model.parameters())
    return str(param.dtype).replace("torch.", "")


def load_qwen(model_dir, device="cuda:0"):
    """`Qwen3TTSModel` with its language model in bfloat16 and its speech tokenizer in float32."""
    import torch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

    model = Qwen3TTSModel.from_pretrained(model_dir, device_map=device, dtype=torch.bfloat16)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(os.path.join(model_dir, "speech_tokenizer"),
                                                  device_map=device,
                                                  dtype=getattr(torch, TOKENIZER_DTYPE))
    model.model.load_speech_tokenizer(tokenizer)
    got = decoder_dtype(model)
    if got != TOKENIZER_DTYPE:
        raise SystemExit("REFUSING: Qwen's waveform decoder loaded as %s, not %s."
                         % (got, TOKENIZER_DTYPE))
    return model

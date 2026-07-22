import json, os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

OUT = "/audition/out/qwen3tts"
os.makedirs(OUT, exist_ok=True)
model = Qwen3TTSModel.from_pretrained(
    "/data/reference/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0", dtype=torch.bfloat16)
print("methods:", [m for m in dir(model) if m.startswith("generate")])

DESIGN = "A warm, natural adult narrator's voice, clear and unaffected."
INSTRUCTS = {
    "victory": "Speak with overwhelming, genuine excitement and joy, almost shouting.",
    "grief": "Speak quietly, with deep grief and a breaking voice.",
    "threat": "A tense, pressed, urgent whisper through gritted teeth.",
    "longform": "Calm, engaged audiobook narration.",
}
script = json.load(open("/audition/stress_script.json"))
gen = getattr(model, "generate_voice_design", None) or model.generate_custom_voice
for line in script["lines"]:
    kwargs = dict(text=line["text"], language="English",
                  instruct=INSTRUCTS[line["id"]])
    try:
        wavs, sr = gen(voice_description=DESIGN, **kwargs)
    except TypeError:
        wavs, sr = gen(speaker=DESIGN, **kwargs)
    name = f"{line['id']}.wav"
    sf.write(os.path.join(OUT, name), wavs[0], sr)
    print(name, len(wavs[0]) / sr, "s")
print("QWEN-RENDERS-DONE")

import json, os, sys
import soundfile as sf
import perth
class _NoopWatermarker:  # perth native component unavailable on this box; internal audition only
    def apply_watermark(self, wav, sample_rate=None, **kw): return wav
perth.PerthImplicitWatermarker = _NoopWatermarker
from chatterbox.tts import ChatterboxTTS

OUT = "/audition/out/chatterbox"
os.makedirs(OUT, exist_ok=True)
model = ChatterboxTTS.from_pretrained(device="cuda")
script = json.load(open("/audition/stress_script.json"))
for line in script["lines"]:
    for exag in (0.25, 0.5, 1.0):
        wav = model.generate(line["text"], exaggeration=exag)
        name = f"{line['id']}_exag{exag:.2f}.wav"
        sf.write(os.path.join(OUT, name), wav.squeeze(0).cpu().numpy(), model.sr)
        print(name, wav.shape[-1] / model.sr, "s")
print("CHATTERBOX-RENDERS-DONE")

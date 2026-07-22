import json, os
import torch
import soundfile as sf
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

OUT = "/audition/out/zonos"
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = "/data/reference/models/Zyphra/Zonos-v0.1-transformer"
model = Zonos.from_local(f"{MODEL_DIR}/config.json", f"{MODEL_DIR}/model.safetensors", device="cuda")

# emotion vector: [happiness, sadness, disgust, fear, surprise, anger, other, neutral]
EMOTIONS = {
    "victory":  [0.85, 0.02, 0.02, 0.02, 0.35, 0.02, 0.05, 0.05],
    "grief":    [0.02, 0.90, 0.02, 0.05, 0.02, 0.02, 0.05, 0.05],
    "threat":   [0.02, 0.05, 0.05, 0.25, 0.05, 0.60, 0.05, 0.05],
    "longform": [0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.85],
}
script = json.load(open("/audition/stress_script.json"))
for line in script["lines"]:
    torch.manual_seed(1234)
    cond = make_cond_dict(text=line["text"], language="en-us",
                          emotion=EMOTIONS[line["id"]])
    codes = model.generate(model.prepare_conditioning(cond))
    wav = model.autoencoder.decode(codes).squeeze().float().cpu().numpy()
    sf.write(os.path.join(OUT, f"{line['id']}.wav"), wav,
             model.autoencoder.sampling_rate)
    print(line["id"], len(wav) / model.autoencoder.sampling_rate, "s")
print("ZONOS-RENDERS-DONE")

import json, os
import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

OUT = "/audition/out/dia"
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = "/data/models/nari-labs/Dia-1.6B-0626"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = DiaForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda")
print("loaded")

# Dia has no instruct channel: affect comes from the script text and
# nonverbal tags. Canonical lines rendered verbatim under [S1]; one extra
# variant showcases the nonverbal-tag feature.
script = json.load(open("/audition/stress_script.json"))
lines = [(l["id"], f"[S1] {l['text']}") for l in script["lines"]]
lines.append(("victory_nonverbal",
              "[S1] We won! (laughs) We actually won the championship!"))

for lid, text in lines:
    torch.manual_seed(1234)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
    audio_tokens = model.generate(**inputs, max_new_tokens=3072,
                                  guidance_scale=3.0, temperature=1.8,
                                  top_p=0.90, top_k=45)
    decoded = processor.batch_decode(audio_tokens)
    processor.save_audio(decoded, os.path.join(OUT, f"{lid}.wav"))
    print(lid, "done", flush=True)
print("DIA-RENDERS-DONE")

"""Higgs v3 benchmark renders. LICENSE WALL: Higgs TTS 3 is NC
(boson-higgs-tts-3 research/non-commercial license) — these renders are
AUDITION/BENCHMARK ONLY and must never train or calibrate anything.

Control interface = inline tags (the model's native affect channel):
sentence-level <|emotion:...|> / <|style:...|>, per official PROMPTING.md.
Longform stays untagged as the neutral baseline.
"""
import json, os
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = "/audition/out/higgs"
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = "/data/reference/models/multimodalart/higgs-audio-v3-tts-4b-transformers"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
).to("cuda").eval()
print("loaded", flush=True)

TAGS = {
    "victory": "<|emotion:elation|>",
    "grief": "<|emotion:sadness|>",
    "threat": "<|style:whispering|><|emotion:fear|>",
    "longform": "",
}
script = json.load(open("/audition/stress_script.json"))
for line in script["lines"]:
    torch.manual_seed(1234)
    text = TAGS[line["id"]] + line["text"]
    wav = model.generate_speech(text, tokenizer, temperature=0.7, top_p=0.95)
    sf.write(os.path.join(OUT, f"{line['id']}.wav"),
             wav.float().numpy(), model.config.sample_rate)
    print(line["id"], "done", flush=True)
print("HIGGS-RENDERS-DONE")

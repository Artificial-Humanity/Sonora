"""MOSS-TTS 8.5B flagship audition (Apache-2.0). README documents only
text/reference/tokens on build_user_message — no instruction channel. We
attempt instruction= (family-shared processor may support it undocumented)
and fall back to text-only, logging which path ran.
"""
import json, os
import torch
import soundfile as sf
from transformers import AutoModel, AutoProcessor

MODEL = "/data/models/OpenMOSS-Team/MOSS-TTS"
OUT = "/audition/out/moss_tts"
os.makedirs(OUT, exist_ok=True)
device = "cuda"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  attn_implementation="sdpa",
                                  torch_dtype=torch.bfloat16).to(device).eval()
print("loaded", flush=True)

INSTRUCTS = {
    "victory": "A deep-voiced adult man overwhelmed with genuine joy and excitement, almost shouting with delight, in American English.",
    "grief": "A middle-aged man speaking quietly through deep grief, voice heavy and close to breaking, slow, in American English.",
    "threat": "A grown man's tense, menacing whisper through gritted teeth — quiet, pressed, dangerous, in American English.",
    "longform": "A warm, natural middle-aged male audiobook narrator, calm and engaged, even pacing, in American English.",
}

def build(text, lid):
    try:
        msg = processor.build_user_message(text=text, instruction=INSTRUCTS[lid])
        return msg, "instructed"
    except TypeError:
        return processor.build_user_message(text=text), "text-only"

script = json.load(open("/audition/stress_script.json"))
with torch.no_grad():
    for line in script["lines"]:
        torch.manual_seed(1234)
        msg, mode = build(line["text"], line["id"])
        batch = processor([[msg]], mode="generation")
        outputs = model.generate(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device),
                                 max_new_tokens=4096)
        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0].float().cpu().numpy()
            sf.write(os.path.join(OUT, f"{line['id']}.wav"), audio,
                     processor.model_config.sampling_rate)
            print(line["id"], mode, f"{len(audio)/processor.model_config.sampling_rate:.1f}s", flush=True)
print("MOSS-TTS-RENDERS-DONE")

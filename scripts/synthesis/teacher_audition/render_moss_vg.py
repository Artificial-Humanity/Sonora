import json, os
import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel, AutoProcessor

MODEL = "/data/reference/models/OpenMOSS-Team/MOSS-VoiceGenerator"
OUT = "/audition/out/moss_voicegen"
os.makedirs(OUT, exist_ok=True)
device = "cuda"
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, normalize_inputs=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  attn_implementation="sdpa",
                                  torch_dtype=torch.bfloat16).to(device).eval()

# Voice anchored in every instruction (Qwen audition lesson: unanchored
# descriptions drift cartoonish under high-arousal styles).
INSTRUCTS = {
    "victory": "A deep-voiced adult man overwhelmed with genuine joy and excitement, almost shouting with delight, in American English.",
    "grief": "A middle-aged man speaking quietly through deep grief, voice heavy and close to breaking, slow, in American English.",
    "threat": "A grown man's tense, menacing whisper through gritted teeth — quiet, pressed, dangerous, in American English.",
    "longform": "A warm, natural middle-aged male audiobook narrator, calm and engaged, even pacing, in American English.",
}
script = json.load(open("/audition/stress_script.json"))
with torch.no_grad():
    for line in script["lines"]:
        conv = [[processor.build_user_message(text=line["text"], instruction=INSTRUCTS[line["id"]])]]
        batch = processor(conv[0:1] if isinstance(conv[0], list) else conv, mode="generation")
        batch = processor([ [processor.build_user_message(text=line["text"], instruction=INSTRUCTS[line["id"]])] ], mode="generation")
        outputs = model.generate(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device))
        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0].float().cpu().numpy()
            sf.write(os.path.join(OUT, f"{line['id']}.wav"), audio,
                     processor.model_config.sampling_rate)
            print(line["id"], len(audio) / processor.model_config.sampling_rate, "s")
print("MOSS-VG-RENDERS-DONE")

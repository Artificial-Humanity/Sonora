"""Two short style-anchor clips (MOSS-VG, instructed) to serve as LongCat
voice-cloning prompts in the affect-transfer test. Synthetic references only —
no real person's voice is ever used as a cloning prompt."""
import os
import torch
import soundfile as sf
from transformers import AutoModel, AutoProcessor

MODEL = "/data/models/OpenMOSS-Team/MOSS-VoiceGenerator"
OUT = "/audition/out/anchors"
os.makedirs(OUT, exist_ok=True)
device = "cuda"
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, normalize_inputs=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  attn_implementation="sdpa",
                                  torch_dtype=torch.bfloat16).to(device).eval()

ANCHORS = {
    "anchor_grief": (
        "The house is so quiet now. I don't know what to do with all this silence.",
        "A middle-aged man speaking quietly through deep grief, voice heavy and close to breaking, slow, in American English."),
    "anchor_victory": (
        "Yes! Yes! I can't believe it, we actually did it!",
        "A deep-voiced adult man overwhelmed with genuine joy and excitement, almost shouting with delight, in American English."),
}
with torch.no_grad():
    for lid, (text, instr) in ANCHORS.items():
        torch.manual_seed(1234)
        batch = processor([[processor.build_user_message(text=text, instruction=instr)]],
                          mode="generation")
        outputs = model.generate(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device))
        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0].float().cpu().numpy()
            sf.write(os.path.join(OUT, f"{lid}.wav"), audio,
                     processor.model_config.sampling_rate)
            print(lid, flush=True)
print("ANCHORS-DONE")

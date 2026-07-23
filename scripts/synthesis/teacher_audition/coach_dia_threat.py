"""Dia direct-menace coaching experiment (portfolio v2 gap: the alley-thug
register). No instruct channel, so coaching = script craft: wording, beat
punctuation, dialogue scene context, temperature, and seed (= voice lottery).
"""
import os
import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

OUT = "/audition/out/dia_coach"
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = "/data/models/nari-labs/Dia-1.6B-0626"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = DiaForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda")
print("loaded", flush=True)

CANON = "[S1] Don't move. Don't even breathe. If you make a sound, they will hear us."
TAKES = [
    # id, text, temperature, seed
    ("t1_canon_lowtemp", CANON, 1.1, 42),
    ("t2_beats",
     "[S1] Don't move. Don't. Even. Breathe. If you make a sound... they will hear us.",
     1.3, 42),
    ("t3_direct_rewrite",
     "[S1] Don't move. Don't even breathe. Make one sound, and you'll regret it.",
     1.2, 42),
    ("t4_mugging_scene",
     "[S1] Wallet. Now. Slow and quiet. [S2] Okay, okay, please... [S1] Don't move. "
     "Don't even breathe. Make one sound, and you'll regret it.",
     1.3, 42),
    ("t5_mugging_scene_alt_voice",
     "[S1] Wallet. Now. Slow and quiet. [S2] Okay, okay, please... [S1] Don't move. "
     "Don't even breathe. Make one sound, and you'll regret it.",
     1.3, 7),
    ("t6_canon_highcfg", CANON, 1.3, 42),
]

for lid, text, temp, seed in TAKES:
    torch.manual_seed(seed)
    gscale = 6.0 if lid.endswith("highcfg") else 3.0
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
    audio_tokens = model.generate(**inputs, max_new_tokens=1536,
                                  guidance_scale=gscale, temperature=temp,
                                  top_p=0.90, top_k=45)
    decoded = processor.batch_decode(audio_tokens)
    processor.save_audio(decoded, os.path.join(OUT, f"{lid}.wav"))
    print(lid, "done", flush=True)
print("DIA-COACH-DONE")

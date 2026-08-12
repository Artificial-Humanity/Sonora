"""Orpheus-3B audition (Apache-2.0; HF gate auto-approved 2026-07-17).
Llama-backbone TTS with preset voices + inline emotion tags; audio tokens
decoded by SNAC 24 kHz. Canonical prompt format: "{voice}: {text}" wrapped
in Orpheus's special tokens.
"""
import json, os
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

OUT = "/audition/out/orpheus"
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = "/data/models/canopylabs/orpheus-3b-0.1-ft"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to("cuda").eval()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
print("loaded", flush=True)

SOH, EOT, EOH = 128259, 128009, 128260   # start-of-human, end-of-text, end-of-human
SOA, EOA = 128257, 128258                # start-of-audio marker, end-of-audio/eos
AUDIO_BASE = 128266

# voice assignment: male for victory/threat, tara (female) for grief/longform
VOICE = {"victory": "dan", "grief": "tara", "threat": "leo", "longform": "tara"}

def render(lid, voice, text):
    torch.manual_seed(1234)
    ids = tok(f"{voice}: {text}", return_tensors="pt").input_ids
    ids = torch.cat([torch.tensor([[SOH]]), ids, torch.tensor([[EOT, EOH]])], dim=1).to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=2048, do_sample=True,
                             temperature=0.6, top_p=0.95, repetition_penalty=1.1,
                             eos_token_id=EOA)
    seq = out[0].tolist()
    if SOA in seq:
        seq = seq[len(seq) - seq[::-1].index(SOA):]        # after last start-of-audio
    codes = [t - AUDIO_BASE for t in seq if AUDIO_BASE <= t < AUDIO_BASE + 4096 * 7]
    codes = codes[: len(codes) // 7 * 7]
    l1, l2, l3 = [], [], []
    for i in range(0, len(codes), 7):
        f = codes[i:i + 7]
        l1.append(f[0])
        l2 += [f[1] - 4096, f[4] - 4 * 4096]
        l3 += [f[2] - 2 * 4096, f[3] - 3 * 4096, f[5] - 5 * 4096, f[6] - 6 * 4096]
    layers = [torch.tensor(l).unsqueeze(0).to("cuda") for l in (l1, l2, l3)]
    with torch.no_grad():
        wav = snac.decode(layers).squeeze().float().cpu().numpy()
    sf.write(os.path.join(OUT, f"{lid}.wav"), wav, 24000)
    print(lid, f"{len(wav)/24000:.1f}s", flush=True)

script = json.load(open("/audition/stress_script.json"))
for line in script["lines"]:
    render(line["id"], VOICE[line["id"]], line["text"])
render("victory_tagged", "dan", "We won! <laugh> We actually won the championship!")
print("ORPHEUS-RENDERS-DONE")

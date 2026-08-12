"""quote-pilot-v3d: Orpheus-3B PRETRAINED zero-shot cloning arm (Apache-2.0).

Casting = reference conditioning: each line's G26 design selected an audited
v1 keep (v3d_bank.json); the prompt is a completed (ref_text -> ref_codes)
example followed by the target text, and the model continues in the
reference voice. SNAC 7-token frame mapping mirrors the proven FT decode.
"""
import json, os
import numpy as np
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

BANK = "/data/model-training/datasets/sonora-expressive-registers/quote-pilot-v3d/v3d_bank.json"
OUT = "/data/model-training/datasets/sonora-expressive-registers/quote-pilot-v3d/orpheus_pt"
MODEL_DIR = "/data/models/canopylabs/orpheus-3b-0.1-pretrained"
os.makedirs(OUT, exist_ok=True)

SOH, EOT, EOH = 128259, 128009, 128260
SOA, EOA = 128257, 128258
AUDIO_BASE = 128266
SEED = 1234

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to("cuda").eval()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
print("loaded", flush=True)

def encode_ref(path, max_s=10.0):
    data, sr = sf.read(path, dtype="float32")
    assert sr == 24000, f"ref not 24k: {path} ({sr})"
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data[: int(max_s * 24000)]).unsqueeze(0)
    with torch.no_grad():
        l1, l2, l3 = snac.encode(wav.unsqueeze(0).to("cuda"))
    l1, l2, l3 = l1[0].tolist(), l2[0].tolist(), l3[0].tolist()
    toks = []
    for i in range(len(l1)):
        toks += [AUDIO_BASE + l1[i],
                 AUDIO_BASE + 4096 + l2[2 * i],
                 AUDIO_BASE + 2 * 4096 + l3[4 * i],
                 AUDIO_BASE + 3 * 4096 + l3[4 * i + 1],
                 AUDIO_BASE + 4 * 4096 + l2[2 * i + 1],
                 AUDIO_BASE + 5 * 4096 + l3[4 * i + 2],
                 AUDIO_BASE + 6 * 4096 + l3[4 * i + 3]]
    return toks

def decode_codes(seq):
    if SOA in seq:
        seq = seq[len(seq) - seq[::-1].index(SOA):]
    codes = [t - AUDIO_BASE for t in seq if AUDIO_BASE <= t < AUDIO_BASE + 4096 * 7]
    codes = codes[: len(codes) // 7 * 7]
    l1, l2, l3 = [], [], []
    for i in range(0, len(codes), 7):
        f = codes[i:i + 7]
        l1.append(f[0])
        l2 += [f[1] - 4096, f[4] - 4 * 4096]
        l3 += [f[2] - 2 * 4096, f[3] - 3 * 4096, f[5] - 5 * 4096, f[6] - 6 * 4096]
    if any(c < 0 or c > 4095 for c in l1 + l2 + l3):
        return None
    layers = [torch.tensor(l).unsqueeze(0).to("cuda") for l in (l1, l2, l3)]
    with torch.no_grad():
        return snac.decode(layers).squeeze().float().cpu().numpy()

bank = json.load(open(BANK))
manifest = open(os.path.join(OUT, "orpheus_pt_manifest.jsonl"), "w")
for row in bank:
    torch.manual_seed(SEED)
    ref_toks = encode_ref(row["ref_wav"])
    ids = ([SOH] + tok(row["ref_text"], add_special_tokens=False).input_ids + [EOT, EOH]
           + [SOA] + ref_toks + [EOA]
           + [SOH] + tok(row["text"], add_special_tokens=False).input_ids + [EOT, EOH])
    ids = torch.tensor([ids]).to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=2048, do_sample=True,
                             temperature=0.6, top_p=0.9, repetition_penalty=1.15,
                             eos_token_id=EOA)
    wav = decode_codes(out[0].tolist())
    lid = f'{row["id"]}_ORP'
    if wav is None or len(wav) < 24000:
        print(lid, "FAILED (bad/short audio)", flush=True)
        continue
    wav_path = os.path.join(OUT, f"{lid}.wav")
    sf.write(wav_path, wav, 24000)
    manifest.write(json.dumps({
        "id": lid, "campaign": "quote-pilot-v3d", "engine": "orpheus3b-pt",
        "engine_license": "apache-2.0", "register": row["register"],
        "intended": row["intended"], "direction": row["direction"],
        "text": row["text"], "seed": SEED, "sr": 24000,
        "wav": f"orpheus_pt/{lid}.wav", "bank_version": "v3d",
        "ref": row["ref_meta"] | {"ref_text": row["ref_text"]},
    }) + "\n")
    print(lid, f"{len(wav)/24000:.1f}s", flush=True)
manifest.close()
print("ORP-RENDERS-DONE", flush=True)

"""quote-pilot-v3d: VibeVoice-Large-Q8 arm (MIT, FabioSarracino selective 8-bit) — v3e footprint comparison.

Casting = native voice-sample conditioning: each line's G26-selected v1 keep
(v3d_bank.json) is passed as the speaker's voice sample; the script text is
the quote line. Known engine quirk to watch in QC/audition: spontaneous
background music/ambience on some generations.
"""
import json, os
import soundfile as sf
import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

BANK = "/data/model-training/datasets/sonora-expressive-registers/quote-pilot-v3d/v3d_bank.json"
OUT = "/data/model-training/datasets/sonora-expressive-registers/quote-pilot-v3d/vibevoice_q8"
MODEL_DIR = "/data/models/FabioSarracino/VibeVoice-Large-Q8"
os.makedirs(OUT, exist_ok=True)
SEED = 1234

processor = VibeVoiceProcessor.from_pretrained(MODEL_DIR)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_DIR, device_map="cuda", attn_implementation="sdpa").eval()
model.set_ddpm_inference_steps(num_steps=10)
print("loaded", flush=True)

bank = json.load(open(BANK))
manifest = open(os.path.join(OUT, "vibevoice_q8_manifest.jsonl"), "w")
for row in bank:
    torch.manual_seed(SEED)
    script = f"Speaker 0: {row['text']}"
    inputs = processor(text=[script], voice_samples=[[row["ref_wav"]]],
                       padding=True, return_tensors="pt", return_attention_mask=True)
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=None, cfg_scale=1.3,
                             tokenizer=processor.tokenizer,
                             generation_config={"do_sample": False}, verbose=False)
    wav = out.speech_outputs[0].squeeze().float().cpu().numpy()
    lid = f'{row["id"]}_VVQ8'
    if wav is None or len(wav) < 24000:
        print(lid, "FAILED (short/empty)", flush=True)
        continue
    sf.write(os.path.join(OUT, f"{lid}.wav"), wav, 24000)
    manifest.write(json.dumps({
        "id": lid, "campaign": "quote-pilot-v3e", "engine": "vibevoice-large-q8", "weights_source": "FabioSarracino/VibeVoice-Large-Q8 (selective bnb-8bit of aoi-ot mirror)",
        "engine_license": "mit", "register": row["register"],
        "intended": row["intended"], "direction": row["direction"],
        "text": row["text"], "seed": SEED, "sr": 24000,
        "wav": f"vibevoice_q8/{lid}.wav", "bank_version": "v3d",
        "ref": row["ref_meta"] | {"ref_text": row["ref_text"]},
    }) + "\n")
    print(lid, f"{len(wav)/24000:.1f}s", flush=True)
manifest.close()
print("VV-RENDERS-DONE", flush=True)

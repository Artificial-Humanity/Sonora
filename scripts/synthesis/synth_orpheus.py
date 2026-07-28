"""Stage-1 renderer: Orpheus 3B (canopylabs, Apache-2.0) — the `-ft` checkpoint.

Reads the script bank, renders every engine=="orpheus" line, writes wavs +
orpheus_manifest.jsonl to --out.

Two fixes over the renderer whose 2026-07-23 verdict was voided, and the verdict
was void because of BOTH:

  1. CHECKPOINT. We auditioned `-pretrained`. That is the BASE model: no named
     voices, no emotion tags, nothing to direct. Every control channel lives in
     `-ft`, from which we had never rendered a single clip.
  2. PROMPT. The old renderers ended at [EOT, EOH] and omitted 128261
     (start-of-AI) and 128257 (start-of-speech). That leaves the model at
     end-of-human, free to choose between emitting the audio header and simply
     continuing as a text LLM — which is exactly the observed "seemed to repeat
     parts of the passage".

The only string that reaches the model is f"{voice}: {text}". There is no
instruction slot; prose sent here is spoken aloud or parsed as a voice name.
Voice and tags are closed sets validated in build_direction() — the shipped
validation function is dead code, so an invented name is interpolated and SPOKEN.

Never read emotions.txt from the repo root. Its 20 bare words look exactly like
a controlled vocabulary, have no brackets and one typo, and are read aloud as
text. Two community reports are people who fell for it.

Container recipe (smoke-proven 2026-07-28): pip transformers soundfile snac.
Nothing exotic — this engine was always the least fragile of the four; what
made its verdict worthless was the checkpoint and the prompt, not the plumbing.
"""
import argparse
import json
import os

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

MODEL_DIR = "/data/models/canopylabs/orpheus-3b-0.1-ft"
SNAC_DIR = "hubertsiuzdak/snac_24khz"

SOH, EOT, EOH = 128259, 128009, 128260   # start-of-human, end-of-text, end-of-human
SOAI, SOA, EOA = 128261, 128257, 128258  # start-of-AI, start-of-speech, end-of-audio
AUDIO_BASE = 128266
SR = 24000

# The reference decoder silently returns NO audio below 28 speech tokens, raising
# nothing. Four 7-token frames is the floor; anything shorter is a failed render,
# not a short one.
MIN_CODES = 28
MS_PER_FRAME = 0.0853        # 85.3 ms per 7 tokens — the documented budget
CHARS_PER_SEC = 14.0         # rough speech rate, for sizing the token budget
BUDGET_HEADROOM = 1.6
REPETITION_PENALTY = 1.1     # maintainers call this REQUIRED for stable generation;
                             # it doubles as a rate control, so pin it, never tune it


def token_budget(text):
    """Cap generation length from the text, not from a fixed 2048.

    Dia's over-generation was our token budget rather than its temperature
    (2026-07-26). Same discipline here: budget the line, leave headroom, cap.
    """
    seconds = max(len(text) / CHARS_PER_SEC, 1.0) * BUDGET_HEADROOM
    return int(min(max(seconds / MS_PER_FRAME * 7, 256), 4096))


def decode_audio(seq, snac, device):
    """Slice after the last start-of-speech, de-interleave 7-token frames, decode."""
    if SOA in seq:
        seq = seq[len(seq) - seq[::-1].index(SOA):]
    codes = [t - AUDIO_BASE for t in seq if AUDIO_BASE <= t < AUDIO_BASE + 4096 * 7]
    codes = codes[: len(codes) // 7 * 7]
    if len(codes) < MIN_CODES:
        return None, len(codes)
    l1, l2, l3 = [], [], []
    for i in range(0, len(codes), 7):
        f = codes[i:i + 7]
        l1.append(f[0])
        l2 += [f[1] - 4096, f[4] - 4 * 4096]
        l3 += [f[2] - 2 * 4096, f[3] - 3 * 4096, f[5] - 5 * 4096, f[6] - 6 * 4096]
    layers = [torch.tensor(l).unsqueeze(0).to(device) for l in (l1, l2, l3)]
    with torch.no_grad():
        wav = snac.decode(layers).squeeze().float().cpu().numpy()
    return wav, len(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "orpheus"]
    if not jobs:
        print("no orpheus jobs in bank")
        return
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16).to("cuda").eval()
    snac = SNAC.from_pretrained(SNAC_DIR).to("cuda").eval()
    print(f"loaded ({len(jobs)} jobs)", flush=True)

    manifest_path = os.path.join(args.out, "orpheus_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            d = job["direction"]
            prompt = f"{d['voice']}: {d['render_text']}"
            ids = tok(prompt, return_tensors="pt").input_ids
            # The trained form CONTINUES past end-of-human into start-of-AI +
            # start-of-speech. Omitting these two is what voided the last verdict.
            ids = torch.cat([torch.tensor([[SOH]]), ids,
                             torch.tensor([[EOT, EOH, SOAI, SOA]])], dim=1).to("cuda")
            torch.manual_seed(job["seed"])
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=token_budget(d["render_text"]),
                                     do_sample=True, temperature=0.6, top_p=0.95,
                                     repetition_penalty=REPETITION_PENALTY,
                                     eos_token_id=EOA)
            wav, n_codes = decode_audio(out[0].tolist(), snac, "cuda")
            if wav is None:
                print(job["id"], f"FAILED ({n_codes} codes < {MIN_CODES}: decoder "
                                 f"returns silence)", flush=True)
                continue
            sf.write(os.path.join(args.out, f"{job['id']}.wav"), wav, SR)

            row = dict(job)
            row.update({
                "wav": f"{job['id']}.wav", "sr": SR,
                "engine_license": "Apache-2.0 (canopylabs/orpheus-3b-0.1-ft)",
                "weights_source": "canopylabs/orpheus-3b-0.1-ft (NOT -pretrained)",
                "campaign": bank["campaign"], "bank_version": bank["version"],
                # community-derived gender mapping — the audition confirms it by ear
                # and it feeds the casting-attribute norms
                "intended_gender": "F" if d["voice"] in ("tara", "leah", "jess", "mia", "zoe") else "M",
                "prompt": prompt,
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/SR:.1f}s voice={d['voice']}", flush=True)
    print("SYNTH-ORPHEUS-DONE")


if __name__ == "__main__":
    main()

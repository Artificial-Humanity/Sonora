"""Stage-1 renderer: Zonos v0.1 transformer (Zyphra, Apache-2.0).

Reads the script bank, renders every engine=="zonos" line, writes wavs +
zonos_manifest.jsonl to --out.

The most numerically directable engine in the portfolio, and the 2026-07-17
audition rendered every clip at the FLOOR of its expressiveness range with no
voice set at all. Both defects are fixed here, and both were in the renderer,
not the model:

  pitch_std — never passed, so it took the default 20.0. Docs: 20-45 is normal
              speech, 60-150 expressive. Victory, grief and threat all rendered
              flat. build_direction() now emits it per line.
  speaker   — never passed, so the model drew its learned UNCONDITIONAL voice,
              an arbitrary draw per seed. That is the whole of the "gender
              coverage gap" the void verdict blamed on Zonos.

The emotion vector is L1-normalised inside the model, so build_direction()
normalises it first and the manifest records the shape actually conditioned on.
Zyphra's own UI ships emotion OFF by default because it destabilises generation:
if clips come back with dropped words or long silences, drop the emotion vector
before suspecting anything else.

NEVER pass vqscore_8, dnsmos_ovrl, speaker_noised or ctc_loss — those
conditioners exist on the HYBRID checkpoint, not ours, and are silently ignored
here (the **kwargs trap that cost us the Qwen voice design). cfg_scale must not
be 1: the model asserts on it, so the generate() default stands.

Container recipe (smoke-proven 2026-07-28) — RUN FROM SOURCE, NOT PIP:
`pip install git+…/Zonos.git` yields an UNIMPORTABLE package, because the wheel
omits the zonos.backbone subpackage entirely (model.py ships, zonos/backbone/
does not). The checkout at /data/toolchain/Zonos on PYTHONPATH is the fix.
Deps: apt espeak-ng; pip phonemizer inflect kanjize soundfile transformers
huggingface-hub **sudachipy sudachidict-full** — the last two are not optional
despite being the Japanese tokenizer, because conditioning.py imports them at
module scope, so an English-only run dies without them.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.speaker_cloning import SpeakerEmbeddingLDA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_select import select_reference, design_age_band  # noqa: E402

MODEL_DIR = "/data/models/Zyphra/Zonos-v0.1-transformer"
FMAX_CLONING = 22050.0
MAX_SECONDS = 30.0   # hard cap AND a training limit; cut text rather than slow the rate


_cloner = None


def _speaker_embedding(model, wav_path):
    """128-d embedding from a 5-30 s reference. Never leave `speaker` unset.

    Deliberately NOT model.make_speaker_embedding(): that constructs
    SpeakerEmbeddingLDA on the model's device, and Zonos builds it inside a
    `with torch.device("cuda")` context. torchaudio's melscale_fbanks mixes
    context-created and explicitly-CPU tensors in _create_triangular_filterbank,
    so the constructor itself dies with "found at least two devices, cuda:0 and
    cpu" before a single clip is read (2026-07-28, torchaudio in rocm/pytorch).

    Building AND running it on CPU sidesteps that without device surgery: the
    device is threaded through nested modules as a plain string, so moving the
    module after construction would leave stale "cpu" attributes behind. It is one
    small ResNet forward per reference, and only the resulting embedding moves to
    the GPU.
    """
    global _cloner
    if _cloner is None:
        _cloner = SpeakerEmbeddingLDA(device="cpu")
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T)          # [channels, samples]
    _, emb = _cloner(wav, sr)
    # mirrors make_speaker_embedding's own return shape/dtype exactly
    return emb.unsqueeze(0).bfloat16().to(getattr(model, "device", "cuda"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "zonos"]
    if not jobs:
        print("no zonos jobs in bank")
        return
    model = Zonos.from_local(f"{MODEL_DIR}/config.json",
                             f"{MODEL_DIR}/model.safetensors", device="cuda")
    sr_out = model.autoencoder.sampling_rate
    print(f"loaded ({len(jobs)} jobs)", flush=True)

    used = set()
    manifest_path = os.path.join(args.out, "zonos_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            d = job["direction"]
            # phoneme count is unknown before phonemization; letters are a safe
            # over-estimate, so this warns early rather than truncating late.
            est = len(job["text"].replace(" ", "")) / max(d["speaking_rate"], 1e-6)
            if est > MAX_SECONDS:
                print(job["id"], f"WARN est {est:.0f}s > {MAX_SECONDS}s cap — cut the text, "
                                 f"do not slow the rate", flush=True)

            ref_wav, ref_text, ref_meta = select_reference(
                d.get("design", ""), job.get("intended", {}), used)
            torch.manual_seed(job["seed"])
            cond = make_cond_dict(text=job["text"], language="en-us",
                                  speaker=_speaker_embedding(model, ref_wav),
                                  emotion=d["emotion"],
                                  pitch_std=d["pitch_std"],
                                  speaking_rate=d["speaking_rate"],
                                  fmax=FMAX_CLONING)
            codes = model.generate(model.prepare_conditioning(cond))
            wav = model.autoencoder.decode(codes).squeeze().float().cpu().numpy()
            sf.write(os.path.join(args.out, f"{job['id']}.wav"), wav, sr_out)

            row = dict(job)
            row.update({
                "wav": f"{job['id']}.wav", "sr": sr_out,
                "engine_license": "Apache-2.0 (Zyphra/Zonos-v0.1-transformer)",
                "weights_source": "Zyphra/Zonos-v0.1-transformer",
                "campaign": bank["campaign"], "bank_version": bank["version"],
                "intended_age": design_age_band(d.get("design", "")),
                "intended_gender": ref_meta["gender"],
                "ref": ref_meta | {"ref_text": ref_text},
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/sr_out:.1f}s pitch_std={d['pitch_std']} "
                             f"rate={d['speaking_rate']} ref={ref_meta['id']}", flush=True)
    print("SYNTH-ZONOS-DONE")


if __name__ == "__main__":
    main()

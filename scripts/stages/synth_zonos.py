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

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from synth_common import rebuild_used_set, write_wav_atomic  # noqa: E402
from ref_select import (  # noqa: E402
    MAX_REF_EXCURSION as _REF_EXCURSION,
    select_reference, pinned_reference, design_age_band,  # noqa: E402
                        REF_BLACKLIST)

MODEL_DIR = "/data/models/Zyphra/Zonos-v0.1-transformer"
FMAX_CLONING = 22050.0
MAX_SECONDS = 30.0   # hard cap AND a training limit; cut text rather than slow the rate

# Speaking-rate ceiling (measured on revisit-v1, 2026-07-28). Both clips the
# director sent at rate 24 came back with the owner's "layered / ghostly" verdict
# AND a separate "spoken too quickly" complaint; all 17 clips at rate <=18 were
# clean of it. Zyphra's own UI presets top out at 21.0 ("high"), so 24 was already
# off the end of the supported range — the model was being asked to fit the
# phonemes into fewer frames than it can articulate, and the overlap it produces
# is what the ear reads as a doubled voice.
#
# Clamped here rather than only documented in the skill file because the skill is
# advice to a director model and this is a correctness bound: a director that
# writes 24 should still get usable audio.
#
# LOWERED 18 -> 16 (stress-v1, 2026-07-29). Rate 18 is articulable — no doubling, no
# audio flaw — but it is not PERFORMABLE: 5 of 10 stress clips at rate 18 were dropped
# with the identical verdict "spoken fairly quickly and without any noticeable emotional
# conveyance... dry like an actor reading a line half-heartedly", and the keeps scored
# 4/5/1/1/2. Separately, 9 of 10 landed under the 4 s speech floor (~23 chars/s at rate
# 18 needs 92+ chars of text; see min-clip-length rule). So between the two ceilings —
# 18 where audio breaks, 16 where delivery does — the corpus wants the lower one.
# 16 itself is untested (measured points are 18 bad, ~15 fine); if fast reads at 16
# still come back flat, the next stop is 15, not another guess upward.
MAX_SPEAKING_RATE = 16.0

# Zonos is NOT immune to the emphasis-peak voice split (routing probe, 2026-07-29):
# rt_news_ZON was dropped for layering — the highest-arousal line, pitch_std 85, on a
# high-excursion bright reference. 1 of 5, against roughly 4 of 8 on Chatterbox, so the
# rate is lower but not zero. Zonos was briefly treated as a safe haven for bright/teen
# female casting on the strength of one unflagged clip; that was the same weak evidence
# that misled us on Chatterbox, and it was wrong here too.
#
# ⚠ AND THE RATE CLAMP IS NOT THE LAYERING FIX (retake-v1, 2026-07-29). Both clips
# re-rendered at the clamped rate 18 layered again — 0 of 2 — while the three clips whose
# defect was silence/stutter and got a NEUTRAL EMOTION VECTOR instead all came back clean,
# 3 of 3 at score 5. So: emotion instability is real and the documented remedy works;
# rate is a pacing control, not a layering guard. Both failures used the same reference
# that also breaks Chatterbox, which is why REF_BLACKLIST is now shared.
# B-L5: one definition, in ref_select (beside REF_BLACKLIST).
MAX_REF_EXCURSION = _REF_EXCURSION


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

    manifest_path = os.path.join(args.out, "zonos_manifest.jsonl")
    # B-M3: the variety-bias `used` set started EMPTY on every resume, because a job whose
    # wav already exists skips before select_reference() runs. A rerolled clip therefore
    # cast as if the reference pool were untouched — so a resumed campaign quietly loses the
    # diversity guarantee, and the casting is not reproducible from the bank either.
    # Rebuilt from the manifest rows that already exist (synth_common, shared with the
    # renderers that already did this).
    used = rebuild_used_set(manifest_path)
    if used:
        print(f"resume: {len(used)} reference(s) already spent", flush=True)
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            d = job["direction"]
            rate = d["speaking_rate"]
            if rate > MAX_SPEAKING_RATE:
                print(job["id"], f"CLAMP speaking_rate {rate} -> {MAX_SPEAKING_RATE} "
                                 f"(doubling above ~20; see MAX_SPEAKING_RATE)", flush=True)
                rate = MAX_SPEAKING_RATE
            # phoneme count is unknown before phonemization; letters are a safe
            # over-estimate, so this warns early rather than truncating late.
            est = len(job["text"].replace(" ", "")) / max(rate, 1e-6)
            if est > MAX_SECONDS:
                print(job["id"], f"WARN est {est:.0f}s > {MAX_SECONDS}s cap — cut the text, "
                                 f"do not slow the rate", flush=True)

            # A bank line may PIN its reference by pool id (same contract as
            # synth_chatterbox): casting is not consulted and the excursion guard
            # does not apply — an explicit choice by the bank author outranks a
            # guard aimed at casting. This is how a book narrator stays ONE voice
            # across hundreds of clips (zonos.md narration section: "Pin ONE
            # reference per narrator"); per-clip casting would re-consult the pool
            # every line. Added for delivery-v1-narration (2026-07-30).
            # B-L2. Uncaught, a casting failure killed the engine's whole remaining bank
            # from that clip on. moss_vg's per-clip try/except is the pattern; a re-run
            # under skip-if-exists retries only what failed.
            try:
                if job.get("ref_id"):
                    ref_wav, ref_text, ref_meta = pinned_reference(job["ref_id"])
                else:
                    ref_wav, ref_text, ref_meta = select_reference(
                        d.get("design", ""), job.get("intended", {}), used,
                        max_excursion=MAX_REF_EXCURSION, exclude=REF_BLACKLIST)
            except (LookupError, ValueError) as exc:
                print(job["id"], f"CASTING FAILED ({type(exc).__name__}: {exc}) "
                                 "— continuing", flush=True)
                continue
            torch.manual_seed(job["seed"])
            # emotion=None means TRULY unconditional — measured distinction that
            # matters (2026-07-30): make_cond_dict's default unconditional_keys is
            # only {vqscore_8, dnsmos_ovrl}, so "omitting" the emotion arg still
            # CONDITIONS on the default neutral mixture, and passing a
            # neutral-dominant vector conditions harder still. Zyphra's UI ships
            # emotion "off" by ADDING it to unconditional_keys — that is what the
            # skill file's "omit emotion for narration" must compile to. The
            # delivery-v1 narration audit showed the documented emotion-conditioning
            # instability (long clause-boundary pauses, rushed resumption, skipped
            # words) on 5 of 9 zonos groups that carried a neutral-dominant VECTOR.
            uncond = {"vqscore_8", "dnsmos_ovrl"}
            emo = d.get("emotion")
            kw = {}
            if emo is None:
                uncond = uncond | {"emotion"}
            else:
                kw["emotion"] = emo
            cond = make_cond_dict(text=job["text"], language="en-us",
                                  speaker=_speaker_embedding(model, ref_wav),
                                  pitch_std=d["pitch_std"],
                                  speaking_rate=rate,
                                  fmax=FMAX_CLONING,
                                  unconditional_keys=uncond, **kw)
            codes = model.generate(model.prepare_conditioning(cond))
            wav = model.autoencoder.decode(codes).squeeze().float().cpu().numpy()
            write_wav_atomic(os.path.join(args.out, f"{job['id']}.wav"), wav, sr_out)

            row = dict(job)
            row.update({
                "wav": f"{job['id']}.wav", "sr": sr_out,
                "engine_license": "Apache-2.0 (Zyphra/Zonos-v0.1-transformer)",
                "weights_source": "Zyphra/Zonos-v0.1-transformer",
                "campaign": bank["campaign"], "bank_version": bank["version"],
                "intended_age": design_age_band(d.get("design", "")),
                "intended_gender": ref_meta["gender"],
                "ref": ref_meta | {"ref_text": ref_text},
                # `direction` carries what the director ASKED for; this is what the
                # model was actually conditioned on. They differ when the clamp fires.
                "applied_speaking_rate": rate,
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/sr_out:.1f}s pitch_std={d['pitch_std']} "
                             f"rate={rate} ref={ref_meta['id']}", flush=True)
    print("SYNTH-ZONOS-DONE")


if __name__ == "__main__":
    main()

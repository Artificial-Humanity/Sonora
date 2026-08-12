#!/usr/bin/env python3
"""Stage-1 renderer: MOSS-VoiceGenerator (Apache-2.0), bank-consuming.

Replaces the moss85 flagship for any DIRECTED work. The 8.5B flagship is an
un-SFT'd pre-trained foundation model — its card does not list `instruction`
among its input types, and OpenMOSS maintainers state instruction control lives
in this sibling instead. That mismatch is the source of the prompt-read-aloud
leakage the owner found (the base model continues its prompt; when the text is
short next to the instruction it vocalises the instruction). See the owner
finding 2026-07-25.

Interface (model card §Input Types): `text` and `instruction` are BOTH required.
No reference audio — timbre comes from the instruction text alone. The card warns
the model is "sensitive to decoding hyperparameters"; the documented defaults for
this checkpoint are used below and should not be changed casually.

Container recipe: pip transformers soundfile.

Usage:
    python scripts/stages/synth_moss_vg.py --bank <bank.json> --out <dir>
"""
import argparse
import json
import os

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

import sys as _sys
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
from synth_common import attempt_seed, write_wav_atomic  # noqa: E402

MODEL_DIR = "/data/models/OpenMOSS-Team/MOSS-VoiceGenerator"

# Model-card defaults for THIS checkpoint (README §1.3). The flagship's defaults
# differ sharply (1.7/0.8/25/1.0) — do not cross-apply them.
AUDIO_TEMPERATURE = 1.5
AUDIO_TOP_P = 0.6
AUDIO_TOP_K = 50
AUDIO_REPETITION_PENALTY = 1.1

# --- Generation budget (measured 2026-07-31) ----------------------------------
# modeling_moss_tts.py:397 defaults max_new_tokens=1000 and this renderer never
# overrode it, so MOSS silently truncated ANY passage needing more than ~31 s of
# speech. It is a hard ceiling, not a stochastic drift: the two longest renders in
# delivery-v1-narration land at 30.88 s and 30.80 s and nothing in the campaign
# exceeds them. the-return_nar_0050_doc_MOS (840 chars) stopped dead at 30.88 s
# mid-sentence; crock-of-gold_nar_0042_neu_MOS (628 chars) finished at 30.80 s with
# zero tail loss — i.e. it fit with nothing to spare.
#
# Implied audio frame rate: 1000 tokens / 30.88 s ~= 32.4 Hz, which makes the token
# cost of a passage ~1.6 tokens per character at narration pace. TOKENS_PER_CHAR
# carries 1.5x headroom over that so a slow or heavily-paused read still completes.
#
# This does NOT explain MOSS's short-passage truncations (113-266 chars) — those sit
# far below the ceiling and remain the stochastic early-EOS defect. Two distinct
# failures wearing the same symptom; only this one has a lever.
AUDIO_FRAME_RATE_HZ = 32.4
TOKENS_PER_CHAR = 2.4
MIN_NEW_TOKENS = 1000          # never go below the model default
MAX_NEW_TOKENS_CAP = 6000      # ~3 min of audio; a runaway guard, not a target


def _token_budget(text):
    """max_new_tokens sized to the passage, with headroom. See the note above."""
    return int(min(MAX_NEW_TOKENS_CAP, max(MIN_NEW_TOKENS, len(text) * TOKENS_PER_CHAR)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "moss_vg"]
    print(f"{len(jobs)} moss_vg jobs -> {args.out}", flush=True)
    if not jobs:
        print("SYNTH-MOSS-VG-DONE (nothing to do)")
        return

    device = "cuda"
    # normalize_inputs=True enables the card's normalize_instruction(): it flattens
    # newlines and DELETES anything inside [...] or {...}, so never put bracketed
    # stage directions in an instruct string for this model.
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                              normalize_inputs=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16).to(device).eval()
    sr = processor.model_config.sampling_rate

    manifest_path = os.path.join(args.out, "moss_vg_manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as mf, torch.no_grad():
        for job in jobs:
            name = f"{job['id']}.wav"
            if os.path.exists(os.path.join(args.out, name)):
                print(job["id"], "exists, skip", flush=True)
                continue
            # B-M5: a re-run under skip-if-exists is the retry, and it used to re-seed
            # identically — so a DETERMINISTIC failure (MOSS's split_with_sizes off-by-one,
            # a decoder that returns silence) reproduced forever and burned the same GPU
            # time to fail the same way. Perturbed per attempt, and the seed actually used
            # is recorded below so the render stays reproducible from the manifest.
            seed_used, attempt = attempt_seed(args.out, job["id"], job["seed"])
            if attempt:
                print(job["id"], f"attempt {attempt + 1} (seed {seed_used})", flush=True)
            torch.manual_seed(seed_used)
            # One bad generation must not kill the bank (2026-07-30: clip
            # windfairies_nar_0040 hit a split_with_sizes off-by-one inside
            # MOSS's own _parse_audio_codes — a malformed audio-code stream —
            # and the uncaught exception orphaned the 7 clips behind it).
            # Stochastic: a re-run under skip-if-exists retries only the
            # failures, which is exactly the recovery this except enables.
            try:
                msg = processor.build_user_message(
                    text=job["text"], instruction=job["direction"]["instruct"])
                batch = processor([[msg]], mode="generation")
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_new_tokens=_token_budget(job["text"]),
                    audio_temperature=AUDIO_TEMPERATURE,
                    audio_top_p=AUDIO_TOP_P,
                    audio_top_k=AUDIO_TOP_K,
                    audio_repetition_penalty=AUDIO_REPETITION_PENALTY,
                )
                decoded = list(processor.decode(outputs))
            except Exception as e:
                print(job["id"], f"FAILED ({type(e).__name__}: {e}) — continuing",
                      flush=True)
                continue
            # B-L6. This looped over EVERY decoded message writing the same `name`, so a
            # multi-message decode overwrote the wav once per message and appended one
            # manifest row per message — all claiming the same id, all describing a file
            # that by then held only the LAST message's audio. Nothing downstream can see
            # it: the wav is valid, the rows are well-formed, and the id is the join key
            # everything else trusts.
            #
            # One clip is one utterance. Take the first message and say so when there were
            # more, rather than silently concatenating (which would change what the clip
            # IS) or writing several (which is the defect).
            if len(decoded) > 1:
                print(job["id"], f"WARN decode returned {len(decoded)} messages; "
                                 "keeping the first", flush=True)
            for message in decoded[:1]:
                audio = message.audio_codes_list[0].float().cpu().numpy()
                write_wav_atomic(os.path.join(args.out, name), audio, sr)
                # dict(job) first: passthrough fields (intended_delivery, book,
                # ref_id …) must reach the manifest — register_audition prefill
                # and the fold read them from here (2026-07-30).
                row = dict(job)
                row.update({
                    "engine": "moss_vg", "wav": name, "sr": sr,
                    # B-M5: the seed ACTUALLY used, which differs from the bank's on a
                    # retry. `row = dict(job)` would otherwise copy the bank value and the
                    # manifest would describe a render that never happened.
                    "seed": seed_used, "attempt": attempt,
                    "engine_license": "Apache-2.0 (MOSS-VoiceGenerator)",
                    "decoding": {"audio_temperature": AUDIO_TEMPERATURE,
                                 "audio_top_p": AUDIO_TOP_P,
                                 "audio_top_k": AUDIO_TOP_K,
                                 "audio_repetition_penalty": AUDIO_REPETITION_PENALTY,
                                 # Recorded so a future truncation can be told apart
                                 # from the pre-2026-07-31 budget ceiling by reading
                                 # the manifest instead of re-deriving it.
                                 "max_new_tokens": _token_budget(job["text"])},
                    "bank_version": bank["version"], "campaign": bank["campaign"],
                })
                mf.write(json.dumps(row) + "\n")
                mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
                print(job["id"], f"{len(audio)/sr:.1f}s", flush=True)
    print("SYNTH-MOSS-VG-DONE")


if __name__ == "__main__":
    main()

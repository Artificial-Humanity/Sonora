"""Stage-1 renderer: Qwen3-TTS-12Hz-1.7B-Base voice cloning (Apache-2.0).

Reads the script bank, renders every engine=="qwen_base" line, writes wavs +
qwen_base_manifest.jsonl (full provenance per clip) to --out.

Interface (CODE-VERIFIED against qwen-tts 0.1.1, the pinned wheel):
`create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)` builds
the clone prompt — ICL mode, so `ref_text` (the reference's exact transcript) is
required — and `generate_voice_clone(text, language, voice_clone_prompt=...)`
renders. There is NO `instruct` parameter on either call: the voice and its
delivery come from the reference clip, the text and sampling, so
`build_direction` emits only the casting call (`design`) for this engine and
nothing in a bank line's `instruct` can reach the model. `generate_voice_clone`
raises unless the loaded checkpoint is the Base model.

Reference resolution is the shared three-way contract (chatterbox/zonos): a line
pins by file (`ref_file`, sha256-checked) or by pool id (`ref_id`), or is cast
from the certified pool by `select_reference`. The clone prompt is cached per
reference and built ONCE — a probe renders one anchor across 300 lines, and
rebuilding the prompt per clip would re-encode the same reference 300 times.
`torch.manual_seed(job["seed"])` sits immediately before `generate_voice_clone`,
after the cache lookup, so a clip's RNG seeding is identical whether its prompt
was built or reused. Throughput: prompt building is timed on a cache miss as
`ref_condition_seconds` (0.0 on a hit, with `prompt_cached` saying which), the
generate call alone as `render_seconds` (see synth_common.render_timing). There
is no `seed` argument and no batching: per-clip seeds hold only at batch size 1.

The checkpoint loads through `qwen_load.load_qwen`, so the float32 waveform
decoder refusal that guards VoiceDesign guards this engine too (both share the
speech tokenizer).

Batching (Step 1c-ii throughput, pre-registered as not gated): a bank may carry
a top-level `batch_size` > 1. `generate_voice_clone` accepts a list of texts
and broadcasts a one-item prompt list across them (CODE-VERIFIED in the same
wheel), so consecutive jobs sharing one reference render in a single call.
⚠ A batched call draws every clip from one RNG stream: per-clip seeds do NOT
hold, the chunk is seeded once from its first line's seed (recorded as the
row's `seed`), and batched rows carry `batch_size`/`batch_seconds` with
`render_seconds` amortised per clip. Batched clips measure throughput only;
they are never the measured cells. Absent `batch_size`, rendering is the
batch-size-1 path above, unchanged.

Container recipe: synth_bank.sh's qwen_base block (the qwen block's recipe:
apt sox; --no-deps qwen-tts==0.1.1; transformers==4.57.3, soundfile, sox,
onnxruntime, einops, librosa; --no-deps accelerate==1.12.0).
"""
import argparse
import json
import os
import time

import soundfile as sf  # noqa: F401  (container import check, as in synth_qwen)
import torch

import sys as _sys  # noqa: E402
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from qwen_load import TOKENIZER_DTYPE, load_qwen  # noqa: E402
from synth_common import rebuild_used_set, render_timing, write_wav_atomic  # noqa: E402
from ref_select import (  # noqa: E402
    REF_BLACKLIST, design_age_band, pinned_reference, pinned_reference_file,
    select_reference)

MODEL_DIR = "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "qwen_base"]
    # Return BEFORE the shard load when the bank holds no qwen_base lines
    # (synth_qwen's B-L3 lesson: an empty job list must not spend a model load).
    if not jobs:
        print("no qwen_base jobs in bank")
        return
    model = load_qwen(MODEL_DIR)
    print(f"loaded ({len(jobs)} jobs)", flush=True)

    manifest_path = os.path.join(args.out, "qwen_base_manifest.jsonl")
    used = rebuild_used_set(manifest_path)
    if used:
        print(f"resume: {len(used)} reference(s) already spent", flush=True)
    prompts = {}      # (ref_wav, ref_text) -> the built voice-clone prompt
    batch_size = bank.get("batch_size", 1)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise SystemExit("REFUSING: bank batch_size must be a positive int, "
                         "got %r." % (batch_size,))
    if batch_size > 1:
        print(f"BATCHED throughput mode: batch_size={batch_size} "
              "(per-clip seeds do not hold; rows are throughput-only)", flush=True)

    def resolved(job):
        """(ref_wav, ref_text, ref_meta), or None after a printed CASTING FAILED.

        Three-way reference resolution, the shared contract. Casting can raise
        (exhausted pool, unknown ref_id, bad ref_file); uncaught it would kill
        the engine's whole remaining bank, so the failure is per-clip and a
        re-run retries only what failed (B-L2)."""
        d = job["direction"]
        try:
            if "ref_file" in job:    # present = a file pin; a falsy one is refused
                return pinned_reference_file(job)
            if job.get("ref_id"):
                return pinned_reference(job["ref_id"])
            return select_reference(d.get("design", ""), job.get("intended", {}), used,
                                    exclude=REF_BLACKLIST)
        except (LookupError, ValueError) as exc:
            print(job["id"], f"CASTING FAILED ({type(exc).__name__}: {exc}) "
                             "— continuing", flush=True)
            return None

    def prompt_for(ref_wav, ref_text):
        """(prompt, ref_cond_sec, prompt_cached), building and timing on a miss."""
        key = (ref_wav, ref_text)
        if key in prompts:
            return prompts[key], 0.0, True
        t0 = time.monotonic()
        prompts[key] = model.create_voice_clone_prompt(
            ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=False)
        torch.cuda.synchronize()
        return prompts[key], time.monotonic() - t0, False

    def write_row(mf, job, ref_text, ref_meta, wav, sr, prompt_cached,
                  ref_cond_sec, render_sec, extra):
        write_wav_atomic(os.path.join(args.out, f"{job['id']}.wav"), wav, sr)
        row = dict(job)
        row.update({
            "engine": "qwen_base", "wav": f"{job['id']}.wav", "sr": sr,
            "engine_license": "Apache-2.0 (Qwen3-TTS-12Hz-1.7B-Base)",
            "speech_tokenizer_dtype": TOKENIZER_DTYPE,
            "bank_version": bank["version"], "campaign": bank["campaign"],
            "intended_age": design_age_band(job["direction"].get("design", "")),
            "intended_gender": ref_meta["gender"],
            "ref": ref_meta | {"ref_text": ref_text},
            "prompt_cached": prompt_cached,
            **render_timing(ref_cond_sec, render_sec, len(wav) / sr),
            **extra,
        })
        mf.write(json.dumps(row) + "\n")
        mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
        print(job["id"], f"{len(wav)/sr:.1f}s ref={ref_meta['id']} "
                         f"prompt={'cached' if prompt_cached else 'built'}",
              flush=True)

    with open(manifest_path, "a", encoding="utf-8") as mf:
        pending = []
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            ref = resolved(job)
            if ref is not None:
                pending.append((job, *ref))
        i = 0
        while i < len(pending):
            job, ref_wav, ref_text, ref_meta = pending[i]
            # Consecutive jobs sharing one reference form one chunk. At
            # batch_size 1 every chunk is a single clip and this loop is the
            # original per-clip path: resolve, prompt, seed, generate.
            chunk = [pending[i]]
            i += 1
            while (i < len(pending) and len(chunk) < batch_size
                   and pending[i][1] == ref_wav and pending[i][2] == ref_text):
                chunk.append(pending[i])
                i += 1
            prompt, ref_cond_sec, prompt_cached = prompt_for(ref_wav, ref_text)
            # Immediately before generate, AFTER the cache lookup, so the RNG
            # state at the call does not depend on whether the prompt was built.
            torch.manual_seed(chunk[0][0]["seed"])
            t1 = time.monotonic()
            if len(chunk) == 1:
                wavs, sr = model.generate_voice_clone(
                    text=chunk[0][0]["text"], language="English",
                    voice_clone_prompt=prompt)
            else:
                wavs, sr = model.generate_voice_clone(
                    text=[c[0]["text"] for c in chunk], language="English",
                    voice_clone_prompt=prompt)
            torch.cuda.synchronize()
            batch_sec = time.monotonic() - t1
            if len(wavs) != len(chunk):
                raise SystemExit("REFUSING: generate_voice_clone returned %d wavs "
                                 "for a chunk of %d texts." % (len(wavs), len(chunk)))
            for n, ((cjob, _rw, _rt, cmeta), wav) in enumerate(zip(chunk, wavs)):
                extra = {}
                if batch_size > 1:
                    # The seed that actually ran is the chunk's first line's;
                    # row = dict(job) would otherwise record a per-clip seed
                    # the batched RNG stream never honoured.
                    extra = {"seed": chunk[0][0]["seed"], "batch_size": len(chunk),
                             "batch_seconds": round(batch_sec, 4)}
                write_row(mf, cjob, ref_text, cmeta, wav, sr,
                          prompt_cached if n == 0 else True,
                          ref_cond_sec if n == 0 else 0.0,
                          batch_sec / len(chunk), extra)
    print("SYNTH-QWEN-BASE-DONE")


if __name__ == "__main__":
    main()

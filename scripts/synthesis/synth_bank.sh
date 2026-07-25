#!/usr/bin/env bash
# synth_bank.sh — render a synth bank (e.g. book_ingest output, or an authored bank)
# across the teacher engines, each in a throwaway rocm/pytorch container with GPU
# passthrough.
#
# Why throwaway containers: the compose-managed `sonora_vocalizer` container has no GPU
# device passthrough (torch.cuda.is_available() == False there), so synth runs in a
# fresh `rocm/pytorch` container with --device /dev/kfd --device /dev/dri (mirrors the
# ollama container's access; validated on the Strix Halo Radeon 8060S). Models are read
# from /data/models (paths hardcoded in the synth_*.py renderers).
#
# Renders run as ai-mgr (uid 105, gid 109, in datashare) with umask 002 so outputs are
# group-owned and group-writable rather than root-owned — lab convention, extended to
# ad-hoc render containers 2026-07-24.
#
# ENGINE SELECTION (owner finding 2026-07-25): moss_vg — MOSS-VoiceGenerator — is the
# instruction-following MOSS. The 8.5B flagship (moss85) is an un-SFT'd base model that
# does not list `instruction` as an input and reads the prompt aloud on short lines; it
# is kept below only for un-directed / cloning work and is OFF by default.
#
# Usage:  synth_bank.sh <bank.json> <out_dir> [engine ...]
#         engines default to: dia moss_vg qwen vibevoice
# Output: <out_dir>/<id>.wav + <engine>_manifest.jsonl per engine.
set -uo pipefail
BANK="${1:?usage: synth_bank.sh <bank.json> <out_dir> [engine ...]}"
OUT="${2:?usage: synth_bank.sh <bank.json> <out_dir> [engine ...]}"
shift 2
ENGINES=("$@")
[ ${#ENGINES[@]} -eq 0 ] && ENGINES=(dia moss_vg qwen vibevoice)

SONORA="$(cd "$(dirname "$0")/../.." && pwd)"   # Sonora repo root (mounted at /sonora)
GPU="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
IMG=rocm/pytorch:latest
mkdir -p "$OUT"

# Some renderers still reach the Hub at load time for trust_remote_code modules
# (MOSS pulls MOSS-Audio-Tokenizer). Without a token those are rate-limited
# anonymous fetches, which is slow and flaky across a multi-engine run. HF_TOKEN
# lives in the owner's .zshenv, so forward it when present.
HF_ENV=""
[ -n "${HF_TOKEN:-}" ] && HF_ENV="-e HF_TOKEN=$HF_TOKEN -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"

# $1 = pip/apt setup, $2 = the python command to run as ai-mgr
run(){
  docker run --rm $GPU $HF_ENV -v /data:/data -v "$SONORA":/sonora "$IMG" bash -c "
    $1
    bash /sonora/scripts/synthesis/container_as_ai_mgr.sh &&
    runuser -u ai-mgr -- bash -c 'umask 002; HF_TOKEN=\${HF_TOKEN:-} $2'"
}

has(){ for e in "${ENGINES[@]}"; do [ "$e" = "$1" ] && return 0; done; return 1; }

if has dia; then
  echo "== DIA =="
  run "pip install -q transformers soundfile >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_dia.py --bank $BANK --out $OUT" \
      || echo "  (dia failed — continuing)"
fi

if has moss_vg; then
  echo "== MOSS-VoiceGenerator =="
  run "pip install -q transformers soundfile >/dev/null 2>&1; pip install -q --no-deps accelerate >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_moss_vg.py --bank $BANK --out $OUT" \
      || echo "  (moss_vg failed — continuing)"
fi

if has moss85; then
  echo "== MOSS-8.5B flagship (base model — un-directed work only) =="
  run "pip install -q transformers soundfile >/dev/null 2>&1; pip install -q --no-deps accelerate >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_moss85.py --bank $BANK --out $OUT" \
      || echo "  (moss85 failed — continuing)"
fi

if has qwen; then
  echo "== QWEN =="
  run "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y sox >/dev/null 2>&1;
       pip install -q --no-deps qwen-tts >/dev/null 2>&1;
       pip install -q transformers==4.57.3 soundfile sox onnxruntime einops librosa >/dev/null 2>&1;
       pip install -q --no-deps accelerate==1.12.0 >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_qwen.py --bank $BANK --out $OUT" \
      || echo "  (qwen failed — continuing)"
fi

if has vibevoice; then
  # The TTS module lives in the COMMUNITY fork — Microsoft restructured it out of
  # the official repo. Installing transformers alone yields
  # "ModuleNotFoundError: No module named 'vibevoice'". Recipe is v3d/v3e-proven
  # and documented in synth_vibevoice.py's own docstring.
  # NB bitsandbytes 8-bit shard load is ~7.5 min on gfx1151 — amortized over the
  # whole bank, which is why vibevoice runs last.
  echo "== VIBEVOICE =="
  run "pip install -q 'git+https://github.com/vibevoice-community/VibeVoice.git' soundfile bitsandbytes accelerate >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_vibevoice.py --bank $BANK --out $OUT" \
      || echo "  (vibevoice failed — continuing)"
fi

echo "== done: $(ls -1 "$OUT"/*.wav 2>/dev/null | wc -l) wav(s) in $OUT =="

# Register the rendered clips into the audition queue (ratings.csv SSOT) so they
# reach the review surface. Idempotent, host-side (uv, not the GPU container); only
# queues clips whose wav lands under DATA_ROOT. Non-fatal if it can't run.
echo "== register audition =="
if command -v uv >/dev/null 2>&1; then
  uv run "$SONORA/scripts/synthesis/register_audition.py" --audio-dir "$OUT" \
    || echo "  (register_audition failed — clips rendered but not queued; run it manually)"
else
  echo "  (uv not found — skipped; run register_audition.py --audio-dir $OUT to queue)"
fi

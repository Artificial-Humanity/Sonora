#!/usr/bin/env bash
# synth_bank.sh — render a synth bank (e.g. book_ingest output, or an authored bank)
# across the teacher engines, each in a throwaway rocm/pytorch container with GPU
# passthrough.
#
# Why throwaway containers: the GitOps-managed `sonora_vocalizer` container has no GPU
# device passthrough (torch.cuda.is_available() == False there), so synth runs in a
# fresh `rocm/pytorch` container with --device /dev/kfd --device /dev/dri (mirrors the
# ollama container's access; validated on the Strix Halo Radeon 8060S). Models are read
# from /data/reference/models (paths hardcoded in the synth_*.py renderers).
#
# Usage:  synth_bank.sh <bank.json> <out_dir>
# Output: <out_dir>/<id>.wav + <engine>_manifest.jsonl per engine.
set -uo pipefail
BANK="${1:?usage: synth_bank.sh <bank.json> <out_dir>}"
OUT="${2:?usage: synth_bank.sh <bank.json> <out_dir>}"
SONORA="$(cd "$(dirname "$0")/../.." && pwd)"   # Sonora repo root (mounted at /sonora)
GPU="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
IMG=rocm/pytorch:latest
mkdir -p "$OUT"
run(){ docker run --rm $GPU -v /data:/data -v "$SONORA":/sonora "$IMG" bash -c "$1"; }

echo "== DIA =="
run "pip install -q transformers soundfile >/dev/null 2>&1; \
     python /sonora/scripts/synthesis/synth_dia.py --bank $BANK --out $OUT" || echo "  (dia failed — continuing)"

echo "== MOSS-8.5B =="
run "pip install -q transformers soundfile >/dev/null 2>&1; pip install -q --no-deps accelerate >/dev/null 2>&1; \
     python /sonora/scripts/synthesis/synth_moss85.py --bank $BANK --out $OUT" || echo "  (moss failed — continuing)"

echo "== QWEN =="
run "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y sox >/dev/null 2>&1; \
     pip install -q --no-deps qwen-tts >/dev/null 2>&1; \
     pip install -q transformers==4.57.3 soundfile sox onnxruntime einops librosa >/dev/null 2>&1; \
     pip install -q --no-deps accelerate==1.12.0 >/dev/null 2>&1; \
     python /sonora/scripts/synthesis/synth_qwen.py --bank $BANK --out $OUT" || echo "  (qwen failed — continuing)"

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

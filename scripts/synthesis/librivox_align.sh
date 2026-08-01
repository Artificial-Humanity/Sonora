#!/usr/bin/env bash
# librivox_align.sh — run the force-align lane in a throwaway rocm/pytorch container.
#
# Same shape as synth_bank.sh and for the same reasons: the host has no torch, no
# torchaudio and no ffmpeg, and the compose-managed sonora_vocalizer has no GPU
# passthrough. Renders as ai-mgr (105:109) with umask 002 so output is group-writable
# rather than root-owned — lab convention, see [[containers-run-as-ai-mgr]].
#
# Usage:  librivox_align.sh <book_dir> <out_dir> [extra args to librivox_align.py...]
#   e.g.  librivox_align.sh /data/.../book-prose/real-audio/pg_6684 \
#                           /data/.../datasets/librivox-v1 --sections 2 --limit 20
set -uo pipefail
BOOK="${1:?usage: librivox_align.sh <book_dir> <out_dir> [args...]}"
OUT="${2:?usage: librivox_align.sh <book_dir> <out_dir> [args...]}"
shift 2

SONORA="$(cd "$(dirname "$0")/../.." && pwd)"
GPU="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
IMG=rocm/pytorch:latest
mkdir -p "$OUT"

HF_ENV=""
[ -n "${HF_TOKEN:-}" ] && HF_ENV="-e HF_TOKEN=$HF_TOKEN -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"

echo "== librivox align =="
docker run --rm $GPU $HF_ENV -v /data:/data -v "$SONORA":/sonora "$IMG" bash -c "
  pip install -q faster-whisper soundfile librosa >/dev/null 2>&1;
  bash /sonora/scripts/synthesis/container_as_ai_mgr.sh &&
  runuser -u ai-mgr -- bash -c 'umask 002; HF_TOKEN=\${HF_TOKEN:-} \
    python /sonora/scripts/synthesis/librivox_align.py \
      --book-dir \"$BOOK\" --out \"$OUT\" $*'" || {
  echo "  !! align failed"; exit 1; }

# The QC gate follows EVERY dataset-generation pass (owner rule 2026-07-31). Real audio
# is not exempt: a bad alignment produces a clip whose canonical text does not match what
# is spoken, which is exactly what the ASR/WER gate catches.
echo "== qc gate =="
if command -v uv >/dev/null 2>&1; then
  uv run "$SONORA/scripts/synthesis/qc_gate.py" --campaign-dir "$OUT" || {
    echo "  !! qc_gate FAILED — clips are in $OUT but will NOT be queued"; exit 1; }
  uv run "$SONORA/scripts/synthesis/register_audition.py" --audio-dir "$OUT/audio" \
      --qc "$OUT/qc_measures.jsonl" \
    || echo "  (register_audition failed — run it manually)"
else
  echo "  !! uv not found — NOT registering. Run qc_gate.py then register_audition.py."
  exit 1
fi

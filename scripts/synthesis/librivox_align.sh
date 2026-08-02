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

# Value-less -e forwards from this shell's environment instead of writing the
# token into the docker argv, where `ps` and `docker inspect` expose it.
HF_ENV=""
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  HF_ENV="-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN"
fi

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
# Host-side: the repo venv directly, not `uv run` (which binds to pyproject.toml
# and ignores the script's inline PEP 723 block) — owner run-mode rule 2026-08-01.
PY="$SONORA/.venv/bin/python"
if [ -x "$PY" ]; then
  "$PY" "$SONORA/scripts/synthesis/qc_gate.py" --campaign-dir "$OUT" || {
    echo "  !! qc_gate FAILED — clips are in $OUT but will NOT be queued"; exit 1; }
  # DELIBERATELY NOT REGISTERING. The aligner fills the POOL; the pool is not the
  # audition queue. Owner design 2026-08-01: segment everything, then stage a linear
  # slice — so `stage_pool.py` is the ONLY thing that writes this lane into
  # ratings.csv, and it applies the ear-confirmed (reader, title) tags as it goes.
  #
  # Registering here auto-queued 652 unaudited rows off one book on 2026-08-01, which
  # is exactly the flood the pool/staging split exists to prevent. QC still runs above
  # (mandatory, every pass) and its findings ride along in qc_flags.txt; stage_pool
  # reads them and sends any flagged clip to the ear when it is staged.
  echo "  pool filled — NOT queued. Stage it with:"
  echo "    .venv/bin/python scripts/synthesis/stage_pool.py --campaign $(basename "$OUT") --stage N --apply"
else
  echo "  !! $PY not found — the mandatory QC gate cannot run."
  echo "  !! Clips are in $OUT but are NOT gated and must not be staged."
  exit 1
fi

#!/usr/bin/env bash
# capture_container_env.sh — freeze the throwaway container's environment into the
# campaign it is producing. Runs INSIDE the container.
#
#   capture_container_env.sh <tag> <env_dir>
#
# A separate script rather than an inlined string on purpose: the three call sites already
# cross two quoting layers (docker `bash -c`, then runuser `bash -c`), and the freeze
# command needs both single and double quotes. Inlining it is how you get a capture that
# silently emits nothing while the render succeeds — which is the exact failure mode this
# is here to end.
#
# Never fatal. A missing freeze is a gap in the record; a freeze that aborts the run
# would be a reproducibility measure that destroys work.
set -u

TAG="${1:?usage: capture_container_env.sh <tag> <env_dir>}"
ENV_DIR="${2:?usage: capture_container_env.sh <tag> <env_dir>}"

mkdir -p "$ENV_DIR" 2>/dev/null || { echo "  (env capture: cannot create $ENV_DIR)"; exit 0; }

OUT="$ENV_DIR/$TAG.txt"
{
  echo "# lane: $TAG"
  echo "# image: ${SONORA_ROCM_IMAGE:-unrecorded}"
  echo "# captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # The one package that is NOT installable from PyPI here and is therefore absent from
  # the freeze below — record it explicitly or the record has a hole exactly where the
  # ROCm-vs-CUDA question gets asked.
  echo "# torch: $(/opt/venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unavailable)"
  echo "#"
  echo "# Written by scripts/capture_container_env.sh (D-M2). This is what ran, not what"
  echo "# pyproject permits. Pins live in scripts/container_env.sh; refresh them FROM"
  echo "# these files rather than by editing a version in place and hoping."
  uv pip freeze --python /opt/venv/bin/python 2>/dev/null \
    || /opt/venv/bin/pip freeze 2>/dev/null \
    || echo "# freeze unavailable"
} > "$OUT" 2>/dev/null || { echo "  (env capture: cannot write $OUT)"; exit 0; }

echo "  env captured: $OUT"
exit 0

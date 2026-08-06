#!/usr/bin/env bash
# score_holdout.sh — run score_holdout.py in a throwaway ROCm container.
#
# Same shape and same reason as eiv_score.sh: the host venv carries the synthesis lane
# only — no torch, and the model stack (conformer, diffusers, lightning) is not
# something the host should grow. The compose-managed sonora_training service is not
# usable for this either, because `up` STARTS the run in its `command:`.
#
# The GPU here is doing teacher-forced forward passes, not training, so the spin-down
# rule does not apply — but it is still a full model per checkpoint, so do not run this
# against a busy card and expect the clip rate to mean anything.
#
# monotonic_align is Cython and ships as a .pyx, so something has to compile it. That
# happens in a COPY of the package under /tmp, never in the mounted repo: the checkout is
# owned by the host user, ai-mgr cannot write to it (Cython fails on core.c), and even
# with the permissions loosened this would leave build artifacts in a working tree the
# owner is editing. The copy is a few MB — matcha/, scripts/, configs/ and the two files
# setup.py reads — and it dies with the container.
#
# PATHS ARE CONTAINER PATHS. Only /data and the repo are mounted, so pass filelists and
# checkpoints as /data/... — a host path like /home/<user>/... does not exist in here.
#
# Usage:  score_holdout.sh [args to score_holdout.py]
#   e.g.  score_holdout.sh --filelist /data/.../libritts_r_holdout_devclean/holdout.txt \
#             --model-config <run>/.hydra/config.yaml --ckpt a=/path/a.ckpt --out out.csv
set -uo pipefail

SONORA="$(cd "$(dirname "$0")/.." && pwd)"
GPU="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
IMG=rocm/pytorch:latest

# Kept to one line so a dependency added here is visibly the same list score_holdout.py
# imports. NOT in requirements.txt: this is an eval lane, and the training image stays lean.
# NOTE: no `phonemizer`. matcha.text.cleaners imports it lazily precisely so the
# espeak-free lane holds, the filelists here are already IPA, and the data config runs
# `no_cleaners` — so pulling in a GPL dependency to satisfy an import that never fires
# would be the G-3/G-4 mistake, voluntarily.
DEPS="einops conformer diffusers lightning hydra-core omegaconf rootutils rich matplotlib gdown wget librosa soundfile cython numpy pyyaml unidecode"

echo "== score holdout =="
docker run --rm $GPU -v /data:/data -v "$SONORA":/sonora "$IMG" bash -c "
  pip install -q uv >/dev/null 2>&1;
  uv pip install -q --python \"\$(which python)\" $DEPS >/dev/null 2>&1;
  bash /sonora/scripts/synthesis/container_as_ai_mgr.sh &&
  mkdir -p /tmp/sonora && cp -a /sonora/setup.py /sonora/README.md /sonora/requirements.txt \
      /sonora/matcha /sonora/scripts /sonora/configs /tmp/sonora/ && chown -R ai-mgr /tmp/sonora &&
  runuser -u ai-mgr -- bash -c 'umask 002; cd /tmp/sonora && \
    python setup.py build_ext --inplace >/dev/null && \
    SONORA_REPO=/tmp/sonora PYTHONPATH=/tmp/sonora python /tmp/sonora/scripts/score_holdout.py $*'" || {
  echo "  !! score_holdout failed"; exit 1; }

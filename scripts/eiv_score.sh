#!/usr/bin/env bash
# eiv_score.sh — run the LAION Empathic-Insight-Voice labeler in a throwaway container.
#
# Same shape and same reason as librivox_align.sh: the host has no torch and no
# transformers, and the compose-managed sonora_vocalizer has no GPU passthrough. This
# wrapper did not exist until 2026-08-01 — every prior EIV pass was launched by hand,
# which is why the corpus scores ended up split across two files with different head
# sets (corpus_v1.jsonl and corpus_families.jsonl) and no record of the invocation.
#
# eiv_score.py APPENDS and is resumable: it reads --out on start-up and skips wavs it
# already holds. Re-running after an interruption is safe and is the intended recovery.
#
# NOTE ON THE GPU: this is a LABELLING pass, not training, so the spin-down rule
# ([[spin-down-inference-before-training]]) does not apply — but the encoder still wants
# VRAM alongside whatever inference stack is up. EmoWhisper-Small + 12 heads is small;
# if it OOMs, stop the renderers rather than shrinking the batch, so the scores stay
# comparable to the existing corpus.
#
# Usage:  eiv_score.sh <out.jsonl> <input...> [-- extra args to eiv_score.py]
#   e.g.  eiv_score.sh /data/.../gap.jsonl /path/to/wavlist.txt -- --heads "Valence,Arousal"
set -uo pipefail
OUT="${1:?usage: eiv_score.sh <out.jsonl> <input...> [-- args]}"
shift
INPUTS=()
EXTRA=()
seen_sep=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_sep=1; continue; fi
  if [ "$seen_sep" -eq 1 ]; then EXTRA+=("$a"); else INPUTS+=("$a"); fi
done
[ "${#INPUTS[@]}" -gt 0 ] || { echo "no inputs given"; exit 1; }

SONORA="$(cd "$(dirname "$0")/.." && pwd)"
GPU="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
# shellcheck source=scripts/container_env.sh
. "$SONORA/scripts/container_env.sh"
IMG="$SONORA_ROCM_IMAGE"
mkdir -p "$(dirname "$OUT")"

# D-M2: the scores are treated as immutable, so the environment that produced them is
# part of the record. It lands beside the scores rather than in a log, because the log is
# not what gets read two months later when a head disagrees with itself.
ENV_DIR="$(dirname "$OUT")/_env"
mkdir -p "$ENV_DIR"

echo "== eiv score -> $OUT =="
docker run --rm $GPU -e SONORA_ROCM_IMAGE="$IMG" \
  -v /data:/data -v "$SONORA":/sonora "$IMG" bash -c "
  $SONORA_UV_BOOTSTRAP;
  $SONORA_UVPIP $SONORA_PIN_TRANSFORMERS librosa soundfile >/dev/null 2>&1;
  bash /sonora/scripts/synthesis/container_as_ai_mgr.sh &&
  $(capture_env_cmd eiv "$ENV_DIR")
  runuser -u ai-mgr -- bash -c 'umask 002; \
    python /sonora/scripts/eiv_score.py --out \"$OUT\" \
      --inputs ${INPUTS[*]} ${EXTRA[*]}'" || {
  echo "  !! eiv_score failed — $OUT holds whatever completed; re-run to resume"; exit 1; }

echo "  scored rows: $(wc -l < "$OUT")"
echo "  environment: $ENV_DIR/eiv.txt"

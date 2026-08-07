#!/usr/bin/env bash
# Run a LiteRT export script FROM THE REPO, with its data on /data.
#
#   scripts/litert_export/run.sh convert_vat.py
#   scripts/litert_export/run.sh build_matcha.py parity
#
# Owner principle (2026-08-06): code executes from the repo checkout; `/data` holds what
# its name implies — datasets, checkpoints, artifacts, venvs, logs. This wrapper is what
# makes that practical for a harness whose scripts used to read and write everything next
# to themselves:
#
#   SONORA_LITERT_WORK  where the DATA is    (checkpoints in, .tflite/.wav/artifacts* out)
#   SONORA_REPO         where the MODEL CODE is (the Sonora repo root, for `import matcha`)
#   the interpreter     the harness venv, which lives with the data because it is 6.6 GB
#
# Until 2026-08-06 the scripts lived on /data too, as a byte copy of this directory, purely
# because running them from the repo would have dumped ~400 MB of graphs into the working
# tree. That copy drifted: convert_vat.py there ran three weeks stale, missing a seam guard
# this repo recorded as landed, and nothing detected it. Splitting code from data is what
# retired the copy rather than merely policing it (AGENTS.md §6, notes/data-mirrors.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SONORA_LITERT_WORK="${SONORA_LITERT_WORK:-/data/toolchain/litert-conversion}"
export SONORA_REPO="${SONORA_REPO:-$(cd "${HERE}/../.." && pwd)}"
PY="${SONORA_LITERT_PY:-${SONORA_LITERT_WORK}/.venv/bin/python}"

usage() { tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//' | head -20; }
[ $# -ge 1 ] || { usage; exit 1; }

name="$1"
script="${HERE}/${name}"; shift
[ -f "$script" ] || { echo "!! no such script: $script" >&2; exit 1; }
[ -x "$PY" ] || {
  echo "!! no harness interpreter at $PY" >&2
  echo "   The venv lives with the data (it is ~6.6 GB). Create it with:" >&2
  echo "   uv venv ${SONORA_LITERT_WORK}/.venv && uv pip install --python $PY \\" >&2
  echo "       litert-torch ai-edge-litert ai-edge-quantizer  # see README.md" >&2
  exit 1
}
[ -d "${SONORA_LITERT_WORK}" ] || { echo "!! no work dir: ${SONORA_LITERT_WORK}" >&2; exit 1; }

echo "== ${name}"
echo "   code:  ${HERE}"
echo "   data:  ${SONORA_LITERT_WORK}"
echo "   model: ${SONORA_REPO}"
# cd to the work dir so any path a script still resolves relatively lands on /data rather
# than in the checkout. Belt to the WORK braces.
cd "${SONORA_LITERT_WORK}"
exec "$PY" "$script" "$@"

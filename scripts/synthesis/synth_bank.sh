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
#         engines default to the survivor portfolio: chatterbox moss_vg orpheus qwen zonos
#         (vibevoice + dia are SET ASIDE, owner 2026-07-29 — see ref_select.SET_ASIDE;
#         their run blocks remain below, opt-in by name, for if they are reinstated)
# Output: <out_dir>/<id>.wav + <engine>_manifest.jsonl per engine.
set -uo pipefail
BANK="${1:?usage: synth_bank.sh <bank.json> <out_dir> [engine ...]}"
OUT="${2:?usage: synth_bank.sh <bank.json> <out_dir> [engine ...]}"
shift 2
ENGINES=("$@")
[ ${#ENGINES[@]} -eq 0 ] && ENGINES=(chatterbox moss_vg orpheus qwen zonos)

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

# --- revisit engines (2026-07-28) -------------------------------------------------
# Off by default and named explicitly, because their prior verdicts were void and a
# clip is only worth what its provenance is worth: audition ONLY output from a fully
# fixed pipeline. Their pip recipes are the first ever recorded for these engines and
# are UNVERIFIED until a smoke render passes — fix them here, not in a shell history.

if has chatterbox; then
  # CLASSIC English weights, not Turbo (Turbo discards `exaggeration`).
  # resemble-perth is imported and patched to a no-op by the renderer: the native
  # watermarker is unavailable here, and these clips are TRAIN-ONLY by owner decision.
  echo "== CHATTERBOX (train-only output) =="
  run "pip install -q --no-deps chatterbox-tts >/dev/null 2>&1;
       pip install -q librosa s3tokenizer diffusers resemble-perth conformer transformers safetensors soundfile omegaconf >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_chatterbox.py --bank $BANK --out $OUT" \
      || echo "  (chatterbox failed — continuing)"
fi

if has zonos; then
  # RUN FROM SOURCE, NOT PIP. `pip install git+…Zonos` yields an UNIMPORTABLE
  # package: the wheel omits the zonos.backbone subpackage entirely (verified
  # 2026-07-28 — model.py ships, zonos/backbone/ does not), so the first import
  # dies on ModuleNotFoundError no matter what deps are present. The checkout at
  # /data/toolchain/Zonos is the source of truth; PYTHONPATH is the whole fix.
  #   git clone --depth 1 https://github.com/Zyphra/Zonos.git /data/toolchain/Zonos
  # espeak-ng is a hard requirement (Zonos phonemizes before conditioning), and
  # sudachipy/sudachidict-full are NOT optional despite being the Japanese
  # tokenizer: conditioning.py imports them at module scope, so an English-only
  # run still fails without them.
  echo "== ZONOS =="
  run "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y espeak-ng >/dev/null 2>&1;
       pip install -q phonemizer inflect kanjize soundfile transformers huggingface-hub sudachipy sudachidict-full >/dev/null 2>&1;" \
      "PYTHONPATH=/data/toolchain/Zonos python /sonora/scripts/synthesis/synth_zonos.py --bank $BANK --out $OUT" \
      || echo "  (zonos failed — continuing)"
fi

if has orpheus; then
  # snac decodes the audio tokens; the renderer pulls snac_24khz from the Hub.
  echo "== ORPHEUS (-ft checkpoint) =="
  run "pip install -q transformers soundfile snac >/dev/null 2>&1;" \
      "python /sonora/scripts/synthesis/synth_orpheus.py --bank $BANK --out $OUT" \
      || echo "  (orpheus failed — continuing)"
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

# Bring every engine to one loudness BEFORE anything measures or auditions the
# clips. Measured 2026-07-28: 5.1 dB spread between per-engine medians. That
# biases the ear in audit (louder reads as more present before the performance is
# judged) and spreads the conditioning signal for every cloning engine, since v1
# is the pool ref_select draws references from. It does NOT leak into the
# instrument's arousal channel — EIV normalises each clip against its own maximum,
# so a constant gain cancels; that claim was asserted here without reading the
# consumer and was retracted the same day (see normalize_loudness.py's docstring).
# Must run before register_audition (audit) and before qc_gate (measurement).
# Originals are kept in $OUT/_pre_loudnorm.
echo "== normalize loudness =="
if command -v uv >/dev/null 2>&1; then
  uv run "$SONORA/scripts/synthesis/normalize_loudness.py" --dir "$OUT" \
    || echo "  (normalize_loudness failed — clips are at raw engine level; DO NOT audition until fixed)"
else
  echo "  (uv not found — skipped; run normalize_loudness.py --dir $OUT before auditioning)"
fi

# Objective QC BEFORE the ear ever sees a clip. This step used to be manual, and
# nothing here invoked it — so every batch reached the audition queue ungated unless
# somebody remembered. Measured cost on the 2026-07-31 reroll: a 3.1 s MOSS clip
# (ASR WER 0.72, 11 of 40 words) went straight to the owner, and a second clip that
# lost 27 of its 47 words was scored 5 by ear because it ended cleanly mid-sentence.
# Both are trivially machine-detectable. The gate is stage 1 only (measures + hard
# gates); qc_verdict.py still owns the intended-vs-measured merge.
# OWNER RULE 2026-07-31: the QC run FOLLOWS EVERY dataset-generation pass. It is not
# advisory and it is not optional, so a failure here stops the pipeline rather than
# quietly queueing ungated clips — an ungated batch is exactly how a 3.1 s truncation
# reached the ear, and how a clip missing 27 of its 47 words got scored 5.
CAMPAIGN_DIR="$(dirname "$OUT")"
echo "== qc gate =="
QC_MEASURES=""
if ! command -v uv >/dev/null 2>&1; then
  echo "  !! uv not found — cannot run the QC gate."
  echo "  !! NOT registering: clips are rendered in $OUT but stay out of the queue."
  echo "  !! Recover with: qc_gate.py --campaign-dir $CAMPAIGN_DIR"
  echo "  !!         then: register_audition.py --audio-dir $OUT --qc $CAMPAIGN_DIR/qc_measures.jsonl"
  exit 1
fi
if uv run "$SONORA/scripts/synthesis/qc_gate.py" --campaign-dir "$CAMPAIGN_DIR"; then
  QC_MEASURES="$CAMPAIGN_DIR/qc_measures.jsonl"
else
  echo "  !! qc_gate FAILED — clips are rendered in $OUT but will NOT be queued."
  echo "  !! Fix the gate, then:"
  echo "  !!   qc_gate.py --campaign-dir $CAMPAIGN_DIR"
  echo "  !!   register_audition.py --audio-dir $OUT --qc $CAMPAIGN_DIR/qc_measures.jsonl"
  exit 1
fi

# Register the rendered clips into the audition queue (ratings.csv SSOT) so they
# reach the review surface. Idempotent, host-side (uv, not the GPU container); only
# queues clips whose wav lands under DATA_ROOT. Non-fatal if it can't run.
# With --qc, structural failures (ASR / 4 s floor) register as `reroll` instead of
# spending an audition slot; advisory flags are still queued AND written to
# qc_flags.txt for pick_audit_subset --flags, because the QC-flag path never relaxes.
echo "== register audition =="
if command -v uv >/dev/null 2>&1; then
  uv run "$SONORA/scripts/synthesis/register_audition.py" --audio-dir "$OUT" \
      ${QC_MEASURES:+--qc "$QC_MEASURES"} \
    || echo "  (register_audition failed — clips rendered but not queued; run it manually)"
else
  echo "  (uv not found — skipped; run register_audition.py --audio-dir $OUT to queue)"
fi

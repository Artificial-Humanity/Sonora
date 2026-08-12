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

# Strip trailing slashes before anything derives a path from $OUT. The QC gate's
# campaign dir is `dirname "$OUT"`, so a single trailing slash used to make it
# the campaign dir itself, the manifest glob match nothing, and the mandatory
# gate "pass" over zero clips.
OUT="${OUT%/}"
while [ "$OUT" != "${OUT%/}" ]; do OUT="${OUT%/}"; done

# Both paths cross two quoting layers (docker bash -c, then runuser bash -c),
# so a space or a quote in either breaks the render in a way that surfaces as a
# confusing engine error. They must also live under /data, which is the only
# tree mounted into the containers.
case "$BANK" in /data/*) ;; *) echo "!! BANK must be an absolute /data path: $BANK" >&2; exit 2;; esac
case "$OUT"  in /data/*) ;; *) echo "!! OUT must be an absolute /data path: $OUT"  >&2; exit 2;; esac
case "$BANK$OUT" in *[\ \'\"]*) echo "!! BANK/OUT must not contain spaces or quotes" >&2; exit 2;; esac
ENGINES=("$@")
[ ${#ENGINES[@]} -eq 0 ] && ENGINES=(chatterbox moss_vg orpheus qwen zonos)

SONORA="$(cd "$(dirname "$0")/../.." && pwd)"   # Sonora repo root (mounted at /sonora)
GPU="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
# Pinned image digest + pinned wheels + the per-campaign freeze, in one place for all
# three container lanes. See its header for what unpinned resolution does today.
# shellcheck source=scripts/container_env.sh
. "$SONORA/scripts/container_env.sh"
IMG="$SONORA_ROCM_IMAGE"
mkdir -p "$OUT"
# One freeze per ENGINE, not per campaign: the engines install different wheel sets into
# different throwaway containers, so a single campaign-level record would describe at most
# one of them and imply the rest.
ENV_DIR="$OUT/_env"
mkdir -p "$ENV_DIR"

# Pre-flight the bank before any GPU time is spent. orpheus and dia render from
# `direction.render_text` while the WER gate, the manifest and the corpus all read
# `line["text"]`; when those two drift apart the clip says one thing and every
# record around it claims another. QC is structurally blind to it — it scores the
# audio against `text`, so it is comparing to the wrong reference — and one wrong
# word in a 35-word line is WER 0.03 regardless. Hard stop: rendering a bank we
# know is inconsistent only manufactures clips that must be thrown away.
if ! "$SONORA/.venv/bin/python" "$SONORA/scripts/stages/check_bank.py" "$BANK"; then
  echo "!! refusing to render an inconsistent bank" >&2
  exit 2
fi

# Some renderers still reach the Hub at load time for trust_remote_code modules
# (MOSS pulls MOSS-Audio-Tokenizer). Without a token those are rate-limited
# anonymous fetches, which is slow and flaky across a multi-engine run. HF_TOKEN
# lives in the owner's .zshenv, so forward it when present.
# Value-less -e forwards from this shell's environment instead of writing the
# token into the docker argv, where `ps` and `docker inspect` would show it.
HF_ENV=""
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  HF_ENV="-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN"
fi

# $1 = engine tag (names the environment freeze), $2 = apt/uv setup, $3 = the python
# command to run as ai-mgr.
run(){
  docker run --rm $GPU $HF_ENV -e SONORA_ROCM_IMAGE="$IMG" \
    -v /data:/data -v "$SONORA":/sonora "$IMG" bash -c "
    $SONORA_UV_BOOTSTRAP;
    $2
    bash /sonora/scripts/container_as_ai_mgr.sh &&
    $(capture_env_cmd "$1" "$ENV_DIR")
    runuser -u ai-mgr -- bash -c 'umask 002; HF_TOKEN=\${HF_TOKEN:-} $3'"
}

has(){ for e in "${ENGINES[@]}"; do [ "$e" = "$1" ] && return 0; done; return 1; }

# Engine failures are non-fatal per engine (one dead engine should not throw
# away the others' renders) but they are counted, so a run where every engine
# died at clip 1 can no longer end in a cheerful exit 0.
FAILED_ENGINES=0
FAILED_ENGINE_NAMES=""
failed(){
  FAILED_ENGINES=$((FAILED_ENGINES + 1))
  FAILED_ENGINE_NAMES="${FAILED_ENGINE_NAMES:+$FAILED_ENGINE_NAMES }$1"
  echo "  ($1 failed — continuing)"
}

# Host-side steps run the repo venv directly. `uv run` binds to pyproject.toml
# and ignores each script's inline PEP 723 block, which is why the owner moved
# host scripts to .venv/bin/python on 2026-08-01. uv still manages the venv:
#   uv pip install --python .venv/bin/python <pkg>
PY="$SONORA/.venv/bin/python"

if has dia; then
  echo "== DIA =="
  run dia "$SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_dia.py --bank $BANK --out $OUT" \
      || failed dia
fi

if has moss_vg; then
  echo "== MOSS-VoiceGenerator =="
  run moss_vg "$SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile >/dev/null 2>&1; $SONORA_UVPIP --no-deps accelerate >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_moss_vg.py --bank $BANK --out $OUT" \
      || failed moss_vg
fi

if has moss85; then
  echo "== MOSS-8.5B flagship (base model — un-directed work only) =="
  run moss85 "$SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile >/dev/null 2>&1; $SONORA_UVPIP --no-deps accelerate >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_moss85.py --bank $BANK --out $OUT" \
      || failed moss85
fi

if has qwen; then
  echo "== QWEN =="
  run qwen "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y sox >/dev/null 2>&1;
       $SONORA_UVPIP --no-deps qwen-tts >/dev/null 2>&1;
       $SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile sox onnxruntime einops librosa >/dev/null 2>&1;
       $SONORA_UVPIP --no-deps accelerate==1.12.0 >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_qwen.py --bank $BANK --out $OUT" \
      || failed qwen
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
  run chatterbox "$SONORA_UVPIP --no-deps chatterbox-tts >/dev/null 2>&1;
       $SONORA_UVPIP librosa s3tokenizer diffusers resemble-perth conformer $SONORA_PIN_TRANSFORMERS safetensors soundfile omegaconf >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_chatterbox.py --bank $BANK --out $OUT" \
      || failed chatterbox
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
  run zonos "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y espeak-ng >/dev/null 2>&1;
       $SONORA_UVPIP phonemizer inflect kanjize soundfile $SONORA_PIN_TRANSFORMERS huggingface-hub sudachipy sudachidict-full >/dev/null 2>&1;" \
      "PYTHONPATH=/data/toolchain/Zonos python /sonora/scripts/stages/synth_zonos.py --bank $BANK --out $OUT" \
      || failed zonos
fi

if has orpheus; then
  # snac decodes the audio tokens; the renderer pulls snac_24khz from the Hub.
  echo "== ORPHEUS (-ft checkpoint) =="
  run orpheus "$SONORA_UVPIP $SONORA_PIN_TRANSFORMERS soundfile snac >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_orpheus.py --bank $BANK --out $OUT" \
      || failed orpheus
fi

if has vibevoice; then
  # The TTS module lives in the COMMUNITY fork — Microsoft restructured it out of
  # the official repo. Installing transformers alone yields
  # "ModuleNotFoundError: No module named 'vibevoice'". Recipe is v3d/v3e-proven
  # and documented in synth_vibevoice.py's own docstring.
  # NB bitsandbytes 8-bit shard load is ~7.5 min on gfx1151 — amortized over the
  # whole bank, which is why vibevoice runs last.
  echo "== VIBEVOICE =="
  run vibevoice "$SONORA_UVPIP 'git+https://github.com/vibevoice-community/VibeVoice.git' soundfile bitsandbytes accelerate >/dev/null 2>&1;" \
      "python /sonora/scripts/stages/synth_vibevoice.py --bank $BANK --out $OUT" \
      || failed vibevoice
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
if [ ! -x "$PY" ]; then
  echo "  !! $PY not found — cannot normalize loudness."
  echo "  !! NOT registering: clips are rendered in $OUT but stay out of the queue."
  echo "  !! Recover with: $PY $SONORA/scripts/stages/normalize_loudness.py --dir $OUT"
  exit 1
fi
# A failure here used to print "DO NOT audition until fixed" and then carry on
# to register the batch for audition — the message and the control flow said
# opposite things. Unnormalized clips bias the ear (louder reads as more
# present) and spread the conditioning signal, so this stops like the QC gate.
if ! "$PY" "$SONORA/scripts/stages/normalize_loudness.py" --dir "$OUT"; then
  echo "  !! normalize_loudness FAILED — clips are at raw engine level."
  echo "  !! NOT registering: they stay out of the queue until this is fixed."
  echo "  !! Recover with: $PY $SONORA/scripts/stages/normalize_loudness.py --dir $OUT"
  exit 1
fi

# Objective QC BEFORE the ear ever sees a clip. This step used to be manual, and
# nothing here invoked it — so every batch reached the audition queue ungated unless
# somebody remembered. Measured cost on the 2026-07-31 reroll: a 3.1 s MOSS clip
# (ASR WER 0.72, 11 of 40 words) went straight to the owner, and a second clip that
# lost 27 of its 47 words was scored 5 by ear because it ended cleanly mid-sentence.
# Both are trivially machine-detectable. The gate is stage 1 only (measures + hard
# gates); qc_verdict.py owns the intended-vs-measured merge and runs right after it.
# OWNER RULE 2026-07-31: the QC run FOLLOWS EVERY dataset-generation pass. It is not
# advisory and it is not optional, so a failure here stops the pipeline rather than
# quietly queueing ungated clips — an ungated batch is exactly how a 3.1 s truncation
# reached the ear, and how a clip missing 27 of its 47 words got scored 5.
CAMPAIGN_DIR="$(dirname "$OUT")"
# The layout the gate assumes is <campaign>/<engine-out>/, so that
# `<campaign>/*/ *_manifest.jsonl` matches this campaign and nothing else. A
# single-level OUT (e.g. /data/model-training/datasets/foo-v1) makes
# CAMPAIGN_DIR the datasets ROOT: the gate then globs every campaign at once
# and drops qc_measures.jsonl in the shared root. newscaster-v1 did exactly
# this on 2026-08-02 and only stayed correct because it happened to be the
# one campaign with top-level manifests.
DATASETS_ROOT="/data/model-training/datasets"
if [ "$CAMPAIGN_DIR" = "$DATASETS_ROOT" ]; then
  echo "!! OUT is a single-level campaign dir ($OUT), so the QC campaign dir"
  echo "!! would be the datasets root — the gate would sweep every campaign"
  echo "!! and write qc_measures.jsonl into the shared root."
  echo "!! Render into <campaign>/<engine-out>, e.g. $OUT/renders" >&2
  exit 2
fi
echo "== qc gate =="
QC_MEASURES=""
if ! "$PY" "$SONORA/scripts/stages/qc_gate.py" --campaign-dir "$CAMPAIGN_DIR"; then
  echo "  !! qc_gate FAILED — clips are rendered in $OUT but will NOT be queued."
  echo "  !! Fix the gate, then:"
  echo "  !!   $PY $SONORA/scripts/stages/qc_gate.py --campaign-dir $CAMPAIGN_DIR"
  echo "  !!   $PY $SONORA/scripts/stages/register_audition.py --audio-dir $OUT --qc $CAMPAIGN_DIR/qc_measures.jsonl"
  exit 1
fi
QC_MEASURES="$CAMPAIGN_DIR/qc_measures.jsonl"
# The gate exiting 0 is not sufficient on its own: it must also have produced
# measurements. An empty measures file means it gated nothing.
if [ ! -s "$QC_MEASURES" ]; then
  echo "  !! qc_gate exited 0 but $QC_MEASURES is empty — nothing was measured."
  echo "  !! NOT registering. Check that $CAMPAIGN_DIR is the campaign dir."
  exit 1
fi

# QC stage 2: the direction check — does the render point where the DIRECTOR said?
#
# Stage 1 above is nine hard gates and every one of them is intrinsic audio quality; not
# one reads `intended{V,A,T}`, which sits unused in the same manifest row. qc_verdict.py
# has existed for that since 2026-07-17 and nothing invoked it — the third instance in
# this file of a written instrument nobody ran, after qc_gate itself and the defect
# detectors below. 695 clips across five campaigns reached the ear with no direction
# check at all.
#
# ADVISORY, AND DELIBERATELY NOT SHAPED LIKE THE GATE ABOVE. `keep` in qc_verdict is a
# LABEL-CONFIRMATION verdict: it asks whether the measured affect points the same way as
# the intended labels, not whether the clip sounds good. gate_calibration.py's own header
# spells out the consequence — "a beautiful clip whose director-assigned labels were wrong
# is a legitimate not-keep" — so wiring this fail-closed would DISCARD GOOD AUDIO because
# Gemma mislabelled it. It runs on the qc_engine_defects pattern instead: always runs,
# writes verdicts, appends axis failures to qc_flags.txt, never drops a clip, non-fatal
# and announced when it fails.
#
# THE ORDER IS THE POINT. The verdicts must be frozen BEFORE register_audition puts these
# clips in front of the ear. gate_calibration.py can only measure the instrument against
# the ear when the instrument's verdict predates the rating — "afterwards there is nothing
# to compare against and the round is spent", and dataset pipeline 1.0 (spot-checking
# instead of auditing every clip) is gated on exactly that measurement. A backfill can
# still measure the drift; it cannot un-spend a round.
echo "== qc verdicts (direction: intended vs measured) =="
# Real-audio banks carry no `intended` labels at all (librivox-v1: 664 rows), so there is
# nothing to confirm and no reason to spend a GPU labelling pass. Ask the verdict script
# rather than re-deriving "is this campaign directed?" here, where the two answers could
# drift apart.
DIRECTED="$("$PY" "$SONORA/scripts/stages/qc_verdict.py" \
    --campaign-dir "$CAMPAIGN_DIR" --count-directed)" || DIRECTED="probe-failed"
case "$DIRECTED" in
  ''|*[!0-9]*)
    # "The probe did not answer" is NOT "this bank has no labels", and collapsing the two
    # would rebuild the hole this fix closes: a directed bank would sail past a message
    # saying it was real audio. Announced, and the count is printed as it stands.
    echo "  !! could not tell whether this campaign carries intended labels ($DIRECTED)."
    echo "  !! Direction check SKIPPED for this bank — clips are gated and still queued."
    echo "  !! Recover with: $PY $SONORA/scripts/stages/qc_verdict.py --campaign-dir $CAMPAIGN_DIR --count-directed"
    ;;
  0)
    # Names the file it read, so "no labels" and "no measures" are told apart by whoever
    # is reading the log rather than looking identical.
    echo "  no intended V/A/T in $CAMPAIGN_DIR/qc_measures.jsonl (real-audio bank)"
    echo "  — nothing to confirm; direction check skipped."
    ;;
  *)
    echo "  $DIRECTED clip(s) carry intended V/A/T"
    # The head set comes from qc_verdict itself: eiv_score.py's DEFAULT is four heads while
    # V is a weighted combination of twelve, and the missing ones used to be silently
    # substituted with 0.0 — which is a fixed offset, not a neutral value. That is what
    # happened to revisit-v1, the last campaign this stage ever ran on.
    EIV_SCORES="$CAMPAIGN_DIR/eiv_scores.jsonl"
    EIV_HEADS="$("$PY" "$SONORA/scripts/stages/qc_verdict.py" --print-heads)" || EIV_HEADS=""
    if [ -z "$EIV_HEADS" ]; then
      echo "  !! could not read the EIV head set — NO direction verdicts for this bank."
      echo "  !! Recover with: $PY $SONORA/scripts/stages/qc_verdict.py --print-heads"
    # eiv_score.py APPENDS and resumes by wav path, so a re-run after an interruption is
    # the intended recovery — but a clip RE-RENDERED in place keeps its path and is
    # skipped. qc_verdict counts those (wav newer than the scores) and says so; the fix is
    # to drop their rows, or the file, and re-run.
    elif ! "$SONORA/scripts/stages/eiv_score.sh" "$EIV_SCORES" "$CAMPAIGN_DIR/qc_filelist.txt" \
          -- --heads "$EIV_HEADS"; then
      echo "  !! eiv_score FAILED — NO direction verdicts for this bank. Clips are gated"
      echo "  !! and will still be queued; the round is spent for gate_calibration."
      echo "  !! Recover with: $SONORA/scripts/stages/eiv_score.sh $EIV_SCORES $CAMPAIGN_DIR/qc_filelist.txt -- --heads \"$EIV_HEADS\""
    elif ! "$PY" "$SONORA/scripts/stages/qc_verdict.py" --campaign-dir "$CAMPAIGN_DIR" \
          --eiv "$EIV_SCORES" --append-flags; then
      echo "  !! qc_verdict FAILED — no qc_verdicts.jsonl for this bank, so audit_sampler"
      echo "  !! will skip it and gate_calibration has nothing frozen to compare."
      echo "  !! Recover with: $PY $SONORA/scripts/stages/qc_verdict.py --campaign-dir $CAMPAIGN_DIR --eiv $EIV_SCORES --append-flags"
    fi
    ;;
esac

# Engine-specific defect detectors (radio timbre for moss_vg, leading gap for zonos).
# ADVISORY — they never drop a clip; they append ids to qc_flags.txt, which
# pick_audit_subset always routes to the ear.
#
# This was a MANUAL step until 2026-08-08, which made it a coverage hole rather than an
# instrument: qc_engine_defects.py exists, in its own words, as "what the tier system
# needs in order to ever hand those engines a ride-along back — coverage traded for
# instrumentation", and a detector nobody runs trades nothing. moss_vg's promotion off
# `scrutinized` (100% heard) is what makes that trade real, so the detector has to fire on
# every bank rather than when someone remembers. Measured on the moss_vg population: 3 of
# 3 ear-confirmed radio clips caught, 23 of 116 clips flagged (19.8%).
#
# Non-fatal by design. A missing detector must not fail a bank whose clips are already
# rendered and gated — but it is announced, because silently skipping the flag pass is
# exactly how the coverage would go missing again.
echo "== engine defect detectors =="
"$PY" "$SONORA/scripts/stages/qc_engine_defects.py" \
    --campaign "$(basename "$CAMPAIGN_DIR")" --append-flags \
  || echo "  !! qc_engine_defects failed — advisory flags NOT appended for this bank."

# Register the rendered clips into the audition queue (ratings.csv SSOT) so they
# reach the review surface. Idempotent, host-side (uv, not the GPU container); only
# queues clips whose wav lands under DATA_ROOT. Non-fatal if it can't run.
# With --qc, every flagged clip is queued for the ear with the finding in its note —
# a QC failure never changes a clip's status (owner: every QC failure is auditioned,
# regardless of engine tier). Flagged ids are ALSO written to qc_flags.txt for
# pick_audit_subset --flags, because the QC-flag path never relaxes.
# (QC-L1: this said structural failures "register as `reroll` instead of spending an
# audition slot". That design was considered and rejected — the instrument tells the
# ear what to check, it does not decide alone — and `_qc_triage` has never done it.)
echo "== register audition =="
"$PY" "$SONORA/scripts/stages/register_audition.py" --audio-dir "$OUT" \
    ${QC_MEASURES:+--qc "$QC_MEASURES"} \
  || echo "  (register_audition failed — clips rendered but not queued; run it manually)"

if [ "$FAILED_ENGINES" -gt 0 ]; then
  echo "!! $FAILED_ENGINES of ${#ENGINES[@]} engine(s) failed: $FAILED_ENGINE_NAMES"
  echo "!! The bank is INCOMPLETE. Clips that did render were gated and queued."
  exit 1
fi

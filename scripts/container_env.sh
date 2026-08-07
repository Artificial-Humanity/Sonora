#!/usr/bin/env bash
# container_env.sh — the pinned container environment for every throwaway-container lane.
#
# Sourced by scripts/synthesis/synth_bank.sh, scripts/synthesis/librivox_align.sh and
# scripts/eiv_score.sh. ONE copy on purpose: the review's recurring finding is two
# implementations of one rule drifting apart (B-L5's MAX_REF_EXCURSION written out three
# times, D-L2's two disagreeing z-guards), and "which environment produced this clip" is
# exactly the rule that must not fork across three scripts.
#
# ---------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS (D-M2, todo § 3)
#
# All three lanes ran `IMG=rocm/pytorch:latest` and `pip install -q <names>` with no
# versions. Raw EIV scores are treated as immutable and the render campaigns as
# reproducible, but neither carried any record of what actually executed — so "re-run the
# campaign" and "re-run the scorer" were both untestable claims.
#
# It is not hypothetical. Measured 2026-08-07 by dry-run resolution inside the pinned
# image: an unpinned `pip install transformers` today yields **transformers==5.14.1**. The
# campaigns that produced the audited corpus ran the 4.x line — qwen is the only engine
# that pinned it (4.57.3), and it pinned it because it mattered. Every other engine would
# silently cross a major version boundary on the next run, and a render that comes out
# *different* rather than *broken* is the failure this repo keeps finding.
#
# Ruled out at the same time, because it is the catastrophic case: none of the seven
# dependency sets pulls a torch or nvidia wheel, so the image's ROCm build
# (2.10.0+rocm7.2.4) survives every install. That was worth measuring rather than assuming
# — a CUDA torch silently replacing the ROCm one renders nothing and explains nothing.
# ---------------------------------------------------------------------------------------

# The image every existing campaign and every existing EIV score ran under. Pinned by
# digest because `:latest` is a moving target that will one day move under a re-run and
# there would be no way to tell that it had.
#
# `rocm/pytorch@sha256:…`, NOT the local `sonora@sha256:…` re-tag that
# `docker inspect` also reports — the point of a digest is that someone else can pull it.
# Verified present on ai-lab-0 2026-08-07; image created 2026-05-28.
: "${SONORA_ROCM_IMAGE:=rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e}"

# Pinned wheels, per lane. Every set below was verified to RESOLVE inside the pinned image
# on 2026-08-07 (`uv pip install --dry-run`, 7/7 clean) — these are checked pins, not
# hopeful ones. Refresh them the same way, never by editing a version in place and hoping.
#
# `transformers` is held at the 4.x the corpus was built under. Holding it is strictly
# safer than the status quo, which is not "some 4.x" but "whatever is newest at run time".
SONORA_PIN_TRANSFORMERS="transformers==4.57.3"

# `pip install uv` is the documented bootstrap for these images and matches the training
# container's own prep chain (see environments/training-container.txt). Everything after
# it goes through uv, per AGENTS.md § 3 — `--python /opt/venv/bin/python`, never
# `--system`, which bypasses the image's venv into Debian's externally-managed Python and
# gets refused under PEP 668.
SONORA_UV_BOOTSTRAP='pip install -q uv >/dev/null 2>&1'
SONORA_UVPIP='uv pip install -q --python /opt/venv/bin/python'

# Per-campaign environment capture. The freeze lands in the campaign's own `_env/`
# directory, beside the artifacts it produced — the only record that survives a throwaway
# container, and what makes the NEXT set of pins evidence rather than a guess.
#
# `capture_env_cmd <tag> <env_dir>` emits the command; the work is in
# scripts/capture_container_env.sh, because inlining a freeze through two `bash -c` layers
# is how you get a capture that silently writes nothing while the render succeeds. Run it
# AS ai-mgr, after container_as_ai_mgr.sh — a root-owned file in a group-writable campaign
# dir is the same papercut the umask 002 convention exists to avoid.
capture_env_cmd() {
  printf '%s' "runuser -u ai-mgr -- bash /sonora/scripts/capture_container_env.sh $1 $2;"
}

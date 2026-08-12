"""Stage-1 renderer: Chatterbox (Resemble, classic English 0.5B, MIT).

Reads the script bank, renders every engine=="chatterbox" line, writes wavs +
chatterbox_manifest.jsonl to --out.

CLASSIC, NOT TURBO — `ChatterboxTTS`, loaded from the local English weights
(t3_cfg/ve/s3gen/tokenizer/conds). Turbo's `generate()` accepts `exaggeration`
and discards it with a log warning, so switching this import silently removes
the only control channel the engine has. Do not.

Two controls, and they are one control with two knobs: `exaggeration` is a rate
profile (low = slower/deadpan, high = faster/dramatic), and raising it without
dropping `cfg_weight` reads rushed. build_direction() always emits both.

Casting: `audio_prompt_path`, a reference clip chosen by ref_select from the
certified pool. The 2026-07-17 audition passed NONE, so all 12 clips used the
built-in conds.pt fallback voice and casting was never exercised — that is the
single biggest difference between this renderer and that void verdict. Only the
first 6 s of a reference reach the speaker prompt (10 s the vocoder), and the
pool window is 4-10 s, so references are used whole or truncated, never averaged.

Publication: every output is Perth-watermarked unconditionally and the native
component is unavailable on this box, so the watermarker is patched to a no-op
below. Owner decision 2026-07-28: strip it, TRAIN on these clips, never publish
them. publish_tier.py sets publish=false for engine=="chatterbox" and fails if
one is ever marked publishable.

Container recipe (smoke-proven 2026-07-28, the first ever recorded for this
engine): pip --no-deps chatterbox-tts; then librosa s3tokenizer diffusers
resemble-perth conformer transformers safetensors soundfile **omegaconf**.
omegaconf is the one --no-deps drops that actually breaks the import chain
(chatterbox/models/s3gen/flow.py), and it fails at import, not at render.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch

import perth


class _NoopWatermarker:
    """perth's native component is unavailable here; see the publication note above."""

    def apply_watermark(self, wav, sample_rate=None, **kw):
        return wav


perth.PerthImplicitWatermarker = _NoopWatermarker

from chatterbox.tts import ChatterboxTTS  # noqa: E402  (must follow the perth patch)

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from synth_common import rebuild_used_set, write_wav_atomic  # noqa: E402
from ref_select import (  # noqa: E402
    MAX_REF_EXCURSION as _REF_EXCURSION,
    BRIGHT_REF_POLICY,
    bright_ref_exaggeration,
    bright_ref_selection_ceiling,
    select_reference, pinned_reference, design_age_band,  # noqa: E402
                        REF_BLACKLIST)

MODEL_DIR = "/data/models/ResembleAI/chatterbox"

# Below ~20 characters Chatterbox hallucinates, and no combination of
# cfg/exaggeration/temperature/seed fixes it (skill file, "Do not"). The 4 s
# owner floor already gates most of these; this catches the rest rather than
# rendering a clip that will be dropped in audit.
MIN_CHARS = 20
MAX_CHARS = 300     # beyond this, chunk — there is a hard 1000-token / ~40 s ceiling

# Reference-pitch ceiling — INTERIM, and the history matters because it is a lesson
# about instruments, not about pitch.
#
# revisit-v1 (observational) said references >=220 Hz split and <=211 Hz did not.
# artifact-probe-chatterbox (2026-07-28) appeared to REFUTE that: the 266 Hz reference
# scored env_ac 0.019, the cleanest in the whole probe, and the ceiling was removed.
# The owner then LISTENED (2026-07-29) and heard the split in all three 266 Hz seeds.
# env_ac had scored those three 0.022 / 0.020 / 0.012 — its three cleanest numbers.
#
# env_ac is a WHOLE-CLIP statistic and this defect is a LOCAL event: the voice splits
# on individual emphasised syllables ("how long", "make", "oh, the third door!") and a
# one-second break inside a seven-second clip averages away to nothing. The refutation
# was an artefact of the measuring instrument. Ear-verified state now:
#
#   266 Hz  tr_embodiment_02_narratorF     SPLITS, 3 of 3 seeds   (owner, 2026-07-29)
#   258 Hz  protective_urgency_00_urgentF  SPLITS                 (owner, 2026-07-29)
#   109 Hz  neutral_narration_00 (control) CLEAN, 2 of 2          (owner, 2026-07-29)
#
# So the ceiling stands, but on the owner's ear, NOT on any measurement we can run.
# Three detectors have failed on this defect (whole-clip env_ac, local CPP dip,
# emphasis-subharmonic delta — the last reached 20% precision on 40 labelled clips).
# qc_artifacts.py DOES NOT CATCH THIS. Do not treat a clean qc_artifacts run as
# evidence that a Chatterbox bank is free of it.
#
# Why 215: the ear-verified split cases are 258 and 266 Hz and the verified-clean
# control is 109 Hz, so the true boundary is somewhere in a very wide gap. 215 is
# inherited from the revisit-v1 population split (offenders from 219.96 Hz, clean to
# 211.4 Hz) and is a guess within that gap, not a measured edge.
#
# INTERIM because the mechanism the owner described — "at certain emphasis points
# (higher pitch, loudness or the combination), the vocal splits" — implies the real
# variable is peak F0 at emphasis, not the reference's median. If lowering
# `exaggeration` keeps those peaks under the break point, bright casting can be kept
# and this ceiling comes out. That test is 4 clips already rendered in
# artifact-probe-chatterbox (cbx_B_f266_*) and needs an ear, not a metric.
MAX_REF_F0 = 215.0          # SUPERSEDED by MAX_REF_EXCURSION — see below

# Pitch-EXCURSION ceiling, and this is the live guard (blind test, 2026-07-29).
# 16 clips auditioned blind at exaggeration 0.3 — 8 bright references (227-335 Hz) and
# 8 normal (170-211 Hz), shuffled, neutral filenames, text fully crossed with band:
#
#   4 of 8 bright flagged        0 of 8 normal flagged      (p ~ 0.04)
#
# Zero false alarms, so the defect is real rather than expectation. But median F0 did
# NOT order the bright band — 227 Hz split and 335 Hz did not, fully interleaved — while
# excursion (p90-p10 F0) did:
#
#   flagged   240  247  287  291      three of four were lilty / singsong / whimsical
#   missed    154  176  250  264
#   normal    104 .. 189
#
# That matches the owner's account of the mechanism exactly: the voice splits at emphasis
# peaks, so what matters is how far the pitch TRAVELS, not where it sits. A steady-but-high
# voice is comparatively safe; a swooping one is not, at any median.
#
# 240 excludes all four audible failures and keeps 47 of 77 female references (61%),
# against 29 (38%) under the median rule — 19 bright-median-but-steady clips come back,
# which is most of what "teen / young female" casting actually needs. The two extra clips
# it excludes (250, 264) are protective_urgency — already blacklisted, confirmed to split
# at another seed — and melodic_keening, so neither is really a false positive.
#
# The defect is STOCHASTIC, so this lowers the rate rather than eliminating it: the same
# reference at a different seed can render clean (cbx_A_f258_s8899) or split
# (cbx_A_f258_s1234). Audition remains the gate — see the qc_artifacts.py blind spot —
# but the blind test showed the owner's ear catches these reliably and without false
# alarms, so detect-and-reroll is a sound workflow rather than a hope.
# B-L5: one definition, in ref_select (beside REF_BLACKLIST).
MAX_REF_EXCURSION = _REF_EXCURSION

# --- the bright-reference trade, made explicit (owner sweep, 2026-07-29) ------------
# An exaggeration sweep on the 266 Hz reference (artifact-probe-exag, 0.1 -> 0.75, all
# else fixed) established that exaggeration MODULATES the split but never gates it:
#
#   0.10  present, and MOVED to a different phrase ("Sorry, may I ask?")
#   0.20  very subtle
#   0.30  minimal — "requires nitpicking to flag"        <- floor
#   0.40  similar to 0.30
#   0.50  slightly more noticeable
#   0.75  spreads across the whole passage after the first break
#
# Severity is U-shaped with a floor near 0.3, not monotonic, and damping below that just
# relocates the break while costing ~30% pacing (0.1 renders 8.0 s against 5.6 s at 0.75).
# Crucially the harm is an INTERACTION, not an exaggeration effect: seven revisit-v1 clips
# ran at 0.75 on low-F0 moss85 references and were all clean. High emphasis is only
# dangerous on a high-pitched reference — exactly what the owner's peak-F0 mechanism
# predicts.
#
# So there are two defensible policies and the choice is the owner's, because it trades
# casting range against a known artifact in TRAINING data:
#
#   "exclude" (default) — high-F0 references are not cast at all. Guaranteed clean, and
#       these clips train Sonora, so a nitpick-level split is still a defect the student
#       can learn. Costs ~60% of the female pool; no bright or teen female casting here.
#   "damp"             — high-F0 references are allowed but their exaggeration is clamped
#       to BRIGHT_REF_EXAG. Keeps the whole pool at the price of a residual artifact that
#       the owner rates as needing nitpicking to find. Choose this only deliberately.
# B-M9. The bright-reference policy — the constant, its three values, and the two
# functions that read it — moved to `ref_select` beside MAX_REF_EXCURSION, which B-L5
# hoisted there for exactly this reason: it is reference-selection policy, it encodes a
# measured finding, and a copy per renderer is how the duplicated constant happened the
# first time. Imported at the top of this file.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    jobs = [l for l in bank["lines"] if l["engine"] == "chatterbox"]
    if not jobs:
        print("no chatterbox jobs in bank")
        return
    model = ChatterboxTTS.from_local(MODEL_DIR, device="cuda")
    print(f"loaded ({len(jobs)} jobs)", flush=True)

    manifest_path = os.path.join(args.out, "chatterbox_manifest.jsonl")
    # B-M3: the variety-bias `used` set started EMPTY on every resume, because a job whose
    # wav already exists skips before select_reference() runs. A rerolled clip therefore
    # cast as if the reference pool were untouched — so a resumed campaign quietly loses the
    # diversity guarantee, and the casting is not reproducible from the bank either.
    # Rebuilt from the manifest rows that already exist (synth_common, shared with the
    # renderers that already did this).
    used = rebuild_used_set(manifest_path)
    if used:
        print(f"resume: {len(used)} reference(s) already spent", flush=True)
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for job in jobs:
            if os.path.exists(os.path.join(args.out, f"{job['id']}.wav")):
                print(job["id"], "exists, skip", flush=True)
                continue
            text = job["text"]
            if len(text) < MIN_CHARS:
                print(job["id"], f"SKIP ({len(text)} chars < {MIN_CHARS}: hallucinates)", flush=True)
                continue
            if len(text) > MAX_CHARS:
                print(job["id"], f"WARN {len(text)} chars > {MAX_CHARS} (40 s ceiling)", flush=True)

            d = job["direction"]
            # A bank line may PIN its reference by pool id. Casting is then not
            # consulted at all, and MAX_REF_F0 does not apply — the probes that
            # measure the ceiling have to be able to render above it, and an
            # explicit choice by the bank author outranks a guard aimed at casting.
            # B-L2. Casting can raise: an exhausted or over-filtered pool, a `ref_id` that
            # is not in it. Uncaught, that killed the ENGINE'S WHOLE REMAINING BANK from
            # whichever clip hit it — the same shape as the moss_vg split_with_sizes crash
            # that orphaned 7 clips on 2026-07-30, and the per-clip try/except there is the
            # pattern. A re-run under skip-if-exists retries only what failed.
            try:
                if job.get("ref_id"):
                    ref_wav, ref_text, ref_meta = pinned_reference(job["ref_id"])
                else:
                    ref_wav, ref_text, ref_meta = select_reference(
                        d.get("design", ""), job.get("intended", {}), used,
                        max_excursion=bright_ref_selection_ceiling(),
                        exclude=REF_BLACKLIST)
            except (LookupError, ValueError) as exc:
                print(job["id"], f"CASTING FAILED ({type(exc).__name__}: {exc}) "
                                 "— continuing", flush=True)
                continue

            # Damping applies to CAST references only, never to a pinned one. Pinning
            # exists so a probe can render deliberately outside the guards, and a probe
            # whose parameters are silently rewritten measures nothing — re-running
            # artifact-probe-chatterbox would otherwise return exaggeration 0.3 for a
            # bank line that plainly reads 0.5, and skip-if-exists would hide the change
            # until someone cleared the directory.
            ref_exc = None if job.get("ref_id") else ref_meta.get("ref_excursion_hz")
            exag, damped = bright_ref_exaggeration(d["exaggeration"], ref_exc)
            if damped:
                print(job["id"], f"DAMP exaggeration {d['exaggeration']} -> {exag} "
                                 f"(ref excursion {ref_exc} Hz)", flush=True)
            torch.manual_seed(job["seed"])
            wav = model.generate(text, audio_prompt_path=ref_wav,
                                 exaggeration=exag,
                                 cfg_weight=d["cfg_weight"])
            wav = wav.squeeze(0).cpu().numpy()
            write_wav_atomic(os.path.join(args.out, f"{job['id']}.wav"), wav, model.sr)

            row = dict(job)
            row.update({
                "wav": f"{job['id']}.wav", "sr": model.sr,
                "engine_license": "MIT (ResembleAI/chatterbox, classic English)",
                "weights_source": "ResembleAI/chatterbox (t3_cfg + ve + s3gen)",
                "watermark": "perth stripped (no-op patch) — TRAIN ONLY, never publish",
                "campaign": bank["campaign"], "bank_version": bank["version"],
                "intended_age": design_age_band(d.get("design", "")),
                "intended_gender": ref_meta["gender"],
                "ref": ref_meta | {"ref_text": ref_text},
                # `direction` is what the director asked for; this is what ran.
                "applied_exaggeration": exag,
                "bright_ref_policy": BRIGHT_REF_POLICY,
                # Recorded because the policy alone does not say whether it BIT on this
                # clip, and "was this render damped" is the question an audit of the
                # split-artifact finding actually asks.
                "bright_ref_damped": damped,
            })
            mf.write(json.dumps(row) + "\n")
            mf.flush()   # a mid-bank abort must not orphan wavs (2026-07-25)
            print(job["id"], f"{len(wav)/model.sr:.1f}s exag={d['exaggeration']} "
                             f"cfg={d['cfg_weight']} exag={exag} ref={ref_meta['id']} "
                             f"ref_f0={ref_meta.get('ref_f0_hz', '?')}Hz "
                             f"exc={ref_meta.get('ref_excursion_hz', '?')}Hz", flush=True)
    print("SYNTH-CHATTERBOX-DONE")


if __name__ == "__main__":
    main()

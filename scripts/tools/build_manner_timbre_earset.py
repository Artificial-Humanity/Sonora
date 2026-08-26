#!/usr/bin/env python3
"""build_manner_timbre_earset — prepare the blind delivery ear test from a rendered probe.

WHY THIS EXISTS (2026-08-25, PR-M2)
-----------------------------------
quality-gap-plan.md names the manner-vs-timbre test as the instrument that should GATE
rung 2 — *direct one real speaker through each of the five lanes and have the ear say
whether the manner changed while the voice did not* — and then says "wiring it in is open
work". This is the wiring. It renders nothing: `probe_delivery_intercept.py` already
produced the clips, and re-rendering them would have changed the checkpoint, the speaker
and the sampler settings all at once.

⚠ WHY THE PROBE'S WAVS CANNOT BE HANDED OVER AS THEY ARE. That probe is a LOUDNESS
measurement and is deliberately not normalised — its own header says so. Across the A = 0
cells of `intercept_ep008` the integrated loudness spans 7.1 dB. Play those to an ear and
the loudest lane is the one that sounds most "delivered", so the test would measure the
thing it is supposed to hold constant. Everything here exists to remove that confound
without introducing a second one.

⚠ LINEAR GAIN ONLY, NEVER `loudnorm`. ffmpeg's single-pass `loudnorm` applies dynamic range
compression, and compression IS a manner cue — it would manufacture the very difference the
ear is being asked to judge. This measures integrated LUFS with pyloudnorm and applies one
scalar per file. Nothing is limited and nothing is compressed; if a target would clip, the
target moves rather than the peaks.

THE DESIGN, and why each part is there
--------------------------------------
* **One repeat per lane** is the judgement set. The lane is the variable; the repeat index
  is sampler noise.
* **A DUPLICATE LANE is included in every text group, and it is the control.** Two clips of
  the SAME lane, different repeats, sit in the group under different blind letters. If the
  ear separates them as confidently as it separates two different lanes, the lane effect is
  not above sampler noise and a "the lanes differ" verdict means nothing. Without this a
  blind grouping task cannot fail, which is the defect that makes most listening tests
  unfalsifiable.
* **Blind letters and a shuffled order**, seeded so the assignment is reproducible and
  recorded in a key file the listener is asked not to open first.
* **Forced grouping, not scoring** — the 5-point scale is saturated on this project and
  cannot separate what it is asked to separate.

⚠ NOTHING IN THE GENERATED README MAY NAME A PREVIOUS ROUND'S ANSWER (#305). A commit
adding the manner question also pasted the recorded v6 verdict — "all sound neutral to me" —
into the listener's own instruction sheet, in bold, directly above the question they were
about to answer, and shipped it live on the still-unheard A=+1 set. The README is read by
someone who has been asked not to open `KEY.json`; telling them the expected answer defeats
the same blinding the key protects. **Rationale for the protocol goes here, in the source,
where the maintainer reads it. The README carries the questions and nothing else.**

Usage:
    .venv/bin/python scripts/tools/build_manner_timbre_earset.py \
        --probe /data/model-training/sonora/probes/intercept_ep008 \
        --out   /data/model-training/sonora/probes/ear_delivery_v6_ep008
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "pyloudnorm"]
# ///
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import shutil
import string
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# The probe holds three A levels; the ear test wants delivery to be the ONLY thing moving,
# so it reads ONE plane and nothing else. A = 0 is the default because it is the cleanest
# statement of "delivery alone".
#
# ⚠ BUT A = 0 CANNOT ANSWER EVERY QUESTION, AND THE FIRST RUN FOUND THE LIMIT. The owner
# heard no manner difference at A = 0 and named what was missing: "there's an energy level to
# Speech that is not present". `Speech` is defined as public address — PROJECTED rather than
# conversational — so projection is its defining quality. If projection is carried by the A
# channel rather than by the delivery one-hot, then at A = 0 the lane has no way to express
# it, and this test returns a null whether or not the lane works. That is a confound in the
# test, not a finding about the model. Hence the flag: render the same comparison at A = +1
# and the two readings separate the explanations.
ENERGY_PLANE = 0.0
# ⚠ #302. This said "mid-range of what the probe actually produced", and warned against a
# target "near the quietest clip" because that "would ask for large positive gain on the rest
# and risk clipping". Both halves were wrong, and the second was wrong about what this very
# constant does. MEASURED 2026-08-26 over probes/intercept_ep008/measures.csv:
#
#            n    min      max      mean     median
#   A=0     30  -35.79   -28.66   -31.98   -31.97
#   all     90  -38.52   -26.93   -32.21   -31.91
#   clips louder than -26.0 LUFS: 0 of 30 in-plane, 0 of 90 overall
#
# So -26.0 is 2.66 dB ABOVE the loudest clip in the plane, not mid-range (which is ~-32), and
# it asks +2.66 to +9.79 dB of every file — the exact thing the old comment warned against.
#
# It is kept anyway, deliberately: all-positive gain is fine as long as nothing clips, the
# peak ceiling below is measured never to fire on any probe on disk, and the A=+1 set already
# built for the owner was rendered at this target. Moving it now would make the two ear sets
# non-comparable to settle a wording defect. The number is sound; only its stated reason was
# not, and a wrong reason is what gets a correct constant "fixed" later.
DEFAULT_TARGET_LUFS = -26.0
PEAK_CEILING = 0.97  # leave a little room; we move the target rather than limit

# ⚠ #308. THE PROTOCOL, ONCE. The `ANSWERS.md` writer used to carry a comment claiming its
# blanks were "in the protocol's order, so the questions cannot drift from the README beside
# it" — while README.md and ANSWERS.md were two independent f-string literals sharing nothing
# but `groups` and `texts`. Nothing derived one from the other and no test compared them, so
# editing the README's wording did not touch the answer sheet. They had ALREADY drifted: the
# README asked group / manner / voice and ANSWERS asked grouping / pair+confidence / manner /
# voice, and the README's separate step 5 ("was any grouping a guess?") duplicated the
# ANSWERS confidence line.
#
# Both artifacts are now rendered from this list, so the claim is true by construction rather
# than by assertion. A step with no `fields` is an instruction, not a question, and appears
# only in the README.
PROTOCOL = (
    {"readme": "Listen to all of them once before writing anything.",
     "heading": None, "fields": ()},
    {"readme": "**Group them.** Which clips share a delivery? Say which letters go together, "
               "then name the pair that shares one and how sure you are. "
               "\"I was guessing\" is a real result here.",
     "heading": "Grouping",
     "fields": ("sets of letters, one set per delivery (exactly one set should hold two):",
                "the pair that shares a delivery:",
                "confidence:      certain / fairly sure / guessing")},
    {"readme": "**Manner.** Do these differ in *how* the line is delivered at all — obvious, "
               "subtle, or none? If audible, what moves: pace, weight, warmth, push, distance?",
     "heading": "Manner",
     "fields": ("audible difference?   obvious / subtle / none",
                "what moves (pace, weight, warmth, push, distance):")},
    {"readme": "**Voice check.** Is it the same person throughout, or does the voice itself "
               "change?",
     "heading": "Voice",
     "fields": ("same voice?      yes / no",
                "if no, which clips and how:")},
)


def read_measures(probe: pathlib.Path) -> list[dict]:
    path = probe / "measures.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, type=pathlib.Path,
                    help="a probe_delivery_intercept.py output directory")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--target-lufs", type=float, default=DEFAULT_TARGET_LUFS)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--force", action="store_true",
                    help="regenerate over an --out that already holds a KEY/ANSWERS pair")
    ap.add_argument("--energy", type=float, default=ENERGY_PLANE,
                    help="which A plane to read (default %(default)s). Use +1 to ask whether "
                         "a lane's manner appears once the energy channel is allowed to move.")
    args = ap.parse_args()

    # ⚠ #297. The README this writes states the speaker, the checkpoint and V/T=0 as FACT.
    # Those live in design.json, so it is read rather than assumed — and a probe without one
    # is REFUSED. That is not defensive padding: `probes/delivery_ep010/` has a
    # schema-compatible measures.csv and no design.json, and it is the exact probe
    # probe_delivery_intercept.py was written about ("the renders survived; the design did
    # not"). Consuming it would produce an ear test asserting four things nobody can check.
    design_path = args.probe / "design.json"
    if not design_path.exists():
        raise SystemExit(
            f"{design_path} is missing. This tool states the speaker, checkpoint and "
            f"V/T settings as fact in the README it writes, so it will not run against a "
            f"probe whose design was not recorded. Re-render with "
            f"probe_delivery_intercept.py, which writes one.")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    for field in ("checkpoint", "spk", "valence", "tension", "sample_rate"):
        if field not in design:
            raise SystemExit(
                f"{design_path} has no {field!r} — this tool either states it as fact in the "
                f"README it writes or measures with it, and will not guess either one.")

    rows = [r for r in read_measures(args.probe) if float(r["energy"]) == args.energy]
    if not rows:
        have = sorted({r["energy"] for r in read_measures(args.probe)})
        raise SystemExit(f"no A={args.energy} rows in {args.probe}/measures.csv (have {have})")

    texts = sorted({r["text"] for r in rows})
    lanes = sorted({r["lane"] for r in rows})
    rng = random.Random(args.seed)

    # ⚠ #310. The merge script got this guard as #295 and this one did not, which is the
    # whole defect: a populated --out here holds a FILLED-IN `ANSWERS.md` and the `KEY.json`
    # that decodes it, and overwriting them destroys a listening session that cannot be
    # repeated — the ear time is spent. Janis proved it by destroying a filled-in sheet with
    # the committed tool, rc=0 and no warning. `RESULT-*.md` is not written by this tool, so
    # it survives; the answers and the key do not.
    collide = [f.name for f in (args.out / "ANSWERS.md", args.out / "KEY.json")
               if f.exists()]
    if collide and not args.force:
        raise SystemExit(
            f"{args.out} already holds {', '.join(collide)}. Refusing to overwrite: if the "
            f"answers are filled in, this destroys a listening session that cannot be "
            f"re-run. Move them aside, or pass --force if regenerating is genuinely what "
            f"you want.")
    args.out.mkdir(parents=True, exist_ok=True)
    # ⚠ #303. This used to read `rows[0].get("sr", 24000)` from measures.csv — a column
    # `probe_delivery_intercept.py` has never written. Its columns are exactly
    # text, lane, energy, rep, lufs, rms_db, dur_s, file, and no probe on disk carries `sr`.
    # So the expression read as a per-probe adaptation and was dead code that always returned
    # its own default. Harmless so far only because every probe happens to be 24 kHz.
    # The rate is recorded authoritatively in design.json, which is already open here.
    # An ITU-R BS.1770 meter at the wrong rate returns a wrong integrated loudness SILENTLY,
    # which would put every per-file gain wrong — defeating the one confound this tool exists
    # to remove, with nothing raised. So it is read, and then checked against each file.
    meter_sr = int(design["sample_rate"])
    meter = pyln.Meter(meter_sr)

    key: list[dict] = []
    gains: list[float] = []
    for text in texts:
        # One repeat per lane, plus a second repeat of ONE lane as the control pair.
        picked: list[tuple[str, dict]] = []
        for lane in lanes:
            cells = [r for r in rows if r["text"] == text and r["lane"] == lane]
            if not cells:
                raise SystemExit(f"lane {lane!r} missing from text {text!r}")
            picked.append((lane, rng.choice(cells)))

        control_lane = rng.choice(lanes)
        pool = [r for r in rows
                if r["text"] == text and r["lane"] == control_lane
                and r["file"] != dict(picked)[control_lane]["file"]]
        if not pool:
            raise SystemExit(
                f"no second repeat available for the control lane {control_lane!r} in "
                f"{text!r} — the probe must carry >1 repeat per cell for this test")
        picked.append((control_lane, rng.choice(pool)))

        rng.shuffle(picked)
        letters = string.ascii_uppercase[:len(picked)]
        for letter, (lane, cell) in zip(letters, picked):
            src = args.probe / cell["file"]
            y, sr = sf.read(str(src))
            # ⚠ #303. The true per-file rate was read here and used to WRITE the output, and
            # never compared against the rate the meter was built at. Silent disagreement is
            # the whole hazard, so it is now loud.
            if sr != meter_sr:
                raise SystemExit(
                    f"{src} is {sr} Hz but design.json says the probe is {meter_sr} Hz. The "
                    f"loudness meter is built at the design rate, and measuring at the wrong "
                    f"rate returns a wrong integrated loudness with no error — which would "
                    f"put every gain in this ear set wrong. Refusing rather than guessing.")
            measured = meter.integrated_loudness(y)
            gain = 10.0 ** ((args.target_lufs - measured) / 20.0)
            peak = float(np.max(np.abs(y))) * gain
            # ⚠ #302. When this fires the clip DOES NOT REACH the target, so loudness is no
            # longer equalised — and it used to say nothing at all, while the README below
            # asserted unconditionally that "louder" is not available as a cue. That would be
            # a false statement to the blind listener about the one confound this tool exists
            # to remove. Recorded per clip and reported, so the README can tell the truth.
            # `bool(...)` is load-bearing: `peak` is a numpy float, so the comparison yields
            # `np.bool_`, which `json.dump` refuses — and KEY.json is written 60 lines later,
            # long after the loop that would have to be re-run.
            capped = bool(peak > PEAK_CEILING)
            if capped:  # move the target, never limit the peaks
                gain *= PEAK_CEILING / peak
            out_name = f"{text}_{letter}.wav"
            sf.write(str(args.out / out_name), y * gain, sr)
            gains.append(20.0 * np.log10(gain))
            key.append({"file": out_name, "text": text, "letter": letter,
                        "lane": lane, "source": cell["file"],
                        "measured_lufs": round(measured, 3),
                        "applied_gain_db": round(20.0 * float(np.log10(gain)), 3),
                        "peak_ceiling_capped": capped,
                        "reached_target_lufs": not capped,
                        "is_control_pair_member": lane == control_lane})

    # The key is what makes the test scoreable, and it is the one file the listener must
    # not read first. It is written beside the clips on purpose — a key kept somewhere
    # else is a key that is lost by the time the answers come back.
    (args.out / "KEY.json").write_text(
        json.dumps({"probe": str(args.probe), "seed": args.seed, "energy": args.energy,
                    "target_lufs": args.target_lufs,
                    # The artifact must carry WHAT IT TESTED. A key naming only the probe
                    # directory is unresolvable once that directory moves or is re-rendered.
                    "checkpoint": design["checkpoint"], "spk": design["spk"],
                    "valence": design["valence"], "tension": design["tension"],
                    "clips": key}, indent=2),
        encoding="utf-8")

    # ⚠ #302. The README must not claim equalisation it did not achieve. Every clip that hit
    # the peak ceiling is BELOW target, so "louder is not available as a cue" would be false
    # for it — in the one document the blind listener reads, about the one confound this tool
    # exists to remove. Measured never to fire on any probe on disk; stated conditionally
    # anyway, because the day it fires is the day nobody is watching for it.
    capped = [k for k in key if k["peak_ceiling_capped"]]
    # ⚠ #324. The #302 fix made README.md conditional and left ANSWERS.md asserting "loudness
    # equalised" flatly — so the two documents the blind listener holds SIDE BY SIDE said
    # opposite things in the capped case, and the answer sheet is the one they read while
    # deciding. Same defect as #308 one layer out: two artifacts, one fact, fixed in one place.
    answers_loudness = ("loudness equalised" if not capped else
                        f"⚠ loudness NOT fully equalised — {len(capped)} of {len(key)} clips "
                        f"are below target (see README.md), so some of these differ in level")
    if capped:
        loudness_note = (
            f"Loudness was targeted at {args.target_lufs:.1f} LUFS by a single gain per file "
            f"(no compression, no limiting). ⚠ **{len(capped)} of {len(key)} clips did NOT "
            f"reach that target** — they hit the peak ceiling first, so they are quieter than "
            f"the rest and **\"louder\" IS partly available as a cue between them**: "
            f"{', '.join(sorted(k['file'] for k in capped))}. Treat any grouping that follows "
            f"loudness with suspicion, and say so in your answers.")
        print(f"  ⚠ {len(capped)} of {len(key)} clips hit the peak ceiling and are BELOW "
              f"target — loudness is not fully equalised; README.md and ANSWERS.md say so",
              file=sys.stderr)
    else:
        loudness_note = (
            f"Loudness has been equalised to {args.target_lufs:.1f} LUFS by a single gain per "
            f"file, so \"louder\" is not available as a cue. No compression or limiting was "
            f"applied, and every clip reached the target.")

    # Both renderings of PROTOCOL. Same list, same order, one source — see #308.
    protocol_readme = chr(10).join(
        f"{i}. {step['readme']}" for i, step in enumerate(PROTOCOL, 1))
    questions = [s for s in PROTOCOL if s["fields"]]

    groups = {t: sorted(k["letter"] for k in key if k["text"] == t) for t in texts}
    controls = {t: sorted({k["lane"] for k in key
                           if k["text"] == t and k["is_control_pair_member"]})
                for t in texts}
    (args.out / "README.md").write_text(f"""# Delivery ear test — manner vs timbre

Blind. Do not open `KEY.json` until the answers are written down.

Every clip is speaker id `{design["spk"]}` saying the SAME sentence, from
`{pathlib.Path(design["checkpoint"]).name}`, with valence {design["valence"]:+g}, tension
{design["tension"]:+g} and **energy (A) {args.energy:+g}** — all read from the probe's own
`design.json`, not assumed here. **The only thing that differs between clips in a group is
the delivery lane** — except that one lane appears TWICE in each group, which is the control.

{loudness_note}

## Groups

{chr(10).join(f"- **{t}** — clips {', '.join(groups[t])} ({len(groups[t])} clips)" for t in texts)}

## For each group, in this order

{protocol_readme}

## What the answers mean

Each group contains **one duplicate lane** — two clips that differ only by sampler noise.
If those two do not land in the same group, the delivery channel is not separable above
noise at this checkpoint, and a confident "the lanes differ" answer elsewhere in the same
group cannot be trusted. That is the point of the control, and it is why a clean-looking
result here is worth something.

If manner changes while the voice stays put, the delivery channel is doing its job. If the
voice moves with it, delivery is entangled with timbre and that is a finding.
""", encoding="utf-8")

    # ⚠ #298. The answer sheet SHIPS. The v6 verdicts were collected against a template
    # hand-written on /data that the repo never had, so the committed tool could not
    # reproduce the artifact its own recorded result rests on.
    # ⚠ #308. Its questions are RENDERED FROM `PROTOCOL`, the same list the README's numbered
    # steps come from, so "the questions cannot drift from the README" is now a property of
    # the code rather than a claim in a comment beside two independent literals.
    def _blank_for(t):
        parts = [f"## Group {t} — clips {', '.join(groups[t])}", ""]
        for i, step in enumerate(questions, 1):
            parts += [f"**{i}. {step['heading']}.**", "", "```"]
            parts += list(step["fields"])
            parts += ["```", ""]
        return chr(10).join(parts)

    blank = chr(10).join(_blank_for(t) for t in texts)
    (args.out / "ANSWERS.md").write_text(
        f"""# Answers — delivery ear test

Fill this in BEFORE opening `KEY.json`.

Each group is {len(groups[texts[0]])} clips containing {len(lanes)} different deliveries — so exactly one pair
shares one. Same speaker, same sentence, {answers_loudness}.

---

{blank}
---

## Overall

```
Did this feel like a real distinction or like listening for noise?

Anything you noticed that these questions did not ask about:
```
""", encoding="utf-8")

    print(f"wrote {len(key)} clips to {args.out}")
    for t in texts:
        print(f"  {t}: {len(groups[t])} clips, control lane {controls[t][0]!r}")
    print(f"  gain applied: {min(gains):+.2f} to {max(gains):+.2f} dB")
    print(f"  key: {args.out / 'KEY.json'} (blind — do not read first)")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the two controlled probes that follow up the 2026-07-28 artifact diagnosis.

The revisit-v1 finding was OBSERVATIONAL: nobody varied one thing at a time, so two
questions it raised cannot be answered from it.

  orpheus — `tara` reverberated in 16 of 16 clips and `dan` in 0 of 4, but the other
      four female voices (`leah` `jess` `mia` `zoe`) have never been rendered here at
      all. Banning tara and silently substituting one of them would be a guess. This
      probe renders all six over the same five lines: tara as the positive control,
      dan as the negative, four unknowns between them.

  chatterbox — references at >=220 Hz doubled 4/4 and those at <=211 Hz 0/16, which is
      why MAX_REF_F0 exists. But that ceiling costs 62% of the female pool, and the
      owner would rather keep bright/teen casting if it can be bought another way.
      `exaggeration` and `cfg_weight` were never varied against a high-F0 reference,
      so they are CONFOUNDED, not exonerated. Two arms:
        A. seed sweep at each reference's ORIGINAL parameters — is the defect even
           deterministic? If it moves with the seed, the F0 story is much weaker and
           the ceiling is the wrong instrument.
        B. a parameter grid at a fixed seed — is there a (exaggeration, cfg_weight)
           setting that renders a 266 Hz reference cleanly? If yes, MAX_REF_F0 comes
           back out and the bright voices are saved.
      Text is held constant per reference (each one's own failing line), so within a
      reference the only variables are the ones being swept.

Loudness was considered as arm C and dropped: the pool is already normalised to
-23.0 LUFS with sd 0.06 dB across all 162 clips, so there is no level variation left
for a loudness guard to act on. Peak, RMS, crest and short-term loudness all overlap
between the doubled and clean groups; F0 was the only clean separator measured.

Usage:
    uv run make_artifact_probe.py --out-dir /data/model-training/datasets
"""
import argparse
import json
import os

# --- orpheus arm -----------------------------------------------------------------
# Five revisit-v1 lines, reused verbatim (with their emotive tags) so results are
# directly comparable to the campaign that raised the question. Chosen to span the
# register range rather than to flatter any voice.
ORPHEUS_LINES = [
    ("victory", "victory_peak", {"V": 0.9, "A": 0.8, "T": -0.5},
     "We did it! After all of it, <laugh> every single one of them said we couldn't, "
     "and we did it anyway!"),
    ("grief", "grief_herenow", {"V": -0.8, "A": -0.3, "T": 0.4},
     "I keep setting a place for her at the table. I know. <sigh> I know she isn't "
     "coming. I do know that."),
    ("narration", "neutral_narration", {"V": 0.0, "A": -0.2, "T": 0.0},
     "The road out of the valley climbed for three miles before it turned, and from "
     "the turn you could see the whole of the town below."),
    ("gossip", "reminiscent_gossip", {"V": 0.3, "A": 0.2, "T": -0.2},
     "Well now, I'm not one to gossip, honey, <chuckle> but you did not hear a single "
     "word of this from me."),
    ("factual", "matter_of_fact", {"V": 0.0, "A": -0.1, "T": 0.0},
     "The ferry goes at six, and if it's blowing hard from the west it doesn't go at "
     "all that day."),
]
# tara first as the positive control, dan as the negative; the rest are the unknowns.
ORPHEUS_VOICES = ["tara", "dan", "leah", "jess", "mia", "zoe"]

# --- chatterbox arm --------------------------------------------------------------
# (pool ref id, measured F0, that reference's original params, its original text)
CBX_REFS = [
    ("tr_embodiment_02_narratorF_s5150", 266, (0.5, 0.4),
     "Sorry, may I ask — how long have you been doing this? It's just, you make it "
     "look very easy."),
    ("protective_urgency_00_urgentF_s1234", 258, (0.75, 0.3),
     "And the third door — oh, the third door! — was the one nobody sensible ever, "
     "ever opened."),
    ("tr_victory_whimsical_01_brightF_s1234", 242, (0.75, 0.3),
     "Pick up, pick up — they said yes! They said yes to all of it, the whole thing, "
     "starting Monday!"),
    ("tr_excited_good_news_01_everydayMF_s5150", 220, (0.5, 0.4),
     "Well now, I'm not one to gossip, honey, but you did not hear a single word of "
     "this from me."),
]
# The negative control: a reference far below the ceiling, which must stay clean or
# the probe itself is not measuring what it claims to.
CBX_CONTROL = ("neutral_narration_00_narratorM_s1234", 109, (0.5, 0.4),
               "The road out of the valley climbed for three miles before it turned, "
               "and from the turn you could see the whole of the town below.")
SEEDS = [1234, 5150, 8899]
# Toward stability: lower exaggeration, higher cfg_weight. The doubled clips all sat at
# exaggeration >= 0.5 with cfg <= 0.4, which is the least-damped corner of this space.
CBX_GRID = [(0.3, 0.5), (0.3, 0.7), (0.5, 0.5), (0.5, 0.7)]


def orpheus_bank():
    lines = []
    for voice in ORPHEUS_VOICES:
        for key, register, vat, text in ORPHEUS_LINES:
            lines.append({
                "id": f"orp_{key}_{voice}",
                "engine": "orpheus", "register": register,
                "expected_register": register, "intended": vat,
                "seed": 1234, "text": text, "probe": "voice",
                "direction": {"voice": voice, "render_text": text},
            })
    return {"version": "1.0", "campaign": "artifact-probe-orpheus",
            "source_campaign": "revisit-v1",
            "note": "Four unproven female voices against tara (positive control, 16/16 "
                    "reverberant) and dan (negative control, 0/4). Same five lines for all.",
            "lines": lines}


def chatterbox_bank():
    lines = []
    # Arm A — is the defect deterministic, or does it move with the seed?
    for ref, f0, (exag, cfg), text in CBX_REFS:
        for seed in SEEDS:
            lines.append({
                "id": f"cbx_A_f{f0}_s{seed}",
                "engine": "chatterbox", "register": "probe",
                "expected_register": "probe", "intended": {"V": 0.3, "A": 0.5, "T": 0.0},
                "seed": seed, "text": text, "probe": "seed",
                "ref_id": ref, "ref_f0_measured": f0,
                "direction": {"design": "probe — reference pinned, casting bypassed",
                              "exaggeration": exag, "cfg_weight": cfg},
            })
    # Arm B — can a parameter setting rescue a high-F0 reference?
    for ref, f0, _orig, text in CBX_REFS:
        for exag, cfg in CBX_GRID:
            lines.append({
                "id": f"cbx_B_f{f0}_e{exag}_c{cfg}",
                "engine": "chatterbox", "register": "probe",
                "expected_register": "probe", "intended": {"V": 0.3, "A": 0.5, "T": 0.0},
                "seed": 1234, "text": text, "probe": "params",
                "ref_id": ref, "ref_f0_measured": f0,
                "direction": {"design": "probe — reference pinned, casting bypassed",
                              "exaggeration": exag, "cfg_weight": cfg},
            })
    # Arm C — negative control at both the original and the most-damped setting.
    ref, f0, (exag, cfg), text = CBX_CONTROL
    for e, c in [(exag, cfg), (0.3, 0.7)]:
        lines.append({
            "id": f"cbx_C_ctl_f{f0}_e{e}_c{c}",
            "engine": "chatterbox", "register": "probe",
            "expected_register": "probe", "intended": {"V": 0.0, "A": 0.0, "T": 0.0},
            "seed": 1234, "text": text, "probe": "control",
            "ref_id": ref, "ref_f0_measured": f0,
            "direction": {"design": "probe — reference pinned, casting bypassed",
                          "exaggeration": e, "cfg_weight": c},
        })
    return {"version": "1.0", "campaign": "artifact-probe-chatterbox",
            "source_campaign": "revisit-v1",
            "note": "A: seed sweep at original params (is it deterministic?). "
                    "B: (exaggeration, cfg_weight) grid on four >=220 Hz references "
                    "(can params rescue bright casting?). C: low-F0 negative control. "
                    "References are PINNED, so MAX_REF_F0 does not apply.",
            "lines": lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/model-training/datasets")
    args = ap.parse_args()
    for bank in (orpheus_bank(), chatterbox_bank()):
        d = os.path.join(args.out_dir, bank["campaign"])
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "bank.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(bank, f, indent=1, ensure_ascii=False)
        print(f"{p}  ({len(bank['lines'])} lines)")


if __name__ == "__main__":
    main()

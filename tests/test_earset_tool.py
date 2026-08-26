"""⚠ #312, SECOND HALF. The first test that RUNS `scripts/tools/build_manner_timbre_earset.py`.

The first commit against #312 covered `merge_libritts_full_corpus.py` and stopped there, while
the issue's title says **either** new tool. That was an under-delivery on my part, not a scope
call — recorded here rather than quietly closed.

The probe is built from scratch in `tmp_path` rather than read from `/data`, so this is
hermetic: no data gate, no skip, and it runs on a fresh clone. Real wavs, because the tool
measures them with a BS.1770 meter and a synthetic path that skipped the audio would prove
nothing about the one thing this tool exists to do.

⚠ THE ASSERTION SHAPE HERE IS DELIBERATE, and it is not "README matches ANSWERS". Both are
checked **against `PROTOCOL`**, the single list they are rendered from. Comparing the two
artifacts to each other would pass if someone re-inlined the same literal into both, which is
precisely the state #308 found them in. Checking each against the derivation is what makes
"the questions cannot drift" a property instead of a claim.
"""

import csv
import json
import pathlib
import subprocess

import numpy as np
import pytest
import soundfile as sf

from scripts_layout import SCRIPTS  # noqa: E402

SCRIPTS.on_path()

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "tools" / "build_manner_timbre_earset.py"
PY = REPO / ".venv" / "bin" / "python"

SR = 24000
LANES = ["Dialogue", "Neutral", "Newscaster", "Speech", "unknown"]
TEXTS = ["t1", "t2"]
REPS = 3


def _protocol():
    """Import PROTOCOL from the tool itself. A second copy here would be the very defect."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_earset", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROTOCOL


@pytest.fixture
def probe(tmp_path):
    """A real probe: design.json, measures.csv, and wavs long enough for a BS.1770 meter.

    ⚠ 2.0 s is not arbitrary. ITU-R BS.1770 integrates over 400 ms blocks and returns -inf for
    anything shorter than one block; a 0.1 s fixture would make every gain infinite and the
    failure would look like a tool bug.
    """
    d = tmp_path / "probe"
    d.mkdir()
    rng = np.random.default_rng(7)
    rows = []
    for text in TEXTS:
        for i, lane in enumerate(LANES):
            for rep in range(REPS):
                # Vary level per lane so the loudness pass has something to equalise.
                amp = 0.02 * (1.0 + 0.35 * i)
                y = (rng.standard_normal(int(SR * 2.0)) * amp).astype(np.float32)
                name = f"{text}_{lane}_E+0_r{rep}.wav"
                sf.write(str(d / name), y, SR)
                rows.append({"text": text, "lane": lane, "energy": "0.0", "rep": str(rep),
                             "lufs": "-30.0", "rms_db": "-30.0", "dur_s": "2.0", "file": name})
    with open(d / "measures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # ⚠ NOTE what is NOT here: an `sr` column. probe_delivery_intercept.py has never written
    # one, which is exactly why reading the meter rate from here was dead code (#303).
    (d / "design.json").write_text(json.dumps({
        "checkpoint": "/fake/checkpoint_epoch=008.ckpt", "spk": 0,
        "valence": 0.0, "tension": 0.0, "sample_rate": SR,
        "texts": {t: f"Sentence {t}." for t in TEXTS},
    }), encoding="utf-8")
    return d


def run(*args):
    return subprocess.run([str(PY), str(TOOL), *map(str, args)],
                          capture_output=True, text=True, cwd=REPO)


def build(probe, out, *extra):
    return run("--probe", probe, "--out", out, *extra)


# ------------------------------------------------------------------ happy path + #308

def test_builds_a_complete_earset(probe, tmp_path):
    out = tmp_path / "set"
    r = build(probe, out)
    assert r.returncode == 0, r.stdout + r.stderr
    for n in ("KEY.json", "README.md", "ANSWERS.md"):
        assert (out / n).exists(), f"{n} missing"
    key = json.loads((out / "KEY.json").read_text())
    assert len(key["clips"]) == len(TEXTS) * (len(LANES) + 1)
    for t in TEXTS:
        clips = [c for c in key["clips"] if c["text"] == t]
        # Exactly one lane appears twice — the control pair the whole test rests on.
        lanes = [c["lane"] for c in clips]
        dupes = {l for l in lanes if lanes.count(l) == 2}
        assert len(dupes) == 1, f"group {t}: expected exactly one duplicated lane, got {dupes}"
        assert sum(c["is_control_pair_member"] for c in clips) == 2
    for c in key["clips"]:
        assert (out / c["file"]).exists()


def test_both_artifacts_are_rendered_from_PROTOCOL(probe, tmp_path):
    """⚠ #308. Each artifact against the DERIVATION, never against the other one."""
    out = tmp_path / "set"
    assert build(probe, out).returncode == 0
    readme = (out / "README.md").read_text()
    answers = (out / "ANSWERS.md").read_text()
    protocol = _protocol()

    # README: every step, numbered, in order, instructions included.
    for i, step in enumerate(protocol, 1):
        first = step["readme"].split(".")[0][:40]
        assert f"{i}. " in readme and first.strip("* ") in readme, \
            f"README lost step {i}: {step['readme'][:60]}"

    # ANSWERS: only the steps with fields, renumbered among themselves, every field present.
    questions = [s for s in protocol if s["fields"]]
    for i, step in enumerate(questions, 1):
        assert f"**{i}. {step['heading']}.**" in answers, \
            f"ANSWERS lost question {i} ({step['heading']})"
        for field in step["fields"]:
            assert field in answers, f"ANSWERS lost field {field!r}"
    # ...and the instruction-only steps must NOT appear as questions.
    for step in protocol:
        if not step["fields"]:
            assert step["heading"] is None
    # One blank per group.
    for t in TEXTS:
        assert f"## Group {t} —" in answers


# ------------------------------------------------------------------ #310

def test_populated_out_is_refused_without_force(probe, tmp_path):
    """⚠ #310. The guard exists because a filled-in ANSWERS.md is a listening session that
    CANNOT be re-run — Janis destroyed one to prove the point."""
    out = tmp_path / "set"
    assert build(probe, out).returncode == 0
    (out / "ANSWERS.md").write_text("MY PRECIOUS FILLED-IN ANSWERS", encoding="utf-8")
    r = build(probe, out)
    assert r.returncode != 0
    assert "Refusing to overwrite" in r.stdout + r.stderr
    assert (out / "ANSWERS.md").read_text() == "MY PRECIOUS FILLED-IN ANSWERS"


def test_force_does_overwrite(probe, tmp_path):
    """Positive control: the guard is a guard, not a wall."""
    out = tmp_path / "set"
    assert build(probe, out).returncode == 0
    (out / "ANSWERS.md").write_text("stale", encoding="utf-8")
    assert build(probe, out, "--force").returncode == 0
    assert "stale" not in (out / "ANSWERS.md").read_text()


# ------------------------------------------------------------------ #297 / #303

def test_probe_without_design_is_refused(probe, tmp_path):
    """⚠ #297. The README states speaker, checkpoint and V/T as FACT."""
    (probe / "design.json").unlink()
    r = build(probe, tmp_path / "set")
    assert r.returncode != 0
    assert "design.json is missing" in (r.stdout + r.stderr).replace(str(probe) + "/", "")


@pytest.mark.parametrize("field", ["sample_rate", "checkpoint", "spk", "valence", "tension"])
def test_design_missing_a_required_field_is_refused(probe, tmp_path, field):
    d = json.loads((probe / "design.json").read_text())
    d.pop(field)
    (probe / "design.json").write_text(json.dumps(d), encoding="utf-8")
    r = build(probe, tmp_path / "set")
    assert r.returncode != 0
    assert f"has no {field!r}" in r.stdout + r.stderr


def test_sample_rate_disagreement_is_refused(probe, tmp_path):
    """⚠ #303. A BS.1770 meter at the wrong rate is wrong SILENTLY, so this must be loud."""
    d = json.loads((probe / "design.json").read_text())
    d["sample_rate"] = 48000
    (probe / "design.json").write_text(json.dumps(d), encoding="utf-8")
    r = build(probe, tmp_path / "set")
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "48000 Hz" in out and f"{SR} Hz" in out


# ------------------------------------------------------------------ #302

def test_readme_claims_equalisation_only_when_it_achieved_it(probe, tmp_path):
    """⚠ #302. Both directions, because the honest branch is the one nobody exercises."""
    ok = tmp_path / "ok"
    assert build(probe, ok).returncode == 0
    text = (ok / "README.md").read_text()
    assert "every clip reached the target" in text
    assert "did NOT reach that target" not in text
    assert all(not c["peak_ceiling_capped"] for c in json.loads((ok / "KEY.json").read_text())["clips"])

    # Force the ceiling with an impossible target.
    capped = tmp_path / "capped"
    r = build(probe, capped, "--target-lufs", "-3.0")
    assert r.returncode == 0
    text = (capped / "README.md").read_text()
    assert "did NOT reach that target" in text
    assert "IS partly available as a cue" in text
    assert "every clip reached the target" not in text
    assert "hit the peak ceiling" in r.stderr, "the operator was not told on stderr"
    clips = json.loads((capped / "KEY.json").read_text())["clips"]
    assert all(c["peak_ceiling_capped"] for c in clips)
    # The README must NAME the affected files, not just count them.
    for c in clips:
        assert c["file"] in text


def test_key_records_what_it_tested(probe, tmp_path):
    """A key naming only the probe directory is unresolvable once that directory moves."""
    out = tmp_path / "set"
    assert build(probe, out).returncode == 0
    key = json.loads((out / "KEY.json").read_text())
    for f in ("checkpoint", "spk", "valence", "tension", "energy", "target_lufs", "seed"):
        assert f in key, f"KEY.json does not record {f}"

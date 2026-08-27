"""`derive_vat_corpus.py --exclude` — the ear's only way to remove a clip from a built corpus.

It exists because the owner rejected a clip on 2026-08-27 that no gate here could see: the
audio of `6003_58761_000005_000009.wav` stops mid-sentence while the transcript runs to the
end. `speech_ok` is a 4 s floor and the clip is 12.28 s; `head_ok` is measured and ungated;
nothing compares transcript length to duration. That class of defect reaches the corpus and
leaves it only by ear, so the removal path is load-bearing and had no test.

⚠ **THE REFUSALS ARE THE POINT, NOT THE REMOVAL.** An exclusion list is an enumeration, and
an enumeration that matches nothing is indistinguishable from one that worked — the run
prints a drop, writes a corpus that still contains the clip, and every log line agrees with
the intent rather than the result. So a stale path must STOP the build rather than be
skipped. Same for the speaker-emptying case: dropping a speaker's last clip renumbers every
speaker after it, which silently invalidates a warm start from a checkpoint trained on the
old map, and the symptom appears far from the cause.

Both directions, per this suite's standing rule: the good list removes exactly what it names
AND each bad list is refused with nothing written. A refusal that fires everywhere and one
that fires nowhere are the same green.

⚠ These fixtures carry **no audio**. `--reuse-from` reads measures, phonemes and filelists
and re-labels, so the whole path is text — which is why the tool can be run for real here
rather than having its guard re-implemented in the test.
"""

import json
import os
import random
import subprocess

import pytest

from scripts_layout import SCRIPTS  # noqa: E402

REPO = SCRIPTS.dirs[-1].parent
TOOL = SCRIPTS / "derive_vat_corpus.py"
PY = REPO / ".venv" / "bin" / "python"

HEADS = ("alpha_db", "cpp", "h1h2")


def _wav(spk, i):
    return f"/corpus/train-other-500/{spk}/ch/{spk}_ch_{i:06d}.wav"


@pytest.fixture()
def donor(tmp_path):
    """A derive_vat_corpus output directory, complete enough for --reuse-from."""
    d = tmp_path / "donor"
    d.mkdir()
    # 100 is the bulk (any single drop leaves it healthy); 200 has two (a drop thins it to
    # one, which RE-LABELS the survivor); 300 has one (a drop would empty it entirely).
    #
    # ⚠ 100 is large on purpose. The 3% hash split is a real guard — it ABORTS when it puts
    # 0 rows in val — so a handful of clips fails there before ever reaching what this file
    # tests. A fixture sized below another guard's floor tests that guard, not yours.
    pop = {"100": 200, "200": 2, "300": 1}
    wavs = [(s, _wav(s, i)) for s, n in pop.items() for i in range(n)]
    idx = {s: i for i, s in enumerate(sorted(pop, key=int))}
    (d / "speakers.json").write_text(
        json.dumps({"libritts_id_to_index": idx, "n_spks": len(idx)}), encoding="utf-8")
    # ⚠ INDEPENDENTLY RANDOM, seeded. The first version of this fixture made every measure a
    # linear function of the row index, which made V, A and T perfectly correlated and
    # tripped the independence gate — a real guard, failing honestly on synthetic data that
    # no corpus would ever look like. A fixture has to clear the guards it is not testing.
    rng = random.Random(20260827)
    with open(d / "measures.jsonl", "w", encoding="utf-8") as fh:
        for s_, w in wavs:
            fh.write(json.dumps({"wav": w, "seconds": rng.uniform(2.0, 15.0),
                                 "lufs": rng.gauss(-18.0, 2.0),
                                 **{h: rng.gauss(0.0, 1.0) for h in HEADS}}) + "\n")
    # every clip in train; val may be empty — the reuse path reads both if present
    with open(d / "train_op.txt", "w", encoding="utf-8") as fh:
        for s, w in wavs:
            fh.write(f"{w}|{idx[s]}|hɛlˈoʊ|0,0,0,0,0,0,0,0\n")
    (d / "val_op.txt").write_text("", encoding="utf-8")

    vj = tmp_path / "valence.json"
    sj = tmp_path / "soft.json"
    vj.write_text(json.dumps({w: rng.gauss(0.0, 1.0) for _, w in wavs}), encoding="utf-8")
    sj.write_text(json.dumps({w: rng.gauss(0.0, 1.0) for _, w in wavs}), encoding="utf-8")
    return d, vj, sj, wavs


def _run(donor, tmp_path, lines, out="out"):
    d, vj, sj, _ = donor
    ex = tmp_path / "exclude.txt"
    ex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dest = tmp_path / out
    p = subprocess.run(
        [str(PY), str(TOOL), "--reuse-from", str(d), "--root", "/corpus/train-other-500",
         "--out", str(dest), "--exclude", str(ex),
         "--valence-json", str(vj), "--soft-json", str(sj)],
        capture_output=True, text=True, cwd=str(REPO))
    return p, dest


def _rows(dest):
    out = []
    for name in ("train_op.txt", "val_op.txt"):
        f = dest / name
        if f.exists():
            out += [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


# --------------------------------------------------------------------------------------
# It removes what it names.
# --------------------------------------------------------------------------------------

def test_excluding_a_real_clip_removes_exactly_that_clip(donor, tmp_path):
    _, _, _, wavs = donor
    victim = _wav("100", 1)
    p, dest = _run(donor, tmp_path, [victim])

    assert p.returncode == 0, p.stdout + p.stderr
    assert "excluded 1 clip(s)" in p.stdout, p.stdout
    rows = _rows(dest)
    assert len(rows) == len(wavs) - 1
    assert not any(r.startswith(victim + "|") for r in rows), "the excluded clip survived"


def test_comments_and_blank_lines_are_ignored(donor, tmp_path):
    victim = _wav("100", 1)
    p, dest = _run(donor, tmp_path, ["# a reason, written down", "", f"{victim}  # why"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert not any(r.startswith(victim + "|") for r in _rows(dest))


# --------------------------------------------------------------------------------------
# The refusals. Each one writes NOTHING.
# --------------------------------------------------------------------------------------

def test_a_path_that_does_not_exist_on_disk_refuses_and_writes_nothing(donor, tmp_path):
    """The defect this guard exists for: a list that quietly matches nothing.

    ⚠ The refusal is keyed on the path not EXISTING, not on it being absent from `kept` —
    those are two different situations and refusing both made the exclusion unrepeatable
    (#369, reopened). This is the one that is a real fault: the clip the author meant to
    remove is still in the output and nothing said so.
    """
    p, dest = _run(donor, tmp_path, ["/corpus/train-other-500/100/ch/NOT_A_REAL_CLIP.wav"])

    assert p.returncode != 0
    assert "do not exist on disk" in (p.stdout + p.stderr)
    assert not dest.exists(), "a refused run must not write a corpus"


def test_a_clip_already_removed_is_a_no_op_not_a_refusal(donor, tmp_path):
    """Re-running the exclusion against an already-clean input must SUCCEED.

    This is the path the build recipe instructs, and the reviewer found it unfollowable by
    running the command the record named: the donor it pointed at had already had the clip
    removed, so passing the flag refused because the clip was absent while omitting it
    refused because the clip was declared. Excluding a clip that is already gone is a no-op,
    and a no-op is the correct answer to "make sure this clip is not in the corpus".
    """
    gone = tmp_path / "already_removed.wav"      # exists on disk, absent from `kept`
    gone.write_bytes(b"")
    p, dest = _run(donor, tmp_path, [str(gone)])

    assert p.returncode == 0, p.stdout + p.stderr
    assert "already absent" in p.stdout, p.stdout
    assert dest.exists(), "an idempotent exclusion must still write the corpus"


def test_emptying_a_speaker_refuses_because_it_renumbers(donor, tmp_path):
    """Speaker 300 has one clip; removing it would renumber every speaker after it."""
    p, dest = _run(donor, tmp_path, [_wav("300", 0)])

    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "no clips" in out and "300" in out, out
    assert not dest.exists()


def test_an_empty_list_refuses_rather_than_running_a_no_op(donor, tmp_path):
    p, dest = _run(donor, tmp_path, ["# every line a comment", ""])
    assert p.returncode != 0
    assert "no-op" in (p.stdout + p.stderr)
    assert not dest.exists()


# --------------------------------------------------------------------------------------
# The consequence the owner has to be told about.
# --------------------------------------------------------------------------------------

def test_dropping_one_of_a_pair_relabels_the_survivor_to_all_zero(donor, tmp_path):
    """Speaker 200 has two clips. Removing one leaves n=1, whose per-speaker z is 0.0.

    This is arithmetic, not a bug, and it is why the exclusion file records it: a reader
    finding a clip that was KEPT with a changed label would otherwise read it as corruption.
    It is also why emptying a speaker is refused while thinning one to a single clip is not
    — the second changes a label, the first changes every index after it.
    """
    p, dest = _run(donor, tmp_path, [_wav("200", 0)])
    assert p.returncode == 0, p.stdout + p.stderr

    survivor = [r for r in _rows(dest) if r.startswith(_wav("200", 1) + "|")]
    assert len(survivor) == 1, "the survivor should still be in the corpus"
    v, a, t = survivor[0].split("|")[3].split(",")[:3]
    assert (float(v), float(a), float(t)) == (0.0, 0.0, 0.0), survivor[0]

    # control: a speaker that kept a real population is NOT all-zero, so the assertion
    # above is measuring the drop rather than a corpus where everything is 0.0.
    others = [r for r in _rows(dest) if r.startswith("/corpus/train-other-500/100/")]
    assert any(set(r.split("|")[3].split(",")[:3]) != {"0.0000"} for r in others), \
        "every label is zero, so the survivor assertion proves nothing"


# --------------------------------------------------------------------------------------
# #369 — an OMITTED --exclude must be loud, not a silent no-op.
# --------------------------------------------------------------------------------------

def _run_declared(donor, tmp_path, declared, pass_flag, out="out"):
    """Run with a configs/data-style exclusion file present, with or without the flag."""
    d, vj, sj, _ = donor
    cfg = tmp_path / "cfgdata"
    cfg.mkdir(exist_ok=True)
    (cfg / "some_corpus.exclude.txt").write_text(
        "# declared by ear\n" + "\n".join(declared) + "\n", encoding="utf-8")
    argv = [str(PY), str(TOOL), "--reuse-from", str(d), "--root", "/corpus/train-other-500",
            "--out", str(tmp_path / out),
            "--valence-json", str(vj), "--soft-json", str(sj)]
    if pass_flag:
        ex = tmp_path / "exclude.txt"
        ex.write_text("\n".join(declared) + "\n", encoding="utf-8")
        argv += ["--exclude", str(ex)]
    env = dict(os.environ, SONORA_EXCLUDE_DIR=str(cfg))
    p = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO), env=env)
    return p, tmp_path / out


def test_omitting_exclude_when_a_clip_is_declared_refuses(donor, tmp_path):
    """The defect: every other refusal fires only when the flag is PASSED.

    Omitting it was a silent no-op by construction, so the build recipe — which this repo
    calls the template later rungs are written from — could reinstate an ear-dropped clip
    with nothing disagreeing. The exclusion FILES are the record; the flag is only how they
    reach a run.
    """
    p, dest = _run_declared(donor, tmp_path, [_wav("100", 5)], pass_flag=False)
    assert p.returncode != 0, p.stdout
    out = p.stdout + p.stderr
    assert "declared excluded by ear" in out, out
    assert not dest.exists(), "a refused run must write nothing"


def test_passing_the_flag_satisfies_the_declaration(donor, tmp_path):
    """The other direction — the refusal must be clearable by doing the right thing."""
    p, dest = _run_declared(donor, tmp_path, [_wav("100", 5)], pass_flag=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not any(r.startswith(_wav("100", 5) + "|") for r in _rows(dest))


def test_a_declared_clip_already_absent_does_not_deadlock(donor, tmp_path):
    """⚠ Found by RUNNING it, not reading it.

    Re-deriving an already-excluded donor has no such clip in `kept`. If the guard demanded
    the flag anyway, passing it would hit the "not in the kept set" refusal while omitting it
    hit this one — no way through. So the guard considers only clips actually PRESENT.
    """
    p, dest = _run_declared(donor, tmp_path, ["/corpus/train-other-500/999/ch/gone.wav"],
                            pass_flag=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert dest.exists()


def test_the_guard_holds_when_root_is_omitted_entirely(donor, tmp_path):
    """#375 — the omit-the-flag guard must not depend on `--root`.

    `--root` is INERT under `--reuse-from` (its only other reader is `find_clips`, in the
    other branch) and it carries a DEFAULT, so a guard anchored on it silently found nothing
    declared and wrote the ear-dropped clip back at exit 0. The population that matters is
    what is about to be WRITTEN, which is `kept` on both code paths.

    This test omits `--root` altogether — the shape the original guard could not see.
    """
    d, vj, sj, _ = donor
    victim = _wav("100", 7)
    cfg = tmp_path / "cfgdata"
    cfg.mkdir(exist_ok=True)
    (cfg / "some_corpus.exclude.txt").write_text(victim + "\n", encoding="utf-8")
    dest = tmp_path / "out"
    p = subprocess.run(
        [str(PY), str(TOOL), "--reuse-from", str(d), "--out", str(dest),
         "--valence-json", str(vj), "--soft-json", str(sj)],   # <- no --root at all
        capture_output=True, text=True, cwd=str(REPO),
        env=dict(os.environ, SONORA_EXCLUDE_DIR=str(cfg)))
    assert p.returncode != 0, p.stdout + p.stderr
    assert "declared excluded by ear" in (p.stdout + p.stderr)
    assert not dest.exists(), "a refused run must write nothing"

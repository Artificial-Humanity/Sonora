"""synth_bank.sh's probe mode (`SYNTH_BANK_PROBE=1`, Phase 1S Step 1c-i), run with stubs.

The other synth_bank.sh tests read its source (`test_audit_sampling._invocations`). These RUN it:
a copy of the script in a fake repo, whose `.venv/bin/python`, `eiv_score.sh` and `docker` are stubs
that log their arguments and exit 0 (the QC gate stub also writes a measures file). The copy differs
from the real script in one thing: its `/data` guards and datasets root point at the test's tmp dir,
because /data is not writable here. Each replacement is asserted to hit exactly once, so a reshaped
guard fails this test rather than running unguarded.

What is checked: probe mode skips qc_verdict/EIV, the defect detectors and register_audition and
says so; loudness normalisation and the QC gate still run and their failures stay fatal; without
the variable the pipeline runs every step as before; any other value is refused.
"""
import os
import pathlib
import shutil
import subprocess

import pytest

from scripts_layout import REPO

STUB_PY = r"""#!/usr/bin/env bash
echo "PY $*" >> "$STUB_LOG"
script="$(basename "$1")"
case "$script" in
  qc_gate.py)
    [ "${STUB_GATE_FAIL:-0}" = 1 ] && exit 1
    shift
    while [ $# -gt 0 ]; do [ "$1" = --campaign-dir ] && dir="$2"; shift; done
    echo '{"id": "c1", "hard_pass": true}' > "$dir/qc_measures.jsonl" ;;
  normalize_loudness.py) [ "${STUB_LOUDNORM_FAIL:-0}" = 1 ] && exit 1 ;;
  qc_verdict.py)
    case " $* " in *" --count-directed "*) echo 3 ;; *" --print-heads "*) echo h1,h2 ;; esac ;;
esac
exit 0
"""
STUB_LOGGER = '#!/usr/bin/env bash\necho "%s $*" >> "$STUB_LOG"\nexit 0\n'


@pytest.fixture
def bank_env(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    (repo / "scripts" / "stages").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    src = (REPO / "scripts" / "stages" / "synth_bank.sh").read_text(encoding="utf-8")
    for old in ('case "$BANK" in /data/*)', 'case "$OUT"  in /data/*)',
                'DATASETS_ROOT="/data/model-training/datasets"'):
        assert src.count(old) == 1, "synth_bank.sh no longer reads %r — update this test" % old
        src = src.replace(old, old.replace("/data", str(data)))
    script = repo / "scripts" / "stages" / "synth_bank.sh"
    script.write_text(src)
    shutil.copy(REPO / "scripts" / "container_env.sh", repo / "scripts" / "container_env.sh")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for path, body in ((repo / ".venv" / "bin" / "python", STUB_PY),
                       (repo / "scripts" / "stages" / "eiv_score.sh", STUB_LOGGER % "EIV"),
                       (stubs / "docker", STUB_LOGGER % "DOCKER")):
        path.write_text(body)
        path.chmod(0o755)
    campaign = data / "model-training" / "datasets" / "probes" / "cell"
    campaign.mkdir(parents=True)
    (campaign / "bank.json").write_text('{"version": "t", "campaign": "t", "lines": []}')
    return {"script": script, "stubs": stubs, "campaign": campaign, "log": tmp_path / "log"}


def _synth(env, probe=None, **extra_env):
    e = dict(os.environ, PATH="%s:%s" % (env["stubs"], os.environ["PATH"]),
             STUB_LOG=str(env["log"]), **extra_env)
    e.pop("SYNTH_BANK_PROBE", None)
    e.pop("HF_TOKEN", None)
    if probe is not None:
        e["SYNTH_BANK_PROBE"] = probe
    res = subprocess.run(["bash", str(env["script"]), str(env["campaign"] / "bank.json"),
                          str(env["campaign"] / "audio"), "chatterbox"],
                         capture_output=True, text=True, env=e, timeout=60)
    log = env["log"].read_text().splitlines() if env["log"].exists() else []
    return res, log


def _ran(log, name):
    return [i for i, line in enumerate(log) if name in line]


def test_probe_mode_skips_the_audition_path_and_says_so(bank_env):
    res, log = _synth(bank_env, probe="1")
    assert res.returncode == 0, res.stdout + res.stderr
    for skipped in ("qc_verdict.py", "EIV", "qc_engine_defects.py", "register_audition.py"):
        assert not _ran(log, skipped), "%s ran in probe mode:\n%s" % (skipped, "\n".join(log))
    assert "PROBE MODE" in res.stdout
    assert res.stdout.count("SKIPPED") == 3, res.stdout


def test_probe_mode_still_runs_check_bank_loudness_and_the_gate_in_order(bank_env):
    _res, log = _synth(bank_env, probe="1")
    order = [_ran(log, s) for s in ("check_bank.py", "DOCKER", "normalize_loudness.py",
                                    "qc_gate.py")]
    assert all(len(o) == 1 for o in order), log
    assert [o[0] for o in order] == sorted(o[0] for o in order), log


def test_the_chatterbox_container_installs_the_pinned_wheel(bank_env):
    _res, log = _synth(bank_env, probe="1")
    docker = "\n".join(log[_ran(log, "DOCKER")[0]:])     # its `bash -c` string spans lines
    assert "--no-deps chatterbox-tts==0.1.7 " in docker


@pytest.mark.parametrize("failing", ["STUB_GATE_FAIL", "STUB_LOUDNORM_FAIL"])
def test_a_failed_gate_or_loudness_is_still_fatal_in_probe_mode(bank_env, failing):
    res, log = _synth(bank_env, probe="1", **{failing: "1"})
    assert res.returncode == 1
    assert not _ran(log, "register_audition.py")
    assert "register_audition" not in res.stdout, "probe mode must not suggest queueing clips"


def test_without_probe_mode_every_step_runs_as_before(bank_env):
    for probe in (None, "0"):
        if bank_env["log"].exists():
            bank_env["log"].unlink()
        res, log = _synth(bank_env, probe=probe)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "PROBE MODE" not in res.stdout and "SKIPPED" not in res.stdout
        steps = ["check_bank.py", "normalize_loudness.py", "qc_gate.py",
                 "qc_verdict.py --campaign-dir", "qc_verdict.py --print-heads", "EIV",
                 "--eiv", "qc_engine_defects.py", "register_audition.py"]
        at = [_ran(log, s) for s in steps]
        assert all(at), dict(zip(steps, at))
        assert [a[0] for a in at] == sorted(a[0] for a in at), log


def test_an_unknown_probe_value_is_refused(bank_env):
    res, log = _synth(bank_env, probe="yes")
    assert res.returncode == 2 and "SYNTH_BANK_PROBE must be 0 or 1" in res.stderr
    assert log == []

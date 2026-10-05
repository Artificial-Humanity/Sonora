"""GemmaServer: the pinned llama.cpp container that serves one Gemma role for one run.

Docker and the health probe are injected, so these run without a GPU and pin exactly what
would be executed: the guards, the `docker run` line, and that the container is always removed.
"""

import subprocess

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import gemma_server as gs  # noqa: E402


class FakeDocker:
    """Records every docker invocation; `responses` maps a docker sub-command to (rc, stdout)."""

    def __init__(self, ps_names=(), running="true", run_rc=0):
        self.calls, self.ps_names, self.running, self.run_rc = [], list(ps_names), running, run_rc

    def __call__(self, *args, timeout=120):
        self.calls.append(list(args))
        out, rc = "", 0
        if args[0] == "ps":
            out = "\n".join(self.ps_names.pop(0) if self.ps_names else [])
        elif args[0] == "run":
            rc = self.run_rc
        elif args[0] == "inspect":
            out = self.running
        return subprocess.CompletedProcess(args, rc, out, "boom" if rc else "")

    def run_line(self):
        return next(c for c in self.calls if c[0] == "run")


@pytest.fixture
def cache(tmp_path):
    hub, drafters = tmp_path / "hub", tmp_path / "drafters"
    for size, file in (("31B", "gemma-4-31B_q4_0-it.gguf"), ("E4B", "gemma-4-E4B_q4_0-it.gguf")):
        snap = hub / f"models--google--gemma-4-{size}-it-qat-q4_0-gguf" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / file).write_bytes(b"GGUF")
    drafters.mkdir()
    for d in ("gemma-4-31B-it-qat-assistant-Q8_0.gguf", "gemma-4-E4B-it-qat-assistant-Q8_0.gguf"):
        (drafters / d).write_bytes(b"GGUF")
    return hub, drafters


def _server(cache, role="director", **kw):
    hub, drafters = cache
    kw.setdefault("docker", FakeDocker())
    kw.setdefault("healthy", lambda: True)
    return gs.GemmaServer(role, hub=hub, drafters=drafters, sleep=lambda s: None, **kw)


def test_the_pins():
    assert gs.IMAGE == ("ghcr.io/ggml-org/llama.cpp@sha256:"
                        "87e42fbed83a1f1b1931e5173e4367880107ab95025032326fef610a5a3f3923")
    assert (gs.PORT, gs.LABEL) == (8014, "sonora=llama")
    assert set(gs.ROLES) == {"director", "volume"}


def test_resolve_mounts_the_whole_repo_and_addresses_the_snapshot(cache):
    hub, drafters = cache
    repo, model, drafter = gs.resolve("director", hub=hub, drafters=drafters)
    assert repo == hub / "models--google--gemma-4-31B-it-qat-q4_0-gguf"
    assert model == "/m/snapshots/abc123/gemma-4-31B_q4_0-it.gguf"
    assert drafter == drafters / "gemma-4-31B-it-qat-assistant-Q8_0.gguf"


def test_resolve_refuses_two_snapshots_or_a_missing_file(cache):
    hub, drafters = cache
    (hub / "models--google--gemma-4-31B-it-qat-q4_0-gguf" / "snapshots" / "def456").mkdir()
    with pytest.raises(gs.ServerError, match="exactly one snapshot"):
        gs.resolve("director", hub=hub, drafters=drafters)
    with pytest.raises(gs.ServerError, match="no such role"):
        gs.resolve("nope", hub=hub, drafters=drafters)
    (drafters / "gemma-4-E4B-it-qat-assistant-Q8_0.gguf").unlink()
    with pytest.raises(gs.ServerError, match="drafter"):
        gs.resolve("volume", hub=hub, drafters=drafters)


def test_server_args_are_the_probed_line():
    assert gs.server_args("/m/snapshots/r/x.gguf", "d.gguf") == [
        "-m", "/m/snapshots/r/x.gguf", "-md", "/d/d.gguf", "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.75", "-c", "8192", "-np", "1",
        "-ngl", "999", "--jinja", "--host", "0.0.0.0", "--port", "8080"]


def test_start_runs_the_pinned_image_read_only_on_localhost_and_stop_removes_it(cache):
    docker = FakeDocker()
    with _server(cache, docker=docker) as srv:
        name = srv.container
        line = docker.run_line()
        assert name.startswith("sonora-llama-") and len(name) == len("sonora-llama-") + 8
        pairs = [line[j:j + 2] for j in range(len(line) - 1)]
        for pair in (["--name", name], ["--label", "sonora=llama"], ["-p", "127.0.0.1:8014:8080"],
                     ["--device", "/dev/kfd"], ["--device", "/dev/dri"]):
            assert pair in pairs
        mounts = [line[j + 1] for j in range(len(line) - 1) if line[j] == "-v"]
        assert all(m.endswith(":ro") for m in mounts) and len(mounts) == 2
        assert line[line.index(gs.IMAGE) + 1:] == gs.server_args(
            "/m/snapshots/abc123/gemma-4-31B_q4_0-it.gguf", "gemma-4-31B-it-qat-assistant-Q8_0.gguf")
        assert "--rm" in line and "--restart" not in line
    assert docker.calls[-1] == ["rm", "-f", name]
    assert srv.container == ""


def test_refuses_beside_a_training_run(cache):
    docker = FakeDocker(ps_names=[["sonora_training"]])
    with pytest.raises(gs.TrainingRunning, match="sonora_training"):
        with _server(cache, docker=docker):
            pass
    assert not any(c[0] == "run" for c in docker.calls)


def test_refuses_beside_another_sonora_server(cache):
    docker = FakeDocker(ps_names=[[], ["sonora-llama-deadbeef"]])
    with pytest.raises(gs.ServerError, match="already running"):
        with _server(cache, docker=docker):
            pass
    assert not any(c[0] == "run" for c in docker.calls)


def test_a_container_that_exits_during_startup_is_reported_and_removed(cache):
    docker = FakeDocker(running="false")
    with pytest.raises(gs.ServerError, match="exited during startup"):
        with _server(cache, docker=docker, healthy=lambda: False):
            pass
    assert docker.calls[-1][:2] == ["rm", "-f"]


def test_a_signal_during_startup_still_removes_the_container(cache):
    docker = FakeDocker()

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with _server(cache, docker=docker, healthy=interrupted):
            pass
    assert docker.calls[-1][:2] == ["rm", "-f"]


def test_a_failed_docker_run_is_a_server_error(cache):
    with pytest.raises(gs.ServerError, match="docker run failed: boom"):
        with _server(cache, docker=FakeDocker(run_rc=125)):
            pass

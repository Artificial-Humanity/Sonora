"""GemmaServer: the pinned llama.cpp container that serves one Gemma role for one run.

Docker and the health probe are injected, so these run without a GPU and pin exactly what
would be executed: the guards, the `docker run` line, and that the container is always removed.
"""

import signal
import subprocess

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import gemma_server as gs  # noqa: E402


class FakeDocker:
    """Records every docker invocation and answers like a small docker.

    `ps` maps a `--filter` value (e.g. "name=sonora_training") to the container names that
    filter matches; `docker ps` answers with the union over the filters it was given, so a
    guard that asks with the wrong filter sees nothing. `ps_rc` makes `ps` fail; `running` is
    what `inspect` prints; `run_rc`/`run_exc` and `rm_rc`/`rm_exc`/`rm_stderr` make `run` and
    `rm` fail (a non-zero status, or an exception such as TimeoutExpired). `timeouts` keeps the
    `timeout=` each sub-command was called with.
    """

    def __init__(self, ps=None, ps_rc=0, running="true", run_rc=0, run_exc=None, rm_rc=0,
                 rm_exc=None, rm_stderr="boom"):
        self.calls, self.timeouts = [], {}
        self.ps, self.ps_rc, self.running = dict(ps or {}), ps_rc, running
        self.run_rc, self.run_exc, self.rm_rc, self.rm_exc, self.rm_stderr = (
            run_rc, run_exc, rm_rc, rm_exc, rm_stderr)

    def __call__(self, *args, timeout=120):
        self.calls.append(list(args))
        self.timeouts[args[0]] = timeout
        out, rc, err = "", 0, ""
        if args[0] == "ps":
            rc, err = self.ps_rc, "ps broke" if self.ps_rc else ""
            filters = [args[j + 1] for j in range(len(args) - 1) if args[j] == "--filter"]
            out = "\n".join(n for f in filters for n in self.ps.get(f, []))
        elif args[0] == "run":
            if self.run_exc:
                raise self.run_exc
            rc, err = self.run_rc, "boom" if self.run_rc else ""
        elif args[0] == "inspect":
            out = self.running
        elif args[0] == "rm":
            if self.rm_exc:
                raise self.rm_exc
            rc, err = self.rm_rc, self.rm_stderr if self.rm_rc else ""
        return subprocess.CompletedProcess(args, rc, out, err)

    def run_line(self):
        return next(c for c in self.calls if c[0] == "run")

    def name(self):
        line = self.run_line()
        return line[line.index("--name") + 1]

    def ps_filters(self):
        return [[c[j + 1] for j in range(len(c) - 1) if c[j] == "--filter"]
                for c in self.calls if c[0] == "ps"]


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


@pytest.fixture(autouse=True)
def host_groups(monkeypatch):
    """The host's video/render groups: fixed gids, so the tests do not depend on the machine."""
    gids = {"video": 44, "render": 105}

    def getgrnam(name):
        if name not in gids:
            raise KeyError(name)
        return type("G", (), {"gr_gid": gids[name]})()

    monkeypatch.setattr(gs.grp, "getgrnam", getgrnam)
    return gids


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
    docker = FakeDocker(ps={"name=sonora_training": ["sonora_training"]})
    with pytest.raises(gs.TrainingRunning, match="sonora_training"):
        with _server(cache, docker=docker):
            pass
    assert not any(c[0] == "run" for c in docker.calls)


def test_refuses_beside_another_sonora_server(cache):
    docker = FakeDocker(ps={"label=sonora=llama": ["sonora-llama-deadbeef"]})
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


def test_a_failing_docker_ps_is_a_server_error_and_starts_nothing(cache):
    docker = FakeDocker(ps_rc=1)
    with pytest.raises(gs.ServerError, match="docker ps failed: ps broke"):
        with _server(cache, docker=docker):
            pass
    assert not any(c[0] == "run" for c in docker.calls)


def test_the_guards_ask_with_the_right_filters(cache):
    docker = FakeDocker()
    with _server(cache, docker=docker):
        pass
    assert docker.ps_filters() == [["name=sonora_training", "name=vocoder_training"],
                                   ["label=sonora=llama"]]


def test_refuses_beside_a_vocoder_training_run(cache):
    docker = FakeDocker(ps={"name=vocoder_training": ["vocoder_training"]})
    with pytest.raises(gs.TrainingRunning, match="vocoder_training"):
        with _server(cache, docker=docker):
            pass
    assert not any(c[0] == "run" for c in docker.calls)


def test_a_docker_run_that_times_out_still_removes_the_container(cache):
    docker = FakeDocker(run_exc=subprocess.TimeoutExpired(["docker", "run"], 900))
    with pytest.raises(gs.ServerError, match="sonora-llama-"):
        with _server(cache, docker=docker):
            pass
    assert docker.calls[-1] == ["rm", "-f", docker.name()]


def test_docker_run_gets_a_long_timeout(cache):
    docker = FakeDocker()
    with _server(cache, docker=docker):
        pass
    assert gs.RUN_TIMEOUT >= 600 and docker.timeouts["run"] == gs.RUN_TIMEOUT


def test_a_keyboard_interrupt_in_docker_run_removes_the_container(cache):
    docker = FakeDocker(run_exc=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        with _server(cache, docker=docker):
            pass
    assert docker.calls[-1] == ["rm", "-f", docker.name()]


def test_a_remove_that_fails_names_the_container(cache):
    docker = FakeDocker(rm_rc=1)
    with pytest.raises(gs.ServerError, match="could not remove") as err:
        with _server(cache, docker=docker):
            pass
    assert docker.name() in str(err.value) and f"docker rm -f {docker.name()}" in str(err.value)


def test_a_remove_that_times_out_names_the_container(cache):
    docker = FakeDocker(rm_exc=subprocess.TimeoutExpired(["docker", "rm"], 180))
    with pytest.raises(gs.ServerError, match="could not remove") as err:
        with _server(cache, docker=docker):
            pass
    assert f"docker rm -f {docker.name()}" in str(err.value)


def test_a_remove_of_a_container_that_is_already_gone_is_fine(cache):
    docker = FakeDocker(rm_rc=1, rm_stderr="Error: No such container: x")
    with _server(cache, docker=docker):
        pass


def test_a_server_that_never_turns_healthy_is_reported_and_removed(cache, monkeypatch):
    monkeypatch.setattr(gs, "START_TIMEOUT", -1)  # the deadline is already past
    docker = FakeDocker(running="true")
    with pytest.raises(gs.ServerError, match="not healthy"):
        with _server(cache, docker=docker, healthy=lambda: False):
            pass
    assert docker.calls[-1] == ["rm", "-f", docker.name()]


def test_the_host_gpu_groups_are_added(cache, host_groups):
    docker = FakeDocker()
    with _server(cache, docker=docker):
        line = docker.run_line()
    pairs = [line[j:j + 2] for j in range(len(line) - 1)]
    assert ["--group-add", "44"] in pairs and ["--group-add", "105"] in pairs


@pytest.mark.parametrize("missing", ["video", "render"])
def test_a_missing_gpu_group_fails_closed(cache, host_groups, missing):
    del host_groups[missing]
    docker = FakeDocker()
    with pytest.raises(gs.ServerError, match=missing):
        with _server(cache, docker=docker):
            pass
    assert not any(c[0] == "run" for c in docker.calls)


def test_sigterm_is_handled_inside_the_with_and_restored_after(cache):
    before = signal.getsignal(signal.SIGTERM)
    with _server(cache):
        inside = signal.getsignal(signal.SIGTERM)
        assert inside is not before and callable(inside)
        with pytest.raises(SystemExit) as stop:
            inside(signal.SIGTERM, None)
        assert stop.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before


def test_sigterm_is_restored_when_start_up_fails(cache):
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(gs.ServerError):
        with _server(cache, docker=FakeDocker(run_rc=125)):
            pass
    assert signal.getsignal(signal.SIGTERM) is before


def test_sigterm_off_the_main_thread_is_left_alone(cache):
    import threading
    seen = []

    def work():
        before = signal.getsignal(signal.SIGTERM)
        with _server(cache):
            seen.append(signal.getsignal(signal.SIGTERM) is before)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert seen == [True]

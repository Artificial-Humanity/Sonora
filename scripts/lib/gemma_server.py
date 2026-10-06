"""The pinned llama.cpp container that serves one Gemma role to Sonora for one run.

Lemonade is user-facing; Sonora's pipelines run their own digest-pinned llama.cpp
(Notes/Sonora/lemonade-migration-design.md § Serving, revised; convention:
Notes/AI-Lab-AMD/project-inference-runtimes.md). Modelled on Kaggle-Gemma4's `LlamaServer`.

⚠ STARTED FOR A RUN AND REMOVED ON EVERY EXIT THE PROCESS CAN SEE. `--rm`, no restart policy,
bound to 127.0.0.1. Any failure, SIGTERM or SIGHUP (on the main thread; both become SystemExit)
removes the container before it propagates: a leaked container holds the shared GPU, and the
one-at-a-time guard would then refuse every later run. SIGKILL or an OOM kill cannot be handled
in-process and leaves the container running; the label guard then refuses the next run and names
the remedy, `docker rm -f <name>`.

⚠ THE IMAGE AND THE WEIGHTS ARE BOTH PINNED: the image by digest, each role's weights by
Hugging Face snapshot revision (`Role.revision`). Other snapshots in the cache are ignored.

⚠ THE WHOLE CACHE REPO IS MOUNTED, NOT THE SNAPSHOT. Hugging Face snapshot files are symlinks
into `../../blobs`; a snapshot-only mount leaves them dangling (found in the 2026-10-05 probe).
"""

import grp
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = ("ghcr.io/ggml-org/llama.cpp@sha256:"
         "87e42fbed83a1f1b1931e5173e4367880107ab95025032326fef610a5a3f3923")  # server-rocm, build 11429
PORT = 8014
LABEL = "sonora=llama"
TRAINING_NAMES = ("sonora_training", "vocoder_training")
START_TIMEOUT = 600
RUN_TIMEOUT = 900  # `docker run` may pull the image first
HUB = Path("/data/huggingface/hub")
DRAFTERS = Path("/data/services/lemonade/recipe/drafters")


@dataclass(frozen=True)
class Role:
    repo: str
    file: str
    drafter: str
    revision: str  # the Hugging Face snapshot the bench validated; the weights are pinned like the image


ROLES = {
    "director": Role("models--google--gemma-4-31B-it-qat-q4_0-gguf", "gemma-4-31B_q4_0-it.gguf",
                     "gemma-4-31B-it-qat-assistant-Q8_0.gguf",
                     "59dde24573e7e61570dba08b18a2e1fe246955ed"),
    "volume": Role("models--google--gemma-4-E4B-it-qat-q4_0-gguf", "gemma-4-E4B_q4_0-it.gguf",
                   "gemma-4-E4B-it-qat-assistant-Q8_0.gguf",
                   "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"),
}


class ServerError(RuntimeError):
    """The Gemma server could not be started, or could not be removed."""


class TrainingRunning(ServerError):
    """A training run holds the GPU."""


def resolve(role, hub=HUB, drafters=DRAFTERS):
    """(repo dir to mount at /m, model path inside the container, drafter host path)."""
    if role not in ROLES:
        raise ServerError(f"no such role {role!r}; roles: {sorted(ROLES)}")
    r = ROLES[role]
    repo = Path(hub) / r.repo
    snap = repo / "snapshots" / r.revision  # other snapshots may coexist; only this one is used
    if not snap.is_dir():
        raise ServerError(f"{repo}: pinned snapshot revision {r.revision} is not in the cache")
    if not (snap / r.file).exists():
        raise ServerError(f"{snap / r.file} does not exist")
    drafter = Path(drafters) / r.drafter
    if not drafter.exists():
        raise ServerError(f"drafter {drafter} does not exist")
    return repo, f"/m/snapshots/{r.revision}/{r.file}", drafter


def server_args(model_in_container, drafter_name):
    return ["-m", model_in_container, "-md", f"/d/{drafter_name}", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.75", "-c", "8192", "-np", "1",
            "-ngl", "999", "--jinja", "--host", "0.0.0.0", "--port", "8080"]


def _docker(*args, timeout=120):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def _group_ids():
    """`--group-add` for the host's GPU groups. Fails CLOSED: without them the container starts
    with no access to /dev/kfd or /dev/dri and would only fail later, inside llama-server."""
    out = []
    for name in ("video", "render"):
        try:
            out += ["--group-add", str(grp.getgrnam(name).gr_gid)]
        except KeyError:
            raise ServerError(f"the host has no {name!r} group; the container would have no "
                              f"GPU access") from None
    return out


_HANDLED = (signal.SIGTERM, signal.SIGHUP)


def _on_signal(signum, frame):
    """Turn SIGTERM (systemd, `timeout`, an orchestrator) or SIGHUP (a closed terminal or ssh
    session) into SystemExit so `__exit__` runs."""
    raise SystemExit(128 + signum)


class GemmaServer:
    def __init__(self, role, *, docker=_docker, healthy=_healthy, sleep=time.sleep, hub=HUB,
                 drafters=DRAFTERS):
        self.role, self._docker, self._healthy, self._sleep = role, docker, healthy, sleep
        self._hub, self._drafters = hub, drafters
        self.container = ""
        self._prev_handlers = {}

    def _ps(self, *filters):
        args = ["ps", "--format", "{{.Names}}"]
        for f in filters:
            args += ["--filter", f]
        r = self._docker(*args)
        if r.returncode != 0:
            raise ServerError(f"docker ps failed: {r.stderr.strip()}")
        return [n for n in r.stdout.split() if n]

    def _arm_signals(self):
        if threading.current_thread() is threading.main_thread():
            for sig in _HANDLED:
                self._prev_handlers[sig] = signal.signal(sig, _on_signal)

    def _restore_signals(self):
        prev, self._prev_handlers = self._prev_handlers, {}
        for sig, handler in prev.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)

    def __enter__(self):
        repo, model, drafter = resolve(self.role, self._hub, self._drafters)
        if running := self._ps(*(f"name={n}" for n in TRAINING_NAMES)):
            raise TrainingRunning(f"a training run is up ({', '.join(running)}); the GPU is its")
        if left := self._ps(f"label={LABEL}"):
            raise ServerError(f"a Sonora Gemma server is already running ({', '.join(left)}); "
                              f"remove it with `docker rm -f {' '.join(left)}` once nothing uses it")
        groups = _group_ids()
        name = f"sonora-llama-{uuid.uuid4().hex[:8]}"
        cmd = ["run", "-d", "--rm", "--name", name, "--label", LABEL,
               "--device", "/dev/kfd", "--device", "/dev/dri", *groups,
               "-p", f"127.0.0.1:{PORT}:8080", "-v", f"{repo}:/m:ro",
               "-v", f"{drafter}:/d/{drafter.name}:ro", IMAGE, *server_args(model, drafter.name)]
        self._arm_signals()
        # The name is recorded BEFORE `docker run`: a timeout or a signal during or after it
        # must still reach the `rm -f` below.
        self.container = name
        try:
            try:
                r = self._docker(*cmd, timeout=RUN_TIMEOUT)
            except subprocess.TimeoutExpired:
                raise ServerError(f"docker run did not return in {RUN_TIMEOUT}s; removing "
                                  f"{name}") from None
            if r.returncode != 0:
                raise ServerError(f"docker run failed: {r.stderr.strip()}")
            deadline = time.monotonic() + START_TIMEOUT
            while not self._healthy():
                state = self._docker("inspect", "-f", "{{.State.Running}}", name).stdout.strip()
                if state != "true":
                    raise ServerError(f"llama-server exited during startup ({self.role})")
                if time.monotonic() > deadline:
                    raise ServerError(f"llama-server not healthy after {START_TIMEOUT}s ({self.role})")
                self._sleep(3)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        try:
            if not self.container:
                return
            name, self.container = self.container, ""
            remedy = (f"could not remove {name}; it may still hold the GPU — "
                      f"run `docker rm -f {name}`")
            try:
                r = self._docker("rm", "-f", name, timeout=180)
            except subprocess.TimeoutExpired:
                raise ServerError(f"{remedy}: docker rm timed out") from None
            if r.returncode != 0 and "No such container" not in r.stderr:
                raise ServerError(f"{remedy}: {r.stderr.strip()}")
        finally:
            self._restore_signals()

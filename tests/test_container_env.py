"""The three container lanes are pinned, and they record what ran (D-M2, todo § 3).

`synth_bank.sh`, `librivox_align.sh` and `eiv_score.sh` each spin a throwaway
`rocm/pytorch` container, install wheels into it, produce artifacts, and vanish. Until
2026-08-07 all three ran `IMG=rocm/pytorch:latest` with unversioned `pip install`, so the
environment that rendered a campaign — or scored the EIV heads the corpus labels are
derived from — was not merely unpinned but unrecoverable.

Measured that day by dry-run resolution inside the pinned image: an unpinned
`pip install transformers` resolves to **5.14.1**, a major version past the 4.x the corpus
was built under. Only qwen pinned it. So this is not a tidiness rule — the next campaign
would have crossed a major version boundary in five of six engines and the renders would
have come out *different*, which is much harder to notice than broken.

Source-level and torch-free, so it runs on the host venv where `make test` runs.
"""

import pathlib
import re
import stat

REPO = pathlib.Path(__file__).resolve().parent.parent

LANES = [
    pathlib.Path("scripts/synthesis/synth_bank.sh"),
    pathlib.Path("scripts/synthesis/librivox_align.sh"),
    pathlib.Path("scripts/eiv_score.sh"),
]
SHARED = pathlib.Path("scripts/container_env.sh")
CAPTURE = pathlib.Path("scripts/capture_container_env.sh")


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def _code_lines(rel):
    """Non-comment, non-blank lines. Comments must stay free to explain the rule."""
    for i, line in enumerate(_read(rel).splitlines(), 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield i, stripped


def test_no_lane_runs_a_floating_image_tag():
    """`:latest` moves. When it moves under a re-run there is no way to tell it did —
    the container is healthy, the script succeeds, and the output is from a different
    stack than the one that produced everything it will be compared against."""
    offenders = []
    for lane in LANES + [SHARED]:
        for n, line in _code_lines(lane):
            if "rocm/pytorch:" in line:
                offenders.append(f"{lane}:{n}: {line}")
    assert not offenders, "floating image tag:\n" + "\n".join(offenders)


def test_the_pinned_digest_is_a_real_digest_and_pullable_by_anyone():
    """`docker inspect` also reports the local `sonora@sha256:…` re-tag. Pinning that
    would name an image only this host can resolve, which is not a pin."""
    shared = _read(SHARED)
    m = re.search(r'SONORA_ROCM_IMAGE:=([^}"\s]+)', shared)
    assert m, "SONORA_ROCM_IMAGE default is gone from scripts/container_env.sh"
    image = m.group(1)
    assert image.startswith("rocm/pytorch@sha256:"), image
    assert len(image.split("sha256:")[1]) == 64, f"not a full digest: {image}"


def test_every_lane_sources_the_one_pin_table():
    """Three copies of a version number is how B-L5 and D-L2 happened."""
    for lane in LANES:
        assert "container_env.sh" in _read(lane), f"{lane} does not source the pin table"


def test_container_installs_go_through_uv():
    """AGENTS.md § 3: containers bootstrap uv, then `uv pip install --python
    /opt/venv/bin/python`. `pip install uv` itself is the sanctioned bootstrap and is the
    only bare pip permitted — it is how the training container's own prep chain starts.
    """
    offenders = []
    for lane in LANES + [SHARED]:
        for n, line in _code_lines(lane):
            if re.search(r"\bpip install\b", line) and "uv pip install" not in line:
                if re.search(r"\bpip install -q uv\b", line):
                    continue  # the bootstrap
                offenders.append(f"{lane}:{n}: {line}")
    assert not offenders, "bare pip in a container lane:\n" + "\n".join(offenders)


def test_uv_never_uses_system_in_these_images():
    """`--system` bypasses the image's /opt/venv into Debian's externally-managed Python,
    which refuses under PEP 668 — it crash-looped two services on 2026-07-13."""
    for lane in LANES + [SHARED]:
        for n, line in _code_lines(lane):
            if "uv pip install" in line:
                assert "--system" not in line, f"{lane}:{n}: {line}"
                assert "/opt/venv/bin/python" in line or "$SONORA_UVPIP" in line, f"{lane}:{n}: {line}"


def test_transformers_is_pinned_in_the_one_place():
    shared = _read(SHARED)
    m = re.search(r'SONORA_PIN_TRANSFORMERS="transformers==([\d.]+)"', shared)
    assert m, "the transformers pin is gone from scripts/container_env.sh"
    assert m.group(1).startswith("4."), (
        f"transformers pinned to {m.group(1)}: the corpus was built on the 4.x line and "
        "crossing a major version silently changes renders rather than breaking them"
    )
    # And no lane may re-pin it locally — that is the drift this table prevents.
    for lane in LANES:
        for n, line in _code_lines(lane):
            if "transformers==" in line:
                assert "$SONORA_PIN_TRANSFORMERS" in line, f"{lane}:{n} pins transformers locally"


def test_every_lane_captures_its_environment():
    """A pin says what SHOULD run. The freeze says what DID. The second is what makes the
    next set of pins evidence instead of a guess, and it is the half that was missing."""
    for lane in LANES:
        source = _read(lane)
        assert "capture_env_cmd" in source, f"{lane} captures no environment"
        assert "_env" in source, f"{lane} names no capture directory"


def test_the_capture_runs_as_ai_mgr_and_after_the_setup_that_creates_it():
    """Root-owned files in a group-writable campaign dir are the papercut the umask 002
    convention exists to end — and `container_as_ai_mgr.sh` is what creates the user, so
    capturing before it would run as root or not at all."""
    shared = _read(SHARED)
    assert "runuser -u ai-mgr" in shared
    for lane in LANES:
        source = _read(lane)
        setup = source.index("container_as_ai_mgr.sh")
        capture = source.index("capture_env_cmd", source.index("docker run"))
        assert setup < capture, f"{lane} captures the environment before ai-mgr exists"


def test_the_capture_is_never_fatal():
    """A reproducibility measure that destroys a finished render is worse than no record.
    Verified live 2026-08-07: with an unwritable target it printed one line and exited 0.
    """
    source = _read(CAPTURE)
    assert source.rstrip().endswith("exit 0")
    assert source.count("exit 0") >= 3, "an early-return path can still fail the run"
    assert (REPO / CAPTURE).stat().st_mode & stat.S_IXUSR


def test_the_capture_records_torch_separately():
    """torch is the one package NOT installable from PyPI here — it is the image's ROCm
    build. A `pip freeze` alone leaves the hole exactly where the ROCm-vs-CUDA question
    gets asked."""
    assert "# torch:" in _read(CAPTURE)

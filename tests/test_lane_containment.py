"""The espeak lane has exactly one caller, and no exporter drops conditioning silently.

These are source-level checks on purpose. The behavioural half lives in
`test_cli_lanes.py`, which needs torch and therefore skips on the host venv — and the host
venv is where `make test` actually runs, so a guard that only fires in the container is a
guard that does not fire. Nothing here imports anything heavier than `pathlib`.

The rule being guarded (todo § 3, closed 2026-08-07): Sonora checkpoints train on op_g2p
IPA against the locked 178-symbol vocab. Feeding one espeak phonemes does not raise — it
produces fluent, confident, wrong audio. So the number of code paths that can reach
`english_cleaners2` without first deciding the lane must be zero, and it must stay zero.
"""

import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def _py_sources(package):
    return sorted((REPO / package).rglob("*.py"))


def _hits(paths, needle, allow=()):
    """Lines containing `needle`, skipping comments and any line matching `allow`."""
    found = []
    for path in paths:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or needle not in stripped:
                continue
            if any(ok in stripped for ok in allow):
                continue
            found.append(f"{path.relative_to(REPO)}:{n}: {stripped}")
    return found


def test_app_py_is_gone():
    """`matcha/app.py` phonemized through `english_cleaners2` unconditionally.

    Deleted rather than repointed at `process_text_for_lane`: `vocalizer.py` supersedes it,
    drives BOTH lanes off `detect_lane`, and does not `.launch(share=True)` from a machine
    holding unreleased checkpoints. Its console script and gradio pin went in the same cut.
    """
    assert not (REPO / "matcha" / "app.py").exists()
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "matcha.app:main" not in pyproject.replace("# `matcha-tts-app = matcha.app:main`", "")


def test_the_legacy_processor_has_exactly_one_caller():
    """`process_text` IS the espeak lane. Only `process_text_for_lane` may call it, and
    only after deciding the lane is legacy. Every other caller is lane-blind by
    construction, which is the defect — not a style preference.

    `matcha/utils/data/hificaptain.py` defines an unrelated local `process_text(infile,
    outpath)`; it is excluded by name, not by module, so a real leak there still fails.
    """
    offenders = _hits(
        _py_sources("matcha"),
        "process_text(",
        allow=("process_text_for_lane", "def process_text(infile", "process_text(filename"),
    )
    offenders = [o for o in offenders if "matcha/cli.py" not in o]
    assert not offenders, "lane-blind espeak callers:\n" + "\n".join(offenders)


def test_onnx_infer_takes_no_default_lane():
    """An ONNX graph does not record which phoneme vocabulary produced its ids, so nothing
    in that lane can detect what `matcha.cli.detect_lane` reads off a checkpoint. Either
    default is silently wrong for half the graphs it will be pointed at; `required=True`
    is the only honest setting, and a later "convenience" default would undo the fix.
    """
    source = (REPO / "matcha" / "onnx" / "infer.py").read_text(encoding="utf-8")
    block = source.split('"--lane"', 1)
    assert len(block) == 2, "--lane is gone from matcha/onnx/infer.py"
    assert "required=True" in block[1].split(")")[0] + block[1][:400]
    assert "process_text_for_lane" in source
    assert "sf.write(output_filename, audio, sample_rate" in source


def test_the_onnx_exporter_guards_conditioned_checkpoints():
    """`get_inputs` builds no VAT input node and `onnx_forward_func` passes no `vat`, so
    this exporter CAN NOT carry conditioning. Without the guard it exports anyway and the
    graph renders neutral speech forever — real speech, so nothing downstream can tell.
    F-H1 was the same shape: an export whose logs said Sonora and whose graph was not.
    """
    source = (REPO / "matcha" / "onnx" / "export.py").read_text(encoding="utf-8")
    assert "def assert_exportable_here(" in source
    assert "assert_exportable_here(matcha)" in source
    # The guard must key on `use_vat`. `vat_dim` defaults to 3 on unconditioned
    # checkpoints too, so a width test would refuse the legacy demos this is still for.
    guard = source.split("def assert_exportable_here(", 1)[1].split("\ndef ", 1)[0]
    assert 'getattr(matcha, "use_vat"' in guard

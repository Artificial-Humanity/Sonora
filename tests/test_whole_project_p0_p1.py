"""Regression tests for the P0 and P1 findings from the 2026-11-02 whole-project review."""

import ast
import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def test_v001_convert_final_declares_width_160_time_embed_dim():
    """V-001 (P0): convert_final.py feeds width-160 sin_pos_emb to the decoder and must
    record time_embed_dim=160 in config.json (matching convert_vat.py's F-M4 fix)."""
    src = (REPO / "scripts/litert_export/convert_final.py").read_text(encoding="utf-8")
    assert "B.sin_pos_emb(t.reshape(-1), 160)" in src
    assert "time_embed_dim=160" in src
    assert "time_embed_dim=1024" not in src


def test_v002_batched_synthesis_uses_global_utterance_index():
    """V-002 (P1): batched_synthesis must derive utterance filenames from a cross-batch
    index so batch 2 does not overwrite batch 1's utterance_000.wav."""
    src = (REPO / "matcha/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "batched_synthesis"
    )
    body_src = ast.get_source_segment(src, fn)
    assert "idx = (i - 1) * args.batch_size + j" in body_src
    assert 'f"utterance_{idx:03d}' in body_src


def test_v003_unknown_cleaner_raises_unknown_cleaner_exception():
    """V-003 (P1): _clean_text must raise UnknownCleanerException rather than AttributeError."""
    from matcha.text import UnknownCleanerException, _clean_text

    with pytest.raises(UnknownCleanerException, match="no_such_cleaner_xyz"):
        _clean_text("hello", ["no_such_cleaner_xyz"])


def test_v004_every_data_config_names_a_cleaner_that_exists():
    """V-004 (P1): every configs/data/*.yaml cleaner entry must exist on matcha.text.cleaners."""
    from matcha.text import cleaners

    for path in sorted((REPO / "configs/data").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name in cfg.get("cleaners") or []:
            assert hasattr(cleaners, name), (
                f"{path.name} references cleaner {name!r}, which does not exist in "
                "matcha.text.cleaners"
            )


def test_v005_v006_extract_tar_and_zip_accept_pathlib_path(tmp_path):
    """V-005 & V-006 (P1): _extract_tar and _extract_zip must convert `from_path` (not the
    `Path` class) to a string when given a `pathlib.Path`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "matcha_utils_data_utils", REPO / "matcha/utils/data/utils.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _extract_tar, _extract_zip = mod._extract_tar, mod._extract_zip

    tar_path = tmp_path / "sample.tar"
    with tarfile.open(tar_path, "w") as tf:
        data = b"hello tar"
        info = tarfile.TarInfo(name="a.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    out_tar = tmp_path / "out_tar"
    out_tar.mkdir()
    extracted_tar = _extract_tar(tar_path, str(out_tar))
    assert extracted_tar == [str(out_tar / "a.txt")]
    assert (out_tar / "a.txt").read_bytes() == b"hello tar"

    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("b.txt", b"hello zip")
    out_zip = tmp_path / "out_zip"
    out_zip.mkdir()
    extracted_zip = _extract_zip(zip_path, str(out_zip))
    assert extracted_zip == ["b.txt"]
    assert (out_zip / "b.txt").read_bytes() == b"hello zip"


def test_v007_hificaptain_does_not_import_torchaudio():
    """V-007 (P1): hificaptain.py must use soundfile + soxr rather than torchaudio.load/save."""
    src = (REPO / "matcha/utils/data/hificaptain.py").read_text(encoding="utf-8")
    assert "torchaudio" not in src
    assert "sf.read" in src and "sf.write" in src


def test_v008_get_durations_passes_weights_only_false_and_eval():
    """V-008 (P1): get_durations_from_trained_model.py must pass weights_only=False and
    put the model in eval() mode."""
    src = (REPO / "matcha/utils/get_durations_from_trained_model.py").read_text(encoding="utf-8")
    assert "weights_only=False" in src
    assert "model.eval()" in src


def test_v013_render_step2_anchors_honors_sonora_ecapa_snap_env(monkeypatch):
    """V-013 (P1): ECAPA_SNAP in render_step2_anchors.py must allow SONORA_ECAPA_SNAP override."""
    src = (REPO / "scripts/tools/render_step2_anchors.py").read_text(encoding="utf-8")
    assert 'os.environ.get(\n    "SONORA_ECAPA_SNAP"' in src


def test_v014_bench_loader_sets_project_root_to_repo_root():
    """V-014 (P1): bench_loader.py must set REPO = _SONORA_REPO (the repo root, not scripts/)."""
    src = (REPO / "scripts/tools/bench_loader.py").read_text(encoding="utf-8")
    assert "REPO = _SONORA_REPO" in src

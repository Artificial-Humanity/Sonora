"""Put this repo's own modules on `sys.path` for the whole suite — once, and derived from disk.

⚠⚠ WHY THIS EXISTS: 50 TESTS COULD VANISH INTO THE WORD "SKIPPED".
`tests/test_license_wall.py` opens with `pytest.importorskip("matcha.data.license_wall")`, and
most test modules here do the same for a script they exercise. Measured 2026-09-11, same file,
same machine, same commit:

    .venv/bin/python -m pytest tests/test_license_wall.py   ->  50 passed
    .venv/bin/pytest            tests/test_license_wall.py  ->   1 skipped

`python -m pytest` puts the current directory on `sys.path`; the console script does not. So the
imports only ever worked by accident of invocation — and in a FULL run, by accident of collection
order, because nine test modules anchored `sys.path` as a side effect of being imported first.
`pytest-randomly` shuffles that order. Nothing was ever wrong with the tests; the question of
whether they ran at all was answered by which command someone typed.

⚠ AND THE FAILURE IS SILENT BY CONSTRUCTION. A module-level `importorskip` that misses reports
one skipped ITEM, not the fifty it swallowed, and "1 skipped" at the end of a green run is
indistinguishable from a deliberate environmental skip. That is this repo's most expensive
shape — `test_doc_claims`'s vacuous pass, `test_stage_coverage`'s unwired stage, #247's
hand-kept list — arriving in the one place that is supposed to catch it.

⚠ DERIVED, NOT TYPED (#247). The buckets under `scripts/` are globbed, because a hand-kept list
beside a directory goes stale the moment a bucket is added — and this file would go stale
silently, which is the whole defect it exists to remove.
`tests/test_no_first_party_import_is_optional.py` is the mechanism that says so out loud: it
asserts every in-repo `importorskip` target actually imports, so a bucket this misses becomes a
failure rather than a skip.

⚠ This does NOT change how scripts run outside pytest. Each entry point still anchors itself the
way AGENTS.md §5c describes; this only removes the accident from the test suite.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent


def _anchors():
    yield _ROOT                                   # `matcha.*`, `tests.*`
    scripts = _ROOT / "scripts"
    if scripts.is_dir():
        # Bare-name imports (`synth_common`, `qc_verdict`, …) resolve from their bucket.
        # ⚠ Verified 2026-09-11: no basename collides across buckets, so a flat path is
        # unambiguous. If that stops being true, two buckets will shadow each other here
        # silently — the guard test is what would surface it.
        for d in sorted(scripts.iterdir()):
            if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")):
                yield d


for _anchor in _anchors():
    _s = str(_anchor)
    if _s not in sys.path:
        sys.path.insert(0, _s)

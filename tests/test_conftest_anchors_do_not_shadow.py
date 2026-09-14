"""`conftest.py` puts six `scripts/` buckets AHEAD of the repo root, the stdlib and site-packages.

⚠⚠ THAT ORDER IS DELIBERATE AND IT IS ALSO A SHADOWING HAZARD (#447). Tests import bucket
modules by bare name — `synth_common`, `qc_verdict` — so the buckets have to win against each
other's absence. But nothing made them win only where intended: a file named `scripts/tools/
json.py` would shadow the standard library for the entire test session, and two buckets holding
the same basename would resolve to whichever sorted last.

⚠ `conftest.py` used to claim the import guard "is what would surface" a collision. It does not,
and the reviewer reproduced that: the shadowed module imports perfectly well — it is simply the
wrong one, which no import check can see. A claim that a guard covers something it does not is
worse than no guard, because it stops anyone building the real one. **This file is the real one.**

⚠ THE POPULATIONS ARE WHAT THIS FILE CAN SEE, WHICH IS NOT EVERYTHING IMPORTABLE. It compares
bucket basenames against other buckets, the stdlib (source and builtins), site-packages
(packages, namespace packages and top-level `.so`) and the repo root (packages and loose `.py`).
It does NOT see platform-absent stdlib names, or anything a `.pth` adds at runtime. The first
version claimed four populations while missing namespace packages, extension modules and the
root's loose modules (#448) — so the limit is stated here rather than implied away.
"""
import pathlib
import sys
import sysconfig

REPO = pathlib.Path(__file__).resolve().parents[1]


def _bucket_modules():
    """{basename: [bucket, ...]} for every module `conftest.py` puts on `sys.path`.

    ⚠ THE DIRECTORIES COME FROM `conftest._anchors()` ITSELF (#449). This used to re-express
    conftest's rule — glob `scripts/*`, skip `__pycache__`, require a `.py` — under a docstring
    saying a second expression "would be the copy that drifts". It was that copy. If conftest
    starts anchoring a seventh directory, this now follows without being edited; before, it
    would have gone on checking six and passed.
    """
    import conftest                                  # already imported by pytest; same object
    out = {}
    for d in conftest._anchors():
        if d == REPO:
            continue
        for f in pathlib.Path(d).glob("*.py"):
            out.setdefault(f.stem, []).append(pathlib.Path(d).name)
    return out


def _stdlib_names():
    return ({p.stem for p in pathlib.Path(sysconfig.get_paths()["stdlib"]).glob("*.py")}
            | {p.name for p in pathlib.Path(sysconfig.get_paths()["stdlib"]).iterdir()
               if p.is_dir() and (p / "__init__.py").exists()}
            | set(sys.builtin_module_names))


def _site_packages_names():
    """Top-level importable names in site-packages.

    ⚠ NAMESPACE PACKAGES AND EXTENSION MODULES COUNT (#448). The first version required an
    `__init__.py`, so it could not see `google` — a namespace package here — and a bucket module
    named `google.py` would have shadowed it, breaking `google.protobuf` and with it
    `ai_edge_litert`, while both guards stayed green. Compiled top-level modules (`.so`) were
    invisible for the same reason.
    """
    sp = pathlib.Path(sysconfig.get_paths()["purelib"])
    if not sp.is_dir():
        return set()
    names = {p.stem for p in sp.glob("*.py")}
    names |= {p.name.split(".")[0] for p in sp.glob("*.so")}
    for d in sp.iterdir():
        if d.is_dir() and d.name != "__pycache__" and not d.name.endswith((".dist-info", ".data")):
            names.add(d.name)                        # packages AND namespace packages
    return names


def _repo_top_level():
    """Top-level importable names at the repo root — packages AND loose modules.

    ⚠ THE LOOSE MODULES WERE MISSING (#448). Buckets are anchored AHEAD of the repo root, so a
    bucket module named `vocalizer` would shadow this repo's own `vocalizer.py`. Measured: the
    root holds `conftest`, `setup` and `vocalizer`, none of which the package-only scan saw.
    """
    names = {d.name for d in REPO.iterdir() if d.is_dir() and (d / "__init__.py").exists()}
    names |= {p.stem for p in REPO.glob("*.py")}
    return names


def test_the_populations_are_non_empty():
    """⚠ THE CONTROL. Every assertion below is a NEGATIVE — "no collisions" — which an empty
    population satisfies in silence. That is the vacuous pass this whole branch is about.
    """
    assert len(_bucket_modules()) > 50, "the bucket scan is broken"
    assert len(_stdlib_names()) > 50, "the stdlib scan is broken"
    assert "matcha" in _repo_top_level(), "the repo-package scan is broken"
    assert _site_packages_names(), "the site-packages scan is broken"


def test_no_bucket_module_shadows_another_bucket_or_the_stdlib_or_site_packages():
    buckets = _bucket_modules()
    names = set(buckets)
    problems = {
        "in two buckets at once": {k: v for k, v in buckets.items() if len(v) > 1},
        "shadowing the stdlib": sorted(names & _stdlib_names()),
        "shadowing site-packages": sorted(names & _site_packages_names()),
        "shadowing a repo package": sorted(names & _repo_top_level()),
    }
    hits = {k: v for k, v in problems.items() if v}
    assert not hits, (
        "conftest.py anchors the scripts/ buckets ahead of everything else, so these names are "
        f"imported in preference to what they collide with, for the whole session: {hits}")


def test_the_collision_check_can_fail():
    """⚠ THE POSITIVE CONTROL, on the case that matters — a bucket module named for a stdlib
    module. Without this, the test above is a negative asserted over data nobody proved the
    checker can react to.
    """
    fake = dict(_bucket_modules())
    fake["json"] = ["tools"]           # the exact hazard: shadowing the stdlib session-wide
    fake["synth_common"] = ["lib", "tools"]
    assert set(fake) & _stdlib_names(), "the checker cannot see a stdlib collision"
    assert [k for k, v in fake.items() if len(v) > 1], "the checker cannot see a duplicate"


def test_the_widened_scans_still_see_what_a_narrow_one_misses():
    """⚠ THE WIDENINGS HAD NO REGRESSION CONTROL, SO REVERTING THEM STAYED GREEN (#450).

    #448 widened two scans: site-packages gained namespace packages and top-level `.so`
    modules, and the repo scan gained loose `.py` files at the root. Because there are zero
    collisions today, a NARROWED scan finds none either — the reviewer measured that reverting
    all three left this file at 3 passed. The widenings guarded the tree; nothing guarded the
    widenings.

    ⚠ ASSERTED AS A STRICT SUPERSET, NOT BY NAMING `google` OR `vocalizer`. A name-specific
    check pins this venv's contents and this repo's root listing, and goes stale the moment
    either changes — the #247 hand-list defect, which is what #449 was. The narrow form is
    computed here only to be the thing the real scan must exceed; it is not a second expression
    of the scan's rule, because the scan is never asked to agree with it — only to contain it.
    """
    sp = pathlib.Path(sysconfig.get_paths()["purelib"])
    narrow_site = ({p.stem for p in sp.glob("*.py")}
                   | {d.name for d in sp.iterdir() if d.is_dir() and (d / "__init__.py").exists()})
    narrow_repo = {d.name for d in REPO.iterdir() if d.is_dir() and (d / "__init__.py").exists()}

    wide_site, wide_repo = _site_packages_names(), _repo_top_level()
    assert narrow_site and narrow_repo, "the narrow scans found nothing, so neither comparison means anything"
    assert wide_site > narrow_site, (
        "the site-packages scan no longer sees more than packages-with-__init__ — namespace "
        "packages and top-level .so modules have gone missing from it, and a bucket module "
        f"named for one of them would shadow it silently (#448). missing: {narrow_site - wide_site or 'narrowed'}")
    assert wide_repo > narrow_repo, (
        "the repo scan no longer sees the root's loose .py modules, so a bucket module named "
        "`vocalizer` or `setup` would shadow this repo's own and nothing would say so")

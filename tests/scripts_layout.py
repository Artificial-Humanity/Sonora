"""Where a `scripts/` module lives, for tests that import or read one.

Before #26 step 3 every test that touched a script hardcoded
`REPO / "scripts" / "synthesis"` — correct while one directory held almost everything, and
20 separate copies of a fact the moment it stopped being true. `scripts/` is now
`stages/ lib/ tools/ gates/ assets/ teacher_audition/ litert_export/`, so the *lookup* is
what tests should depend on, not the layout.

    from scripts_layout import SCRIPTS

    SCRIPTS / "qc_gate.py"      # -> the real path, whichever bucket it is in
    SCRIPTS.src("qc_gate.py")   # -> its source text
    SCRIPTS.on_path()           # -> every bucket importable, for `import synth_common`

A basename that does not exist raises, so a test cannot silently read nothing — which is
how `scripts/gates/test_skill_files.py` spent weeks comparing against an app that had moved
(see `tests/test_asset_paths.py`).
"""

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# stages first: `qc_gate.py` and friends are stages that other code also imports, and a
# stage is the authoritative copy.
BUCKETS = ("stages", "lib", "tools", "gates", "assets", "teacher_audition", "litert_export")
ASSETS = REPO / "scripts" / "assets"


class _Scripts:
    """Resolves a script by basename across the buckets."""

    dirs = tuple(REPO / "scripts" / b for b in BUCKETS) + (REPO / "scripts",)

    def __truediv__(self, name):
        for d in self.dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"no script named {name!r} under scripts/ — looked in "
            f"{[d.name for d in self.dirs]}. If it was deleted, the test that names it "
            f"should go too; if it moved out of these buckets, add the bucket here."
        )

    def src(self, name):
        return (self / name).read_text(encoding="utf-8")

    def on_path(self):
        """Make flat `import <module>` work for every bucket, plus `import matcha`."""
        for d in (*reversed(self.dirs), REPO):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))

    def __str__(self):
        raise TypeError(
            "SCRIPTS is not a single directory any more (#26 step 3). Use `SCRIPTS / name` "
            "for a path, `SCRIPTS.src(name)` for source, or `SCRIPTS.on_path()` to import."
        )

    __fspath__ = __str__


SCRIPTS = _Scripts()

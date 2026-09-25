"""run_in_rocm.sh must install the SAME hydra as training, or what it writes will not load.

A warm-start checkpoint built through the runner with an unpinned hydra-core pickled
`hydra._internal.target_policy`, which the training container's hydra 1.3.2 lacks; the
trainer failed at load (2026-09-25). The pins live in pyproject.toml; this holds the runner
to them rather than restating them.
"""

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_pin(pkg):
    with open(os.path.join(REPO, "pyproject.toml"), encoding="utf-8") as f:
        m = re.search(r'"%s==([^"]+)"' % re.escape(pkg), f.read())
    assert m, "%s is not pinned in pyproject.toml" % pkg
    return m.group(1)


def test_the_runner_pins_hydra_to_the_training_versions():
    with open(os.path.join(REPO, "scripts/stages/run_in_rocm.sh"), encoding="utf-8") as f:
        deps = re.search(r'^DEPS="([^"]*)"', f.read(), re.M).group(1).split()
    for pkg in ("hydra-core", "hydra-colorlog"):
        pinned = [d for d in deps if d.split("==")[0] == pkg]
        assert pinned == ["%s==%s" % (pkg, _pyproject_pin(pkg))], (
            "run_in_rocm.sh installs %s as %r; training pins %s==%s"
            % (pkg, pinned, pkg, _pyproject_pin(pkg)))

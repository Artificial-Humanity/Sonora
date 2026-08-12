"""Export-gate smoke: the FiLM op chain through litert-torch at fixed shapes.

Converts (VATTrunk -> FiLMLayer) exactly as it will sit inside the split
graphs — raw [1, 3, T] vat input, [1, C, T] features, [1, 1, T] mask — and
checks (a) GPU-clean op report, (b) tflite-vs-torch parity with random
weights (zero-init would trivially pass; random weights actually exercise
the multiply/add lowering).
"""
import os
import sys

# The litert-torch conversion harness workspace (build_matcha.py + venv).
HARNESS = os.environ.get("SONORA_LITERT_HARNESS", "/data/toolchain/litert-conversion")
sys.path.insert(0, HARNESS)
import _stub  # noqa: F401,E402

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("litert_export",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

import build_matcha as B
from matcha.models.components.film import FiLMLayer, VATTrunk

T, C, COND = 256, 192, 256


class FilmChain(nn.Module):
    """The conditioning subgraph as it will appear in-graph."""

    def __init__(self):
        super().__init__()
        self.trunk = VATTrunk(3, COND)
        self.film = FiLMLayer(COND, C)
        # Random weights: exercise the real lowering, not the zero shortcut.
        for p in self.parameters():
            nn.init.normal_(p, 0, 0.05)

    def forward(self, x, vat, mask):
        c = self.trunk(vat * mask)
        return self.film(x, c, mask)


torch.manual_seed(0)
m = FilmChain().eval()
x = torch.randn(1, C, T)
vat = torch.randn(1, 3, T)
mask = torch.ones(1, 1, T)
mask[0, 0, 200:] = 0.0

with torch.no_grad():
    ref = m(x, vat, mask).numpy()

path = B.convert(m, (x, vat, mask), os.path.join(HARNESS, "film_gate.tflite"))
clean = B.opcheck(path, "film_chain")
cm = B.tfl_load(path)
out = B.tfl_run(cm, x.numpy(), vat.numpy(), mask.numpy())[0]

corr = float(np.corrcoef(ref.reshape(-1), out.reshape(-1))[0, 1])
maxdiff = float(np.abs(ref - out).max())
print(f"GPU-clean: {clean}")
print(f"parity: corr={corr:.6f} max|diff|={maxdiff:.3e}")
ok = bool(clean and corr > 0.9999)
print("EXPORT GATE:", "PASS" if ok else "FAIL")
# Exit code, so this is usable as a pipeline gate. Printing FAIL and exiting
# 0 made the parity check decorative in any scripted lane.
sys.exit(0 if ok else 1)

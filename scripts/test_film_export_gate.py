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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
print("EXPORT GATE:", "PASS" if clean and corr > 0.9999 else "FAIL")

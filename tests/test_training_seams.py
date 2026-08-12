"""Regression tests for the two training-loop seams carried as E-M5 and E-M6.

Both are silent failures. E-M5 corrupted only a *logged* number, which is worse than a
crash: it inflated `sub_loss/val_diff_loss` against train and so invited every curve
after 2026-08-01 to be read as overfitting. E-M6 aborted a run hours in, and only
sometimes — it needs a validation pass followed by a training batch whose text is no
longer than the longest text validation happened to see.

These need torch and the model's own dependency stack, which live in the ROCm training
container. The host `.venv` carries the synthesis lane only, so they skip there:

    docker compose run --rm --entrypoint pytest sonora_training tests/test_training_seams.py
"""

import pathlib
import pytest

torch = pytest.importorskip("torch")
flow_matching = pytest.importorskip("matcha.models.components.flow_matching")
text_encoder = pytest.importorskip("matcha.models.components.text_encoder")


class _MaskedEstimator(torch.nn.Module):
    """Stands in for `Decoder`, carrying the one property this test turns on: its
    output is masked (`decoder.py` ends `return output * mask`), so the padded frames
    contribute nothing to the prediction while `u` still holds noise there."""

    def forward(self, y, mask, mu, t, spks=None, cond=None):  # pylint: disable=unused-argument
        return (mu * 0.5) * mask


def _cfm():
    cfm_params = type("P", (), {"solver": "euler", "sigma_min": 1e-4})()
    model = flow_matching.BASECFM(n_feats=8, cfm_params=cfm_params)
    model.estimator = _MaskedEstimator()
    return model


def _diff_loss(model, n_pad):
    """`compute_loss` on one 16-frame clip padded out to 16 + n_pad frames."""
    n_feats, valid = 8, 16
    total = valid + n_pad

    x1 = torch.zeros(1, n_feats, total)
    x1[:, :, :valid] = torch.linspace(-1.0, 1.0, valid).expand(n_feats, valid)
    mu = torch.zeros(1, n_feats, total)
    mu[:, :, :valid] = 0.25
    mask = torch.zeros(1, 1, total)
    mask[:, :, :valid] = 1.0

    loss, _ = model.compute_loss(x1=x1, mask=mask, mu=mu)
    return float(loss)


@pytest.fixture(name="fixed_noise")
def _fixed_noise(monkeypatch):
    """Pin the two random draws inside `compute_loss` so padding is the only variable.

    `z = ones` is deliberate rather than convenient: it makes `u` in the padded region
    exactly `-(1 - sigma_min)`, i.e. the noise floor E-M5 was summing. A zero draw would
    make the test pass either way.
    """
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    monkeypatch.setattr(torch, "rand", lambda size, **kw: torch.full(size, 0.5, **kw))


def test_diff_loss_is_padding_invariant(fixed_noise):  # pylint: disable=unused-argument
    """E-M5: the logged loss must describe the clip, not the batch it was padded into."""
    model = _cfm()
    unpadded = _diff_loss(model, n_pad=0)
    padded = _diff_loss(model, n_pad=48)

    assert unpadded > 0.0
    assert padded == pytest.approx(unpadded, rel=1e-6)


def test_rope_cache_survives_inference_mode():
    """E-M6: a cache first built during validation must still be usable for backward.

    Mirrors the real sequence — `on_validation_end` synthesises under
    `@torch.inference_mode()` with some long text, then training resumes on a batch that
    is no longer than that, so the cache is reused rather than rebuilt.
    """
    rope = text_encoder.RotaryPositionalEmbeddings(4)

    with torch.inference_mode():
        rope(torch.zeros(1, 1, 12, 8))

    assert rope.cos_cached is not None
    assert not rope.cos_cached.is_inference()
    assert not rope.sin_cached.is_inference()

    x = torch.nn.Parameter(torch.randn(1, 1, 6, 8))
    rope(x).sum().backward()

    assert x.grad is not None


# --- TR-L4: an unbound variable on a path no config reaches yet -----------------------


def test_configure_optimizers_survives_a_scheduler_without_last_epoch():
    """`current_epoch` was assigned only inside `if "last_epoch" in signature(...)`, while
    `scheduler.last_epoch = current_epoch` ran unconditionally — so a scheduler that does
    NOT take `last_epoch` raised `NameError`. Unreachable today (no config defines a
    scheduler at all), and the first scheduler experiment would have hit it immediately.
    """
    import inspect as _inspect

    base = pytest.importorskip("matcha.models.baselightningmodule")
    src = pathlib.Path(base.__file__).read_text(encoding="utf-8")
    body = src[src.index("def configure_optimizers"):src.index("def ", src.index("def configure_optimizers") + 10)]
    # The assignment must dominate the use, not sit under the branch that may not run.
    assert body.index("current_epoch = -1") < body.index('if "last_epoch" in')
    assert body.index("current_epoch = -1") < body.index("scheduler.last_epoch = current_epoch")

    # And the shape of the bug, reproduced so the fix has something to be a fix OF.
    def old(has_last_epoch):
        if has_last_epoch:
            current_epoch = -1
        return current_epoch                      # noqa: F821 — the defect
    assert old(True) == -1
    with pytest.raises(UnboundLocalError):
        old(False)
    assert "last_epoch" in _inspect.signature(
        torch.optim.lr_scheduler.ExponentialLR).parameters, (
        "the schedulers that DO take it are why the branch exists")

"""What the mobile host is told about the control surface (F-H2, F-M7).

`config.json` is the ONLY thing a host can read. If it and the graph disagree, nothing
on-device can detect it — the graph runs, the audio is fluent, and the dial is wrong. So
these are contract tests on the manifest, not on the conversion.

Torch-free: they rebuild the manifest's control block from the same module `convert_vat`
builds it from. Importing the converter itself would drag in litert, diffusers and a GPU.
"""

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

delivery = pytest.importorskip("matcha.delivery")
CONVERTER = REPO / "scripts" / "litert_export" / "convert_vat.py"
SRC = CONVERTER.read_text(encoding="utf-8")


def test_the_converter_takes_its_width_from_the_contract():
    """It hardcoded `VAT_DIM = 3` and a three-name channel map, which gave it its own
    opinion about the control surface. A graph and its manifest disagreeing is the one
    failure a host cannot detect."""
    assert "VAT_DIM = delivery.VAT_DIM" in SRC
    assert re.search(r"N_SPKS, SPK_DIM, VAT_COND = 247, 64, 256", SRC), \
        "VAT_DIM is back in the constants tuple"
    assert "CHANNELS = dict(enumerate((\"valence\", \"energy\", \"tension\")" in SRC
    assert "delivery.DELIVERY_LANES))" in SRC


def test_the_manifest_records_the_two_sigma_clamp():
    """No bound was recorded ANYWHERE. V/A/T are per-speaker z-scores clamped at 2 sigma
    in derivation, so ±1 is the edge of the TRAINED range — but a host reading the old
    manifest had no way to know, and a request for valence 5 renders fluent speech off the
    manifold rather than failing."""
    assert "control=dict(" in SRC
    assert "min=-1.0, max=1.0, neutral=0.0" in SRC
    assert 'clamp="reject"' in SRC
    assert "2 sigma" in SRC


def test_the_manifest_says_the_delivery_block_is_categorical():
    """Handed eight floats and three continuous names, a host will reasonably crossfade
    all eight and blend Newscaster into Dialogue — a vector the model never saw and that
    `lane_of_vector` refuses outright."""
    assert 'encoding="one_hot"' in SRC
    assert "DO NOT INTERPOLATE" in SRC
    assert "values=list(delivery.DELIVERY_LANES)" in SRC
    assert "unknown=dict(vector=[0.0] * delivery.DELIVERY_DIM" in SRC


def test_the_manifest_documents_cfg_which_exports_nothing():
    """F-M7. CFG is pure host orchestration — the graph is identical for both passes, so
    NOTHING about it appears in the artifacts and a host cannot discover it. The one
    number that makes it safe (≥25 ODE steps) lives nowhere else."""
    assert "guidance=dict(" in SRC
    assert "supported=True, exported=False" in SRC
    assert "min_n_timesteps_above_1=25" in SRC
    assert "v_uncond + s * (v_cond - v_uncond)" in SRC


def test_a_mismatched_checkpoint_is_refused_with_the_reason():
    """A NARROWER pre-v2 checkpoint is refused deliberately: its graph has no delivery
    inputs, but the manifest written here declares the lane vocabulary, so a host reading
    it would send eight floats to a three-channel graph."""
    assert "exports the " in SRC and "contract width" in SRC
    assert "predates contract v2" in SRC
    assert "different artifact, not a truncation" in SRC


def test_g5_probes_a_lane_one_hot_not_at_minus_one():
    """-1 on a one-hot lane is not "the opposite of Newscaster", it is a vector the model
    never saw and the host is forbidden to send. Certifying the wire with it would be a
    green light for a path nobody may take."""
    assert "categorical = idx in CATEGORICAL_CHANNELS" in SRC
    block = SRC[SRC.index("categorical = idx in CATEGORICAL_CHANNELS"):]
    block = block[:block.index("gates_or_die()")]
    assert "if categorical:" in block
    assert "one-hot ->" in block


def test_g6_requires_the_lanes_to_be_distinguishable():
    """Five inputs wired to one summing junction pass G5 five times and give the host five
    names for one behaviour — which reads on-device as a weak channel, not a broken one."""
    assert "G6 delivery lanes are distinguishable" in SRC
    assert "closest pair" in SRC


def test_no_width_three_literal_survives_in_a_driven_vector():
    """`torch.tensor([0.0, a, 0.0])` would raise on a width change — loud, but from inside
    a builder, and the reader would have to work out that energy is slot 1 of however
    many."""
    assert 'torch.tensor([0.0, a, 0.0])' not in SRC
    assert 'vat_i[SLOT["energy"]] = a' in SRC
    assert "render([0.0] * VAT_DIM)" in SRC


# --- the manifest is machine-usable, not just present --------------------------------


def _control_block():
    """Rebuild `control` exactly as the converter does, without importing it."""
    channels = dict(enumerate(("valence", "energy", "tension") + delivery.DELIVERY_LANES))
    continuous = tuple(range(delivery.VAT_BASE_DIM))
    categorical = tuple(range(delivery.VAT_BASE_DIM, delivery.VAT_DIM))
    return channels, continuous, categorical


def _validate(vec):
    """The validation a host can perform from `control` alone. Returns a reason or None."""
    _channels, continuous, _categorical = _control_block()
    for i in continuous:
        if not -1.0 <= vec[i] <= 1.0:
            return f"channel {i} outside [-1, 1]"
    try:
        delivery.lane_of_vector(vec)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize(("vec", "ok"), [
    ([0.4, -0.2, 0.1, 0, 0, 0, 1, 0], True),    # in range, one lane hot
    ([0.0, 0.0, 0.0, 0, 0, 0, 0, 0], True),     # neutral + unknown == contract v1
    ([5.0, 0, 0, 0, 0, 0, 0, 0], False),        # out of the trained range
    ([0, 0, 0, 0.5, 0.5, 0, 0, 0], False),      # interpolated between two lanes
    ([0, 0, 0, 1, 0, 0, 1, 0], False),          # two lanes hot
])
def test_a_host_can_validate_before_it_renders(vec, ok):
    """The point of the contract: every one of these was previously accepted silently and
    produced fluent audio. Three of the five are wrong."""
    assert (_validate(vec) is None) is ok


def test_the_channel_map_covers_every_channel_exactly_once():
    channels, continuous, categorical = _control_block()
    assert len(channels) == delivery.VAT_DIM
    assert set(continuous) | set(categorical) == set(range(delivery.VAT_DIM))
    assert not set(continuous) & set(categorical)
    assert len(set(channels.values())) == delivery.VAT_DIM, "a duplicate channel name"


def test_the_shipped_manifest_predates_the_contract_and_is_readable():
    """The artifacts on /data are from the 3-channel era. This is not a failure — it
    records WHY re-export is required, and proves the old manifest genuinely carried no
    bound and no categorical marker for a host to read."""
    path = pathlib.Path("/data/toolchain/litert-conversion/artifacts_vat/config.json")
    if not path.is_file():
        pytest.skip("no exported artifacts on this machine")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["vat_dim"] == 3, "the shipped artifacts are pre-contract-v2"
    assert "control" not in cfg, "no control contract — this is F-H2, as filed"
    assert not any(k in cfg for k in ("vat_min", "vat_max", "clamp")), \
        "the old manifest recorded no bound at all"

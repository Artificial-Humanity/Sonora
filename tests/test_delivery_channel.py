"""The delivery channel, contract v2 (§1 of notes/todo.md, closed 2026-08-07).

The migration's whole risk is that a WIDTH mistake and a MEANING mistake look identical
from outside: both produce a model that renders fluent speech. `scripts/test_vat_dim_seams.py`
covers width (22 checks, run in the training container — it needs torch). This file covers
meaning, and is torch-free so it runs on the host where `make test` runs.

`vat_dim` is 8, not the 4 the todo proposed. The reasoning is in `matcha/delivery.py`; the
short form is that a single ordered channel asserts the five lanes lie on one continuum,
and `seed_delivery.py` records that they do not — Dialogue vs Neutral is a property of the
TEXT, Newscaster vs Documentary a property of the RENDER.
"""

import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

delivery = pytest.importorskip("matcha.delivery")


def test_the_width_is_three_plus_five():
    assert delivery.VAT_BASE_DIM == 3
    assert delivery.DELIVERY_DIM == 5
    assert delivery.VAT_DIM == 8


def test_the_model_config_is_the_single_source_of_width():
    """Every data config interpolates `${model.vat_dim}`, so this one line decides what
    the filelist parse, the collate and the FiLM trunk all expect. If it and the encoding
    disagree, the seam guards catch the shape and nothing catches the meaning."""
    text = (REPO / "configs" / "model" / "matcha.yaml").read_text(encoding="utf-8")
    assert int(re.search(r"^vat_dim:\s*(\d+)", text, re.M).group(1)) == delivery.VAT_DIM
    for name in ("libritts_r_vat.yaml", "libritts_r_vat_v2.yaml", "libritts_r_vat_v3c.yaml"):
        data = (REPO / "configs" / "data" / name).read_text(encoding="utf-8")
        assert "vat_dim: ${model.vat_dim}" in data, f"{name} pins its own width"


def test_unknown_is_exactly_the_v1_vector_padded():
    """The contract's compatibility property: `unknown ≡ zero vector ≡ v1 behavior`. A
    corpus re-derived at the new width with no delivery labels — which is every LibriTTS
    clip, since that corpus predates the axis — produces the conditioning it always did."""
    assert delivery.vat_vector(0.4, -0.2, 0.1) == [0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert delivery.delivery_onehot(None) == [0.0] * 5
    assert delivery.delivery_onehot("") == [0.0] * 5


@pytest.mark.parametrize("lane", delivery.DELIVERY_LANES)
def test_every_lane_round_trips(lane):
    assert delivery.lane_of_vector(delivery.vat_vector(0, 0, 0, lane)) == lane


def test_exactly_one_channel_is_hot_per_lane():
    for lane in delivery.DELIVERY_LANES:
        block = delivery.delivery_onehot(lane)
        assert sum(block) == 1.0 and set(block) == {0.0, 1.0}


def test_the_lane_order_is_the_wire_format():
    """Appending is a contract change. REORDERING silently reinterprets every checkpoint
    and every filelist ever written, which is worse than appending and is never the right
    edit — so the order is asserted, not merely documented."""
    assert delivery.DELIVERY_LANES == (
        "Dialogue", "Neutral", "Documentary", "Newscaster", "Speech")


def test_an_unrecognised_lane_is_refused_not_silently_unknown():
    """A typo would otherwise become an unconditioned clip that still trains, and the
    corpus balance report would count it as unlabelled — a defect with no symptom."""
    for bad in ("Newscasterr", "dialogue", "Narration", "Speech "):
        if bad.strip() in delivery.DELIVERY_LANES:
            continue
        with pytest.raises(ValueError, match="unknown delivery lane"):
            delivery.delivery_index(bad)


def test_blank_is_the_only_spelling_of_no_label():
    assert delivery.delivery_index("") is None
    assert delivery.delivery_index(None) is None
    assert delivery.delivery_index("   ") is None


def test_a_non_one_hot_block_is_refused():
    """Delivery is CATEGORICAL. Two lanes set means two labels were merged upstream; a
    fractional value means someone interpolated between categories, which has no meaning
    — and is exactly what a mobile host would do if it treated these like V/A/T."""
    with pytest.raises(ValueError, match="not one-hot"):
        delivery.lane_of_vector([0, 0, 0, 1, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="not one-hot"):
        delivery.lane_of_vector([0, 0, 0, 0.5, 0, 0, 0, 0])


def test_a_wrong_width_vector_names_the_disagreement():
    with pytest.raises(ValueError, match="expected 8 channels, got 4"):
        delivery.lane_of_vector([0, 0, 0, 1])


def test_embodiment_has_no_lane_and_the_code_says_why():
    """ARCHITECTURE §1, owner 2026-08-02: an embodiment clip's delivery does not have a
    value, it CHANGES partway through. A sixth label would teach the model that
    embodiment is a uniform manner of speaking, which is the one thing it is not."""
    assert "Embodiment" not in delivery.DELIVERY_LANES
    src = (REPO / "matcha" / "delivery.py").read_text(encoding="utf-8")
    assert "EMBODIMENT IS NOT A LANE" in src


def test_the_vocabulary_has_exactly_one_definition():
    """"Which number means Newscaster" is precisely the rule that must not fork — getting
    it wrong produces fluent, confident, wrongly-delivered audio rather than an error.

    This found a real one: `stage_pool.LANES` listed the same five lanes in a DIFFERENT
    ORDER. Harmless while it only drove reports; not harmless at all now that position is
    the wire format of the one-hot block, because adding a lane to one and not the other
    would silently reinterpret every filelist ever written.

    A full FIVE-lane tuple is the thing being guarded. Subsets (`NARRATION_LANES`) and
    per-lane config dicts are different concepts and are allowed — but the subset was
    itself duplicated between book_ingest and ref_select, and now is not.
    """
    offenders = []
    for path in list((REPO / "matcha").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")):
        if path.name == "delivery.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "DELIVERY_LANES" in stripped:
                continue
            if sum(f'"{ln}"' in stripped for ln in delivery.DELIVERY_LANES) >= 4:
                offenders.append(f"{path.relative_to(REPO)}:{n}: {stripped}")
    assert not offenders, "a second delivery vocabulary:\n" + "\n".join(offenders)


def test_the_narration_subset_also_has_one_definition():
    """Same B-L5 shape one level down: one file gets a new narration lane, the other does
    not, and an engine is directed as narration in one and as dialogue in the next."""
    sys.path.insert(0, str(REPO / "scripts" / "synthesis"))
    book_ingest = pytest.importorskip("book_ingest")
    ref_select = pytest.importorskip("ref_select")
    stage_pool = pytest.importorskip("stage_pool")
    assert ref_select._NARRATION_LANES is book_ingest.NARRATION_LANES
    assert stage_pool.LANES is delivery.DELIVERY_LANES
    assert set(book_ingest.NARRATION_LANES) <= set(delivery.DELIVERY_LANES)


# --- the consumers ---------------------------------------------------------------------


def test_the_cli_names_the_lane_rather_than_asking_for_five_zeros():
    """A delivery block typed by hand is one that will eventually be typed wrong,
    silently, into a channel whose failure mode is a fluent render in the wrong manner."""
    src = (REPO / "matcha" / "cli.py").read_text(encoding="utf-8")
    assert '"--delivery"' in src
    assert "choices=list(delivery.DELIVERY_LANES)" in src
    assert "delivery.vat_vector(" in src


def test_the_vocalizer_ships_a_dial_in_the_same_phase():
    """The standing rule: every new model capability ships with a Vocalizer control in the
    same phase. A capability with no control cannot be vetted, and an unvetted
    conditioning channel is one whose failure mode we learn about from a training run
    instead of from a listen."""
    src = (REPO / "vocalizer.py").read_text(encoding="utf-8")
    assert "delivery_lane = gr.Dropdown(" in src, "no delivery control on the vetting surface"
    assert "delivery.DELIVERY_LANES" in src
    assert "delivery.vat_vector(" in src
    # A dropdown, not a slider: interpolating between Newscaster and Dialogue is meaningless.
    assert "delivery_lane = gr.Slider(" not in src


def test_the_vocalizer_still_renders_a_pre_v2_checkpoint():
    """A 3-channel checkpoint predates delivery and must still render — the trunk's width
    guard would otherwise refuse it with a shape error a listener cannot act on. And the
    info line must SAY the dial did nothing: a control that silently no-ops is how a
    vetting surface produces a confident wrong verdict."""
    src = (REPO / "vocalizer.py").read_text(encoding="utf-8")
    assert "ckpt_vat_dim <= delivery.VAT_BASE_DIM" in src
    assert "delivery IGNORED" in src


def test_the_http_api_rejects_a_typod_lane_as_a_400():
    """Not silently unknown, which would render a neutral clip while the caller believed
    it had asked for Newscaster."""
    src = (REPO / "vocalizer.py").read_text(encoding="utf-8")
    assert "def _delivery_of(body):" in src
    assert "raise ClientError(str(exc))" in src


def test_the_corpus_derivation_joins_a_lane_rather_than_inventing_one():
    """Delivery is corpus metadata from the ear, not a measure — there is no signal in the
    audio to derive it from. Absent a source every clip is unknown, which is CORRECT for
    LibriTTS."""
    src = (REPO / "scripts" / "derive_vat_corpus.py").read_text(encoding="utf-8")
    assert "--delivery-from" in src
    assert "from matcha.delivery import vat_vector" in src
    assert "def _load_delivery(path):" in src


def test_the_export_converter_takes_the_width_from_here():
    """F-H2 closed 2026-08-07. The converter no longer has its own opinion about the
    control surface — a graph and its manifest disagreeing is the one failure a mobile
    host cannot detect, because the manifest is all it can read. Full contract coverage
    is in `tests/test_export_contract.py`."""
    src = (REPO / "scripts" / "litert_export" / "convert_vat.py").read_text(encoding="utf-8")
    assert "VAT_DIM = delivery.VAT_DIM" in src
    assert "from matcha import delivery" in src
    assert "DO NOT INTERPOLATE" in src


def test_the_width_is_not_the_vad_octants():
    """Asked on the day the width landed, which is how we know the inference is easy.

    The 8 VAD octants are the sign combinations of a 3-axis space (2³ = 8 regions of
    ±valence/±arousal/±dominance) — a partition of ONE continuous space. Ours is 3 + 5:
    three continuous channels and a categorical block, with no arithmetic relating them.
    The collision is a coincidence, and a trap, because our continuous triple genuinely is
    VAD-adjacent. Guarded here so the reasoning survives the next reader of `vat_dim: 8`.
    """
    assert delivery.VAT_DIM == delivery.VAT_BASE_DIM + delivery.DELIVERY_DIM
    assert delivery.VAT_DIM != 2 ** delivery.VAT_BASE_DIM or delivery.DELIVERY_DIM == 5
    src = (REPO / "matcha" / "delivery.py").read_text(encoding="utf-8")
    assert "NOT THE EIGHT VAD OCTANTS" in src
    # And the third axis is tension, not dominance — rescoped LAX/TIGHT 2026-07-20.
    assert "tension" in src and "not dominance" in src

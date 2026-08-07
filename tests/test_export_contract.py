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


# --- F-M1 / F-M5: the referee ---------------------------------------------------------

REFEREE = REPO / "scripts" / "export_fidelity_referee.py"
REF_SRC = REFEREE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def referee():
    """Import the referee with onnxruntime / litert stubbed — neither is on the host venv
    and neither is needed to exercise the binder or the metrics."""
    import importlib.util
    import types

    for mod, attrs in [("onnxruntime", {"InferenceSession": object}),
                       ("ai_edge_litert", {}),
                       ("ai_edge_litert.interpreter", {"Interpreter": object})]:
        m = types.ModuleType(mod)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(mod, m)
    spec = importlib.util.spec_from_file_location("_referee", REFEREE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conditioned_graph():
    import numpy as np
    return [
        {"name": "serving_default_x:0", "index": 0, "dtype": np.int64, "shape": [1, 50]},
        {"name": "serving_default_x_lengths:0", "index": 1, "dtype": np.int64, "shape": [1]},
        {"name": "serving_default_scales:0", "index": 2, "dtype": np.float32, "shape": [2]},
        {"name": "serving_default_spks:0", "index": 3, "dtype": np.int64, "shape": [1]},
        {"name": "serving_default_vat:0", "index": 4, "dtype": np.float32, "shape": [1, 8]},
    ]


def test_the_referee_can_bind_a_conditioned_graph(referee):
    """F-M1's headline: it could not score the conditioned lane AT ALL."""
    feed = referee.build_inputs(referee.DEFAULT_IDS, 50, 0.0, 1.0, 245,
                                [0.4, -0.2, 0.1, 0, 0, 0, 1, 0])
    bound = referee._match(_conditioned_graph(), feed, "tflite")
    assert bound == {0: "x", 1: "x_lengths", 2: "scales", 3: "spks", 4: "vat"}


def test_the_old_dtype_heuristic_bound_the_speaker_silently_wrong():
    """The reason this was a defect and not an inconvenience.

    The heuristic was "int + 2-D means tokens, int means lengths, anything else means
    scales". On a conditioned graph `spks` is int64 shape (1,) — IDENTICAL to
    `x_lengths` — so it received the token count as a speaker id. No shape error, no
    exception: the referee just compared a render of the wrong speaker and reported a
    fidelity number for it. (`vat` collided with `scales` too, but those shapes differ,
    so that half at least raised.)
    """
    import numpy as np

    old = {}
    for d in _conditioned_graph():
        if d["dtype"] in (np.int64, np.int32) and len(d["shape"]) == 2:
            old[d["index"]] = "x"
        elif d["dtype"] in (np.int64, np.int32):
            old[d["index"]] = "x_lengths"
        else:
            old[d["index"]] = "scales"
    assert old[3] == "x_lengths", "spks would have been bound to the token count"
    assert old[4] == "scales"


def test_an_unrecognised_input_is_refused_not_guessed(referee):
    """Guessing is what this tool exists to stop."""
    feed = referee.build_inputs(referee.DEFAULT_IDS, 50, 0.0, 1.0, None, None)
    import numpy as np
    details = _conditioned_graph()[:3] + [
        {"name": "mystery", "index": 9, "dtype": np.float32, "shape": [1, 8]}]
    with pytest.raises(SystemExit, match="cannot bind"):
        referee._match(details, feed, "tflite")


def test_supplying_conditioning_to_an_unconditioned_graph_is_refused(referee):
    feed = referee.build_inputs(referee.DEFAULT_IDS, 50, 0.0, 1.0, 245, None)
    with pytest.raises(SystemExit, match="no such input"):
        referee._match(_conditioned_graph()[:3], feed, "tflite")


def test_a_half_gain_error_passes_cosine_and_fails_the_new_gates(referee):
    """F-M5, demonstrated rather than asserted.

    `tflite = 0.5 * onnx` — every sample exactly half, which is what a mis-scaled
    dequantization or a dropped factor of two looks like. On a model whose flagship axis
    is a LOUDNESS dial this is the worst possible blind spot: on-device it reads as "the
    energy channel is weak", not as a broken export.
    """
    import numpy as np

    ref = np.sin(np.linspace(0, 400, 4000)) * (0.3 + 0.2 * np.sin(np.linspace(0, 9, 4000)))
    bad = 0.5 * ref

    cos, _n = referee.cosine(ref, bad)
    assert cos == pytest.approx(1.0, abs=1e-9), "cosine is scale-invariant, as expected"
    assert referee.gain_error_db(ref, bad) == pytest.approx(-6.02, abs=0.02)
    assert referee.nrmse(ref, bad) == pytest.approx(0.5, abs=1e-6)
    # and the defaults catch it
    assert abs(referee.gain_error_db(ref, bad)) > 0.5
    assert referee.nrmse(ref, bad) > 0.15


def test_an_honest_render_passes_all_three(referee):
    import numpy as np

    rng = np.random.default_rng(0)
    ref = np.sin(np.linspace(0, 400, 4000)) * 0.3
    close = ref + rng.normal(0, 0.002, ref.shape)      # fp16-ish noise
    cos, _ = referee.cosine(ref, close)
    assert cos >= 0.99
    assert abs(referee.gain_error_db(ref, close)) <= 0.5
    assert referee.nrmse(ref, close) <= 0.15


def test_the_referee_gates_on_all_three():
    """RMSE used to be computed, printed, and thrown away; the exit code read cosine
    alone."""
    tail = REF_SRC[REF_SRC.index("if args.temperature == 0.0:"):]
    assert '("gain", abs(gain) <= args.max_gain_db' in tail
    assert '("nrmse", err <= args.max_nrmse' in tail
    assert "ok = all(c[1] for c in checks)" in tail


def test_the_referee_shares_the_delivery_encoding():
    """A referee with its own idea of what a lane means would certify the wrong vector."""
    assert "from matcha import delivery" in REF_SRC
    assert "delivery.vat_vector(" in REF_SRC


def test_the_converter_gained_the_same_scale_sensitive_gates():
    """G3 was Pearson-only for the same reason and with the same blind spot."""
    assert "G3b e2e gain parity" in SRC
    assert "G3c e2e sample-wise error" in SRC
    assert "def gain_error_db(" in SRC and "def nrmse(" in SRC


# --- F-M2: rename_tflite_tensors -------------------------------------------------------

RENAMER = REPO / "scripts" / "rename_tflite_tensors.py"


@pytest.fixture(scope="module")
def renamer():
    """Pure planning logic, split out so it is testable without flatbuffers or the
    onnx2tf-generated schema module — neither of which is on the host venv."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_renamer", RENAMER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FLOAT32, INT32, INT64 = 0, 2, 4


def _t(index, name, shape, typ):
    return {"index": index, "name": name, "shape": shape, "type": typ}


def _healthy():
    return [
        _t(0, "serving_default_x:0", [1, 50], INT64),
        _t(1, "serving_default_x_lengths:0", [1], INT64),
        _t(2, "serving_default_scales:0", [2], FLOAT32),
        _t(8, "StatefulPartitionedCall:0", [1, 120000], FLOAT32),
        _t(9, "StatefulPartitionedCall:1", [1], INT32),
    ]


def test_a_healthy_graph_renames_cleanly(renamer):
    plan, problems = renamer.plan_renames(_healthy(), {0, 1, 2, 8, 9},
                                          renamer.DEFAULT_RENAMES)
    assert problems == []
    assert plan == {0: "x", 1: "x_lengths", 2: "scales", 8: "wav", 9: "wav_lengths"}


def test_swapped_outputs_are_refused_not_renamed(renamer):
    """THE emission-order bug. `StatefulPartitionedCall:0 -> wav` assumed TF emits the
    waveform first; nothing guarantees it. Swap them and the old code renamed the LENGTHS
    tensor `wav` — and Prosodia matches by name, so it would read a one-element int tensor
    as audio. Checked against shape and dtype now: a waveform is a large float tensor, a
    length is a tiny integer one, and nothing else in the graph is close."""
    swapped = _healthy()[:3] + [
        _t(8, "StatefulPartitionedCall:0", [1], INT32),
        _t(9, "StatefulPartitionedCall:1", [1, 120000], FLOAT32),
    ]
    plan, problems = renamer.plan_renames(swapped, {0, 1, 2, 8, 9},
                                          renamer.DEFAULT_RENAMES)
    assert "wav" not in plan.values() and "wav_lengths" not in plan.values()
    assert len(problems) == 2
    assert any("EMISSION-ORDER" in p for p in problems)


def test_renaming_nothing_is_a_failure_not_a_success(renamer):
    """`renamed 0 tensors` exited 0, so a graph this table does not recognise — a TF
    upgrade is enough — was copied through untouched and called a success. The failure
    then surfaced in Prosodia as a missing input, several steps from the cause."""
    plan, problems = renamer.plan_renames(
        [_t(0, "args_0", [1, 50], INT64)], {0}, renamer.DEFAULT_RENAMES)
    assert plan == {} and problems == []
    src = RENAMER.read_text(encoding="utf-8")
    assert "if not renamed:" in src
    assert "renamed 0 tensors" in src and "raise SystemExit(" in src


def test_an_intermediate_tensor_is_not_renamed(renamer):
    """Only graph I/O carries the contract; renaming an intermediate that happens to share
    a name is at best noise and at worst a second tensor answering to `x`."""
    plan, problems = renamer.plan_renames(
        [_t(5, "serving_default_x:0", [1, 50], INT64)], set(), renamer.DEFAULT_RENAMES)
    assert plan == {}
    assert "not a graph input or output" in problems[0]


def test_a_waveform_too_small_to_be_one_is_refused(renamer):
    """The dtype check alone would pass a float tensor of two elements as `wav`."""
    tiny = _healthy()[:3] + [_t(8, "StatefulPartitionedCall:0", [1, 2], FLOAT32)]
    _plan, problems = renamer.plan_renames(tiny, {0, 1, 2, 8}, renamer.DEFAULT_RENAMES)
    assert any("at least" in p for p in problems)


def test_two_tensors_cannot_claim_the_same_contract_name(renamer):
    dup = _healthy() + [_t(10, "serving_default_x:0", [1, 50], INT64)]
    _plan, problems = renamer.plan_renames(dup, {0, 1, 2, 8, 9, 10},
                                           renamer.DEFAULT_RENAMES)
    assert any("claimed twice" in p for p in problems)


def test_the_conditioned_inputs_are_in_the_table(renamer):
    """The table predated contract v2, so a conditioned export's `spks` and `vat` kept
    their mangled names and Prosodia could not bind them."""
    assert renamer.DEFAULT_RENAMES["serving_default_spks:0"] == "spks"
    assert renamer.DEFAULT_RENAMES["serving_default_vat:0"] == "vat"


def test_an_absent_optional_rename_needs_an_explicit_flag(renamer):
    """An unconditioned export legitimately has no spks/vat. That is `--allow-missing`,
    not a silent pass."""
    plan, _ = renamer.plan_renames(_healthy(), {0, 1, 2, 8, 9}, renamer.DEFAULT_RENAMES)
    present = {t["name"] for t in _healthy()}
    missing, _absent = renamer.check_complete(plan, renamer.DEFAULT_RENAMES, present)
    assert missing == [], "everything present was renamed"
    src = RENAMER.read_text(encoding="utf-8")
    assert "--allow-missing" in src


# --- F-M3 / F-M6: the Kotlin replica --------------------------------------------------

REPLICA = REPO / "scripts" / "litert_export" / "kotlin_replica.py"
REPLICA_SRC = REPLICA.read_text(encoding="utf-8")


def test_the_replica_looks_in_both_places_its_artifacts_live():
    """F-M3. It read everything out of `artifacts/` and died at MODULE SCOPE on a bare
    FileNotFoundError for the first missing one. THREE of its seven prerequisites are not
    conversion outputs at all — the G2P graph, its meta and the 275k dictionary are
    VENDORED assets under /data/models/litert-community/Matcha-TTS. The script was
    unrunnable as written and the error named neither the reason nor the real directory.
    """
    assert "SONORA_LITERT_ASSETS" in REPLICA_SRC
    assert "def _preflight()" in REPLICA_SRC
    assert "def _find(" in REPLICA_SRC


def test_the_preflight_names_every_missing_file_and_its_producer():
    """One FileNotFoundError names one file. A preflight names all of them, and where each
    comes from — which is the actual question when three come from somewhere else."""
    block = REPLICA_SRC[REPLICA_SRC.index("def _preflight()"):]
    block = block[:block.index("_PATHS = _preflight()")]
    assert "convert_g2p_matcha.py" in block
    assert "vendored asset" in block
    assert "is missing" in block and "prerequisites" in block


def test_the_dictionary_may_be_gzipped():
    """The vendored dictionary ships as `g2p_dict.txt.gz`; the original opened the bare
    name, so even pointed at the right directory it would have failed."""
    assert "import gzip" in REPLICA_SRC
    assert 'name + ".gz"' in REPLICA_SRC


def test_the_replica_can_drive_the_conditioned_lane():
    """F-M6. It pointed at the Phase 0 LJSpeech graphs (22.05 kHz, unconditioned) and had
    no way to supply spk or vat — so it validated the Kotlin port against a model that is
    not the one shipping. The port exists FOR the conditioned actor."""
    assert "def conditioning(" in REPLICA_SRC
    assert '"vat":    dict(art="artifacts_vat"' in REPLICA_SRC
    assert "spk_vec=None, vat_vec=None" in REPLICA_SRC
    # and the conditioning must actually reach both graphs
    assert "te_args += [spk_vec, _tok(vat_vec, MAX_TEXT, tmask)]" in REPLICA_SRC
    assert "dec_args += [spk_vec, vat_y]" in REPLICA_SRC


def test_the_replica_reads_the_control_contract_rather_than_assuming_it():
    """A replica with its own idea of the bounds proves nothing about the host, which
    reads config.json. It must consume the same manifest."""
    assert 'CONTROL = _config.get("control") or {}' in REPLICA_SRC
    assert 'CONTROL.get("continuous")' in REPLICA_SRC
    assert 'CONTROL.get("categorical")' in REPLICA_SRC


def test_the_replica_hardcodes_no_mel_stats_for_the_conditioned_lane():
    """MEL_MEAN/MEL_STD were module constants that happened to match Phase 0. They are a
    property of the corpus, so the conditioned lane's differ — and a wrong denormalization
    is a quiet, whole-clip loudness and timbre error."""
    assert 'MEL_MEAN = _L["mel_mean"] if _L["mel_mean"] is not None else _config["mel_mean"]' \
        in REPLICA_SRC


def test_a_delivery_lane_against_pre_v2_artifacts_is_refused():
    """The artifacts on /data predate contract v2. Silently ignoring `--delivery` there
    would render neutral audio while the caller believed it had asked for Newscaster —
    the exact failure the control contract exists to close."""
    assert "predate contract v2" in REPLICA_SRC
    assert "Re-export." in REPLICA_SRC or "Re-export" in REPLICA_SRC

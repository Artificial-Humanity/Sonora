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

import ast
import csv
import json
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


def test_no_narration_lane_is_retired():
    """A narration lane is one new lines are DIRECTED INTO, so it must be assignable.

    Two live rules read `NARRATION_LANES` and neither can mean anything for a retired
    lane: `ref_select.MAX_SHARE_NARRATION` caps a lane's per-engine share, and
    `book_ingest.build_direction` forces zonos's emotion vector off and caps chatterbox's
    exaggeration on one. Membership would be a rule that can only fire on a row nothing
    may write — and, read the other way, a lane still counted as narration is a lane a
    campaign builder may still be sizing renders for.

    `matcha.delivery` derives the tuple by filtering `RETIRED_LANES` rather than restating
    the names, so retiring a narration lane later cannot leave a stale member behind. This
    asserts the property, not the spelling.
    """
    assert not set(delivery.NARRATION_LANES) & set(delivery.RETIRED_LANES)
    assert set(delivery.NARRATION_LANES) <= set(delivery.ACTIVE_DELIVERY_LANES)


# --- the consumers ---------------------------------------------------------------------


def _one_clip_campaign(tmp_path, lane):
    """A rendered one-clip bank whose manifest claims `lane`, and an empty ratings.csv.

    The CSV carries the app's real header on purpose: `_build_row` filters to the header
    actually on disk, so the `delivery` column only survives if the sheet has one.
    """
    audio = tmp_path / "campaign-x" / "audio"
    audio.mkdir(parents=True)
    (audio / "clip.wav").write_bytes(b"RIFF")          # only is_file() is checked
    (audio / "bank_manifest.jsonl").write_text(json.dumps({
        "id": "lane-0001", "wav": "clip.wav", "engine": "qwen", "campaign": "campaign-x",
        "register": "neutral_narration", "intended_delivery": lane}) + "\n",
        encoding="utf-8")
    ratings = tmp_path / "ratings"
    ratings.mkdir(parents=True)
    (ratings / "ratings.csv").write_text(
        "campaign,id,engine,register,gender,delivery,score,note,status,link\n",
        encoding="utf-8")
    return audio, ratings / "ratings.csv"


def _register_audition(tmp_path, monkeypatch, lane):
    """-> (run, rows) for the synthesis pipeline's ratings.csv writer.

    Imported inside the test rather than at module scope because it resolves DATA_ROOT and
    RATINGS_DIR from the environment at IMPORT time.
    """
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(tmp_path / "ratings"))
    monkeypatch.setenv("BOOK_PROSE_ROOT", str(tmp_path / "book-prose"))
    sys.path.insert(0, str(REPO / "scripts" / "synthesis"))
    sys.modules.pop("register_audition", None)
    mod = pytest.importorskip("register_audition")
    audio, csv_path = _one_clip_campaign(tmp_path, lane)
    monkeypatch.setattr(sys, "argv", ["register_audition", "--audio-dir", str(audio)])

    def rows():
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    return mod.main, rows


@pytest.mark.parametrize("lane", delivery.RETIRED_LANES)
def test_the_synthesis_writer_refuses_a_retired_lane(tmp_path, monkeypatch, lane):
    """The retirement has to hold where new rows are actually MADE (issue #12).

    `register_audition` is the synthesis pipeline's write site for `delivery`: it turns a
    rendered bank's manifests into `unaudited` ratings.csv rows, and it copied
    `intended_delivery` through verbatim. `check_assignable` was called only by the corpus
    builder, several days downstream — so a stale bank could queue a retired lane, the
    auditor could certify it by ear, and the merge would refuse the row afterwards with
    the render already bought and a verdict attached to a label that cannot be kept.

    Nothing may be appended when it raises. `append_guarded` runs only after every row of
    every target is built, so the refusal costs the run and not the sheet.
    """
    run, rows = _register_audition(tmp_path, monkeypatch, lane)
    with pytest.raises(ValueError, match="RETIRED"):
        run()
    assert rows() == [], (
        f"a {lane!r} row reached ratings.csv — the writer refused and appended anyway")


@pytest.mark.parametrize("lane", ["Newscaster", "Neutral", ""])
def test_the_synthesis_writer_still_pre_fills_an_assignable_lane(tmp_path, monkeypatch,
                                                                 lane):
    """The guard means something only if it passes what it should.

    A blank is the commonest case by far — most banks state no lane at all — and blank is
    the spelling of `unknown`, not an error. Refusing it, or blanking a live lane to make
    the guard "safe", would break the verify-don't-enter standard the column exists for.
    """
    run, rows = _register_audition(tmp_path, monkeypatch, lane)
    run()
    assert [(r["id"], r["delivery"]) for r in rows()] == [("lane-0001", lane)]


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
    instead of from a listen.

    THE GENERIC HALF OF THIS TEST HAS MOVED. That a dial exists, and that its shape does
    not misrepresent the dimension, is now checked for EVERY declared dimension by
    `tests/test_direction_surface.py`. This version asserted one widget by name, so it
    could never fail for a channel it did not mention — which is exactly how the speaker
    dimension went two corpus generations silently clamped while this test passed.

    What stays here is the part that is specific to delivery: that the Vocalizer takes the
    vocabulary and the encoding FROM `matcha.delivery` rather than spelling either out
    itself. A dial that agreed with nothing would still satisfy the generic check.
    """
    direction = pytest.importorskip("matcha.direction")
    # Registered, therefore covered by the generic parity check rather than only here.
    assert direction.dimension("delivery_lane").kind == "categorical"
    src = (REPO / "vocalizer.py").read_text(encoding="utf-8")
    assert "delivery.DELIVERY_LANES" in src, "the dial builds its own lane list"
    assert "delivery.vat_vector(" in src, "the dial builds its own conditioning vector"


#: `vocalizer.py` cannot be imported here — it pulls in torch and gradio, neither of which
#: the host has — so the dial is guarded by reading it. As the AST, NOT as text: the two
#: findings this file has already lost to text matching (#17, #27) were both assertions
#: that stayed green on a line that no longer said what they were checking. A parse tree
#: cannot be satisfied by a comment, a docstring, or a substring that happens to survive.
def _vocalizer_ast():
    return ast.parse((REPO / "vocalizer.py").read_text(encoding="utf-8"))


def _assigned_call(tree, target, attr):
    """The `target = <mod>.attr(...)` call node, or None."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(isinstance(t, ast.Name) and t.id == target for t in node.targets):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == attr:
            return node.value
    return None


def test_the_delivery_dials_choices_are_flat_strings():
    """The bug that reached the owner mid-ep010: `(label, value)` tuple choices are a
    Gradio 4 feature, the container runs 3.43.2, and there the tuple ITSELF becomes the
    value — so every one of the five lanes arrived at `delivery_index` as
    `"('Newscaster', 'Newscaster')"` and the closed vocabulary refused it. The dial was
    unusable for every lane while `/v1/audio/speech`, which passes plain strings, was fine.

    Nothing pinned that. Restoring the tuple-choices line verbatim left the whole suite
    green, because the existing guards ask whether `delivery.DELIVERY_LANES` is mentioned
    and both forms mention it. This asks the question that actually distinguishes them:
    is there a tuple anywhere in the `choices=` expression?
    """
    call = _assigned_call(_vocalizer_ast(), "delivery_lane", "Dropdown")
    assert call is not None, "the delivery dial is no longer a `gr.Dropdown` assignment"
    choices = [kw.value for kw in call.keywords if kw.arg == "choices"]
    assert len(choices) == 1, "the dial's choices are not a single `choices=` keyword"
    offenders = [ast.unparse(n) for n in ast.walk(choices[0]) if isinstance(n, ast.Tuple)]
    assert not offenders, (
        "tuple choices are back in the delivery dial and Gradio 3 will pass the tuple "
        "through as the value: " + ", ".join(offenders))
    built = ast.unparse(choices[0])
    assert "delivery.DELIVERY_LANES" in built, "the dial builds its own lane list"
    assert "UNKNOWN_UI" in built, "the dial spells the unknown label itself"


def test_on_synth_maps_the_display_sentinel_back_to_the_contracts_unknown():
    """`DELIVERY_UNKNOWN` is `""`, which a dropdown can neither display nor let you
    re-select, so the picker shows `UNKNOWN_UI` and `on_synth` maps it back. Delete that
    mapping and the DEFAULT selection raises `unknown delivery lane 'unknown'` on every
    render — the dial broken again, from the other end. That mutation also left the suite
    green before this test."""
    fn = next((n for n in ast.walk(_vocalizer_ast())
               if isinstance(n, ast.FunctionDef) and n.name == "on_synth"), None)
    assert fn is not None, "`on_synth` is gone; the mapping has nowhere to live"
    mapped = any(
        "UNKNOWN_UI" in ast.unparse(node.test)
        and any("delivery.DELIVERY_UNKNOWN" in ast.unparse(s) for s in node.body)
        for node in ast.walk(fn) if isinstance(node, ast.If))
    assert mapped, ("`on_synth` no longer maps UNKNOWN_UI back to "
                    "delivery.DELIVERY_UNKNOWN; the default selection now raises")


def test_the_uis_unknown_label_is_spelled_once_and_is_not_a_real_lane():
    """Two halves of the same hazard, both latent on today's vocabulary.

    The label must not collide with a lane. `on_synth` rewrites `d == UNKNOWN_UI` to `""`,
    so a lane that ever spells itself `"unknown"` would be silently downgraded to an
    UNCONDITIONED render — the precise failure `delivery_index` was written to make loud by
    refusing typos. That vocabulary change would be made in `matcha/delivery.py`, by
    someone with no reason to open `vocalizer.py`; this is the line that would tell them.

    And it must be spelled once. The info line under the audio reports the same state, and
    its job is to name the lane that ACTUALLY ran. Spelled from a second literal, renaming
    the label to something clearer leaves the picker saying `— none —` while the readout
    says `delivery=unknown`, with nothing telling a listener they are the same state.
    """
    tree = _vocalizer_ast()
    assign = next((n for n in tree.body
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "UNKNOWN_UI"
                           for t in n.targets)), None)
    assert assign is not None, "UNKNOWN_UI is no longer a module-level constant"
    label = ast.literal_eval(assign.value)

    assert label not in delivery.DELIVERY_LANES, (
        f"the UI's unknown label {label!r} is now a real delivery lane; on_synth would "
        "rewrite that lane to DELIVERY_UNKNOWN and render it unconditioned")
    assert label != delivery.DELIVERY_UNKNOWN, "the sentinel is the thing it stands in for"

    echoes = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and n.value == label
              and n.lineno != assign.value.lineno]
    assert not echoes, (f"{label!r} is spelled again outside UNKNOWN_UI at line(s) "
                        f"{echoes}; use the constant so the picker and the info line "
                        "cannot drift apart")


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


def test_the_open_emotion_block_decision_is_recorded_where_it_would_be_made():
    """An open decision that lives only in a conversation is an omission with a good
    story. The 3-continuous + 8-categorical hybrid is standard in emotional TTS and we
    deliberately do not have one; the reasoning and its gate belong in the three places
    someone would actually look — the contract canon, the open-work list, and the module
    they would edit to add channels."""
    arch = (REPO / "notes" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "categorical EMOTION block is an open question" in arch
    assert "gated on Phase 1" in arch

    todo = (REPO / "notes" / "todo.md").read_text(encoding="utf-8")
    assert "The categorical emotion block — OPEN DECISION" in todo
    assert "Gate on Phase 1" in todo

    src = (REPO / "matcha" / "delivery.py").read_text(encoding="utf-8")
    assert "THIS IS NOT AN EMOTION TAXONOMY" in src
    assert "APPENDS as channels 8+" in src


def test_the_recorded_gate_names_the_evidence_it_waits_on():
    """The decision is not "maybe later" — it is gated on a specific measurement. Valence
    FAILED its standing test, and that failure is currently diagnosed as a corpus-label
    limit rather than an architectural one; Phase 1 is the experiment that separates them.
    Spiking before that read would confound a data problem with a representation one."""
    todo = " ".join((REPO / "notes" / "todo.md").read_text(encoding="utf-8").split())
    assert "corpus-label limit, not architectural" in todo
    assert "sob and a laugh have similar energy" in todo
    assert "1,189 labelled keeps; an 8-way emotion block starts at zero" in todo


# --- TR-L3: the loader checked the width and not the shape ----------------------------


def test_the_loader_refuses_an_interpolated_delivery_block():
    """`lane_of_vector`'s one-hot refusal was called only by READERS — the CLI and the
    manifest tools — never by the training loader. So a future writer emitting
    `...,0.5,0.5,0,0,0` would train silently as an interpolated category: the meaning-error
    `matcha/delivery.py`'s own docstring says has no meaning.

    Derive and merge both write `int(x)` and the shipped v5 was verified 0/1 on disk, so
    this is a seam guard rather than a live fix — which is exactly when it is cheap to add.
    """
    with pytest.raises(ValueError, match="not one-hot"):
        delivery.lane_of_vector([0.1, 0.2, 0.3, 0.5, 0.5, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not one-hot"):
        delivery.lane_of_vector([0.1, 0.2, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0])


def test_a_valid_block_and_the_unknown_block_still_pass():
    """The all-zero block is `unknown`, not a violation — it is the conditioning every
    contract-v1 checkpoint trained under, and refusing it would refuse the corpus."""
    assert delivery.lane_of_vector([0.1, 0.2, 0.3, 0, 0, 0, 0, 0]) == delivery.DELIVERY_UNKNOWN
    for i, lane in enumerate(delivery.DELIVERY_LANES):
        block = [0.0] * len(delivery.DELIVERY_LANES)
        block[i] = 1.0
        assert delivery.lane_of_vector([0.1, 0.2, 0.3, *block]) == lane


def test_the_training_loader_calls_it():
    """Torch-only, so the behaviour cannot run on the host — this notices if the call is
    removed. The width check alone passed an interpolated block for as long as the
    delivery channel has existed."""
    src = (REPO / "matcha" / "data" / "text_mel_datamodule.py").read_text(encoding="utf-8")
    assert "lane_of_vector(vat.tolist())" in src
    assert src.index("vat.numel() != self.vat_dim") < src.index("lane_of_vector(vat.tolist())"), \
        "width must be checked first, or the one-hot check reads a mis-sized block"


# --- TR-L5: the default checkpoint cadence -------------------------------------------


def test_the_checkpoint_cadence_matches_how_the_lineage_is_measured():
    """TR-L5. `every_n_epochs: 100` outlived the evidence: v5 delivered 100% of its gain by
    epoch 9, so a default-config run retained only `last.ckpt` and the per-epoch
    checkpoints the holdout methodology reads did not exist unless someone remembered a
    launch-time override. The v5 run clearly used one; where it lived was recorded nowhere.
    """
    cfg = (pathlib.Path(__file__).resolve().parent.parent
           / "configs" / "callbacks" / "model_checkpoint.yaml").read_text(encoding="utf-8")
    assert "every_n_epochs: 1 " in cfg or "every_n_epochs: 1\n" in cfg
    assert "save_top_k: 10" in cfg, "the retention bound is what keeps this free"

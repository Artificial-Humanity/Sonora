"""Regression coverage for the §6 synthesis-campaign fixes (2026-08-07).

Two shapes recur here and both are invisible at render time:

* **A knob that is recorded but not read.** `BRIGHT_REF_POLICY` was written into every
  chatterbox manifest row while nothing consulted it, and the value written did not
  describe what the code did — so the provenance was not merely incomplete, it was wrong.
* **A builder that forks the engine contract.** AGENTS.md makes `build_direction()` the
  single source of truth for what each renderer receives, "never bypass it".
  `make_bulk_bank.py` bypassed it, and paid for it in three places at once.

Source- and logic-level, torch-free: every renderer here imports a GPU stack at module
scope, so nothing in this file imports one.
"""

import json
import os
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTH = REPO / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))


def _src(name):
    return (SYNTH / name).read_text(encoding="utf-8")


# --- B-M9: the bright-reference policy ------------------------------------------------


def test_the_recorded_policy_describes_what_the_code_does():
    """It read "exclude" and was stamped into every manifest row, while the code ran a
    hybrid nothing had a name for: `select_reference(max_excursion=…)` drops references
    above 240 Hz excursion (that half IS exclude) and the damping block then clamps
    exaggeration for anything at or above 80% of it. The manifests were WRONG, not merely
    uninformative — and a provenance field nobody can trust is worse than none."""
    src = _src("ref_select.py")
    m = re.search(r'^BRIGHT_REF_POLICY = "([a-z+]+)"', src, re.M)
    assert m, "BRIGHT_REF_POLICY is gone"
    assert m.group(1) in ("exclude", "damp", "exclude+damp")
    # The default must remain what has actually been running: every existing chatterbox
    # clip was rendered under it, and silently changing the rendering while fixing a
    # provenance bug would be its own defect.
    assert m.group(1) == "exclude+damp"


def test_the_knob_is_read_not_just_recorded():
    """Defined in `ref_select` beside MAX_REF_EXCURSION (B-L5's reason: casting policy
    learned from heard failures, and a copy per renderer is how the duplicated constant
    happened the first time); CALLED by the renderer, which is the half that was missing.
    """
    policy = _src("ref_select.py")
    assert "def bright_ref_selection_ceiling(" in policy
    assert "def bright_ref_exaggeration(" in policy

    renderer = _src("synth_chatterbox.py")
    assert "max_excursion=bright_ref_selection_ceiling()" in renderer
    assert 'bright_ref_exaggeration(d["exaggeration"], ref_exc)' in renderer
    # and it must not have grown a second copy on the way back
    assert "BRIGHT_REF_POLICY = " not in renderer


@pytest.mark.parametrize(("policy", "excursion", "expected", "damped"), [
    # "exclude": the ceiling is applied at SELECTION, so nothing is damped afterwards.
    ("exclude", 200.0, 0.5, False),
    ("exclude", 100.0, 0.5, False),
    # "damp": the whole pool is castable, bright references are clamped instead.
    ("damp", 200.0, 0.3, True),      # 200 >= 240 * 0.8
    ("damp", 100.0, 0.5, False),
    # the hybrid, which is what has been running
    ("exclude+damp", 200.0, 0.3, True),
    ("exclude+damp", 100.0, 0.5, False),
])
def test_each_policy_does_what_its_name_says(policy, excursion, expected, damped):
    cbx = pytest.importorskip("ref_select")
    assert cbx.bright_ref_exaggeration(0.5, excursion, policy) == (expected, damped)


def test_a_pinned_reference_is_never_damped():
    """Pinning exists so a probe can render deliberately outside the guards, and a probe
    whose parameters are silently rewritten measures nothing."""
    cbx = pytest.importorskip("ref_select")
    assert cbx.bright_ref_exaggeration(0.5, None, "exclude+damp") == (0.5, False)


def test_selection_ceiling_matches_the_policy():
    cbx = pytest.importorskip("ref_select")
    assert cbx.bright_ref_selection_ceiling("damp") is None
    assert cbx.bright_ref_selection_ceiling("exclude") == cbx.MAX_REF_EXCURSION
    assert cbx.bright_ref_selection_ceiling("exclude+damp") == cbx.MAX_REF_EXCURSION


def test_whether_damping_bit_is_recorded_per_clip():
    """The policy alone does not say whether it fired on THIS clip, and "was this render
    damped" is the question an audit of the split-artifact finding actually asks."""
    assert '"bright_ref_damped": damped' in _src("synth_chatterbox.py")


# --- B-L2: a casting failure must not kill the rest of the bank -----------------------


@pytest.mark.parametrize("renderer", ["synth_chatterbox.py", "synth_zonos.py",
                                      "synth_vibevoice.py"])
def test_casting_is_wrapped_per_clip(renderer):
    """Uncaught, a `LookupError` from an exhausted or over-filtered pool killed the
    engine's whole REMAINING bank from that clip on — the same shape as the moss_vg
    split_with_sizes crash that orphaned 7 clips on 2026-07-30. That renderer's per-clip
    try/except is the pattern; these three lacked it."""
    src = _src(renderer)
    block = src[src.index("select_reference("):]
    assert "except (LookupError, ValueError)" in src, f"{renderer}: casting is unguarded"
    assert "CASTING FAILED" in block


# --- B-L4: vibevoice forked the casting contract --------------------------------------


def test_vibevoice_applies_the_same_casting_guards_as_everyone_else():
    """It called `select_reference(design, intended, used)` and nothing else — no
    `max_excursion`, no `exclude=REF_BLACKLIST`, no `ref_id` pinning. Both guards encode
    MEASURED findings, and this renderer opted out of both by forking the call. VibeVoice
    is SET_ASIDE, which is exactly why it is a reinstatement trap."""
    src = _src("synth_vibevoice.py")
    assert "max_excursion=MAX_REF_EXCURSION" in src
    assert "exclude=REF_BLACKLIST" in src
    assert "pinned_reference(job[\"ref_id\"])" in src


# --- B-L3: qwen loaded the model before checking for work -----------------------------


def test_qwen_checks_for_jobs_before_loading_the_model():
    src = _src("synth_qwen.py")
    jobs_at = src.index('jobs = [l for l in bank["lines"]')
    guard_at = src.index('if not jobs:')
    load_at = src.index("Qwen3TTSModel.from_pretrained")
    assert jobs_at < guard_at < load_at, "the shard load still precedes the zero-jobs check"


# --- B-L6: a multi-message decode clobbered the wav -----------------------------------


@pytest.mark.parametrize("renderer", ["synth_moss_vg.py", "synth_moss85.py"])
def test_only_one_message_becomes_one_clip(renderer):
    """These looped over EVERY decoded message writing the same filename, so a
    multi-message decode overwrote the wav once per message AND appended one manifest row
    per message — all claiming the same id, all describing a file that by then held only
    the last message's audio. Nothing downstream can see it: the wav is valid, the rows
    are well-formed, and the id is the join key everything else trusts."""
    src = _src(renderer)
    assert "decoded[:1]" in src, f"{renderer} still writes every decoded message"
    assert "keeping the first" in src, f"{renderer} drops extra messages silently"


# --- B-M6: the manifest must carry the bank line's passthrough fields -----------------


@pytest.mark.parametrize("renderer", ["synth_dia.py", "synth_moss85.py"])
def test_the_manifest_row_starts_from_the_bank_line(renderer):
    """Naming keys explicitly dropped every passthrough field — intended_delivery, book,
    ref_id, source_ref, chunk_type — which `register_audition`'s prefill and the corpus
    fold read from the manifest. The other six renderers were moved to `dict(job)` first;
    these two were missed, and being SET_ASIDE is why nobody noticed."""
    src = _src(renderer)
    assert "row = dict(job)" in src, f"{renderer} still names its manifest keys"
    assert "mf.write(json.dumps(row)" in src


# --- B-M7: make_bulk_bank bypassed the direction SSOT ---------------------------------


@pytest.fixture(scope="module")
def rebuilt_bank(tmp_path_factory):
    out = tmp_path_factory.mktemp("bulk") / "bulk_bank.json"
    import subprocess
    r = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), str(SYNTH / "make_bulk_bank.py"),
         "--spec", str(SYNTH / "bulk_spec.json"), "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_builder_routes_through_build_direction():
    """AGENTS.md, in as many words: `build_direction()` "is the single source of truth for
    what each engine actually receives — never bypass it"."""
    src = _src("make_bulk_bank.py")
    assert "from book_ingest import build_direction" in src
    assert "build_direction(" in src


def test_dia_lines_carry_the_end_of_audio_guard(rebuilt_bank):
    """The trailing `[S1]` is not decoration. nari-labs' generation guidelines prescribe
    repeating the speaker tag to stop Dia improvising a tail, and the forked builder
    emitted `f"[S1] {dia_tags}{text}"` with no trailing tag — so the bank rendered long
    and wrong while looking correctly directed."""
    dia = [l for l in rebuilt_bank["lines"] if l["engine"] == "dia"]
    assert dia
    for line in dia:
        rt = line["direction"]["render_text"]
        assert rt.startswith("[S1] ") and rt.endswith(" [S1]"), rt[:80]


def test_dia_inline_tags_survive_the_ssot(rebuilt_bank):
    """`bulk_spec.json` sets `dia_tags: "(laughs) "`. Routing through a function with no
    slot for it would have silently dropped real direction — which is the defect the
    2026-07-25 relay audit found, not a fix for it. The slot was added instead."""
    tagged = [l for l in rebuilt_bank["lines"]
              if l["engine"] == "dia" and "(laughs)" in l["direction"]["render_text"]]
    assert tagged, "the spec's inline tag was dropped"


def test_moss85_quality_survives_the_ssot(rebuilt_bank):
    """`synth_moss85.py` reads `direction["quality"]` — a real renderer input — and
    `build_direction` had no slot for it. That gap is WHY the builder forked; closing it
    is what makes "never bypass" a rule rather than a wish."""
    spec = json.loads((SYNTH / "bulk_spec.json").read_text(encoding="utf-8"))
    wanted = {j["quality"] for r in spec["registers"].values() for j in r["jobs"]
              if j.get("quality")}
    assert wanted, "the spec no longer exercises this path"

    got = {l["direction"]["quality"] for l in rebuilt_bank["lines"]
           if l["engine"] == "moss85" and "quality" in l["direction"]}
    assert got == wanted, f"quality lost in translation: wanted {wanted}, got {got}"

    # Only the jobs that ASK for it get it — an empty `quality` must not become a key.
    blank = [l for l in rebuilt_bank["lines"]
             if l["direction"].get("quality") in ("", None) and "quality" in l["direction"]]
    assert not blank


def test_qwen_lines_no_longer_carry_a_key_the_renderer_cannot_read(rebuilt_bank):
    """The sharpest consequence of the bypass. `synth_qwen.py` reads ONLY
    `direction["instruct"]`; the forked builder wrote the voice design to a separate
    `design` key, so all 87 qwen lines in the shipped bank rendered with the instruct half
    alone and the design — gender, age, timbre, accent — never reached the model.

    That is verbatim the owner finding of 2026-07-25 that `build_direction` was written to
    end, reproduced in a builder nobody re-checked.
    """
    qwen = [l for l in rebuilt_bank["lines"] if l["engine"] == "qwen"]
    assert qwen
    for line in qwen:
        assert "design" not in line["direction"], "a key synth_qwen.py never reads"
        assert line["direction"]["instruct"]

    shipped = json.loads((SYNTH / "bulk_bank.json").read_text(encoding="utf-8"))
    stranded = [l for l in shipped["lines"]
                if l["engine"] == "qwen" and "design" in l["direction"]]
    assert stranded, "the shipped bank should still show the defect this test describes"


def test_ids_name_their_engine(rebuilt_bank):
    """Two jobs differing only by engine produced the SAME id and would have collided in
    the shared output directory. The current spec escapes that only because each engine
    happens to use a distinct `voice` name — a convention, not a guard."""
    ids = [l["id"] for l in rebuilt_bank["lines"]]
    assert len(set(ids)) == len(ids)
    for line in rebuilt_bank["lines"]:
        assert f"_{line['engine']}_" in line["id"], line["id"]


# --- B-L8 / B-L10 ---------------------------------------------------------------------


def test_the_newscaster_variety_bias_is_not_dead_code():
    """`used.add(rid)` accumulated the same ids that go into `exclude`, and exclude ⊇ used
    at every iteration — so the bias could only ever apply to references already dropped
    outright. Two mechanisms where one is inert is how a later reader concludes the bias
    is doing work it is not."""
    src = _src("make_newscaster_bank.py")
    assert "used.add(rid)" not in src
    assert "used=used" not in src


def test_the_container_identity_script_asserts_its_postcondition():
    """Every step is deliberately tolerant (a second run must not fail because the group
    exists), which also hid the one failure that matters: ai-mgr not being created. The
    script exited 0 — `echo` succeeds even when the substitution in it does not — and the
    caller's `&&` proceeded to a `runuser` failure several lines from the cause. In
    synth_bank.sh, where a failing engine is non-fatal by design, that reads as all five
    engines failing."""
    src = (SYNTH / "container_as_ai_mgr.sh").read_text(encoding="utf-8")
    assert "if ! id ai-mgr" in src
    assert "exit 1" in src
    assert os.access(SYNTH / "container_as_ai_mgr.sh", os.R_OK)


# --- B-M2: the two paths that cross two quoting layers --------------------------------


def test_bank_and_out_are_validated_before_they_are_interpolated():
    """Filed as open; the guards actually landed in `b9c7a7e` with the Critical/High
    round and the todo entry went stale. Pinned here rather than re-implemented.

    `$BANK` and `$OUT` are interpolated unquoted into a `docker bash -c` string and then
    into a `runuser bash -c` string inside it. Quoting through two layers is not
    practical, so the fix is VALIDATION: both must be absolute `/data` paths (the only
    tree mounted into the containers) and neither may contain a space or a quote. Without
    it a stray space surfaces as a confusing engine error several layers from the cause.
    """
    src = (SYNTH / "synth_bank.sh").read_text(encoding="utf-8")
    assert 'case "$BANK" in /data/*)' in src
    assert 'case "$OUT"  in /data/*)' in src
    assert 'case "$BANK$OUT" in *[\\ \\\'\\"]*)' in src
    # and the single-level-OUT refusal, which is the same family
    assert 'DATASETS_ROOT="/data/model-training/datasets"' in src


# --- B-L7: the guardless fork of ref_select scoring -----------------------------------


def test_the_v3d_builder_shares_the_helpers_it_used_to_copy():
    """`design_gender` and the VAT accessor were duplicated BYTE FOR BYTE from
    ref_select — the B-L5 shape again, where a copied rule can be fixed in one place and
    not the other."""
    src = _src("make_v3d_bank.py")
    assert "from ref_select import" in src
    assert "def design_gender" not in src
    assert "def keep_vat" not in src


def test_the_v3d_builder_applies_the_casting_guards():
    """It scored references itself — gender, VAT proximity, duration window, variety bias
    — with none of the guards. It cannot simply CALL `select_reference` (it casts from the
    campaign's own keeps file, not the certified pool, and that difference is legitimate),
    but the guards are properties of the reference CLIP and apply either way."""
    src = _src("make_v3d_bank.py")
    assert "REF_BLACKLIST" in src and "MAX_REF_EXCURSION" in src
    assert 'k.get("id") in REF_BLACKLIST' in src


def test_an_unfillable_line_does_not_take_the_bank_with_it():
    """`cands[0]` had no emptiness check. With the guards narrowing the pool that is a
    reachable state, and one unfillable line must not cost the other nine."""
    src = _src("make_v3d_bank.py")
    assert "if not cands:" in src
    assert "no eligible reference" in src


# --- issue #44: the folded target did not reproduce the round it claimed to --------------


def test_the_narration_round_reproduces_the_bank_it_closed():
    """`delivery-v1-narration-r2` is rendered, audited and closed (2026-08-04), so this
    builder's numbers exist to REPRODUCE it, not to size a new campaign — and after the
    Documentary retirement they did not.

    The target is not consumed as a lane total; it is divided across the lane's books
    (`each = want // len(books)`), and `len(books)` went 1 + 4 -> 5 when
    `voyage-of-the-beagle` moved into Neutral. `{"Neutral": 141}` therefore selected 140
    lines — `141 // 5 * 5` — with beagle dropping 45 -> 28 and the other four rising
    24 -> 28. Both halves are silent: nothing crashes, and the report prints whatever it
    selected.

    Asserted on the invariant the issue names — the per-book split sums to the lane total,
    with no floor division between them — plus the composition the shipped bank actually
    has (141 = 45 + 24 + 24 + 24 + 24, read off `bank.json`).
    """
    import make_narration_bank as mb

    assert mb.PER_BOOK == {"voyage-of-the-beagle": 45, "conan-stories": 24,
                           "up-from-slavery": 24, "franklin-autobiography": 24,
                           "walden": 24}
    assert sum(mb.PER_BOOK.values()) == 141
    # No book is sized by a rule the lane table cannot see, and no lane total is a second
    # literal that has to agree with the per-book one.
    assert set(mb.PER_BOOK) == set(mb.LANES)
    assert sum(mb.TARGETS.values()) == sum(mb.PER_BOOK.values())
    # The arithmetic that replaced it, kept as the thing being guarded against: dividing
    # the same total across the same five books loses a line AND reshuffles the split.
    each = 141 // len(mb.LANES)
    assert each * len(mb.LANES) == 140 and each != mb.PER_BOOK["voyage-of-the-beagle"]


def test_the_builder_records_that_the_casting_cannot_be_reproduced():
    """The text selection reproduces; the AUDIO does not, and that limit has to be stated
    where the numbers are or the next reader will believe a re-run rebuilds the round.

    Engines are dealt per LANE. The closed round dealt two lanes under two measured mixes
    (Documentary 45 + Neutral 96, giving orpheus 4 + 14 and moss_vg 2 + 10); a re-run
    deals one lane of 141 under Neutral's mix alone, so the engine split and every
    per-(book, engine) frozen voice come out different. It is not recoverable in
    principle either: `ENGINE_MIX_BY_LANE` carries no Documentary mix and
    `ref_select.mix_for_lane` refuses the lane outright.
    """
    src = _src("make_narration_bank.py")
    assert "THE CASTING CANNOT BE REPRODUCED" in src
    assert "PER BOOK RATHER THAN PER LANE" in src

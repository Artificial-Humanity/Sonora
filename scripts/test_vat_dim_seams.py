# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""test_vat_dim_seams — prove the vat_dim seams fail loudly, before the migration.

The delivery channel makes VAT four-wide (direction-contract-v2: 5 lanes + unknown as
zero). `vat_dim=3` is currently assumed at four places that never checked it, and the
review (E-M1/F-H2/T8) flagged them as the migration's blast radius:

  1. the FiLM trunk       — every conditioning tensor entering the model core
  2. the filelist parse   — `v,a,t` split with no field count
  3. the collate          — all-or-none `torch.stack`, the SILENT one
  4. the export converter — a hardcoded VAT_DIM that also writes config.json

A guard nobody has seen fail is a guess. This drives each one with the wrong width and
asserts on the error, so the assertions are known-good BEFORE anything relies on them.

Run:  .venv/bin/python scripts/test_vat_dim_seams.py
"""
import pathlib
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from matcha.data.text_mel_datamodule import TextMelBatchCollate, TextMelDataset  # noqa: E402
from matcha.models.components.film import VATTrunk  # noqa: E402


def _load_from_convert_vat(name):
    """Pull one function out of convert_vat.py without importing the module.

    convert_vat.py opens with `import _stub`, a shim that only exists inside the
    litert conversion harness at /data/toolchain/litert-conversion, so the module is
    not importable from the repo. The helper under test is pure — it reads a tensor
    shape — so compile just its definition and test the real source rather than a copy
    that could drift from it.
    """
    import ast

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts/litert_export/convert_vat.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"torch": torch}
            exec(compile(ast.Module([node], []), "<convert_vat>", "exec"), ns)  # noqa: S102
            return ns[name]
    raise AssertionError(f"{name} not found in convert_vat.py — the export guard is gone")


detect_vat_dim = _load_from_convert_vat("detect_vat_dim")


def _config_vat_dim():
    """`vat_dim` out of configs/model/matcha.yaml, without pulling in hydra.

    Read as text on purpose: the data configs interpolate `${model.vat_dim}`, so this one
    line decides what the filelist parse, the collate and the trunk all expect. If it and
    `matcha/delivery.py` ever disagree, every clip is conditioned with a vector of the
    wrong shape or the wrong meaning, and the seam guards above only catch the shape.
    """
    import re

    text = (pathlib.Path(__file__).resolve().parents[1]
            / "configs/model/matcha.yaml").read_text(encoding="utf-8")
    m = re.search(r"^vat_dim:\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None

PASS, FAIL = [], []


def check(name, fn, want):
    """`fn` must raise something whose message contains `want`."""
    try:
        fn()
    except Exception as e:  # pylint: disable=broad-except
        if want in str(e):
            PASS.append(name)
        else:
            FAIL.append(f"{name}: raised, but message lacked {want!r} -> {e}")
        return
    FAIL.append(f"{name}: DID NOT RAISE — the seam is unguarded")



def _assert(cond):
    """check_ok judges on "did not raise"; these checks judge on a boolean."""
    if not cond:
        raise AssertionError("condition is false")
    return True

def check_ok(name, fn):
    """`fn` must NOT raise — the guards must not break the working path."""
    try:
        fn()
        PASS.append(name)
    except Exception as e:  # pylint: disable=broad-except
        FAIL.append(f"{name}: raised on a VALID input -> {e}")


# --- 1. the FiLM trunk ------------------------------------------------------------
trunk3 = VATTrunk(vat_dim=3, cond_dim=8)
check("trunk: 4 channels into a 3-wide trunk",
      lambda: trunk3(torch.zeros(2, 4, 16)), "vat_dim=3")
check("trunk: 3 channels into a 4-wide trunk",
      lambda: VATTrunk(vat_dim=4, cond_dim=8)(torch.zeros(2, 3, 16)), "vat_dim=4")
check("trunk: per-utterance (B, vat_dim) not expanded",
      lambda: trunk3(torch.zeros(2, 3)), "expects [B, vat_dim, T]")
check_ok("trunk: correct width passes", lambda: trunk3(torch.zeros(2, 3, 16)))
check_ok("trunk: 4-wide accepts 4 channels",
         lambda: VATTrunk(vat_dim=4, cond_dim=8)(torch.zeros(2, 4, 16)))

# --- 2. the filelist parse --------------------------------------------------------
def _dataset(line, vat_dim):
    """A dataset over one filelist line, asked for that line."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(line + "\n")
        path = fh.name
    ds = TextMelDataset(path, 1, ["no_cleaners"], True, 1024, 80, 24000, 256, 1024,
                        0, 12000, {"mel_mean": 0, "mel_std": 1}, 42, False,
                        load_vat=True, vat_dim=vat_dim)
    return ds


# get_datapoint reads the VAT field before it ever touches audio, so the width check
# fires without a real wav on disk. That is the point — it is a filelist check.
check("filelist: 3-field row against a 4-dim model",
      lambda: _dataset("nope.wav|1 2 3|0.1,0.2,0.3", 4).get_datapoint(
          ["nope.wav", "1 2 3", "0.1,0.2,0.3"]),
      "vat_dim=4")
check("filelist: 4-field row against a 3-dim model",
      lambda: _dataset("nope.wav|1 2 3|0.1,0.2,0.3,0.4", 3).get_datapoint(
          ["nope.wav", "1 2 3", "0.1,0.2,0.3,0.4"]),
      "vat_dim=3")

# --- 3. the collate ---------------------------------------------------------------
def _item(vat, n=4):
    return {"y": torch.zeros(80, n), "x": torch.zeros(n, dtype=torch.long),
            "spk": None, "filepath": f"clip{n}.wav", "x_text": "x",
            "durations": None, "vat": vat}


collate = TextMelBatchCollate(n_spks=1)
check("collate: one item missing its VAT drops the whole batch",
      lambda: collate([_item(torch.zeros(3), 4), _item(None, 5)]),
      "Mixed presence")
check_ok("collate: all present",
         lambda: collate([_item(torch.zeros(3), 4), _item(torch.zeros(3), 5)]))
check_ok("collate: none present (unconditioned run)",
         lambda: collate([_item(None, 4), _item(None, 5)]))

# The silent failure this replaced, stated as a property: a batch that keeps its
# conditioning must never come back with vat=None.
mixed_ok = collate([_item(torch.zeros(3), 4), _item(torch.zeros(3), 5)])
if mixed_ok["vat"] is None or mixed_ok["vat"].shape != (2, 3):
    FAIL.append(f"collate: fully-labelled batch returned vat={mixed_ok['vat']}")
else:
    PASS.append("collate: fully-labelled batch keeps its conditioning")

# --- 4. the export converter ------------------------------------------------------
check_ok("export: detect_vat_dim reads the trunk conv",
         lambda: None if detect_vat_dim(
             {"encoder.vat_trunk.net.0.weight": torch.zeros(256, 4, 1)}) == 4
         else (_ for _ in ()).throw(AssertionError("read the wrong width")))
check_ok("export: unconditioned checkpoint reads as None",
         lambda: None if detect_vat_dim({"encoder.emb.weight": torch.zeros(2, 2)})
         is None else (_ for _ in ()).throw(AssertionError("should be None")))

# --- 5. the delivery channel (contract v2, added with the migration) ---------------
#
# The seams above prove a WIDTH disagreement fails loudly. These prove the width is now
# the one the contract specifies and that the encoding is self-consistent — because the
# failure mode of a correct-width, wrong-MEANING vector is a fluent render in the wrong
# manner, which no shape check can see.
from matcha import delivery  # noqa: E402

check_ok("delivery: production width is 3 V/A/T + 5 one-hot lanes",
         lambda: None if delivery.VAT_DIM == 8 and delivery.VAT_BASE_DIM == 3
         else (_ for _ in ()).throw(AssertionError(f"VAT_DIM={delivery.VAT_DIM}")))

check_ok("delivery: the model config agrees with the encoding",
         lambda: None if _config_vat_dim() == delivery.VAT_DIM
         else (_ for _ in ()).throw(AssertionError(
             f"configs/model/matcha.yaml says {_config_vat_dim()}, "
             f"matcha/delivery.py says {delivery.VAT_DIM}")))

check_ok("delivery: unknown is the v1 vector padded with zeros",
         lambda: None if delivery.vat_vector(0.4, -0.2, 0.1) == [0.4, -0.2, 0.1] + [0.0] * 5
         else (_ for _ in ()).throw(AssertionError("unknown is not all-zero")))

check_ok("delivery: every lane round-trips through the vector",
         lambda: None if all(
             delivery.lane_of_vector(delivery.vat_vector(0, 0, 0, ln)) == ln
             for ln in delivery.DELIVERY_LANES)
         else (_ for _ in ()).throw(AssertionError("a lane did not round-trip")))

check_ok("delivery: a blank label reads back as unknown",
         lambda: None if delivery.lane_of_vector(delivery.vat_vector(0, 0, 0, "")) == ""
         else (_ for _ in ()).throw(AssertionError("blank did not round-trip")))

check("delivery: an unrecognised lane is refused, not treated as unknown",
      lambda: delivery.delivery_index("Newscasterr"), "unknown delivery lane")
check("delivery: two lanes at once is refused",
      lambda: delivery.lane_of_vector([0, 0, 0, 1, 1, 0, 0, 0]), "not one-hot")
check("delivery: a fractional lane is refused",
      lambda: delivery.lane_of_vector([0, 0, 0, 0.5, 0, 0, 0, 0]), "not one-hot")

# The trunk must accept the production width — the check that would have caught a
# migration that bumped the config and forgot the model.
check_ok("delivery: the trunk accepts the production width",
         lambda: VATTrunk(vat_dim=delivery.VAT_DIM, cond_dim=8)(
             torch.zeros(2, delivery.VAT_DIM, 16)))


# --- 5. the warm start's widening (2026-08-07/08) -----------------------------------
#
# The seam between a 3-wide checkpoint and an 8-wide model. make_warmstart DROPPED every
# shape-mismatched tensor and let random init stand, which discarded the whole learned
# V/A/T -> FiLM pathway on a run whose only purpose was to change the width. Widening is
# an ALLOWLIST because the same operation is right for one tensor and silently wrong for
# another: channel position is a contract, speaker-row position is not.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_warmstart as _mw  # noqa: E402

check_ok("warmstart: the VAT trunk widens, new channels at zero",
         lambda: _assert(
             (lambda o, d: o is not None and torch.equal(o[:, :3], d)
              and bool((o[:, 3:] == 0).all()))(
                 _mw._widen("encoder.vat_trunk.net.0.weight",
                            (d := torch.randn(256, 3, 1)), torch.randn(256, 8, 1)), d)))

check_ok("warmstart: the speaker table is REFUSED without a proven index map",
         lambda: _assert(_mw._widen("spk_emb.weight", torch.randn(247, 64),
                                    torch.randn(2655, 64)) is None))

check_ok("warmstart: with proof it widens and new speakers keep the fresh init",
         lambda: _assert(
             (lambda o, d, m: o is not None and torch.equal(o[:247], d)
              and torch.equal(o[247:], m[247:]))(
                 _mw._widen("spk_emb.weight", (d := torch.randn(247, 64)),
                            (m := torch.randn(2655, 64)), allow_speakers=True), d, m)))

check_ok("warmstart: a shrink is never a widen",
         lambda: _assert(_mw._widen("spk_emb.weight", torch.randn(2655, 64),
                                    torch.randn(247, 64), allow_speakers=True) is None))

check_ok("warmstart: an un-allowlisted tensor is never reshaped",
         lambda: _assert(_mw._widen("decoder.estimator.mid_block1.0.block.0.weight",
                                    torch.randn(256, 3, 1), torch.randn(256, 8, 1)) is None))

check_ok("warmstart: a reordered speaker map fails the prefix proof",
         lambda: _assert(
             not _mw._speaker_prefix_ok({"libritts_id_to_index": {"a": 0, "b": 1}},
                                        {"libritts_id_to_index": {"b": 0, "a": 1}})[0]
             and _mw._speaker_prefix_ok({"libritts_id_to_index": {"a": 0, "b": 1}},
                                        {"libritts_id_to_index": {"a": 0, "b": 1, "c": 2}})[0]))


# --- report -----------------------------------------------------------------------
for p in PASS:
    print(f"  PASS  {p}")
for f in FAIL:
    print(f"  FAIL  {f}")
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)

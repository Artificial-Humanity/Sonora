#!/usr/bin/env python
"""Rename TFLite tensor names back to the Matcha/Prosodia I/O contract.

TF's TFLiteConverter (used for the weights-only-fp16 / f32-I/O export) mangles tensor
names into ``serving_default_x:0`` / ``StatefulPartitionedCall:N`` form, but the Prosodia
engine (`crates/actor/src/engine.rs`) matches inputs by exact name (``x``, ``scales``) or
substring (``x_lengths``). This round-trips the flatbuffer through the object API and
restores the contract names.

F-M2, TWO DEFECTS, both fixed 2026-08-07:

**Outputs were mapped by EMISSION ORDER.** ``StatefulPartitionedCall:0 -> wav`` and
``:1 -> wav_lengths`` assume TF emits the waveform first. Nothing guarantees that. If the
order ever flips — a TF upgrade, a different converter path, a graph rebuilt with its
returns swapped — this renames the LENGTHS tensor ``wav`` and the waveform
``wav_lengths``, and Prosodia matches by name, so it would read a two-element int tensor
as audio. The rename is now checked against SHAPE and DTYPE: a waveform is a large float
tensor, a length is a tiny integer one, and a swap is refused rather than performed.

**``renamed 0 tensors`` exited 0.** A graph whose names this table does not recognise —
again, a TF upgrade is enough — was copied through untouched and reported as success. The
failure then surfaced in Prosodia as an input it could not find, several steps away from
the cause. Every expected rename must now fire, or the tool refuses.

Requires the ``schema_generated.py`` produced by onnx2tf (kept alongside the exports, e.g.
in the HF repo's ``baseline-ljspeech-22k/``) on sys.path or via ``--schema-dir``.

Example:
    python scripts/tools/rename_tflite_tensors.py in.tflite out.tflite \
        --schema-dir /path/to/dir_with_schema_generated
"""
import argparse
import sys

DEFAULT_RENAMES = {
    "serving_default_x:0": "x",
    "serving_default_x_lengths:0": "x_lengths",
    "serving_default_scales:0": "scales",
    "serving_default_spks:0": "spks",
    "serving_default_vat:0": "vat",
    "StatefulPartitionedCall:0": "wav",
    "StatefulPartitionedCall:1": "wav_lengths",
}

# What the contract names must LOOK like, so a rename can be checked rather than trusted.
# `kind` is "float" or "int"; `min_elems` / `max_elems` bound the element count.
#
# The two that matter are `wav` and `wav_lengths`, because they are the pair the old code
# distinguished only by emission order. A waveform at 24 kHz is tens of thousands of
# samples; a length is one or two integers. Nothing else in the graph is close.
EXPECTED = {
    "x":           dict(kind="int",   min_elems=2,     max_elems=None),
    "x_lengths":   dict(kind="int",   min_elems=1,     max_elems=8),
    "scales":      dict(kind="float", min_elems=1,     max_elems=8),
    "spks":        dict(kind="int",   min_elems=1,     max_elems=8),
    "vat":         dict(kind="float", min_elems=1,     max_elems=None),
    "wav":         dict(kind="float", min_elems=1024,  max_elems=None),
    "wav_lengths": dict(kind="int",   min_elems=1,     max_elems=8),
}

# TFLite TensorType enum values that are integral. Taken from the schema rather than
# guessed at call time so this stays readable without the generated module to hand:
#   0 FLOAT32  1 FLOAT16  2 INT32  3 UINT8  4 INT64  5 STRING  6 BOOL  7 INT16 ...
INT_TYPES = {2, 3, 4, 6, 7, 9, 10, 16, 17}


def _kind(type_code):
    return "int" if int(type_code) in INT_TYPES else "float"


def _elems(shape):
    n = 1
    for d in shape or ():
        n *= max(int(d), 1)
    return n


def plan_renames(tensors, io_indices, renames):
    """Decide every rename, or explain why not. Pure — no flatbuffers, so it is testable.

    `tensors` is a list of `{"name", "shape", "type", "index"}`. `io_indices` is the set of
    tensor indices that are graph inputs or outputs.

    Returns `(plan, problems)` where `plan` maps tensor index -> new name.
    """
    plan, problems = {}, []
    seen = {}
    for t in tensors:
        old = t["name"]
        if old not in renames:
            continue
        new = renames[old]
        # Only graph I/O carries the contract. An intermediate that happens to share a
        # name is not what Prosodia binds to, and renaming it is at best noise.
        if t["index"] not in io_indices:
            problems.append(f"{old!r} -> {new!r}: not a graph input or output")
            continue
        want = EXPECTED.get(new)
        if want is not None:
            kind, n = _kind(t["type"]), _elems(t["shape"])
            if kind != want["kind"]:
                problems.append(
                    f"{old!r} -> {new!r}: is {kind}, contract says {want['kind']} "
                    f"(shape {t['shape']}). THIS IS THE EMISSION-ORDER BUG: outputs were "
                    "mapped by position, so a swapped pair renamed the lengths 'wav' and "
                    "Prosodia would read an int tensor as audio.")
                continue
            if want["min_elems"] is not None and n < want["min_elems"]:
                problems.append(
                    f"{old!r} -> {new!r}: {n} element(s), contract expects at least "
                    f"{want['min_elems']} (shape {t['shape']})")
                continue
            if want["max_elems"] is not None and n > want["max_elems"]:
                problems.append(
                    f"{old!r} -> {new!r}: {n} element(s), contract expects at most "
                    f"{want['max_elems']} (shape {t['shape']})")
                continue
        if new in seen:
            problems.append(f"{new!r} claimed twice: {seen[new]!r} and {old!r}")
            continue
        seen[new] = old
        plan[t["index"]] = new
    return plan, problems


def check_complete(plan, renames, present_names):
    """-> problems for renames that were ASKED for and did not happen.

    `renamed 0 tensors` used to exit 0. A graph whose names this table does not recognise
    was copied through untouched and called a success, and the failure surfaced in Prosodia
    as a missing input several steps from the cause.
    """
    done = set(plan.values())
    missing = [f"{old!r} -> {new!r}: no such tensor in the graph"
               for old, new in sorted(renames.items())
               if new not in done and old in present_names]
    absent = sorted(old for old in renames if old not in present_names)
    return missing, absent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--schema-dir", default=None, help="directory containing schema_generated.py")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="extra rename (repeatable); overrides defaults on conflict")
    ap.add_argument("--allow-missing", action="store_true",
                    help="proceed when a DEFAULT rename finds no matching tensor. Never "
                         "relaxes the shape/dtype checks — those catch a wrong rename, "
                         "not an absent one.")
    args = ap.parse_args()

    if args.schema_dir:
        sys.path.insert(0, args.schema_dir)
    import flatbuffers
    import schema_generated as sg

    renames = dict(DEFAULT_RENAMES)
    explicit = set()
    for spec in args.rename:
        old, new = spec.split("=", 1)
        renames[old] = new
        explicit.add(old)

    model = sg.ModelT.InitFromPackedBuf(open(args.src, "rb").read(), 0)

    total_plan, all_problems, present = {}, [], set()
    for si, sub in enumerate(model.subgraphs):
        tensors = []
        for ti, t in enumerate(sub.tensors):
            name = t.name.decode() if isinstance(t.name, (bytes, bytearray)) else t.name
            tensors.append({"name": name, "shape": list(t.shape or []),
                            "type": t.type, "index": ti})
            present.add(name)
        io_indices = {int(i) for i in (list(sub.inputs or []) + list(sub.outputs or []))}
        plan, problems = plan_renames(tensors, io_indices, renames)
        total_plan[si] = plan
        all_problems += problems

    missing, absent = check_complete(
        {k: v for p in total_plan.values() for k, v in p.items()}, renames, present)

    if all_problems:
        print("!! refusing to rename:", file=sys.stderr)
        for p in all_problems:
            print(f"   {p}", file=sys.stderr)
        raise SystemExit(1)

    renamed = sum(len(p) for p in total_plan.values())
    if not renamed:
        raise SystemExit(
            f"!! renamed 0 tensors — none of {sorted(renames)} is in {args.src}.\n"
            "   Exiting 0 here used to copy the file through untouched and report success;\n"
            "   the failure then surfaced in Prosodia as an input it could not find. The\n"
            "   converter's naming has probably changed — inspect the graph and extend\n"
            "   DEFAULT_RENAMES or pass --rename OLD=NEW."
        )
    if missing and not args.allow_missing:
        print("!! some renames did not fire:", file=sys.stderr)
        for m in missing:
            print(f"   {m}", file=sys.stderr)
        print("   Pass --allow-missing if this graph legitimately lacks them "
              "(an unconditioned export has no spks/vat).", file=sys.stderr)
        raise SystemExit(1)
    for old in absent:
        if old in explicit:
            print(f"   note: --rename {old}=… matched nothing", file=sys.stderr)

    for si, sub in enumerate(model.subgraphs):
        for ti, new in total_plan[si].items():
            sub.tensors[ti].name = new.encode()

    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    open(args.dst, "wb").write(b.Output())
    for si, plan in total_plan.items():
        for ti, new in sorted(plan.items()):
            print(f"  subgraph {si} tensor {ti} -> {new}")
    print(f"renamed {renamed} tensors -> {args.dst}")


if __name__ == "__main__":
    main()

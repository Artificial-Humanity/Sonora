#!/usr/bin/env python
"""Rename TFLite tensor names back to the Matcha/Prosodia I/O contract.

TF's TFLiteConverter (used for the weights-only-fp16 / f32-I/O export) mangles
tensor names into ``serving_default_x:0`` / ``StatefulPartitionedCall:N`` form,
but the Prosodia engine (`crates/actor/src/engine.rs`) matches inputs by exact
name (``x``, ``scales``) or substring (``x_lengths``). This round-trips the
flatbuffer through the object API and restores the contract names.

Requires the ``schema_generated.py`` produced by onnx2tf (kept alongside the
exports, e.g. in the HF repo's ``v1-ljspeech/``) on sys.path or via
``--schema-dir``.

Example:
    python scripts/rename_tflite_tensors.py in.tflite out.tflite \
        --schema-dir /path/to/dir_with_schema_generated
"""
import argparse
import sys

DEFAULT_RENAMES = {
    "serving_default_x:0": "x",
    "serving_default_x_lengths:0": "x_lengths",
    "serving_default_scales:0": "scales",
    "StatefulPartitionedCall:0": "wav",
    "StatefulPartitionedCall:1": "wav_lengths",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--schema-dir", default=None, help="directory containing schema_generated.py")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="extra rename (repeatable); overrides defaults on conflict")
    args = ap.parse_args()

    if args.schema_dir:
        sys.path.insert(0, args.schema_dir)
    import flatbuffers
    import schema_generated as sg

    renames = dict(DEFAULT_RENAMES)
    for spec in args.rename:
        old, new = spec.split("=", 1)
        renames[old] = new

    model = sg.ModelT.InitFromPackedBuf(open(args.src, "rb").read(), 0)
    renamed = 0
    for sub in model.subgraphs:
        for t in sub.tensors:
            name = t.name.decode() if isinstance(t.name, (bytes, bytearray)) else t.name
            if name in renames:
                t.name = renames[name].encode()
                renamed += 1
    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    open(args.dst, "wb").write(b.Output())
    print(f"renamed {renamed} tensors -> {args.dst}")


if __name__ == "__main__":
    main()

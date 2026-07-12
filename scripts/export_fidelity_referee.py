#!/usr/bin/env python
"""Export-fidelity referee — the standing ONNX-vs-TFLite gate for Sonora.

Runs identical input through the ONNX (the oracle) and a converted TFLite, and
reports end-to-end waveform correlation. Optionally transcribes both with
faster-whisper for an intelligibility check.

Why a dedicated gate: onnx2tf's own ``-cotof`` self-report can pass while the
artifact is numerically broken (it skips nondeterministic ops and, on a crashed
validator, emits a false ``cosine=1``). A real-input check against the ONNX is
the only trustworthy signal. This tool was written to catch — and then confirm
the fix of — the 2026-07-11 encoder-LayerNorm conversion bug (garbled TFLite
audio; ONNX rendered cleanly).

Determinism note: the e2e graph contains a ``RandomNormalLike`` (CFM decoder
noise) whose RNG differs between onnxruntime and the TFLite interpreter, so a
stochastic render is a different-but-valid sample and end-to-end cosine is
meaningless at temperature > 0. Matcha computes ``z = randn * temperature``, so
``--temperature 0`` zeroes the noise and makes the two graphs bit-comparable —
that is the mode to use for a fidelity number (expect cosine ~1.0). Use
temperature > 0 + ``--asr`` only for an intelligibility spot-check.

Example:
    python scripts/export_fidelity_referee.py \
        model_e2e.onnx model_e2e_float32.tflite --temperature 0
"""
import argparse
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

# "The morning light." phonemized via matcha english_cleaners2 (18 tokens).
DEFAULT_IDS = [81, 83, 16, 55, 156, 76, 158, 123, 56, 102, 112, 16, 54, 156, 43, 102, 62, 4]
SAMPLE_RATE = 22050  # Phase 0 = LJSpeech + hifigan_T2_v1


def build_inputs(ids, static_limit, temperature, length_scale):
    x = np.zeros((1, static_limit), np.int64)
    x[0, : len(ids)] = ids
    x_lengths = np.array([len(ids)], np.int64)
    scales = np.array([temperature, length_scale], np.float32)
    return x, x_lengths, scales


def run_onnx(path, inputs):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {i.name: v for i, v in zip(sess.get_inputs(), inputs)}
    wav, wav_lengths = sess.run(None, feed)
    return wav.reshape(-1)[: int(wav_lengths.reshape(-1)[0])]


def run_tflite(path, inputs):
    x, x_lengths, scales = inputs
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    for d in interp.get_input_details():
        if d["dtype"] in (np.int64, np.int32) and len(d["shape"]) == 2:
            v = x
        elif d["dtype"] in (np.int64, np.int32):
            v = x_lengths
        else:
            v = scales
        interp.set_tensor(d["index"], v.astype(d["dtype"]))
    interp.invoke()
    outs = interp.get_output_details()
    return max((interp.get_tensor(o["index"]) for o in outs), key=lambda a: a.size).reshape(-1)


def cosine(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", help="path to the ONNX oracle")
    ap.add_argument("tflite", help="path to the TFLite artifact under test")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) = deterministic, RNG-free fidelity check; >0 = stochastic (use --asr only)")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--static-limit", type=int, default=50, help="baked token limit of the e2e graph")
    ap.add_argument("--ids", type=int, nargs="+", default=None, help="phoneme ids (default: 'The morning light.')")
    ap.add_argument("--asr", action="store_true", help="also transcribe both with faster-whisper")
    ap.add_argument("--write-wav", metavar="DIR", default=None, help="dump ref_onnx.wav / ref_tflite.wav here")
    ap.add_argument("--threshold", type=float, default=0.99, help="pass threshold for deterministic cosine")
    args = ap.parse_args()

    ids = args.ids or DEFAULT_IDS
    inputs = build_inputs(ids, args.static_limit, args.temperature, args.length_scale)
    o = run_onnx(args.onnx, inputs)
    t = run_tflite(args.tflite, inputs)[: len(o)]
    cos, n = cosine(o, t)
    rmse = float(np.sqrt(np.mean((o[:n] - t[:n]) ** 2)))
    print(f"ONNX   len={len(o)} rms={np.sqrt(np.mean(o ** 2)):.4f}")
    print(f"TFLite len={n} rms={np.sqrt(np.mean(t[:n] ** 2)):.4f}")
    print(f"cosine={cos:.4f} rmse={rmse:.4f}")

    if args.write_wav or args.asr:
        import soundfile as sf
        import os
        d = args.write_wav or "/tmp"
        op, tp = os.path.join(d, "ref_onnx.wav"), os.path.join(d, "ref_tflite.wav")
        sf.write(op, o.astype(np.float32), SAMPLE_RATE)
        sf.write(tp, t[:n].astype(np.float32), SAMPLE_RATE)
        if args.asr:
            from faster_whisper import WhisperModel
            m = WhisperModel("base.en", device="cpu", compute_type="int8")
            for label, w in (("ONNX", op), ("TFLite", tp)):
                segs, _ = m.transcribe(w)
                print(f"ASR[{label}]: {' '.join(s.text for s in segs).strip()!r}")

    if args.temperature == 0.0:
        ok = cos >= args.threshold
        print(f"{'PASS' if ok else 'FAIL'} (deterministic cosine {cos:.4f} vs threshold {args.threshold})")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

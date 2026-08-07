#!/usr/bin/env python
"""Export-fidelity referee — the standing ONNX-vs-TFLite gate for Sonora.

Runs identical input through the ONNX (the oracle) and a converted TFLite, and reports
end-to-end waveform agreement. Optionally transcribes both with faster-whisper for an
intelligibility check.

Why a dedicated gate: onnx2tf's own ``-cotof`` self-report can pass while the artifact is
numerically broken (it skips nondeterministic ops and, on a crashed validator, emits a
false ``cosine=1``). A real-input check against the ONNX is the only trustworthy signal.
This tool was written to catch — and then confirm the fix of — the 2026-07-11
encoder-LayerNorm conversion bug (garbled TFLite audio; ONNX rendered cleanly).

TWO THINGS IT GOT WRONG, both fixed 2026-08-07:

**F-M1 — it bound graph inputs by a dtype heuristic** ("int + 2-D means tokens, int means
lengths, anything else means scales"). That is fine for the three-input Phase 0 e2e graph
and wrong for everything since: a CONDITIONED graph has `spk` (float), `vat` (float) and
`scales` (float), and the heuristic hands all three the same tensor. So the referee could
not score the conditioned lane at all — it either raised a shape error from inside the
interpreter, or, when shapes happened to agree, silently compared the wrong render.
Inputs are bound BY NAME now, with an explicit shape-based fallback and a refusal on
ambiguity. Guessing is what the tool exists to stop.

**F-M5 — it gated on a scale-INVARIANT metric.** Cosine similarity (and Pearson) cannot
see a systematic gain error: ``tflite = 0.5 * onnx``, every sample exactly half, scores
**1.000000** and passes a 0.99 threshold with room to spare. RMSE was computed and printed
and never gated on. That is the worst possible blind spot for this model, whose flagship
axis is a LOUDNESS dial — a mis-scaled dequantization or a dropped factor of two reads
on-device as "the energy channel is weak", not as a broken export. The gate is now cosine
AND gain AND normalised RMSE, all three.

Determinism note: the e2e graph contains a ``RandomNormalLike`` (CFM decoder noise) whose
RNG differs between onnxruntime and the TFLite interpreter, so a stochastic render is a
different-but-valid sample and end-to-end cosine is meaningless at temperature > 0. Matcha
computes ``z = randn * temperature``, so ``--temperature 0`` zeroes the noise and makes the
two graphs bit-comparable — that is the mode to use for a fidelity number. Use
temperature > 0 + ``--asr`` only for an intelligibility spot-check.

Example:
    python scripts/export_fidelity_referee.py \
        model_e2e.onnx model_e2e_float32.tflite --temperature 0

    # conditioned lane (contract v2): name the delivery lane rather than typing zeros
    python scripts/export_fidelity_referee.py sonora_e2e.onnx sonora_e2e.tflite \
        --temperature 0 --spk 245 --vat 0.4,-0.2,0.1 --delivery Newscaster \
        --sample-rate 24000
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
try:
    from matcha import delivery
except Exception:  # noqa: BLE001 - the referee must still run on a pre-v2 checkout
    delivery = None

# "The morning light." phonemized via matcha english_cleaners2 (18 tokens).
DEFAULT_IDS = [81, 83, 16, 55, 156, 76, 158, 123, 56, 102, 112, 16, 54, 156, 43, 102, 62, 4]

# Every name a Sonora/Matcha graph has used for each logical input, in the order the
# builders emit them. Bound by NAME first because that is the only binding that survives a
# graph gaining an input — which is exactly what contract v2 did.
INPUT_ALIASES = {
    "x": ("x", "ids", "tokens", "serving_default_x:0", "input_ids"),
    "x_lengths": ("x_lengths", "lengths", "serving_default_x_lengths:0"),
    "scales": ("scales", "serving_default_scales:0"),
    "spks": ("spks", "spk", "speaker", "serving_default_spks:0"),
    "vat": ("vat", "vat_tok", "conditioning", "serving_default_vat:0"),
}


def build_inputs(ids, static_limit, temperature, length_scale, spk, vat):
    """-> {logical name: array}. Only the inputs the caller actually supplied."""
    x = np.zeros((1, static_limit), np.int64)
    x[0, : len(ids)] = ids
    feed = {
        "x": x,
        "x_lengths": np.array([len(ids)], np.int64),
        "scales": np.array([temperature, length_scale], np.float32),
    }
    if spk is not None:
        feed["spks"] = np.array([spk], np.int64)
    if vat is not None:
        feed["vat"] = np.asarray(vat, np.float32).reshape(1, -1)
    return feed


def _match(details, feed, what):
    """Bind graph inputs to logical names. By NAME, then by shape. Never by guess.

    F-M1's fix. The old heuristic read dtype and rank, which cannot distinguish `spk`,
    `vat` and `scales` — three float inputs on every conditioned graph. Ambiguity is an
    ERROR here rather than a coin flip: this tool's entire job is to be the thing that
    does not quietly compare the wrong tensors.
    """
    bound, used = {}, set()
    for d in details:
        name = str(d["name"]).lower()
        for logical, aliases in INPUT_ALIASES.items():
            if logical in used:
                continue
            if any(a.lower() == name or name.endswith(":" + a.lower())
                   or a.lower() in name.split("/")[-1] for a in aliases):
                bound[d["index"] if what == "tflite" else d.name] = logical
                used.add(logical)
                break

    unbound = [d for d in details
               if (d["index"] if what == "tflite" else d.name) not in bound]
    if unbound:
        # Fall back to SHAPE, which distinguishes what dtype cannot: scales is (2,), spk
        # is (1,), vat is (1, V) or (1, V, T). Any residual ambiguity is refused.
        for d in unbound:
            shape = tuple(int(s) for s in d["shape"]) if what == "tflite" else None
            cands = [lg for lg, v in feed.items()
                     if lg not in used and shape is not None and tuple(v.shape) == shape]
            if len(cands) == 1:
                bound[d["index"] if what == "tflite" else d.name] = cands[0]
                used.add(cands[0])
            else:
                key = d["name"] if what == "tflite" else d.name
                raise SystemExit(
                    f"!! cannot bind {what} input {key!r} (shape {shape}).\n"
                    f"   Known names: {sorted(a for v in INPUT_ALIASES.values() for a in v)}\n"
                    f"   Supplied: {sorted(feed)}\n"
                    "   Binding by dtype was the F-M1 defect; this refuses rather than "
                    "guessing. Add the graph's name to INPUT_ALIASES, or supply the "
                    "missing input (--spk / --vat)."
                )
    missing = set(feed) - used
    if missing and any(m in ("spks", "vat") for m in missing):
        raise SystemExit(
            f"!! supplied {sorted(missing)} but the {what} graph has no such input.\n"
            "   This graph is unconditioned; drop the flag or use the conditioned export."
        )
    return bound


def run_onnx(path, feed):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    bound = _match(sess.get_inputs(), feed, "onnx")
    wav, wav_lengths = sess.run(None, {name: feed[lg] for name, lg in bound.items()})
    return wav.reshape(-1)[: int(wav_lengths.reshape(-1)[0])]


def run_tflite(path, feed):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    details = interp.get_input_details()
    bound = _match(details, feed, "tflite")
    by_index = {d["index"]: d for d in details}
    for index, logical in bound.items():
        interp.set_tensor(index, feed[logical].astype(by_index[index]["dtype"]))
    interp.invoke()
    outs = interp.get_output_details()
    return max((interp.get_tensor(o["index"]) for o in outs), key=lambda a: a.size).reshape(-1)


# --- agreement metrics ----------------------------------------------------------------
#
# Three, because one of them is blind to the failure this model is most exposed to.


def cosine(a, b):
    """Scale-INVARIANT. Catches shape/phase/content errors; blind to gain."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), n


def gain_error_db(a, b):
    """How much louder or quieter `b` is than `a`. THE ONE COSINE CANNOT SEE.

    `b = 0.5 * a` scores cosine 1.000000 and −6.02 dB here. That is not a contrived
    case — it is what a mis-scaled dequantization or a dropped factor of two looks like,
    and energy is this model's flagship axis.
    """
    n = min(len(a), len(b))
    ra = float(np.sqrt(np.mean(np.square(a[:n]))))
    rb = float(np.sqrt(np.mean(np.square(b[:n]))))
    return 20 * np.log10(max(rb, 1e-9) / max(ra, 1e-9))


def nrmse(a, b):
    """Sample-wise error normalised by the reference's RMS, so one threshold means the
    same thing on a whisper and on a shout."""
    n = min(len(a), len(b))
    ra = float(np.sqrt(np.mean(np.square(a[:n]))))
    return float(np.sqrt(np.mean(np.square(a[:n] - b[:n])))) / max(ra, 1e-9)


def parse_vat(args):
    """-> the conditioning vector, or None. Shares `matcha.delivery`'s encoding."""
    if args.vat is None and not args.delivery:
        return None
    try:
        vals = [float(v) for v in (args.vat or "0,0,0").split(",")]
    except ValueError:
        raise SystemExit(f"!! --vat wants comma-separated floats, got {args.vat!r}") from None
    if args.delivery:
        if delivery is None:
            raise SystemExit("!! --delivery needs matcha.delivery on the path")
        if len(vals) != delivery.VAT_BASE_DIM:
            raise SystemExit(
                f"!! --delivery expects {delivery.VAT_BASE_DIM} V/A/T values in --vat, "
                f"got {len(vals)}; the lane supplies the rest")
        try:
            vals = delivery.vat_vector(*vals, args.delivery)
        except ValueError as exc:
            raise SystemExit(f"!! {exc}") from None
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", help="path to the ONNX oracle")
    ap.add_argument("tflite", help="path to the TFLite artifact under test")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) = deterministic, RNG-free fidelity check; "
                         ">0 = stochastic (use --asr only)")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--static-limit", type=int, default=50,
                    help="baked token limit of the e2e graph")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="phoneme ids (default: 'The morning light.')")
    ap.add_argument("--spk", type=int, default=None,
                    help="speaker id (multi-speaker graphs). F-M1: the referee could not "
                         "supply this at all before 2026-08-07")
    ap.add_argument("--vat", default=None, metavar="V,A,T",
                    help="conditioning, comma-separated floats in [-1, 1]")
    ap.add_argument("--delivery", default=None,
                    choices=list(delivery.DELIVERY_LANES) if delivery else None,
                    help="delivery lane (contract v2); omit for unknown")
    ap.add_argument("--sample-rate", type=int, default=22050,
                    help="22050 = Phase 0 LJSpeech; 24000 = the Sonora lane")
    ap.add_argument("--asr", action="store_true", help="also transcribe both with faster-whisper")
    ap.add_argument("--write-wav", metavar="DIR", default=None,
                    help="dump ref_onnx.wav / ref_tflite.wav here")
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="pass threshold for deterministic cosine")
    ap.add_argument("--max-gain-db", type=float, default=0.5,
                    help="F-M5: max |gain error| in dB. Cosine cannot see this at all; "
                         "a dropped factor of two is 6 dB")
    ap.add_argument("--max-nrmse", type=float, default=0.15,
                    help="F-M5: max RMSE normalised by the reference RMS")
    args = ap.parse_args()

    feed = build_inputs(args.ids or DEFAULT_IDS, args.static_limit, args.temperature,
                        args.length_scale, args.spk, parse_vat(args))
    o = run_onnx(args.onnx, feed)
    t = run_tflite(args.tflite, feed)[: len(o)]
    cos, n = cosine(o, t)
    gain = gain_error_db(o, t)
    err = nrmse(o, t)
    print(f"ONNX   len={len(o)} rms={np.sqrt(np.mean(o ** 2)):.4f}")
    print(f"TFLite len={n} rms={np.sqrt(np.mean(t[:n] ** 2)):.4f}")
    print(f"cosine={cos:.6f}  gain={gain:+.3f}dB  nrmse={err:.4f}")

    if args.write_wav or args.asr:
        import soundfile as sf
        d = args.write_wav or "/tmp"
        op, tp = os.path.join(d, "ref_onnx.wav"), os.path.join(d, "ref_tflite.wav")
        sf.write(op, o.astype(np.float32), args.sample_rate)
        sf.write(tp, t[:n].astype(np.float32), args.sample_rate)
        if args.asr:
            from faster_whisper import WhisperModel
            m = WhisperModel("base.en", device="cpu", compute_type="int8")
            for label, w in (("ONNX", op), ("TFLite", tp)):
                segs, _ = m.transcribe(w)
                print(f"ASR[{label}]: {' '.join(s.text for s in segs).strip()!r}")

    if args.temperature == 0.0:
        # F-M5. All three, because cosine alone passed `0.5 * reference` at 1.000000 and
        # the RMSE that would have caught it was printed and thrown away.
        checks = [
            ("cosine", cos >= args.threshold, f"{cos:.6f} >= {args.threshold}"),
            ("gain", abs(gain) <= args.max_gain_db,
             f"|{gain:+.3f}| <= {args.max_gain_db} dB"),
            ("nrmse", err <= args.max_nrmse, f"{err:.4f} <= {args.max_nrmse}"),
        ]
        for name, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:6} {detail}")
        ok = all(c[1] for c in checks)
        print(f"{'PASS' if ok else 'FAIL'} (deterministic)")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

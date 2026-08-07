# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validates the Kotlin port by replicating its exact host logic in Python.

Replicates MatchaG2P + MatchaSynthesizer — per-word tflite G2P + manual
assembly + the integer length-regulator + Euler loop — and synthesizes
through the same fp16 graphs. Confirms the Android pipeline before an
on-device build. There is no mobile app yet, so this file is the spec the
port has to meet rather than a description of one that exists.

The G2P half now lives in `device_g2p.py`, so the same front end can be
compared against `matcha.text.op_g2p` by gate G7 in convert_vat.py without
importing this module's graphs (D-C1 on the device).

Run: python kotlin_replica.py
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import json
import math
import os

import numpy as np

import build_matcha as B
import device_g2p
from ai_edge_litert.compiled_model import CompiledModel

# F-M3. This read every artifact out of `artifacts/` and died at MODULE SCOPE on a bare
# FileNotFoundError for the first one missing. Three of the seven it needs are not
# conversion outputs at all — `dp_g2p_matcha_fp16.tflite`, `g2p_meta.json` and the G2P
# dictionary are VENDORED assets that live under /data/models/litert-community/Matcha-TTS.
# So the script was unrunnable as written, and the error named neither the reason nor the
# directory the file actually sits in.
#
# F-M6. It also pointed at the Phase 0 LJSpeech lane (22.05 kHz, unconditioned) and had no
# way to drive spk or vat. It therefore validated the Kotlin port against a model that is
# not the one shipping — the port's whole point is the conditioned 24 kHz actor.
#
# `--lane vat` selects the conditioned artifacts and the conditioned host pipeline;
# `--lane legacy` keeps the original Phase 0 behaviour byte for byte.
ASSETS = os.environ.get("SONORA_LITERT_ASSETS",
                        "/data/models/litert-community/Matcha-TTS")

LANES = {
    "legacy": dict(art="artifacts", prefix="matcha", sample_rate=22050, time_dim=160,
                   mel_mean=-5.536622, mel_std=2.116101, length_scale=0.95),
    "vat":    dict(art="artifacts_vat", prefix="sonora", sample_rate=24000, time_dim=224,
                   mel_mean=None, mel_std=None, length_scale=1.0),
}

_LANE = os.environ.get("SONORA_KOTLIN_LANE", "legacy")
if _LANE not in LANES:
    raise SystemExit(f"SONORA_KOTLIN_LANE must be one of {sorted(LANES)}, got {_LANE!r}")
_L = LANES[_LANE]

ART = os.path.join(B.WORK, _L["art"])
MAX_TEXT = 256
MAX_MEL = 512
N_FEATS = 80
N_CHANNELS = 192
HOP = 256
SAMPLE_RATE = _L["sample_rate"]
LENGTH_SCALE = _L["length_scale"]
TIME_DIM = _L["time_dim"]


def _find(name, produced_by):
    """Resolve one prerequisite across the conversion artifacts and the vendored assets.

    The whole of F-M3 is that these live in two places and the script only knew about one.
    Accepts a `.gz` sibling, because the vendored dictionary ships compressed and the
    original code opened the bare name.
    """
    for root in (ART, ASSETS):
        for cand in (os.path.join(root, name), os.path.join(root, name + ".gz")):
            if os.path.isfile(cand):
                return cand
    return None


def _preflight():
    """Name every missing prerequisite AND where it comes from, before anything opens one.

    Replaces a module-scope FileNotFoundError that named one file, no directory, and no
    producer — on a script that could not be run at all.
    """
    need = [
        ("config.json", f"convert_{'vat' if _LANE == 'vat' else 'final'}.py"),
        ("emb.npy", f"convert_{'vat' if _LANE == 'vat' else 'final'}.py"),
        ("g2p_meta.json", "convert_g2p_matcha.py, or the vendored assets"),
        ("g2p_dict.txt", "vendored asset (ships as g2p_dict.txt.gz)"),
        # D-C1 on the device. Named here so a missing table is a preflight line next to
        # its producer, rather than a contraction silently resolving to its
        # apostrophe-stripped letters six frames deeper — which is how it went unseen
        # through two corpus versions.
        (device_g2p.CONTRACTIONS_ASSET, "convert_vat.py (from matcha.text.op_g2p)"),
        ("dp_g2p_matcha_fp16.tflite", "convert_g2p_matcha.py (emits the f32 name) "
                                      "or the vendored assets"),
        (f"{_L['prefix']}_textenc_fp16.tflite", f"convert_{'vat' if _LANE == 'vat' else 'final'}.py"),
        (f"{_L['prefix']}_decoder_fp16.tflite", f"convert_{'vat' if _LANE == 'vat' else 'final'}.py"),
        (f"{_L['prefix']}_vocoder{'24k' if _LANE == 'vat' else ''}_fp16.tflite",
         f"convert_{'vat' if _LANE == 'vat' else 'final'}.py"),
    ]
    found, missing = {}, []
    for name, producer in need:
        path = _find(name, producer)
        if path is None:
            missing.append(f"  {name:34} <- {producer}")
        else:
            found[name] = path
    if missing:
        raise SystemExit(
            f"!! kotlin_replica ({_LANE} lane) is missing {len(missing)} of "
            f"{len(need)} prerequisites.\n"
            f"   Looked in: {ART}\n"
            f"          and: {ASSETS}\n\n"
            + "\n".join(missing)
            + "\n\n   Three of these are VENDORED assets, not conversion outputs — that "
              "is F-M3:\n"
              "   the script only ever looked in artifacts/, so it could not run at all.\n"
              "   Point SONORA_LITERT_ASSETS at the asset directory if it has moved."
        )
    return found


_PATHS = _preflight()

with open(_PATHS["config.json"]) as _f:
    _config = json.load(_f)
SYMBOLS = _config["symbols"]
SYM2ID = {s: i for i, s in enumerate(SYMBOLS) if len(s) == 1}
# The lane's mel stats belong to the EXPORT, not to this file. Phase 0's were hardcoded
# here and happened to match; the conditioned lane's are per-corpus and do not.
MEL_MEAN = _L["mel_mean"] if _L["mel_mean"] is not None else _config["mel_mean"]
MEL_STD = _L["mel_std"] if _L["mel_std"] is not None else _config["mel_std"]
# Contract v2: how wide the conditioning tensor is, and what its channels mean. Read from
# the manifest the host will read, not assumed.
VAT_DIM = int(_config.get("vat_dim") or 0)
CONTROL = _config.get("control") or {}

emb = np.load(_PATHS["emb.npy"])
# Multi-speaker lanes carry a speaker table the Phase 0 lane does not have.
spk_emb = None
_spk_path = os.path.join(ART, "spk_emb.npy")
if os.path.isfile(_spk_path):
    spk_emb = np.load(_spk_path)

# The text front end lives in device_g2p.py, which is the executable spec for the mobile
# port and the thing gate G7 checks against the training front end. It used to be inline
# here as `DICT.get(token) or phon_word(token)` — a flat lookup with no contraction table,
# so every apostrophe word on the device took the path D-C1 named on the host
# (`we'll` -> `wˈɛl`, the word *well*). Measured on the probe corpus the day it moved out:
# 69 of 86 probe sentences phonemized differently from the corpus the model trained on.
G2P = device_g2p.DeviceG2P(assets_dir=ASSETS, artifacts_dir=ART)

te = CompiledModel.from_file(_PATHS[f"{_L['prefix']}_textenc_fp16.tflite"])
dec = CompiledModel.from_file(_PATHS[f"{_L['prefix']}_decoder_fp16.tflite"])
voc = CompiledModel.from_file(
    _PATHS[f"{_L['prefix']}_vocoder{'24k' if _LANE == 'vat' else ''}_fp16.tflite"])

def run(model, *inputs):
    """Runs a LiteRT CompiledModel on numpy inputs.

    Args:
        model: The CompiledModel to run.
        *inputs: One numpy array per signature input, in signature order.

    Returns:
        A list of output arrays shaped per the signature details.
    """
    signatures = model.get_signature_list()
    key = list(signatures)[0]
    in_details = model.get_input_tensor_details(key)
    out_details = model.get_output_tensor_details(key)
    input_buffers = model.create_input_buffers(0)
    output_buffers = model.create_output_buffers(0)
    for name, buffer, x in zip(signatures[key]["inputs"], input_buffers,
                               inputs):
        dtype = np.dtype(in_details[name]["dtype"])
        buffer.write(np.ascontiguousarray(x, dtype=dtype))
    model.run_by_index(0, input_buffers, output_buffers)
    outputs = []
    for name, buffer in zip(signatures[key]["outputs"], output_buffers):
        detail = out_details[name]
        count = int(np.prod(detail["shape"]))
        flat = buffer.read(count, np.dtype(detail["dtype"]))
        outputs.append(flat.reshape(detail["shape"]))
    return outputs


def phonemize(text: str):
    """MatchaG2P.phonemize replica: text -> (matcha symbol ids, IPA).

    Args:
        text: Input sentence.

    Returns:
        A (matcha symbol id list, IPA string) tuple.
    """
    ipa_string = G2P.phonemize(text)
    return [SYM2ID[c] for c in ipa_string if c in SYM2ID], ipa_string


def sin_pos_emb(t: float) -> np.ndarray:
    """MatchaSynthesizer.sinusoidalPositionEmbedding replica.

    Args:
        t: ODE time in [0, 1].

    Returns:
        The (1, TIME_DIM) embedding.
    """
    half = TIME_DIM // 2
    out = np.zeros(TIME_DIM, np.float32)
    k = -math.log(10000) / (half - 1)
    for i in range(half):
        angle = 1000.0 * t * math.exp(i * k)
        out[i] = math.sin(angle)
        out[half + i] = math.cos(angle)
    return out[None]


def conditioning(spk_id=None, vat=None, delivery_lane=None):
    """-> (spk vector, vat vector) for the conditioned lane, or (None, None).

    F-M6. This replica had no conditioning path at all, so it validated the Kotlin port
    against the UNCONDITIONED Phase 0 graphs — a model that is not the one shipping. The
    port exists for the conditioned actor; a replica that cannot drive spk or vat proves
    the parts of it that were never in doubt.

    The vector is built through the same rules the manifest publishes, so a lane named
    here means what it means everywhere else. Out-of-range and non-one-hot input is
    refused rather than clamped — a host that sends it gets speech, which is the trap the
    control contract exists to close.
    """
    if VAT_DIM == 0:
        if spk_id is not None or vat is not None or delivery_lane:
            raise SystemExit(
                "!! this lane's graphs take no conditioning (vat_dim 0 in config.json).\n"
                "   Run with SONORA_KOTLIN_LANE=vat against the conditioned artifacts.")
        return None, None

    if spk_emb is None:
        raise SystemExit("!! conditioned lane but no spk_emb.npy beside the graphs")
    sid = 0 if spk_id is None else int(spk_id)
    if not 0 <= sid < spk_emb.shape[0]:
        raise SystemExit(f"!! speaker {sid} outside 0..{spk_emb.shape[0] - 1}")
    spk_vec = spk_emb[sid].reshape(1, -1).astype(np.float32)

    vals = list(vat) if vat is not None else [0.0, 0.0, 0.0]
    cont = CONTROL.get("continuous") or {}
    lo, hi = float(cont.get("min", -1.0)), float(cont.get("max", 1.0))
    n_cont = len(cont.get("channels") or range(3))
    if len(vals) != n_cont:
        raise SystemExit(f"!! --vat wants {n_cont} value(s), got {len(vals)}")
    out = [v for v in vals if not lo <= v <= hi]
    if out:
        raise SystemExit(
            f"!! VAT values must be within [{lo}, {hi}]; out of range: {out}\n"
            "   These are per-speaker z-scores clamped at 2 sigma in derivation, so the\n"
            "   bound is the edge of the TRAINED range, not a display convention.")

    # The delivery block, from the manifest's own vocabulary — the host reads exactly
    # this, so the replica must too.
    groups = CONTROL.get("categorical") or []
    for g in groups:
        lanes = list(g.get("values") or ())
        block = [0.0] * len(lanes)
        if delivery_lane:
            if delivery_lane not in lanes:
                raise SystemExit(
                    f"!! unknown {g.get('group', 'categorical')} value "
                    f"{delivery_lane!r}; the vocabulary is closed at {lanes}")
            block[lanes.index(delivery_lane)] = 1.0
        vals += block
    if not groups and delivery_lane:
        raise SystemExit(
            "!! these artifacts predate contract v2 — config.json declares no delivery\n"
            "   vocabulary, so there is no channel for a lane to drive. Re-export.")
    if len(vals) != VAT_DIM:
        raise SystemExit(f"!! built {len(vals)} channels, graph takes {VAT_DIM}")
    return spk_vec, np.asarray(vals, np.float32)


def synth(phoneme_ids, nsteps: int = 10, seed: int = 0,
          spk_vec=None, vat_vec=None):
    """MatchaSynthesizer.synthesize replica through the fp16 tflite graphs.

    Args:
        phoneme_ids: Matcha symbol ids (without interspersed zeros).
        nsteps: Euler ODE step count.
        seed: RNG seed for the reproducible noise.
        spk_vec: speaker embedding row, conditioned lane only.
        vat_vec: the conditioning vector, conditioned lane only.

    Returns:
        A (waveform, mel frame count) tuple.
    """
    # Per-token expansion, exactly as the host does it: a per-utterance vector repeated
    # across the sequence and masked. Kept here rather than in `conditioning` because the
    # mask lengths are a property of THIS render.
    def _tok(vec, width, mask):
        return (vec.reshape(1, -1, 1) * np.ones((1, 1, width), np.float32)) * mask
    text_length = min(len(phoneme_ids) * 2 + 1, MAX_TEXT)
    ids = np.zeros(MAX_TEXT, np.int64)
    i = 1
    for p in phoneme_ids:
        if i >= MAX_TEXT:
            break
        ids[i] = p
        i += 2
    tmask_row = [1.0 if k < text_length else 0.0 for k in range(MAX_TEXT)]
    tmask = np.array([[tmask_row]], np.float32)
    embx = emb[ids].reshape(1, MAX_TEXT, N_CHANNELS).astype(np.float32)
    te_args = [embx, tmask]
    if vat_vec is not None:
        te_args += [spk_vec, _tok(vat_vec, MAX_TEXT, tmask)]
    outputs = run(te, *te_args)
    mu, logw = ((outputs[0], outputs[1]) if outputs[0].shape[1] == 80
                else (outputs[1], outputs[0]))
    mu = mu[0]  # [80, MAX_TEXT]
    logw = logw[0, 0]  # [MAX_TEXT]
    wceil = np.ceil(np.exp(logw) * tmask[0, 0]) * LENGTH_SCALE
    cum = np.cumsum(wceil)
    ylen = int(min(max(int(cum[-1]), 1), MAX_MEL))
    mu_y = np.zeros((N_FEATS, MAX_MEL), np.float32)
    p = 0
    for f in range(ylen):
        while p < MAX_TEXT - 1 and cum[p] <= f:
            p += 1
        mu_y[:, f] = mu[:, p]
    ymask_row = [1.0 if f < ylen else 0.0 for f in range(MAX_MEL)]
    ymask = np.array([[ymask_row]], np.float32)
    rng = np.random.RandomState(seed)
    x = np.zeros((1, N_FEATS, MAX_MEL), np.float32)
    x[0, :, :ylen] = rng.randn(N_FEATS, ylen).astype(np.float32)
    dt = 1.0 / nsteps
    t = 0.0
    mu_y_batched = mu_y[None]
    vat_y = None if vat_vec is None else _tok(vat_vec, MAX_MEL, ymask)
    for _ in range(nsteps):
        dec_args = [x, mu_y_batched, sin_pos_emb(t), ymask]
        if vat_vec is not None:
            dec_args += [spk_vec, vat_y]
        v = run(dec, *dec_args)[0]
        x = x + dt * v
        t += dt
    mel = np.zeros((1, N_FEATS, MAX_MEL), np.float32)
    mel[0, :, :ylen] = x[0, :, :ylen] * MEL_STD + MEL_MEAN
    wav = run(voc, mel)[0].reshape(-1)[:ylen * HOP]
    return np.clip(wav, -1, 1), ylen


def main():
    """Synthesizes the check sentences through the Kotlin-replica path."""
    import argparse

    import soundfile as sf

    ap = argparse.ArgumentParser(
        description="Replicate the Kotlin host pipeline. Lane is chosen by "
                    "SONORA_KOTLIN_LANE (legacy | vat).")
    ap.add_argument("--spk", type=int, default=None, help="speaker id (conditioned lane)")
    ap.add_argument("--vat", default=None, metavar="V,A,T",
                    help="conditioning, comma-separated floats")
    ap.add_argument("--delivery", default=None, metavar="LANE",
                    help="delivery lane from config.json's vocabulary")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    vat_vals = [float(v) for v in args.vat.split(",")] if args.vat else None
    spk_vec, vat_vec = conditioning(args.spk, vat_vals, args.delivery)

    # espeak is the LEGACY lane's phonemizer and needs the GPL extra. The conditioned lane
    # trains on op_g2p IPA, so a side-by-side against espeak there is not a check, it is
    # two different vocabularies — see matcha/cli.py process_text_for_lane.
    espeak_compare = _LANE == "legacy"

    # The third sentence is contraction-dense on purpose: it is the one an ear can check
    # against D-C1's exemplars, where the defect was audible as the wrong vowel rather
    # than as anything a gate reported.
    sentences = ["Hello, this is Matcha running on the mobile GPU.",
                 "The quick brown fox jumps over the lazy dog.",
                 "Don't worry — we'll know he's right, and it won't be James's fault."]
    print(f"lane={_LANE} sr={SAMPLE_RATE} vat_dim={VAT_DIM} "
          f"spk={args.spk} delivery={args.delivery or 'unknown'}")
    for i, sentence in enumerate(sentences):
        ids, ipa = phonemize(sentence)
        print(f"[{i}] {sentence!r}")
        print(f"    kotlin-G2P IPA: {ipa!r}")
        if espeak_compare:
            try:
                from matcha.text import text_to_sequence
                print("    espeak     IPA: "
                      f"{text_to_sequence(sentence, ['english_cleaners2'])[1]!r}")
            except Exception as exc:  # noqa: BLE001 - the espeak extra is optional
                print(f"    espeak     IPA: unavailable ({type(exc).__name__})")
        wav, ylen = synth(ids, nsteps=args.steps, spk_vec=spk_vec, vat_vec=vat_vec)
        suffix = f"_{args.delivery}" if args.delivery else ""
        out = os.path.join(ART, f"kotlin_{i}{suffix}.wav")
        sf.write(out, wav, SAMPLE_RATE)
        print(f"    {len(ids)} phonemes, {ylen} frames, "
              f"{len(wav) / SAMPLE_RATE:.2f}s -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()

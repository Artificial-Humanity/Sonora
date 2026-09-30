"""WHERE DID OUR EXTRA HUM COME IN? The lineage bench: one voice set through the warm-start chain.

THE QUESTION (2026-09-30)
-------------------------
The baseline bench (`render_ear_baseline.py`) found stock `matcha_vctk` adds a little hum
over its own vocoder (+0.75, p = 0.031) and vat7 ep005 much more (+2.50; +1.75 per
HNR-matched pair, p = 0.002). Every Sonora checkpoint descends from stock by warm-starts:

    matcha_vctk -> derisk-energy ep099 -> vat3 ep099 -> vat5 ep019 -> vat6 ep008 -> vat7 ep005

This renders the same voices through the five Sonora checkpoints to find the step where
the excess entered. No training.

THE DESIGN
----------
Each item pairs one checkpoint's render of a recording with that recording's round trip
through the 24 kHz vocoder, so the gap is the hum that checkpoint adds (as in the baseline
bench). Eight voices, each through all five checkpoints, each checkpoint speaking a
DIFFERENT recording of the voice: 40 items. The reading (`lineage_bench.reading`) compares
each step voice by voice.

⚠ THE VOICES ARE LibriTTS-R train-clean-100's, because derisk-energy and vat3 trained on
that alone. Its 247 speakers hold indices 0-246 in EVERY lineage corpus (checked: the five
`speakers.json` agree on that prefix), so a voice is the same embedding row throughout.

⚠ EACH CHECKPOINT SPEAKS ITS OWN CORPUS'S PHONEMES for the recording. The corpora re-derived
their IPA over time (v3c's apostrophe cleanup among others), and a checkpoint fed another
corpus's phonemes would be judged on a front end it never saw.

⚠ vat5 ep019 IS NO LONGER ON DISK: it was reclaimed on 2026-08-29.
`vat5_finetune/SELECTED.md` records that `warmstart/vat6_init.ckpt` was built from it and
is indistinguishable from it by every check still possible; that stands in for it here. It
carries v6's mel statistics in its hparams, so its buffers are reset to v5's, the data
ep019 trained on (a 0.015 log-mel difference, but the right one).

⚠ EVERY ITEM IS A SENTENCE ITS CHECKPOINT TRAINED ON, and exposure to these voices falls
along the chain: they are ~100% of derisk's and vat3's data, ~72% of v5's, ~9% of v7's. A
rise at a step is therefore confounded with memorisation of these clips fading, and the
pre-registration says so. No recording of these voices is held out of all five corpora.

⚠ NEUTRAL CONDITIONING IS ALL ZEROS AT EITHER WIDTH. derisk-energy and vat3 take 3 values,
the rest 8; `delivery.vat_vector(0, 0, 0, DELIVERY_UNKNOWN)` is eight zeros, so every
checkpoint gets the same neutral input. Same steps, temperature and length scale for all.

    # host
    .venv/bin/python scripts/tools/render_ear_lineage.py select \\
        --out-speakers /data/model-training/sonora/lineage_bench/speakers.json
    # container
    SONORA_EXTRA_DEPS="pyloudnorm" scripts/stages/run_in_rocm.sh \\
        scripts/tools/render_ear_lineage.py render \\
        --speakers /data/model-training/sonora/lineage_bench/speakers.json \\
        --out /data/model-training/sonora/eartest/lineage_hum \\
        --key-out /data/model-training/sonora/.keystage/lineage_hum.key.json
"""

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import baseline_bench as bb                          # noqa: E402
from lib import lineage_bench as lb                           # noqa: E402

ROOT = "/data/model-training/sonora"
T = ROOT + "/logs/train"
# (name, checkpoint, corpus directory under data/), in lineage order.
LINEAGE = [
    ("derisk", T + "/derisk_energy/runs/2026-07-15_00-20-31/checkpoints/checkpoint_epoch=099.ckpt",
     "libritts_r_vat"),
    ("vat3", T + "/vat3_finetune/runs/2026-07-20_01-45-32/checkpoints/checkpoint_epoch=099.ckpt",
     "libritts_r_vat_v2"),
    ("vat5", ROOT + "/warmstart/vat6_init.ckpt", "libritts_r_emilia_vat_v5"),
    ("vat6", T + "/vat6_finetune/runs/2026-08-10_23-48-23/checkpoints/checkpoint_epoch=008.ckpt",
     "libritts_r_emilia_expressive_vat_v6"),
    ("vat7", T + "/vat7_finetune/runs/2026-08-29_01-45-09/checkpoints/"
             "checkpoint_epoch=005_step=0063107.ckpt", "libritts_r_full_vat_v7"),
]
NAMES = [n for n, _c, _d in LINEAGE]
PREFIX = 247            # train-clean-100's speakers: indices 0-246 in every lineage corpus

SETS = {
    "hum": {
        "title": "How much machine is in each one?",
        "ask": ("Two versions of the same sentence. Some are synthesized and some are "
                "recordings passed through a vocoder. Ignore which you would rather listen "
                "to and ignore the reading. Rate EACH clip on its own for the ROBOTIC "
                "quality you have described — the buzz or hum under the voice, the sense of "
                "something mechanical trying to sound human. 0 means you cannot hear it at "
                "all. 5 means it sounds like a Freak-a-Zoid robot. Giving both clips the "
                "same rating is a real answer."),
        "scale": {"max": 5, "anchors": {
            "0": "cannot hear the hum at all",
            "5": "a Freak-a-Zoid robot"}},
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus_rows(corpus):
    """{wav: (speaker index, phonemes)} over TRAIN rows, for the prefix voices only.

    Train only, so every item is a sentence its checkpoint trained on: a recording in one
    corpus's val split and the others' train would make that one item unlike the rest."""
    out = {}
    for name in ("train_op.txt",):
        for line in (Path("data") / corpus / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                wav, spk, phon = line.split("|")[:3]
                if int(spk) < PREFIX:
                    out[wav] = (int(spk), phon)
    return out


# ------------------------------------------------------------------------------ select

def select(args):
    rng = random.Random(args.seed)
    maps = [json.loads((Path("data") / d / "speakers.json").read_text())["libritts_id_to_index"]
            for _n, _c, d in LINEAGE]
    prefix = [{k: v for k, v in m.items() if v < PREFIX} for m in maps]
    if any(p != prefix[0] for p in prefix) or len(prefix[0]) != PREFIX:
        raise SystemExit("REFUSING: the lineage corpora no longer agree on speakers 0-%d, so "
                         "a voice index would mean different people." % (PREFIX - 1))

    rows = {n: corpus_rows(d) for n, _c, d in LINEAGE}
    common = set.intersection(*(set(r) for r in rows.values()))
    for w in common:
        if len({rows[n][w][0] for n in NAMES}) != 1:
            raise SystemExit("REFUSING: %s carries different speaker indices across the "
                             "corpora." % w)

    docs = []
    for pat in args.prior:
        for f in sorted(glob.glob(pat)):
            docs.append(json.loads(Path(f).read_text()))
    heard = bb.prior_speakers(docs)
    hnr = {int(k): v["hnr"] for k, v in
           json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    pool = {s: hnr[s] for s in range(PREFIX) if s in hnr and s not in heard}
    print("%d unheard train-clean-100 voices with a measured HNR (%d heard before)"
          % (len(pool), len(heard & set(range(PREFIX)))))

    spks = lb.spread(pool, args.n)
    clips = {}
    for s in spks:
        ok = []
        for w in sorted(x for x in common if rows["vat7"][x][0] == s):
            if Path(w).is_file():
                sec = sf.info(w).duration
                if args.min_seconds <= sec <= args.max_seconds:
                    ok.append((w, sec))
        clips[s] = ok
    assigned = lb.assign_clips(clips, NAMES, rng, target=args.target_seconds)

    items = []
    for s in spks:
        for n in NAMES:
            w = assigned[s][n]
            items.append({"spk": s, "hnr": hnr[s], "ckpt": n, "wav": w,
                          "seconds": round(sf.info(w).duration, 2),
                          "phonemes": rows[n][w][1]})
        print("  spk%-4d HNR %5.2f  %d usable recordings" % (s, hnr[s], len(clips[s])))

    out = {"rule": ("%d unheard LibriTTS-R train-clean-100 voices spread over speaker HNR; "
                    "for each, one distinct recording per checkpoint, in the TRAIN split of all "
                    "five lineage corpora, %.1f-%.1fs, the %d nearest %.1fs; each checkpoint speaks "
                    "its own corpus's phonemes; seed %d."
                    % (args.n, args.min_seconds, args.max_seconds, len(NAMES),
                       args.target_seconds, args.seed)),
           "lineage": [{"name": n, "ckpt": c, "corpus": d} for n, c, d in LINEAGE],
           "inputs": {"hnr_json": [args.hnr_json, sha256(args.hnr_json)], "prior": args.prior},
           "items": items}
    Path(args.out_speakers).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote %s  sha256 %s" % (args.out_speakers, sha256(args.out_speakers)))


# ------------------------------------------------------------------------------ render

def render(args):
    import torch
    import yaml

    from lib import ear_bench
    from matcha.cli import load_matcha, to_waveform
    from matcha.text import text_to_sequence
    from matcha.utils.audio import mel_spectrogram
    from matcha.utils.utils import intersperse

    spec = json.loads(Path(args.speakers).read_text())
    if [x["name"] for x in spec["lineage"]] != NAMES:
        raise SystemExit("REFUSING: the speakers file names a different lineage.")
    rng = random.Random(args.seed)
    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    sr = bench.sample_rate
    cfg = yaml.safe_load(Path(args.data_config).read_text())
    if cfg["sample_rate"] != sr:
        raise SystemExit("REFUSING: the vocoder is %d Hz and %s says %d."
                         % (sr, args.data_config, cfg["sample_rate"]))

    def roundtrip(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32"))[None]
        mel = mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                              cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                              cfg["f_max"], center=False)
        with torch.no_grad():
            return to_waveform(mel, bench.vocoder, None).numpy()

    spks = sorted({it["spk"] for it in spec["items"]})
    swaps = lb.side_swaps(spks, NAMES, rng)
    order = list(range(len(spec["items"])))
    rng.shuffle(order)
    pair_of = {idx: "item_%02d" % i for i, idx in enumerate(order)}

    made = {}
    for name, ckpt, corpus in LINEAGE:
        model = load_matcha(name, ckpt, ear_bench.DEVICE)
        hp = model.hparams
        # The training data's statistics come from the CORPUS config, not the checkpoint:
        # comparing with its hparams would only prove the load hook copied them.
        want = yaml.safe_load((Path("configs/data") / ("%s.yaml" % corpus)).read_text())
        want = (float(want["data_statistics"]["mel_mean"]),
                float(want["data_statistics"]["mel_std"]))
        got = (float(model.mel_mean), float(model.mel_std))
        if abs(got[0] - want[0]) > 1e-3 or abs(got[1] - want[1]) > 1e-3:
            if name != "vat5":
                raise SystemExit("REFUSING: %s normalises with %r but its corpus %s has %r."
                                 % (name, got, corpus, want))
            print("  %s: stand-in carries %r; set to its own corpus's %r" % (name, got, want))
            model.mel_mean.fill_(want[0])
            model.mel_std.fill_(want[1])
        vat_dim = int(hp["vat_dim"])
        for idx, it in enumerate(spec["items"]):
            if it["ckpt"] != name:
                continue
            pair_key = pair_of[idx]
            seq, _ = text_to_sequence(it["phonemes"], ["no_cleaners"])
            x = torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]
            torch.manual_seed(ear_bench.seed_for(pair_key, args.seed))
            with torch.no_grad():
                o = model.synthesise(x, torch.tensor([x.shape[-1]]),
                                     n_timesteps=ear_bench.N_TIMESTEPS,
                                     temperature=ear_bench.TEMPERATURE,
                                     length_scale=ear_bench.LENGTH_SCALE,
                                     spks=torch.tensor([it["spk"]], dtype=torch.long),
                                     vat=torch.zeros(1, vat_dim), guidance=ear_bench.GUIDANCE)
                made[idx] = to_waveform(o["mel"], bench.vocoder, None).numpy()
        del model

    clips = Path(args.out) / "clips"
    served, truth, limited = [], {}, 0
    for idx in order:
        it, pair_key = spec["items"][idx], pair_of[idx]
        x, xsr = sf.read(it["wav"], dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        if xsr != sr:
            raise SystemExit("REFUSING: %s is %d Hz, not %d." % (it["wav"], xsr, sr))
        audio = {"rt": roundtrip(x), it["ckpt"]: made[idx]}
        lo, hi = "rt", it["ckpt"]
        sides = [("A", hi), ("B", lo)] if swaps[(it["spk"], it["ckpt"])] else [("A", lo), ("B", hi)]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            a, bound = ear_bench.match_loudness(audio[lbl], sr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), a, sr, "PCM_24")
            bench.key[nm] = {"label": lbl, "pair": pair_key, "side": side_key}
            item[side_key] = nm
        served.append(item)
        truth[pair_key] = {"kind": "rt_vs_%s" % it["ckpt"], "ckpt": it["ckpt"],
                           "spk": it["spk"], "hnr": it["hnr"], "A_label": sides[0][1],
                           "B_label": sides[1][1], "source": it["wav"]}
    served.sort(key=lambda i: i["id"])
    for pair_key in sorted(truth):
        t = truth[pair_key]
        print("  %s  %-6s spk%-4d HNR %5.2f" % (pair_key, t["ckpt"], t["spk"], t["hnr"]))

    bench.write(Path(args.out).name, SETS, served,
                {"lineage": spec["lineage"], "speakers": [args.speakers, sha256(args.speakers)],
                 "lufs": args.lufs, "peak_limited_sides": limited},
                key_out=args.key_out)
    k = json.loads(Path(args.key_out).read_text())
    k["items"] = truth
    Path(args.key_out).write_text(json.dumps(k, indent=2))
    if limited:
        print("\n⚠ %d side(s) hit the peak ceiling before reaching %.1f LUFS."
              % (limited, args.lufs))
    print("\nRENDERED %d items. Key: %s" % (len(served), args.key_out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--hnr-json", default=ROOT + "/pitch_error/speaker_hnr_all.json")
    s.add_argument("--prior", nargs="+", default=[ROOT + "/pitch_error/_*.json",
                                                  ROOT + "/eartest/_keys/*.key.json"])
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--min-seconds", type=float, default=3.5)
    s.add_argument("--max-seconds", type=float, default=8.0)
    s.add_argument("--target-seconds", type=float, default=5.0)
    s.add_argument("--seed", type=int, default=20260930)
    s.add_argument("--out-speakers", required=True)

    r = sub.add_parser("render")
    r.add_argument("--speakers", required=True)
    r.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    r.add_argument("--out", required=True)
    r.add_argument("--key-out", required=True)
    r.add_argument("--lufs", type=float, default=-23.0)
    r.add_argument("--salt", default="lineage-hum-v1")
    r.add_argument("--seed", type=int, default=2718)

    args = ap.parse_args()
    (select if args.cmd == "select" else render)(args)


if __name__ == "__main__":
    main()

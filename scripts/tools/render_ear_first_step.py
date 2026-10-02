"""DID OUR FIRST STEP ADD THE HUM? The first-step bench: stock against derisk-energy, vat7 as check.

THE QUESTION (2026-10-02)
-------------------------
The baseline bench (`render_ear_baseline.py`) found vat7 ep005 adds +1.75 more hum than stock
`matcha_vctk` per HNR-matched pair (p = 0.002). The lineage bench (`render_ear_lineage.py`)
was INVALID (vat7 p = 0.094 with 8 voices), but its gap looked flat from derisk-energy to
vat7 and followed the voice. The one step it did not test is the first:

    matcha_vctk -> derisk-energy ep099

which changed the corpus (VCTK -> LibriTTS-R train-clean-100), the sample rate and vocoder
(22.05 kHz universal -> 24 kHz fine-tune), and added V/A/T conditioning, all at once. Does
derisk-energy already hum more than stock at the same voice roughness? No training.

THE DESIGN
----------
16 matches. Each is one VCTK voice and one unheard LibriTTS-R train-clean-100 voice of
nearly the same speaker HNR (`baseline_bench.match_speakers`, spread over the range both
cover), with three items:

  stock    stock on a VCTK recording, against that recording through the stock vocoder
  derisk   derisk-energy on a LibriTTS-R recording, against it through the 24 kHz vocoder
  vat7     vat7 ep005 on a DIFFERENT recording of the same LibriTTS-R voice, likewise

48 items. vat7 over stock repeats the baseline bench's result: if this bench cannot hear
that, it says nothing (`first_step_bench.reading`).

⚠ THE STOCK HALF IS THE BASELINE BENCH'S, UNCHANGED: upstream's symbol table, VITS's espeak
phonemes, the universal vocoder with its denoiser, resampled to 24 kHz after vocoding. See
`render_ear_baseline.py` for why each is so. VCTK speaker HNR is read from the baseline
bench's speakers file (pinned by sha256) rather than measured again.

⚠ EACH SONORA CHECKPOINT SPEAKS ITS OWN CORPUS'S PHONEMES with all-zero conditioning, as in
the lineage bench, and its mel statistics are checked against its corpus config.

⚠ CLIP LENGTHS ARE MATCHED: each LibriTTS-R recording is among the nearest in length to its
match's VCTK recording. More seconds is more exposure to the hum.

⚠ ALL THREE ITEMS OF A MATCH ARE SERVED THE SAME WAY ROUND (`first_step_bench.side_plan`).

⚠ VOICES ARE UNHEARD: LibriTTS-R indices from every earlier key and pitch-error file, AND
the VCTK voices the baseline bench used, are excluded.

    # host
    .venv/bin/python scripts/tools/render_ear_first_step.py select \\
        --out-speakers /data/model-training/sonora/first_step_bench/speakers.json
    # container
    SONORA_EXTRA_DEPS="pyloudnorm soxr" scripts/stages/run_in_rocm.sh \\
        scripts/tools/render_ear_first_step.py render \\
        --speakers /data/model-training/sonora/first_step_bench/speakers.json \\
        --out /data/model-training/sonora/eartest/first_step_hum \\
        --key-out /data/model-training/sonora/.keystage/first_step_hum.key.json
"""

import argparse
import glob
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import baseline_bench as bb                          # noqa: E402
from lib import first_step_bench as fs                        # noqa: E402
from lib import lineage_bench as lb                           # noqa: E402

ROOT = "/data/model-training/sonora"
T = ROOT + "/logs/train"
BASELINE_DIR = ROOT + "/baseline_bench"
VCTK_ZIP = "/data/model-training/datasets/VCTK/VCTK-Corpus-0.92.zip"
# (name, checkpoint, corpus directory under data/), as in render_ear_lineage.LINEAGE.
OURS = [
    ("derisk", T + "/derisk_energy/runs/2026-07-15_00-20-31/checkpoints/checkpoint_epoch=099.ckpt",
     "libritts_r_vat"),
    ("vat7", T + "/vat7_finetune/runs/2026-08-29_01-45-09/checkpoints/"
             "checkpoint_epoch=005_step=0063107.ckpt", "libritts_r_full_vat_v7"),
]
NAMES = [n for n, _c, _d in OURS]
PREFIX = 247            # train-clean-100's speakers: indices 0-246 in every lineage corpus

# Stock Matcha, exactly as `render_ear_baseline.py` renders it.
STOCK_SR, STOCK_MEL = 22050, (1024, 80, 22050, 256, 1024, 0, 8000)
STOCK_MEL_MEAN, STOCK_MEL_STD = -6.630575, 2.482914
DENOISER_STRENGTH = 0.00025

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


def read_vits(path):
    """VITS filelist -> {utterance: (speaker, sid, text)}."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p, sid, text = line.split("|", 2)
        spk, utt = p.split("/")[1], Path(p).stem
        out[utt] = (spk, int(sid), text)
    return out


def vctk_path(utt):
    return "wav48_silence_trimmed/%s/%s_mic1.flac" % (utt.split("_")[0], utt)


def vctk_audio(z, utt):
    x, sr = sf.read(io.BytesIO(z.read(vctk_path(utt))), dtype="float64")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def corpus_rows(corpus):
    """{wav: (speaker index, phonemes)} over TRAIN rows, for the prefix voices only."""
    out = {}
    for line in (Path("data") / corpus / "train_op.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            wav, spk, phon = line.split("|")[:3]
            if int(spk) < PREFIX:
                out[wav] = (int(spk), phon)
    return out


# ------------------------------------------------------------------------------ select

def select(args):
    rng = random.Random(args.seed)
    maps = [json.loads((Path("data") / d / "speakers.json").read_text())["libritts_id_to_index"]
            for _n, _c, d in OURS]
    prefix = [{k: v for k, v in m.items() if v < PREFIX} for m in maps]
    if prefix[0] != prefix[1] or len(prefix[0]) != PREFIX:
        raise SystemExit("REFUSING: the two corpora no longer agree on speakers 0-%d, so a "
                         "voice index would mean different people." % (PREFIX - 1))
    rows = {n: corpus_rows(d) for n, _c, d in OURS}
    common = set(rows["derisk"]) & set(rows["vat7"])
    for w in common:
        if rows["derisk"][w][0] != rows["vat7"][w][0]:
            raise SystemExit("REFUSING: %s carries different speaker indices." % w)

    docs = []
    for pat in args.prior:
        for f in sorted(glob.glob(pat)):
            docs.append(json.loads(Path(f).read_text()))
    heard, vheard = bb.prior_speakers(docs), fs.heard_vctk(docs)

    raw, cleaned = read_vits(args.vctk_filelist), read_vits(args.vctk_cleaned)
    if set(raw) != set(cleaned):
        raise SystemExit("REFUSING: the raw and .cleaned VITS filelists list different "
                         "utterances.")
    z = zipfile.ZipFile(args.vctk_zip)
    have = set(z.namelist())
    by_spk = {}
    for utt, (spk, _sid, _t) in raw.items():
        if vctk_path(utt) in have:
            by_spk.setdefault(spk, []).append(utt)
    base = json.loads(Path(args.vctk_hnr_from).read_text())
    vctk = {s: h for s, h in base["vctk_hnr"].items() if s not in vheard and s in by_spk}

    # Each voice's recordings in the train split of BOTH corpora, within the length bounds.
    # A voice with fewer than two is not eligible, so the spread is decided before any clip.
    hnr = {int(k): v["hnr"] for k, v in
           json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    lib_clips = {}
    for w in sorted(common):
        s = rows["vat7"][w][0]
        if s in hnr and s not in heard and Path(w).is_file():
            sec = sf.info(w).duration
            if args.min_seconds <= sec <= args.max_seconds:
                lib_clips.setdefault(s, []).append((w, sec))
    pool = {s: hnr[s] for s, c in lib_clips.items() if len(c) >= len(NAMES)}
    print("%d VCTK voices unheard (%d heard); %d LibriTTS-R train-clean-100 voices eligible "
          "(%d heard)" % (len(vctk), len(vheard), len(pool), len(heard & set(range(PREFIX)))))

    matches = bb.match_speakers(vctk, pool, args.n, set(), args.max_gap)
    items = []
    for i, m in enumerate(matches):
        cand = []
        for u in sorted(by_spk[m["vctk"]]):
            sec = sf.info(io.BytesIO(z.read(vctk_path(u)))).duration
            if args.min_seconds <= sec <= args.max_seconds:
                cand.append((u, sec))
        if not cand:
            raise SystemExit("REFUSING: VCTK %s has no mic1 clip of %.1f-%.1fs."
                             % (m["vctk"], args.min_seconds, args.max_seconds))
        utt, vsec = rng.choice(cand)
        _spk, sid, text = cleaned[utt]
        got = lb.assign_clips({m["libri"]: lib_clips[m["libri"]]}, NAMES, rng,
                              target=vsec)[m["libri"]]
        item = {"match": i, "vctk": m["vctk"], "vctk_hnr": m["vctk_hnr"], "vctk_sid": sid,
                "vctk_utt": utt, "vctk_seconds": round(vsec, 2), "vctk_phonemes": text,
                "vctk_text": raw[utt][2], "libri": m["libri"], "libri_hnr": m["libri_hnr"]}
        for n in NAMES:
            w = got[n]
            item[n] = {"wav": w, "seconds": round(sf.info(w).duration, 2),
                       "phonemes": rows[n][w][1]}
        items.append(item)
        print("  match %2d  VCTK %-5s HNR %5.2f %4.1fs  <->  spk%-4d HNR %5.2f  %4.1fs / %4.1fs"
              % (i, m["vctk"], m["vctk_hnr"], vsec, m["libri"], m["libri_hnr"],
                 item["derisk"]["seconds"], item["vat7"]["seconds"]))

    out = {"rule": ("%d unheard VCTK voices spread over speaker HNR, each matched to the "
                    "nearest unheard LibriTTS-R train-clean-100 voice (max gap %.2f dB); one "
                    "VCTK clip of %.1f-%.1fs each, and for each of %s a distinct LibriTTS-R "
                    "recording in the TRAIN split of both corpora, among the %d nearest in "
                    "length to it; each checkpoint speaks its own corpus's phonemes; seed %d."
                    % (args.n, args.max_gap, args.min_seconds, args.max_seconds,
                       "/".join(NAMES), len(NAMES), args.seed)),
           "ours": [{"name": n, "ckpt": c, "corpus": d} for n, c, d in OURS],
           "inputs": {"vctk_filelist": [args.vctk_filelist, sha256(args.vctk_filelist)],
                      "vctk_cleaned": [args.vctk_cleaned, sha256(args.vctk_cleaned)],
                      "vctk_hnr_from": [args.vctk_hnr_from, sha256(args.vctk_hnr_from)],
                      "hnr_json": [args.hnr_json, sha256(args.hnr_json)],
                      "prior": args.prior},
           "items": items}
    Path(args.out_speakers).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_speakers).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote %s  sha256 %s" % (args.out_speakers, sha256(args.out_speakers)))


# ------------------------------------------------------------------------------ render

def render(args):
    import soxr
    import torch
    import yaml

    from lib import ear_bench
    from matcha.cli import load_matcha, load_vocoder, to_waveform
    from matcha.text import text_to_sequence
    from matcha.utils.audio import mel_spectrogram
    from matcha.utils.utils import intersperse

    spec = json.loads(Path(args.speakers).read_text())
    if [x["name"] for x in spec["ours"]] != NAMES:
        raise SystemExit("REFUSING: the speakers file names different checkpoints.")
    rng = random.Random(args.seed)
    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    sr = bench.sample_rate
    cfg = yaml.safe_load(Path(args.data_config).read_text())
    mel_keys = ("n_fft", "n_feats", "sample_rate", "hop_length", "win_length", "f_min", "f_max")
    for _n, _c, corpus in OURS:
        c = yaml.safe_load((Path("configs/data") / ("%s.yaml" % corpus)).read_text())
        if any(c[k] != cfg[k] for k in mel_keys):
            raise SystemExit("REFUSING: %s's mel differs from %s, so one round trip cannot "
                             "serve both checkpoints." % (corpus, args.data_config))
    if cfg["sample_rate"] != sr:
        raise SystemExit("REFUSING: the vocoder is %d Hz and %s says %d."
                         % (sr, args.data_config, cfg["sample_rate"]))
    # `mel_spectrogram` caches its mel basis by fmax and its window by device only.
    if STOCK_MEL[6] == cfg["f_max"] or STOCK_MEL[4] != cfg["win_length"]:
        raise SystemExit("REFUSING: the stock and ours mel configs now collide in "
                         "matcha.utils.audio's cache.")

    # --- stock, as render_ear_baseline.py renders it
    stock = load_matcha("matcha_vctk", args.stock_ckpt, ear_bench.DEVICE)
    if (stock.n_spks, stock.n_vocab) != (109, 178):
        raise SystemExit("REFUSING: the stock checkpoint has n_spks %d, n_vocab %d; "
                         "matcha_vctk has 109 and 178." % (stock.n_spks, stock.n_vocab))
    got = (float(stock.mel_mean), float(stock.mel_std))
    if abs(got[0] - STOCK_MEL_MEAN) > 1e-3 or abs(got[1] - STOCK_MEL_STD) > 1e-3:
        raise SystemExit("REFUSING: the stock model loaded mel stats %r, not VCTK's %r."
                         % (got, (STOCK_MEL_MEAN, STOCK_MEL_STD)))
    s_voc, s_den = load_vocoder("hifigan_univ_v1", args.stock_vocoder, ear_bench.DEVICE)
    z = zipfile.ZipFile(args.vctk_zip)

    def stock_wave(mel):
        with torch.no_grad():
            w = to_waveform(mel, s_voc, s_den, DENOISER_STRENGTH).numpy()
        return soxr.resample(w.astype("float64"), STOCK_SR, sr).astype("float32")

    def ours_rt(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32"))[None]
        mel = mel_spectrogram(y, *(cfg[k] for k in mel_keys), center=False)
        with torch.no_grad():
            return to_waveform(mel, bench.vocoder, None).numpy()

    plan = [(it, f) for it in spec["items"] for f in fs.FAMILIES]
    rng.shuffle(plan)
    pair_of = {(it["match"], f): "item_%02d" % i for i, (it, f) in enumerate(plan)}
    swaps = fs.side_plan(len(spec["items"]), rng)

    made = {}
    for it in spec["items"]:
        pair_key = pair_of[(it["match"], "stock")]
        x, xsr = vctk_audio(z, it["vctk_utt"])
        y = torch.from_numpy(soxr.resample(x, xsr, STOCK_SR).astype("float32"))[None]
        ids = intersperse(bb.encode_original(it["vctk_phonemes"]), 0)
        xt = torch.tensor(ids, dtype=torch.long)[None]
        torch.manual_seed(ear_bench.seed_for(pair_key, args.seed))
        with torch.no_grad():
            o = stock.synthesise(xt, torch.tensor([xt.shape[-1]]),
                                 n_timesteps=ear_bench.N_TIMESTEPS,
                                 temperature=ear_bench.TEMPERATURE,
                                 spks=torch.tensor([it["vctk_sid"]], dtype=torch.long),
                                 length_scale=ear_bench.LENGTH_SCALE)
        made[pair_key] = {"stockrt": stock_wave(mel_spectrogram(y, *STOCK_MEL, center=False)),
                          "stock": stock_wave(o["mel"])}
    del stock

    for name, ckpt, corpus in OURS:
        model = load_matcha(name, ckpt, ear_bench.DEVICE)
        want = yaml.safe_load((Path("configs/data") / ("%s.yaml" % corpus)).read_text())
        want = (float(want["data_statistics"]["mel_mean"]),
                float(want["data_statistics"]["mel_std"]))
        got = (float(model.mel_mean), float(model.mel_std))
        if abs(got[0] - want[0]) > 1e-3 or abs(got[1] - want[1]) > 1e-3:
            raise SystemExit("REFUSING: %s normalises with %r but its corpus %s has %r."
                             % (name, got, corpus, want))
        vat_dim = int(model.hparams["vat_dim"])
        for it in spec["items"]:
            pair_key, src = pair_of[(it["match"], name)], it[name]
            x, xsr = sf.read(src["wav"], dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            if xsr != sr:
                raise SystemExit("REFUSING: %s is %d Hz, not %d." % (src["wav"], xsr, sr))
            seq, _ = text_to_sequence(src["phonemes"], ["no_cleaners"])
            xt = torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]
            torch.manual_seed(ear_bench.seed_for(pair_key, args.seed))
            with torch.no_grad():
                o = model.synthesise(xt, torch.tensor([xt.shape[-1]]),
                                     n_timesteps=ear_bench.N_TIMESTEPS,
                                     temperature=ear_bench.TEMPERATURE,
                                     length_scale=ear_bench.LENGTH_SCALE,
                                     spks=torch.tensor([it["libri"]], dtype=torch.long),
                                     vat=torch.zeros(1, vat_dim), guidance=ear_bench.GUIDANCE)
                made[pair_key] = {name + "rt": ours_rt(x),
                                  name: to_waveform(o["mel"], bench.vocoder, None).numpy()}
        del model

    clips = Path(args.out) / "clips"
    served, truth, limited = [], {}, 0
    for it, fam in plan:
        pair_key = pair_of[(it["match"], fam)]
        lo, hi = fam + "rt", fam
        sides = [("A", hi), ("B", lo)] if swaps[it["match"]][fam] else [("A", lo), ("B", hi)]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            a, bound = ear_bench.match_loudness(made[pair_key][lbl], sr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), a, sr, "PCM_24")
            bench.key[nm] = {"label": lbl, "pair": pair_key, "side": side_key}
            item[side_key] = nm
        served.append(item)
        # ⚠ A VCTK name goes under `vctk_spk`, never `spk`: later benches read `spk` from
        # every key in `_keys/` as a LibriTTS-R index to exclude.
        if fam == "stock":
            who, hnr, src = {"vctk_spk": it["vctk"]}, it["vctk_hnr"], it["vctk_utt"]
        else:
            who, hnr, src = {"spk": it["libri"]}, it["libri_hnr"], it[fam]["wav"]
        truth[pair_key] = dict(who, kind="%srt_vs_%s" % (fam, fam), family=fam,
                               match=it["match"], hnr=hnr, A_label=sides[0][1],
                               B_label=sides[1][1], source=src)
        print("  %s  %-6s match %2d  HNR %5.2f" % (pair_key, fam, it["match"], hnr))
    served.sort(key=lambda i: i["id"])

    bench.write(Path(args.out).name, SETS, served,
                {"stock_ckpt": args.stock_ckpt, "ours": spec["ours"],
                 "speakers": [args.speakers, sha256(args.speakers)], "lufs": args.lufs,
                 "peak_limited_sides": limited},
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
    s.add_argument("--vctk-zip", default=VCTK_ZIP)
    s.add_argument("--vctk-filelist",
                   default=BASELINE_DIR + "/vctk_audio_sid_text_train_filelist.txt")
    s.add_argument("--vctk-cleaned",
                   default=BASELINE_DIR + "/vctk_audio_sid_text_train_filelist.txt.cleaned")
    s.add_argument("--vctk-hnr-from", default=BASELINE_DIR + "/speakers.json")
    s.add_argument("--hnr-json", default=ROOT + "/pitch_error/speaker_hnr_all.json")
    s.add_argument("--prior", nargs="+", default=[ROOT + "/pitch_error/_*.json",
                                                  ROOT + "/eartest/_keys/*.key.json"])
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--min-seconds", type=float, default=3.0)
    s.add_argument("--max-seconds", type=float, default=8.0)
    s.add_argument("--max-gap", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=20261002)
    s.add_argument("--out-speakers", required=True)

    r = sub.add_parser("render")
    r.add_argument("--speakers", required=True)
    r.add_argument("--stock-ckpt", default=ROOT + "/warmstart/matcha_vctk.ckpt")
    r.add_argument("--stock-vocoder",
                   default=BASELINE_DIR + "/UNIVERSAL_V1_g_02500000")  # a readable copy
    r.add_argument("--vctk-zip", default=VCTK_ZIP)
    r.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    r.add_argument("--out", required=True)
    r.add_argument("--key-out", required=True)
    r.add_argument("--lufs", type=float, default=-23.0)
    r.add_argument("--salt", default="first-step-hum-v1")
    r.add_argument("--seed", type=int, default=1618)

    args = ap.parse_args()
    (select if args.cmd == "select" else render)(args)


if __name__ == "__main__":
    main()

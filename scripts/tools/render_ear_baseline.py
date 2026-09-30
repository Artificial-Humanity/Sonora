"""DOES STOCK MATCHA HUM TOO? The baseline bench: stock `matcha_vctk` against ours.

THE QUESTION (2026-09-30)
-------------------------
Every lever tried on the hum has come back null: vocoder fine-tune, stale mel buffers, ODE
steps, temperature, rough-voice data weighting, and the DiT decoder. Every Sonora
checkpoint descends by warm-starts from stock `matcha_vctk` (derisk-energy -> vat3 -> vat5
-> vat6 -> vat7). So: is the hum in Matcha-TTS itself, or did our training history put it
there?

  * Stock hums as much as ours -> the recipe is at fault; our history is exonerated.
  * Stock is clean and ours hums -> our history introduced it; bisect the lineage next.

THE DESIGN
----------
Each item is ONE source recording served two ways: the model's render of its sentence, and
the recording's round trip through the same vocoder. The within-item difference is the hum
the model adds beyond its vocoder, which is what the `ceiling` bench measured for ours
(+1.75, p = 0.015).

Half the items are stock (VCTK speaker, stock model, stock vocoder `UNIVERSAL_V1` with its
denoiser) and half ours (LibriTTS-R speaker, vat7 ep005, the 24 kHz vocoder). The halves are
MATCHED ONE TO ONE ON SPEAKER HNR, measured with the same estimator
(`scripts/lib/periodicity.py`, the `measure_speaker_hnr.py` settings, audio at 24 kHz):
severity tracks HNR in every bench so far, so an unmatched comparison would measure the two
corpora's roughness mix instead of the two models.

⚠ THE STOCK MODEL READS UPSTREAM'S SYMBOL TABLE, NOT OURS. Sonora replaced a duplicated "'"
in the table with "ᵊ", which moves "'" and "ᵻ" — see `baseline_bench.original_matcha_symbols`.

⚠ ITS INPUT IS VITS'S ESPEAK PHONEMIZATION (the `.cleaned` filelist), NOT A BYTE-EXACT COPY
OF WHAT IT TRAINED ON. Upstream phonemized on the fly with `english_cleaners2` and whatever
espeak-ng it ran. Checked 2026-09-30 on the 12 bench sentences: upstream's cleaner under
espeak-ng 1.52 agrees on 8, and the other 4 differ in a stress mark on "I", one vowel
(oː/ɔː), and "US", which 1.52 reads as the word "us" and VITS as "U.S.". Neither is exact;
VITS's is the older espeak and reads "US" correctly, so it is used. No espeak runs here.

⚠ THE FAMILIES ARE AUDIBLY DIFFERENT. Stock output has nothing above 11 kHz and VCTK voices
are mostly British, so the listener can tell a stock item from ours. Each gap is within one
item (same vocoder, same band), so that cannot move a gap directly — but the matched
contrast is not blind to family, and the listener knows the hypothesis.

⚠ CLIP LENGTHS ARE MATCHED. More seconds is more exposure to the hum, so each LibriTTS-R
clip is the one closest in length to its matched VCTK clip.

⚠ SAME RENDER SETTINGS FOR BOTH MODELS: `ear_bench`'s steps, temperature and length scale.
Stock's CLI would speak VCTK at length scale 0.85; the bench does not, so that no setting
differs between the families.

⚠ ONE SAMPLE RATE SERVED. Stock renders are 22.05 kHz and are resampled to 24 kHz after
vocoding, so both halves reach the listener identically.

Two steps, both deterministic:

    # host: measure VCTK HNR, match speakers, fix every source clip -> speakers file
    .venv/bin/python scripts/tools/render_ear_baseline.py select \\
        --out-speakers /data/model-training/sonora/baseline_bench/speakers.json

    # container: render the bench from that file
    SONORA_EXTRA_DEPS="pyloudnorm soxr" scripts/stages/run_in_rocm.sh \\
        scripts/tools/render_ear_baseline.py render \\
        --speakers /data/model-training/sonora/baseline_bench/speakers.json \\
        --ours-ckpt /data/.../checkpoint_epoch=005_step=0063107.ckpt \\
        --out /data/model-training/sonora/eartest/baseline_hum \\
        --key-out /data/model-training/sonora/.keystage/baseline_hum.key.json
"""

import argparse
import glob
import hashlib
import io
import json
import random
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import baseline_bench as bb                          # noqa: E402
from lib import periodicity                                   # noqa: E402
from lib.corpus_filelist import read_corpus                   # noqa: E402
from matcha.delivery import VAT_DIM                           # noqa: E402

ROOT = "/data/model-training/sonora"
BENCH_DIR = ROOT + "/baseline_bench"
VCTK_ZIP = "/data/model-training/datasets/VCTK/VCTK-Corpus-0.92.zip"
HNR_SR = 24000
# measure_speaker_hnr.py's settings, so both corpora sit on one HNR axis.
HNR_ARGS = dict(fmin=55.0, fmax=350.0, rms_floor=0.01, periodicity=0.3)

# Stock Matcha's mel, from upstream `configs/data/vctk.yaml` (inherits ljspeech's).
# Positional, in `matcha.utils.audio.mel_spectrogram`'s order: n_fft, n_mels, sr, hop, win,
# f_min, f_max.
STOCK_SR, STOCK_MEL = 22050, (1024, 80, 22050, 256, 1024, 0, 8000)
STOCK_MEL_MEAN, STOCK_MEL_STD = -6.630575, 2.482914
DENOISER_STRENGTH = 0.00025   # upstream CLI default for the universal vocoder

KIND_STOCK, KIND_OURS = "stockrt_vs_stock", "oursrt_vs_ours"

SETS = {
    "hum": {
        "title": "How much machine is in each one?",
        "ask": ("Two versions of the same sentence. Some are synthesized and some are "
                "recordings passed through a vocoder. Ignore which you would rather listen to and ignore the "
                "reading. Rate EACH clip on its own for the ROBOTIC quality you have "
                "described — the buzz or hum under the voice, the sense of something "
                "mechanical trying to sound human. 0 means you cannot hear it at all. 5 "
                "means it sounds like a Freak-a-Zoid robot. Giving both clips the same "
                "rating is a real answer."),
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


def vctk_audio(z, utt):
    spk = utt.split("_")[0]
    x, sr = sf.read(io.BytesIO(z.read("wav48_silence_trimmed/%s/%s_mic1.flac" % (spk, utt))),
                    dtype="float64")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def speaker_hnr(clips):
    """Median over clips of each clip's median voiced-frame HNR (measure_speaker_hnr.py)."""
    import soxr
    h = []
    for x, sr in clips:
        _f0, hh = periodicity.frames(soxr.resample(x, sr, HNR_SR), HNR_SR, **HNR_ARGS)
        if len(hh):
            h.append(float(np.median(hh)))
    return round(statistics.median(h), 3) if h else None


# ------------------------------------------------------------------------------ select

def select(args):
    rng = random.Random(args.seed)
    raw, cleaned = read_vits(args.vctk_filelist), read_vits(args.vctk_cleaned)
    if set(raw) != set(cleaned):
        raise SystemExit("REFUSING: the raw and .cleaned VITS filelists list different "
                         "utterances.")
    z = zipfile.ZipFile(args.vctk_zip)
    have = set(z.namelist())
    by_spk = {}
    for utt, (spk, sid, _t) in raw.items():
        if "wav48_silence_trimmed/%s/%s_mic1.flac" % (spk, utt) in have:
            by_spk.setdefault(spk, []).append(utt)

    print("measuring HNR for %d VCTK speakers x %d clips" % (len(by_spk),
                                                            args.clips_per_speaker))
    vctk_hnr = {}
    for spk in sorted(by_spk):
        utts = rng.sample(sorted(by_spk[spk]), min(args.clips_per_speaker, len(by_spk[spk])))
        h = speaker_hnr([vctk_audio(z, u) for u in utts])
        if h is not None:
            vctk_hnr[spk] = h

    docs = []
    for pat in args.prior:
        for f in sorted(glob.glob(pat)):
            docs.append(json.loads(Path(f).read_text()))
    exclude = bb.prior_speakers(docs)
    print("excluding %d speakers already heard in earlier benches" % len(exclude))

    libri_all = {int(k): v for k, v in
                 json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    rows = {}
    for line, wav, s, _n in read_corpus(args.corpus, "train", VAT_DIM):
        rows.setdefault(s, []).append((wav, line.split("|")[2]))

    def libri_clips(s):
        """[(wav, phonemes, seconds)] of this speaker's train rows long enough to serve."""
        out = []
        for wav, phon in sorted(rows.get(s, [])):
            if Path(wav).is_file():
                sec = sf.info(wav).duration
                if sec >= args.min_seconds:
                    out.append((wav, phon, sec))
        return out

    def vctk_clip(spk):
        """(utterance, seconds), drawn among this speaker's clips long enough to serve."""
        cand = []
        for u in sorted(by_spk[spk]):
            sec = sf.info(io.BytesIO(z.read("wav48_silence_trimmed/%s/%s_mic1.flac"
                                            % (spk, u)))).duration
            if sec >= args.min_seconds:
                cand.append((u, sec))
        return rng.choice(cand) if cand else None

    # A matched LibriTTS-R speaker with no clip long enough is excluded and the match
    # re-run, so the spread over HNR is decided before any clip is.
    dropped = set()
    while True:
        libri = {s: v["hnr"] for s, v in libri_all.items() if s in rows and s not in dropped}
        matches = bb.match_speakers(vctk_hnr, libri, args.n, exclude, args.max_gap)
        lib_clips = {m["libri"]: libri_clips(m["libri"]) for m in matches}
        bad = {s for s, c in lib_clips.items() if not c}
        if not bad:
            break
        dropped |= bad

    items = []
    for i, m in enumerate(matches):
        pick = vctk_clip(m["vctk"])
        if pick is None:
            raise SystemExit("REFUSING: VCTK %s has no mic1 clip of at least %.1fs."
                             % (m["vctk"], args.min_seconds))
        utt, vsec = pick
        spk, sid, text = cleaned[utt]
        by_wav = {w: (ph, sec) for w, ph, sec in lib_clips[m["libri"]]}
        wav = bb.closest_duration([(w, sec) for w, (_p, sec) in by_wav.items()], vsec)
        phon, lsec = by_wav[wav]
        items.append({"match": i, "vctk": m["vctk"], "vctk_hnr": m["vctk_hnr"],
                      "vctk_sid": sid, "vctk_utt": utt, "vctk_seconds": round(vsec, 2),
                      "vctk_phonemes": text, "vctk_text": raw[utt][2],
                      "libri": m["libri"], "libri_hnr": m["libri_hnr"],
                      "libri_partition": libri_all[m["libri"]].get("partition"),
                      "libri_wav": wav, "libri_seconds": round(lsec, 2),
                      "libri_phonemes": phon})
        print("  match %2d  VCTK %-5s HNR %5.2f %4.1fs  <->  spk%-5d HNR %5.2f %4.1fs"
              % (i, m["vctk"], m["vctk_hnr"], vsec, m["libri"], m["libri_hnr"], lsec))

    out = {"rule": ("%d VCTK speakers spread over speaker HNR, each matched to the "
                    "nearest unused LibriTTS-R v7 train speaker not heard in any earlier "
                    "bench (max gap %.2f dB); one VCTK clip of >= %.1fs each, and the LibriTTS-R "
                    "clip closest to it in length; seed %d."
                    % (args.n, args.max_gap, args.min_seconds, args.seed)),
           "inputs": {"vctk_filelist": [args.vctk_filelist, sha256(args.vctk_filelist)],
                      "vctk_cleaned": [args.vctk_cleaned, sha256(args.vctk_cleaned)],
                      "hnr_json": [args.hnr_json, sha256(args.hnr_json)],
                      "corpus": args.corpus, "prior": args.prior},
           "vctk_hnr": vctk_hnr, "items": items}
    Path(args.out_speakers).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote %s  sha256 %s" % (args.out_speakers, sha256(args.out_speakers)))


# ------------------------------------------------------------------------------ render

def render(args):
    import soxr
    import torch
    import yaml

    from lib import ear_bench
    from matcha import delivery
    from matcha.cli import load_matcha, load_vocoder, to_waveform
    from matcha.utils.audio import mel_spectrogram
    from matcha.utils.utils import intersperse

    spec = json.loads(Path(args.speakers).read_text())
    rng = random.Random(args.seed)
    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    sr = bench.sample_rate
    cfg = yaml.safe_load(Path(args.data_config).read_text())
    if cfg["sample_rate"] != sr:
        raise SystemExit("REFUSING: the vocoder is %d Hz and %s says %d."
                         % (sr, args.data_config, cfg["sample_rate"]))

    stock = load_matcha("matcha_vctk", args.stock_ckpt, ear_bench.DEVICE)
    if (stock.n_spks, stock.n_vocab) != (109, 178):
        raise SystemExit("REFUSING: the stock checkpoint has n_spks %d, n_vocab %d; "
                         "matcha_vctk has 109 and 178." % (stock.n_spks, stock.n_vocab))
    # `mel_spectrogram` caches its mel basis by fmax and its window by device only, so
    # the two families' mels share a cache: correct only while the fmax values differ and
    # the windows are the same length.
    if STOCK_MEL[6] == cfg["f_max"] or STOCK_MEL[4] != cfg["win_length"]:
        raise SystemExit("REFUSING: the stock and ours mel configs now collide in "
                         "matcha.utils.audio's cache (fmax %s/%s, win %s/%s)."
                         % (STOCK_MEL[6], cfg["f_max"], STOCK_MEL[4], cfg["win_length"]))
    got = (float(stock.mel_mean), float(stock.mel_std))
    if abs(got[0] - STOCK_MEL_MEAN) > 1e-3 or abs(got[1] - STOCK_MEL_STD) > 1e-3:
        raise SystemExit("REFUSING: the stock model loaded mel stats %r, not VCTK's %r. "
                         "A load hook re-normalising it would make this a different model."
                         % (got, (STOCK_MEL_MEAN, STOCK_MEL_STD)))
    s_voc, s_den = load_vocoder("hifigan_univ_v1", args.stock_vocoder, ear_bench.DEVICE)
    z = zipfile.ZipFile(args.vctk_zip)

    def stock_wave(mel):
        with torch.no_grad():
            w = to_waveform(mel, s_voc, s_den, DENOISER_STRENGTH).numpy()
        return soxr.resample(w.astype("float64"), STOCK_SR, sr).astype("float32")

    def stock_rt(x, xsr):
        y = torch.from_numpy(soxr.resample(x, xsr, STOCK_SR).astype("float32"))[None]
        return stock_wave(mel_spectrogram(y, *STOCK_MEL, center=False))

    def stock_model(pair_key, it):
        ids = intersperse(bb.encode_original(it["vctk_phonemes"]), 0)
        x = torch.tensor(ids, dtype=torch.long)[None]
        torch.manual_seed(ear_bench.seed_for(pair_key, args.seed))
        with torch.no_grad():
            o = stock.synthesise(x, torch.tensor([x.shape[-1]]),
                                 n_timesteps=ear_bench.N_TIMESTEPS,
                                 temperature=ear_bench.TEMPERATURE,
                                 spks=torch.tensor([it["vctk_sid"]], dtype=torch.long),
                                 length_scale=ear_bench.LENGTH_SCALE)
        return stock_wave(o["mel"])

    def ours_rt(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32"))[None]
        mel = mel_spectrogram(y, cfg["n_fft"], cfg["n_feats"], cfg["sample_rate"],
                              cfg["hop_length"], cfg["win_length"], cfg["f_min"],
                              cfg["f_max"], center=False)
        with torch.no_grad():
            return to_waveform(mel, bench.vocoder, None).numpy()

    flips = bb.balanced_flips(len(spec["items"]), rng)
    plan = [(it, fam, f[0] if fam == "stock" else f[1])
            for it, f in zip(spec["items"], flips) for fam in ("stock", "ours")]
    rng.shuffle(plan)

    clips = Path(args.out) / "clips"
    served, truth, limited = [], {}, 0
    for i, (it, fam, swap) in enumerate(plan):
        pair_key = "item_%02d" % i
        if fam == "stock":
            x, xsr = vctk_audio(z, it["vctk_utt"])
            made = {"stockrt": stock_rt(x, xsr), "stock": stock_model(pair_key, it)}
            kind, spk, hnr, src = KIND_STOCK, it["vctk"], it["vctk_hnr"], it["vctk_utt"]
        else:
            x, xsr = sf.read(it["libri_wav"], dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            if xsr != sr:
                raise SystemExit("REFUSING: %s is %d Hz, not %d." % (it["libri_wav"], xsr, sr))
            side = "model_spk%d_%s" % (it["libri"], Path(it["libri_wav"]).stem)
            name = ear_bench.opaque(pair_key, side, args.salt)
            bench.render(pair_key, side, args.ours_ckpt, "(phonemes)", it["libri"],
                         (0.0, 0.0, 0.0), delivery.DELIVERY_UNKNOWN, "ours",
                         phonemes=it["libri_phonemes"])
            model, _ = sf.read(str(clips / ("%s.wav" % name)), dtype="float32")
            made = {"oursrt": ours_rt(x), "ours": model}
            kind, spk, hnr, src = KIND_OURS, it["libri"], it["libri_hnr"], it["libri_wav"]

        lo, hi = kind.split("_vs_")
        sides = [("A", hi), ("B", lo)] if swap else [("A", lo), ("B", hi)]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            audio, bound = ear_bench.match_loudness(made[lbl], sr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), audio, sr, "PCM_24")
            bench.key[nm] = {"label": lbl, "pair": pair_key, "side": side_key}
            item[side_key] = nm
        served.append(item)
        # ⚠ A VCTK name goes under `vctk_spk`, never `spk`: later benches read `spk` from
        # every key in `_keys/` as a LibriTTS-R index to exclude.
        truth[pair_key] = {"kind": kind, "family": fam, "match": it["match"],
                           ("vctk_spk" if fam == "stock" else "spk"): spk,
                           "hnr": hnr, "A_label": sides[0][1], "B_label": sides[1][1],
                           "source": src}
        print("  %s  %-5s match %2d  %-7s HNR %5.2f" % (pair_key, fam, it["match"], spk, hnr))

    # The ours-side renders were written under a model_* name by Bench.render; they are
    # served under their opaque side names above, so the intermediates go.
    for nm in list(bench.key):
        if bench.key[nm]["side"].startswith("model_"):
            (clips / ("%s.wav" % nm)).unlink(missing_ok=True)
            del bench.key[nm]

    bench.write(Path(args.out).name, SETS, served,
                {"stock_ckpt": args.stock_ckpt, "ours_ckpt": args.ours_ckpt,
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
    s.add_argument("--vctk-filelist", default=BENCH_DIR + "/vctk_audio_sid_text_train_filelist.txt")
    s.add_argument("--vctk-cleaned",
                   default=BENCH_DIR + "/vctk_audio_sid_text_train_filelist.txt.cleaned")
    s.add_argument("--hnr-json", default=ROOT + "/pitch_error/speaker_hnr_all.json")
    s.add_argument("--corpus", default="data/libritts_r_full_vat_v7")
    s.add_argument("--prior", nargs="+", default=[ROOT + "/pitch_error/_*.json",
                                                  ROOT + "/eartest/_keys/*.key.json"])
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--clips-per-speaker", type=int, default=5)
    s.add_argument("--min-seconds", type=float, default=3.0)
    s.add_argument("--max-gap", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=20260930)
    s.add_argument("--out-speakers", required=True)

    r = sub.add_parser("render")
    r.add_argument("--speakers", required=True)
    r.add_argument("--stock-ckpt", default=ROOT + "/warmstart/matcha_vctk.ckpt")
    r.add_argument("--stock-vocoder",
                   default=BENCH_DIR + "/UNIVERSAL_V1_g_02500000")  # a readable copy
    r.add_argument("--ours-ckpt", required=True)
    r.add_argument("--vctk-zip", default=VCTK_ZIP)
    r.add_argument("--data-config", default="configs/data/libritts_r_full_vat_v7.yaml")
    r.add_argument("--out", required=True)
    r.add_argument("--key-out", required=True)
    r.add_argument("--lufs", type=float, default=-23.0)
    r.add_argument("--salt", default="baseline-hum-v1")
    r.add_argument("--seed", type=int, default=3141)

    args = ap.parse_args()
    (select if args.cmd == "select" else render)(args)


if __name__ == "__main__":
    main()

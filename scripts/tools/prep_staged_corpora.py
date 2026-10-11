"""Build the staged rebuild's 22.05 kHz corpora.

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md (`scripts/lib/staged_data.py`
holds the rules; this is the glue). Idempotent: a 22.05 kHz copy that already exists is kept.

    # host
    .venv/bin/python scripts/tools/prep_staged_corpora.py vctk
    .venv/bin/python scripts/tools/prep_staged_corpora.py libritts
    .venv/bin/python scripts/tools/prep_staged_corpora.py original   # arm L, after libritts
    # container (needs torch; CPU)
    scripts/stages/run_in_rocm.sh scripts/tools/prep_staged_corpora.py stats
    scripts/stages/run_in_rocm.sh scripts/tools/prep_staged_corpora.py stats --corpus libritts_22k
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import staged_data as sd                             # noqa: E402

DATA = "/data/model-training/sonora/data"     # the repo's data/ (absent in the container)
BASELINE_DIR = "/data/model-training/sonora/baseline_bench"
VCTK_ZIP = "/data/model-training/datasets/VCTK/VCTK-Corpus-0.92.zip"
LIBRITTS_TGZ = "/data/model-training/datasets/LibriTTS/train-clean-100.tar.gz"
LIBRITTS_TGZ_MD5 = "4a8c202b78fe1bc0c47916a98f3a2ea8"     # OpenSLR 60's published MD5
_ZIP = None


def sha256(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mkdir(d):
    # group-writable and setgid: the container writes here as ai-mgr (group datashare)
    os.makedirs(d, exist_ok=True)
    os.chmod(d, 0o2775)


def resample_to(x, sr, dst):
    """Mono float -> 22.05 kHz PCM_16 at `dst`. Returns 1 if a sample was clipped."""
    import soxr
    y = soxr.resample(np.asarray(x, dtype="float64"), sr, sd.TARGET_SR)
    clipped = int(np.abs(y).max() > 1.0)
    mkdir(os.path.dirname(dst))
    sf.write(dst, np.clip(y, -1.0, 1.0), sd.TARGET_SR, "PCM_16")
    return clipped


def done(dst):
    return os.path.isfile(dst) and sf.info(dst).samplerate == sd.TARGET_SR


def _vctk_job(job):
    global _ZIP
    utt, spk = job
    dst = sd.vctk_22k_path(spk, utt)
    if done(dst):
        return 0
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(VCTK_ZIP)
    name = "wav48_silence_trimmed/%s/%s_mic1.flac" % (spk, utt)
    x, sr = sf.read(io.BytesIO(_ZIP.read(name)), dtype="float64")
    return resample_to(x.mean(axis=1) if x.ndim > 1 else x, sr, dst)


def _libri_job(src):
    dst = sd.to_22k(src)
    if done(dst):
        return 0
    x, sr = sf.read(src, dtype="float64")
    return resample_to(x.mean(axis=1) if x.ndim > 1 else x, sr, dst)


def _original_job(job):
    src, dst = job
    if done(dst):
        return 0
    x, sr = sf.read(src, dtype="float64")          # exactly as `_libri_job` reads S1's
    return resample_to(x.mean(axis=1) if x.ndim > 1 else x, sr, dst)


def vctk(args):
    names = set(zipfile.ZipFile(VCTK_ZIP).namelist())
    have = {Path(n).name[:-len("_mic1.flac")] for n in names if n.endswith("_mic1.flac")}
    out = Path(DATA) / "vctk_22k"
    mkdir(out)
    report = {"inputs": {}, "rows": {}, "missing": {}}
    jobs = []
    for split, src in (("train", BASELINE_DIR + "/vctk_audio_sid_text_train_filelist.txt.cleaned"),
                       ("val", BASELINE_DIR + "/vctk_audio_sid_text_val_filelist.txt.cleaned")):
        rows = sd.read_vits(src)
        lines, missing = sd.c0_lines(rows, have)
        (out / ("%s.txt" % split)).write_text("\n".join(lines) + "\n", encoding="utf-8")
        report["inputs"][split] = [src, sha256(src)]
        report["rows"][split], report["missing"][split] = len(lines), missing
        jobs += [(u, s) for u, s, _sid, _t in rows if u in have]
    with ProcessPoolExecutor(args.workers) as ex:
        clipped = sum(ex.map(_vctk_job, jobs, chunksize=64))
    report["clipped_files"] = clipped
    report["steps_per_epoch"] = sd.steps_per_epoch(report["rows"]["train"])
    report["zip"] = [VCTK_ZIP, sha256(VCTK_ZIP)]
    (out / "provenance.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("vctk_22k: train %d, val %d rows; %d/%d VITS rows had no mic1 recording; "
          "%d clipped; %d steps/epoch"
          % (report["rows"]["train"], report["rows"]["val"],
             len(report["missing"]["train"]), len(report["missing"]["val"]),
             clipped, report["steps_per_epoch"]))


def libritts(args):
    src_dir, out = Path(DATA) / "libritts_r_vat", Path(DATA) / "libritts_r_22k"
    mkdir(out)
    report = {"inputs": {}, "rows": {}}
    jobs = []
    for split in ("train", "val"):
        src = src_dir / ("%s_op.txt" % split)
        rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines = [sd.s1_line(l) for l in rows]
        (out / ("%s_op.txt" % split)).write_text("\n".join(lines) + "\n", encoding="utf-8")
        report["inputs"][split] = [str(src), sha256(src)]
        report["rows"][split] = len(lines)
        jobs += [l.split("|")[0] for l in rows]
    shutil.copyfile(src_dir / "speakers.json", out / "speakers.json")
    with ProcessPoolExecutor(args.workers) as ex:
        clipped = sum(ex.map(_libri_job, jobs, chunksize=64))
    report["clipped_files"] = clipped
    report["steps_per_epoch"] = sd.steps_per_epoch(report["rows"]["train"])
    (out / "provenance.json").write_text(json.dumps(report, indent=1))
    print("libritts_r_22k: train %d, val %d rows; %d clipped; %d steps/epoch"
          % (report["rows"]["train"], report["rows"]["val"], clipped,
             report["steps_per_epoch"]))


def original(args):
    """Arm L: S1's rows with each recording replaced by the same utterance's ORIGINAL
    LibriTTS recording, resampled to 22.05 kHz exactly as S1's copies were. A row whose
    original is missing is dropped and counted, never substituted; more than 1 % missing
    refuses before anything is written."""
    md5 = sha256(LIBRITTS_TGZ, "md5")
    if md5 != LIBRITTS_TGZ_MD5:
        raise SystemExit("REFUSING: %s has MD5 %s; OpenSLR publishes %s."
                         % (LIBRITTS_TGZ, md5, LIBRITTS_TGZ_MD5))
    src_dir, out = Path(DATA) / "libritts_r_22k", Path(DATA) / "libritts_22k"
    report = {"inputs": {}, "rows": {}, "dropped": {},
              "tarball": [LIBRITTS_TGZ, "md5", md5]}
    split_lines, jobs, n_s1 = {}, [], 0
    for split in ("train", "val"):
        src = src_dir / ("%s_op.txt" % split)
        rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        have = {sd.original_of(l.split("|", 1)[0]) for l in rows}
        have = {w for w in have if os.path.isfile(w)}
        lines, missing = sd.l_lines(rows, have)
        n_s1 += len(rows)
        split_lines[split] = lines
        report["inputs"][split] = [str(src), sha256(src)]
        report["rows"][split], report["dropped"][split] = len(lines), missing
        kept = [l.split("|", 1)[0] for l in rows]
        jobs += [(sd.original_of(w), sd.l22k_of(w)) for w in kept if sd.original_of(w) in have]
    n_missing = sum(len(v) for v in report["dropped"].values())
    if sd.too_many_missing(n_missing, n_s1):
        raise SystemExit("REFUSING: %d of S1's %d rows have no original recording (more than "
                         "%.0f %%): L is not trained; the owner decides."
                         % (n_missing, n_s1, 100 * sd.MAX_MISSING_SHARE))
    mkdir(out)
    for split, lines in split_lines.items():
        (out / ("%s_op.txt" % split)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(src_dir / "speakers.json", out / "speakers.json")
    with ProcessPoolExecutor(args.workers) as ex:
        clipped = sum(ex.map(_original_job, jobs, chunksize=64))
    report["clipped_files"] = clipped
    report["steps_per_epoch"] = sd.steps_per_epoch(report["rows"]["train"])
    (out / "provenance.json").write_text(json.dumps(report, indent=1))
    print("libritts_22k: train %d, val %d rows; dropped %d train, %d val (no original); "
          "%d clipped; %d steps/epoch; tarball MD5 %s"
          % (report["rows"]["train"], report["rows"]["val"], len(report["dropped"]["train"]),
             len(report["dropped"]["val"]), clipped, report["steps_per_epoch"], md5))


def stats(args):
    from matcha.data.text_mel_datamodule import TextMelDataModule
    from matcha.utils.generate_data_statistics import compute_data_statistics

    d = Path(DATA) / args.corpus
    dm = TextMelDataModule(
        name="%s_stats" % args.corpus, train_filelist_path=str(d / "train_op.txt"),
        valid_filelist_path=str(d / "val_op.txt"), batch_size=256, num_workers=args.workers,
        pin_memory=False, cleaners=["no_cleaners"], add_blank=True, n_spks=247,
        data_statistics=None, seed=1234, load_durations=False, load_vat=False, vat_dim=None,
        **dict(zip(sd.MEL_KEYS, sd.STOCK_MEL)))
    dm.setup()
    params = compute_data_statistics(dm.train_dataloader(), sd.STOCK_MEL[1])
    (d / "mel_statistics.json").write_text(json.dumps(params))
    print("mel_statistics:", json.dumps(params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("vctk", "libritts", "original", "stats"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--corpus", choices=("libritts_r_22k", "libritts_22k"),
                    default="libritts_r_22k", help="stats: S1's corpus or L's")
    args = ap.parse_args()
    {"vctk": vctk, "libritts": libritts, "original": original, "stats": stats}[args.cmd](args)


if __name__ == "__main__":
    main()

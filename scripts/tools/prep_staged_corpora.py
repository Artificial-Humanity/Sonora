"""Build the staged rebuild's 22.05 kHz corpora.

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md and, for d1, g and f1,
Sonora/hum-fix-batch-preregistration.md (`scripts/lib/staged_data.py` holds the rules; this is
the glue). Idempotent: a 22.05 kHz copy that already exists is kept.

    # host
    .venv/bin/python scripts/tools/prep_staged_corpora.py vctk
    .venv/bin/python scripts/tools/prep_staged_corpora.py libritts
    .venv/bin/python scripts/tools/prep_staged_corpora.py original   # arm L, after libritts
    .venv/bin/python scripts/tools/prep_staged_corpora.py d1         # after vctk
    .venv/bin/python scripts/tools/prep_staged_corpora.py g          # after libritts
    .venv/bin/python scripts/tools/prep_staged_corpora.py f1         # after libritts
    # container (needs torch; CPU)
    scripts/stages/run_in_rocm.sh scripts/tools/prep_staged_corpora.py stats
    scripts/stages/run_in_rocm.sh scripts/tools/prep_staged_corpora.py stats --corpus libritts_22k
    scripts/stages/run_in_rocm.sh scripts/tools/prep_staged_corpora.py stats --corpus f1_dense
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
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


def _f1_job(src):
    dst = sd.libri_r_22k(src)
    if done(dst):
        return 0, 0
    x, sr = sf.read(src, dtype="float64")          # exactly as `_libri_job` reads S1's
    return resample_to(x.mean(axis=1) if x.ndim > 1 else x, sr, dst), 1


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


def read_rows(path):
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def d1(args):
    """D1: C0's train rows, 105 per speaker (`staged_data.d1_subsample`); C0's val rows
    unchanged. Filelists only: the audio is C0's."""
    src_dir, out = Path(DATA) / "vctk_22k", Path(DATA) / "vctk_d1"
    train = read_rows(src_dir / "train.txt")
    kept = sd.d1_subsample(train)
    mkdir(out)
    (out / "train.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    shutil.copyfile(src_dir / "val.txt", out / "val.txt")
    n_spk = len({l.split("|")[1] for l in kept})
    report = {"inputs": {s: [str(src_dir / ("%s.txt" % s)), sha256(src_dir / ("%s.txt" % s))]
                         for s in ("train", "val")},
              "seed": sd.D1_SEED, "per_speaker": sd.D1_PER_SPEAKER, "speakers": n_spk,
              "rows": {"train": len(kept), "val": len(read_rows(out / "val.txt"))},
              "steps_per_epoch": sd.steps_per_epoch(len(kept))}
    (out / "provenance.json").write_text(json.dumps(report, indent=1))
    print("vctk_d1: %d speakers x %d = train %d rows (of %d), val %d rows; %d steps/epoch"
          % (n_spk, sd.D1_PER_SPEAKER, len(kept), len(train), report["rows"]["val"],
             report["steps_per_epoch"]))


def g(args):
    """G′: S1's filelists with every `ᵻ` moved to `ᵊ`, as `{train,val}_g.txt` beside S1's.
    Every row is asserted equal to `upstream_as_ours` of S1's phonemes (`staged_data.g_line`)
    before anything is written."""
    d = Path(DATA) / "libritts_r_22k"
    report = {"inputs": {}, "rows": {}, "replaced": {}, "rows_touched": {}}
    lines = {}
    for split in ("train", "val"):
        src = d / ("%s_op.txt" % split)
        got = [sd.g_line(l) for l in read_rows(src)]
        lines[split] = [l for l, _n in got]
        report["inputs"][split] = [str(src), sha256(src)]
        report["rows"][split] = len(got)
        report["replaced"][split] = sum(n for _l, n in got)
        report["rows_touched"][split] = sum(1 for _l, n in got if n)
    for split, ls in lines.items():
        (d / ("%s_g.txt" % split)).write_text("\n".join(ls) + "\n", encoding="utf-8")
    report["steps_per_epoch"] = sd.steps_per_epoch(report["rows"]["train"])
    report["equals_upstream_as_ours"] = "asserted on every row"
    (d / "provenance_g.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("libritts_r_22k _g: train %d, val %d rows; %s -> %s: %d train + %d val replaced, "
          "in %d + %d rows; %d steps/epoch"
          % (report["rows"]["train"], report["rows"]["val"], sd.G_FROM, sd.G_TO,
             report["replaced"]["train"], report["replaced"]["val"],
             report["rows_touched"]["train"], report["rows_touched"]["val"],
             report["steps_per_epoch"]))


F1_EXPECT = {"speakers": 90, "train": 31287, "val": 960}     # the registration's counts


def f1(args):
    """F1: every LibriTTS-R speaker with >= 300 train rows after S1's 1-16 s filter, from
    S1's train filelist and the derived our-G2P train-clean-360 / train-other-500 filelists
    (their lengths re-filtered from `measures.jsonl`), re-indexed into a table of its own.
    Val: those speakers' rows in the same filelists' val splits, filtered the same way.
    Refuses before writing anything if the counts are not the registered ones. The audio
    copies are made first and the filelists, speaker table and provenance last."""
    sources = [("libritts_r_vat", None),                        # S1's rows (1-16 s already)
               ("_derived_train_clean_360", "measures.jsonl"),
               ("_derived_train_other_500", "measures.jsonl")]
    v7 = json.loads((Path(DATA) / "libritts_r_full_vat_v7" / "speakers.json")
                    .read_text())["libritts_id_to_index"]
    report = {"inputs": {}, "filtered_out": {}}
    rows = {"train": [], "val": []}
    spk_source = {}
    for name, measures in sources:
        d = Path(DATA) / name
        idx_to_id = {i: k for k, i in json.loads((d / "speakers.json").read_text())
                     ["libritts_id_to_index"].items()}
        sec = None
        if measures:
            sec = {}
            for line in (d / measures).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    m = json.loads(line)
                    sec[m["wav"]] = m["seconds"]
            report["inputs"]["%s/%s" % (name, measures)] = sha256(d / measures)
        for split in ("train", "val"):
            src = d / ("%s_op.txt" % split)
            report["inputs"]["%s/%s" % (name, src.name)] = sha256(src)
            dropped = 0
            for line in read_rows(src):
                sd.check_speaker_column(line, idx_to_id)
                wav = line.split("|", 1)[0]
                if sec is not None and not sd.f1_length_ok(sec[wav]):
                    dropped += 1
                    continue
                spk = sd.libritts_id(wav)
                if spk_source.setdefault(spk, name) != name:
                    raise SystemExit("REFUSING: LibriTTS speaker %d is in both %s and %s."
                                     % (spk, spk_source[spk], name))
                rows[split].append(line)
            report["filtered_out"]["%s/%s" % (name, split)] = dropped
    spk_of = lambda line: sd.libritts_id(line.split("|", 1)[0])      # noqa: E731
    ids = sd.f1_dense_speakers([l.split("|", 1)[0] for l in rows["train"]])
    table = sd.f1_table(ids, v7)
    index = table["libritts_id_to_index"]
    kept = {s: [l for l in rows[s] if str(spk_of(l)) in index] for s in ("train", "val")}
    lines = {s: [sd.f1_line(l, index) for l in kept[s]] for s in kept}
    got = {"speakers": len(ids), "train": len(lines["train"]), "val": len(lines["val"])}
    if got != F1_EXPECT:
        raise SystemExit("REFUSING: F1 counts %s, registered %s. Nothing written; the "
                         "owner decides." % (got, F1_EXPECT))
    out = Path(DATA) / "f1_dense"
    mkdir(out)
    jobs = [l.split("|", 1)[0] for s in ("train", "val") for l in kept[s]]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_f1_job, jobs, chunksize=64))
    # The filelists and the speaker table come AFTER the audio: a crash mid-resample then
    # leaves no filelist pointing at audio that is not there.
    for split, ls in lines.items():
        (out / ("%s_op.txt" % split)).write_text("\n".join(ls) + "\n", encoding="utf-8")
    (out / "speakers.json").write_text(json.dumps(table, indent=1))
    report.update({
        "rows": {"train": got["train"], "val": got["val"]}, "speakers": got["speakers"],
        "rows_by_source": {s: dict(Counter(spk_source[spk_of(l)] for l in kept[s]))
                           for s in kept},
        "min_rows": sd.F1_MIN_ROWS, "seconds": [sd.F1_MIN_SECONDS, sd.F1_MAX_SECONDS],
        "copies_written": sum(w for _c, w in res), "copies_kept": sum(1 - w for _c, w in res),
        "clipped_files": sum(c for c, _w in res),
        "clipped_note": "counted over the copies THIS run wrote; kept copies count 0",
        "steps_per_epoch": sd.steps_per_epoch(got["train"])})
    (out / "provenance.json").write_text(json.dumps(report, indent=1))
    print("f1_dense: %d speakers, train %d, val %d rows; %d copies written (%d clipped), "
          "%d kept; %d steps/epoch"
          % (got["speakers"], got["train"], got["val"], report["copies_written"],
             report["clipped_files"], report["copies_kept"], report["steps_per_epoch"]))


# corpus: its train/val filelists and speaker-table size, for `stats`
STATS_CORPORA = {"libritts_r_22k": 247, "libritts_22k": 247, "f1_dense": 90}


def stats(args):
    from matcha.data.text_mel_datamodule import TextMelDataModule
    from matcha.utils.generate_data_statistics import compute_data_statistics

    d = Path(DATA) / args.corpus
    dm = TextMelDataModule(
        name="%s_stats" % args.corpus, train_filelist_path=str(d / "train_op.txt"),
        valid_filelist_path=str(d / "val_op.txt"), batch_size=256, num_workers=args.workers,
        pin_memory=False, cleaners=["no_cleaners"], add_blank=True,
        n_spks=STATS_CORPORA[args.corpus],
        data_statistics=None, seed=1234, load_durations=False, load_vat=False, vat_dim=None,
        **dict(zip(sd.MEL_KEYS, sd.STOCK_MEL)))
    dm.setup()
    params = compute_data_statistics(dm.train_dataloader(), sd.STOCK_MEL[1])
    (d / "mel_statistics.json").write_text(json.dumps(params))
    print("mel_statistics:", json.dumps(params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("vctk", "libritts", "original", "d1", "g", "f1", "stats"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--corpus", choices=sorted(STATS_CORPORA),
                    default="libritts_r_22k", help="stats: S1's corpus, L's or F1's")
    args = ap.parse_args()
    {"vctk": vctk, "libritts": libritts, "original": original, "d1": d1, "g": g, "f1": f1,
     "stats": stats}[args.cmd](args)


if __name__ == "__main__":
    main()

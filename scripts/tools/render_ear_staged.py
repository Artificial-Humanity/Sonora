"""WHICH PART OF THE FIRST STEP ADDS THE HUM? The staged-rebuild bench, checks 1-4.

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md (owner, 2026-10-06; check 4
2026-10-11); the readings are `staged_bench.reading`. Checks 1-3 are 20 units x 3 arms = 60
items; check 4 is at most 20 units x 2 arms. An item is one arm's render and the vocoder
round trip of the same real recording, each rated for hum 0-5, on the app, scale and
loudness of the earlier hum benches.

  check 1  stock (VCTK voice) | R, derisk (its HNR-matched LibriTTS-R voice)   the gate
  check 2  stock, C0 (one VCTK voice) | S1 (its HNR-matched LibriTTS-R voice)
  check 3  S1, S2, R (one LibriTTS-R voice)
  check 4  S1, L (one LibriTTS-R voice)      does LibriTTS-R's restoration add the hum?

⚠ EVERY ARM RENDERS THROUGH ITS OWN AUDIO PATH. stock, C0 and S1: 22.05 kHz mels through the
universal vocoder with its denoiser, resampled to the bench rate after vocoding, exactly as
`render_ear_baseline.py` renders stock. S2, R and derisk: 24 kHz through our vocoder. Each
round trip goes through its arm's path. L renders on S1's path, but round-trips its OWN
corpus's copy of the utterance: the 22.05 kHz copy of the ORIGINAL LibriTTS recording, where
S1 round-trips LibriTTS-R's (`round_trip_source`).

⚠ CHECK 4 TAKES EVERY ELIGIBLE VOICE IF THERE ARE 20 OR FEWER, and refuses under 16. A voice
is eligible with 2 recordings of 4-8 s (on the 24 kHz LibriTTS-R file) that are in both S1's
and L's train filelists and whose original recording is within 0.1 s of the same length.

⚠ A VOICE THAT SPEAKS SEVERAL ARMS GETS A DIFFERENT RECORDING FOR EACH (`assign_clips`), the
ones nearest in length to the unit's VCTK clip (checks 1-2) or to --target-seconds (checks 3-4).
Every recording is from its arm's training split.

⚠ stock and C0 read VITS's espeak phonemes through upstream's table (`encode_original`),
which is what both trained on. Every LibriTTS-R arm reads our G2P with no cleaners; the VAT
arms (R, derisk) get all-zero conditioning.

⚠ VOICES ARE UNHEARD: every key in `_keys/`, every pitch-error file, the first-step speakers
file (`--prior`, each pattern must match a file) and every earlier staged speakers file
(`--prior-optional`, may match nothing) is excluded. A speakers file is never overwritten.
Pass each glob as its own QUOTED argument: an unquoted `$P` is one word in zsh.

    # host, before L trains: the pre-flight count. A dry run may omit an untrained arm's
    # checkpoint; its speakers file is marked as one, never lands in staged_bench/, and
    # render refuses it.
    .venv/bin/python scripts/tools/render_ear_staged.py select --check 4 --dry-run \\
        --ckpt S1=<staged_s1 checkpoint_epoch=009_step=0009209.ckpt> \\
        --out-speakers <scratch>/check4_dryrun_speakers.json
    # host (check 2 before check 3: check 3 then excludes check 2's voices)
    .venv/bin/python scripts/tools/render_ear_staged.py select --check 1 \\
        --ckpt R=<staged_r checkpoint_epoch=009_step=…ckpt> \\
        --out-speakers /data/model-training/sonora/staged_bench/check1_speakers.json
    # host: the container runs as ai-mgr and cannot create a directory in eartest/
    install -d -m 2775 /data/model-training/sonora/eartest/staged_check1
    # container (CPU), from a clean checkout of the registered commit
    SONORA_EXTRA_DEPS="pyloudnorm soxr" scripts/stages/run_in_rocm.sh \\
        scripts/tools/render_ear_staged.py render --check 1 --commit <registered sha> \\
        --speakers /data/model-training/sonora/staged_bench/check1_speakers.json \\
        --out /data/model-training/sonora/eartest/staged_check1 \\
        --key-out /data/model-training/sonora/.keystage/staged_check1.key.json
"""

import argparse
import fnmatch
import glob
import hashlib
import io
import json
import os
import random
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import baseline_bench as bb                          # noqa: E402
from lib import first_step_bench as fs                        # noqa: E402
from lib import lineage_bench as lb                           # noqa: E402
from lib import staged_bench as sb                            # noqa: E402
from lib import staged_data as sd                             # noqa: E402
from tools import render_ear_first_step as rfs                # noqa: E402

ROOT = rfs.ROOT
DATA = ROOT + "/data"
FIXED = {"stock": ROOT + "/warmstart/matcha_vctk.ckpt",
         "derisk": ROOT + "/logs/train/derisk_energy/runs/2026-07-15_00-20-31/checkpoints/"
                          "checkpoint_epoch=099.ckpt"}
NEW = ("R", "C0", "S1", "S2", "L")
# arm: (data config holding its mel statistics, n_spks, use_vat)
EXPECT = {"stock": ("vctk", 109, False), "C0": ("vctk", 109, False),
          "S1": ("libritts_r_22k", 247, False), "S2": ("libritts_r_vat", 247, False),
          "R": ("libritts_r_vat", 247, True), "derisk": ("libritts_r_vat", 247, True),
          "L": ("libritts_22k", 247, False)}
# A 22.05 kHz LibriTTS arm's round trip: its own corpus's copy of the 24 kHz LibriTTS-R row.
RT_22K = {"libri22": sd.to_22k, "orig22": sd.to_l22k}
ORIG_MAX_DRIFT = Fraction(1, 10)     # check 4: original vs LibriTTS-R length, seconds, strict
PRIOR = [ROOT + "/pitch_error/_*.json", ROOT + "/eartest/_keys/*.key.json",
         ROOT + "/first_step_bench/speakers.json"]
OURS_MEL = ("n_fft", "n_feats", "sample_rate", "hop_length", "win_length", "f_min", "f_max")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_ckpts(arms, given, allow_missing=False):
    """{arm: checkpoint} for one check. stock and derisk are fixed; R, C0, S1, S2 and L must
    be given as ARM=PATH, and must be the step checkpoint that ended the tenth epoch.
    `allow_missing` (a dry run only) maps an arm given no checkpoint to None, for an arm not
    trained yet; a checkpoint that IS given is still checked."""
    got = {}
    for g in given:
        arm, _, path = g.partition("=")
        if arm in FIXED:
            raise SystemExit("REFUSING: %s is a fixed arm (%s)." % (arm, FIXED[arm]))
        got[arm] = path
    out = {}
    for a in arms:
        if a in FIXED:
            out[a] = FIXED[a]
        elif a not in got and allow_missing:
            out[a] = None
        elif a not in got:
            raise SystemExit("REFUSING: arm %s needs --ckpt %s=<checkpoint>." % (a, a))
        elif not sb.ten_epoch_ckpt(got[a]):
            raise SystemExit("REFUSING: %s's checkpoint %s is not the one that ended "
                             "10 epochs (checkpoint_epoch=009_step=…)." % (a, got[a]))
        else:
            out[a] = got[a]
    return out


def corpus_rows(corpus):
    """{wav: (speaker index, phonemes)} over TRAIN rows of data/<corpus>/train_op.txt."""
    out = {}
    for line in (Path(DATA) / corpus / "train_op.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            wav, spk, phon = line.split("|")[:3]
            out[wav] = (int(spk), phon)
    return out


def round_trip_source(arm, wav24):
    """The 22.05 kHz copy a LibriTTS arm on the universal vocoder round-trips for the 24 kHz
    LibriTTS-R row `wav24`: S1 its LibriTTS-R copy, L its copy of the ORIGINAL recording."""
    return RT_22K[sb.FAMILY[arm]](wav24)


def exact_duration(path):
    """A file's length in seconds, exactly (frames / rate): a float difference can put 5.1 s
    against 5.0 s inside 0.1 s."""
    i = sf.info(path)
    return Fraction(i.frames, i.samplerate)


def same_utterance_length(sec_r, sec_orig):
    """Check 4: is the original recording within 0.1 s (strictly) of the LibriTTS-R file?"""
    return abs(sec_r - sec_orig) < ORIG_MAX_DRIFT


def default_prior(check):
    """--prior when none is given. Check 4 also REQUIRES every earlier staged speakers file
    (pre-registered: all required, each must match a file)."""
    if check == 4:
        return PRIOR + [ROOT + "/staged_bench/check%d_speakers.json" % i for i in (1, 2, 3)]
    return list(PRIOR)


def require_default_prior(check, prior):
    """Check 4 REQUIRES every entry of `default_prior(4)` (pre-registered): an explicit
    --prior that leaves one out would re-draw heard voices without a word."""
    if check != 4:
        return
    missing = [p for p in default_prior(4) if p not in prior]
    if missing:
        raise SystemExit("REFUSING: check 4's --prior must hold every default pattern; "
                         "missing: %s." % ", ".join(missing))


def require_registered_s1(path):
    """Check 4 compares the registered S1 checkpoint, byte for byte."""
    if path is None or sha256(path) != sb.CHECK4_S1_SHA256:
        raise SystemExit("REFUSING: check 4's S1 checkpoint %s is not the registered one "
                         "(sha256 %s)." % (path, sb.CHECK4_S1_SHA256))


def speakers_rule(args, c, n_units):
    """The `rule` text of a speakers file. `n_units` is the units actually drawn."""
    spread = "checks 3-4" if args.check == 4 else "check 3"
    return ("check %d: arms %s; %d units; clips %.1f-%.1fs; VCTK/LibriTTS-R matched "
            "within %.2f dB HNR (checks 1-2) or LibriTTS-R voices spread over HNR "
            "(%s, target %.1fs); a distinct training recording per arm; seed %d."
            % (args.check, "/".join(c["arms"]), n_units, args.min_seconds,
               args.max_seconds, args.max_gap, spread, args.target_seconds, args.seed))


def truth_entry(check, arm, it, sides):
    """One item's record in the key's truth. A VCTK name goes under `vctk_spk`, never `spk`:
    later benches read `spk` from every key in `_keys/` as a LibriTTS-R index to exclude.
    Check 4 alone also records `rt_source`, the file the round trip was made from (checks 1-3
    keys keep their format)."""
    if sb.FAMILY[arm] == "vctk22":
        who, hnr = {"vctk_spk": it["vctk"]}, it["vctk_hnr"]
    else:
        who, hnr = {"spk": it["libri"]}, it["libri_hnr"]
    entry = dict(who, kind="%srt_vs_%s" % (arm, arm), arm=arm, unit=it["unit"],
                 hnr=hnr, A_label=sides[0][1], B_label=sides[1][1],
                 source=it["arms"][arm]["source"])
    if check == 4 and sb.FAMILY[arm] in RT_22K:
        entry["rt_source"] = round_trip_source(arm, it["arms"][arm]["source"])
    return entry


def pick_voices(pool, n, check):
    """The units' voices, spread over HNR (`lineage_bench.spread`). Check 4 takes every
    eligible voice when there are `n` or fewer, and refuses under its minimum."""
    least = sb.CHECKS[check].get("min_units")
    if least is None:
        return lb.spread(pool, n)
    if len(pool) < least:
        raise SystemExit("REFUSING: %d eligible voices; check %d needs at least %d. It is not "
                         "run, and the owner decides." % (len(pool), check, least))
    return lb.spread(pool, min(n, len(pool)))


def refuse_dry_run_target(path, patterns=()):
    """A dry run never writes where a real selection or an exclusion reads: select refuses an
    existing speakers file, so a dry-run file in staged_bench/ would block the real draw, and
    a file a `--prior` / `--prior-optional` pattern matches would be read as a heard-voice
    list by the next draw. Nothing under the bench data root is a scratch directory."""
    here = Path(path).resolve()
    if here.is_relative_to(Path(ROOT).resolve()):
        raise SystemExit("REFUSING: a dry run may not write under %s (the bench data root; the "
                         "real selection goes in staged_bench/)." % ROOT)
    for pat in patterns:
        if fnmatch.fnmatchcase(str(here), pat) or fnmatch.fnmatchcase(os.path.abspath(path), pat):
            raise SystemExit("REFUSING: a dry run may not write %s: the --prior pattern %r "
                             "matches it." % (path, pat))


def read_spec(path, check):
    """The speakers file render takes, refused if it is another check's or a dry run's."""
    spec = json.loads(Path(path).read_text())
    if spec.get("check") != check:
        raise SystemExit("REFUSING: %s is check %s's, not check %d's."
                         % (path, spec.get("check"), check))
    if spec.get("dry_run"):
        raise SystemExit("REFUSING: %s is a dry run's speakers file, not a selection." % path)
    return spec


# ------------------------------------------------------------------------------ select

def load_prior(patterns, optional, skip):
    """The earlier benches' files, as parsed JSON. Every pattern in `patterns` must match a
    file: an exclusion that matches nothing re-draws heard voices without a word (an
    unquoted `$P` in zsh arrives as ONE pattern and matches nothing). The `optional`
    patterns may match nothing, for files that do not exist before the first draw. `skip`
    (the speakers file being written) is never read."""
    docs = []
    for pat, may_be_empty in [(p, False) for p in patterns] + [(p, True) for p in optional]:
        files = [f for f in sorted(glob.glob(pat)) if Path(f).resolve() != Path(skip).resolve()]
        if not files and not may_be_empty:
            raise SystemExit("REFUSING: --prior pattern %r matches no file; an exclusion that "
                             "matches nothing re-draws heard voices." % pat)
        docs += [json.loads(Path(f).read_text()) for f in files]
    return docs


def select(args):
    c = sb.CHECKS[args.check]
    out_path = Path(args.out_speakers)
    if out_path.exists():
        raise SystemExit("REFUSING: %s exists. Voices are drawn once (pre-registration)."
                         % out_path)
    if args.dry_run:
        refuse_dry_run_target(out_path, list(args.prior) + list(args.prior_optional))
    ckpts = resolve_ckpts(c["arms"], args.ckpt, args.dry_run)
    for a, p in ckpts.items():
        if p is not None and not Path(p).is_file():
            raise SystemExit("REFUSING: %s's checkpoint %s does not exist." % (a, p))
    if args.check == 4:
        require_registered_s1(ckpts["S1"])
    rng = random.Random(args.seed + args.check)

    docs = load_prior(args.prior, args.prior_optional, out_path)
    vheard, lheard = fs.heard_vctk(docs), sb.heard_libri(docs)

    rows = corpus_rows("libritts_r_vat")
    s1_have = set(corpus_rows("libritts_r_22k")) if "S1" in c["arms"] else None
    l_have = set(corpus_rows("libritts_22k")) if "L" in c["arms"] else None
    orig_sec = {}
    hnr = {int(k): v["hnr"] for k, v in
           json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    lib_clips = {}
    for w in sorted(rows):
        s = rows[w][0]
        if s not in hnr or s in lheard or not Path(w).is_file():
            continue
        if s1_have is not None and sd.to_22k(w) not in s1_have:
            continue
        if l_have is not None and sd.to_l22k(w) not in l_have:
            continue
        sec = sf.info(w).duration
        if args.min_seconds <= sec <= args.max_seconds:
            if l_have is not None:
                o = sd.to_original(w)
                if not (Path(o).is_file()
                        and same_utterance_length(exact_duration(w), exact_duration(o))):
                    continue
                orig_sec[w] = sf.info(o).duration
            lib_clips.setdefault(s, []).append((w, sec))
    pool = {s: hnr[s] for s, cl in lib_clips.items() if len(cl) >= len(c["libri_arms"])}
    print("%d LibriTTS-R voices eligible (%d heard)" % (len(pool), len(lheard)))

    def libri_arms(voice, target):
        got = lb.assign_clips({voice: lib_clips[voice]}, c["libri_arms"], rng, target)[voice]
        secs = dict(lib_clips[voice])
        out = {a: {"source": w, "seconds": round(secs[w], 2), "phonemes": rows[w][1]}
               for a, w in got.items()}
        for a, w in got.items():
            if w in orig_sec:
                out[a]["original_seconds"] = round(orig_sec[w], 2)
        return out

    items = []
    if c["vctk_arms"]:
        raw = rfs.read_vits(args.vctk_filelist)
        cleaned = rfs.read_vits(args.vctk_cleaned)
        if set(raw) != set(cleaned):
            raise SystemExit("REFUSING: the raw and .cleaned VITS filelists differ.")
        z = zipfile.ZipFile(args.vctk_zip)
        have = set(z.namelist())
        by_spk = {}
        for utt, (spk, _sid, _t) in raw.items():
            if rfs.vctk_path(utt) in have:
                by_spk.setdefault(spk, []).append(utt)
        base = json.loads(Path(args.vctk_hnr_from).read_text())
        vctk = {s: h for s, h in base["vctk_hnr"].items() if s not in vheard and s in by_spk}
        print("%d VCTK voices unheard (%d heard)" % (len(vctk), len(vheard)))
        for i, m in enumerate(bb.match_speakers(vctk, pool, args.n, set(), args.max_gap)):
            cand = []
            for u in sorted(by_spk[m["vctk"]]):
                sec = sf.info(io.BytesIO(z.read(rfs.vctk_path(u)))).duration
                if args.min_seconds <= sec <= args.max_seconds:
                    cand.append((u, sec))
            if len(cand) < len(c["vctk_arms"]):
                raise SystemExit("REFUSING: VCTK %s has %d mic1 clips of %.1f-%.1fs."
                                 % (m["vctk"], len(cand), args.min_seconds, args.max_seconds))
            target = rng.choice(cand)[1]
            vgot = lb.assign_clips({m["vctk"]: cand}, c["vctk_arms"], rng, target)[m["vctk"]]
            secs = dict(cand)
            arms = {a: {"source": u, "seconds": round(secs[u], 2), "sid": cleaned[u][1],
                        "phonemes": cleaned[u][2], "text": raw[u][2]} for a, u in vgot.items()}
            arms.update(libri_arms(m["libri"], target))
            items.append({"unit": i, "vctk": m["vctk"], "vctk_hnr": m["vctk_hnr"],
                          "libri": m["libri"], "libri_hnr": m["libri_hnr"], "arms": arms})
    else:
        for i, s in enumerate(pick_voices(pool, args.n, args.check)):
            items.append({"unit": i, "libri": s, "libri_hnr": hnr[s],
                          "arms": libri_arms(s, args.target_seconds)})
    for it in items:
        print("  unit %2d  %s" % (it["unit"], "  ".join(
            "%s %s %.1fs" % (a, Path(str(x["source"])).stem, x["seconds"])
            for a, x in sorted(it["arms"].items()))))

    out = {"check": args.check,
           "rule": speakers_rule(args, c, len(items) if args.check == 4 else args.n),
           "ckpts": {a: [p, sha256(p)] if p else None for a, p in ckpts.items()},
           "inputs": {"vctk_filelist": [args.vctk_filelist, sha256(args.vctk_filelist)],
                      "vctk_cleaned": [args.vctk_cleaned, sha256(args.vctk_cleaned)],
                      "vctk_hnr_from": [args.vctk_hnr_from, sha256(args.vctk_hnr_from)],
                      "hnr_json": [args.hnr_json, sha256(args.hnr_json)],
                      "prior": args.prior, "prior_optional": args.prior_optional},
           "items": items}
    if args.check == 4:
        out["rule"] += (" Check 4: every eligible voice if %d or fewer (at least %d); "
                        "recordings in both S1's and L's train filelists whose original is "
                        "within %s s of the LibriTTS-R file's length."
                        % (args.n, c["min_units"], float(ORIG_MAX_DRIFT)))
        out["eligible"] = len(pool)
    if args.dry_run:
        out["dry_run"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote %s  sha256 %s" % (out_path, sha256(out_path)))


# ------------------------------------------------------------------------------ render

def render(args):
    spec = read_spec(args.speakers, args.check)
    c = sb.CHECKS[args.check]

    import soxr
    import torch
    import yaml

    from lib import ear_bench
    from matcha.cli import load_matcha, load_vocoder, to_waveform
    from matcha.text import text_to_sequence
    from matcha.utils.audio import mel_spectrogram
    from matcha.utils.utils import intersperse

    if set(spec["ckpts"]) != set(c["arms"]):
        raise SystemExit("REFUSING: the speakers file names arms %s." % sorted(spec["ckpts"]))
    for a, (p, digest) in spec["ckpts"].items():
        if sha256(p) != digest:
            raise SystemExit("REFUSING: %s's checkpoint %s changed since select." % (a, p))
    rng = random.Random(args.seed + args.check)
    ear_bench.refuse_if_judged(args.out)
    ear_bench.prove_writable(args.out)
    bench = ear_bench.Bench(args.out, args.salt, args.seed, "vat")
    sr = bench.sample_rate
    ours = yaml.safe_load(Path("configs/data/libritts_r_vat.yaml").read_text())
    if ours["sample_rate"] != sr:
        raise SystemExit("REFUSING: the vocoder is %d Hz and libritts_r_vat says %d."
                         % (sr, ours["sample_rate"]))
    # `mel_spectrogram` caches its mel basis by fmax and its window by device only.
    if sd.STOCK_MEL[6] == ours["f_max"] or sd.STOCK_MEL[4] != ours["win_length"]:
        raise SystemExit("REFUSING: the 22 and 24 kHz mel configs collide in "
                         "matcha.utils.audio's cache.")

    u_voc, u_den = load_vocoder("hifigan_univ_v1", args.stock_vocoder, ear_bench.DEVICE)

    def univ(mel):
        with torch.no_grad():
            w = to_waveform(mel, u_voc, u_den, rfs.DENOISER_STRENGTH).numpy()
        return soxr.resample(w.astype("float64"), sd.TARGET_SR, sr).astype("float32")

    def univ_rt(x, xsr):
        y = np.asarray(x, dtype="float64")
        if xsr != sd.TARGET_SR:
            y = soxr.resample(y, xsr, sd.TARGET_SR)
        y = torch.from_numpy(y.astype("float32"))[None]
        return univ(mel_spectrogram(y, *sd.STOCK_MEL, center=False))

    def ours_rt(x):
        y = torch.from_numpy(np.asarray(x, dtype="float32"))[None]
        mel = mel_spectrogram(y, *(ours[k] for k in OURS_MEL), center=False)
        with torch.no_grad():
            return to_waveform(mel, bench.vocoder, None).numpy()

    def mono(x):
        return x.mean(axis=1) if x.ndim > 1 else x

    def load_arm(arm):
        corpus, n_spks, use_vat = EXPECT[arm]
        model = load_matcha(arm, spec["ckpts"][arm][0], ear_bench.DEVICE)
        want = yaml.safe_load((Path("configs/data") / ("%s.yaml" % corpus)).read_text())
        want = (float(want["data_statistics"]["mel_mean"]),
                float(want["data_statistics"]["mel_std"]))
        got = (float(model.mel_mean), float(model.mel_std))
        if abs(got[0] - want[0]) > 1e-3 or abs(got[1] - want[1]) > 1e-3:
            raise SystemExit("REFUSING: %s normalises with %r; %s has %r."
                             % (arm, got, corpus, want))
        if (model.n_spks, model.n_vocab, bool(model.hparams.get("use_vat", False))) \
                != (n_spks, 178, use_vat):
            raise SystemExit("REFUSING: %s has n_spks %d, n_vocab %d, use_vat %s; expected "
                             "%d, 178, %s." % (arm, model.n_spks, model.n_vocab,
                                               model.hparams.get("use_vat"), n_spks, use_vat))
        return model, (int(model.hparams["vat_dim"]) if use_vat else 0)

    def synth(model, ids, spk, vat_dim, pair_key):
        xt = torch.tensor(intersperse(ids, 0), dtype=torch.long)[None]
        kw = {"vat": torch.zeros(1, vat_dim), "guidance": ear_bench.GUIDANCE} if vat_dim else {}
        torch.manual_seed(ear_bench.seed_for(pair_key, args.seed))
        with torch.no_grad():
            return model.synthesise(xt, torch.tensor([xt.shape[-1]]),
                                    n_timesteps=ear_bench.N_TIMESTEPS,
                                    temperature=ear_bench.TEMPERATURE,
                                    spks=torch.tensor([spk], dtype=torch.long),
                                    length_scale=ear_bench.LENGTH_SCALE, **kw)["mel"]

    plan = [(it, a) for it in spec["items"] for a in c["arms"]]
    rng.shuffle(plan)
    pair_of = {(it["unit"], a): "item_%02d" % i for i, (it, a) in enumerate(plan)}
    swaps = sb.side_plan(len(spec["items"]), c["arms"], rng)
    z = zipfile.ZipFile(args.vctk_zip) if c["vctk_arms"] else None

    made = {}
    for arm in c["arms"]:
        model, vat_dim = load_arm(arm)
        fam = sb.FAMILY[arm]
        for it in spec["items"]:
            pair_key, src = pair_of[(it["unit"], arm)], it["arms"][arm]
            if fam == "vctk22":
                x, xsr = rfs.vctk_audio(z, src["source"])
                rt = univ_rt(x, xsr)
                wave = univ(synth(model, bb.encode_original(src["phonemes"]), src["sid"], 0,
                                  pair_key))
            else:
                seq, _ = text_to_sequence(src["phonemes"], ["no_cleaners"])
                mel = synth(model, seq, it["libri"], vat_dim, pair_key)
                if fam in RT_22K:
                    rt_src = round_trip_source(arm, src["source"])
                    x, xsr = sf.read(rt_src, dtype="float64")
                    if xsr != sd.TARGET_SR:
                        raise SystemExit("REFUSING: %s is %d Hz." % (rt_src, xsr))
                    rt, wave = univ_rt(mono(x), xsr), univ(mel)
                else:
                    x, xsr = sf.read(src["source"], dtype="float32")
                    if xsr != sr:
                        raise SystemExit("REFUSING: %s is %d Hz, not %d." % (src["source"], xsr, sr))
                    with torch.no_grad():
                        wave = to_waveform(mel, bench.vocoder, None).numpy()
                    rt = ours_rt(mono(x))
            made[pair_key] = {arm + "rt": rt, arm: wave}
        del model

    clips = Path(args.out) / "clips"
    served, truth, limited = [], {}, 0
    for it, arm in plan:
        pair_key = pair_of[(it["unit"], arm)]
        lo, hi = arm + "rt", arm
        sides = [("A", hi), ("B", lo)] if swaps[it["unit"]][arm] else [("A", lo), ("B", hi)]
        item = {"id": pair_key, "set": "hum", "text": "(recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, lbl in sides:
            a, bound = ear_bench.match_loudness(made[pair_key][lbl], sr, args.lufs)
            limited += int(bound)
            nm = ear_bench.opaque(pair_key, side_key, args.salt)
            sf.write(str(clips / ("%s.wav" % nm)), a, sr, "PCM_24")
            bench.written += 1                 # render_meta.json and the summary count these
            bench.key[nm] = {"label": lbl, "pair": pair_key, "side": side_key}
            item[side_key] = nm
        served.append(item)
        truth[pair_key] = truth_entry(args.check, arm, it, sides)
        hnr = truth[pair_key]["hnr"]
        print("  %s  %-6s unit %2d  HNR %5.2f" % (pair_key, arm, it["unit"], hnr))
    served.sort(key=lambda i: i["id"])

    bench.write(Path(args.out).name, rfs.SETS, served,
                {"check": args.check, "commit": args.commit, "ckpts": spec["ckpts"],
                 "speakers": [args.speakers, sha256(args.speakers)], "lufs": args.lufs,
                 "peak_limited_sides": limited},
                key_out=args.key_out)
    k = json.loads(Path(args.key_out).read_text())
    k["check"], k["items"] = args.check, truth
    Path(args.key_out).write_text(json.dumps(k, indent=2))
    if limited:
        print("\n⚠ %d side(s) hit the peak ceiling before reaching %.1f LUFS."
              % (limited, args.lufs))
    print("\nRENDERED %d items. Key: %s" % (len(served), args.key_out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--check", type=int, choices=(1, 2, 3, 4), required=True)
    s.add_argument("--ckpt", action="append", default=[], metavar="ARM=PATH")
    s.add_argument("--dry-run", action="store_true",
                   help="pre-flight count: an arm given no --ckpt (not trained yet) is recorded "
                        "as null; the file is marked a dry run, may not be written into "
                        "staged_bench/, and render refuses it")
    s.add_argument("--vctk-zip", default=rfs.VCTK_ZIP)
    s.add_argument("--vctk-filelist",
                   default=rfs.BASELINE_DIR + "/vctk_audio_sid_text_train_filelist.txt")
    s.add_argument("--vctk-cleaned",
                   default=rfs.BASELINE_DIR + "/vctk_audio_sid_text_train_filelist.txt.cleaned")
    s.add_argument("--vctk-hnr-from", default=rfs.BASELINE_DIR + "/speakers.json")
    s.add_argument("--hnr-json", default=ROOT + "/pitch_error/speaker_hnr_all.json")
    s.add_argument("--prior", nargs="+", default=None,
                   help="earlier benches' files; each pattern must match at least one file "
                        "(default: `default_prior`; check 4 adds staged_bench/check{1,2,3})")
    s.add_argument("--prior-optional", nargs="*",
                   default=[ROOT + "/staged_bench/*_speakers.json"],
                   help="patterns that may match nothing (no staged speakers file exists "
                        "before check 1 is drawn)")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--min-seconds", type=float, default=4.0)
    s.add_argument("--max-seconds", type=float, default=8.0)
    s.add_argument("--target-seconds", type=float, default=6.0)
    s.add_argument("--max-gap", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=20261006)
    s.add_argument("--out-speakers", required=True)

    r = sub.add_parser("render")
    r.add_argument("--check", type=int, choices=(1, 2, 3, 4), required=True)
    r.add_argument("--speakers", required=True)
    r.add_argument("--commit", required=True,
                   help="the registered commit this render runs from (Task 10); recorded in "
                        "the render's metadata. The container copy has no .git to ask.")
    r.add_argument("--stock-vocoder", default=rfs.BASELINE_DIR + "/UNIVERSAL_V1_g_02500000")
    r.add_argument("--vctk-zip", default=rfs.VCTK_ZIP)
    r.add_argument("--out", required=True)
    r.add_argument("--key-out", required=True)
    r.add_argument("--lufs", type=float, default=-23.0)
    r.add_argument("--salt", default=None)
    r.add_argument("--seed", type=int, default=1618)

    args = ap.parse_args()
    if args.cmd == "select" and args.prior is None:
        args.prior = default_prior(args.check)
    elif args.cmd == "select":
        require_default_prior(args.check, args.prior)
    if args.cmd == "render" and args.salt is None:
        args.salt = "staged-check%d-v1" % args.check
    (select if args.cmd == "select" else render)(args)


if __name__ == "__main__":
    main()

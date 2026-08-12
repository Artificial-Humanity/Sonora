"""audit_sampler — select the human-audit subset from QC'd campaign renders.

Owner bandwidth doesn't cover auditing every clip (campaign scale, 2026-07-24),
so per-clip certification moves to a statistical gate: instruments certify the
bulk, humans audit a *targeted sample*. This script sits between qc_verdict.py
and the audition queue — it reads a book's qc_verdicts.jsonl and registers only
the selected clips into ratings.csv (via register_audition's row conventions),
with the selection reason in the note column.

Selection criteria (agreed 2026-07-24):
  1. disagreement-first  — hard_pass clips the instruments rejected on axis
     direction (keep=False): the exact clips where intended labels and measured
     affect disagree; the owner adjudicates instrument vs label.
  2. age-mismatch        — priority trigger: measured render F0 percentile
     (within-gender, pool-anchored) far from the intended_age band target.
     Age labels must describe the actual audio (training-attribution rule).
  3. near-miss tails     — hard_pass clips that squeaked past a gate
     (ASR WER 0.25-0.35, DNSMOS ovr 2.5-2.7): where instrument blind spots live.
  4. novelty             — engine x register combos with little audited history
     get extra eyes (a premier engine's first production outing = all novel).
  5. strata floors       — every engine x register x gender stratum present in
     the keeps contributes at least one sampled clip.
  6. random 5%           — unconditional humility quota over the keeps; catches
     failure modes none of the triggers anticipate.
Escalation: any stratum with >=2 instrument-rejects out of its clips is flagged
for full audit in the console report (acceptance-sampling rung).

Run (after qc_gate + eiv_score + qc_verdict have produced qc_verdicts.jsonl):
  .venv/bin/python scripts/tools/audit_sampler.py --book <book-prose/slug> [--dry-run]
  .venv/bin/python scripts/tools/audit_sampler.py --all [--dry-run]

Deps: numpy, librosa (render F0 for the age trigger), soundfile.
"""
import argparse
import csv
import json
import os
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib", "stages"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import register_audition as reg  # noqa: E402  (path/header/link conventions)
from ref_select import design_gender  # noqa: E402

POOL_ROOT = Path("/data/model-training/datasets/sonora-expressive-registers/v1")
POOL_ACOUSTICS = POOL_ROOT / "pool_acoustics.json"
POOL_META = POOL_ROOT / "metadata.jsonl"

# intended_age -> within-gender F0 percentile target. These must match ref_select's
# AGE_BANDS: "adult" is the 0.70 band (an EXPLICIT claim, checkable), not a stand-in
# for "unmarked" — unmarked designs now carry "" and are skipped below.
AGE_TARGETS = {"child": 0.95, "teen": 0.80, "adult": 0.70,
               "middle-aged": 0.30, "elderly": 0.10}
AGE_FLAG_DELTA = 0.40      # |measured_pct - target_pct| beyond this = mismatch
AGE_SKIP = {""}            # unmarked designs only: no age claim to betray
NEAR_MISS_WER = (0.25, 0.35)
NEAR_MISS_DNSMOS = (2.5, 2.7)
RANDOM_RATE = 0.05
NOVELTY_MIN_AUDITED = 6    # combo with fewer human-scored clips = still novel
NOVELTY_CAP = 2            # extra clips per novel combo per book
ESCALATE_FAILS = 2         # >= this many instrument-rejects in a stratum...
ESCALATE_RATE = 0.2        # ...AND >= this reject rate -> flag for full audit


def _pool_f0_percentile():
    """gender -> sorted F0 medians from the certified pool (the anchor scale;
    reference cloning copies the ref's F0 register within ~2%, so the pool
    distribution is the right yardstick for renders)."""
    ac = json.loads(POOL_ACOUSTICS.read_text())
    by_g = defaultdict(list)
    for line in POOL_META.open():
        m = json.loads(line)
        a = ac.get(m["file"])
        if a:
            by_g[m.get("gender", "?")[:1].upper()].append(a["f0_median"])
    return {g: np.sort(np.array(v)) for g, v in by_g.items() if len(v) >= 10}


def _render_f0_median(wav_path):
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=12.0)
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    voiced = f0[(f0 > 60) & (f0 < 500)]
    return float(np.median(voiced)) if len(voiced) > 20 else None


def _row_gender(r):
    g = (r.get("intended_gender") or (r.get("ref") or {}).get("gender") or "")
    if not g:
        g = design_gender((r.get("direction") or {}).get("design", ""))
    return g[:1].upper()


def _audited_combo_counts():
    """(engine, register) -> human-scored row count in ratings.csv."""
    counts = defaultdict(int)
    if reg.RATINGS_CSV.is_file():
        with open(reg.RATINGS_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("score") or "").strip():
                    counts[(r.get("engine", ""), r.get("register", ""))] += 1
    return counts


def sample_book(book: Path, pool_pct, combo_counts, rng, verbose=True):
    """Return {clip_id: [reasons]} for one book (needs qc_verdicts.jsonl)."""
    vpath = book / "qc_verdicts.jsonl"
    if not vpath.is_file():
        print(f"  ! {book.name}: no qc_verdicts.jsonl (run qc_verdict first); skipped",
              file=sys.stderr)
        return {}, []
    rows = [json.loads(l) for l in vpath.open() if l.strip()]
    picked = defaultdict(list)

    # 1. disagreement-first: structurally sound, label-rejected.
    # `is False`, not falsy: qc_verdict scores an axis it could not measure as None, and a
    # clip nobody scored is not a clip that disagreed. Sampling it as a disagreement sends
    # the ear a finding that was never made — and `not r.get("keep")` is true of an
    # unmeasured clip too, since unmeasured cannot confirm a keep.
    for r in rows:
        if r.get("hard_pass") and not r.get("keep"):
            checks = r.get("axis_checks") or {}
            fails = [a for a, ok in checks.items() if ok is False]
            if not fails:
                continue
            picked[r["id"]].append(f"axis-disagree:{'+'.join(fails)}")

    # 2. age-mismatch (priority): measured render F0 vs intended band
    for r in rows:
        age = r.get("intended_age")
        if not age or age in AGE_SKIP or not r.get("hard_pass"):
            continue
        g = _row_gender(r)
        if g not in pool_pct:
            continue
        f0 = _render_f0_median(r["wav_abs"])
        if f0 is None:
            continue
        pct = float(np.searchsorted(pool_pct[g], f0) / len(pool_pct[g]))
        target = AGE_TARGETS[age]
        if abs(pct - target) > AGE_FLAG_DELTA:
            picked[r["id"]].append(f"age-mismatch:{age}@p{target:.0%}->p{pct:.0%}")

    # 3. near-miss tails
    for r in rows:
        if not r.get("hard_pass"):
            continue
        if NEAR_MISS_WER[0] < r.get("asr_wer", 0) <= NEAR_MISS_WER[1]:
            picked[r["id"]].append(f"near-miss:wer={r['asr_wer']:.2f}")
        if NEAR_MISS_DNSMOS[0] <= r.get("dnsmos_ovr", 9) < NEAR_MISS_DNSMOS[1]:
            picked[r["id"]].append(f"near-miss:dnsmos={r['dnsmos_ovr']:.2f}")

    keeps = [r for r in rows if r.get("keep")]

    # 4. novelty: low-history engine x register combos
    by_combo = defaultdict(list)
    for r in keeps:
        by_combo[(r["engine"], r["register"])].append(r)
    for combo, rs in sorted(by_combo.items()):
        if combo_counts.get(combo, 0) < NOVELTY_MIN_AUDITED:
            fresh = [r for r in rs if r["id"] not in picked]
            for r in rng.sample(fresh, min(NOVELTY_CAP, len(fresh))):
                picked[r["id"]].append(f"novel-combo:{combo[0]}/{combo[1]}")

    # 6. random humility quota (before floors so floors see the full picture)
    n_rand = max(1, round(RANDOM_RATE * len(keeps))) if keeps else 0
    for r in rng.sample(keeps, min(n_rand, len(keeps))):
        if "random" not in picked[r["id"]]:
            picked[r["id"]].append("random")

    # 5. strata floors over engine x register x gender
    by_stratum = defaultdict(list)
    for r in keeps:
        by_stratum[(r["engine"], r["register"], _row_gender(r))].append(r)
    for stratum, rs in sorted(by_stratum.items()):
        if not any(r["id"] in picked for r in rs):
            r = rng.choice(rs)
            picked[r["id"]].append(f"stratum-floor:{'/'.join(stratum)}")

    # escalation report (acceptance-sampling rung)
    escalations = []
    strat_all = defaultdict(lambda: [0, 0])
    for r in rows:
        s = strat_all[(r["engine"], r["register"], _row_gender(r))]
        s[0] += 1
        s[1] += 0 if r.get("keep") else 1
    for stratum, (n, fails) in sorted(strat_all.items()):
        if fails >= ESCALATE_FAILS and fails / n >= ESCALATE_RATE:
            escalations.append((stratum, fails, n))

    if verbose:
        print(f"  {book.name}: {len(picked)}/{len(rows)} sampled "
              f"({len(keeps)} keeps, {len(rows) - len(keeps)} rejects)")
    return dict(picked), escalations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", action="append", default=[], metavar="DIR")
    ap.add_argument("--all", action="store_true",
                    help=f"every book under {reg.BOOK_PROSE_ROOT} with qc_verdicts")
    ap.add_argument("--seed", type=int, default=51, help="deterministic sampling")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    books = [Path(b).resolve() for b in args.book]
    if args.all:
        books += sorted(d for d in reg.BOOK_PROSE_ROOT.iterdir()
                        if (d / "qc_verdicts.jsonl").is_file())
    if not books:
        ap.error("give --book <dir> (repeatable) or --all")

    pool_pct = _pool_f0_percentile()
    combo_counts = _audited_combo_counts()
    fields, existing_ids = reg._read_header_and_ids()
    print(f"ratings.csv: {reg.RATINGS_CSV} ({len(existing_ids)} existing ids); "
          f"pool F0 anchor: {{{', '.join(f'{g}:{len(v)}' for g, v in pool_pct.items())}}}")

    all_rows, all_escalations = [], []
    for book in books:
        rng = random.Random(args.seed + zlib.crc32(book.name.encode()))
        picked, escalations = sample_book(book, pool_pct, combo_counts, rng)
        all_escalations += [(book.name, *e) for e in escalations]
        if not picked:
            continue
        verdicts = {r["id"]: r for l in (book / "qc_verdicts.jsonl").open()
                    if l.strip() for r in [json.loads(l)]}
        sample_out = book / "audit_sample.jsonl"
        if not args.dry_run:
            with open(sample_out, "w", encoding="utf-8") as f:
                for cid, reasons in sorted(picked.items()):
                    f.write(json.dumps({"id": cid, "reasons": reasons}) + "\n")
        for cid, reasons in sorted(picked.items()):
            if cid in existing_ids:
                continue
            r = verdicts[cid]
            wav = Path(r["wav_abs"])
            if not wav.is_file() or not reg._under_data_root(wav):
                print(f"  ! {cid}: wav missing/outside data root; not queued",
                      file=sys.stderr)
                continue
            row = reg._build_row(r, wav, book.name, fields)
            if "note" in row:
                row["note"] = "audit-sample: " + "; ".join(reasons)
            all_rows.append(row)
            existing_ids.add(cid)

    for book, stratum, fails, n in all_escalations:
        print(f"  !! ESCALATE {book} {'/'.join(stratum)}: {fails}/{n} "
              f"instrument-rejects — audit this stratum fully")

    if not all_rows:
        print("nothing new to queue.")
        return
    if args.dry_run:
        print(f"[dry-run] {len(all_rows)} audit rows would be appended")
        return
    # Use register_audition's guarded append, not a bare one. The audition app
    # saves by read-all -> modify -> write tmp -> os.replace, so rows appended
    # inside that window vanish when the swap lands — silently, after this
    # script has already printed success and written audit_sample.jsonl. The
    # clips at risk are exactly the ones chosen because the instruments
    # distrust them (disagreement, near-miss, novelty), and the race window
    # spans the whole slow F0-measurement run that precedes this point.
    reg.append_guarded(all_rows, fields)   # prints its own confirmation
    print("AUDIT-SAMPLER-DONE")


if __name__ == "__main__":
    main()

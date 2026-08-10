# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy", "soundfile", "librosa>=0.10", "unidecode", "pyyaml",
# ]
# ///
"""v6 = v5's rows verbatim + the expressive-registers append set (rung 2).

The third corpus shape, and the same merge discipline as `merge_emilia_corpus.py`:
**every rung ADDS rows without re-rolling what came before**, which is what makes
rung n's holdout number comparable to rung n-1's. v5's train/val files pass through
BYTE-IDENTICAL and the hash split is re-checked against them rather than trusted.

WHY THIS IS A THIRD SCRIPT. `derive_vat_corpus.py` walks a LibriTTS-R speaker/chapter
tree; `merge_emilia_corpus.py` reads extracted-tar keeps with per-speaker manifests.
This bank is flat, keyed by `ratings.csv`, ear-certified, and its labels were computed
ahead of time by `label_expressive_registers.py`. Four things differ from Emilia and
each one is a place a copied script would have failed silently:

  1. **Labels arrive pre-computed.** V/A/T come from `labels_v6.jsonl` (global anchor,
     A centred per campaign), not from a label pass run here. This script REFUSES to
     compute a label — if a clip is not in that file it is dropped, never labelled on
     the fly, because a second labelling path is a second scale.
  2. **Delivery is KNOWN for 822 of 832.** Emilia's rows are all `unknown`/all-zero.
     These carry real lanes, and they are the only delivery signal in the corpus:
     v5 holds 48 labelled rows in 42,442.
  3. **One speaker id per clip** (owner, 2026-08-09). Identity is not recoverable —
     880 of 1,004 ids carry no `_sSEED` suffix and only 114 rows in any manifest carry
     a reference/voice field — so per-engine ids would assert that all 147 zonos clips
     are one person. The precedent is already in the corpus: v5 carries 756 one-clip
     speakers.
  4. **148 clips are 44.1 kHz.** The corpus contract is 24 kHz and `_audio_ok` refuses
     anything else, so audio is STAGED to 24 kHz first — the same shape as Emilia's
     `emilia_kept_24k`. Staging at 24 kHz is also exactly how the measures were taken
     (`librosa.load(sr=24000)`), so labels and audio agree by construction.

⚠ **TEXT COMES FROM THREE PLACES AND ONE OF THEM NEEDS A FALLBACK.** Per-directory
`*_manifest.jsonl`, then `v1/metadata.jsonl`. 21 rows are `mk_`-prefixed twins whose
audio is a v1 clip and whose text is filed under the UN-prefixed id — without the
fallback they resolve to nothing, and a clip with no text is a clip that cannot be
phonemised. All 832 resolve with it.

⚠ **The dedup kept the `mk_` side, not the original.** `quality-gap-plan.md` § Rung 2
prerequisite 3 says "the original-campaign row is kept and the `mk_` twin dropped" —
the artifact does the reverse for these 21. The OUTCOME is sound and verified here
(832 rows, 832 distinct wavs, 0 already in v5, 0 `mk_`/original pairs both present),
so this is a documentation mismatch rather than a corpus defect. It is called out
because "one row per wav" is the property that matters and it is asserted below, not
assumed.

Run:
    .venv/bin/python scripts/merge_expressive_registers.py \
        --out data/libritts_r_emilia_expressive_vat_v6 --dry-run
"""

import argparse
import collections
import json
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "synthesis"))

BASE = "/data/model-training/sonora/data/libritts_r_emilia_vat_v5"
LABELS = ("/data/model-training/sonora/expressive_registers_measures/labels_v6.jsonl")
BANK = "/data/model-training/datasets/sonora-expressive-registers"
STAGE = "/data/model-training/datasets/expressive_registers_24k"
TARGET_SR = 24000


def _resolve_text(rows):
    """-> {clip_id: text}, from the manifests then v1/metadata.jsonl, with the mk_ fallback."""
    import glob

    text = {}
    for d in sorted({os.path.dirname(r["wav"]) for r in rows}):
        for m in sorted(glob.glob(os.path.join(d, "*_manifest.jsonl"))):
            with open(m, encoding="utf-8") as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    if rec.get("id") and rec.get("text"):
                        text.setdefault(rec["id"], rec["text"])
    v1 = os.path.join(BANK, "v1", "metadata.jsonl")
    if os.path.exists(v1):
        with open(v1, encoding="utf-8") as fh:
            for ln in fh:
                rec = json.loads(ln)
                if rec.get("id") and rec.get("text"):
                    text.setdefault(rec["id"], rec["text"])

    out = {}
    for r in rows:
        cid = r["id"]
        got = text.get(cid)
        if got is None and cid.startswith("mk_"):
            got = text.get(cid[3:])       # the twin's audio is a v1 clip; its text is too
        if got:
            out[cid] = got
    return out


def _stage_24k(rows, stage_dir, force=False):
    """Write every clip as 24 kHz mono PCM under `stage_dir/wavs`, and return {id: path}.

    Idempotent: a staged file already at 24 kHz with the expected frame count is left
    alone, so a re-run costs a stat rather than a resample. The corpus points at the
    staged tree and NOT at the bank, so a later reroll in the bank cannot silently
    change what a built corpus trained on — the failure the loudnorm sidecar hit when
    it keyed on path and a reroll re-rendered in place.
    """
    import librosa
    import soundfile as sf

    wavs = os.path.join(stage_dir, "wavs")
    os.makedirs(wavs, exist_ok=True)
    out, resampled, reused = {}, 0, 0
    for r in rows:
        dst = os.path.join(wavs, r["id"] + ".wav")
        if not force and os.path.exists(dst):
            try:
                if sf.info(dst).samplerate == TARGET_SR:
                    out[r["id"]] = dst
                    reused += 1
                    continue
            except Exception:
                pass
        y, _ = librosa.load(r["wav"], sr=TARGET_SR, mono=True)
        sf.write(dst, y, TARGET_SR, subtype="PCM_16")
        out[r["id"]] = dst
        resampled += 1
    print(f"staged {len(out)} clips -> {wavs}  ({resampled} written, {reused} reused)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="corpus directory to write")
    ap.add_argument("--base", default=BASE, help=f"corpus to merge into (default {BASE})")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--restage", action="store_true", help="re-resample even if staged")
    ap.add_argument("--no-neural-oov", action="store_true")
    ap.add_argument("--max-drop-frac", type=float, default=0.10,
                    help="abort if more than this fraction of the append set is dropped. "
                         "Tighter than the Emilia merge's 0.35 on purpose: these rows are "
                         "ear-certified and pre-labelled, so a large drop means a BUG here, "
                         "not a dirty source.")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    from matcha.text import op_g2p
    from matcha.text.op_g2p import OpenPhonemizerG2P
    from derive_vat_corpus import VAL_FRACTION, _in_val
    from matcha.delivery import DELIVERY_LANES, vat_vector

    try:
        from matcha.data.license_wall import enforce as license_check
    except ImportError:                                    # older tree
        license_check = None

    # ---- the append set, already labelled --------------------------------------------
    rows = [json.loads(ln) for ln in open(args.labels, encoding="utf-8") if ln.strip()]
    print(f"append set: {len(rows)} rows from {args.labels}")

    # One row per wav is THE property that makes a speaker table honest here, because
    # every clip gets its own speaker id. Two ids on one wav would hand one voice two
    # rows — the exact collision merge_emilia_corpus's check exists to stop.
    by_wav = collections.Counter(r["wav"] for r in rows)
    dupes = [w for w, n in by_wav.items() if n > 1]
    if dupes:
        raise SystemExit(f"ABORT: {len(dupes)} wav(s) appear under more than one id, e.g. "
                         f"{dupes[:2]} — one voice would get two speaker rows.")

    parts, base_spk = {}, None
    for name in ("train_op.txt", "val_op.txt"):
        with open(os.path.join(args.base, name), encoding="utf-8") as fh:
            parts[name] = [ln.rstrip("\n") for ln in fh if ln.strip()]
    with open(os.path.join(args.base, "speakers.json"), encoding="utf-8") as fh:
        base_spk = json.load(fh)
    print(f"base {args.base}: {len(parts['train_op.txt'])} train / "
          f"{len(parts['val_op.txt'])} val · n_spks {base_spk['n_spks']}")

    # v5's own split must reproduce under the shared hash, or this merge is re-rolling it.
    for name, side in (("val_op.txt", True), ("train_op.txt", False)):
        wrong = [r for r in parts[name] if _in_val(r) != side]
        if wrong:
            raise SystemExit(
                f"ABORT: {len(wrong)} of {len(parts[name])} base {name} rows land on the "
                f"other side under the shared hash split. SPLIT_SALT or the wav paths have "
                f"moved, and merging would re-roll a split that exists not to move.\n"
                f"  first: {wrong[0].split('|', 1)[0]}")
    print("base split reproduces under the shared hash: OK")

    base_wavs = {ln.split("|", 1)[0] for ln in parts["train_op.txt"] + parts["val_op.txt"]}
    already = [r for r in rows if r["wav"] in base_wavs]
    if already:
        raise SystemExit(f"ABORT: {len(already)} append row(s) whose audio is ALREADY in the "
                         f"base, e.g. {already[0]['id']} — the dedup did not hold.")
    print("no append row duplicates base audio: OK")

    texts = _resolve_text(rows)
    print(f"text resolved for {len(texts)}/{len(rows)}")

    staged = _stage_24k(rows, args.stage, force=args.restage)

    # ---- build the new rows ----------------------------------------------------------
    g2p = OpenPhonemizerG2P(use_neural_oov=not args.no_neural_oov, homographs=True)
    dropped, examples = collections.Counter(), collections.defaultdict(list)
    new, lanes = [], collections.Counter()

    def drop(reason, cid, detail=""):
        dropped[reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(f"{cid}: {detail}"[:110])

    from derive_vat_corpus import MAX_SECONDS, MIN_SECONDS
    import soundfile as sf

    for r in sorted(rows, key=lambda x: x["id"]):
        cid = r["id"]
        wav = staged[cid]
        info = sf.info(wav)
        seconds = info.frames / info.samplerate
        if info.samplerate != TARGET_SR:
            drop("audio", cid, f"sample_rate {info.samplerate}")
            continue
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            # The 14 over-length rows were dropped upstream (owner, 2026-08-10); anything
            # landing here is a staging bug, not a corpus decision.
            drop("duration", cid, f"{seconds:.1f}s outside [{MIN_SECONDS}, {MAX_SECONDS}]")
            continue

        text = (texts.get(cid) or "").strip()
        if not text:
            drop("no text", cid)
            continue
        if op_g2p.carries_digits(text):
            drop("digits", cid, text[:60])
            continue
        before = g2p.stats["oov_misses"]
        ipa = g2p.phonemize(text)
        if g2p.stats["oov_misses"] > before:
            drop("unresolved word", cid, text[:60])
            continue
        if g2p.validate(ipa):
            drop("vocab violation", cid, text[:60])
            continue

        lane = r.get("delivery") or ""
        if lane and lane not in DELIVERY_LANES:
            # Documentary was retired into Neutral on 2026-08-10. A lane that is no longer
            # in the vocabulary must stop the build rather than silently become `unknown`,
            # which is a real conditioning value and would look exactly like a blank.
            drop("unknown delivery lane", cid, lane)
            continue
        lanes[lane or "(blank)"] += 1

        vec = vat_vector(r["v"], r["a"], r["t"], lane)
        lab = ",".join([f"{x:.4f}" for x in vec[:3]] + [f"{int(x)}" for x in vec[3:]])
        new.append((cid, f"{wav}|{{spk}}|{ipa}|{lab}", seconds))

    n_drop = sum(dropped.values())
    print(f"\nkept {len(new)} of {len(rows)}; dropped {n_drop}:")
    for reason, n in dropped.most_common():
        print(f"  {n:5d}  {reason}")
        for ex in examples[reason]:
            print(f"         {ex}")
    print(f"delivery lanes kept: {dict(lanes)}")

    s_ = g2p.stats
    tot = sum(s_.values()) or 1
    print(f"G2P: dict {100 * s_['dict_hits'] / tot:.2f}% | contractions "
          f"{s_['contraction_hits']} | neural {s_['neural_hits']} | homograph flips "
          f"{s_['homograph_flips']} | unresolved {s_['oov_misses']}")

    if n_drop / max(len(rows), 1) > args.max_drop_frac:
        raise SystemExit(
            f"ABORT: dropped {100 * n_drop / len(rows):.1f}% of an ear-certified, "
            f"pre-labelled append set, over the {100 * args.max_drop_frac:.0f}% ceiling. "
            f"At this stage a large drop is a bug in this script, not a dirty source.")
    if not new:
        raise SystemExit("ABORT: no append row survived. Nothing to merge.")

    # ---- speaker numbering: one id per clip, appended after the base ------------------
    base_index = {}
    for key in ("libritts_id_to_index", "emilia_id_to_index"):
        base_index.update(base_spk.get(key, {}))
    next_index = base_spk["n_spks"]
    if sorted(base_index.values()) != list(range(next_index)):
        raise SystemExit("ABORT: the base speaker map is not contiguous 0..n-1; appending "
                         "to it would collide or leave holes.")
    er_index = {f"er_{cid}": next_index + i for i, (cid, _, _) in enumerate(sorted(new))}
    collision = set(base_index) & set(er_index)
    if collision:
        raise SystemExit(f"ABORT: {len(collision)} speaker id(s) appear in both namespaces, "
                         f"e.g. {sorted(collision)[:3]} — one voice would get two rows.")
    n_spks = next_index + len(er_index)

    rendered = [row.format(spk=er_index[f"er_{cid}"]) for cid, row, _ in new]
    seconds_added = sum(s for _, _, s in new)
    val_new = sorted(r for r in rendered if _in_val(r))
    train_new = sorted(r for r in rendered if not _in_val(r))
    print(f"\nspeakers: {next_index} base + {len(er_index)} expressive = {n_spks}")
    print(f"hash split on the new rows: {len(val_new)} val / {len(train_new)} train "
          f"({len(val_new) / len(rendered):.2%}, target {VAL_FRACTION:.0%})")

    train = parts["train_op.txt"] + train_new
    val = parts["val_op.txt"] + val_new
    print(f"merged: {len(train)} train / {len(val)} val "
          f"(+{len(rendered)} rows, +{seconds_added / 3600:.2f} h)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(args.out, exist_ok=True)
    for name, part in (("val_op.txt", val), ("train_op.txt", train)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(part) + "\n")
        print(f"wrote {len(part)} rows -> {os.path.join(args.out, name)}")

    if license_check:
        license_check([os.path.join(args.out, n) for n in ("train_op.txt", "val_op.txt")])
        print("licence wall: accepted")
    else:
        print("⚠ license_check not importable — declare the staged dir in "
              "configs/data_licenses.yaml before training.")

    with open(os.path.join(args.out, "speakers.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_spks": n_spks, "base_id_to_index": base_index,
                   "expressive_id_to_index": er_index}, fh, indent=2)

    report = {
        "base": args.base,
        "derivation": (
            "v6 = v5 rows verbatim + the sonora-expressive-registers append set. Base "
            "labels are unchanged (LibriTTS per-speaker z, Emilia global anchor). The "
            "appended rows are labelled on the SAME global anchor as Emilia "
            "(label_expressive_registers.py), with the A channel centred PER CAMPAIGN "
            "because the bank is loudnorm'd to -23 LUFS against LibriTTS-R's -18.16 and "
            "the uncorrected label pinned A at -1 on 94.4% of rows. One speaker id per "
            "clip (owner 2026-08-09): identity is not recoverable and per-engine ids "
            "would assert that every clip from one engine is one person."),
        "base_rows": {"train": len(parts["train_op.txt"]), "val": len(parts["val_op.txt"])},
        "expressive": {
            "candidates": len(rows),
            "kept": len(new),
            "dropped": dict(dropped),
            "seconds_total": float(seconds_added),
            "train": len(train_new),
            "val": len(val_new),
            "delivery_lanes": dict(lanes),
            "labels_from": args.labels,
            "staged_audio": args.stage,
        },
        "n_spks": n_spks,
        "n_spks_base": next_index,
        "license": ("LibriTTS-R CC-BY-4.0 + Emilia-YODAS CC-BY-4.0 (YODAS subset ONLY) + "
                    "sonora-expressive-registers (ours; teacher renders + cleared real "
                    "audio — see configs/data_licenses.yaml)"),
    }
    with open(os.path.join(args.out, "derivation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("report ->", os.path.join(args.out, "derivation_report.json"))
    print("\nNEXT, and neither is optional:")
    print("  1. data_statistics must be RE-MEASURED in-container. They cannot be inherited "
          "from v5 — this changes both the audio set and the split.")
    print("  2. scripts/test_vat_dim_seams.py must pass (vat_dim is unchanged at 8, but "
          "the filelist is new).")


if __name__ == "__main__":
    main()

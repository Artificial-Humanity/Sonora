# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy", "soundfile", "unidecode", "pyyaml", "ai-edge-litert", "ml_dtypes",
# ]
# ///
"""Phase 1 #1 — merge the Emilia-YODAS keeps into the v4 VAT corpus.

HOW TO RUN IT (owner decision 2026-08-01): the repo **.venv**, not `uv run`.

    .venv/bin/python scripts/merge_emilia_corpus.py --out data/libritts_r_emilia_vat_v5

THE ONE THING THIS SCRIPT IS FOR: v4's rows pass through BYTE-IDENTICAL, and the Emilia
rows are appended with speaker indices that continue v4's numbering. That is not a
stylistic preference — it is what makes the warm start legal. `make_warmstart
--donor-speakers` widens `spk_emb.weight` only after proving the donor's index map is a
PREFIX of the new one, because row *i* of a speaker table is a person and renumbering
silently reassigns every voice the model has learned. If this script ever rewrites a
LibriTTS row, that proof fails loudly, which is the intended behaviour.

WHY IT IS A SCRIPT AND NOT `derive_vat_corpus --root`
-----------------------------------------------------
The derive lane computes V/A/T as a **per-speaker z-score**, which is correct for LibriTTS
(247 speakers, ~127 clips each) and destroys Emilia (2,408 speakers, median 3 clips). The
full argument is in `anchor_emilia_labels.py`; the short version is that re-centring a
tail-selected clip on its own speaker's mean hands 756 one-clip speakers a label of exactly
0.0, and those clips were selected FOR being extreme. Emilia is therefore labelled on the
**global anchor** — the owner's option 1, 2026-08-08 — and that anchor is derived from v4,
which is the corpus this merges into. So the two halves reach the model on one scale
without either half being relabelled.

Nothing here reimplements a rule that exists elsewhere. The labels come from
`anchor_emilia_labels`, the split from `derive_vat_corpus._in_val`, the conditioning vector
from `matcha.delivery.vat_vector`, the phonemes from `matcha.text.op_g2p`, the WER
threshold from `synth_common.ASR_MAX_WER`, and the provenance check from
`matcha.data.license_wall`. Every one of those has already been the site of a
two-implementations-drifted defect.

WHAT GETS DROPPED, AND WHY EACH DROP IS A DROP RATHER THAN A FIX
----------------------------------------------------------------
The todo item says "all 13,141 rows". It will not be all 13,141, and the gap is reported
per reason rather than as a total, because each reason is a different kind of unusable.

**DIGITS (~13%).** D-M3: the tokenizer DELETES digits rather than expanding them, so
"320 members" trains as "members" against audio that says "three hundred and twenty
members" — a transcript missing a word the audio speaks, which no downstream gate can
detect. `derive_vat_corpus` refuses the whole corpus over this; here it is a per-clip drop,
because 13% of a corpus is not a reason to refuse the other 87%.

NOT normalised to words, deliberately. It reads like a table job and is not: "1 Chronicles"
is *First* Chronicles, "Ezr. 232" is "two thirty-two", "100%" is "a hundred percent", "3D"
is "three-dee". 1,262 distinct digit tokens over 2,771 occurrences. Guessing wrong produces
a transcript that looks right, matches the audio nowhere, and is invisible — the exact
failure D-M3 exists to prevent, arriving through the fix instead of the bug. Recovering
these is a separate, testable piece of work; the drop is unbiased across the mining criteria
(T+ 13.9%, V+ 14.9%, V- 10.8%, A+ 12.5%) and spread over 809 speakers, so it costs volume
and not tail-selection.

**CAPTION vs ASR (WER > ASR_MAX_WER).** `process_emilia_tail` recorded a faster-whisper
transcript per clip and wrote down that "the drop threshold is a merge-time decision (stage
2, labels/filelists) so the filter can be tuned without re-running ASR". This is stage 2.
A high WER here means the CAPTION is unreliable, never that the audio is bad — YODAS audio
is a real human recording and cannot be wrong about what it says.

⚠ The ASR is `base.en` int8 on CPU, so a good part of the WER is the cross-check being
weak rather than the caption being wrong ("Clade 2" -> "Clay two", "Tochigi" -> "told
Chiki", one clip transcribed as "=== SARSAT ==="). That argues for a LOOSE threshold, and
0.35 is loose: it drops 3.6%, and the band it drops is where captions are visibly missing
speech the audio contains ("team of the season's sony. You're probably thinking..." against
a caption that starts at "Now you're probably thinking"). A tight threshold here would
mostly measure whisper.

**UNRESOLVED WORDS.** `derive_vat_corpus` aborts the run if any word is unresolved, because
an unresolved word ships as literal letters and `validate()` cannot catch it (a-z are in the
vocab). LibriTTS is read literature and hits zero; Emilia is conversational web speech and
will not. The abort is right at corpus scale and wrong per clip, so this drops the clip and
keeps the count — the per-clip analogue of the same refusal.

**UNLABELLABLE / BAD AUDIO.** No probe measure or EIV head (cannot be labelled at all — the
anchor script already refuses to write these), wrong sample rate, or duration outside the
derive lane's own [MIN_SECONDS, MAX_SECONDS]. Read from the wav, not from the manifest: the
manifest duration is the source mp3's and the corpus points at the transcode.

A DROP-RATE CEILING, because a filter that eats the corpus should stop rather than succeed.
`--max-drop-frac` (default 0.35) aborts before writing anything.
"""

import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "synthesis"))

BASE = "data/libritts_r_vat_v4"
KEPT_24K = "/data/model-training/datasets/emilia_kept_24k"


def _emilia_wav(base):
    return os.path.join(KEPT_24K, "wavs", base + ".wav")


def _audio_ok(path):
    """(seconds, None) if this wav can enter the corpus, else (None, reason)."""
    import soundfile as sf

    from derive_vat_corpus import MAX_SECONDS, MIN_SECONDS, SAMPLE_RATE

    try:
        info = sf.info(path)
    except Exception:
        return None, "unreadable"
    if info.samplerate != SAMPLE_RATE:
        return None, f"sample_rate {info.samplerate}"
    seconds = info.frames / info.samplerate
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return None, "duration"
    return seconds, None


def _base_rows(base_dir):
    """v4's rows exactly as they are on disk, plus the map they were numbered with."""
    parts = {}
    for name in ("train_op.txt", "val_op.txt"):
        with open(os.path.join(base_dir, name), encoding="utf-8") as fh:
            parts[name] = [ln.rstrip("\n") for ln in fh if ln.strip()]
    with open(os.path.join(base_dir, "speakers.json"), encoding="utf-8") as fh:
        spk = json.load(fh)
    return parts, spk


def _check_split_agrees(rows, in_val, side):
    """v4's own split must reproduce under the shared hash, or the merge is re-rolling it.

    The hash split's whole promise is that a clip's side is a property OF THE CLIP, so
    growth never moves anything. That promise is checkable rather than assumed, and this is
    the moment to check it: if `_in_val` disagrees with the file v4 shipped, then either the
    salt moved or the paths did, and every cross-version val comparison silently stops
    meaning anything — which is the defect the hash split replaced.
    """
    wrong = [r for r in rows if in_val(r) != side]
    if wrong:
        raise SystemExit(
            f"ABORT: {len(wrong)} of {len(rows)} v4 {'val' if side else 'train'} rows land "
            f"on the other side under the shared hash split. SPLIT_SALT or the wav paths "
            f"have moved, and merging would re-roll a split that exists not to move.\n"
            f"  first: {wrong[0].split('|', 1)[0]}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="corpus directory to write")
    ap.add_argument("--base", default=BASE, help=f"corpus to merge into (default {BASE})")
    ap.add_argument("--max-drop-frac", type=float, default=0.35,
                    help="abort if this fraction of Emilia keeps is dropped (default 0.35)")
    ap.add_argument("--no-neural-oov", action="store_true",
                    help="dictionary-only G2P; every OOV word drops its clip")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    import anchor_emilia_labels as anchor_mod
    from synth_common import ASR_MAX_WER, edge_loss

    from derive_vat_corpus import VAL_FRACTION, _in_val
    from matcha.data.license_wall import classify_path
    from matcha.data.license_wall import enforce as license_check
    from matcha.delivery import vat_vector
    from matcha.text.op_g2p import OpenPhonemizerG2P

    # Pre-flight the provenance of everything this will touch, BEFORE ~13k phonemizations.
    # The authoritative check is `license_check` on the written filelists at the end; this
    # is the same question asked early, because the answer does not change and finding out
    # afterwards costs the whole run. An undeclared `--out` is the likely miss: the wall
    # classifies the filelist's own directory, so a new corpus dir needs a manifest entry
    # even though it holds no audio.
    for p in (args.out, args.base, os.path.join(KEPT_24K, "wavs")):
        hit = classify_path(p)
        if hit is None:
            raise SystemExit(
                f"ABORT: {p} matches no declared dataset. The licence wall would refuse "
                f"this corpus at load; declare it in configs/data_licenses.yaml first.")
        if hit[1] == "nc":
            raise SystemExit(f"ABORT: {p} -> {hit[0]} ({hit[2]}) is NON-COMMERCIAL. "
                             f"NC data is de-risk-only and must not enter a corpus.")

    # The anchor is derived from the corpus we are merging INTO, so the two halves land on
    # one scale. Pointing it at a different corpus than --base would label Emilia against a
    # distribution the LibriTTS rows do not have.
    anchor_mod.CORPUS = args.base

    parts, base_spk = _base_rows(args.base)
    base_index = base_spk["libritts_id_to_index"]
    print(f"base: {args.base} — {len(parts['train_op.txt'])} train / "
          f"{len(parts['val_op.txt'])} val, n_spks {base_spk['n_spks']}")
    _check_split_agrees(parts["train_op.txt"], _in_val, False)
    _check_split_agrees(parts["val_op.txt"], _in_val, True)
    print("  hash split reproduces v4's own train/val exactly — safe to grow")

    print("deriving the global anchor from the base corpus...")
    anchor = anchor_mod.libritts_anchor()
    print(f"  {anchor['n_clips']} clips, {anchor['n_heads']} with EIV heads, "
          f"{len(anchor['weights'])} weighted heads")

    rows, missing = anchor_mod.emilia_rows()
    print(f"emilia: {len(rows)} keeps carry both a probe measure and EIV heads"
          + (f", {len(missing)} do not and cannot be labelled" if missing else ""))

    asr = {}
    with open(os.path.join(KEPT_24K, "asr.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            asr[os.path.splitext(r["file"])[0]] = r

    g2p = OpenPhonemizerG2P(use_neural_oov=not args.no_neural_oov, homographs=True)
    dropped = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    emilia_rows_out, spk_seen, edges = [], {}, []

    def drop(reason, base, detail=""):
        dropped[reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(f"{base}: {detail}"[:110])

    for i, (base, row) in enumerate(sorted(rows.items())):
        wav = _emilia_wav(base)
        seconds, why = _audio_ok(wav)
        if why:
            drop("audio", base, why)
            continue

        rec = asr.get(base)
        if rec is None:
            drop("no ASR cross-check", base)
            continue
        if rec["wer"] > ASR_MAX_WER:
            drop("caption WER", base, f"wer={rec['wer']:.2f} :: {rec['text'].strip()[:60]}")
            continue

        text = (row.get("text") or "").strip()
        if not text:
            drop("empty caption", base)
            continue
        # D-M3, per clip. See the module docstring for why these are not normalized.
        if re.search(r"[0-9]", text):
            drop("digits", base, text[:60])
            continue

        # The per-clip form of derive's global OOV abort: an unresolved word ships as bare
        # letters and validate() cannot see it, so the clip goes rather than the run.
        before = g2p.stats["oov_misses"]
        ipa = g2p.phonemize(text)
        if g2p.stats["oov_misses"] > before:
            drop("unresolved word", base, text[:60])
            continue
        if g2p.validate(ipa):
            drop("vocab violation", base, text[:60])
            continue

        spk = row.get("speaker")
        if not spk:
            drop("no speaker id", base)
            continue
        spk_seen.setdefault(spk, None)

        v, a, t = anchor_mod.label(row, anchor)
        # Delivery is `unknown` for every Emilia row, which is the all-zero block and
        # therefore exactly contract-v1 conditioning. These clips were mined on acoustics
        # and nobody has heard one, so a lane here would be a guess about mode of address.
        vec = vat_vector(v, a, t, "")
        lab = ",".join([f"{x:.4f}" for x in vec[:3]] + [f"{int(x)}" for x in vec[3:]])
        emilia_rows_out.append((spk, f"{wav}|{{spk}}|{ipa}|{lab}", seconds))

        e = edge_loss(text, rec["asr"])
        edges.append((e["head_frac"], e["tail_frac"]))
        if (i + 1) % 2000 == 0:
            print(f"  processed {i + 1}/{len(rows)}")

    for base in missing:
        drop("no probe measure / EIV head", base)

    total = len(rows) + len(missing)
    n_drop = sum(dropped.values())
    print(f"\nkept {len(emilia_rows_out)} of {total} Emilia keeps "
          f"({100 * len(emilia_rows_out) / total:.1f}%); dropped {n_drop}:")
    for reason, n in dropped.most_common():
        print(f"  {n:6d}  {reason}  ({100 * n / total:.2f}%)")
        for ex in examples[reason]:
            print(f"          {ex}")

    s_ = g2p.stats
    tot_g2p = sum(s_.values()) or 1
    print(f"G2P: dict {100 * s_['dict_hits'] / tot_g2p:.2f}% | "
          f"contractions {s_['contraction_hits']} | neural {s_['neural_hits']} | "
          f"homograph flips {s_['homograph_flips']} | unresolved {s_['oov_misses']}")

    if edges:
        import numpy as np

        h = np.array([e[0] for e in edges])
        t = np.array([e[1] for e in edges])
        # Diagnostic, not a gate. Recorded because caption-vs-ASR edge disagreement is the
        # measure that would tighten this filter next, and it costs nothing to write down
        # now against re-running ASR later. No threshold is claimed for it here: on the
        # synthesis side the head half turned out ANTI-correlated with the ear.
        print(f"caption/ASR edge disagreement (advisory): head p90 {np.percentile(h, 90):.3f} "
              f"/ >0.05 {(h > 0.05).sum()} | tail p90 {np.percentile(t, 90):.3f} "
              f"/ >0.05 {(t > 0.05).sum()}")

    if n_drop / total > args.max_drop_frac:
        raise SystemExit(
            f"ABORT: dropped {100 * n_drop / total:.1f}% of the keeps, over the "
            f"{100 * args.max_drop_frac:.0f}% ceiling. A filter that eats the corpus should "
            f"stop rather than quietly succeed; raise --max-drop-frac only with a reason."
        )
    if not emilia_rows_out:
        raise SystemExit("ABORT: no Emilia row survived. Nothing to merge.")

    # Speaker numbering: v4's map is preserved verbatim and the new ids are appended after
    # it, in sorted order so the map is reproducible from the keep set alone.
    next_index = base_spk["n_spks"]
    if sorted(base_index.values()) != list(range(next_index)):
        raise SystemExit("ABORT: the base speaker map is not contiguous 0..n-1; appending "
                         "to it would collide or leave holes.")
    emilia_index = {spk: next_index + i for i, spk in enumerate(sorted(spk_seen))}
    collision = set(base_index) & set(emilia_index)
    if collision:
        raise SystemExit(f"ABORT: {len(collision)} speaker id(s) appear in both namespaces, "
                         f"e.g. {sorted(collision)[:3]} — one voice would get two rows.")
    n_spks = next_index + len(emilia_index)

    new_rows = [r.format(spk=emilia_index[spk]) for spk, r, _ in emilia_rows_out]
    seconds_added = sum(s for _, _, s in emilia_rows_out)
    val_new = sorted(r for r in new_rows if _in_val(r))
    train_new = sorted(r for r in new_rows if not _in_val(r))
    print(f"\nspeakers: {next_index} LibriTTS + {len(emilia_index)} Emilia = {n_spks}")
    print(f"hash split on the new rows: {len(val_new)} val / {len(train_new)} train "
          f"({len(val_new) / len(new_rows):.2%}, target {VAL_FRACTION:.0%})")
    if not val_new:
        raise SystemExit("ABORT: the hash split put 0 new rows in val.")

    train = parts["train_op.txt"] + train_new
    val = parts["val_op.txt"] + val_new
    print(f"merged: {len(train)} train / {len(val)} val "
          f"(+{len(new_rows)} rows, +{seconds_added / 3600:.1f} h)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(args.out, exist_ok=True)
    for name, part in (("val_op.txt", val), ("train_op.txt", train)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(part) + "\n")
        print(f"wrote {len(part)} rows -> {os.path.join(args.out, name)}")

    # The wall runs on the FILELISTS, and it classifies the filelist's own directory as
    # well as every audio directory inside it — so a new corpus dir must be declared even
    # though it holds no audio. Run here rather than only at load: the same check that
    # would refuse this at `fit()` should refuse it now, while re-deriving is minutes
    # instead of a queued GPU slot. `--out` is pre-flighted before the expensive work
    # above for the same reason.
    license_check([os.path.join(args.out, n) for n in ("train_op.txt", "val_op.txt")])
    print("licence wall: accepted")
    with open(os.path.join(args.out, "speakers.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_spks": n_spks, "libritts_id_to_index": base_index,
                   "emilia_id_to_index": emilia_index}, fh, indent=2)

    report = {
        "base": args.base,
        "derivation": (
            "v5 = v4 rows verbatim + Emilia-YODAS keeps. LibriTTS labels are per-speaker "
            "z@2sigma (unchanged from v4); Emilia labels are the SAME composites z-scored "
            "against v4's GLOBAL distribution (anchor_emilia_labels, owner option 1 "
            "2026-08-08), because 2,408 speakers at a median of 3 clips have no per-speaker "
            "statistic. Delivery is `unknown` (all-zero) for every Emilia row."),
        "base_rows": {"train": len(parts["train_op.txt"]), "val": len(parts["val_op.txt"])},
        "emilia": {
            "candidates": total,
            "kept": len(emilia_rows_out),
            "dropped": dict(dropped),
            "seconds_total": float(seconds_added),
            "asr_max_wer": ASR_MAX_WER,
            "train": len(train_new),
            "val": len(val_new),
        },
        "n_spks": n_spks,
        "n_spks_base": next_index,
        "license": ("LibriTTS-R CC-BY-4.0 + Emilia-YODAS CC-BY-4.0 (YODAS subset ONLY — "
                    "see configs/data_licenses.yaml)"),
    }
    with open(os.path.join(args.out, "derivation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("report ->", os.path.join(args.out, "derivation_report.json"))
    print("\nNEXT: data_statistics must be RE-MEASURED in-container. They cannot be "
          "inherited from v4 — v4 could only inherit v3c's because the audio and the split "
          "were identical, and this changes both.")


if __name__ == "__main__":
    main()

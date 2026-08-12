"""Stage 1 of the reverse-conveyance pipeline (direction-interface-brief §3):
consolidate the instruments' utterance-level decode of every TRUSTED audio+text
pair into one notation store — the input the quantizer + Gemma tagging pass
will read ("instruments listen; Gemma translates"). Text-adjoined sources only
(owner, 2026-07-19).

Sources merged:
  * libritts_r_vat_v2 filelists (train+val) — wav, speaker, V/A/T labels;
    transcript from the adjacent LibriTTS-R *.normalized.txt; acoustic
    measures (measures.jsonl: LUFS, alpha ratio, CPP, H1-H2, seconds); EIV
    head scores (corpus_v1 + corpus_families) and the JL-calibrated valence
    combo score.
  * sonora-expressive-registers v1 metadata.jsonl — certified keeps only
    (owner_audit.score >= 1; retakes excluded): text, register, gender,
    intended VAT, measured_z, qc, owner audit; EIV heads joined from the
    campaign eiv_scores.jsonl files where present (book-prose clips have no
    EIV pass yet — fields null).

Output: /data/model-training/sonora/markup_prep/utterance_notation.jsonl
(one row per clip) + a coverage report on stdout. Pure joins — no GPU, no
new measurement. Span-level (per-token) measures are NOT here yet: that
substrate is the cdminix/libritts-r-aligned layer (cleared CC-BY-4.0, not
on disk) or a forced-alignment pass — see the brief §6.

Run:  .venv/bin/python scripts/tools/derive_markup_measures.py
"""
import argparse
import json
from pathlib import Path

SON = Path("/data/model-training/sonora")
# D-L5: this was pinned to `libritts_r_vat_v2` while v3c is the corpus to train on, so the
# span-markup spike would have been prepared against superseded labels — v3b fixed the
# phonemes and v3c re-drew the split. Defaults to current and is overridable, rather than
# being a constant somebody has to notice.
DEFAULT_CORPUS = "libritts_r_vat_v3c"
EIV = SON / "eiv_scores"
REG = Path("/data/model-training/datasets/sonora-expressive-registers")
OUT = SON / "markup_prep" / "utterance_notation.jsonl"


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help=f"corpus dir under data/ (default: {DEFAULT_CORPUS})")
    args = ap.parse_args()
    global CORPUS
    CORPUS = SON / "data" / args.corpus
    if not CORPUS.is_dir():
        raise SystemExit(f"no such corpus: {CORPUS}")

    # Invert the derive's own map so the LibriTTS speaker id can be recorded beside the
    # contiguous index. Written by derive_vat_corpus as {libritts_id: index}.
    global index_to_libritts
    index_to_libritts = {}
    spk_file = CORPUS / "speakers.json"
    if spk_file.is_file():
        m = json.load(open(spk_file)).get("libritts_id_to_index", {})
        index_to_libritts = {int(i): lid for lid, i in m.items()}
    else:
        print(f"!! no speakers.json in {CORPUS}; 'speaker' will be null")

    print(f"corpus: {CORPUS.name}  ({len(index_to_libritts)} speakers mapped)")
    measures = {r["wav"]: r for r in jsonl(CORPUS / "measures.jsonl")}
    eiv = {r["wav"]: {k: v for k, v in r.items() if k != "wav"} for r in jsonl(EIV / "corpus_v1.jsonl")}
    for r in jsonl(EIV / "corpus_families.jsonl"):
        eiv.setdefault(r["wav"], {}).update({k: v for k, v in r.items() if k != "wav"})
    combo = json.load(open(EIV / "corpus_valence_combo.json"))

    # EIV for the synthetic campaigns, keyed by filename stem (paths differ post-rename).
    camp_eiv = {}
    for pat in ("synthetic_v1/*/eiv_scores.jsonl", "book-prose/*/eiv_scores.jsonl"):
        for f in Path("/data/model-training/datasets").glob(pat):
            for r in jsonl(f):
                camp_eiv[Path(r["wav"]).stem] = {k: v for k, v in r.items() if k != "wav"}

    rows, miss_txt, miss_meas = [], 0, 0
    for fl in ("train_op.txt", "val_op.txt"):
        for line in open(CORPUS / fl):
            wav, spk, _ipa, lab = line.rstrip("\n").split("|")
            v, a, t = (float(x) for x in lab.split(","))
            txt = Path(wav).with_suffix("").with_suffix(".normalized.txt")
            if not txt.is_file():
                miss_txt += 1
                continue
            m = measures.get(wav)
            if not m:
                miss_meas += 1
            rows.append({
                "source": CORPUS.name, "wav": wav,
                "id": Path(wav).stem,
                # D-L5: `spk` is the CONTIGUOUS INDEX the derive assigns (0..n-1), not the
                # LibriTTS speaker id — and the mapping is re-derived per corpus version,
                # so v2's index 100 need not be v3c's. Storing it under "speaker" invited a
                # join against reader_profiles.json or a LibriTTS path that would match the
                # wrong voice without erroring. Both are recorded, named for what they are.
                "speaker_index": int(spk),
                "speaker": index_to_libritts.get(int(spk)),
                "text": txt.read_text().strip(),
                "labels": {"V": v, "A": a, "T": t},
                "measures": {k: m[k] for k in ("seconds", "lufs", "alpha_db", "cpp", "h1h2")} if m else None,
                "eiv": eiv.get(wav), "eiv_valence_combo": combo.get(wav),
            })
    n_corpus = len(rows)

    n_reg = 0
    for m in jsonl(REG / "v1" / "metadata.jsonl"):
        score = (m.get("owner_audit") or {}).get("score")
        if not score:  # unaudited or retake-pending — not "approved and trusted"
            continue
        rows.append({
            "source": "expressive-registers-v1", "wav": str(REG / "v1" / m["file"]),
            "id": m["id"], "speaker": None, "engine": m.get("engine"),
            "text": m.get("text"),
            "register": m.get("register"), "gender": m.get("gender"),
            "labels": None, "intended_vat": m.get("intended_vat"),
            "measured_z": m.get("measured_z"), "qc": m.get("qc"),
            "owner_audit": m.get("owner_audit"),
            "measures": None,
            "eiv": camp_eiv.get(m["id"]), "eiv_valence_combo": None,
        })
        n_reg += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_eiv_reg = sum(1 for r in rows if r["source"] == "expressive-registers-v1" and r["eiv"])
    print(f"corpus rows: {n_corpus} (missing transcript: {miss_txt}, missing measures: {miss_meas})")
    print(f"expressive-registers certified rows: {n_reg} (with EIV: {n_eiv_reg})")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()

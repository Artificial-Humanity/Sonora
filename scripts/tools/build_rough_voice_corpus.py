"""A v7 subset of only the ROUGH voices, for the probe that asks whether the decoder CAN
render them.

THE QUESTION (2026-09-25)
-------------------------
The hum is in the mel the acoustic model predicts, and sampling does not remove it (steps
tied, temperature not confirmed). The owner hears "buzz" on voices with low natural HNR and
"layering" on clean ones, and severity tracks natural HNR (rho -0.537). Two explanations
predict different things:

  * the decoder is capable, and rough voices are simply under-trained — then fine-tuning on
    rough voices alone lowers the buzz on rough voices it has NOT seen in the fine-tune;
  * the decoder cannot represent them — then it does not, and the DiT decoder spike
    (model-decisions.md § Decoder v2) moves onto the critical path.

This builds the fine-tune corpus. Rows are v7's, VERBATIM, so speaker ids, phonemes and VAT
are what the warm-start model was trained on; `n_spks` and the mel statistics stay v7's.

⚠⚠ THE BENCH SPEAKERS ARE HELD OUT WHOLE, including from validation. Their rows appear in
neither list, so the bench measures generalisation to rough voices, not memorisation.

⚠ ONLY SPEAKERS WITH A MEASURED HNR can be classified, so the subset is LibriTTS-R only.
Emilia and the expressive-register clips have no entry in `speaker_hnr_all.json`.

Usage:
    python scripts/tools/build_rough_voice_corpus.py \\
        --corpus data/libritts_r_full_vat_v7 \\
        --hnr-json /data/model-training/sonora/pitch_error/speaker_hnr_all.json \\
        --hold-out /data/model-training/sonora/pitch_error/_rough_probe_bench_speakers.json \\
        --out data/libritts_r_full_vat_v7_rough
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hnr-json", required=True)
    ap.add_argument("--hold-out", required=True,
                    help="JSON {items: {..: {spk}}}: speakers kept out of both lists")
    ap.add_argument("--max-hnr", type=float, default=4.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hnr = {int(k): v["hnr"] for k, v in
           json.loads(Path(args.hnr_json).read_text())["speakers"].items()}
    held = {int(v["spk"]) for v in json.loads(Path(args.hold_out).read_text())["items"].values()}
    if not held:
        raise SystemExit("REFUSING: --hold-out names no speakers.")
    rough = {s for s, h in hnr.items() if h < args.max_hnr} - held

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("REFUSING: %s is not empty. A corpus directory is written once." % out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("train_op.txt", "val_op.txt"):
        kept = []
        for line in (Path(args.corpus) / name).read_text(encoding="utf-8").splitlines():
            if line.strip() and int(line.split("|")[1]) in rough:
                kept.append(line)
        leaked = {int(l.split("|")[1]) for l in kept} & held
        if leaked:
            raise SystemExit("REFUSING: held-out speakers reached %s: %s" % (name, sorted(leaked)))
        (out / name).write_text("\n".join(kept) + "\n", encoding="utf-8")
        counts[name] = len(kept)
    report = {"base": args.corpus, "max_hnr": args.max_hnr, "speakers": len(rough),
              "held_out": sorted(held), "rows": counts,
              "note": "Rows are v7's verbatim. n_spks and data_statistics stay v7's."}
    (out / "derivation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("rough-voice corpus: %d speakers, train %d rows, val %d rows, %d held out -> %s"
          % (len(rough), counts["train_op.txt"], counts["val_op.txt"], len(held), out))


if __name__ == "__main__":
    main()

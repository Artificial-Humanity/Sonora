"""Stage-2 prep: build LongCat batch_inference.py's meta.lst from dataset-v1
anchors. Each certified clip transfers its performance onto a SIBLING line of
the same register (different text, cycled), inheriting the anchor's register
and intended labels. Emits meta.lst (uid|prompt_text|prompt_wav|gen_text) and
longcat_manifest.jsonl rows for the QC gate.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="sonora-expressive-registers/v1 dir")
    ap.add_argument("--spec", required=True, help="bulk_spec.json (register line pools)")
    ap.add_argument("--out", required=True, help="transfer campaign dir")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "longcat"), exist_ok=True)

    spec = json.load(open(args.spec, encoding="utf-8"))
    pools = {reg: [(l["text"] if isinstance(l, dict) else l) for l in r["lines"]]
             for reg, r in spec["registers"].items()}
    rows = [json.loads(l) for l in open(os.path.join(args.dataset, "metadata.jsonl"),
                                        encoding="utf-8")]
    lst, manifest = [], []
    for r in rows:
        pool = [t for t in pools.get(r["register"], []) if t != r["text"]]
        if not pool:
            continue
        # cycle sibling choice by line index so anchors spread across the pool
        li = int(r["id"].split("_")[-3]) if r["id"].split("_")[-3].isdigit() else 0
        target = pool[li % len(pool)]
        uid = f"tr_{r['id']}"
        wav_abs = os.path.join(args.dataset, r["file"])
        lst.append(f"{uid}|{r['text']}|{wav_abs}|{target}")
        manifest.append({
            "id": uid, "engine": "longcat", "wav": f"{uid}.wav",
            "register": r["register"], "intended": r["intended_vat"],
            "text": target, "seed": 1024,
            "direction": {"anchor": r["id"], "anchor_text": r["text"],
                          "anchor_engine": r["engine"],
                          "anchor_owner_score": (r.get("owner_audit") or {}).get("score")},
            "engine_license": "MIT (LongCat-AudioDiT-3.5B); anchor lineage Apache-2.0",
            "campaign": "transfer1",
        })
    with open(os.path.join(args.out, "meta.lst"), "w", encoding="utf-8") as f:
        f.write("\n".join(lst) + "\n")
    with open(os.path.join(args.out, "longcat", "longcat_manifest.jsonl"), "w",
              encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"{len(lst)} transfer pairs -> {args.out}/meta.lst")


if __name__ == "__main__":
    main()

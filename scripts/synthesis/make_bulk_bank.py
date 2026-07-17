"""Expand bulk_spec.json (registers x line pools x job templates x seeds) into
a flat bank the synth_* renderers consume. Job ids are deterministic:
<register>_<lineNN>_<voice>_s<seed>.

Usage: python make_bulk_bank.py --spec bulk_spec.json --out bulk_bank.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf-8"))
    lines_out = []
    for reg, rdef in spec["registers"].items():
        for li, line in enumerate(rdef["lines"]):
            text = line["text"] if isinstance(line, dict) else line
            extras = line if isinstance(line, dict) else {}
            for job in rdef["jobs"]:
                for seed in job["seeds"]:
                    direction = {}
                    if job["engine"] == "qwen":
                        direction["design"] = job["design"]
                        instruct = job["instruct"]
                        for k, v in extras.items():
                            if k != "text":
                                instruct = instruct.replace("{" + k + "}", v)
                        direction["instruct"] = instruct
                    elif job["engine"] == "moss85":
                        direction["instruct"] = job["instruct"]
                        if job.get("quality"):
                            direction["quality"] = job["quality"]
                    elif job["engine"] == "dia":
                        direction["render_text"] = f"[S1] {job.get('dia_tags', '')}{text}"
                        direction["temperature"] = job.get("temperature", 1.8)
                        direction["guidance"] = job.get("guidance", 3.0)
                    lines_out.append({
                        "id": f"{reg}_{li:02d}_{job['voice']}_s{seed}",
                        "engine": job["engine"], "register": reg,
                        "intended": rdef["intended"], "seed": seed,
                        "text": text, "direction": direction,
                    })
    bank = {"version": spec["version"], "campaign": spec["campaign"],
            "license_note": spec["license_note"], "lines": lines_out}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
    by_eng = {}
    for l in lines_out:
        by_eng[l["engine"]] = by_eng.get(l["engine"], 0) + 1
    print(f"{len(lines_out)} jobs -> {args.out}  {by_eng}")


if __name__ == "__main__":
    main()

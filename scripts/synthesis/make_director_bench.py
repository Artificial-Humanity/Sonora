"""v3 director bench: same 10 quotes through G2/G4/G26, everything else pinned.

Task #3 (owner design 2026-07-21): a three-way director comparison — gemma-4-
e2b-qat vs gemma-4-e4b-qat vs gemma-4-26b-a4b-qat — on identical inputs. Ten
quotes from the (fully kept) quote-pilot-v2 bank; per quote the TTS engine and
seed are PINNED from v2, so the 30 rendered clips differ only in who directed
them. Ids/filenames carry the arm: qp3_<nn>_<class>_G2 / _G4 / _G26.

Also the timing benchmark: every director call is wall-clocked; the first call
per model is flagged cold (includes ollama load). Emits the combined bank plus
director_bench.json (per-call latencies, retries, per-arm summary).

Usage:
    uv run --no-project python scripts/synthesis/make_director_bench.py \
        --v2-bank /data/model-training/datasets/book-prose/necromancers/quote_pilot_v2_bank.json \
        --out-bank /data/model-training/datasets/book-prose/necromancers/quote_pilot_v3_bank.json \
        --out-bench /data/model-training/datasets/book-prose/necromancers/director_bench.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_quote_pilot_bank import call_director

ARMS = [("G2", "gemma-4-e2b-qat"), ("G4", "gemma-4-e4b-qat"),
        ("G26", "gemma-4-26b-a4b-qat")]


def pick_quotes(v2_lines, n=10, neutrals=2):
    expr = [l for l in v2_lines if "_neutral" not in l["id"] and "_ab_" not in l["id"]]
    neut = [l for l in v2_lines if "_neutral" in l["id"]]
    return expr[: n - neutrals] + neut[:neutrals]


def user_message(l):
    sr = l["source_ref"]
    attr = (f'{sr["verb"]} {sr["speaker"]}' if sr.get("verb") else "none in text")
    return (
        f"Scene context (the two sentences before the line):\n"
        f"{sr.get('context_pre') or '(chapter opening)'}\n\n"
        f'A character speaks (attribution: "{attr}"). Their line:\n“{l["text"]}”\n\n'
        f"What follows the line:\n{sr.get('context_post') or '(paragraph ends)'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-bank", required=True)
    ap.add_argument("--out-bank", required=True)
    ap.add_argument("--out-bench", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ollama", default="http://localhost:11434/api/chat")
    args = ap.parse_args()

    v2 = json.load(open(args.v2_bank, encoding="utf-8"))
    quotes = pick_quotes(v2["lines"], args.n)
    print(f"{len(quotes)} quotes; arms: {[a for a, _ in ARMS]}")

    lines, bench = [], []
    for arm, model in ARMS:  # grouped by model so cold-load lands on call #1
        print(f"\n== {arm} ({model}) ==")
        for i, l in enumerate(quotes):
            user = user_message(l)
            t0 = time.monotonic()
            d = call_director(user, model, args.ollama)
            dt = time.monotonic() - t0
            cls = l["id"].split("_", 2)[2]  # qp2_<nn>_<class>
            new_id = f"qp3_{i:02d}_{cls}_{arm}"
            bench.append({"arm": arm, "model": model, "id": new_id,
                          "seconds": round(dt, 2), "cold": i == 0,
                          "ok": d is not None})
            print(f"  [{i + 1}/{len(quotes)}] {dt:6.1f}s "
                  f"{'COLD ' if i == 0 else ''}{new_id}"
                  + ("" if d else "  DIRECTOR FAILED"))
            if d is None:
                continue
            direction = {"design": d["voice_design"], "instruct": d["instruct"]}
            if l["engine"] == "dia":
                direction.update({"render_text": f"[S1] {l['text']}",
                                  "temperature": 1.8, "guidance": 4.0})
            lines.append({
                "id": new_id, "engine": l["engine"], "register": d["register"],
                "intended": {"V": round(float(d["valence"]), 2),
                             "A": round(float(d["arousal"]), 2),
                             "T": round(float(d["tension"]), 2)},
                "seed": 1234, "text": l["text"], "direction": direction,
                "source_ref": {**l["source_ref"], "bench_arm": arm,
                               "director_model": model, "v2_id": l["id"]},
            })

    bank = {"version": 1, "campaign": "quote-pilot-v3",
            "license_note": v2["license_note"] + " Director-bench: per-line "
                            "director model in source_ref.director_model.",
            "lines": lines}
    json.dump(bank, open(args.out_bank, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    summary = {}
    for arm, model in ARMS:
        xs = [b for b in bench if b["arm"] == arm]
        warm = sorted(b["seconds"] for b in xs if not b["cold"])
        summary[arm] = {
            "model": model,
            "cold_first_call_s": next(b["seconds"] for b in xs if b["cold"]),
            "warm_median_s": warm[len(warm) // 2] if warm else None,
            "warm_min_s": warm[0] if warm else None,
            "warm_max_s": warm[-1] if warm else None,
            "failures": sum(1 for b in xs if not b["ok"]),
        }
    json.dump({"calls": bench, "summary": summary},
              open(args.out_bench, "w", encoding="utf-8"), indent=2)
    print(f"\n{len(lines)} lines -> {args.out_bank}")
    for arm, s in summary.items():
        print(f"  {arm}: cold {s['cold_first_call_s']}s, warm median "
              f"{s['warm_median_s']}s [{s['warm_min_s']}-{s['warm_max_s']}], "
              f"failures {s['failures']}")


if __name__ == "__main__":
    main()

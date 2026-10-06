"""Director obedience bench: does the casting pass obey the zonos skill file on this server?

Pre-registered in Notes/Sonora/lemonade-migration-design.md § The bench. 24 fixed narration
passages (scripts/assets/director_bench_passages.json), each cast for zonos exactly as
`book_ingest.casting_pass` casts it — same prompt, same schema, same sampling — one call per
passage, scored on the skill file's Narration rules (emotion omitted, rate 14-16, pitch
20-45). A candidate arm is judged against a reference arm; the verdict rules are in
`verdict` and must not change after a run.

Run:
  .venv/bin/python scripts/tools/director_obedience_bench.py --build-passages
  .venv/bin/python scripts/tools/director_obedience_bench.py --reference RESULTS.json [--out DIR]
The reference scores are the paired run recorded in the design note's Verdict section; the
retired server's reference arm is recorded in bench_20261005T230855.json (under --out).
"""

import argparse
import collections
import glob
import json
import random
import statistics
import time
from pathlib import Path

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib", "tools"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from book_ingest import (CASTING_SCHEMA, _extract_json, _json_schema,  # noqa: E402
                         _validate_casting, casting_messages)
from gemma_client import DIRECTOR, VOLUME, chat  # noqa: E402
from gemma_server import GemmaServer  # noqa: E402
import judge_passages  # noqa: E402

ENGINE = "zonos"
SEED = 20261005
N = 24
BANK_GLOB = "/data/model-training/datasets/book-prose/*/*_bank.json"
ASSET = Path(_SONORA_REPO) / "scripts" / "assets" / "director_bench_passages.json"
RATE = (14, 16)      # zonos.md, Narration: rate 14-16
PITCH = (20, 45)     # zonos.md, Narration: pitch_std 20-45
GATED = ("emotion_omitted", "rate", "pitch")


def select_passages(bank_paths, n=N, seed=SEED):
    """`n` zonos narration lines, round-robin over books, seeded. Refuses a short population."""
    by_book, seen = collections.defaultdict(list), set()
    for path in sorted(bank_paths):
        bank = json.loads(Path(path).read_text(encoding="utf-8"))
        lines = bank["lines"] if isinstance(bank, dict) else bank
        book = Path(path).parent.name
        for x in lines:
            if (x.get("chunk_type") == "narration" and x.get("engine") == ENGINE
                    and x.get("register") and x.get("intended") and x["id"] not in seen):
                seen.add(x["id"])
                by_book[book].append({"id": x["id"], "book": book, "text": x["text"],
                                      "register": x["register"],
                                      **{k: x["intended"][k] for k in ("V", "A", "T")}})
    eligible = sum(len(v) for v in by_book.values())
    if eligible < n:
        raise ValueError(f"only {eligible} eligible {ENGINE} narration lines, need {n}")
    rng = random.Random(seed)
    queues = {}
    for book in sorted(by_book):
        queue = sorted(by_book[book], key=lambda r: r["id"])
        rng.shuffle(queue)
        queues[book] = queue
    picked = []
    while len(picked) < n:
        for book in sorted(queues):
            if queues[book] and len(picked) < n:
                picked.append(queues[book].pop())
    return picked


def parsed(d):
    return (isinstance(d, dict) and all(k in d for k in CASTING_SCHEMA[ENGINE])
            and _validate_casting(ENGINE, d))


def score(castings):
    ok = [d for d in castings if parsed(d)]
    return {"n": len(castings), "parsed": len(ok),
            # The schema REQUIRES the key and allows null, so "omitted" means null.
            "emotion_omitted": sum(d["emotion"] is None for d in ok),
            "rate": sum(RATE[0] <= d["speaking_rate"] <= RATE[1] for d in ok),
            "pitch": sum(PITCH[0] <= d["pitch_std"] <= PITCH[1] for d in ok),
            "distinct": len({json.dumps(d, sort_keys=True) for d in ok})}


def verdict(reference, candidate, volume_parsed):
    """Pre-registered (Notes/Sonora/lemonade-migration-design.md). Do not edit after a run."""
    n = reference["n"]
    if n == 0 or candidate["n"] != n:
        raise ValueError(f"cannot judge: reference n={n}, candidate n={candidate['n']}")
    if reference["parsed"] < n or any(reference[c] < n - 1 for c in GATED):
        return "INVALID", [f"reference arm below validity: {reference}"]
    reasons = []
    if candidate["parsed"] < n:
        reasons.append(f"candidate parsed {candidate['parsed']}/{n}")
    for c in GATED:
        if candidate[c] < reference[c] - 1:
            reasons.append(f"{c}: candidate {candidate[c]} < reference {reference[c]} - 1")
    if volume_parsed < n:
        reasons.append(f"volume smoke parsed {volume_parsed}/{n}")
    return ("FAIL", reasons) if reasons else ("PASS", [])


def run_arm(call, passages):
    rows = []
    for i, p in enumerate(passages, 1):
        labels = {"V": p["V"], "A": p["A"], "T": p["T"], "register": p["register"]}
        system, user = casting_messages(p["text"], ENGINE, labels)
        t0 = time.monotonic()
        try:
            content, error = call(system, user, _json_schema(ENGINE)), None
        except Exception as e:   # one bad call is a scored miss, not a crash
            content, error = None, repr(e)
        casting = _extract_json(content)
        rows.append({"id": p["id"], "seconds": round(time.monotonic() - t0, 2),
                     "content": content, "casting": casting, "error": error})
        print(f"  [{i}/{len(passages)}] {p['id']} parsed={parsed(casting)} "
              f"{rows[-1]['seconds']}s{' ' + error if error else ''}", flush=True)
    return rows


def candidate_call(system, user, schema):
    return chat(system, user, model=DIRECTOR, max_tokens=900, temperature=0.2, schema=schema,
                timeout=300)   # the reference arm's budget


def volume_smoke(passages):
    ok = 0
    for p in passages:
        d, err = judge_passages.ask(VOLUME, p["text"])
        ok += d is not None
        if err:
            print(f"  volume {p['id']}: {err}", flush=True)
    return ok


def _arm(source, rows):
    return {"source": source, "rows": rows,
            "seconds_median": statistics.median(r["seconds"] for r in rows),
            "scores": score([r["casting"] for r in rows])}


def _save(path, out):
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-passages", action="store_true",
                    help="rewrite the committed passage asset from the book-prose banks")
    ap.add_argument("--out", default="/data/model-training/sonora/lemonade_bench")
    ap.add_argument("--reference",
                    help="results JSON whose reference arm to compare against (required "
                         "for a run)")
    args = ap.parse_args()

    if args.build_passages:
        passages = select_passages(glob.glob(BANK_GLOB))
        ASSET.write_text(json.dumps(passages, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        print(f"wrote {len(passages)} passages from "
              f"{len({p['book'] for p in passages})} books -> {ASSET}")
        return

    if not args.reference:
        ap.error("--reference RESULTS.json is required: the reference arm's server is "
                 "retired, so a run compares against its recorded scores")
    passages = json.loads(ASSET.read_text(encoding="utf-8"))
    out = {"design": "Notes/Sonora/lemonade-migration-design.md", "engine": ENGINE,
           "passages": len(passages), "arms": {}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / time.strftime("bench_%Y%m%dT%H%M%S.json")
    try:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))["arms"]["reference"]
        out["arms"]["reference"] = {"source": f"{args.reference}: {ref['source']}",
                                    "scores": ref["scores"]}

        print(f"== candidate arm: {DIRECTOR} ==", flush=True)
        with GemmaServer(DIRECTOR):   # one server at a time: they would not fit together
            chat(None, "Reply with an empty JSON object.", model=DIRECTOR, max_tokens=16,
                 temperature=0.0, as_json=True)   # warm: a cold load is not a casting call
            out["arms"]["candidate"] = _arm(DIRECTOR, run_arm(candidate_call, passages))
        _save(path, out)
        print(f"== volume smoke: {VOLUME} ==", flush=True)
        with GemmaServer(VOLUME):
            out["volume_parsed"] = volume_smoke(passages)
        out["verdict"], out["reasons"] = verdict(out["arms"]["reference"]["scores"],
                                                 out["arms"]["candidate"]["scores"],
                                                 out["volume_parsed"])
    except BaseException as e:   # whatever finished stays on disk, with the reason it stopped
        out["error"] = repr(e)
        raise
    finally:
        _save(path, out)

    cols = ("parsed", "emotion_omitted", "rate", "pitch", "distinct")
    print("\narm        " + "  ".join(f"{c:>15}" for c in cols) + "   s/call")
    for name, arm in out["arms"].items():
        s = arm["scores"]
        print(f"{name:<10} " + "  ".join(f"{s[c]:>12}/{s['n']}" for c in cols)
              + f"   {arm.get('seconds_median', '-')}")
    print(f"volume smoke parsed {out['volume_parsed']}/{len(passages)}")
    print(f"\nVERDICT: {out['verdict']}" + "".join(f"\n  - {r}" for r in out["reasons"]))
    print(f"results: {path}")


if __name__ == "__main__":
    main()

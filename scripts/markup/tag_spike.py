"""SCM v0.1 tagging spike (markup-schema-brief.md §5 step 3).

Selects ~100 trusted clips from the utterance notation store (50 owner-certified
expressive-registers + 50 LibriTTS-R spread across the V/A/T range), quantizes the
instrument decode into symbols, has Gemma (ollama, AR 26B-A4B by default) fill the
interpretive SCM fields blind (register is NOT given — recovery is scored), then
validates + verifies each object and registers the results as the `audit-markup-v0`
campaign in the Auditions app (clip + inline projection in the note).

Outputs under /data/model-training/sonora/markup_prep/spike_v0/:
  scm_rows.jsonl   one SCM sidecar per clip + verifier verdicts
  report.json      schema-valid rate, VAT verify rate, register recovery
Run:  uv run python scripts/markup/tag_spike.py [--model gemma-4-26b-a4b-qat]
      [--limit N] [--no-register]
"""
import argparse
import csv
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scm  # noqa: E402

SON = Path("/data/model-training/sonora")
NOTATION = SON / "markup_prep" / "utterance_notation.jsonl"
OUT_DIR = SON / "markup_prep" / "spike_v0"
RATINGS = Path("/data/model-training/datasets/sonora-expressive-registers/ratings.csv")
OLLAMA = "http://localhost:11434/api/chat"


def bin_sym(v):
    return "-2" if v < -0.6 else "-1" if v < -0.2 else "0" if v < 0.2 else "+1" if v < 0.6 else "+2"


def clamp2(z):
    return max(-1.0, min(1.0, z / 2.0))


def row_vat(r):
    if r.get("labels"):
        return {c: r["labels"][c] for c in ("V", "A", "T")}
    mz = r.get("measured_z") or {}
    return {c: clamp2(mz[c]) for c in ("V", "A", "T") if c in mz}


def select(rows, n_reg=50, n_corpus=50):
    reg = [r for r in rows if r["source"] == "expressive-registers-v1"]
    # spread across registers, best-audited first
    by_reg = {}
    for r in sorted(reg, key=lambda r: -(r.get("owner_audit") or {}).get("score", 0)):
        by_reg.setdefault(r.get("register"), []).append(r)
    picked, i = [], 0
    while len(picked) < min(n_reg, len(reg)):
        added = False
        for g in by_reg.values():
            if i < len(g) and len(picked) < n_reg:
                picked.append(g[i]); added = True
        if not added:
            break
        i += 1
    corpus = [r for r in rows if r["source"] == "libritts_r_vat_v2"]
    extremity = lambda r: sum(abs(v) for v in row_vat(r).values())
    corpus_sorted = sorted(corpus, key=extremity)
    rng = random.Random(1234)
    neutrals = rng.sample(corpus_sorted[:5000], 15)
    extremes = corpus_sorted[-35:]
    return picked + neutrals + extremes


def prompt_for(r, lexicon, include_register_list=True):
    vat = row_vat(r)
    sym = ", ".join(f"{c}:{bin_sym(v)}" for c, v in vat.items())
    eiv = r.get("eiv") or {}
    top = sorted(eiv.items(), key=lambda kv: -kv[1])[:3]
    emo = ", ".join(f"{k}:{v:.2f}" for k, v in top) if top else "n/a"
    secs = (r.get("measures") or {}).get("seconds") or (r.get("qc") or {}).get("duration") or "?"
    gender = r.get("gender") or "unknown"
    lex = " | ".join(sorted(lexicon))
    return (
        "You annotate speech clips with Sonora Conveyance Markup (SCM v0.1).\n"
        "Given the TEXT of an utterance and INSTRUMENT readings from its audio recording, "
        "output ONE JSON object exactly of the form:\n"
        '{"scm":"0.1","utterance":{"vat":{"V":0.0,"A":0.0,"T":0.0},'
        '"register":"<one from the lexicon>","style":["adj1","adj2"]},'
        '"direction":"<one sentence>"}\n'
        "- vat: continuous values in [-1,1] CONSISTENT with the instrument bins "
        "(V=valence neg/pos, A=arousal/energy low/high, T=phonation breathy(-)/pressed(+)); "
        "a bin of +1 means roughly 0.2..0.6, +2 means 0.6..1.0, etc.\n"
        f"- register: the single best fit from this lexicon: {lex}\n"
        "- style: up to 3 adjectives describing the delivery you infer.\n"
        "- direction: one sentence a voice director would give to reproduce this exact "
        "delivery of this text.\n"
        f'TEXT: "{r.get("text", "")}"\n'
        f"INSTRUMENTS: vat bins {sym}; duration {secs}s; strongest EIV emotion heads: {emo}; "
        f"speaker gender: {gender}\n"
        "Output the JSON object only."
    )


def ask(model, prompt):
    req = urllib.request.Request(OLLAMA, method="POST", data=json.dumps({
        "model": model, "stream": False, "think": False, "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "num_predict": 700, "num_ctx": 8192},
    }).encode())
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-26b-a4b-qat")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-register-audit", action="store_true",
                    help="skip appending the audit-markup-v0 campaign rows")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(NOTATION) if l.strip()]
    # controlled lexicon = live register vocabulary from ratings.csv (non-audit campaigns)
    lexicon = {r["register"] for r in csv.DictReader(open(RATINGS))
               if r["register"] and not r["campaign"].startswith("audit-")}
    picks = select(rows)
    if args.limit:
        picks = picks[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results, n_valid, n_verified, reg_hits, reg_total = [], 0, 0, 0, 0
    for i, r in enumerate(picks):
        p = prompt_for(r, lexicon)
        try:
            raw = ask(args.model, p)
            obj = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            results.append({"id": r["id"], "error": str(e)[:200]})
            print(f"[{i+1}/{len(picks)}] {r['id']}: ERROR {str(e)[:80]}")
            continue
        obj.setdefault("scm", "0.1")
        obj["id"], obj["text"], obj["wav"] = r["id"], r.get("text"), r["wav"]
        errs = scm.validate(obj, lexicon)
        ok_vat, flags = scm.verify_vat(obj, row_vat(r))
        obj["provenance"] = {
            "source": f"instruments+{args.model}", "schema_errors": errs,
            "verified": (not errs) and ok_vat,
            "verifier": {"pass": ok_vat, "checked": ["vat"], "flags": flags},
        }
        if not errs:
            n_valid += 1
        if (not errs) and ok_vat:
            n_verified += 1
        if r["source"] == "expressive-registers-v1" and r.get("register"):
            reg_total += 1
            claimed = (obj.get("utterance") or {}).get("register")
            hit = claimed == r["register"]
            reg_hits += hit
            obj["provenance"]["register_truth"] = r["register"]
            obj["provenance"]["register_hit"] = hit
        results.append(obj)
        print(f"[{i+1}/{len(picks)}] {r['id']}: "
              f"{'ok' if obj['provenance']['verified'] else 'FLAG'}"
              f"{' reg=' + str(obj['provenance'].get('register_hit')) if reg_total and r['source'] != 'libritts_r_vat_v2' else ''}")

    with open(OUT_DIR / "scm_rows.jsonl", "w") as f:
        for o in results:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    report = {
        "model": args.model, "clips": len(picks),
        "schema_valid": n_valid, "vat_verified": n_verified,
        "register_recovery": {"hits": reg_hits, "total": reg_total},
        "errors": sum(1 for x in results if "error" in x),
    }
    json.dump(report, open(OUT_DIR / "report.json", "w"), indent=1)
    print("REPORT:", json.dumps(report))

    if args.no_register_audit:
        return
    # ---- register audit-markup-v0 campaign (clip + inline projection in note) ----
    ratings_rows = list(csv.DictReader(open(RATINGS)))
    fields = list(csv.DictReader(open(RATINGS)).fieldnames)
    existing = {x["id"] for x in ratings_rows}
    rdir = RATINGS.parent
    new = []
    for o in results:
        if "error" in o or o["provenance"]["schema_errors"]:
            continue
        rid = f"mk_{o['id']}"
        if rid in existing:
            continue
        import os
        link = os.path.relpath(o["wav"], rdir)
        note = scm.render_inline(o)[:480]
        new.append({k: "" for k in fields} | {
            "campaign": "audit-markup-v0", "id": rid, "engine": "scm-spike",
            "register": (o.get("utterance") or {}).get("register") or "",
            "status": "unaudited", "note": note, "link": link})
    shutil.copy2(RATINGS, str(RATINGS) + ".pre-markup-audit.bak")
    with open(RATINGS, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerows(new)
    print(f"registered {len(new)} audit-markup-v0 rows in the Auditions app")


if __name__ == "__main__":
    main()

"""Build the revisit-v1 campaign bank — Chatterbox, Zonos and Orpheus.

These three engines' prior verdicts were voided on 2026-07-28 (wrong checkpoint,
malformed prompt, expressiveness knob at its floor, no reference clip ever passed).
Their pipelines were rebuilt and smoke-proven the same day. This assembles the bank
that lets the owner's ear overturn or confirm those verdicts.

WHY THE TEACHER-AB-V1 TEXTS, VERBATIM. A revisit that invents its own texts produces
an island: "Chatterbox scored 14/20" answers nothing, because the question is not
whether it is good, it is whether it belongs in a portfolio whose gold standard is
measured. Reusing the exact 20 texts makes every clip directly comparable to the
2026-07-26 A/B — qwen 19/20, vibevoice 19/20, moss_vg 18/20, dia 12/20, median
DNSMOS 3.46 / 3.40 / 3.27 / 2.53 — line for line, register for register, through
`pair_key`. Same texts, same seeds, same intended V/A/T.

THE ACCENT PROBES ARE DEMOTED, NOT DROPPED. Four of the 20 were accent probes.
Accent is structurally unreachable on all three of these engines: Chatterbox leaks
reference accent uncontrollably and its only official mention treats accent as a
defect, Orpheus has no accent channel at all, and Zonos's is documented UNVERIFIED
with an explicit "do not build a campaign on it". Scoring an engine on a dimension
it cannot receive is the precise failure the relay audit exists to end. The texts
are good expressive material, so they stay — but they carry `probe: "register"`,
their ids no longer say "accent", and `source_id` preserves the lineage.

ONE DELIBERATE CONTRADICTION, FLAGGED. chatterbox.md says do not send V-negative,
low-arousal grief: the 2026-07-17 audition found it rendered as casual chat. That
line is sent anyway, marked `probe: "v_neg_recheck"`. The failure was recorded
through the same broken configuration as everything else about that audition — no
reference clip, a dial nobody had characterised — so it is exactly what a revisit
is for. If it fails again with casting exercised and the dial understood, the
finding is real and the skill file's warning is earned rather than inherited.

Resumable: each completed line is appended to <out>.partial.jsonl, so a director
pass that dies at line 47 of 60 does not restart from zero.

Usage:
    uv run make_revisit_bank.py --out /data/model-training/datasets/revisit-v1/bank.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book_ingest as bi
from ref_select import route_engines  # noqa: E402

SOURCE_BANK = "/data/model-training/datasets/teacher-ab-v1/bank.json"
ENGINES = [("chatterbox", "CBX"), ("zonos", "ZON"), ("orpheus", "ORP")]

# The four accent probes, renamed so no id claims a dimension these engines cannot
# reach. Keyed by the source pair_key.
SLUG_OVERRIDE = {
    "accent_rp": "reassurance",
    "accent_south": "gossip",
    "accent_scots": "factual",
    "accent_none": "casual",
}

# Deliberate re-tests of a specific prior failure, keyed by (engine, pair_key).
RECHECK = {
    ("chatterbox", "grief"): "v_neg_recheck",
}


def source_lines():
    """One record per unique text from teacher-ab-v1, in bank order."""
    with open(SOURCE_BANK, encoding="utf-8") as f:
        bank = json.load(f)
    seen, out = set(), []
    for l in bank["lines"]:
        if l["pair_key"] in seen:
            continue
        seen.add(l["pair_key"])
        parts = l["id"].split("_")
        out.append({
            "idx": int(parts[1]),
            "slug": SLUG_OVERRIDE.get(l["pair_key"], "_".join(parts[2:-1])),
            "source_id": "_".join(parts[:-1]),
            "text": l["text"],
            "register": l["register"],
            "expected_register": l["expected_register"],
            "intended": l["intended"],
            "seed": l["seed"],
            "pair_key": l["pair_key"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", nargs="*", default=[e for e, _ in ENGINES])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    partial_path = args.out + ".partial.jsonl"

    done = {}
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
        print(f"resuming: {len(done)} line(s) already built", flush=True)

    src = source_lines()
    print(f"{len(src)} source texts x {len(args.engines)} engine(s)", flush=True)

    lines, failures = [], []
    with open(partial_path, "a", encoding="utf-8") as pf:
        for engine, suffix in ENGINES:
            # An unselected engine still contributes its CACHED lines. Skipping the
            # engine outright would write a bank containing only the selected one —
            # so re-running a single engine would silently delete the other two.
            selected = engine in args.engines
            for s in src:
                lid = f"rev_{s['idx']:02d}_{s['slug']}_{suffix}"
                if lid in done:
                    lines.append(done[lid])
                    continue
                if not selected:
                    continue
                # The line's labels go in as fixed context. Without them the
                # director picked Chatterbox's AROUSAL dial without being told the
                # arousal, and flattened it to (0.25, 0.3) across 18 of 20 registers.
                cast = bi.casting_pass(s["text"], engine, labels={
                    **s["intended"], "register": s["register"]})
                if not cast:
                    print(f"  {lid}: DIRECTOR FAILED", flush=True)
                    failures.append(lid)
                    continue
                # Engine routing: a casting call may be one an engine must not take.
                # Rule lives in ref_select.route_engines so it is stated once and shows
                # up in this log rather than in per-builder engine lists. Empty today —
                # Chatterbox's bright-female ban was withdrawn 2026-07-29 when the guard
                # moved to pitch excursion — but the mechanism stays wired so the next
                # such finding is one dict entry, not a builder edit.
                _kept, _dropped = route_engines(cast.get("voice_design", ""), [engine])
                if not _kept:
                    for _e, _why in _dropped:
                        print(f"  {lid}: ROUTED AWAY from {_e} — {_why}", flush=True)
                    continue
                tag = dict(cast)
                tag["engine"] = engine
                eng, direction = bi.build_direction(tag, s["text"])
                row = {
                    "id": lid,
                    "engine": eng,
                    "register": s["register"],
                    "expected_register": s["expected_register"],
                    "intended": s["intended"],
                    "seed": s["seed"],
                    "text": s["text"],
                    "pair_key": s["pair_key"],
                    "probe": RECHECK.get((engine, s["pair_key"]), "register"),
                    "source_id": s["source_id"],
                    "direction": direction,
                }
                lines.append(row)
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                pf.flush()   # a director pass that dies at 47/60 must not restart at 0
                print(f"  {lid}: {json.dumps(direction, ensure_ascii=False)[:110]}", flush=True)

    bank = {
        "version": "1.0",
        "campaign": "revisit-v1",
        "source_campaign": "teacher-ab-v1",
        "note": ("Same 20 texts/seeds/intended-VAT as teacher-ab-v1 so verdicts are "
                 "comparable line-for-line via pair_key. Accent probes demoted to "
                 "register probes — accent is unreachable on all three engines."),
        "lines": lines,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)

    by_engine = {}
    for l in lines:
        by_engine[l["engine"]] = by_engine.get(l["engine"], 0) + 1
    print(f"\nwrote {args.out}: {len(lines)} line(s) {by_engine}")
    if failures:
        print(f"DIRECTOR FAILURES ({len(failures)}): {', '.join(failures)}")
        print("re-run to retry only these (completed lines are cached)")


if __name__ == "__main__":
    main()

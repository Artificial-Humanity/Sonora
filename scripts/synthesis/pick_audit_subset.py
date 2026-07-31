# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""pick_audit_subset — risk-stratified audit sampling (owner protocol, 2026-07-30).

The corpus push renders in ~300-clip batches, and the owner cannot realistically
audition every file. This tool selects a HIGH-RISK SUBSET for the ear and parks the
rest as status `deferred` (visible in the app under the `deferred` filter, never in
todo). The batch structure makes this sound rather than merely convenient:
delivery-mix banks freeze ONE voice per (book, lane) group, so a group's systematic
risks — wrong persona, wrong lane feel, gender flip, register drift — are perfectly
correlated within the group. Hearing a few clips certifies the group's voice; what
remains on the unheard clips is per-clip stochastic risk, which stress-v1/v2 bounded
at <=~14% worst-case (0/20) and ~5% observed (orpheus).

Selection, per campaign:
  1. Every objectively-flagged clip (--flags file, one id per line) — always ear.
  2. Group certification: N clips per (book, delivery) group, positionally spread
     (first/last/middle...), N by the engine's tier.
  3. A deterministic tail sample of what remains (seed 42), per engine.

THREE TIERS (owner 2026-07-31), set by measured failure rate:

  tier          engines              per group   tail   unheard clips fold?
  trusted       qwen, chatterbox         1        3%    yes
  normal        orpheus (+1), others     2       10%    yes
  scrutinized   moss_vg, zonos           4       25%    NO

The third tier differs in KIND. Group certification assumes defects are SYSTEMATIC:
one voice per (book, lane) means a casting or lane error is perfectly correlated
within the group, so a few clips speak for the rest. Zonos and MOSS violate that
assumption — their defects are stochastic per clip (pause-and-rush, radio timbre,
early EOS) at roughly one in three, and no amount of sampling certifies a neighbour
that carries an independent 1-in-3 risk. So for those engines the unheard clips are
simply not folded. On delivery-v1-narration they were 13% of ride-along volume and
91% of expected contamination; refusing to fold them costs 17 auditions.

What does NOT relax at any tier is criterion 1: every QC-flagged clip goes to the
ear. The flags are what caught the real failures in round 1, so the instrument path
carries the weight the ear gives up. Trust is revocable: a drop in a trusted engine's
spot check should pull it back to the normal tier before the next campaign, not just
promote the one failed group.

Verdict handling after audition (`certify`):
  - group with ZERO drops among its audited clips -> CERTIFIED: its deferred clips
    are recorded in <campaign dir>/group_certification.json. They KEEP status
    `deferred` in ratings.csv — the score column stays ear-only, and the certified
    lexicon never counts unheard clips. The dataset fold ingests them from the
    certification artifact with provenance owner_audit="group-sampled".
  - group with any drop -> listed for promotion; `promote` flips its deferred rows
    back to unaudited so they land in todo.

ratings.csv is written with the audition app's own idiom (read-modify-write to a
temp file + os.replace) plus an mtime guard: if the file moves between read and
replace, the write is retried from a fresh read. The app is a LIVE WRITER — do not
edit ratings.csv with anything less careful.

Usage:
  pick_audit_subset.py select  --campaign delivery-v1-narration [--flags qc_flags.txt]
  pick_audit_subset.py promote --campaign delivery-v1-narration --group "mike|Neutral"
  pick_audit_subset.py certify --campaign delivery-v1-narration
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

RATINGS = "/data/model-training/datasets/sonora-expressive-registers/ratings.csv"
DATASETS = "/data/model-training/datasets"
# --- Three-tier audit policy (owner 2026-07-31) -------------------------------
# Coverage is set per engine by measured failure rate, and the third tier changes
# KIND, not just amount. Rates from delivery-v1-narration (heard clips):
#   chatterbox 0/38 · qwen 0/37 · orpheus 1/19 (5%) · zonos 20/64 (31%) · moss_vg 19/54 (35%)
#
# Why scrutinized engines get `ride_along: False` rather than simply a bigger sample:
# group certification rests on defects being SYSTEMATIC — one voice per (book, lane)
# means a casting or lane error is perfectly correlated within the group, so hearing a
# few clips speaks for the rest. Zonos and MOSS do not fail that way. Their defects are
# stochastic per clip (pause-and-rush, radio timbre, early EOS) at roughly one in three,
# and an independent 1-in-3 risk on each unheard clip is not certified by hearing its
# neighbours. Measured on this campaign: moss_vg + zonos are 13% of ride-along volume
# but 91% of expected contamination (5.7 of 6.2 bad clips). Refusing to fold their
# unheard clips removes nearly all of that risk for 17 auditions.
# Detectors that catch those specific defects objectively are what could earn these
# engines a ride-along back — coverage traded for instrumentation, same bargain qwen made.
TIERS = {
    "trusted":     {"per_group": 1, "tail": 0.03, "ride_along": True},
    "normal":      {"per_group": 2, "tail": 0.10, "ride_along": True},
    # Owner 2026-07-31: "I would hear all moss_vg and zonos renders until such a time
    # as we can bring their drift rates down." tail 1.00 means the sample IS the batch,
    # so ride_along never comes up — every clip is heard or it does not exist. This is
    # the honest form of the earlier 54%-coverage design: since unheard clips of these
    # engines were never going to fold, sampling them only wasted the render. Their
    # ENGINE_MIX share was cut to match (ref_select.ENGINE_MIX), so hearing everything
    # costs less listening overall, not more.
    "scrutinized": {"per_group": 4, "tail": 1.00, "ride_along": False},
}
ENGINE_TIER = {
    "qwen": "trusted", "chatterbox": "trusted",
    "moss_vg": "scrutinized", "zonos": "scrutinized",
}
DEFAULT_TIER = "normal"
# orpheus keeps a third certification clip on top of its tier: its 1-in-20 emphasis-peak
# voice split (stress-v1/v2) is a real defect the 2-clip default was never sized for.
CERT_BONUS = {"orpheus": 1}
SEED = 42


def tier_of(engine):
    return TIERS[ENGINE_TIER.get(engine, DEFAULT_TIER)]


def cert_count(engine):
    return tier_of(engine)["per_group"] + CERT_BONUS.get(engine, 0)


# Back-compat for readers of the old names.
TRUSTED_ENGINES = {e for e, t in ENGINE_TIER.items() if t == "trusted"}
SCRUTINIZED_ENGINES = {e for e, t in ENGINE_TIER.items() if t == "scrutinized"}


def _read():
    with open(RATINGS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, os.stat(RATINGS).st_mtime


def _write_guarded(mutate, attempts=5):
    """Read -> mutate(rows) -> temp+replace, retried if the file moved underneath."""
    for _ in range(attempts):
        rows, mtime = _read()
        changed = mutate(rows)
        if not changed:
            return 0
        if os.stat(RATINGS).st_mtime != mtime:
            time.sleep(0.5)
            continue
        fields = list(rows[0].keys())
        tmp = RATINGS + ".tmp-subset"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        if os.stat(RATINGS).st_mtime != mtime:
            os.unlink(tmp)
            time.sleep(0.5)
            continue
        os.replace(tmp, RATINGS)
        return changed
    print("ratings.csv kept moving — is someone auditioning right now? Aborting.")
    sys.exit(1)


def _bank_groups(campaign):
    """id -> (group_key, engine) from the campaign's bank."""
    bank = json.load(open(f"{DATASETS}/{campaign}/bank.json", encoding="utf-8"))
    out = {}
    for l in bank["lines"]:
        grp = f'{l.get("book", "?")}|{l.get("intended_delivery", "?")}'
        out[l["id"]] = (grp, l["engine"])
    return out


def cmd_select(args):
    flagged = set()
    if args.flags and os.path.exists(args.flags):
        flagged = {ln.strip() for ln in open(args.flags) if ln.strip()}
    groups_of = _bank_groups(args.campaign)
    rows, _ = _read()
    camp = [r for r in rows if r["campaign"] == args.campaign
            and r["status"] == "unaudited"]
    by_group = defaultdict(list)
    for r in camp:
        grp, eng = groups_of.get(r["id"], ("?", "?"))
        by_group[(grp, eng)].append(r["id"])

    keep_ids = set(i for i in flagged if any(i in ids for ids in by_group.values()))
    for (grp, eng), ids in sorted(by_group.items()):
        ids = sorted(ids)
        n = min(cert_count(eng), len(ids))
        # positional spread: first, last, then interior midpoints
        picks = [0, len(ids) - 1] + [len(ids) // 2, len(ids) // 4]
        chosen = []
        for p in picks:
            if len(chosen) >= n:
                break
            if ids[p] not in chosen:
                chosen.append(ids[p])
        keep_ids.update(chosen)
    # Tail sample per ENGINE, not over the pooled remainder: a trusted engine's thin
    # tail must not be diluted (or padded) by how many clips other engines contributed
    # to the same batch. Same seed, so selection stays deterministic per campaign.
    rng = random.Random(SEED)
    rest_by_engine = defaultdict(list)
    for (grp, eng), ids in sorted(by_group.items()):
        rest_by_engine[eng] += [i for i in sorted(ids) if i not in keep_ids]
    for eng, rest in sorted(rest_by_engine.items()):
        frac = tier_of(eng)["tail"]
        keep_ids.update(rng.sample(sorted(rest), round(len(rest) * frac)))

    defer_ids = set(i for ids in by_group.values() for i in ids) - keep_ids

    def mutate(rows):
        changed = 0
        for r in rows:
            if r["id"] in defer_ids and r["status"] == "unaudited":
                r["status"] = "deferred"
                changed += 1
        return changed

    n = _write_guarded(mutate)
    print(f"{args.campaign}: {len(keep_ids)} in the audit queue "
          f"({len(flagged & keep_ids)} QC-flagged), {n} deferred, "
          f"{len(by_group)} groups")
    for (grp, eng), ids in sorted(by_group.items()):
        in_q = sum(1 for i in ids if i in keep_ids)
        tname = ENGINE_TIER.get(eng, DEFAULT_TIER)
        trust = "" if tname == "normal" else f"  [{tname}]"
        print(f"  {grp:42s} {eng:11s} {in_q}/{len(ids)} queued{trust}")


def cmd_promote(args):
    groups_of = _bank_groups(args.campaign)
    target = {i for i, (g, _) in groups_of.items() if g == args.group}

    def mutate(rows):
        changed = 0
        for r in rows:
            if r["id"] in target and r["status"] == "deferred":
                r["status"] = "unaudited"
                changed += 1
        return changed

    n = _write_guarded(mutate)
    print(f"promoted {n} deferred clips of group {args.group!r} back to the queue")


def cmd_certify(args):
    groups_of = _bank_groups(args.campaign)
    rows, _ = _read()
    camp = [r for r in rows if r["campaign"] == args.campaign]
    by_group = defaultdict(lambda: {"keep": [], "drop": [], "todo": [], "deferred": [],
                                    "reroll": []})
    for r in camp:
        grp, eng = groups_of.get(r["id"], ("?", "?"))
        g = by_group[(grp, eng)]
        if r["status"] == "deferred":
            g["deferred"].append(r["id"])
        elif r["status"] == "unaudited":
            g["todo"].append(r["id"])
        elif r["score"] == "x" or r["status"] == "dropped":
            g["drop"].append(r["id"])
        elif r["status"] == "reroll":
            # A reroll is a clip awaiting a fresh take, not a keep. It used to fall
            # through to the `score` branch below and be counted as one, so a group
            # could certify while still holding flagged clips (4 such groups on
            # delivery-v1-narration, 2026-07-31). It does not fail the group — the
            # voice is fine, the take is not — but it must not pad the keep count
            # that certification is read off.
            g["reroll"].append(r["id"])
        elif r["score"]:
            g["keep"].append(r["id"])
    certified, failed, pending = {}, [], []
    for (grp, eng), g in sorted(by_group.items()):
        if g["todo"]:
            pending.append(grp)
            state = f"PENDING ({len(g['todo'])} unheard)"
        elif g["drop"]:
            failed.append(grp)
            state = f"FAILED ({len(g['drop'])} drop) -> promote its {len(g['deferred'])} deferred"
        elif not tier_of(eng)["ride_along"]:
            # Scrutinized engine: the group's VOICE is certified, but its unheard clips
            # are not, because this engine's defects are per-clip rather than per-group.
            # Nothing rides along; the deferred clips simply never enter the fold.
            certified[grp] = []
            state = (f"VOICE OK ({len(g['keep'])} heard) — {len(g['deferred'])} deferred "
                     f"NOT folded [scrutinized]")
        else:
            certified[grp] = g["deferred"]
            state = f"CERTIFIED ({len(g['keep'])} keeps heard, {len(g['deferred'])} ride along)"
        if g["reroll"]:
            state += f"  [{len(g['reroll'])} awaiting reroll]"
        print(f"  {grp:42s} {eng:11s} {state}")
    # A drop inside a TRUSTED engine's spot check is a protocol event, not just a
    # failed group: that engine bought thin coverage with its track record, and the
    # record just broke. Say so loudly — the fix is to revoke the tier, then re-select.
    revoke = sorted({eng for (grp, eng), g in by_group.items()
                     if eng in TRUSTED_ENGINES and g["drop"]})
    for eng in revoke:
        print(f"\n  !! TRUST BROKEN: {eng} dropped a clip under spot-check coverage. "
              f"Move it out of the trusted tier in ENGINE_TIER and re-run `select` before "
              f"the next campaign — promoting the one group is not enough.")

    if not pending:
        out = f"{DATASETS}/{args.campaign}/group_certification.json"
        json.dump({"campaign": args.campaign, "protocol": "risk-stratified v1",
                   "trusted_engines": sorted(TRUSTED_ENGINES),
                   "trust_revoked": revoke,
                   "certified_groups": certified,
                   "failed_groups": failed,
                   "provenance": "owner_audit=group-sampled"},
                  open(out, "w"), indent=1)
        n = sum(len(v) for v in certified.values())
        print(f"\nwrote {out}: {n} group-certified clips across {len(certified)} groups; "
              f"{len(failed)} groups need promotion")
    else:
        print(f"\n{len(pending)} groups still have unheard queue clips — certify later")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--campaign", required=True)
    s.add_argument("--flags", default=None)
    s.set_defaults(fn=cmd_select)
    p = sub.add_parser("promote")
    p.add_argument("--campaign", required=True)
    p.add_argument("--group", required=True)
    p.set_defaults(fn=cmd_promote)
    c = sub.add_parser("certify")
    c.add_argument("--campaign", required=True)
    c.set_defaults(fn=cmd_certify)
    args = ap.parse_args()
    args.fn(args)

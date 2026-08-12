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

  tier          engines                    per group   tail   unheard clips fold?
  trusted       qwen, chatterbox               1        3%    yes
  normal        orpheus, zonos, moss_vg, …     2       10%    yes
  scrutinized   (none named — the DEFAULT)     4       25%    NO

The third tier differs in KIND. Group certification assumes defects are SYSTEMATIC:
one voice per (book, lane) means a casting or lane error is perfectly correlated
within the group, so a few clips speak for the rest. An engine whose defects are
STOCHASTIC PER CLIP violates that assumption — no amount of sampling certifies a
neighbour that carries an independent risk of its own — so its unheard clips are
simply not folded.

**As of 2026-08-08 no engine is named scrutinized**; it is now purely the default for
engines with no track record, which is what the onboarding rule asks for. moss_vg was
the last one, and it left the tier the way the tier is meant to be left: its stochastic
defects (radio timbre, early EOS) became instruments — `qc_engine_defects.radio_timbre`
and the pause/tail/speech gates — rather than staying a reason to hear everything. That
is the trade the tier system exists to make possible; see the note on the entry itself.

Zonos sat here too until 2026-08-04. It was moved to `normal` once its instability
was traced to the emotion vector rather than the engine — a conditioning bug is
systematic, which is exactly the kind of defect group certification DOES cover.

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
import math
import os
import random
import sys

from collections import defaultdict

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import synth_common  # noqa: E402
import reader_profile  # noqa: E402

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
# few clips speaks for the rest. MOSS does not fail that way: its defects are stochastic
# per clip (radio timbre, early EOS), and an independent per-clip risk is not certified
# by hearing a neighbour.
#
# Zonos's 31% above is a SUPERSEDED figure — it averaged two populations, and with the
# emotion vector off it measured 93.7%, then 37/38 by ear on delivery-v1-narration-r2.
# That is a conditioning bug, which IS systematic, so it was promoted to `normal` on
# 2026-08-04. The 13%-of-volume / 91%-of-contamination arithmetic below was computed
# when both engines sat in this tier; moss_vg alone now carries far less of it.
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
    # zonos PROMOTED scrutinized -> normal, 2026-08-04, on the bank the promotion
    # had been waiting for. Its instability was never the engine: it tracked the
    # neutral-dominant emotion vector (93.5% keep with emotion truly off vs 27% with
    # the vector). What was missing was proof the DIRECTOR path could produce
    # true-unconditional zonos in production — every earlier narration bank had
    # `emotion: null` hand-patched in, so the evidence and the pipeline had never
    # been the same thing.
    #
    # delivery-v1-narration-r2 closed that gap: 44/44 zonos lines directed with
    # emotion off, no hand-patching, 38 heard -> 37 keeps (97%). The one non-keep is
    # a bookkeeping retirement, not an ear verdict — on clips the ear actually judged
    # it is 37/37, above chatterbox, which is already trusted.
    "zonos": "normal",
    # moss_vg PROMOTED scrutinized -> normal, 2026-08-08, on the same bargain zonos took
    # four days earlier: coverage traded for instrumentation.
    #
    # The hold rested on two things and neither survived re-measurement. The "24% rate on
    # expressive material" is not reproducible from anything on disk — the source
    # (`4abfd3f`) measured moss_vg 19/54 = 35% REJECTED on delivery-v1-narration, a
    # NARRATION campaign. And 11 of its 20 non-keeps are bookkeeping retirements ("retired
    # unheard: group failed certification twice", "superseded by _QWE, kept at 5"), not ear
    # rejections of the clip; on real ear verdicts it is 95 heard / 90.5% keep. The
    # dialogue re-test the hold was waiting for already existed: 19 heard at 89.5%, against
    # a 73.5% lane mean and above chatterbox, which is trusted.
    #
    # What actually justified 100% listening was the DEFECT, and it is now instrumented at
    # both ends. The nine genuine rejections were 5 structural (early EOS, 2.82 s and
    # 3.04 s dead air, 0.67 s and 3.68 s of speech) — every one of which fails pause_ok,
    # tail_ok or speech_ok today — and 4 timbre. `qc_engine_defects.radio_timbre` catches
    # 3 of those 3 radio clips (out/in-band 0.014 / 0.072 / 0.094 against a 0.10 bar) and
    # flags 19.8% of the batch for the ear, and synth_bank.sh now runs it on every bank
    # rather than leaving it to whoever remembers.
    #
    # ⚠ The residual is real and named: `windfairies_nar_0034`, "highly robotic… like an
    # old-fashioned phone IVR" — 28.8 s, 26.2 s of speech, 0.38 s worst pause. Flat-but-
    # fluent prosody passes every instrument we have. `normal` sampling (2/group + 10%) is
    # what covers it now; if that mode recurs, this is the line to revisit.
    "moss_vg": "normal",
    # Everything below was riding on DEFAULT_TIER when that default was
    # "normal". Naming them keeps their measured tier explicit now that the
    # default is conservative — otherwise tightening the default would have
    # silently promoted 15 established engines to 100%-heard and multiplied
    # the audit load, which is not what the onboarding rule asks for. The rule
    # is about engines with NO track record; these have one.
    "orpheus": "normal",        # 80% keep / mean 4.49 once `tara` is excluded
    "vibevoice": "normal", "dia": "normal", "moss85": "normal",
    "longcat": "normal",
    # Real human audio and licensed corpora: no engine drift to sample for.
    # The new-reader ear rule governs librivox, not this tier table.
    "librivox": "normal", "libritts-r": "normal", "emilia-yodas": "normal",
    "scm-spike-v1": "normal",
}
# Unknown engines fall to the CONSERVATIVE tier, not the folding one. A brand-new
# engine has no measured drift rate, so "normal" handed it ride-along folding
# privileges by default — the opposite of the onboarding rule, which says an engine
# earns trust with evidence. This also catches the id-join failure below, where the
# engine reads "?": a scrutinized "?" group folds nothing, so a ratings row whose id
# does not match the bank can no longer demote a real scrutinized clip into a tier
# that folds its siblings unheard.
DEFAULT_TIER = "scrutinized"
# orpheus keeps a third certification clip on top of its tier: its 1-in-20 emphasis-peak
# voice split (stress-v1/v2) is a real defect the 2-clip default was never sized for.
CERT_BONUS = {"orpheus": 1}
SEED = 42


def tier_of(engine):
    return TIERS[ENGINE_TIER.get(engine, DEFAULT_TIER)]


def _warn_unmatched(unmatched, campaign):
    """Say so, loudly, when ratings ids do not join to the bank/manifest.

    The join fails for real reasons — an MF/FM rename propagated to ratings but
    not the bank, a manifest line that failed json.loads and was skipped — and
    the ids land in a synthetic ("?", "?") group. They are scrutinized now so
    nothing folds, but a silent "?" group is still a sign the campaign's
    bookkeeping has drifted, and it used to print nothing at all.
    """
    if not unmatched:
        return
    print(f"  !! {len(unmatched)} ratings id(s) in {campaign} do not match any "
          f"bank/manifest record: {sorted(unmatched)[:5]}"
          + (" ..." if len(unmatched) > 5 else ""), file=sys.stderr)
    print("     They are grouped as ('?','?') and treated as SCRUTINIZED "
          "(nothing folds). Fix the join before trusting this campaign's "
          "certification.", file=sys.stderr)


def cert_count(engine):
    return tier_of(engine)["per_group"] + CERT_BONUS.get(engine, 0)


# Back-compat for readers of the old names.
TRUSTED_ENGINES = {e for e, t in ENGINE_TIER.items() if t == "trusted"}
SCRUTINIZED_ENGINES = {e for e, t in ENGINE_TIER.items() if t == "scrutinized"}


def _read():
    """Read-only view of ratings.csv, for the report paths.

    It used to also return the mtime, for a guard that lived in `_write_guarded`. Both
    remaining callers discarded it (`rows, _ = _read()`) and the guard is the shared
    transaction's now, so returning it was one more copy of the rule with no reader.
    """
    with open(RATINGS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_guarded(mutate):
    """Read -> mutate(rows) -> write, through the one shared transaction (C-M6).

    This carried its own flavour: read, stamp the mtime, mutate, re-stamp, temp+replace,
    and RETRY up to five times with a half-second sleep if the file moved. The retry was
    the interesting difference and it is gone on purpose — `ratings_transaction` takes an
    flock, which serialises our scripts against each other outright, so the only writer
    left to lose a race with is the audition app. Against a human editing in a browser,
    retrying is a way to eventually win a race we should not be running; aborting and
    saying so is the behaviour every other writer now has.

    `mutate` returns the number of rows it changed. Zero means leave the file alone —
    not rewrite it byte-identically, which would take a backup and move the mtime for a
    run that changed nothing.
    """
    changed = 0
    with synth_common.ratings_transaction(RATINGS, tag="subset") as (_hdr, rows):
        changed = mutate(rows)
        if not changed:
            raise synth_common.DryRun
    return changed


def _bank_groups(campaign):
    """id -> (group_key, engine), from the campaign's bank OR its librivox manifest.

    Synthesis campaigns have a bank.json whose lines carry book + intended_delivery.
    The REAL-AUDIO lane has no bank at all — its "bank" is a book someone read aloud —
    so fall back to the aligner's manifest, grouping by book|section and treating
    `librivox` as the engine. Without this, `select` cannot run on a force-aligned
    campaign, and the new-reader rule below could never fire on the very lane it exists
    to serve.

    register_audition writes real-audio campaigns as `book-<dir>`, so strip that prefix
    when resolving the directory.
    """
    import glob
    stripped = campaign[len("book-"):] if campaign.startswith("book-") else campaign
    for base in (campaign, stripped):
        # Every bank in the campaign, not just bank.json. A REROLL keeps the clip's
        # id, so reading bank.json alone still finds it — but a RE-CAST does not:
        # the id carries the engine, so moving a line to another engine mints a NEW
        # id that exists only in the reroll bank. Reading bank.json alone left those
        # clips grouped ('?','?') and treated as scrutinized, so a legitimate keep
        # was silently excluded from its group's evidence. bank.json is applied
        # first and the reroll rounds override it, so a line reflects its LAST cast.
        banks = ([f"{DATASETS}/{base}/bank.json"]
                 + sorted(glob.glob(f"{DATASETS}/{base}/bank-*.json")))
        banks = [b for b in banks if os.path.exists(b)]
        if not banks:
            continue
        out = {}
        for path in banks:
            for l in json.load(open(path, encoding="utf-8"))["lines"]:
                grp = f'{l.get("book", "?")}|{l.get("intended_delivery", "?")}'
                out[l["id"]] = (grp, l["engine"])
        return out

    out = {}
    for man in glob.glob(f"{DATASETS}/{stripped}/**/librivox_manifest.jsonl",
                         recursive=True):
        with open(man, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id"):
                    grp = f'{rec.get("book", "?")}|sec{rec.get("section", "?")}'
                    out[rec["id"]] = (grp, rec.get("engine", "librivox"))
    if not out:
        raise SystemExit(f"no bank.json and no librivox manifest for {campaign}")
    return out


PROFILES_PATH = "/data/model-training/datasets/reader_profiles.json"


def _clip_readers():
    """clip id -> (reader, title), from every librivox manifest on disk."""
    import glob
    out = {}
    for man in glob.glob("/data/model-training/datasets/**/librivox_manifest.jsonl",
                         recursive=True):
        with open(man, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id") and rec.get("reader"):
                    out[rec["id"]] = (rec["reader"], rec.get("book") or "?")
    return out


def _confirmed_reader_titles():
    """{(reader, title)} the ear has already confirmed.

    Keyed on the PAIR, not the reader (owner 2026-08-01). A LibriVox display name is not
    an identity — two people can share one — and a long-running reader legitimately
    reads as "adult" in early recordings and "middle-aged" in later ones. So a new TITLE
    by a known name still earns one ear pass; only then may that title's clips be filled.

    C-M8: the all-three test used to live here as its own expression, and `stage_pool`
    carried a looser one — so a pair could be confirmed enough to fold a whole title into
    the corpus and still be force-queued here. Both now call the same predicate.
    """
    prof = reader_profile.load_profiles(PROFILES_PATH)
    return {(reader, title)
            for reader, entry in prof.items()
            for title in (entry.get("titles") or {})
            if reader_profile.is_confirmed_pair(prof, reader, title)}


def cmd_select(args):
    flagged = set()
    if args.flags and os.path.exists(args.flags):
        flagged = {ln.strip() for ln in open(args.flags) if ln.strip()}
    groups_of = _bank_groups(args.campaign)
    rows = _read()
    # C-M1, second half. The requeue branch in `mutate` below draws from `keep_ids`, and
    # `keep_ids` is built entirely out of `by_group` — so restricting the pool to
    # `unaudited` made "deferral is revisable" unreachable code: an id in `keep_ids` was
    # by construction an unaudited row's id, and the `elif ... == "deferred"` branch could
    # never fire. Deferred rows are candidates again on every pass; `mutate` is what
    # distinguishes the two states, not this filter.
    #
    # Rows the ear has already ruled on (keep/drop/reroll/…) stay out: a verdict is not
    # revisable by sampling. This also makes selection STABLE across re-runs — the group's
    # id list no longer shrinks each pass, so the positional spread picks the same clips.
    camp = [r for r in rows if r["campaign"] == args.campaign
            and r["status"] in ("unaudited", "deferred")]
    by_group = defaultdict(list)
    unmatched = []
    for r in camp:
        if r["id"] not in groups_of:
            unmatched.append(r["id"])
        grp, eng = groups_of.get(r["id"], ("?", "?"))
        by_group[(grp, eng)].append(r["id"])
    _warn_unmatched(unmatched, args.campaign)

    keep_ids = {i for i in flagged if any(i in ids for ids in by_group.values())}

    # NEW-READER RULE (owner 2026-08-01): "if we have a set of clips from a new reader,
    # it would do us well to have me review at least one clip, always. The voice
    # identity attributes I apply can then be widened to the rest."
    #
    # Enforced HERE rather than trusted to sampling, because it is a precondition for
    # propagation, not a coverage preference: reader_profile.py learns gender/age/accent
    # only from AUDITED clips, so a reader whose clips are all deferred can never be
    # profiled, and every future book they narrate arrives untagged. One guaranteed clip
    # unlocks that title. Cross-title reuse is only ever a pre-fill hint, never an
    # assertion of identity — see reader_profile.py for why the pair is the unit.
    readers = _clip_readers()
    confirmed = _confirmed_reader_titles()
    campaign_ids = {i for ids in by_group.values() for i in ids}
    new_readers = defaultdict(list)
    for cid in sorted(campaign_ids):
        pair = readers.get(cid)
        if pair and pair not in confirmed:
            new_readers[pair].append(cid)
    for rd, ids in sorted(new_readers.items()):
        if not (set(ids) & keep_ids):        # nothing of theirs is queued yet
            keep_ids.add(sorted(ids)[len(ids) // 2])   # a mid clip, not an edge one
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
        # C-L6. `round(len(rest) * frac)` is ZERO for a trusted engine's 3% tail until
        # the remainder reaches 17 clips — and banker's rounding makes it zero at exactly
        # 16.67 too. So the tail sample, which is the whole mechanism for noticing that a
        # trusted engine has started drifting, silently did not exist on small batches.
        # Trusted is the tier where it matters most: those engines are audited 1 per
        # group precisely BECAUSE the tail is watching them.
        #
        # Ceiling, not round: a non-zero fraction of a non-empty remainder means at least
        # one clip. An engine either has a tail sample or it has none because `frac` is
        # zero, and no third case where the arithmetic quietly decided for us.
        n = math.ceil(len(rest) * frac) if frac else 0
        if rest and frac and n == 0:            # unreachable with ceil; kept as a tripwire
            n = 1
        keep_ids.update(rng.sample(sorted(rest), min(n, len(rest))))

    defer_ids = {i for ids in by_group.values() for i in ids} - keep_ids

    counts = {"deferred": 0, "requeued": 0}

    def mutate(rows):
        for r in rows:
            if r["id"] in defer_ids and r["status"] == "unaudited":
                r["status"] = "deferred"
                counts["deferred"] += 1
            # C-M1. The flip only ever went one way, so a clip that was deferred by an
            # earlier `select` and QC-flagged by a LATER pass stayed deferred forever —
            # invisible in todo, never heard. `keep_ids` said it belonged in the queue and
            # nothing acted on that, which quietly breaks this tool's one non-negotiable
            # rule: "What does NOT relax at any tier is criterion 1: every QC-flagged clip
            # goes to the ear." QC runs after every generation pass and `--flags` grows;
            # deferral has to be revisable or the flags cannot reach the ear.
            #
            # Written for `keep_ids` rather than for flagged ids alone, because the same
            # asymmetry bites a re-run that samples differently: whatever select decides
            # belongs in the queue, belongs in the queue.
            elif r["id"] in keep_ids and r["status"] == "deferred":
                r["status"] = "unaudited"
                counts["requeued"] += 1
        return counts["deferred"] + counts["requeued"]

    _write_guarded(mutate)
    n = counts["deferred"]
    print(f"{args.campaign}: {len(keep_ids)} in the audit queue "
          f"({len(flagged & keep_ids)} QC-flagged), {n} deferred, "
          f"{len(by_group)} groups")
    if counts["requeued"]:
        print(f"  re-queued {counts['requeued']} previously-deferred clip(s) that this "
              f"pass selected (QC flags or a changed sample)")
    if new_readers:
        print(f"  reader/title pairs needing an ear pass ({len(new_readers)}): "
              + ", ".join(f"{r} / {t}" for r, t in sorted(new_readers)))
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
    rows = _read()
    camp = [r for r in rows if r["campaign"] == args.campaign]
    by_group = defaultdict(lambda: {"keep": [], "drop": [], "todo": [], "deferred": [],
                                    "reroll": []})
    _warn_unmatched([r["id"] for r in camp if r["id"] not in groups_of], args.campaign)
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
        elif not g["keep"]:
            # Every heard clip was a reroll, so the group certifies on zero
            # positive evidence and folds its deferred siblings on the strength
            # of nothing. Certification needs at least one actual keep.
            failed.append(grp)
            state = (f"NO EVIDENCE (0 keeps heard, {len(g['reroll'])} reroll) "
                     f"-> promote its {len(g['deferred'])} deferred")
        else:
            certified[grp] = g["deferred"]
            state = f"CERTIFIED ({len(g['keep'])} keeps heard, {len(g['deferred'])} ride along)"
        if g["reroll"]:
            state += f"  [{len(g['reroll'])} awaiting reroll]"
        print(f"  {grp:42s} {eng:11s} {state}")
    # A drop inside a TRUSTED engine's spot check is a protocol event, not just a
    # failed group: that engine bought thin coverage with its track record, and the
    # record just broke. Say so loudly — the fix is to revoke the tier, then re-select.
    #
    # BUT READ THE DROP'S NOTE FIRST. Drops are not typed here, so a clip rejected
    # for its TEXT counts against the ENGINE that rendered it faithfully. This alarm
    # fired on delivery-v1-narration-r2 for a passage narrating figures that do not
    # exist in audio — a text-selection defect no engine could have rendered usably —
    # and was overridden (owner 2026-08-04); chatterbox stayed trusted. Typing drops
    # by cause was considered and declined as premature. Until it exists, treat this
    # as a prompt to go read the verdict, not as a verdict itself.
    # See notes/delivery-mix-campaign.md, "EXCEPTION".
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

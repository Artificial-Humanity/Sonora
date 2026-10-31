"""Unblind one staged-rebuild or hum-fix check (1-9) and apply its pre-registered reading.

Reads the key written by `render_ear_staged.py render` and the app's verdicts, and prints
each arm's gap (model minus round trip) with its floor, the check's contrasts, the number
of tests (uncorrected), and the outcome from `staged_bench.reading`. Exact; no randomness.

    .venv/bin/python scripts/tools/unblind_ear_staged.py --check 1 \\
        --key /data/model-training/sonora/.keystage/staged_check1.key.json \\
        --test /data/model-training/sonora/eartest/staged_check1

Checks 5-7 are one listening session over three test folders (A, B, C), and are read only when
every item of all three is rated: name the folders with --session-tests (this check's --test
among them). Checks 8 (D) and 9 (Q) are single-folder sessions and are read only when every
item of --test is rated. --session-closed reads a hum-fix check anyway, leaving out the items not fully rated.
"""

import argparse
import csv
import json
import math
import statistics as st
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import staged_bench as sb                            # noqa: E402

OUTCOME = {
    (1, "go"): ("R hums more than stock after 10 epochs: the recipe's extra hum is reachable "
                "at this budget. GO: train C0, S1 and S2, then checks 2 and 3."),
    (1, "stop"): ("R over stock is not shown. STOP: the extra hum needs more than 10 epochs to "
                  "appear on the derisk recipe. Nothing else trains; the owner decides whether "
                  "to repeat at a longer budget. Not shown is not absent."),
    (1, "invalid"): ("derisk over stock is not shown, or the round-trip floors are not "
                     "comparable: no reading. The owner decides whether to re-draw voices."),
    (2, "shown"): "",
    (2, "not_shown"): ("Neither C0 nor S1 is shown to hum more than stock. Not shown is not "
                       "absent."),
    (2, "invalid"): "The round-trip floors are not comparable: no reading.",
    (3, "shown"): "",
    (3, "not_shown"): ("Neither the 24 kHz path nor the conditioning is shown to add the hum. "
                       "Not shown is not absent."),
    (3, "invalid"): ("Neither R over S1 nor S1's own excess is shown: the hum is not visible "
                     "on these voices, so no reading."),
    (4, "shown"): "",
    (4, "not_shown"): ("S1 over L is not shown: the restoration is not shown to add the hum. "
                       "The speakers and G2P are next. Not shown is not absent."),
    (4, "invalid"): ("S1's own excess is not shown, or the round-trip floors are not comparable, "
                     "or fewer than 16 voices were rated in both arms: no reading. The owner "
                     "decides; no voices remain to re-draw from."),
    # the hum-fix batch (Notes: Sonora/hum-fix-batch-preregistration.md); arm G is its G′
    (5, "shown"): "",
    (5, "not_shown"): ("D1 over C0 is not shown: fewer clips per speaker is not shown to add "
                       "hum (density is not shown to matter). D1 manipulates distinct clips per "
                       "speaker at S1's step budget, so how often a speaker is heard is only "
                       "weakly manipulated. Not shown is not absent."),
    (5, "invalid"): ("The round-trip floors are not comparable (D1's within 1 point of C0's, "
                     "both under 4), or fewer than 16 voices were rated in both arms: no "
                     "reading. The owner decides."),
    (6, "shown"): "",
    (6, "not_shown"): ("S1 over G′ is not shown: the ᵻ remap is not shown to reduce the hum. "
                       "Not shown is not absent."),
    (6, "invalid"): ("S1's own excess is not shown, or the round-trip floors are not comparable "
                     "(G′'s within 1 point of S1's, both under 4), or fewer than 16 voices were "
                     "rated in both arms: no reading. The owner decides."),
    (7, "shown"): "",
    (7, "not_shown"): ("S1 over F1 is not shown: dense speakers are not shown to reduce the hum. "
                       "Not shown is not absent."),
    (7, "invalid"): ("S1's own excess is not shown, or the round-trip floors are not comparable "
                     "(F1's within 1 point of S1's, both under 4), or fewer than 16 pairs were "
                     "rated in both arms: no reading. The owner decides."),
    (8, "shown"): "",
    (8, "not_shown"): ("S1 over E is not shown: replacing op_g2p with an espeak front end "
                       "(eSpeak NG 1.52.0, 99.4% character-identical to VITS's) is not shown "
                       "to reduce the hum. Not shown is not absent."),
    (8, "invalid"): ("S1's own excess is not shown, or the round-trip floors are not comparable "
                     "(E's within 1 point of S1's, both under 4), or fewer than 16 voices were "
                     "rated in both arms: no reading. The owner decides."),
    (9, "shown"): "",
    (9, "not_shown"): ("neither S1 over Q nor F1 over Q is shown: the synthetic corpus is not "
                       "shown to hum less than S1 or than the dense real speakers. Not shown is "
                       "not absent."),
    (9, "invalid"): ("S1's own excess is not shown, or the round-trip floors are not comparable "
                     "(F1's and Q's within 1 point of S1's, all under 4), or fewer than 16 "
                     "triplets were rated in all three arms: no reading. The owner decides."),
}
SHOWN = {
    "fine_tuning": ("C0 hums more than stock: fine-tuning itself adds the hum. The cause is "
                    "the recipe, not the data."),
    "data": ("S1 hums more than stock and C0 is not shown to: the new data adds it (corpus, "
             "speakers or G2P, bundled here)."),
    "audio_path": "S2 hums more than S1: the 24 kHz path or our vocoder adds the hum.",
    "conditioning": "R hums more than S2: the VAT/FiLM conditioning adds it.",
    "restoration": ("S1 hums more than L: the restoration adds the hum (Miipher denoising, "
                    "dereverberation and WaveFit resynthesis, as a whole, not which part). "
                    "Known limit: the original audio is noisier and more reverberant, and "
                    "noise in L's renders could mask hum; the floor rule guards the round "
                    "trips only."),
    "clips_per_speaker": ("D1 hums more than C0: fewer clips per speaker adds hum. Production "
                          "data should favour fewer speakers heard more (F1's recipe is the "
                          "candidate)."),
    "symbol": ("S1 hums more than G′: the remap reduces the hum. A symbol-table defect "
               "contributes; the fix is a one-line remap in our pipeline, but it changes "
               "Sonora's text-to-id contract, so Prosodia (which pins it) is consulted before "
               "it lands."),
    "dense_speakers": ("S1 hums more than F1: dense speakers reduce the hum, and training on "
                       "them is a usable fix as it stands. Cross-voice: each pair rests on HNR "
                       "and F0 matching, and F1 bundles density with other changes (partition, "
                       "clip length, G2P version); each pair's partition is listed below."),
    "text_frontend": ("S1 hums more than E: at a 10-epoch budget, training on op_g2p instead of "
                      "the espeak input stock's text encoder learned adds hum. This does not "
                      "show op_g2p is worse in itself (S1 re-learns its input, E does not); "
                      "candidate fixes each need their own run."),
    "synthetic_reduces_hum": ("S1 hums more than Q: a synthetic corpus with no LibriVox audio in "
                              "the chain hums less than LibriTTS-R. The recording domain is the "
                              "candidate cause, not the speakers alone; F1 and Q are compared "
                              "below."),
    "synthetic_beats_dense_real": ("F1 hums more than Q: at matched speaker density (90 speakers, "
                                   "300+ clips each), the synthetic corpus hums less than dense "
                                   "LibriTTS-R. This separates the recording domain from density."),
}
# How each check's tests are counted in the print-out.
TESTS = {4: ("three p-values reported (S1's excess, S1 − L, L's excess), two of them "
             "deciding the reading (S1's excess and S1 − L); uncorrected"),
         5: ("three p-values reported (D1 − C0, D1's excess, C0's excess), one of them "
             "deciding the reading (D1 − C0); uncorrected"),
         6: ("three p-values reported (S1's excess, S1 − G′, G′'s excess), two of them "
             "deciding the reading (S1's excess and S1 − G′); uncorrected"),
         7: ("three p-values reported (S1's excess, S1 − F1, F1's excess), two of them "
             "deciding the reading (S1's excess and S1 − F1); uncorrected"),
         8: ("three p-values reported (S1's excess, S1 − E, E's excess), two of them "
             "deciding the reading (S1's excess and S1 − E); uncorrected"),
         9: ("six p-values reported (S1 − Q, F1 − Q, S1 − F1, and S1's, F1's and Q's own "
             "excess); three of them deciding the reading (S1's excess for validity, S1 − Q "
             "and F1 − Q); uncorrected")}


def unrated(test_dir):
    """(items, unrated ids) of one test folder: `items.json` names the items, and an item is
    rated when its (last) row in verdicts/verdicts.csv holds both severities."""
    ipath, vpath = Path(test_dir) / "items.json", Path(test_dir) / "verdicts" / "verdicts.csv"
    if not ipath.is_file():
        raise SystemExit("REFUSING: no %s — the session's items are not known." % ipath)
    ids = [i["id"] for i in json.loads(ipath.read_text())["items"]]
    done = {}
    if vpath.is_file():
        with vpath.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done[r["item"]] = r.get("sev_a", "") != "" and r.get("sev_b", "") != ""
    return len(ids), [i for i in ids if not done.get(i)]


def require_key_matches_folder(key, test_dir):
    """Refuse unless `key` is the key of the clips `test_dir` serves. Item ids are item_00..
    in every render of a check, so a key from another render (re-render after a crash, a new
    seed or --out) passes every other guard and maps each verdict to the wrong side: a
    confident, wrong reading. `items.json` says which opaque clip the listener heard on each
    side; the key says which arm label that clip is."""
    ipath = Path(test_dir) / "items.json"
    if not ipath.is_file():
        raise SystemExit("REFUSING: no %s — the key cannot be matched to the clips that were "
                         "served." % ipath)
    served = {i["id"]: i for i in json.loads(ipath.read_text())["items"]}
    truth, clips = key["items"], key.get("clips", {})
    if set(served) != set(truth):
        raise SystemExit("REFUSING: the key is not this folder's: it has %d items, %s has %d, "
                         "and they differ (e.g. %s)."
                         % (len(truth), ipath, len(served),
                            sorted(set(served) ^ set(truth))[:3]))
    wrong = [i for i, t in sorted(truth.items())
             if any(clips.get(served[i][side], {}).get("label") != t[side + "_label"]
                    for side in ("A", "B"))]
    if wrong:
        raise SystemExit("REFUSING: the key is not this folder's: for %d of %d items the clip "
                         "served on a side is not the arm the key records there (e.g. %s). "
                         "Use the key written by the render that wrote %s."
                         % (len(wrong), len(truth), ", ".join(wrong[:3]), ipath))


def report_peak_limited(test_dir):
    """Print the render's `peak_limited_sides` when non-zero. A side that hit the peak ceiling
    was quieter than its target; the renderer then matched its pair at that level, but the
    session was quieter than the registered -23 LUFS and the reader should know."""
    meta = Path(test_dir) / "render_meta.json"
    if not meta.is_file():
        return
    n = json.loads(meta.read_text()).get("peak_limited_sides") or 0
    if n:
        print("⚠ %d side(s) of this render hit the peak ceiling before reaching the target "
              "loudness (render_meta.json); their pairs were matched at the quieter level.\n"
              % n)


def severity(row, field):
    """A verdict's 0-5 severity as an int. "3.0" is read as 3; anything else is refused."""
    raw = row[field].strip()
    try:
        v = float(raw)
    except ValueError:
        v = None
    if v is None or v != int(v) or not 0 <= v <= 5:
        raise SystemExit("REFUSING: %s=%r for %r is not a whole number from 0 to 5."
                         % (field, row[field], row["item"]))
    return int(v)


def require_session_rated(args):
    """Checks 5-7: refuse unless every item of the session's three folders is rated, or
    --session-closed says the session is over. Check 8: single-folder session (--test only)."""
    if args.check not in sb.HUM_FIX_CHECKS:
        if args.session_tests or args.session_closed:
            raise SystemExit("REFUSING: --session-tests and --session-closed are for checks "
                             "5-9.")
        return
    if args.check in (8, 9):
        if args.session_tests:
            raise SystemExit("REFUSING: --session-tests is for checks 5-7 (the three-check "
                             "session); check %d is a single-folder session." % args.check)
        meta = Path(args.test) / "render_meta.json"
        if not meta.is_file():
            raise SystemExit("REFUSING: no %s — which check this folder was rendered for is "
                             "not known." % meta)
        test_check = json.loads(meta.read_text()).get("check")
        if test_check != args.check:
            raise SystemExit("REFUSING: --test %s was rendered for check %s, not --check %d."
                             % (args.test, test_check, args.check))
        total, bad = unrated(args.test)
        if bad and not args.session_closed:
            left = ["%s/%s" % (Path(args.test).name, i) for i in bad]
            raise SystemExit("REFUSING: %d of %d items in %s are not fully rated (e.g. %s). "
                             "Rate them all, or pass --session-closed to read the check without "
                             "them." % (len(left), total, Path(args.test).name,
                                        ", ".join(left[:3])))
        return
    if not args.session_tests:
        raise SystemExit("REFUSING: check %d needs --session-tests (the session's three test "
                         "folders)." % args.check)
    if str(Path(args.test).resolve()) not in {str(Path(d).resolve()) for d in args.session_tests}:
        raise SystemExit("REFUSING: --test %s is not among --session-tests." % args.test)
    resolved = [Path(d).resolve() for d in args.session_tests]
    if len(set(resolved)) != 3:
        raise SystemExit("REFUSING: --session-tests names the same folder more than once "
                         "(%s); the session is three distinct folders." % ", ".join(args.session_tests))
    checks = {}
    for d in args.session_tests:
        meta = Path(d) / "render_meta.json"
        if not meta.is_file():
            raise SystemExit("REFUSING: no %s — which check this folder was rendered for is "
                             "not known." % meta)
        checks[str(Path(d).resolve())] = json.loads(meta.read_text()).get("check")
    if sorted(checks.values(), key=str) != [5, 6, 7]:
        raise SystemExit("REFUSING: the three --session-tests folders must be the check 5, 6 "
                         "and 7 benches, one each; their render_meta.json say %s."
                         % ", ".join("%s=%s" % (Path(d).name, c) for d, c in checks.items()))
    test_check = checks[str(Path(args.test).resolve())]
    if test_check != args.check:
        raise SystemExit("REFUSING: --test %s was rendered for check %s, not --check %d."
                         % (args.test, test_check, args.check))
    total, left = 0, []
    for d in args.session_tests:
        n, bad = unrated(d)
        total += n
        left += ["%s/%s" % (Path(d).name, i) for i in bad]
    if left and not args.session_closed:
        raise SystemExit("REFUSING: %d of %d items in the session's three folders are not fully "
                         "rated (e.g. %s). Rate them all, or pass --session-closed to read the "
                         "check without them." % (len(left), total, ", ".join(left[:3])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, choices=tuple(sb.CHECKS), required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--session-tests", nargs=3, metavar="DIR",
                    help="checks 5-7: the session's three test folders, this check's among them")
    ap.add_argument("--session-closed", action="store_true",
                    help="checks 5-9: read the check although items are not fully rated; they "
                         "are left out")
    args = ap.parse_args(argv)
    arms = sb.CHECKS[args.check]["arms"]

    key = json.loads(Path(args.key).read_text())
    if key.get("check") != args.check:
        raise SystemExit("REFUSING: the key is check %s's, not check %d's."
                         % (key.get("check"), args.check))
    truth = key["items"]
    if {t["arm"] for t in truth.values()} != set(arms):
        raise SystemExit("REFUSING: the key's arms are not %s." % (arms,))
    require_session_rated(args)
    require_key_matches_folder(key, args.test)
    report_peak_limited(args.test)
    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit("REFUSING: no %s — nothing was judged." % vpath)

    gaps = {a: {} for a in arms}
    sev = {a: [] for a in arms}
    floor = {a: [] for a in arms}
    model_u, rt_u = {a: {} for a in arms}, {a: {} for a in arms}
    with vpath.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key." % r["item"])
            if r.get("sev_a", "") == "" or r.get("sev_b", "") == "":
                continue
            t = truth[r["item"]]
            a = t["arm"]
            s = {t["A_label"]: severity(r, "sev_a"), t["B_label"]: severity(r, "sev_b")}
            gaps[a][t["unit"]] = s[a] - s[a + "rt"]
            sev[a].append(s[a])
            floor[a].append(s[a + "rt"])
            model_u[a][t["unit"]], rt_u[a][t["unit"]] = s[a], s[a + "rt"]
    if "min_units" in sb.CHECKS[args.check]:
        # Checks 4-9 read the floors, levels and contrasts on one unit set: the units rated in
        # every arm. A unit rated in one arm only (--session-closed) would otherwise sit in that
        # arm's floor but in no contrast, and a lone high round-trip rating there could turn a
        # comparable pair of floors into a false INVALID. (A unit with S1 and F1 but no Q is
        # left out of the floors and of every printed count.)
        common = sb.rated_in_all(gaps, arms)
        gaps = {a: {u: gaps[a][u] for u in common} for a in arms}
        model_u = {a: {u: model_u[a][u] for u in common} for a in arms}
        rt_u = {a: {u: rt_u[a][u] for u in common} for a in arms}
        sev = {a: [model_u[a][u] for u in sorted(common)] for a in arms}
        floor = {a: [rt_u[a][u] for u in sorted(common)] for a in arms}
    rated = sum(len(v) for v in gaps.values())
    if rated < len(truth):
        print("⚠ %d of %d items not fully rated or not rated in every arm; they are left out.\n"
              % (len(truth) - rated, len(truth)))
    if not all(floor.values()) and "min_units" in sb.CHECKS[args.check]:
        # Pre-registered: fewer than 16 voices rated in both arms is INVALID, not a crash.
        print("OUTCOME: INVALID — %s" % OUTCOME[(args.check, "invalid")])
        print("  an arm has no rated item: fewer than 16 voices rated in both arms.")
        return {"outcome": "invalid", "shown": []}
    if not all(floor.values()):
        raise SystemExit("REFUSING: an arm has no rated item.")
    # Exact: the validity rule is strict (< 1 point, < 4), and a float mean of integer ratings
    # decides exact-1.0 gaps by rounding (3/20 vs 23/20 computes 0.9999999999999999).
    floors = {a: Fraction(sum(v), len(v)) for a, v in floor.items()}

    r = sb.reading(args.check, gaps, floors)
    print("CHECK %d — GAP PER ARM (model - round trip; one-sided p)" % args.check)
    for a in arms:
        m, p, n = r["level"][a]
        print("  %-6s n=%2d  model %.2f  round trip %.2f  gap %+.2f  p = %.4f"
              % (a, n, st.mean(sev[a]), float(floors[a]), m, p))
    if args.check in TESTS:
        print("\nCONTRASTS (per unit; one-sided) — %s" % TESTS[args.check])
    else:
        print("\nCONTRASTS (per unit; one-sided; uncorrected — %d tests in this check)"
              % r["n_tests"])
    for k, (m, p, n) in r["contrasts"].items():
        a, b = k.split("-")
        d = [gaps[a][u] - gaps[b][u] for u in sorted(set(gaps[a]) & set(gaps[b]))]
        print("  %-12s n=%2d  mean %+.2f  p = %.4f  %d up / %d down / %d same"
              % (k, n, m, p, sum(x > 0 for x in d), sum(x < 0 for x in d),
                 sum(x == 0 for x in d)))
    if args.check == 4:
        a, b = sb.CHECKS[4]["contrasts"][0]
        d = sb.split_contrast(model_u, rt_u, a, b)
        print("  descriptive, not tested: %s − %s on the model ratings alone %+.2f (n=%d); on "
              "the round-trip ratings alone %+.2f (n=%d); %s − %s on the gaps %+.2f"
              % (a, b, d["model"][0], d["model"][1], d["round_trip"][0], d["round_trip"][1],
                 b, a, -r["contrasts"]["%s-%s" % (a, b)][0]))
    if r["floors_ok"] is False:
        print("\n⚠ round-trip floors not comparable (within 1 point of %s's, all under 4)"
              % sb.CHECKS[args.check]["floor_ref"])

    print("\nOUTCOME: %s — %s" % (r["outcome"].upper(), OUTCOME[(args.check, r["outcome"])]))
    for s in r["shown"]:
        print("  shown: %s" % SHOWN[s])

    hnr = {}
    for t in truth.values():
        hnr.setdefault(t["unit"], {})[t["arm"]] = t["hnr"]
    print("\nunit  HNR per arm          " + "  ".join("%-6s" % a for a in arms))
    for u in sorted(hnr):
        print("%4d  %-20s %s" % (u, "/".join("%.2f" % hnr[u][a] for a in arms if a in hnr[u]),
                                 "  ".join("%+-6d" % gaps[a][u] if u in gaps[a] else "  .   "
                                           for a in arms)))
    if args.check == 7:
        print_pairs(truth)
    if args.check == 9:
        print_triplets(truth, sb.rated_in_all(gaps, arms))
    return r


def print_pairs(truth):
    """Check 7's pairs: each unit's S1 voice and F1 voice (v7 index), partition, HNR and F0.
    HNR is partly a recording property, so a pair that crosses partitions is visible."""
    by = {}
    for t in truth.values():
        by.setdefault(t["unit"], {})[t["arm"]] = t
    print("\nPAIRS (v7 index, partition, HNR, F0)")
    for u in sorted(by):
        s, f = by[u].get("S1"), by[u].get("F1")
        print("%4d  %s  |  %s" % (
            u, "S1 v7 %d  %-16s HNR %.2f  F0 %.0f" % (s["spk"], s["partition"], s["hnr"], s["f0"])
            if s else "S1 .",
            "F1 v7 %d (LibriTTS %d)  %-16s HNR %.2f  F0 %.0f"
            % (f["spk"], f["libritts_id"], f["partition"], f["hnr"], f["f0"]) if f else "F1 ."))


def print_triplets(truth, units):
    """Check 9's triplets (S1, F1, Q) on the units the readout used: each voice's HNR, F0 and
    partition, the `widened` flags (0.5 dB window widened to 1.0), and the realised HNR/F0
    gaps. The deciding F1 - Q contrast rests on voices matched only through S1, so their
    realised gap is shown, not only the registered window."""
    by = {}
    for t in truth.values():
        by.setdefault(t["unit"], {})[t["arm"]] = t
    print("\nTRIPLETS (voice, HNR, F0; widened = the 0.5 dB window was widened to 1.0)")
    dh, df = [], []
    for u in sorted(units):
        s, f, q = by[u]["S1"], by[u]["F1"], by[u]["Q"]
        w = q.get("widened") or {}
        dh.append(abs(f["hnr"] - q["hnr"]))
        df.append(abs(math.log(f["f0"] / q["f0"])))
        print("%4d  S1 %d %-16s %.2f %.0f  |  F1 %d %-16s %.2f %.0f  |  Q %s %.2f %.0f  |  "
              "widened F1 %s Q %s  |ΔHNR F1-Q| %.2f"
              % (u, s["spk"], s["partition"], s["hnr"], s["f0"], f["spk"], f["partition"],
                 f["hnr"], f["f0"], q["q_spk"], q["hnr"], q["f0"],
                 "yes" if w.get("F1") else "no", "yes" if w.get("Q") else "no", dh[-1]))
    if dh:
        print("F1 vs Q: |ΔHNR| mean %.2f max %.2f dB; |Δln F0| mean %.3f max %.3f; %d of %d "
              "triplets widened"
              % (st.mean(dh), max(dh), st.mean(df), max(df),
                 sum(1 for u in units if any((by[u]["Q"].get("widened") or {}).values())),
                 len(dh)))


if __name__ == "__main__":
    main()

"""Pairing speakers who share a PITCH and differ (or do not) in PERIODICITY.

⚠⚠ ONE COPY, BECAUSE THE CONTRAST AND ITS CONTROL MUST BE BUILT THE SAME WAY. This bench
design rests on two pair sets from one corpus: a CONTRAST set whose members differ in HNR,
and a CONTROL set whose members do not. If the two were built by separate code they could
drift on the pitch tolerance, the partition rule or the row-count rule, and the control
would then differ from the contrast in more than the thing under test — which is the exact
failure the control exists to catch. `build_hnr_error_filelist.py` and
`render_ear_hnr_speaker.py` both call this.

Why pairing at all: F0 and HNR correlate at +0.905 across the 1526 eligible speakers of
LibriTTS-R, and the off-diagonal population is 7% at every sample size, all of it
off-diagonal by a hair. The corpus cannot supply a decorrelated draw, so pitch is held
fixed by construction instead.
"""


def matched_pairs(spk, rows_of, max_f0_gap, hnr_lo, hnr_hi, row_ratio, match_partition):
    """Disjoint speaker pairs within `max_f0_gap` Hz whose HNR gap lies in [lo, hi].

    `spk` maps speaker -> {"f0", "hnr", "partition"}; `rows_of(s)` gives the training row
    count used by the volume control. Returns `(rough, clean, hnr_gap, mean_f0, d_f0)`
    tuples, `rough` being the LOWER-HNR speaker.

    ⚠ THE WIDEST GAP IS TAKEN FOR A CONTRAST AND THE NARROWEST FOR A CONTROL, which is why
    the choice is keyed on `hnr_lo` rather than left to the caller. A control built by
    taking the widest gap under its ceiling would sit as close to the contrast as the
    ceiling allows, and a control is only worth having when it is as far from the contrast
    as the corpus permits.
    """
    want_wide = hnr_lo > 0
    cand = sorted(spk, key=lambda s: spk[s]["f0"])
    used, pairs = set(), []
    for i, a in enumerate(cand):
        if a in used:
            continue
        best = None
        for b in cand[i + 1:]:
            if b in used:
                continue
            if spk[b]["f0"] - spk[a]["f0"] > max_f0_gap:
                break                   # sorted by F0, so nothing further can qualify
            if match_partition and spk[a]["partition"] != spk[b]["partition"]:
                continue
            ra, rb = rows_of(a), rows_of(b)
            if row_ratio and max(ra, rb) / min(ra, rb) > row_ratio:
                continue
            gap = abs(spk[a]["hnr"] - spk[b]["hnr"])
            if not (hnr_lo <= gap <= hnr_hi):
                continue
            if best is None or (gap > best[0] if want_wide else gap < best[0]):
                best = (gap, b)
        if best is None:
            continue
        gap, b = best
        rough, clean = (a, b) if spk[a]["hnr"] < spk[b]["hnr"] else (b, a)
        used.add(a)
        used.add(b)
        pairs.append((rough, clean, gap,
                      (spk[rough]["f0"] + spk[clean]["f0"]) / 2.0,
                      spk[rough]["f0"] - spk[clean]["f0"]))
    return pairs

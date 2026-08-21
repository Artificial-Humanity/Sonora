"""SCM v0.1 — Sonora Conveyance Markup core (markup-schema-brief.md, RATIFIED 2026-07-19).

Sidecar-canonical: these helpers validate a sidecar object, render the human-facing
inline projection (never parsed back), and verify measurable claims against the
instrument decode (utterance VAT within `VAT_TOL`; spans activate when the span decode
layer lands). Interpretive fields (style, direction) are never load-bearing.
"""

import math

import schemas

VAT_TOL = 0.35  # ratified tolerance (owner amendment 2026-07-19: widened from
                # 0.25 to match the 0.4 quantizer bin width; copy-through declined)
SPAN_TYPES = {"emphasis", "pause_after", "pace", "pitch_move", "nonverbal", "quote"}
PAUSE_BINS = {"micro", "short", "med", "long"}
GENDERS = {"Male", "Female", "Undefined"}
AGE_BANDS = {"child", "young", "adult", "senior"}


def _show(v, limit=40):
    """`repr(v)`, truncated.

    ⚠ The value that motivated this is a JSON integer too large for a float: its repr runs
    to 400 digits, and an error message that scrolls the terminal hides the ones printed
    beside it. That case has had **its own branch** since issue #145 — it is not
    "non-finite", which this docstring called it until #150; `10**400` is an exact int and
    perfectly finite. What it cannot do is fit in a float64.
    """
    r = repr(v)
    return r if len(r) <= limit else r[:limit] + f"… ({len(r)} chars)"


def validate(obj, lexicon):
    """Return a list of schema violations (empty = valid SCM v0.1)."""
    errs = []
    if obj.get("scm") != "0.1":
        errs.append("scm version missing/wrong")
    utt = obj.get("utterance") or {}
    vat = utt.get("vat") or {}
    # ⚠⚠ STRICT ON TYPE, AND THAT IS NOT THE SAME QUESTION `coerce_axis` ANSWERS.
    # Corrected 2026-08-19 after checking the ratified brief, which the first fix did not:
    # `docs/markup-schema-brief.md` §"Field semantics" says `utterance.vat` is
    # **continuous [−1,1]**, and §"Numbers vs symbols" says the sidecar keeps VAT
    # continuous because IT IS THE CONTRACT — symbols are presentation, never storage.
    #
    # So a JSON string is a schema violation here even though `coerce_axis("0.7")` is 0.7.
    # `coerce_axis` answers "is this a usable number for a LABEL", which is the tolerant
    # question a director's output gets asked. `validate` answers "does this sidecar
    # conform to a ratified wire format", which is the strict one. Routing this through
    # `coerce_axis` (issue #117's fix) collapsed the two and made the validator CERTIFY a
    # sidecar the contract forbids — the same over-broad move #113's own guard had to be
    # narrowed to avoid, made one file over.
    #
    # ⚠ The BOUNDS still come from `schemas`, because those genuinely are one constant and
    # a literal here was a second copy of them.
    for ch in schemas.AXES:
        v = vat.get(ch)
        # ⚠ THREE MESSAGES, BECAUSE THERE ARE THREE REPAIRS (issue #132). "missing or not a
        # number" told an operator holding `"0.7"` that their axis was ABSENT — so the
        # obvious action is to go and produce a label that is already there, and the actual
        # fault (a string where the contract wants a number) goes unread. The classification
        # was right and the instruction beside it was wrong, which is this branch's most
        # expensive recurring shape.
        if ch not in vat or v is None:
            errs.append(f"vat.{ch} is missing — the sidecar must carry all three axes")
        elif not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append(f"vat.{ch} is {type(v).__name__} {_show(v)}, not a number — the sidecar "
                        f"stores VAT continuous (markup-schema-brief.md, RATIFIED v0.1). "
                        f"Emit it as a JSON number, not a string.")
        elif schemas.coerce_axis(v) is None:
            # ⚠ A FOURTH REPAIR, AND IT WORE THE THIRD ONE'S INSTRUCTION (issue #137).
            # `isinstance(nan, float)` is True, so NaN passed the type check and then failed
            # the range comparison — every comparison against NaN is False — and was
            # reported as "out of [-1,1]". It is not out of range; it is not a number.
            #
            # ⚠⚠ THE OUTER TEST IS `coerce_axis`, NOT `math.isnan(v) or math.isinf(v)`
            # (issue #142). That was the remedy #137 suggested, marked UNVERIFIED, and I
            # implemented it verbatim without checking it: `math.isnan` CONVERTS its
            # argument to a float first, so on a JSON integer too large for one it raises
            # `OverflowError` — and `scm.validate`, whose contract is to RETURN a list of
            # violations, raised instead. The identical conversion is what #141 had just
            # been filed about, one module over.
            #
            # ⚠ `math.isnan` DOES appear below, and that is not a contradiction (issue
            # #149). What makes it safe is the `isinstance(v, float)` CONJUNCT ON THAT LINE
            # — not anything above it. The only `isinstance` above is `(int, float)`, which
            # admits the oversized int, and that int reaches here: `coerce_axis` catches its
            # `OverflowError` and returns None, which is why the `else:` below exists at all.
            #
            # ⚠ SO DO NOT DELETE THAT CONJUNCT AS REDUNDANT. It is the whole guard: without
            # it `math.isnan(10**400)` raises and #142 is back. Reading the `else:` as dead
            # undoes #145 the same way. A previous version of this paragraph said `v` was
            # "already proved a float by the isinstance test above", which is false and
            # invited exactly those two edits.
            #
            # ⚠ The TYPE check above stays separate and strict, because that is the
            # contract question (#117): a JSON string is a violation even though
            # `coerce_axis("0.7")` is 0.7.
            # ⚠ TWO DIAGNOSES, BECAUSE `coerce_axis` RETURNS None FOR TWO DIFFERENT THINGS
            # (issue #145). It refuses NaN and the infinities, AND it refuses a value that
            # cannot be converted to a float at all — a JSON integer larger than float64
            # can hold, which is #141's own case. `10**400` IS a finite number; it is an
            # exact Python int. Telling its author it "is not a finite number" is false of
            # the value in front of them, and the repair differs: a NaN comes from a broken
            # instrument, an oversized int comes from a wrong SCALE.
            #
            # ⚠ NO "write it as null" (issue #144). The branch four lines above rejects a
            # null axis outright — "the sidecar must carry all three axes" — and the
            # ratified brief has no null form for `utterance.vat`. Sending the reader there
            # is one validator handing out two contradictory instructions.
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                errs.append(f"vat.{ch} is {_show(v)}, not a finite number — re-derive it "
                            f"from the instrument")
            else:
                errs.append(f"vat.{ch} is {_show(v)}, too large to represent as a float — "
                            f"re-derive it on the clamped per-speaker z scale")
        elif not schemas.AXIS_MIN <= v <= schemas.AXIS_MAX:
            errs.append(f"vat.{ch} is {_show(v)}, out of "
                        f"[{schemas.AXIS_MIN:g},{schemas.AXIS_MAX:g}] — clamp it to the "
                        f"trained range or re-derive it")
    reg = utt.get("register")
    if reg is not None and reg not in lexicon:
        errs.append(f"register '{reg}' not in controlled lexicon")
    style = utt.get("style", [])
    if not isinstance(style, list) or len(style) > 3:
        errs.append("style must be a list of <=3 adjectives")
    n_tok = len((obj.get("text") or "").split())
    cast = obj.get("cast") or {}
    for key, c in cast.items():
        if c.get("gender") not in GENDERS | {None}:
            errs.append(f"cast.{key}.gender invalid")
        if c.get("age_band") not in AGE_BANDS | {None}:
            errs.append(f"cast.{key}.age_band invalid")
    for i, s in enumerate(obj.get("spans", []) or []):
        t = s.get("type")
        if t not in SPAN_TYPES:
            errs.append(f"spans[{i}].type '{t}' unknown")
        tok = s.get("tok")
        if (not isinstance(tok, list) or len(tok) != 2
                or not all(isinstance(x, int) for x in tok)
                or not 0 <= tok[0] <= tok[1] < max(n_tok, 1)):
            errs.append(f"spans[{i}].tok invalid for {n_tok}-token text")
        if t == "pause_after" and s.get("bin") not in PAUSE_BINS:
            errs.append(f"spans[{i}].bin invalid")
        if t == "emphasis" and s.get("level") not in (1, 2, 3):
            errs.append(f"spans[{i}].level invalid")
        if t == "quote" and s.get("cast") not in cast:
            errs.append(f"spans[{i}].cast '{s.get('cast')}' not in cast map")
    if not isinstance(obj.get("direction", ""), str):
        errs.append("direction must be a string")
    return errs


def verify_vat(obj, measured):
    """Check utterance VAT against instrument values {V,A,T}; returns (ok, flags)."""
    flags = []
    vat = (obj.get("utterance") or {}).get("vat") or {}
    # ⚠ ALL THREE READERS IN THIS MODULE GO THROUGH `coerce_axis` (issue #117). Only
    # `validate` was delegated on 2026-08-19, which made it CERTIFY a numeric-string axis
    # that these two still crashed on — `abs("0.7" - m)` is a TypeError and the renderer
    # below raises "Sign not allowed in string format specifier". Before that change all
    # three refused a string together; after it, "no schema violations" stopped implying
    # "this module can process it", and `tag_spike` writes `verified: true` beside a value
    # its own renderer cannot format.
    for ch in schemas.AXES:
        m, c = schemas.coerce_axis(measured.get(ch)), schemas.coerce_axis(vat.get(ch))
        if m is None or c is None:
            continue
        if abs(c - m) > VAT_TOL:
            flags.append(f"vat.{ch} claimed {c:+.2f} vs measured {m:+.2f} (>±{VAT_TOL})")
    return (not flags, flags)


def render_inline(obj):
    """Deterministic inline projection (view only, single line for the audit note)."""
    utt = obj.get("utterance") or {}
    vat = utt.get("vat") or {}
    # ⚠ `coerce_axis`, and `or 0.0` for the absent case — see verify_vat above (issue #117).
    head = "[" + " ".join(f"{c}{schemas.coerce_axis(vat.get(c)) or 0.0:+.2f}" for c in ("V", "A", "T"))
    if utt.get("register"):
        head += f" · {utt['register']}"
    if utt.get("style"):
        head += " · " + ",".join(utt["style"])
    head += "]"
    toks = (obj.get("text") or "").split()
    opens, closes = {}, {}
    for s in sorted(obj.get("spans", []) or [], key=lambda s: s.get("tok", [0])[0]):
        tok = s.get("tok") or [0, 0]
        t = s.get("type")
        if t == "emphasis":
            o, c = f"<em{s.get('level', 1)}>", f"</em{s.get('level', 1)}>"
        elif t == "pause_after":
            o, c = "", f"<pause:{s.get('bin')}/>"
        elif t == "pace":
            o, c = f"<pace:{s.get('value')}>", "</pace>"
        elif t == "pitch_move":
            o, c = "", f"<pitch:{s.get('value')}/>"
        elif t == "nonverbal":
            o, c = f"<{s.get('value')}>", f"</{s.get('value')}>"
        elif t == "quote":
            o, c = f"<quote:{s.get('cast')}>", "</quote>"
        else:
            continue
        if o:
            opens[tok[0]] = opens.get(tok[0], "") + o
        closes[tok[1]] = c + closes.get(tok[1], "")
    body = " ".join(opens.get(i, "") + w + closes.get(i, "") for i, w in enumerate(toks))
    out = f"{head} {body}"
    if obj.get("direction"):
        out += f" | dir: {obj['direction']}"
    return out

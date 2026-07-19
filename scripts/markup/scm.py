"""SCM v0.1 — Sonora Conveyance Markup core (markup-schema-brief.md, RATIFIED 2026-07-19).

Sidecar-canonical: these helpers validate a sidecar object, render the human-facing
inline projection (never parsed back), and verify measurable claims against the
instrument decode (utterance VAT within ±0.25; spans activate when the span decode
layer lands). Interpretive fields (style, direction) are never load-bearing.
"""

VAT_TOL = 0.25  # ratified verifier tolerance on the clamped [-1, 1] scale
SPAN_TYPES = {"emphasis", "pause_after", "pace", "pitch_move", "nonverbal", "quote"}
PAUSE_BINS = {"micro", "short", "med", "long"}
GENDERS = {"Male", "Female", "Undefined"}
AGE_BANDS = {"child", "young", "adult", "senior"}


def validate(obj, lexicon):
    """Return a list of schema violations (empty = valid SCM v0.1)."""
    errs = []
    if obj.get("scm") != "0.1":
        errs.append("scm version missing/wrong")
    utt = obj.get("utterance") or {}
    vat = utt.get("vat") or {}
    for ch in ("V", "A", "T"):
        v = vat.get(ch)
        if not isinstance(v, (int, float)) or not -1.0 <= v <= 1.0:
            errs.append(f"vat.{ch} missing or out of [-1,1]")
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
    for ch in ("V", "A", "T"):
        m, c = measured.get(ch), vat.get(ch)
        if m is None or c is None:
            continue
        if abs(c - m) > VAT_TOL:
            flags.append(f"vat.{ch} claimed {c:+.2f} vs measured {m:+.2f} (>±{VAT_TOL})")
    return (not flags, flags)


def render_inline(obj):
    """Deterministic inline projection (view only, single line for the audit note)."""
    utt = obj.get("utterance") or {}
    vat = utt.get("vat") or {}
    head = "[" + " ".join(f"{c}{vat.get(c, 0):+.2f}" for c in ("V", "A", "T"))
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

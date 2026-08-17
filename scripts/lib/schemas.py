"""Validated loaders for the data contracts that enter this repo from files.

WHY THIS EXISTS. 77 files call `json.load` against no schema at all, and the tree holds one
dataclass and zero `TypedDict`. Where an invariant IS known it tends to live in a gate test —
which runs in CI and protects no runtime read. So a malformed asset is caught on the machine
that runs the suite and nowhere else.

⚠ THE TARGET IS THE SILENT DEGRADE, NOT THE CRASH. A file that fails to parse loudly is
already survivable. What is not is a loader that answers with a plausible empty value, because
downstream guards are usually written as `if VOCAB and x not in VOCAB` — and an empty vocab
switches the guard off at the same moment it empties the prompt that guard was protecting.
Both halves then fail in the same direction with nothing on stdout. That is the shape
`build_direction()` was already made fatal for on the engine axis (`book_ingest` §"unknown
engine"); this module is the same decision for the axes that were left behind.

Pydantic was already an installed dependency (2.13.4, transitively) before this module, so
nothing here adds one.

⚠ VOCABULARIES ARE DERIVED, NEVER RE-SPELLED. Every controlled list below is imported from
the module that owns it. A literal copied into this file would be exactly the defect the file
exists to prevent — issue #28 was one word ("unknown") spelled twice from two independent
literals, and this is the file most likely to become the second literal for all of them.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

from pydantic import BaseModel, ConfigDict, field_validator

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(_REPO, "scripts", "assets")

# ⚠ NAMED AS A CONSTANT ENDING IN THE FILENAME, not joined inside the loader.
# `tests/test_asset_paths.py` only checks a path whose TAIL names a tracked file, so
# `os.path.join(ASSETS, "register_lexicon.json")` buried in a function body is invisible to
# it — the asset would stop being falsifiable at the moment its loading became strict, which
# is precisely backwards. Consolidating the two old call sites into one already costs the
# ratchet a count; it should not also cost it the check.
# ⚠ SPELLED AS ONE JOIN FROM THE REPO ANCHOR, not `join(ASSETS, ...)`. The evaluator in
# `tests/test_asset_paths.py` resolves a join rooted at the repo constant; it cannot follow a
# two-step join through an intermediate variable, so the `ASSETS`-based spelling type-checked,
# ran correctly, and was invisible to the guard. Correct code that no guard can see is the
# thing this whole branch is about.
REGISTER_LEXICON_PATH = os.path.join(_REPO, "scripts", "assets", "register_lexicon.json")

_SNAKE = re.compile(r"[a-z0-9_]+")


class SchemaError(ValueError):
    """A data contract was violated. Deliberately not caught anywhere in this module.

    ⚠ Callers must not wrap a load in `except Exception` and substitute a default. That is
    the failure this module exists to remove, and it reads as defensive programming right up
    until the default silently disables a downstream guard.
    """


class RegisterLexicon(BaseModel):
    """`scripts/assets/register_lexicon.json` — the controlled register vocabulary.

    The invariants below were already known and already checked, in
    `scripts/gates/test_skill_files.py::test_register_lexicon`. That gate stays: it is what
    tells a human the *asset* is wrong. This model is what stops a *runtime read* proceeding
    on a broken one — the two are different jobs and the gate could never do this one.
    """

    model_config = ConfigDict(extra="allow")  # provenance keys are free to grow

    lexicon: List[str]
    source: str = ""
    min_keeps: int = 0
    n_certified_keeps: int = 0
    n_distinct_labels_seen: int = 0

    @field_validator("lexicon")
    @classmethod
    def _check(cls, v: List[str]) -> List[str]:
        # ⚠ EMPTY IS THE ONE THAT MATTERS. The other three produce odd behaviour; this one
        # produces NO behaviour — the director is handed "pick exactly from: " and the
        # `if REGISTER_LEXICON and ...` guard downstream stops testing anything.
        if not v:
            raise SchemaError(
                "register lexicon is empty. An empty controlled vocabulary is not a degraded "
                "vocabulary — it removes the list from the director's prompt AND switches off "
                "the off-lexicon guard that would have caught the result. Regenerate it with "
                "scripts/tools/build_register_lexicon.py."
            )
        if v != sorted(v):
            raise SchemaError("register lexicon is not sorted; diffs would be unstable")
        if len(v) != len(set(v)):
            dupes = sorted({x for x in v if v.count(x) > 1})
            raise SchemaError("register lexicon has duplicates: %s" % dupes[:5])
        bad = [x for x in v if not _SNAKE.fullmatch(x)]
        if bad:
            raise SchemaError("register labels must be snake_case: %s" % bad[:5])
        return v


def load_register_lexicon(path: str | None = None) -> List[str]:
    """The controlled register vocabulary, or an exception. Never a plausible empty list.

    ⚠ This replaced three separate readers that disagreed about failure:
      * `book_ingest._lexicon()`  — `except Exception: return []`, the silent degrade above;
      * `make_teacher_ab_bank`    — bare `json.load(...)["lexicon"]` at import time, which is
                                    loud but raises `KeyError`/`JSONDecodeError` with no
                                    indication of what the file was for;
      * `gates/test_skill_files`  — the only place the invariants were stated, in a test.
    """
    p = path or REGISTER_LEXICON_PATH
    try:
        with open(p, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as e:
        raise SchemaError("register lexicon not found at %s" % p) from e
    except json.JSONDecodeError as e:
        raise SchemaError("register lexicon at %s is not valid JSON: %s" % (p, e)) from e
    if not isinstance(raw, dict):
        raise SchemaError("register lexicon at %s must be a JSON object, got %s"
                          % (p, type(raw).__name__))
    return RegisterLexicon(**raw).lexicon


# --- delivery lanes ---------------------------------------------------------------------
# Imported, never re-spelled. `matcha/delivery.py` owns the vocabulary and the retired/active
# split, and its ORDER IS THE WIRE FORMAT — a copy here that drifted by one entry would
# silently reinterpret every filelist and every checkpoint written at that `vat_dim`.


def _delivery():
    from matcha import delivery
    return delivery


def validate_delivery_label(lane: str, *, producing: bool) -> str:
    """Check a delivery lane, on the right side of the retired/active split.

    ⚠ `producing` IS NOT A STYLE CHOICE, and getting it backwards is silent both ways.
    `matcha/delivery.py` states the rule: a NEW row may only be labelled with an ACTIVE lane,
    while READERS stay on the full vocabulary because v5's filelists and ep019's weights both
    carry the retired channel and must keep decoding.

      producing=True   labelling a row, writing a manifest  -> ACTIVE_DELIVERY_LANES
      producing=False  conditioning a render, decoding      -> DELIVERY_LANES (full)

    Passing `producing=False` where a row is being written lets a retired lane back into the
    corpus; passing `True` where a render is conditioned refuses a lane the checkpoint can
    legitimately still decode.
    """
    d = _delivery()
    if lane == d.DELIVERY_UNKNOWN:
        return lane
    allowed = d.ACTIVE_DELIVERY_LANES if producing else d.DELIVERY_LANES
    if lane not in allowed:
        retired = producing and lane in d.DELIVERY_LANES
        raise SchemaError(
            "delivery lane %r is not %s. Allowed: %s.%s"
            % (lane, "assignable" if producing else "known", list(allowed),
               "  It is a RETIRED lane: the channel still decodes, but no new row may be "
               "labelled with it." if retired else "")
        )
    return lane

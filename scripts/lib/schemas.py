"""Validated loaders for the data contracts that enter this repo from files.

WHY THIS EXISTS. Most JSON entering this repo is read against no schema at all, and the tree
holds one `@dataclass` (`matcha/direction.py:59`) and zero `TypedDict`. Where an invariant IS
known it tends to live in a gate test — which runs in CI and protects no runtime read. So a
malformed asset is caught on the machine that runs the suite and nowhere else.

⚠ THE NUMBER NOW CARRIES ITS METHOD, because the first version did not and could not be
reproduced (issue #97). It said "77 files", which matches no counting rule anyone could find:
the plausible ones give 90 (any `json.load`, including `json.loads`, tests included), 80 (same,
tests excluded) and 61 (`json.load(` — an actual file read — tests excluded). A motivating
figure that a reader cannot re-derive is worse than no figure, because it gets quoted onward.
Measured 2026-08-17, and re-runnable:

    git ls-files '*.py' | grep -v '^tests/' | xargs grep -lE 'json\\.load\\(' | wc -l   # 61

⚠ THE TARGET IS THE SILENT DEGRADE, NOT THE CRASH. A file that fails to parse loudly is
already survivable. What is not is a loader that answers with a plausible empty value, because
downstream guards are usually written as `if VOCAB and x not in VOCAB` — and an empty vocab
switches the guard off at the same moment it empties the prompt that guard was protecting.
Both halves then fail in the same direction with nothing on stdout. That is the shape
`build_direction()` was already made fatal for on the engine axis (`book_ingest` §"unknown
engine"); this module is the same decision for the axes that were left behind.

⚠ PYDANTIC IS A CORE DEPENDENCY BECAUSE OF THIS MODULE, and an earlier draft of this
paragraph said the opposite (issue #96) — "already an installed dependency (2.13.4,
transitively), so nothing here adds one" — while `pyproject.toml` in the SAME COMMIT added it
to `[project.dependencies]` and explained why. Both statements cannot be true, and the
reassuring one was false in the way that matters: pydantic was reachable only through
`fastapi`, which lives in the `vocalizer` extra, so a dev box had it by accident and a
training or eval image would not have had it at all. `book_ingest` imports this module at
module scope, which puts it on the teacher-synthesis path every container takes. The failure
would have been an `ImportError` in a container, where no test could see it.

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

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

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
    """A data contract was violated. The ONE exception type this module's loaders raise.

    ⚠ Callers must not wrap a load in `except Exception` and substitute a default. That is
    the failure this module exists to remove, and it reads as defensive programming right up
    until the default silently disables a downstream guard.

    ⚠ RAISING THIS INSIDE A PYDANTIC VALIDATOR DOES NOT REACH THE CALLER (issue #94, measured).
    Pydantic catches `ValueError` — which this subclasses — out of a `field_validator` and
    converts it into `pydantic_core.ValidationError`. So every lexicon invariant below used to
    surface as a type the caller was never told about, INCLUDING the empty case that
    `RegisterLexicon._check` calls the one that matters. A caller who wrote the narrow
    `except SchemaError` that this module's own guard test reasons about would have caught a
    missing file and let an empty vocabulary straight through.

    The tests could not see it: they asserted on the MESSAGE through `pytest.raises(Exception)`,
    and pydantic preserves the message ("Value error, <text>"), so they passed either way. The
    asymmetry was the tell — `intended_vat`'s tests, which do not go through pydantic, pinned
    `SchemaError` exactly. The type was pinned where it happened to hold and left loose where
    it did not. Loaders now convert at the boundary; `tests/test_schemas.py` pins the type on
    every path.
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
    # ⚠ THE CONVERSION IS THE POINT (#94). Everything above raises `SchemaError` directly;
    # everything pydantic checks would otherwise escape as `ValidationError`, so the same
    # broken asset raised two unrelated types depending on HOW it was broken. Converting here
    # rather than in the validator keeps one boundary: the model stays a plain pydantic model
    # and this function stays the only thing callers have to reason about.
    try:
        return RegisterLexicon(**raw).lexicon
    except ValidationError as e:
        raise SchemaError("register lexicon at %s is invalid: %s" % (p, e)) from e


# --- intended V/A/T, at the point of WRITE -----------------------------------------------
# `qc_measures.jsonl` is read by seven modules. One of them, `qc_verdict.py`, was hardened by
# hand after issue #58 — a JSON string like "0.7" was dropped as silently as "very sad", so a
# fully directed bank counted as ZERO directed clips and `synth_bank.sh` announced it as real
# audio and skipped the direction check. That fix is good and stays.
#
# ⚠ BUT IT FIXED THE READER. Its own docstring names the root and leaves it open: "book_ingest
# writes {"V": tag.get("valence", 0.0), ...} straight out of the LLM and validates only
# `register` and `engine`". So six other readers still receive whatever the director emitted,
# and each would have to re-derive the same defence. Validating on write is one place instead
# of seven, and it keeps the bad value out of the file rather than out of one consumer.

AXES = ("V", "A", "T")

# The trained range. `matcha/cli.py --vat` documents "comma-separated floats in [-1, 1]" and
# derivation clamps per-speaker z-scores at 2 sigma; a label outside this is not a strong
# label, it is a mistake that will be clamped silently downstream.
AXIS_MIN, AXIS_MAX = -1.0, 1.0


def coerce_axis(v):
    """One axis value as a float, or None when it carries no usable number.

    ⚠ THE SINGLE DEFINITION. `qc_verdict.coerce_axis` delegates here rather than keeping its
    own copy: two implementations of "what counts as a number" would drift, and the drift
    would be invisible because both sides would still return floats for the easy cases.

    A JSON string that spells a number IS a label ("0.7" is 0.7); a string that spells
    anything else is not, and the difference has to stay visible (issue #58).
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def intended_vat(tag, *, strict=True):
    """The `intended` block for a manifest row, from a director's raw tag.

    ⚠ AN ABSENT AXIS STAYS ABSENT. It is written as `None`, never as `0.0`.
    `tag.get("valence", 0.0)` — what both writers used — turns "the director said nothing"
    into "the director said neutral", and those are different facts. `qc_verdict`'s reader
    goes to real trouble to distinguish them ("an absent axis ... is not collected, because
    there is nothing wrong with it") and the writer's default made that branch UNREACHABLE
    from this path: every row arrived with all three axes present and numeric.

    `strict=True` refuses a value that is present and unreadable, because that is the case
    where the director produced something and nobody can tell what. `strict=False` records it
    as absent instead, for callers that must not fail a whole run on one bad line.
    """
    src = tag if isinstance(tag, dict) else {}
    long_names = {"V": "valence", "A": "arousal", "T": "tension"}
    out = {}
    for ax in AXES:
        raw = src.get(long_names[ax], src.get(ax))
        if raw is None:
            out[ax] = None
            continue
        v = coerce_axis(raw)
        if v is None:
            if strict:
                raise SchemaError(
                    "intended %s is present but unreadable: %r. A directed clip whose label "
                    "cannot be parsed is not an undirected clip — write it as absent (null) "
                    "or fix the director." % (long_names[ax], raw))
            out[ax] = None
            continue
        if not (AXIS_MIN <= v <= AXIS_MAX):
            if strict:
                raise SchemaError(
                    "intended %s = %r is outside the trained range [%s, %s]. Downstream this "
                    "is clamped silently, so the label would not mean what it says."
                    % (long_names[ax], v, AXIS_MIN, AXIS_MAX))
            out[ax] = None
            continue
        out[ax] = v
    return out


def intended_labels(tag, *, strict=False):
    """The same axes as `intended_vat`, but an absent axis is OMITTED rather than `None`.

    ⚠ TWO SHAPES BECAUSE THERE ARE TWO CONTRACTS, and conflating them broke a live run
    (issues #90, #91, #92 — all reproduced).

      * `intended_vat`    -> the MANIFEST. A durable JSON record, where an explicit `null`
                             is a fact: the director was asked and said nothing. Dropping the
                             key would make "not asked" and "asked, no answer" identical.
      * `intended_labels` -> an IN-MEMORY label dict handed to a prompt builder, where the
                             reader's own idiom is `for k in ("V","A","T") if k in labels`.

    That `if k in labels` guard is PRE-EXISTING and was written for exactly this — an axis the
    director never produced. Making every key always present defeated it silently, and the
    line under it formats with `:+.2f`, so a `None` reached `__format__` and took the whole
    book-ingest run down. The guard was right; the value shape was wrong.

    Defaults to lenient: a caller building a prompt should degrade to fewer labels, never
    abort a retry loop. The manifest write is where a bad axis is fatal.
    """
    return {k: v for k, v in intended_vat(tag, strict=strict).items() if v is not None}


def fmt_axis(v, spec="+.1f", absent="  · "):
    """One axis for HUMAN OUTPUT, where `None` must print rather than raise.

    ⚠ Progress lines are not diagnostics — they are the only thing a long run shows, and this
    one sat AFTER the checkpoint `write`/`flush`/`fsync`. So a `TypeError` here killed the run
    with the offending row already durable: resume skipped it, then died on the next chunk
    missing an axis. A run made exactly one chunk of progress per restart (issue #91).
    """
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else absent


# --- delivery lanes: DELIBERATELY NOT HERE ---------------------------------------------
#
# ⚠ A `validate_delivery_label()` lived here for one commit and was REMOVED (issue #95). It
# had zero production call sites — its only four callers were its own tests — but "unwired"
# was the smaller half of the problem.
#
# `matcha.delivery.check_assignable()` already owns this rule, and it is wired at the sites
# that matter: `scripts/stages/register_audition.py` (the synthesis pipeline's write site for
# `delivery`), `scripts/lib/ref_select.py`, and the corpus merge. A second implementation here
# is exactly what this module's own docstring says it is most at risk of becoming — "the
# second literal" — and it would have been the more dangerous copy, because it encoded the
# active/retired split as a `producing=` BOOLEAN that a caller can get backwards in silence:
# `producing=False` on a write lets a retired lane back into the corpus, `producing=True` on a
# render refuses a lane the checkpoint can legitimately still decode.
#
# The right shape for a new caller is `from matcha.delivery import check_assignable`, not a
# re-export through this module. Adding a thin wrapper here would restore the fork with an
# extra layer of indirection over it.

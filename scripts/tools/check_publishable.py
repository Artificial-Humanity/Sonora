#!/usr/bin/env python
"""Run the publish wall over a Lightning checkpoint before promoting it (ARCHITECTURE §7).

The LiteRT export calls the wall itself; the registry promotion is a hand process with no
code on it, so this is the runnable form of the §7 checklist item. Exit 0 = every filelist
in the lineage was opened and nothing in it is `publish: forbidden`; 1 = refused; 2 = the
lineage is UNKNOWN or partly unread, which is not a clean result; 3 = it could not run — no
torch, or the checkpoint could not be read.

⚠ A CHECKPOINT THAT WILL NOT LOAD USED TO EXIT 1, WHICH THIS DOCSTRING ASSIGNS TO "refused".
`torch.load` raised through `main()`, so a mistyped path produced a traceback and status 1 —
indistinguishable, to anything reading the exit code, from the wall refusing an artifact that
must not ship. That is the wrong direction to be wrong in: a typo read as a refusal is
survivable, a refusal read as a typo is not, and a promoter scripting this could not tell
them apart either way.

    "${SONORA_LITERT_PY:-/data/toolchain/litert-conversion/.venv/bin/python}" \
        scripts/tools/check_publishable.py /path/to/checkpoint.ckpt

⚠ NOT `.venv/bin/python`, which this line and ARCHITECTURE §7 both named until #428. Loading a
checkpoint needs torch, and the repo venv deliberately has none: AGENTS.md §3 populates it from
the `test` dependency group, which excludes torch on purpose. So the §7 checklist item died at
`import torch` on the very checkout it was written from — measured. The interpreters that do
have torch are the LiteRT harness venv (6.6 GB, living with the data) and the training
container.

⚠⚠ THIS LINE SAID "a deliberate second copy / change both", AND THE PAIR WAS NOT A PAIR (#434).
`scripts/litert_export/run.sh` OWNS the default — it composes it from `SONORA_LITERT_WORK` —
and more than one file then spelled it out: this docstring, so the command is pasteable
rather than a shape the reader has to complete, and `REVIEWER_TORCH_PY` in the review lane's
config. Each described itself as one half of a pair with `run.sh` and neither mentioned the
other, so "change both" reached some of them and left this line naming an interpreter the
export lane no longer used — re-creating, on this exact line, the defect #428 closed.

⚠ THAT SECOND COPY IS GONE (2026-09-15, with the review lane), so this docstring and `run.sh`
really are a pair now. The lesson is kept because it was general and cost two findings: a
file calling itself "one half of a pair" without naming the other half cannot be maintained
by a reader who only has one of them.

⚠ NO SIZE IS STATED HERE — NOT "three", NOT "both", NOT "all of them" (#434, three passes).
Every sentence that said how many places hold this path was correct when written and unable to
fail afterwards: "a deliberate second copy", then "THREE PLACES", then — four lines below the
sentence removing it — "All three are now pinned". A number in prose is not a mechanism.

⚠ THE MECHANISM MOVED (2026-09-15). `_INTERPRETER_COPIES` lived in
`tests/test_request_review.py` and was deleted with the review lane, leaving the copies free
to drift — the exact state the paragraph above describes as already paid for once. It is now
`_INTERPRETER_COPIES` in `tests/test_license_wall.py`, which does the same two things:
compares every enrolled copy against `run.sh`, and scans the tracked tree for the literal so
an unenrolled copy fails rather than hides. Change `run.sh` and it names whichever copy did
not follow; add a copy without enrolling it and the completeness check says so.

The guarded import in `main()` says the interpreter part at the moment it fails, because a
corrected sentence four lines above a still-wrong command leaves the defect where people
paste from.
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    args = ap.parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        print(f"!! cannot run: {exc}. This tool loads a checkpoint, so it needs torch, and the "
              "repo .venv deliberately has none (AGENTS.md §3 — the `test` dependency group "
              "excludes it). Run it with the LiteRT harness interpreter, which "
              "scripts/litert_export/run.sh resolves as SONORA_LITERT_PY, or inside the "
              "training container. See this script's docstring.", file=sys.stderr)
        return 3
    from matcha.data.license_wall import (LicenseWallError, lineage_filelists, lineage_gaps,
                                          refuse_unpublishable)
    # ⚠ A LOAD FAILURE IS "COULD NOT RUN", NOT "REFUSED". Unguarded, `torch.load` raised
    # through and the process exited 1 — the code this tool documents as a publish refusal — so
    # a mistyped path and an artifact that must not ship were the same answer to anything
    # reading the status. Caught here rather than in the caller because the exit code is this
    # tool's whole interface: §7 is a hand checklist, and the only machine-readable thing it
    # produces is the number.
    # ⚠⚠ `Exception`, NOT A TUPLE OF TYPES, AND THE TUPLE WAS THE DEFECT (#436). The first fix
    # here listed `(OSError, RuntimeError, EOFError, ValueError)`, which is a hand-kept
    # enumeration over an open-ended set — the #247 shape, one layer down. Measured under the
    # granted interpreter: a file that exists but is not a checkpoint raises
    # `_pickle.UnpicklingError` (PickleError -> Exception), which that tuple does not name, so it
    # escaped and the process exited 1 — the code this tool documents as REFUSED. A truncated
    # checkpoint raises RuntimeError and a directory IsADirectoryError, both of which it did
    # catch, which is why the hole looked closed.
    #
    # The classification is "the load failed", not "the load failed with one of these types", so
    # the handler says that. Its blast radius is exactly one statement — `torch.load` and nothing
    # else is inside the `try` — and `BaseException` is deliberately not caught, so a
    # KeyboardInterrupt still interrupts.
    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # pylint: disable=broad-except
        # ⚠ THE INSTRUCTION, NOT ONLY THE CLASSIFICATION (#437, AGENTS.md §5). This said
        # "Check the path.", which is right for a missing path and wrong for the three other
        # causes that reach here — a garbage file, a truncated one, a directory. The exception
        # type is the only thing that knows which, so the message points at it instead of
        # guessing, and the classification it carries is the part that must not be lost.
        print(f"!! cannot run: could not read the checkpoint {args.ckpt}: "
              f"{type(exc).__name__}: {exc}. This is NOT a publish refusal — the wall never "
              "ran, so this says nothing about whether the artifact may ship. The exception "
              "above is what distinguishes a wrong path from a file that is not a loadable "
              "checkpoint.", file=sys.stderr)
        return 3
    lineage = lineage_filelists(ck)
    if lineage:
        print("lineage:")
        for p in lineage:
            print("  ", p)
    try:
        unread = refuse_unpublishable(lineage, what=args.ckpt, root=_REPO)
    except LicenseWallError as exc:
        print(f"REFUSED: {exc}")
        return 1
    # ⚠ ONE DEFINITION OF "NOT CLEAN", shared with the export path (#425, #426). This used to
    # test `not lineage` and `unread` separately here, and the export tested only the second,
    # so the two readers disagreed about what an empty lineage meant.
    gaps = lineage_gaps(lineage, unread)
    if gaps:
        for note in gaps:
            print(f"!! {note}")
        return 2
    print("publishable: no `publish: forbidden` corpus anywhere in the lineage")
    return 0


if __name__ == "__main__":
    sys.exit(main())

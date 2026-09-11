#!/usr/bin/env python
"""Run the publish wall over a Lightning checkpoint before promoting it (ARCHITECTURE §7).

The LiteRT export calls the wall itself; the registry promotion is a hand process with no
code on it, so this is the runnable form of the §7 checklist item. Exit 0 = every filelist
in the lineage was opened and nothing in it is `publish: forbidden`; 1 = refused; 2 = the
lineage is UNKNOWN or partly unread, which is not a clean result; 3 = it could not run.

    "${SONORA_LITERT_PY:-/data/toolchain/litert-conversion/.venv/bin/python}" \
        scripts/tools/check_publishable.py /path/to/checkpoint.ckpt

⚠ NOT `.venv/bin/python`, which this line and ARCHITECTURE §7 both named until #428. Loading a
checkpoint needs torch, and the repo venv deliberately has none: AGENTS.md §3 populates it from
the `test` dependency group, which excludes torch on purpose. So the §7 checklist item died at
`import torch` on the very checkout it was written from — measured. The interpreters that do
have torch are the LiteRT harness venv (6.6 GB, living with the data) and the training
container.

⚠ The default above is a deliberate second copy of the one `scripts/litert_export/run.sh`
resolves, so that this command is pasteable rather than a shape the reader has to complete;
change one, change both. `run.sh` is not a route to this script — it runs its own directory
only. The guarded import in `main()` says the same thing at the moment it fails, because a
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
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
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

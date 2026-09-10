#!/usr/bin/env python
"""Run the publish wall over a Lightning checkpoint before promoting it (ARCHITECTURE §7).

The LiteRT export calls the wall itself; the registry promotion is a hand process with no
code on it, so this is the runnable form of the §7 checklist item. Exit 0 = every filelist
in the lineage was opened and nothing in it is `publish: forbidden`; 1 = refused; 2 = the
lineage is UNKNOWN or partly unread, which is not a clean result.

    .venv/bin/python scripts/tools/check_publishable.py /path/to/checkpoint.ckpt
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
    import torch
    from matcha.data.license_wall import (LicenseWallError, lineage_filelists,
                                          refuse_unpublishable)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lineage = lineage_filelists(ck)
    if not lineage:
        print("lineage UNKNOWN: the checkpoint carries no datamodule hparams and no lineage "
              "key (pre-wall?). This is not a clean result.")
        return 2
    print("lineage:")
    for p in lineage:
        print("  ", p)
    try:
        unread = refuse_unpublishable(lineage, what=args.ckpt, root=_REPO)
    except LicenseWallError as exc:
        print(f"REFUSED: {exc}")
        return 1
    if unread:
        for p in unread:
            print(f"!! could not open {p} under {_REPO}: its audio was NOT classified")
        print("lineage partly UNKNOWN — not a clean result")
        return 2
    print("publishable: no `publish: forbidden` corpus anywhere in the lineage")
    return 0


if __name__ == "__main__":
    sys.exit(main())

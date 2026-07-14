"""The license wall: refuses to train on non-permissive or undeclared data.

Enforces the permissive/commercial boundary of north star §8.2 in code, so
CC-BY-NC sources (Expresso, original-subset Emilia) can never silently enter
the production corpus. Every dataset reachable from a training filelist must
be declared in configs/data_licenses.yaml.

Escape hatch for §7 de-risk experiments: SONORA_LICENSE_WALL=derisk permits
class-`nc` data with a loud banner — artifacts from such runs are tainted and
must never be promoted to the registry. There is deliberately no "off".
"""

import os

import yaml

_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "data_licenses.yaml",
)

_manifest_cache = None


class LicenseWallError(RuntimeError):
    pass


def _manifest():
    global _manifest_cache
    if _manifest_cache is None:
        with open(_MANIFEST_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)["datasets"]
        _manifest_cache = {}
        for name, entry in raw.items():
            for d in entry["dirs"]:
                _manifest_cache[d.lower()] = (name, entry["class"], entry["license"])
    return _manifest_cache


def classify_path(path):
    """Returns (dataset_name, class, license) or None if no component matches."""
    table = _manifest()
    for component in os.path.normpath(path).split(os.sep):
        hit = table.get(component.lower())
        if hit:
            return hit
    return None


def enforce(filelist_paths):
    """Validates filelists + every audio path inside them against the manifest.

    Raises LicenseWallError on class-`nc` data (unless SONORA_LICENSE_WALL=derisk)
    and on paths that match no declared dataset.
    """
    mode = os.environ.get("SONORA_LICENSE_WALL", "enforce")
    nc_hits, unknown = [], []
    for filelist in filelist_paths:
        seen_dirs = set()
        rows = []
        with open(filelist, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line.split("|")[0])
        for audio_path in rows:
            d = os.path.dirname(audio_path)
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
        for p in [filelist, *seen_dirs]:
            hit = classify_path(p)
            if hit is None:
                unknown.append(p)
            elif hit[1] == "nc":
                nc_hits.append((p, hit[0], hit[2]))
    if unknown:
        raise LicenseWallError(
            "License wall: undeclared dataset path(s) in training filelists — "
            f"declare them in configs/data_licenses.yaml first: {sorted(set(unknown))[:5]}"
        )
    if nc_hits:
        detail = "; ".join(f"{p} -> {name} ({lic})" for p, name, lic in nc_hits[:5])
        if mode == "derisk":
            banner = "=" * 76
            print(
                f"\n{banner}\n"
                "LICENSE WALL — NON-COMMERCIAL DE-RISK RUN\n"
                f"NC-licensed data in the corpus: {detail}\n"
                "Checkpoints/exports from this run are TAINTED: de-risk use only,\n"
                "never promote to the registry or any shipped artifact.\n"
                f"{banner}\n",
                flush=True,
            )
        else:
            raise LicenseWallError(
                f"License wall: NC-licensed data in training filelists: {detail}. "
                "NC sources are de-risk-only (north star §8.2). For a de-risk "
                "experiment set SONORA_LICENSE_WALL=derisk (taints the run)."
            )

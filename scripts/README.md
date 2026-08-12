# `scripts/` — what goes where

Until 2026-08-12 this directory was 114 files in two folders, and **"nothing invokes this"
was the normal, healthy state** for most of them. That is why `qc_verdict.py` could be named
in a `synth_bank.sh` comment for a month without running: an unwired *stage* looked exactly
like the ~80 files that were never meant to be called (issues #24, #26).

The layout is the answer to *"what is this file?"*. The enforcement is
[`pipeline_manifest.py`](pipeline_manifest.py) plus `tests/test_stage_coverage.py`.

| directory | what belongs there | how many |
|---|---|---|
| **`stages/`** | Pipeline stages, and the four orchestrator shells that drive them. Every file here is declared in `pipeline_manifest.py`, and the ratchet fails if a declared stage is not invoked *or* an invoked script is not declared. | 17 py + 4 sh |
| **`lib/`** | Imported by other non-test Python. `synth_common` (27 importers), `ref_select` (11), `book_ingest` (7)… | 10 |
| **`tools/`** | Run by hand, deliberately and repeatedly. Legitimate, and *named* so it is not mistaken for a stage. | 46 |
| **`gates/`** | Standalone gate scripts, executed as subprocesses by `tests/test_gate_scripts.py`. Named `test_*.py` and **not** collected by pytest. | 6 |
| **`assets/`** | Checked-in data the scripts read: the per-engine director skill files and the register lexicon. One home, so a reader does not resolve it relative to wherever *it* happens to live. | 4 + 9 |
| **`teacher_audition/`** | **Provenance, not a lane.** Clips these rendered are still rated in `ratings.csv`, and its README is the per-engine interface record. See the standing note at the top of that README before proposing a deletion. | 14 py + 2 sh |
| **`litert_export/`** | The on-device export lane. Self-contained, with its own `run.sh` dispatcher that takes the script name as `$1` — so static reachability cannot see it, which `pipeline_manifest.DYNAMIC_DISPATCH` declares rather than leaves as a mystery. | 12 py |

Top level holds only `pipeline_manifest.py` (the declaration), `fix_pr.sh` (a developer
tool), and the three container helpers two of the orchestrators source.

## Imports: flat, from a repo-root anchor

Every file under `scripts/<bucket>/` is **exactly two levels down**, which is deliberate —
it makes one expression correct everywhere:

```python
_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, _os.path.join(_SONORA_REPO, "scripts", "lib")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
```

It replaced 87 scattered `sys.path.insert(0, dirname(__file__))` calls, which worked only
while every sibling shared one directory. Two things about it:

* **Imports stay flat** (`import synth_common`), not `from scripts.lib import synth_common`.
  Package-qualified imports need the repo root on `sys.path` at *every* entry point, and the
  container lanes run `python /sonora/scripts/stages/synth_dia.py` with nothing but the
  script's own directory on the path. Converting them means rewriting the launch path of
  every lane that cannot be tested off-GPU. It is the remaining piece, not a done one.
* **The prologue aliases its own `os`/`sys`** (`_os`, `_sys`). `ref_select.py` imports
  `sys as _sys` and nothing else, so a prologue that assumed a bare `sys` broke on import.

⚠ **`pyproject.toml` still excludes `scripts*` from the package, deliberately.** #26 proposed
dropping it "so bucket 2 can be imported rather than path-hacked", and that reason does not
hold: implicit namespace packages already make `scripts.lib` importable whenever the repo
root is on `sys.path`, exclude or not. The exclude governs what a built wheel *ships*, and
nothing installs this repo. Dropping it would put `scripts/`, `scripts/assets/` and the rest
into `setuptools.packages.find`'s results for no gain.

## Adding something

* **A pipeline stage** → put it in `stages/`, wire it into its orchestrator, and declare it
  in `pipeline_manifest.py` **in the same commit**. See AGENTS.md §5c.
* **A shell under `scripts/`** → it must appear in exactly one of the manifest's
  `ORCHESTRATORS` / `DYNAMIC_DISPATCH` / `NOT_ORCHESTRATORS`, with the reason.
* **Checked-in data a script reads** → `assets/`, and resolve it from `_SONORA_REPO`.
  `tests/test_asset_paths.py` asserts that every in-repo path built from `__file__` actually
  points at the thing; it exists because that class fails when something *reads* the path —
  hours into a GPU render — not at import.

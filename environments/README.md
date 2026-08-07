# Recorded environments (T4 / G-1)

What actually ran, per lane. Before 2026-08-06 the answer was nowhere: `uv.lock` was a
three-line stub with no packages in it, and the venv that derived `libritts_r_vat` v1
through v3c and produced every render campaign was unrecorded. An artifact you cannot
tie to an environment is an artifact you cannot reproduce or debug.

| file | lane | has torch? |
|---|---|---|
| [`host-venv.txt`](host-venv.txt) | `.venv` — corpus derivation, synthesis, audit tooling | no, deliberately |
| [`training-container.txt`](training-container.txt) | `rocm/pytorch` — training, eval, scoring | yes, the image's ROCm build |

## Why snapshots and not a `uv.lock`

**Torch is not installable from PyPI here.** The ROCm build ships in the base image
(`torch==2.10.0+rocm7.2.4`), and a resolver asked to lock `torch>=2.0.0` would have
happily pinned a CUDA wheel — producing a lockfile that looks authoritative, installs
cleanly on a laptop, and describes an environment we have never trained in. That is worse
than the stub, because the stub was obviously empty.

So these are `uv pip freeze` output: a record of an environment that existed and did the
work, not a resolution of what `pyproject.toml` permits. `pyproject.toml` remains the
single source of truth for *constraints*; these files are the evidence of what those
constraints resolved to in practice.

The stub `uv.lock` was deleted rather than filled in for the same reason. If the project
ever moves to a `uv sync`-managed venv it should come back — but only for the host lane,
which is the one that genuinely installs from PyPI.

## Refreshing

Host — keep the header, replace the body:

```
uv pip freeze --python .venv/bin/python
```

Container — re-run the compose prep chain against the base image and freeze the result,
so the record stays a record rather than a guess. The exact chain is in the file's own
header, and it is copied verbatim from `AI-Lab-AMD/docker-compose.yml`.

Refresh both after any dependency change, and note in `notes/CHANGELOG.md` that you did —
a stale snapshot asserting the wrong versions is the failure mode these files exist to
prevent.

## Still open

Per-campaign environment capture (a freeze dropped into each render campaign dir) and
image-digest pinning for `synth_bank.sh` / `librivox_align.sh` / `eiv_score.sh` — only
qwen pins its wheels today. EIV raw scores are called immutable but carry no version
stamp (D-M2). Tracked in [`notes/todo.md`](../notes/todo.md) § 3.

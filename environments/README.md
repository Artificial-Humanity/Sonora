# Recorded environments (T4 / G-1)

What actually ran, per lane. An artifact you cannot tie to an environment is an artifact
you cannot reproduce or debug.

| file | lane | has torch? |
|---|---|---|
| [`host-venv.txt`](host-venv.txt) | `.venv` — corpus derivation, synthesis, audit tooling | no, deliberately |
| [`training-container.txt`](training-container.txt) | `rocm/pytorch` — training, eval, scoring | yes, the image's ROCm build |

## Why snapshots and not a `uv.lock`

**Torch is not installable from PyPI here.** The ROCm build ships in the base image
(`torch==2.10.0+rocm7.2.4`), and a resolver asked to lock `torch>=2.0.0` would pin a
CUDA wheel — producing a lockfile that looks authoritative, installs cleanly on a laptop,
and describes an environment we have never trained in.

So these are `uv pip freeze` output: a record of an environment that existed and did the
work, not a resolution of what `pyproject.toml` permits. `pyproject.toml` remains the
single source of truth for *constraints*; these files are the evidence of what those
constraints resolved to in practice.

There is no `uv.lock`, for the same reason. If the project ever moves to a `uv sync`-managed
venv, add one — but only for the host lane, which is the one that genuinely installs from PyPI.

## Refreshing

Host — keep the header, replace the body:

```
uv pip freeze --python .venv/bin/python
```

Container — re-run the compose prep chain against the base image and freeze the result,
so the record stays a record rather than a guess. The exact chain is in the file's own
header, and it is copied verbatim from `AI-Lab-AMD/docker-compose.yml`.

Refresh both after any dependency change, and say so in the commit message —
a stale snapshot asserting the wrong versions is the failure mode these files exist to
prevent.

## The container lanes (D-M2)

The three throwaway-container lanes — `scripts/stages/synth_bank.sh`,
`scripts/stages/librivox_align.sh`, `scripts/stages/eiv_score.sh` — source one pin table,
[`scripts/container_env.sh`](../scripts/container_env.sh):

* **The image is pinned by digest**, `rocm/pytorch@sha256:4449f856…`, the one every
  existing campaign and every existing EIV score ran under. Override with
  `SONORA_ROCM_IMAGE` to test a new base; re-pin the default from a run that passed.
* **`transformers` is held at 4.57.3**, the line the corpus was built on. By dry-run
  resolution inside the pinned image: unpinned, it resolves to a 5.x release (5.14.1 when the pin was set). All seven
  dependency sets resolve against the held version (7/7 clean) — these are checked pins,
  not hopeful ones.
* **Installs go through uv**, per AGENTS.md § 3 (`--python /opt/venv/bin/python`, never
  `--system`). `pip install uv` stays as the bootstrap, matching the training container's
  own prep chain above.
* **Every run freezes itself into the campaign** at `<campaign>/_env/<lane>.txt`, via
  [`scripts/capture_container_env.sh`](../scripts/capture_container_env.sh) — as `ai-mgr`,
  with the ROCm torch build recorded separately, since it is the one package a `pip
  freeze` cannot see. Never fatal: a missing record is a gap, but a reproducibility
  measure that destroys a finished render is worse than no record at all.

**No dependency set pulls a torch or nvidia wheel**, so the image's ROCm build survives every
install. That is the case that would matter most: a CUDA torch silently displacing it
renders nothing and explains nothing.

That is the mechanism. The evidence it produces is what the *next* refresh of the pins
should be read from — `tests/test_container_env.py` holds the rules in place.

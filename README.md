# Sonora — Expressive & Directable TTS Actor Training

Sonora is a training codebase for a directable, castable, mobile-friendly
text-to-speech actor. It uses Matcha-TTS conditional flow-matching with affect,
delivery and speaker conditioning. Prosodia consumes the exported actor artifacts.

## Start here

Read [AGENTS.md](AGENTS.md), [PERSONA.md](PERSONA.md) and [WORKFLOW.md](WORKFLOW.md)
before changing the repo.

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) defines the model contract, corpus
  rules, validation gates and artifact promotion.
* [docs/README.md](docs/README.md) indexes the binding design and policy documents.
* [scripts/README.md](scripts/README.md) describes the script layout and pipeline.
* `notes/STATE.md` and `notes/todo.md` contain current state and open work.
  `notes/quality-gap-plan.md` defines sequencing; `notes/training-operations.md`
  is the training runbook. These notes are private and are not included in public
  clones. See `AGENTS.md` for the local symlink setup.

## Workspace and storage

* Edit code in `Sonora/github`, this repository.
* Use the sibling `Sonora/huggingface` checkout for model artifacts and their registry.
* Keep datasets, caches, checkpoints and training outputs on fast storage, separate
  from source. Configure paths through [configs/paths/default.yaml](configs/paths/default.yaml)
  and the run configuration; `data/` can be a symlink to dataset storage.
* Treat `/data` source copies as deployment targets. Follow `AGENTS.md` §§6–7 for
  source locations, deployment and drift checks.

## Training environment

Run training in a GPU-enabled Docker container. The lab's ROCm service is
`sonora_training`, defined in the sibling `AI-Lab-AMD` repo. Use its compose
configuration and the private training runbook for launches and checkpoint/resume
settings.

Inside the container, navigate to the project root and install with uv:

```bash
uv pip install --no-build-isolation -e .
```

Pre-install `Cython` if needed. `--no-build-isolation` reuses the container's
PyTorch/NumPy dependencies. Follow `AGENTS.md` §3 for uv and interpreter selection.

Phonemization uses `op_g2p`: OpenPhonemizer's dictionary with a DeepPhonemizer TFLite
fallback for out-of-vocabulary words. Training and runtime share the locked IPA
vocabulary. `espeak-ng` is not required and is prohibited in the runtime path.

## Export and validation

Follow [scripts/litert_export/README.md](scripts/litert_export/README.md) for the
split-graph LiteRT export: text encoder, decoder and vocoder, with the ODE loop on
the host. Use `convert_final.py` for the 22.05 kHz baseline and `convert_vat.py` for
the 24 kHz/multi-speaker/VAT lane. The ONNX/onnx2tf monolith is the fallback path.

Run the export gates, compare per-graph and end-to-end waveform parity against
PyTorch, and verify that the target runtime loads the graphs and synthesizes audio.
Check the converter's supported conditioning width and the current export gaps in
`notes/STATE.md` and `notes/todo.md` before choosing a checkpoint. Do not infer
support for a conditioning channel from training support alone.

## License and credits

Sonora is licensed under the Apache License, Version 2.0. It builds on
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS), licensed under MIT.
See [NOTICE](NOTICE) for attribution.

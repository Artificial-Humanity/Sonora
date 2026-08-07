# Matcha-TTS → LiteRT conversion

Scripts that produce the four `.tflite` graphs used by the Android sample, from the official Matcha-TTS checkpoints, with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

`uv` is this organization's Python standard ([AGENTS.md](../../AGENTS.md) §3) — no bare `pip`:

The venv and the checkpoints live **with the data**, not in the checkout — it is ~6.6 GB:

```bash
W=/data/toolchain/litert-conversion            # $SONORA_LITERT_WORK
uv venv "$W/.venv" && uv pip install --python "$W/.venv/bin/python" \
    --no-deps matcha-tts diffusers einops conformer deep-phonemizer
uv pip install --python "$W/.venv/bin/python" litert-torch ai-edge-litert ai-edge-quantizer
# checkpoints, into $W (auto-downloaded by the matcha-tts CLI, or):
#   matcha_ljspeech.ckpt + generator_v1   (github.com/shivammehta25/Matcha-TTS-checkpoints v1.0)
#   openphonemizer_best_model.pt           (hf://openphonemizer/ckpt/best_model.pt)
```

Never `uv run` — it resolves against the repo `pyproject.toml` and ignores the inline
PEP 723 blocks these scripts carry.

## Run

`run.sh` executes a script **from here** with its data on `/data`, wiring up
`SONORA_LITERT_WORK`, `SONORA_REPO` and the harness interpreter:

```bash
scripts/litert_export/run.sh convert_final.py 512   # textenc + CFM decoder + vocoder (fp16)
scripts/litert_export/run.sh convert_g2p_matcha.py  # DeepPhonemizer G2P (fp16)
scripts/litert_export/run.sh convert_vat.py         # the 24 kHz / VAT lane
```

Outputs land in `$SONORA_LITERT_WORK/artifacts/`: `matcha_textenc_fp16.tflite`, `matcha_decoder_fp16.tflite`, `matcha_vocoder_fp16.tflite`, `dp_g2p_matcha_fp16.tflite`, plus the host tables (`emb.bin`, `g2p_dict.txt.gz`, `config.json`, `g2p_meta.json`).

`convert_vat.py` additionally writes **`g2p_contractions.json`** — the apostrophe tables, exported from `matcha.text.op_g2p` rather than transcribed, because the dictionary holds no apostrophe keys and a front end without them phonemizes `we'll` as the word *well*. A device that cannot find it must refuse to phonemize; falling back to the plain dictionary is the defect, not a degraded mode.

## Files

| File | What |
|---|---|
| `build_matcha.py` | the re-authoring recipe (GroupNorm→4D, Mish→SELECT-free softplus, ConvTranspose1d→ZeroStuffConvT1d, diffusers Attention→manual additive-masked, SinusoidalPosEmb host-side) + real-weight conversion + per-graph parity (corr 1.0). |
| `convert_final.py` | converts + fp16-quantizes the three acoustic graphs; end-to-end waveform parity. |
| `convert_g2p_matcha.py` | converts the DeepPhonemizer (espeak-IPA) G2P to a fixed `[1,96]` graph. |
| `device_g2p.py` | the **device text front end** — an independent replica of `matcha.text.op_g2p`, and the spec the mobile port has to meet. Gate G7 holds it to phoneme-string parity with the training front end. |
| `e2e_masked.py`, `e2e_matcha.py` | end-to-end host-orchestration parity (pad-to-max + runtime mask). |
| `kotlin_replica.py` | replicates the exact Android host logic in Python (validates the Kotlin port). |
| `probe_tx_standalone.py`, `probe_decoder_taps.py` | the on-device bisection that isolated the Mali ML Drift transformer-fusion bug (decoder → CPU). |

## Re-authoring → GPU-clean

Every graph converts GPU-clean (per-graph tflite-vs-torch corr **1.000000**; end-to-end waveform corr ≥0.99). Fixed shapes (256 phonemes, 512 mel frames) with a runtime float mask let one compiled graph handle any length. See `build_matcha.py` for the op-by-op recipe.

⚠ **`build_matcha.py` is the 22.05 kHz Phase-0 lane.** The 24 kHz / multi-speaker / VAT
converter is `convert_vat.py`, **here, in this directory** — it was migrated in on
2026-08-04 and this sentence claimed otherwise until 2026-08-06. Before trusting either,
read [notes/todo.md](../../notes/todo.md) §2: the gate suite prints PASS/FAIL and **exits 0
regardless**, and valence/tension have never been driven nonzero through a converted graph.

## Provenance, and which copy is authoritative

Migrated into source control 2026-07-22 from `/data/toolchain/litert-conversion/` on
ai-lab-0, where the **working directory** (venv, checkpoints, logs, `artifacts*/` outputs)
remains. These scripts produced the LiteRT/TFLite exports published in
`artificial-humanity/Sonora` → `baseline-ljspeech-22k/` (end-to-end graphs and the
`litert-split/` mobile lane) and the VAT-ready split export in `derisk-energy-24k/`.

**Run these from here, with `run.sh`:**

```
scripts/litert_export/run.sh convert_vat.py
scripts/litert_export/run.sh build_matcha.py parity
```

`SONORA_LITERT_WORK` (default `/data/toolchain/litert-conversion`) is where the **data**
lives — the checkpoints read, and the `.tflite`/`.wav`/`artifacts*/` written. It defaults
to this directory when unset, so an old invocation still behaves as it always did.

That split is what let the `/data` **code** copy be retired rather than merely policed.
The migration had left the scripts there too, and by 2026-08-06 two had drifted:
`convert_vat.py` was three weeks stale and missing the `detect_vat_dim` seam guard this
repo recorded as landed, and the README still documented a bare-`pip` install predating
the uv standard. Nothing detected either — both directories looked healthy, and the only
symptom was that a fix believed to have shipped had not.

Owner principle (AGENTS.md §6): code executes from the repo; `/data` holds what its name
implies. `tests/test_data_mirrors.py::test_litert_harness_has_no_code_copy` fails if a
`.py` reappears on `/data`.

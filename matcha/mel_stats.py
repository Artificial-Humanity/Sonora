"""Which mel statistics a checkpoint's outputs are in. Torch-free, so the rule is testable here.

⚠⚠ A CHECKPOINT CARRIES TWO COPIES, AND ON A WARM START THEY DISAGREE (found 2026-09-22).

    hyper_parameters["data_statistics"]      the config of the run that trained it
    state_dict["mel_mean"], ["mel_std"]      buffers — what `synthesise` denormalises with

`update_data_statistics` sets the buffers from the config in `__init__`, and then
`load_state_dict` OVERWRITES them with whatever the loaded checkpoint holds. A fine-tune
warm-started from a donor therefore keeps the DONOR's buffers for its whole life: training
never reads them (the datamodule normalises with the config), so nothing goes wrong until
inference. `vat7_finetune` trained on mean -5.544 / std 2.430 and carried -6.631 / 2.483,
which put every render's mel 1.09 log units (9.4 dB) below any mel the vocoder was trained
on — measured on 64 aligned renders against the real mels of the same clips, uniform across
all 80 bins.

The model's outputs are in the space its training data was normalised into, so the config's
statistics are the right ones. The buffers are replaced on load, not trusted.

⚠ ON LOAD MEANS THROUGH ONE OF TWO DOORS. The Lightning hook covers `load_from_checkpoint`
and a trainer `ckpt_path`. A bare `model.load_state_dict(sd)` skips it, so every such path
that turns a mel back into audio calls `correct_state_dict` itself: measure_harmonicity.py,
render_vat_sweep.py and render_guidance_demo.py. render_aligned_mels.py denormalises with
the config explicitly instead. Paths that only compare NORMALISED mels (score_holdout.py,
measure_synth_mel_error.py) and test_vat_identity.py, which compares two renders of one
load, do not read the buffers in a way that matters.
"""


def corrected_mel_buffers(hparams, state_dict):
    """The buffer values `state_dict` should carry, or {} when there is nothing to correct.

    `hparams` is the LOADING module's own — on `load_from_checkpoint` that is the
    checkpoint's config, and on a warm start (`ckpt_path=<donor>`) it is the NEW run's,
    which is the corpus the run is about to normalise with. Reading the donor's
    `hyper_parameters` instead would reproduce the bug on every warm start.
    """
    stats = (hparams or {}).get("data_statistics")
    if not stats or state_dict is None or not {"mel_mean", "mel_std"} <= set(state_dict):
        return {}
    want = {"mel_mean": float(stats["mel_mean"]), "mel_std": float(stats["mel_std"])}
    have = {k: float(state_dict[k]) for k in want if k in state_dict}
    if all(abs(have.get(k, float("nan")) - v) < 1e-6 for k, v in want.items()):
        return {}
    return want


def correct_state_dict(hparams, state_dict):
    """Apply `corrected_mel_buffers` to `state_dict` in place, before `load_state_dict`.

    Returns what was replaced, as {name: (old, new)}, so a caller can say so. The new value
    keeps the old tensor's dtype and device via `new_tensor`; a plain float stays a float,
    which is what lets the rule be tested without torch.
    """
    fix = corrected_mel_buffers(hparams, state_dict)
    changed = {}
    for k, v in fix.items():
        old = state_dict[k]
        changed[k] = (float(old), v)
        state_dict[k] = old.new_tensor(v) if hasattr(old, "new_tensor") else v
    return changed

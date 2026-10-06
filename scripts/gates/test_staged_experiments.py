"""Contract gate for the staged rebuild's four experiments.

IN-CONTAINER (needs torch, hydra and /data; CPU only):
    scripts/stages/run_in_rocm.sh scripts/gates/test_staged_experiments.py

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md. Every new arm is a
10-epoch fine-tune from stock that differs from it ONLY in the named changes. This proves
the configs say so before a GPU hour is spent:

1. Each experiment composes to its arm: corpus, sample rate, mel ceiling, speakers, VAT on
   or off, 10 epochs, and a run name equal to the experiment name (the launcher resumes
   from logs/train/<experiment>/).
2. R's model hyper-parameters equal derisk-energy ep099's on every key both record.
3. C0, S1 and S2 share R's encoder, decoder, CFM and optimizer settings.
4. Each warm start loads into its composed model with strict=True, at epoch 0 and step 0.
5. Each datamodule yields a first training batch (80 mel bins; R also a 3-wide VAT).
6. R's composed `data`, `model` and `trainer` sections equal the July derisk run's own
   `.hydra/config.yaml` on every key both carry, except the keys in `ALLOWED`. Every arm uses
   July's length-blind shuffle (`data.bucket_multiplier: 0`). Model hparams (check 2) are
   not the whole recipe; batching changed after July.
"""
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import functools  # noqa: E402

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, ListConfig, OmegaConf  # noqa: E402

DATA = "/data/model-training/sonora/data"
# Read from the environment so tests/test_gate_scripts.py's SLOW probe names a variable this
# script actually uses.
WS = _os.environ.get("SONORA_STAGED_WARMSTART_DIR", "/data/model-training/sonora/warmstart")
DERISK_RUN = "/data/model-training/sonora/logs/train/derisk_energy/runs/2026-07-15_00-20-31"
DERISK = DERISK_RUN + "/checkpoints/checkpoint_epoch=099.ckpt"
# Keys of the July config R may differ on. max_epochs: July ran open-ended (-1) and was
# stopped by hand; R is 10. The other two are loader plumbing and do not change what is
# learned. Any other difference is a recipe change and stops the gate.
ALLOWED = {"trainer.max_epochs", "data.num_workers", "data.pin_memory"}
# experiment: (data name, sample rate, f_max, n_spks, use_vat, warm start)
ARMS = {
    "staged_r": ("libritts_r_vat", 24000, 12000, 247, True, WS + "/derisk_energy_init.ckpt"),
    "staged_c0": ("vctk_22k", 22050, 8000, 109, False, WS + "/staged_c0_init.ckpt"),
    "staged_s1": ("libritts_r_22k", 22050, 8000, 247, False, WS + "/staged_s1_init.ckpt"),
    "staged_s2": ("libritts_r_vat", 24000, 12000, 247, False, WS + "/staged_s2_init.ckpt"),
}
SHARED = ("encoder", "decoder", "cfm", "optimizer")


def plain(v):
    # Hydra builds the optimizer as a functools.partial, and partials compare by identity,
    # so two identical recipes would never compare equal. Compare what it names instead.
    if isinstance(v, functools.partial):
        return ("partial", f"{v.func.__module__}.{v.func.__qualname__}",
                tuple(plain(a) for a in v.args), {k: plain(x) for k, x in v.keywords.items()})
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (DictConfig, ListConfig)):
        return OmegaConf.to_container(v, resolve=True)
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    return v


def composed(exp):
    with initialize_config_dir(version_base="1.3",
                               config_dir=_os.path.join(_SONORA_REPO, "configs")):
        return compose(config_name="train.yaml", overrides=[f"experiment={exp}"])


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def july_diff(cfg):
    """[(key, July value, R value)] over data/model/trainer keys both configs carry."""
    july = OmegaConf.to_container(OmegaConf.load(DERISK_RUN + "/.hydra/config.yaml"),
                                  resolve=False)
    now = OmegaConf.to_container(cfg, resolve=False)
    out = []
    for sec in ("data", "model", "trainer"):
        a, b = flat(july.get(sec) or {}, sec + "."), flat(now.get(sec) or {}, sec + ".")
        out += [(k, a[k], b[k]) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
        new = sorted(set(b) - set(a))
        if new:
            print(f"  note  {sec}: keys new since July (no July value to compare): {new}")
    return out


def check_arm(exp):
    name, sr, fmax, n_spks, use_vat, warm = ARMS[exp]
    cfg = composed(exp)
    assert cfg.data.get("bucket_multiplier") == 0, f"{exp}: batching is not July's shuffle"
    if exp == "staged_r":
        diff = [d for d in july_diff(cfg) if d[0] not in ALLOWED]
        assert not diff, f"R's recipe differs from the July derisk run: {diff}"
        print("  ok  R's data/model/trainer config matches July's outside", sorted(ALLOWED))
    got = (cfg.data.name, cfg.data.sample_rate, cfg.data.f_max, cfg.data.n_spks,
           bool(cfg.model.use_vat), cfg.trainer.max_epochs, cfg.run_name)
    want = (name, sr, fmax, n_spks, use_vat, 10, exp)
    assert got == want, f"{exp}: composed {got}, pre-registered {want}"
    model = instantiate(cfg.model)
    w = torch.load(warm, map_location="cpu", weights_only=False)
    assert (w["epoch"], w["global_step"]) == (0, 0), f"{warm} is not an epoch-0 warm start"
    model.load_state_dict(w["state_dict"], strict=True)
    dm = instantiate(cfg.data, num_workers=0, batch_size=2,
                     train_filelist_path=cfg.data.train_filelist_path.replace("data/", DATA + "/", 1),
                     valid_filelist_path=cfg.data.valid_filelist_path.replace("data/", DATA + "/", 1))
    dm.setup()
    b = next(iter(dm.train_dataloader()))
    assert b["y"].shape[1] == 80, f"{exp}: mel has {b['y'].shape[1]} bins"
    if use_vat:
        assert b["vat"] is not None and b["vat"].shape[1] == 3, f"{exp}: VAT {b['vat']}"
    else:
        assert b["vat"] is None, f"{exp}: a VAT-free arm loaded VAT"
    print(f"  ok  {exp}: {name} {sr} Hz fmax {fmax}, {n_spks} speakers, vat={use_vat}, "
          f"warm start {warm.rsplit('/', 1)[-1]} loads strict")
    return {k: plain(v) for k, v in model.hparams.items()}


def main():
    hp = {exp: check_arm(exp) for exp in ARMS}
    d = torch.load(DERISK, map_location="cpu", weights_only=False)["hyper_parameters"]
    diff = [k for k in set(d) & set(hp["staged_r"]) if plain(d[k]) != hp["staged_r"][k]]
    assert not diff, f"R differs from derisk ep099 on {sorted(diff)}"
    print(f"  ok  R matches derisk ep099 on {len(set(d) & set(hp['staged_r']))} hparams")
    for exp in ("staged_c0", "staged_s1", "staged_s2"):
        diff = [k for k in SHARED if hp[exp][k] != hp["staged_r"][k]]
        assert not diff, f"{exp} differs from R on {diff}"
    print("  ok  C0, S1, S2 share R's encoder, decoder, CFM and optimizer")
    print("STAGED EXPERIMENTS: PASS")


if __name__ == "__main__":
    main()

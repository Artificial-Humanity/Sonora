"""Contract gates for Decoder v2, the DiT estimator (matcha/models/components/dit_decoder.py).

IN-CONTAINER (needs torch):  scripts/stages/run_in_rocm.sh scripts/gates/test_dit_decoder.py

1. Shape and mask: output is [B, 80, T] and exactly zero on padded frames.
2. adaLN-Zero: every block is the identity at init.
3. Padding invariance: a clip's valid frames come out the same alone and padded in a batch.
   This is what proves the key mask, the conv masking and the long skips agree; a leak
   through any of them shows here as a difference, and in training as a model that learns
   the batch's padding.
4. VAT zero-init: at init, VAT changes nothing (the CFG/dropout neutral); once the
   projection is non-zero, VAT changes the output (the plumbing is connected).
5. CFM selection: `type: dit` builds a DiTDecoder, no `type` builds the U-Net (every saved
   checkpoint's hparams), and an unknown type refuses.
6. The experiment composes, and the full model trains one step and synthesises, finite.
7. A fixed-shape `torch.export` trace succeeds. This is NOT the split-graph export gate,
   which is an adoption gate of its own. It only proves nothing here blocks tracing.
8. ON THE GPU, UNDER THE RUN'S fp16 AUTOCAST (skipped without a GPU): padding invariance
   again, and one full-size training step of the real config at batch 32 x 2,064 frames
   (the 22 s ceiling) with finite loss and grads, and its peak memory. Gates 1-7 run in
   fp32 on the CPU, which says nothing about the ROCm attention kernel, fp16 or memory
   (review, 2026-09-26).
"""
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Gate 8 runs under the kernel-search mode the run itself uses (configs/experiment/
# vat7_dit_spike.yaml), and a cold default-mode search would take minutes per shape here.
_os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import torch
from omegaconf import OmegaConf

from matcha.models.components.decoder import Decoder
from matcha.models.components.dit_decoder import DiTConvBlock, DiTDecoder
from matcha.models.components.flow_matching import CFM

_FAILURES = []


def check(name, ok, detail=""):
    if not ok:
        _FAILURES.append(name)
    print(f"{name}: {detail}  {'PASS' if ok else 'FAIL'}")
    return ok


torch.manual_seed(0)
N_FEATS, SPK, VAT = 80, 64, 8


def make(vat=True):
    return DiTDecoder(in_channels=2 * N_FEATS, out_channels=N_FEATS, hidden_channels=64,
                      filter_channels=128, n_layers=4, n_heads=4, dropout=0.0,
                      spk_emb_dim=SPK, vat_dim=VAT, vat_cond_dim=32 if vat else 0).eval()


def randomise_zero_inits(dec):
    """Give the zero-init heads real weights, so the blocks are not the identity."""
    with torch.no_grad():
        for m in dec.modules():
            if isinstance(m, DiTConvBlock):
                m.ada_ln.weight.normal_(0, 0.05)
                m.ada_ln.bias.normal_(0, 0.05)
        if dec.vat_proj is not None:
            dec.vat_proj.weight.normal_(0, 0.05)


# 1. shape + mask
dec = make()
randomise_zero_inits(dec)
B, T = 3, 50
lengths = torch.tensor([50, 37, 12])
mask = (torch.arange(T)[None] < lengths[:, None]).float().unsqueeze(1)
x, mu = torch.randn(B, N_FEATS, T), torch.randn(B, N_FEATS, T)
spks, cond, t = torch.randn(B, SPK), torch.randn(B, VAT, T), torch.rand(B)
with torch.no_grad():
    out = dec(x, mask, mu, t, spks, cond)
check("1 shape", tuple(out.shape) == (B, N_FEATS, T), f"{tuple(out.shape)}")
check("1 padded frames are zero", float((out * (1 - mask)).abs().max()) == 0.0)

# 2. adaLN-Zero identity
blk = DiTConvBlock(64, 128, 4, p_dropout=0.0).eval()
h = torch.randn(2, 64, 20)
m2 = torch.ones(2, 1, 20)
with torch.no_grad():
    y = blk(h, torch.randn(2, 64, 20), m2, m2.bool().unsqueeze(1))
check("2 block is identity at init", torch.equal(y, h), f"max|diff| {float((y - h).abs().max()):.2e}")

# 3. padding invariance: clip 2 (12 frames) alone vs inside the padded batch
with torch.no_grad():
    alone = dec(x[2:3, :, :12], mask[2:3, :, :12], mu[2:3, :, :12], t[2:3], spks[2:3], cond[2:3, :, :12])
diff = float((alone - out[2:3, :, :12]).abs().max())
check("3 padding invariance", diff < 1e-4, f"max|diff| {diff:.2e}")

# 4. VAT zero-init, then connected
fresh = make()
with torch.no_grad():
    a = fresh(x, mask, mu, t, spks, cond)
    b = fresh(x, mask, mu, t, spks, None)
check("4 VAT inert at init (adaLN-Zero: no block reads c yet)", torch.equal(a, b))
randomise_zero_inits(fresh)
with torch.no_grad():
    a = fresh(x, mask, mu, t, spks, cond)
    b = fresh(x, mask, mu, t, spks, None)
check("4 VAT connected once trained", float((a - b).abs().max()) > 1e-3,
      f"max|diff| {float((a - b).abs().max()):.2e}")

# 5. CFM selection
cfm_params = OmegaConf.create({"name": "CFM", "solver": "euler", "sigma_min": 1e-4})
dit_params = {"type": "dit", "hidden_channels": 64, "filter_channels": 128, "n_layers": 2,
              "n_heads": 4, "kernel_size": 3, "dropout": 0.0, "use_lsc": True}
unet_params = {"channels": [256, 256], "dropout": 0.05, "attention_head_dim": 64,
               "n_blocks": 1, "num_mid_blocks": 2, "num_heads": 2, "act_fn": "snakebeta"}
c1 = CFM(2 * N_FEATS, N_FEATS, cfm_params, OmegaConf.create(dit_params), n_spks=10, spk_emb_dim=SPK,
         use_vat=True, vat_dim=VAT)
c2 = CFM(2 * N_FEATS, N_FEATS, cfm_params, unet_params, n_spks=10, spk_emb_dim=SPK,
         use_vat=True, vat_dim=VAT)
check("5 type: dit -> DiTDecoder", isinstance(c1.estimator, DiTDecoder))
check("5 no type -> U-Net", isinstance(c2.estimator, Decoder) and not isinstance(c2.estimator, DiTDecoder))
try:
    CFM(2 * N_FEATS, N_FEATS, cfm_params, {"type": "wavenet"}, n_spks=10, spk_emb_dim=SPK)
    check("5 unknown type refuses", False)
except ValueError:
    check("5 unknown type refuses", True)
check("5 DiT speaker enters conditioning, not input",
      c1.estimator.in_proj.in_channels == N_FEATS + 64 and c1.estimator.spk_proj.in_features == SPK,
      f"in_proj in={c1.estimator.in_proj.in_channels}")

# 6. the experiment composes; one training step and a synthesis are finite
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

with initialize_config_dir(version_base="1.3", config_dir=_os.path.join(_SONORA_REPO, "configs")):
    cfg = compose(config_name="train.yaml", overrides=["experiment=vat7_dit_spike"])
check("6 experiment selects the DiT", cfg.model.decoder.type == "dit", f"decoder={dict(cfg.model.decoder)}")
check("6 experiment keeps VAT on", bool(cfg.model.use_vat))
# The end must land on the launcher's checkpoint cadence (3,506), or restarts loop.
check("6 experiment ends on its own, on a checkpoint",
      cfg.trainer.get("max_steps", -1) == 105180 and 105180 % 3506 == 0,
      f"max_steps={cfg.trainer.get('max_steps')}")
check("6 experiment sets FAST kernel search", cfg.get("miopen_find_mode") == "FAST")
model = instantiate(cfg.model)
est = model.decoder.estimator
n_dec = sum(p.numel() for p in est.parameters())
n_all = sum(p.numel() for p in model.parameters())
check("6 full model builds a DiT", isinstance(est, DiTDecoder),
      f"decoder {n_dec / 1e6:.1f}M of {n_all / 1e6:.1f}M params")
model.train()
bx = torch.randint(1, 170, (2, 30))
bxl = torch.tensor([30, 22])
by = torch.randn(2, N_FEATS, 120)
byl = torch.tensor([120, 90])
bspk = torch.tensor([0, 5])
bvat = torch.zeros(2, cfg.model.vat_dim, 30)
dur, prior, diff_loss, _ = model(bx, bxl, by, byl, spks=bspk, vat=bvat)
loss = dur + prior + diff_loss
loss.backward()
grads = [p.grad for p in est.parameters() if p.grad is not None]
check("6 training step finite", bool(torch.isfinite(loss)), f"loss {float(loss):.3f}")
check("6 decoder receives gradient", len(grads) > 0 and all(torch.isfinite(g).all() for g in grads),
      f"{len(grads)} tensors")
model.eval()
res = model.synthesise(bx, bxl, n_timesteps=4, temperature=0.667, spks=bspk, vat=bvat)
check("6 synthesis finite", bool(torch.isfinite(res["mel"]).all()), f"mel {tuple(res['mel'].shape)}")

# 6b. batch of one (compute_loss's `t.squeeze()` is 0-d) and scalar t at B>1 without speakers
model.train()
d1, p1, f1, _ = model(bx[:1], bxl[:1], by[:1], byl[:1], spks=bspk[:1], vat=bvat[:1])
check("6b training step at B=1 finite", bool(torch.isfinite(d1 + p1 + f1).all()))
nospk = make()
randomise_zero_inits(nospk)
nospk.spk_proj = None
with torch.no_grad():
    o = nospk(x, mask, mu, torch.tensor(0.3), None, cond)
check("6b scalar t, B=3, no speaker", tuple(o.shape) == (B, N_FEATS, T) and bool(torch.isfinite(o).all()))

# 7. fixed-shape trace
try:
    ep = torch.export.export(dec, (x[:1], mask[:1], mu[:1], t[:1], spks[:1], cond[:1]))
    got = ep.module()(x[:1], mask[:1], mu[:1], t[:1], spks[:1], cond[:1])
    d7 = float((got - out[:1]).abs().max())
    check("7 torch.export trace", d7 < 1e-4, f"max|diff| vs eager {d7:.2e}")
except Exception as exc:  # noqa: BLE001
    check("7 torch.export trace", False, f"{type(exc).__name__}: {exc}")

# 8. GPU, fp16 autocast, full size
if torch.cuda.is_available():
    dev = torch.device("cuda")
    g = dec.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        full = g(x.to(dev), mask.to(dev), mu.to(dev), t.to(dev), spks.to(dev), cond.to(dev))
        one = g(x[2:3, :, :12].to(dev), mask[2:3, :, :12].to(dev), mu[2:3, :, :12].to(dev),
                t[2:3].to(dev), spks[2:3].to(dev), cond[2:3, :, :12].to(dev))
    d8 = float((one.float() - full[2:3, :, :12].float()).abs().max())
    check("8 GPU fp16 padding invariance", d8 < 2e-2, f"max|diff| {d8:.2e}")

    big = instantiate(cfg.model)
    # Live gates: at init every adaLN gate is 0, so attention and the FFN would carry no
    # gradient and the step would test nothing about their fp16 backward (review).
    randomise_zero_inits(big.decoder.estimator)
    big = big.to(dev).train()
    opt = torch.optim.Adam(big.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    BB, TT, TX = 32, 2064, 260
    torch.cuda.reset_peak_memory_stats()
    gx = torch.randint(1, 170, (BB, TX), device=dev)
    gxl = torch.full((BB,), TX, device=dev)
    gxl[1::2] = TX // 2
    gy = torch.randn(BB, N_FEATS, TT, device=dev)
    gyl = torch.full((BB,), TT, device=dev)
    gyl[1::2] = TT // 2
    gspk = torch.randint(0, cfg.model.n_spks, (BB,), device=dev)
    gvat = torch.zeros(BB, cfg.model.vat_dim, TX, device=dev)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            d, p, f, _ = big(gx, gxl, gy, gyl, spks=gspk, vat=gvat)
            gl = d + p + f
        scaler.scale(gl).backward()
        scaler.unscale_(opt)
        gg = [q.grad for q in big.decoder.estimator.parameters() if q.grad is not None]
        ok = bool(torch.isfinite(gl)) and all(torch.isfinite(q).all() for q in gg)
        peak = torch.cuda.max_memory_allocated() / 2**30
        check("8 GPU fp16 full-size step finite", ok,
              f"B={BB} T={TT} loss {float(gl):.3f}, peak {peak:.1f} GiB")
    except Exception as exc:  # noqa: BLE001
        check("8 GPU fp16 full-size step finite", False, f"{type(exc).__name__}: {exc}")
else:
    print("8 GPU gates: SKIPPED (no GPU)")

print()
if _FAILURES:
    print(f"FAILED: {', '.join(_FAILURES)}")
    raise SystemExit(1)
print("all DiT decoder gates PASS")

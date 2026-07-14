"""Identity-at-init + smoke tests for the VAT FiLM conditioning code.

1. Warm-start identity: a use_vat model loaded from the Phase-0 checkpoint
   (strict=False) with vat=zeros must produce bit-identical synthesise()
   output to the unmodified model (zero-init FiLM guarantee).
2. Non-zero VAT must actually change the output (the plumbing is connected).
3. Training forward must run with per-utterance vat + cond dropout and
   produce finite losses, including through the out_size cut path.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

# Any plain (non-VAT) trained checkpoint works as the warm-start reference.
CKPT = os.environ.get(
    "SONORA_PHASE0_CKPT",
    "/data/model-training/sonora/logs/train/ljspeech/runs/2026-07-11_04-02-23/checkpoints/checkpoint_epoch=199.ckpt",
)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
hp = dict(ckpt["hyper_parameters"])
sd = ckpt["state_dict"]
print("hparams keys:", sorted(hp.keys()))

from matcha.models.matcha_tts import MatchaTTS

base_kwargs = {k: v for k, v in hp.items()}


def build(use_vat):
    kwargs = dict(base_kwargs)
    kwargs.update(use_vat=use_vat)
    model = MatchaTTS(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    if use_vat:
        assert all("film" in m or "vat_trunk" in m for m in missing), \
            f"non-FiLM missing keys: {[m for m in missing if 'film' not in m and 'vat_trunk' not in m][:5]}"
        print(f"  vat model: {len(missing)} new FiLM/trunk tensors, all others loaded")
    else:
        assert not missing, f"missing keys: {missing[:5]}"
    model.eval()
    return model


base = build(False)
vat_model = build(True)

ids = torch.tensor([[0, 45, 0, 32, 0, 61, 0, 12, 0, 51, 0, 44, 0, 4, 0]], dtype=torch.long)
lengths = torch.tensor([ids.shape[-1]])

torch.manual_seed(42)
out_base = base.synthesise(ids, lengths, n_timesteps=4, temperature=0.667)
torch.manual_seed(42)
out_zero = vat_model.synthesise(ids, lengths, n_timesteps=4, temperature=0.667, vat=torch.zeros(1, 3))
torch.manual_seed(42)
out_none = vat_model.synthesise(ids, lengths, n_timesteps=4, temperature=0.667)
torch.manual_seed(42)
out_hot = vat_model.synthesise(ids, lengths, n_timesteps=4, temperature=0.667,
                               vat=torch.tensor([[0.8, -0.5, 0.9]]))

d_zero = (out_base["decoder_outputs"] - out_zero["decoder_outputs"]).abs().max().item()
d_none = (out_base["decoder_outputs"] - out_none["decoder_outputs"]).abs().max().item()
d_hot = (out_base["decoder_outputs"] - out_hot["decoder_outputs"]).abs().max().item()
print(f"identity (vat=0):    max|diff| = {d_zero:.3e}  {'PASS' if d_zero == 0 else 'FAIL'}")
print(f"identity (vat=None): max|diff| = {d_none:.3e}  {'PASS' if d_none == 0 else 'FAIL'}")
print(f"identity (vat hot):  max|diff| = {d_hot:.3e}  {'PASS (zero-init: hot vat inert at init)' if d_hot == 0 else 'FAIL'}")

# Plumbing: nudge one decoder FiLM head — hot vat must now change the output.
with torch.no_grad():
    vat_model.decoder.estimator.mid_films[0].head.weight.add_(0.01)
torch.manual_seed(42)
out_nudge = vat_model.synthesise(ids, lengths, n_timesteps=4, temperature=0.667,
                                 vat=torch.tensor([[0.8, -0.5, 0.9]]))
d_nudge = (out_base["decoder_outputs"] - out_nudge["decoder_outputs"]).abs().max().item()
print(f"plumbing (nudged head, vat hot): max|diff| = {d_nudge:.3e}  {'PASS' if d_nudge > 0 else 'FAIL'}")
with torch.no_grad():
    vat_model.decoder.estimator.mid_films[0].head.weight.sub_(0.01)

# Per-token vat path.
torch.manual_seed(42)
vat_tok = torch.zeros(1, 3, ids.shape[-1])
vat_tok[0, :, 5:9] = 0.7
out_tok = vat_model.synthesise(ids, lengths, n_timesteps=4, temperature=0.667, vat=vat_tok)
print("per-token vat synthesise: OK, frames =", int(out_tok["mel_lengths"][0]))

# Training forward with cond dropout + out_size cut. Fresh instance: the
# rotary-PE cache from inference_mode synthesise() can't be reused in autograd.
vat_model = build(True)
vat_model.train()
y = torch.randn(2, 80, 172)
y_lengths = torch.tensor([172, 144])
x2 = ids.repeat(2, 1)
x2_lengths = lengths.repeat(2)
vat_b = torch.tensor([[0.5, -0.2, 0.1], [-0.3, 0.6, -0.8]])
dur_loss, prior_loss, diff_loss, _ = vat_model(
    x2, x2_lengths, y, y_lengths, out_size=100, vat=vat_b)
ok = all(torch.isfinite(loss).item() for loss in (dur_loss, prior_loss, diff_loss))
print(f"training forward (out_size cut + dropout): dur={dur_loss:.3f} prior={prior_loss:.3f} "
      f"diff={diff_loss:.3f}  {'PASS' if ok else 'FAIL'}")

# Gradient reaches the FiLM heads (after a backward, heads must have grads).
(dur_loss + prior_loss + diff_loss).backward()
head_grads = [p.grad.abs().sum().item() for n, p in vat_model.named_parameters()
              if "film" in n and p.grad is not None]
print(f"FiLM heads with grads: {len(head_grads)}, nonzero grad sum: "
      f"{sum(1 for g in head_grads if g > 0)}")

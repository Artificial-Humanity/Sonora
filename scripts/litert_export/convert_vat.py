# Copyright 2026 Artificial Humanity.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sonora derisk-energy (24 kHz / multi-speaker / VAT) -> LiteRT split graphs.

Extends the proven build_matcha.py recipe to the derisk_energy checkpoint:
  sonora_textenc_fp16.tflite  in: emb(1,Tt,192), tmask(1,1,Tt), spk(1,64),
                                  vat(1,3,Tt)      out: mu(1,80,Tt), logw
  sonora_decoder_fp16.tflite  in: x,mu(1,80,Tm), t_emb(1,224), ymask(1,1,Tm),
                                  spk(1,64), vat_y(1,3,Tm)   out: v(1,80,Tm)
  sonora_vocoder24k_fp16.tflite  in: mel(1,80,Tm) -> wav(1,1,Tm*256) @ 24 kHz
plus host tables: emb.{npy,bin} (178x192), spk_emb.{npy,bin} (247x64),
config.json (symbols, shapes, mel stats, sr 24000, vat_dim).

Model code comes from the Sonora fork (FiLM/VATTrunk), NOT the stock
matcha-tts package. Vocoder is the fine-tuned 24k HiFi-GAN (g_02510000) —
architecturally HiFi-GAN v1, so the ZeroStuffConvT1d swap applies unchanged.

Gates (all must pass):
  G1 wrapper-vs-model: full-length masked wrapper == the fork's own
     TextEncoder.forward / Decoder.forward (proves the orchestration
     replicates the real model, FiLM paths included)
  G2 per-graph: fp16 tflite vs re-authored torch, corr ~ 1.0
  G3 e2e: tflite host pipeline vs torch host pipeline, waveform corr
  G4 energy monotonicity through the full tflite pipeline (RMS dB at
     vat -1/0/+1) + sample wavs for listening

Run: .venv/bin/python convert_vat.py [MAX_MEL]
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import json
import math
import os
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn as nn

import build_matcha as B  # noqa: E402  (stubs matcha.utils at import)
from e2e_matcha import generate_path, sequence_mask  # noqa: E402

# _stub's scipy fakes are for macOS probes; on this box real scipy works and
# the fakes actively break transformers->sklearn->scipy.stats (no milp in the
# stubbed scipy.optimize). Drop them so the real modules load on demand.
for _m in ("scipy.optimize", "scipy.sparse.linalg._propack"):
    sys.modules.pop(_m, None)

# diffusers.models.lora imports transformers CLIP at module load, which drags
# in torchao and (flakily) dies on duplicate op registration. Sonora's
# transformer.py only wants LoRACompatibleLinear from it — an nn.Linear with
# an ignored lora `scale` arg — so pre-register a faithful stand-in and the
# real module (and all of transformers) never loads.
_lora = types.ModuleType("diffusers.models.lora")


class _LoRACompatibleLinear(nn.Linear):
    def forward(self, hidden_states, scale=1.0):  # noqa: ARG002
        return super().forward(hidden_states)


_lora.LoRACompatibleLinear = _LoRACompatibleLinear
sys.modules["diffusers.models.lora"] = _lora

SONORA = os.environ.get(
    "SONORA_REPO", "/home/lmcfarlin/Projects/Artificial-Humanity/Sonora")
sys.path.insert(0, SONORA)

# Registry copy replaced by safetensors 2026-07-16 (HF picklescan); the full
# Lightning checkpoint lives only in the training-run logs now.
CKPT = os.environ.get(
    "SONORA_VAT_CKPT",
    "/data/model-training/sonora/logs/train/derisk_energy/runs/"
    "2026-07-15_00-20-31/checkpoints/checkpoint_epoch=099.ckpt")
VOC_CKPT = os.environ.get(
    "SONORA_VOC24K", "/data/model-training/vocoder/cp_hifigan_24k/g_02510000")
VOC_CFG = os.environ.get(
    "SONORA_VOC24K_CFG",
    "/data/model-training/vocoder/hifi-gan/config_24k_80band.json")
VAL_FILELIST = os.path.join(SONORA, "data/libritts_r_vat/val_op.txt")

ART = os.path.join(B.HERE, "artifacts_vat")
MAX_TEXT = 256
MAX_MEL = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_TIMESTEPS = 10
LENGTH_SCALE = 1.0  # the derisk gate's setting (not LJSpeech's 0.95)
SR = 24000
N_SPKS, SPK_DIM, VAT_DIM, VAT_COND = 247, 64, 3, 256
# Multi-speaker widens the U-Net input (x+mu+spk), and matcha sizes the
# sinusoidal time embedding to in_channels — 224 here, not LJSpeech's 160.
IN_CH = 160 + SPK_DIM


# ---------------------------------------------------------------- builders
def load_ckpt():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    stats = hp["data_statistics"]
    return ck["state_dict"], float(stats["mel_mean"]), float(stats["mel_std"])


def build_text_encoder_vat(sd):
    from matcha.models.components.text_encoder import TextEncoder

    enc = NS(n_feats=80, n_channels=192, filter_channels=768,
             filter_channels_dp=256, n_heads=2, n_layers=6, kernel_size=3,
             p_dropout=0.0, spk_emb_dim=SPK_DIM, n_spks=1, prenet=True)
    dp = NS(filter_channels_dp=256, kernel_size=3, p_dropout=0.0)
    te = TextEncoder("RoPE Encoder", enc, dp, n_vocab=178, n_spks=N_SPKS,
                     spk_emb_dim=SPK_DIM, use_vat=True, vat_dim=VAT_DIM,
                     vat_cond_dim=VAT_COND)
    weights = {k[len("encoder."):]: v for k, v in sd.items()
               if k.startswith("encoder.")}
    te.load_state_dict(weights, strict=True)
    return te.eval()


def build_decoder_vat(sd):
    from matcha.models.components.decoder import Decoder

    dec = Decoder(in_channels=IN_CH, out_channels=80,
                  channels=(256, 256), dropout=0.0, attention_head_dim=64,
                  n_blocks=1, num_mid_blocks=2, num_heads=2,
                  act_fn="snakebeta", vat_dim=VAT_DIM, vat_cond_dim=VAT_COND)
    weights = {k[len("decoder.estimator."):]: v for k, v in sd.items()
               if k.startswith("decoder.estimator.")}
    dec.load_state_dict(weights, strict=True)
    return dec.eval()


def build_hifigan_24k():
    # The 24k fine-tune is architecturally HiFi-GAN v1 (same upsample stack);
    # only sr/fmax metadata differ, so matcha's Generator class loads it.
    from matcha.hifigan.env import AttrDict
    from matcha.hifigan.models import Generator

    h = AttrDict(json.load(open(VOC_CFG)))
    generator = Generator(h)
    ck = torch.load(VOC_CKPT, map_location="cpu", weights_only=False)
    generator.load_state_dict(ck["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator


# ------------------------------------------------------- masked re-authoring
def reauth_text_encoder_masked_vat(te):
    """build_matcha.reauth_text_encoder_masked + spk/vat graph inputs."""
    from einops import rearrange

    n_channels = te.n_channels
    for module in te.modules():
        if type(module).__name__ == "MultiHeadAttention":

            def mha_forward(self, x, c, attn_mask=None):
                q = self.conv_q(x)
                k = self.conv_k(c)
                v = self.conv_v(c)
                heads = self.n_heads
                q = rearrange(q, "b (h c) t -> b h t c", h=heads)
                k = rearrange(k, "b (h c) t -> b h t c", h=heads)
                v = rearrange(v, "b (h c) t -> b h t c", h=heads)
                q = self.query_rotary_pe(q)
                k = self.key_rotary_pe(k)
                scale = math.sqrt(self.k_channels)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                if attn_mask is not None:
                    scores = scores + (attn_mask - 1.0) * 1e4  # additive
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v).transpose(2, 3).contiguous().view(
                    q.shape[0], heads * self.k_channels, -1)
                return self.conv_o(out)

            module.forward = types.MethodType(mha_forward, module)

    class TEWrapVat(nn.Module):
        def __init__(self, te):
            super().__init__()
            self.te = te
            self.n_channels = n_channels

        def forward(self, emb_x, tmask, spk, vat):
            te = self.te
            x = (emb_x * math.sqrt(self.n_channels)).transpose(1, -1)
            x = te.prenet(x, tmask)
            # (1,64,1)*(1,1,T) broadcast MUL, not repeat(): repeat lowers to
            # BROADCAST_TO (GPU-banned); MUL broadcasts natively AND zeroes
            # the padded columns (downstream masking makes that a no-op for
            # the valid region — verified by G1/G3).
            spk_t = spk.unsqueeze(-1) * tmask
            x = torch.cat([x, spk_t], dim=1)
            vat_cond = te.vat_trunk(vat * tmask)
            x = te.encoder(x, tmask, vat_cond)
            return te.proj_m(x) * tmask, te.proj_w(x, tmask)

    return TEWrapVat(te).eval()


def _decoder_forward_clean_vat(self, x, mask, mu, t, spks=None, cond=None):
    """The fork's Decoder.forward with the GPU-hostile ::2 slices replaced.

    Same dec2 (RESHAPE+SLICE) trick as build_matcha._decoder_forward_clean,
    applied to BOTH the half-res masks and the FiLM conditioning track.
    """
    from einops import pack, rearrange

    def dec2(m):
        b, c, length = m.shape
        return m.reshape(b, c, length // 2, 2)[:, :, :, 0]

    t = self.time_embeddings(t)
    t = self.time_mlp(t)
    x = pack([x, mu], "b * t")[0]
    if spks is not None:
        # Broadcast MUL against the mask, not einops repeat() — repeat lowers
        # to BROADCAST_TO (GPU-banned); padded columns are masked downstream.
        spks = spks.unsqueeze(-1) * mask
        x = pack([x, spks], "b * t")[0]

    c = None
    if self.vat_trunk is not None:
        if cond is None:
            cond = torch.zeros(x.shape[0], self.vat_dim, x.shape[-1],
                               dtype=x.dtype, device=x.device)
        c = self.vat_trunk(cond * mask)

    hiddens = []
    masks = [mask]
    conds = [c]
    for level, (resnet, transformer_blocks, downsample) in enumerate(
            self.down_blocks):
        mask_down = masks[-1]
        x = resnet(x, mask_down, t)
        x = rearrange(x, "b c t -> b t c")
        mask_down = rearrange(mask_down, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_down, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_down = rearrange(mask_down, "b t -> b 1 t")
        if c is not None:
            x = self.down_films[level](x, conds[-1], mask_down)
        hiddens.append(x)
        x = downsample(x * mask_down)
        masks.append(dec2(mask_down))
        conds.append(dec2(conds[-1]) if c is not None else None)

    masks = masks[:-1]
    mask_mid = masks[-1]
    conds = conds[:-1]
    cond_mid = conds[-1]
    for level, (resnet, transformer_blocks) in enumerate(self.mid_blocks):
        x = resnet(x, mask_mid, t)
        x = rearrange(x, "b c t -> b t c")
        mask_mid = rearrange(mask_mid, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_mid, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_mid = rearrange(mask_mid, "b t -> b 1 t")
        if c is not None:
            x = self.mid_films[level](x, cond_mid, mask_mid)

    for level, (resnet, transformer_blocks, upsample) in enumerate(
            self.up_blocks):
        mask_up = masks.pop()
        cond_up = conds.pop() if c is not None else None
        x = resnet(torch.cat([x, hiddens.pop()], dim=1), mask_up, t)
        x = rearrange(x, "b c t -> b t c")
        mask_up = rearrange(mask_up, "b 1 t -> b t")
        for tb in transformer_blocks:
            x = tb(hidden_states=x, attention_mask=mask_up, timestep=t)
        x = rearrange(x, "b t c -> b c t")
        mask_up = rearrange(mask_up, "b t -> b 1 t")
        if c is not None:
            x = self.up_films[level](x, cond_up, mask_up)
        x = upsample(x * mask_up)

    x = self.final_block(x, mask_up)
    output = self.final_proj(x * mask_up)
    return output * mask


def reauth_decoder_masked_vat(dec, T):
    """build_matcha.reauth_decoder_masked with spk/vat-aware trace + forward."""
    dec.time_embeddings = nn.Identity()
    spk0 = torch.zeros(1, SPK_DIM)
    vat0 = torch.zeros(1, VAT_DIM, T)
    lengths = B._trace_convtranspose_lengths(
        dec, lambda: dec(torch.randn(1, 80, T), torch.ones(1, 1, T),
                         torch.randn(1, 80, T), torch.randn(1, IN_CH),
                         spk0, vat0))
    B.swap_convtranspose(dec, lengths)
    B.swap_norm_act(dec)
    for module in dec.modules():
        if type(module).__name__ == "Attention":

            def attn_forward(self, hidden_states, encoder_hidden_states=None,
                             attention_mask=None, **kwargs):
                b, seq, _ = hidden_states.shape
                heads = self.heads
                q = self.to_q(hidden_states)
                k = self.to_k(hidden_states)
                v = self.to_v(hidden_states)
                head_dim = q.shape[-1] // heads
                q = q.reshape(b, seq, heads, head_dim).transpose(1, 2)
                k = k.reshape(b, seq, heads, head_dim).transpose(1, 2)
                v = v.reshape(b, seq, heads, head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                if attention_mask is not None:
                    # diffusers SDPA semantics: raw 0/1 mask added as bias.
                    scores = scores + attention_mask.reshape(b, 1, 1, seq)
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v).transpose(1, 2)
                out = out.reshape(b, seq, heads * head_dim)
                return self.to_out[0](out)

            module.forward = types.MethodType(attn_forward, module)

    dec.forward = types.MethodType(_decoder_forward_clean_vat, dec)

    class DecWrapVat(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.d = decoder

        def forward(self, x, mu, t_emb, mask, spk, vat_y):
            return self.d(x, mask, mu, t_emb, spk, vat_y)

    return DecWrapVat(dec).eval()


# --------------------------------------------------------------- host side
def load_symbol_map():
    from matcha.text.symbols import symbols
    return {s: i for i, s in enumerate(symbols)}, list(symbols)


def phonemes_to_ids(phonemes, sym_to_id):
    # no_cleaners text_to_sequence + intersperse(0), as in training.
    seq = [sym_to_id[ch] for ch in phonemes]
    out = [0] * (2 * len(seq) + 1)
    out[1::2] = seq
    return torch.tensor(out, dtype=torch.long)[None]


def pick_val_rows(n, min_chars=50, max_chars=120):
    # max_chars bounds the interspersed token count (2*len+1) under MAX_TEXT.
    rows, seen = [], set()
    with open(VAL_FILELIST, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4 or parts[1] in seen:
                continue
            if not (min_chars <= len(parts[2]) <= max_chars):
                continue
            seen.add(parts[1])
            rows.append({"spk": int(parts[1]), "phonemes": parts[2]})
            if len(rows) == n:
                break
    return rows


def host_pipeline_vat(text_enc, decoder, vocoder, t_embed, emb_w, spk_vec,
                      ids, vat_scalar, mel_mean, mel_std, z=None):
    """Pad-to-max host synthesise() with spk + vat threading.

    vat_scalar: (vat_dim,) per-utterance control; expanded per-token, then
    through the duration alignment to per-frame (mirrors MatchaTTS.synthesise).
    """
    t_x = ids.shape[-1]
    ids_pad = torch.zeros(1, MAX_TEXT, dtype=torch.long)
    ids_pad[0, :t_x] = ids[0]
    tmask = torch.zeros(1, 1, MAX_TEXT)
    tmask[0, 0, :t_x] = 1.0
    emb_x = emb_w[ids_pad]
    vat_tok = vat_scalar.reshape(1, VAT_DIM, 1).repeat(1, 1, MAX_TEXT) * tmask
    mu_x, logw = text_enc(emb_x, tmask, spk_vec, vat_tok)
    w = torch.exp(logw) * tmask
    w_ceil = torch.ceil(w) * LENGTH_SCALE
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    ymask = sequence_mask(y_lengths.float(), MAX_MEL).unsqueeze(1)
    attn_mask = tmask.unsqueeze(-1) * ymask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    expand = attn.squeeze(1).transpose(1, 2)
    mu_y = torch.matmul(expand, mu_x.transpose(1, 2)).transpose(1, 2)
    vat_y = torch.matmul(expand, vat_tok.transpose(1, 2)).transpose(1, 2)
    if z is None:
        z = torch.randn(1, 80, MAX_MEL)
    x = z.clone() * ymask
    t_span = torch.linspace(0, 1, N_TIMESTEPS + 1)
    t = t_span[0]
    dt = t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        t_emb = t_embed(t.reshape(1))
        v = decoder(x, mu_y, t_emb, ymask, spk_vec, vat_y)
        x = x + dt * v
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    mel = (x * mel_std + mel_mean) * ymask
    wav = np.clip(vocoder(mel), -1, 1).reshape(-1)
    return wav[:int(y_lengths.item()) * 256], int(y_lengths.item()), z


def corr(a, b):
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def rms_db(wav):
    return 20 * math.log10(max(float(np.sqrt(np.mean(wav ** 2))), 1e-9))


# --------------------------------------------------------------------- main
def main():
    os.makedirs(ART, exist_ok=True)
    sd, mel_mean, mel_std = load_ckpt()
    emb_w = sd["encoder.emb.weight"]
    spk_w = sd["spk_emb.weight"]
    sym_to_id, symbols = load_symbol_map()
    print(f"MAX_TEXT={MAX_TEXT} MAX_MEL={MAX_MEL} sr={SR} "
          f"mel_mean={mel_mean:.4f} mel_std={mel_std:.4f} "
          f"spk_table={tuple(spk_w.shape)}")

    rows = pick_val_rows(3)
    assert rows, f"no usable rows in {VAL_FILELIST}"

    # --- G1: masked wrapper == the fork's own forward (full length) ---
    te_orig = build_text_encoder_vat(sd)
    row = rows[0]
    ids = phonemes_to_ids(row["phonemes"], sym_to_id)
    lengths = torch.tensor([ids.shape[-1]])
    spk_vec = spk_w[torch.tensor([row["spk"]])]
    vat = torch.tensor([[0.0, 0.7, 0.0]])
    with torch.no_grad():
        mu_ref, logw_ref, _ = te_orig(
            ids, lengths, spk_vec,
            vat=vat.unsqueeze(-1).expand(-1, -1, ids.shape[-1]))
        te_wrap_pure = reauth_text_encoder_masked_vat(build_text_encoder_vat(sd))
        tmask_full = torch.ones(1, 1, ids.shape[-1])
        vat_tok = vat.reshape(1, VAT_DIM, 1).repeat(1, 1, ids.shape[-1])
        mu_w, logw_w = te_wrap_pure(emb_w[ids], tmask_full, spk_vec, vat_tok)
    g1_te = float((mu_ref - mu_w).abs().max())
    print(f"\nG1 textenc wrapper-vs-model max|diff| = {g1_te:.3e} "
          f"{'PASS' if g1_te < 1e-4 else 'FAIL'}")

    dec_orig = build_decoder_vat(sd)
    T = 128
    x0 = torch.randn(1, 80, T)
    mu0 = torch.randn(1, 80, T)
    m_full = torch.ones(1, 1, T)
    vat_y0 = torch.zeros(1, VAT_DIM, T)
    vat_y0[0, 1] = 0.7
    with torch.no_grad():
        v_ref = dec_orig(x0, m_full, mu0, torch.tensor([0.5]),
                         spks=spk_vec, cond=vat_y0)
        dec_host = build_decoder_vat(sd)
        dec_host.time_embeddings = nn.Identity()
        dec_host.forward = types.MethodType(_decoder_forward_clean_vat,
                                            dec_host)
        t_emb0 = B.sin_pos_emb(torch.tensor([0.5]), IN_CH)
        v_host = dec_host(x0, m_full, mu0, t_emb0, spk_vec, vat_y0)
    g1_dec = float((v_ref - v_host).abs().max())
    print(f"G1 decoder wrapper-vs-model max|diff| = {g1_dec:.3e} "
          f"{'PASS' if g1_dec < 1e-3 else 'FAIL'}")

    # --- torch reference pipeline modules (masked, full precision) ---
    te_true_m = reauth_text_encoder_masked_vat(build_text_encoder_vat(sd))
    dec_true = build_decoder_vat(sd)
    dec_true.time_embeddings = nn.Identity()
    dec_true.forward = types.MethodType(_decoder_forward_clean_vat, dec_true)
    gen_true_m = build_hifigan_24k()

    def t_embed(t):
        return B.sin_pos_emb(t.reshape(-1), IN_CH)

    def te_true(emb_x, tmask, spk, vat_tok):
        with torch.no_grad():
            return te_true_m(emb_x, tmask, spk, vat_tok)

    def dec_true_fn(x, mu, t_emb, mask, spk, vat_y):
        with torch.no_grad():
            return dec_true(x, mask, mu, t_emb, spk, vat_y)

    def gen_true(mel):
        with torch.no_grad():
            return gen_true_m(mel).numpy()

    # --- convert fp32 -> fp16 ---
    te_r = reauth_text_encoder_masked_vat(build_text_encoder_vat(sd))
    dec_r = reauth_decoder_masked_vat(build_decoder_vat(sd), MAX_MEL)
    gen_r = B.reauth_hifigan(build_hifigan_24k(), MAX_MEL)

    ex = emb_w[torch.zeros(1, MAX_TEXT, dtype=torch.long)]
    tm0 = torch.ones(1, 1, MAX_TEXT)
    spk0 = torch.zeros(1, SPK_DIM)
    vat_t0 = torch.zeros(1, VAT_DIM, MAX_TEXT)
    xm = torch.randn(1, 80, MAX_MEL)
    mum = torch.randn(1, 80, MAX_MEL)
    tem = t_embed(torch.zeros(1))
    mm = torch.ones(1, 1, MAX_MEL)
    vat_m0 = torch.zeros(1, VAT_DIM, MAX_MEL)
    mel0 = torch.randn(1, 80, MAX_MEL)

    print("\n=== convert fp32 + quantize fp16 ===")
    specs = [("textenc", te_r, (ex, tm0, spk0, vat_t0)),
             ("decoder", dec_r, (xm, mum, tem, mm, spk0, vat_m0)),
             ("vocoder24k", gen_r, (mel0,))]
    fp16 = {}
    for name, module, inputs in specs:
        p32 = B.convert(module, inputs,
                        os.path.join(ART, f"sonora_{name}.tflite"))
        p16 = B.to_fp16(p32, os.path.join(ART, f"sonora_{name}_fp16.tflite"))
        clean = B.opcheck(p16, name + "_fp16")
        print(f"  {name}: fp32 {os.path.getsize(p32)/1e6:.1f}MB -> "
              f"fp16 {os.path.getsize(p16)/1e6:.1f}MB  GPU-clean={clean}")
        fp16[name] = p16

    cm_te = B.tfl_load(fp16["textenc"])
    cm_dec = B.tfl_load(fp16["decoder"])
    cm_gen = B.tfl_load(fp16["vocoder24k"])

    def tfl_te(emb_x, tmask, spk, vat_tok):
        outputs = B.tfl_run(cm_te, emb_x.numpy(), tmask.numpy(), spk.numpy(),
                            vat_tok.numpy())
        a, b = outputs
        mu, logw = (a, b) if a.shape[1] == 80 else (b, a)
        return torch.from_numpy(mu.copy()), torch.from_numpy(logw.copy())

    def tfl_dec(x, mu, t_emb, mask, spk, vat_y):
        outputs = B.tfl_run(cm_dec, x.numpy(), mu.numpy(), t_emb.numpy(),
                            mask.numpy(), spk.numpy(), vat_y.numpy())
        return torch.from_numpy(outputs[0].copy())

    def tfl_gen(mel):
        return B.tfl_run(cm_gen, mel.numpy())[0]

    # --- G2: per-graph fp16 parity on real activations ---
    print("\n=== G2 per-graph fp16 parity ===")
    with torch.no_grad():
        ids_pad = torch.zeros(1, MAX_TEXT, dtype=torch.long)
        ids_pad[0, :ids.shape[-1]] = ids[0]
        tmask = torch.zeros(1, 1, MAX_TEXT)
        tmask[0, 0, :ids.shape[-1]] = 1.0
        vat_tok = vat.reshape(1, VAT_DIM, 1).repeat(1, 1, MAX_TEXT) * tmask
        mu_t, logw_t = te_true(emb_w[ids_pad], tmask, spk_vec, vat_tok)
        mu_f, logw_f = tfl_te(emb_w[ids_pad], tmask, spk_vec, vat_tok)
        c_te = corr(mu_t.numpy().ravel(), mu_f.numpy().ravel())
        v_t = dec_true_fn(xm, mum, tem, mm, spk_vec, vat_m0)
        v_f = tfl_dec(xm, mum, tem, mm, spk_vec, vat_m0)
        c_dec = corr(v_t.numpy().ravel(), v_f.numpy().ravel())
        w_t = gen_true(mel0)
        w_f = tfl_gen(mel0)
        c_gen = corr(np.ravel(w_t), np.ravel(w_f))
    print(f"  textenc mu corr={c_te:.6f}  decoder v corr={c_dec:.6f}  "
          f"vocoder wav corr={c_gen:.6f}")

    # --- G3 + G4: e2e parity + energy monotonicity ---
    print("\n=== G3 e2e parity / G4 energy monotonicity ===")
    sweep_db = {}
    try:
        import soundfile as sf
    except Exception:
        sf = None
    for i, row in enumerate(rows):
        ids_i = phonemes_to_ids(row["phonemes"], sym_to_id)
        if ids_i.shape[-1] > MAX_TEXT:
            continue
        spk_i = spk_w[torch.tensor([row["spk"]])]
        for a in (-1.0, 0.0, 1.0):
            vat_i = torch.tensor([0.0, a, 0.0])
            torch.manual_seed(1234)
            wav_t, ylen, z = host_pipeline_vat(
                te_true, dec_true_fn, gen_true, t_embed, emb_w, spk_i, ids_i,
                vat_i, mel_mean, mel_std)
            if ylen > MAX_MEL:
                print(f"  SKIP row {i} (frames {ylen} > {MAX_MEL})")
                break
            wav_f, *_ = host_pipeline_vat(
                tfl_te, tfl_dec, tfl_gen, t_embed, emb_w, spk_i, ids_i,
                vat_i, mel_mean, mel_std, z=z)
            c = corr(wav_t, wav_f)
            sweep_db.setdefault(i, {})[a] = rms_db(wav_f)
            print(f"  row {i} spk {row['spk']} energy {a:+.0f}: "
                  f"corr={c:.5f} frames={ylen} rms={rms_db(wav_f):.1f}dB")
            if sf and i == 0:
                sf.write(os.path.join(ART, f"sample_e{a:+.0f}.wav"), wav_f, SR)
    mono = all(d[-1.0] < d[0.0] < d[1.0]
               for d in sweep_db.values() if len(d) == 3)
    print(f"  energy monotonic (rms dB, all rows): "
          f"{'PASS' if mono else 'FAIL'}")

    # --- host tables + config ---
    np.save(os.path.join(ART, "emb.npy"), emb_w.numpy().astype(np.float32))
    emb_w.numpy().astype("<f4").tofile(os.path.join(ART, "emb.bin"))
    np.save(os.path.join(ART, "spk_emb.npy"), spk_w.numpy().astype(np.float32))
    spk_w.numpy().astype("<f4").tofile(os.path.join(ART, "spk_emb.bin"))
    cfg = dict(symbols=symbols, n_vocab=178, n_channels=192, n_feats=80,
               MAX_TEXT=MAX_TEXT, MAX_MEL=MAX_MEL, mel_mean=mel_mean,
               mel_std=mel_std, hop=256, sample_rate=SR,
               length_scale=LENGTH_SCALE, sigma_min=1e-4,
               n_timesteps_default=N_TIMESTEPS, time_embed_dim=1024,
               in_channels=IN_CH, n_spks=N_SPKS, spk_emb_dim=SPK_DIM,
               vat_dim=VAT_DIM,
               vat_channels={"valence": 0, "energy": 1, "tension": 2},
               checkpoint=os.path.basename(CKPT),
               vocoder=os.path.basename(VOC_CKPT))
    with open(os.path.join(ART, "config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\nartifacts in {ART}:")
    for name in sorted(os.listdir(ART)):
        print(f"  {name} {os.path.getsize(os.path.join(ART, name))/1e6:.1f}MB")


if __name__ == "__main__":
    main()

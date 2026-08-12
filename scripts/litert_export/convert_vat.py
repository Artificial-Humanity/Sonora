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
                                  vat(1,V,Tt)      out: mu(1,80,Tt), logw
  sonora_decoder_fp16.tflite  in: x,mu(1,80,Tm), t_emb(1,224), ymask(1,1,Tm),
                                  spk(1,64), vat_y(1,V,Tm)   out: v(1,80,Tm)
  sonora_vocoder24k_fp16.tflite  in: mel(1,80,Tm) -> wav(1,1,Tm*256) @ 24 kHz
plus host tables: emb.{npy,bin} (178x192), spk_emb.{npy,bin} (247x64),
config.json (symbols, shapes, mel stats, sr 24000, and the CONTROL CONTRACT).

V is `matcha.delivery.VAT_DIM` — 8 under contract v2: three continuous V/A/T channels
plus a five-wide one-hot delivery block. It was a hardcoded 3 here until 2026-08-07, which
gave this converter its own opinion about the control surface; a graph and its manifest
disagreeing is the one failure a mobile host cannot detect, because the manifest is all it
can read.

THE CONTROL CONTRACT (F-H2), which is the manifest's job and did not exist:
  * V/A/T are continuous on [-1, 1] — per-speaker z-scores clamped at 2 sigma in
    derivation, so +/-1 is the edge of the TRAINED range, not a display convention. No
    bound was recorded anywhere, and out-of-range input renders fluent speech off the
    manifold rather than failing.
  * The delivery channels are ONE-HOT and must never be interpolated. Handed eight floats
    and three continuous names, a host will reasonably crossfade all eight and blend
    Newscaster into Dialogue — a vector the model never saw.
  * CFG guidance is host orchestration and exports NOTHING (F-M7), so a host cannot
    discover it from the artifacts; `control.guidance` documents the method and the
    >= 25 ODE steps that make it safe.

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
  G5 per-channel differential: drive every channel INDEPENDENTLY and require each
     to move the waveform. Until 2026-08-06 nothing did: G2 ran vat = zeros and
     G3/G4 drove [0, a, 0], so valence and tension had never once been nonzero
     through a converted graph, and a dropped or mis-sliced input would have
     passed every gate and shipped. Continuous channels sweep +/-1; delivery
     lanes are probed ONE-HOT against unknown, because -1 on a lane is a vector
     the host is forbidden to send and certifying a wire with it would be a
     green light for a path nobody may take.
  G6 delivery lanes are distinguishable: every lane being individually connected
     is not sufficient — five inputs wired to one summing junction pass G5 five
     times and give the host five names for one behaviour. Reads on-device as
     "delivery does not do much", which is indistinguishable from a weak channel.
  G7 front-end parity: the DEVICE G2P and the TRAINING G2P must produce identical
     phoneme strings. Everything above certifies the graph; nothing certified the
     text that reaches it, and a front end that phonemizes `we'll` as *well* feeds
     a perfect graph the wrong input. `kotlin_replica` had no contraction table —
     D-C1, the finding that poisoned corpus v1 through v3, still live on the device
     side on 2026-08-07 — and it diverged on 69 of 86 probe sentences. The tables
     ship as `g2p_contractions.json` from `matcha.text.op_g2p`, so the two cannot
     drift apart by omission, and this gate proves the device reproduces the host
     FROM that asset rather than from a transcription of it.

Every gate is RECORDED, not printed: `gates_or_die()` refuses to write artifacts
unless all pass. Before 2026-08-06 they printed PASS/FAIL and the script exited 0
either way, leaving a complete, shippable artifact set behind a failed export.

Run: scripts/litert_export/run.sh convert_vat.py [MAX_MEL]
     (code from the repo, data on /data — AGENTS.md §6)
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import json
import math
import os
import pathlib
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

# F-H1. The default was `…/Artificial-Humanity/Sonora`, which stopped being a repo on
# 2026-07-22 when the umbrella layout was flattened — it is now a plain container holding
# `github/` and `huggingface/`. So `sys.path.insert` added a directory with no `matcha`
# package in it, the import fell through to whatever `matcha` the harness venv happened to
# hold (documented as the stock PyPI `matcha-tts`), and the converter exported UPSTREAM's
# architecture while every log line said Sonora. That is the worst possible shape for this
# script: a wrong export converts cleanly and its graphs run.
#
# Derive it from this file's own location instead of hardcoding a path at all — this file
# lives at <repo>/scripts/litert_export/, so the repo root is two parents up, and the
# default is correct wherever the checkout sits. The env var still wins for the /data
# working copy. Verified, not assumed: the import target must actually be there.
SONORA = os.environ.get("SONORA_REPO") or str(pathlib.Path(__file__).resolve().parents[2])
if not (pathlib.Path(SONORA) / "matcha" / "models" / "matcha_tts.py").is_file():
    raise SystemExit(
        f"SONORA_REPO={SONORA!r} does not contain matcha/models/matcha_tts.py.\n"
        "  Point it at the Sonora repo root (…/Sonora/github). Without this the import\n"
        "  falls through to whatever `matcha` is installed and exports the wrong model."
    )
sys.path.insert(0, SONORA)

# Imported AFTER the path insert and the existence check above, on purpose: this is the
# single definition of the conditioning vector's width and meaning, and taking it from a
# `matcha` that is not the repo's is the F-H1 failure in a different coat.
from matcha import delivery  # noqa: E402

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

ART = os.path.join(B.WORK, "artifacts_vat")
MAX_TEXT = 256
MAX_MEL = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_TIMESTEPS = 10
LENGTH_SCALE = 1.0  # the derisk gate's setting (not LJSpeech's 0.95)
SR = 24000
N_SPKS, SPK_DIM, VAT_COND = 247, 64, 256

# F-H2. The conditioning width and the meaning of every channel come from
# `matcha/delivery.py`, which is the single definition shared with the trunk, the corpus
# derivation, the CLI and the Vocalizer. This file used to hardcode 3 and a three-name
# map; a converter with its own opinion about the control surface is how a graph and its
# manifest come to disagree, and the manifest is the only thing the mobile host can read.
VAT_DIM = delivery.VAT_DIM

# Channel order, in one place: the G5 probe and config.json must agree, and a
# manifest that disagrees with the graph mislabels the whole control surface.
CHANNELS = dict(enumerate(("valence", "energy", "tension") + delivery.DELIVERY_LANES))

# WHICH CHANNELS ARE CONTINUOUS AND WHICH ARE CATEGORICAL — the half of F-H2 that did not
# exist before the delivery migration, and the one a host cannot infer.
#
# V/A/T are continuous on [-1, 1]: per-speaker z-scores clamped at 2 sigma during
# derivation, so ±1 IS the edge of the trained range. Values beyond it do not make the
# channel stronger, they move the FiLM activation off the manifold — and the model still
# renders speech, which is the trap.
#
# The delivery block is ONE-HOT and must never be interpolated. A host handed eight floats
# and told three are continuous will reasonably treat all eight the same way and crossfade
# between Newscaster and Dialogue, which has no meaning: `delivery.lane_of_vector` refuses
# exactly that vector, and the model was never trained on one.
CONTINUOUS_CHANNELS = tuple(range(delivery.VAT_BASE_DIM))
CATEGORICAL_CHANNELS = tuple(range(delivery.VAT_BASE_DIM, delivery.VAT_DIM))
SLOT = {name: idx for idx, name in CHANNELS.items()}
# Multi-speaker widens the U-Net input (x+mu+spk), and matcha sizes the
# sinusoidal time embedding to in_channels — 224 here, not LJSpeech's 160.
IN_CH = 160 + SPK_DIM


# ---------------------------------------------------------------- builders
def detect_vat_dim(sd):
    """The checkpoint's own VAT width, read off the trunk's first conv.

    Weight shape is [cond_dim, vat_dim, 1], so the channel count is not a guess.
    Returns None for a checkpoint with no trunk (an unconditioned baseline).
    """
    for k in ("encoder.vat_trunk.net.0.weight",
              "decoder.estimator.vat_trunk.net.0.weight"):
        if k in sd:
            return int(sd[k].shape[1])
    return None


def load_ckpt():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    stats = hp["data_statistics"]
    sd = ck["state_dict"]
    # THE EXPORT WIDTH SEAM. VAT_DIM is a hardcoded 3 above, and every exported shape
    # is built from it — the [1, VAT_DIM, T] graph inputs, the per-channel probes, and
    # the vat_dim written into config.json for the mobile host to read.
    #
    # `load_state_dict(strict=True)` does catch a mismatch, but only as an unnamed
    # tensor-shape error from inside a builder, after the constant has already sized
    # graph inputs. And the config.json this writes is what the host trusts: exporting a
    # 4-channel checkpoint under a 3-channel constant would ship a manifest that lies
    # about the model's control surface. Refuse early and name the edit.
    # THE EXPORT WIDTH SEAM, now driven by the contract rather than by a constant.
    # `VAT_DIM` is `delivery.VAT_DIM`, so a checkpoint at the contract width exports and
    # anything else is refused — including, deliberately, a NARROWER pre-v2 checkpoint.
    # Exporting one of those would produce a graph with no delivery inputs whose
    # config.json is nonetheless read by a host that now knows about lanes; the host would
    # send eight floats to a three-channel graph. It is not a downgrade path, it is a
    # different artifact, and it needs its own manifest rather than this one truncated.
    found = detect_vat_dim(sd)
    if found is not None and found != VAT_DIM:
        raise SystemExit(
            f"{CKPT}\n  checkpoint has vat_dim={found}; this converter exports the "
            f"contract width, VAT_DIM={VAT_DIM}\n"
            f"  ({delivery.VAT_BASE_DIM} continuous V/A/T + {delivery.DELIVERY_DIM} "
            "one-hot delivery lanes — matcha/delivery.py).\n"
            "  A narrower checkpoint predates contract v2. Its graph has no delivery\n"
            "  inputs, but the config.json written here declares the lane vocabulary, so\n"
            "  a host reading it would send eight floats to a three-channel graph. That\n"
            "  is a different artifact, not a truncation of this one.\n"
            "  Re-derive the corpus at the contract width and retrain, or export the old\n"
            "  checkpoint with the converter revision that matches it (git history)."
        )
    return sd, float(stats["mel_mean"]), float(stats["mel_std"])


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


# F-C1. Every gate in this file printed PASS/FAIL and NOTHING acted on it: main() ran to
# completion, wrote artifacts/ and config.json, and exited 0 whether the graphs matched the
# model or not. A failed export was indistinguishable from a good one at the shell, and it
# left a complete, shippable artifact set behind — which is worse than not exporting.
#
# GATES is the ledger. Nothing writes an artifact until every entry has passed.
GATES = []


def gate(name, ok, detail):
    """Record a gate result and print it. Returns `ok` so callers can branch."""
    GATES.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return bool(ok)


def gates_or_die():
    """Refuse to emit artifacts unless every gate passed."""
    failed = [(n, d) for n, ok, d in GATES if not ok]
    print(f"\n=== gates: {len(GATES) - len(failed)}/{len(GATES)} passed ===")
    if failed:
        for n, d in failed:
            print(f"  FAIL  {n}: {d}", file=sys.stderr)
        raise SystemExit(
            f"\n{len(failed)} gate(s) failed — refusing to write artifacts.\n"
            "  A converted graph that fails parity still RUNS and still produces audio;\n"
            "  shipping one is how a wrong export reaches a device looking healthy."
        )


def corr(a, b):
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def gain_error_db(a, b):
    """F-M5. How much QUIETER or louder `b` is than `a`, in dB. Pearson cannot see this.

    Correlation is scale-invariant by construction, so `b = 0.5 * a` — a systematic fp16
    gain error, every sample exactly half — scores corr = 1.0000 and passes a 0.99 gate
    with room to spare. That is not a hypothetical shape of bug: it is what a mis-scaled
    dequantization, a dropped `* 2`, or a wrong output scale factor looks like, and it is
    the flagship axis of this model. The ENERGY channel is a loudness dial; an export that
    halves every render is exactly the failure a correlation gate is blind to, and the
    symptom on-device would be "the energy channel is weak", not "the export is broken".

    Reported as dB rather than a ratio so the number is comparable to `rms_db` above and
    to the G4 sweep, where the whole question is also loudness.
    """
    n = min(len(a), len(b))
    ra = float(np.sqrt(np.mean(np.square(a[:n]))))
    rb = float(np.sqrt(np.mean(np.square(b[:n]))))
    return 20 * math.log10(max(rb, 1e-9) / max(ra, 1e-9))


def nrmse(a, b):
    """RMS error between the two waveforms, normalised by the reference's RMS.

    The scale-SENSITIVE companion to `corr`. Unlike raw RMSE this is comparable across
    renders of different loudness, so one threshold means the same thing on a whisper and
    on a shout.
    """
    n = min(len(a), len(b))
    ra = float(np.sqrt(np.mean(np.square(a[:n]))))
    return float(np.sqrt(np.mean(np.square(a[:n] - b[:n])))) / max(ra, 1e-9)


def rms_db(wav):
    return 20 * math.log10(max(float(np.sqrt(np.mean(wav ** 2))), 1e-9))


def g2p_parity_gate():
    """G7. Certify the device text front end against the training one. -> the asset bytes.

    Every other gate in this file asks whether the converted GRAPH matches the model.
    None of them ask whether the text that reaches the graph matches the text the model
    was trained on, and that is a separate failure with the same signature: fluent audio,
    wrong phonemes, nothing to see at the shell. `kotlin_replica` phonemized with a flat
    dictionary lookup and no apostrophe handling until 2026-08-07 — D-C1's exact shape,
    still live on the device side five days after the host was fixed — and the dictionary
    holds no apostrophe keys, so `we'll` resolved to the letters `well` and arrived as a
    successful lookup.

    Returns the serialized `g2p_contractions.json` bytes so main() writes the same string
    this gate certified. The round trip is not decoration: the device reads JSON, so the
    thing under test has to be what JSON preserves, not the Python dict it came from.
    """
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
    import device_g2p

    from matcha.text import op_g2p

    tables = op_g2p.contraction_tables()
    serialized = json.dumps(tables, ensure_ascii=False, indent=2, sort_keys=True)
    shipped = json.loads(serialized)

    host = op_g2p.OpenPhonemizerG2P()
    # D-M4 is off by default and the exported tables carry no homograph resolver, so a
    # host that resolves them would be certifying a front end the device cannot run. If a
    # derivation ever turns them on, this fails here rather than on someone's ear.
    if not gate("G7 exported front end matches the corpus front end",
                not host.homographs,
                f"homographs={host.homographs} (the device has no context resolver)"):
        return serialized

    device = device_g2p.DeviceG2P(tables=shipped, assets_dir=host.assets_dir)
    sentences = device_g2p.probe_sentences(shipped)
    bad = device_g2p.compare(host.phonemize, device.phonemize, sentences)
    for sentence, want, got in bad[:5]:
        print(f"    {sentence!r}\n      host: {want!r}\n      dev : {got!r}", flush=True)
    gate("G7 host/device G2P parity", not bad,
         f"{len(sentences) - len(bad)}/{len(sentences)} probe sentences identical "
         f"({len(tables['contractions'])} contractions, {len(tables['clitics'])} clitics)")
    return serialized


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
    print()
    gate("G1 textenc wrapper-vs-model", g1_te < 1e-4, f"max|diff| = {g1_te:.3e} (< 1e-4)")

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
    gate("G1 decoder wrapper-vs-model", g1_dec < 1e-3, f"max|diff| = {g1_dec:.3e} (< 1e-3)")

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
    # Thresholds are the values this lane has actually held: build_matcha reports per-graph
    # corr 1.000000 and e2e >= 0.99. 0.9995 leaves room for fp16 noise without admitting a
    # graph that is merely correlated.
    gate("G2 textenc fp16 parity", c_te >= 0.9995, f"mu corr = {c_te:.6f}")
    gate("G2 decoder fp16 parity", c_dec >= 0.9995, f"v corr = {c_dec:.6f}")
    gate("G2 vocoder fp16 parity", c_gen >= 0.9995, f"wav corr = {c_gen:.6f}")

    # --- G3 + G4: e2e parity + energy monotonicity ---
    print("\n=== G3 e2e parity / G4 energy monotonicity ===")
    sweep_db = {}
    e2e_corrs = []
    e2e_gain, e2e_nrmse = [], []   # F-M5: the scale-SENSITIVE companions to corr
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
            # Energy alone, at the contract width. This was a literal 3-vector, which
            # would have raised a shape error the moment the width changed — loud, but
            # from inside a builder rather than here, and the reader would have to work
            # out that "energy" is slot 1 of however many.
            vat_i = torch.zeros(VAT_DIM)
            vat_i[SLOT["energy"]] = a
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
            e2e_corrs.append(c)
            # F-M5: the scale-SENSITIVE companions. corr alone cannot see a systematic
            # gain error, which is precisely the shape a mis-scaled dequantization takes.
            e2e_gain.append(gain_error_db(wav_t, wav_f))
            e2e_nrmse.append(nrmse(wav_t, wav_f))
            sweep_db.setdefault(i, {})[a] = rms_db(wav_f)
            print(f"  row {i} spk {row['spk']} energy {a:+.0f}: "
                  f"corr={c:.5f} gain={e2e_gain[-1]:+.2f}dB nrmse={e2e_nrmse[-1]:.4f} "
                  f"frames={ylen} rms={rms_db(wav_f):.1f}dB")
            if sf and i == 0:
                sf.write(os.path.join(ART, f"sample_e{a:+.0f}.wav"), wav_f, SR)
    complete = [d for d in sweep_db.values() if len(d) == 3]
    mono = bool(complete) and all(d[-1.0] < d[0.0] < d[1.0] for d in complete)
    # `all()` over an EMPTY sequence is True, so a run where every row was skipped for
    # length used to report PASS on zero evidence. Requiring at least one complete sweep is
    # the difference between "monotonic" and "never checked".
    gate("G4 energy monotonic", mono, f"{len(complete)} row(s) swept, rms dB strictly rising")
    e2e_min = min(e2e_corrs) if e2e_corrs else float("nan")
    gate("G3 e2e waveform parity", bool(e2e_corrs) and e2e_min >= 0.99,
         f"min corr over {len(e2e_corrs)} render(s) = {e2e_min:.5f} (>= 0.99)")

    # G3b / G3c — F-M5. `corr` is scale-invariant, so `wav_f = 0.5 * wav_t` scores 1.0000
    # and sails through the gate above. On a model whose flagship axis is a LOUDNESS dial
    # that is the worst possible blind spot: on-device it reads as "the energy channel is
    # weak", not as a broken export. 0.5 dB is well inside fp16 round-trip noise (measured
    # runs sit near 0.0) and well below the ~6 dB a dropped factor of two would give.
    worst_gain = max((abs(g) for g in e2e_gain), default=float("nan"))
    gate("G3b e2e gain parity", bool(e2e_gain) and worst_gain <= 0.5,
         f"worst |gain error| over {len(e2e_gain)} render(s) = {worst_gain:.3f} dB "
         "(<= 0.5) — corr cannot see this")
    worst_nrmse = max(e2e_nrmse, default=float("nan"))
    gate("G3c e2e sample-wise error", bool(e2e_nrmse) and worst_nrmse <= 0.15,
         f"worst RMSE/RMS over {len(e2e_nrmse)} render(s) = {worst_nrmse:.4f} (<= 0.15)")

    # --- G5: PER-CHANNEL DIFFERENTIAL PROBE ---------------------------------------
    #
    # The other half of F-C1, and the part no existing gate covered: **valence and tension
    # had never once been driven nonzero through a converted graph.** G2 runs vat = zeros;
    # G3/G4 drive [0, a, 0], which is energy alone. So a graph that dropped the valence
    # input, wired tension to the wrong slice, or fed both into a dead branch would have
    # passed every gate in this file and shipped — and on a device it would simply be a
    # model whose valence dial does nothing, which reads as a training failure rather than
    # an export bug.
    #
    # Drive each channel INDEPENDENTLY and require the output to move. The assertion is
    # deliberately weak — "this input changes the audio" — because direction and magnitude
    # are the model's business, not the converter's. What must be true is that the wire is
    # connected, and that is exactly what was never checked.
    print("\n=== G5 per-channel differential probe ===")
    probe = next((r for r in rows
                  if phonemes_to_ids(r["phonemes"], sym_to_id).shape[-1] <= MAX_TEXT), None)
    if probe is None:
        gate("G5 per-channel differential", False, "no row short enough to probe")
    else:
        ids_p = phonemes_to_ids(probe["phonemes"], sym_to_id)
        spk_p = spk_w[torch.tensor([probe["spk"]])]

        def render(vat_vec):
            torch.manual_seed(1234)
            wav, ylen, z = host_pipeline_vat(
                te_true, dec_true_fn, gen_true, t_embed, emb_w, spk_p, ids_p,
                torch.tensor(vat_vec), mel_mean, mel_std)
            wav_f, *_ = host_pipeline_vat(
                tfl_te, tfl_dec, tfl_gen, t_embed, emb_w, spk_p, ids_p,
                torch.tensor(vat_vec), mel_mean, mel_std, z=z)
            return wav_f

        base = render([0.0] * VAT_DIM)
        for idx, name in sorted(CHANNELS.items(), key=lambda kv: kv[0]):
            # F-H2: probe each channel the way the CONTRACT says it is driven.
            #
            # The continuous channels sweep ±1, the edge of the trained range. The
            # delivery channels are ONE-HOT: -1 on a lane is not "the opposite of
            # Newscaster", it is a vector the model never saw and that
            # `delivery.lane_of_vector` refuses outright. Driving it here would have the
            # gate certify the wire using an input the host is forbidden to send — a
            # green light for a path nobody may take.
            #
            # So a lane is probed at 1.0 against neutral only. That is also exactly what
            # the host will do, which is the point of a gate.
            categorical = idx in CATEGORICAL_CHANNELS
            hi = render([1.0 if k == idx else 0.0 for k in range(VAT_DIM)])
            d_hi = float(np.abs(hi[:min(len(base), len(hi))]
                                - base[:min(len(base), len(hi))]).mean())
            if categorical:
                moved, detail = d_hi, f"mean |delta| vs unknown: one-hot -> {d_hi:.2e}"
            else:
                lo = render([-1.0 if k == idx else 0.0 for k in range(VAT_DIM)])
                n = min(len(base), len(hi), len(lo))
                d_hi = float(np.abs(hi[:n] - base[:n]).mean())
                d_lo = float(np.abs(lo[:n] - base[:n]).mean())
                moved = min(d_hi, d_lo)
                detail = f"mean |delta| vs neutral: +1 -> {d_hi:.2e}, -1 -> {d_lo:.2e}"
            # 1e-5 mean absolute sample delta on a [-1, 1] waveform: far above fp16 noise,
            # far below "audibly different". A disconnected input gives exactly 0.
            kind = "lane" if categorical else "channel"
            gate(f"G5 {name} {kind} is connected", moved > 1e-5, detail)

        # G6: the delivery block is a GROUP, and the graph must distinguish its members.
        # Every lane being individually connected is not sufficient — five inputs wired to
        # the same summing junction would pass G5 five times and give the host five names
        # for one behaviour. Cheap to check, and the failure it catches is one that reads
        # on-device as "delivery does nothing much".
        if CATEGORICAL_CHANNELS:
            lane_renders = {}
            for idx in CATEGORICAL_CHANNELS:
                lane_renders[CHANNELS[idx]] = render(
                    [1.0 if k == idx else 0.0 for k in range(VAT_DIM)])
            names = sorted(lane_renders)
            worst, worst_pair = float("inf"), ("", "")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    wa, wb = lane_renders[a], lane_renders[b]
                    n = min(len(wa), len(wb))
                    d = float(np.abs(wa[:n] - wb[:n]).mean())
                    if d < worst:
                        worst, worst_pair = d, (a, b)
            gate("G6 delivery lanes are distinguishable", worst > 1e-5,
                 f"closest pair {worst_pair[0]}/{worst_pair[1]}: mean |delta| = {worst:.2e}")

    # --- G7: THE TEXT FRONT END ---------------------------------------------------
    print("\n=== G7 host/device G2P parity ===")
    g2p_json = g2p_parity_gate()

    gates_or_die()

    # --- host tables + config ---
    # The apostrophe tables the device front end cannot work without. Written here rather
    # than transcribed into the port, because D-C1 is a defect of omission and a table
    # kept in sync by hand is the same defect on a delay.
    with open(os.path.join(ART, "g2p_contractions.json"), "w", encoding="utf-8") as f:
        f.write(g2p_json)
    np.save(os.path.join(ART, "emb.npy"), emb_w.numpy().astype(np.float32))
    emb_w.numpy().astype("<f4").tofile(os.path.join(ART, "emb.bin"))
    np.save(os.path.join(ART, "spk_emb.npy"), spk_w.numpy().astype(np.float32))
    spk_w.numpy().astype("<f4").tofile(os.path.join(ART, "spk_emb.bin"))
    cfg = dict(symbols=symbols, n_vocab=178, n_channels=192, n_feats=80,
               MAX_TEXT=MAX_TEXT, MAX_MEL=MAX_MEL, mel_mean=mel_mean,
               mel_std=mel_std, hop=256, sample_rate=SR,
               length_scale=LENGTH_SCALE, sigma_min=1e-4,
               n_timesteps_default=N_TIMESTEPS,
               # F-M4: this said 1024 while the masked graphs consume
               # `in_channels` — matcha sizes the sinusoidal time embedding
               # to in_channels, 224 here (see IN_CH). The mobile host trusts
               # this manifest to size its t_emb buffer, so 1024 was a lie it
               # had no way to detect except by the graph rejecting the shape.
               time_embed_dim=IN_CH,
               in_channels=IN_CH, n_spks=N_SPKS, spk_emb_dim=SPK_DIM,
               vat_dim=VAT_DIM,
               vat_channels={v: k for k, v in CHANNELS.items()},
               # F-H2. `vat_dim` and a name-to-slot map tell a host WHERE each control is
               # and nothing about what may be sent there. Both halves of that gap are
               # real, and both fail silently on-device:
               #
               #   * no bound was ever recorded. V/A/T are per-speaker z-scores clamped at
               #     2 sigma in derivation, so ±1 is the EDGE of the trained range — but
               #     a host reading this manifest had no way to know, and a request for
               #     valence 5 does not render "more valence", it moves the FiLM
               #     activation off the manifold and still produces fluent speech.
               #   * nothing said which channels are categorical. Handed eight floats and
               #     three continuous names, a host will reasonably crossfade all eight
               #     and blend Newscaster into Dialogue — a vector the model never saw and
               #     that `delivery.lane_of_vector` refuses outright.
               #
               # `control` is the contract, machine-readable. A host that reads only
               # `vat_dim` still works exactly as before; one that reads this can validate
               # before it renders instead of after someone listens.
               control=dict(
                   continuous=dict(
                       channels=list(CONTINUOUS_CHANNELS),
                       names=[CHANNELS[i] for i in CONTINUOUS_CHANNELS],
                       min=-1.0, max=1.0, neutral=0.0,
                       clamp="reject",
                       note=("Per-speaker z-scores clamped at 2 sigma during corpus "
                             "derivation, so +/-1 is the edge of the TRAINED range, not "
                             "a display convention. Out-of-range input does not "
                             "strengthen the channel; it leaves the manifold and still "
                             "renders fluent speech. Reject rather than clamp, so the "
                             "caller learns its request was wrong."),
                   ),
                   categorical=[dict(
                       group="delivery",
                       channels=list(CATEGORICAL_CHANNELS),
                       encoding="one_hot",
                       values=list(delivery.DELIVERY_LANES),
                       unknown=dict(vector=[0.0] * delivery.DELIVERY_DIM,
                                    meaning="no delivery label; equivalent to contract v1"),
                       note=("EXACTLY ONE of these channels may be 1.0 and the rest 0.0, "
                             "or all of them 0.0 for unknown. DO NOT INTERPOLATE: there "
                             "is no vector between Newscaster and Dialogue, the model was "
                             "never trained on one, and a fractional value is refused by "
                             "the reference implementation (matcha/delivery.py, "
                             "lane_of_vector)."),
                   )],
                   guidance=dict(
                       # F-M7: CFG is pure HOST orchestration. The graph is unchanged —
                       # nothing about it exports — so a host that does not know this
                       # cannot discover it from the artifacts, and the one number that
                       # makes it safe (the ODE step count) lives nowhere else.
                       supported=True, exported=False, default=1.0,
                       recommended_range=[1.0, 3.0],
                       min_n_timesteps_above_1=25,
                       method=("Run the decoder TWICE per ODE step — once with the "
                               "caller's vat, once with vat = all zeros — and "
                               "extrapolate: v = v_uncond + s * (v_cond - v_uncond). The "
                               "graph is identical for both passes; guidance is not an "
                               "input and there is nothing to export."),
                       note=("Validated by ear at s = 2-3 with >= 25 ODE steps "
                             "(2026-07-16); at 10 steps solver artifacts dominate and the "
                             "result is worse than s = 1. NEVER amplify by scaling the "
                             "vat input instead — raw out-of-range VAT saturates."),
                   ),
               ),
               # The text front end, declared for the same reason `control` is: the
               # manifest is the only thing a host can read, and the front end is as
               # capable of silently disagreeing with the training corpus as the control
               # surface is. `homographs` is the D-M4 switch — false here means the
               # exported tables carry no context resolver, so a device that renders
               # `live` must render the dictionary's adjective, exactly as the corpus
               # does. If a derivation ever turns it on, G7 fails until the resolver is
               # ported, rather than the two front ends drifting quietly apart.
               g2p=dict(
                   dictionary="g2p_dict.txt.gz",
                   neural_oov="dp_g2p_matcha_fp16.tflite",
                   contractions="g2p_contractions.json",
                   homographs=False,
                   normalization=["ascii_fold", "lowercase", "expand_abbreviations",
                                  "remove_brackets", "hyphen_to_space",
                                  "collapse_whitespace"],
                   note=("The dictionary holds NO apostrophe keys and the neural charset "
                         "has no \"'\", so a front end that looks a contraction up "
                         "directly gets its apostrophe-stripped letters back and cannot "
                         "tell that it failed: we'll -> wˈɛl, which is the word *well*. "
                         "Resolve apostrophe words through g2p_contractions.json first. "
                         "The reference implementation is "
                         "scripts/litert_export/device_g2p.py, and gate G7 holds it to "
                         "phoneme-string parity with the training front end."),
               ),
               contract_version=2,
               checkpoint=os.path.basename(CKPT),
               vocoder=os.path.basename(VOC_CKPT))
    with open(os.path.join(ART, "config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\nartifacts in {ART}:")
    for name in sorted(os.listdir(ART)):
        print(f"  {name} {os.path.getsize(os.path.join(ART, name))/1e6:.1f}MB")


if __name__ == "__main__":
    main()

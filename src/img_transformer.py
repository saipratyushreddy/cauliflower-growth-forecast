"""
Attention-over-history image forecaster (the image-track CNN-Transformer).

* TOKENS: each input frame -> ONE token. The shared Step C / ConvLSTM encoder (stem + downs) maps a frame to an 8x8x256
  bottleneck; the token is LayerNorm(GlobalAvgPool(bottleneck)) projected to d_model. Not spatial-patch tokens.
* TIME ENCODING: per token, a fixed continuous sinusoidal encoding of (target_day - input_day) (NOT day_after_planting):
  geometric frequency set, 128 dims, wavelengths 16-1000 days, chosen by src/pe_aliasing_check.py on the real range 2..84.
  A dedicated learned OUTPUT token sits at offset 0 (the target time).
* ATTENTION: standard Transformer encoder over [output token, frame tokens (chronological)], FULL available history
  (up to 14 frames), padding mask for shorter sequences, no K truncation, no causal mask (all history precedes the target).
* DECODER INPUTS: one attention-pooling step (the output token as query over the frame-token outputs) gives weights a_t
  over frames. The decoder's bottleneck AND all five skip connections are the a_t-weighted combination of the per-frame
  feature maps, so, unlike Step C / ConvLSTM (skips from the last frame only), any frame can supply detail. The output
  token also FiLM-modulates the pooled bottleneck (zero-initialised), replacing Step C's offset FiLM.
* Frames are encoded only where real (no padded compute), in chunks under activation checkpointing, so full-length
  histories fit in memory.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from img_film_unet import FiLMUNet

PE_DIM = 128
PE_LMIN, PE_LMAX = 16.0, 1000.0


def time_encoding(offsets, dim=PE_DIM, lmin=PE_LMIN, lmax=PE_LMAX):
    """Continuous encoding of (target_day - input_day) in days. offsets: float tensor [...]. Returns [..., dim]."""
    m = dim // 2
    lam = lmin * (lmax / lmin) ** (torch.arange(m, device=offsets.device, dtype=torch.float32) / (m - 1))
    a = offsets.float().unsqueeze(-1) * (2 * np.pi / lam)
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class TemporalTransformerUNet(FiLMUNet):
    def __init__(self, widths=(32, 64, 128, 192, 256, 256), d_model=128, nhead=4, num_layers=2, dim_ff=256,
                 dropout=0.1, chunk=32, grad_checkpoint=True):
        super().__init__(widths, 64)
        del self.film                                         # Step C's scalar-offset FiLM is replaced by the output token
        cb = widths[-1]
        self.d_model, self.chunk, self.grad_checkpoint = d_model, chunk, grad_checkpoint
        self.tok_norm = nn.LayerNorm(cb)
        self.tok_proj = nn.Linear(cb, d_model)
        self.pe_proj = nn.Linear(PE_DIM, d_model)
        self.out_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.out_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.tf_norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.film_out = nn.Sequential(nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 2 * cb))
        nn.init.zeros_(self.film_out[-1].weight)
        nn.init.zeros_(self.film_out[-1].bias)

    def _encode(self, x):
        h = self.stem(x * 2 - 1)
        skips = [h]
        for d in self.downs:
            h = d(h)
            skips.append(h)
        return tuple(skips)                                   # skips[-1] is the bottleneck

    def encode_frames(self, x):
        outs = []
        for i in range(0, x.shape[0], self.chunk):
            xc = x[i:i + self.chunk]
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self._encode, xc, use_reentrant=False))
            else:
                outs.append(self._encode(xc))
        return [torch.cat([o[l] for o in outs], 0) for l in range(len(outs[0]))]

    def forward(self, x_flat, b_idx, t_idx, offsets, B, Kmax, return_weights=False):
        """x_flat [N,3,H,W] valid frames in [0,1]; b_idx/t_idx [N] sample and chronological slot of each frame;
        offsets [N] = target_day - input_day (days) of each frame. Returns images [B,3,H,W]."""
        feats = self.encode_frames(x_flat)
        z = feats[-1]                                          # [N,C,8,8] bottleneck of every real frame
        skips = feats[:-1]
        tok = self.tok_proj(self.tok_norm(z.float().mean((2, 3)))) + self.pe_proj(time_encoding(offsets))
        seq = torch.zeros(B, Kmax, self.d_model, device=tok.device, dtype=tok.dtype)
        seq[b_idx, t_idx] = tok
        pad = torch.ones(B, Kmax, dtype=torch.bool, device=tok.device)
        pad[b_idx, t_idx] = False
        out0 = (self.out_token + self.pe_proj(time_encoding(torch.zeros(1, device=tok.device)))[0]).expand(B, 1, -1)
        hseq = self.tf(torch.cat([out0, seq], 1),
                       src_key_padding_mask=torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tok.device), pad], 1))
        hseq = self.tf_norm(hseq)
        o, ht = hseq[:, 0], hseq[:, 1:]
        score = (self.q(o).unsqueeze(1) * self.k(ht)).sum(-1) / np.sqrt(self.d_model)     # [B,Kmax]
        w = torch.softmax(score.masked_fill(pad, float("-inf")), dim=1)                    # attention weights over real frames
        wf = w[b_idx, t_idx]                                                               # [N]

        def pool(feat):
            f = feat * wf.view(-1, 1, 1, 1).to(feat.dtype)
            return torch.zeros(B, *feat.shape[1:], device=feat.device, dtype=feat.dtype).index_add_(0, b_idx, f)

        h = pool(z)
        gamma, beta = self.film_out(o).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.mid(h)
        pooled = [pool(s) for s in skips]
        for up in self.ups:
            h = up(h, pooled.pop())
        img = torch.sigmoid(self.head(h))
        return (img, w) if return_weights else img

"""
History-aware forecaster: Step C's shared encoder on the last K input frames -> ConvLSTM over the
8x8x256 bottleneck features, each step FiLM-conditioned on the day gap to the NEXT frame (for the
last frame the "next frame" is the target, so its gap is the target offset) -> the final ConvLSTM
hidden state gets Step C's target-offset FiLM -> mid block -> decoder with skips from the LAST frame
only (earlier frames contribute through the recurrent state only).

Padding: sequences shorter than K are LEFT-padded (first real frame repeated) with mask=0; masked
steps leave the ConvLSTM state unchanged (zeros), so padding has no effect. The last frame is always real.
K=1 is a single ConvLSTM step on the last frame (no history).
"""
import torch
import torch.nn as nn

from img_film_unet import FiLMUNet


class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid, ksize=3):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, ksize, padding=ksize // 2)
        nn.init.zeros_(self.conv.bias)
        self.conv.bias.data[c_hid:2 * c_hid] = 1.0  # forget-gate bias

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class TemporalFiLMUNet(FiLMUNet):
    """Inherits Step C's stem/downs/film (target offset)/mid/ups/head so the encoder and decoder match."""

    def __init__(self, widths=(32, 64, 128, 192, 256, 256), film_hidden=64):
        super().__init__(widths, film_hidden)
        cb = widths[-1]
        self.film_step = nn.Sequential(nn.Linear(1, film_hidden), nn.SiLU(), nn.Linear(film_hidden, 2 * cb))
        nn.init.zeros_(self.film_step[-1].weight)
        nn.init.zeros_(self.film_step[-1].bias)
        self.cell = ConvLSTMCell(cb, cb)

    def forward(self, frames, gaps, mask, target_offset):
        """frames [B,K,3,H,W] in [0,1]; gaps [B,K] normalized gap to the next frame (last = target offset);
        mask [B,K] (1 = real frame); target_offset [B] normalized."""
        B, K = frames.shape[:2]
        x = frames.flatten(0, 1)
        h = self.stem(x * 2 - 1)
        skips = [h]
        for d in self.downs:
            h = d(h)
            skips.append(h)
        skips.pop()                                              # bottleneck is not a skip
        last = torch.arange(B, device=x.device) * K + (K - 1)
        skips = [s[last] for s in skips]                         # skips from the LAST frame only
        C = h.shape[1]
        z = h.view(B, K, C, h.shape[-2], h.shape[-1])
        gam, bet = self.film_step(gaps.reshape(-1, 1)).chunk(2, dim=1)
        z = z * (1 + gam.view(B, K, C, 1, 1)) + bet.view(B, K, C, 1, 1)
        hs = torch.zeros_like(z[:, 0])
        cs = torch.zeros_like(z[:, 0])
        for t in range(K):
            h_new, c_new = self.cell(z[:, t], (hs, cs))
            m = mask[:, t].view(B, 1, 1, 1)
            hs = m * h_new + (1 - m) * hs
            cs = m * c_new + (1 - m) * cs
        gamma, beta = self.film(target_offset.view(-1, 1)).chunk(2, dim=1)
        h = hs * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.mid(h)
        for up in self.ups:
            h = up(h, skips.pop())
        return torch.sigmoid(self.head(h))

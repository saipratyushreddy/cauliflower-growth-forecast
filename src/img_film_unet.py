"""
Single-frame image-to-image forecaster: CNN encoder -> FiLM (scale/shift from the scalar
day-offset) at the bottleneck -> U-Net decoder with skip connections.

Input : last observed image x in [0,1], [B,3,256,256]; day offset (normalized scalar) [B].
Output: predicted image in [0,1] via sigmoid, [B,3,256,256]. The output is generated
        directly (no residual-to-input shortcut), so copy-forward is not hard-wired in.

Encoder: stem + 4 downsampling stages (256 -> 128 -> 64 -> 32 -> 16), then one more
down to an 8x8 bottleneck. FiLM:  h = h * (1 + gamma(d)) + beta(d), where (gamma, beta) =
MLP(d); the last MLP layer is zero-initialised so training starts from "no conditioning".
GroupNorm (not BatchNorm) is used so small batches behave consistently in train/eval.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = gn(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.down = nn.Conv2d(cin, cin, 3, stride=2, padding=1)
        self.block = ResBlock(cin, cout)

    def forward(self, x):
        return self.block(self.down(x))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.block = ResBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class FiLMUNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 192, 256, 256), film_hidden=64):
        super().__init__()
        w = widths
        self.stem = ResBlock(3, w[0])                       # 256
        self.downs = nn.ModuleList([Down(w[i], w[i + 1]) for i in range(len(w) - 1)])  # 128,64,32,16,8
        self.film = nn.Sequential(nn.Linear(1, film_hidden), nn.SiLU(), nn.Linear(film_hidden, 2 * w[-1]))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.mid = ResBlock(w[-1], w[-1])
        self.ups = nn.ModuleList([Up(w[i + 1], w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.head = nn.Conv2d(w[0], 3, 1)

    def forward(self, x, day_offset):
        h = self.stem(x * 2 - 1)
        skips = [h]
        for d in self.downs:
            h = d(h)
            skips.append(h)
        skips.pop()                                         # bottleneck itself is not a skip
        gamma, beta = self.film(day_offset.view(-1, 1)).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.mid(h)
        for up in self.ups:
            h = up(h, skips.pop())
        return torch.sigmoid(self.head(h))


def gaussian_window(size=11, sigma=1.5, device="cpu", channels=3):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])
    return w.expand(channels, 1, size, size).contiguous()


def ssim_torch(x, y, window=None, C1=0.01 ** 2, C2=0.03 ** 2, reduce=True):
    """Differentiable SSIM (11x11 Gaussian, sigma 1.5, valid padding), images in [0,1]."""
    c = x.shape[1]
    w = window if window is not None else gaussian_window(device=x.device, channels=c)
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean() if reduce else s.mean(dim=(1, 2, 3))

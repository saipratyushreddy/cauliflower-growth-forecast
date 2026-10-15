"""
Building blocks of the sharpness experiments (all on the cheap single-frame model, so each ingredient can be ablated):

  ResFiLMUNet     Step C's FiLM U-Net whose output is the INPUT image plus a learned residual (zero-initialised head, so training starts exactly at copy-forward).
  VGGPerceptual   L1 distance between pretrained VGG16 feature maps (relu1_2, relu2_2, relu3_3, relu4_3) of prediction and target; also reported as a metric.
                  Needs torchvision's ImageNet VGG16 weights (downloaded once); `pretrained=False` exists for smoke tests only.
  sharpness metrics (no pretrained weights): per image pair, the ratio prediction/target of (a) mean Sobel gradient magnitude of the luma and (b) variance of the
                  Laplacian of the luma. 1 = as sharp as the real photo, < 1 = blurrier. Pooled ratios (mean of the numerators / mean of the denominators) are reported.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import laplace, sobel

from img_film_unet import FiLMUNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResFiLMUNet(FiLMUNet):
    """out = clamp(x + head(decoder), 0, 1); head zero-initialised => output == input at initialisation (copy-forward)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, day_offset):
        h = self.stem(x * 2 - 1)
        skips = [h]
        for d in self.downs:
            h = d(h)
            skips.append(h)
        skips.pop()
        gamma, beta = self.film(day_offset.view(-1, 1)).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.mid(h)
        for up in self.ups:
            h = up(h, skips.pop())
        return (x + self.head(h)).clamp(0, 1)


class VGGPerceptual(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        w = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.vgg16(weights=w).features.eval()
        self.blocks = nn.ModuleList([f[:4], f[4:9], f[9:16], f[16:23]])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pred, target, reduce=True):
        x, y = (pred - self.mean) / self.std, (target - self.mean) / self.std
        d = 0.0
        for b in self.blocks:
            x, y = b(x), b(y)
            d = d + (x - y).abs().mean(dim=(1, 2, 3))
        return d.mean() if reduce else d


def _luma(img_u8):
    return (img_u8.astype(np.float64) @ np.array([0.299, 0.587, 0.114]))


def sharpness_pair(pred_u8, target_u8):
    """(grad_pred, grad_target, lap_pred, lap_target) for one pair of uint8 HxWx3 images."""
    out = []
    for im in (pred_u8, target_u8):
        g = _luma(im)
        gm = np.hypot(sobel(g, axis=0), sobel(g, axis=1)).mean()
        out.append((gm, laplace(g).var()))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def sharpness_table(preds, targets):
    """preds/targets: [N,H,W,3] uint8 -> dict of pooled ratios and per-pair arrays."""
    a = np.array([sharpness_pair(p, t) for p, t in zip(preds, targets)])
    return {"grad_pred": a[:, 0], "grad_target": a[:, 1], "lap_pred": a[:, 2], "lap_target": a[:, 3],
            "grad_ratio": a[:, 0].mean() / a[:, 1].mean(), "lap_ratio": a[:, 2].mean() / a[:, 3].mean()}

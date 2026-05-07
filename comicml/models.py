"""
comicml_models.py — Shared model architecture definitions.

Used by both comicml_train.py (training) and comicscan/comicml.py (inference).
Kept in sync manually — changes here should be mirrored in comicscan/comicml.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 (imported for completeness)
from torchvision import models

# Default input resolution. Stored per-checkpoint so different models can use
# different sizes. 768 gives ~7 orig-px per feature cell at 600 DPI.
INPUT_SIZE = 512

# ImageNet normalization (ResNet-18 is pretrained on it)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CornerRegressor(nn.Module):
    """ResNet-18 backbone + linear head predicting 8 normalized corner coords."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 8)
        self.net = backbone

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class CornerHeatmapRegressor(nn.Module):
    """ResNet-18 encoder + deconv decoder predicting a 4-channel corner heatmap.

    At inference, coordinates are extracted via soft-argmax for sub-pixel
    accuracy. The decoder upsamples by 8× (stride 32 → stride 4),
    producing a heatmap at input_size / 4 resolution.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        up = self.decoder(feats)
        return self.final(up)  # [B, 4, H/4, W/4] — raw logits


def _make_heatmap_targets(corners_norm, hmap_size, sigma=2.0, device=None):
    """Build Gaussian heatmap targets from normalized corners.

    corners_norm: [B, 8] in [0, 1]. Returns [B, 4, H, W] float tensor.
    sigma is in heatmap-pixels; 2.0 at 192×192 covers ~5 px FWHM ≈ 20 orig-px.
    """
    B = corners_norm.shape[0]
    H = W = hmap_size
    coords = corners_norm.view(B, 4, 2)
    px = coords[..., 0] * (W - 1)
    py = coords[..., 1] * (H - 1)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    px = px.view(B, 4, 1, 1)
    py = py.view(B, 4, 1, 1)
    return torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))


def _soft_argmax_2d(heatmap, temperature=1.0):
    """Differentiable sub-pixel argmax.

    heatmap: [B, C, H, W] logits. Returns [B, C, 2] normalized coords in [0, 1]
    as (x, y).
    """
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1) / temperature
    probs = torch.softmax(flat, dim=-1).view(B, C, H, W)
    xs = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, W) / (W - 1)
    ys = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, H, 1) / (H - 1)
    x = (probs * xs).sum(dim=(2, 3))
    y = (probs * ys).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)  # [B, C, 2]

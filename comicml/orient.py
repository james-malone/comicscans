#!/usr/bin/env python3
"""
comicml.orient — Binary page-orientation classifier (upright vs upside-down).

OCR-based orientation (comicscans.detect_orientation) fails on text-sparse
pages — covers, splash pages, dark action panels — because there's no text to
read in either orientation. This model instead learns *visual* upright cues
(faces, figures, sky-at-top, panel layout) from the existing gt_rotate180
labels, so it works with zero text.

Training data is free: every ground-truth page already carries gt_rotate180,
so we make each page upright, then present it upright (label 0) or flipped
180° (label 1) on the fly — perfectly balanced, no extra labeling.

Usage:
    python3 -m comicml.orient train --epochs 20 --output orient_resnet18.pt
    python3 -m comicml.orient eval  --model orient_resnet18.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .models import IMAGENET_MEAN, IMAGENET_STD, pick_device

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR    = _PROJECT_ROOT / "models"
GROUND_TRUTH  = _PROJECT_ROOT / "data" / "ground_truth.json"
INPUT_SIZE    = 256          # orientation is a global cue — low res is plenty
HOLDOUT_DIRS  = ["DS9E20", "DS9E23", "DS9_1996_5", "VOY_5"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_upright_cache(entries, size):
    """Decode each page once, rotate to its true upright orientation, resize to
    `size`, and keep it in memory (small). Returns [(rgb_uint8, scan_dir), ...].

    Caching upright thumbnails up front means epochs don't re-decode the 600 DPI
    originals — the orientation label is then just an on-the-fly 180° flip."""
    cache = []
    for i, e in enumerate(entries):
        img = cv2.imread(e["filepath"])
        if img is None:
            continue
        if e.get("gt_rotate180"):
            img = cv2.rotate(img, cv2.ROTATE_180)   # now upright
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cache.append((img, e["scan_dir"].rsplit("/", 1)[-1]))
        if (i + 1) % 300 == 0:
            print(f"  cached {i+1}/{len(entries)}", flush=True)
    return cache


class OrientationDataset(Dataset):
    """Yields (image_tensor, label) where label=1 means the presented image is
    upside-down. Each upright thumbnail is flipped 180° with p=0.5 so the two
    classes are balanced. Augmentation is orientation-preserving only (no
    vertical changes): horizontal mirror, brightness/contrast, small rotation."""

    def __init__(self, cache, augment=False):
        self.cache = cache
        self.augment = augment
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        img = self.cache[idx][0].copy()
        label = 0
        if random.random() < 0.5:
            img = np.ascontiguousarray(np.rot90(img, 2))   # 180° → upside-down
            label = 1
        if self.augment:
            if random.random() < 0.5:                       # mirror keeps up/down
                img = np.ascontiguousarray(img[:, ::-1])
            if random.random() < 0.5:
                alpha = 1.0 + (random.random() - 0.5) * 0.3
                beta = (random.random() - 0.5) * 30
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.3:                       # small skew, orientation intact
                h, w = img.shape[:2]
                ang = (random.random() - 0.5) * 8
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return self.normalize(t), torch.tensor([float(label)])


def _build_model(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def _split(entries):
    hold = set(HOLDOUT_DIRS)
    tr = [e for e in entries if e["scan_dir"].rsplit("/", 1)[-1] not in hold]
    ho = [e for e in entries if e["scan_dir"].rsplit("/", 1)[-1] in hold]
    return tr, ho


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def _accuracy(model, device, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(device))
            pred = (torch.sigmoid(out).cpu() > 0.5).float()
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def train(args):
    device = pick_device()
    print(f"Device: {device}")
    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    entries = [e for e in json.loads(GROUND_TRUTH.read_text()) if e.get("gt_corners")]
    tr_e, ho_e = _split(entries)
    print(f"Caching upright thumbnails ({INPUT_SIZE}px): {len(tr_e)} train, {len(ho_e)} holdout…")
    tr_cache = _load_upright_cache(tr_e, INPUT_SIZE)
    ho_cache = _load_upright_cache(ho_e, INPUT_SIZE)
    print(f"Train {len(tr_cache)} / Holdout {len(ho_cache)} pages from {HOLDOUT_DIRS}")

    tr_loader = DataLoader(OrientationDataset(tr_cache, augment=True),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    ho_loader = DataLoader(OrientationDataset(ho_cache, augment=False),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = _build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    out_path = args.output if Path(args.output).is_absolute() else MODELS_DIR / args.output
    log_path = out_path.with_name(out_path.stem + "_log.jsonl")
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time(); running = 0.0; n = 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = loss_fn(model(imgs), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * imgs.size(0); n += imgs.size(0)
        acc = _accuracy(model, device, ho_loader)
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "input_size": INPUT_SIZE,
                        "holdout_dirs": HOLDOUT_DIRS, "epoch": epoch, "val_acc": acc}, out_path)
        with open(log_path, "a") as lf:
            lf.write(json.dumps({"epoch": epoch + 1, "total_epochs": args.epochs,
                                 "train_loss": round(running / max(n, 1), 5),
                                 "val_acc": round(acc, 4), "is_best": is_best}) + "\n")
        print(f"epoch {epoch+1:>3}/{args.epochs}  loss={running/max(n,1):.4f}  "
              f"holdout_acc={acc*100:.2f}%  ({time.time()-t0:.1f}s){'  *' if is_best else ''}",
              flush=True)
    print(f"\nBest holdout orientation accuracy: {best_acc*100:.2f}%")
    print(f"Saved to {out_path}")


def load_orient_model(model_path, device=None):
    device = device or pick_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = _build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._input_size = ckpt.get("input_size", INPUT_SIZE)
    return model, device


def predict_upside_down(model, device, image_bgr):
    """Return (is_upside_down, probability) for a BGR scan as given."""
    size = getattr(model, "_input_size", INPUT_SIZE)
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t).unsqueeze(0).to(device)
    with torch.no_grad():
        p = float(torch.sigmoid(model(t)).item())
    return p > 0.5, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train", help="Train the orientation classifier")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--num-workers", type=int, default=4)
    pt.add_argument("--seed", type=int, default=137)
    pt.add_argument("--output", default="orient_resnet18.pt")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)


if __name__ == "__main__":
    main()

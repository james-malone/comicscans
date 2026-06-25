#!/usr/bin/env python3
"""
comicml.train — Training and evaluation CLI for the comic page corner CNN.

Paths are resolved relative to the project root (two levels up from this
file):
    <root>/data/ground_truth.json   — training labels
    <root>/models/<name>.pt         — output checkpoints + JSONL logs

Trained models land in <root>/models/ so inference (comicml.inference) picks
them up automatically without any extra config.

Usage (run as a module — required for the relative import of .models):
    python3 -m comicml.train train \\
        --input-size 768 --epochs 280 --warm-restarts 40 \\
        --seed 137 --output comicml_model_reg_768_1420pg_e280.pt

    python3 -m comicml.train eval --model comicml_model_reg_768_1420pg_e280.pt

    python3 -m comicml.train predict path/to/Scan.jpeg \\
        --model comicml_model_reg_768_1420pg_e280.pt
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
from torchvision import transforms

from .models import (
    INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CornerRegressor, CornerHeatmapRegressor,
    _make_heatmap_targets, _soft_argmax_2d,
)

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root.
#   <root>/
#     comicml/train.py     ← __file__
#     models/              ← .pt + _log.jsonl live here
#     data/                ← ground_truth.json
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR    = _PROJECT_ROOT / "models"
DATA_DIR      = _PROJECT_ROOT / "data"

GROUND_TRUTH_FILE  = DATA_DIR / "ground_truth.json"
DEFAULT_MODEL_FILE = MODELS_DIR / "comicml_model_reg_768_1420pg_e280.pt"


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def _load_entries():
    if not GROUND_TRUTH_FILE.exists():
        print(f"Missing ground truth file: {GROUND_TRUTH_FILE}")
        print("Run `python3 comiceval.py collect <scan-root>...` to regenerate it.")
        sys.exit(1)
    return json.loads(GROUND_TRUTH_FILE.read_text())


def _split_entries(entries, train_dirs, holdout_dirs):
    """Split ground truth entries into train and holdout sets by scan_dir."""
    train, holdout = [], []
    train_set = set(train_dirs)
    holdout_set = set(holdout_dirs)
    overlap = train_set & holdout_set
    if overlap:
        raise ValueError(f"scan dirs appear in both train and holdout: {sorted(overlap)}")
    for e in entries:
        name = e["scan_dir"].rsplit("/", 1)[-1]
        if name in train_set:
            train.append(e)
        elif name in holdout_set:
            holdout.append(e)
    return train, holdout


def _resolve_train_dirs(entries, holdout_dirs, explicit=None):
    """Training dir list. An explicit list wins; otherwise default to EVERY
    collected dir except the holdout.

    Deriving the default from the ground truth (rather than a hand-maintained
    allowlist) means freshly collected comics are trained on automatically —
    a stale allowlist would silently drop exactly the data you just added.
    """
    if explicit:
        return explicit
    holdout_set = set(holdout_dirs)
    all_dirs = sorted({e["scan_dir"].rsplit("/", 1)[-1] for e in entries})
    return [d for d in all_dirs if d not in holdout_set]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PageCornerDataset(Dataset):
    """Loads (image, 8-dim normalized corner target) pairs.

    Image is resized to input_size × input_size (ignoring aspect). Corners are
    normalized to [0, 1] using original image dims so the network learns
    position as a fraction of each axis, independent of the aspect-distort.

    Augmentation at train time: horizontal flip, affine jitter, brightness/
    contrast, color jitter, Gaussian noise, random erasing.
    """

    def __init__(self, entries, augment=False, input_size=INPUT_SIZE):
        self.entries = entries
        self.augment = augment
        self.input_size = input_size
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = cv2.imread(entry["filepath"])
        if img is None:
            raise RuntimeError(f"Could not read {entry['filepath']}")
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)

        H, W = img.shape[:2]
        corners = np.array(entry["gt_corners"], dtype=np.float32)

        norm = corners.copy()
        norm[:, 0] /= W
        norm[:, 1] /= H

        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                flipped = norm.copy()
                flipped[:, 0] = 1.0 - flipped[:, 0]
                norm = np.array([flipped[1], flipped[0], flipped[3], flipped[2]])

            if random.random() < 0.5:
                sx = 1.0 + (random.random() - 0.5) * 0.06
                sy = 1.0 + (random.random() - 0.5) * 0.06
                tx = (random.random() - 0.5) * 0.04
                ty = (random.random() - 0.5) * 0.04
                h, w = img.shape[:2]
                M = np.array([[sx, 0, tx * w], [0, sy, ty * h]], dtype=np.float32)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                norm[:, 0] = norm[:, 0] * sx + tx
                norm[:, 1] = norm[:, 1] * sy + ty
                norm = np.clip(norm, 0.0, 1.0)

            if random.random() < 0.5:
                alpha = 1.0 + (random.random() - 0.5) * 0.2
                beta = (random.random() - 0.5) * 20
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            if random.random() < 0.3:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= 1.0 + (random.random() - 0.5) * 0.4
                hsv[:, :, 0] += (random.random() - 0.5) * 10
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                hsv[:, :, 0] = hsv[:, :, 0] % 180
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            if random.random() < 0.3:
                noise = np.random.normal(0, 5, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            if random.random() < 0.3:
                h, w = img.shape[:2]
                eh = random.randint(h // 10, h // 4)
                ew = random.randint(w // 10, w // 4)
                ey = random.randint(0, h - eh)
                ex = random.randint(0, w - ew)
                img[ey:ey+eh, ex:ex+ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = self.normalize(img_t)
        target = torch.from_numpy(norm.flatten()).float()
        meta = {
            "filepath": entry["filepath"],
            "orig_w": W,
            "orig_h": H,
            "scan_dir": entry["scan_dir"],
            "page_index": entry["page_index"],
        }
        return img_t, target, meta


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

def _corner_px_error(pred_norm, target_norm, orig_w, orig_h):
    """Mean corner distance in original-image pixels for a batch."""
    B = pred_norm.shape[0]
    pred = pred_norm.view(B, 4, 2)
    tgt = target_norm.view(B, 4, 2)
    scale = torch.stack([orig_w.float(), orig_h.float()], dim=1).view(B, 1, 2)
    pred_px = pred * scale
    tgt_px = tgt * scale
    d = torch.sqrt(((pred_px - tgt_px) ** 2).sum(dim=2))
    return d.mean().item(), d


# ---------------------------------------------------------------------------
# Model loading (for eval)
# ---------------------------------------------------------------------------

def _load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "regression")
    if model_type == "heatmap":
        model = CornerHeatmapRegressor(pretrained=False).to(device)
    else:
        model = CornerRegressor(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._input_size = ckpt.get("input_size", INPUT_SIZE)
    model._model_type = model_type
    return model, ckpt


def _predict_single(model, device, image_bgr, size):
    H, W = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t).unsqueeze(0).to(device)
    model_type = getattr(model, "_model_type", "regression")
    with torch.no_grad():
        out = model(t)
        if model_type == "heatmap":
            coords = _soft_argmax_2d(out).view(1, 4, 2)
            pred = coords.cpu().numpy().reshape(4, 2)
        else:
            pred = out.cpu().numpy().reshape(4, 2)
    pred[:, 0] *= W
    pred[:, 1] *= H
    return pred.tolist()


def _predict_corners(model, device, image_bgr, rotate180=False, input_size=None, tta=True):
    if rotate180:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
    size = input_size if input_size is not None else getattr(model, "_input_size", INPUT_SIZE)
    pred_a = _predict_single(model, device, image_bgr, size)
    if not tta:
        return pred_a
    W = image_bgr.shape[1]
    flipped = cv2.flip(image_bgr, 1)
    pred_b = _predict_single(model, device, flipped, size)
    pred_b = [[W - x, y] for x, y in pred_b]
    pred_b = [pred_b[1], pred_b[0], pred_b[3], pred_b[2]]
    return [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in zip(pred_a, pred_b)]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Relative output paths land in <root>/models/ so inference picks them up
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = MODELS_DIR / args.output
    else:
        output_path = DEFAULT_MODEL_FILE
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix("").with_name(output_path.stem + "_log.jsonl")

    # Guard: refuse to start if another training process appears to be writing
    # the same log file (within the last 5 minutes). Prevents two runs
    # clobbering each other's output (the .pt file and interleaved JSONL).
    if log_path.exists():
        age = time.time() - log_path.stat().st_mtime
        if age < 300:
            raise SystemExit(
                f"ERROR: {log_path} was modified {age:.0f}s ago — another training "
                f"run may be in progress and writing to the same output.\n"
                f"If you're sure it's safe, delete or rename the log file and retry:\n"
                f"  rm {log_path}"
            )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")

    entries = _load_entries()
    holdout_dirs = args.holdout.split(",")
    train_dirs = _resolve_train_dirs(entries, holdout_dirs,
                                     args.train.split(",") if args.train else None)
    train_entries, holdout_entries = _split_entries(entries, train_dirs, holdout_dirs)

    train_entries = [e for e in train_entries if e["has_correction"]]
    holdout_entries = [e for e in holdout_entries if e["has_correction"]]

    print(f"Train: {len(train_entries)} pages from {train_dirs}")
    print(f"Holdout: {len(holdout_entries)} pages from {holdout_dirs}")

    input_size = args.input_size
    print(f"Input resolution: {input_size}×{input_size}")
    train_ds = PageCornerDataset(train_entries, augment=True, input_size=input_size)
    val_ds   = PageCornerDataset(holdout_entries, augment=False, input_size=input_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    model_type = "heatmap" if args.heatmap else "regression"
    print(f"Model type: {model_type}")
    if model_type == "heatmap":
        model = CornerHeatmapRegressor(pretrained=True).to(device)
        hmap_size = input_size // 4
        print(f"Heatmap resolution: {hmap_size}×{hmap_size}  σ={args.hmap_sigma}")
    else:
        model = CornerRegressor(pretrained=True).to(device)
        hmap_size = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.warm_restarts > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.warm_restarts, T_mult=2)
        print(f"LR schedule: cosine warm restarts T_0={args.warm_restarts} T_mult=2")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"LR schedule: cosine annealing T_max={args.epochs}")

    l1_loss  = nn.SmoothL1Loss()
    mse_loss = nn.MSELoss()

    def forward_loss(imgs, targets):
        out = model(imgs)
        if model_type == "heatmap":
            coords = _soft_argmax_2d(out).view(imgs.size(0), 8)
            coord_loss = l1_loss(coords, targets)
            target_hmap = _make_heatmap_targets(
                targets, hmap_size, sigma=args.hmap_sigma, device=device)
            probs = torch.softmax(out.view(out.size(0), 4, -1), dim=-1).view_as(out)
            tgt_sum = target_hmap.sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
            target_probs = target_hmap / tgt_sum
            reg = mse_loss(probs, target_probs)
            loss = coord_loss + args.hmap_reg * reg
        else:
            loss = l1_loss(out, targets)
            coords = out
        return loss, coords

    best_val_px = float("inf")
    best_epoch = 0

    # Clear any previous log for this output path
    log_path.write_text("")
    print(f"Training log → {log_path}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_px_sum = 0.0
        train_n = 0
        for imgs, targets, meta in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            loss, coords = forward_loss(imgs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            orig_w = meta["orig_w"].to(device)
            orig_h = meta["orig_h"].to(device)
            px, _ = _corner_px_error(coords.detach(), targets, orig_w, orig_h)
            train_px_sum += px * imgs.size(0)
            train_n += imgs.size(0)
        scheduler.step()

        model.eval()
        val_px_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for imgs, targets, meta in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                _, coords = forward_loss(imgs, targets)
                orig_w = meta["orig_w"].to(device)
                orig_h = meta["orig_h"].to(device)
                px, _ = _corner_px_error(coords, targets, orig_w, orig_h)
                val_px_sum += px * imgs.size(0)
                val_n += imgs.size(0)

        train_px  = train_px_sum / max(train_n, 1)
        val_px    = val_px_sum / max(val_n, 1)
        tl        = train_loss / max(train_n, 1)
        elapsed   = time.time() - t0
        is_best   = val_px < best_val_px

        if is_best:
            best_val_px = val_px
            best_epoch = epoch
            torch.save({
                "model_state":  model.state_dict(),
                "input_size":   input_size,
                "model_type":   model_type,
                "hmap_sigma":   args.hmap_sigma,
                "epoch":        epoch,
                "val_px":       val_px,
                "train_dirs":   train_dirs,
                "holdout_dirs": holdout_dirs,
                "seed":         args.seed,
            }, output_path)

        marker = " *" if is_best else ""
        print(f"epoch {epoch+1:>3d}/{args.epochs}  "
              f"train_loss={tl:.5f}  "
              f"train_px={train_px:7.2f}  val_px={val_px:7.2f}  "
              f"({elapsed:.1f}s){marker}", flush=True)

        # Append structured log entry for the dashboard
        with open(log_path, "a") as lf:
            lf.write(json.dumps({
                "epoch":       epoch + 1,
                "total_epochs": args.epochs,
                "train_loss":  round(tl, 6),
                "train_px":    round(train_px, 3),
                "val_px":      round(val_px, 3),
                "is_best":     is_best,
                "elapsed":     round(elapsed, 1),
            }) + "\n")

        if args.patience and epoch - best_epoch >= args.patience:
            print(f"Early stopping: val_px has not improved since epoch "
                  f"{best_epoch + 1} ({args.patience} epochs ago)")
            break

    print(f"\nBest holdout mean corner error: {best_val_px:.2f} px")
    print(f"Model saved to {output_path}")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = args.model if args.model else DEFAULT_MODEL_FILE
    if not Path(model_path).is_absolute():
        model_path = MODELS_DIR / model_path
    model, ckpt = _load_model(model_path, device)
    print(f"Loaded {model_path}  (epoch {ckpt.get('epoch','?')}, "
          f"best val {ckpt.get('val_px','?'):.2f} px)")

    entries = _load_entries()
    holdout_dirs = ckpt.get("holdout_dirs", [])
    if args.all:
        eval_entries = [e for e in entries if e["has_correction"]]
        print(f"Evaluating on ALL {len(eval_entries)} corrected pages")
    else:
        _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []), holdout_dirs)
        eval_entries = [e for e in eval_entries if e["has_correction"]]
        print(f"Evaluating on {len(eval_entries)} holdout pages ({holdout_dirs})")

    cnn_dists = []
    per_dir   = {}
    per_page  = []

    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None:
            continue
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        size = getattr(model, "_input_size", INPUT_SIZE)
        pred_a = _predict_single(model, device, img, size)
        W = img.shape[1]
        flipped = cv2.flip(img, 1)
        pred_b = _predict_single(model, device, flipped, size)
        pred_b = [[W - x, y] for x, y in pred_b]
        pred_b = [pred_b[1], pred_b[0], pred_b[3], pred_b[2]]
        pred = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in zip(pred_a, pred_b)]

        gt = entry["gt_corners"]
        d = float(np.mean([np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred, gt)]))
        cnn_dists.append(d)
        name = entry["scan_dir"].rsplit("/", 1)[-1]
        per_dir.setdefault(name, []).append(d)
        per_page.append({
            "filepath":   entry["filepath"],
            "scan_dir":   entry["scan_dir"],
            "page_index": entry["page_index"],
            "error_px":   round(d, 2),
            "pred":       [[round(x, 1), round(y, 1)] for x, y in pred],
            "gt":         gt,
        })

    arr = np.array(cnn_dists)
    print(f"\n=== CNN Corner Regressor ===")
    print(f"Pages:   {len(arr)}")
    print(f"Mean:    {arr.mean():.2f} px")
    print(f"Median:  {np.median(arr):.2f} px")
    print(f"P95:     {np.percentile(arr, 95):.2f} px")
    print(f"Max:     {arr.max():.2f} px")
    print(f"\nPer directory:")
    for name in sorted(per_dir):
        v = np.array(per_dir[name])
        print(f"  {name:<20s}  n={len(v):<3d}  mean={v.mean():7.2f}  "
              f"median={np.median(v):7.2f}  max={v.max():7.2f} px")

    if args.output_json:
        out = Path(args.output_json)
        out.write_text(json.dumps(per_page, indent=2))
        print(f"\nPer-page results → {out}")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_cli(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = args.model if args.model else DEFAULT_MODEL_FILE
    if not Path(model_path).is_absolute():
        model_path = MODELS_DIR / model_path
    model, _ = _load_model(model_path, device)
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read {args.image}")
        sys.exit(1)
    corners = _predict_corners(model, device, img)
    print(json.dumps({"corners": corners, "image_size": [img.shape[1], img.shape[0]]}, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a new corner regression model")
    p_train.add_argument("--train", default=None,
        help="Comma-separated scan dir names for training "
             "(default: all collected dirs except the holdout)")
    p_train.add_argument("--holdout", default="DS9E20,DS9E23,DS9_1996_5",
        help="Comma-separated scan dir names held out for validation")
    p_train.add_argument("--epochs", type=int, default=120)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--input-size", type=int, default=768,
        help="CNN input resolution (default 768; 512 is faster but less accurate)")
    p_train.add_argument("--warm-restarts", type=int, default=40,
        help="Cosine warm restart period in epochs (0 = plain cosine decay)")
    p_train.add_argument("--seed", type=int, default=137)
    p_train.add_argument("--patience", type=int, default=0,
                         help="stop early if val_px has not improved in N epochs (0 = off)")
    p_train.add_argument("--output", type=str, default=None,
        help="Output filename (relative → placed in <root>/models/). "
             f"Default: {DEFAULT_MODEL_FILE.name}")
    p_train.add_argument("--heatmap", action="store_true",
        help="Use heatmap regression head instead of direct coord regression")
    p_train.add_argument("--hmap-sigma", type=float, default=2.0)
    p_train.add_argument("--hmap-reg",   type=float, default=1.0)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate a trained model on its holdout set")
    p_eval.add_argument("--model", default=None,
        help="Model filename or path (relative → looked up in <root>/models/)")
    p_eval.add_argument("--all", action="store_true",
        help="Evaluate on ALL corrected pages, not just holdout")
    p_eval.add_argument("--output-json", default=None,
        help="Write per-page results to this JSON file (for dashboard import)")

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Predict corners for a single image")
    p_pred.add_argument("image")
    p_pred.add_argument("--model", default=None)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "predict":
        predict_cli(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decompose hybrid-vs-GT error into systematic inward bias + page-localization noise.

For each holdout page:
  signed_dx = GT.x - pred.x
  signed_dy = GT.y - pred.y
  inward_x  = signed_dx * INWARD_X[corner]   (positive = GT is inward of pred)
  inward_y  = signed_dy * INWARD_Y[corner]

Mean of these tells us the systematic direction the model is off. If GT is
systematically inward of pred, we can post-shift inward by that amount.
"""
import numpy as np
import cv2
import torch

from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners_ensemble, refine_corners_linefit,
    _get_cached_ensemble, MODEL_FILE,
)

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
INWARD_X = [+1, -1, -1, +1]
INWARD_Y = [+1, +1, -1, -1]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ensemble, _ = _get_cached_ensemble()
    _, ckpt = _load_model(MODEL_FILE, device)

    entries = _load_entries()
    _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []),
                                     ckpt.get("holdout_dirs", []))
    eval_entries = [e for e in eval_entries if e["has_correction"]]
    print(f"Ensemble: {len(ensemble)} models  Holdout: {len(eval_entries)} pages\n")

    # Per-corner lists
    inward_x = [[] for _ in range(4)]
    inward_y = [[] for _ in range(4)]
    abs_err = [[] for _ in range(4)]

    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None: continue
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        dpi = entry.get("dpi", 600)
        cnn, dis = predict_corners_ensemble(ensemble, device, img)
        pred = refine_corners_linefit(img, cnn, dpi=dpi, tta_disagreements=dis)
        gt = entry["gt_corners"]
        for i in range(4):
            sdx = gt[i][0] - pred[i][0]
            sdy = gt[i][1] - pred[i][1]
            inward_x[i].append(sdx * INWARD_X[i])
            inward_y[i].append(sdy * INWARD_Y[i])
            abs_err[i].append(np.hypot(sdx, sdy))

    print(f"{'corner':>6s}  {'|err|_mean':>10s}  {'inX_mean':>9s}  {'inY_mean':>9s}  "
          f"{'inX_med':>8s}  {'inY_med':>8s}  {'pct_inw':>8s}")
    print("-" * 75)
    for i, name in enumerate(CORNER_NAMES):
        ix = np.array(inward_x[i])
        iy = np.array(inward_y[i])
        ae = np.array(abs_err[i])
        net = (ix + iy) / np.sqrt(2)
        pct = 100 * (net > 0).mean()
        print(f"{name:>6s}  {ae.mean():>10.2f}  "
              f"{ix.mean():>+9.2f}  {iy.mean():>+9.2f}  "
              f"{np.median(ix):>+8.2f}  {np.median(iy):>+8.2f}  "
              f"{pct:>7.1f}%")

    all_ix = np.concatenate(inward_x)
    all_iy = np.concatenate(inward_y)
    all_net = (all_ix + all_iy) / np.sqrt(2)
    print(f"\nAll corners (n={len(all_ix)}):")
    print(f"  Mean inward-X shift GT needs vs pred: {all_ix.mean():+.2f} px")
    print(f"  Mean inward-Y shift GT needs vs pred: {all_iy.mean():+.2f} px")
    print(f"  Median inward-X:                      {np.median(all_ix):+.2f} px")
    print(f"  Median inward-Y:                      {np.median(all_iy):+.2f} px")
    print(f"  Pages where GT is inward of pred:     {100*(all_net>0).mean():.1f}%")

    # What would applying a fixed post-shift achieve?
    print("\n--- If we post-shift prediction inward by (+X, +Y) px ---")
    print(f"{'shiftX':>7s}  {'shiftY':>7s}  {'new_mean_err':>13s}  {'Δmean':>7s}")
    baseline = np.concatenate([np.array(e) for e in abs_err]).mean()
    print(f"{'0':>7s}  {'0':>7s}  {baseline:>13.2f}  {'+0.00':>7s}")
    # Try a few candidate shifts
    candidates = [
        (0, 5), (0, 10), (0, 15), (0, 20),
        (5, 10), (5, 15),
        (int(round(all_ix.mean())), int(round(all_iy.mean()))),
        (int(round(np.median(all_ix))), int(round(np.median(all_iy)))),
    ]
    for sx, sy in candidates:
        errs = []
        for i in range(4):
            # Applying an inward shift of (sx, sy) to the prediction shifts the
            # prediction in the INWARD direction for that corner, reducing the
            # GT-pred residual by (sx, sy) along inward axes.
            ix = np.array(inward_x[i]) - sx
            iy = np.array(inward_y[i]) - sy
            errs.extend(np.hypot(ix * INWARD_X[i], iy * INWARD_Y[i]))
        # Since INWARD_X/Y are ±1, hypot is preserved. Simplify:
        errs = []
        for i in range(4):
            ix = np.array(inward_x[i]) - sx
            iy = np.array(inward_y[i]) - sy
            errs.extend(np.hypot(ix, iy))
        m = np.mean(errs)
        print(f"{sx:>+7d}  {sy:>+7d}  {m:>13.2f}  {m-baseline:>+7.2f}")


if __name__ == "__main__":
    main()

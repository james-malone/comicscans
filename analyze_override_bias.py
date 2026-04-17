#!/usr/bin/env python3
"""Measure whether user overrides systematically move corners inward.

For each page with both an auto-detection and a user override, compute
per-corner displacement, decomposed into "inward" (toward page center)
vs "tangential" components. Reports mean inward shift per corner and
overall to confirm (or falsify) the hypothesis that manual corrections
are primarily a uniform inward crop.
"""
import json
import glob
import numpy as np

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
# Unit vectors pointing INWARD (toward page center) for each corner
# assuming a ~rectangular page with TL at top-left etc.
INWARD_X = [+1, -1, -1, +1]  # TL, TR, BR, BL
INWARD_Y = [+1, +1, -1, -1]

def main():
    sessions = sorted(glob.glob("raw-scans/*/.comicscans_session.json"))
    print(f"Found {len(sessions)} session files\n")

    per_dir_totals = {}
    global_in_x = [[] for _ in range(4)]   # per corner
    global_in_y = [[] for _ in range(4)]
    global_net_inward = [[] for _ in range(4)]  # signed magnitude along inward axis

    for sess_path in sessions:
        with open(sess_path) as f:
            s = json.load(f)
        issue = sess_path.split("/")[-2]
        det = s.get("detections", {})
        ov = s.get("overrides", {})
        dir_in = [[] for _ in range(4)]
        for pid, override in ov.items():
            if pid not in det:
                continue
            d_corners = det[pid]["corners"]
            o_corners = override["corners"]
            if len(d_corners) != 4 or len(o_corners) != 4:
                continue
            for i in range(4):
                dx = o_corners[i][0] - d_corners[i][0]
                dy = o_corners[i][1] - d_corners[i][1]
                # project onto inward unit vectors
                inward_shift_x = dx * INWARD_X[i]
                inward_shift_y = dy * INWARD_Y[i]
                global_in_x[i].append(inward_shift_x)
                global_in_y[i].append(inward_shift_y)
                # scalar net inward along the 45° diagonal
                net = (inward_shift_x + inward_shift_y) / np.sqrt(2)
                global_net_inward[i].append(net)
                dir_in[i].append(net)
        if any(dir_in):
            per_dir_totals[issue] = dir_in

    # Global per-corner stats
    print(f"{'corner':>6s}  {'n':>4s}  {'inX_mean':>9s}  {'inY_mean':>9s}  "
          f"{'netIn_mean':>10s}  {'netIn_med':>9s}  {'pct_inward':>10s}")
    print("-" * 75)
    for i, name in enumerate(CORNER_NAMES):
        ix = np.array(global_in_x[i])
        iy = np.array(global_in_y[i])
        net = np.array(global_net_inward[i])
        pct_inward = 100.0 * (net > 0).mean()
        print(f"{name:>6s}  {len(net):>4d}  "
              f"{ix.mean():>+9.2f}  {iy.mean():>+9.2f}  "
              f"{net.mean():>+10.2f}  {np.median(net):>+9.2f}  "
              f"{pct_inward:>9.1f}%")

    # Aggregate across all corners
    all_net = np.concatenate(global_net_inward)
    all_ix = np.concatenate(global_in_x)
    all_iy = np.concatenate(global_in_y)
    print(f"\nAll corners (n={len(all_net)}):")
    print(f"  mean inward-x: {all_ix.mean():+.2f} px")
    print(f"  mean inward-y: {all_iy.mean():+.2f} px")
    print(f"  mean net inward (45°): {all_net.mean():+.2f} px  (median {np.median(all_net):+.2f})")
    print(f"  Fraction of corners moved inward: {100*(all_net>0).mean():.1f}%")
    print(f"  P25: {np.percentile(all_net,25):+.1f}  P75: {np.percentile(all_net,75):+.1f}  "
          f"P90: {np.percentile(all_net,90):+.1f}")

    # Per-issue summary (net inward mean across all corners)
    print(f"\nPer-issue mean net-inward (px):")
    for issue in sorted(per_dir_totals):
        merged = np.concatenate([np.array(x) for x in per_dir_totals[issue] if x])
        if len(merged) == 0: continue
        print(f"  {issue:<22s}  n={len(merged):>4d}  mean={merged.mean():+6.2f}  median={np.median(merged):+6.2f}")


if __name__ == "__main__":
    main()

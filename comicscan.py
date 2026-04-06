#!/usr/bin/env python3
"""
comicscan.py — Process raw comic book scans into clean, aligned page images.

Usage:
    python3 comicscan.py <input_dir> [--output <output_dir>] [--quality 93] [--preview]

Rotation options (for upside-down pages):
    --rotate 2,4,6,8       Rotate specific pages 180°
    --rotate-range 2-14    Rotate a range of pages 180°
    --rotate-even           Rotate all even-numbered pages 180°
    --rotate-odd            Rotate all odd-numbered pages 180°

Example:
    python3 comicscan.py raw-scans/DS9E17/
    python3 comicscan.py raw-scans/DS9E17/ --output output/DS9E17 --rotate-even
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def parse_scan_filename(filename):
    """Parse scan filenames and return the page index.

    'Scan.jpeg' -> 0, 'Scan 1.jpeg' -> 1, 'Scan 35.jpeg' -> 35
    """
    stem = Path(filename).stem
    match = re.match(r'^Scan(?:\s+(\d+))?$', stem)
    if not match:
        return None
    return int(match.group(1)) if match.group(1) else 0


def load_scans(input_dir):
    """Load and sort scan files from the input directory."""
    input_path = Path(input_dir)
    scans = []

    for f in input_path.iterdir():
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'):
            idx = parse_scan_filename(f.name)
            if idx is not None:
                scans.append((idx, f))

    if not scans:
        print(f"Error: No scan files found in {input_dir}")
        sys.exit(1)

    scans.sort(key=lambda x: x[0])
    print(f"Found {len(scans)} scan files (pages 0-{scans[-1][0]})")

    # Check for gaps
    indices = [s[0] for s in scans]
    expected = list(range(max(indices) + 1))
    missing = set(expected) - set(indices)
    if missing:
        print(f"Warning: Missing page indices: {sorted(missing)}")

    return scans


def get_source_dpi(filepath):
    """Read DPI from the source image."""
    try:
        img = Image.open(filepath)
        dpi = img.info.get('dpi', (300, 300))
        return int(dpi[0])
    except Exception:
        return 300


def parse_rotate_pages(args, total_pages):
    """Build a set of page indices to rotate 180° from CLI arguments."""
    pages_to_rotate = set()

    if args.rotate:
        for part in args.rotate.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                pages_to_rotate.update(range(int(start), int(end) + 1))
            else:
                pages_to_rotate.add(int(part))

    if args.rotate_range:
        start, end = args.rotate_range.split('-', 1)
        pages_to_rotate.update(range(int(start), int(end) + 1))

    if args.rotate_even:
        pages_to_rotate.update(range(0, total_pages, 2))

    if args.rotate_odd:
        pages_to_rotate.update(range(1, total_pages, 2))

    return pages_to_rotate


# ---------------------------------------------------------------------------
# Two-page bleed / spine detection
# ---------------------------------------------------------------------------

def detect_spine(gray, content_bottom):
    """Detect a comic spine/binding shadow in the scan.

    When a comic is opened on a scanner, the spine creates a dark vertical
    band (mean brightness < 60, std < 20) that separates the two visible
    pages. Returns the column position of the spine center, or None.
    """
    h = min(content_bottom, gray.shape[0])
    w = gray.shape[1]

    # Search 10-90% of the image width (covers both left and right spine positions,
    # including cases where the page was rotated 180° moving spine to the other side)
    search_start = int(w * 0.1)
    search_end = int(w * 0.9)

    # Analyze each column in the search region
    # The spine is a vertical band where brightness is consistently very low
    # with very low variance (uniform dark shadow)
    best_spine = None
    best_spine_width = 0

    col = search_start
    while col < search_end:
        strip = gray[:h, col]
        col_mean = strip.mean()
        col_std = strip.std()

        # Spine characteristics: very dark, very uniform
        if col_mean < 60 and col_std < 20:
            # Found a potential spine column - measure the full width
            spine_start = col
            while col < search_end:
                s = gray[:h, col]
                if s.mean() >= 60 or s.std() >= 20:
                    break
                col += 1
            spine_end = col
            spine_width = spine_end - spine_start

            # A real spine is at least 15 pixels wide
            if spine_width >= 15 and spine_width > best_spine_width:
                best_spine = (spine_start + spine_end) // 2
                best_spine_width = spine_width
        else:
            col += 1

    if best_spine is not None:
        return best_spine, best_spine_width

    return None


# ---------------------------------------------------------------------------
# Skew detection
# ---------------------------------------------------------------------------

def detect_skew(gray_image):
    """Detect skew angle using Hough lines on panel borders and page edges.

    Focuses on the outer border region of the page. Only returns small
    corrections (0.1° - 5°). Scans on a flatbed never need large rotation.
    """
    h, w = gray_image.shape

    # Focus on the outer 15% border strips where panel edges and page
    # edges give the best skew signal
    mask = np.zeros_like(gray_image)
    border_w = max(w // 7, 50)
    border_h = max(h // 7, 50)
    mask[:border_h, :] = 255       # top strip
    mask[-border_h:, :] = 255      # bottom strip
    mask[:, :border_w] = 255       # left strip
    mask[:, -border_w:] = 255      # right strip

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    edges = cv2.bitwise_and(edges, mask)

    # Require long lines for reliable angle detection
    min_line_len = min(h, w) // 5
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_line_len, maxLineGap=15)

    if lines is None or len(lines) == 0:
        return 0.0

    # Collect angles from near-horizontal and near-vertical lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Weight longer lines more heavily
        weight = length / min_line_len

        # Near-horizontal lines (within 5° of horizontal)
        if abs(angle) < 5:
            angles.extend([angle] * int(weight))
        elif abs(angle) > 175:
            corrected = angle - 180 if angle > 0 else angle + 180
            angles.extend([corrected] * int(weight))
        # Near-vertical lines (within 5° of vertical)
        elif abs(abs(angle) - 90) < 5:
            corrected = angle - 90 if angle > 0 else angle + 90
            angles.extend([corrected] * int(weight))

    if len(angles) < 3:
        return 0.0

    median_angle = np.median(angles)

    # Only correct if skew is noticeable but not extreme
    if abs(median_angle) < 0.1 or abs(median_angle) > 5.0:
        return 0.0

    return float(median_angle)


# ---------------------------------------------------------------------------
# Page boundary detection
# ---------------------------------------------------------------------------

def detect_page_bounds(image):
    """Detect the comic page boundaries within a scanner image.

    Scans inward from each edge to find content boundaries. Also detects
    spine/fold lines for two-page bleed handling.

    Returns dict with keys: top, bottom, left, right, angle, spine_col
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Determine scanner bed color from the bottom-right corner
    corner = gray[h - 50:h, w - 50:w]
    bed_mean = corner.mean()
    mean_thresh = max(bed_mean - 30, 180)
    std_thresh = 25

    def is_content_row(y):
        row = gray[y, :]
        return row.mean() < mean_thresh or row.std() > std_thresh

    def is_content_col(x):
        col = gray[:, x]
        return col.mean() < mean_thresh or col.std() > std_thresh

    # Find content boundaries
    top = 0
    for y in range(h):
        if is_content_row(y):
            top = y
            break

    bottom = h
    for y in range(h - 1, -1, -1):
        if is_content_row(y):
            bottom = y + 1
            break

    left = 0
    for x in range(w):
        if is_content_col(x):
            left = x
            break

    right = w
    for x in range(w - 1, -1, -1):
        if is_content_col(x):
            right = x + 1
            break

    # Check for two-page bleed (spine detection)
    spine_result = detect_spine(gray, bottom)
    spine_col = None

    if spine_result is not None:
        spine_center, spine_width = spine_result
        # Determine which side has the primary page (larger area)
        left_area = (spine_center - spine_width // 2) - left
        right_area = right - (spine_center + spine_width // 2)

        if left_area > right_area:
            # Primary page is on the left - crop at the spine
            right = spine_center - spine_width // 2
            spine_col = spine_center
        else:
            # Primary page is on the right - crop at the spine
            left = spine_center + spine_width // 2
            spine_col = spine_center

    # Detect skew angle on the cropped content region
    cropped_gray = gray[top:bottom, left:right]
    angle = detect_skew(cropped_gray)

    return {
        'top': top, 'bottom': bottom, 'left': left, 'right': right,
        'angle': angle, 'spine_col': spine_col,
    }


# ---------------------------------------------------------------------------
# Deskew and crop
# ---------------------------------------------------------------------------

def deskew_and_crop(image, bounds):
    """Apply deskew rotation and crop to detected page bounds."""
    top = bounds['top']
    bottom = bounds['bottom']
    left = bounds['left']
    right = bounds['right']
    angle = bounds['angle']

    if abs(angle) > 0.05:
        img_h, img_w = image.shape[:2]
        center = (img_w / 2, img_h / 2)
        # Negate the angle: if the page is skewed by -1°, we rotate +1° to correct
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(img_h * sin + img_w * cos)
        new_h = int(img_h * cos + img_w * sin)
        rotation_matrix[0, 2] += (new_w - img_w) / 2
        rotation_matrix[1, 2] += (new_h - img_h) / 2

        image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        # Adjust crop bounds for rotation offset
        dx = (new_w - img_w) / 2
        dy = (new_h - img_h) / 2
        left = int(left + dx)
        right = int(right + dx)
        top = int(top + dy)
        bottom = int(bottom + dy)

    cropped = image[top:bottom, left:right]
    return cropped


# ---------------------------------------------------------------------------
# Dimension normalization
# ---------------------------------------------------------------------------

def normalize_dimensions(pages, target_w, target_h):
    """Center-composite each page onto a black canvas of uniform dimensions."""
    normalized = []
    for page in pages:
        h, w = page.shape[:2]
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        paste_w = min(w, target_w)
        paste_h = min(h, target_h)

        src_x = max(0, (w - target_w) // 2)
        src_y = max(0, (h - target_h) // 2)
        dst_x = max(0, (target_w - w) // 2)
        dst_y = max(0, (target_h - h) // 2)

        canvas[dst_y:dst_y + paste_h, dst_x:dst_x + paste_w] = \
            page[src_y:src_y + paste_h, src_x:src_x + paste_w]

        normalized.append(canvas)
    return normalized


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_pages(pages, output_dir, quality, dpi):
    """Save processed pages as JPEG files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(pages):
        filename = output_path / f"Scan {i}.jpg"
        img = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        img.save(str(filename), 'JPEG', quality=quality, dpi=(dpi, dpi))
        size_mb = filename.stat().st_size / (1024 * 1024)
        print(f"  Saved {filename.name} ({img.width}x{img.height}, {size_mb:.1f} MB)")

    return output_path


def preview_pages(pages, indices=None):
    """Open pages for preview using the system image viewer."""
    if indices is None:
        n = len(pages)
        indices = [0, n // 2, n - 1]

    import tempfile
    preview_files = []
    for i in indices:
        page = pages[i]
        img = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        tmp = tempfile.NamedTemporaryFile(suffix=f'_page{i}.jpg', delete=False)
        img.save(tmp.name, 'JPEG', quality=93)
        preview_files.append(tmp.name)
        print(f"  Preview: page {i} -> {tmp.name}")

    for f in preview_files:
        subprocess.Popen(['open', f])

    response = input("\nProceed with saving all pages? [Y/n] ").strip().lower()
    return response != 'n'


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(input_dir, output_dir=None, quality=93, preview=False,
            pages_to_rotate=None):
    """Main processing pipeline."""
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = Path('output') / input_path.name

    if pages_to_rotate is None:
        pages_to_rotate = set()

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    if pages_to_rotate:
        print(f"Rotate 180°: pages {sorted(pages_to_rotate)}")
    print()

    # Step 1: Load and sort scans
    print("Step 1: Loading scans...")
    scans = load_scans(input_dir)
    dpi = get_source_dpi(scans[0][1])
    print(f"  Source DPI: {dpi}")
    print()

    # Step 2: Detect, deskew, crop (and rotate if needed)
    print("Step 2: Processing pages...")
    cropped_pages = []
    for idx, filepath in scans:
        print(f"  Page {idx}: {filepath.name}")
        image = cv2.imread(str(filepath))
        if image is None:
            print(f"    Error: Could not read {filepath}")
            continue

        # Rotate 180° if requested
        if idx in pages_to_rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)
            print(f"    Rotated 180°")

        # Detect page bounds
        bounds = detect_page_bounds(image)
        det_w = bounds['right'] - bounds['left']
        det_h = bounds['bottom'] - bounds['top']
        margins = (f"T={bounds['top']} "
                   f"B={image.shape[0] - bounds['bottom']} "
                   f"L={bounds['left']} "
                   f"R={image.shape[1] - bounds['right']}")

        extra = ""
        if bounds['spine_col'] is not None:
            extra += f" SPINE@{bounds['spine_col']}"
        if abs(bounds['angle']) > 0.05:
            extra += f" skew={bounds['angle']:.2f}°"

        print(f"    Detected: {det_w}x{det_h}, margins: {margins}{extra}")

        # Deskew and crop
        cropped = deskew_and_crop(image, bounds)
        ch, cw = cropped.shape[:2]
        print(f"    Cropped:  {cw}x{ch}")
        cropped_pages.append(cropped)
    print()

    # Step 3: Normalize dimensions
    print("Step 3: Normalizing dimensions...")
    widths = [p.shape[1] for p in cropped_pages]
    heights = [p.shape[0] for p in cropped_pages]
    target_w = int(np.median(widths))
    target_h = int(np.median(heights))
    print(f"  Median dimensions: {target_w}x{target_h}")
    print(f"  Width  range: {min(widths)}-{max(widths)} (spread: {max(widths)-min(widths)})")
    print(f"  Height range: {min(heights)}-{max(heights)} (spread: {max(heights)-min(heights)})")

    normalized = normalize_dimensions(cropped_pages, target_w, target_h)
    print(f"  All {len(normalized)} pages normalized to {target_w}x{target_h}")
    print()

    # Preview if requested
    if preview:
        print("Step 3.5: Preview...")
        if not preview_pages(normalized):
            print("Aborted by user.")
            return None
        print()

    # Step 4: Save
    print("Step 4: Saving processed pages...")
    saved_path = save_pages(normalized, output_dir, quality, dpi)
    print()
    print(f"Done! {len(normalized)} pages saved to {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description='Process raw comic book scans into clean, aligned page images.')
    parser.add_argument('input_dir',
                        help='Directory containing raw scan images')
    parser.add_argument('--output', '-o',
                        help='Output directory (default: output/<input_name>)')
    parser.add_argument('--quality', '-q', type=int, default=93,
                        help='JPEG quality 1-100 (default: 93)')
    parser.add_argument('--preview', '-p', action='store_true',
                        help='Preview pages before saving')

    # 180° rotation options
    rotate_group = parser.add_argument_group('rotation',
        'Options for correcting upside-down pages')
    rotate_group.add_argument('--rotate',
        help='Comma-separated page indices to rotate 180° (e.g., "2,4,6,8" or "2-8")')
    rotate_group.add_argument('--rotate-range',
        help='Range of pages to rotate 180° (e.g., "2-14")')
    rotate_group.add_argument('--rotate-even', action='store_true',
        help='Rotate all even-numbered pages 180°')
    rotate_group.add_argument('--rotate-odd', action='store_true',
        help='Rotate all odd-numbered pages 180°')

    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    # Count pages for rotate calculation
    scans = load_scans(args.input_dir)
    total_pages = max(idx for idx, _ in scans) + 1
    pages_to_rotate = parse_rotate_pages(args, total_pages)

    # Suppress the duplicate "Found N scan files" from process()
    print()
    process(args.input_dir, args.output, args.quality, args.preview,
            pages_to_rotate)


if __name__ == '__main__':
    main()

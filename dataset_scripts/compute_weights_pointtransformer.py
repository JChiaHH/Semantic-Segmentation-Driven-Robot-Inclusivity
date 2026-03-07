#!/usr/bin/env python3
"""
Compute class_weights for Open3D-ML PointTransformer (S3DIS format).

Unlike RandLA-Net / KPConv (SemanticKITTI), the S3DIS dataset config
expects RAW POINT COUNTS per class — not inverse-frequency weights.
Open3D-ML's SemSegLoss computes the weighting internally.

This script reads your S3DIS-format Annotations and counts points
per class, outputting the list to paste into pointtransformer_s3dis.yml.

It also optionally prints the power-inverse-frequency weights (same
method as your RandLA-Net script) in case you want to compare or
use a custom loss.

Author: Jeremy
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import glob

# ===========================================================
# CONFIG – MODIFY THESE
# ===========================================================

# Path to your S3DIS dataset root
S3DIS_ROOT = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis"
    "/Stanford3dDataset_v1.2_Aligned_Version"
)

# Which Areas to count (train only — exclude your test area)
# If you used the spatial_grid_splitting_s3dis.py defaults:
#   Area_1 = train,  Area_5 = val
TRAIN_AREAS = [1]       # Only count training data for weights
# VAL_AREAS = [5]       # Not used for weight computation

# Your class name → class ID mapping
# MUST match the LABEL_TO_CLASS dict in spatial_grid_splitting_s3dis.py
CLASS_TO_ID = {
    "Unlabelled":                  0,
    "Wall":                        1,
    "Staircase":                   2,
    "Fixed_Obstacles":             3,
    "Temporary_Ramps":             4,
    "Safety_Barriers_And_Signs":   5,
    "Temporary_Utilities":         6,
    "Scaffold_Structure":          7,
    "Semi_Fixed_Obstacles":        8,   # Note: underscore, not hyphen
    "Large_Materials":             9,
    "Stored_Equipment":           10,
    "Mobile_Machines_And_Vehicles": 11,
    "Movable_Objects":            12,
    "Containers_And_Pallets":     13,
    "Small_Tools":                14,
    "Portable_Objects":           15,
}

NUM_CLASSES = 16

# Fallback class for unknown filenames
FALLBACK_CLASS = "clutter"
FALLBACK_ID = 0

# ── Optional: power-inverse-frequency params (for comparison) ──
ALSO_COMPUTE_POWER_INV = True
alpha = 0.5
bg_id = 0
bg_scale = 0.10
max_ratio = 5.0
min_ratio = 0.1

# ===========================================================
# COUNT POINTS PER CLASS
# ===========================================================


def class_name_from_filename(fname: str) -> str:
    """
    Extract class name from S3DIS annotation filename.
    e.g. 'class3_1.txt' → 'class3'
          'wall_2.txt'   → 'wall'
    """
    # Remove instance index: split on last underscore before .txt
    stem = Path(fname).stem  # e.g. 'class3_1'
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem  # fallback: use whole stem


def main():
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_files = 0
    total_rooms = 0

    for area_idx in TRAIN_AREAS:
        area_dir = S3DIS_ROOT / f"Area_{area_idx}"
        if not area_dir.exists():
            print(f"[WARN] Area dir not found: {area_dir}")
            continue

        rooms = sorted([d for d in area_dir.iterdir() if d.is_dir()])
        total_rooms += len(rooms)

        for room_dir in rooms:
            anno_dir = room_dir / "Annotations"
            if not anno_dir.exists():
                continue

            txt_files = sorted(anno_dir.glob("*.txt"))
            for txt_file in txt_files:
                class_name = class_name_from_filename(txt_file.name)
                class_id = CLASS_TO_ID.get(class_name, FALLBACK_ID)

                # Count lines = number of points
                try:
                    data = np.loadtxt(str(txt_file))
                    n_pts = data.shape[0] if data.ndim == 2 else (1 if data.ndim == 1 else 0)
                except Exception as e:
                    print(f"[WARN] Could not read {txt_file}: {e}")
                    n_pts = 0

                counts[class_id] += n_pts
                total_files += 1

    # ===========================================================
    # PRINT RESULTS
    # ===========================================================

    print(f"\n{'=' * 60}")
    print(f"S3DIS class_weights computation")
    print(f"{'=' * 60}")
    print(f"Areas scanned:  {TRAIN_AREAS}")
    print(f"Total rooms:    {total_rooms}")
    print(f"Total txt files: {total_files}")
    print(f"Total points:   {counts.sum():,}")

    print(f"\nRaw point counts per class:")
    for cid in range(NUM_CLASSES):
        # Find class name
        cname = [k for k, v in CLASS_TO_ID.items() if v == cid]
        cname = cname[0] if cname else f"class{cid}"
        print(f"  {cid:3d} ({cname:>20s}): {counts[cid]:>12,}")

    # Format for YAML
    counts_list = counts.tolist()
    print(f"\n{'=' * 60}")
    print("Paste into pointtransformer_s3dis.yml:")
    print(f"  class_weights: {counts_list}")
    print(f"{'=' * 60}")

    # ===========================================================
    # OPTIONAL: Power inverse-frequency weights (for comparison)
    # ===========================================================

    if ALSO_COMPUTE_POWER_INV:
        print(f"\n{'─' * 60}")
        print("Power inverse-frequency weights (same method as RandLA-Net):")
        print(f"{'─' * 60}")

        freq = counts.astype(float) + 1.0  # avoid div-by-zero

        # Power inverse frequency
        w_inv = (1.0 / freq) ** alpha

        # Normalize: foreground mean = 1.0
        fg_mask = np.arange(NUM_CLASSES) != bg_id
        mean_fg = w_inv[fg_mask].mean()
        w_inv /= mean_fg

        # Scale background
        w_inv[bg_id] *= bg_scale

        # Clamp
        w_inv = np.clip(w_inv, min_ratio, max_ratio)

        pretty = [round(float(x), 6) for x in w_inv]
        print(f"  Power inv-freq weights: {pretty}")

        # Median frequency
        median_f = np.median(freq[1:])
        w_med = median_f / freq
        fg_mean_med = w_med[fg_mask].mean()
        w_med /= fg_mean_med
        w_med[bg_id] *= bg_scale
        w_med = np.clip(w_med, min_ratio, max_ratio)
        pretty_med = [round(float(x), 6) for x in w_med]
        print(f"  Median freq weights:    {pretty_med}")

        # Effective number (class-balanced)
        beta = 0.99
        w_cb = (1.0 - beta) / (1.0 - np.power(beta, freq))
        fg_mean_cb = w_cb[fg_mask].mean()
        w_cb /= fg_mean_cb
        w_cb[bg_id] *= bg_scale
        w_cb = np.clip(w_cb, min_ratio, max_ratio)
        pretty_cb = [round(float(x), 6) for x in w_cb]
        print(f"  Class-balanced weights: {pretty_cb}")

        print(f"\nNOTE: PointTransformer S3DIS config uses RAW COUNTS.")
        print(f"      The above inverse-freq weights are for reference only,")
        print(f"      or if you modify the loss function manually.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Spatial splitter for S3DIS format (NO CLI ARGS).

Converts .ply files → S3DIS directory layout compatible with
Open3D-ML's PointTransformer (and other S3DIS models).

- Splits by XY grid cells (spatial hold-out)
- Prevents leakage between train / val / test
- Maps grid cells → "rooms" within S3DIS "Areas"

Open3D-ML S3DIS expected layout:
    Stanford3dDataset_v1.2_Aligned_Version/
    ├── Area_1/                          # ← train
    │   ├── room_000000/
    │   │   └── Annotations/
    │   │       ├── Unlabelled_1.txt     # x y z r g b  (one .txt per instance)
    │   │       ├── Wall_1.txt
    │   │       └── ...
    │   ├── room_000001/
    │   │   └── Annotations/
    │   │       └── ...
    │   └── ...
    └── Area_val/                        # ← validation
        └── ...

Each .txt file inside Annotations/ is named <className>_<instanceIdx>.txt
and contains lines of: x y z r g b

Author: Jeremy (adapted from spatial_grid_splitting.py)
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import open3d as o3d
from collections import defaultdict

# =====================================================
# =============== USER CONFIG (EDIT HERE) ==============
# =====================================================

# INPUT: folder containing .ply files
INPUT_DIR = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_prep"
)

# OUTPUT: S3DIS root (the script creates Stanford3dDataset_v1.2_Aligned_Version inside)
OUTPUT_ROOT = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis"
)

# Grid cell size in meters
CELL_SIZE_M = 10.0

# Axes used for spatial grid (XY)
GRID_AXES = (0, 1)

# Drop tiny cells
MIN_POINTS_PER_CELL = 2000

# Cap very large cells
MAX_POINTS_PER_FRAME = 80000

# Random seed for deterministic subsampling
RNG_SEED = 42

# ── S3DIS Area mapping ──────────────────────────────
# Open3D-ML S3DIS loader expects folders named Area_<N>.
# We use Area_1 for train and Area_val for validation.
#
# IMPORTANT: You will need to set test_area_idx in your yml
# to match, OR use a custom dataset class / symlink.
# See NEXT STEPS printed at the end of this script.
TRAIN_AREA = "1"       # All train cells go to Area_1
VAL_AREA   = "val"     # Val cells → Area_val
TEST_AREA  = "test"    # Test cells → Area_test (optional)

# Deterministic spatial split rule (by X index)
# For synthetic data (90/10/0):
SPLIT_MOD = 10
TRAIN_BANDS = {0, 1, 2, 3, 4, 5, 6, 7, 8}   # 90%
VAL_BANDS   = {9}                              # 10%
TEST_BANDS  = set()                            # 0%

# ── Learning map (raw PLY label → training ID) ─────
# Same mapping used in your SemanticKITTI pipeline.
# Raw labels not listed here or mapped to 255 are IGNORED.
LEARNING_MAP = {
    0: 0,    # Unlabelled / background
    1: 1,    # Wall
    3: 2,    # Staircase
    4: 3,    # Fixed_Obstacles
    5: 4,    # Temporary_Ramps
    6: 5,    # Safety_Barriers_And_Signs
    7: 6,    # Temporary_Utilities
    8: 7,    # Scaffold_Structure
    9: 8,    # Semi-Fixed_Obstacles
    10: 9,   # Large_Materials
    11: 10,  # Stored_Equipment
    12: 11,  # Mobile_Machines_And_Vehicles
    13: 12,  # Movable_Objects
    14: 13,  # Containers_And_Pallets
    15: 14,  # Small_Tools
    17: 15,  # Portable_Objects

    # ignored
    2: 255,
    16: 255,
    18: 255,
    255: 255,
}

# ── Training ID → class name (16 classes) ──────────
# S3DIS Annotations use class names as filename prefixes.
LABEL_TO_CLASS = {
    0:  "Unlabelled",
    1:  "Wall",
    2:  "Staircase",
    3:  "Fixed_Obstacles",
    4:  "Temporary_Ramps",
    5:  "Safety_Barriers_And_Signs",
    6:  "Temporary_Utilities",
    7:  "Scaffold_Structure",
    8:  "Semi_Fixed_Obstacles",
    9:  "Large_Materials",
    10: "Stored_Equipment",
    11: "Mobile_Machines_And_Vehicles",
    12: "Movable_Objects",
    13: "Containers_And_Pallets",
    14: "Small_Tools",
    15: "Portable_Objects",
}

# Fallback class name for unknown labels
FALLBACK_CLASS = "Unlabelled"

# Does your PLY have RGB colors? If not, zeros will be used.
# Set to True if your .ply files contain r,g,b or red,green,blue fields.
HAS_RGB = True

# =====================================================
# =============== END OF USER CONFIG ==================
# =====================================================

S3DIS_SUBDIR = "Stanford3dDataset_v1.2_Aligned_Version"


def get_area_dir(area_key: str) -> Path:
    """Return the Area_<key> directory path."""
    d = OUTPUT_ROOT / S3DIS_SUBDIR / f"Area_{area_key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_ply(p: Path):
    """Load a PLY file and return xyz, rgb, labels."""
    tpcd = o3d.t.io.read_point_cloud(str(p))
    xyz = tpcd.point["positions"].numpy().astype(np.float32)

    # ── RGB ──
    rgb = None
    if HAS_RGB:
        for k in ["colors", "red"]:
            if k in tpcd.point:
                if k == "colors":
                    rgb = tpcd.point[k].numpy().astype(np.float32)
                    # Open3D may store as [0,1]; S3DIS expects [0,255]
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255.0)
                    rgb = rgb.astype(np.uint8)
                break
        # Try individual r/g/b channels
        if rgb is None:
            r_arr, g_arr, b_arr = None, None, None
            for rk in ["red", "r", "R"]:
                if rk in tpcd.point:
                    r_arr = tpcd.point[rk].numpy().reshape(-1)
                    break
            for gk in ["green", "g", "G"]:
                if gk in tpcd.point:
                    g_arr = tpcd.point[gk].numpy().reshape(-1)
                    break
            for bk in ["blue", "b", "B"]:
                if bk in tpcd.point:
                    b_arr = tpcd.point[bk].numpy().reshape(-1)
                    break
            if r_arr is not None and g_arr is not None and b_arr is not None:
                rgb = np.stack([r_arr, g_arr, b_arr], axis=1).astype(np.float32)
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255.0)
                rgb = rgb.astype(np.uint8)

    if rgb is None:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)

    # ── Labels ──
    label = None
    for k in ["label", "labels", "scalar_Label", "scalar_label",
              "classification", "class"]:
        if k in tpcd.point:
            label = tpcd.point[k].numpy().astype(np.uint32).reshape(-1)
            break

    if label is None:
        raise RuntimeError(f"No label field found in {p.name}")

    return xyz, rgb, label


def compute_cells(xyz: np.ndarray):
    """Compute grid cell indices for each point."""
    xy = xyz[:, list(GRID_AXES)]
    cx = np.floor(xy[:, 0] / CELL_SIZE_M).astype(np.int32)
    cy = np.floor(xy[:, 1] / CELL_SIZE_M).astype(np.int32)
    return cx, cy


def choose_split(cx: int) -> str:
    """Return the Area key for a given cell X index."""
    band = cx % SPLIT_MOD
    if band in TRAIN_BANDS:
        return TRAIN_AREA
    if band in VAL_BANDS:
        return VAL_AREA
    if band in TEST_BANDS:
        return TEST_AREA
    return TRAIN_AREA  # default fallback


def write_s3dis_room(area_dir: Path, room_name: str,
                     xyz: np.ndarray, rgb: np.ndarray,
                     labels: np.ndarray):
    """
    Write one 'room' in S3DIS Annotations format.

    Creates:
      area_dir/room_name/Annotations/<className>_<instIdx>.txt

    Each .txt line:  x y z r g b
    """
    room_dir = area_dir / room_name
    anno_dir = room_dir / "Annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    # Shift points so the room origin is at (min_x, min_y, min_z)
    xyz_shifted = xyz - xyz.min(axis=0)

    # Group points by label → each label becomes one "instance" txt file
    unique_labels = np.unique(labels)

    # Track instance counts per class name (for naming files)
    class_instance_count: Dict[str, int] = defaultdict(int)

    for lbl in unique_labels:
        # Skip ignored labels
        if int(lbl) == 255:
            continue

        mask = labels == lbl
        class_name = LABEL_TO_CLASS.get(int(lbl), FALLBACK_CLASS)

        pts = xyz_shifted[mask]
        clr = rgb[mask]

        class_instance_count[class_name] += 1
        inst_idx = class_instance_count[class_name]

        fname = f"{class_name}_{inst_idx}.txt"
        out_path = anno_dir / fname

        # Write x y z r g b  (space-separated)
        data = np.hstack([pts, clr.astype(np.float32)])  # (N, 6)
        np.savetxt(str(out_path), data, fmt="%.6f %.6f %.6f %d %d %d")

    return len(unique_labels)


def main():
    rng = np.random.default_rng(RNG_SEED)

    ply_files = sorted(INPUT_DIR.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in {INPUT_DIR}")

    print(f"[INFO] Found {len(ply_files)} PLY files")
    print(f"[INFO] CELL_SIZE_M = {CELL_SIZE_M} m")
    print(f"[INFO] Output: {OUTPUT_ROOT / S3DIS_SUBDIR}")
    print(f"[INFO] Train → Area_{TRAIN_AREA}, Val → Area_{VAL_AREA}, "
          f"Test → Area_{TEST_AREA}")
    print(f"[INFO] Num classes: {len(LABEL_TO_CLASS)}")

    # Build vectorized learning_map lookup
    max_raw = max(LEARNING_MAP.keys()) + 1
    lmap_arr = np.full(max_raw + 1, 255, dtype=np.uint32)
    for raw_id, train_id in LEARNING_MAP.items():
        if raw_id <= max_raw:
            lmap_arr[raw_id] = train_id

    # Counter for room names per area
    room_count: Dict[str, int] = defaultdict(int)

    for ply in ply_files:
        print(f"\n[PROCESS] {ply.name}")
        xyz, rgb, label = load_ply(ply)

        # Remap raw labels → training IDs via learning_map
        # Labels mapped to 255 will be skipped
        safe_label = np.clip(label, 0, max_raw).astype(np.uint32)
        label = lmap_arr[safe_label]

        cx, cy = compute_cells(xyz)
        cell_key = (cx.astype(np.int64) << 32) ^ \
                   (cy.astype(np.int64) & 0xFFFFFFFF)

        order = np.argsort(cell_key)
        cell_key_sorted = cell_key[order]

        splits = np.flatnonzero(
            np.r_[True, cell_key_sorted[1:] != cell_key_sorted[:-1]]
        )
        ends = np.r_[splits[1:], len(order)]

        written = 0
        for s, e in zip(splits, ends):
            idx = order[s:e]

            if idx.size < MIN_POINTS_PER_CELL:
                continue

            if MAX_POINTS_PER_FRAME and idx.size > MAX_POINTS_PER_FRAME:
                idx = idx[rng.choice(idx.size, MAX_POINTS_PER_FRAME,
                                     replace=False)]

            area_key = choose_split(cx[idx[0]])
            area_dir = get_area_dir(area_key)

            room_name = f"room_{room_count[area_key]:06d}"
            room_count[area_key] += 1

            write_s3dis_room(
                area_dir, room_name,
                xyz[idx], rgb[idx], label[idx]
            )
            written += 1

        print(f"[INFO] Wrote {written} rooms from {ply.name}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("[DONE] S3DIS-format dataset created.")
    print(f"  Output root: {OUTPUT_ROOT / S3DIS_SUBDIR}")
    for area_key in sorted(room_count.keys()):
        role = {TRAIN_AREA: "train", VAL_AREA: "val",
                TEST_AREA: "test"}.get(area_key, "?")
        print(f"  Area_{area_key} ({role}): {room_count[area_key]} rooms")
    print("=" * 60)

    # ── Reminder ──
    print("\n[NEXT STEPS]")
    print("1. Update your pointtransformer_s3dis.yml:")
    print(f"   dataset_path: {OUTPUT_ROOT / S3DIS_SUBDIR}")
    print(f"   test_area_idx: val")
    print(f"   num_classes: {len(LABEL_TO_CLASS)}")
    print("   class_weights: [<compute from your data>]")
    print("2. Run compute_weights_pointtransformer.py to get class_weights.")
    print()
    print("NOTE: Open3D-ML S3DIS loader expects test_area_idx as an integer.")
    print("      If it fails with Area_val, you may need to either:")
    print("      (a) Rename Area_val → Area_5 and set test_area_idx: 5, or")
    print("      (b) Subclass the S3DIS dataset to accept string area names.")


if __name__ == "__main__":
    main()

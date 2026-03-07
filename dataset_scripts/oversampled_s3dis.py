#!/usr/bin/env python3
"""
Oversample rare classes for S3DIS format (folder-based, not pkl).

Creates new synthetic "rooms" by extracting patches centered on rare-class points.
Compatible with spatial_grid_splitting_s3dis.py output and compute_weights_pointtransformer.py.

Author: Jeremy (corrected version)
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# =========================================================
#                       CONFIG
# =========================================================

DATASET_ROOT = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis"
    "/Stanford3dDataset_v1.2_Aligned_Version"
)

# Which Area(s) to oversample (TRAIN only - exclude val/test)
TRAIN_AREAS = [1]

# Which Area is used for validation/test (excluded from oversampling)
TEST_AREA_IDX = 2

# Output: create a new Area for oversampled data, or add to existing train area
# Option 1: Add to same Area_1 (simpler, recommended)
# Option 2: Create Area_1_oversampled (requires custom dataset loader)
ADD_TO_EXISTING_AREA = True  # If True, adds new rooms to Area_1

# Rare classes to oversample (TRAINING IDs 0-15, not raw IDs!)
# Based on your class_weights, the rarest classes are:
#   14: Small_Tools (2,559 pts)
#   8:  Semi_Fixed_Obstacles (6,958 pts) 
#   4:  Temporary_Ramps (26,787 pts)
#   13: Containers_And_Pallets (95,413 pts)
#   15: Portable_Objects (116,988 pts)
#   6:  Temporary_Utilities (116,978 pts)
RARE_CLASSES = [14, 8, 4, 13, 15, 6]

# Patch extraction params
RADIUS = 3.0  # meters
MIN_TOTAL_POINTS_PER_PATCH = 2000
MAX_POINTS_PER_PATCH = 40000  # Cap to avoid huge patches

# Per-class minimum count of that rare class inside the patch
MIN_RARE_POINTS_PER_CLASS = {
    14: 2,    # Small_Tools (very rare)
    8:  3,    # Semi_Fixed_Obstacles
    4:  5,    # Temporary_Ramps
    13: 10,   # Containers_And_Pallets
    15: 10,   # Portable_Objects
    6:  10,   # Temporary_Utilities
}

# How many patches to try per room per rare class
PATCHES_PER_ROOM_PER_CLASS = 10

# Global cap on new rooms
MAX_NEW_ROOMS = 500

SEED = 42

# Training ID → Class name (must match spatial_grid_splitting_s3dis.py)
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

# =========================================================
#                    HELPER FUNCTIONS
# =========================================================


def load_room(room_dir: Path):
    """
    Load all points from a room's Annotations folder.
    Returns: xyz (N,3), rgb (N,3), labels (N,)
    """
    anno_dir = room_dir / "Annotations"
    if not anno_dir.exists():
        return None, None, None

    all_xyz = []
    all_rgb = []
    all_labels = []

    for txt_file in sorted(anno_dir.glob("*.txt")):
        # Extract class name from filename (e.g., "Wall_1.txt" → "Wall")
        stem = txt_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            class_name = parts[0]
        else:
            class_name = stem

        # Find training ID
        label_id = None
        for lid, cname in LABEL_TO_CLASS.items():
            if cname == class_name:
                label_id = lid
                break
        if label_id is None:
            label_id = 0  # Fallback to Unlabelled

        try:
            data = np.loadtxt(str(txt_file))
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[0] == 0:
                continue

            xyz = data[:, 0:3].astype(np.float32)
            rgb = data[:, 3:6].astype(np.uint8) if data.shape[1] >= 6 else np.zeros((data.shape[0], 3), dtype=np.uint8)
            labels = np.full(data.shape[0], label_id, dtype=np.int32)

            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_labels.append(labels)
        except Exception as e:
            print(f"[WARN] Could not load {txt_file}: {e}")
            continue

    if not all_xyz:
        return None, None, None

    return (
        np.vstack(all_xyz),
        np.vstack(all_rgb),
        np.concatenate(all_labels)
    )


def write_room(area_dir: Path, room_name: str, xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray):
    """
    Write a room in S3DIS Annotations format.
    """
    room_dir = area_dir / room_name
    anno_dir = room_dir / "Annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    # Shift to local origin
    xyz_shifted = xyz - xyz.min(axis=0)

    # Group by label
    unique_labels = np.unique(labels)
    class_instance_count = defaultdict(int)

    for lbl in unique_labels:
        if int(lbl) == 255:
            continue

        mask = labels == lbl
        class_name = LABEL_TO_CLASS.get(int(lbl), "Unlabelled")

        pts = xyz_shifted[mask]
        clr = rgb[mask]

        class_instance_count[class_name] += 1
        inst_idx = class_instance_count[class_name]

        fname = f"{class_name}_{inst_idx}.txt"
        out_path = anno_dir / fname

        data = np.hstack([pts, clr.astype(np.float32)])
        np.savetxt(str(out_path), data, fmt="%.6f %.6f %.6f %d %d %d")


def extract_patch(xyz, rgb, labels, center_idx, rare_class):
    """
    Extract a spherical patch around center point.
    Returns (patch_xyz, patch_rgb, patch_labels) or (None, None, None) if invalid.
    """
    center = xyz[center_idx]
    dists = np.linalg.norm(xyz - center, axis=1)
    mask = dists < RADIUS

    n_pts = mask.sum()
    if n_pts < MIN_TOTAL_POINTS_PER_PATCH:
        return None, None, None

    # Check rare class count in patch
    rare_in_patch = (labels[mask] == rare_class).sum()
    min_needed = MIN_RARE_POINTS_PER_CLASS.get(rare_class, 5)
    if rare_in_patch < min_needed:
        return None, None, None

    # Subsample if too large
    if n_pts > MAX_POINTS_PER_PATCH:
        indices = np.where(mask)[0]
        indices = np.random.choice(indices, MAX_POINTS_PER_PATCH, replace=False)
        mask = np.zeros_like(mask)
        mask[indices] = True

    return xyz[mask], rgb[mask], labels[mask]


# =========================================================
#                         MAIN
# =========================================================


def main():
    np.random.seed(SEED)

    global_room_id = 0
    global_count = 0

    # Find the highest existing room number in train area(s)
    for area_idx in TRAIN_AREAS:
        area_dir = DATASET_ROOT / f"Area_{area_idx}"
        if not area_dir.exists():
            continue
        existing_rooms = [d.name for d in area_dir.iterdir() if d.is_dir() and d.name.startswith("room_")]
        for rname in existing_rooms:
            try:
                num = int(rname.split("_")[1])
                global_room_id = max(global_room_id, num + 1)
            except:
                pass

    print(f"[INFO] Starting room ID for oversampled data: {global_room_id}")
    print(f"[INFO] Rare classes (training IDs): {RARE_CLASSES}")
    print(f"[INFO] Max new rooms: {MAX_NEW_ROOMS}")

    # Process each training area
    for area_idx in TRAIN_AREAS:
        area_dir = DATASET_ROOT / f"Area_{area_idx}"
        if not area_dir.exists():
            print(f"[WARN] Area not found: {area_dir}")
            continue

        rooms = sorted([d for d in area_dir.iterdir() if d.is_dir() and d.name.startswith("room_")])
        print(f"\n[INFO] Processing Area_{area_idx} with {len(rooms)} rooms")

        for room_dir in rooms:
            if global_count >= MAX_NEW_ROOMS:
                break

            xyz, rgb, labels = load_room(room_dir)
            if xyz is None:
                continue

            # For each rare class, try to extract patches
            for rc in RARE_CLASSES:
                if global_count >= MAX_NEW_ROOMS:
                    break

                rare_idx = np.where(labels == rc)[0]
                if rare_idx.size == 0:
                    continue

                # Choose random centers from rare class points
                n_centers = min(PATCHES_PER_ROOM_PER_CLASS, rare_idx.size)
                chosen = np.random.choice(rare_idx, n_centers, replace=False)

                for center_i in chosen:
                    if global_count >= MAX_NEW_ROOMS:
                        break

                    p_xyz, p_rgb, p_labels = extract_patch(xyz, rgb, labels, center_i, rc)
                    if p_xyz is None:
                        continue

                    # Write new room
                    new_room_name = f"room_{global_room_id:06d}"

                    if ADD_TO_EXISTING_AREA:
                        out_area_dir = area_dir
                    else:
                        out_area_dir = DATASET_ROOT / f"Area_{area_idx}_oversampled"
                        out_area_dir.mkdir(parents=True, exist_ok=True)

                    write_room(out_area_dir, new_room_name, p_xyz, p_rgb, p_labels)

                    rare_count = (p_labels == rc).sum()
                    print(f"[✓] {new_room_name} | src={room_dir.name} | "
                          f"class {rc} ({LABEL_TO_CLASS[rc]}) | "
                          f"{p_xyz.shape[0]} pts (rare={rare_count})")

                    global_room_id += 1
                    global_count += 1

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print(f"[DONE] Created {global_count} oversampled rooms")
    print("=" * 60)

    print("\n[NEXT STEPS]")
    print("1. Re-run compute_weights_pointtransformer.py to get updated class_weights")
    print("2. Update pointtransformer_s3dis.yml with new class_weights")
    print("3. Delete any cached .pkl files in cache_dir before training")
    print(f"\n   rm -rf ./logs/cache/*")


if __name__ == "__main__":
    main()
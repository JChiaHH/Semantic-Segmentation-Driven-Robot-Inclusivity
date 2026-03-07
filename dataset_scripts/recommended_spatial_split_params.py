#!/usr/bin/env python3
"""
Analyze PLY point clouds and recommend spatial split parameters
for KPConv / SemanticKITTI-style datasets.

- Reads all .ply files in INPUT_DIR
- Computes spatial extent and point density
- Recommends:
    - CELL_SIZE_M
    - SPLIT_MOD
    - TRAIN / VAL / TEST band layout
"""

from pathlib import Path
import numpy as np
import open3d as o3d

# =====================================================
# =============== USER CONFIG (EDIT HERE) ==============
# =====================================================

INPUT_DIR = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_prep"
)

# axes used for splitting (XY grid)
GRID_AXES = (0, 1)

# target points per frame (KPConv sweet spot)
TARGET_POINTS_PER_FRAME = (30000, 70000)

# =====================================================


def load_xyz(ply_path: Path) -> np.ndarray:
    tpcd = o3d.t.io.read_point_cloud(str(ply_path))
    return tpcd.point["positions"].numpy().astype(np.float32)


def main():
    ply_files = sorted(INPUT_DIR.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in {INPUT_DIR}")

    print(f"\n[INFO] Found {len(ply_files)} PLY files")

    all_xyz = []
    for p in ply_files:
        xyz = load_xyz(p)
        all_xyz.append(xyz)
        print(f"  {p.name}: {xyz.shape[0]:,} points")

    xyz = np.vstack(all_xyz)

    # -------------------------------------------------
    # Spatial extent
    # -------------------------------------------------
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = maxs - mins

    x_extent = extent[GRID_AXES[0]]
    y_extent = extent[GRID_AXES[1]]

    print("\n[SPATIAL EXTENT]")
    print(f"  X range: {mins[0]:.2f} → {maxs[0]:.2f}  ({x_extent:.2f} m)")
    print(f"  Y range: {mins[1]:.2f} → {maxs[1]:.2f}  ({y_extent:.2f} m)")
    print(f"  Z range: {mins[2]:.2f} → {maxs[2]:.2f}  ({extent[2]:.2f} m)")

    # -------------------------------------------------
    # Density estimation
    # -------------------------------------------------
    area = x_extent * y_extent
    total_points = xyz.shape[0]
    density = total_points / max(area, 1e-6)

    print("\n[POINT DENSITY]")
    print(f"  Total points: {total_points:,}")
    print(f"  XY area: {area:.2f} m²")
    print(f"  Density: {density:.1f} pts / m²")

    # -------------------------------------------------
    # Recommend CELL_SIZE
    # -------------------------------------------------
    # Estimate how many points per cell for different sizes
    candidate_sizes = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0]
    best_size = None

    print("\n[CELL SIZE CANDIDATES]")
    for s in candidate_sizes:
        expected_pts = density * (s * s)
        print(f"  {s:>4.1f} m → ~{int(expected_pts):,} pts per cell")

        if (TARGET_POINTS_PER_FRAME[0]
            <= expected_pts
            <= TARGET_POINTS_PER_FRAME[1]
            and best_size is None):
            best_size = s

    if best_size is None:
        # fallback heuristic
        best_size = 10.0 if max(x_extent, y_extent) < 30 else 5.0

    # -------------------------------------------------
    # Recommend SPLIT_MOD
    # -------------------------------------------------
    site_width = max(x_extent, y_extent)

    if site_width < 25:
        split_mod = 4
        train_bands = "{0,1}"
        val_bands = "{2}"
        test_bands = "{3}"
    elif site_width < 45:
        split_mod = 6
        train_bands = "{0,1,2,3}"
        val_bands = "{4}"
        test_bands = "{5}"
    else:
        split_mod = 10
        train_bands = "{0,1,2,3,4,5,6}"
        val_bands = "{7}"
        test_bands = "{8,9}"

    # -------------------------------------------------
    # Final recommendation
    # -------------------------------------------------
    print("\n================= RECOMMENDATION =================")
    print(f"CELL_SIZE_M = {best_size}")
    print(f"SPLIT_MOD   = {split_mod}")
    print(f"TRAIN_BANDS = {train_bands}")
    print(f"VAL_BANDS   = {val_bands}")
    print(f"TEST_BANDS  = {test_bands}")
    print("=================================================\n")

    print("[WHY]")
    print("- CELL_SIZE chosen to give ~30k–70k points per frame (KPConv-friendly)")
    print("- SPLIT_MOD chosen so val/test regions are spatially distinct")
    print("- No random splitting → no leakage")

    print("\n[NOTE]")
    print("After applying these values, VISUALLY CHECK one train/val/test frame.")
    print("If regions still look similar, increase CELL_SIZE_M by +2.5 or +5.0.")


if __name__ == "__main__":
    main()

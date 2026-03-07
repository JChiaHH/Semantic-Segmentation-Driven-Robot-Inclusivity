#!/usr/bin/env python3
"""
Spatial splitter for SemanticKITTI (NO CLI ARGS).

- Splits by XY grid cells (spatial hold-out)
- Prevents leakage between train / val / test
- Designed for single-site LiDAR captures

Author: Jeremy
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import open3d as o3d

# =====================================================
# =============== USER CONFIG (EDIT HERE) ==============
# =====================================================

# INPUT: folder containing .ply files
INPUT_DIR = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_prep"
)

# OUTPUT: SemanticKITTI root folder
OUTPUT_ROOT = Path(
    "/home/jeremychia/Documents/Point_clouds/Training/dataset_prep"
)

# Grid cell size in meters (IMPORTANT)
CELL_SIZE_M = 10.0  #12.5      # try 5.0 first; use 10.0 if site is large

# Axes used for spatial grid (XY)
GRID_AXES = (0, 1)

# Drop tiny cells
MIN_POINTS_PER_CELL = 2000

# Cap very large cells (keeps KPConv stable)
MAX_POINTS_PER_FRAME = 80000

# Random seed for deterministic subsampling
RNG_SEED = 42

# SemanticKITTI sequences
TRAIN_SEQ = "00"
VAL_SEQ   = "val"
TEST_SEQ  = ""

# Deterministic spatial split rule (by X index)
# SPLIT_MOD = 10
# TRAIN_BANDS = {0, 1, 2, 3, 4, 5, 6}   # 70%
# VAL_BANDS   = {7}                     # 10%
# TEST_BANDS  = {8, 9}                  # 20%

#For synthetic data
SPLIT_MOD = 10
TRAIN_BANDS = {0, 1, 2, 3, 4, 5, 6,7,8}   # 90
VAL_BANDS   = {9}                     # 10
TEST_BANDS  = {}                  # 0

# =====================================================


def ensure_dirs(seq: str):
    vel = OUTPUT_ROOT / "sequences" / seq / "velodyne"
    lab = OUTPUT_ROOT / "sequences" / seq / "labels"
    vel.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    return vel, lab


def load_ply(p: Path):
    tpcd = o3d.t.io.read_point_cloud(str(p))

    xyz = tpcd.point["positions"].numpy().astype(np.float32)

    intensity = None
    for k in ["intensity", "reflectance"]:
        if k in tpcd.point:
            intensity = tpcd.point[k].numpy().astype(np.float32).reshape(-1)
            break

    label = None
    for k in ["label", "labels", "scalar_Label", "scalar_label"]:
        if k in tpcd.point:
            label = tpcd.point[k].numpy().astype(np.uint32).reshape(-1)
            break

    if label is None:
        raise RuntimeError(f"No label field found in {p.name}")

    return xyz, intensity, label


def compute_cells(xyz: np.ndarray):
    xy = xyz[:, GRID_AXES]
    cx = np.floor(xy[:, 0] / CELL_SIZE_M).astype(np.int32)
    cy = np.floor(xy[:, 1] / CELL_SIZE_M).astype(np.int32)
    return cx, cy


def choose_split(cx: int) -> str:
    band = cx % SPLIT_MOD
    if band in TRAIN_BANDS:
        return TRAIN_SEQ
    if band in VAL_BANDS:
        return VAL_SEQ
    return TEST_SEQ


def write_frame(vel_dir, lab_dir, fid, xyz, intensity, label):
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)

    xyzi = np.hstack([xyz, intensity[:, None]])
    stem = f"{fid:06d}"

    (vel_dir / f"{stem}.bin").write_bytes(xyzi.astype(np.float32).tobytes())
    (lab_dir / f"{stem}.label").write_bytes(label.astype(np.uint32).tobytes())


def main():
    rng = np.random.default_rng(RNG_SEED)

    ply_files = sorted(INPUT_DIR.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in {INPUT_DIR}")

    print(f"[INFO] Found {len(ply_files)} PLY files")
    print(f"[INFO] CELL_SIZE_M = {CELL_SIZE_M} m")

    seq_dirs = {
        TRAIN_SEQ: ensure_dirs(TRAIN_SEQ),
        VAL_SEQ: ensure_dirs(VAL_SEQ),
        #TEST_SEQ: ensure_dirs(TEST_SEQ),
    }

    frame_id = {TRAIN_SEQ: 0, VAL_SEQ: 0, TEST_SEQ: 0}

    for ply in ply_files:
        print(f"\n[PROCESS] {ply.name}")
        xyz, intensity, label = load_ply(ply)

        cx, cy = compute_cells(xyz)
        cell_key = (cx.astype(np.int64) << 32) ^ (cy.astype(np.int64) & 0xffffffff)

        order = np.argsort(cell_key)
        cell_key = cell_key[order]

        splits = np.flatnonzero(np.r_[True, cell_key[1:] != cell_key[:-1]])
        ends = np.r_[splits[1:], len(order)]

        written = 0
        for s, e in zip(splits, ends):
            idx = order[s:e]
            if idx.size < MIN_POINTS_PER_CELL:
                continue

            if MAX_POINTS_PER_FRAME and idx.size > MAX_POINTS_PER_FRAME:
                idx = idx[rng.choice(idx.size, MAX_POINTS_PER_FRAME, replace=False)]

            seq = choose_split(cx[idx[0]])
            vel, lab = seq_dirs[seq]

            write_frame(
                vel, lab,
                frame_id[seq],
                xyz[idx],
                intensity[idx] if intensity is not None else None,
                label[idx]
            )

            frame_id[seq] += 1
            written += 1

        print(f"[INFO] Wrote {written} spatial frames")

    print("\n[DONE]")
    for s in [TRAIN_SEQ, VAL_SEQ, TEST_SEQ]:
        print(f"  seq {s}: {frame_id[s]} frames")


if __name__ == "__main__":
    main()

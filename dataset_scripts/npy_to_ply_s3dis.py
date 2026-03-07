#!/usr/bin/env python3
"""
Convert Open3D-ML S3DIS test predictions (.npy) back to .ply for CloudCompare.

Input:
- prediction .npy files from run_pipeline.py (e.g. ./test/S3DIS/Area_3_*.npy)
- matching original_pkl/Area_3_*.pkl files (contain XYZ)

Output PLY fields:
- x y z red green blue label label_train label_raw label_gt

Notes:
- Open3D-ML predictions are train IDs (0..15).
- `label_raw` is recovered via learning_map_inv and `label` is set to raw ID
  for compatibility with existing CloudCompare scalar color scales.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


# Train-ID color map (from pointtransformer_s3dis.yml color_map).
COLOR_MAP_TRAIN = np.array(
    [
        [0, 0, 0],            # 0 Unlabelled
        [255, 170, 255],      # 1 Wall
        [249, 241, 0],        # 2 Staircase
        [85, 170, 255],       # 3 Fixed_Obstacles
        [255, 0, 0],          # 4 Temporary_Ramps
        [142, 106, 36],       # 5 Safety_Barriers_And_Signs
        [255, 0, 127],        # 6 Temporary_Utilities
        [170, 255, 255],      # 7 Scaffold_Structure
        [170, 170, 255],      # 8 Semi_Fixed_Obstacles
        [85, 85, 0],          # 9 Large_Materials
        [159, 98, 0],         # 10 Stored_Equipment
        [104, 104, 56],       # 11 Mobile_Machines_And_Vehicles
        [132, 132, 140],      # 12 Movable_Objects
        [170, 255, 0],        # 13 Containers_And_Pallets
        [179, 20, 176],       # 14 Small_Tools
        [255, 85, 127],       # 15 Portable_Objects
    ],
    dtype=np.uint8,
)


# Train ID -> raw ID (from pointtransformer_s3dis.yml learning_map_inv).
LEARNING_MAP_INV = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 17,
}


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels_train: np.ndarray,
    labels_raw: np.ndarray,
    labels_gt: np.ndarray,
) -> None:
    n = xyz.shape[0]
    if (
        rgb.shape[0] != n
        or labels_train.shape[0] != n
        or labels_raw.shape[0] != n
        or labels_gt.shape[0] != n
    ):
        raise ValueError(
            f"Length mismatch for {path.name}: xyz={n}, rgb={rgb.shape[0]}, "
            f"train={labels_train.shape[0]}, raw={labels_raw.shape[0]}, gt={labels_gt.shape[0]}"
        )

    vertices = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "<i4"),
            ("label_train", "<i4"),
            ("label_raw", "<i4"),
            ("label_gt", "<i4"),
        ],
    )
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["red"] = rgb[:, 0].astype(np.uint8)
    vertices["green"] = rgb[:, 1].astype(np.uint8)
    vertices["blue"] = rgb[:, 2].astype(np.uint8)
    # Keep `label` as raw ID for CloudCompare color scales built on raw labels.
    vertices["label"] = labels_raw.astype(np.int32)
    vertices["label_train"] = labels_train.astype(np.int32)
    vertices["label_raw"] = labels_raw.astype(np.int32)
    vertices["label_gt"] = labels_gt.astype(np.int32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int label\n"
        "property int label_train\n"
        "property int label_raw\n"
        "property int label_gt\n"
        "end_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def load_xyz_and_gt_from_pkl(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)) or len(data) < 1:
        raise ValueError(f"Unexpected pkl format: {pkl_path}")

    pc_label = np.asarray(data[0])
    if pc_label.ndim != 2 or pc_label.shape[1] < 3:
        raise ValueError(f"Invalid point array in {pkl_path}: shape={pc_label.shape}")

    xyz = pc_label[:, :3].astype(np.float32)
    if pc_label.shape[1] >= 7:
        gt = pc_label[:, 6].astype(np.int32)
    else:
        gt = np.zeros((pc_label.shape[0],), dtype=np.int32)
    return xyz, gt


def labels_to_rgb_train(labels_train: np.ndarray) -> np.ndarray:
    labels_safe = labels_train.copy()
    invalid = (labels_safe < 0) | (labels_safe >= COLOR_MAP_TRAIN.shape[0])
    labels_safe[invalid] = 0
    return COLOR_MAP_TRAIN[labels_safe]


def train_to_raw(labels_train: np.ndarray) -> np.ndarray:
    labels_raw = np.zeros(labels_train.shape, dtype=np.int32)
    for train_id, raw_id in LEARNING_MAP_INV.items():
        labels_raw[labels_train == train_id] = raw_id
    # Unknown/unmapped labels default to 0 (Unlabelled).
    return labels_raw


def read_ply_xyz_rgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Invalid point cloud: {ply_path}")
    if colors.shape[0] != points.shape[0]:
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    if colors.size > 0 and colors.max() <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0.0, 255.0).astype(np.float32)
    return points, colors


def as_str(x) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            x = x.item()
        elif x.size == 1:
            x = x.reshape(()).item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Open3D-ML S3DIS predictions (.npy) to .ply")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("/home/jeremychia/Documents/Open3D-ML/test/S3DIS"),
        help="Directory containing prediction .npy files",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/"
            "Stanford3dDataset_v1.2_Aligned_Version"
        ),
        help="S3DIS dataset root (used to resolve source ply files)",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path(
            "/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/"
            "Stanford3dDataset_v1.2_Aligned_Version/original_pkl_meta"
        ),
        help="Directory containing the metadata .npz files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for .ply files (default: <pred-dir>/ply)",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir
    meta_dir = args.meta_dir
    out_dir = args.out_dir or (pred_dir / "ply")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")

    npy_files = sorted(pred_dir.glob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {pred_dir}")

    print(f"[INFO] Found {len(npy_files)} prediction files in {pred_dir}")
    print(f"[INFO] Writing PLY to {out_dir}")

    ply_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    converted = 0
    for npy_path in npy_files:
        stem = npy_path.stem
        meta_chunks = sorted(meta_dir.glob(f"{stem}__cell_*.npz"))
        if not meta_chunks:
            print(f"[WARN] Missing metadata for {stem} (no matching .npz files)")
            continue

        meta0 = np.load(meta_chunks[0], allow_pickle=True)
        source_ply = Path(as_str(meta0.get("source_ply", "")))
        if not source_ply.exists():
            print(f"[WARN] Source ply missing for {stem}: {source_ply}")
            continue

        original_cnt = int(np.asarray(meta0["original_point_count"]).reshape(()))
        if source_ply not in ply_cache:
            points, colors = read_ply_xyz_rgb(source_ply)
            if points.shape[0] != original_cnt:
                print(
                    f"[WARN] Point count mismatch for {source_ply.name}: "
                    f"{points.shape[0]} vs {original_cnt}"
                )
            ply_cache[source_ply] = (points, colors)
        else:
            points, colors = ply_cache[source_ply]

        labels_train = np.load(npy_path).astype(np.int32).reshape(-1)
        if labels_train.shape[0] != original_cnt:
            print(
                f"[WARN] Skipping {stem}: label count {labels_train.shape[0]} != expected {original_cnt}"
            )
            continue

        labels_raw = train_to_raw(labels_train)
        labels_gt = np.zeros_like(labels_train, dtype=np.int32)
        rgb = labels_to_rgb_train(labels_train)
        out_path = out_dir / f"{stem}.ply"
        write_binary_ply(out_path, points[:original_cnt], colors[:original_cnt], labels_train, labels_raw, labels_gt)

        uniq_tr, cnt_tr = np.unique(labels_train, return_counts=True)
        uniq_raw, cnt_raw = np.unique(labels_raw, return_counts=True)
        summary_tr = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq_tr, cnt_tr))
        summary_raw = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq_raw, cnt_raw))
        print(
            f"[OK] {out_path.name} ({original_cnt:,} points)\n"
            f"     train[{summary_tr}]\n"
            f"     raw  [{summary_raw}]"
        )
        converted += 1

    print(f"[DONE] Converted {converted}/{len(npy_files)} files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# ================== CONFIG ==================

# Point clouds (.bin) and labels (.label) to export
# BIN_DIR  = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/test/velodyne")
# LAB_DIR  = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/test/predictions2")

#BIN_DIR  = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/testfullpcd/velodyne")
#LAB_DIR  = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/testfullpcd/predictions")

BIN_DIR = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/testfullpcd/velodyne")
LAB_DIR = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/testfullpcd/labels")

# Output PLYs here
#OUT_DIR  = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/testfullpcd/plyfiles_predicted2")
OUT_DIR = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences/testfullpcd/predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Write ASCII PLY (easy to inspect, slower) vs binary
WRITE_ASCII = True

# If you want to force interpretation:
#   "auto"  -> detect train-vs-raw per file (recommended)
#   "raw"   -> always treat .label as raw semantic IDs (dataset GT style)
#   "train" -> always treat .label as train IDs (model prediction style)
LABEL_MODE = "raw"

# ================== YOUR MAPS ==================
# raw semantic id -> train id (0..12) or 255(ignore)
# learning_map = {
#     0: 0,   # Unlabelled / background
#     1: 1,   # Wall
#     3: 2,   # Staircase
#     4: 3,   # Fixed_Obstacles
#     6: 4,   # Safety_Barriers_And_Signs
#     7: 5,   # Temporary_Utilities
#     8: 6,   # Scaffold_Structure
#     10: 7,  # Large_Materials
#     11: 8,  # Stored_Equipment
#     12: 9,  # Mobile_Machines_And_Vehicles
#     13: 10, # Movable_Objects
#     14: 11, # Containers_And_Pallets
#     17: 12, # Portable_Objects

#     # NOTE:
#     # If your dataset truly does not contain those raw classes,
#     # you can leave them out. Anything not listed becomes IGNORE_ID.
#     # 2: 255, 5: 255, 9: 255, 15: 255, 16: 255, 18: 255,
# }

# # train id -> raw semantic id (for visualization)
# learning_map_inv = {
#     0: 0,   # background
#     1: 1,   # Wall
#     2: 3,   # Staircase
#     3: 4,   # Fixed_Obstacles
#     4: 6,   # Safety_Barriers_And_Signs
#     5: 7,   # Temporary_Utilities
#     6: 8,   # Scaffold_Structure
#     7: 10,  # Large_Materials
#     8: 11,  # Stored_Equipment
#     9: 12,  # Mobile_Machines_And_Vehicles
#     10: 13, # Movable_Objects
#     11: 14, # Containers_And_Pallets
#     12: 17, # Portable_Objects
# }



# raw semantic id  -> train id
learning_map = {
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

    # ignored classes
    2: 255,
    16: 255,
    18: 255,
    255: 255,
}

learning_map_inv = {
    0: 0,    # background
    1: 1,    # Wall
    2: 3,    # Staircase
    3: 4,    # Fixed_Obstacles

    4: 5,    # Temporary_Ramps
    5: 6,    # Safety_Barriers_And_Signs
    6: 7,    # Temporary_Utilities
    7: 8,    # Scaffold_Structure

    8: 9,    # Semi-Fixed_Obstacles
    9: 10,   # Large_Materials
    10: 11,  # Stored_Equipment
    11: 12,  # Mobile_Machines_And_Vehicles
    12: 13,  # Movable_Objects
    13: 14,  # Containers_And_Pallets
    14: 15,  # Small_Tools
    15: 17,  # Portable_Objects,
}

IGNORE_ID = 255

# ================== IO HELPERS ==================

def read_bin(bin_path: Path) -> np.ndarray:
    """SemanticKITTI velodyne .bin: float32 [x,y,z,intensity]"""
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise RuntimeError(f"{bin_path.name}: .bin size not divisible by 4 floats")
    return arr.reshape((-1, 4))

def read_label(label_path: Path) -> np.ndarray:
    """
    DO NOT TOUCH (per your instruction):
    SemanticKITTI .label: uint32 where lower 16 bits = semantic ID, upper 16 bits = instance ID.
    Works for BOTH:
      - GT labels (raw semantic IDs in lower 16 bits)
      - prediction labels (often just train IDs stored as uint32; lower 16 bits still valid)
    """
    lab = np.fromfile(label_path, dtype=np.uint32)
    semantic = (lab & 0xFFFF).astype(np.int32)
    return semantic

def apply_learning_map(raw_sem: np.ndarray) -> np.ndarray:
    """Map raw semantic IDs -> train IDs, unknown -> IGNORE_ID (255) by default."""
    out = np.full(raw_sem.shape, IGNORE_ID, dtype=np.int32)
    for k, v in learning_map.items():
        out[raw_sem == k] = v
    return out

def train_to_raw(train_ids: np.ndarray) -> np.ndarray:
    """Map train IDs -> raw semantic IDs; ignore (255) becomes 0 for visualization."""
    out = np.zeros(train_ids.shape, dtype=np.int32)
    for tid, rid in learning_map_inv.items():
        out[train_ids == tid] = rid
    out[train_ids == IGNORE_ID] = 0
    return out

def looks_like_train_ids(vals: np.ndarray) -> bool:
    """
    Heuristic: if all (or almost all) labels are within {0..max_train_id} or 255,
    then it's probably train-ID labels already.
    """
    max_train_id = max(learning_map_inv.keys())
    # sample for speed on huge clouds
    if vals.size > 500000:
        idx = np.random.choice(vals.size, size=500000, replace=False)
        v = vals[idx]
    else:
        v = vals

    allowed_min = 0
    allowed_max = max_train_id
    ok = (v == IGNORE_ID) | ((v >= allowed_min) & (v <= allowed_max))
    # allow a tiny bit of noise, but not much
    return (ok.mean() > 0.999)

def write_ply(path: Path, xyz: np.ndarray, fields: dict):
    """
    Write PLY with x,y,z + extra scalar fields (int).
    fields: dict[str, np.ndarray] each shape (N,)
    """
    n = xyz.shape[0]
    for name, arr in fields.items():
        if arr.shape[0] != n:
            raise RuntimeError(f"Field {name} has {arr.shape[0]} != {n}")

    if WRITE_ASCII:
        with path.open("w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            for name in fields.keys():
                f.write(f"property int {name}\n")
            f.write("end_header\n")

            for i in range(n):
                row = [f"{xyz[i,0]:.6f}", f"{xyz[i,1]:.6f}", f"{xyz[i,2]:.6f}"]
                for name in fields.keys():
                    row.append(str(int(fields[name][i])))
                f.write(" ".join(row) + "\n")
    else:
        with path.open("wb") as f:
            header = []
            header.append("ply")
            header.append("format binary_little_endian 1.0")
            header.append(f"element vertex {n}")
            header.append("property float x")
            header.append("property float y")
            header.append("property float z")
            for name in fields.keys():
                header.append(f"property int {name}")
            header.append("end_header\n")
            f.write(("\n".join(header)).encode("ascii"))

            data = [xyz.astype(np.float32)]
            for name in fields.keys():
                data.append(fields[name].astype(np.int32).reshape(-1, 1))
            packed = np.concatenate(data, axis=1)
            f.write(packed.tobytes())

# ================== MAIN ==================

def main():
    bin_files = sorted(BIN_DIR.glob("*.bin"))
    if not bin_files:
        raise RuntimeError(f"No .bin files in {BIN_DIR}")

    print(f"[INFO] Found {len(bin_files)} bin files")

    for b in bin_files:
        stem = b.stem
        l = LAB_DIR / f"{stem}.label"
        if not l.exists():
            print(f"[WARN] Missing label for {stem}, skipping")
            continue

        xyzi = read_bin(b)
        xyz = xyzi[:, :3]
        sem = read_label(l)

        if sem.shape[0] != xyz.shape[0]:
            raise RuntimeError(f"{stem}: points {xyz.shape[0]} != labels {sem.shape[0]}")

        # Decide interpretation
        if LABEL_MODE == "raw":
            mode = "raw"
        elif LABEL_MODE == "train":
            mode = "train"
        else:
            mode = "train" if looks_like_train_ids(sem) else "raw"

        if mode == "train":
            # sem already IS train IDs (model output)
            train = sem.astype(np.int32)
            raw_from_train = train_to_raw(train)

            # You don't truly have GT raw here, so we export a "raw view" recovered from train IDs.
            raw = raw_from_train
        else:
            # sem is GT raw semantic IDs (dataset labels)
            raw = sem.astype(np.int32)
            train = apply_learning_map(raw)
            raw_from_train = train_to_raw(train)

        ignored_mask = (train == IGNORE_ID).astype(np.int32)

        # Print quick sanity summary
        uniq_raw, cnt_raw = np.unique(raw, return_counts=True)
        uniq_tr, cnt_tr = np.unique(train, return_counts=True)

        print(f"\n[{stem}] mode={mode}")
        print(f"  label_raw present: {dict(zip(uniq_raw.tolist(), cnt_raw.tolist()))}")
        print(f"  label_train present: {dict(zip(uniq_tr.tolist(), cnt_tr.tolist()))}")
        print(f"  ignored points: {ignored_mask.sum()} / {ignored_mask.size}")

        out_path = OUT_DIR / f"{stem}.ply"

        fields = {
            # If mode=raw: true GT raw labels from dataset
            # If mode=train: recovered raw view from predicted train IDs (for CC color-scale)
            "label_raw": raw.astype(np.int32),

            # Always what training “sees” (or what the model predicted)
            #"label_train": train.astype(np.int32),

            # Always the raw ids recovered from train ids (ignore->0)
            #"label_raw_from_train": raw_from_train.astype(np.int32),

            #"is_ignored": ignored_mask.astype(np.int32),

            # Helpful: 0=raw mode, 1=train mode (CloudCompare-friendly)
            #"label_mode": (np.ones_like(train, dtype=np.int32) if mode == "train" else np.zeros_like(train, dtype=np.int32)),
        }

        write_ply(out_path, xyz, fields)
        print(f"  -> wrote {out_path}")

    print("\n[DONE] CloudCompare tips:")
    print("  - Use SF 'label_train' to see what the model predicts / what training uses")
    print("  - Use SF 'label_raw_from_train' for stable coloring with your raw-ID color scale")
    print("  - If exporting GT labels, SF 'label_raw' is the true raw dataset label")
    print("  - SF 'label_mode' tells you whether that file was interpreted as raw(0) or train(1)")

if __name__ == "__main__":
    main()

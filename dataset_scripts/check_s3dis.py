#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from collections import defaultdict

# ===== CONFIG =====
root = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version")

# If you split into Area_1, Area_2, etc
areas = ["Area_1"]  # add more if you want

# OPTIONAL: map raw class name (from filename) -> your training id
# If you DON'T want mapping, set USE_LABEL_MAP = False and it will count by class name.
USE_LABEL_MAP = True
LABEL_MAP = {
    "Unlabelled": 0,
    "Wall": 1,
    "Staircase": 2,
    "Fixed_Obstacles": 3,
    "Temporary_Ramps": 4,
    "Safety_Barriers_And_Signs": 5,
    "Temporary_Utilities": 6,
    "Scaffold_Structure": 7,
    "Semi_Fixed_Obstacles": 8,
    "Large_Materials": 9,
    "Stored_Equipment": 10,
    "Mobile_Machines_And_Vehicles": 11,
    "Movable_Objects": 12,
    "Containers_And_Pallets": 13,
    "Small_Tools": 14,
    "Portable_Objects": 15,
}

# ===== GLOBAL ACCUMULATORS =====
global_counts = defaultdict(int)
global_total = 0
skipped_empty = 0
skipped_bad = 0
skipped_unknown = 0

def parse_class_name_from_filename(txt_path: Path) -> str:
    """
    Your files look like: Containers_And_Pallets_-1.txt
    We want: Containers_And_Pallets
    Works even if class contains underscores.
    """
    stem = txt_path.stem  # e.g. "Containers_And_Pallets_-1"
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]

for area in areas:
    area_dir = root / area
    if not area_dir.exists():
        print(f"[WARN] {area} not found at {area_dir}")
        continue

    print(f"\n========== {area} ==========")

    for room in area_dir.glob("*"):
        anno_dir = room / "Annotations"
        if not anno_dir.exists():
            continue

        for txt_file in anno_dir.glob("*.txt"):
            # load points (expected: x y z r g b) — NO label column in standard S3DIS
            try:
                data = np.loadtxt(txt_file)
            except Exception as e:
                skipped_bad += 1
                print(f"[WARN] Failed to read {txt_file}: {e}")
                continue

            # handle single-line files (become 1D)
            if data.ndim == 1:
                if data.size == 0:
                    skipped_empty += 1
                    continue
                data = data.reshape(1, -1)

            # empty or weird columns
            if data.size == 0 or data.shape[0] == 0:
                skipped_empty += 1
                continue

            # S3DIS annotation usually has 6 columns: xyzrgb
            if data.shape[1] < 3:
                skipped_bad += 1
                print(f"[WARN] Skipping {txt_file}, unexpected column format: {data.shape}")
                continue

            n_pts = data.shape[0]
            class_name = parse_class_name_from_filename(txt_file)

            if USE_LABEL_MAP:
                if class_name not in LABEL_MAP:
                    skipped_unknown += n_pts
                    # You can also map unknowns to Unlabelled if you prefer:
                    # class_id = LABEL_MAP["Unlabelled"]
                    # global_counts[class_id] += n_pts
                    continue
                class_id = LABEL_MAP[class_name]
                global_counts[class_id] += n_pts
            else:
                global_counts[class_name] += n_pts

            global_total += n_pts

print("\n================ FINAL GLOBAL SUMMARY ================")
print(f"Total points: {global_total}")
print(f"Skipped empty files: {skipped_empty}")
print(f"Skipped bad files:   {skipped_bad}")
if USE_LABEL_MAP:
    print(f"Skipped unknown-class points (not in LABEL_MAP): {skipped_unknown}")
print("")

for k in sorted(global_counts, key=lambda x: (x if isinstance(x, int) else str(x))):
    print(f"class {k}: {global_counts[k]}")

print("======================================================")
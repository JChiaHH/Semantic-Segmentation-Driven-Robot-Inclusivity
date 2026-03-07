#!/usr/bin/env python3
import numpy as np
import glob

# ===========================================================
# CONFIG – MODIFY ONLY THESE TWO PATHS
# ===========================================================

from pathlib import Path

# ===========================================================
# NEW DIRECTORY LOGIC
# ===========================================================

root = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset/sequences")
seqs = ["00", "oversampled1"]

TRAIN_LABELS = []
for s in seqs:
    TRAIN_LABELS += glob.glob(str(root / s / "labels" / "*.label"))

print(f"Found {len(TRAIN_LABELS)} training label files")


# Your raw → training ID mapping EXACTLY as used in your YAML
# learning_map = {
#     0: 0,   # Unlabelled / background
#     1: 1,   # Wall
#     3: 2,   # Staircase  (your new class — now included!)
#     4: 3,   # Fixed_Obstacles
#     6: 4,   # Safety_Barriers_And_Signs
#     7: 5,   # Temporary_Utilities
#     8: 6,   # Scaffold_Structure
#     10: 7,  # Large_Materials
#     11: 8,  # Stored_Equipment
#     12: 9,  # Mobile_Machines_And_Vehicles
#     13: 10, # Movable_Objects
#     14: 11, # Containers_And_Pallets
#     15: 12, #small tools
#     17: 13, # Portable_Objects

#     # Everything not used → background
#     2: 255,
#     5: 255,
#     9: 255,
    
#     16: 255,
#     18: 255,
#     255: 255
# }
learning_map = {
    0: 0,    # Unlabelled / background
    1: 1,    # Wall
    3: 2,    # Staircase
    4: 3,    # Fixed_Obstacles

    5: 4,    # Temporary_Ramps        (moved here)
    6: 5,    # Safety_Barriers_And_Signs
    7: 6,    # Temporary_Utilities
    8: 7,    # Scaffold_Structure

    9: 8,    # Semi-Fixed_Obstacles   (moved here)
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
    255: 255
}


# ===========================================================
# Weighting hyperparams
# ===========================================================

num_classes = 16
bg_id = 0              # background class ID
beta = 0.99            # for effective number of samples
alpha = 0.5            # exponent for power inverse frequency
max_ratio = 5.0        # clamp max
min_ratio = 0.1        # clamp min
bg_scale = 0.10        # lower weight for background (you used 0.10)

# ===========================================================
# COUNT TRAIN LABELS
# ===========================================================

counts = np.zeros(num_classes, dtype=np.int64)

for lf in TRAIN_LABELS:
    raw = np.fromfile(lf, dtype=np.uint32)
    train_ids = np.vectorize(learning_map.get)(raw)
    for tid in range(num_classes):
        counts[tid] += np.sum(train_ids == tid)

print("\nTraining-ID frequencies:")
for tid, val in enumerate(counts):
    print(f"  train_id {tid}: {val}")

# Prevent zero division
freq = counts.astype(float) + 1.0

# ===========================================================
# 1) POWER INVERSE FREQUENCY
# ===========================================================

w_inv = (1.0 / freq) ** alpha


# ===========================================================
# 2) MEDIAN FREQUENCY BALANCING
# ===========================================================

median_f = np.median(freq[1:])  # exclude background
w_med = median_f / freq


# ===========================================================
# 3) EFFECTIVE NUMBER OF SAMPLES
# ===========================================================

w_cb = (1.0 - beta) / (1.0 - np.power(beta, freq))


# ===========================================================
# NORMALIZE + CLIP
# ===========================================================

def normalize_and_clip(w, name):
    w = w.copy()

    # Normalize so foreground mean = 1.0
    fg_mask = np.arange(num_classes) != bg_id
    mean_fg = w[fg_mask].mean()
    w /= mean_fg

    # scale background
    w[bg_id] *= bg_scale

    # clamp
    w = np.clip(w, min_ratio, max_ratio)

    print(f"\n{name} (normalized & clipped):")
    pretty = [float(f"{x:.6f}") for x in w]
    print(pretty)
    return pretty

# Output three versions
weights_inv = normalize_and_clip(w_inv, "Power inverse-freq")
weights_med = normalize_and_clip(w_med, "Median frequency")
weights_cb  = normalize_and_clip(w_cb, "Class-balanced (effective N)")

print("\n===================================")
print("Paste this into YAML:")
print("class_weights:", weights_inv)
print("===================================\n")

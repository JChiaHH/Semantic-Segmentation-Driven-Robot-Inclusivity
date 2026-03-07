from pathlib import Path
import numpy as np
from collections import defaultdict

# === CONFIG === Use file directory before sequence folder. 
#root = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset")
root = Path("/home/jeremychia/Documents/Point_clouds/Training/dataset")

#seqs = ["tinytrain"]
#seqs = ["00", "01", "oversampled","oversampled2", "oversampled3", "oversampled4", "oversampled5", "oversampled6"]
#seqs = ["00", "01", "oversampled","oversampled2", "oversampled3"]
seqs = [ "synthetic"]

# === GLOBAL ACCUMULATOR ===
global_counts = defaultdict(int)
global_total_points = 0


# === PROCESS EACH SEQUENCE ===
for seq in seqs:
    label_dir = root / "sequences" / seq / "labels"
    if not label_dir.exists():
        print(f"\n[WARN] Label dir not found: {label_dir}")
        continue

    print(f"\n================ Sequence {seq} ================")
    label_files = sorted(label_dir.glob("*.label"))
    if not label_files:
        print("  (no .label files)")
        continue

    # per-sequence accumulator
    seq_counts = defaultdict(int)
    seq_total_points = 0

    # per-file counts
    for lf in label_files:
        arr = np.fromfile(lf, dtype=np.uint32)
        uniq, counts = np.unique(arr, return_counts=True)

        print(f"  {lf.name}:")
        for u, c in zip(uniq, counts):
            print(f"    class {u}: {c} points")

            # update sequence-level and global-level counts
            seq_counts[int(u)] += int(c)
            seq_total_points += int(c)

            global_counts[int(u)] += int(c)
            global_total_points += int(c)

    # === PER-SEQUENCE SUMMARY ===
    print(f"\n  --- Summary for sequence {seq} ---")
    for cid in sorted(seq_counts):
        print(f"    class {cid}: {seq_counts[cid]} points")
    print(f"  Total points in seq {seq}: {seq_total_points}")

    # # === GLOBAL SUMMARY SO FAR ===
    # print("\n  --- GLOBAL SUMMARY (so far) ---")
    # for cid in sorted(global_counts):
    #     print(f"    class {cid}: {global_counts[cid]} points")
    # print(f"  Total points across processed sequences: {global_total_points}")
    # print("====================================================")


# === FINAL GLOBAL SUMMARY ===
print("\n================ FINAL GLOBAL SUMMARY ================")
print(f"Total points across ALL sequences: {global_total_points}\n")
for cid in sorted(global_counts):
    print(f"  class {cid}: {global_counts[cid]} points")
print("======================================================\n")

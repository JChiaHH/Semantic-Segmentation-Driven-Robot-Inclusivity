#!/usr/bin/env python3
"""
Merge chunked S3DIS prediction .npy files back to one .npy per original cloud.

Expected inputs:
- Predictions from run_pipeline.py: <pred-dir>/<chunk_stem>.npy
- Metadata from convert_ply_to_s3dis.py: <meta-dir>/<chunk_stem>.npz

Output:
- <out-dir>/<merged_name>.npy (one merged prediction per original cloud)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def as_str(x) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            x = x.item()
        elif x.size == 1:
            x = x.reshape(()).item()
    if isinstance(x, bytes):
        return x.decode('utf-8')
    return str(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split chunk predictions into one .npy per source cloud"
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("/home/jeremychia/Documents/Open3D-ML/test/S3DIS"),
        help="Directory with chunk prediction .npy files",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path(
            "/home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/"
            "Stanford3dDataset_v1.2_Aligned_Version/original_pkl_meta"
        ),
        help="Directory with chunk metadata .npz files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for merged .npy files (default: <pred-dir>/merged)",
    )
    parser.add_argument(
        "--fill-label",
        type=int,
        default=0,
        help="Label used for points not covered by any predicted chunk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_dir = args.pred_dir
    meta_dir = args.meta_dir
    out_dir = args.out_dir or (pred_dir / 'merged')
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")

    meta_files = sorted(meta_dir.glob('*.npz'))
    if not meta_files:
        raise RuntimeError(f"No metadata .npz files found in {meta_dir}")

    groups = defaultdict(list)
    for meta_path in meta_files:
        m = np.load(meta_path, allow_pickle=True)
        merged_name = as_str(m['merged_name']) if 'merged_name' in m else meta_path.stem.split('__cell_')[0]
        n_total = int(np.asarray(m['original_point_count']).reshape(()))
        idx = np.asarray(m['point_indices'], dtype=np.int64)
        groups[merged_name].append(
            {
                'meta_path': meta_path,
                'chunk_stem': meta_path.stem,
                'n_total': n_total,
                'indices': idx,
            }
        )

    print(f"[INFO] Groups to merge: {len(groups)}")
    print(f"[INFO] Reading predictions from: {pred_dir}")
    print(f"[INFO] Writing merged outputs to: {out_dir}")

    report_lines = []
    merged_count = 0

    for merged_name, chunks in sorted(groups.items()):
        n_total_ref = chunks[0]['n_total']
        merged_pred = np.full((n_total_ref,), args.fill_label, dtype=np.int32)
        assigned = np.zeros((n_total_ref,), dtype=np.uint16)

        total_chunks = len(chunks)
        used_chunks = 0
        missing_chunks = 0
        overlap_points = 0

        for ch in chunks:
            if ch['n_total'] != n_total_ref:
                raise RuntimeError(
                    f"Inconsistent original_point_count for {merged_name}: "
                    f"{ch['n_total']} vs {n_total_ref}"
                )

            pred_path = pred_dir / f"{ch['chunk_stem']}.npy"
            if not pred_path.exists():
                missing_chunks += 1
                continue

            pred = np.load(pred_path).astype(np.int32).reshape(-1)
            idx = ch['indices']

            if pred.shape[0] != idx.shape[0]:
                print(
                    f"[WARN] Skip chunk {ch['chunk_stem']}: "
                    f"pred_len={pred.shape[0]} idx_len={idx.shape[0]}"
                )
                missing_chunks += 1
                continue

            overlap_mask = assigned[idx] > 0
            overlap_points += int(overlap_mask.sum())

            merged_pred[idx] = pred
            assigned[idx] += 1
            used_chunks += 1

        covered = int((assigned > 0).sum())
        unassigned = int((assigned == 0).sum())
        coverage = covered / float(n_total_ref) if n_total_ref > 0 else 0.0

        out_path = out_dir / f"{merged_name}.npy"
        np.save(out_path, merged_pred)

        report = (
            f"{merged_name}: chunks {used_chunks}/{total_chunks}, "
            f"missing_chunks={missing_chunks}, coverage={coverage:.2%}, "
            f"unassigned={unassigned}, overlap_points={overlap_points}, "
            f"saved={out_path.name}"
        )
        print(f"[OK] {report}")
        report_lines.append(report)
        merged_count += 1

    report_path = out_dir / 'merge_report.txt'
    report_path.write_text("\n".join(report_lines) + "\n", encoding='utf-8')

    print(f"[DONE] Merged {merged_count} cloud(s)")
    print(f"[DONE] Report: {report_path}")


if __name__ == '__main__':
    main()

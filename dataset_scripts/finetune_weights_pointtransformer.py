#!/usr/bin/env python3
"""
Fine-tune PointTransformer class_weights for Open3D-ML S3DIS config.

Why this script exists:
- Open3D-ML expects `dataset.class_weights` as class counts in YAML.
- The loss internally converts counts to CE weights:
    ce_w = 1 / ((count / sum(count)) + 0.02)
- So class-count edits are non-intuitive; this script previews both counts and
  resulting CE weights before and after tuning.

Typical usage:
  python finetune_weights_pointtransformer.py \
    --config /home/jeremychia/Documents/Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml \
    --mode ce \
    --ce-mult "14:0.55,15:0.7,13:0.8,0:1.15"

Notes:
- This script does NOT edit your YAML; it prints a new `class_weights` line.
- Use `--mode ce` if your intent is to directly damp/boost loss impact.
- Use `--mode counts` if you prefer manipulating raw counts directly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


CLASS_NAMES = [
    "Unlabelled",
    "Wall",
    "Staircase",
    "Fixed_Obstacles",
    "Temporary_Ramps",
    "Safety_Barriers_And_Signs",
    "Temporary_Utilities",
    "Scaffold_Structure",
    "Semi_Fixed_Obstacles",
    "Large_Materials",
    "Stored_Equipment",
    "Mobile_Machines_And_Vehicles",
    "Movable_Objects",
    "Containers_And_Pallets",
    "Small_Tools",
    "Portable_Objects",
]


def parse_yaml_class_weights(config_path: Path) -> np.ndarray:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*class_weights:\s*\[([^\]]+)\]", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find 'class_weights: [...]' in {config_path}")

    raw = match.group(1)
    vals: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))

    if len(vals) == 0:
        raise ValueError("Parsed empty class_weights list from config.")
    return np.asarray(vals, dtype=np.float64)


def parse_inline_weights(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty --base-weights list.")
    return np.asarray(vals, dtype=np.float64)


def parse_multiplier_map(spec: str, num_classes: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not spec.strip():
        return out
    pairs = [p.strip() for p in spec.split(",") if p.strip()]
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"Invalid pair '{p}', expected '<class_id>:<factor>'")
        k, v = p.split(":", 1)
        cid = int(k.strip())
        fac = float(v.strip())
        if cid < 0 or cid >= num_classes:
            raise ValueError(f"class_id {cid} out of range [0, {num_classes - 1}]")
        if fac <= 0:
            raise ValueError(f"factor must be > 0, got {fac} for class {cid}")
        out[cid] = fac
    return out


def ce_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / np.sum(counts)
    return 1.0 / (freq + 0.02)


def fix_integer_sum(counts: np.ndarray, target_sum: int) -> np.ndarray:
    """Round then enforce exact sum by adjusting largest buckets first."""
    out = np.maximum(1, np.rint(counts).astype(np.int64))
    diff = int(target_sum - int(out.sum()))
    if diff == 0:
        return out

    order = np.argsort(out)
    if diff > 0:
        # Add to largest counts first.
        idx_cycle = order[::-1]
        i = 0
        while diff > 0:
            out[idx_cycle[i % len(out)]] += 1
            i += 1
            diff -= 1
    else:
        # Remove from largest counts first while staying >= 1.
        idx_cycle = order[::-1]
        i = 0
        while diff < 0:
            idx = idx_cycle[i % len(out)]
            if out[idx] > 1:
                out[idx] -= 1
                diff += 1
            i += 1
            if i > 10_000_000:
                raise RuntimeError("Failed to rebalance count sum safely.")
    return out


def counts_from_target_ce(target_ce: np.ndarray, total_points: int) -> np.ndarray:
    """
    Invert Open3D CE formula:
      target_ce = 1 / (p + 0.02)  ->  p = (1 / target_ce) - 0.02
    Then convert p to integer counts with the same total.
    """
    p = (1.0 / target_ce) - 0.02
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    raw_counts = p * float(total_points)
    return fix_integer_sum(raw_counts, total_points)


def apply_count_mode(
    base_counts: np.ndarray,
    count_mult: Dict[int, float],
    keep_total: bool,
) -> np.ndarray:
    out = base_counts.astype(np.float64).copy()
    for cid, fac in count_mult.items():
        out[cid] *= fac
    out = np.clip(out, 1.0, None)

    if keep_total:
        scale = float(base_counts.sum()) / float(out.sum())
        out = out * scale
    return fix_integer_sum(out, int(base_counts.sum()))


def apply_ce_mode(
    base_counts: np.ndarray,
    ce_mult: Dict[int, float],
    ce_power: float,
) -> np.ndarray:
    if ce_power <= 0:
        raise ValueError("--ce-power must be > 0")

    base_ce = ce_from_counts(base_counts)
    target_ce = np.power(base_ce, ce_power)
    for cid, fac in ce_mult.items():
        target_ce[cid] *= fac
    target_ce = np.clip(target_ce, 1e-6, None)
    return counts_from_target_ce(target_ce, int(base_counts.sum()))


def print_table(before_counts: np.ndarray, after_counts: np.ndarray) -> None:
    before_ce = ce_from_counts(before_counts)
    after_ce = ce_from_counts(after_counts)
    names = CLASS_NAMES
    if len(names) < len(before_counts):
        names = names + [f"class_{i}" for i in range(len(names), len(before_counts))]

    print("\nPer-class summary:")
    print("id  class                               count_before -> count_after   ce_before -> ce_after")
    for i in range(len(before_counts)):
        print(
            f"{i:2d}  {names[i]:<34s}"
            f"{int(before_counts[i]):>12,d} -> {int(after_counts[i]):>11,d}   "
            f"{before_ce[i]:>8.3f} -> {after_ce[i]:>8.3f}"
        )

    print("\nTotals:")
    print(f"  total_count_before: {int(before_counts.sum()):,}")
    print(f"  total_count_after:  {int(after_counts.sum()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune class weights for PointTransformer S3DIS")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/jeremychia/Documents/Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml"),
        help="Config file to read current class_weights from.",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="",
        help="Optional inline class_weights list (comma-separated). Overrides --config.",
    )
    parser.add_argument(
        "--mode",
        choices=["counts", "ce"],
        default="ce",
        help="`counts`: edit counts directly, `ce`: edit effective CE weights then invert to counts.",
    )
    parser.add_argument(
        "--count-mult",
        type=str,
        default="",
        help="Class multipliers for count mode, e.g. '0:1.2,14:2.0'",
    )
    parser.add_argument(
        "--ce-mult",
        type=str,
        default="",
        help="Class multipliers for CE mode, e.g. '14:0.55,15:0.7,0:1.15'",
    )
    parser.add_argument(
        "--ce-power",
        type=float,
        default=1.0,
        help="Global CE flattening/sharpening in CE mode. <1 flattens rare-class dominance.",
    )
    parser.add_argument(
        "--keep-total",
        action="store_true",
        default=True,
        help="Keep total points unchanged (always true for CE mode).",
    )
    args = parser.parse_args()

    if args.base_weights.strip():
        before = parse_inline_weights(args.base_weights)
    else:
        before = parse_yaml_class_weights(args.config)

    n_cls = len(before)
    if args.mode == "counts":
        mult = parse_multiplier_map(args.count_mult, n_cls)
        after = apply_count_mode(before, mult, keep_total=args.keep_total)
    else:
        mult = parse_multiplier_map(args.ce_mult, n_cls)
        after = apply_ce_mode(before, mult, ce_power=args.ce_power)

    print("\n" + "=" * 72)
    print("PointTransformer Weight Fine-tuning")
    print("=" * 72)
    print(f"mode: {args.mode}")
    if args.mode == "counts":
        print(f"count multipliers: {mult if mult else '{}'}")
    else:
        print(f"ce multipliers: {mult if mult else '{}'}")
        print(f"ce power: {args.ce_power}")

    print_table(before, after)

    after_list = [int(x) for x in after.tolist()]
    print("\nPaste into pointtransformer_s3dis.yml:")
    print(f"  class_weights: {after_list}")
    print("=" * 72)


if __name__ == "__main__":
    main()


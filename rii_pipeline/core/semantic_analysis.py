import sys
import math
import time
import numpy as np
from collections import deque

from PyQt5.QtGui import QImage

from config import PCD_PACKAGE_DIR
if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)
from pcd_package.pcd_tools import load_xyz_and_labels

from core.RII_horizontal import _score_accessibility_from_masks, _footprint_inflation_pixels
from core.rendering import _build_bg


SEMANTIC_LABEL_NAMES = {
    0: "Unlabelled / Background",
    1: "Wall",
    3: "Staircase",
    4: "Fixed Obstacles",
    5: "Temporary Ramps",
    6: "Safety Barriers and Signs",
    7: "Temporary Utilities",
    8: "Scaffold Structure",
    9: "Semi-Fixed Obstacles",
    10: "Large Materials",
    11: "Stored Equipment",
    12: "Mobile Machines and Vehicles",
    13: "Movable Objects",
    14: "Containers and Pallets",
    15: "Small Tools",
    17: "Portable Objects",
}

# All valid raw label IDs (used for iteration instead of range(16))
SEMANTIC_RAW_LABEL_IDS = sorted(SEMANTIC_LABEL_NAMES.keys())

SEMANTIC_FIXATION_GROUPS = {
    0: "Fixed",        # Unlabelled / Background
    1: "Fixed",        # Wall
    3: "Fixed",        # Staircase
    4: "Fixed",        # Fixed Obstacles
    5: "Semi-Fixed",   # Temporary Ramps
    6: "Semi-Fixed",   # Safety Barriers and Signs
    7: "Semi-Fixed",   # Temporary Utilities
    8: "Semi-Fixed",   # Scaffold Structure
    9: "Semi-Fixed",   # Semi-Fixed Obstacles
    10: "Movable",     # Large Materials
    11: "Movable",     # Stored Equipment
    12: "Movable",     # Mobile Machines and Vehicles
    13: "Movable",     # Movable Objects
    14: "Portable",    # Containers and Pallets
    15: "Portable",    # Small Tools
    17: "Portable",    # Portable Objects
}

SEMANTIC_RECOMMENDATIONS = {
    0: "Large unlabeled regions are reducing interpretability. Revisit the CloudCompare annotations so the gap can be tied to the correct construction elements.",
    1: "Wall-related gap is a hard structural constraint. Improve RII Horizontal by changing the route, robot footprint, or deployment zone rather than trying to move the wall.",
    3: "Staircases are fixed vertical barriers. Improve horizontal accessibility with alternative ramped routes, lift access, or a different robot platform.",
    4: "Fixed obstacles are permanent constraints. Consider site-layout changes, alternate access corridors, or a smaller robot footprint.",
    5: "Temporary ramps are semi-fixed and usually offer the best short-term intervention: adjust placement, width, or gradient so the robot can traverse them safely.",
    6: "Barrier/sign placement is a recoverable gap. Reposition compliant safety barriers and signs to reopen the corridor while maintaining site safety rules.",
    7: "Temporary utilities are often recoverable. Reroute hoses, cables, or power runs overhead or along protected edges to free the travel corridor.",
    8: "Scaffold placement is a semi-fixed constraint. Leave a protected robot corridor or reschedule traversal around scaffold-heavy phases.",
    9: "Semi-fixed obstacles should be reviewed during phase planning. Reposition them or create a temporary bypass to recover accessible area.",
    10: "Large materials are movable. Improve RII Horizontal through material staging zones, just-in-time delivery, or clearing buffer space around routes.",
    11: "Stored equipment is movable. Define dedicated parking/storage zones so equipment does not reduce corridor clearance.",
    12: "Mobile machines and vehicles are dynamic movable constraints. Use time windows, exclusion zones, or coordinated traffic rules to recover access.",
    13: "Movable objects indicate housekeeping issues. Regular clearing or better local storage can recover this portion of the RII gap quickly.",
    14: "Containers and pallets are portable and highly actionable. Relocating pallet stacks or container staging areas should recover this gap first.",
    15: "Small tools are portable clutter. Apply 5S-style housekeeping or tool-drop zones to restore traversable space.",
    17: "Portable objects are the most easily recoverable source of lost accessibility. Focus cleanup here before changing semi-fixed or fixed site elements.",
}

SEMANTIC_3D_COLORS = {
    0: (80, 80, 80),      # Unlabelled — dark gray (black in CC, brightened for visibility)
    1: (255, 170, 255),   # Wall — pink
    3: (249, 241, 0),     # Staircase — yellow
    4: (85, 170, 255),    # Fixed Obstacles — blue
    5: (255, 0, 0),       # Temporary Ramps — red
    6: (142, 106, 36),    # Safety Barriers — brown
    7: (255, 0, 127),     # Temporary Utilities — magenta-red
    8: (170, 255, 255),   # Scaffold Structure — cyan
    9: (170, 170, 255),   # Semi-Fixed Obstacles — periwinkle
    10: (85, 85, 0),      # Large Materials — olive
    11: (132, 132, 140),  # Stored Equipment — gray
    12: (170, 255, 0),    # Mobile Machines — lime
    13: (179, 20, 176),   # Movable Objects — purple
    14: (0, 170, 255),    # Containers — sky blue
    15: (255, 85, 127),   # Small Tools — coral
    17: (250, 255, 147),  # Portable Objects — pale yellow
}

SEMANTIC_REMOVABLE_FIXATIONS = ("Portable", "Movable", "Semi-Fixed")
SEMANTIC_LAYER_SEQUENCE = [
    ("Actual site state", ()),
    ("Portable removed", ("Portable",)),
    ("Portable + Movable removed", ("Portable", "Movable")),
    ("Structural maximum (only Fixed remains)", ("Portable", "Movable", "Semi-Fixed")),
]


def load_semantic_pcd(pcd_path):
    """Load a CloudCompare-labeled PCD or PLY using the shared parser."""
    try:
        pts, labels, label_field = load_xyz_and_labels(pcd_path)
    except Exception:
        return None, None, None

    if labels is not None:
        labels = np.clip(labels, 0, 17)
    return pts, labels, label_field


def project_labels_to_2d_grid(pts, labels, yaml_data, map_w, map_h):
    """Project 3D semantic labels onto the 2D occupancy grid.

    For each grid cell, assigns the most common label among all 3D points
    that fall within that cell's horizontal footprint.

    Returns:
        label_grid: (map_h * map_w) int array, -1 = no label data
    """
    res = yaml_data['resolution']
    ox, oy = yaml_data['origin'][0], yaml_data['origin'][1]

    label_grid = np.full(map_h * map_w, -1, dtype=np.int32)

    # Convert 3D points to grid indices
    gx = ((pts[:, 0] - ox) / res).astype(np.int32)
    gy = ((pts[:, 1] - oy) / res).astype(np.int32)

    # Filter points within grid bounds
    valid = (gx >= 0) & (gx < map_w) & (gy >= 0) & (gy < map_h)
    gx, gy, lab = gx[valid], gy[valid], labels[valid]

    # For each cell, use the most frequent label (mode)
    # Use accumulation: for each cell, track label counts
    cell_indices = gy * map_w + gx

    # Group by cell and find mode
    for label_val in SEMANTIC_RAW_LABEL_IDS:
        mask = lab == label_val
        if not np.any(mask):
            continue
        cells = cell_indices[mask]
        unique_cells, cell_counts = np.unique(cells, return_counts=True)
        for ci, count in zip(unique_cells, cell_counts):
            if label_grid[ci] == -1:
                label_grid[ci] = label_val
            else:
                # Check if this label has more points in this cell
                # Simple approach: last-wins for ties, most-common otherwise
                # We'll do a simpler approach: highest count wins
                pass  # first-assigned approach for speed

    # More accurate approach: direct assignment by most common
    # Build count arrays per cell
    if len(cell_indices) > 0:
        label_counts = np.zeros((map_h * map_w, 16), dtype=np.int32)
        for i in range(len(cell_indices)):
            ci = cell_indices[i]
            li = lab[i]
            label_counts[ci, li] += 1

        has_data = label_counts.sum(axis=1) > 0
        label_grid[has_data] = label_counts[has_data].argmax(axis=1)

    return label_grid


def analyze_semantic_rii(ref_result, act_result, label_grid, yaml_data, label_names=None):
    """Analyze which semantic categories contribute to the RII gap.

    Args:
        ref_result: reference coverage result dict
        act_result: actual coverage result dict
        label_grid: (h*w) array of semantic labels (-1 = no data)
        yaml_data: map YAML config
        label_names: dict mapping label_id -> name

    Returns:
        dict with:
          - total_missed_area: float (m²)
          - label_breakdown: list of {label, name, area, pct, recommendation}
          - top_recommendations: list of strings
    """
    if label_names is None:
        label_names = SEMANTIC_LABEL_NAMES

    w, h = ref_result['w'], ref_result['h']
    res = yaml_data['resolution']
    cell_area = res * res

    ref_cov = ref_result['covPx']
    act_cov = act_result['covPx']

    # Missed = covered by reference but NOT by actual
    missed = (ref_cov == 1) & (act_cov == 0)
    total_missed = float(np.sum(missed)) * cell_area

    # Break down by semantic label
    breakdown = []
    fixation_totals = {}
    for label_id in SEMANTIC_RAW_LABEL_IDS:
        in_label = (label_grid == label_id)
        label_missed = missed & in_label
        area = float(np.sum(label_missed)) * cell_area
        if area > 0:
            pct = (area / total_missed * 100) if total_missed > 0 else 0
            fixation = SEMANTIC_FIXATION_GROUPS.get(label_id, "Unknown")
            fixation_totals[fixation] = fixation_totals.get(fixation, 0.0) + area
            breakdown.append({
                'label': label_id,
                'name': label_names.get(label_id, f"Label {label_id}"),
                'fixation': fixation,
                'area': area,
                'pct': pct,
                'recommendation': SEMANTIC_RECOMMENDATIONS.get(label_id, "Review this area."),
            })

    # Also count missed area with no label data
    no_label_missed = missed & (label_grid == -1)
    no_label_area = float(np.sum(no_label_missed)) * cell_area
    if no_label_area > 0:
        pct = (no_label_area / total_missed * 100) if total_missed > 0 else 0
        breakdown.append({
            'label': -1,
            'name': "No Label Data",
            'fixation': "Unknown",
            'area': no_label_area,
            'pct': pct,
            'recommendation': "No semantic data for this area — ensure the labeled point cloud covers the full map extent.",
        })

    # Sort by area descending
    breakdown.sort(key=lambda x: x['area'], reverse=True)

    # Prioritize interventions that are easiest to act on first.
    priority = {"Portable": 0, "Movable": 1, "Semi-Fixed": 2, "Fixed": 3, "Unknown": 4}
    actionable = [b for b in breakdown if b['label'] not in (-1, 0, 1) and b['area'] > 0]
    actionable.sort(key=lambda b: (priority.get(b['fixation'], 99), -b['area']))
    top_recs = []
    for b in actionable[:3]:
        top_recs.append(
            f"• {b['name']} [{b['fixation']}] ({b['pct']:.1f}% of gap, {b['area']:.2f} m²): {b['recommendation']}"
        )

    fixation_breakdown = [
        {
            "fixation": fixation,
            "area": area,
            "pct": (area / total_missed * 100) if total_missed > 0 else 0.0,
        }
        for fixation, area in sorted(
            fixation_totals.items(),
            key=lambda item: (priority.get(item[0], 99), -item[1]),
        )
    ]

    return {
        'total_missed_area': total_missed,
        'label_breakdown': breakdown,
        'fixation_breakdown': fixation_breakdown,
        'top_recommendations': top_recs,
    }


def compute_semantic_layered_rii(act_result, label_grid, logf=None, progress_cb=None):
    """Recompute horizontal RII while progressively removing fixation groups."""
    L = logf if logf else lambda m, c="": None
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result["resolution"])
    params = dict(act_result.get("params", {}))
    use_stc = bool(act_result.get("useSTC"))
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]),
        dtype=np.uint8,
    ).reshape(h, w)
    floor_mask = act_result.get("floorPx")
    floor2d = np.asarray(floor_mask, dtype=np.uint8).reshape(h, w) if isinstance(floor_mask, np.ndarray) else None
    label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
    cell_area = res * res

    layers = []
    for idx, (name, excluded_fixations) in enumerate(SEMANTIC_LAYER_SEQUENCE):
        removed_mask = np.zeros((h, w), dtype=bool)
        if idx == 0:
            layer_result = dict(act_result)
        else:
            excluded_labels = [
                label_id
                for label_id, fixation in SEMANTIC_FIXATION_GROUPS.items()
                if fixation in excluded_fixations
            ]
            if excluded_labels:
                removed_mask = (source_blocked == 1) & np.isin(label2d, excluded_labels)
            test_blocked = np.asarray(source_blocked, dtype=np.uint8).copy()
            test_blocked[removed_mask] = 0
            layer_result = _score_accessibility_from_masks(
                test_blocked,
                floor2d,
                res,
                params,
                f"LAYER {idx}",
                L,
                use_stc=use_stc,
            )
            layer_result.update(
                sourceBlocked=test_blocked.ravel().copy(),
                params=dict(params),
                resolution=float(res),
                origin=tuple(act_result.get("origin", (0.0, 0.0))),
                removedCells=int(removed_mask.sum()),
                removedArea=float(removed_mask.sum()) * cell_area,
            )

        rii = float(layer_result.get("riiHorizontal", 0.0))
        accessible_area = float(layer_result.get("accessibleArea", layer_result.get("reachableArea", 0.0)))
        total_floor_area = float(layer_result.get("totalFloorArea", 0.0))
        removed_area = float(layer_result.get("removedArea", 0.0))
        if idx == 0:
            removed_area = 0.0
        layers.append({
            "layer": idx,
            "name": name,
            "excludedFixations": list(excluded_fixations),
            "riiHorizontal": rii,
            "accessibleArea": accessible_area,
            "totalFloorArea": total_floor_area,
            "removedArea": removed_area,
        })
        if progress_cb:
            progress_cb(idx + 1, len(SEMANTIC_LAYER_SEQUENCE), name)

    for idx, layer in enumerate(layers):
        prev = layers[idx - 1] if idx > 0 else None
        layer["deltaPts"] = 0.0 if prev is None else float(layer["riiHorizontal"] - prev["riiHorizontal"])
        layer["deltaArea"] = 0.0 if prev is None else float(layer["accessibleArea"] - prev["accessibleArea"])

    portable_delta = float(layers[1]["riiHorizontal"] - layers[0]["riiHorizontal"])
    movable_delta = float(layers[2]["riiHorizontal"] - layers[1]["riiHorizontal"])
    semi_fixed_delta = float(layers[3]["riiHorizontal"] - layers[2]["riiHorizontal"])
    return {
        "layers": layers,
        "rii_actual": float(layers[0]["riiHorizontal"]),
        "rii_structural_max": float(layers[-1]["riiHorizontal"]),
        "delta_portable": portable_delta,
        "delta_movable": movable_delta,
        "delta_semi_fixed": semi_fixed_delta,
        "improvement_potential": float(layers[-1]["riiHorizontal"] - layers[0]["riiHorizontal"]),
        "delta_portable_area": float(layers[1]["accessibleArea"] - layers[0]["accessibleArea"]),
        "delta_movable_area": float(layers[2]["accessibleArea"] - layers[1]["accessibleArea"]),
        "delta_semi_fixed_area": float(layers[3]["accessibleArea"] - layers[2]["accessibleArea"]),
    }


def simulate_removed_fixations(act_result, label_grid, excluded_fixations, label="FIXATION", logf=None):
    """Recompute horizontal RII after removing every blocked cell from the selected fixation groups."""
    excluded = tuple(dict.fromkeys(excluded_fixations))
    if not excluded:
        raise ValueError("Select one or more fixation groups first")

    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result["resolution"])
    params = dict(act_result.get("params", {}))
    use_stc = bool(act_result.get("useSTC"))
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]),
        dtype=np.uint8,
    ).reshape(h, w)
    floor_mask = act_result.get("floorPx")
    floor2d = np.asarray(floor_mask, dtype=np.uint8).reshape(h, w) if isinstance(floor_mask, np.ndarray) else None
    label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
    excluded_labels = [
        label_id
        for label_id, fixation in SEMANTIC_FIXATION_GROUPS.items()
        if fixation in excluded
    ]
    remove_mask = (source_blocked == 1) & np.isin(label2d, excluded_labels)
    modified = np.asarray(source_blocked, dtype=np.uint8).copy()
    modified[remove_mask] = 0
    improved = _score_accessibility_from_masks(
        modified,
        floor2d,
        res,
        params,
        label,
        logf,
        use_stc=use_stc,
    )
    improved.update(
        sourceBlocked=modified.ravel().copy(),
        params=dict(params),
        resolution=float(res),
        origin=tuple(act_result.get("origin", (0.0, 0.0))),
        removedCells=int(remove_mask.sum()),
        removedArea=float(remove_mask.sum()) * res * res,
        removedMode="fixation",
        excludedFixations=list(excluded),
    )
    return improved


def render_semantic_missed(ref_result, act_result, label_grid, bg_pgm=None):
    """Render an image showing missed areas colored by semantic label, overlaid on map.pgm."""
    w, h = ref_result['w'], ref_result['h']
    ref_cov = ref_result['covPx'].reshape(h, w)[::-1, :]
    act_cov = act_result['covPx'].reshape(h, w)[::-1, :]
    lg = label_grid.reshape(h, w)[::-1, :]
    blk = ref_result['blocked'].reshape(h, w)[::-1, :]

    LABEL_COLORS = SEMANTIC_3D_COLORS

    buf = _build_bg(h, w, ref_result, bg_pgm)
    _blend = bg_pgm is not None

    # Actual covered = green (blended with background)
    cov_mask = act_cov == 1
    if _blend:
        buf[cov_mask] = (0.35 * np.array([0, 200, 130], dtype=np.float32) + 0.65 * buf[cov_mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[cov_mask] = [0, 200, 130]

    # Missed areas colored by semantic label
    missed = (ref_cov == 1) & (act_cov == 0)
    for label_id in SEMANTIC_RAW_LABEL_IDS:
        mask = missed & (lg == label_id)
        if np.any(mask):
            if _blend:
                buf[mask] = (0.45 * np.array(LABEL_COLORS[label_id], dtype=np.float32) + 0.55 * buf[mask].astype(np.float32)).astype(np.uint8)
            else:
                buf[mask] = LABEL_COLORS[label_id]

    # Missed with no label data = dim red
    mask_no_label = missed & (lg == -1)
    buf[mask_no_label] = [100, 30, 30]

    # Blocked areas (only darken if no bg_pgm, since bg already shows them)
    if not _blend:
        buf[(act_cov == 0) & (ref_cov == 0) & (blk == 1)] = [20, 24, 28]

    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def _binary_components(mask2d: np.ndarray) -> list[np.ndarray]:
    height, width = mask2d.shape
    seen = np.zeros_like(mask2d, dtype=np.uint8)
    comps = []
    for row in range(height):
        for col in range(width):
            if mask2d[row, col] == 0 or seen[row, col]:
                continue
            q = deque([(row, col)])
            seen[row, col] = 1
            flat = []
            while q:
                rr, cc = q.popleft()
                flat.append(rr * width + cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask2d[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = 1
                        q.append((nr, nc))
            comps.append(np.asarray(flat, dtype=np.int32))
    return comps


def _integral_rect_sum(integral: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
    if r0 >= r1 or c0 >= c1:
        return 0
    total = int(integral[r1 - 1, c1 - 1])
    if r0 > 0:
        total -= int(integral[r0 - 1, c1 - 1])
    if c0 > 0:
        total -= int(integral[r1 - 1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(integral[r0 - 1, c0 - 1])
    return total


def identify_semantic_removal_candidates(
    act_result,
    label_grid,
    yaml_data,
    max_candidates: int = 120,
    min_component_area_m2: float = 0.04,
    progress_cb=None,
):
    """Find connected removable-object components that plausibly reduce horizontal accessibility."""
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(yaml_data["resolution"])
    ox, oy = float(yaml_data["origin"][0]), float(yaml_data["origin"][1])
    cell_area = res * res

    label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
    source_blocked = np.asarray(act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8).reshape(h, w)
    floor2d = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)
    cov2d = np.asarray(act_result["covPx"], dtype=np.uint8).reshape(h, w)
    inaccessible_floor = (floor2d == 1) & (cov2d == 0)
    inaccessible_integral = np.cumsum(
        np.cumsum(inaccessible_floor.astype(np.int32), axis=0),
        axis=1,
    )

    inflX, inflY, isRect = _footprint_inflation_pixels(act_result.get("params", {}), res)
    min_cells = max(1, int(math.ceil(min_component_area_m2 / max(cell_area, 1e-9))))

    priority = {"Portable": 0, "Movable": 1, "Semi-Fixed": 2}
    candidates = []
    candidate_id = 1
    removable_label_ids = [
        label_id for label_id, fixation in SEMANTIC_FIXATION_GROUPS.items()
        if fixation in SEMANTIC_REMOVABLE_FIXATIONS
    ]
    total_labels = max(1, len(removable_label_ids))
    for label_index, label_id in enumerate(removable_label_ids, start=1):
        fixation = SEMANTIC_FIXATION_GROUPS.get(label_id, "Unknown")
        base = ((source_blocked == 1) & (label2d == label_id)).astype(np.uint8)
        if not np.any(base):
            if progress_cb:
                progress_cb(label_index, total_labels, fixation, label_id)
            continue
        for flat_idx in _binary_components(base):
            if flat_idx.size < min_cells:
                continue
            rows = flat_idx // w
            cols = flat_idx % w
            r0 = max(0, int(rows.min()) - inflY)
            r1 = min(h, int(rows.max()) + inflY + 1)
            c0 = max(0, int(cols.min()) - inflX)
            c1 = min(w, int(cols.max()) + inflX + 1)
            # Approximate unlockable area from the inflation-expanded component bbox.
            # Exact recomputation still uses the real component mask; this fast pass is
            # only for ranking and listing semantic candidates interactively.
            potential_unlock_cells = _integral_rect_sum(
                inaccessible_integral,
                r0,
                r1,
                c0,
                c1,
            )
            if potential_unlock_cells <= 0:
                continue
            x0 = ox + float(cols.min()) * res
            x1 = ox + float(cols.max() + 1) * res
            y0 = oy + float(rows.min()) * res
            y1 = oy + float(rows.max() + 1) * res
            candidates.append({
                "id": candidate_id,
                "label": int(label_id),
                "name": SEMANTIC_LABEL_NAMES.get(label_id, f"Label {label_id}"),
                "fixation": fixation,
                "area": float(flat_idx.size) * cell_area,
                "cells": int(flat_idx.size),
                "potentialUnlockArea": float(potential_unlock_cells) * cell_area,
                "potentialUnlockCells": potential_unlock_cells,
                "indices": flat_idx,
                "bboxWorld": (x0, x1, y0, y1),
                "recommendation": SEMANTIC_RECOMMENDATIONS.get(label_id, "Review this object."),
            })
            candidate_id += 1
        if progress_cb:
            progress_cb(label_index, total_labels, fixation, label_id)

    candidates.sort(
        key=lambda c: (
            priority.get(c["fixation"], 99),
            -c["potentialUnlockArea"],
            -c["area"],
        )
    )
    return candidates[:max_candidates]


def simulate_removed_candidates(act_result, candidate_list, selected_ids, label="IMPROVED", logf=None):
    """Recompute horizontal RII after clearing the selected removable semantic components."""
    selected = {int(v) for v in selected_ids}
    if not selected:
        raise ValueError("No removable objects were selected")

    w, h = int(act_result["w"]), int(act_result["h"])
    source_blocked = np.asarray(act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8).copy()
    remove_mask = np.zeros_like(source_blocked, dtype=np.uint8)
    picked = []
    for candidate in candidate_list:
        if int(candidate["id"]) not in selected:
            continue
        remove_mask[candidate["indices"]] = 1
        picked.append(candidate)
    if not picked:
        raise ValueError("Selected removable objects are no longer available")

    source_blocked[remove_mask == 1] = 0
    floor_mask = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)
    improved = _score_accessibility_from_masks(
        source_blocked.reshape(h, w),
        floor_mask,
        float(act_result["resolution"]),
        dict(act_result.get("params", {})),
        label,
        logf,
        use_stc=bool(act_result.get("useSTC")),
    )
    improved.update(
        sourceBlocked=source_blocked.copy(),
        params=dict(act_result.get("params", {})),
        resolution=float(act_result["resolution"]),
        origin=tuple(act_result.get("origin", (0.0, 0.0))),
        selectedCandidateIds=sorted(selected),
        removedCells=int(remove_mask.sum()),
        removedArea=float(remove_mask.sum()) * float(act_result["resolution"]) * float(act_result["resolution"]),
    )
    return improved


def render_semantic_candidates(ref_result, act_result, label_grid, candidates, selected_ids=None, focused_id=None, bg_pgm=None):
    """Render the semantic gap plus removable-object candidates, overlaid on map.pgm."""
    selected = {int(v) for v in (selected_ids or [])}
    focused = None if focused_id is None else int(focused_id)
    w, h = ref_result['w'], ref_result['h']
    ref_cov = ref_result['covPx'].reshape(h, w)[::-1, :]
    act_cov = act_result['covPx'].reshape(h, w)[::-1, :]
    lg = label_grid.reshape(h, w)[::-1, :]
    blk = np.asarray(act_result.get('sourceBlocked', act_result['blocked']), dtype=np.uint8).reshape(h, w)[::-1, :]

    buf = _build_bg(h, w, act_result, bg_pgm)
    _blend = bg_pgm is not None

    missed = (ref_cov == 1) & (act_cov == 0)
    # Color missed areas by semantic label (matching the 3D viewer palette)
    for label_id in SEMANTIC_RAW_LABEL_IDS:
        mask = missed & (lg == label_id)
        if np.any(mask):
            if _blend:
                buf[mask] = (0.45 * np.array(SEMANTIC_3D_COLORS[label_id], dtype=np.float32) + 0.55 * buf[mask].astype(np.float32)).astype(np.uint8)
            else:
                buf[mask] = SEMANTIC_3D_COLORS[label_id]
    # Missed with no label data = dim red
    mask_no_label = missed & (lg == -1)
    if np.any(mask_no_label):
        buf[mask_no_label] = [100, 30, 30]

    cov_mask = act_cov == 1
    if _blend:
        buf[cov_mask] = (0.35 * np.array([0, 200, 130], dtype=np.float32) + 0.65 * buf[cov_mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[cov_mask] = [0, 200, 130]

    if not _blend:
        buf[(act_cov == 0) & (blk == 1)] = [20, 24, 28]

    for candidate in candidates:
        rows = candidate["indices"] // w
        cols = candidate["indices"] % w
        disp_rows = h - 1 - rows
        cid = int(candidate["id"])
        lid = candidate.get("label", -1)
        if cid == focused:
            color = np.array([80, 220, 255], dtype=np.uint8)
        elif cid in selected:
            color = np.array([255, 90, 180], dtype=np.uint8)
        else:
            color = np.array(SEMANTIC_3D_COLORS.get(lid, (255, 190, 0)), dtype=np.uint8)
        buf[disp_rows, cols] = color
        if cid == focused:
            r0 = max(0, int(disp_rows.min()) - 2)
            r1 = min(h - 1, int(disp_rows.max()) + 2)
            c0 = max(0, int(cols.min()) - 2)
            c1 = min(w - 1, int(cols.max()) + 2)
            buf[r0:r1 + 1, c0] = [80, 220, 255]
            buf[r0:r1 + 1, c1] = [80, 220, 255]
            buf[r0, c0:c1 + 1] = [80, 220, 255]
            buf[r1, c0:c1 + 1] = [80, 220, 255]

    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()

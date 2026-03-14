"""RII Horizontal — inflation, BFS reachability, path planners, coverage computation."""

import os
import sys
import time
import math
import numpy as np
from collections import deque

from core.map_io import parse_pgm, parse_yaml

from config import PCD_PACKAGE_DIR
if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)
from pcd_package.pcd_tools import estimate_ground_preserving_preset


STC_ANALYSIS_CELL_M = 0.20
TERRAIN_SIDECAR_MIN_BAND_M = 2.0
BLOCKED_MAP_VIEW = "Obstacle Map"
PRIMARY_SELECTION_VIEW = "Obstacle Map"


def derive_terrain_sidecar_bounds(
    points: np.ndarray,
    obstacle_min_z: float,
    obstacle_max_z: float,
    min_band_m: float = TERRAIN_SIDECAR_MIN_BAND_M,
):
    """Choose a floor-preserving z range for RII terrain sidecars."""
    preset = estimate_ground_preserving_preset(points)
    obstacle_min_z = float(obstacle_min_z)
    obstacle_max_z = float(obstacle_max_z)
    band_ok = obstacle_max_z > obstacle_min_z and (obstacle_max_z - obstacle_min_z) >= float(min_band_m)
    if band_ok:
        return obstacle_min_z, obstacle_max_z, {
            "source": "requested",
            "floor_anchor_z": float(preset["floor_anchor_z"]),
            "cleanup_min_z": float(preset["cleanup_min_z"]),
            "cleanup_max_z": float(preset["cleanup_max_z"]),
        }
    return float(preset["cleanup_min_z"]), float(preset["cleanup_max_z"]), {
        "source": "preset_cleanup",
        "floor_anchor_z": float(preset["floor_anchor_z"]),
        "cleanup_min_z": float(preset["cleanup_min_z"]),
        "cleanup_max_z": float(preset["cleanup_max_z"]),
    }


def _largest_component_on_coarse_mask(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Approximate STC-reachable area as the largest connected free region on a coarse grid."""
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw = math.ceil(w / step)
    ch = math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels = np.full((ch, cw), -1, dtype=np.int32)
    component_sizes = []
    for row in range(ch):
        for col in range(cw):
            if coarse[row, col] == 0 or labels[row, col] != -1:
                continue
            comp_id = len(component_sizes)
            q = deque([(row, col)])
            labels[row, col] = comp_id
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < ch and 0 <= nc < cw and coarse[nr, nc] == 1 and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            component_sizes.append(size)

    if not component_sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []

    largest_id = int(np.argmax(component_sizes))
    largest_tiles = (labels == largest_id)
    up = np.repeat(np.repeat(largest_tiles, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & (accessible2d.astype(bool))).astype(np.uint8)
    return mask, len(component_sizes), int(component_sizes[largest_id]), step, _stc_stroke_on_mask(largest_tiles)


def _stc_stroke_on_mask(mask: np.ndarray):
    """Return an STC-style Euler tour over a 4-connected boolean grid."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return []
    start = (int(idx[0][0]), int(idx[0][1]))

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parent = np.full((height, width, 2), -1, dtype=np.int32)
    nbrs = ((0, 1), (0, -1), (1, 0), (-1, 0))

    stack = [start]
    seen[start] = True
    while stack:
        row, col = stack.pop()
        for dr, dc in nbrs:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                parent[nr, nc] = (row, col)
                stack.append((row, col))
                stack.append((nr, nc))
                break

    children = [[[] for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            pr, pc = parent[row, col]
            if pr >= 0:
                children[int(pr)][int(pc)].append((row, col))
    for row in range(height):
        for col in range(width):
            children[row][col].sort()

    stroke = [start]
    stack2 = [(start, 0)]
    while stack2:
        node, child_idx = stack2[-1]
        row, col = node
        ch_list = children[row][col]
        if child_idx < len(ch_list):
            child = ch_list[child_idx]
            stack2[-1] = (node, child_idx + 1)
            stroke.append(child)
            stack2.append((child, 0))
        else:
            stack2.pop()
            if stack2:
                stroke.append(stack2[-1][0])
    return stroke


# ── Path planner registry ──────────────────────────────────────────────
PLANNER_NAMES = [
    "Spanning Tree Coverage (STC)",
    "Boustrophedon Cellular Decomposition (BCD)",
    "Wavefront Coverage",
    "Morse-based Cellular Decomposition (Morse)",
    "Frontier-based Exploration (Frontier)",
]
# Map display name → internal dispatch key
_PLANNER_KEY = {
    "Spanning Tree Coverage (STC)": "STC",
    "Boustrophedon Cellular Decomposition (BCD)": "BCD",
    "Wavefront Coverage": "Wavefront",
    "Morse-based Cellular Decomposition (Morse)": "Morse",
    "Frontier-based Exploration (Frontier)": "Frontier",
}


def _connect_path(waypoints, free_mask):
    """Insert BFS shortest-path segments between non-adjacent waypoints.

    Given a list of (row, col) waypoints on a boolean *free_mask*, return a
    fully 4-connected path that never crosses blocked cells.  Adjacent
    waypoints (Manhattan distance 1) are kept as-is; distant ones are bridged
    with BFS through free space.
    """
    if len(waypoints) <= 1:
        return list(waypoints)
    h, w = free_mask.shape
    connected = [waypoints[0]]
    for i in range(1, len(waypoints)):
        prev = waypoints[i - 1]
        cur = waypoints[i]
        dr = abs(cur[0] - prev[0])
        dc = abs(cur[1] - prev[1])
        if dr + dc <= 1:
            connected.append(cur)
            continue
        # BFS from prev to cur on free_mask
        seen = np.zeros((h, w), dtype=bool)
        parent = {}
        q = deque([prev])
        seen[prev] = True
        found = False
        while q:
            rr, cc = q.popleft()
            if (rr, cc) == cur:
                found = True
                break
            for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + d_r, cc + d_c
                if 0 <= nr < h and 0 <= nc < w and free_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    parent[(nr, nc)] = (rr, cc)
                    q.append((nr, nc))
        if found:
            seg = []
            node = cur
            while node != prev:
                seg.append(node)
                node = parent[node]
            seg.reverse()
            connected.extend(seg)
        else:
            # Unreachable — just append the waypoint (gap in path)
            connected.append(cur)
    return connected


def _bfs_largest_component(mask2d):
    """Return (labels, component_sizes) for 4-connected components on a boolean 2D mask."""
    h, w = mask2d.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not mask2d[r, c] or labels[r, c] != -1:
                continue
            cid = len(sizes)
            q = deque([(r, c)])
            labels[r, c] = cid
            sz = 0
            while q:
                rr, cc = q.popleft()
                sz += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask2d[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            sizes.append(sz)
    return labels, sizes


def _keep_largest(accessible2d, labels, sizes):
    """Zero out everything except the largest connected component; return (mask, n_comps, largest_size)."""
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0
    lid = int(np.argmax(sizes))
    keep = (labels == lid)
    mask = (keep & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid])


# ── BCD (Boustrophedon Cellular Decomposition) ────────────────────────
def _run_bcd(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Boustrophedon path over the largest connected accessible region.

    Chobanyan & Choset, "Coverage of Known Spaces: The Boustrophedon
    Cellular Decomposition", 1998.

    The decomposition is approximated on a coarse grid: each column is swept
    top-to-bottom then bottom-to-top in alternating directions (ox-plough
    pattern). Only the largest connected component is kept.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Boustrophedon sweep: alternate column direction
    waypoints = []
    for col in range(cw):
        rows = range(ch) if col % 2 == 0 else range(ch - 1, -1, -1)
        for row in rows:
            if comp[row, col]:
                waypoints.append((row, col))

    path = _connect_path(waypoints, comp)
    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


# ── Wavefront Coverage ─────────────────────────────────────────────────
def _run_wavefront(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Wavefront (distance-transform) coverage path.

    Zelinsky et al., "Planning Paths of Complete Coverage of an
    Unstructured Environment by a Mobile Robot", 1993.

    A BFS wavefront expands from the centre of the largest component,
    assigning distance values. The path then follows cells in descending
    distance order (farthest-first), producing a spiral-inward trajectory.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Find centroid of largest component as wavefront seed
    idx = np.argwhere(comp)
    cr, cc = int(idx[:, 0].mean()), int(idx[:, 1].mean())
    if not comp[cr, cc]:
        cr, cc = int(idx[0, 0]), int(idx[0, 1])

    # BFS wavefront from centroid
    dist = np.full((ch, cw), -1, dtype=np.int32)
    dist[cr, cc] = 0
    q = deque([(cr, cc)])
    while q:
        rr, rc_ = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, rc_ + dc
            if 0 <= nr < ch and 0 <= nc < cw and comp[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[rr, rc_] + 1
                q.append((nr, nc))

    # Collect cells with valid distance, sort descending (farthest first → spiral inward)
    cells = [(int(dist[r, c]), r, c) for r in range(ch) for c in range(cw) if dist[r, c] >= 0]
    cells.sort(key=lambda x: -x[0])

    # Greedy nearest-neighbour ordering to reduce path jumps
    waypoints = _greedy_nearest_order([(r, c) for _, r, c in cells])
    path = _connect_path(waypoints, comp)

    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


def _greedy_nearest_order(cells):
    """Reorder cells via greedy nearest-neighbour to produce a smooth path."""
    if len(cells) <= 1:
        return list(cells)
    remaining = set(range(len(cells)))
    order = [0]
    remaining.discard(0)
    while remaining:
        cr, cc = cells[order[-1]]
        best = None
        best_d = float('inf')
        for i in remaining:
            d = abs(cells[i][0] - cr) + abs(cells[i][1] - cc)
            if d < best_d:
                best_d = d
                best = i
        order.append(best)
        remaining.discard(best)
    return [cells[i] for i in order]


# ── Morse-based Cellular Decomposition ─────────────────────────────────
def _run_morse(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Morse-based cellular decomposition coverage path.

    Acar & Choset, "Sensor-Based Coverage of Unknown Environments: Incremental
    Construction of Morse Decompositions", 2002.

    The free space is sliced into vertical strips at each column. Connected
    vertical segments within a column form Morse cells. Cells are linked to
    neighbours in adjacent columns and the path traverses cells in a
    depth-first order, sweeping each cell top-to-bottom or bottom-to-top.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Decompose into vertical segments (Morse cells) per column
    cells_by_col = {}  # col -> list of (start_row, end_row) tuples
    for col in range(cw):
        segs = []
        in_seg = False
        start = 0
        for row in range(ch):
            if comp[row, col]:
                if not in_seg:
                    start = row
                    in_seg = True
            else:
                if in_seg:
                    segs.append((start, row - 1))
                    in_seg = False
        if in_seg:
            segs.append((start, ch - 1))
        cells_by_col[col] = segs

    # Build adjacency graph between segments in neighbouring columns
    seg_id = {}
    seg_list = []
    for col in sorted(cells_by_col.keys()):
        for seg in cells_by_col[col]:
            seg_id[(col, seg)] = len(seg_list)
            seg_list.append((col, seg))

    adj = [[] for _ in seg_list]
    for col in range(cw - 1):
        for s1 in cells_by_col.get(col, []):
            for s2 in cells_by_col.get(col + 1, []):
                if s1[1] >= s2[0] and s2[1] >= s1[0]:  # overlapping row ranges
                    i, j = seg_id[(col, s1)], seg_id[(col + 1, s2)]
                    adj[i].append(j)
                    adj[j].append(i)

    # DFS traversal of segment graph
    visited = [False] * len(seg_list)
    waypoints = []
    if seg_list:
        stack = [0]
        sweep_down = True
        while stack:
            sid = stack.pop()
            if visited[sid]:
                continue
            visited[sid] = True
            col, (r0, r1) = seg_list[sid]
            if sweep_down:
                for r in range(r0, r1 + 1):
                    waypoints.append((r, col))
            else:
                for r in range(r1, r0 - 1, -1):
                    waypoints.append((r, col))
            sweep_down = not sweep_down
            for nb in adj[sid]:
                if not visited[nb]:
                    stack.append(nb)

    path = _connect_path(waypoints, comp)
    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


# ── Frontier-based Exploration ─────────────────────────────────────────
def _run_frontier(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Frontier-based exploration coverage path.

    Yamauchi, "A Frontier-Based Approach for Autonomous Exploration", 1997.

    In a coverage context the 'frontier' is the boundary between visited
    and unvisited free cells. Starting from the centroid, the planner
    repeatedly moves to the nearest frontier cell, marking cells as visited
    until all reachable cells are covered. This greedy strategy naturally
    prioritises nearby uncovered regions.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Start at centroid of largest component
    idx = np.argwhere(comp)
    cr, cc = int(idx[:, 0].mean()), int(idx[:, 1].mean())
    if not comp[cr, cc]:
        cr, cc = int(idx[0, 0]), int(idx[0, 1])

    visited = np.zeros((ch, cw), dtype=bool)
    path = []
    cur = (cr, cc)
    visited[cur] = True
    path.append(cur)
    total_free = int(comp.sum())

    # Precompute BFS distance field for nearest-frontier lookups
    def _bfs_to_nearest_frontier(start):
        """BFS from start; return first unvisited free cell found."""
        q2 = deque([start])
        seen = np.zeros((ch, cw), dtype=bool)
        seen[start] = True
        parent = {}
        while q2:
            rr, rc_ = q2.popleft()
            if comp[rr, rc_] and not visited[rr, rc_]:
                # Trace back to start to build path segment
                seg = []
                node = (rr, rc_)
                while node != start:
                    seg.append(node)
                    node = parent[node]
                seg.reverse()
                return seg
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, rc_ + dc
                if 0 <= nr < ch and 0 <= nc < cw and comp[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    parent[(nr, nc)] = (rr, rc_)
                    q2.append((nr, nc))
        return None

    while len(path) < total_free:
        seg = _bfs_to_nearest_frontier(cur)
        if seg is None:
            break
        for cell in seg:
            visited[cell] = True
            path.append(cell)
        cur = path[-1]

    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


# ── Planner dispatcher ─────────────────────────────────────────────────
_PLANNER_DISPATCH = {
    "STC": lambda acc, res: _largest_component_on_coarse_mask(acc, res),
    "BCD": _run_bcd,
    "Wavefront": _run_wavefront,
    "Morse": _run_morse,
    "Frontier": _run_frontier,
}


def run_planner(name: str, accessible2d: np.ndarray, resolution: float):
    """Run a named path planner.  Returns (mask, n_components, largest_size, step, path)."""
    key = _PLANNER_KEY.get(name, name)  # accept display name or short key
    fn = _PLANNER_DISPATCH.get(key)
    if fn is None:
        raise ValueError(f"Unknown planner: {name!r}. Choose from {list(_PLANNER_DISPATCH)}")
    return fn(accessible2d, resolution)


def _footprint_inflation_pixels(params, resolution: float) -> tuple:
    shape = params.get('shape', 'circular')
    halfW = params.get('halfW', params.get('radius', 0.35))
    halfL = params.get('halfL', params.get('radius', 0.35))
    inflX = max(0, math.ceil(halfW / resolution))
    inflY = max(0, math.ceil(halfL / resolution))
    isRect = (shape == 'rectangular')
    return inflX, inflY, isRect


def _dilate_binary_mask(mask2d: np.ndarray, inflX: int, inflY: int, isRect: bool) -> np.ndarray:
    base = np.asarray(mask2d, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("mask2d must be 2D")
    if inflX <= 0 and inflY <= 0:
        return base.copy()

    h, w = base.shape
    out = np.zeros((h, w), dtype=np.uint8)
    inflSq = inflX * inflX
    for dy in range(-inflY, inflY + 1):
        dy2 = dy * dy
        for dx in range(-inflX, inflX + 1):
            if not isRect and dx * dx + dy2 > inflSq:
                continue
            sy0 = max(0, -dy)
            sy1 = min(h, h - dy)
            sx0 = max(0, -dx)
            sx1 = min(w, w - dx)
            dy0 = max(0, dy)
            dy1 = min(h, h + dy)
            dx0 = max(0, dx)
            dx1 = min(w, w + dx)
            if sy1 > sy0 and sx1 > sx0:
                out[dy0:dy1, dx0:dx1] |= base[sy0:sy1, sx0:sx1]
    return out


def _score_accessibility_from_masks(
    blocked2d: np.ndarray,
    floor_mask2d,
    resolution: float,
    params: dict,
    label: str,
    logf=None,
    use_stc: bool = False,
    trav_mask2d=None,
    planner: str | None = None,
) -> dict:
    L = logf if logf else lambda m, c="": None
    t0 = time.time()
    h, w = blocked2d.shape
    inflX, inflY, isRect = _footprint_inflation_pixels(params, resolution)
    L(f"[{label}] Inflate: {'rect' if isRect else 'circle'} {inflX}x{inflY}px", "info")

    blocked_src = np.asarray(blocked2d, dtype=np.uint8)
    inflated2d = _dilate_binary_mask(blocked_src, inflX, inflY, isRect)
    if inflX > 0 or inflY > 0:
        L(f"[{label}] Inflation done: {time.time()-t0:.2f}s", "info")

    accessible2d = (inflated2d == 0).astype(np.uint8)
    if trav_mask2d is not None and floor_mask2d is not None:
        trav = np.asarray(trav_mask2d, dtype=np.uint8)
        floor = np.asarray(floor_mask2d, dtype=np.uint8)
        known_non_trav = (floor > 0) & (trav == 0)
        before = int(accessible2d.sum())
        accessible2d[known_non_trav] = 0
        excluded = before - int(accessible2d.sum())
        if excluded > 0:
            L(f"[{label}] Terrain constraint excluded {excluded} known-non-traversable cells", "info")
    if floor_mask2d is None:
        floor_mask2d = (inflated2d == 0).astype(np.uint8)
        L(f"[{label}] Floor denominator fallback: using post-mask free cells.", "warn")
    else:
        floor_mask2d = np.asarray(floor_mask2d, dtype=np.uint8)

    accessible2d &= floor_mask2d

    # Backwards compat: use_stc=True without planner → STC
    if planner is None and use_stc:
        planner = "STC"

    stc_components = 0
    stc_largest_tiles = 0
    stc_step = 1
    stc_path = []
    planner_name = _PLANNER_KEY.get(planner, planner) if planner else ""
    if planner:
        accessible2d, stc_components, stc_largest_tiles, stc_step, stc_path = run_planner(
            planner, accessible2d, resolution
        )
        L(
            f"[{label}] {planner} planner: largest connected region kept "
            f"({stc_components} components, largest={stc_largest_tiles} coarse cells, cell={stc_step * resolution:.3f} m)",
            "info",
        )

    accessible_cells = int(accessible2d.sum())
    accessible_area = float(accessible_cells) * resolution * resolution
    total_floor_cells = int(floor_mask2d.sum())
    total_floor_area = float(total_floor_cells) * resolution * resolution
    rii_horizontal = (accessible_area / total_floor_area * 100.0) if total_floor_area > 0 else 0.0

    L(f"[{label}] Total floor: {total_floor_area:.2f}m² ({total_floor_cells} px)", "info")
    L(f"[{label}] Inflated accessible: {accessible_area:.2f}m² ({accessible_cells} px)", "info")
    L(f"[{label}] Area paint: {time.time()-t0:.2f}s", "info")
    L(f"[{label}] Done: RII Horizontal={rii_horizontal:.1f}% ({accessible_area:.2f}/{total_floor_area:.2f}m²)", "success")

    return dict(
        coveredArea=accessible_area,
        reachableArea=accessible_area,
        accessibleArea=accessible_area,
        accessibleCells=accessible_cells,
        totalFloorArea=total_floor_area,
        totalFloorCells=total_floor_cells,
        riiHorizontal=rii_horizontal,
        useSTC=bool(planner),
        planner=planner_name or "",
        stcComponents=int(stc_components),
        stcLargestTiles=int(stc_largest_tiles),
        stcStep=int(stc_step),
        stcPath=stc_path,
        reachableCells=accessible_cells,
        waypoints=accessible_cells,
        blocked=inflated2d.ravel().copy(),
        floorPx=floor_mask2d.ravel().copy(),
        covPx=accessible2d.ravel().copy(),
        coarsePath=[],
        step=1,
        cw=w,
        ch=h,
        w=w,
        h=h,
    )


def run_coverage(
    pgm_path,
    yaml_path,
    params,
    start_x,
    start_y,
    sel_mask,
    label,
    logf=None,
    traversable_pgm_path=None,
    floor_pgm_path=None,
    use_stc=False,
    planner=None,
):
    """Horizontal RII area computation."""
    L = logf if logf else lambda m, c="": None
    t_total = time.time()
    P = lambda m: print(f"  [{label}] {m}")

    w, h, pixels = parse_pgm(pgm_path)
    yd = parse_yaml(yaml_path)
    res = yd['resolution']
    ox, oy = yd['origin'][0], yd['origin'][1]
    ft = yd['free_thresh']
    neg = yd['negate']
    fpt = math.ceil(255 * (1 - ft))

    P(f"Map: {w}x{h}, res={res}, fpt={fpt}")
    L(f"[{label}] Map: {w}x{h}, res={res}", "info")

    t0 = time.time()
    pix2d = pixels.reshape(h, w)
    flipped = pix2d[::-1, :].ravel().copy()
    if neg: flipped = 255 - flipped
    blocked = (flipped < fpt).astype(np.uint8)
    P(f"Blocked map: {time.time()-t0:.3f}s, blocked={int(np.sum(blocked))}, free={int(np.sum(blocked==0))}")
    L(f"[{label}] Blocked map: {time.time()-t0:.3f}s", "info")

    floor_mask = None
    if floor_pgm_path and os.path.isfile(floor_pgm_path):
        t0 = time.time()
        fw, fh, floor_pixels = parse_pgm(floor_pgm_path)
        if fw == w and fh == h:
            floor_flip = floor_pixels.reshape(h, w)[::-1, :].ravel().copy()
            if neg:
                floor_flip = 255 - floor_flip
            floor_mask = (floor_flip >= fpt).astype(np.uint8)
            L(f"[{label}] Floor mask: {time.time()-t0:.3f}s", "info")
        else:
            L(
                f"[{label}] Floor mask size mismatch: "
                f"{fw}x{fh} vs map {w}x{h}. Ignoring floor sidecar.",
                "warn",
            )

    trav_mask = None
    if traversable_pgm_path and os.path.isfile(traversable_pgm_path):
        t0 = time.time()
        tw, th, trav_pixels = parse_pgm(traversable_pgm_path)
        if tw == w and th == h:
            trav_flip = trav_pixels.reshape(h, w)[::-1, :].ravel().copy()
            if neg:
                trav_flip = 255 - trav_flip
            trav_mask = (trav_flip >= fpt).astype(np.uint8)
            if floor_mask is None:
                floor_mask = trav_mask.copy()
                L(f"[{label}] Floor denominator fallback: using traversability sidecar.", "warn")
            L(f"[{label}] Traversability mask: {time.time()-t0:.3f}s", "info")
        else:
            L(
                f"[{label}] Traversability mask size mismatch: "
                f"{tw}x{th} vs map {w}x{h}. Ignoring sidecar.",
                "warn",
            )

    if trav_mask is not None:
        before_blk = int(np.sum(blocked))
        blocked[trav_mask == 1] = 0
        unblocked = before_blk - int(np.sum(blocked))
        if unblocked > 0:
            L(f"[{label}] Unblocked {unblocked} traversable-floor cells from obstacle map", "info")

    if sel_mask is not None:
        t0 = time.time()
        blocked[sel_mask == 0] = 1
        if floor_mask is not None:
            floor_mask[sel_mask == 0] = 0
        if trav_mask is not None:
            trav_mask[sel_mask == 0] = 0
        L(f"[{label}] Selection mask: {time.time()-t0:.3f}s", "info")

    source_blocked = blocked.copy()
    result = _score_accessibility_from_masks(
        source_blocked.reshape(h, w),
        floor_mask.reshape(h, w) if floor_mask is not None else None,
        res,
        params,
        label,
        L,
        use_stc=use_stc,
        trav_mask2d=trav_mask.reshape(h, w) if trav_mask is not None else None,
        planner=planner,
    )
    result.update(
        sourceBlocked=source_blocked.copy(),
        params=dict(params),
        resolution=float(res),
        origin=(float(ox), float(oy)),
        requestedStartWorld=(float(start_x), float(start_y)),
        effectiveStartWorld=None,
        requestedStartCell=None,
        effectiveStartCell=None,
        startAdjusted=False,
        startAdjustmentReason=None,
        startComponentSize=0,
        largestComponentSize=0,
    )
    return result

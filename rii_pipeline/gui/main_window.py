"""Main application window — MainWin."""

import os
import sys
import math
import time
import shlex
import signal
import shutil
import tempfile
import threading

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QApplication,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QSplitter, QStackedWidget, QTabBar,
    QGroupBox, QFrame, QLabel,
    QPushButton, QLineEdit, QComboBox,
    QDoubleSpinBox, QSpinBox,
    QProgressBar, QTextEdit,
    QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QImage, QColor

from config import (
    DEFAULT_PCD_IN as DEF_IN,
    PCD_PACKAGE_DIR,
    PRECLEAN_DIR as PRECLEAN,
    WORKSPACE,
)

if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)

from pcd_package.pcd_tools import (
    estimate_ground_preserving_preset,
    load_xyz_points,
)

from core.map_io import (
    parse_pgm, parse_yaml,
    resolve_point_cloud_path,
    filtered_point_cloud_filename,
    filtered_point_cloud_stem_candidates,
    rewrite_nav2_yaml_image,
    traversability_sidecar_path,
    floor_sidecar_path,
)
from core.semantic_selection import (
    selection_kind, selection_bounds_px,
    selection_center_px, selection_mask_from_display,
    selection_to_world_bounds,
)
from core.RII_horizontal import (
    BLOCKED_MAP_VIEW, PRIMARY_SELECTION_VIEW,
    PLANNER_NAMES,
    run_coverage,
)
from core.rendering import (
    render_coverage_fast, render_compare_fast,
    render_stc_path_fast, make_info_image,
)
from core.semantic_analysis import (
    SEMANTIC_LABEL_NAMES, SEMANTIC_FIXATION_GROUPS,
    SEMANTIC_3D_COLORS, SEMANTIC_REMOVABLE_FIXATIONS,
    load_semantic_pcd,
    project_labels_to_2d_grid, analyze_semantic_rii,
    compute_semantic_layered_rii,
    simulate_removed_fixations,
    identify_semantic_removal_candidates,
    simulate_removed_candidates,
    render_semantic_candidates,
)
from core.RII_vertical import (
    compute_rii_vertical, compute_combined_rii,
    identify_wall_segments, colorize_cloud_with_walls,
)
from gui.workers import ShellW, ViewW, MapBuildW
from gui.widgets import (
    MapW, DragScrollArea,
    PointCloudPreviewW, PointCloudW,
    PYQTGRAPH_GL_AVAILABLE,
)


class MainWin(QMainWindow):
    ui_log_sig = pyqtSignal(str, str)
    ref_result_sig = pyqtSignal(object, str)
    act_result_sig = pyqtSignal(object, str)
    ref_error_sig = pyqtSignal(str)
    act_error_sig = pyqtSignal(str)
    sem_loaded_sig = pyqtSignal(int, object, object, object)
    sem_load_error_sig = pyqtSignal(int, str)
    sem_result_sig = pyqtSignal(int, object, object, object)
    sem_error_sig = pyqtSignal(int, str)
    sem_improved_sig = pyqtSignal(int, object)
    sem_improved_error_sig = pyqtSignal(int, str)
    sem_progress_sig = pyqtSignal(int, int, str)
    rv_result_sig = pyqtSignal(object)
    rv_error_sig = pyqtSignal(str)
    rv_progress_sig = pyqtSignal(int, str)

    def __init__(s):
        super().__init__()
        s.setWindowTitle("Robot Inclusivity Index (RII)")
        s.setMinimumSize(1300, 800)
        default_in = resolve_point_cloud_path(DEF_IN, ["GlobalMap"])
        s._wk = []
        s._cache_root = None
        s._init_session_cache()
        s.pcd_in = default_in if os.path.isfile(default_in) else ""
        s.ref_r = s.act_r = None; s._imgs = {}; s._clouds = {}; s._map_w = s._map_h = 0
        s._loaded_map_path = None
        s._pgm_pixels = None  # raw PGM pixels for selection mask
        s._sem_pts = None; s._sem_labels = None; s._label_grid = None
        s._sem_analysis = None
        s._sem_candidates = []
        s._sem_improved = None
        s._sem_layered_result = None
        s._sem_focused_candidate_id = None
        s._sem_session_token = 0
        s._sem_load_active = False
        s._sem_analysis_active = False
        s._build(); s._theme()
        s.ui_log_sig.connect(s._log)
        s.ref_result_sig.connect(s._ref_done)
        s.act_result_sig.connect(s._act_done)
        s.ref_error_sig.connect(s._ref_failed)
        s.act_error_sig.connect(s._act_failed)
        s.sem_loaded_sig.connect(s._sem_loaded)
        s.sem_load_error_sig.connect(s._sem_load_failed)
        s.sem_result_sig.connect(s._sem_done)
        s.sem_error_sig.connect(s._sem_failed)
        s.sem_improved_sig.connect(s._sem_improved_done)
        s.sem_improved_error_sig.connect(s._sem_improved_failed)
        s.sem_progress_sig.connect(s._sem_progress)
        s.rv_result_sig.connect(s._rv_done)
        s.rv_error_sig.connect(s._rv_failed)
        s.rv_progress_sig.connect(s._rv_progress)
        s._rv_active = False
        s._rv_result = None
        s._rv_wall_segments = []
        s._rv_focused_wall_id = None
        s._log(s._viewer_backend_startup_message(), "info" if PYQTGRAPH_GL_AVAILABLE else "warn")
        s._log(f"Session cache: {s._cache_root}", "info")
        s._log("Pipeline ready. Steps 1→6.", "info")

    def _viewer_backend_startup_message(s):
        if PYQTGRAPH_GL_AVAILABLE:
            return "Embedded 3D point-cloud viewer ready via pyqtgraph.opengl."
        return "pyqtgraph.opengl is unavailable; using lightweight point-cloud preview."

    def _theme(s):
        s.setStyleSheet("""
            QMainWindow, QWidget { background: #0a0e14; color: #c5cdd8; }
            QLabel { color: #6b7a8d; font-size: 12px; }
            QScrollArea { border: none; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background: #0a0e14; border: 1px solid #2a3545; border-radius: 5px;
                padding: 6px 10px; color: #c5cdd8; font-family: monospace; font-size: 12px; min-height: 26px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: #00e5a0; }
            QGroupBox {
                background: #1a2230; border: 1px solid #2a3545; border-radius: 8px;
                margin-top: 16px; padding: 16px; padding-top: 26px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px;
                color: #5e6a7d; font-size: 12px; font-weight: bold; }
            QProgressBar { background: #2a3545; border: none; border-radius: 2px; height: 4px; }
            QProgressBar::chunk { background: #00e5a0; border-radius: 2px; }
            QTextEdit { background: #111820; border: 1px solid #2a3545; border-radius: 6px;
                color: #6b7a8d; font-family: monospace; font-size: 11px; }
            QSplitter::handle { background: #2a3545; width: 8px; }
            QSplitter::handle:hover { background: #00e5a0; }
        """)

    def _B(s, c):
        return (f"QPushButton {{ background: {c}; color: #0a0e14; border: none; border-radius: 7px; "
                f"padding: 11px; font-weight: bold; font-size: 13px; }}"
                f"QPushButton:hover {{ border: 2px solid white; }}"
                f"QPushButton:disabled {{ background: #333; color: #666; }}")

    def _log(s, m, c=""):
        cl = {"info": "#4a9eff", "success": "#00e5a0", "warn": "#ff6b4a", "gold": "#ffd700"}.get(c, "#6b7a8d")
        s.log_box.append(f'<span style="color:{cl}">[{time.strftime("%H:%M:%S")}] {m}</span>')

    def _init_session_cache(s):
        cache_base = os.path.join(tempfile.gettempdir(), "rii_pipeline_cache")
        os.makedirs(cache_base, exist_ok=True)
        s._cache_root = tempfile.mkdtemp(prefix="session_", dir=cache_base)
        s.pcd_out = os.path.join(s._cache_root, "pcd")
        s.map_dir = os.path.join(s._cache_root, "map")
        os.makedirs(s.pcd_out, exist_ok=True)
        os.makedirs(s.map_dir, exist_ok=True)

    def _browse_dir(s, le, attr):
        d = QFileDialog.getExistingDirectory(s, "Select", getattr(s, attr))
        if d: le.setText(d); setattr(s, attr, d)

    def _browse_point_cloud(s, le, attr):
        start = getattr(s, attr)
        base = start if os.path.isdir(start) else os.path.dirname(start) if start else ""
        f, _ = QFileDialog.getOpenFileName(s, "Select Point Cloud", base, "Point Cloud (*.pcd *.ply)")
        if f:
            le.setText(f)
            setattr(s, attr, f)

    def _set_clean_param(s, name, value):
        widget = s.cp[name]
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        else:
            widget.setValue(float(value))

    def _sync_map_z_from_cleanup(s):
        follow = hasattr(s, "map_z_follow_cleanup") and s.map_z_follow_cleanup.isChecked()
        if hasattr(s, "oz1") and hasattr(s, "oz2"):
            s.oz1.setEnabled(not follow)
            s.oz2.setEnabled(not follow)
            if follow and hasattr(s, "cp"):
                s.oz1.setValue(float(s.cp["min_z"].value()))
                s.oz2.setValue(float(s.cp["max_z"].value()))

    def _flash_widgets(s, widgets, duration_ms=1600):
        active = [w for w in widgets if w is not None]
        if not active:
            return
        for widget in active:
            widget.setStyleSheet("background:#173425;border:1px solid #66d9a3;color:#eafff4;")

        def clear():
            for widget in active:
                widget.setStyleSheet("")

        QTimer.singleShot(duration_ms, clear)

    def _apply_ground_preset(s, log=True):
        preset = {
            "voxel": 0.05,
            "sor_k": 50,
            "sor_std": 0.0,
            "ror_radius": 0.20,
            "ror_min": 2,
        }
        z_preset = {
            "cleanup_min_z": -3.0,
            "cleanup_max_z": 3.0,
            "map_min_z": 0.05,
            "map_max_z": 1.00,
            "floor_anchor_z": 0.0,
        }
        source_desc = "fallback absolute z preset"
        pi = s.e_in.text().strip() if hasattr(s, "e_in") else ""
        if os.path.isfile(pi):
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                points = load_xyz_points(pi)
                z_preset = estimate_ground_preserving_preset(points)
                source_desc = f"floor-anchored from {os.path.basename(pi)}"
            except Exception as exc:
                if log:
                    s._log(f"Preset auto-anchor failed; using fallback z values. {exc}", "warn")
            finally:
                QApplication.restoreOverrideCursor()
        preset["min_z"] = z_preset["cleanup_min_z"]
        preset["max_z"] = z_preset["cleanup_max_z"]
        for key, value in preset.items():
            s._set_clean_param(key, value)
        if hasattr(s, "map_z_follow_cleanup"):
            s.map_z_follow_cleanup.setChecked(False)
        if hasattr(s, "oz1") and hasattr(s, "oz2"):
            s.oz1.setValue(float(z_preset["map_min_z"]))
            s.oz2.setValue(float(z_preset["map_max_z"]))
        s._sync_map_z_from_cleanup()
        changed_widgets = [s.cp.get(key) for key in preset]
        if hasattr(s, "oz1") and hasattr(s, "oz2"):
            changed_widgets.extend([s.oz1, s.oz2])
        s._flash_widgets(changed_widgets)
        if hasattr(s, "preset_status"):
            s.preset_status.setText(
                "Preset applied: "
                f"floor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"cleanup z={z_preset['cleanup_min_z']:.2f}..{z_preset['cleanup_max_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m, "
                "voxel=0.05, sor_k=50, sor_std=0.0, ror_radius=0.20, ror_min=2 "
                f"({source_desc})"
            )
        if log:
            s._log(
                "Applied ground-preserving cleanup preset. "
                f"Floor anchor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"cleanup z={z_preset['cleanup_min_z']:.2f}..{z_preset['cleanup_max_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m.",
                "info",
            )

    def _apply_map_ground_preset(s, log=True):
        pi = s.e_in.text().strip() if hasattr(s, "e_in") else ""
        if not os.path.isfile(pi):
            QMessageBox.warning(s, "Error", f"Raw point cloud not found:\n{pi}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        source_desc = "fallback absolute z preset"
        try:
            points = load_xyz_points(pi)
            z_preset = estimate_ground_preserving_preset(points)
            source_desc = f"floor-anchored from {os.path.basename(pi)}"
        except Exception as exc:
            QMessageBox.warning(s, "Error", f"Failed to estimate map z preset:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        s.oz1.setValue(float(z_preset["map_min_z"]))
        s.oz2.setValue(float(z_preset["map_max_z"]))
        s._flash_widgets([s.oz1, s.oz2])
        if hasattr(s, "map_preset_status"):
            s.map_preset_status.setText(
                "Preset applied: "
                f"floor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m "
                f"({source_desc})"
            )
        if log:
            s._log(
                "Applied raw-cloud map z preset. "
                f"Floor anchor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m.",
                "info",
            )

    def _browse_pgm(s):
        f, _ = QFileDialog.getOpenFileName(s, "PGM", "", "PGM (*.pgm)")
        if f:
            s.e_pgm.setText(f); s._load_map(f)
            # Auto-fill yaml if matching file exists
            y = f.replace('.pgm', '.yaml')
            if os.path.isfile(y) and not s.e_yaml.text():
                s.e_yaml.setText(y)

    def _browse_yaml(s):
        f, _ = QFileDialog.getOpenFileName(s, "YAML", "", "YAML (*.yaml *.yml)")
        if f:
            s.e_yaml.setText(f)
            s._update_actual_start_bounds(f)

    def _save_filtered_pcd_as(s):
        src = resolve_point_cloud_path(
            s.e_out.text(),
            filtered_point_cloud_stem_candidates(s.e_in.text().strip()),
        )
        if not os.path.isfile(src):
            QMessageBox.warning(s, "Error", f"Run Step 2 first\n{src}")
            return
        dst, _ = QFileDialog.getSaveFileName(
            s,
            "Save Filtered Point Cloud",
            os.path.join(WORKSPACE, filtered_point_cloud_filename(s.e_in.text().strip())),
            "Point Cloud (*.pcd)",
        )
        if not dst:
            return
        if not dst.lower().endswith(".pcd"):
            dst += ".pcd"
        shutil.copy2(src, dst)
        s._log(f"Saved filtered PCD: {dst}", "success")

    def _save_map_bundle(s):
        src_dir = s.e_save.text().strip()
        pgm = os.path.join(src_dir, "map.pgm")
        yml = os.path.join(src_dir, "map.yaml")
        if not (os.path.isfile(pgm) and os.path.isfile(yml)):
            QMessageBox.warning(s, "Error", f"Run Step 2 first\n{pgm}\n{yml}")
            return
        dst_yaml, _ = QFileDialog.getSaveFileName(
            s,
            "Save Map Bundle As",
            os.path.join(WORKSPACE, "map.yaml"),
            "Nav2 Map YAML (*.yaml)",
        )
        if not dst_yaml:
            return
        if not dst_yaml.lower().endswith(".yaml"):
            dst_yaml += ".yaml"
        dst_pgm = os.path.splitext(dst_yaml)[0] + ".pgm"
        shutil.copy2(pgm, dst_pgm)
        with open(yml, "r", encoding="utf-8") as handle:
            yaml_text = handle.read()
        yaml_text = rewrite_nav2_yaml_image(yaml_text, os.path.basename(dst_pgm))
        with open(dst_yaml, "w", encoding="utf-8") as handle:
            handle.write(yaml_text)
        extras = [
            (
                floor_sidecar_path(pgm),
                floor_sidecar_path(dst_pgm),
                os.path.splitext(yml)[0] + "_floor.yaml",
                os.path.splitext(dst_yaml)[0] + "_floor.yaml",
            ),
            (
                traversability_sidecar_path(pgm),
                traversability_sidecar_path(dst_pgm),
                os.path.splitext(yml)[0] + "_traversable.yaml",
                os.path.splitext(dst_yaml)[0] + "_traversable.yaml",
            ),
        ]
        for src_extra_pgm, dst_extra_pgm, src_extra_yaml, dst_extra_yaml in extras:
            if not os.path.isfile(src_extra_pgm):
                continue
            shutil.copy2(src_extra_pgm, dst_extra_pgm)
            if os.path.isfile(src_extra_yaml):
                with open(src_extra_yaml, "r", encoding="utf-8") as handle:
                    extra_yaml_text = handle.read()
                extra_yaml_text = rewrite_nav2_yaml_image(extra_yaml_text, os.path.basename(dst_extra_pgm))
                with open(dst_extra_yaml, "w", encoding="utf-8") as handle:
                    handle.write(extra_yaml_text)
            s._log(f"Saved sidecar map: {dst_extra_pgm}", "success")
            if os.path.isfile(dst_extra_yaml):
                s._log(f"Saved sidecar config: {dst_extra_yaml}", "success")
        s._log(f"Saved map: {dst_pgm}", "success")
        s._log(f"Saved map config: {dst_yaml}", "success")

    def _selected_map_source_path(s):
        path = s.e_in.text().strip()
        label = "raw input point cloud"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Raw point cloud not found:\n{path}")
        return path, label

    def _map_world_bounds(s, yaml_path=None):
        if s._map_w <= 0 or s._map_h <= 0:
            return None
        if yaml_path is None:
            _, yaml_path = s._get_pgm()
        if not yaml_path or not os.path.isfile(yaml_path):
            return None
        yd = parse_yaml(yaml_path)
        res = yd["resolution"]
        ox, oy = yd["origin"][0], yd["origin"][1]
        return (ox, ox + s._map_w * res, oy, oy + s._map_h * res)

    def _selection_world_bounds(s, yaml_path=None):
        if not s.mw.sel or s._map_w <= 0 or s._map_h <= 0:
            return None
        if yaml_path is None:
            _, yaml_path = s._get_pgm()
        if not yaml_path or not os.path.isfile(yaml_path):
            return None
        yd = parse_yaml(yaml_path)
        return selection_to_world_bounds(s.mw.sel, s._map_w, s._map_h, yd)

    def _selection_center_world(s, yaml_path=None):
        if not s.mw.sel or s._map_w <= 0 or s._map_h <= 0:
            return None
        if yaml_path is None:
            _, yaml_path = s._get_pgm()
        if not yaml_path or not os.path.isfile(yaml_path):
            return None
        yd = parse_yaml(yaml_path)
        center = selection_center_px(s.mw.sel)
        if center is None:
            return None
        res = yd["resolution"]
        ox, oy = yd["origin"][0], yd["origin"][1]
        cx, cy = center
        return (ox + cx * res, oy + (s._map_h - 1 - cy) * res)

    def _update_actual_start_bounds(s, yaml_path=None):
        if not hasattr(s, "sx_") or not hasattr(s, "sy_"):
            return
        bounds = s._map_world_bounds(yaml_path)
        if bounds is None:
            return
        min_x, max_x, min_y, max_y = bounds
        s.sx_.setRange(min_x, max_x)
        s.sy_.setRange(min_y, max_y)
        s.sx_.setValue(min(max(s.sx_.value(), min_x), max_x))
        s.sy_.setValue(min(max(s.sy_.value(), min_y), max_y))

    def _set_actual_start_from_selection(s):
        if not hasattr(s, "sx_") or not hasattr(s, "sy_"):
            return
        _, yaml_path = s._get_pgm()
        center = s._selection_center_world(yaml_path)
        if center is None:
            bounds = s._map_world_bounds(yaml_path)
            if bounds is None:
                return
            min_x, max_x, min_y, max_y = bounds
            center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
            s._log("No selection active. Actual start set to map center.", "info")
        else:
            s._log(f"Actual start set to selection center: ({center[0]:.2f}, {center[1]:.2f})", "info")
        s.sx_.setValue(center[0])
        s.sy_.setValue(center[1])

    def _coverage_start_note(s, result):
        if not result or not result.get("startAdjusted"):
            return ""
        eff = result.get("effectiveStartWorld")
        if not eff:
            return ""
        reason = result.get("startAdjustmentReason")
        if reason == "tiny_island":
            return (
                "Start was on an isolated reachable pocket. "
                f"Used main connected region at ({eff[0]:.2f}, {eff[1]:.2f})."
            )
        if reason == "blocked":
            return (
                "Start was blocked after inflation. "
                f"Used nearest reachable cell at ({eff[0]:.2f}, {eff[1]:.2f})."
            )
        return f"Used reachable start at ({eff[0]:.2f}, {eff[1]:.2f})."

    def _result_area(s, result):
        if not result:
            return 0.0
        return float(result.get("accessibleArea", result.get("reachableArea", result.get("coveredArea", 0.0))))

    def _result_floor_area(s, result):
        if not result:
            return 0.0
        return float(result.get("totalFloorArea", 0.0))

    def _results_share_map(s):
        if not s.ref_r or not s.act_r:
            return False
        if int(s.ref_r.get("w", -1)) != int(s.act_r.get("w", -1)):
            return False
        if int(s.ref_r.get("h", -1)) != int(s.act_r.get("h", -1)):
            return False
        ref_map = s.ref_r.get("pgm_path")
        act_map = s.act_r.get("pgm_path")
        if ref_map and act_map:
            return os.path.abspath(ref_map) == os.path.abspath(act_map)
        return True

    def _next_semantic_session_token(s):
        s._sem_session_token += 1
        return s._sem_session_token

    def _clear_semantic_progress(s):
        if hasattr(s, "sem_prog_lbl"):
            s.sem_prog_lbl.clear()
            s.sem_prog_lbl.hide()
        if hasattr(s, "sem_prog"):
            s.sem_prog.setValue(0)
            s.sem_prog.hide()

    def _set_semantic_candidate_placeholder(s, message):
        if not hasattr(s, "sem_candidate_list"):
            return
        s.sem_candidate_list.blockSignals(True)
        s.sem_candidate_list.clear()
        placeholder = QListWidgetItem(message)
        placeholder.setFlags(Qt.NoItemFlags)
        s.sem_candidate_list.addItem(placeholder)
        s.sem_candidate_list.blockSignals(False)

    def _update_semantic_ready_state(s):
        ready = (
            not s._sem_load_active
            and not s._sem_analysis_active
            and s.ref_r is not None
            and s.act_r is not None
            and s._sem_pts is not None
            and s._sem_labels is not None
        )
        if hasattr(s, "bsem"):
            s.bsem.setEnabled(ready)
        candidate_ready = bool(s._sem_candidates) and not s._sem_load_active and not s._sem_analysis_active
        if hasattr(s, "sem_candidate_list"):
            s._set_semantic_candidate_controls_enabled(candidate_ready)

    def _invalidate_semantic_state(
        s,
        *,
        keep_loaded_cloud=True,
        candidate_message="Run semantic analysis to populate removable-object candidates.",
        status_message=None,
        status_color="#6b7a8d",
        clear_progress=True,
    ):
        token = s._next_semantic_session_token()
        s._sem_load_active = False
        s._sem_analysis_active = False
        if not keep_loaded_cloud:
            s._sem_pts = None
            s._sem_labels = None
        s._label_grid = None
        s._sem_analysis = None
        s._sem_candidates = []
        s._sem_improved = None
        s._sem_layered_result = None
        s._sem_focused_candidate_id = None
        if hasattr(s, "sem_filter"):
            s.sem_filter.blockSignals(True)
            s.sem_filter.setCurrentIndex(0)
            s.sem_filter.blockSignals(False)
        if hasattr(s, "sem_layered_status"):
            s.sem_layered_status.clear()
            s.sem_layered_status.hide()
        if hasattr(s, "sem_candidate_status"):
            s.sem_candidate_status.setText(
                "Run semantic analysis to populate removable-object candidates, filter them by fixation, and recompute an Optimised RII score."
            )
            s.sem_candidate_status.setStyleSheet("color:#ffcc66;font-size:11px")
            s.sem_candidate_status.setVisible(True)
        s._set_semantic_candidate_placeholder(candidate_message)
        s._hide_semantic_whatif_card()
        s.mw.clear_focus()
        if hasattr(s, 'bsem_3d'):
            s.bsem_3d.setEnabled(keep_loaded_cloud and s._sem_pts is not None and s._sem_labels is not None)
        s._imgs["Semantic"] = make_info_image("Run semantic analysis to view object candidates and Optimised RII changes.")
        if s._is_view_active("Semantic"):
            s._switch_view("Semantic")
        if status_message is not None and hasattr(s, "sem_status"):
            s.sem_status.setText(status_message)
            s.sem_status.setStyleSheet(f"color:{status_color};font-size:11px")
        if clear_progress:
            s._clear_semantic_progress()
        s._update_semantic_ready_state()
        return token

    def _clear_step5_results(s, reason=None):
        s.ref_r = None
        s.act_r = None
        if hasattr(s, "lref"):
            s.lref.setText("")
        if hasattr(s, "lact"):
            s.lact.setText("")
        if hasattr(s, "lref_note"):
            s.lref_note.setText("")
        if hasattr(s, "lact_note"):
            s.lact_note.setText("")
        if hasattr(s, "riif"):
            s.riif.hide()
        if hasattr(s, "sem_riif"):
            s._hide_semantic_whatif_card()
        s._imgs["Reference Coverage"] = make_info_image("Run Step 3 to view reference coverage.")
        s._imgs["Actual Coverage"] = make_info_image("Run Step 3 to view actual coverage.")
        s._imgs["Compare"] = make_info_image("Run both Reference and Actual on the same map to compare coverage.")
        s._invalidate_semantic_state(
            keep_loaded_cloud=True,
            candidate_message="Run Step 3 and then semantic analysis to repopulate removable-object candidates for the current map.",
            clear_progress=True,
        )
        s._update_stc_path_view()
        if reason:
            s._log(reason, "info")

    def _use_stc_mode(s):
        """Return selected planner name, or None if 'Without Path Planner'."""
        if not hasattr(s, "rii_mode") or s.rii_mode.currentIndex() == 0:
            return None
        return s.planner_combo.currentText()

    def _toggle_planner_combo(s):
        s.planner_row.setVisible(s.rii_mode.currentIndex() == 1)

    def _update_stc_path_view(s):
        ref = s.ref_r if s.ref_r and s.ref_r.get("useSTC") else None
        act = s.act_r if s.act_r and s.act_r.get("useSTC") else None
        planner_label = ""
        for r in (act, ref):
            if r and r.get("planner"):
                planner_label = r["planner"]
                break
        if ref is None and act is None:
            s._imgs["Planner Path"] = make_info_image("Run Step 3 with a path planner to view the coverage path.")
        else:
            s._imgs["Planner Path"] = render_stc_path_fast(ref, act, bg_pgm=getattr(s, '_pgm_pixels', None), planner_label=planner_label)
        if s._is_view_active("Planner Path"):
            s._switch_view("Planner Path")

    def _toggle_shape(s, prefix):
        if prefix == 'r':
            c = s.rs.currentIndex() == 0; s.rc.setVisible(c); s.rrc.setVisible(not c)
        else:
            c = s.as_.currentIndex() == 0; s.ac.setVisible(c); s.arc.setVisible(not c)

    def _get_params(s, prefix):
        """Return the robot footprint parameters used for horizontal RII."""
        if prefix == 'r':
            if s.rs.currentIndex() == 0:
                r = s.rr.value()
                return dict(shape='circular', radius=r, halfW=r, halfL=r)
            else:
                w = s.rw.value(); l = s.rl.value()
                return dict(shape='rectangular', radius=math.hypot(w/2, l/2), halfW=w/2, halfL=l/2)
        else:
            if s.as_.currentIndex() == 0:
                r = s.ar.value()
                return dict(shape='circular', radius=r, halfW=r, halfL=r)
            else:
                w = s.aw.value(); l = s.al.value()
                return dict(shape='rectangular', radius=math.hypot(w/2, l/2), halfW=w/2, halfL=l/2)

    def _get_pgm(s):
        p = s.e_pgm.text()
        if not p: p = os.path.join(s.e_save.text(), "map.pgm")
        y = s.e_yaml.text()
        if not y: y = p.replace('.pgm', '.yaml')
        if not os.path.isfile(p):
            QMessageBox.warning(s, "Error", f"PGM not found:\n{p}"); return None, None
        if not os.path.isfile(y):
            QMessageBox.warning(s, "Error", f"YAML not found:\n{y}\n\nPlease browse for the .yaml file in Step 3."); return None, None
        return p, y

    def _switch_view(s, nm):
        # Select the matching tab without re-triggering the signal
        if hasattr(s, 'view_tab_bar'):
            for i in range(s.view_tab_bar.count()):
                if s.view_tab_bar.tabText(i) == nm:
                    s.view_tab_bar.blockSignals(True)
                    s.view_tab_bar.setCurrentIndex(i)
                    s.view_tab_bar.blockSignals(False)
                    break
        if nm in ("3D Viewer", "Clean Cloud", "Vertical Coverage"):
            s.mw.clear_focus()
            s.view_stack.setCurrentWidget(s.pcw)
            if nm in s._clouds:
                s.pcw.set_cloud(s._clouds[nm])
            else:
                s.pcw.clear_cloud(f"No {nm.lower()} loaded")
            return
        s.view_stack.setCurrentWidget(s.mw)
        if nm != "Semantic":
            s.mw.clear_focus()
        if nm in s._imgs:
            s.mw.set_qi(s._imgs[nm])
        else:
            s.mw.set_qi(make_info_image(f"No {nm} image"))

    def _is_view_active(s, nm):
        """Check if a given view tab is currently selected."""
        if hasattr(s, 'view_tab_bar'):
            return s.view_tab_bar.tabText(s.view_tab_bar.currentIndex()) == nm
        return False

    def _active_view_name(s):
        """Return the name of the currently selected view tab."""
        if hasattr(s, 'view_tab_bar'):
            return s.view_tab_bar.tabText(s.view_tab_bar.currentIndex())
        return PRIMARY_SELECTION_VIEW

    # ── Split View ──
    def _toggle_split_view(s):
        """Show or hide the secondary split viewer panel."""
        show = s.btn_split_view.isChecked()
        s._split_panel.setVisible(show)
        if show:
            s._switch_split_view(s._split_tab_bar.tabText(s._split_tab_bar.currentIndex()))
            s._split_splitter.setSizes([1, 1])  # equal widths

    def _switch_split_view(s, nm):
        """Switch the secondary split panel to the named view."""
        for i in range(s._split_tab_bar.count()):
            if s._split_tab_bar.tabText(i) == nm:
                s._split_tab_bar.blockSignals(True)
                s._split_tab_bar.setCurrentIndex(i)
                s._split_tab_bar.blockSignals(False)
                break
        if nm in ("3D Viewer", "Clean Cloud", "Vertical Coverage"):
            s._split_view_stack.setCurrentWidget(s._split_pcw)
            if nm in s._clouds:
                s._split_pcw.set_cloud(s._clouds[nm])
            else:
                s._split_pcw.clear_cloud(f"No {nm.lower()} loaded")
            return
        s._split_view_stack.setCurrentWidget(s._split_mw)
        if nm in s._imgs:
            s._split_mw.set_qi(s._imgs[nm])
        else:
            s._split_mw.set_qi(make_info_image(f"No {nm} image"))

    def _set_img(s, nm, qi):
        s._imgs[nm] = qi; s._switch_view(nm)

    def _set_cloud(s, nm, cloud):
        s._clouds[nm] = cloud
        s._switch_view(nm)

    def _fallback_point_cloud_viewer(s, reason):
        if isinstance(s.pcw, PointCloudPreviewW):
            return
        current = s._active_view_name()
        old = s.pcw
        idx = s.view_stack.indexOf(old)
        preview = PointCloudPreviewW()
        if idx >= 0:
            s.view_stack.removeWidget(old)
            old.hide()
            old.setParent(None)
            s.view_stack.insertWidget(idx, preview)
        s.pcw = preview
        s._log(f"Embedded 3D viewer backend failed; using software preview instead. {reason}", "warn")
        if current in ("3D Viewer", "Clean Cloud"):
            s._switch_view(current)

    def _load_map(s, pgm):
        previous_size = (s._map_w, s._map_h)
        w, h, pixels = parse_pgm(pgm)
        previous_path = s._loaded_map_path
        s._map_w, s._map_h = w, h
        s._pgm_pixels = pixels
        qi = QImage(pixels.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
        s._imgs[BLOCKED_MAP_VIEW] = qi
        s._load_map_sidecars(pgm)
        s._switch_view(PRIMARY_SELECTION_VIEW)
        if previous_path and (
            os.path.abspath(previous_path) != os.path.abspath(pgm) or previous_size != (w, h)
        ):
            s._clear_step5_results("Loaded a different map. Rerun Step 3 on the current map.")
        s._loaded_map_path = pgm
        yaml_guess = pgm.replace(".pgm", ".yaml")
        if os.path.isfile(yaml_guess):
            s._update_actual_start_bounds(yaml_guess)

    def _load_map_sidecars(s, pgm):
        sidecars = [
            ("Floor Mask", floor_sidecar_path(pgm), "No floor sidecar for this map bundle."),
            ("Traversable Ground", traversability_sidecar_path(pgm), "No traversability sidecar for this map bundle."),
        ]
        for name, path, placeholder in sidecars:
            if os.path.isfile(path):
                try:
                    w, h, pixels = parse_pgm(path)
                    qi = QImage(pixels.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
                    s._imgs[name] = qi
                except Exception as exc:
                    s._imgs[name] = make_info_image(f"Failed to load {os.path.basename(path)}.\n{exc}")
            else:
                s._imgs[name] = make_info_image(placeholder)

    # ── Build selection mask matching HTML lines 342-344 ──
    def _make_sel_mask(s):
        """Create a selection mask in OccGrid coords from rectangle or freeform input."""
        if not s.mw.sel or s._map_w == 0:
            return None
        return selection_mask_from_display(s.mw.sel, s._map_w, s._map_h)

    # ══════════════════════════════════════════════════════════════════════
    # _build — construct all UI widgets
    # ══════════════════════════════════════════════════════════════════════
    def _build(s):
        cw = QWidget(); s.setCentralWidget(cw)
        ml = QHBoxLayout(cw); ml.setContentsMargins(0, 0, 0, 0)
        sp = QSplitter(Qt.Horizontal); sp.setChildrenCollapsible(False); sp.setHandleWidth(8); ml.addWidget(sp)

        ls = DragScrollArea()
        ls.setMinimumWidth(360)
        lw = QWidget(); ll = QVBoxLayout(lw)
        ll.setSpacing(8); ll.setContentsMargins(12, 12, 12, 12)

        def mkbr_dir(le, attr):
            b = QPushButton("📂"); b.setFixedWidth(34)
            b.clicked.connect(lambda: s._browse_dir(le, attr)); return b

        def mkbr_cloud(le, attr):
            b = QPushButton("📂"); b.setFixedWidth(34)
            b.clicked.connect(lambda: s._browse_point_cloud(le, attr)); return b

        # Step 1
        g1 = QGroupBox("Step 1 — View Raw Point Cloud"); l1 = QVBoxLayout()
        h1 = QHBoxLayout(); s.e_in = QLineEdit(s.pcd_in)
        s.e_in.setPlaceholderText("Pick a raw .pcd or .ply file")
        h1.addWidget(QLabel("Input File:")); h1.addWidget(s.e_in, 1); h1.addWidget(mkbr_cloud(s.e_in, 'pcd_in'))
        l1.addLayout(h1)
        s.b1 = QPushButton("👁  View Raw Point Cloud"); s.b1.setStyleSheet(s._B("#4a9eff"))
        s.b1.clicked.connect(s._step1); l1.addWidget(s.b1)
        g1.setLayout(l1); ll.addWidget(g1)

        # Step 2
        g4 = QGroupBox("Step 2 — Generate 2D Map"); l4 = QVBoxLayout()
        h4 = QHBoxLayout(); s.e_save = QLineEdit(s.map_dir)
        s.e_save.setReadOnly(True)
        h4.addWidget(QLabel("Cache Dir:")); h4.addWidget(s.e_save, 1)
        l4.addLayout(h4)
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Map Source:"))
        src_label = QLabel("Raw input point cloud")
        src_label.setStyleSheet("color:#c5cdd8;font-weight:bold")
        src_row.addWidget(src_label, 1)
        l4.addLayout(src_row)
        h4p = QHBoxLayout()
        s.b4preset = QPushButton("Apply Ground-Based Z Preset")
        s.b4preset.setStyleSheet(s._B("#66d9a3"))
        s.b4preset.clicked.connect(lambda: s._apply_map_ground_preset())
        h4p.addWidget(s.b4preset)
        h4p.addStretch()
        l4.addLayout(h4p)
        s.map_preset_status = QLabel("")
        s.map_preset_status.setWordWrap(True)
        s.map_preset_status.setStyleSheet("color:#66d9a3;font-size:11px")
        l4.addWidget(s.map_preset_status)
        s.map_source_hint = QLabel(
            "Adjust pt_min_z and pt_max_z below, then click 'Generate 2D Map' until the traversable ground map "
            "closely matches the Reference Coverage map (Step 3). You can also edit the traversable map manually "
            "using the Draw/Erase tools below."
        )
        s.map_source_hint.setWordWrap(True)
        l4.addWidget(s.map_source_hint)
        zh = QHBoxLayout()
        zh.addWidget(QLabel("pt_min_z (m):"))
        s.oz1 = QDoubleSpinBox(); s.oz1.setRange(-20, 20); s.oz1.setValue(0.05); s.oz1.setDecimals(2)
        zh.addWidget(s.oz1)
        zh.addWidget(QLabel("pt_max_z (m):"))
        s.oz2 = QDoubleSpinBox(); s.oz2.setRange(-20, 20); s.oz2.setValue(1.00); s.oz2.setDecimals(2)
        zh.addWidget(s.oz2); l4.addLayout(zh)
        th = QHBoxLayout()
        th.addWidget(QLabel("max_slope (deg):"))
        s.t_slope = QDoubleSpinBox(); s.t_slope.setRange(1, 89); s.t_slope.setValue(35.0); s.t_slope.setDecimals(1)
        th.addWidget(s.t_slope)
        th.addWidget(QLabel("max_step (m):"))
        s.t_step = QDoubleSpinBox(); s.t_step.setRange(0.01, 9999.0); s.t_step.setValue(0.25); s.t_step.setDecimals(2)
        th.addWidget(s.t_step)
        th.addWidget(QLabel("max_rough (m):"))
        s.t_rough = QDoubleSpinBox(); s.t_rough.setRange(0.01, 9999.0); s.t_rough.setValue(0.15); s.t_rough.setDecimals(2)
        th.addWidget(s.t_rough)
        l4.addLayout(th)
        l4.addWidget(QLabel("Terrain thresholds for traversability sidecar. Increase values to mark more ground as traversable."))
        s.b4 = QPushButton("🗺️  Generate 2D Map"); s.b4.setStyleSheet(s._B("#aa66ff"))
        s.b4.clicked.connect(s._step4); l4.addWidget(s.b4)
        s.b4save = QPushButton("💾  Save Map (.pgm + .yaml) As..."); s.b4save.setStyleSheet(s._B("#dca7ff"))
        s.b4save.clicked.connect(s._save_map_bundle); l4.addWidget(s.b4save)
        l4.addWidget(QLabel("Positive pt_min_z / pt_max_z values are treated as offsets above the detected floor for shifted clouds like Area_3."))
        l4.addWidget(QLabel("For road-accessibility analysis, keep pt_min_z above the ground plane so floor hits do not become occupied cells."))
        l4.addWidget(QLabel("Outputs are cached in a temporary session folder unless you explicitly save them."))
        g4.setLayout(l4); ll.addWidget(g4)

        # ── Edit Traversable Ground ──
        g_edit = QGroupBox("Edit Traversable Ground"); l_edit = QVBoxLayout(); l_edit.setSpacing(6)
        l_edit.addWidget(QLabel("Draw or erase on the traversable ground map. Apply to save changes for RII computation."))
        edit_row1 = QHBoxLayout()
        s.btn_edit_draw = QPushButton("Draw"); s.btn_edit_draw.setCheckable(True)
        s.btn_edit_draw.setStyleSheet("QPushButton{background:#1a2230;color:#6b7a8d;border:1px solid #2a3545;border-radius:4px;padding:4px 10px}"
                                       "QPushButton:checked{background:#00e5a040;color:#00e5a0;border-color:#00e5a0}")
        s.btn_edit_erase = QPushButton("Erase"); s.btn_edit_erase.setCheckable(True)
        s.btn_edit_erase.setStyleSheet(s.btn_edit_draw.styleSheet())
        s.btn_edit_draw.clicked.connect(lambda: s._toggle_edit_mode("draw"))
        s.btn_edit_erase.clicked.connect(lambda: s._toggle_edit_mode("erase"))
        edit_row1.addWidget(s.btn_edit_draw); edit_row1.addWidget(s.btn_edit_erase)
        l_edit.addLayout(edit_row1)
        edit_row2 = QHBoxLayout()
        edit_row2.addWidget(QLabel("Brush:"))
        s.edit_brush_shape = QComboBox(); s.edit_brush_shape.addItems(["Circle", "Rectangle", "Free Draw"])
        s.edit_brush_shape.currentTextChanged.connect(s._on_brush_shape_changed)
        edit_row2.addWidget(s.edit_brush_shape)
        # Circle / Free Draw size
        s._brush_size_label = QLabel("Radius (m):")
        edit_row2.addWidget(s._brush_size_label)
        s.edit_brush_size = QDoubleSpinBox(); s.edit_brush_size.setRange(0.01, 9999.0); s.edit_brush_size.setValue(0.50); s.edit_brush_size.setDecimals(2); s.edit_brush_size.setSingleStep(0.1)
        s.edit_brush_size.valueChanged.connect(s._update_brush_size_px)
        edit_row2.addWidget(s.edit_brush_size)
        # Rectangle W/H
        s._brush_w_label = QLabel("W (m):")
        edit_row2.addWidget(s._brush_w_label)
        s.edit_brush_w = QDoubleSpinBox(); s.edit_brush_w.setRange(0.01, 9999.0); s.edit_brush_w.setValue(0.50); s.edit_brush_w.setDecimals(2); s.edit_brush_w.setSingleStep(0.1)
        s.edit_brush_w.valueChanged.connect(s._update_brush_rect_px)
        edit_row2.addWidget(s.edit_brush_w)
        s._brush_h_label = QLabel("H (m):")
        edit_row2.addWidget(s._brush_h_label)
        s.edit_brush_h = QDoubleSpinBox(); s.edit_brush_h.setRange(0.01, 9999.0); s.edit_brush_h.setValue(0.50); s.edit_brush_h.setDecimals(2); s.edit_brush_h.setSingleStep(0.1)
        s.edit_brush_h.valueChanged.connect(s._update_brush_rect_px)
        edit_row2.addWidget(s.edit_brush_h)
        # Initially show circle controls, hide rectangle controls
        s._brush_w_label.hide(); s.edit_brush_w.hide()
        s._brush_h_label.hide(); s.edit_brush_h.hide()
        l_edit.addLayout(edit_row2)
        s.edit_ref_overlay = QCheckBox("Show Obstacle Map Reference")
        s.edit_ref_overlay.setStyleSheet("QCheckBox{color:#6b7a8d;font-size:11px}")
        s.edit_ref_overlay.setToolTip("Overlay the obstacle map semi-transparently so you can match the traversability map to it")
        s.edit_ref_overlay.toggled.connect(lambda v: s.mw.set_reference_overlay_visible(v))
        l_edit.addWidget(s.edit_ref_overlay)

        edit_row3 = QHBoxLayout()
        s.btn_edit_apply = QPushButton("Apply to Traversability Map")
        s.btn_edit_apply.setStyleSheet(s._B("#00e5a0"))
        s.btn_edit_apply.clicked.connect(s._apply_trav_edit)
        s.btn_edit_revert = QPushButton("Revert")
        s.btn_edit_revert.setStyleSheet(s._B("#ff6b4a"))
        s.btn_edit_revert.clicked.connect(s._revert_trav_edit)
        edit_row3.addWidget(s.btn_edit_apply); edit_row3.addWidget(s.btn_edit_revert)
        l_edit.addLayout(edit_row3)
        g_edit.setLayout(l_edit); ll.addWidget(g_edit)

        # Step 3
        g5 = QGroupBox("Step 3 — RII Horizontal"); l5 = QVBoxLayout(); l5.setSpacing(8)
        hm = QHBoxLayout()
        s.e_pgm = QLineEdit(""); s.e_pgm.setPlaceholderText("Auto from Step 2 or browse")
        bm = QPushButton("📂"); bm.setFixedWidth(34); bm.clicked.connect(s._browse_pgm)
        hm.addWidget(QLabel(".pgm:")); hm.addWidget(s.e_pgm, 1); hm.addWidget(bm); l5.addLayout(hm)
        hy = QHBoxLayout()
        s.e_yaml = QLineEdit(""); s.e_yaml.setPlaceholderText("Auto from .pgm path or browse")
        by = QPushButton("📂"); by.setFixedWidth(34); by.clicked.connect(s._browse_yaml)
        hy.addWidget(QLabel(".yaml:")); hy.addWidget(s.e_yaml, 1); hy.addWidget(by); l5.addLayout(hy)

        sh = QHBoxLayout()
        sh.addWidget(QLabel("Selection:"))
        s.sel_mode = QComboBox()
        s.sel_mode.addItems(["Rectangle", "Spline / Freeform"])
        sh.addWidget(s.sel_mode, 1)
        l5.addLayout(sh)
        s.bsel = QPushButton("✎  Select Area on Floor Mask (optional)")
        s.bsel.setStyleSheet(s._B("#ffd700")); s.bsel.clicked.connect(s._enable_sel); l5.addWidget(s.bsel)
        s.slbl = QLabel(""); s.slbl.setStyleSheet("color:#ffd700;font-size:11px"); l5.addWidget(s.slbl)
        mh = QHBoxLayout()
        mh.addWidget(QLabel("Mode:"))
        s.rii_mode = QComboBox()
        s.rii_mode.addItems(["Without Path Planner", "With Path Planner"])
        s.rii_mode.currentIndexChanged.connect(s._toggle_planner_combo)
        mh.addWidget(s.rii_mode, 1)
        l5.addLayout(mh)
        ph = QHBoxLayout()
        ph.addWidget(QLabel("Planner:"))
        s.planner_combo = QComboBox()
        s.planner_combo.addItems(PLANNER_NAMES)
        ph.addWidget(s.planner_combo, 1)
        s.planner_row = QWidget()
        s.planner_row.setLayout(ph)
        s.planner_row.hide()
        l5.addWidget(s.planner_row)

        # Reference robot
        l5.addWidget(QLabel("─── Reference Robot (comparison only) ───"))
        l5.addWidget(QLabel("Optional benchmark footprint for comparison only. It does not set the RII Horizontal denominator."))
        rsh = QHBoxLayout(); rsh.addWidget(QLabel("Shape:"))
        s.rs = QComboBox(); s.rs.addItems(["circular", "rectangular"])
        s.rs.currentIndexChanged.connect(lambda: s._toggle_shape('r')); rsh.addWidget(s.rs, 1); l5.addLayout(rsh)
        s.rc = QWidget(); rc_ = QHBoxLayout(s.rc); rc_.setContentsMargins(0, 0, 0, 0)
        rc_.addWidget(QLabel("Radius (m):"))
        s.rr = QDoubleSpinBox(); s.rr.setRange(.001, 5); s.rr.setValue(.035); s.rr.setDecimals(3)
        rc_.addWidget(s.rr, 1); l5.addWidget(s.rc)
        s.rrc = QWidget(); rrc_ = QHBoxLayout(s.rrc); rrc_.setContentsMargins(0, 0, 0, 0)
        rrc_.addWidget(QLabel("W (m):")); s.rw = QDoubleSpinBox(); s.rw.setRange(.01, 5); s.rw.setValue(.07); s.rw.setDecimals(3); rrc_.addWidget(s.rw, 1)
        rrc_.addWidget(QLabel("L (m):")); s.rl = QDoubleSpinBox(); s.rl.setRange(.01, 5); s.rl.setValue(.07); s.rl.setDecimals(3); rrc_.addWidget(s.rl, 1)
        s.rrc.hide(); l5.addWidget(s.rrc)
        s.bref = QPushButton("▶  Run Reference"); s.bref.setStyleSheet(s._B("#00c0ff"))
        s.bref.clicked.connect(s._run_ref); l5.addWidget(s.bref)
        s.lref = QLabel(""); s.lref.setStyleSheet("color:#00c0ff;font-size:13px;font-weight:bold"); l5.addWidget(s.lref)
        s.lref_note = QLabel("")
        s.lref_note.setWordWrap(True)
        s.lref_note.setStyleSheet("color:#8fdcff;font-size:11px")
        l5.addWidget(s.lref_note)

        # Actual robot
        l5.addWidget(QLabel("─── Actual Robot (your real platform) ───"))
        l5.addWidget(QLabel("RII Horizontal = inflated accessible area / total floor area. 'With Path Planner' keeps only the largest connected inflated-accessible region."))
        ash = QHBoxLayout(); ash.addWidget(QLabel("Shape:"))
        s.as_ = QComboBox(); s.as_.addItems(["circular", "rectangular"]); s.as_.setCurrentIndex(1)
        s.as_.currentIndexChanged.connect(lambda: s._toggle_shape('a')); ash.addWidget(s.as_, 1); l5.addLayout(ash)
        s.ac = QWidget(); ac_ = QHBoxLayout(s.ac); ac_.setContentsMargins(0, 0, 0, 0)
        ac_.addWidget(QLabel("Radius (m):"))
        s.ar = QDoubleSpinBox(); s.ar.setRange(.01, 5); s.ar.setValue(.35); s.ar.setDecimals(3)
        ac_.addWidget(s.ar, 1); s.ac.hide(); l5.addWidget(s.ac)
        s.arc = QWidget(); arc_ = QHBoxLayout(s.arc); arc_.setContentsMargins(0, 0, 0, 0)
        arc_.addWidget(QLabel("W (m):")); s.aw = QDoubleSpinBox(); s.aw.setRange(.01, 5); s.aw.setValue(.6); s.aw.setDecimals(3); arc_.addWidget(s.aw, 1)
        arc_.addWidget(QLabel("L (m):")); s.al = QDoubleSpinBox(); s.al.setRange(.01, 5); s.al.setValue(.4); s.al.setDecimals(3); arc_.addWidget(s.al, 1)
        l5.addWidget(s.arc)
        s.bact = QPushButton("▶  Run Actual"); s.bact.setStyleSheet(s._B("#00e5a0"))
        s.bact.clicked.connect(s._run_act); l5.addWidget(s.bact)
        s.lact = QLabel(""); s.lact.setStyleSheet("color:#00e5a0;font-size:13px;font-weight:bold"); l5.addWidget(s.lact)
        s.lact_note = QLabel("")
        s.lact_note.setWordWrap(True)
        s.lact_note.setStyleSheet("color:#7ae5c4;font-size:11px")
        l5.addWidget(s.lact_note)
        cov_hint = QLabel("Area tabs: colored = inflated accessible area, light = floor but inaccessible, dark = blocked after robot inflation.")
        cov_hint.setWordWrap(True)
        l5.addWidget(cov_hint)
        stc_hint = QLabel("Planner Path tab: reference path = blue, actual path = green when a path planner is selected.")
        stc_hint.setWordWrap(True)
        l5.addWidget(stc_hint)

        # RII display
        s.riif = QFrame()
        s.riif.setStyleSheet("QFrame{background:#1a1800;border:2px solid #ffd70060;border-radius:10px;padding:12px}")
        rf = QVBoxLayout(s.riif)
        rf.setSpacing(6)
        s.riit = QLabel("RII Horizontal"); s.riit.setAlignment(Qt.AlignCenter)
        s.riit.setStyleSheet("color:#ffd700;font-size:18px;font-weight:bold;letter-spacing:1px")
        rf.addWidget(s.riit)
        s.riiv = QLabel("—"); s.riiv.setAlignment(Qt.AlignCenter)
        s.riiv.setMinimumHeight(58)
        s.riiv.setWordWrap(False)
        s.riiv.setStyleSheet("color:#ffd700;font-size:42px;font-weight:bold;font-family:monospace")
        rf.addWidget(s.riiv)
        s.riis = QLabel(""); s.riis.setAlignment(Qt.AlignCenter)
        s.riis.setStyleSheet("color:#6b7a8d;font-size:12px"); rf.addWidget(s.riis)
        s.riif.hide(); l5.addWidget(s.riif)

        g5.setLayout(l5); ll.addWidget(g5)

        # Step 4 — RII Horizontal Analysis
        g6 = QGroupBox("Step 4 — RII Horizontal Analysis"); l6 = QVBoxLayout(); l6.setSpacing(8)
        l6.addWidget(QLabel(
            "Load a CloudCompare-labeled PCD or PLY (labels 0-15 from the paper taxonomy)\n"
            "to explain the RII Horizontal gap by fixation group and recommended interventions."
        ))

        hsem = QHBoxLayout()
        s.e_sem_pcd = QLineEdit(""); s.e_sem_pcd.setPlaceholderText("Path to labeled .pcd or .ply file")
        bsem = QPushButton("📂"); bsem.setFixedWidth(34)
        bsem.clicked.connect(s._browse_sem_pcd)
        hsem.addWidget(QLabel("Labeled Cloud:")); hsem.addWidget(s.e_sem_pcd, 1); hsem.addWidget(bsem)
        l6.addLayout(hsem)

        s.sem_status = QLabel(""); s.sem_status.setStyleSheet("color:#6b7a8d;font-size:11px")
        l6.addWidget(s.sem_status)

        s.bsem_3d = QPushButton("View Semantic Labels in 3D Viewer")
        s.bsem_3d.setStyleSheet(
            "QPushButton{background:#ff66cc;color:#0a0e14;border:none;border-radius:7px;"
            "padding:8px;font-weight:bold;font-size:11px}"
            "QPushButton:hover{border:2px solid white}"
            "QPushButton:disabled{background:#333;color:#666}")
        s.bsem_3d.setToolTip("Show the loaded semantic point cloud in the 3D Viewer, colored by label class")
        s.bsem_3d.clicked.connect(s._show_semantic_3d)
        s.bsem_3d.setEnabled(False)
        l6.addWidget(s.bsem_3d)

        def _sem_btn_style(color):
            return (f"QPushButton {{ background: {color}; color: #0a0e14; border: none; border-radius: 7px; "
                    f"padding: 8px; font-weight: bold; font-size: 11px; }}"
                    f"QPushButton:hover {{ border: 2px solid white; }}"
                    f"QPushButton:disabled {{ background: #333; color: #666; }}")

        s.bsem = QPushButton("🔬  Analyze RII_Horizontal"); s.bsem.setStyleSheet(s._B("#ff66cc"))
        s.bsem.clicked.connect(s._run_semantic_analysis); l6.addWidget(s.bsem)

        s.sem_prog_lbl = QLabel("")
        s.sem_prog_lbl.setStyleSheet("color:#7ad9ff;font-size:11px")
        s.sem_prog_lbl.hide()
        l6.addWidget(s.sem_prog_lbl)
        s.sem_prog = QProgressBar()
        s.sem_prog.setRange(0, 100)
        s.sem_prog.setValue(0)
        s.sem_prog.setTextVisible(True)
        s.sem_prog.setFormat("%p%")
        s.sem_prog.setMaximumHeight(18)
        s.sem_prog.hide()
        l6.addWidget(s.sem_prog)

        s.sem_candidate_status = QLabel("")
        s.sem_candidate_status.setWordWrap(True)
        s.sem_candidate_status.setStyleSheet("color:#ffcc66;font-size:11px")
        s.sem_candidate_status.setText(
            "Run semantic analysis to populate removable-object candidates, filter them by fixation, and recompute an Optimised RII score."
        )
        l6.addWidget(s.sem_candidate_status)

        s.sem_candidate_hdr = QLabel("Object candidates to remove")
        s.sem_candidate_hdr.setStyleSheet("color:#ffcc66;font-size:12px;font-weight:bold")
        s.sem_candidate_hdr.setWordWrap(True)
        l6.addWidget(s.sem_candidate_hdr)

        sem_filter_row = QHBoxLayout()
        s.sem_filter_lbl = QLabel("Filter:")
        sem_filter_row.addWidget(s.sem_filter_lbl)
        s.sem_filter = QComboBox()
        s.sem_filter.addItem("All Fixations", None)
        s.sem_filter.addItem("Portable", "Portable")
        s.sem_filter.addItem("Movable", "Movable")
        s.sem_filter.addItem("Semi-Fixed", "Semi-Fixed")
        s.sem_filter.currentIndexChanged.connect(s._apply_semantic_candidate_filter)
        s.sem_filter.setEnabled(False)
        s.sem_filter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sem_filter_row.addWidget(s.sem_filter)
        s.bsem_select_filtered = QPushButton("Select Filtered")
        s.bsem_select_filtered.setStyleSheet(_sem_btn_style("#ffd700"))
        s.bsem_select_filtered.clicked.connect(s._select_filtered_semantic_candidates)
        s.bsem_select_filtered.setEnabled(False)
        sem_filter_row.addWidget(s.bsem_select_filtered)
        l6.addLayout(sem_filter_row)

        s.sem_candidate_list = QListWidget()
        s.sem_candidate_list.setMaximumHeight(170)
        s.sem_candidate_list.setStyleSheet(
            "QListWidget{background:#111820;border:1px solid #2a3545;border-radius:6px;color:#c5cdd8;font-size:11px;}"
            "QListWidget::item{padding:4px;}"
        )
        s.sem_candidate_list.itemChanged.connect(s._semantic_candidate_selection_changed)
        s.sem_candidate_list.currentItemChanged.connect(s._semantic_candidate_current_changed)
        s.sem_candidate_list.itemClicked.connect(lambda item: s._semantic_candidate_current_changed(item, None))
        placeholder = QListWidgetItem("Run semantic analysis to populate removable-object candidates.")
        placeholder.setFlags(Qt.NoItemFlags)
        s.sem_candidate_list.addItem(placeholder)
        s.sem_candidate_list.setEnabled(False)
        l6.addWidget(s.sem_candidate_list)

        s.bsem_select_portable = QPushButton("Portable")
        s.bsem_select_portable.setStyleSheet(_sem_btn_style("#ffc857"))
        s.bsem_select_portable.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Portable"}))
        s.bsem_select_portable.setEnabled(False)
        s.bsem_select_movable = QPushButton("Movable")
        s.bsem_select_movable.setStyleSheet(_sem_btn_style("#ff9f43"))
        s.bsem_select_movable.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Movable"}))
        s.bsem_select_movable.setEnabled(False)
        s.bsem_select_semi_fixed = QPushButton("Semi-Fixed")
        s.bsem_select_semi_fixed.setStyleSheet(_sem_btn_style("#f368e0"))
        s.bsem_select_semi_fixed.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Semi-Fixed"}))
        s.bsem_select_semi_fixed.setEnabled(False)
        s.bsem_select_portable_movable = QPushButton("Portable + Movable")
        s.bsem_select_portable_movable.setStyleSheet(_sem_btn_style("#ffcc66"))
        s.bsem_select_portable_movable.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Portable", "Movable"}))
        s.bsem_select_portable_movable.setEnabled(False)
        s.bsem_select_all_candidates = QPushButton("All Removable")
        s.bsem_select_all_candidates.setStyleSheet(_sem_btn_style("#7ad9ff"))
        s.bsem_select_all_candidates.clicked.connect(lambda: s._set_semantic_candidates_by_fixation(set(SEMANTIC_REMOVABLE_FIXATIONS)))
        s.bsem_select_all_candidates.setEnabled(False)
        s.bsem_clear_candidates = QPushButton("Clear Selection")
        s.bsem_clear_candidates.setStyleSheet(_sem_btn_style("#7a8ba3"))
        s.bsem_clear_candidates.clicked.connect(s._clear_semantic_candidates)
        s.bsem_clear_candidates.setEnabled(False)
        for btn in (
            s.bsem_select_portable,
            s.bsem_select_movable,
            s.bsem_select_semi_fixed,
            s.bsem_select_portable_movable,
            s.bsem_select_all_candidates,
            s.bsem_clear_candidates,
        ):
            btn.setMinimumHeight(34)

        sem_btn_grid = QGridLayout()
        sem_btn_grid.setHorizontalSpacing(8)
        sem_btn_grid.setVerticalSpacing(8)
        sem_btn_grid.addWidget(s.bsem_select_portable, 0, 0)
        sem_btn_grid.addWidget(s.bsem_select_movable, 0, 1)
        sem_btn_grid.addWidget(s.bsem_select_semi_fixed, 1, 0)
        sem_btn_grid.addWidget(s.bsem_select_portable_movable, 1, 1)
        sem_btn_grid.addWidget(s.bsem_select_all_candidates, 2, 0)
        sem_btn_grid.addWidget(s.bsem_clear_candidates, 2, 1)
        l6.addLayout(sem_btn_grid)

        s.bsem_recompute = QPushButton("▶  Recompute Optimised RII")
        s.bsem_recompute.setStyleSheet(s._B("#00e5a0"))
        s.bsem_recompute.clicked.connect(s._recompute_semantic_improvement)
        s.bsem_recompute.setEnabled(False)
        l6.addWidget(s.bsem_recompute)

        s.sem_riif = QFrame()
        s.sem_riif.setStyleSheet("QFrame{background:#1a1800;border:2px solid #ffd70060;border-radius:10px;padding:12px}")
        sem_rf = QVBoxLayout(s.sem_riif)
        sem_rf.setSpacing(6)
        # ── Side-by-side comparison: Current → Optimised ──
        sem_cmp = QHBoxLayout(); sem_cmp.setSpacing(12)
        # Current RII (left)
        cur_col = QVBoxLayout(); cur_col.setSpacing(2)
        s.sem_cur_title = QLabel("Current"); s.sem_cur_title.setAlignment(Qt.AlignCenter)
        s.sem_cur_title.setStyleSheet("color:#6b7a8d;font-size:12px;font-weight:bold")
        cur_col.addWidget(s.sem_cur_title)
        s.sem_cur_val = QLabel("—"); s.sem_cur_val.setAlignment(Qt.AlignCenter)
        s.sem_cur_val.setStyleSheet("color:#6b7a8d;font-size:28px;font-weight:bold;font-family:monospace")
        cur_col.addWidget(s.sem_cur_val)
        sem_cmp.addLayout(cur_col)
        # Arrow + Delta (center)
        delta_col = QVBoxLayout(); delta_col.setSpacing(2)
        s.sem_arrow = QLabel("→"); s.sem_arrow.setAlignment(Qt.AlignCenter)
        s.sem_arrow.setStyleSheet("color:#ffd700;font-size:22px;font-weight:bold")
        delta_col.addWidget(s.sem_arrow)
        s.sem_delta = QLabel(""); s.sem_delta.setAlignment(Qt.AlignCenter)
        s.sem_delta.setStyleSheet("color:#00e5a0;font-size:14px;font-weight:bold")
        delta_col.addWidget(s.sem_delta)
        sem_cmp.addLayout(delta_col)
        # Optimised RII (right)
        opt_col = QVBoxLayout(); opt_col.setSpacing(2)
        s.sem_opt_title = QLabel("Optimised"); s.sem_opt_title.setAlignment(Qt.AlignCenter)
        s.sem_opt_title.setStyleSheet("color:#ffd700;font-size:12px;font-weight:bold")
        opt_col.addWidget(s.sem_opt_title)
        s.sem_riiv = QLabel("—"); s.sem_riiv.setAlignment(Qt.AlignCenter)
        s.sem_riiv.setStyleSheet("color:#ffd700;font-size:28px;font-weight:bold;font-family:monospace")
        opt_col.addWidget(s.sem_riiv)
        sem_cmp.addLayout(opt_col)
        sem_rf.addLayout(sem_cmp)
        # Subtitle line
        s.sem_riis = QLabel(""); s.sem_riis.setAlignment(Qt.AlignCenter)
        s.sem_riis.setWordWrap(True)
        s.sem_riis.setStyleSheet("color:#6b7a8d;font-size:11px")
        sem_rf.addWidget(s.sem_riis)
        s.sem_riif.hide()
        l6.addWidget(s.sem_riif)

        s.sem_layered_status = QTextEdit(); s.sem_layered_status.setReadOnly(True)
        s.sem_layered_status.setMaximumHeight(145); s.sem_layered_status.setMinimumHeight(90)
        s.sem_layered_status.setStyleSheet(
            "background:#111820;border:1px solid #2a3545;border-radius:6px;"
            "color:#7ad9ff;font-family:monospace;font-size:11px;padding:8px"
        )
        s.sem_layered_status.hide()
        l6.addWidget(s.sem_layered_status)

        # Recommendations display
        s.sem_report = QTextEdit(); s.sem_report.setReadOnly(True)
        s.sem_report.setMaximumHeight(250); s.sem_report.setMinimumHeight(100)
        s.sem_report.setStyleSheet("background:#111820;border:1px solid #2a3545;border-radius:6px;"
                                   "color:#c5cdd8;font-family:monospace;font-size:11px;padding:8px")
        s.sem_report.hide()
        l6.addWidget(s.sem_report)

        g6.setLayout(l6); ll.addWidget(g6)

        # Step 5 — RII Vertical (Wall Paint Coverage)
        g7 = QGroupBox("Step 5 — RII Vertical (Wall Reachability)"); l7 = QVBoxLayout(); l7.setSpacing(8)
        l7.addWidget(QLabel(
            "Compute wall surface reachability from accessible floor using STVL raycasting.\n"
            "Requires: labelled point cloud (Step 4) + RII Horizontal (Step 3)."
        ))

        # Wall height band
        whb = QHBoxLayout()
        whb.addWidget(QLabel("Wall min h (m):"))
        s.rv_wall_min_h = QDoubleSpinBox(); s.rv_wall_min_h.setRange(0.0, 10.0); s.rv_wall_min_h.setValue(0.40); s.rv_wall_min_h.setDecimals(2); s.rv_wall_min_h.setSingleStep(0.1)
        whb.addWidget(s.rv_wall_min_h)
        whb.addWidget(QLabel("max h (m):"))
        s.rv_wall_max_h = QDoubleSpinBox(); s.rv_wall_max_h.setRange(0.1, 20.0); s.rv_wall_max_h.setValue(2.00); s.rv_wall_max_h.setDecimals(2); s.rv_wall_max_h.setSingleStep(0.1)
        whb.addWidget(s.rv_wall_max_h)
        l7.addLayout(whb)

        # Raycasting params
        rcp = QHBoxLayout()
        rcp.addWidget(QLabel("Voxel (m):"))
        s.rv_voxel = QDoubleSpinBox(); s.rv_voxel.setRange(0.01, 1.0); s.rv_voxel.setValue(0.05); s.rv_voxel.setDecimals(3); s.rv_voxel.setSingleStep(0.01)
        rcp.addWidget(s.rv_voxel)
        rcp.addWidget(QLabel("Reach (m):"))
        s.rv_reach = QDoubleSpinBox(); s.rv_reach.setRange(0.1, 5.0); s.rv_reach.setValue(1.0); s.rv_reach.setDecimals(2); s.rv_reach.setSingleStep(0.1)
        rcp.addWidget(s.rv_reach)
        rcp.addWidget(QLabel("Angle (°):"))
        s.rv_angle = QDoubleSpinBox(); s.rv_angle.setRange(1.0, 45.0); s.rv_angle.setValue(10.0); s.rv_angle.setDecimals(1); s.rv_angle.setSingleStep(5.0)
        rcp.addWidget(s.rv_angle)
        l7.addLayout(rcp)

        # Paint tool params
        ptp = QHBoxLayout()
        ptp.addWidget(QLabel("Paint width (m):"))
        s.rv_paint_w = QDoubleSpinBox(); s.rv_paint_w.setRange(0.01, 9999.0); s.rv_paint_w.setValue(0.25); s.rv_paint_w.setDecimals(2); s.rv_paint_w.setSingleStep(0.05)
        ptp.addWidget(s.rv_paint_w)
        ptp.addWidget(QLabel("Vertical span (m):"))
        s.rv_paint_vspan = QDoubleSpinBox(); s.rv_paint_vspan.setRange(0.01, 9999.0); s.rv_paint_vspan.setValue(0.30); s.rv_paint_vspan.setDecimals(2); s.rv_paint_vspan.setSingleStep(0.05)
        ptp.addWidget(s.rv_paint_vspan)
        ptp.addWidget(QLabel("Sweep step (m):"))
        s.rv_sweep = QDoubleSpinBox(); s.rv_sweep.setRange(0.01, 9999.0); s.rv_sweep.setValue(0.20); s.rv_sweep.setDecimals(2); s.rv_sweep.setSingleStep(0.05)
        ptp.addWidget(s.rv_sweep)
        l7.addLayout(ptp)

        # Sampling params
        smp = QHBoxLayout()
        smp.addWidget(QLabel("Ground stride (px):"))
        s.rv_stride = QSpinBox(); s.rv_stride.setRange(1, 20); s.rv_stride.setValue(3)
        smp.addWidget(s.rv_stride)
        smp.addWidget(QLabel("Max samples (count):"))
        s.rv_max_samples = QSpinBox(); s.rv_max_samples.setRange(1000, 200000); s.rv_max_samples.setValue(60000); s.rv_max_samples.setSingleStep(5000)
        smp.addWidget(s.rv_max_samples)
        l7.addLayout(smp)

        # Wall label IDs
        wli = QHBoxLayout()
        wli.addWidget(QLabel("Wall label IDs:"))
        s.rv_wall_ids = QLineEdit("1"); s.rv_wall_ids.setPlaceholderText("Comma-separated label IDs treated as wall (e.g. 1)")
        s.rv_wall_ids.setToolTip("Semantic label IDs to treat as wall surface. Default: 1 (Wall)")
        wli.addWidget(s.rv_wall_ids, 1)
        l7.addLayout(wli)

        # ── Wall segment detection & selection ──
        s.brv_detect = QPushButton("🔍  Detect Wall Segments")
        s.brv_detect.setStyleSheet(s._B("#ff9900"))
        s.brv_detect.clicked.connect(s._detect_wall_segments)
        l7.addWidget(s.brv_detect)

        s.rv_wall_status = QLabel(""); s.rv_wall_status.setStyleSheet("color:#6b7a8d;font-size:11px")
        l7.addWidget(s.rv_wall_status)

        l7.addWidget(QLabel("Wall segments (click to visualise in 3D, check to include in RII_V):"))
        s.rv_wall_list = QListWidget()
        s.rv_wall_list.setMaximumHeight(150)
        s.rv_wall_list.setStyleSheet(
            "QListWidget{background:#0d1117;color:#c5cdd8;border:1px solid #2a3545;font-size:11px}"
            "QListWidget::item{padding:3px}"
            "QListWidget::item:selected{background:#ff990030;color:#ff9900}"
        )
        s.rv_wall_list.currentItemChanged.connect(s._rv_wall_current_changed)
        s.rv_wall_list.itemChanged.connect(s._rv_wall_check_changed)
        l7.addWidget(s.rv_wall_list)

        rv_wbtn = QHBoxLayout()
        s.brv_sel_all = QPushButton("Select All"); s.brv_sel_all.setFixedHeight(28)
        s.brv_sel_all.setStyleSheet("QPushButton{background:#ff9900;color:#0a0e14;border:none;border-radius:5px;padding:4px 12px;font-weight:bold;font-size:11px}")
        s.brv_sel_all.clicked.connect(lambda: s._rv_wall_select_all(True))
        rv_wbtn.addWidget(s.brv_sel_all)
        s.brv_sel_none = QPushButton("Clear All"); s.brv_sel_none.setFixedHeight(28)
        s.brv_sel_none.setStyleSheet("QPushButton{background:#333;color:#c5cdd8;border:none;border-radius:5px;padding:4px 12px;font-weight:bold;font-size:11px}")
        s.brv_sel_none.clicked.connect(lambda: s._rv_wall_select_all(False))
        rv_wbtn.addWidget(s.brv_sel_none)
        l7.addLayout(rv_wbtn)

        # Combined RII gamma
        gml = QHBoxLayout()
        gml.addWidget(QLabel("Combined γ:"))
        s.rv_gamma = QDoubleSpinBox(); s.rv_gamma.setRange(0.0, 1.0); s.rv_gamma.setValue(0.50); s.rv_gamma.setDecimals(2); s.rv_gamma.setSingleStep(0.1)
        s.rv_gamma.setToolTip(
            "γ (gamma) controls the balance between Operational Efficiency (OE) and Surface Continuity (SC) "
            "in the combined RII score.\n\n"
            "Formula:  Combined = TCR × (γ · OE + (1-γ) · SC)\n\n"
            "• γ = 1.0  →  Combined depends only on OE (how much floor can reach walls)\n"
            "• γ = 0.0  →  Combined depends only on SC (how contiguous the painted walls are)\n"
            "• γ = 0.5  →  Equal weight to OE and SC (default)\n\n"
            "Where:\n"
            "  TCR = Task Coverage Rate = painted wall area / total wall area\n"
            "  OE  = Operational Efficiency = fraction of floor that can reach walls\n"
            "  SC  = Surface Continuity = largest contiguous painted region / total painted"
        )
        gml.addWidget(s.rv_gamma)
        # Gamma explanation label
        s._gamma_info = QLabel(
            "<span style='color:#6b7a8d;font-size:10px'>"
            "γ balances Operational Efficiency (OE) vs Surface Continuity (SC).<br>"
            "<b>Combined = TCR × (γ · OE + (1-γ) · SC)</b><br>"
            "γ→1: prioritise floor reachability &nbsp;|&nbsp; γ→0: prioritise wall contiguity &nbsp;|&nbsp; γ=0.5: equal weight"
            "</span>"
        )
        s._gamma_info.setWordWrap(True)
        gml.addWidget(s._gamma_info, 1)
        l7.addLayout(gml)

        s.brv = QPushButton("▶  Compute RII Vertical (selected walls)"); s.brv.setStyleSheet(s._B("#ff9900"))
        s.brv.clicked.connect(s._run_rii_vertical); l7.addWidget(s.brv)

        s.rv_prog_lbl = QLabel(""); s.rv_prog_lbl.setStyleSheet("color:#ff9900;font-size:11px"); s.rv_prog_lbl.hide()
        l7.addWidget(s.rv_prog_lbl)
        s.rv_prog = QProgressBar(); s.rv_prog.setRange(0, 100); s.rv_prog.setValue(0)
        s.rv_prog.setTextVisible(True); s.rv_prog.setFormat("%p%"); s.rv_prog.setMaximumHeight(18); s.rv_prog.hide()
        l7.addWidget(s.rv_prog)

        # ── RII Vertical result card ──
        s.rv_riif = QFrame()
        s.rv_riif.setStyleSheet("QFrame{background:#1a1200;border:2px solid #ff990060;border-radius:10px;padding:12px}")
        rv_rf = QVBoxLayout(s.rv_riif); rv_rf.setSpacing(6)
        s.rv_riit = QLabel("RII Vertical — Task Coverage Rate (TCR)"); s.rv_riit.setAlignment(Qt.AlignCenter)
        s.rv_riit.setStyleSheet("color:#ff9900;font-size:18px;font-weight:bold;letter-spacing:1px")
        rv_rf.addWidget(s.rv_riit)
        s.rv_riiv = QLabel("—"); s.rv_riiv.setAlignment(Qt.AlignCenter); s.rv_riiv.setMinimumHeight(48)
        s.rv_riiv.setStyleSheet("color:#ff9900;font-size:36px;font-weight:bold;font-family:monospace")
        rv_rf.addWidget(s.rv_riiv)
        s.rv_riis = QLabel(""); s.rv_riis.setAlignment(Qt.AlignCenter); s.rv_riis.setWordWrap(True)
        s.rv_riis.setStyleSheet("color:#6b7a8d;font-size:11px")
        rv_rf.addWidget(s.rv_riis)
        s.rv_riif.hide()
        l7.addWidget(s.rv_riif)

        # ── Combined RII card ──
        s.rv_combf = QFrame()
        s.rv_combf.setStyleSheet("QFrame{background:#0a1a1a;border:2px solid #00e5a060;border-radius:10px;padding:12px}")
        rv_cf = QVBoxLayout(s.rv_combf); rv_cf.setSpacing(6)

        # Side-by-side: RII_H | RII_V | Combined
        rv_cmp = QHBoxLayout(); rv_cmp.setSpacing(8)
        # RII_H (left)
        hcol = QVBoxLayout(); hcol.setSpacing(2)
        s.rv_ch_title = QLabel("RII Horizontal"); s.rv_ch_title.setAlignment(Qt.AlignCenter)
        s.rv_ch_title.setStyleSheet("color:#ffd700;font-size:11px;font-weight:bold")
        hcol.addWidget(s.rv_ch_title)
        s.rv_ch_val = QLabel("—"); s.rv_ch_val.setAlignment(Qt.AlignCenter)
        s.rv_ch_val.setStyleSheet("color:#ffd700;font-size:22px;font-weight:bold;font-family:monospace")
        hcol.addWidget(s.rv_ch_val)
        rv_cmp.addLayout(hcol)
        # RII_V (middle)
        vcol = QVBoxLayout(); vcol.setSpacing(2)
        s.rv_cv_title = QLabel("RII Vertical"); s.rv_cv_title.setAlignment(Qt.AlignCenter)
        s.rv_cv_title.setStyleSheet("color:#ff9900;font-size:11px;font-weight:bold")
        vcol.addWidget(s.rv_cv_title)
        s.rv_cv_val = QLabel("—"); s.rv_cv_val.setAlignment(Qt.AlignCenter)
        s.rv_cv_val.setStyleSheet("color:#ff9900;font-size:22px;font-weight:bold;font-family:monospace")
        vcol.addWidget(s.rv_cv_val)
        rv_cmp.addLayout(vcol)
        # Combined (right)
        ccol = QVBoxLayout(); ccol.setSpacing(2)
        s.rv_cc_title = QLabel("Combined"); s.rv_cc_title.setAlignment(Qt.AlignCenter)
        s.rv_cc_title.setStyleSheet("color:#00e5a0;font-size:11px;font-weight:bold")
        ccol.addWidget(s.rv_cc_title)
        s.rv_cc_val = QLabel("—"); s.rv_cc_val.setAlignment(Qt.AlignCenter)
        s.rv_cc_val.setStyleSheet("color:#00e5a0;font-size:22px;font-weight:bold;font-family:monospace")
        ccol.addWidget(s.rv_cc_val)
        rv_cmp.addLayout(ccol)
        rv_cf.addLayout(rv_cmp)

        # Detail line
        s.rv_comb_detail = QLabel(""); s.rv_comb_detail.setAlignment(Qt.AlignCenter)
        s.rv_comb_detail.setWordWrap(True)
        s.rv_comb_detail.setStyleSheet("color:#6b7a8d;font-size:11px")
        rv_cf.addWidget(s.rv_comb_detail)

        # Formula reminder
        s.rv_comb_formula = QLabel("Combined = Task Coverage Rate (TCR) × (γ · Operational Efficiency (OE) + (1-γ) · Surface Continuity (SC))\nWeighted = 0.5 · RII_H + 0.5 · RII_V")
        s.rv_comb_formula.setAlignment(Qt.AlignCenter); s.rv_comb_formula.setWordWrap(True)
        s.rv_comb_formula.setStyleSheet("color:#444;font-size:10px;font-style:italic")
        rv_cf.addWidget(s.rv_comb_formula)
        s.rv_combf.hide()
        l7.addWidget(s.rv_combf)

        g7.setLayout(l7); ll.addWidget(g7)

        ll.addStretch()
        ls.setWidget(lw); sp.addWidget(ls)

        # RIGHT panel
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)

        _tab_ss = ("QTabBar{background:transparent;border:none}"
                    "QTabBar::tab{background:#1a2230;color:#6b7a8d;border:1px solid #2a3545;"
                    "border-radius:5px;padding:4px 12px;margin:2px 2px;font-size:11px}"
                    "QTabBar::tab:selected{background:#00e5a020;color:#00e5a0;border-color:#00e5a0}")
        _tab_names = [PRIMARY_SELECTION_VIEW, "Traversable Ground", "3D Viewer", "Reference Coverage", "Actual Coverage", "Compare", "Planner Path", "Semantic", "Vertical Coverage"]
        _tab_tooltips = {
            PRIMARY_SELECTION_VIEW: "2D projection of the raw point cloud.\nBlack = obstacle hit, White = free space.\nUsed as the base map for RII computation and area selection.",
            "Traversable Ground": "Terrain analysis sidecar.\nShows which ground cells pass slope, step-height and roughness checks.\nUsed to exclude non-traversable terrain from accessible area.",
        }

        # Primary tab bar row (with split-view toggle)
        tab_row = QHBoxLayout(); tab_row.setContentsMargins(0, 0, 0, 0); tab_row.setSpacing(0)
        s.view_tab_bar = QTabBar()
        s.view_tab_bar.setMovable(True)
        s.view_tab_bar.setExpanding(False)
        s.view_tab_bar.setFixedHeight(32)
        s.view_tab_bar.setStyleSheet(_tab_ss)
        s.vb = {}
        for nm in _tab_names:
            idx = s.view_tab_bar.addTab(nm)
            if nm in _tab_tooltips:
                s.view_tab_bar.setTabToolTip(idx, _tab_tooltips[nm])
            s.vb[nm] = idx
        s.view_tab_bar.setCurrentIndex(0)
        s.view_tab_bar.currentChanged.connect(lambda idx: s._switch_view(s.view_tab_bar.tabText(idx)))
        tab_row.addWidget(s.view_tab_bar, 1)
        s.btn_split_view = QPushButton("Split View")
        s.btn_split_view.setCheckable(True)
        s.btn_split_view.setFixedHeight(28)
        s.btn_split_view.setStyleSheet(
            "QPushButton{background:#1a2230;color:#6b7a8d;border:1px solid #2a3545;"
            "border-radius:4px;padding:2px 12px;font-size:11px;font-weight:bold}"
            "QPushButton:checked{background:#00e5a030;color:#00e5a0;border-color:#00e5a0}")
        s.btn_split_view.setToolTip("Toggle side-by-side split view — drag the divider to resize each panel")
        s.btn_split_view.clicked.connect(s._toggle_split_view)
        tab_row.addWidget(s.btn_split_view)
        rl.addLayout(tab_row)

        s.prog = QProgressBar(); s.prog.setTextVisible(False); s.prog.setMaximumHeight(4)
        rl.addWidget(s.prog)

        # Primary viewer (always present)
        s.view_stack = QStackedWidget()
        s.mw = MapW(); s.mw.sel_changed.connect(s._on_sel); s.view_stack.addWidget(s.mw)
        s.pcw = PointCloudW()
        if hasattr(s.pcw, "gl_failed"):
            s.pcw.gl_failed.connect(s._fallback_point_cloud_viewer)
        s.view_stack.addWidget(s.pcw)

        # Split viewer container (splitter holds primary + secondary)
        s._split_splitter = QSplitter(Qt.Horizontal)
        s._split_splitter.setChildrenCollapsible(False)
        s._split_splitter.setHandleWidth(10)
        s._split_splitter.setStyleSheet(
            "QSplitter::handle{background:#2a3545;border-radius:3px}"
            "QSplitter::handle:hover{background:#00e5a0}"
        )
        s._split_splitter.addWidget(s.view_stack)

        # Secondary viewer panel (built once, shown/hidden)
        s._split_panel = QWidget()
        _sp_layout = QVBoxLayout(s._split_panel); _sp_layout.setContentsMargins(0, 0, 0, 0); _sp_layout.setSpacing(0)
        s._split_tab_bar = QTabBar()
        s._split_tab_bar.setMovable(True)
        s._split_tab_bar.setExpanding(False)
        s._split_tab_bar.setFixedHeight(32)
        s._split_tab_bar.setStyleSheet(_tab_ss)
        for nm in _tab_names:
            idx2 = s._split_tab_bar.addTab(nm)
            if nm in _tab_tooltips:
                s._split_tab_bar.setTabToolTip(idx2, _tab_tooltips[nm])
        s._split_tab_bar.setCurrentIndex(0)
        s._split_tab_bar.currentChanged.connect(lambda idx: s._switch_split_view(s._split_tab_bar.tabText(idx)))
        _sp_layout.addWidget(s._split_tab_bar)
        s._split_view_stack = QStackedWidget()
        s._split_mw = MapW()
        s._split_view_stack.addWidget(s._split_mw)
        s._split_pcw = PointCloudW()
        if hasattr(s._split_pcw, "gl_failed"):
            s._split_pcw.gl_failed.connect(lambda reason: s._log(f"Split 3D viewer: {reason}", "warn"))
        s._split_view_stack.addWidget(s._split_pcw)
        _sp_layout.addWidget(s._split_view_stack, 1)
        s._split_splitter.addWidget(s._split_panel)
        s._split_panel.hide()

        rl.addWidget(s._split_splitter, 1)
        s.log_box = QTextEdit(); s.log_box.setReadOnly(True); s.log_box.setMaximumHeight(140)
        rl.addWidget(s.log_box)
        sp.addWidget(rw); sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        sp.setSizes([470, 910])

    # ── Selection ──
    def _enable_sel(s):
        p, _ = s._get_pgm()
        if not p: return
        if BLOCKED_MAP_VIEW not in s._imgs: s._load_map(p)
        mode = "freeform" if hasattr(s, "sel_mode") and s.sel_mode.currentIndex() == 1 else "rectangle"
        s._switch_view(PRIMARY_SELECTION_VIEW); s.mw.clear_sel(); s.mw.enable_sel(mode)
        if mode == "freeform":
            s._log(f"Draw a freeform loop on {PRIMARY_SELECTION_VIEW.lower()} to select area.", "gold")
        else:
            s._log(f"Drag on {PRIMARY_SELECTION_VIEW.lower()} to select a rectangular area.", "gold")

    def _on_sel(s, r):
        if not r: return
        p, y = s._get_pgm()
        if not p: return
        yd = parse_yaml(y)
        res = yd['resolution']
        bounds = selection_bounds_px(r)
        if bounds is None:
            return
        x1, y1, x2, y2 = bounds
        wm = (x2 - x1 + 1) * res
        hm = (y2 - y1 + 1) * res
        mask = selection_mask_from_display(r, s._map_w, s._map_h)
        area_m2 = 0.0 if mask is None else float(mask.sum()) * res * res
        if selection_kind(r) == "freeform":
            s.slbl.setText(f"Selected freeform: {area_m2:.1f}m² (bbox {wm:.1f}×{hm:.1f}m)")
            s._log(f"Freeform area: {area_m2:.1f}m²", "gold")
        else:
            s.slbl.setText(f"Selected rectangle: {wm:.1f}×{hm:.1f}m = {area_m2:.1f}m²")
            s._log(f"Rectangle area: {wm:.1f}×{hm:.1f}m", "gold")
        center = s._selection_center_world(y)
        if center is not None:
            s._log(f"Selection center: ({center[0]:.2f}, {center[1]:.2f})", "info")

    # ── Steps 1-4 ──
    def _step1(s):
        p = s.e_in.text().strip()
        if not os.path.isfile(p): QMessageBox.warning(s, "Error", p); return
        s.b1.setEnabled(False)
        w = ViewW(p, "3D Viewer")
        w.log.connect(s._log)
        w.loaded.connect(lambda cloud: s._set_cloud("3D Viewer", cloud))
        w.done.connect(lambda *_: s.b1.setEnabled(True))
        s._wk.append(w); w.start()

    def _step2(s):
        if not hasattr(s, 'e_out') or not hasattr(s, 'cp') or not hasattr(s, 'b2'):
            s._log("Step 2 (cleanup) is not available in this pipeline version.", "warn")
            return
        pi = s.e_in.text().strip()
        if not os.path.isfile(pi): QMessageBox.warning(s, "Error", pi); return
        od = s.e_out.text(); os.makedirs(od, exist_ok=True)
        po = os.path.join(od, filtered_point_cloud_filename(pi))
        args = " ".join(f"--{k} {v.value()}" for k, v in s.cp.items())
        s.b2.setEnabled(False)
        cmd = (
            f"cd {shlex.quote(PRECLEAN)} && "
            f"python3 pre_map.py --in {shlex.quote(pi)} --out {shlex.quote(po)} {args}"
        )
        w = ShellW(cmd, "Clean", False)
        w.log.connect(s._log); w.done.connect(lambda *_: s.b2.setEnabled(True))
        s._wk.append(w); w.start()

    def _step3(s):
        if not hasattr(s, 'e_out'):
            s._log("Step 3 (view clean cloud) is not available in this pipeline version.", "warn")
            return
        p = resolve_point_cloud_path(
            s.e_out.text(),
            filtered_point_cloud_stem_candidates(s.e_in.text().strip()),
        )
        if not os.path.isfile(p): QMessageBox.warning(s, "Error", p); return
        w = ViewW(p, "Clean Cloud")
        w.log.connect(s._log)
        w.loaded.connect(lambda cloud: s._set_cloud("Clean Cloud", cloud))
        w.done.connect(lambda *_: None)
        s._wk.append(w); w.start()

    def _step4(s):
        try:
            p, src_label = s._selected_map_source_path()
        except FileNotFoundError as exc:
            QMessageBox.warning(s, "Error", str(exc))
            return
        s.b4.setEnabled(False); s.prog.setValue(0); sd = s.e_save.text()
        mz = s.oz1.value()
        xz = s.oz2.value()
        s._log(
            f"Generating Step 2 map from the {src_label}: {os.path.basename(p)}",
            "info",
        )
        w = MapBuildW(p, sd, mz, xz, s.t_slope.value(), s.t_step.value(), s.t_rough.value())
        w.log.connect(s._log); w.prog.connect(s.prog.setValue)
        def done(ok, msg):
            s.b4.setEnabled(True)
            if ok:
                pgm = os.path.join(sd, "map.pgm"); s.e_pgm.setText(pgm)
                yml = os.path.join(sd, "map.yaml"); s.e_yaml.setText(yml)
                s._log(f"Map: {pgm}", "success")
                if os.path.isfile(pgm): s._load_map(pgm)
        w.done.connect(done); s._wk.append(w); w.start()

    # ── Traversability Map Editing ──
    def _toggle_edit_mode(s, mode):
        """Activate draw/erase editing on the traversable ground map."""
        pgm = s.e_pgm.text()
        if not pgm:
            pgm = os.path.join(s.e_save.text(), "map.pgm")
        trav_path = traversability_sidecar_path(pgm)
        if not os.path.isfile(trav_path):
            QMessageBox.warning(s, "No Traversability Map",
                                "Generate a 2D map first (Step 2) to create a traversability sidecar.")
            s.btn_edit_draw.setChecked(False)
            s.btn_edit_erase.setChecked(False)
            return
        # Switch to Traversable Ground view
        s._switch_view("Traversable Ground")
        # Load traversability PGM as overlay if not already editing
        if not s.mw._edit_active:
            w, h, pixels = parse_pgm(trav_path)
            overlay = pixels.reshape(h, w)
            s.mw.enable_edit(overlay)
            # Set obstacle map as reference overlay for comparison
            if BLOCKED_MAP_VIEW in s._imgs:
                s.mw.set_reference_overlay(s._imgs[BLOCKED_MAP_VIEW])
            s._log("Edit mode enabled — draw or erase on the traversable ground map.", "info")
        # Set draw/erase mode
        s.mw.set_edit_mode(mode)
        s.btn_edit_draw.setChecked(mode == "draw")
        s.btn_edit_erase.setChecked(mode == "erase")

    def _get_map_resolution(s):
        """Return the map resolution (m/px) from the loaded YAML, or a default."""
        y = s.e_yaml.text()
        if y and os.path.isfile(y):
            try:
                yd = parse_yaml(y)
                return float(yd['resolution'])
            except Exception:
                pass
        return 0.05  # default fallback

    def _update_brush_size_px(s, meters):
        """Convert brush radius from metres to pixels and update MapW."""
        res = s._get_map_resolution()
        px = max(1, int(round(meters / res)))
        s.mw.set_brush_size(px)

    def _update_brush_rect_px(s, _=None):
        """Convert rectangle W/H from metres to pixels and update MapW."""
        res = s._get_map_resolution()
        hw = max(1, int(round(s.edit_brush_w.value() / res)))
        hh = max(1, int(round(s.edit_brush_h.value() / res)))
        s.mw.set_brush_rect_size(hw, hh)

    def _on_brush_shape_changed(s, text):
        """Show/hide the appropriate size controls based on brush shape."""
        s.mw.set_brush_shape("free" if text == "Free Draw" else text.lower())
        is_rect = (text == "Rectangle")
        # Show rectangle W/H controls
        s._brush_w_label.setVisible(is_rect)
        s.edit_brush_w.setVisible(is_rect)
        s._brush_h_label.setVisible(is_rect)
        s.edit_brush_h.setVisible(is_rect)
        # Show circle/free radius control
        s._brush_size_label.setVisible(not is_rect)
        s.edit_brush_size.setVisible(not is_rect)

    def _apply_trav_edit(s):
        """Write the edited overlay back to the traversability PGM file."""
        if not s.mw._edit_active or s.mw._edit_overlay is None:
            QMessageBox.information(s, "Nothing to Apply", "Enable edit mode first (Draw or Erase).")
            return
        pgm = s.e_pgm.text()
        if not pgm:
            pgm = os.path.join(s.e_save.text(), "map.pgm")
        trav_path = traversability_sidecar_path(pgm)
        overlay = s.mw.get_edit_overlay()
        h, w = overlay.shape
        # Write as PGM P5
        with open(trav_path, 'wb') as f:
            f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
            f.write(overlay.tobytes())
        s._log(f"Traversability map saved: {trav_path}", "success")
        # Disable edit mode and clear reference overlay
        s.mw.disable_edit()
        s.mw.set_reference_overlay(None)
        s.mw.set_reference_overlay_visible(False)
        s.btn_edit_draw.setChecked(False)
        s.btn_edit_erase.setChecked(False)
        if hasattr(s, 'edit_ref_overlay'):
            s.edit_ref_overlay.setChecked(False)
        # Reload sidecar images so the view reflects the saved state
        s._load_map_sidecars(pgm)
        s._switch_view("Traversable Ground")

    def _revert_trav_edit(s):
        """Discard edits and reload the original traversability PGM from disk."""
        s.mw.disable_edit()
        s.mw.set_reference_overlay(None)
        s.mw.set_reference_overlay_visible(False)
        s.btn_edit_draw.setChecked(False)
        s.btn_edit_erase.setChecked(False)
        if hasattr(s, 'edit_ref_overlay'):
            s.edit_ref_overlay.setChecked(False)
        pgm = s.e_pgm.text()
        if not pgm:
            pgm = os.path.join(s.e_save.text(), "map.pgm")
        # Reload from disk
        s._load_map_sidecars(pgm)
        s._switch_view("Traversable Ground")
        s._log("Edit reverted — original traversability map restored.", "info")

    # ── Step 5: Coverage ──
    def _run_ref(s):
        pgm, yml = s._get_pgm()
        if not pgm: return
        if s._map_w == 0: s._load_map(pgm)
        s.bref.setEnabled(False); s._log("Running reference...", "info"); s.prog.setValue(10)
        s.lref_note.setText("")
        params = s._get_params('r')
        sel_mask = s._make_sel_mask()
        trav_sidecar = traversability_sidecar_path(pgm)
        floor_sidecar = floor_sidecar_path(pgm)
        planner = s._use_stc_mode()
        yd = parse_yaml(yml)
        res = yd['resolution']; ox = yd['origin'][0]; oy = yd['origin'][1]
        w, h = s._map_w, s._map_h
        center = s._selection_center_world(yml)
        if center is not None:
            cx, cy = center
        else:
            cx = ox + w * res / 2; cy = oy + h * res / 2

        def _worker():
            try:
                r = run_coverage(
                    pgm,
                    yml,
                    params,
                    cx,
                    cy,
                    sel_mask,
                    "REF",
                    lambda m, c: s.ui_log_sig.emit(m, c),
                    trav_sidecar,
                    floor_sidecar,
                    planner=planner,
                )
                s.ref_result_sig.emit(r, pgm)
            except Exception as e:
                import traceback
                traceback.print_exc()
                s.ref_error_sig.emit(str(e))
        threading.Thread(target=_worker, daemon=True).start()

    def _ref_done(s, r, pgm):
        s.bref.setEnabled(True); s.prog.setValue(100)
        if r:
            r["pgm_path"] = pgm
            s.ref_r = r
            s.lref.setText(f"Ref: {s._result_area(r):.2f} m²")
            s.lref_note.setText(s._coverage_start_note(r))
            qi = render_coverage_fast(r, (0, 180, 255), bg_pgm=getattr(s, '_pgm_pixels', None))
            s._set_img("Reference Coverage", qi)
            s._update_stc_path_view()
            s._check_rii(pgm)
            s._update_semantic_ready_state()
            s._log("Reference accessible area ready.", "success")

    def _ref_failed(s, msg):
        s.bref.setEnabled(True)
        s.prog.setValue(0)
        s._log(f"Reference reachable-area evaluation failed: {msg}", "warn")

    def _run_act(s):
        pgm, yml = s._get_pgm()
        if not pgm: return
        if s._map_w == 0: s._load_map(pgm)
        s.bact.setEnabled(False); s._log("Running actual...", "info"); s.prog.setValue(10)
        s.lact_note.setText("")
        params = s._get_params('a')
        sel_mask = s._make_sel_mask()
        trav_sidecar = traversability_sidecar_path(pgm)
        floor_sidecar = floor_sidecar_path(pgm)
        planner = s._use_stc_mode()

        def _worker():
            try:
                r = run_coverage(
                    pgm,
                    yml,
                    params,
                    0.0,
                    0.0,
                    sel_mask,
                    "ACTUAL",
                    lambda m, c: s.ui_log_sig.emit(m, c),
                    trav_sidecar,
                    floor_sidecar,
                    planner=planner,
                )
                s.act_result_sig.emit(r, pgm)
            except Exception as e:
                import traceback
                traceback.print_exc()
                s.act_error_sig.emit(str(e))
        threading.Thread(target=_worker, daemon=True).start()

    def _act_done(s, r, pgm):
        s.bact.setEnabled(True); s.prog.setValue(100)
        if r:
            r["pgm_path"] = pgm
            s.act_r = r
            s.lact.setText(f"Actual: {s._result_area(r):.2f} m²")
            s.lact_note.setText(s._coverage_start_note(r))
            s._set_img("Actual Coverage", render_coverage_fast(r, (0, 229, 160), bg_pgm=getattr(s, '_pgm_pixels', None)))
            s._update_stc_path_view()
            s._check_rii(pgm)
            s._update_semantic_ready_state()
            s._log("Actual accessible area ready.", "success")

    def _act_failed(s, msg):
        s.bact.setEnabled(True)
        s.prog.setValue(0)
        s._log(f"Actual accessibility evaluation failed: {msg}", "warn")

    def _check_rii(s, pgm):
        if not s.act_r:
            return
        aa = s._result_area(s.act_r)
        floor_area = s._result_floor_area(s.act_r)
        rii = (aa / floor_area * 100) if floor_area > 0 else 0
        planner_name = s.act_r.get("planner", "")
        mode_label = f"With {planner_name}" if planner_name else "Without Path Planner"
        s.riiv.setText(f"{rii:.1f}%")
        s.riis.setText(f"{aa:.2f} / {floor_area:.2f} m² × 100  |  {mode_label}")
        s.riif.show()
        s._log(f"★ RII Horizontal = {rii:.1f}%", "gold")
        if s.ref_r and s._results_share_map():
            s._set_img("Compare", render_compare_fast(s.ref_r, s.act_r, bg_pgm=getattr(s, '_pgm_pixels', None)))
        elif s.ref_r:
            s._set_img("Compare", make_info_image("Reference and Actual are from different maps.\nRerun both Step 3 evaluations on the current map."))
            s._log("Compare view skipped because Reference and Actual are from different maps.", "warn")
        if s.ref_r and s._sem_pts is not None and s._sem_labels is not None:
            s._invalidate_semantic_state(
                keep_loaded_cloud=True,
                candidate_message="Step 3 results changed. Click 'Analyze RII_Horizontal' to re-run semantic analysis.",
                clear_progress=True,
            )
            s._log("Step 3 results changed — semantic analysis reset. Re-run Step 4 when ready.", "warn")

    # ── Step 6: Semantic Analysis ──
    def _browse_sem_pcd(s):
        f, _ = QFileDialog.getOpenFileName(s, "Labeled Point Cloud", "", "Point Cloud (*.pcd *.ply)")
        if f:
            s.e_sem_pcd.setText(f)
            s._log(f"Semantic point cloud: {f}", "info")
            token = s._invalidate_semantic_state(
                keep_loaded_cloud=False,
                candidate_message="Loading labeled cloud...",
                status_message="Loading...",
                status_color="#4a9eff",
                clear_progress=True,
            )
            s._sem_load_active = True
            s._update_semantic_ready_state()

            def _load():
                try:
                    pts, labels, field_name = load_semantic_pcd(f)
                    if token != s._sem_session_token:
                        return
                    if pts is None:
                        s.sem_load_error_sig.emit(token, "Failed to load point cloud")
                        return
                    s.sem_loaded_sig.emit(token, pts, labels, field_name)
                except Exception as e:
                    if token == s._sem_session_token:
                        s.sem_load_error_sig.emit(token, f"Error: {e}")

            threading.Thread(target=_load, daemon=True).start()

    def _run_semantic_analysis(s):
        if not s.ref_r or not s.act_r:
            QMessageBox.warning(s, "Error", "Run both Reference and Actual RII Horizontal evaluations first (Step 3).")
            return
        if s._sem_load_active:
            QMessageBox.warning(s, "Error", "Wait for the labeled cloud to finish loading first.")
            return
        if s._sem_analysis_active:
            QMessageBox.warning(s, "Error", "Semantic analysis is already running for the current inputs.")
            return
        if s._sem_pts is None:
            QMessageBox.warning(s, "Error", "Load a labeled PCD or PLY file first.")
            return
        if s._sem_labels is None:
            QMessageBox.warning(s, "Error", "No semantic labels found in the point cloud file.\n"
                                "Ensure it has a scalar field (e.g., 'classification') from CloudCompare.")
            return

        _, yml = s._get_pgm()
        if not yml: return
        yd = parse_yaml(yml)

        token = s._invalidate_semantic_state(
            keep_loaded_cloud=True,
            candidate_message="Semantic analysis in progress...",
            clear_progress=True,
        )
        s._sem_analysis_active = True
        s._update_semantic_ready_state()
        s._log("Running semantic analysis...", "info")
        s._sem_progress(token, 0, "Preparing semantic analysis...")
        pts = s._sem_pts
        labels = s._sem_labels
        map_w = s._map_w
        map_h = s._map_h
        ref_r = s.ref_r
        act_r = s.act_r
        bg_pgm = getattr(s, '_pgm_pixels', None)

        def _analyze():
            try:
                def _is_current():
                    return token == s._sem_session_token

                # Project 3D labels to 2D grid
                s.sem_progress_sig.emit(token, 10, "Projecting semantic labels onto the current map...")
                label_grid = project_labels_to_2d_grid(
                    pts, labels, yd, map_w, map_h)
                if not _is_current():
                    return

                # Analyze
                s.sem_progress_sig.emit(token, 30, "Summarizing semantic accessibility gap...")
                analysis = analyze_semantic_rii(ref_r, act_r, label_grid, yd)
                if not _is_current():
                    return
                s.sem_progress_sig.emit(token, 40, "Computing layered semantic RII scenarios...")
                analysis["layered_rii"] = compute_semantic_layered_rii(
                    act_r,
                    label_grid,
                    logf=lambda m, c="": s.ui_log_sig.emit(m, c),
                    progress_cb=lambda done, total, name: s.sem_progress_sig.emit(
                        token,
                        40 + int(round(25 * done / max(total, 1))),
                        f"Computing layered semantic RII ({done}/{total}): {name}",
                    ),
                )
                if not _is_current():
                    return
                analysis["_label_grid"] = label_grid
                s.ui_log_sig.emit("Finding removable-object candidates...", "info")
                s.sem_progress_sig.emit(token, 70, "Finding removable-object candidates...")
                candidates = identify_semantic_removal_candidates(
                    act_r,
                    label_grid,
                    yd,
                    progress_cb=lambda done, total, fixation, label_id: s.sem_progress_sig.emit(
                        token,
                        70 + int(round(20 * done / max(total, 1))),
                        f"Finding removable-object candidates ({done}/{total}) - {fixation} label {label_id}",
                    ),
                )
                if not _is_current():
                    return
                s.ui_log_sig.emit(
                    f"Found {len(candidates)} removable-object candidate(s).",
                    "info" if candidates else "warn",
                )

                # Render semantic view
                s.sem_progress_sig.emit(token, 95, "Rendering semantic candidate view...")
                sem_img = render_semantic_candidates(ref_r, act_r, label_grid, candidates, selected_ids=[], bg_pgm=bg_pgm)
                if not _is_current():
                    return

                s.sem_result_sig.emit(token, analysis, sem_img, candidates)
            except Exception as e:
                import traceback; traceback.print_exc()
                if token == s._sem_session_token:
                    s.sem_error_sig.emit(token, str(e))

        threading.Thread(target=_analyze, daemon=True).start()

    def _show_semantic_3d(s):
        """Show the semantic point cloud in the 3D viewer, colored by label class."""
        if s._sem_pts is None or s._sem_labels is None:
            QMessageBox.warning(s, "Error", "Load a labeled point cloud first.")
            return
        pts = s._sem_pts
        labels = s._sem_labels
        n = pts.shape[0]
        max_points = 2_000_000 if PYQTGRAPH_GL_AVAILABLE else 250_000
        sampled = n > max_points
        if sampled:
            rng = np.random.default_rng(42)
            keep = rng.choice(n, size=max_points, replace=False)
            pts = pts[keep]
            labels = labels[keep]

        colors = np.full((len(pts), 3), 128, dtype=np.uint8)
        legend = []
        for label_id, color in SEMANTIC_3D_COLORS.items():
            mask = labels == label_id
            if np.any(mask):
                colors[mask] = color
                name = SEMANTIC_LABEL_NAMES.get(label_id, f"Label {label_id}")
                legend.append((name, color))

        cloud = {
            "points": np.ascontiguousarray(pts, dtype=np.float32),
            "colors": colors,
            "legend": legend,
            "label": "Semantic Labels (3D)",
            "path": s.e_sem_pcd.text() if hasattr(s, 'e_sem_pcd') else "",
            "total_points": int(s._sem_pts.shape[0]),
            "display_points": int(pts.shape[0]),
            "sampled": sampled,
        }
        s._set_cloud("3D Viewer", cloud)
        s._log("Showing semantic labels in 3D Viewer — colors indicate label class.", "success")

    def _sem_loaded(s, token, pts, labels, field_name):
        if token != s._sem_session_token:
            return
        s._sem_load_active = False
        s._sem_pts = pts
        s._sem_labels = labels
        n = len(pts)
        if labels is not None:
            unique = np.unique(labels)
            msg = f"Loaded {n:,} pts, field='{field_name}', labels: {list(unique)}"
        else:
            msg = f"Loaded {n:,} pts, no label field found"
        s.sem_status.setText(msg)
        s.sem_status.setStyleSheet("color:#00e5a0;font-size:11px")
        s._set_semantic_candidate_placeholder("Run semantic analysis to populate removable-object candidates.")
        s._clear_semantic_progress()
        s._update_semantic_ready_state()
        # Enable 3D semantic view button when labels are available
        if hasattr(s, 'bsem_3d'):
            s.bsem_3d.setEnabled(labels is not None)
        s._log(msg, "success")

    def _sem_load_failed(s, token, msg):
        if token != s._sem_session_token:
            return
        s._sem_load_active = False
        s._sem_pts = None
        s._sem_labels = None
        s.sem_status.setText(msg)
        s.sem_status.setStyleSheet("color:#ff6b4a;font-size:11px")
        s._set_semantic_candidate_placeholder("Load a labeled cloud, then run semantic analysis to populate removable-object candidates.")
        s._clear_semantic_progress()
        s._update_semantic_ready_state()
        s._log(msg, "warn")

    def _sem_progress(s, token, value, message):
        if token != s._sem_session_token:
            return
        if hasattr(s, "sem_prog_lbl"):
            s.sem_prog_lbl.setText(message)
            s.sem_prog_lbl.show()
        if hasattr(s, "sem_prog"):
            s.sem_prog.setValue(max(0, min(100, int(value))))
            s.sem_prog.show()
        if hasattr(s, "prog"):
            s.prog.setValue(max(0, min(100, int(value))))

    def _sem_failed(s, token, msg):
        if token != s._sem_session_token:
            return
        s._sem_analysis_active = False
        s._update_semantic_ready_state()
        s._sem_progress(token, 0, f"Semantic analysis failed: {msg}")
        s._log(f"Semantic error: {msg}", "warn")

    def _selected_semantic_candidate_ids(s):
        ids = []
        if not hasattr(s, "sem_candidate_list"):
            return ids
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(int(item.data(Qt.UserRole)))
        return ids

    def _current_semantic_candidate_filter(s):
        if not hasattr(s, "sem_filter"):
            return None
        return s.sem_filter.currentData()

    def _apply_semantic_candidate_filter(s):
        if not hasattr(s, "sem_candidate_list"):
            return
        wanted = s._current_semantic_candidate_filter()
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            fixation = item.data(Qt.UserRole + 1)
            item.setHidden(bool(wanted and fixation != wanted))
        current = s.sem_candidate_list.currentItem()
        if current is not None and current.isHidden():
            s.sem_candidate_list.setCurrentItem(None)
        s._update_semantic_candidate_status()

    def _select_filtered_semantic_candidates(s):
        if not hasattr(s, "sem_candidate_list"):
            return
        s.sem_candidate_list.blockSignals(True)
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            item.setCheckState(Qt.Checked if not item.isHidden() else Qt.Unchecked)
        s.sem_candidate_list.blockSignals(False)
        s._semantic_candidate_selection_changed()

    def _selected_semantic_fixation_groups(s):
        groups = []
        if hasattr(s, "sem_fix_portable") and s.sem_fix_portable.isChecked():
            groups.append("Portable")
        if hasattr(s, "sem_fix_movable") and s.sem_fix_movable.isChecked():
            groups.append("Movable")
        if hasattr(s, "sem_fix_semi_fixed") and s.sem_fix_semi_fixed.isChecked():
            groups.append("Semi-Fixed")
        return groups

    def _set_semantic_fixation_groups(s, groups, run_recompute=False):
        wanted = set(groups)
        for fixation, cb in (
            ("Portable", getattr(s, "sem_fix_portable", None)),
            ("Movable", getattr(s, "sem_fix_movable", None)),
            ("Semi-Fixed", getattr(s, "sem_fix_semi_fixed", None)),
        ):
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(fixation in wanted)
                cb.blockSignals(False)
        s._update_semantic_fixation_status()
        if run_recompute:
            s._recompute_semantic_fixations()

    def _update_semantic_fixation_status(s):
        if not hasattr(s, "sem_fixation_status"):
            return
        groups = s._selected_semantic_fixation_groups()
        has_analysis = bool(s._sem_analysis and s._sem_analysis.get("layered_rii"))
        visible = has_analysis or bool(groups)
        s.sem_fixation_status.setVisible(visible)
        if not visible:
            return
        if groups:
            s.sem_fixation_status.setText(
                "Selected fixation groups for full-group recompute: "
                + ", ".join(groups)
            )
        else:
            s.sem_fixation_status.setText(
                "Use the fixation controls below to remove all Portable, Movable, or Semi-Fixed obstacles and recompute RII."
            )

    def _update_semantic_layered_status(s):
        layered = s._sem_analysis.get("layered_rii") if s._sem_analysis else None
        s._sem_layered_result = layered
        if not hasattr(s, "sem_layered_status"):
            return
        if not layered or not layered.get("layers"):
            s.sem_layered_status.clear()
            s.sem_layered_status.hide()
            return
        lines = ["Layered RII decomposition"]
        for layer in layered["layers"]:
            excluded = ", ".join(layer["excludedFixations"]) if layer["excludedFixations"] else "None"
            lines.append(
                f"{layer['name']}: {layer['riiHorizontal']:.1f}%  "
                f"(excluded: {excluded}; Δ {layer['deltaPts']:+.1f} pts, {layer['deltaArea']:+.2f} m²)"
            )
        lines.append(
            f"Portable {layered['delta_portable']:+.1f} pts | "
            f"Movable {layered['delta_movable']:+.1f} pts | "
            f"Semi-Fixed {layered['delta_semi_fixed']:+.1f} pts"
        )
        lines.append(
            f"Structural max {layered['rii_structural_max']:.1f}% | "
            f"overall potential {layered['improvement_potential']:+.1f} pts"
        )
        s.sem_layered_status.setPlainText("\n".join(lines))
        s.sem_layered_status.show()
        s._update_semantic_fixation_status()

    def _semantic_candidate_by_id(s, candidate_id):
        if candidate_id is None:
            return None
        return next((c for c in s._sem_candidates if int(c["id"]) == int(candidate_id)), None)

    def _semantic_candidate_bounds_px(s, candidate):
        if not candidate or not s.act_r:
            return None
        w = int(s.act_r["w"])
        h = int(s.act_r["h"])
        flat = np.asarray(candidate.get("indices"), dtype=np.int32)
        if flat.size == 0:
            return None
        rows = flat // w
        cols = flat % w
        disp_rows = h - 1 - rows
        return (
            int(cols.min()),
            int(disp_rows.min()),
            int(cols.max()),
            int(disp_rows.max()),
        )

    def _focus_semantic_candidate(s, candidate_id, switch_view=True):
        candidate = s._semantic_candidate_by_id(candidate_id)
        if candidate is None:
            s._sem_focused_candidate_id = None
            s.mw.clear_focus()
            s._update_semantic_candidate_view()
            return
        already_focused = int(candidate["id"]) == s._sem_focused_candidate_id
        s._sem_focused_candidate_id = int(candidate["id"])
        s._update_semantic_candidate_view()
        if switch_view:
            s._switch_view("Semantic")
        bounds = s._semantic_candidate_bounds_px(candidate)
        if bounds is not None:
            s.mw.focus_rect(
                bounds,
                label=f"#{candidate['id']} {candidate['name']}",
            )
            if not already_focused:
                s._log(
                    f"Focused candidate #{candidate['id']} {candidate['name']} in the Semantic view.",
                    "info",
                )

    def _semantic_candidate_current_changed(s, current, _previous):
        if current is None:
            s._focus_semantic_candidate(None, switch_view=False)
            return
        candidate_id = current.data(Qt.UserRole)
        if candidate_id is None:
            return
        s._focus_semantic_candidate(int(candidate_id), switch_view=True)

    def _populate_semantic_candidates(s, candidates):
        s._sem_candidates = list(candidates)
        s._sem_focused_candidate_id = None
        s.mw.clear_focus()
        if hasattr(s, "sem_filter"):
            s.sem_filter.blockSignals(True)
            s.sem_filter.setCurrentIndex(0)
            s.sem_filter.blockSignals(False)
        s.sem_candidate_list.blockSignals(True)
        s.sem_candidate_list.clear()
        if candidates:
            for candidate in candidates:
                x0, x1, y0, y1 = candidate["bboxWorld"]
                text = (
                    f"#{candidate['id']} {candidate['name']} [{candidate['fixation']}]  "
                    f"unlock≈{candidate['potentialUnlockArea']:.2f} m²  "
                    f"object≈{candidate['area']:.2f} m²  "
                    f"bbox=({x0:.1f},{y0:.1f})→({x1:.1f},{y1:.1f})"
                )
                item = QListWidgetItem(text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, int(candidate["id"]))
                item.setData(Qt.UserRole + 1, candidate["fixation"])
                item.setToolTip(candidate["recommendation"])
                s.sem_candidate_list.addItem(item)
        else:
            placeholder = QListWidgetItem("No removable candidates were found for the current semantic map.")
            placeholder.setFlags(Qt.NoItemFlags)
            s.sem_candidate_list.addItem(placeholder)
        s.sem_candidate_list.blockSignals(False)

        has_candidates = bool(candidates)
        s._set_semantic_candidate_controls_enabled(has_candidates)
        s._apply_semantic_candidate_filter()
        s._hide_semantic_whatif_card()
        s._update_semantic_candidate_status()

    def _update_semantic_candidate_status(s, message=None, color="#ffcc66"):
        has_candidates = bool(s._sem_candidates)
        selected = len(s._selected_semantic_candidate_ids())
        visible = 0
        total = 0
        if hasattr(s, "sem_candidate_list"):
            total = s.sem_candidate_list.count()
            for i in range(total):
                if not s.sem_candidate_list.item(i).isHidden():
                    visible += 1
        filter_name = s._current_semantic_candidate_filter()
        filter_text = filter_name if filter_name else "All Fixations"
        if message is not None:
            s.sem_candidate_status.setText(message)
            s.sem_candidate_status.setStyleSheet(f"color:{color};font-size:11px")
            s.sem_candidate_status.setVisible(True)
            return
        if has_candidates:
            if selected:
                s.sem_candidate_status.setText(
                    f"{selected} candidate(s) selected. Showing {visible}/{total} under {filter_text}. "
                    "Click 'Recompute Optimised RII' to estimate the updated score. "
                    "Use the filter or the fixation buttons below to select all candidate objects in a fixation group."
                )
                s.sem_candidate_status.setStyleSheet("color:#7ad9ff;font-size:11px")
            else:
                s.sem_candidate_status.setText(
                    f"{visible}/{total} removable-object candidates shown under {filter_text}. "
                    "Amber = candidate, pink = selected candidate, cyan = focused candidate. "
                    "Click a row to center it in the Semantic tab. "
                    "Use the filter or the fixation buttons below to select all candidate objects in a fixation group."
                )
                s.sem_candidate_status.setStyleSheet("color:#ffcc66;font-size:11px")
        else:
            s.sem_candidate_status.setText(
                "No removable Portable / Movable / Semi-Fixed object components are currently unlocking additional floor area."
            )
            s.sem_candidate_status.setStyleSheet("color:#6b7a8d;font-size:11px")
        s.sem_candidate_status.setVisible(True)

    def _hide_semantic_whatif_card(s):
        if hasattr(s, "sem_cur_val"):
            s.sem_cur_val.setText("—")
        if hasattr(s, "sem_riiv"):
            s.sem_riiv.setText("—")
        if hasattr(s, "sem_delta"):
            s.sem_delta.clear()
        if hasattr(s, "sem_riis"):
            s.sem_riis.clear()
        if hasattr(s, "sem_riif"):
            s.sem_riif.hide()

    def _set_semantic_candidate_controls_enabled(s, enabled):
        for widget_name in (
            "sem_filter",
            "bsem_select_filtered",
            "sem_candidate_list",
            "bsem_select_portable",
            "bsem_select_movable",
            "bsem_select_semi_fixed",
            "bsem_select_portable_movable",
            "bsem_select_all_candidates",
            "bsem_clear_candidates",
            "bsem_recompute",
        ):
            widget = getattr(s, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _update_semantic_candidate_view(s):
        if not (s.ref_r and s.act_r and s._label_grid is not None):
            return
        qi = render_semantic_candidates(
            s.ref_r,
            s.act_r,
            s._label_grid,
            s._sem_candidates,
            selected_ids=s._selected_semantic_candidate_ids(),
            focused_id=s._sem_focused_candidate_id,
            bg_pgm=getattr(s, '_pgm_pixels', None),
        )
        s._imgs["Semantic"] = qi
        if s._is_view_active("Semantic"):
            s._switch_view("Semantic")
        # Also update 3D viewer with semantic overlay
        s._update_semantic_3d_view()

    def _update_semantic_3d_view(s):
        """Build a 3D point cloud colorized with semantic candidate highlights."""
        if not hasattr(s, '_sem_pts') or s._sem_pts is None or len(s._sem_candidates) == 0:
            return
        pts = s._sem_pts
        labels = s._sem_labels
        max_display = 500_000
        step = max(1, pts.shape[0] // max_display)
        pts_sub = pts[::step]
        labels_sub = labels[::step]
        # Base colors: gray for all points
        colors = np.full((pts_sub.shape[0], 3), 140, dtype=np.uint8)
        selected_ids = s._selected_semantic_candidate_ids()
        focused_id = s._sem_focused_candidate_id
        for cand in s._sem_candidates:
            x0, x1, y0, y1 = cand["bboxWorld"]
            lid = cand["label"]
            mask = (
                (labels_sub == lid) &
                (pts_sub[:, 0] >= x0) & (pts_sub[:, 0] <= x1) &
                (pts_sub[:, 1] >= y0) & (pts_sub[:, 1] <= y1)
            )
            if not mask.any():
                continue
            cid = cand["id"]
            if cid == focused_id:
                colors[mask] = [0, 220, 255]    # cyan = focused
            elif cid in selected_ids:
                colors[mask] = [255, 120, 180]   # pink = selected
            else:
                colors[mask] = SEMANTIC_3D_COLORS.get(lid, (255, 180, 50))
        cloud = {"points": pts_sub, "colors": colors}
        s._clouds["3D Viewer"] = cloud
        if s._is_view_active("3D Viewer"):
            s.pcw.set_cloud(cloud)

    def _semantic_candidate_selection_changed(s, *_):
        s._sem_improved = None
        s._hide_semantic_whatif_card()
        s._update_semantic_candidate_status()
        s._update_semantic_candidate_view()

    def _set_semantic_candidates_by_fixation(s, fixations):
        if not s._sem_candidates:
            return
        if hasattr(s, "sem_filter"):
            target = next(iter(fixations)) if len(fixations) == 1 else None
            idx = s.sem_filter.findData(target)
            if idx >= 0:
                s.sem_filter.blockSignals(True)
                s.sem_filter.setCurrentIndex(idx)
                s.sem_filter.blockSignals(False)
                s._apply_semantic_candidate_filter()
        s.sem_candidate_list.blockSignals(True)
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            fixation = item.data(Qt.UserRole + 1)
            item.setCheckState(Qt.Checked if fixation in fixations else Qt.Unchecked)
        s.sem_candidate_list.blockSignals(False)
        s._semantic_candidate_selection_changed()

    def _select_semantic_candidates_portable(s):
        s._set_semantic_candidates_by_fixation({"Portable", "Movable"})

    def _clear_semantic_candidates(s):
        if not s._sem_candidates:
            return
        s.sem_candidate_list.blockSignals(True)
        for i in range(s.sem_candidate_list.count()):
            s.sem_candidate_list.item(i).setCheckState(Qt.Unchecked)
        s.sem_candidate_list.blockSignals(False)
        s._semantic_candidate_selection_changed()

    def _recompute_semantic_improvement(s):
        if s._sem_load_active or s._sem_analysis_active:
            QMessageBox.warning(s, "Error", "Wait for semantic analysis to finish before recomputing the Optimised RII.")
            return
        if not s.act_r or not s._sem_candidates:
            QMessageBox.warning(s, "Error", "Run Step 4 semantic analysis first.")
            return
        selected_ids = s._selected_semantic_candidate_ids()
        if not selected_ids:
            QMessageBox.warning(s, "Error", "Select one or more removable-object candidates first.")
            return
        s.bsem_recompute.setEnabled(False)
        s._log("Recomputing improved RII Horizontal from selected semantic removals...", "info")
        token = s._sem_session_token

        def _recompute():
            try:
                improved = simulate_removed_candidates(
                    s.act_r,
                    s._sem_candidates,
                    selected_ids,
                    label="IMPROVED",
                    logf=lambda m, c: s.ui_log_sig.emit(m, c),
                )
                if token == s._sem_session_token:
                    s.sem_improved_sig.emit(token, improved)
            except Exception as e:
                import traceback; traceback.print_exc()
                if token == s._sem_session_token:
                    s.sem_improved_error_sig.emit(token, str(e))

        threading.Thread(target=_recompute, daemon=True).start()

    def _recompute_semantic_fixations(s):
        if not s.act_r or s._label_grid is None:
            QMessageBox.warning(s, "Error", "Run Step 4 semantic analysis first.")
            return
        fixations = s._selected_semantic_fixation_groups()
        if not fixations:
            QMessageBox.warning(s, "Error", "Select one or more fixation groups first.")
            return
        if hasattr(s, "bsem_fix_recompute"):
            s.bsem_fix_recompute.setEnabled(False)
        s._log(
            "Recomputing RII from fixation groups: " + ", ".join(fixations),
            "info",
        )
        token = s._sem_session_token

        def _recompute():
            try:
                improved = simulate_removed_fixations(
                    s.act_r,
                    s._label_grid,
                    fixations,
                    label="FIXATION",
                    logf=lambda m, c: s.ui_log_sig.emit(m, c),
                )
                if token == s._sem_session_token:
                    s.sem_improved_sig.emit(token, improved)
            except Exception as e:
                import traceback; traceback.print_exc()
                if token == s._sem_session_token:
                    s.sem_improved_error_sig.emit(token, str(e))

        threading.Thread(target=_recompute, daemon=True).start()

    def _sem_improved_done(s, token, improved):
        if token != s._sem_session_token:
            return
        s.bsem_recompute.setEnabled(True)
        if hasattr(s, "bsem_fix_recompute"):
            s.bsem_fix_recompute.setEnabled(True)
        s._sem_improved = improved
        current_rii = (s._result_area(s.act_r) / max(s._result_floor_area(s.act_r), 1e-9)) * 100.0
        improved_rii = (s._result_area(improved) / max(s._result_floor_area(improved), 1e-9)) * 100.0
        gain = improved_rii - current_rii
        area_gain = s._result_area(improved) - s._result_area(s.act_r)
        if improved.get("removedMode") == "fixation":
            groups = ", ".join(improved.get("excludedFixations", [])) or "None"
            scenario = f"{groups} removed"
        else:
            ids = s._selected_semantic_candidate_ids()
            scenario = f"{len(ids)} selected object(s) removed"
        floor_area = s._result_floor_area(improved)
        aa = s._result_area(improved)
        s.sem_cur_val.setText(f"{current_rii:.1f}%")
        s.sem_riiv.setText(f"{improved_rii:.1f}%")
        delta_color = "#00e5a0" if gain >= 0 else "#ff6b4a"
        s.sem_delta.setText(f"{gain:+.1f} pts")
        s.sem_delta.setStyleSheet(f"color:{delta_color};font-size:14px;font-weight:bold")
        s.sem_riis.setText(
            f"{scenario}  |  {area_gain:+.2f} m² ({aa:.2f} / {floor_area:.2f} m²)"
        )
        s.sem_riif.show()
        s._update_semantic_candidate_status(
            f"Optimised RII recompute complete. Removed {improved.get('removedArea', 0.0):.2f} m² of blocked footprint.",
            color="#00e5a0",
        )
        s._log(
            f"Optimised RII Horizontal = {improved_rii:.1f}% "
            f"(+{gain:.1f} pts, +{area_gain:.2f} m² accessible)",
            "success",
        )

    def _sem_improved_failed(s, token, msg):
        if token != s._sem_session_token:
            return
        s.bsem_recompute.setEnabled(True)
        if hasattr(s, "bsem_fix_recompute"):
            s.bsem_fix_recompute.setEnabled(True)
        s._hide_semantic_whatif_card()
        s._update_semantic_candidate_status(f"Optimised RII recompute failed: {msg}", color="#ff6b4a")
        s._log(f"Optimised RII recompute failed: {msg}", "warn")

    def _sem_done(s, token, analysis, sem_img, candidates):
        if token != s._sem_session_token:
            return
        s._sem_analysis_active = False
        s._label_grid = analysis.pop("_label_grid", None)
        s._sem_progress(token, 100, f"Semantic analysis complete. {len(candidates)} removable-object candidate(s) ready.")
        s._set_img("Semantic", sem_img)
        s._sem_analysis = analysis
        s._populate_semantic_candidates(candidates)
        s._update_semantic_layered_status()
        s._hide_semantic_whatif_card()
        s._update_semantic_ready_state()

        # Build report HTML
        html = '<b style="color:#ff66cc;font-size:14px">RII Horizontal — Gap Analysis</b><br><br>'

        floor_area = s._result_floor_area(s.act_r)
        aa = s._result_area(s.act_r)
        rii = (aa / floor_area * 100) if floor_area > 0 else 0
        html += f'<b>RII Horizontal:</b> <span style="color:#ffd700">{rii:.1f}%</span><br>'
        html += f'<b>Total missed area:</b> {analysis["total_missed_area"]:.2f} m²<br><br>'

        if analysis.get('fixation_breakdown'):
            html += '<b>Gap by fixation group:</b><br>'
            for item in analysis['fixation_breakdown']:
                html += (
                    f'<span style="color:#c5cdd8">{item["fixation"]}: '
                    f'{item["area"]:.2f} m² ({item["pct"]:.1f}%)</span><br>'
                )
            html += '<br>'

        layered = analysis.get("layered_rii")
        if layered and layered.get("layers"):
            html += '<b style="color:#7ad9ff">Layered RII decomposition:</b><br>'
            html += '<table style="width:100%;font-size:11px;border-collapse:collapse">'
            html += '<tr style="color:#6b7a8d"><td>Scenario</td><td>Excluded</td><td>RII_H</td><td>Δ pts</td><td>Δ area</td></tr>'
            for layer in layered["layers"]:
                excluded = ", ".join(layer["excludedFixations"]) if layer["excludedFixations"] else "None"
                color = "#7ad9ff" if layer["layer"] == len(layered["layers"]) - 1 else "#c5cdd8"
                html += (
                    f'<tr style="color:{color}">'
                    f'<td>{layer["name"]}</td>'
                    f'<td>{excluded}</td>'
                    f'<td>{layer["riiHorizontal"]:.1f}%</td>'
                    f'<td>{layer["deltaPts"]:+.1f}</td>'
                    f'<td>{layer["deltaArea"]:+.2f} m²</td>'
                    f'</tr>'
                )
            html += '</table><br>'
            html += (
                f'<span style="color:#c5cdd8">Portable contribution: '
                f'<span style="color:#ffd700">{layered["delta_portable"]:+.1f} pts</span> '
                f'({layered["delta_portable_area"]:+.2f} m²)</span><br>'
            )
            html += (
                f'<span style="color:#c5cdd8">Movable contribution: '
                f'<span style="color:#ffd700">{layered["delta_movable"]:+.1f} pts</span> '
                f'({layered["delta_movable_area"]:+.2f} m²)</span><br>'
            )
            html += (
                f'<span style="color:#c5cdd8">Semi-fixed contribution: '
                f'<span style="color:#ffd700">{layered["delta_semi_fixed"]:+.1f} pts</span> '
                f'({layered["delta_semi_fixed_area"]:+.2f} m²)</span><br>'
            )
            html += (
                f'<span style="color:#c5cdd8">Structural maximum: '
                f'<span style="color:#7ad9ff">{layered["rii_structural_max"]:.1f}%</span> '
                f'(overall improvement potential {layered["improvement_potential"]:+.1f} pts)</span><br><br>'
            )

        html += '<b>Accessibility gap by surface type:</b><br>'
        html += '<table style="width:100%;font-size:11px;border-collapse:collapse">'
        html += '<tr style="color:#6b7a8d"><td>Label</td><td>Fixation</td><td>Area (m²)</td><td>% of Gap</td></tr>'

        for b in analysis['label_breakdown']:
            color = "#c5cdd8"
            if b['pct'] > 20: color = "#ff6b4a"
            elif b['pct'] > 10: color = "#ffd700"
            elif b['pct'] > 5: color = "#ff9940"
            html += (f'<tr style="color:{color}">'
                     f'<td>{b["label"]}: {b["name"]}</td>'
                     f'<td>{b["fixation"]}</td>'
                     f'<td>{b["area"]:.2f}</td>'
                     f'<td>{b["pct"]:.1f}%</td></tr>')

        html += '</table><br>'

        if analysis['top_recommendations']:
            html += '<b style="color:#00e5a0">Top Recommendations:</b><br>'
            for rec in analysis['top_recommendations']:
                html += f'<span style="color:#c5cdd8">{rec}</span><br>'

        if candidates:
            html += '<br><b style="color:#ffcc66">Removable-object candidates:</b><br>'
            for candidate in candidates[:6]:
                html += (
                    f'<span style="color:#c5cdd8">#{candidate["id"]} {candidate["name"]} '
                    f'[{candidate["fixation"]}] unlock≈{candidate["potentialUnlockArea"]:.2f} m², '
                    f'object≈{candidate["area"]:.2f} m²</span><br>'
                )

        s.sem_report.setHtml(html)
        s.sem_report.show()
        if layered:
            s._log(
                "Layered semantic RII: "
                f"portable {layered['delta_portable']:+.1f} pts, "
                f"movable {layered['delta_movable']:+.1f} pts, "
                f"semi-fixed {layered['delta_semi_fixed']:+.1f} pts, "
                f"structural max {layered['rii_structural_max']:.1f}%.",
                "info",
            )
        s._log(f"Semantic analysis complete — {len(analysis['label_breakdown'])} categories in gap", "success")

    # ════ RII Vertical — Wall Segments ════
    def _detect_wall_segments(s):
        if s._sem_pts is None or s._sem_labels is None:
            QMessageBox.warning(s, "Error", "Load a labelled point cloud first (Step 4).")
            return
        try:
            wall_ids = {int(x.strip()) for x in s.rv_wall_ids.text().split(",") if x.strip()}
        except ValueError:
            QMessageBox.warning(s, "Error", "Invalid wall label IDs.")
            return
        if not wall_ids:
            QMessageBox.warning(s, "Error", "Wall label IDs cannot be empty.")
            return

        s._log("Detecting wall segments...", "info")
        s._rv_wall_segments = identify_wall_segments(
            s._sem_pts, s._sem_labels, wall_label_ids=wall_ids,
            voxel_size=0.20, min_area_m2=0.5,
        )
        s._rv_focused_wall_id = None
        s.rv_wall_list.blockSignals(True)
        s.rv_wall_list.clear()
        for seg in s._rv_wall_segments:
            text = (f"Wall #{seg['id']}  area={seg['area_m2']:.1f}m²  "
                    f"h={seg['height_span']:.1f}m  w={seg['width_span']:.1f}m  "
                    f"pts={seg['num_points']}")
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, seg["id"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            s.rv_wall_list.addItem(item)
        s.rv_wall_list.blockSignals(False)
        s.rv_wall_status.setText(f"{len(s._rv_wall_segments)} wall segment(s) found. Click to view in 3D.")
        s._log(f"Found {len(s._rv_wall_segments)} wall segments.", "success")
        # Show the full wall cloud in 3D
        s._rv_update_3d_wall_view()

    def _rv_wall_current_changed(s, current, _previous):
        if current is None:
            s._rv_focused_wall_id = None
            s._rv_update_3d_wall_view()
            return
        wall_id = current.data(Qt.UserRole)
        if wall_id is None:
            return
        s._rv_focused_wall_id = int(wall_id)
        s._rv_update_3d_wall_view()
        s._log(f"Focused wall #{wall_id} in 3D viewer.", "info")

    def _rv_wall_check_changed(s, item):
        # Update the 3D view to reflect selection state
        s._rv_update_3d_wall_view()

    def _rv_wall_select_all(s, checked):
        s.rv_wall_list.blockSignals(True)
        for i in range(s.rv_wall_list.count()):
            s.rv_wall_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        s.rv_wall_list.blockSignals(False)
        s._rv_update_3d_wall_view()

    def _rv_selected_wall_ids(s):
        """Return set of checked wall segment IDs."""
        ids = set()
        for i in range(s.rv_wall_list.count()):
            item = s.rv_wall_list.item(i)
            if item.checkState() == Qt.Checked:
                wid = item.data(Qt.UserRole)
                if wid is not None:
                    ids.add(int(wid))
        return ids

    def _rv_update_3d_wall_view(s):
        """Show the labelled point cloud in the 3D viewer with walls highlighted."""
        if s._sem_pts is None or s._sem_labels is None or not s._rv_wall_segments:
            return
        try:
            wall_ids = {int(x.strip()) for x in s.rv_wall_ids.text().split(",") if x.strip()}
        except ValueError:
            wall_ids = {1}

        selected = s._rv_selected_wall_ids()
        focused = getattr(s, '_rv_focused_wall_id', None)

        all_pts = s._sem_pts
        all_labels = s._sem_labels
        max_display = 600_000

        # Keep ALL wall points, subsample only non-wall points
        wall_mask = np.isin(all_labels, np.array(sorted(wall_ids), dtype=np.int32))
        wall_indices = np.where(wall_mask)[0]
        nonwall_indices = np.where(~wall_mask)[0]

        if len(wall_indices) + len(nonwall_indices) > max_display and len(nonwall_indices) > 0:
            budget = max(max_display - len(wall_indices), max_display // 4)
            step = max(1, len(nonwall_indices) // budget)
            nonwall_sub = nonwall_indices[::step]
        else:
            nonwall_sub = nonwall_indices

        # Combine: wall points first, then subsampled non-wall
        keep_indices = np.concatenate([wall_indices, nonwall_sub])
        keep_indices.sort()
        pts = all_pts[keep_indices]
        labels = all_labels[keep_indices]

        # Remap segment point_indices to the new array
        idx_remap = np.full(all_pts.shape[0], -1, dtype=np.int64)
        idx_remap[keep_indices] = np.arange(len(keep_indices))

        segments_mapped = []
        for seg in s._rv_wall_segments:
            new_idx = idx_remap[seg["point_indices"]]
            new_idx = new_idx[new_idx >= 0]  # drop any that weren't kept
            if len(new_idx) > 0:
                segments_mapped.append(dict(seg, point_indices=new_idx))

        colors = colorize_cloud_with_walls(
            pts, labels, segments_mapped,
            selected_wall_ids=selected,
            focused_wall_id=focused,
            wall_label_ids=wall_ids,
        )

        cloud = dict(
            points=pts,
            colors=colors,
            label="Wall Segments (3D)",
            path=s.e_sem_pcd.text() if hasattr(s, 'e_sem_pcd') else "",
            total_points=all_pts.shape[0],
            display_points=pts.shape[0],
            sampled=pts.shape[0] < all_pts.shape[0],
        )
        # Show wall segments in the main 3D Viewer tab
        s._clouds["3D Viewer"] = cloud
        if s._is_view_active("3D Viewer"):
            s.pcw.set_cloud(cloud)
        else:
            s._set_cloud("3D Viewer", cloud)

    # ════ RII Vertical — Computation ════
    def _run_rii_vertical(s):
        if not s.act_r:
            QMessageBox.warning(s, "Error", "Run Actual RII Horizontal first (Step 3).")
            return
        if s._sem_pts is None or s._sem_labels is None:
            QMessageBox.warning(s, "Error", "Load a labelled point cloud first (Step 4).")
            return
        if s._rv_active:
            QMessageBox.warning(s, "Error", "RII Vertical is already running.")
            return

        # Parse wall IDs
        try:
            wall_ids = {int(x.strip()) for x in s.rv_wall_ids.text().split(",") if x.strip()}
        except ValueError:
            QMessageBox.warning(s, "Error", "Invalid wall label IDs. Use comma-separated integers (e.g. '1' or '1,7').")
            return
        if not wall_ids:
            QMessageBox.warning(s, "Error", "Wall label IDs cannot be empty.")
            return

        s._rv_active = True
        s.brv.setEnabled(False)
        s.rv_prog.show(); s.rv_prog.setValue(0)
        s.rv_prog_lbl.show(); s.rv_prog_lbl.setText("Starting RII Vertical computation...")
        s.rv_riif.hide(); s.rv_combf.hide()
        s._log("Running RII Vertical (wall reachability via raycasting)...", "info")

        pts = s._sem_pts.copy()
        labels = s._sem_labels.copy()
        act_r = dict(s.act_r)
        # Copy numpy arrays to avoid threading issues
        for k in ("covPx", "floorPx", "blocked", "sourceBlocked"):
            if k in act_r and isinstance(act_r[k], np.ndarray):
                act_r[k] = act_r[k].copy()

        params = dict(
            wall_label_ids=wall_ids,
            voxel_size=s.rv_voxel.value(),
            max_reach=s.rv_reach.value(),
            angle_step_deg=s.rv_angle.value(),
            wall_min_h=s.rv_wall_min_h.value(),
            wall_max_h=s.rv_wall_max_h.value(),
            ground_stride=s.rv_stride.value(),
            max_ground_samples=s.rv_max_samples.value(),
            paint_width=s.rv_paint_w.value(),
            paint_vertical_span=s.rv_paint_vspan.value(),
            sweep_step=s.rv_sweep.value(),
        )
        gamma = s.rv_gamma.value()

        def _compute():
            try:
                rv = compute_rii_vertical(
                    pts, labels, act_r,
                    logf=lambda m, c="": s.ui_log_sig.emit(m, c),
                    progress_cb=lambda v: s.rv_progress_sig.emit(v, ""),
                    **params,
                )
                rii_h = float(act_r.get("riiHorizontal", 0.0))
                combined = compute_combined_rii(rii_h, rv, gamma=gamma)
                rv["combined"] = combined
                s.rv_result_sig.emit(rv)
            except Exception as exc:
                import traceback
                s.rv_error_sig.emit(f"{exc}\n{traceback.format_exc()}")

        threading.Thread(target=_compute, daemon=True).start()

    def _rv_progress(s, value, message):
        s.rv_prog.setValue(value)
        if message:
            s.rv_prog_lbl.setText(message)

    def _show_rv_painted_cloud(s, rv):
        """After RII Vertical, show full point cloud with wall coverage overlay in the Vertical Coverage tab."""
        wall_band = rv.get("wall_band", set())
        painted = rv.get("painted_voxels", set())
        voxel_origin = rv.get("voxel_origin")
        vs = rv.get("voxel_size", 0.05)
        if not wall_band or voxel_origin is None:
            return

        # -- Build wall voxel points with green/red colors --
        wall_pts_list = []
        wall_colors_list = []
        for k in wall_band:
            center = voxel_origin + (np.array(k, dtype=np.float32) + 0.5) * vs
            wall_pts_list.append(center)
            if k in painted:
                wall_colors_list.append([0, 200, 100])   # green = reachable
            else:
                wall_colors_list.append([255, 70, 50])    # red = unreachable
        wall_pts = np.array(wall_pts_list, dtype=np.float32)
        wall_colors = np.array(wall_colors_list, dtype=np.uint8)

        # -- Combine with the full semantic point cloud colored by label --
        if s._sem_pts is not None and s._sem_labels is not None:
            bg_pts = s._sem_pts
            bg_labels = s._sem_labels
            # Exclude wall-label points (they're shown as voxels already)
            wall_label_ids = set()
            for item in s.rv_wall_list.findItems("*", Qt.MatchWildcard):
                if item.checkState() == Qt.Checked:
                    try:
                        wall_label_ids.add(int(item.data(Qt.UserRole)))
                    except Exception:
                        pass
            if not wall_label_ids:
                wall_label_ids = {1}
            non_wall_mask = ~np.isin(bg_labels, np.array(sorted(wall_label_ids), dtype=np.int32))
            bg_pts = bg_pts[non_wall_mask]
            bg_labels = bg_labels[non_wall_mask]
            # Color by semantic label, dimmed to 40% brightness
            bg_colors = np.full((bg_pts.shape[0], 3), 50, dtype=np.uint8)
            legend = [("Reachable wall", (0, 200, 100)), ("Unreachable wall", (255, 70, 50))]
            for label_id, color in SEMANTIC_3D_COLORS.items():
                mask = bg_labels == label_id
                if np.any(mask):
                    bg_colors[mask] = (np.array(color, dtype=np.float32) * 0.4).astype(np.uint8)
                    if label_id not in (0, 1):  # skip unlabelled and wall
                        name = SEMANTIC_LABEL_NAMES.get(label_id, f"Label {label_id}")
                        legend.append((name, color))
            pts = np.concatenate([bg_pts, wall_pts], axis=0)
            colors = np.concatenate([bg_colors, wall_colors], axis=0)
            total = bg_pts.shape[0] + len(wall_band)
        else:
            pts = wall_pts
            colors = wall_colors
            legend = [("Reachable wall", (0, 200, 100)), ("Unreachable wall", (255, 70, 50))]
            total = len(wall_band)

        painted_pct = len(painted) / max(1, len(wall_band)) * 100
        label = (f"Vertical Coverage  |  Wall voxels: {len(wall_band):,}  |  "
                 f"Painted: {len(painted):,} ({painted_pct:.1f}%)")
        cloud = dict(
            points=pts,
            colors=colors,
            legend=legend,
            label=label,
            total_points=total,
            display_points=len(pts),
            sampled=False,
        )
        s._set_cloud("Vertical Coverage", cloud)

    def _rv_done(s, rv):
        s._rv_active = False
        s._rv_result = rv
        s.brv.setEnabled(True)
        s.rv_prog.setValue(100)
        s.rv_prog_lbl.setText("RII Vertical complete.")

        tcr_pct = rv.get("riiVertical", 0.0)
        sc_pct = rv.get("sc", 0.0) * 100.0
        painted = rv.get("painted_area_m2", 0.0)
        total_wall = rv.get("total_wall_area_m2", 0.0)
        rays_w = rv.get("rays_wall", 0)
        rays_o = rv.get("rays_obstacle", 0)
        rays_m = rv.get("rays_miss", 0)

        s.rv_riiv.setText(f"{tcr_pct:.1f}%")
        s.rv_riis.setText(
            f"{painted:.2f} / {total_wall:.2f} m²  |  "
            f"Surface Continuity (SC)={sc_pct:.0f}%  |  "
            f"Rays: {rays_w} wall, {rays_o} obstacle, {rays_m} miss  |  "
            f"{rv.get('ground_samples', 0)} ground samples"
        )
        s.rv_riif.show()

        # Combined card
        comb = rv.get("combined", {})
        s.rv_ch_val.setText(f"{comb.get('rii_h', 0):.1f}%")
        s.rv_cv_val.setText(f"{comb.get('rii_v', 0):.1f}%")
        s.rv_cc_val.setText(f"{comb.get('combined_paint', 0):.1f}%")
        s.rv_comb_detail.setText(
            f"Task Coverage Rate (TCR)={comb.get('tcr', 0):.1f}%  |  "
            f"Operational Efficiency (OE)={comb.get('oe', 0):.1f}%  |  "
            f"Surface Continuity (SC)={comb.get('sc', 0):.1f}%  |  "
            f"γ={comb.get('gamma', 0.5):.2f}  |  "
            f"Weighted avg={comb.get('weighted_avg', 0):.1f}%"
        )
        s.rv_combf.show()
        s._log(f"RII Vertical = {tcr_pct:.1f}%, Combined = {comb.get('combined_paint', 0):.1f}%", "success")
        # Show painted vs unpainted walls in the Vertical Coverage 3D window
        s._show_rv_painted_cloud(rv)

    def _rv_failed(s, msg):
        s._rv_active = False
        s.brv.setEnabled(True)
        s.rv_prog_lbl.setText("RII Vertical failed.")
        s.rv_prog_lbl.setStyleSheet("color:#ff6b4a;font-size:11px")
        s._log(f"RII Vertical failed: {msg}", "warn")

    def closeEvent(s, e):
        for w in s._wk:
            if hasattr(w, 'cancel'): w.cancel()
        for w in s._wk:
            if isinstance(w, QThread) and w.isRunning():
                w.wait(250)
        if s._cache_root and os.path.isdir(s._cache_root):
            shutil.rmtree(s._cache_root, ignore_errors=True)
        e.accept()

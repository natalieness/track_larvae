"""
Simple SLEAP overlay GUI for sideview rig videos.

Dependencies (not vendored):
  - pandas
  - numpy
  - opencv-python
  - matplotlib

Usage:
  python scripts/sideview-rig/sleap_video_gui.py
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(_ROOT_DIR, "data", "behaviour")
DEFAULT_VIDEO = os.path.join(_ROOT_DIR, "data", "behaviour", "GH-2026-03-10_14-42-25_SV35.mp4")


@dataclass
class TrackState:
    track_id: int
    color: str
    last_manual_point: Optional[Tuple[int, float, float]] = None
    prev_manual_point: Optional[Tuple[int, float, float]] = None
    last_centroid: Optional[Tuple[float, float]] = None
    prev_centroid: Optional[Tuple[float, float]] = None
    geom_count: Optional[np.ndarray] = None
    geom_mean: Optional[np.ndarray] = None
    geom_m2: Optional[np.ndarray] = None
    geom_err_count: int = 0
    geom_err_mean: float = 0.0
    geom_err_m2: float = 0.0
    body_frac_count: int = 0
    body_frac_mean: float = 0.0
    body_frac_m2: float = 0.0
    ml_weights: Optional[np.ndarray] = None
    ml_updates: int = 0
    ml_last_margin: Optional[float] = None
    loss_history: List[float] = field(default_factory=list)
    step_history: List[int] = field(default_factory=list)
    step_idx: int = 0
    blocked_frames: set = field(default_factory=set)
    skipped_missing_frames: set = field(default_factory=set)
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=5))


def _find_col(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _maybe_lower(c):
    return c.lower() if isinstance(c, str) else c


def _first_arraylike(series):
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim >= 2:
            return arr
    return None


def _detect_points_column(df: pd.DataFrame):
    candidates = ["points", "pred_points", "predicted_points", "locations", "positions", "xy"]
    for name in candidates:
        col = _find_col(list(df.columns), [name])
        if col is None:
            continue
        arr = _first_arraylike(df[col])
        if arr is not None:
            return col

    for col in df.columns:
        arr = _first_arraylike(df[col])
        if arr is not None:
            return col
    return None


def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize SLEAP predictions to long format:
      frame, track, node, x, y, score
    """
    cols = list(df.columns)
    col_lower = [_maybe_lower(c) for c in cols]

    frame_col = _find_col(cols, ["frame", "frame_idx", "frame_index", "frame_id"])
    if frame_col is None:
        raise ValueError("Could not find a frame column (expected frame/frame_idx).")

    x_col = _find_col(cols, ["x", "x_px", "x_pix", "xpos"])
    y_col = _find_col(cols, ["y", "y_px", "y_pix", "ypos"])
    track_col = _find_col(cols, ["track", "track_id", "instance_id", "trackid"])
    node_col = _find_col(cols, ["node", "node_id", "point_id", "bodypart", "part", "node_name"])
    score_col = _find_col(cols, ["score", "confidence", "prob", "p"])

    if x_col and y_col:
        if track_col:
            instance_id = df[track_col].astype(int)
        else:
            instance_id = df[frame_col].groupby(df[frame_col]).cumcount().astype(int)
        out = pd.DataFrame({
            "frame": df[frame_col].astype(int),
            "x": df[x_col].astype(float),
            "y": df[y_col].astype(float),
            "instance_id": instance_id,
        })
        if track_col:
            out["track"] = df[track_col].astype(int)
        else:
            out["track"] = 0
        if node_col:
            out["node"] = df[node_col].astype(str)
        else:
            out["node"] = "point"
        if score_col:
            out["score"] = df[score_col].astype(float)
        else:
            out["score"] = np.nan
        return out

    # Attempt array-column format: per-row points array
    points_col = _detect_points_column(df)
    scores_col = _find_col(cols, ["scores", "confidence", "prob", "p", "point_scores"])
    if points_col is not None:
        instance_id = df[frame_col].groupby(df[frame_col]).cumcount().astype(int)
        records = []
        for i, row in df.iterrows():
            frame_val = int(row[frame_col])
            inst_val = int(instance_id.loc[i])
            pts = np.asarray(row[points_col])
            if pts.ndim == 2:
                if pts.shape[1] != 2 and pts.shape[0] == 2:
                    pts = pts.T
                if pts.shape[1] != 2:
                    continue
                scores = None
                if scores_col is not None:
                    scores = np.asarray(row[scores_col])
                    if scores.ndim == 2 and scores.shape[0] == 1:
                        scores = scores[0]
                for node_idx in range(pts.shape[0]):
                    x_val, y_val = pts[node_idx, 0], pts[node_idx, 1]
                    track_val = int(row[track_col]) if track_col else 0
                    score_val = np.nan
                    if scores is not None and scores.ndim >= 1 and node_idx < scores.shape[-1]:
                        score_val = float(scores[node_idx])
                    records.append({
                        "frame": frame_val,
                        "track": track_val,
                        "instance_id": inst_val,
                        "node": str(node_idx),
                        "x": float(x_val),
                        "y": float(y_val),
                        "score": score_val,
                    })
            elif pts.ndim == 3 and pts.shape[-1] == 2:
                scores = None
                if scores_col is not None:
                    scores = np.asarray(row[scores_col])
                for inst_idx in range(pts.shape[0]):
                    for node_idx in range(pts.shape[1]):
                        x_val, y_val = pts[inst_idx, node_idx, 0], pts[inst_idx, node_idx, 1]
                        track_val = int(row[track_col]) if track_col else inst_idx
                        score_val = np.nan
                        if scores is not None and scores.ndim == 2:
                            if inst_idx < scores.shape[0] and node_idx < scores.shape[1]:
                                score_val = float(scores[inst_idx, node_idx])
                        records.append({
                            "frame": frame_val,
                            "track": track_val,
                            "instance_id": inst_val,
                            "node": str(node_idx),
                            "x": float(x_val),
                            "y": float(y_val),
                            "score": score_val,
                        })
        if records:
            return pd.DataFrame.from_records(records)

    # Attempt wide format: columns like head.x/head.y, head_x/head_y, or x_head/y_head
    pairs = {}
    for col in cols:
        if not isinstance(col, str):
            continue
        lower = col.lower()
        if lower.startswith("x_") and len(lower) > 2:
            node = col[2:]
            pairs.setdefault(node, {})["x"] = col
            continue
        if lower.startswith("y_") and len(lower) > 2:
            node = col[2:]
            pairs.setdefault(node, {})["y"] = col
            continue
        if lower.startswith("score_") and len(lower) > 6:
            node = col[6:]
            pairs.setdefault(node, {})["score"] = col
            continue
        for sep in [".", "_", " "]:
            if lower.endswith(sep + "x"):
                node = col[: -len(sep + "x")]
                pairs.setdefault(node, {})["x"] = col
            elif lower.endswith(sep + "y"):
                node = col[: -len(sep + "y")]
                pairs.setdefault(node, {})["y"] = col

    if not pairs:
        raise ValueError("Could not detect x/y columns, wide-format pairs, or array-column points.")

    track_col = track_col or _find_col(cols, ["track", "track_id", "instance_id", "trackid"])
    score_col = score_col or _find_col(cols, ["score", "confidence", "prob", "p", "instance_score"])

    instance_id = df[frame_col].groupby(df[frame_col]).cumcount().astype(int)
    records = []
    for node, xy in pairs.items():
        if "x" not in xy or "y" not in xy:
            continue
        sub = pd.DataFrame({
            "frame": df[frame_col].astype(int),
            "x": df[xy["x"]].astype(float),
            "y": df[xy["y"]].astype(float),
            "node": str(node),
            "instance_id": instance_id,
        })
        if track_col:
            sub["track"] = df[track_col].astype(int)
        else:
            sub["track"] = 0
        if "score" in xy:
            sub["score"] = df[xy["score"]].astype(float)
        elif score_col:
            sub["score"] = df[score_col].astype(float)
        else:
            sub["score"] = np.nan
        records.append(sub)

    if not records:
        raise ValueError("Wide-format detection found no valid x/y pairs.")

    return pd.concat(records, ignore_index=True)


def _video_for_predictions(pred_path: str) -> str | None:
    base = os.path.basename(pred_path)
    if base.endswith(".predictions.feather"):
        stem = base[: -len(".predictions.feather")]
        folder = os.path.dirname(pred_path)
        candidate = os.path.join(folder, f"{stem}.mp4")
        if os.path.isfile(candidate):
            return candidate
        # Fallback: try any mp4 that starts with the same stem
        for name in sorted(os.listdir(folder)):
            if name.startswith(stem) and name.endswith(".mp4"):
                return os.path.join(folder, name)
    return None


def _default_manual_csv(pred_path: str | None, video_path: str | None) -> str:
    base_dir = None
    stem = None
    for path in [pred_path, video_path]:
        if path:
            base_dir = os.path.dirname(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem and stem.endswith(".predictions"):
                stem = stem[: -len(".predictions")]
            break
    if base_dir is None:
        base_dir = os.getcwd()
    if stem is None:
        stem = "manual_track"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{stem}.manual_track.{ts}.csv")


def _lighten_hex(color: str, amount: float = 0.5) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "#ffffff"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return "#ffffff"
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


class SleapVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SLEAP Video Proofreader")
        self.bg = "#1e1e1e"
        self.fg = "#e6e6e6"
        self.accent = "#2a2a2a"
        self.root.configure(bg=self.bg)
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("default")
        except tk.TclError:
            pass
        self.style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=self.accent,
            background="#4caf50",
        )

        self.pred_path = None
        self.video_path = None
        self.df = None
        self.cap = None
        self.frame_idx = 0
        self.total_frames = None
        self.selected_instance_id = None
        self.selected_node = None
        self.last_click_frame = None
        self.last_click_xy = None
        self.last_selected_xy = None
        self.manual_track = []
        self.last_auto_frame = None
        self.auto_advance = tk.BooleanVar(value=True)
        self.active_learning = tk.BooleanVar(value=True)
        self.conf_threshold = 3.0
        self._updating_scale = False
        self.zoom = 1.0
        self.pan_offset = (0.0, 0.0)
        self._dragging = False
        self._drag_start = None
        self._pan_start = None
        self._drag_moved = False
        self._drag_start_px = None
        self.last_mouse_xy = None
        self.frame_shape = None
        self._timeline_dirty = True
        self._timeline_vline = None
        self.z_hold_active = False
        self.z_hold_delay_ms = 60
        self.z_last_frame = None
        self.geom_pairs = []
        self.geom_index = {}
        self.track_count = 1
        self.active_track = 0
        self.track_colors = [
            "#ff9f1a",
            "#6bc5ff",
            "#6bff95",
            "#c46bff",
            "#ff6bd6",
            "#ffd36b",
            "#6bffea",
            "#ff6b6b",
        ]
        self.tracks: Dict[int, TrackState] = {}
        self.geom_weight = 1.0
        self.geom_threshold = 4.0
        self.geom_adaptive = tk.BooleanVar(value=False)
        self.geom_adapt_k = 2.0
        self.geom_adapt_min = 10
        self.geom_weight_min = 0.1
        self.body_frac_base = 0.1
        self.body_frac_k = 2.5
        self.body_frac_min_count = 5
        self.body_frac_min = 0.2
        self.body_frac_max = 1.5
        self.ml_mode = tk.BooleanVar(value=False)
        self.ml_lr = 0.1
        self.ml_margin = 0.5
        self.ml_min_margin = 0.3
        self.ml_min_updates = 5
        self.motion_err_thresh = 0.1
        self.loss_history = []
        self.step_history = []
        self.step_idx = 0
        self.auto_debug = ""
        self.auto_assigned_last = False
        self.auto_run_active = False
        self.auto_run_target = None
        self.auto_run_on_click = tk.BooleanVar(value=True)
        self.auto_run_delay_ms = 1
        self.auto_run_render_stride = 3
        self.auto_run_debug = ""
        self.auto_run_paused = False
        self.auto_run_track_id = None
        self.display_scale = 0.5
        self.reverse = tk.BooleanVar(value=False)
        self.show_seg_overlay = tk.BooleanVar(value=True)
        self.add_point_mode = False
        self.direction = 1
        self._rebuild_state = None
        self.nav_stride = 1
        self.fast_nav_stride = 5

        self._build_ui()
        self._bind_keys()

        # Try default paths if present
        if os.path.isdir(DEFAULT_DATA_DIR):
            self.pred_path = self._pick_default_feather(DEFAULT_DATA_DIR)
            if self.pred_path:
                self._load_predictions(self.pred_path)
        if self.video_path is None and os.path.isfile(DEFAULT_VIDEO):
            self._load_video(DEFAULT_VIDEO)

        self._ensure_track_count(self.track_count)
        self._render()

    def _build_ui(self):
        toolbar_top = tk.Frame(self.root, bg=self.bg)
        toolbar_top.pack(side=tk.TOP, fill=tk.X)

        toolbar_middle = tk.Frame(self.root, bg=self.bg)
        toolbar_middle.pack(side=tk.TOP, fill=tk.X)

        toolbar_bottom = tk.Frame(self.root, bg=self.bg)
        toolbar_bottom.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar_top, text="Load Predictions", command=self._choose_predictions).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Load Video", command=self._choose_video).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Save CSV", command=self._save_manual_csv).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Load CSV", command=self._load_manual_csv).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Warm-start", command=self._warm_start_csv).pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar_top, text="Frame:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(10, 2))
        self.frame_entry = tk.Entry(toolbar_top, width=6, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.frame_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_top, text="Go", command=self._jump_to_frame).pack(side=tk.LEFT, padx=4)

        tk.Button(toolbar_top, text="Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Next", command=self._next_frame).pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar_top, text="Run to:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(6, 2))
        self.run_to_entry = tk.Entry(toolbar_top, width=5, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.run_to_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_top, text="Auto-run", command=self._start_auto_run).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Stop Run", command=self._stop_auto_run).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_top, text="Clear Frame", command=self._clear_frame_labels).pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            toolbar_middle, text="Auto-assign", variable=self.active_learning, bg=self.bg, fg=self.fg, selectcolor=self.accent
        ).pack(side=tk.LEFT, padx=6)
        tk.Checkbutton(
            toolbar_middle, text="ML mode", variable=self.ml_mode, bg=self.bg, fg=self.fg, selectcolor=self.accent
        ).pack(side=tk.LEFT, padx=6)
        tk.Checkbutton(
            toolbar_middle,
            text="Auto-run after click",
            variable=self.auto_run_on_click,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.accent,
        ).pack(side=tk.LEFT, padx=6)
        tk.Checkbutton(
            toolbar_middle,
            text="Auto-advance on click",
            variable=self.auto_advance,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.accent,
        ).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(
            toolbar_middle,
            text="Reverse",
            variable=self.reverse,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.accent,
            command=self._toggle_reverse,
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(toolbar_middle, text="Max jump:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(10, 2))
        self.body_frac_entry = tk.Entry(toolbar_middle, width=5, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.body_frac_entry.insert(0, f"{self.body_frac_base:.2f}")
        self.body_frac_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_middle, text="Set", command=self._set_body_frac_base).pack(side=tk.LEFT, padx=4)
        tk.Label(toolbar_middle, text="Geom thr:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(10, 2))
        self.geom_thr_entry = tk.Entry(toolbar_middle, width=4, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.geom_thr_entry.insert(0, f"{self.geom_threshold:.2f}")
        self.geom_thr_entry.pack(side=tk.LEFT)
        tk.Label(toolbar_middle, text="Geom w:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(6, 2))
        self.geom_w_entry = tk.Entry(toolbar_middle, width=4, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.geom_w_entry.insert(0, f"{self.geom_weight:.2f}")
        self.geom_w_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_middle, text="Set", command=self._set_geom_params).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            toolbar_middle,
            text="Adaptive geom",
            variable=self.geom_adaptive,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.accent,
        ).pack(side=tk.LEFT, padx=6)
        tk.Label(toolbar_middle, text="Stride:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(10, 2))
        self.render_stride_entry = tk.Entry(toolbar_middle, width=4, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.render_stride_entry.insert(0, str(self.auto_run_render_stride))
        self.render_stride_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_middle, text="Set", command=self._set_render_stride).pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar_bottom, text="Fast nav:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(10, 2))
        self.fast_nav_entry = tk.Entry(toolbar_bottom, width=4, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.fast_nav_entry.insert(0, str(self.fast_nav_stride))
        self.fast_nav_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_bottom, text="Set", command=self._set_fast_nav_stride).pack(side=tk.LEFT, padx=4)
        tk.Label(toolbar_bottom, text="Tracks:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(10, 2))
        self.track_count_entry = tk.Entry(toolbar_bottom, width=3, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.track_count_entry.insert(0, str(self.track_count))
        self.track_count_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_bottom, text="Set", command=self._set_track_count).pack(side=tk.LEFT, padx=4)
        tk.Label(toolbar_bottom, text="Active:", bg=self.bg, fg=self.fg).pack(side=tk.LEFT, padx=(6, 2))
        self.active_track_entry = tk.Entry(toolbar_bottom, width=3, bg=self.accent, fg=self.fg, insertbackground=self.fg)
        self.active_track_entry.insert(0, str(self.active_track))
        self.active_track_entry.pack(side=tk.LEFT)
        tk.Button(toolbar_bottom, text="Go", command=self._set_active_track_from_entry).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_bottom, text="Zoom -", command=self._zoom_out).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_bottom, text="Zoom +", command=self._zoom_in).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar_bottom, text="Skip Empty", command=self._skip_to_next_points).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            toolbar_bottom, text="Seg overlay", variable=self.show_seg_overlay,
            bg=self.bg, fg=self.fg, selectcolor=self.accent,
        ).pack(side=tk.LEFT, padx=6)
        self.add_point_btn = tk.Button(
            toolbar_bottom, text="+ Add Point", command=self._toggle_add_point_mode,
            bg=self.bg, fg=self.fg,
        )
        self.add_point_btn.pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(self.root, text="", anchor="w", bg=self.bg, fg=self.fg)
        self.status.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=200,
            mode="determinate",
            variable=self.progress_var,
            style="Dark.Horizontal.TProgressbar",
        )
        self.progress.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        self.frame_scale = tk.Scale(
            self.root,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_scale,
            bg=self.bg,
            fg=self.fg,
            troughcolor=self.accent,
            highlightthickness=0,
        )
        self.frame_scale.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
        self._timeline_ratio = None
        self._build_figure()

    def _bind_keys(self):
        self.root.bind("<Left>", lambda _e: self._prev_frame())
        self.root.bind("<Right>", lambda _e: self._next_frame())
        self.root.bind("-", lambda _e: self._zoom_out())
        self.root.bind("+", lambda _e: self._zoom_in())
        self.root.bind("=", lambda _e: self._zoom_in())
        self.root.bind("s", lambda _e: self._skip_to_next_points())
        self.root.bind("S", lambda _e: self._skip_to_next_points())
        self.root.bind("r", lambda _e: self._start_auto_run())
        self.root.bind("R", lambda _e: self._start_auto_run())
        self.root.bind("<space>", lambda _e: self._stop_auto_run())
        self.root.bind("<Escape>", lambda _e: self._stop_auto_run())
        self.root.bind("a", lambda _e: self._prev_frame())
        self.root.bind("A", lambda _e: self._prev_frame())
        self.root.bind("d", lambda _e: self._next_frame())
        self.root.bind("D", lambda _e: self._next_frame())
        self.root.bind("x", lambda _e: self._clear_frame_labels())
        self.root.bind("X", lambda _e: self._clear_frame_labels())
        self.root.bind("<Shift-Right>", lambda _e: self._next_frame(stride=self.fast_nav_stride))
        self.root.bind("<Shift-Left>", lambda _e: self._prev_frame(stride=self.fast_nav_stride))
        self.root.bind("<Shift-D>", lambda _e: self._next_frame(stride=self.fast_nav_stride))
        self.root.bind("<Shift-A>", lambda _e: self._prev_frame(stride=self.fast_nav_stride))
        self.root.bind("<KeyPress-z>", lambda _e: self._start_z_hold())
        self.root.bind("<KeyPress-Z>", lambda _e: self._start_z_hold())
        self.root.bind("<KeyRelease-z>", lambda _e: self._stop_z_hold())
        self.root.bind("<KeyRelease-Z>", lambda _e: self._stop_z_hold())

    def _timeline_ratio_for_tracks(self, count: int) -> float:
        if count <= 8:
            return 1.0
        if count <= 12:
            return 1.1
        if count <= 18:
            return 1.2
        if count <= 25:
            return 1.3
        return 1.4

    def _mark_timeline_dirty(self):
        self._timeline_dirty = True

    def _build_figure(self):
        if not hasattr(self, "fig"):
            self.fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.bg)
        else:
            self.fig.clear()
        ratio = self._timeline_ratio_for_tracks(self.track_count)
        self._timeline_ratio = ratio
        gs = self.fig.add_gridspec(
            2,
            2,
            height_ratios=[4, ratio],
            width_ratios=[4, 1],
            hspace=0.08,
            wspace=0.08,
        )
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax.set_axis_off()
        self.ax.set_aspect("auto")
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.ax_loss.set_title("Loss", color=self.fg)
        self.ax_loss.set_xlabel("step", color=self.fg)
        self.ax_loss.set_ylabel("loss", color=self.fg)
        self.ax_loss.tick_params(colors=self.fg)
        self.ax_loss.set_facecolor(self.bg)
        self.ax_timeline = self.fig.add_subplot(gs[1, :])
        self.ax_timeline.set_facecolor(self.bg)
        self.ax_timeline.tick_params(colors=self.fg, labelsize=8)
        self._timeline_vline = None
        self._timeline_dirty = True
        if not hasattr(self, "canvas"):
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
            self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
            self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        else:
            self.canvas.draw_idle()

    def _maybe_update_layout(self):
        if not hasattr(self, "fig"):
            return
        ratio = self._timeline_ratio_for_tracks(self.track_count)
        if self._timeline_ratio != ratio:
            self._build_figure()

    def _pick_default_feather(self, directory):
        files = [f for f in os.listdir(directory) if f.endswith(".feather")]
        files.sort()
        if not files:
            return None
        return os.path.join(directory, files[0])

    def _choose_predictions(self):
        path = filedialog.askopenfilename(
            title="Select SLEAP predictions (.feather)",
            initialdir=DEFAULT_DATA_DIR if os.path.isdir(DEFAULT_DATA_DIR) else os.getcwd(),
            filetypes=[("Feather", "*.feather"), ("All files", "*")],
        )
        if path:
            self._load_predictions(path)
            self.frame_idx = 0
            self._render()

    def _choose_video(self):
        path = filedialog.askopenfilename(
            title="Select video (.mp4)",
            initialdir=os.path.dirname(DEFAULT_VIDEO) if os.path.isfile(DEFAULT_VIDEO) else os.getcwd(),
            filetypes=[("Video", "*.mp4"), ("All files", "*")],
        )
        if path:
            self._load_video(path)
            self.frame_idx = 0
            self._render()

    def _save_manual_csv(self):
        if not self.manual_track:
            messagebox.showinfo("No data", "No manual points recorded yet.")
            return
        default_path = _default_manual_csv(self.pred_path, self.video_path)
        path = filedialog.asksaveasfilename(
            title="Save manual track CSV",
            initialdir=os.path.dirname(default_path),
            initialfile=os.path.basename(default_path),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not path:
            return
        save_records = []
        for order_idx, rec in enumerate(self.manual_track):
            out = dict(rec)
            out.setdefault("order", order_idx)
            save_records.append(out)
        df = pd.DataFrame(save_records)
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Saved {len(df)} points to:\\n{path}")

    def _load_manual_csv(self):
        path = filedialog.askopenfilename(
            title="Load manual track CSV",
            initialdir=DEFAULT_DATA_DIR if os.path.isdir(DEFAULT_DATA_DIR) else os.getcwd(),
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Load failed", f"{exc}")
            return
        if "frame" not in df.columns or "x" not in df.columns or "y" not in df.columns:
            messagebox.showerror("Load failed", "CSV must include columns: frame, x, y")
            return

        records = []
        frames_needing = set()
        for order_idx, (_, row) in enumerate(df.iterrows()):
            frame = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            node = str(row["node"]) if "node" in df.columns else "point"
            inst_id = int(row["instance_id"]) if "instance_id" in df.columns and not pd.isna(row["instance_id"]) else None
            track_id = int(row["track_id"]) if "track_id" in df.columns and not pd.isna(row["track_id"]) else self.active_track
            source = str(row["source"]) if "source" in df.columns and not pd.isna(row["source"]) else "load"

            if inst_id is None and self.df is not None:
                frames_needing.add(frame)

            csv_order = int(row["order"]) if "order" in df.columns and not pd.isna(row["order"]) else order_idx
            records.append({
                "frame": frame,
                "x": x,
                "y": y,
                "node": node,
                "instance_id": inst_id,
                "track_id": track_id,
                "source": source,
                "order": csv_order,
            })

        if frames_needing and self.df is not None:
            cache = self._build_frame_points_cache(frames_needing)
            for rec in records:
                if rec["instance_id"] is not None:
                    continue
                frame = rec["frame"]
                if frame not in cache:
                    continue
                pts, inst_ids, nodes = cache[frame]
                if pts.size == 0:
                    continue
                dists = np.linalg.norm(pts - np.array([rec["x"], rec["y"]], dtype=float), axis=1)
                idx = int(np.argmin(dists))
                rec["instance_id"] = int(inst_ids[idx])
                rec["node"] = str(nodes[idx])

        if self.manual_track:
            replace = messagebox.askyesno(
                "Replace existing?",
                "Replace existing manual labels with this CSV? (No = merge)",
            )
        else:
            replace = True

        if replace:
            new_track = records
        else:
            merged = {}
            for m in self.manual_track:
                key = (int(m["frame"]), int(m.get("track_id", 0)), m.get("instance_id"), str(m.get("node")))
                merged[key] = m
            for m in records:
                key = (int(m["frame"]), int(m.get("track_id", 0)), m.get("instance_id"), str(m.get("node")))
                merged[key] = m
            new_track = list(merged.values())

        self.auto_run_debug = "loading csv..."
        self._set_status()
        max_track = max((int(r.get("track_id", 0)) for r in records), default=0)
        self._ensure_track_count(max_track + 1)
        self.root.after(10, lambda: self._finish_load_manual_csv(new_track, path))

    def _finish_load_manual_csv(self, new_track, path):
        self.manual_track = list(new_track)
        self._mark_timeline_dirty()
        def done():
            self._render()
            messagebox.showinfo("Loaded", f"Loaded {len(self.manual_track)} points from:\\n{path}")
        self._start_rebuild_models(self.manual_track, on_done=done, train_ml=self.ml_mode.get())

    def _warm_start_csv(self):
        path = filedialog.askopenfilename(
            title="Warm-start from CSV",
            initialdir=DEFAULT_DATA_DIR if os.path.isdir(DEFAULT_DATA_DIR) else os.getcwd(),
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Warm-start failed", f"{exc}")
            return
        if "frame" not in df.columns or "x" not in df.columns or "y" not in df.columns:
            messagebox.showerror("Warm-start failed", "CSV must include columns: frame, x, y")
            return

        records = []
        frames_needing = set()
        for order_idx, (_, row) in enumerate(df.iterrows()):
            frame = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            node = str(row["node"]) if "node" in df.columns else "point"
            inst_id = int(row["instance_id"]) if "instance_id" in df.columns and not pd.isna(row["instance_id"]) else None
            track_id = int(row["track_id"]) if "track_id" in df.columns and not pd.isna(row["track_id"]) else self.active_track

            if inst_id is None and self.df is not None:
                frames_needing.add(frame)

            csv_order = int(row["order"]) if "order" in df.columns and not pd.isna(row["order"]) else order_idx
            records.append({
                "frame": frame,
                "x": x,
                "y": y,
                "node": node,
                "instance_id": inst_id,
                "track_id": track_id,
                "source": "warmstart",
                "order": csv_order,
            })

        if frames_needing and self.df is not None:
            cache = self._build_frame_points_cache(frames_needing)
            for rec in records:
                if rec["instance_id"] is not None:
                    continue
                frame = rec["frame"]
                if frame not in cache:
                    continue
                pts, inst_ids, nodes = cache[frame]
                if pts.size == 0:
                    continue
                dists = np.linalg.norm(pts - np.array([rec["x"], rec["y"]], dtype=float), axis=1)
                idx = int(np.argmin(dists))
                rec["instance_id"] = int(inst_ids[idx])
                rec["node"] = str(nodes[idx])

        self.auto_run_debug = "warm-start..."
        self._set_status()
        max_track = max((int(r.get("track_id", 0)) for r in records), default=0)
        self._ensure_track_count(max_track + 1)
        self.root.after(10, lambda: self._finish_warm_start(records, path))

    def _finish_warm_start(self, records, path):
        saved_track = self.manual_track
        def done():
            self.manual_track = saved_track
            self._mark_timeline_dirty()
            self._render()
            messagebox.showinfo("Warm-start", f"Warm-started from:\\n{path}")
        self.manual_track = records
        self._start_rebuild_models(self.manual_track, on_done=done, train_ml=self.ml_mode.get())

    def _load_predictions(self, path):
        try:
            df_raw = pd.read_feather(path)
            self.df = normalize_predictions(df_raw)
            self._init_geom_model()
            self._ensure_track_count(self.track_count)
            self.pred_path = path
            auto_vid = _video_for_predictions(path)
            if auto_vid and (self.video_path is None or self.video_path != auto_vid):
                self._load_video(auto_vid)
            self._apply_reverse_start()
            self._set_status()
        except Exception as exc:
            messagebox.showerror("Prediction load failed", f"{exc}")

    def _load_video(self, path):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Video load failed", f"Could not open: {path}")
            return
        self.cap = cap
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total if total > 0 else None
        self.video_path = path
        if self.total_frames is not None:
            self.frame_scale.config(to=max(0, self.total_frames - 1))
        self._apply_reverse_start()
        self._set_status()

    def _set_status(self):
        pred = self.pred_path if self.pred_path else "(no predictions)"
        vid = self.video_path if self.video_path else "(no video)"
        inst = self.selected_instance_id if self.selected_instance_id is not None else "-"
        node = self.selected_node if self.selected_node is not None else "-"
        if self.total_frames is not None and self.total_frames > 0:
            frame_text = f"{self.frame_idx} / {self.total_frames - 1}"
        else:
            frame_text = f"{self.frame_idx}"
        manual_count = len([m for m in self.manual_track if int(m.get("track_id", 0)) == int(self.active_track)])
        self.status.config(
            text=(
                f"Pred: {pred} | Video: {vid} | Frame: {frame_text} | "
                f"Track: {self.active_track}/{self.track_count - 1} | "
                f"Manual pts: {manual_count} | Instance: {inst} | Node: {node} | "
                f"Auto: {self.auto_debug} | Run: {self.auto_run_debug}"
            )
        )

    def _frame_bounds(self):
        max_frame = None
        if self.total_frames is not None and self.total_frames > 0:
            max_frame = self.total_frames - 1
        elif self.df is not None:
            max_frame = int(self.df["frame"].max())
        if max_frame is None:
            max_frame = 0
        return 0, max_frame

    def _apply_reverse_start(self):
        min_f, max_f = self._frame_bounds()
        if self.reverse.get():
            self.direction = -1
            self.frame_idx = max_f
        else:
            self.direction = 1
            self.frame_idx = min_f

    def _toggle_reverse(self):
        self._apply_reverse_start()
        self._render()

    def _jump_to_frame(self):
        try:
            idx = int(self.frame_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid frame", "Frame must be an integer.")
            return
        self.frame_idx = max(0, idx)
        self._render()

    def _prev_frame(self, stride=None):
        stride = self.nav_stride if stride is None else stride
        min_f, max_f = self._frame_bounds()
        self.frame_idx = min(max_f, max(min_f, self.frame_idx - self.direction * stride))
        self._render()

    def _next_frame(self, stride=None):
        stride = self.nav_stride if stride is None else stride
        min_f, max_f = self._frame_bounds()
        self.frame_idx = min(max_f, max(min_f, self.frame_idx + self.direction * stride))
        self._render()

    def _skip_to_next_points(self):
        if self.df is None:
            return
        min_f, max_f = self._frame_bounds()
        start = self.frame_idx + self.direction
        if self.direction > 0:
            rng = range(start, max_f + 1)
        else:
            rng = range(start, min_f - 1, -1)
        missing = []
        next_frame = None
        for f in rng:
            sub = self._frame_points(f)
            if sub is not None and not sub.empty:
                next_frame = f
                break
            missing.append(f)
        if next_frame is not None:
            ts = self.tracks.get(self.active_track)
            if ts is not None and missing:
                ts.skipped_missing_frames.update(missing)
            self.frame_idx = next_frame
            self._render()
            return
        messagebox.showinfo("Skip Empty", "No further frames with valid points.")

    def _start_auto_run(self, start_from_next=True):
        if self.auto_run_active:
            self.auto_run_active = False
            return
        self.auto_run_paused = False
        self.auto_run_track_id = self.active_track
        target = None
        if self.run_to_entry.get().strip():
            try:
                target = int(self.run_to_entry.get().strip())
            except ValueError:
                messagebox.showwarning("Invalid target", "Run-to frame must be an integer.")
                return
        self.auto_run_target = target
        if start_from_next:
            self.frame_idx += self.direction
        self.auto_run_active = True
        self._auto_run_step()

    def _stop_auto_run(self):
        self.auto_run_active = False
        self.auto_run_debug = "stopped(manual)"
        self.auto_run_paused = True
        self.auto_run_track_id = None

    def _clear_frame_labels(self):
        if not self.manual_track:
            return
        before = len(self.manual_track)
        self.manual_track = [
            m for m in self.manual_track
            if not (int(m["frame"]) == int(self.frame_idx) and int(m.get("track_id", 0)) == int(self.active_track))
        ]
        removed = before - len(self.manual_track)
        if removed > 0:
            self._mark_timeline_dirty()
            # Avoid immediate re-auto-assign on the same frame.
            self._render(do_auto=False)

    def _auto_run_step(self):
        if not self.auto_run_active:
            return
        self.auto_run_debug = "running"
        min_f, max_f = self._frame_bounds()
        if self.frame_idx < min_f or self.frame_idx > max_f:
            self.auto_run_active = False
            self.auto_run_debug = "stopped(bounds)"
            return
        if self.auto_run_target is not None:
            if self.direction > 0 and self.frame_idx > self.auto_run_target:
                self.auto_run_active = False
                self.auto_run_debug = "stopped(target)"
                return
            if self.direction < 0 and self.frame_idx < self.auto_run_target:
                self.auto_run_active = False
                self.auto_run_debug = "stopped(target)"
                return
        run_track = self.auto_run_track_id if self.auto_run_track_id is not None else self.active_track
        frame_df = self._frame_points(self.frame_idx)
        self._maybe_auto_assign(frame_df, track_id=run_track)
        if not self.auto_assigned_last:
            # Skip empty or already-labeled frames; stop on low confidence.
            if self.auto_debug in ("no points", "labeled", "already"):
                self.frame_idx += self.direction
                self.root.after(self.auto_run_delay_ms, self._auto_run_step)
                return
            self.auto_run_active = False
            self.auto_run_debug = f"stopped({self.auto_debug})"
            self._render(do_auto=False)
            return
        if self.auto_run_target is not None:
            if self.direction > 0 and self.frame_idx >= self.auto_run_target:
                self.auto_run_active = False
                self.auto_run_debug = "stopped(target)"
                self._render(do_auto=False)
                return
            if self.direction < 0 and self.frame_idx <= self.auto_run_target:
                self.auto_run_active = False
                self.auto_run_debug = "stopped(target)"
                self._render(do_auto=False)
                return
        if self.auto_run_render_stride > 1:
            if self.frame_idx % self.auto_run_render_stride == 0:
                self._render(do_auto=False)
        else:
            self._render(do_auto=False)
        self.frame_idx += self.direction
        self.root.after(self.auto_run_delay_ms, self._auto_run_step)

    def _on_scale(self, value):
        if self._updating_scale:
            return
        try:
            idx = int(float(value))
        except ValueError:
            return
        self.frame_idx = max(0, idx)
        self._render()

    def _zoom_in(self):
        self._set_zoom(min(8.0, self.zoom * 1.25))
        self._render()

    def _zoom_out(self):
        self._set_zoom(max(1.0, self.zoom / 1.25))
        self._render()

    def _set_zoom(self, new_zoom):
        new_zoom = float(new_zoom)
        if self.frame_shape is None or self.last_mouse_xy is None:
            self.zoom = new_zoom
            return
        w = float(self.frame_shape[1])
        h = float(self.frame_shape[0])
        if w <= 0 or h <= 0:
            self.zoom = new_zoom
            return
        cx, cy = w / 2.0, h / 2.0

        # Current view bounds
        width = w / self.zoom
        height = h / self.zoom
        cur_center_x = cx + self.pan_offset[0]
        cur_center_y = cy + self.pan_offset[1]
        x_min = cur_center_x - width / 2.0
        y_min = cur_center_y - height / 2.0

        x0, y0 = self.last_mouse_xy
        if x0 is None or y0 is None:
            self.zoom = new_zoom
            return

        u = (x0 - x_min) / width if width > 0 else 0.5
        v = (y0 - y_min) / height if height > 0 else 0.5
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)

        new_width = w / new_zoom
        new_height = h / new_zoom
        new_x_min = x0 - u * new_width
        new_y_min = y0 - v * new_height
        new_center_x = new_x_min + new_width / 2.0
        new_center_y = new_y_min + new_height / 2.0
        self.pan_offset = (new_center_x - cx, new_center_y - cy)
        self.zoom = new_zoom

    def _view_bounds(self, frame_shape):
        h, w = frame_shape
        cx, cy = w / 2.0, h / 2.0
        width = w / self.zoom
        height = h / self.zoom
        center_x = cx + self.pan_offset[0]
        center_y = cy + self.pan_offset[1]
        if width < w:
            center_x = min(max(center_x, width / 2.0), w - width / 2.0)
        else:
            center_x = cx
        if height < h:
            center_y = min(max(center_y, height / 2.0), h - height / 2.0)
        else:
            center_y = cy
        # Clamp pan offset to bounds so it doesn't drift out of range.
        self.pan_offset = (center_x - cx, center_y - cy)
        x_min = center_x - width / 2.0
        x_max = center_x + width / 2.0
        y_min = center_y - height / 2.0
        y_max = center_y + height / 2.0
        return x_min, x_max, y_min, y_max

    def _init_geom_model(self):
        if self.df is None or "node" not in self.df.columns:
            self.geom_pairs = []
            self.geom_index = {}
            return
        nodes = sorted({str(n) for n in self.df["node"].unique()})
        pairs = []
        index = {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pair = (nodes[i], nodes[j])
                idx = len(pairs)
                pairs.append(pair)
                index[pair] = idx
                index[(pair[1], pair[0])] = idx
        self.geom_pairs = pairs
        self.geom_index = index

    def _new_track_state(self, track_id: int) -> TrackState:
        n = len(self.geom_pairs)
        color = self.track_colors[track_id % len(self.track_colors)]
        return TrackState(
            track_id=track_id,
            color=color,
            geom_count=np.zeros(n, dtype=int),
            geom_mean=np.zeros(n, dtype=float),
            geom_m2=np.zeros(n, dtype=float),
        )

    def _ensure_track_count(self, count: int):
        count = max(1, int(count))
        max_existing = max(self.tracks.keys(), default=-1)
        if max_existing + 1 > count:
            count = max_existing + 1
        self.track_count = count
        for tid in range(count):
            if tid not in self.tracks:
                self.tracks[tid] = self._new_track_state(tid)
            else:
                # Ensure geometry arrays match current node set
                ts = self.tracks[tid]
                if ts.geom_count is None or len(ts.geom_count) != len(self.geom_pairs):
                    self._reset_track_state(ts)
        if self.active_track >= self.track_count:
            self.active_track = 0
        if hasattr(self, "track_count_entry"):
            self.track_count_entry.delete(0, tk.END)
            self.track_count_entry.insert(0, str(self.track_count))
        self._maybe_update_layout()

    def _set_active_track(self, track_id: int):
        if track_id < 0 or track_id >= self.track_count:
            return
        if self.auto_run_active and self.auto_run_track_id is not None and self.auto_run_track_id != track_id:
            self._stop_auto_run()
        self.active_track = track_id
        if hasattr(self, "active_track_entry"):
            self.active_track_entry.delete(0, tk.END)
            self.active_track_entry.insert(0, str(self.active_track))
        self._set_status()
        self._render()

    def _node_map_from_df(self, inst_df):
        node_map = {}
        for _, row in inst_df.iterrows():
            node = str(row["node"])
            x = float(row["x"])
            y = float(row["y"])
            if np.isnan(x) or np.isnan(y):
                continue
            node_map[node] = (x, y)
        return node_map

    def _ordered_nodes(self, nodes):
        def norm(name: str) -> str:
            s = str(name).strip().lower()
            s = s.replace("-", " ").replace("_", " ")
            s = " ".join(s.split())
            return s

        norm_map = {str(n): norm(n) for n in nodes}
        used = set()
        ordered = []

        def pick(exact, contains=()):
            exact_set = {norm(x) for x in exact}
            contains_set = {norm(x) for x in contains}
            exact_hits = [orig for orig, n in norm_map.items() if orig not in used and n in exact_set]
            if exact_hits:
                exact_hits.sort()
                chosen = exact_hits[0]
                ordered.append(chosen)
                used.add(chosen)
                return
            if contains_set:
                contains_hits = [
                    orig for orig, n in norm_map.items()
                    if orig not in used and any(tok in n for tok in contains_set)
                ]
                if contains_hits:
                    contains_hits.sort()
                    chosen = contains_hits[0]
                    ordered.append(chosen)
                    used.add(chosen)

        pick(
            ["head", "hd", "h"],
            contains=["head", "cephalic", "anterior"],
        )
        pick(
            ["mouth hook", "mouthhook", "mouth_hook", "mh", "mouth"],
            contains=["mouth", "hook"],
        )
        pick(
            ["body", "bd", "b"],
            contains=["body", "midbody", "mid body", "mid"],
        )
        pick(
            ["tail", "tl", "t"],
            contains=["tail", "posterior"],
        )
        pick(
            ["spiracle", "spiracles", "posterior spiracle", "posterior spiracles", "sp", "s"],
            contains=["spiracle"],
        )

        return ordered if len(ordered) >= 2 else []

    def _build_frame_points_cache(self, frames):
        if self.df is None or not frames:
            return {}
        sub = self.df[self.df["frame"].isin(frames)]
        cache = {}
        for frame, fdf in sub.groupby("frame"):
            pts = fdf[["x", "y"]].to_numpy(dtype=float)
            inst = fdf["instance_id"].to_numpy(dtype=int)
            nodes = fdf["node"].astype(str).to_numpy()
            cache[int(frame)] = (pts, inst, nodes)
        return cache

    def _build_node_map_cache(self, frames):
        if self.df is None or not frames:
            return {}
        sub = self.df[self.df["frame"].isin(frames)]
        cache = {}
        for (frame, inst_id), inst_df in sub.groupby(["frame", "instance_id"]):
            cache[(int(frame), int(inst_id))] = self._node_map_from_df(inst_df)
        return cache

    def _build_frame_df_cache(self, frames):
        if self.df is None or not frames:
            return {}
        sub = self.df[self.df["frame"].isin(frames)]
        return {int(frame): fdf for frame, fdf in sub.groupby("frame")}

    def _pair_distances(self, node_map):
        if not self.geom_pairs:
            return None
        dists = np.full(len(self.geom_pairs), np.nan, dtype=float)
        for i, (a, b) in enumerate(self.geom_pairs):
            if a in node_map and b in node_map:
                xa, ya = node_map[a]
                xb, yb = node_map[b]
                dists[i] = float(np.hypot(xa - xb, ya - yb))
        return dists

    _GEOM_EMA_ALPHA = 0.98

    def _update_geom_model(self, ts: TrackState, dists):
        if dists is None or ts.geom_count is None:
            return
        alpha = self._GEOM_EMA_ALPHA
        for i, d in enumerate(dists):
            if not np.isfinite(d):
                continue
            if ts.geom_count[i] == 0:
                ts.geom_mean[i] = d
                ts.geom_m2[i] = 0.0
            else:
                old_mean = ts.geom_mean[i]
                ts.geom_mean[i] = alpha * ts.geom_mean[i] + (1.0 - alpha) * d
                ts.geom_m2[i] = alpha * ts.geom_m2[i] + (1.0 - alpha) * (d - old_mean) ** 2
            ts.geom_count[i] += 1

    def _update_geom_err_stats(self, ts: TrackState, geom_err: Optional[float]):
        if geom_err is None or not np.isfinite(geom_err):
            return
        ts.geom_err_count += 1
        delta = geom_err - ts.geom_err_mean
        ts.geom_err_mean += delta / ts.geom_err_count
        delta2 = geom_err - ts.geom_err_mean
        ts.geom_err_m2 += delta * delta2

    def _geom_stats(self, ts: TrackState):
        if ts.geom_err_count < self.geom_adapt_min:
            return None
        var = ts.geom_err_m2 / max(1, ts.geom_err_count - 1)
        std = float(np.sqrt(var)) if var > 1e-6 else 0.0
        return ts.geom_err_mean, std

    def _geom_threshold_value(self, ts: TrackState):
        if not self.geom_adaptive.get():
            return self.geom_threshold
        stats = self._geom_stats(ts)
        if stats is None:
            return self.geom_threshold
        mean, std = stats
        return max(self.geom_threshold, mean + self.geom_adapt_k * std)

    def _geom_weight_value(self, ts: TrackState):
        if not self.geom_adaptive.get():
            return self.geom_weight
        stats = self._geom_stats(ts)
        if stats is None:
            return self.geom_weight
        _, std = stats
        scale = 1.0 / (1.0 + std)
        return max(self.geom_weight_min, self.geom_weight * scale)

    def _geom_error(self, ts: TrackState, dists):
        if dists is None or ts.geom_count is None:
            return None
        errs = []
        for i, d in enumerate(dists):
            if not np.isfinite(d):
                continue
            if ts.geom_count[i] < 2:
                continue
            var = ts.geom_m2[i]  # EMA variance stored directly
            std = np.sqrt(var) if var > 1e-6 else 1.0
            z = (d - ts.geom_mean[i]) / std
            errs.append(z * z)
        if not errs:
            return None
        return float(np.mean(errs))

    def _instance_dists(self, frame_idx, instance_id):
        if self.df is None:
            return None
        inst_df = self.df[
            (self.df["frame"] == frame_idx) & (self.df["instance_id"] == instance_id)
        ]
        if inst_df.empty:
            return None
        node_map = self._node_map_from_df(inst_df)
        return self._pair_distances(node_map)

    def _instance_node_map(self, frame_idx, instance_id):
        if self.df is None:
            return None
        inst_df = self.df[
            (self.df["frame"] == frame_idx) & (self.df["instance_id"] == instance_id)
        ]
        if inst_df.empty:
            return None
        return self._node_map_from_df(inst_df)

    def _body_length_from_nodes(self, node_map):
        if "head" in node_map and "tail" in node_map:
            x1, y1 = node_map["head"]
            x2, y2 = node_map["tail"]
            return float(np.hypot(x1 - x2, y1 - y2))
        if "head" in node_map and "body" in node_map and "tail" in node_map:
            xh, yh = node_map["head"]
            xb, yb = node_map["body"]
            xt, yt = node_map["tail"]
            return float(np.hypot(xh - xb, yh - yb) + np.hypot(xb - xt, yb - yt))
        if "head" in node_map and "body" in node_map:
            xh, yh = node_map["head"]
            xb, yb = node_map["body"]
            return float(np.hypot(xh - xb, yh - yb))
        if "body" in node_map and "tail" in node_map:
            xb, yb = node_map["body"]
            xt, yt = node_map["tail"]
            return float(np.hypot(xb - xt, yb - yt))
        return None

    def _body_length_model(self, ts: TrackState):
        if not self.geom_index or ts.geom_mean is None:
            return None
        ht = self.geom_index.get(("head", "tail"))
        hb = self.geom_index.get(("head", "body"))
        bt = self.geom_index.get(("body", "tail"))
        if ht is not None:
            return float(ts.geom_mean[ht])
        if hb is not None and bt is not None:
            return float(ts.geom_mean[hb] + ts.geom_mean[bt])
        if hb is not None:
            return float(ts.geom_mean[hb])
        if bt is not None:
            return float(ts.geom_mean[bt])
        return None

    def _update_body_frac(self, ts: TrackState, body_frac):
        if body_frac is None or not np.isfinite(body_frac):
            return
        ts.body_frac_count += 1
        delta = body_frac - ts.body_frac_mean
        ts.body_frac_mean += delta / ts.body_frac_count
        delta2 = body_frac - ts.body_frac_mean
        ts.body_frac_m2 += delta * delta2

    def _body_frac_threshold(self, ts: TrackState):
        if ts.body_frac_count < self.body_frac_min_count:
            return self.body_frac_base
        var = ts.body_frac_m2 / max(1, ts.body_frac_count - 1)
        std = np.sqrt(var) if var > 1e-6 else 0.0
        learned = ts.body_frac_mean + self.body_frac_k * std
        w = min(1.0, ts.body_frac_count / 50.0)
        thresh = (1 - w) * self.body_frac_base + w * learned
        return max(self.body_frac_min, min(self.body_frac_max, thresh))

    def _set_body_frac_base(self):
        try:
            val = float(self.body_frac_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Max jump must be a number.")
            return
        if val <= 0:
            messagebox.showwarning("Invalid value", "Max jump must be > 0.")
            return
        self.body_frac_base = val

    def _set_geom_params(self):
        try:
            thr = float(self.geom_thr_entry.get().strip())
            w = float(self.geom_w_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Geom threshold/weight must be numbers.")
            return
        if thr <= 0 or w <= 0:
            messagebox.showwarning("Invalid value", "Geom threshold/weight must be > 0.")
            return
        self.geom_threshold = thr
        self.geom_weight = w
        self._set_status()

    def _set_render_stride(self):
        try:
            val = int(self.render_stride_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Stride must be an integer.")
            return
        if val <= 0:
            messagebox.showwarning("Invalid value", "Stride must be > 0.")
            return
        self.auto_run_render_stride = val

    def _set_fast_nav_stride(self):
        try:
            val = int(self.fast_nav_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Fast nav must be an integer.")
            return
        if val <= 0:
            messagebox.showwarning("Invalid value", "Fast nav must be > 0.")
            return
        self.fast_nav_stride = val

    def _set_track_count(self):
        try:
            val = int(self.track_count_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Tracks must be an integer.")
            return
        if val <= 0:
            messagebox.showwarning("Invalid value", "Tracks must be > 0.")
            return
        self._ensure_track_count(val)
        self.track_count_entry.delete(0, tk.END)
        self.track_count_entry.insert(0, str(self.track_count))
        if self.active_track >= self.track_count:
            self.active_track = 0
        self.active_track_entry.delete(0, tk.END)
        self.active_track_entry.insert(0, str(self.active_track))
        self._set_status()
        self._render()

    def _set_active_track_from_entry(self):
        try:
            val = int(self.active_track_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Active track must be an integer.")
            return
        if val < 0 or val >= self.track_count:
            messagebox.showwarning("Invalid value", "Active track out of range.")
            return
        self._set_active_track(val)

    def _init_ml_model(self, ts: TrackState):
        ts.ml_weights = None
        ts.ml_updates = 0
        ts.ml_last_margin = None

    def _reset_track_state(self, ts: TrackState):
        n = len(self.geom_pairs)
        ts.geom_count = np.zeros(n, dtype=int)
        ts.geom_mean = np.zeros(n, dtype=float)
        ts.geom_m2 = np.zeros(n, dtype=float)
        ts.geom_err_count = 0
        ts.geom_err_mean = 0.0
        ts.geom_err_m2 = 0.0
        ts.body_frac_count = 0
        ts.body_frac_mean = 0.0
        ts.body_frac_m2 = 0.0
        ts.last_manual_point = None
        ts.prev_manual_point = None
        ts.last_centroid = None
        ts.prev_centroid = None
        ts.loss_history = []
        ts.step_history = []
        ts.step_idx = 0
        ts.blocked_frames = set()
        ts.skipped_missing_frames = set()
        ts.centroid_history = deque(maxlen=5)
        self._init_ml_model(ts)

    def _start_rebuild_models(self, records, on_done=None, train_ml=False):
        # Reset models and history, then replay labels in time order in chunks.
        self._init_geom_model()
        max_track = max((int(r.get("track_id", 0)) for r in records), default=0)
        self._ensure_track_count(max_track + 1)
        for ts in self.tracks.values():
            self._reset_track_state(ts)

        if not records:
            if on_done:
                on_done()
            return

        ordered = sorted(records, key=lambda r: (int(r.get("order", 10**12)), int(r.get("track_id", 0))))
        self._rebuild_state = {
            "records": ordered,
            "idx": 0,
            "total": len(ordered),
            "on_done": on_done,
            "train_ml": train_ml,
        }
        self.auto_run_debug = "rebuild 0%"
        if hasattr(self, "progress"):
            self.progress.configure(maximum=max(1, len(ordered)))
            self.progress_var.set(0)
        self._set_status()
        self.root.after(1, self._rebuild_step)

    def _rebuild_step(self):
        state = self._rebuild_state
        if state is None:
            return
        records = state["records"]
        idx = state["idx"]
        total = state["total"]
        train_ml = state["train_ml"]

        chunk = 200
        end = min(total, idx + chunk)
        frames = {int(records[i]["frame"]) for i in range(idx, end) if records[i].get("instance_id") is not None}
        node_cache = self._build_node_map_cache(frames)
        frame_cache = self._build_frame_df_cache(frames)
        for i in range(idx, end):
            m = records[i]
            frame_idx = int(m["frame"])
            inst_id = m.get("instance_id")
            track_id = int(m.get("track_id", 0))
            ts = self.tracks.get(track_id)
            if inst_id is None or ts is None:
                continue
            node_map = node_cache.get((frame_idx, int(inst_id)))
            if node_map is None:
                continue
            dists = self._pair_distances(node_map)
            self._update_geom_model(ts, dists)
            pred = self._predict_position(ts)
            body_len = self._body_length_from_nodes(node_map)
            body_frac = None
            centroid = self._instance_centroid(node_map)
            if pred is not None and body_len is not None and body_len > 0 and centroid is not None:
                body_frac = float(np.hypot(pred[0] - centroid[0], pred[1] - centroid[1]) / body_len)
            self._update_body_frac(ts, body_frac)

            if train_ml:
                frame_df = frame_cache.get(frame_idx)
                if pred is not None:
                    self._update_ml_model(ts, frame_df, inst_id, pred)

            if pred is not None:
                motion_err = float(np.hypot(pred[0] - float(m["x"]), pred[1] - float(m["y"])))
                geom_err = self._geom_error(ts, dists)
                loss = motion_err + (self._geom_weight_value(ts) * geom_err if geom_err is not None else 0.0)
                ts.loss_history.append(loss)
                ts.step_history.append(ts.step_idx)
                ts.step_idx += 1
                self._update_geom_err_stats(ts, geom_err)

            if ts.last_manual_point is not None:
                ts.prev_manual_point = ts.last_manual_point
            ts.last_manual_point = (frame_idx, float(m["x"]), float(m["y"]))
            if centroid is not None:
                if ts.last_centroid is not None:
                    ts.prev_centroid = ts.last_centroid
                ts.last_centroid = centroid
                ts.centroid_history.append(centroid)

        state["idx"] = end
        if total > 0:
            pct = int(100 * end / total)
            self.auto_run_debug = f"rebuild {pct}%"
            if hasattr(self, "progress"):
                self.progress_var.set(end)
            self._set_status()
        if end < total:
            self.root.after(1, self._rebuild_step)
        else:
            self._rebuild_state = None
            self.auto_run_debug = "rebuild done"
            if hasattr(self, "progress"):
                self.progress_var.set(total)
            self._set_status()
            if state["on_done"]:
                state["on_done"]()

    def _instance_centroid(self, node_map):
        if not node_map:
            return None
        xs = [v[0] for v in node_map.values()]
        ys = [v[1] for v in node_map.values()]
        if not xs or not ys:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _feature_vector(self, ts: TrackState, inst_df, target):
        node_map = self._node_map_from_df(inst_df)
        centroid = self._instance_centroid(node_map)
        if centroid is None:
            return None
        body_len = self._body_length_from_nodes(node_map)
        model_body = self._body_length_model(ts)
        if body_len is None or not np.isfinite(body_len):
            body_len = model_body
        if body_len is None or body_len <= 0:
            body_len = 1.0
        dist = float(np.hypot(centroid[0] - target[0], centroid[1] - target[1]))
        body_frac = dist / body_len
        dists = self._pair_distances(node_map)
        geom_err = self._geom_error(ts, dists)
        if geom_err is None:
            geom_err = 0.0

        vel_err = 0.0
        if ts.last_centroid is not None and ts.prev_centroid is not None:
            pred_dx = ts.last_centroid[0] - ts.prev_centroid[0]
            pred_dy = ts.last_centroid[1] - ts.prev_centroid[1]
            cand_dx = centroid[0] - ts.last_centroid[0]
            cand_dy = centroid[1] - ts.last_centroid[1]
            vel_err = float(np.hypot(cand_dx - pred_dx, cand_dy - pred_dy)) / body_len

        len_ratio = 1.0
        if model_body is not None and model_body > 0:
            len_ratio = body_len / model_body

        pts = inst_df[["x", "y"]].to_numpy(dtype=float)
        spread = float(np.median(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        if not np.isfinite(spread) or spread <= 0:
            spread = 1.0
        norm_dist = dist / spread

        mean_score = 0.5
        if "score" in inst_df.columns:
            s = inst_df["score"].dropna()
            if len(s) > 0:
                mean_score = float(s.mean())

        return np.array([1.0, norm_dist, geom_err, body_frac, vel_err, len_ratio, mean_score], dtype=float)

    def _motion_error(self, ts: TrackState, centroid, body_len):
        if centroid is None or body_len is None or body_len <= 0:
            return None
        if ts.last_centroid is None or ts.prev_centroid is None:
            return None
        pred_dx = ts.last_centroid[0] - ts.prev_centroid[0]
        pred_dy = ts.last_centroid[1] - ts.prev_centroid[1]
        cand_dx = centroid[0] - ts.last_centroid[0]
        cand_dy = centroid[1] - ts.last_centroid[1]
        return float(np.hypot(cand_dx - pred_dx, cand_dy - pred_dy)) / body_len

    def _ensure_ml_weights(self, ts: TrackState, dim):
        if ts.ml_weights is None or len(ts.ml_weights) != dim:
            ts.ml_weights = np.zeros(dim, dtype=float)

    def _ml_scores(self, ts: TrackState, frame_df, target):
        feats = {}
        scores = {}
        inst_dfs = {}
        for inst_id, inst_df in frame_df.groupby("instance_id"):
            fv = self._feature_vector(ts, inst_df, target)
            if fv is None:
                continue
            self._ensure_ml_weights(ts, len(fv))
            score = float(np.dot(ts.ml_weights, fv))
            feats[int(inst_id)] = fv
            scores[int(inst_id)] = score
            inst_dfs[int(inst_id)] = inst_df
        return feats, scores, inst_dfs

    def _update_ml_model(self, ts: TrackState, frame_df, chosen_inst_id, target):
        if frame_df is None or frame_df.empty:
            return
        feats, scores, _ = self._ml_scores(ts, frame_df, target)
        if chosen_inst_id not in feats:
            return
        if len(scores) <= 1:
            return
        # Best competing instance
        sorted_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_score = sorted_ids[0]
        if int(best_id) == int(chosen_inst_id):
            neg_id, neg_score = sorted_ids[1]
        else:
            neg_id, neg_score = best_id, best_score
        pos_score = scores[int(chosen_inst_id)]
        margin = pos_score - neg_score
        ts.ml_last_margin = margin
        if margin < self.ml_margin:
            ts.ml_weights += self.ml_lr * (feats[int(chosen_inst_id)] - feats[int(neg_id)])
            ts.ml_weights *= 0.999
            ts.ml_updates += 1

    def _read_frame(self, frame_idx):
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _frame_points(self, frame_idx):
        if self.df is None:
            return None
        sub = self.df[self.df["frame"] == frame_idx]
        if sub.empty:
            return sub
        return sub.dropna(subset=["x", "y"])

    def _frame_has_track_label(self, frame_idx, track_id):
        return any(
            int(m.get("frame")) == int(frame_idx) and int(m.get("track_id", 0)) == int(track_id)
            for m in self.manual_track
        )

    def _used_instances(self, frame_idx, exclude_track=None):
        used = set()
        for m in self.manual_track:
            if int(m.get("frame")) != int(frame_idx):
                continue
            if exclude_track is not None and int(m.get("track_id", 0)) == int(exclude_track):
                continue
            inst = m.get("instance_id")
            if inst is None or (isinstance(inst, float) and np.isnan(inst)):
                continue
            used.add(int(inst))
        return used

    def _nearest_point(self, frame_idx, target_xy, frame_df=None):
        frame_df = frame_df if frame_df is not None else self._frame_points(frame_idx)
        if frame_df is None or frame_df.empty:
            return None
        pts = frame_df[["x", "y"]].to_numpy(dtype=float)
        click = np.array(target_xy, dtype=float)
        dists = np.linalg.norm(pts - click, axis=1)
        idx = int(np.argmin(dists))
        return frame_df.iloc[idx]

    def _predict_position(self, ts: TrackState):
        if ts.last_manual_point is None:
            return None
        last_frame, last_mp_x, last_mp_y = ts.last_manual_point
        if ts.last_centroid is not None:
            last_x, last_y = ts.last_centroid
        else:
            last_x, last_y = last_mp_x, last_mp_y

        hist = ts.centroid_history
        if len(hist) >= 3:
            dxs = [hist[j][0] - hist[j - 1][0] for j in range(1, len(hist))]
            dys = [hist[j][1] - hist[j - 1][1] for j in range(1, len(hist))]
            return (last_x + float(np.median(dxs)), last_y + float(np.median(dys)))

        if ts.prev_centroid is not None:
            return (last_x + (last_x - ts.prev_centroid[0]),
                    last_y + (last_y - ts.prev_centroid[1]))
        if ts.prev_manual_point is not None:
            prev_frame, prev_x, prev_y = ts.prev_manual_point
            dt = last_frame - prev_frame
            if dt <= 0:
                return (last_x, last_y)
            return (last_x + (last_x - prev_x) / dt, last_y + (last_y - prev_y) / dt)
        return (last_x, last_y)

    def _record_manual_point(self, ts: TrackState, track_id: int, frame_idx, row, source):
        if ts.blocked_frames:
            ts.blocked_frames.discard(int(frame_idx))
        pred = self._predict_position(ts)
        inst_id = int(row["instance_id"]) if "instance_id" in row else None
        dists = self._instance_dists(frame_idx, inst_id) if inst_id is not None else None
        geom_err = self._geom_error(ts, dists)
        node_map = self._instance_node_map(frame_idx, inst_id) if inst_id is not None else None
        body_len = self._body_length_from_nodes(node_map) if node_map is not None else None
        centroid = self._instance_centroid(node_map) if node_map is not None else None
        body_frac = None
        if pred is not None and body_len is not None and body_len > 0:
            body_frac = float(np.hypot(pred[0] - float(row["x"]), pred[1] - float(row["y"])) / body_len)
        if pred is not None:
            motion_err = float(np.hypot(pred[0] - float(row["x"]), pred[1] - float(row["y"])))
        else:
            motion_err = None
        if motion_err is not None:
            loss = motion_err + (self._geom_weight_value(ts) * geom_err if geom_err is not None else 0.0)
            ts.loss_history.append(loss)
            ts.step_history.append(ts.step_idx)
            ts.step_idx += 1
        self._update_geom_err_stats(ts, geom_err)

        if ts.last_manual_point is not None:
            ts.prev_manual_point = ts.last_manual_point
        ts.last_manual_point = (int(frame_idx), float(row["x"]), float(row["y"]))
        if centroid is not None:
            if ts.last_centroid is not None:
                ts.prev_centroid = ts.last_centroid
            ts.last_centroid = centroid
            ts.centroid_history.append(centroid)

        # Replace existing label for this track/frame
        self.manual_track = [
            m for m in self.manual_track
            if not (int(m["frame"]) == int(frame_idx) and int(m.get("track_id", 0)) == int(track_id))
        ]
        self.manual_track.append({
            "frame": int(frame_idx),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "node": str(row["node"]),
            "instance_id": inst_id,
            "track_id": int(track_id),
            "source": source,
        })
        self._mark_timeline_dirty()
        if source in ("click", "interp", "auto") and inst_id is not None:
            self._update_geom_model(ts, dists)
            self._update_body_frac(ts, body_frac)
            frame_df = self._frame_points(frame_idx)
            if pred is not None:
                self._update_ml_model(ts, frame_df, inst_id, pred)

    def _maybe_auto_assign(self, frame_df, track_id=None):
        self.auto_debug = ""
        self.auto_assigned_last = False
        track_id = self.active_track if track_id is None else track_id
        ts = self.tracks.get(track_id)
        if ts is None:
            self.auto_debug = "no track"
            return
        if not self.active_learning.get():
            self.auto_debug = "off"
            return
        if self.frame_idx in ts.blocked_frames:
            self.auto_debug = "blocked"
            return
        if ts.last_manual_point is None:
            self.auto_debug = "need click"
            return
        if frame_df is None or frame_df.empty:
            self.auto_debug = "no points"
            return
        if self.last_auto_frame == self.frame_idx:
            self.auto_debug = "already"
            return
        if self._frame_has_track_label(self.frame_idx, track_id):
            self.auto_debug = "labeled"
            return

        used_other = self._used_instances(self.frame_idx, exclude_track=track_id)
        if used_other:
            frame_df = frame_df[~frame_df["instance_id"].isin(used_other)]
        if frame_df is None or frame_df.empty:
            self.auto_debug = "no candidates"
            return

        target = self._predict_position(ts)
        if target is None:
            self.auto_debug = "no target"
            return

        use_ml = self.ml_mode.get() and ts.ml_updates >= self.ml_min_updates
        if use_ml:
            feats, scores, inst_dfs = self._ml_scores(ts, frame_df, target)
            if not scores:
                self.auto_debug = "ml no feats"
                return
            body_thresh = self._body_frac_threshold(ts)
            valid = []
            for inst_id, score in scores.items():
                fv = feats[inst_id]
                body_frac = fv[3]
                geom_err = fv[2]
                vel_err = fv[4]
                if body_frac > body_thresh:
                    continue
                if geom_err > self._geom_threshold_value(ts):
                    continue
                if vel_err > self.motion_err_thresh:
                    continue
                valid.append((inst_id, score))
            if not valid:
                self.auto_debug = "ml gated"
                return
            valid.sort(key=lambda kv: kv[1], reverse=True)
            top_id, top_score = valid[0]
            if len(valid) > 1:
                margin = top_score - valid[1][1]
            else:
                margin = top_score
            ts.ml_last_margin = margin
            if margin < self.ml_min_margin:
                self.auto_debug = f"ml margin {margin:.2f}"
                return
            best = {
                "instance_id": top_id,
                "df": inst_dfs[top_id],
                "norm_dist": feats[top_id][1],
                "geom_err": feats[top_id][2],
                "body_frac": feats[top_id][3],
                "vel_err": feats[top_id][4],
                "score": top_score,
            }
        else:
            best = None
            model_body_len = self._body_length_model(ts)
            body_thresh = self._body_frac_threshold(ts)
            for inst_id, inst_df in frame_df.groupby("instance_id"):
                pts = inst_df[["x", "y"]].to_numpy(dtype=float)
                centroid = pts.mean(axis=0)
                spread = np.median(np.linalg.norm(pts - centroid, axis=1))
                if not np.isfinite(spread) or spread <= 0:
                    spread = 1.0
                dist = float(np.linalg.norm(centroid - np.array(target)))
                norm_dist = dist / spread
                node_map = self._node_map_from_df(inst_df)
                body_len = self._body_length_from_nodes(node_map)
                if body_len is None or not np.isfinite(body_len):
                    body_len = model_body_len
                if body_len is not None and body_len > 0:
                    body_frac = dist / body_len
                    if body_frac > body_thresh:
                        continue
                    vel_err = self._motion_error(ts, centroid, body_len)
                    if vel_err is not None and vel_err > self.motion_err_thresh:
                        continue
                dists = self._pair_distances(node_map)
                geom_err = self._geom_error(ts, dists)
                score = norm_dist + (self._geom_weight_value(ts) * geom_err if geom_err is not None else 0.0)
                if best is None or score < best["score"]:
                    best = {
                        "instance_id": inst_id,
                        "df": inst_df,
                        "norm_dist": norm_dist,
                        "score": score,
                        "geom_err": geom_err,
                        "body_len": body_len,
                        "body_frac": dist / body_len if body_len else None,
                        "vel_err": vel_err,
                    }

        if best is None or best["norm_dist"] > self.conf_threshold:
            self.auto_debug = f"far {best['norm_dist']:.2f}" if best else "far"
            return
        if best.get("body_frac") is not None and best["body_frac"] > body_thresh:
            self.auto_debug = f"body {best['body_frac']:.2f}>{body_thresh:.2f}"
            return
        geom_thresh = self._geom_threshold_value(ts)
        if best.get("geom_err") is not None and best["geom_err"] > geom_thresh:
            self.auto_debug = f"geom {best['geom_err']:.2f}"
            return
        if best.get("vel_err") is not None and best["vel_err"] > self.motion_err_thresh:
            self.auto_debug = f"motion {best['vel_err']:.2f}"
            return

        row = self._nearest_point(self.frame_idx, target, frame_df=best["df"])
        if row is None:
            return
        self.selected_instance_id = int(row["instance_id"])
        self.selected_node = row["node"]
        self.last_selected_xy = (float(row["x"]), float(row["y"]))
        self._record_manual_point(ts, track_id, self.frame_idx, row, source="auto")
        self.last_auto_frame = self.frame_idx
        mode_tag = "ml" if use_ml else "geom"
        if best.get("geom_err") is not None:
            self.auto_debug = f"{mode_tag} ok d={best['norm_dist']:.2f} g={best['geom_err']:.2f}"
        else:
            self.auto_debug = f"{mode_tag} ok d={best['norm_dist']:.2f}"
        self.auto_assigned_last = True

    def _handle_click(self, x, y, allow_auto_run=True, allow_auto_advance=True, silent=False):
        frame_df = self._frame_points(self.frame_idx)
        if frame_df is None or frame_df.empty:
            if not silent:
                messagebox.showinfo("No points", "No valid SLEAP points on this frame.")
            return False
        used_other = self._used_instances(self.frame_idx, exclude_track=self.active_track)
        if used_other:
            frame_df = frame_df[~frame_df["instance_id"].isin(used_other)]
        if frame_df is None or frame_df.empty:
            if not silent:
                messagebox.showinfo("No points", "All instances on this frame are already assigned to other tracks.")
            return False
        pts = frame_df[["x", "y"]].to_numpy(dtype=float)
        click = np.array([x, y], dtype=float)
        dists = np.linalg.norm(pts - click, axis=1)
        idx = int(np.argmin(dists))
        row = frame_df.iloc[idx]
        # If this frame was auto-assigned to a different instance for this track, discard future autos for that instance.
        if self.manual_track:
            existing = [
                p for p in self.manual_track
                if int(p.get("frame")) == int(self.frame_idx)
                and int(p.get("track_id", 0)) == int(self.active_track)
                and p.get("source") == "auto"
            ]
            for p in existing:
                if p.get("instance_id") is not None and int(p["instance_id"]) != int(row["instance_id"]):
                    bad_inst = int(p["instance_id"])
                    next_manual = None
                    for m in self.manual_track:
                        if int(m.get("track_id", 0)) != int(self.active_track):
                            continue
                        if int(m.get("frame")) <= int(self.frame_idx):
                            continue
                        if m.get("source") in ("auto", None):
                            continue
                        f = int(m.get("frame"))
                        if next_manual is None or f < next_manual:
                            next_manual = f
                    limit_frame = next_manual
                    self.manual_track = [
                        m for m in self.manual_track
                        if not (
                            m.get("source") == "auto"
                            and int(m.get("frame")) >= int(self.frame_idx)
                            and (limit_frame is None or int(m.get("frame")) < limit_frame)
                            and int(m.get("track_id", 0)) == int(self.active_track)
                            and m.get("instance_id") is not None
                            and int(m.get("instance_id")) == bad_inst
                        )
                    ]
                    self._mark_timeline_dirty()
                    break

        self.selected_instance_id = int(row["instance_id"]) if "instance_id" in row else None
        self.selected_node = row["node"]
        self.last_click_frame = self.frame_idx
        self.last_click_xy = (float(x), float(y))
        self.last_selected_xy = (float(row["x"]), float(row["y"]))
        current_point = (self.frame_idx, float(row["x"]), float(row["y"]))

        # If we skipped frames, interpolate and snap to nearest SLEAP points.
        ts = self.tracks.get(self.active_track)
        if ts is None:
            return False
        if ts.last_manual_point is not None:
            last_frame, last_x, last_y = ts.last_manual_point
            if self.frame_idx > last_frame + 1:
                for f in range(last_frame + 1, self.frame_idx):
                    alpha = (f - last_frame) / (self.frame_idx - last_frame)
                    interp_x = last_x + alpha * (current_point[1] - last_x)
                    interp_y = last_y + alpha * (current_point[2] - last_y)
                    interp_row = self._nearest_point(f, (interp_x, interp_y))
                    if interp_row is None:
                        if f in ts.skipped_missing_frames:
                            gap_row = {
                                "frame": f,
                                "x": float(interp_x),
                                "y": float(interp_y),
                                "node": str(row["node"]),
                                "instance_id": None,
                            }
                            self._record_manual_point(ts, self.active_track, f, gap_row, source="gap")
                            ts.skipped_missing_frames.discard(f)
                        continue
                    # Skip if instance already assigned to other track on this frame
                    if int(interp_row["instance_id"]) in self._used_instances(f, exclude_track=self.active_track):
                        continue
                    self._record_manual_point(ts, self.active_track, f, interp_row, source="interp")

        self._record_manual_point(ts, self.active_track, self.frame_idx, row, source="click")

        self._render()
        if allow_auto_run and self.active_learning.get() and self.auto_run_on_click.get() and not self.auto_run_paused:
            self._start_auto_run(start_from_next=True)
        elif allow_auto_advance and self.auto_advance.get():
            self.root.after(60, self._next_frame)
        return True

    def _toggle_add_point_mode(self):
        self.add_point_mode = not self.add_point_mode
        if self.add_point_mode:
            self.add_point_btn.config(bg="#ff9f1a", fg="#1e1e1e")
        else:
            self.add_point_btn.config(bg=self.bg, fg=self.fg)

    def _on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self.add_point_mode:
            self._handle_new_point(event.xdata, event.ydata)
        else:
            self._handle_click(event.xdata, event.ydata, allow_auto_run=True, allow_auto_advance=True, silent=False)

    def _handle_new_point(self, x, y):
        ts = self.tracks.get(self.active_track)
        if ts is None:
            return
        self.manual_track.append({
            "frame": int(self.frame_idx),
            "x": float(x),
            "y": float(y),
            "node": "new_point",
            "instance_id": None,
            "track_id": int(self.active_track),
            "source": "new",
        })
        self._mark_timeline_dirty()
        self.last_click_frame = self.frame_idx
        self.last_click_xy = (float(x), float(y))
        self._render(do_auto=False)

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._dragging = True
            self._drag_start = (event.xdata, event.ydata)
            self._pan_start = self.pan_offset
            self._drag_start_px = (event.x, event.y)
            self._drag_moved = False

    def _on_mouse_release(self, event):
        if event.button == 1:
            was_drag = self._drag_moved
            self._dragging = False
            self._drag_start = None
            self._pan_start = None
            self._drag_start_px = None
            self._drag_moved = False
            if not was_drag:
                self._on_click(event)

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.last_mouse_xy = (event.xdata, event.ydata)
        if not self._dragging:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self._drag_start is None or self._pan_start is None:
            return
        if self._drag_start_px is not None:
            dx_px = event.x - self._drag_start_px[0]
            dy_px = event.y - self._drag_start_px[1]
            if (dx_px * dx_px + dy_px * dy_px) < 9:
                return
            self._drag_moved = True
        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]
        self.pan_offset = (self._pan_start[0] - dx, self._pan_start[1] - dy)
        self._render()

    def _start_z_hold(self):
        if self.z_hold_active:
            return
        self.z_hold_active = True
        self.z_last_frame = None
        self._z_hold_step()

    def _stop_z_hold(self):
        self.z_hold_active = False

    def _z_hold_step(self):
        if not self.z_hold_active:
            return
        if self._dragging:
            self.root.after(self.z_hold_delay_ms, self._z_hold_step)
            return
        if self.last_mouse_xy is None:
            self.root.after(self.z_hold_delay_ms, self._z_hold_step)
            return
        if self.z_last_frame == self.frame_idx:
            self.root.after(self.z_hold_delay_ms, self._z_hold_step)
            return

        x, y = self.last_mouse_xy
        self._handle_click(x, y, allow_auto_run=False, allow_auto_advance=False, silent=True)
        self.z_last_frame = self.frame_idx
        self._next_frame()
        self.root.after(self.z_hold_delay_ms, self._z_hold_step)

    def _update_timeline(self):
        ax = self.ax_timeline
        if not self._timeline_dirty and self._timeline_vline is not None:
            self._timeline_vline.set_xdata([self.frame_idx, self.frame_idx])
            return
        ax.clear()
        ax.set_facecolor(self.bg)
        min_f, max_f = self._frame_bounds()
        ax.set_xlim(min_f, max_f)
        ax.set_ylim(-0.5, self.track_count - 0.5)
        ticks = list(range(self.track_count))
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(i) for i in ticks], color=self.fg)
        label_size = 8 if self.track_count <= 12 else 7 if self.track_count <= 20 else 6
        ax.tick_params(axis="x", colors=self.fg, labelsize=8)
        ax.tick_params(axis="y", colors=self.fg, labelsize=label_size)
        ax.set_xlabel("Frame", color=self.fg, fontsize=9)
        ax.set_ylabel("Track", color=self.fg, fontsize=9)
        for tid in range(self.track_count):
            ax.hlines(tid, min_f, max_f, color="#333333", linewidth=1, alpha=0.6)

        if self.manual_track:
            manual_frames = {}
            auto_frames = {}
            interp_frames = {}
            gap_frames = {}
            for m in self.manual_track:
                tid = int(m.get("track_id", 0))
                frame = int(m.get("frame", 0))
                src = str(m.get("source", "click"))
                if src == "auto":
                    auto_frames.setdefault(tid, []).append(frame)
                elif src == "interp":
                    interp_frames.setdefault(tid, []).append(frame)
                elif src == "gap":
                    gap_frames.setdefault(tid, []).append(frame)
                else:
                    manual_frames.setdefault(tid, []).append(frame)

            for tid, ts in self.tracks.items():
                base = ts.color
                auto_color = _lighten_hex(base, 0.55)
                interp_color = "#ff5252"
                gap_color = "#9e9e9e"
                if tid in manual_frames:
                    ax.scatter(
                        manual_frames[tid],
                        [tid] * len(manual_frames[tid]),
                        s=8,
                        c=base,
                        marker="|",
                        linewidths=1.2,
                    )
                if tid in auto_frames:
                    ax.scatter(
                        auto_frames[tid],
                        [tid] * len(auto_frames[tid]),
                        s=8,
                        c=auto_color,
                        marker="|",
                        linewidths=1.2,
                    )
                if tid in interp_frames:
                    ax.scatter(
                        interp_frames[tid],
                        [tid] * len(interp_frames[tid]),
                        s=8,
                        c=interp_color,
                        marker="|",
                        linewidths=1.2,
                    )
                if tid in gap_frames:
                    ax.scatter(
                        gap_frames[tid],
                        [tid] * len(gap_frames[tid]),
                        s=8,
                        c=gap_color,
                        marker="|",
                        linewidths=1.2,
                    )

        self._timeline_vline = ax.axvline(self.frame_idx, color="#ffffff", linewidth=1.0, alpha=0.7)
        self._timeline_dirty = False

    def _segment_larvae(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return mask

    def _draw_seg_overlays(self, frame_rgb, frame_df):
        mask = self._segment_larvae(frame_rgb)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 200]
        if not valid:
            return
        for tid, ts in self.tracks.items():
            picked = [
                p for p in self.manual_track
                if int(p.get("frame")) == int(self.frame_idx)
                and int(p.get("track_id", 0)) == int(tid)
                and p.get("instance_id") is not None
            ]
            if not picked:
                continue
            inst_ids = {int(p["instance_id"]) for p in picked}
            inst_df = frame_df[frame_df["instance_id"].isin(inst_ids)]
            if inst_df.empty:
                continue
            cx = float(inst_df["x"].mean())
            cy = float(inst_df["y"].mean())
            best_c, best_d = None, float("inf")
            for c in valid:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                d = np.hypot(M["m10"] / M["m00"] - cx, M["m01"] / M["m00"] - cy)
                if d < best_d:
                    best_d, best_c = d, c
            if best_c is not None and best_d < 150:
                pts = best_c.reshape(-1, 2).astype(float)
                xs = np.append(pts[:, 0], pts[0, 0])
                ys = np.append(pts[:, 1], pts[0, 1])
                self.ax.plot(xs, ys, color=ts.color, linewidth=1.5, alpha=0.85, zorder=5)

    def _render(self, do_auto=True):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_aspect("auto")
        self.ax.set_facecolor(self.bg)

        frame = self._read_frame(self.frame_idx)
        if frame is not None:
            self.frame_shape = frame.shape[:2]
            if self.zoom > 1.0:
                x_min, x_max, y_min, y_max = self._view_bounds(self.frame_shape)
                h, w = self.frame_shape
                x0 = int(max(0, np.floor(x_min)))
                x1 = int(min(w, np.ceil(x_max)))
                y0 = int(max(0, np.floor(y_min)))
                y1 = int(min(h, np.ceil(y_max)))
                crop = frame[y0:y1, x0:x1]
                if self.display_scale < 1.0:
                    crop = cv2.resize(
                        crop,
                        (max(1, int(crop.shape[1] * self.display_scale)),
                         max(1, int(crop.shape[0] * self.display_scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                self.ax.imshow(
                    crop,
                    origin="upper",
                    extent=[x0, x1, y1, y0],
                )
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_max, y_min)
            else:
                view = frame
                if self.display_scale < 1.0:
                    view = cv2.resize(
                        frame,
                        (max(1, int(frame.shape[1] * self.display_scale)),
                         max(1, int(frame.shape[0] * self.display_scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                self.ax.imshow(
                    view,
                    origin="upper",
                    extent=[0, self.frame_shape[1], self.frame_shape[0], 0],
                )
                self.ax.set_xlim(0, self.frame_shape[1])
                self.ax.set_ylim(self.frame_shape[0], 0)
        else:
            self.ax.text(0.5, 0.5, "No frame", ha="center", va="center")

        frame_df = self._frame_points(self.frame_idx)
        if do_auto:
            self._maybe_auto_assign(frame_df)
        if frame_df is not None and not frame_df.empty:
            nodes = frame_df["node"].unique() if "node" in frame_df.columns else ["point"]
            colors = ["#ff6b6b", "#6bc5ff", "#ffd36b", "#6bff95", "#c46bff", "#ff6bd6", "#6bffea"]
            if "instance_id" in frame_df.columns:
                for inst_id, inst_df in frame_df.groupby("instance_id"):
                    node_map = self._node_map_from_df(inst_df)
                    if len(node_map) < 2:
                        continue
                    ordered = self._ordered_nodes(list(node_map.keys()))
                    xs = []
                    ys = []
                    for node in ordered:
                        if node in node_map:
                            x, y = node_map[node]
                            xs.append(x)
                            ys.append(y)
                    if len(xs) < 2:
                        continue
                    self.ax.plot(
                        xs,
                        ys,
                        color="#f0f0f0",
                        linewidth=1.2,
                        alpha=0.6,
                        zorder=1,
                    )
            for i, node in enumerate(nodes):
                if "node" in frame_df.columns:
                    sub = frame_df[frame_df["node"] == node]
                else:
                    sub = frame_df
                color = colors[i % len(colors)]
                self.ax.scatter(sub["x"], sub["y"], s=18, c=color, edgecolors="black", linewidths=0.6, zorder=3)
            if self.manual_track and "instance_id" in frame_df.columns:
                for tid, ts in self.tracks.items():
                    picked = [
                        p for p in self.manual_track
                        if int(p.get("frame")) == int(self.frame_idx)
                        and int(p.get("track_id", 0)) == int(tid)
                        and p.get("instance_id") is not None
                    ]
                    if not picked:
                        continue
                    inst_ids = {int(p["instance_id"]) for p in picked}
                    inst_df = frame_df[frame_df["instance_id"].isin(inst_ids)]
                    if inst_df.empty:
                        continue
                    self.ax.scatter(
                        inst_df["x"],
                        inst_df["y"],
                        s=70,
                        c="none",
                        edgecolors=ts.color,
                        linewidths=1.8,
                        marker="o",
                        zorder=4,
                    )
            if self.manual_track:
                for tid, ts in self.tracks.items():
                    tagged = [
                        p for p in self.manual_track
                        if int(p.get("frame")) == int(self.frame_idx)
                        and int(p.get("track_id", 0)) == int(tid)
                        and p.get("source") in ("interp", "gap")
                    ]
                    if not tagged:
                        continue
                    for p in tagged:
                        try:
                            x = float(p.get("x"))
                            y = float(p.get("y"))
                        except (TypeError, ValueError):
                            continue
                        source = p.get("source")
                        if source == "interp":
                            marker = "D"
                            size = 70
                        else:
                            marker = "s"
                            size = 70
                        self.ax.scatter(
                            [x],
                            [y],
                            s=size,
                            c="none",
                            edgecolors=ts.color,
                            linewidths=1.8,
                            marker=marker,
                            alpha=0.9,
                            zorder=5,
                        )
            if self.manual_track:
                for tid, ts in self.tracks.items():
                    new_pts = [
                        p for p in self.manual_track
                        if int(p.get("frame")) == int(self.frame_idx)
                        and int(p.get("track_id", 0)) == int(tid)
                        and p.get("source") == "new"
                    ]
                    for p in new_pts:
                        try:
                            xp = float(p.get("x"))
                            yp = float(p.get("y"))
                        except (TypeError, ValueError):
                            continue
                        self.ax.scatter(
                            [xp], [yp],
                            s=90,
                            c=ts.color,
                            edgecolors="white",
                            linewidths=1.2,
                            marker="P",
                            alpha=0.95,
                            zorder=6,
                        )
            if frame is not None and self.show_seg_overlay.get():
                self._draw_seg_overlays(frame, frame_df)
        if self.last_click_frame == self.frame_idx and self.last_click_xy is not None:
            x_click, y_click = self.last_click_xy
            self.ax.scatter(
                [x_click],
                [y_click],
                s=140,
                c="none",
                edgecolors="#ffea00",
                linewidths=2.5,
                marker="o",
                zorder=6,
            )
        if self.last_click_frame == self.frame_idx and self.last_selected_xy is not None:
            x_sel, y_sel = self.last_selected_xy
            self.ax.scatter(
                [x_sel],
                [y_sel],
                s=120,
                c="none",
                edgecolors="#00ff7f",
                linewidths=2.0,
                marker="*",
                zorder=6,
            )
        ts = self.tracks.get(self.active_track)
        if ts is not None and ts.last_manual_point is not None:
            frame_last, x_last, y_last = ts.last_manual_point
            if self.frame_idx == frame_last + self.direction:
                self.ax.scatter(
                    [x_last],
                    [y_last],
                    s=90,
                    c="#00bcd4",
                    linewidths=2.0,
                    marker="x",
                )
            self.ax.text(
                0.01,
                0.01,
                f"Track {self.active_track} last: frame {frame_last} @ ({x_last:.1f}, {y_last:.1f})",
                transform=self.ax.transAxes,
                color="#00bcd4",
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2),
            )

        self._update_timeline()
        self._render_loss()
        self._set_status()
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(self.frame_idx))
        if self.total_frames is not None:
            self._updating_scale = True
            self.frame_scale.set(self.frame_idx)
            self._updating_scale = False
        self.canvas.draw_idle()

    def _render_loss(self):
        self.ax_loss.clear()
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("step")
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.tick_params(colors=self.fg)
        self.ax_loss.set_facecolor(self.bg)
        self.ax_loss.title.set_color(self.fg)
        self.ax_loss.xaxis.label.set_color(self.fg)
        self.ax_loss.yaxis.label.set_color(self.fg)
        ts = self.tracks.get(self.active_track)
        if ts is not None and ts.loss_history:
            self.ax_loss.plot(ts.step_history, ts.loss_history, color="#ff9f1a", linewidth=1.5)


def main():
    root = tk.Tk()
    app = SleapVideoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

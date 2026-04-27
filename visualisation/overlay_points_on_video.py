import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

DEFAULT_VIDEO = "data/behaviour/GH-2026-03-10_14-42-25_SV35.mp4"
DEFAULT_TRACKS = "data/files/FULL-F_2026-03-10_14-42-25_SV35.tracks.manual_track.20260427_140125.csv"
DEFAULT_SLEAP = "data/behaviour/2026-03-10_14-42-25_SV35.tracks.feather"

# One BGR color per track_id (cycles if more than len)
TRACK_COLORS_BGR = [
    (80, 80, 255),   # red
    (255, 160, 50),  # blue
    (50, 220, 255),  # yellow
    (80, 220, 80),   # green
    (200, 80, 220),  # purple
    (220, 80, 180),  # pink
    (200, 220, 80),  # cyan
    (80, 130, 255),  # orange
]

NODE_ORDER = ["head", "mouthhooks", "body", "tail", "spiracle"]
DOT_RADIUS = 6
LINE_THICKNESS = 1


def normalize_feather(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format SLEAP feather to long format with instance_id = within-frame rank."""
    nodes = [c[2:] for c in df.columns if c.startswith("x_")]
    # instance_id = within-frame cumcount on original feather order, matching the GUI
    instance_id = df.groupby("frame", sort=False).cumcount().astype(int).values
    records = []
    for node in nodes:
        records.append(pd.DataFrame({
            "frame": df["frame"].astype(int).values,
            "sleap_track_id": df["track_id"].astype(int).values,
            "instance_id": instance_id,
            "x": df[f"x_{node}"].astype(float).values,
            "y": df[f"y_{node}"].astype(float).values,
            "node": node,
        }))
    return pd.concat(records, ignore_index=True)


def build_overlay(long_df: pd.DataFrame, tracks_df: pd.DataFrame):
    """Return dict: frame_idx -> list of (x, y, node, track_id, color_bgr)."""
    # Unique (frame, instance_id) -> track_id from manual tracks
    # Use the first occurrence per (frame, instance_id) to get track_id
    assignment = (
        tracks_df[["frame", "instance_id", "track_id"]]
        .drop_duplicates(subset=["frame", "instance_id"])
    )

    # Join feather long data to manual track assignments
    merged = long_df.merge(
        assignment.rename(columns={"track_id": "manual_track_id"}),
        on=["frame", "instance_id"],
        how="inner",
    )

    track_ids = sorted(merged["manual_track_id"].unique())
    color_map = {tid: TRACK_COLORS_BGR[i % len(TRACK_COLORS_BGR)] for i, tid in enumerate(track_ids)}

    merged = merged.dropna(subset=["x", "y"])
    overlay = {}
    for fid, grp in merged.groupby("frame"):
        overlay[int(fid)] = [
            (float(r.x), float(r.y), r.node, int(r.manual_track_id), color_map[int(r.manual_track_id)])
            for r in grp.itertuples(index=False)
        ]
    return overlay, color_map


def draw_on_frame(frame_bgr: np.ndarray, points) -> np.ndarray:
    """Draw keypoints and skeleton lines for one frame."""
    out = frame_bgr.copy()
    # Group by track_id to draw skeleton lines
    by_track: dict[int, dict[str, tuple]] = {}
    for x, y, node, tid, color in points:
        by_track.setdefault(tid, {})[node] = (int(round(x)), int(round(y)), color)

    for tid, node_map in by_track.items():
        color = next(iter(node_map.values()))[2]
        ordered = [n for n in NODE_ORDER if n in node_map]
        for i in range(len(ordered) - 1):
            a, b = node_map[ordered[i]], node_map[ordered[i + 1]]
            cv2.line(out, a[:2], b[:2], color, LINE_THICKNESS, cv2.LINE_AA)
        for node, (px, py, col) in node_map.items():
            cv2.circle(out, (px, py), DOT_RADIUS, col, -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), DOT_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser(description="Overlay SLEAP predictions on video for manually tracked frames.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input mp4 video path")
    parser.add_argument("--tracks", default=DEFAULT_TRACKS, help="Manual tracks CSV path")
    parser.add_argument("--sleap_predictions", default=DEFAULT_SLEAP, help="SLEAP predictions feather path")
    parser.add_argument("--output", default=None, help="Output mp4 path (default: <video_stem>_overlay.mp4)")
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(args.video)[0]
        args.output = stem + "_overlay.mp4"

    print(f"Loading tracks:      {args.tracks}")
    tracks_df = pd.read_csv(args.tracks)
    print(f"  {len(tracks_df)} rows, frames {tracks_df['frame'].min()}–{tracks_df['frame'].max()}, "
          f"{tracks_df['track_id'].nunique()} tracks")

    print(f"Loading predictions: {args.sleap_predictions}")
    feather_df = pd.read_feather(args.sleap_predictions)
    feather_df["frame"] = feather_df["frame"].astype(int)
    feather_df["track_id"] = feather_df["track_id"].astype(int)

    relevant_frames = set(tracks_df["frame"].unique())
    feather_sub = feather_df[feather_df["frame"].isin(relevant_frames)].copy()
    print(f"  {len(feather_sub)} prediction rows across {len(relevant_frames)} relevant frames")

    print("Normalizing predictions...")
    long_df = normalize_feather(feather_sub)
    del feather_sub
    overlay, color_map = build_overlay(long_df, tracks_df)
    del long_df, tracks_df
    print(f"  Overlay built for {len(overlay)} frames, {len(color_map)} tracks")

    print(f"Opening video:       {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {w}x{h} @ {fps:.2f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    min_frame = min(relevant_frames)
    max_frame = max(relevant_frames)
    n_frames = len(relevant_frames)
    span = max_frame - min_frame + 1
    print(f"Writing {span} frames ({min_frame}–{max_frame}, {n_frames} with annotations) to: {args.output}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
    written = 0
    for fid in range(min_frame, max_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: read failed at frame {fid}")
            break
        points = overlay.get(fid)
        writer.write(draw_on_frame(frame, points) if points else frame)
        written += 1
        if written % 500 == 0:
            print(f"  {written}/{span} frames written")

    cap.release()
    writer.release()
    print("Done.")


if __name__ == "__main__":
    main()

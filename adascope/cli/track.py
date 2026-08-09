"""Fahrzeugtracking in der Bildebene: annotiertes Video plus Track-CSV.

Die Eingangsstufe fuer die BEV-Belegung. Die CSV enthaelt je Frame Box, Rolle,
Confidence und Bbox-Unterkante -- letztere ist die einzige Groesse, die
`lanes.bev.project_footprint()` in die Bodenebene uebernehmen darf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ..io import VideoWriter, iter_source, write_named
from ..render.primitives import FONT
from ._common import (
    add_config_args, add_source_args, add_tracking_args, build_tracker,
    iter_limited, load_settings,
)

FIELDS = ["frame", "timestamp_s", "source", "track_id", "role", "class", "confidence",
          "x1", "y1", "x2", "y2", "bottom_center_x", "bottom_y"]


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    add_source_args(parser)
    add_tracking_args(parser)
    parser.add_argument("--output", type=Path, default=Path("outputs/vehicle_tracks.mp4"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/vehicle_tracks.csv"))


def run(args: argparse.Namespace) -> int:
    if args.source is None:
        raise SystemExit("--source fehlt (Videodatei oder Frame-Ordner)")
    settings = load_settings(args)
    tracker = build_tracker(args, settings)
    if tracker is None:
        raise SystemExit("track braucht einen Detektor -- --no-detect ist hier sinnlos")

    frames, fps = iter_source(args.source, args.stride, args.fps)
    rows: list[dict] = []
    with VideoWriter(args.output, fps / max(args.stride, 1)) as writer:
        for index, name, frame in iter_limited(frames, args.stride, args.max_frames):
            vehicles = tracker.update(frame)
            writer.write(draw_tracks(frame, vehicles, index, settings))
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle.bbox
                bx, by = vehicle.bottom_center
                rows.append({
                    "frame": index, "timestamp_s": round(index / fps, 3), "source": name,
                    "track_id": vehicle.track_id, "role": vehicle.role,
                    "class": vehicle.label, "confidence": round(vehicle.confidence, 5),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bottom_center_x": round(bx, 2), "bottom_y": round(by, 2),
                })
            if len(rows) and (index == 0 or index % 100 == 0):
                print(f"Frame {index} | {len(vehicles)} Fahrzeuge | Ego-ID {tracker.ego_track_id}")

    write_named(rows, args.csv, FIELDS)
    print(f"Video: {args.output}\nCSV:   {args.csv}")
    return 0


def draw_tracks(frame, vehicles, frame_index: int, settings) -> "cv2.Mat":
    debug = settings.debug
    output = frame.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        color = debug.color("ego") if vehicle.role == "ego" else debug.color("untracked")
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        identity = "?" if vehicle.track_id is None else str(vehicle.track_id)
        label = f"{vehicle.role.upper()} ID={identity} {vehicle.label} {vehicle.confidence:.2f}"
        cv2.putText(output, label, (x1, max(18, y1 - 7)), FONT, .45, color, 2, cv2.LINE_AA)
        bx, by = vehicle.bottom_center
        cv2.drawMarker(output, (round(bx), round(by)), color, cv2.MARKER_CROSS, 10, 2)
    cv2.putText(output, f"frame {frame_index}", (12, 22), FONT, .55, (255, 255, 255), 2)
    return output

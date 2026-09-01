"""Stufe 4a - Rohe Homographie-Kandidaten gegen temporales Gate.

Links wird das Paar gezeigt, das der aktuelle Einzel-Frame ungeprueft anbieten
wuerde. Rechts steht die tatsaechlich akzeptierte, geglaettete oder gehaltene
Geometrie. Ein roter Kandidat wurde verworfen; der Grund und die Messwerte
stehen in Bild und CSV.

Stellschrauben: pipeline.homography_max_point_jump,
homography_max_width_change_ratio, homography_max_vanishing_jump,
homography_max_top_width_ratio, homography_min_pair_continuity,
homography_min_pair_support
"""

from __future__ import annotations

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.bev import outer_solid_pair, source_points
from adascope.lanes.detection import detect_lanes
from adascope.lanes.pipeline import HomographyTracker


def tracker_from(settings) -> HomographyTracker:
    cfg = settings.pipeline
    return HomographyTracker(
        settings.lane, settings.bev, cfg.max_hold,
        smoothing=cfg.homography_smoothing,
        max_point_jump=cfg.homography_max_point_jump,
        max_width_change_ratio=cfg.homography_max_width_change_ratio,
        max_vanishing_jump=cfg.homography_max_vanishing_jump,
        max_top_width_ratio=cfg.homography_max_top_width_ratio,
        min_pair_continuity=cfg.homography_min_pair_continuity,
        min_pair_support=cfg.homography_min_pair_support,
    )


def draw_geometry(img, src, color, title):
    out = img.copy()
    if src is not None:
        pts = np.rint(src).astype(int)
        bl, br, tr, tl = pts
        cv2.line(out, tuple(bl), tuple(tl), color, 3)
        cv2.line(out, tuple(br), tuple(tr), color, 3)
        for t in np.linspace(0.0, 1.0, 7):
            left = np.rint(tl * (1 - t) + bl * t).astype(int)
            right = np.rint(tr * (1 - t) + br * t).astype(int)
            cv2.line(out, tuple(left), tuple(right), (90, 90, 90), 1)
        for point in pts:
            cv2.circle(out, tuple(point), 4, color, -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse(__doc__, stufe="stage_04a_temporal_gate")
    st = Stage(args, "stage_04a_temporal_gate", "pipeline.homography_*")
    tracker = tracker_from(st.settings)
    previous_raw = previous_used = None

    for i, name, img, lane in st.frames():
        tracker.lane = lane
        result = detect_lanes(img, lane)
        pair = outer_solid_pair(result, lane, st.settings.bev)
        raw = source_points(pair, lane) if pair else None
        _, state = tracker.update(result)
        used = tracker.accepted_src

        raw_width = float(raw[1, 0] - raw[0, 0]) if raw is not None else 0.0
        used_width = float(used[1, 0] - used[0, 0]) if used is not None else 0.0
        raw_delta = abs(raw_width - previous_raw) if raw_width and previous_raw else 0.0
        used_delta = abs(used_width - previous_used) if used_width and previous_used else 0.0
        if raw_width:
            previous_raw = raw_width
        if state == "fresh" and used_width:
            previous_used = used_width

        metrics = tracker.metrics
        reason = tracker.last_rejection
        st.row(frame=i, zustand=state, ablehnungsgrund=reason,
               roh_breite=round(raw_width, 1), roh_delta=round(raw_delta, 1),
               benutzt_breite=round(used_width, 1), benutzt_delta=round(used_delta, 1),
               punkt_sprung=round(metrics.get("point_jump", 0.0), 1),
               breite_delta=round(metrics.get("width_change_ratio", 0.0), 3),
               fluchtpunkt_sprung=round(metrics.get("vanishing_jump", 0.0), 1),
               continuity=round(metrics.get("pair_continuity", 0.0), 2),
               support=round(metrics.get("pair_support", 0.0)),
               gehalten=tracker.held_frames)

        raw_color = (0, 0, 255) if reason and raw is not None else (0, 200, 255)
        left = draw_geometry(img, raw, raw_color,
                             f"ROH: {raw_width:.0f}px" + (f"  reject:{reason}" if reason else ""))
        right = draw_geometry(img, used, (255, 170, 0),
                              f"BENUTZT: {state} {used_width:.0f}px")
        st.snapshot(i, "01_raw_candidate", left)
        st.snapshot(i, "02_temporal_result", right)
        st.show(i, np.hstack([left, right]))
    st.finish()


if __name__ == "__main__":
    main()

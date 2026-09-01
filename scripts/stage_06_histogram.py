"""Stufe 6 - Spaltenhistogramm und Peaks.

Hier entstehen die Spurgrenzen. Zu viele Peaks: `peak_min_distance`
zu klein oder `peak_min_pixels` zu niedrig. Zu wenige: umgekehrt.

`peak_min_distance` muss deutlich unter der Spurbreite liegen, aber
deutlich ueber der Strichbreite -- ein zu kleiner Wert erzeugt
Scheinkorridore von einem Drittel Spurbreite.

Stellschrauben: bev.peak_min_pixels, bev.peak_min_distance, bev.histogram_blur, bev.peak_window

    python scripts/stage_06_histogram.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.bev import (
    build_lane_mask, lane_histogram, peaks_from_histogram,
    restrict_to_driving_area, warp_lane_mask,
)
from adascope.lanes.detection import detect_lanes
from adascope.lanes.pipeline import HomographyTracker, road_vehicles
from adascope.detection import YoloVehicleTracker


def zusatz(ap) -> None:
    ap.add_argument("--detect", action=argparse.BooleanOptionalAction, default=None,
                    help="Fahrzeugboxen wie in der echten Pipeline aus der Maske stanzen")


def main() -> None:
    args = parse(__doc__, zusatz, stufe="stage_06_histogram")
    st = Stage(args, "stage_06_histogram",
               "bev.peak_min_pixels, bev.peak_min_distance, "
               "bev.histogram_blur, bev.peak_window")
    bev = st.settings.bev
    tracker = HomographyTracker(st.settings.lane, bev,
                                st.settings.pipeline.max_hold,
                                smoothing=st.settings.pipeline.homography_smoothing,
                                max_point_jump=st.settings.pipeline.homography_max_point_jump,
                                max_width_change_ratio=st.settings.pipeline.homography_max_width_change_ratio,
                                max_vanishing_jump=st.settings.pipeline.homography_max_vanishing_jump,
                                max_top_width_ratio=st.settings.pipeline.homography_max_top_width_ratio,
                                min_pair_continuity=st.settings.pipeline.homography_min_pair_continuity,
                                min_pair_support=st.settings.pipeline.homography_min_pair_support)
    detector = YoloVehicleTracker(st.settings.tracking) if args.detect else None
    print(f"  Detektion     {'an' if detector else 'AUS (Rohmaske)'}")
    for i, name, img, lane in st.frames():
        fahrzeuge = road_vehicles(detector.update(img), lane,
                                  st.settings.pipeline.road_margin) if detector else []
        tracker.lane = lane
        H, zustand = tracker.update(detect_lanes(img, lane))
        if H is None:
            st.row(frame=i, zustand=zustand,
                   ablehnungsgrund=tracker.last_rejection,
                   n_fahrzeuge=len(fahrzeuge), n_peaks=0,
                   abstand_median=0.0, abstand_min=0, hist_max=0.0)
            leer = np.zeros((bev.height, bev.width), np.uint8)
            st.snapshot(i, "01_bev_peaks", leer, "Keine Homographie")
            st.show(i, leer, f"f{i}  keine Homographie")
            continue
        maske = build_lane_mask(img, lane, [v.bbox for v in fahrzeuge])
        maske = restrict_to_driving_area(maske, tracker.accepted_src)
        gewarpt = warp_lane_mask(maske, H, bev)
        hist = lane_histogram(gewarpt, bev)
        peaks = peaks_from_histogram(hist, bev)
        abstaende = [b - a for a, b in zip(peaks, peaks[1:])]
        st.row(frame=i, zustand=zustand,
               ablehnungsgrund=tracker.last_rejection,
               n_fahrzeuge=len(fahrzeuge), n_peaks=len(peaks),
               abstand_median=round(float(np.median(abstaende)), 1) if abstaende else 0.0,
               abstand_min=min(abstaende) if abstaende else 0,
               hist_max=round(float(hist.max()), 1))

        # Histogramm als Kurve unter der BEV-Maske.
        hoehe = 160
        plot = np.zeros((hoehe, bev.width, 3), np.uint8)
        skala = hoehe / max(float(hist.max()), 1.0)
        for x in range(bev.width - 1):
            cv2.line(plot, (x, hoehe - int(hist[x] * skala)),
                     (x + 1, hoehe - int(hist[x + 1] * skala)), (200, 200, 200), 1)
        y_schwelle = hoehe - int(bev.peak_min_pixels * skala)
        cv2.line(plot, (0, y_schwelle), (bev.width, y_schwelle), (0, 140, 255), 1)
        for x in peaks:
            cv2.line(plot, (x, 0), (x, hoehe), (0, 255, 255), 1)
        oben = cv2.cvtColor(gewarpt, cv2.COLOR_GRAY2BGR)
        for x in peaks:
            cv2.line(oben, (x, 0), (x, bev.height), (0, 255, 255), 1)
        st.snapshot(i, "01_bev_peaks", oben,
                    "BEV-Maske + gefundene Peaks")
        st.snapshot(i, "02_histogram", plot,
                    "Spaltenhistogramm + Schwelle + Peaks")
        st.show(i, np.vstack([oben, plot]), f"f{i}  {len(peaks)} Peaks")
    st.finish()


if __name__ == "__main__":
    main()

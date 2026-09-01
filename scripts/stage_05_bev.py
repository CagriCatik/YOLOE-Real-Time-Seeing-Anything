"""Stufe 5 - Gewarpte Spurmaske in der Bodenebene.

Die Maske aus Stufe 1, durch die Homographie aus Stufe 4 gezogen.
Hier muessen die Markierungen als senkrechte Saeulen erscheinen. Sind
sie schraeg oder gebogen, stimmt die Homographie nicht -- dann ist
Stufe 4 zu pruefen, nicht diese.

Stellschrauben: bev.width, bev.height, bev.x_left/x_right, bev.y_near/y_far

    python scripts/stage_05_bev.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.bev import (
    build_lane_mask, restrict_to_driving_area, warp_lane_mask,
)
from adascope.lanes.detection import detect_lanes
from adascope.lanes.pipeline import HomographyTracker, road_vehicles
from adascope.detection import YoloVehicleTracker


def zusatz(ap) -> None:
    ap.add_argument("--detect", action=argparse.BooleanOptionalAction, default=None,
                    help="Fahrzeugboxen wie in der echten Pipeline aus der Maske stanzen")


def main() -> None:
    args = parse(__doc__, zusatz, stufe="stage_05_bev")
    st = Stage(args, "stage_05_bev",
               "bev.width, bev.height, bev.x_left/x_right, bev.y_near/y_far")
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
                   n_fahrzeuge=len(fahrzeuge),
                   entfernte_pixel=0, ausserhalb_richtung=0,
                   bev_pixel=0, belegte_spalten=0,
                   saeulen_anteil=0.0)
            st.show(i, np.zeros((bev.height, bev.width), np.uint8),
                    f"f{i}  keine Homographie")
            continue
        rohmaske = build_lane_mask(img, lane, [])
        maske = build_lane_mask(img, lane, [v.bbox for v in fahrzeuge])
        richtungsmaske = restrict_to_driving_area(maske, tracker.accepted_src)
        gewarpt = warp_lane_mask(richtungsmaske, H, bev)
        vergleich = (img.astype(np.float32) * 0.30).astype(np.uint8)
        behalten = richtungsmaske > 0
        entfernt = (maske > 0) & ~behalten
        vergleich[behalten] = (0, 220, 0)
        vergleich[entfernt] = (0, 0, 255)
        cv2.polylines(vergleich,
                      [np.rint(tracker.accepted_src).astype(np.int32)], True,
                      (0, 255, 255), 2)
        spalten = (gewarpt > 0).sum(axis=0)
        st.row(frame=i, zustand=zustand,
               ablehnungsgrund=tracker.last_rejection,
               n_fahrzeuge=len(fahrzeuge),
               entfernte_pixel=int(((rohmaske > 0) & (maske == 0)).sum()),
               ausserhalb_richtung=int(((maske > 0) & (richtungsmaske == 0)).sum()),
               bev_pixel=int((gewarpt > 0).sum()),
               belegte_spalten=int((spalten > 0).sum()),
               saeulen_anteil=round(float((spalten > bev.peak_min_pixels).mean()), 3))
        st.snapshot(i, "01_camera_mask_raw", rohmaske, "Rohmaske vor Fahrzeugfilter")
        st.snapshot(i, "02_camera_mask_clean", maske,
                    f"Pipeline-Maske: {len(fahrzeuge)} Fahrzeugboxen entfernt")
        st.snapshot(i, "03_direction_filter_overlay", vergleich,
                    "GRUEN weitergegeben | ROT ausserhalb Richtungsfahrbahn")
        st.snapshot(i, "04_driving_area_mask", richtungsmaske,
                    "Nur eigene Richtungsfahrbahn (Downstream-Eingang)")
        st.snapshot(i, "05_bev_mask", gewarpt, "Binaere Maske in Bodenebene")
        st.show(i, gewarpt, f"f{i}  H: {zustand}")
    st.finish()


if __name__ == "__main__":
    main()

"""Stufe 1 - Weissmaske und ROI.

Zeigt, was die Spurerkennung ueberhaupt als Markierung sieht. Ist hier
nichts, kann keine spaetere Stufe es reparieren. Zu viel ist ebenso
schaedlich: Fahrbahntextur und HMI-Grafik werden dann zu Scheinlinien.

Der weisse Anteil sollte im Bereich weniger Prozent der ROI liegen.

Stellschrauben: lane.white_l_min, lane.white_l_max, lane.roi_polygon

    python scripts/stage_01_mask.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.bev import build_lane_mask


def main() -> None:
    args = parse(__doc__, stufe="stage_01_mask")
    st = Stage(args, "stage_01_mask",
               "lane.white_l_min, lane.white_l_max, lane.roi_polygon")
    for i, name, img, lane in st.frames():
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        weiss = cv2.inRange(hls[:, :, 1], lane.white_l_min, lane.white_l_max)
        roi = np.zeros(weiss.shape, np.uint8)
        cv2.fillPoly(roi, [np.array(lane.roi_polygon, np.int32)], 255)
        maske = build_lane_mask(img, lane, [])

        anteil = 100.0 * (maske > 0).sum() / max((roi > 0).sum(), 1)
        st.row(frame=i, weiss_in_roi_prozent=round(anteil, 2),
               weiss_gesamt=int((weiss > 0).sum()),
               roi_flaeche=int((roi > 0).sum()))

        # Links das Original mit ROI-Umriss, rechts die Maske.
        links = img.copy()
        cv2.polylines(links, [np.array(lane.roi_polygon, np.int32)], True,
                      (90, 90, 90), 1)
        # Benannte Zwischenbilder beantworten getrennt: Was war der Eingang,
        # was bestand die Helligkeitsschwelle und was blieb durch die ROI?
        st.snapshot(i, "01_original_roi", links, "Original + ROI")
        st.snapshot(i, "02_hls_lightness", hls[:, :, 1],
                    "HLS-L: Eingang der Weiss-Schwelle")
        st.snapshot(i, "03_white_threshold", weiss,
                    f"{lane.white_l_min} <= HLS-L <= {lane.white_l_max}")
        st.snapshot(i, "04_roi_mask", roi, "ROI-Maske")
        st.snapshot(i, "05_lane_mask", maske, "Weiss UND ROI")
        st.show(i, np.hstack([links, cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)]),
                f"f{i}  weiss in ROI {anteil:.2f} %")
    st.finish()


if __name__ == "__main__":
    main()

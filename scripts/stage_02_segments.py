"""Stufe 2 - Kanten und Hough-Segmente.

Aus der Maske werden Kanten und daraus gerade Segmente. Zu wenige
Segmente heisst: die Hough-Schwellen sind zu streng oder die Maske ist
leer. Sehr viele heisst: Textur und Grafik kommen mit durch.

Stellschrauben: lane.canny_*, lane.hough_threshold, hough_min_len, hough_max_gap

    python scripts/stage_02_segments.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.detection import build_masked_edges, extract_segments


def main() -> None:
    args = parse(__doc__, stufe="stage_02_segments")
    st = Stage(args, "stage_02_segments",
               "lane.canny_*, lane.hough_threshold, hough_min_len, hough_max_gap")
    for i, name, img, lane in st.frames():
        kanten = build_masked_edges(img, lane)
        segmente = extract_segments(kanten, lane)

        raw = cv2.HoughLinesP(
            kanten, 1, np.pi / 180, lane.hough_threshold,
            minLineLength=lane.hough_min_len, maxLineGap=lane.hough_max_gap)

        # Segment ist (x_bottom, m, (x1, y1, x2, y2)).
        laengen = [float(np.hypot(r[2] - r[0], r[3] - r[1]))
                   for _, _, r in segmente]
        st.row(frame=i, n_segmente=len(segmente),
               kantenpixel=int((kanten > 0).sum()),
               laenge_median=round(float(np.median(laengen)), 1) if laengen else 0.0)

        bild = cv2.cvtColor(kanten, cv2.COLOR_GRAY2BGR)
        kandidaten = img.copy()
        if raw is not None:
            for x1, y1, x2, y2 in raw[:, 0, :]:
                dx, dy = float(x2 - x1), float(y2 - y1)
                angle = 0.0 if dx == dy == 0 else abs(np.degrees(np.arctan2(dy, dx)))
                angle = min(angle, 180 - angle)
                farbe = (80, 80, 80) if angle >= lane.min_line_angle_deg else (0, 0, 255)
                cv2.line(kandidaten, (x1, y1), (x2, y2), farbe, 1)
        for _, _, (x1, y1, x2, y2) in segmente:
            cv2.line(bild, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.line(kandidaten, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 255, 0), 2)
        st.snapshot(i, "01_edges", kanten, "Canny nach Weissmaske + ROI")
        st.snapshot(i, "02_hough_filter", kandidaten,
                    "Hough: gruen akzeptiert, rot Winkel verworfen")
        st.show(i, bild, f"f{i}  {len(segmente)} Segmente")
    st.finish()


if __name__ == "__main__":
    main()

"""Stufe 4 - Randlinienpaar und Homographie.

Die Homographie wird aus genau zwei Linien aufgespannt. Findet sich
kein Paar, ist der Frame stumm -- alle folgenden Stufen entfallen.

`basis_breite` ist der Abstand der beiden Stuetzpunkte am unteren
Bildrand. Weil die BEV-Skala darauf normiert ist, bedeutet jede
Aenderung eine Neuskalierung der gesamten Bodenebene: `basis_delta`
ist damit das Wackeln in Zahlen. Die Glaettung senkt es.

Stellschrauben: pipeline.homography_smoothing, homography_max_point_jump,
homography_max_width_change_ratio, homography_max_vanishing_jump,
lane.y_top/y_bottom, bev.min_pair_separation

    python scripts/stage_04_homography.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.bev import outer_solid_pair
from adascope.lanes.detection import detect_lanes
from adascope.lanes.pipeline import HomographyTracker


def main() -> None:
    args = parse(__doc__, stufe="stage_04_homography")
    st = Stage(args, "stage_04_homography",
               "pipeline.homography_smoothing, homography_max_*_jump, "
               "lane.y_top/y_bottom, bev.min_pair_separation")
    tracker = HomographyTracker(st.settings.lane, st.settings.bev,
                                st.settings.pipeline.max_hold,
                                smoothing=st.settings.pipeline.homography_smoothing,
                                max_point_jump=st.settings.pipeline.homography_max_point_jump,
                                max_width_change_ratio=st.settings.pipeline.homography_max_width_change_ratio,
                                max_vanishing_jump=st.settings.pipeline.homography_max_vanishing_jump,
                                max_top_width_ratio=st.settings.pipeline.homography_max_top_width_ratio,
                                min_pair_continuity=st.settings.pipeline.homography_min_pair_continuity,
                                min_pair_support=st.settings.pipeline.homography_min_pair_support)
    vorher = None
    for i, name, img, lane in st.frames():
        tracker.lane = lane
        ergebnis = detect_lanes(img, lane)
        paar = outer_solid_pair(ergebnis, lane, st.settings.bev)
        H, zustand = tracker.update(ergebnis)

        kandidat_breite = (paar[1].x_bottom - paar[0].x_bottom) if paar else 0.0
        akzeptiert = tracker.accepted_src
        breite = float(akzeptiert[1, 0] - akzeptiert[0, 0]) if akzeptiert is not None else 0.0
        delta = abs(breite - vorher) if (breite and vorher is not None) else 0.0
        if zustand == "fresh" and breite:
            vorher = breite
        metric = tracker.metrics
        st.row(frame=i, zustand=zustand, paar_gefunden=int(paar is not None),
               ablehnungsgrund=tracker.last_rejection,
               kandidat_breite=round(kandidat_breite, 1),
               basis_breite=round(breite, 1), basis_delta=round(delta, 1),
               punkt_sprung=round(metric.get("point_jump", 0.0), 1),
               breite_delta=round(metric.get("width_change_ratio", 0.0), 3),
               fluchtpunkt_sprung=round(metric.get("vanishing_jump", 0.0), 1),
               paar_continuity=round(metric.get("pair_continuity", 0.0), 2),
               paar_support=round(metric.get("pair_support", 0.0)),
               gehalten=tracker.held_frames)

        bild = img.copy()
        # Alle nicht gewaehlten Kandidaten bleiben sichtbar. So ist sofort
        # erkennbar, ob eine Gegenfahrbahnkante angeboten, aber korrekt
        # verworfen wurde.
        gewaehlt = {id(L) for L in paar} if paar else set()
        for L in ergebnis.lines:
            if L.role.endswith("solid") and id(L) not in gewaehlt:
                cv2.line(bild, (L.x_at(lane.y_top), lane.y_top),
                         (L.x_at(lane.y_bottom), lane.y_bottom), (100, 100, 100), 2)
                cv2.putText(bild, "verworfen", (L.x_at(lane.y_bottom), lane.y_bottom - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
                            cv2.LINE_AA)
        if paar and tracker.last_rejection:
            for L in paar:
                cv2.line(bild, (L.x_at(lane.y_top), lane.y_top),
                         (L.x_at(lane.y_bottom), lane.y_bottom), (0, 0, 255), 2)

        # Das Gitter zeigt die TATSAECHLICH verwendeten (ggf. gehaltenen und
        # geglaetteten) Stuetzpunkte, nicht nur den angebotenen Kandidaten.
        if akzeptiert is not None:
            bl, br, tr, tl = akzeptiert
            for y in np.linspace(lane.y_top, lane.y_bottom, 7):
                t = (float(y) - lane.y_top) / max(lane.y_bottom - lane.y_top, 1)
                left = tl * (1 - t) + bl * t
                right = tr * (1 - t) + br * t
                cv2.line(bild, tuple(np.rint(left).astype(int)),
                         tuple(np.rint(right).astype(int)), (90, 90, 90), 1)
            for point, farbe in zip((bl, br, tr, tl),
                                    ((255, 120, 0), (0, 120, 255),
                                     (0, 120, 255), (255, 120, 0))):
                cv2.circle(bild, tuple(np.rint(point).astype(int)), 5, farbe, -1)
            cv2.line(bild, tuple(np.rint(bl).astype(int)), tuple(np.rint(tl).astype(int)),
                     (255, 120, 0), 3)
            cv2.line(bild, tuple(np.rint(br).astype(int)), tuple(np.rint(tr).astype(int)),
                     (0, 120, 255), 3)
        status = zustand + (f" ({tracker.last_rejection})" if tracker.last_rejection else "")
        st.snapshot(i, "01_support_geometry", bild,
                    f"Homographie: {status}, {st.settings.bev.boundary_pair_strategy}")
        st.show(i, bild, f"f{i}  H: {status}  Basis {breite:.0f} px  "
                         f"Delta {delta:.1f}")
    st.finish()


if __name__ == "__main__":
    main()

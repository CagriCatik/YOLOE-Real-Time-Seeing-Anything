"""Stufe 3 - Cluster, Linienfit, Rollen und Durchgezogenheit.

Segmente werden zu Linien zusammengefasst und bekommen eine Rolle.

WICHTIG: Die Rollennamen sind POSITIONELL. `solid` heisst nur 'weiter
aussen als die ego-naechste', nicht 'durchgezogen'. Was tatsaechlich
gemessen wird, steht in `continuity` -- Anteil der y-Spanne, den die
Segmente abdecken. Beide werden hier nebeneinander gezeigt.

Kreuzende Linien werden verworfen: zwei Spurgrenzen schneiden sich
nicht. Die Spalte `verworfen` zaehlt sie.

Stellschrauben: lane.cluster_method, cluster_max_dist, cluster_max_slope_diff,
cluster_max_lateral_gap, cluster_max_top_dist, cluster_max_y_gap,
cluster_vanishing_x_tolerance, min_cluster_support, robust_trim

    python scripts/stage_03_lines.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.detection import (
    build_masked_edges, cluster_segments, drop_crossing_lines, extract_segments,
    fit_lanes, classify_lanes,
    segment_incompatibility,
)

FARBE = {"left_solid": (0, 170, 255), "right_solid": (0, 170, 255),
         "left_dashed": (0, 255, 0), "right_dashed": (0, 255, 0),
         "unknown": (0, 0, 255)}


def main() -> None:
    args = parse(__doc__, stufe="stage_03_lines")
    st = Stage(args, "stage_03_lines",
               "lane.cluster_method, cluster_max_dist, cluster_max_slope_diff, "
               "cluster_max_lateral_gap, cluster_max_top_dist, "
               "cluster_vanishing_x_tolerance, cluster_max_y_gap")
    for i, name, img, lane in st.frames():
        segmente = extract_segments(build_masked_edges(img, lane), lane)
        cluster = cluster_segments(segmente, lane)
        roh = fit_lanes(cluster, lane)
        linien = drop_crossing_lines(roh, lane)
        ergebnis = classify_lanes(linien, lane)

        ablehnungen = Counter(segment_incompatibility(a, b, lane)
                              for a, b in combinations(segmente, 2))
        kompatibel = ablehnungen.pop("", 0)

        cont = [L.continuity for L in linien]
        st.row(frame=i, cluster_methode=lane.cluster_method,
               n_cluster=len(cluster), n_linien_roh=len(roh),
               segment_paare=sum(ablehnungen.values()) + kompatibel,
               kompatible_paare=kompatibel,
               reject_slope=ablehnungen["slope"],
               reject_lokal=ablehnungen["lateral_gap"],
               reject_flucht=ablehnungen["vanishing_projection"],
               reject_fluchtregion=ablehnungen["vanishing_region"],
               verworfen=len(roh) - len(linien),
               n_solid=sum(1 for L in linien if L.role.endswith("solid")),
               n_dashed=sum(1 for L in linien if L.role.endswith("dashed")),
               continuity_median=round(float(np.median(cont)), 2) if cont else 0.0,
               continuity_min=round(float(min(cont)), 2) if cont else 0.0)

        bild = img.copy()
        cluster_bild = img.copy()
        palette = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
                   (255, 255, 80), (255, 80, 255), (80, 255, 255)]
        for k, gruppe in enumerate(cluster):
            farbe = palette[k % len(palette)]
            for _, _, (x1, y1, x2, y2) in gruppe:
                cv2.line(cluster_bild, (x1, y1), (x2, y2), farbe, 2)
            if gruppe:
                x, _, (_, y, _, _) = gruppe[0]
                cv2.putText(cluster_bild, f"C{k} n={len(gruppe)}", (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, farbe, 1, cv2.LINE_AA)

        fits = img.copy()
        behalten = {id(L) for L in linien}
        for L in roh:
            farbe = (0, 255, 0) if id(L) in behalten else (0, 0, 255)
            cv2.line(fits, (L.x_at(lane.y_top), lane.y_top),
                     (L.x_at(lane.y_bottom), lane.y_bottom), farbe, 2)
        for L in ergebnis.lines:
            p0 = (L.x_at(lane.y_top), lane.y_top)
            p1 = (L.x_at(lane.y_bottom), lane.y_bottom)
            cv2.line(bild, p0, p1, FARBE.get(L.role, (200, 200, 200)), 2)
            cv2.putText(bild, f"{L.role} c={L.continuity:.2f} s{L.support}",
                        (p1[0] - 60, lane.y_bottom + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, FARBE.get(L.role,
                        (200, 200, 200)), 1, cv2.LINE_AA)
        st.snapshot(i, "01_clusters", cluster_bild,
                    f"{lane.cluster_method}: Clusterfarben; n = Segment-Support")
        st.snapshot(i, "02_fits", fits,
                    "Fits: gruen behalten, rot kreuzend verworfen")
        st.show(i, bild, f"f{i}  {lane.cluster_method}  {len(linien)} Linien, "
                         f"{len(roh) - len(linien)} kreuzend verworfen")
    st.finish()


if __name__ == "__main__":
    main()

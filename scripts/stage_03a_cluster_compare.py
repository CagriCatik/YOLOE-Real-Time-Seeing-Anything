"""Stufe 3a - Greedy- gegen Union-Find-Clustering direkt vergleichen.

Beide Verfahren bekommen exakt dieselben Hough-Segmente. Links steht das alte,
reihenfolgeabhaengige Greedy-Verfahren, rechts das produktive geometrische
Union-Find-Verfahren. Farben bezeichnen Cluster; weisse Geraden sind die Fits,
die nach Support- und Kreuzungsfilter uebrig bleiben.

Stellschrauben: lane.cluster_method, cluster_max_dist,
cluster_max_slope_diff, cluster_max_lateral_gap, cluster_max_top_dist,
cluster_vanishing_x_tolerance, cluster_max_y_gap
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.lanes.detection import (
    build_masked_edges, cluster_segments, drop_crossing_lines,
    extract_segments, fit_lanes,
)

PALETTE = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
           (255, 255, 80), (255, 80, 255), (80, 255, 255),
           (180, 120, 255), (120, 255, 180)]


def render(img, segments, lane, method):
    cfg = replace(lane, cluster_method=method)
    clusters = cluster_segments(segments, cfg)
    raw = fit_lanes(clusters, cfg)
    lines = drop_crossing_lines(raw, cfg)
    out = img.copy()
    for index, group in enumerate(clusters):
        color = PALETTE[index % len(PALETTE)]
        for _, _, (x1, y1, x2, y2) in group:
            cv2.line(out, (x1, y1), (x2, y2), color, 2)
        if group:
            x = round(float(np.median([item[0] for item in group])))
            y = min(min(item[2][1], item[2][3]) for item in group)
            cv2.putText(out, f"C{index} n={len(group)}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    for line in lines:
        cv2.line(out, (line.x_at(lane.y_top), lane.y_top),
                 (line.x_at(lane.y_bottom), lane.y_bottom), (255, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(out, f"{method}: {len(clusters)} Cluster -> {len(lines)} Fits | KANDIDATEN",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)
    return out, clusters, raw, lines


def main() -> None:
    args = parse(__doc__, stufe="stage_03a_cluster_compare")
    st = Stage(args, "stage_03a_cluster_compare",
               "lane.cluster_max_*, cluster_vanishing_x_tolerance")
    for i, name, img, lane in st.frames():
        segments = extract_segments(build_masked_edges(img, lane), lane)
        old, old_clusters, old_raw, old_lines = render(img, segments, lane, "greedy")
        new, new_clusters, new_raw, new_lines = render(
            img, segments, lane, "union_find")

        st.row(frame=i, n_segmente=len(segments),
               greedy_cluster=len(old_clusters), union_cluster=len(new_clusters),
               greedy_fits=len(old_lines), union_fits=len(new_lines),
               greedy_kreuzend=len(old_raw) - len(old_lines),
               union_kreuzend=len(new_raw) - len(new_lines),
               cluster_delta=len(new_clusters) - len(old_clusters),
               fit_delta=len(new_lines) - len(old_lines))
        st.snapshot(i, "01_greedy", old)
        st.snapshot(i, "02_union_find", new)
        st.show(i, np.hstack([old, new]))
    st.finish()


if __name__ == "__main__":
    main()

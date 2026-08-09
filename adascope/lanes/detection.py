"""
Klassische Fahrspurerkennung (OpenCV) für die HCP3-HMI-Lane-Change-Ansicht.

Pipeline (rein funktional, Seiteneffekte nur beim Rendern/IO):

    Bild ─▶ White-Mask + ROI ─▶ Hough-Segmente ─▶ Steigungsfilter
         ─▶ 2D-Clustering (x_bottom, Steigung) ─▶ Liniennfit x = m·y + b
         ─▶ Rollenzuordnung (links/ego/rechts) ─▶ Overlay

Die gesamte Kalibrierung steckt in `LaneConfig` (config/lane.yaml). Für die
Portierung auf ein anderes Fahrzeugprojekt wird nur diese Datei ausgetauscht,
der Pipeline-Code bleibt unverändert.

Dieses Modul rendert nicht und schreibt nichts — Overlays liegen in
`adascope.render`, das Ausführen in `adascope.cli`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from ..config import LaneConfig

Role = Literal["left_solid", "left_dashed", "right_dashed", "right_solid", "unknown"]


@dataclass
class LaneLine:
    m: float          # x = m·y + b
    b: float
    x_bottom: float
    support: int
    role: Role = "unknown"

    def x_at(self, y: float) -> int:
        return int(self.m * y + self.b)


@dataclass
class LaneResult:
    lines: list[LaneLine]
    ego_left: LaneLine | None = None
    ego_right: LaneLine | None = None
    debug: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage 1 – Masken                                                            #
# --------------------------------------------------------------------------- #
def build_masked_edges(img: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    h, w = img.shape[:2]
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls[:, :, 1], cfg.white_l_min, cfg.white_l_max)

    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [np.array(cfg.roi_polygon, np.int32)], 255)

    masked = cv2.bitwise_and(white, roi)
    return cv2.Canny(masked, 50, 150)


# --------------------------------------------------------------------------- #
# Stage 2 – Segmente + Steigungsfilter                                        #
# --------------------------------------------------------------------------- #
def extract_segments(edges: np.ndarray, cfg: LaneConfig) -> list[tuple]:
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180, cfg.hough_threshold,
        minLineLength=cfg.hough_min_len, maxLineGap=cfg.hough_max_gap,
    )
    if raw is None:
        return []

    segments = []
    for x1, y1, x2, y2 in raw[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dy == 0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)
        if angle < cfg.min_line_angle_deg:      # Fahrzeugdach / Horizont raus
            continue
        m = dx / dy                              # x = m·y + b (stabil für vertikale)
        x_bottom = m * (cfg.y_bottom - y1) + x1
        segments.append((x_bottom, m, (x1, y1, x2, y2)))
    return segments


# --------------------------------------------------------------------------- #
# Stage 3 – Clustering in (x_bottom, Steigung)                                #
# --------------------------------------------------------------------------- #
def cluster_segments(segments: list[tuple], cfg: LaneConfig) -> list[list[tuple]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s[0])
    clusters: list[list[tuple]] = [[segments[0]]]
    for seg in segments[1:]:
        ref = clusters[-1][-1]
        dist = abs(seg[0] - ref[0]) + cfg.cluster_slope_weight * abs(seg[1] - ref[1])
        if dist < cfg.cluster_max_dist:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])
    return clusters


# --------------------------------------------------------------------------- #
# Stage 4 – Fit pro Cluster                                                   #
# --------------------------------------------------------------------------- #
def robust_line(pts: np.ndarray, cfg: LaneConfig) -> tuple[float, float]:
    """Ausgleichsgerade x = m·y + b, gegen Ausreisser gehärtet.

    Zwei Durchgänge: erst Kleinstquadrate über alle Punkte, dann die
    `robust_trim`-Fraktion mit dem größten Abstand verwerfen und neu fitten.
    Ein einzelner Leitplankenreflex oder eine Fahrzeugkontur im Cluster
    verschiebt sonst `x(y_bottom)` um zweistellige Pixelbeträge — und genau
    dieser Punkt speist die Homographie.
    """
    y, x = pts[:, 1], pts[:, 0]
    m, b = np.polyfit(y, x, 1)
    keep_count = int(round(len(y) * (1 - cfg.robust_trim)))
    if cfg.robust_trim <= 0 or len(y) < cfg.robust_min_points or keep_count < 3:
        return float(m), float(b)
    keep = np.argsort(np.abs(x - (m * y + b)))[:keep_count]
    m, b = np.polyfit(y[keep], x[keep], 1)
    return float(m), float(b)


def fit_lanes(clusters: list[list[tuple]], cfg: LaneConfig) -> list[LaneLine]:
    lanes: list[LaneLine] = []
    for cluster in clusters:
        if len(cluster) < cfg.min_cluster_support:
            continue
        pts = np.array(
            [p for _, _, seg in cluster
             for p in ((seg[0], seg[1]), (seg[2], seg[3]))]
        )
        m, b = robust_line(pts, cfg)                   # x = m·y + b
        lanes.append(LaneLine(m=m, b=b, x_bottom=m * cfg.y_bottom + b,
                              support=len(cluster)))
    return sorted(lanes, key=lambda L: L.x_bottom)


# --------------------------------------------------------------------------- #
# Stage 5 – Rollenzuordnung relativ zum Ego-Referenzpunkt                     #
# --------------------------------------------------------------------------- #
def classify_lanes(lanes: list[LaneLine], cfg: LaneConfig) -> LaneResult:
    left = [L for L in lanes if L.x_bottom < cfg.ego_x_bottom]
    right = [L for L in lanes if L.x_bottom >= cfg.ego_x_bottom]

    ego_left = max(left, key=lambda L: L.x_bottom) if left else None
    ego_right = min(right, key=lambda L: L.x_bottom) if right else None

    for L in lanes:
        if L is ego_left:
            L.role = "left_dashed"
        elif L is ego_right:
            L.role = "right_dashed"
        elif ego_left and L.x_bottom < ego_left.x_bottom:
            L.role = "left_solid"
        elif ego_right and L.x_bottom > ego_right.x_bottom:
            L.role = "right_solid"

    return LaneResult(lines=lanes, ego_left=ego_left, ego_right=ego_right)


def detect_lanes(img: np.ndarray, cfg: LaneConfig = LaneConfig()) -> LaneResult:
    edges = build_masked_edges(img, cfg)
    segments = extract_segments(edges, cfg)
    clusters = cluster_segments(segments, cfg)
    lanes = fit_lanes(clusters, cfg)
    result = classify_lanes(lanes, cfg)
    result.debug = {"n_segments": len(segments), "n_clusters": len(clusters)}
    return result

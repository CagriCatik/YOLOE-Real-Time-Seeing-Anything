"""
Klassische Fahrspurerkennung (OpenCV) für die HCP3-HMI-Lane-Change-Ansicht.

Pipeline (rein funktional, Seiteneffekte nur beim Rendern/IO):

    Bild ─▶ White-Mask + ROI ─▶ Hough-Segmente ─▶ Steigungsfilter
         ─▶ 2D-Clustering (x_bottom, Steigung) ─▶ Liniennfit x = m·y + b
         ─▶ Rollenzuordnung (links/ego/rechts) ─▶ Overlay

Die gesamte Kalibrierung steckt in `LaneConfig` (configs/lane.yaml). Für die
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
    # GEMESSENE Durchgezogenheit: Anteil der y-Spanne, den die Segmente des
    # Clusters tatsaechlich abdecken. 1.0 = luekenlos, kleine Werte = Striche.
    #
    # Bis hierher waren `solid` und `dashed` reine POSITIONSANGABEN -- keine
    # Stufe hat je geprueft, ob eine Linie durchgezogen ist. Das sah wie eine
    # Messung aus und war keine.
    continuity: float = 1.0

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
    return cv2.Canny(masked, cfg.canny_low, cfg.canny_high)


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
def _greedy_clusters(segments: list[tuple], cfg: LaneConfig) -> list[list[tuple]]:
    """Historisches Verfahren fuer reproduzierbare A/B-Vergleiche."""
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


def _segment_model(segment: tuple) -> tuple[float, float, float, float]:
    """(m, b, y_min, y_max) fuer ``x = m*y + b``."""
    _, m, (x1, y1, _x2, y2) = segment
    b = float(x1) - float(m) * float(y1)
    return float(m), b, float(min(y1, y2)), float(max(y1, y2))


def segment_incompatibility(a: tuple, b: tuple, cfg: LaneConfig) -> str:
    """Leerer String bei Kompatibilitaet, sonst der erste Ablehnungsgrund.

    Neben Steigung und ``x_bottom`` werden der lokale Querabstand, die
    vertikale Luecke und die Projektion bei ``y_top`` geprueft. Letztere ist
    die Fluchtpunkt-Kompatibilitaet: lokal aneinanderstossende Segmente, die in
    unterschiedliche Fernrichtungen zeigen, bilden keinen Bruecken-Cluster.
    """
    if (a[0] < cfg.ego_x_bottom) != (b[0] < cfg.ego_x_bottom):
        return "side"
    ma, ba, alo, ahi = _segment_model(a)
    mb, bb, blo, bhi = _segment_model(b)
    if abs(ma - mb) > cfg.cluster_max_slope_diff:
        return "slope"
    if abs(float(a[0]) - float(b[0])) > cfg.cluster_max_dist:
        return "bottom_projection"
    top_a, top_b = ma * cfg.y_top + ba, mb * cfg.y_top + bb
    top_points = sorted(cfg.roi_polygon, key=lambda point: point[1])[:2]
    vanishing_x = sum(point[0] for point in top_points) / len(top_points)
    if (cfg.cluster_vanishing_x_tolerance > 0
            and (abs(top_a - vanishing_x) > cfg.cluster_vanishing_x_tolerance
                 or abs(top_b - vanishing_x) > cfg.cluster_vanishing_x_tolerance)):
        return "vanishing_region"
    if abs(top_a - top_b) > cfg.cluster_max_top_dist:
        return "vanishing_projection"

    overlap_lo, overlap_hi = max(alo, blo), min(ahi, bhi)
    if overlap_lo <= overlap_hi:
        y_probe, y_gap = (overlap_lo + overlap_hi) / 2.0, 0.0
    elif ahi < blo:
        y_probe, y_gap = (ahi + blo) / 2.0, blo - ahi
    else:
        y_probe, y_gap = (bhi + alo) / 2.0, alo - bhi
    if y_gap > cfg.cluster_max_y_gap:
        return "vertical_gap"
    lateral = abs((ma * y_probe + ba) - (mb * y_probe + bb))
    return "" if lateral <= cfg.cluster_max_lateral_gap else "lateral_gap"


def segments_compatible(a: tuple, b: tuple, cfg: LaneConfig) -> bool:
    """Ob zwei Hough-Segmente dieselbe physische Markierung beschreiben."""
    return not segment_incompatibility(a, b, cfg)


def _union_find_clusters(segments: list[tuple], cfg: LaneConfig) -> list[list[tuple]]:
    """Deterministische Zusammenhangskomponenten kompatibler Segmente."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (float(s[0]), float(s[1]), tuple(s[2])))
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i, first in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            second = ordered[j]
            if second[0] - first[0] > cfg.cluster_max_dist:
                break
            if segments_compatible(first, second, cfg):
                union(i, j)

    grouped: dict[int, list[tuple]] = {}
    for i, segment in enumerate(ordered):
        grouped.setdefault(find(i), []).append(segment)
    return sorted(grouped.values(),
                  key=lambda group: float(np.median([s[0] for s in group])))


def cluster_segments(segments: list[tuple], cfg: LaneConfig) -> list[list[tuple]]:
    """Cluster mit konfigurierbarem, standardmaessig deterministischem Verfahren."""
    if cfg.cluster_method == "greedy":
        return _greedy_clusters(segments, cfg)
    return _union_find_clusters(segments, cfg)


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


def measure_continuity(cluster: list[tuple]) -> float:
    """Anteil der y-Spanne eines Clusters, der von Segmenten belegt ist.

    Eine durchgezogene Linie deckt ihre Spanne nahezu vollstaendig ab, eine
    gestrichelte laesst Luecken. Das ist der Unterschied, den die Rollennamen
    bisher behauptet, aber nie gemessen haben.
    """
    # Index 1 und 3 sind die y-Koordinaten; das Segment ist (x1, y1, x2, y2).
    # Zuerst standen hier 0 und 2 -- damit wurde die x-Spanne gemessen und die
    # Durchgezogenheit kam praktisch immer als 1.00 heraus.
    spans = sorted((min(seg[1], seg[3]), max(seg[1], seg[3]))
                   for _, _, seg in cluster)
    if not spans:
        return 0.0
    gesamt = spans[-1][1] - spans[0][0]
    if gesamt <= 0:
        return 1.0
    belegt, ende = 0.0, spans[0][0]
    for lo, hi in spans:                       # ueberlappende Segmente vereinen
        if hi <= ende:
            continue
        belegt += hi - max(lo, ende)
        ende = hi
    return min(belegt / gesamt, 1.0)


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
                              support=len(cluster),
                              continuity=measure_continuity(cluster)))
    return sorted(lanes, key=lambda L: L.x_bottom)


# --------------------------------------------------------------------------- #
# Stage 4b - Geometrisch unmoegliche Linien verwerfen                          #
# --------------------------------------------------------------------------- #
def drop_crossing_lines(lanes: list[LaneLine], cfg: LaneConfig) -> list[LaneLine]:
    """Linien entfernen, die sich innerhalb der ROI schneiden.

    Zwei Spurgrenzen kreuzen sich nicht. Tun zwei gefittete Linien es doch, ist
    mindestens eine keine Markierung -- in HMI-Aufnahmen typischerweise die
    Trapezkante der Darstellung selbst, die diagonal durchs Bild laeuft.

    Genau das war im Debugvideo zu sehen: eine als `right_solid` gefuehrte Linie
    verlief von oben Mitte quer durch das Ego-Fahrzeug nach unten links und
    kreuzte dabei alle anderen. Weil `right_solid` positionell "am weitesten
    aussen" heisst, gewann ausgerechnet sie die Rolle -- und wurde Stuetzpunkt
    der Homographie.

    Aufloesung nach Support: die Linie mit den wenigeren Clustersegmenten
    fliegt. Eine echte Markierung hat mehr Belege als eine Bildkante, die nur
    an wenigen Stellen als hell durchgeht.
    """
    if len(lanes) < 2:
        return lanes
    y0, y1 = float(cfg.y_top), float(cfg.y_bottom)
    konflikte: dict[int, int] = {}
    for i, a in enumerate(lanes):
        for j in range(i + 1, len(lanes)):
            b = lanes[j]
            oben = a.m * y0 + a.b - (b.m * y0 + b.b)
            unten = a.m * y1 + a.b - (b.m * y1 + b.b)
            if oben == 0.0 or unten == 0.0 or (oben < 0) != (unten < 0):
                konflikte[i] = konflikte.get(i, 0) + 1
                konflikte[j] = konflikte.get(j, 0) + 1
    if not konflikte:
        return lanes
    # Wiederholt die schlechteste Linie entfernen, bis nichts mehr kreuzt.
    # Ein einzelner Durchgang genuegt nicht: eine Diagonale kreuzt mehrere,
    # und nach ihrem Entfernen sind die uebrigen wieder in Ordnung.
    schlechteste = min(konflikte, key=lambda i: (lanes[i].support, -konflikte[i]))
    rest = [L for k, L in enumerate(lanes) if k != schlechteste]
    return drop_crossing_lines(rest, cfg)


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
    lanes = drop_crossing_lines(lanes, cfg)
    result = classify_lanes(lanes, cfg)
    result.debug = {"n_segments": len(segments), "n_clusters": len(clusters),
                    "cluster_method": cfg.cluster_method}
    return result

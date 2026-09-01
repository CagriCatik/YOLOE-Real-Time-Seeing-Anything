"""Kalibrierung der Spurerkennung in der Bildebene.

Diese Werte waren als Dataclass-Defaults in `lane_detection.py` verdrahtet und
auf genau einen Screenshot getuned -- inklusive Letterbox-Balken. Sie gehoeren
in eine Datei, weil sie sich pro Kamera, pro HMI-Layout und pro Aufloesung
aendern, der Pipeline-Code aber nicht.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .loader import as_int_pairs, require_positive, require_range

# Trapez der Strassenflaeche in Pixeln, zum Fluchtpunkt konvergierend.
DEFAULT_ROI_POLYGON = ((15, 300), (470, 20), (760, 20), (1195, 295))


@dataclass(frozen=True)
class LaneConfig:
    """Kalibrierflaeche von `lanes.detection`."""

    # Aufloesung, in deren Pixeln die Werte unten gemessen wurden. Weicht ein
    # Frame davon ab, skaliert `scaled_to()` die Laengen mit -- aber nur, wenn
    # das SEITENVERHAELTNIS passt. Bei anderem Seitenverhaeltnis zeigt das Bild
    # einen anderen Ausschnitt, und Skalieren legt die ROI ueber falschen
    # Inhalt; dann muss vorher zugeschnitten werden (`adascope crop`).
    reference_size: tuple[int, int] = (1209, 457)
    roi_polygon: tuple[tuple[int, int], ...] = DEFAULT_ROI_POLYGON
    # Weiss-Extraktion ueber den HLS-L-Kanal
    white_l_min: int = 130
    white_l_max: int = 255
    # Canny auf der binaeren Weissmaske. Bislang standen 50/150 trotz der
    # dokumentierten ``lane.canny_*``-Stellschrauben fest im Algorithmus.
    canny_low: int = 50
    canny_high: int = 150
    # Hough
    hough_threshold: int = 30
    hough_min_len: int = 30
    hough_max_gap: int = 60
    # Referenzbaselines fuer Fit und Zeichnung
    # Ab dieser GEMESSENEN Durchgezogenheit gilt eine Linie als tauglicher
    # Stuetzpunkt der Homographie. Eine gestrichelte Linie als Stuetzpunkt
    # laesst die Bodenebene mit jedem Strich springen.
    solid_min_continuity: float = 0.55
    y_bottom: int = 295
    y_top: int = 55
    # Filter und Clustering
    min_line_angle_deg: float = 20.0     # verwirft ~horizontale (Fahrzeugdaecher)
    cluster_slope_weight: float = 500.0  # Gewicht der Steigung im Distanzmass
    cluster_max_dist: float = 130.0
    cluster_method: str = "union_find"
    cluster_max_slope_diff: float = 0.35
    cluster_max_lateral_gap: float = 28.0
    cluster_max_top_dist: float = 65.0
    # 0 deaktiviert das absolute Fluchtpunktband (z.B. fuer starke Kurven).
    cluster_vanishing_x_tolerance: float = 0.0
    cluster_max_y_gap: float = 100.0
    min_cluster_support: int = 2         # verwirft schwache Einzeldashes
    # Getrimmter Ausgleich: die Punkte mit dem groessten Abstand zur ersten
    # Ausgleichsgeraden werden verworfen und neu gefittet. Gemessen ueber 1573
    # Cluster: beim kritischen obersten Dezil faellt das Residuum von 19.2 auf
    # 3.1 px (-84 %), waehrend ein Polynom 2. Grades nur -16 % bringt -- die
    # Stoerung sind Ausreisser (Leitplanke, Schatten, Fahrzeugkontur), nicht
    # Kruemmung. Deterministisch statt RANSAC, weil der Ausreisseranteil klein
    # ist und ein reproduzierbares Ergebnis mehr wert ist als Zufallsproben.
    robust_trim: float = 0.25            # 0.0 schaltet ab (reiner Kleinstquadrate-Fit)
    robust_min_points: int = 6           # darunter lohnt das Trimmen nicht
    # Ego-Referenzpunkt (Kamera-/Bildmitte am unteren Rand)
    ego_x_bottom: float = 600.0

    def __post_init__(self) -> None:
        if self.y_top >= self.y_bottom:
            raise ValueError(f"y_top={self.y_top} muss ueber y_bottom={self.y_bottom} liegen")
        if self.white_l_min >= self.white_l_max:
            raise ValueError("white_l_min muss kleiner als white_l_max sein")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("lane.canny_low/high brauchen 0 <= low < high <= 255")
        require_range(self.min_line_angle_deg, 0, 90, "min_line_angle_deg")
        require_positive(self.cluster_max_dist, "cluster_max_dist")
        if self.cluster_method not in {"union_find", "greedy"}:
            raise ValueError("lane.cluster_method muss union_find oder greedy sein")
        for name in ("cluster_max_slope_diff", "cluster_max_lateral_gap",
                     "cluster_max_top_dist", "cluster_max_y_gap"):
            require_positive(getattr(self, name), f"lane.{name}")
        if self.cluster_vanishing_x_tolerance < 0:
            raise ValueError("lane.cluster_vanishing_x_tolerance darf nicht negativ sein")
        if self.min_cluster_support < 1:
            raise ValueError("min_cluster_support muss mindestens 1 sein")
        require_range(self.robust_trim, 0, 0.5, "lane.robust_trim")
        if self.robust_min_points < 3:
            raise ValueError("lane.robust_min_points muss mindestens 3 sein")

    # --- Aufloesungsanpassung ------------------------------------------- #
    # Zulaessige Abweichung des Seitenverhaeltnisses, bis zu der noch skaliert
    # wird. 1428x534 (2.674) gegen 1209x457 (2.646) sind 1 % -- derselbe
    # Ausschnitt, nur groesser. 1920x1080 (1.78) ist ein anderer Ausschnitt.
    ASPECT_TOLERANCE = 0.05

    def matches(self, width: int, height: int) -> bool:
        """Passt diese Kalibrierung ohne Zuschnitt auf ein Bild dieser Groesse?"""
        ref_w, ref_h = self.reference_size
        return abs((width / height) / (ref_w / ref_h) - 1) <= self.ASPECT_TOLERANCE

    def scaled_to(self, width: int, height: int) -> "LaneConfig":
        """Auf eine andere Bildgroesse umgerechnet.

        Skaliert werden alle Laengen in Bildpixeln. Nicht skaliert werden
        Winkel, Helligkeitsschwellen und Stueckzahlen -- die sind unabhaengig
        von der Aufloesung.
        """
        ref_w, ref_h = self.reference_size
        if (width, height) == (ref_w, ref_h):
            return self
        if not self.matches(width, height):
            raise ValueError(
                f"Kalibrierung fuer {ref_w}x{ref_h} passt nicht auf {width}x{height} "
                f"(Seitenverhaeltnis {width / height:.2f} statt {ref_w / ref_h:.2f}). "
                "Das Bild zeigt einen anderen Ausschnitt. Abhilfe: `--crop` "
                "anhaengen (schneidet auf detection.crop_box), oder eine eigene "
                "lane.yaml im Szenario hinterlegen.")
        sx, sy = width / ref_w, height / ref_h
        s = (sx + sy) / 2                      # fuer richtungslose Laengen
        return replace(
            self,
            reference_size=(width, height),
            roi_polygon=tuple((round(x * sx), round(y * sy)) for x, y in self.roi_polygon),
            hough_min_len=round(self.hough_min_len * s),
            hough_max_gap=round(self.hough_max_gap * s),
            y_bottom=round(self.y_bottom * sy),
            y_top=round(self.y_top * sy),
            cluster_slope_weight=self.cluster_slope_weight * s,
            cluster_max_dist=self.cluster_max_dist * s,
            cluster_max_lateral_gap=self.cluster_max_lateral_gap * s,
            cluster_max_top_dist=self.cluster_max_top_dist * s,
            cluster_vanishing_x_tolerance=self.cluster_vanishing_x_tolerance * sx,
            cluster_max_y_gap=self.cluster_max_y_gap * sy,
            ego_x_bottom=self.ego_x_bottom * sx,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LaneConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"lane: unbekannte Schluessel {sorted(unknown)}")
        values = dict(raw)
        if "roi_polygon" in values:
            values["roi_polygon"] = as_int_pairs(values["roi_polygon"], "lane.roi_polygon")
        if "reference_size" in values:
            size = values["reference_size"]
            if len(size) != 2:
                raise ValueError("lane.reference_size braucht [Breite, Hoehe]")
            values["reference_size"] = (int(size[0]), int(size[1]))
        return cls(**values)

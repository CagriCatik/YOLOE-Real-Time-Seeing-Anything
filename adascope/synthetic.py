"""Synthetische Fahrszenen mit bekannter Wahrheit.

Wozu
----
Auf echtem Material ist nicht entscheidbar, ob ein gemeldetes `cut_in` richtig
war -- es gibt keine Annotation. Hier wird der umgekehrte Weg gegangen: die
Trajektorie wird VORGEGEBEN, das Bild daraus erzeugt, und die Pipeline muss die
vorgegebenen Ereignisse zurueckliefern. Das prueft die ganze Kette

    Spurerkennung -> Homographie -> Korridore -> Indizierung -> State Machine

ohne Modell, ohne Annotation und in Millisekunden.

Wie
---
Nicht das Bild wird beschrieben, sondern die Bodenebene. Eine gewaehlte
Homographie `H` bildet Bild auf BEV ab; die Spurlinien werden durch
Rueckprojektion ihrer BEV-Positionen gezeichnet, Fahrzeuge durch
Rueckprojektion ihres Footprints. Damit ist jede laterale Position in
BEV-Pixeln exakt vorgegeben und nicht aus dem Bild geschaetzt.

Grenzen
-------
Die Szene ist gerade, gleichmaessig ausgeleuchtet und rauschfrei. Sie beweist,
dass die LOGIK stimmt -- nicht, dass die Spurerkennung auf echtem Material
haelt. Dafuer sind die annotierten Szenarien da (`ground_truth/`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import BevConfig, LaneConfig
from .detection import TrackedVehicle

# Strichmuster der gestrichelten Markierungen in BEV-Pixeln (Strich, Periode).
DASH_LENGTH, DASH_PERIOD = 26, 70


@dataclass
class SyntheticRoad:
    """Gerade Fahrbahn mit frei setzbaren Fahrzeugen.

    `lane_width` ist die Spurbreite in BEV-Pixeln. Die Randlinien liegen auf
    `bev.x_left` / `bev.x_right`, damit die von der Pipeline zurueckgerechnete
    Homographie mit der hier gewaehlten uebereinstimmt.
    """

    lane: LaneConfig = field(default_factory=LaneConfig)
    bev: BevConfig = field(default_factory=BevConfig)
    lanes: int = 3
    noise_sigma: float = 0.0          # Bildrauschen, um den Fit zu stoeren
    stray_marks: int = 0              # falsche helle Striche (Ausreisser)
    # Seitliche Verschiebung der Fahrbahn am FERNEN Ende, in BEV-Pixeln.
    # 0 = gerade. Der Verlauf ist quadratisch ueber die Tiefe, wie eine reale
    # Klothoide in guter Naeherung. Positiv = Rechtskurve.
    curvature: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.width, self.height = self.lane.reference_size
        self.lane_width = (self.bev.x_right - self.bev.x_left) / self.lanes
        self.H = self._homography()
        self.H_inv = np.linalg.inv(self.H)

    # -- Geometrie --------------------------------------------------------- #
    def _homography(self) -> np.ndarray:
        """Bild -> BEV. Frei gewaehlt, aber perspektivisch plausibel."""
        near, far = self.lane.y_bottom, self.lane.y_top
        src = np.float32([[120, near], [self.width - 120, near],
                          [self.width / 2 + 150, far], [self.width / 2 - 150, far]])
        dst = np.float32([[self.bev.x_left, self.bev.y_near],
                          [self.bev.x_right, self.bev.y_near],
                          [self.bev.x_right, self.bev.y_far],
                          [self.bev.x_left, self.bev.y_far]])
        return cv2.getPerspectiveTransform(src, dst)

    def offset_at(self, y_bev: float) -> float:
        """Seitliche Verschiebung der ganzen Fahrbahn bei dieser Tiefe.

        Quadratisch ueber die Tiefe: am Nahende null, am Fernende `curvature`.
        Damit ist die Kurve dort am staerksten, wo auch das Histogramm am
        ehesten verschmiert.
        """
        if not self.curvature:
            return 0.0
        span = self.bev.y_near - self.bev.y_far
        depth = max(0.0, min(1.0, (self.bev.y_near - y_bev) / span))
        return self.curvature * depth ** 2

    def boundary_x(self, index: int, y_bev: float | None = None) -> float:
        """BEV-x der `index`-ten Spurgrenze, von links ab 0.

        Ohne `y_bev` am Nahende -- dort, wo die Fahrzeuge stehen und wo die
        Trajektorien der Tests definiert sind.
        """
        base = self.bev.x_left + index * self.lane_width
        return base + self.offset_at(self.bev.y_near if y_bev is None else y_bev)

    def lane_center(self, index: int, y_bev: float | None = None) -> float:
        """BEV-x der Mitte von Spur `index` (0 = ganz links)."""
        return self.boundary_x(index, y_bev) + self.lane_width / 2

    def to_image(self, points_bev) -> np.ndarray:
        pts = np.asarray(points_bev, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    # -- Bild -------------------------------------------------------------- #
    def frame(self) -> np.ndarray:
        """Ein Frame der leeren Fahrbahn.

        Die aeusseren Linien sind durchgezogen (daraus rechnet die Pipeline die
        Homographie), die inneren gestrichelt -- wie auf der Autobahn und wie es
        das Spaltenhistogramm erwartet.
        """
        image = np.full((self.height, self.width, 3), 30, np.uint8)
        for index in range(self.lanes + 1):
            solid = index in (0, self.lanes)
            # Basis ohne Versatz -- den legt `_draw_boundary` je Tiefe drauf.
            self._draw_boundary(image, self.bev.x_left + index * self.lane_width,
                                solid)
        for _ in range(self.stray_marks):
            self._draw_stray(image)
        if self.noise_sigma:
            noise = self._rng.normal(0, self.noise_sigma, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        return image

    def _draw_boundary(self, image: np.ndarray, base_x: float, solid: bool) -> None:
        """Eine Grenze zeichnen -- als Polylinie, damit Kurven mitkommen."""
        spans = ([(self.bev.y_far, self.bev.y_near)] if solid else
                 [(y, min(y + DASH_LENGTH, self.bev.y_near))
                  for y in range(self.bev.y_far, self.bev.y_near, DASH_PERIOD)])
        for y0, y1 in spans:
            ys = np.linspace(y0, y1, max(int((y1 - y0) / 12) + 2, 2))
            points = self.to_image([[base_x + self.offset_at(y), y] for y in ys])
            cv2.polylines(image, [points.round().astype(np.int32)], False,
                          (235, 235, 235), 3, cv2.LINE_AA)

    def _draw_stray(self, image: np.ndarray) -> None:
        """Ein heller Strich, der zu keiner Spur gehoert.

        Modelliert Leitplankenreflexe, Schattenkanten und Fahrzeugkonturen --
        die Punkte, die den Ausgleichsgeraden-Fit verziehen.
        """
        x = self._rng.uniform(self.bev.x_left, self.bev.x_right)
        y = self._rng.uniform(self.bev.y_far, self.bev.y_near - 120)
        dx = self._rng.uniform(-45, 45)
        (ax, ay), (bx, by) = self.to_image([[x, y], [x + dx, y + 110]])
        cv2.line(image, (round(ax), round(ay)), (round(bx), round(by)),
                 (215, 215, 215), 3, cv2.LINE_AA)

    # -- Fahrzeuge ---------------------------------------------------------- #
    def vehicle(self, track_id: int, x_bev: float, y_bev: float,
                width_bev: float = 42.0, role: str = "co") -> TrackedVehicle:
        """Fahrzeug, dessen FOOTPRINT exakt auf [x-w/2, x+w/2] bei y liegt.

        Nur die Bbox-Unterkante wird von der Pipeline projiziert; die Oberkante
        ist frei und wird nur so gesetzt, dass eine plausible Box entsteht.
        """
        left, right = x_bev - width_bev / 2, x_bev + width_bev / 2
        (x1, y1), (x2, y2) = self.to_image([[left, y_bev], [right, y_bev]])
        bottom = round(max(y1, y2))
        box_width = abs(x2 - x1)
        top = round(bottom - max(box_width * 0.8, 12))
        return TrackedVehicle(track_id, 2, "car", 0.9,
                              (round(min(x1, x2)), top, round(max(x1, x2)), bottom), role)

    def ego(self, x_bev: float | None = None, y_bev: float | None = None) -> TrackedVehicle:
        return self.vehicle(0, self.lane_center(self.lanes // 2) if x_bev is None else x_bev,
                            self.bev.y_near - 40 if y_bev is None else y_bev, role="ego")


# --------------------------------------------------------------------------- #
# Trajektorien                                                                #
# --------------------------------------------------------------------------- #
def glide(start: float, end: float, frames: int) -> list[float]:
    """Gleichmaessiger Uebergang; `frames` Werte einschliesslich beider Enden."""
    return list(np.linspace(start, end, frames)) if frames > 1 else [end]


def hold(value: float, frames: int) -> list[float]:
    return [value] * frames


def path(*segments: list[float]) -> list[float]:
    """Segmente zu einer Trajektorie verketten."""
    return [value for segment in segments for value in segment]

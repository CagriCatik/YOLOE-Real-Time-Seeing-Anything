"""
Bodenebene: Homographie, Spurkorridore und Footprint-Projektion.

Warum BEV nur fuer die Fahrbahn und nicht fuer die Fahrzeuge
------------------------------------------------------------
Die Homographie ist ausschliesslich in der Bodenebene gueltig. Fahrzeuge haben
Bauhoehe -> beim Warpen zerlaufen sie radial vom Kamerapunkt weg und ueber-
decken in BEV mehrere Spuren. Eine Flaechensegmentierung des FAHRZEUGS im BEV
ist deshalb strukturell falsch und war in der ersten Version die Fehlerquelle.

Aufgabenteilung
---------------
    Bodenebene (BEV)  Fahrbahnbereiche / Spurkorridore   Homographie gueltig
    Bildebene         Fahrzeugdetektion (YOLO11-Bbox)    Homographie ungueltig
    Bruecke           NUR die Bbox-Unterkante            liegt in der Bodenebene

Ablauf
------
1. Bildebene: die beiden durchgezogenen Randlinien per Hough (hohe Stuetzzahl)
   -> Homographie H.
2. Fahrzeug-Bboxen aus der Weiss-Maske ausstanzen (Detektor unterdrueckt
   Falschpixel der Spurerkennung -> kooperierende Module).
3. Maske nach BEV warpen, Spaltenhistogramm -> ALLE Spurgrenzen. Gestrichelte
   Linien akkumulieren hier zu Saeulen; genau die kurzen Fern-Dashes, die
   Hough in der Bildebene verliert, werden so stabil.
4. Bbox-Unterkanten projizieren -> laterales Footprint-Segment.
5. Ueberlappung Footprint x Ego-Korridor -> KONTINUIERLICHE Einscher-Rate.

Die Einscher-Rate ist bewusst kein Boolean: sie ist der stetige Eingang fuer
die State Machine in `lanes.events`. Die Spurnummern liefert `lanes.indexing`,
das Verketten ueber Frames `lanes.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from ..config import BevConfig, LaneConfig
from .detection import LaneLine, LaneResult, detect_lanes


@dataclass
class Footprint:
    name: str
    x_left: float
    x_right: float
    y: float

    @property
    def width(self) -> float:
        return max(self.x_right - self.x_left, 1e-6)

    @property
    def center(self) -> float:
        return (self.x_left + self.x_right) / 2


# --------------------------------------------------------------------------- #
# Stufe 1 - Homographie aus den durchgezogenen Randlinien                     #
# --------------------------------------------------------------------------- #
# Mindestabstand der beiden Randlinien am unteren Bildrand, in Pixeln.
# Darunter ist die Homographie entartet.
MIN_PAIR_SEPARATION = 40.0


def outer_solid_pair(result: LaneResult) -> tuple[LaneLine, LaneLine] | None:
    """Aeusserste durchgezogene Linie je Seite, oder None.

    Bewusst nicht `{L.role: L for L in lines}`: bei mehreren `left_solid`
    behaelt die Komprehension die zuletzt einsortierte -- das ist die dem Ego
    naechste, nicht die Fahrbahnkante. Die Homographie wird dadurch stumm
    schiefgezogen, ohne dass ein Fehler auftritt. Auf Videomaterial trat das in
    41 % der Frames auf, und in 63 % fehlte eine der beiden Rollen ganz.

    Warum es einen Rueckfall auf die aeusserste Linie gibt
    -----------------------------------------------------
    `left_solid` wird nur an Linien vergeben, die WEITER AUSSEN liegen als die
    ego-naechste. Faehrt das Ego auf der linken Spur, ist die Fahrbahnkante
    zugleich die ego-naechste Linie -- sie heisst dann `left_dashed`, und die
    Rolle `left_solid` existiert im ganzen Video nicht.

    Gemessen: auf `adjusting_speed_scenario_9` ist `left_solid` in **jedem**
    Frame leer, waehrend im Mittel 2.2 `right_solid` gefunden werden. Die
    Aufnahme lief mit 0 % Homographie, obwohl beide Randlinien durchgehend
    sichtbar waren.

    Die Rollennamen sind an dieser Stelle positionell, nicht gemessen: keine
    Stufe prueft, ob eine Linie wirklich durchgezogen ist. Was die Homographie
    braucht, ist die **aeusserste Linie je Seite** -- danach wird gesucht, mit
    den Rollen als bevorzugtem, nicht als einzigem Weg dorthin.
    """
    left = [L for L in result.lines if L.role == "left_solid"]
    right = [L for L in result.lines if L.role == "right_solid"]
    # Rueckfall je Seite einzeln: eine vorhandene `*_solid`-Rolle behaelt
    # Vorrang, damit Material, das heute laeuft, exakt gleich weiterlaeuft.
    if not left and result.ego_left is not None:
        left = [L for L in result.lines
                if L.x_bottom <= result.ego_left.x_bottom]
    if not right and result.ego_right is not None:
        right = [L for L in result.lines
                 if L.x_bottom >= result.ego_right.x_bottom]
    if not left or not right:
        return None
    outer_left = min(left, key=lambda L: L.x_bottom)
    outer_right = max(right, key=lambda L: L.x_bottom)
    # Zwei praktisch deckungsgleiche Linien ergeben eine entartete Homographie,
    # die alles Nachgelagerte still verzerrt statt zu scheitern.
    if outer_right.x_bottom - outer_left.x_bottom < MIN_PAIR_SEPARATION:
        return None
    return outer_left, outer_right


def homography_from_pair(pair: tuple[LaneLine, LaneLine], lcfg: LaneConfig,
                         bcfg: BevConfig) -> np.ndarray:
    """Bildet die beiden Randlinien auf das BEV-Zielrechteck ab.

    Damit ist die BEV-Skala auf den Abstand *dieser* Linien normiert und nicht
    metrisch -- schliessen sie mal zwei und mal vier Spuren ein, bedeutet
    dieselbe Pixelbreite eine andere reale Breite.
    """
    ls, rs = pair
    src = np.float32([
        [ls.x_at(lcfg.y_bottom), lcfg.y_bottom],
        [rs.x_at(lcfg.y_bottom), lcfg.y_bottom],
        [rs.x_at(lcfg.y_top), lcfg.y_top],
        [ls.x_at(lcfg.y_top), lcfg.y_top],
    ])
    dst = np.float32([
        [bcfg.x_left, bcfg.y_near], [bcfg.x_right, bcfg.y_near],
        [bcfg.x_right, bcfg.y_far], [bcfg.x_left, bcfg.y_far],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def build_homography(img: np.ndarray, lcfg: LaneConfig,
                     bcfg: BevConfig) -> np.ndarray:
    """Einzelbild-Bequemlichkeit: erkennen und abbilden in einem Schritt.

    Fuer Sequenzen stattdessen `lanes.pipeline.HomographyTracker` verwenden --
    der haelt die letzte gueltige Abbildung ueber Aussetzer hinweg, statt hier
    zu scheitern.
    """
    pair = outer_solid_pair(detect_lanes(img, lcfg))
    if pair is None:
        raise ValueError("keine zwei durchgezogenen Randlinien gefunden")
    return homography_from_pair(pair, lcfg, bcfg)


# --------------------------------------------------------------------------- #
# Stufe 2 - Spurmaske mit Fahrzeugunterdrueckung                              #
# --------------------------------------------------------------------------- #
def build_lane_mask(img: np.ndarray, lcfg: LaneConfig,
                    boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls[:, :, 1], lcfg.white_l_min, lcfg.white_l_max)
    roi = np.zeros(white.shape, np.uint8)
    cv2.fillPoly(roi, [np.array(lcfg.roi_polygon, np.int32)], 255)
    mask = cv2.bitwise_and(white, roi)
    for x1, y1, x2, y2 in boxes:            # Detektor raeumt die Spurmaske auf
        mask[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = 0
    return mask


# --------------------------------------------------------------------------- #
# Stufe 3 - Spurgrenzen per Spaltenhistogramm im BEV                          #
# --------------------------------------------------------------------------- #
def lane_histogram(mask_bev: np.ndarray) -> np.ndarray:
    """Geglaettetes Spaltenhistogramm der BEV-Maske (Basis der Spurfindung)."""
    hist = (mask_bev > 0).sum(axis=0).astype(np.float32)
    return cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 0).ravel()


def peaks_from_histogram(hist: np.ndarray, cfg: BevConfig) -> list[int]:
    peaks: list[int] = []
    for x in range(4, len(hist) - 4):
        if hist[x] != hist[x - 4:x + 5].max() or hist[x] < cfg.peak_min_pixels:
            continue
        if peaks and x - peaks[-1] < cfg.peak_min_distance:
            if hist[x] > hist[peaks[-1]]:
                peaks[-1] = x
        else:
            peaks.append(x)
    return peaks


def find_lane_boundaries(mask_bev: np.ndarray, cfg: BevConfig) -> list[int]:
    return peaks_from_histogram(lane_histogram(mask_bev), cfg)


def corridors_from(boundaries: list[int]) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in zip(boundaries, boundaries[1:])]


# --------------------------------------------------------------------------- #
# Stufe 4 - Bruecke: nur die Bbox-Unterkante wird projiziert                  #
# --------------------------------------------------------------------------- #
def project_footprint(name: str, bbox: tuple[int, int, int, int],
                      H: np.ndarray) -> Footprint:
    x1, _, x2, y2 = bbox
    pts = np.float32([[[x1, y2]], [[x2, y2]]])
    w = cv2.perspectiveTransform(pts, H)[:, 0, :]
    return Footprint(name, float(min(w[0, 0], w[1, 0])),
                     float(max(w[0, 0], w[1, 0])), float(w[:, 1].mean()))


# --------------------------------------------------------------------------- #
# Stufe 5 - Belegung und Einscher-Klassifikation                              #
# --------------------------------------------------------------------------- #
def assign_lane(fp: Footprint, corridors: list[tuple[float, float]]) -> tuple[int, dict[int, float]]:
    ratios: dict[int, float] = {}
    for i, (lo, hi) in enumerate(corridors):
        overlap = max(0.0, min(fp.x_right, hi) - max(fp.x_left, lo))
        ratios[i] = overlap / fp.width
    best = max(ratios, key=ratios.get)
    return (best if ratios[best] > 0 else -1), ratios


def footprint_is_plausible(fp: Footprint, lane_width: float, cfg: BevConfig) -> bool:
    """Ein projiziertes Fahrzeug muss eine physikalisch moegliche Breite haben.

    Im Fernfeld divergiert die Rueckprojektion: kleine Bbox-Fehler in der
    Bildebene werden zu grossen lateralen Fehlern in der Bodenebene. Eine
    Footprint-Breite jenseits einer knappen Spurbreite ist ein Projektions-
    artefakt, keine Messung -> das Sample wird verworfen statt geraten. Das
    definiert die nutzbare Detektionsreichweite.

    `lane_width` kommt von `lanes.indexing`, nicht aus einem Median ueber
    `corridors[1:]`. Der wuerde voraussetzen, dass Korridor 0 immer der
    Standstreifen ist -- ueber eine Sequenz beginnt die Korridorliste je nach
    Sichtbarkeit mal mit dem Standstreifen, mal mit einer Fahrspur.
    """
    return (cfg.fp_width_min_ratio * lane_width
            <= fp.width <= cfg.fp_width_max_ratio * lane_width)




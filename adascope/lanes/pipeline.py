"""Sequenz-Pipeline: verkettet die Spurstufen ueber eine Frame-Folge.

    lanes.detection    Spurlinien in der Bildebene
    detection.tracking YOLO11 + ByteTrack, Bildebene
    (hier)             Homographie mit Persistenz
    lanes.bev          BEV-Maske, Spaltenhistogramm, Korridore, Footprints
    lanes.indexing     ego-relative Spurnummern, verschmolzene Korridore teilen
    lanes.events       temporale Ereignisse: cut_in / cut_out / aborted

Der Zustand liegt in genau zwei Objekten: der gehaltenen Homographie und der
Ereignis-State-Machine. Alles andere wird pro Frame neu berechnet.

Warum die Homographie ueberhaupt gehalten wird
----------------------------------------------
Nur ein Teil der Frames liefert beide durchgezogenen Randlinien. Gemessen ueber
2117 Frames: 39 % `fresh`, 46 % `held`, 15 % `none`. Ohne Persistenz waere die
Pipeline auf zwei Dritteln des Materials tot.

Warum zwei Spurnummerierungen nebeneinander
-------------------------------------------
Der positionsbasierte Index aus `assign_lane()` bleibt erhalten, obwohl
`lanes.indexing` ihn ersetzt: nur so laesst sich im Debug-Video direkt
vergleichen, wie oft er springt und ob die ego-relative Nummer an derselben
Stelle stabil bleibt. Er ist eine Messgroesse, kein Eingang.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from ..config import (
    BevConfig, BoundaryTrackConfig, EgoMotionConfig, EventConfig, IndexConfig,
    LaneConfig, PipelineConfig, Settings, WindowConfig,
)
from ..detection import TrackedVehicle
from .bev import (
    Footprint, assign_lane, build_lane_mask, corridors_from, footprint_is_plausible,
    homography_from_points, lane_histogram, source_points, outer_solid_pair,
    peaks_from_histogram, restrict_to_driving_area, warp_lane_mask,
    project_footprint,
)
from .boundaries import Boundaries
from .detection import LaneResult, detect_lanes
from .windows import find_boundaries as find_boundaries_windows
from .egomotion import EgoMotion, EgoMotionDetector
from .events import CutInTracker, Event, State
from .stabilize import BoundaryStabilizer
from .tracking_ids import BoundaryTracker
from .indexing import Lane, build_lane_index, ego_overlap, locate

HomographyState = Literal["fresh", "held", "none"]


# --------------------------------------------------------------------------- #
# Homographie mit Persistenz                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class HomographyTracker:
    """Haelt die letzte gueltige Homographie ueber Aussetzer hinweg."""

    lane: LaneConfig = field(default_factory=LaneConfig)
    bev: BevConfig = field(default_factory=BevConfig)
    max_hold: int = 25
    H: np.ndarray | None = None
    held_frames: int = 0
    # Glaettung der STUETZPUNKTE, nicht der Matrix. 0 = aus (jeder Frame neu),
    # 1 = eingefroren. Die Matrixeintraege sind nichtlinear verkoppelt; ein
    # Mittel ueber sie ergibt keine gueltige Homographie, ein Mittel ueber die
    # vier Punkte schon.
    #
    # Warum ueberhaupt: die BEV-Skala ist auf den Abstand der beiden Randlinien
    # normiert. Gemessen zittert dieser Abstand je Frame um 5-11 px im Median
    # -- damit atmet die gesamte Bodenebene in jedem Frame neu ein. Genau das
    # ist das Wackeln im Debugvideo.
    smoothing: float = 0.0
    max_point_jump: float = 90.0
    max_width_change_ratio: float = 0.25
    max_vanishing_jump: float = 160.0
    max_top_width_ratio: float = 1.0
    min_pair_continuity: float = 0.45
    min_pair_support: int = 2
    _src: np.ndarray | None = None
    candidate_src: np.ndarray | None = None
    last_rejection: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def accepted_src(self) -> np.ndarray | None:
        return None if self._src is None else self._src.copy()

    @staticmethod
    def _vanishing_point(src: np.ndarray) -> np.ndarray | None:
        """Schnittpunkt der linken und rechten Stuetzlinie in der Bildebene."""
        bl, br, tr, tl = src
        left = np.cross([bl[0], bl[1], 1.0], [tl[0], tl[1], 1.0])
        right = np.cross([br[0], br[1], 1.0], [tr[0], tr[1], 1.0])
        point = np.cross(left, right)
        if abs(float(point[2])) < 1e-6:
            return None
        return np.asarray(point[:2] / point[2], dtype=np.float32)

    def _validate(self, pair, src: np.ndarray) -> str:
        bottom_width = float(src[1, 0] - src[0, 0])
        top_width = float(src[2, 0] - src[3, 0])
        confidence = min(pair[0].continuity, pair[1].continuity)
        support = min(pair[0].support, pair[1].support)
        self.metrics = {
            "bottom_width": bottom_width, "top_width": top_width,
            "top_width_ratio": top_width / max(bottom_width, 1e-6),
            "pair_continuity": float(confidence), "pair_support": float(support),
            "point_jump": 0.0, "width_change_ratio": 0.0,
            "vanishing_jump": 0.0,
        }
        if bottom_width <= 0 or top_width <= 0:
            return "invalid_order"
        if top_width / bottom_width > self.max_top_width_ratio:
            return "bad_perspective"
        if confidence < self.min_pair_continuity:
            return "low_continuity"
        if support < self.min_pair_support:
            return "low_support"
        if self._src is None:
            return ""

        jumps = np.linalg.norm(src - self._src, axis=1)
        point_jump = float(jumps.max())
        old_bottom = float(self._src[1, 0] - self._src[0, 0])
        old_top = float(self._src[2, 0] - self._src[3, 0])
        width_change = max(
            abs(bottom_width - old_bottom) / max(abs(old_bottom), 1.0),
            abs(top_width - old_top) / max(abs(old_top), self.bev.min_pair_separation),
        )
        old_vp, new_vp = self._vanishing_point(self._src), self._vanishing_point(src)
        vanishing_jump = (float(np.linalg.norm(new_vp - old_vp))
                          if old_vp is not None and new_vp is not None else 0.0)
        self.metrics.update(point_jump=point_jump,
                            width_change_ratio=float(width_change),
                            vanishing_jump=vanishing_jump)
        if point_jump > self.max_point_jump:
            return "point_jump"
        if width_change > self.max_width_change_ratio:
            return "width_jump"
        if vanishing_jump > self.max_vanishing_jump:
            return "vanishing_jump"
        return ""

    def _hold(self, reason: str) -> tuple[np.ndarray | None, HomographyState]:
        self.last_rejection = reason
        if self.H is not None and self.held_frames < self.max_hold:
            self.held_frames += 1
            return self.H, "held"
        self.H, self.held_frames, self._src = None, 0, None
        return None, "none"

    def update(self, result: LaneResult) -> tuple[np.ndarray | None, HomographyState]:
        pair = outer_solid_pair(result, self.lane, self.bev)
        if pair is not None:
            src = source_points(pair, self.lane)
            self.candidate_src = src.copy()
            rejection = self._validate(pair, src)
            if rejection:
                return self._hold(rejection)
            if self._src is not None and self.smoothing > 0.0:
                a = self.smoothing
                src = a * self._src + (1.0 - a) * src
            self._src = src
            self.H = homography_from_points(src, self.bev)
            self.held_frames = 0
            self.last_rejection = ""
            return self.H, "fresh"
        self.candidate_src = None
        self.metrics = {}
        return self._hold("no_pair")


# --------------------------------------------------------------------------- #
# Ergebnis je Frame                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class VehicleOccupancy:
    """Ein Co-Fahrzeug in der Bodenebene, ego-relativ eingeordnet."""

    track: str
    footprint: Footprint
    rel: int | None             # 0 = Ego-Spur, -1 links, +1 rechts; None = keine
    ego_overlap: float
    valid: bool                 # Footprint-Breite plausibel
    state: State | None = None  # Zustand der State Machine
    # FR-1.4: laterale Position in BEV-Pixeln und die naechstgelegene
    # Grenze mit ihrer Kennung -- beides ohne jede Spurnummer.
    lateral_pos: float = 0.0
    boundary_id: int | None = None
    confidence: float = 0.0

    @property
    def label(self) -> str:
        if not self.valid:
            return f"{self.track} invalid"
        rel = "?" if self.rel is None else f"{self.rel:+d}"
        return f"{self.track} rel{rel} ego={self.ego_overlap:.2f}"


@dataclass
class FrameAnalysis:
    index: int
    name: str
    image: np.ndarray
    lanes: LaneResult
    vehicles: list[TrackedVehicle]
    H: np.ndarray | None
    h_state: HomographyState
    held_frames: int
    mask: np.ndarray                          # Weissmaske in der Bildebene
    h_rejection: str = ""
    h_metrics: dict[str, float] = field(default_factory=dict)
    driving_area_src: np.ndarray | None = None
    mask_bev: np.ndarray | None = None
    histogram: np.ndarray | None = None
    # Grenzen als Kurven. Bei `histogram` sind es Polynome vom Grad 0, bei
    # `windows` vom konfigurierten Grad -- der Rest des Codes sieht nur eine
    # Darstellung.
    boundary_fit: Boundaries = field(default_factory=Boundaries)
    # Bei der Hoehe des Ego-Footprints ausgewertet. Genau dort findet der
    # Belegungsvergleich statt; ueber die volle Hoehe gemittelte Grenzen waeren
    # in einer Kurve schlicht falsch.
    boundaries: list[float] = field(default_factory=list)
    corridors: list[tuple[float, float]] = field(default_factory=list)
    ego_lane_pos: int = -1                    # positionsbasiert, nur als Messgroesse
    ego_footprint: Footprint | None = None
    lanes_rel: list[Lane] = field(default_factory=list)
    lane_width: float = 0.0
    index_note: str = ""                      # warum die Indizierung ausfiel
    # Anteil des EIGENEN Footprints in der eigenen Spur. 1.0 = vollstaendig
    # innerhalb; faellt er, verlaesst das Ego seine Spur. Dieselbe Groesse wie
    # `ego_overlap` fuer Fremdfahrzeuge, nur auf das Ego angewandt -- das ist
    # das Lane-Departure-Signal, fuer das `lanes.events` noch keinen EventKind
    # hat. Es wird gemessen und angezeigt, nicht klassifiziert.
    ego_in_lane: float = 1.0
    # FR-1.4/FR-5.1: stabile Kennung je Grenze, in der Reihenfolge von
    # `boundaries`. Beantwortet 'dieselbe Linie wie eben?', nicht 'die
    # wievielte von links?'.
    boundary_ids: list[int] = field(default_factory=list)
    # FR-3: Urteil ueber die eigene Querbewegung aus der Linienstruktur.
    ego_motion: EgoMotion = field(default_factory=EgoMotion)
    occupancies: list[VehicleOccupancy] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    def states(self) -> list[dict]:
        """Per-Frame-Zustand je Fahrzeug nach FR-1.4.

        {frame_id, fahrzeug, lateral_pos, aktive_grenze_id, confidence} --
        das Ego eingeschlossen, damit EGO und CO denselben Satz liefern.
        """
        rows = []
        if self.ego_footprint is not None:
            rows.append({
                "frame_id": self.index, "fahrzeug": "EGO",
                "lateral_pos": round(self.ego_footprint.center, 2),
                "aktive_grenze_id": self.nearest_boundary(self.ego_footprint.center),
                "confidence": round(self.ego_motion_confidence(), 3),
            })
        for occ in self.occupancies:
            rows.append({
                "frame_id": self.index, "fahrzeug": occ.track,
                "lateral_pos": round(occ.lateral_pos, 2),
                "aktive_grenze_id": occ.boundary_id,
                "confidence": round(occ.confidence, 3),
            })
        return rows

    def nearest_boundary(self, x: float) -> int | None:
        """Kennung der Grenze, die dieser Position am naechsten liegt."""
        if not self.boundaries or not self.boundary_ids:
            return None
        index = min(range(len(self.boundaries)),
                    key=lambda i: abs(self.boundaries[i] - x))
        return self.boundary_ids[index] if index < len(self.boundary_ids) else None

    def ego_motion_confidence(self) -> float:
        """Belastbarkeit der Ego-Aussage: wie viele Grenzen sie stuetzen."""
        if not self.boundary_ids:
            return 0.0
        return min(self.ego_motion.boundaries_used / max(len(self.boundary_ids), 1), 1.0)

    @property
    def worst_state(self) -> str:
        order = ["outside", "invalid", "encroaching", "inside"]
        seen = [o.state or ("invalid" if not o.valid else "outside")
                for o in self.occupancies]
        return max(seen, key=order.index) if seen else "outside"


# --------------------------------------------------------------------------- #
def road_vehicles(vehicles: list[TrackedVehicle], lane: LaneConfig,
                  margin: int) -> list[TrackedVehicle]:
    """Detektionen unterhalb der Fahrbahn verwerfen.

    Die HMI-Ansicht zeigt unter der Kamerabildflaeche ein Kombiinstrument, und
    dessen stilisiertes Fahrzeug-Icon wird von YOLO zuverlaessig als `car`
    detektiert -- mit eigener, ueber die ganze Sequenz stabiler Track-ID. Ohne
    diesen Filter erzeugt ein Bildschirmelement einen Dauer-Cut-In-Kandidaten.

    Kriterium ist die Bbox-Unterkante, weil genau sie projiziert wird: liegt sie
    unter der Fahrbahn-Referenzlinie, hat der Punkt keine Bodenebene.
    """
    return [v for v in vehicles if v.bbox[3] <= lane.y_bottom + margin]


def ego_reference_footprint(vehicles: list[TrackedVehicle], H: np.ndarray,
                            lane: LaneConfig, cfg: PipelineConfig) -> Footprint:
    """Footprint des Ego-Fahrzeugs; ohne Detektion der kalibrierte Bildpunkt.

    Die HMI-Ansicht ist eine Chase-Cam, das Ego-Fahrzeug ist also sichtbar und
    wird normal detektiert. Der Fallback haelt die Geometrie-Ansichten ohne YOLO
    lauffaehig (`--no-detect`).
    """
    ego = next((v for v in vehicles if v.role == "ego"), None)
    if ego is not None:
        return project_footprint("ego", ego.bbox, H)
    half = cfg.ego_fallback_half_width
    box = (int(lane.ego_x_bottom - half), lane.y_bottom - cfg.ego_fallback_height,
           int(lane.ego_x_bottom + half), lane.y_bottom)
    return project_footprint("ego", box, H)


# --------------------------------------------------------------------------- #
class SequencePipeline:
    """Zustandsbehaftete Pipeline ueber eine Frame-Folge."""

    def __init__(self, settings: Settings | None = None, *,
                 lane: LaneConfig | None = None, bev: BevConfig | None = None,
                 indexing: IndexConfig | None = None, events: EventConfig | None = None,
                 pipeline: PipelineConfig | None = None,
                 windows: WindowConfig | None = None,
                 boundaries: BoundaryTrackConfig | None = None,
                 egomotion: EgoMotionConfig | None = None):
        """`settings` ist der Normalfall; die Einzel-Configs sind fuer Tests.

        Damit braucht ein Test kein Dateisystem: er baut die eine Config, die er
        variieren will, und laesst den Rest auf den Code-Defaults.
        """
        self.lane = lane or (settings.lane if settings else LaneConfig())
        self.bev = bev or (settings.bev if settings else BevConfig())
        self.indexing = indexing or (settings.indexing if settings else IndexConfig())
        self.cfg = pipeline or (settings.pipeline if settings else PipelineConfig())
        self.windows = windows or (settings.windows if settings else WindowConfig())
        bcfg = boundaries or (settings.boundaries if settings else BoundaryTrackConfig())
        # Reihenfolge ist Absicht: erst entscheiden, WELCHE Grenzen es gibt
        # (stabilizer), dann WELCHE KENNUNG jede traegt (tracker).
        self.stabilizer = BoundaryStabilizer(bcfg, self.bev.peak_min_pixels)
        self.boundary_tracker = BoundaryTracker(bcfg)
        self.ego_motion = EgoMotionDetector(
            egomotion or (settings.egomotion if settings else EgoMotionConfig()))
        self.homography = HomographyTracker(self.lane, self.bev, self.cfg.max_hold,
            smoothing=self.cfg.homography_smoothing,
            max_point_jump=self.cfg.homography_max_point_jump,
            max_width_change_ratio=self.cfg.homography_max_width_change_ratio,
            max_vanishing_jump=self.cfg.homography_max_vanishing_jump,
            max_top_width_ratio=self.cfg.homography_max_top_width_ratio,
            min_pair_continuity=self.cfg.homography_min_pair_continuity,
            min_pair_support=self.cfg.homography_min_pair_support)
        self.fsm = CutInTracker(events or (settings.events if settings else EventConfig()))
        self.log: list[Event] = []
        # Startpositionen des Vorframes fuer die Fenstersuche.
        self._previous_starts: list[int] = []

    def _fit_calibration(self, img: np.ndarray) -> None:
        """Die Spurkalibrierung einmal auf die tatsaechliche Bildgroesse ziehen.

        `lane.yaml` ist in Pixeln eines Referenzzuschnitts notiert. Kommt ein
        anders grosses, aber gleich proportioniertes Bild, wird mitskaliert;
        bei anderem Seitenverhaeltnis zeigt das Bild einen anderen Ausschnitt
        und `scaled_to()` sagt das mit einer verwertbaren Meldung.
        """
        height, width = img.shape[:2]
        if (width, height) == self.lane.reference_size:
            return
        self.lane = self.lane.scaled_to(width, height)
        self.homography.lane = self.lane

    def process(self, index: int, name: str, img: np.ndarray,
                vehicles: list[TrackedVehicle]) -> FrameAnalysis:
        self._fit_calibration(img)
        vehicles = road_vehicles(vehicles, self.lane, self.cfg.road_margin)
        lanes = detect_lanes(img, self.lane)
        H, h_state = self.homography.update(lanes)

        # Der Detektor raeumt die Spurmaske auf: Fahrzeugpixel raus, bevor
        # gewarpt wird -- sonst erzeugen Fahrzeugdaecher Histogramm-Peaks.
        mask = build_lane_mask(img, self.lane, [v.bbox for v in vehicles])

        # Stage 3 remains deliberately permissive so every candidate is
        # inspectable.  From here onward, however, only the accepted
        # directional carriageway may contribute pixels.  Otherwise a valid
        # but irrelevant marking on the opposite carriageway survives the
        # homography and becomes a false histogram boundary.
        accepted_src = self.homography.accepted_src
        if H is not None and accepted_src is not None:
            mask = restrict_to_driving_area(mask, accepted_src)

        # Wird schrittweise angereichert; jeder fruehe return ist ein
        # Stufenausfall, den die Debug-Ansichten als solchen zeigen sollen.
        fa = FrameAnalysis(index, name, img, lanes, vehicles, H, h_state,
                           self.homography.held_frames, mask)
        fa.h_rejection = self.homography.last_rejection
        fa.h_metrics = dict(self.homography.metrics)
        fa.driving_area_src = accepted_src
        if H is None:
            # Die Bodenebene ist ganz weg, nicht nur kurz ueberbrueckt. Was
            # danach kommt, kann eine voellig andere Geometrie sein -- die
            # gemerkten Grenzen dahin mitzunehmen waere geraten, nicht gemessen.
            self.stabilizer.reset()
            fa.index_note = "keine Homographie"
            return self._finish(fa, {})

        fa.mask_bev = warp_lane_mask(mask, H, self.bev)
        fa.histogram = lane_histogram(fa.mask_bev, self.bev)
        fa.boundary_fit = self._find_boundaries(fa.mask_bev, fa.histogram)
        # Kennungen VOR jeder Auswertung vergeben: sie sind die Verbindung
        # zum Vorframe und damit die Grundlage der Ego-Bewegungsanalyse.
        fa.boundary_ids = self.boundary_tracker.update(fa.boundary_fit)
        if len(fa.boundary_fit) < self.indexing.min_corridors + 1:
            fa.boundaries = fa.boundary_fit.at(fa.boundary_fit.y_reference)
            fa.corridors = corridors_from(fa.boundaries)
            fa.index_note = f"nur {max(len(fa.corridors), 0)} Korridor(e)"
            return self._finish(fa, {})

        # Erst den Ego-Footprint bestimmen, dann die Grenzen AUF SEINER HOEHE
        # auswerten: dort findet der Vergleich statt. Fuer gerade Spuren ist
        # das Ergebnis identisch, in einer Kurve ist es der Unterschied.
        fa.ego_footprint = ego_reference_footprint(vehicles, H, self.lane, self.cfg)
        fa.boundaries = fa.boundary_fit.at(fa.ego_footprint.y)
        fa.corridors = corridors_from(fa.boundaries)
        if len(fa.corridors) < self.indexing.min_corridors:
            fa.index_note = f"nur {len(fa.corridors)} Korridor(e)"
            return self._finish(fa, {})
        fa.ego_lane_pos, _ = assign_lane(fa.ego_footprint, fa.corridors)

        try:
            fa.lanes_rel, fa.lane_width = build_lane_index(
                fa.corridors, fa.ego_footprint.x_left, fa.ego_footprint.x_right,
                self.indexing)
        except ValueError as exc:               # Ego in keiner plausiblen Spur
            fa.index_note = str(exc)
            return self._finish(fa, {})
        if not fa.lanes_rel:
            fa.index_note = "keine plausible Spur"
            return self._finish(fa, {})
        fa.ego_in_lane = ego_overlap(fa.lanes_rel, fa.ego_footprint.x_left,
                                     fa.ego_footprint.x_right)
        # FR-3.1/3.2: eigener Spurwechsel aus der Linienstruktur, nicht aus
        # den Fremdfahrzeugen.
        fa.ego_motion = self.ego_motion.update(fa.boundary_fit, fa.boundary_ids,
                                               fa.lane_width)

        observations: dict[str, tuple[int, float]] = {}
        for vehicle in vehicles:
            if vehicle.role == "ego":
                continue
            track = f"ID{vehicle.track_id}"
            fp = project_footprint(track, vehicle.bbox, H)
            valid = footprint_is_plausible(fp, fa.lane_width, self.bev)
            rel, _ = locate(fa.lanes_rel, fp.x_left, fp.x_right)
            overlap = ego_overlap(fa.lanes_rel, fp.x_left, fp.x_right)
            boundary_id = fa.nearest_boundary(fp.center)
            occupancy = VehicleOccupancy(
                track, fp, rel, overlap, valid,
                lateral_pos=fp.center, boundary_id=boundary_id,
                confidence=(self.boundary_tracker.confidence(boundary_id)
                            if boundary_id is not None else 0.0))
            fa.occupancies.append(occupancy)
            # Ungueltige Samples werden weggelassen, nicht mit Ersatzwerten
            # gefuellt -- die State Machine glaettet sonst falsche Ereignisse.
            if valid and rel is not None:
                observations[track] = (rel, overlap)
        return self._finish(fa, observations)

    def _find_boundaries(self, mask_bev: np.ndarray,
                         histogram: np.ndarray | None = None) -> Boundaries:
        """Grenzen nach dem konfigurierten Verfahren suchen und stabilisieren."""
        if histogram is None:
            histogram = lane_histogram(mask_bev, self.bev)
        if not self.windows.uses_windows:
            peaks = peaks_from_histogram(histogram, self.bev)
            found = Boundaries.from_positions(peaks, mask_bev.shape[0])
        else:
            found = find_boundaries_windows(mask_bev, self.bev, self.windows,
                                            self._previous_starts)
        found = self.stabilizer.update(found, histogram)
        if found:
            # Der Startpunkt der naechsten Fenstersuche kommt aus dem
            # STABILISIERTEN Ergebnis -- sonst traegt die Suche das Flackern,
            # das gerade herausgerechnet wurde, in den naechsten Frame.
            self._previous_starts = [round(x) for x in found.at(found.y_reference)]
        return found

    def _finish(self, fa: FrameAnalysis,
                observations: dict[str, tuple[int, float]]) -> FrameAnalysis:
        """State Machine in JEDEM Frame takten, auch ohne Beobachtungen.

        Nur so altern fehlende Tracks korrekt aus (`max_missing`) und das
        Nachlauffenster nach einem eigenen Spurwechsel laeuft weiter.
        """
        lateral = {o.track: o.lateral_pos for o in fa.occupancies}
        boundary_at = {o.track: o.boundary_id for o in fa.occupancies
                       if o.boundary_id is not None}
        fa.events = self.fsm.update(fa.index, observations, fa.ego_in_lane,
                                    lateral, boundary_at)
        fa.events.extend(self._ego_events(fa))
        self.log.extend(fa.events)
        for occ in fa.occupancies:
            occ.state = self.fsm.state_of(occ.track)
        return fa

    def _ego_events(self, fa: FrameAnalysis) -> list[Event]:
        """Ereignis aus der Linienstruktur-Analyse (FR-3.1 bis FR-3.3).

        Ein nicht belegter Wechsel wird als `certain=False` gemeldet, nicht
        verschwiegen -- FR-3.3 verlangt ausdruecklich eine Markierung statt
        eines stillen Ereignisses.
        """
        motion = fa.ego_motion
        if motion.verdict == "kein_wechsel":
            return []
        confidence = fa.ego_motion_confidence()
        return [Event(
            fa.index, "ego_lane_change", None,
            f"Verschiebung {motion.shift_lanes:.2f} Spurbreiten, Streuung {motion.spread:.2f}",
            direction=motion.direction,
            boundary_id=fa.nearest_boundary(
                fa.ego_footprint.center if fa.ego_footprint else 0.0),
            frame_start=max(fa.index - self.ego_motion.cfg.window, 0),
            frame_end=fa.index,
            confidence=confidence,
            certain=motion.verdict == "wechsel")]

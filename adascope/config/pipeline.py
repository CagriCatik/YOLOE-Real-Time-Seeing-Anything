"""Konfiguration der Sequenz-Pipeline (Zustand ueber Frames hinweg)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Kalibrierflaeche von `lanes.pipeline`."""

    # Frames, die eine ausgefallene Homographie gehalten wird (~1 s bei 25 fps).
    max_hold: int = 25
    # Detektionen, deren Bbox-Unterkante weiter als dieser Abstand unter
    # `LaneConfig.y_bottom` liegt, werden verworfen. Grund: das HMI zeigt unter
    # der Kamerabildflaeche ein Kombiinstrument, dessen Fahrzeug-Icon von YOLO
    # stabil als `car` detektiert wird -- ein Dauer-Cut-In-Kandidat aus einem
    # Bildschirmelement.
    road_margin: int = 25
    # Ersatz-Ego-Box, wenn ohne Detektor gearbeitet wird (`--no-detect`):
    # halbe Fahrzeugbreite und Boxhoehe in Bildpixeln um `ego_x_bottom`.
    ego_fallback_half_width: int = 45
    ego_fallback_height: int = 60

    def __post_init__(self) -> None:
        for name in ("max_hold", "road_margin"):
            if getattr(self, name) < 0:
                raise ValueError(f"pipeline.{name} darf nicht negativ sein")
        if self.ego_fallback_half_width <= 0 or self.ego_fallback_height <= 0:
            raise ValueError("pipeline: Ersatz-Ego-Box braucht positive Masse")
        if not 0 <= self.homography_smoothing < 1:
            raise ValueError("pipeline.homography_smoothing muss in [0, 1) liegen")
        for name in ("homography_max_point_jump", "homography_max_vanishing_jump"):
            if getattr(self, name) <= 0:
                raise ValueError(f"pipeline.{name} muss positiv sein")
        if not 0 < self.homography_max_width_change_ratio <= 1:
            raise ValueError("pipeline.homography_max_width_change_ratio muss in (0, 1] liegen")
        if not 0 < self.homography_max_top_width_ratio <= 1:
            raise ValueError("pipeline.homography_max_top_width_ratio muss in (0, 1] liegen")
        if not 0 <= self.homography_min_pair_continuity <= 1:
            raise ValueError("pipeline.homography_min_pair_continuity muss in [0, 1] liegen")
        if self.homography_min_pair_support < 1:
            raise ValueError("pipeline.homography_min_pair_support muss mindestens 1 sein")

    # Glaettung der Homographie-Stuetzpunkte ueber die Zeit, 0 bis <1.
    # 0.0 = jeder Frame neu (das alte Verhalten), 0.8 = traege.
    #
    # Gemessen zittert der Abstand der beiden Randlinien je Frame um 5-11 px
    # im Median. Weil die BEV-Skala genau darauf normiert ist, atmet die
    # Bodenebene damit in jedem Frame neu ein -- das ist das Wackeln.
    homography_smoothing: float = 0.75
    # Plausibilitaetsgates fuer neue Stuetzpunktpaare. Ein abgelehnter
    # Kandidat fuehrt zu `held`, nicht zu einer neuen falschen Bodenebene.
    homography_max_point_jump: float = 90.0
    homography_max_width_change_ratio: float = 0.25
    homography_max_vanishing_jump: float = 160.0
    homography_max_top_width_ratio: float = 1.0
    homography_min_pair_continuity: float = 0.45
    homography_min_pair_support: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"pipeline: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)

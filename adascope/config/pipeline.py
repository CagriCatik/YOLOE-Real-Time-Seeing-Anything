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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"pipeline: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)

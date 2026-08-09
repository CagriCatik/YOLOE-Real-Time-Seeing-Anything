"""Kalibrierung der Bodenebene (Bird's Eye View) und der Belegungsschwellen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loader import require_positive, require_range


@dataclass(frozen=True)
class BevConfig:
    """Zielrechteck der Homographie plus Schwellen von `lanes.bev`."""

    width: int = 500
    height: int = 700
    # Auf diese vier Kanten werden die beiden durchgezogenen Randlinien
    # abgebildet. ACHTUNG: damit ist die BEV-Skala auf den Abstand *dieser*
    # Linien normiert, nicht metrisch -- siehe README, "Die BEV-Skala ist nicht
    # metrisch". Eine Aenderung hier aendert die Bedeutung jeder Pixelbreite.
    x_left: int = 81
    x_right: int = 419
    y_near: int = 690
    y_far: int = 20
    # Histogramm-Spurfindung
    peak_min_pixels: int = 10
    # Mindestabstand zweier Peaks. Muss deutlich unter der schmalsten echten
    # Spurbreite und deutlich ueber der Breite der Fehlpeaks liegen. Bei 25 px
    # ueberlebten auf Video Kleinstkorridore von 25..42 px: das Einzelbild blieb
    # korrekt, aber die Spurbreitenschaetzung griff genau diese Fehlpeaks ab.
    peak_min_distance: int = 55
    # Einscher-Schwellen (Anteil der Fahrzeugbreite im Ego-Korridor)
    thr_encroaching: float = 0.10
    thr_in_lane: float = 0.50
    # Plausibilitaetsgrenzen der projizierten Footprint-Breite, als Vielfaches
    # der gemessenen Spurbreite (PKW ~0.5, LKW ~0.75)
    fp_width_min_ratio: float = 0.25
    fp_width_max_ratio: float = 0.95

    def __post_init__(self) -> None:
        require_positive(self.width, "bev.width")
        require_positive(self.height, "bev.height")
        if self.x_left >= self.x_right:
            raise ValueError("x_left muss kleiner als x_right sein")
        if self.y_far >= self.y_near:
            raise ValueError("y_far muss kleiner als y_near sein (0 = fern)")
        require_range(self.thr_encroaching, 0, 1, "bev.thr_encroaching")
        require_range(self.thr_in_lane, 0, 1, "bev.thr_in_lane")
        if self.thr_encroaching > self.thr_in_lane:
            raise ValueError("thr_encroaching darf thr_in_lane nicht ueberschreiten")
        if self.fp_width_min_ratio >= self.fp_width_max_ratio:
            raise ValueError("fp_width_min_ratio muss kleiner als fp_width_max_ratio sein")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BevConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"bev: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)

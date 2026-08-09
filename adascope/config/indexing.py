"""Konfiguration der ego-relativen Spurnummerierung."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loader import require_positive, require_range


@dataclass(frozen=True)
class IndexConfig:
    """Kalibrierflaeche von `lanes.indexing`."""

    # Mindestzahl an Korridoren, ab der ein Frame ausgewertet wird.
    #
    # Stand 1 statt frueher 2, und das ist ein Messergebnis. Eine EINSPURIGE
    # Fahrbahn hat zwei Grenzen und genau einen Korridor -- eine vollstaendige,
    # gueltige Szene. Auf `lane_departure_1_lane` (489 Frames) haben 312 Frames
    # genau zwei Grenzen; die Forderung nach zwei Korridoren verwarf sie alle
    # und liess von der Aufnahme 36 % uebrig.
    #
    # Bitter daran: die Fahrzeugdetektion arbeitete dabei korrekt. Ohne sie
    # erzeugten Fahrzeugpixel Scheingrenzen, die auf drei kamen -- die Aufnahme
    # lief also nur deshalb, WEIL die Spurmaske verschmutzt war.
    #
    # Bei einem Korridor gibt es keine Nachbarspur, also auch kein cut_in oder
    # cut_out. Das Spurverlassen des Ego bleibt messbar, und genau darum geht
    # es in dieser Aufnahme.
    min_corridors: int = 1

    # Toleranz, innerhalb der eine Korridorbreite als ganzzahliges Vielfaches
    # der Spurbreite gilt.
    multiple_tolerance: float = 0.18
    # Maximale Anzahl Spuren, in die ein Korridor aufgeteilt werden darf.
    max_merge: int = 4
    # Spurbreite in BEV-Pixeln. None -> je Frame aus den Korridoren schaetzen.
    # Ein fester Wert ist nur gueltig, solange die BEV-Skala konstant ist, also
    # solange die beiden Randlinien dieselbe Anzahl Spuren einschliessen.
    lane_width: float | None = None
    # Faktor, bis zu dem eine Korridorbreite noch zum Minimum-Cluster der
    # Spurbreitenschaetzung zaehlt.
    width_cluster_factor: float = 1.15

    def __post_init__(self) -> None:
        require_range(self.multiple_tolerance, 0, 0.5, "indexing.multiple_tolerance")
        if self.max_merge < 1:
            raise ValueError("indexing.max_merge muss mindestens 1 sein")
        if self.lane_width is not None:
            require_positive(self.lane_width, "indexing.lane_width")
        if self.width_cluster_factor < 1:
            raise ValueError("indexing.width_cluster_factor muss mindestens 1 sein")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IndexConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"indexing: unbekannte Schluessel {sorted(unknown)}")
        values = dict(raw)
        if values.get("lane_width") in (0, 0.0):     # 0 in YAML = "schaetzen"
            values["lane_width"] = None
        return cls(**values)

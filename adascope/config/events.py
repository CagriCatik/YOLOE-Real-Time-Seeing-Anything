"""Konfiguration der temporalen Ereignisableitung (Cut-In / Cut-Out)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loader import require_range


@dataclass(frozen=True)
class EventConfig:
    """Kalibrierflaeche von `lanes.events`."""

    thr_encroaching: float = 0.10
    thr_inside: float = 0.50
    # Aufeinanderfolgende Frames, bevor ein Zustand uebernommen wird.
    confirm_frames: int = 3
    # Frames ohne Messung, nach denen ein Track verworfen wird.
    max_missing: int = 5
    # Mindestens so viele Tracks muessen sich gleichsinnig verschieben, damit es
    # als eigener Spurwechsel gilt. Auf Material mit meist nur einem sichtbaren
    # Fremdfahrzeug kann die Szenenebene damit nie greifen -- siehe README.
    ego_shift_min_tracks: int = 2
    # Ein eigener Spurwechsel wird nur anerkannt, wenn der Ego-Footprint dabei
    # auch wirklich seine Spurgrenze beruehrt: `ego_in_lane` muss unter diesen
    # Wert fallen. 1.0 schaltet den Beleg ab (altes Verhalten).
    ego_departure_max: float = 0.90
    # Nachlauffenster nach einem eigenen Spurwechsel: waehrend Ego die Spur
    # wechselt, ist der ego-relative Bezugsrahmen selbst in Bewegung.
    ego_refractory_frames: int = 6

    def __post_init__(self) -> None:
        require_range(self.thr_encroaching, 0, 1, "events.thr_encroaching")
        require_range(self.thr_inside, 0, 1, "events.thr_inside")
        require_range(self.ego_departure_max, 0, 1, "events.ego_departure_max")
        if self.thr_encroaching > self.thr_inside:
            raise ValueError("thr_encroaching darf thr_inside nicht ueberschreiten")
        for name in ("confirm_frames", "max_missing", "ego_refractory_frames"):
            if getattr(self, name) < 0:
                raise ValueError(f"events.{name} darf nicht negativ sein")
        if self.ego_shift_min_tracks < 1:
            raise ValueError("events.ego_shift_min_tracks muss mindestens 1 sein")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EventConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"events: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)

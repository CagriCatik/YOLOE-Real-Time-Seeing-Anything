"""Konfiguration des Fahrzeugtrackings (YOLO11 + ByteTrack) in der Bildebene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import require_positive, require_range

# Relativ zum Projektwurzelverzeichnis; wird beim Laden aufgeloest.
DEFAULT_WEIGHTS = "models/yolo11n.pt"
# COCO: car=2, motorcycle=3, bus=5, truck=7
DEFAULT_CLASS_IDS = (2, 3, 5, 7)
# Erwartete Ego-Glyphenregion in normalisierten Bildkoordinaten (x0,y0,x1,y1).
DEFAULT_EGO_ZONE = (0.40, 0.30, 0.62, 0.68)


@dataclass(frozen=True)
class TrackingConfig:
    weights: str = DEFAULT_WEIGHTS
    tracker: str = "bytetrack.yaml"
    confidence: float = 0.25
    iou: float = 0.50
    image_size: int = 1280
    device: str | int | None = None
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS
    ego_zone: tuple[float, float, float, float] = DEFAULT_EGO_ZONE
    # Frames, die die gewaehlte Ego-Identitaet ohne Detektion gehalten wird.
    ego_memory_frames: int = 15

    def __post_init__(self) -> None:
        require_range(self.confidence, 0, 1, "tracking.confidence")
        require_range(self.iou, 0, 1, "tracking.iou")
        require_positive(self.image_size, "tracking.image_size")
        if self.tracker not in ("bytetrack.yaml", "botsort.yaml"):
            raise ValueError(f"unbekannter Tracker: {self.tracker!r}")
        x0, y0, x1, y1 = self.ego_zone
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("ego_zone muss 0 <= x0 < x1 <= 1 und 0 <= y0 < y1 <= 1 erfuellen")
        if not self.class_ids:
            raise ValueError("class_ids darf nicht leer sein")

    def resolve_weights(self, root: str | Path) -> str:
        """Relative Gewichtspfade gegen die Projektwurzel aufloesen."""
        path = Path(self.weights)
        return str(path if path.is_absolute() else Path(root) / path)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrackingConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"tracking: unbekannte Schluessel {sorted(unknown)}")
        values = dict(raw)
        if "class_ids" in values:
            values["class_ids"] = tuple(int(v) for v in values["class_ids"])
        if "ego_zone" in values:
            zone = values["ego_zone"]
            if len(zone) != 4:
                raise ValueError("tracking.ego_zone braucht vier Werte")
            values["ego_zone"] = tuple(float(v) for v in zone)
        return cls(**values)

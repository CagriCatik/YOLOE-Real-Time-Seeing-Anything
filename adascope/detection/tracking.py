"""YOLO11-Fahrzeugdetektion mit persistenten ByteTrack-IDs.

Der Tracker bleibt in der BILDEBENE. Nachgelagerter BEV-Code darf ausschliess-
lich die Unterkante von ``bbox`` projizieren, niemals das ganze Fahrzeugbild
oder eine Segmentierungsmaske -- nur die Unterkante liegt in der Bodenebene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..config import TrackingConfig

VehicleRole = Literal["ego", "co"]


@dataclass
class TrackedVehicle:
    track_id: int | None
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    role: VehicleRole = "co"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2, float(y2)


class YoloVehicleTracker:
    """Zustandsbehafteter YOLO11-/ByteTrack-Adapter fuer aufeinanderfolgende Frames."""

    def __init__(self, config: TrackingConfig | None = None, weights: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics fehlt: pip install -e .") from exc
        self.config = config or TrackingConfig()
        self.model = YOLO(weights or self.config.weights)
        self.ego_track_id: int | None = None
        self._ego_missing = 0

    def reset(self) -> None:
        """Track-Zustand verwerfen, Modell behalten.

        Fuer aufeinanderfolgende, unabhaengige Sequenzen: ohne Reset laufen
        Track-IDs und die Ego-Identitaet aus der vorigen Sequenz weiter.

        Zurueckgesetzt wird ueber `BYTETracker.reset()`, NICHT ueber ein
        `persist=False` am naechsten Frame: Ultralytics registriert seinen
        Callback beim ersten `track()`-Aufruf und backt den damaligen
        `persist`-Wert hinein. Ein erster Aufruf mit `persist=False` laesst den
        Tracker danach in *jedem* Frame neu anfangen -- gemessen: Ego-Index-
        Spruenge von 0.8 % auf 16.7 %, plus 45 erfundene Ereignisse.
        """
        self.ego_track_id, self._ego_missing = None, 0
        for tracker in getattr(getattr(self.model, "predictor", None), "trackers", []) or []:
            tracker.reset()

    def update(self, frame: np.ndarray) -> list[TrackedVehicle]:
        options = {
            "source": frame,
            "persist": True,
            "tracker": self.config.tracker,
            "classes": list(self.config.class_ids),
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.image_size,
            "verbose": False,
        }
        if self.config.device is not None:
            options["device"] = self.config.device
        result = self.model.track(**options)[0]
        vehicles = result_to_vehicles(result)
        self._assign_roles(vehicles, frame.shape[1], frame.shape[0])
        return vehicles

    def _assign_roles(self, vehicles: list[TrackedVehicle], width: int, height: int) -> None:
        # Die gewaehlte Ego-Identitaet halten, solange ByteTrack sie noch sieht.
        if self.ego_track_id is not None:
            current = next((v for v in vehicles if v.track_id == self.ego_track_id), None)
            if current is not None:
                current.role = "ego"
                self._ego_missing = 0
                return
            self._ego_missing += 1
            if self._ego_missing <= self.config.ego_memory_frames:
                return
            self.ego_track_id = None

        ego = select_ego_candidate(vehicles, width, height, self.config.ego_zone)
        if ego is not None:
            ego.role = "ego"
            self.ego_track_id = ego.track_id
            self._ego_missing = 0


def result_to_vehicles(result) -> list[TrackedVehicle]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(result, "names", {})
    coords = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().int().tolist()
    raw_ids = boxes.id.detach().cpu().int().tolist() if boxes.id is not None else [None] * len(coords)
    output = []
    for xyxy, confidence, class_id, track_id in zip(coords, confidences, classes, raw_ids):
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        output.append(TrackedVehicle(
            int(track_id) if track_id is not None else None,
            int(class_id), str(label), float(confidence),
            tuple(round(float(value)) for value in xyxy),
        ))
    return output


def select_ego_candidate(vehicles: list[TrackedVehicle], width: int, height: int,
                         zone: tuple[float, float, float, float]) -> TrackedVehicle | None:
    """Unterstes sichtbares Fahrzeug innerhalb der konfigurierten Ankerzone."""
    x0, y0, x1, y1 = zone
    candidates = []
    for vehicle in vehicles:
        cx, cy = vehicle.center
        if x0 <= cx / width <= x1 and y0 <= cy / height <= y1:
            bx1, by1, bx2, by2 = vehicle.bbox
            area = max(0, bx2 - bx1) * max(0, by2 - by1)
            candidates.append((cy, area, vehicle))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None

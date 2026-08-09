"""Modell-Adapter. Der einzige Ort, an dem `ultralytics` importiert wird.

Zwei Modelle mit verschiedenen Aufgaben, bewusst getrennt gehalten:

    YoloeDetector       open-vocabulary, Textprompts, EIN Frame ohne Gedaechtnis
    YoloVehicleTracker  geschlossenes COCO-Set, ByteTrack, Zustand ueber Frames

Der Rest des Pakets kennt nur `BBox` bzw. `TrackedVehicle` -- kein Domaenen-
oder Renderingcode importiert `ultralytics` oder `torch`.
"""

from .detector import BBox, Detector, YoloeDetector
from .tracking import TrackedVehicle, VehicleRole, YoloVehicleTracker, select_ego_candidate

__all__ = [
    "BBox", "Detector", "TrackedVehicle", "VehicleRole", "YoloVehicleTracker",
    "YoloeDetector", "select_ego_candidate",
]

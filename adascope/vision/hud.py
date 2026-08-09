"""Open-vocabulary HUD element detection."""

from dataclasses import dataclass
from ..detection import BBox, YoloeDetector


@dataclass(frozen=True)
class HudBox:
    box: BBox
    element: str
    confidence: float


class HudDetector:
    def __init__(self, checkpoint: str, prompts: dict[str, list[str]], conf: float):
        self.label_elements = {prompt: name for name, values in prompts.items() for prompt in values}
        self.detector = YoloeDetector(checkpoint, list(self.label_elements))
        self.conf = conf

    def detect_hud(self, frame):
        boxes, _ = self.detector.predict_labelled(frame, self.conf)
        return [HudBox(box, self.label_elements[box.label], box.confidence)
                for box in boxes if box.label in self.label_elements]


def hud_flags(detections, elements):
    found = {item.element for item in detections}
    return {element: element in found for element in elements}

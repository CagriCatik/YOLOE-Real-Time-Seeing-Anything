"""Open-vocabulary driving-area detections and lane assignment."""

from dataclasses import dataclass
from ..detection import BBox, YoloeDetector


@dataclass(frozen=True)
class AreaBox:
    box: BBox
    color: str
    confidence: float
    tip_x: float | None = None


class DrivingAreaDetector:
    def __init__(self, checkpoint: str, prompts: dict[str, list[str]], conf: float):
        self.label_colors = {prompt: color for color, values in prompts.items() for prompt in values}
        self.detector = YoloeDetector(checkpoint, list(self.label_colors))
        self.conf = conf

    def detect_areas(self, frame):
        boxes, _ = self.detector.predict_labelled(frame, self.conf)
        return [AreaBox(box, self.label_colors.get(box.label, ""), box.confidence) for box in boxes]


def lane_colors_from_areas(areas, ego_cx: float, frame_size, config):
    width, height = frame_size
    chosen = {"left": None, "ego": None, "right": None}
    ranked = {name: (-1, -1.0) for name in chosen}
    # Rangfolge kommt aus der Config, nicht aus einer Konstante hier.
    priority = {color: rank for rank, color in enumerate(config.priority, 1)}
    for area in areas:
        if area.color not in priority:
            continue
        x1 = min(width, max(0, area.box.x1)); x2 = min(width, max(0, area.box.x2))
        y1 = min(height, max(0, area.box.y1)); y2 = min(height, max(0, area.box.y2))
        if x2 <= x1 or y2 <= y1 or ((x2 - x1) * (y2 - y1)) / (width * height) > config.max_box_frac:
            continue
        x = area.tip_x if area.tip_x is not None else (x1 + x2) / 2
        frac = x / width
        lane = "ego" if abs(frac - ego_cx) <= config.side_dead_zone else ("left" if frac < ego_cx else "right")
        score = (priority[area.color], area.confidence)
        if score > ranked[lane]:
            ranked[lane] = score
            chosen[lane] = area.color
    return chosen

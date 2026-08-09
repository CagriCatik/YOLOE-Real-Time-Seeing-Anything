"""Pure per-frame lane analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from .carpet import detect_carpet
from .driving_area import lane_colors_from_areas
from .geometry import assign_region, polys_from_rois
from .hud import hud_flags

EGO = "ego_vehicle"
LANES = ("left", "ego", "right")


def lane_state(color, lane):
    if color == "red":
        return "blocked"
    if lane == "ego":
        return "drivable" if color == "white" else "clear"
    return "available" if color == "green" else "clear"


@dataclass
class FrameResult:
    name: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    lane_states: dict[str, str] = field(default_factory=dict)
    assignments: list[tuple[object, str | None]] = field(default_factory=list)
    area_boxes: list[object] = field(default_factory=list)
    hud: dict[str, bool] | None = None
    hud_boxes: list[object] = field(default_factory=list)

    def csv_row(self):
        row = {"frame": self.name}
        row.update({f"veh_{lane}": self.counts.get(lane, 0) for lane in LANES})
        row.update({f"state_{lane}": self.lane_states.get(lane, "clear") for lane in LANES})
        if self.hud is not None:
            row.update({f"hud_{name}": int(value) for name, value in self.hud.items()})
        return row


def analyse_frame(detector, frame, config, conf=None, name="", area_detector=None, hud_detector=None):
    height, width = frame.shape[:2]
    polys = polys_from_rois(config.rois, width, height)
    counts = {lane: 0 for lane in LANES}
    assignments = []
    for box in detector.detect(frame, config.model.conf if conf is None else conf):
        cx, cy = box.center
        if config.ego_box and _inside_normalized(cx, cy, width, height, config.ego_box):
            lane = EGO
        else:
            lane = assign_region(polys, cx, cy)
            if lane in counts:
                counts[lane] += 1
        assignments.append((box, lane))
    areas = area_detector.detect_areas(frame) if area_detector else []
    if area_detector:
        ego_cx = (config.ego_box[0] + config.ego_box[2]) / 2 if config.ego_box else .5
        colors = lane_colors_from_areas(areas, ego_cx, (width, height), config.driving_area)
    else:
        colors = detect_carpet(frame, config.rois, config.carpet)
    hud_boxes = hud_detector.detect_hud(frame) if hud_detector else []
    flags = hud_flags(hud_boxes, config.hud.prompts) if hud_detector else None
    return FrameResult(name, counts, {lane: lane_state(colors.get(lane), lane) for lane in LANES},
                       assignments, areas, flags, hud_boxes)


def _inside_normalized(x, y, width, height, box):
    x0, y0, x1, y1 = box
    return x0 <= x / width <= x1 and y0 <= y / height <= y1

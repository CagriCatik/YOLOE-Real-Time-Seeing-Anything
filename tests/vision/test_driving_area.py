"""ROI-free YOLO-E driving-area logic with a fake labelled detector (no model load).

On the 100x100 test frame the ego centre is x=50 (ego_box 0.40-0.60) and the
side dead zone is 0.05 -> a carpet tip within x 45-55 reads as the ego lane.
"""

from __future__ import annotations

from adascope.vision.analysis import analyse_frame
from adascope.config import DetectionConfig
from adascope.detection import BBox
from adascope.vision.driving_area import AreaBox, lane_colors_from_areas

from conftest import FakeDetector


class FakeAreaDetector:
    """Returns a fixed list of labelled area boxes, ignoring the frame."""

    def __init__(self, areas: list[AreaBox]):
        self.areas = areas

    def detect_areas(self, frame) -> list[AreaBox]:
        return list(self.areas)


def _lanes(areas, config, ego_cx=0.5):
    return lane_colors_from_areas(areas, ego_cx, (100, 100), config.driving_area)


def test_box_centre_right_of_ego_is_right_lane(config):
    areas = [AreaBox(BBox(60, 20, 90, 60), "green", 0.5)]   # centre x=75
    assert _lanes(areas, config) == {"left": None, "ego": None, "right": "green"}


def test_box_centre_left_of_ego_is_left_lane(config):
    areas = [AreaBox(BBox(10, 20, 40, 60), "green", 0.5)]   # centre x=25
    assert _lanes(areas, config) == {"left": "green", "ego": None, "right": None}


def test_box_centred_on_ego_is_ego_lane(config):
    areas = [AreaBox(BBox(30, 20, 72, 60), "red", 0.5)]     # centre x=51, in dead zone
    assert _lanes(areas, config) == {"left": None, "ego": "red", "right": None}


def test_mask_tip_overrides_box_centre(config):
    # Swoosh box is centred on ego (centre x=50) but its tip points left.
    areas = [AreaBox(BBox(25, 20, 75, 60), "green", 0.5, tip_x=30.0)]
    assert _lanes(areas, config) == {"left": "green", "ego": None, "right": None}


def test_whole_frame_box_ignored(config):
    # A near-frame-sized detection carries no lane information; max_box_frac drops it.
    areas = [AreaBox(BBox(0, 0, 100, 100), "red", 0.9)]
    assert _lanes(areas, config) == {"left": None, "ego": None, "right": None}


def test_red_beats_green_in_same_lane(config):
    areas = [
        AreaBox(BBox(60, 20, 90, 60), "green", 0.9),
        AreaBox(BBox(62, 22, 88, 58), "red", 0.2),
    ]
    assert _lanes(areas, config)["right"] == "red"


def test_higher_confidence_wins_for_same_color(config):
    areas = [
        AreaBox(BBox(60, 20, 90, 60), "green", 0.2),
        AreaBox(BBox(62, 22, 88, 58), "green", 0.8),
    ]
    lanes = _lanes(areas, config)
    assert lanes["right"] == "green"


def test_unknown_color_label_ignored(config):
    areas = [AreaBox(BBox(60, 20, 90, 60), "purple", 0.9)]
    assert _lanes(areas, config)["right"] is None


def test_box_clipped_to_frame_bounds(config):
    areas = [AreaBox(BBox(60, -50, 150, 60), "green", 0.5)]
    assert _lanes(areas, config)["right"] == "green"


def test_analyse_frame_with_area_detector(config, blank_frame):
    # ego_box 0.40-0.60 -> ego centre 0.5; green tip right, white centred on ego.
    area_det = FakeAreaDetector([
        AreaBox(BBox(55, 10, 90, 50), "green", 0.5, tip_x=80.0),   # right -> available
        AreaBox(BBox(40, 20, 60, 90), "white", 0.5, tip_x=50.0),   # ego -> drivable
    ])
    result = analyse_frame(FakeDetector([]), blank_frame, config, conf=0.1,
                           area_detector=area_det)
    assert result.lane_states == {"left": "clear", "ego": "drivable", "right": "available"}
    assert len(result.area_boxes) == 2


def test_analyse_frame_without_area_detector_uses_hsv(config, blank_frame):
    # Black frame, HSV path: no carpet anywhere.
    result = analyse_frame(FakeDetector([]), blank_frame, config, conf=0.1)
    assert result.lane_states == {"left": "clear", "ego": "clear", "right": "clear"}
    assert result.area_boxes == []


def test_config_without_driving_area_section_gets_defaults(raw_config):
    raw_config.pop("driving_area")
    cfg = DetectionConfig.from_dict(raw_config)
    assert cfg.driving_area.enabled is True   # YOLOE is the default backend
    assert set(cfg.driving_area.prompts) == {"green", "red", "white"}
    assert 0 < cfg.driving_area.max_box_frac <= 1
    assert 0 <= cfg.driving_area.side_dead_zone < 0.5


def test_config_rejects_unknown_prompt_colors(raw_config):
    raw_config["driving_area"]["prompts"]["purple"] = ["purple carpet"]
    try:
        DetectionConfig.from_dict(raw_config)
    except ValueError as exc:
        assert "purple" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown prompt colour")

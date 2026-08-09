"""HUD element detection logic with a fake labelled detector (no model load)."""

from __future__ import annotations

from adascope.vision.analysis import analyse_frame
from adascope.config import DetectionConfig
from adascope.detection import BBox
from adascope.vision.hud import HudBox, hud_flags

from conftest import FakeDetector


class FakeHudDetector:
    """Returns a fixed list of labelled HUD boxes, ignoring the frame."""

    def __init__(self, boxes: list[HudBox]):
        self.boxes = boxes

    def detect_hud(self, frame) -> list[HudBox]:
        return list(self.boxes)


def test_hud_flags_present_and_absent():
    detections = [HudBox(BBox(10, 0, 30, 10), "takeover_request", 0.8)]
    flags = hud_flags(detections, ["takeover_request", "speed_limit_sign"])
    assert flags == {"takeover_request": True, "speed_limit_sign": False}


def test_hud_flags_dedupe_multiple_detections():
    detections = [
        HudBox(BBox(10, 0, 30, 10), "speed_limit_sign", 0.8),
        HudBox(BBox(50, 0, 70, 10), "speed_limit_sign", 0.6),
    ]
    flags = hud_flags(detections, ["speed_limit_sign"])
    assert flags == {"speed_limit_sign": True}


def test_analyse_frame_with_hud_detector(config, blank_frame):
    hud_det = FakeHudDetector([HudBox(BBox(40, 0, 60, 10), "takeover_request", 0.9)])
    result = analyse_frame(FakeDetector([]), blank_frame, config, conf=0.1,
                           hud_detector=hud_det)
    assert result.hud == {"takeover_request": True, "speed_limit_sign": False}
    assert len(result.hud_boxes) == 1


def test_csv_row_includes_hud_columns_when_enabled(config, blank_frame):
    hud_det = FakeHudDetector([HudBox(BBox(40, 0, 60, 10), "speed_limit_sign", 0.9)])
    row = analyse_frame(FakeDetector([]), blank_frame, config, conf=0.1,
                        hud_detector=hud_det).csv_row()
    assert row["hud_takeover_request"] == 0
    assert row["hud_speed_limit_sign"] == 1


def test_csv_row_has_no_hud_columns_without_detector(config, blank_frame):
    row = analyse_frame(FakeDetector([]), blank_frame, config, conf=0.1).csv_row()
    assert not [k for k in row if k.startswith("hud_")]


def test_config_without_hud_section_gets_defaults(raw_config):
    raw_config.pop("hud")
    cfg = DetectionConfig.from_dict(raw_config)
    assert cfg.hud.enabled is False
    # Default prompt set covers the cluster's icons, warnings and pop-ups.
    assert {"takeover_request", "warning_popup", "lane_assist_warning",
            "speed_limit_sign", "acc_indicator", "lane_change_arrow",
            "sensor_arc"} <= set(cfg.hud.prompts)

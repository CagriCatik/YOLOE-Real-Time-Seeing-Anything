"""Per-frame analysis using a fake detector (no model load)."""

from __future__ import annotations

from adascope.vision.analysis import EGO, analyse_frame, lane_state
from adascope.detection import BBox

from conftest import FakeDetector


def test_counts_other_vehicles_per_lane(config, blank_frame):
    # Boxes centred in each lane (frame is 100x100), all in the upper half so none
    # fall in the ego_box (y 0.7-1.0): left ~x15, ego ~x50, right ~x85.
    detector = FakeDetector([
        BBox(10, 20, 20, 40),   # center (15, 30) -> left
        BBox(45, 20, 55, 40),   # center (50, 30) -> ego (a real lead car)
        BBox(80, 20, 90, 40),   # center (85, 30) -> right
        BBox(82, 5, 88, 25),    # center (85, 15) -> right (second)
    ])
    result = analyse_frame(detector, blank_frame, config, conf=0.1, name="f.jpg")
    assert result.counts == {"left": 1, "ego": 1, "right": 2}
    assert len(result.assignments) == 4


def test_ego_vehicle_excluded_from_counts(config, blank_frame):
    # Box centred at (50, 85) -> normalized (0.5, 0.85) -> inside ego_box -> dropped.
    detector = FakeDetector([BBox(45, 80, 55, 90)])
    result = analyse_frame(detector, blank_frame, config, conf=0.1)
    assert result.counts == {"left": 0, "ego": 0, "right": 0}
    assert result.assignments[0][1] == EGO


def test_box_outside_all_rois_not_counted(config, blank_frame):
    detector = FakeDetector([BBox(31, 20, 33, 40)])  # center x=0.32 -> gap
    result = analyse_frame(detector, blank_frame, config, conf=0.1)
    assert result.counts == {"left": 0, "ego": 0, "right": 0}
    assert result.assignments[0][1] is None


def test_side_lane_state_mapping():
    assert lane_state("green", "left") == "available"
    assert lane_state("red", "right") == "blocked"
    assert lane_state("white", "left") == "clear"   # side white = lane marking
    assert lane_state(None, "right") == "clear"


def test_ego_lane_state_mapping():
    assert lane_state("white", "ego") == "drivable"
    assert lane_state("red", "ego") == "blocked"
    assert lane_state("green", "ego") == "clear"     # sensor-arc bleed, ignored
    assert lane_state(None, "ego") == "clear"


def test_csv_row_schema(config, blank_frame):
    detector = FakeDetector([BBox(45, 20, 55, 40)])
    row = analyse_frame(detector, blank_frame, config, conf=0.1, name="frame_0001.jpg").csv_row()
    assert row["frame"] == "frame_0001.jpg"
    assert row["veh_ego"] == 1
    assert set(row) == {
        "frame", "veh_left", "veh_ego", "veh_right",
        "state_left", "state_ego", "state_right",
    }

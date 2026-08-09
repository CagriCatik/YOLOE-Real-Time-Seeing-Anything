"""Carpet HSV detection: green/red/white precedence and the min-fraction gate."""

from __future__ import annotations

import numpy as np

from adascope.vision.carpet import detect_carpet


def _fill(frame, roi_fracs, bgr):
    """Paint a BGR colour over a single ROI's bounding rect (ROIs here are rects)."""
    h, w = frame.shape[:2]
    xs = [int(x * w) for x, _ in roi_fracs]
    ys = [int(y * h) for _, y in roi_fracs]
    frame[min(ys):max(ys), min(xs):max(xs)] = bgr


def test_detects_green_and_red_per_side(config):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _fill(frame, config.rois["left"], (0, 255, 0))    # pure green BGR
    _fill(frame, config.rois["right"], (0, 0, 255))   # pure red BGR
    out = detect_carpet(frame, config.rois, config.carpet)
    assert out["left"] == "green"
    assert out["right"] == "red"


def test_empty_frame_has_no_carpet(config):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = detect_carpet(frame, config.rois, config.carpet)
    assert out == {"left": None, "ego": None, "right": None}


def test_white_detected_in_ego_lane(config):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _fill(frame, config.rois["ego"], (130, 130, 130))   # low-sat mid-value -> white
    out = detect_carpet(frame, config.rois, config.carpet)
    assert out["ego"] == "white"


def test_detect_in_sides_reported(config):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = detect_carpet(frame, config.rois, config.carpet)
    assert set(out) == {"left", "ego", "right"}       # ego now included for white path

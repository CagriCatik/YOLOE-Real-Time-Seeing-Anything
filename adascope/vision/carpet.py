"""HSV-based driving-area colour detection."""

from __future__ import annotations
import cv2
import numpy as np
from .geometry import polys_from_rois


def _mask_range(hsv, bounds):
    return cv2.inRange(hsv, np.asarray(bounds[0], np.uint8), np.asarray(bounds[1], np.uint8))


def detect_carpet(frame, rois, config):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = _mask_range(hsv, config.green_hsv)
    red = np.zeros((height, width), np.uint8)
    for hsv_range in config.red_hsv:
        red = cv2.bitwise_or(red, _mask_range(hsv, hsv_range))
    white = _mask_range(hsv, config.white_hsv) if config.white_hsv else None
    output = {name: None for name in rois}
    polygons = polys_from_rois(rois, width, height)
    for name in config.detect_in:
        if name not in polygons:
            continue
        roi_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(roi_mask, [polygons[name]], 255)
        area = max(1, cv2.countNonZero(roi_mask))
        fractions = {
            "green": cv2.countNonZero(cv2.bitwise_and(green, roi_mask)) / area,
            "red": cv2.countNonZero(cv2.bitwise_and(red, roi_mask)) / area,
        }
        # Red wins ties: a blocking indication is the safety-conservative result.
        if fractions["red"] >= config.min_frac and fractions["red"] >= fractions["green"]:
            output[name] = "red"
        elif fractions["green"] >= config.min_frac:
            output[name] = "green"
        elif white is not None and cv2.countNonZero(cv2.bitwise_and(white, roi_mask)) / area >= config.white_min_frac:
            output[name] = "white"
    return output

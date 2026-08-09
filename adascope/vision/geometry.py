"""Resolution-independent lane geometry helpers."""

from __future__ import annotations
import cv2
import numpy as np


def poly_from_fractions(points, width: int, height: int) -> np.ndarray:
    return np.asarray([[round(x * width), round(y * height)] for x, y in points], dtype=np.int32)


def polys_from_rois(rois, width: int, height: int):
    return {name: poly_from_fractions(points, width, height) for name, points in rois.items()}


def assign_region(polys, x: float, y: float):
    for name, polygon in polys.items():
        if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
            return name
    return None

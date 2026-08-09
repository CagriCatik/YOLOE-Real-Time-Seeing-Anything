"""Gemeinsame Fixtures.

Leitidee: **kein Test laedt ein Modell und keiner braucht das Dateisystem.**
Jede Config-Dataclass ist ohne Datei konstruierbar, der Detektor liegt hinter
einem Port. Deshalb reichen ein Fake-Detektor und ein synthetisches Bild.
"""

from __future__ import annotations

import numpy as np
import pytest

from adascope.config import (
    BevConfig, DetectionConfig, EventConfig, IndexConfig, LaneConfig,
    BoundaryTrackConfig, EgoMotionConfig, PipelineConfig, Settings, WindowConfig,
)
from adascope.detection import BBox, TrackedVehicle


# --------------------------------------------------------------------------- #
# YOLOE-Domaene                                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def raw_config() -> dict:
    """Kleine, aber vollstaendige Detektionskonfiguration ueber alle Felder."""
    return {
        "model": {"checkpoint": "yoloe-11l-seg.pt", "classes": ["car", "truck"], "conf": 0.1},
        "rois": {
            # Drei ueberschneidungsfreie Spuren quer durchs Bild (Bruchteile).
            "left": [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]],
            "ego": [[0.35, 0.0], [0.65, 0.0], [0.65, 1.0], [0.35, 1.0]],
            "right": [[0.7, 0.0], [1.0, 0.0], [1.0, 1.0], [0.7, 1.0]],
        },
        "roi_colors": {"left": [255, 180, 0], "ego": [255, 255, 255], "right": [0, 180, 255]},
        "crop_box": [0.1, 0.1, 0.9, 0.9],
        # Ego-Glyph sitzt unten mittig: Detektionen dort zaehlen nicht.
        "ego_box": [0.40, 0.7, 0.60, 1.0],
        "carpet": {
            "detect_in": ["left", "ego", "right"],
            "min_frac": 0.04,
            "white_min_frac": 0.18,
            "green_hsv": [[38, 70, 70], [90, 255, 255]],
            "red_hsv": [[[0, 90, 80], [12, 255, 255]], [[168, 90, 80], [180, 255, 255]]],
            "white_hsv": [[0, 0, 95], [179, 45, 170]],
        },
        "driving_area": {
            "enabled": False, "conf": 0.03, "max_box_frac": 0.5, "side_dead_zone": 0.05,
            "prompts": {"green": ["green carpet"], "red": ["red carpet"], "white": ["white path"]},
        },
        "hud": {
            "enabled": False, "conf": 0.15,
            "prompts": {"takeover_request": ["yellow steering wheel warning icon"],
                        "speed_limit_sign": ["round speed limit sign with red ring"]},
        },
    }


@pytest.fixture
def config(raw_config) -> DetectionConfig:
    return DetectionConfig.from_dict(raw_config)


class FakeDetector:
    """Liefert eine feste Boxliste und ignoriert den Frame -- kein Modell noetig."""

    def __init__(self, boxes: list[BBox]):
        self.boxes = boxes

    def detect(self, frame, conf: float) -> list[BBox]:
        return list(self.boxes)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Spur-Domaene                                                                #
# --------------------------------------------------------------------------- #
@pytest.fixture
def settings() -> Settings:
    """Settings ausschliesslich aus Code-Defaults -- ohne YAML-Dateien."""
    from pathlib import Path

    from adascope.config import DebugConfig, TrackingConfig

    return Settings(
        root=Path("."), config_dir=Path("config"), scenario=None,
        lane=LaneConfig(), bev=BevConfig(), tracking=TrackingConfig(),
        indexing=IndexConfig(), events=EventConfig(), pipeline=PipelineConfig(),
        windows=WindowConfig(), boundaries=BoundaryTrackConfig(),
        egomotion=EgoMotionConfig(), debug=DebugConfig(),
    )


@pytest.fixture
def lane_config() -> LaneConfig:
    return LaneConfig()


@pytest.fixture
def bev_config() -> BevConfig:
    return BevConfig()


def make_vehicle(bbox: tuple[int, int, int, int], track_id: int = 1,
                 role: str = "co") -> TrackedVehicle:
    return TrackedVehicle(track_id, 2, "car", 0.9, bbox, role)


@pytest.fixture
def straight_lanes_frame(lane_config) -> np.ndarray:
    """Synthetische Szene: vier gerade, senkrechte Markierungen.

    Senkrecht statt perspektivisch, damit die erwarteten Ergebnisse aus der
    Geometrie folgen und nicht aus einem Referenzbild abgelesen werden muessen.
    """
    import cv2

    frame = np.zeros((457, 1209, 3), np.uint8)
    for x in (200, 450, 700, 950):
        cv2.line(frame, (x, lane_config.y_top), (x, lane_config.y_bottom),
                 (255, 255, 255), 4)
    return frame

"""YOLOE-Domaene: ROI-Geometrie, Carpet-Farben, Driving Area, HUD, Analyse.

Reine Domaenenlogik -- haengt an numpy/OpenCV und am `Detector`-Port, nie an
`ultralytics` oder argparse. Dadurch ist sie mit einem Fake-Detektor testbar,
ohne ein 70-MB-Modell zu laden.
"""

from .analysis import FrameResult, analyse_frame
from .carpet import detect_carpet
from .driving_area import DrivingAreaDetector, lane_colors_from_areas
from .geometry import assign_region, poly_from_fractions, polys_from_rois
from .hud import HudDetector, hud_flags

__all__ = [
    "DrivingAreaDetector", "FrameResult", "HudDetector", "analyse_frame",
    "assign_region", "detect_carpet", "hud_flags", "lane_colors_from_areas",
    "poly_from_fractions", "polys_from_rois",
]

"""Spur-Domaene: klassische CV-Spurerkennung, Bodenebene, Ereignisse.

Die Stufen bauen strikt aufeinander auf und kennen jeweils nur die vorige:

    detection  Bildebene    Hough-Linien mit Rolle
    bev        Bodenebene   Homographie, Korridore, Footprints
    indexing   Bodenebene   ego-relative Spurnummern
    events     Zeit         cut_in / cut_out / aborted / ego_lane_change
    pipeline   Sequenz      verkettet alles und haelt den Zustand

Kein Modul hier rendert, schreibt Dateien oder importiert `ultralytics`.
"""

from .bev import (
    Footprint, assign_lane, build_homography, build_lane_mask, corridors_from,
    find_lane_boundaries, footprint_is_plausible, homography_from_pair,
    lane_histogram, outer_solid_pair, peaks_from_histogram, project_footprint,
    restrict_to_driving_area, source_points, warp_lane_mask,
)
from .boundaries import Boundaries, fit_curve
from .egomotion import EgoMotion, EgoMotionDetector
from .tracking_ids import BoundaryTracker, crossing_direction
from .detection import LaneLine, LaneResult, Role, detect_lanes, robust_line
from .windows import find_boundaries as find_boundaries_windows, start_positions, trace
from .events import CutInTracker, Event, EventKind, State
from .mapping import LaneNumbering, assign_lane_numbers
from .indexing import (
    Lane, build_lane_index, ego_overlap, estimate_lane_width,
    estimate_lane_width_by_multiples, locate,
)
from .pipeline import (
    FrameAnalysis, HomographyState, HomographyTracker, SequencePipeline,
    VehicleOccupancy, ego_reference_footprint, road_vehicles,
)

__all__ = [
    "Boundaries", "BoundaryTracker", "LaneNumbering", "assign_lane_numbers", "CutInTracker", "EgoMotion",
    "EgoMotionDetector", "crossing_direction", "Event", "EventKind", "Footprint", "FrameAnalysis",
    "HomographyState", "HomographyTracker", "Lane", "LaneLine", "LaneResult",
    "Role", "SequencePipeline", "State", "VehicleOccupancy", "assign_lane",
    "build_homography", "build_lane_index", "build_lane_mask", "corridors_from",
    "detect_lanes", "ego_overlap", "ego_reference_footprint",
    "estimate_lane_width", "estimate_lane_width_by_multiples", "find_lane_boundaries", "footprint_is_plausible",
    "homography_from_pair", "lane_histogram", "locate", "outer_solid_pair",
    "peaks_from_histogram", "project_footprint", "road_vehicles", "robust_line",
    "restrict_to_driving_area", "source_points", "warp_lane_mask",
    "find_boundaries_windows", "fit_curve", "start_positions", "trace",
]

"""Bodenebene: Linienwahl, Homographie, Histogramm, Footprint-Plausibilitaet."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from adascope.config import BevConfig, LaneConfig
from adascope.lanes.bev import (
    Footprint, assign_lane, build_homography, build_lane_mask, corridors_from,
    footprint_is_plausible, homography_from_pair, lane_histogram, outer_solid_pair,
    peaks_from_histogram, project_footprint,
)
from adascope.lanes.detection import LaneLine, LaneResult


def line(x_bottom: float, role: str) -> LaneLine:
    """Senkrechte Linie bei x_bottom mit fester Rolle."""
    return LaneLine(m=0.0, b=x_bottom, x_bottom=x_bottom, support=5, role=role)


# --------------------------------------------------------------------------- #
# Linienwahl fuer die Homographie                                             #
# --------------------------------------------------------------------------- #
def test_picks_the_outermost_solid_line_per_side():
    """Der stumme Fehler, den `{L.role: L}` hatte.

    Bei mehreren `left_solid` behielt die Komprehension die zuletzt einsortierte
    -- die dem Ego naechste, nicht die Fahrbahnkante. Die Homographie wurde
    dadurch schiefgezogen, ohne dass ein Fehler auftrat.
    """
    result = LaneResult(lines=[line(64, "left_solid"), line(318, "left_solid"),
                               line(500, "right_dashed"), line(853, "right_solid")])
    left, right = outer_solid_pair(result)
    assert left.x_bottom == 64          # nicht 318
    assert right.x_bottom == 853


def test_no_pair_when_a_solid_role_is_missing():
    result = LaneResult(lines=[line(100, "left_dashed"), line(400, "right_dashed")])
    assert outer_solid_pair(result) is None


def test_build_homography_fails_loudly_without_a_pair(lane_config, bev_config):
    # Frueher ein KeyError mitten in der Pipeline; jetzt eine Aussage.
    with pytest.raises(ValueError, match="Randlinien"):
        build_homography(np.zeros((457, 1209, 3), np.uint8), lane_config, bev_config)


def test_homography_maps_the_pair_onto_the_target_rectangle(lane_config, bev_config):
    pair = (line(100, "left_solid"), line(900, "right_solid"))
    H = homography_from_pair(pair, lane_config, bev_config)
    near = cv2.perspectiveTransform(
        np.float32([[[100, lane_config.y_bottom]], [[900, lane_config.y_bottom]]]), H)
    assert near[0, 0, 0] == pytest.approx(bev_config.x_left, abs=0.5)
    assert near[1, 0, 0] == pytest.approx(bev_config.x_right, abs=0.5)


# --------------------------------------------------------------------------- #
# Maske                                                                       #
# --------------------------------------------------------------------------- #
def test_vehicle_boxes_are_punched_out_of_the_lane_mask(lane_config):
    """Der Detektor raeumt die Spurmaske auf -- kooperierende Module.

    Ohne das erzeugen helle Fahrzeugdaecher Saeulen im Spaltenhistogramm und
    damit Spurgrenzen, wo keine sind.
    """
    frame = np.full((457, 1209, 3), 255, np.uint8)
    box = (400, 100, 500, 200)
    without = build_lane_mask(frame, lane_config, [])
    with_box = build_lane_mask(frame, lane_config, [box])
    assert without[150, 450] > 0
    assert with_box[150, 450] == 0


# --------------------------------------------------------------------------- #
# Spaltenhistogramm                                                           #
# --------------------------------------------------------------------------- #
def test_peaks_need_the_configured_minimum_distance():
    """Der gemessene Fehlermodus in einem Satz.

    Zwei dicht beieinanderliegende Saeulen sind eine Markierung, keine zwei
    Spurgrenzen. Mit peak_min_distance=25 ueberlebten beide und erzeugten
    Kleinstkorridore; mit 55 bleibt die staerkere uebrig.
    """
    hist = np.zeros(500, np.float32)
    hist[200], hist[230] = 40, 60          # 30 px auseinander
    assert len(peaks_from_histogram(hist, BevConfig(peak_min_distance=25))) == 2
    assert peaks_from_histogram(hist, BevConfig(peak_min_distance=55)) == [230]


def test_peaks_below_the_pixel_threshold_are_ignored():
    hist = np.zeros(500, np.float32)
    hist[100] = 5
    assert peaks_from_histogram(hist, BevConfig(peak_min_pixels=10)) == []


def test_histogram_counts_set_pixels_per_column():
    mask = np.zeros((100, 20), np.uint8)
    mask[:40, 10] = 255
    assert lane_histogram(mask).argmax() == 10


def test_corridors_are_the_gaps_between_boundaries():
    assert corridors_from([10, 90, 170]) == [(10.0, 90.0), (90.0, 170.0)]
    assert corridors_from([10]) == []


# --------------------------------------------------------------------------- #
# Footprint                                                                   #
# --------------------------------------------------------------------------- #
def test_only_the_bottom_edge_is_projected(lane_config, bev_config):
    """Die Bruecke zwischen den Ebenen ist genau eine Kante.

    Ein Fahrzeug hat Bauhoehe; nur seine Radaufstandslinie liegt in der
    Bodenebene. Deshalb darf die Projektion nicht von y1 abhaengen.
    """
    H = homography_from_pair((line(100, "left_solid"), line(900, "right_solid")),
                             lane_config, bev_config)
    flach = project_footprint("a", (400, 250, 500, 280), H)
    hoch = project_footprint("b", (400, 50, 500, 280), H)
    assert flach.x_left == pytest.approx(hoch.x_left)
    assert flach.x_right == pytest.approx(hoch.x_right)


@pytest.mark.parametrize("width,expected", [
    (18, False),    # 0.24 Spurbreiten -- unter fp_width_min_ratio=0.25
    (20, True),     # 0.26 -- gerade noch innerhalb
    (40, True),     # PKW
    (60, True),     # LKW
    (73, False),    # 0.96 -- knapp ueber fp_width_max_ratio=0.95
    (90, False),    # 1.18 Spurbreiten: physikalisch unmoeglich, Fernfeldartefakt
])
def test_footprint_plausibility_against_the_lane_width(width, expected, bev_config):
    fp = Footprint("x", 100.0, 100.0 + width, 300.0)
    assert footprint_is_plausible(fp, 76.0, bev_config) is expected


def test_assign_lane_picks_the_largest_overlap():
    fp = Footprint("x", 85.0, 125.0, 300.0)     # 15 px in L0, 25 px in L1
    index, ratios = assign_lane(fp, [(0.0, 100.0), (100.0, 200.0)])
    assert index == 1
    assert ratios[0] == pytest.approx(0.375, abs=0.01)


def test_assign_lane_returns_minus_one_without_overlap():
    fp = Footprint("x", 900.0, 940.0, 300.0)
    assert assign_lane(fp, [(0.0, 100.0)])[0] == -1

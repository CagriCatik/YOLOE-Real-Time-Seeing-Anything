"""Deterministisches geometrisches Hough-Clustering."""

from dataclasses import replace
from itertools import permutations

from adascope.config import LaneConfig
from adascope.lanes.detection import (
    cluster_segments, segment_incompatibility, segments_compatible,
)


def segment(m: float, b: float, y1: int, y2: int, cfg: LaneConfig):
    x1, x2 = round(m * y1 + b), round(m * y2 + b)
    return (m * cfg.y_bottom + b, m, (x1, y1, x2, y2))


def signature(clusters):
    return sorted(sorted(round(item[0], 2) for item in group) for group in clusters)


def test_union_find_is_independent_of_hough_input_order():
    cfg = LaneConfig(cluster_method="union_find")
    items = [segment(-0.8, 520, 80, 140, cfg),
             segment(-0.8, 520, 160, 230, cfg),
             segment(0.7, 520, 90, 155, cfg),
             segment(0.7, 520, 175, 240, cfg)]
    expected = signature(cluster_segments(items, cfg))
    assert len(expected) == 2
    for order in permutations(items):
        assert signature(cluster_segments(list(order), cfg)) == expected


def test_rejects_a_diagonal_with_incompatible_slope():
    cfg = LaneConfig(cluster_method="union_find")
    lane_mark = segment(-0.8, 520, 100, 180, cfg)
    diagonal = segment(0.8, 50, 150, 230, cfg)
    assert not segments_compatible(lane_mark, diagonal, cfg)


def test_rejects_segments_that_only_meet_locally_but_diverge_at_the_top():
    cfg = replace(LaneConfig(), cluster_max_lateral_gap=50,
                  cluster_max_top_dist=20)
    first = segment(-0.8, 520, 180, 240, cfg)
    second = segment(-0.5, 450, 180, 240, cfg)
    assert not segments_compatible(first, second, cfg)


def test_rejects_a_coherent_diagonal_outside_the_road_vanishing_region():
    cfg = LaneConfig(cluster_max_slope_diff=0.5,
                     cluster_vanishing_x_tolerance=220)
    first = segment(1.2, 40, 100, 160, cfg)
    second = segment(1.2, 40, 180, 240, cfg)
    assert segment_incompatibility(first, second, cfg) == "vanishing_region"

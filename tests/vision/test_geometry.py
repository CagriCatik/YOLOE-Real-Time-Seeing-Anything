"""Geometry: fraction->pixel polygons and point-in-polygon ROI assignment."""

from __future__ import annotations

from adascope.vision.geometry import assign_region, poly_from_fractions, polys_from_rois


def test_poly_from_fractions_scales_to_pixels():
    poly = poly_from_fractions([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], 200, 100)
    assert poly.tolist() == [[0, 0], [200, 0], [200, 100]]


def test_assign_region_picks_containing_polygon(config):
    polys = polys_from_rois(config.rois, 100, 100)
    assert assign_region(polys, 15, 50) == "left"     # x=0.15 -> left lane
    assert assign_region(polys, 50, 50) == "ego"      # x=0.50 -> ego lane
    assert assign_region(polys, 85, 50) == "right"    # x=0.85 -> right lane


def test_assign_region_returns_none_outside_all(config):
    polys = polys_from_rois(config.rois, 100, 100)
    assert assign_region(polys, 32, 50) is None       # x=0.32 falls in the gap


def test_assign_region_respects_order_on_overlap():
    # Two overlapping polygons: 'a' is tested first and should win.
    rois = {
        "a": [[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]],
        "b": [[0.4, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.0]],
    }
    polys = polys_from_rois(rois, 100, 100)
    assert assign_region(polys, 50, 50) == "a"        # x=0.5 in both -> first wins

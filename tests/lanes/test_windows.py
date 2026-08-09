"""Sliding Windows gegen Spaltenhistogramm auf gekruemmter Fahrbahn.

Der Vertrag, den diese Tests festschreiben:

1. Auf GERADER Fahrbahn liefern beide Verfahren dasselbe -- Umstellen darf
   nichts kaputt machen.
2. Auf GEKRUEMMTER Fahrbahn verliert das Histogramm Grenzen, die Fenster
   folgen ihnen. Das ist der ganze Grund fuer das Verfahren.
3. Die Grenzen werden dort ausgewertet, wo das Fahrzeug steht -- nicht ueber
   die volle Hoehe gemittelt.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from adascope.config import BevConfig, IndexConfig, WindowConfig
from adascope.lanes import SequencePipeline
from adascope.lanes.boundaries import Boundaries, fit_curve
from adascope.lanes.windows import find_boundaries, start_positions, trace
from adascope.synthetic import SyntheticRoad


def road_with(settings, curvature: float) -> SyntheticRoad:
    return SyntheticRoad(lane=settings.lane, bev=settings.bev, lanes=3,
                         curvature=curvature)


def mask_of(settings, road: SyntheticRoad) -> np.ndarray:
    """Die BEV-Spurmaske derselben Szene, die auch die Pipeline sieht."""
    from adascope.lanes.bev import build_lane_mask

    mask = build_lane_mask(road.frame(), settings.lane, [])
    return cv2.warpPerspective(mask, road.H, (settings.bev.width, settings.bev.height))


def pipeline_with(settings, road: SyntheticRoad, method: str) -> SequencePipeline:
    return SequencePipeline(settings,
                            indexing=IndexConfig(lane_width=road.lane_width),
                            windows=WindowConfig(method=method))


# --------------------------------------------------------------------------- #
# 1. Gerade Fahrbahn: beide Verfahren einig                                   #
# --------------------------------------------------------------------------- #
def test_both_methods_agree_on_a_straight_road(settings):
    road = road_with(settings, curvature=0.0)
    expected = [road.boundary_x(i) for i in range(4)]

    found = find_boundaries(mask_of(settings, road), settings.bev, WindowConfig())
    positions = found.at(settings.bev.y_near - 60)

    assert len(positions) == 4
    for got, want in zip(positions, expected):
        assert got == pytest.approx(want, abs=8)


def test_switching_the_method_does_not_change_a_straight_result(settings):
    road = road_with(settings, curvature=0.0)
    frame, vehicles = road.frame(), [road.ego(), road.vehicle(1, road.lane_center(0), 450)]

    by_histogram = pipeline_with(settings, road, "histogram").process(0, "f", frame, vehicles)
    by_windows = pipeline_with(settings, road, "windows").process(0, "f", frame, vehicles)

    assert len(by_histogram.lanes_rel) == len(by_windows.lanes_rel) == 3
    assert [L.rel for L in by_histogram.lanes_rel] == [L.rel for L in by_windows.lanes_rel]
    assert by_histogram.occupancies[0].rel == by_windows.occupancies[0].rel


# --------------------------------------------------------------------------- #
# 2. Kurve: hier trennt sich das Verfahren                                    #
# --------------------------------------------------------------------------- #
def test_histogram_columns_collapse_with_curvature(settings):
    """Die Begruendung fuer das ganze Modul, gemessen.

    Das Spaltenhistogramm summiert ueber die volle Hoehe. In einer Kurve wandert
    dieselbe Linie ueber viele Spalten -- die Summe verteilt sich. Gemessen faellt
    die hoechste Saeule schon bei maessiger Kruemmung um rund 75 %:

        Kruemmung    0 ->  658
        Kruemmung   60 ->  168   (-74 %)
        Kruemmung  120 ->  137   (-79 %)
    """
    from adascope.lanes.bev import lane_histogram

    straight = lane_histogram(mask_of(settings, road_with(settings, 0.0))).max()
    curved = lane_histogram(mask_of(settings, road_with(settings, 120.0))).max()
    assert curved < straight * 0.4


def test_histogram_loses_a_boundary_in_a_strong_curve(settings):
    """Ab genug Kruemmung faellt eine Grenze ganz aus -- die Fenster nicht."""
    from adascope.lanes.bev import lane_histogram, peaks_from_histogram

    mask = mask_of(settings, road_with(settings, 300.0))
    assert len(peaks_from_histogram(lane_histogram(mask), settings.bev)) < 4
    assert len(find_boundaries(mask, settings.bev, WindowConfig())) == 4


def test_a_moderate_curve_still_yields_four_peaks_but_wrong_ones(settings):
    """Peaks zu zaehlen genuegt nicht -- sie muessen auch stimmen.

    Bei maessiger Kruemmung findet das Histogramm noch vier Peaks, aber sie
    liegen auf einem Mittelwert ueber die Tiefe. Im Fernfeld ist das deutlich
    daneben; die Fenster treffen dort.
    """
    from adascope.lanes.bev import lane_histogram, peaks_from_histogram

    road = road_with(settings, curvature=150.0)
    mask = mask_of(settings, road)
    far = settings.bev.y_near - 420
    expected = sorted(road.boundary_x(i, far) for i in range(4))

    peaks = sorted(peaks_from_histogram(lane_histogram(mask), settings.bev))
    windows = find_boundaries(mask, settings.bev, WindowConfig()).at(far)

    error = lambda xs: max(abs(g - w) for g, w in zip(xs, expected))
    assert len(peaks) == 4
    assert error(windows) < error(peaks) / 2


def test_windows_follow_the_curve(settings):
    road = road_with(settings, curvature=150.0)
    found = find_boundaries(mask_of(settings, road), settings.bev, WindowConfig())

    assert len(found) == 4
    assert found.is_curved


def test_windows_hit_the_true_boundary_at_both_ends(settings):
    """Nah UND fern richtig -- eine Gerade durch die Kurve schafft nur eins."""
    road = road_with(settings, curvature=150.0)
    found = find_boundaries(mask_of(settings, road), settings.bev, WindowConfig())

    for depth in (settings.bev.y_near - 60, settings.bev.y_near - 400):
        expected = sorted(road.boundary_x(i, depth) for i in range(4))
        for got, want in zip(found.at(depth), expected):
            assert got == pytest.approx(want, abs=20), f"bei y={depth}"


def test_a_straight_fit_would_be_wrong_in_the_far_field(settings):
    """Warum Grad 2 und nicht Grad 1.

    Mit `poly_degree=1` filtern die Fenster zwar Ausreisser, koennen der Kurve
    aber nicht folgen -- im Fernfeld liegt die Grenze dann deutlich daneben.
    """
    road = road_with(settings, curvature=150.0)
    mask = mask_of(settings, road)
    far = settings.bev.y_near - 420
    expected = sorted(road.boundary_x(i, far) for i in range(4))

    linear = find_boundaries(mask, settings.bev, WindowConfig(poly_degree=1))
    quadratic = find_boundaries(mask, settings.bev, WindowConfig(poly_degree=2))

    error = lambda b: max(abs(g - w) for g, w in zip(b.at(far), expected))
    assert error(quadratic) < error(linear) / 2


# --------------------------------------------------------------------------- #
# 3. Grenzen dort auswerten, wo das Fahrzeug steht                            #
# --------------------------------------------------------------------------- #
def test_boundaries_are_evaluated_at_the_vehicle_depth(settings):
    """Auf einer Kurve unterscheiden sich nah und fern deutlich.

    Ueber die volle Hoehe gemittelte Grenzen waeren an beiden Enden falsch.
    """
    road = road_with(settings, curvature=150.0)
    found = find_boundaries(mask_of(settings, road), settings.bev, WindowConfig())
    near = found.at(settings.bev.y_near - 40)
    far = found.at(settings.bev.y_near - 420)
    assert abs(far[0] - near[0]) > 40


def test_pipeline_assigns_the_curved_lane_correctly(settings):
    """Der Endnachweis: ein Fahrzeug im Fernfeld einer Kurve.

    Mit dem Histogramm ist die Zuordnung dort nicht verlaesslich; mit Fenstern
    landet das Fahrzeug in der Spur, in die es gesetzt wurde.
    """
    road = road_with(settings, curvature=150.0)
    far = settings.bev.y_near - 380
    vehicles = [road.ego(), road.vehicle(1, road.lane_center(0, far), far)]

    analysis = pipeline_with(settings, road, "windows").process(
        0, "f", road.frame(), vehicles)

    assert len(analysis.lanes_rel) == 3
    assert analysis.occupancies and analysis.occupancies[0].rel == -1


# --------------------------------------------------------------------------- #
# Bausteine                                                                   #
# --------------------------------------------------------------------------- #
def test_trace_walks_upward_and_reports_its_support(settings):
    road = road_with(settings, curvature=0.0)
    mask = mask_of(settings, road)
    ys, xs, hits = trace(mask, int(road.boundary_x(0)), WindowConfig())
    assert hits >= 4
    assert len(ys) == len(xs) == hits
    assert ys[0] > ys[-1]                    # von unten nach oben


def test_start_positions_come_from_the_near_band_only(settings):
    road = road_with(settings, curvature=150.0)
    starts = start_positions(mask_of(settings, road), settings.bev, WindowConfig())
    # Im Nahbereich ist die Kurve noch klein -- die Starts liegen bei den
    # Nahpositionen, nicht irgendwo dazwischen.
    for start in starts:
        assert min(abs(start - road.boundary_x(i)) for i in range(4)) < 20


def test_boundary_without_enough_windows_is_dropped(settings):
    """Ein einzelner Fleck darf keine Spurlinie werden."""
    mask = np.zeros((700, 500), np.uint8)
    mask[640:690, 200:210] = 255                 # nur im untersten Fenster
    found = find_boundaries(mask, settings.bev, WindowConfig(min_windows_hit=4))
    assert len(found) == 0


def test_degree_is_lowered_when_there_are_too_few_points():
    """Ein Polynom 2. Grades durch zwei Punkte ist Rauschen mit drei Zahlen."""
    coefficients = fit_curve(np.array([10.0, 20.0]), np.array([100.0, 110.0]),
                             degree=2, min_points_per_degree=3)
    assert len(coefficients) == 2                # auf Grad 1 gesenkt


def test_histogram_positions_become_degree_zero_curves():
    """Damit der nachgelagerte Code nur eine Darstellung kennt."""
    boundaries = Boundaries.from_positions([300, 100, 200], y_reference=690)
    assert not boundaries.is_curved
    assert boundaries.at(0) == boundaries.at(690) == [100.0, 200.0, 300.0]
    assert boundaries.corridors_at(500) == [(100.0, 200.0), (200.0, 300.0)]


def test_empty_boundaries_are_falsy_and_safe():
    empty = Boundaries()
    assert not empty and len(empty) == 0
    assert empty.at(100) == [] and empty.corridors_at(100) == []


# --------------------------------------------------------------------------- #
# Die Grenze des Verfahrens                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curvature", [150.0, 300.0, 900.0, 2000.0])
def test_stage_one_survives_curvature_so_the_windows_get_their_turn(settings, curvature):
    """Die Kurven-Obergrenze war ein Artefakt der Rollenzuweisung.

    Frueher hielt dieser Test fest: ab Kruemmung 220 findet Stufe 1 kein
    Randlinienpaar mehr, die Homographie faellt aus, und die Fenster kommen gar
    nicht zum Zug. Die Erklaerung dafuer -- der Geradenfit zerfaelle in der
    Kurve -- war falsch.

    Tatsaechlich vergab `classify_lanes` die Rolle `left_solid` nur an Linien
    WEITER AUSSEN als die ego-naechste. In der Kurve wandert die aeussere Linie
    aus dem Bild, es bleibt eine links vom Ego, sie heisst `left_dashed`, und
    `outer_solid_pair` fand nichts. Mit dem Rueckfall auf die aeusserste Linie
    je Seite haelt Stufe 1 ueber den gesamten geprueften Bereich.

    Der Test sagt bewusst NICHT, dass die Geometrie dort gut ist -- nur, dass
    die Fenster ueberhaupt zum Zug kommen. Wie gut sie es dann machen, pruefen
    die Tests darueber.
    """
    from adascope.lanes.bev import outer_solid_pair
    from adascope.lanes.detection import detect_lanes

    road = road_with(settings, curvature=curvature)
    assert outer_solid_pair(detect_lanes(road.frame(), settings.lane)) is not None

    analysis = pipeline_with(settings, road, "windows").process(
        0, "f", road.frame(), [road.ego()])
    assert analysis.h_state == "fresh"
    assert analysis.index_note != "keine Homographie"

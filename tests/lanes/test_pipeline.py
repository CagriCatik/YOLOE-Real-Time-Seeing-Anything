"""Sequenz-Pipeline: Homographie-Persistenz, Fahrbahnfilter, Stufenausfaelle.

Alle Tests laufen ohne Modell: `SequencePipeline.process()` bekommt die
Fahrzeugliste uebergeben, der Detektor ist nicht Teil der Pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from adascope.config import BevConfig, LaneConfig, PipelineConfig
from adascope.detection import TrackedVehicle
from adascope.lanes.detection import LaneLine, LaneResult
from adascope.lanes.pipeline import (
    HomographyTracker, SequencePipeline, road_vehicles,
)


def line(x_bottom: float, role: str) -> LaneLine:
    return LaneLine(m=0.0, b=x_bottom, x_bottom=x_bottom, support=5, role=role)


def pair_result() -> LaneResult:
    return LaneResult(lines=[line(100, "left_solid"), line(900, "right_solid")])


def no_pair_result() -> LaneResult:
    return LaneResult(lines=[line(400, "left_dashed")])


def vehicle(bbox, track_id=1, role="co") -> TrackedVehicle:
    return TrackedVehicle(track_id, 2, "car", 0.9, bbox, role)


# --------------------------------------------------------------------------- #
# Homographie-Persistenz                                                      #
# --------------------------------------------------------------------------- #
def test_fresh_when_both_solid_lines_are_present():
    tracker = HomographyTracker(LaneConfig(), BevConfig(), max_hold=3)
    H, state = tracker.update(pair_result())
    assert state == "fresh" and H is not None


def test_holds_the_last_valid_homography_over_a_dropout():
    """Nur 39 % der Frames liefern beide Randlinien -- ohne Halten waere die
    Pipeline auf zwei Dritteln des Materials tot."""
    tracker = HomographyTracker(LaneConfig(), BevConfig(), max_hold=3)
    fresh, _ = tracker.update(pair_result())
    held, state = tracker.update(no_pair_result())
    assert state == "held"
    assert np.array_equal(held, fresh)


def test_hold_counter_is_reported_and_expires():
    tracker = HomographyTracker(LaneConfig(), BevConfig(), max_hold=2)
    tracker.update(pair_result())
    assert tracker.update(no_pair_result())[1] == "held" and tracker.held_frames == 1
    assert tracker.update(no_pair_result())[1] == "held" and tracker.held_frames == 2
    # Danach lieber nichts als eine veraltete Geometrie.
    assert tracker.update(no_pair_result()) == (None, "none")


def test_none_before_any_valid_homography():
    tracker = HomographyTracker(LaneConfig(), BevConfig())
    assert tracker.update(no_pair_result()) == (None, "none")


def test_recovers_to_fresh_after_a_dropout():
    tracker = HomographyTracker(LaneConfig(), BevConfig(), max_hold=1)
    tracker.update(pair_result())
    tracker.update(no_pair_result())
    tracker.update(no_pair_result())                      # -> none
    assert tracker.update(pair_result())[1] == "fresh"
    assert tracker.held_frames == 0


# --------------------------------------------------------------------------- #
# Fahrbahnfilter                                                              #
# --------------------------------------------------------------------------- #
def test_detections_below_the_road_line_are_discarded():
    """Das Fahrzeug-Icon im Kombiinstrument bekommt eine stabile Track-ID.

    Ohne diesen Filter erzeugt ein Bildschirmelement einen Dauer-Cut-In-
    Kandidaten. Kriterium ist die Bbox-Unterkante, weil genau sie projiziert
    wird.
    """
    lane = LaneConfig()
    auf_der_fahrbahn = vehicle((500, 200, 560, lane.y_bottom - 10))
    im_kombiinstrument = vehicle((880, 350, 940, lane.y_bottom + 100), track_id=17)
    kept = road_vehicles([auf_der_fahrbahn, im_kombiinstrument], lane, margin=25)
    assert [v.track_id for v in kept] == [1]


def test_margin_allows_vehicles_just_below_the_reference_line():
    lane = LaneConfig()
    grenzfall = vehicle((500, 200, 560, lane.y_bottom + 20))
    assert road_vehicles([grenzfall], lane, margin=25) == [grenzfall]
    assert road_vehicles([grenzfall], lane, margin=5) == []


# --------------------------------------------------------------------------- #
# Stufenausfaelle sind Ergebnisse, keine Ausnahmen                            #
# --------------------------------------------------------------------------- #
def test_blank_frame_yields_a_reported_failure_not_a_crash(settings):
    """Im Videobetrieb ist der Stufenausfall der Normalfall.

    Die Pipeline muss ihn benennen (`index_note`), damit die Debug-Ansichten
    ihn anzeigen koennen, statt den Frame zu verschlucken.
    """
    pipeline = SequencePipeline(settings)
    fa = pipeline.process(0, "leer", np.zeros((457, 1209, 3), np.uint8), [])
    assert fa.H is None
    assert fa.h_state == "none"
    assert fa.index_note == "keine Homographie"
    assert fa.lanes_rel == [] and fa.events == []


def test_ego_in_lane_defaults_to_one_when_nothing_was_measured(settings):
    pipeline = SequencePipeline(settings)
    fa = pipeline.process(0, "leer", np.zeros((457, 1209, 3), np.uint8), [])
    assert fa.ego_in_lane == 1.0


def test_worst_state_is_outside_without_occupancies(settings):
    pipeline = SequencePipeline(settings)
    fa = pipeline.process(0, "leer", np.zeros((457, 1209, 3), np.uint8), [])
    assert fa.worst_state == "outside"


def test_state_machine_is_clocked_even_on_failed_frames(settings):
    """Sonst altern fehlende Tracks nicht aus und das Nachlauffenster steht."""
    pipeline = SequencePipeline(settings)
    blank = np.zeros((457, 1209, 3), np.uint8)
    for index in range(5):
        pipeline.process(index, f"f{index}", blank, [])
    assert pipeline.fsm.state_of("ID1") is None


# --------------------------------------------------------------------------- #
# Konfigurierbarkeit                                                          #
# --------------------------------------------------------------------------- #
def test_single_configs_can_be_injected_without_settings():
    # Ein Test soll genau die Config bauen, die er variieren will.
    pipeline = SequencePipeline(pipeline=PipelineConfig(max_hold=99))
    assert pipeline.homography.max_hold == 99
    assert pipeline.bev == BevConfig()


def test_settings_supply_every_stage(settings):
    pipeline = SequencePipeline(settings)
    assert pipeline.lane is settings.lane
    assert pipeline.bev is settings.bev
    assert pipeline.indexing is settings.indexing

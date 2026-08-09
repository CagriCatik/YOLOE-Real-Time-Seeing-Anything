"""Abnahmetests entlang der Anforderungen FR-1 bis FR-7.

Jeder Test nennt die Anforderung, die er prueft. Die beiden TC-Szenarien aus
FR-7.1 (EGO 2->4, CO 2->1) sind hier SYNTHETISCH nachgebaut: dort ist die
Wahrheit vorgegeben und die Richtung damit ueberpruefbar (FR-7.2), was auf
unannotiertem echtem Material nicht geht.
"""

from __future__ import annotations

import numpy as np
import pytest

from adascope.config import (
    BoundaryTrackConfig, EgoMotionConfig, EventConfig, IndexConfig,
)
from adascope.ground_truth import ExpectedEvent, GroundTruth, score_events
from adascope.lanes import SequencePipeline
from adascope.lanes.boundaries import Boundaries
from adascope.lanes.egomotion import EgoMotionDetector
from adascope.lanes.mapping import assign_lane_numbers
from adascope.lanes.tracking_ids import BoundaryTracker, crossing_direction
from adascope.synthetic import SyntheticRoad, glide, hold, path

CONFIRM = 2


@pytest.fixture
def road(settings) -> SyntheticRoad:
    return SyntheticRoad(lane=settings.lane, bev=settings.bev, lanes=4)


def pipeline_for(settings, road, **overrides) -> SequencePipeline:
    return SequencePipeline(settings, indexing=IndexConfig(lane_width=road.lane_width),
                            events=EventConfig(confirm_frames=CONFIRM), **overrides)


def drive(pipeline, road, tracks, frames=None, y_bev=430.0, ego_x=None):
    events, states = [], []
    count = frames or len(next(iter(tracks.values()), []))
    for index in range(count):
        vehicles = [road.ego(x_bev=None if ego_x is None else ego_x[index])]
        vehicles += [road.vehicle(tid, bahn[index], y_bev) for tid, bahn in tracks.items()]
        analysis = pipeline.process(index, f"f{index}", road.frame(), vehicles)
        events.extend(analysis.events)
        states.extend(analysis.states())
    return events, states


# --------------------------------------------------------------------------- #
# FR-1.2 / FR-2.3 -- Richtung                                                 #
# --------------------------------------------------------------------------- #
def test_fr12_direction_is_reported_without_any_lane_number():
    assert crossing_direction(300.0, 200.0) == "links"
    assert crossing_direction(200.0, 300.0) == "rechts"
    assert crossing_direction(200.0, 200.0) == "unbestimmt"


def test_fr23_cut_in_from_the_left_is_reported_as_rechts(settings, road):
    """Ein Fahrzeug, das von links einschert, bewegt sich nach RECHTS."""
    pipeline = pipeline_for(settings, road)
    ego_lane, left_lane = 2, 1
    trajectory = path(hold(road.lane_center(left_lane), 6),
                      glide(road.lane_center(left_lane), road.lane_center(ego_lane), 10),
                      hold(road.lane_center(ego_lane), 8))
    events, _ = drive(pipeline, road, {1: trajectory})
    cut_ins = [e for e in events if e.kind == "cut_in"]
    assert len(cut_ins) == 1
    assert cut_ins[0].direction == "rechts"


def test_fr23_cut_in_from_the_right_is_reported_as_links(settings, road):
    pipeline = pipeline_for(settings, road)
    ego_lane, right_lane = 2, 3
    trajectory = path(hold(road.lane_center(right_lane), 6),
                      glide(road.lane_center(right_lane), road.lane_center(ego_lane), 10),
                      hold(road.lane_center(ego_lane), 8))
    events, _ = drive(pipeline, road, {1: trajectory})
    cut_ins = [e for e in events if e.kind == "cut_in"]
    assert len(cut_ins) == 1 and cut_ins[0].direction == "links"


# --------------------------------------------------------------------------- #
# FR-1.4 -- Per-Frame-Zustand                                                 #
# --------------------------------------------------------------------------- #
def test_fr14_state_record_carries_all_required_fields(settings, road):
    pipeline = pipeline_for(settings, road)
    _, states = drive(pipeline, road, {1: hold(road.lane_center(1), 6)})
    assert states
    required = {"frame_id", "fahrzeug", "lateral_pos", "aktive_grenze_id", "confidence"}
    for row in states:
        assert set(row) == required


def test_fr14_ego_and_co_use_the_same_record(settings, road):
    pipeline = pipeline_for(settings, road)
    _, states = drive(pipeline, road, {1: hold(road.lane_center(1), 4)})
    vehicles = {row["fahrzeug"] for row in states}
    assert "EGO" in vehicles and "ID1" in vehicles


def test_fr14_confidence_grows_while_a_boundary_stays_visible(settings, road):
    pipeline = pipeline_for(settings, road)
    _, states = drive(pipeline, road, {1: hold(road.lane_center(1), 12)})
    of_id1 = [row["confidence"] for row in states if row["fahrzeug"] == "ID1"]
    assert of_id1[-1] > of_id1[0]


# --------------------------------------------------------------------------- #
# FR-1.4 / FR-5.1 -- Grenzen-IDs                                              #
# --------------------------------------------------------------------------- #
def test_boundary_keeps_its_identity_while_it_drifts():
    tracker = BoundaryTracker(BoundaryTrackConfig(max_shift=35))
    first = tracker.update(Boundaries.from_positions([100, 200, 300], 600))
    second = tracker.update(Boundaries.from_positions([110, 208, 312], 600))
    assert first == second


def test_a_boundary_that_disappears_briefly_keeps_its_identity():
    """Eine Strichluecke darf keine neue Identitaet erzeugen."""
    tracker = BoundaryTracker(BoundaryTrackConfig(max_missing=5))
    before = tracker.update(Boundaries.from_positions([100, 200, 300], 600))
    for _ in range(3):
        tracker.update(Boundaries.from_positions([100, 300], 600))   # mittlere fehlt
    after = tracker.update(Boundaries.from_positions([100, 202, 300], 600))
    assert after == before


def test_a_new_boundary_gets_a_new_identity():
    tracker = BoundaryTracker()
    tracker.update(Boundaries.from_positions([100, 200], 600))
    identities = tracker.update(Boundaries.from_positions([100, 200, 400], 600))
    assert len(set(identities)) == 3


def test_fr51_event_carries_direction_boundary_and_frame_span(settings, road):
    pipeline = pipeline_for(settings, road)
    trajectory = path(hold(road.lane_center(1), 6),
                      glide(road.lane_center(1), road.lane_center(2), 10),
                      hold(road.lane_center(2), 8))
    events, _ = drive(pipeline, road, {1: trajectory})
    event = next(e for e in events if e.kind == "cut_in")
    assert event.direction in ("links", "rechts")
    assert event.boundary_id is not None
    start, end = event.frames
    assert start < end                       # eine Spanne, nicht ein Frame


# --------------------------------------------------------------------------- #
# FR-3 -- Ego aus der Linienstruktur                                          #
# --------------------------------------------------------------------------- #
def shift_structure(detector, positions_over_time, lane_width=100.0):
    """Eine Grenzenfolge durch den Detektor schicken."""
    tracker = BoundaryTracker(BoundaryTrackConfig(max_shift=60))
    verdicts = []
    for positions in positions_over_time:
        boundaries = Boundaries.from_positions(positions, 600)
        identities = tracker.update(boundaries)
        verdicts.append(detector.update(boundaries, identities, lane_width))
    return verdicts


def test_fr31_parallel_shift_is_reported_as_a_lane_change():
    """Alle Grenzen wandern gleich weit -- das ist eine Translation."""
    detector = EgoMotionDetector(EgoMotionConfig(window=10, shift_fraction=0.5))
    frames = [[100 + 8 * step, 200 + 8 * step, 300 + 8 * step, 400 + 8 * step]
              for step in range(12)]
    verdicts = shift_structure(detector, frames)
    assert any(v.verdict == "wechsel" for v in verdicts)
    assert next(v for v in verdicts if v.verdict == "wechsel").direction == "links"


def test_fr32_a_curve_is_not_reported_as_a_lane_change():
    """Der Kurven-Confounder: nahe Grenzen wandern wenig, ferne viel.

    Dieselbe mittlere Verschiebung wie oben, aber ungleich verteilt. Das ist
    eine Drehung und darf keinen Wechsel ergeben.
    """
    detector = EgoMotionDetector(EgoMotionConfig(window=10, shift_fraction=0.5))
    frames = [[100 + 2 * step, 200 + 6 * step, 300 + 10 * step, 400 + 14 * step]
              for step in range(12)]
    verdicts = shift_structure(detector, frames)
    assert not any(v.verdict == "wechsel" for v in verdicts)


def test_fr33_an_unproven_change_is_marked_uncertain_not_hidden():
    """FR-3.3 verlangt eine MARKIERUNG, kein Schweigen."""
    detector = EgoMotionDetector(EgoMotionConfig(window=10, shift_fraction=0.5))
    frames = [[100 + 2 * step, 200 + 6 * step, 300 + 10 * step, 400 + 14 * step]
              for step in range(12)]
    verdicts = shift_structure(detector, frames)
    uncertain = [v for v in verdicts if v.verdict == "unsicher"]
    assert uncertain, "Drehung wurde stillschweigend verworfen statt markiert"
    assert "Kurve nicht ausgeschlossen" in uncertain[0].reason


def test_fr33_uncertainty_can_be_switched_off():
    detector = EgoMotionDetector(EgoMotionConfig(window=10, shift_fraction=0.5,
                                                 report_uncertain=False))
    frames = [[100 + 2 * step, 200 + 6 * step, 300 + 10 * step, 400 + 14 * step]
              for step in range(12)]
    assert not any(v.verdict == "unsicher" for v in shift_structure(detector, frames))


def test_ego_path_needs_enough_boundaries():
    """Mit zwei Grenzen ist Translation nicht von Drehung zu unterscheiden."""
    detector = EgoMotionDetector(EgoMotionConfig(window=6, min_boundaries=3))
    frames = [[100 + 10 * step, 200 + 10 * step] for step in range(8)]
    verdicts = shift_structure(detector, frames)
    assert all(v.verdict != "wechsel" for v in verdicts)
    assert any("durchgehend verfolgte Grenzen" in v.reason for v in verdicts)


def test_uncertain_ego_verdict_reaches_the_event_stream(settings, road):
    """Ein unsicheres Urteil wird gemeldet -- mit `certain=False` (FR-3.3).

    Der Detektor wird hier durch einen Stub ersetzt: die synthetische Fahrbahn
    steht still, ihre Grenzen verschieben sich also gar nicht, und ein echter
    Kurven-Confounder liesse sich damit nicht erzeugen. Geprueft wird die
    Uebergabe -- dass ein `unsicher` nicht unterwegs verlorengeht.
    """
    from adascope.lanes.egomotion import EgoMotion

    pipeline = pipeline_for(settings, road)

    class UncertainStub:
        cfg = EgoMotionConfig()

        def update(self, *_args):
            return EgoMotion("unsicher", "links", 0.8, 0.9, 4,
                             "Kurve nicht ausgeschlossen")

    pipeline.ego_motion = UncertainStub()
    analysis = pipeline.process(0, "f", road.frame(), [road.ego()])

    ego_events = [e for e in analysis.events if e.kind == "ego_lane_change"]
    assert len(ego_events) == 1
    assert ego_events[0].certain is False
    assert ego_events[0].direction == "links"
    assert "UNSICHER" in str(ego_events[0])


def test_a_proven_ego_change_is_not_flagged_uncertain(settings, road):
    from adascope.lanes.egomotion import EgoMotion

    pipeline = pipeline_for(settings, road)

    class CertainStub:
        cfg = EgoMotionConfig()

        def update(self, *_args):
            return EgoMotion("wechsel", "rechts", 0.9, 0.1, 4, "parallel")

    pipeline.ego_motion = CertainStub()
    analysis = pipeline.process(0, "f", road.frame(), [road.ego()])
    event = next(e for e in analysis.events if e.kind == "ego_lane_change")
    assert event.certain is True and event.direction == "rechts"


# --------------------------------------------------------------------------- #
# FR-4 -- Optionale Mapping-Schicht                                           #
# --------------------------------------------------------------------------- #
def test_fr41_absolute_numbers_when_the_structure_is_complete(settings, road):
    pipeline = pipeline_for(settings, road)
    analysis = pipeline.process(0, "f", road.frame(),
                                [road.ego(), road.vehicle(1, road.lane_center(1), 430)])
    numbering = assign_lane_numbers(analysis, expected_lanes=len(analysis.lanes_rel))
    assert numbering.complete
    assert numbering.of("EGO") is not None
    assert 1 <= numbering.of("EGO") <= numbering.total_lanes


def test_fr42_returns_none_when_the_structure_is_incomplete(settings, road):
    """Lieber keine Nummer als eine geratene -- das war der Fehlermodus."""
    pipeline = pipeline_for(settings, road)
    analysis = pipeline.process(0, "f", road.frame(), [road.ego()])
    numbering = assign_lane_numbers(analysis, expected_lanes=99)
    assert not numbering.complete
    assert numbering.of("EGO") is None
    assert "statt 99" in numbering.reason


def test_fr42_synthetic_boundaries_block_numbering(settings, road):
    """Eine virtuell rekonstruierte Grenze ist eine Annahme, keine Beobachtung."""
    from adascope.lanes.indexing import Lane

    class FakeAnalysis:
        lanes_rel = [Lane(0, 100, -1), Lane(100, 200, 0, synthetic=True)]
        occupancies: list = []

    numbering = assign_lane_numbers(FakeAnalysis())
    assert not numbering.complete and "virtuell" in numbering.reason


def test_fr43_the_core_runs_without_the_mapping_layer(settings, road):
    """Kein Rueckkanal: die Kernanalyse kennt die Mapping-Schicht nicht."""
    import adascope.lanes.pipeline as core

    assert "mapping" not in core.__dict__
    pipeline = pipeline_for(settings, road)
    analysis = pipeline.process(0, "f", road.frame(), [road.ego()])
    assert analysis.lanes_rel          # Kern liefert Ergebnisse ohne die Schicht


# --------------------------------------------------------------------------- #
# FR-7 -- Abnahme: die beiden TC-Szenarien, synthetisch                       #
# --------------------------------------------------------------------------- #
def test_fr71_tc_co_change_two_to_one_is_detected_with_direction(settings, road):
    """TC: CO wechselt von Spur 2 nach Spur 1 -- also nach LINKS.

    Synthetisch, weil die Richtung nur gegen eine bekannte Wahrheit pruefbar
    ist (FR-7.2). Auf echtem Material braucht es dafuer eine Annotation.
    """
    pipeline = pipeline_for(settings, road)
    lane2, lane1 = 1, 0                       # 0-basiert: Spur 2 und Spur 1
    trajectory = path(hold(road.lane_center(lane2), 6),
                      glide(road.lane_center(lane2), road.lane_center(lane1), 10),
                      hold(road.lane_center(lane1), 8))
    events, _ = drive(pipeline, road, {1: trajectory},
                      ego_x=[road.lane_center(lane2)] * 24)

    truth = GroundTruth((ExpectedEvent(13, "cut_out", "any", "links"),), tolerance=8)
    score = score_events(truth, [e for e in events if e.kind == "cut_out"])
    assert score.perfect, score.as_text()


def test_fr72_a_wrong_direction_is_caught_by_the_scoring(settings, road):
    """Der Nachweis, dass die Richtungspruefung wirklich prueft."""
    pipeline = pipeline_for(settings, road)
    trajectory = path(hold(road.lane_center(1), 6),
                      glide(road.lane_center(1), road.lane_center(2), 10),
                      hold(road.lane_center(2), 8))
    events, _ = drive(pipeline, road, {1: trajectory})

    correct = GroundTruth((ExpectedEvent(13, "cut_in", "any", "rechts"),), tolerance=8)
    wrong = GroundTruth((ExpectedEvent(13, "cut_in", "any", "links"),), tolerance=8)
    cut_ins = [e for e in events if e.kind == "cut_in"]
    assert score_events(correct, cut_ins).perfect
    assert not score_events(wrong, cut_ins).perfect

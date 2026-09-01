"""Bewertung gemeldeter Ereignisse gegen die Annotation.

Diese Schicht entscheidet, ob ein Lauf als bestanden gilt. Sie muss deshalb
selbst geprueft sein -- eine zu nachsichtige Zuordnung wuerde jeden Nachweis
wertlos machen.
"""

from __future__ import annotations

import pytest

from adascope.ground_truth import ExpectedEvent, GroundTruth, score_events
from adascope.lanes.events import Event


def truth(*events, tolerance: int = 5) -> GroundTruth:
    return GroundTruth(tuple(ExpectedEvent(*e) for e in events), tolerance)


def detected(*events) -> list[Event]:
    return [Event(frame, kind, track) for frame, kind, track in events]


def test_event_within_tolerance_counts_as_a_hit():
    score = score_events(truth((100, "cut_in")), detected((103, "cut_in", "ID1")))
    assert score.matched == 1 and score.perfect
    assert score.offsets == [3]


def test_event_outside_tolerance_is_a_miss_and_a_false_alarm():
    """Zu spaet ist nicht erkannt: sonst waere die Toleranz bedeutungslos."""
    score = score_events(truth((100, "cut_in"), tolerance=5),
                         detected((120, "cut_in", "ID1")))
    assert score.matched == 0
    assert len(score.missed) == 1 and len(score.spurious) == 1


def test_wrong_kind_does_not_match():
    score = score_events(truth((100, "cut_in")), detected((100, "cut_out", "ID1")))
    assert score.matched == 0 and not score.perfect


def test_track_any_accepts_every_id():
    """Track-IDs haengen am Detektor, das Ereignis nicht."""
    score = score_events(truth((100, "cut_in", "any")), detected((100, "cut_in", "ID42")))
    assert score.perfect


def test_named_track_must_agree():
    score = score_events(truth((100, "cut_in", "ID3")), detected((100, "cut_in", "ID9")))
    assert score.matched == 0


def test_a_burst_counts_as_one_hit_plus_false_alarms():
    """Entprellung soll Salven verhindern -- eine Salve darf nicht belohnt werden."""
    score = score_events(truth((100, "cut_in")),
                         detected((99, "cut_in", "ID1"), (100, "cut_in", "ID1"),
                                  (101, "cut_in", "ID1")))
    assert score.matched == 1 and len(score.spurious) == 2
    assert not score.perfect


def test_each_expectation_binds_at_most_one_detection():
    score = score_events(truth((100, "cut_in"), (104, "cut_in")),
                         detected((100, "cut_in", "ID1"), (104, "cut_in", "ID2")))
    assert score.matched == 2 and score.perfect


def test_empty_annotation_means_nothing_may_be_reported():
    """Die staerkste Aussage: hier passiert nachweislich nichts."""
    assert score_events(truth(), []).perfect
    score = score_events(truth(), detected((50, "cut_in", "ID1")))
    assert not score.perfect and len(score.spurious) == 1


def test_negative_only_annotation_reports_positive_recall_as_na():
    score = score_events(truth(), [])
    assert "Recall N/A" in score.label()
    assert "Positiv-Recall" in score.as_text()


def test_recall_and_precision_are_reported():
    score = score_events(truth((10, "cut_in"), (50, "cut_out")),
                         detected((10, "cut_in", "ID1"), (99, "cut_in", "ID2")))
    assert score.recall == 0.5
    assert score.precision == 0.5


# --------------------------------------------------------------------------- #
# Laden                                                                       #
# --------------------------------------------------------------------------- #
def test_missing_annotation_is_none_not_an_error(tmp_path):
    assert GroundTruth.load("gibt-es-nicht", tmp_path) is None


def test_loads_events_sorted_by_frame(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "tolerance: 3\nevents:\n  - {frame: 50, kind: cut_out}\n"
        "  - {frame: 10, kind: cut_in, track: ID2}\n", encoding="utf-8")
    loaded = GroundTruth.load("a", tmp_path)
    assert [e.frame for e in loaded.events] == [10, 50]
    assert loaded.events[0].track == "ID2"
    assert loaded.tolerance == 3


def test_empty_event_list_loads_as_a_statement(tmp_path):
    (tmp_path / "a.yaml").write_text("events: []\n", encoding="utf-8")
    loaded = GroundTruth.load("a", tmp_path)
    assert loaded is not None and loaded.events == ()


def test_event_without_a_frame_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("events:\n  - {kind: cut_in}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frame"):
        GroundTruth.load("a", tmp_path)


def test_unknown_key_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("erwartet: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unbekannte Schluessel"):
        GroundTruth.load("a", tmp_path)


def test_loads_perception_frames_and_acceptance(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "events: []\n"
        "perception:\n"
        "  - {frame: 8, boundaries_bev: [81, 190, 302], lane_count: 2, "
        "ego_lane_position: 1}\n"
        "acceptance: {lane_count_accuracy_min: 0.95}\n",
        encoding="utf-8")
    loaded = GroundTruth.load("a", tmp_path)
    assert loaded.perception[0].frame == 8
    assert loaded.perception[0].lane_count == 2
    assert loaded.acceptance.lane_count_accuracy_min == 0.95


def test_duplicate_perception_frame_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "perception:\n"
        "  - {frame: 8, lane_count: 2}\n"
        "  - {frame: 8, lane_count: 3}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="eindeutig"):
        GroundTruth.load("a", tmp_path)

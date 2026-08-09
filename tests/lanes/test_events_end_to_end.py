"""Cut-In und Cut-Out ueber die GANZE Kette, gegen bekannte Wahrheit.

Anders als `test_events.py`, das die State Machine isoliert mit fertigen
Beobachtungen fuettert, laeuft hier alles mit:

    gezeichnetes Bild -> Hough -> Homographie -> Spaltenhistogramm
    -> Korridore -> ego-relative Indizierung -> Footprints -> State Machine

Die Trajektorie ist vorgegeben, also ist auch das erwartete Ereignis bekannt.
Ein Fehlschlag hier heisst: irgendwo in dieser Kette geht die Zuordnung
verloren -- nicht, dass ein Schwellwert Geschmackssache ist.
"""

from __future__ import annotations

import numpy as np
import pytest

from adascope.config import EventConfig, IndexConfig, Settings
from adascope.lanes import SequencePipeline
from adascope.synthetic import SyntheticRoad, glide, hold, path

CONFIRM = 2


@pytest.fixture
def road(settings) -> SyntheticRoad:
    return SyntheticRoad(lane=settings.lane, bev=settings.bev, lanes=3)


def pipeline_for(settings, road: SyntheticRoad, **event_overrides) -> SequencePipeline:
    """Pipeline mit fester Spurbreite -- die Szene kennt sie, also vorgeben."""
    return SequencePipeline(
        settings,
        indexing=IndexConfig(lane_width=road.lane_width),
        events=EventConfig(confirm_frames=CONFIRM, **event_overrides))


def drive(pipeline: SequencePipeline, road: SyntheticRoad, tracks: dict[int, list[float]],
          y_bev: float = 430.0, mask_top: int = 0, ego_x: list[float] | None = None):
    """Fahrzeuge entlang vorgegebener BEV-x-Bahnen fahren lassen.

    `tracks` bildet Track-ID auf die laterale Bahn ab, ein Wert je Frame.
    `mask_top` schwaerzt die obersten Bildzeilen -- so faellt der Fernbereich
    der Fahrbahn aus, wie bei Strichluecke oder Kuppe.
    """
    frames = len(next(iter(tracks.values())))
    events = []
    for index in range(frames):
        image = road.frame()
        if mask_top:
            image[:mask_top] = 0
        ego = road.ego(x_bev=None if ego_x is None else ego_x[index])
        vehicles = [ego] + [road.vehicle(tid, bahn[index], y_bev)
                            for tid, bahn in tracks.items()]
        events.extend(pipeline.process(index, f"f{index}", image, vehicles).events)
    return events


def kinds(events) -> list[str]:
    return [e.kind for e in events]


# --------------------------------------------------------------------------- #
# Die Grundfaelle                                                             #
# --------------------------------------------------------------------------- #
def test_vehicle_moving_into_the_ego_lane_is_reported_once(settings, road):
    """Ein Einschervorgang -- genau ein cut_in, nicht null und nicht drei."""
    pipeline = pipeline_for(settings, road)
    trajectory = path(hold(road.lane_center(0), 6),                       # links
                      glide(road.lane_center(0), road.lane_center(1), 10),  # herueber
                      hold(road.lane_center(1), 8))                       # in Ego-Spur
    events = drive(pipeline, road, {1: trajectory})
    assert kinds(events) == ["cut_in"]
    assert events[0].track == "ID1"


def test_vehicle_leaving_the_ego_lane_is_reported_once(settings, road):
    pipeline = pipeline_for(settings, road)
    trajectory = path(hold(road.lane_center(1), 6),
                      glide(road.lane_center(1), road.lane_center(2), 10),
                      hold(road.lane_center(2), 8))
    assert kinds(drive(pipeline, road, {1: trajectory})) == ["cut_out"]


def test_vehicle_staying_in_its_lane_produces_nothing(settings, road):
    pipeline = pipeline_for(settings, road)
    assert drive(pipeline, road, {1: hold(road.lane_center(0), 25)}) == []


def test_aborted_lane_change_produces_no_cut_in(settings, road):
    """Bis an die Linie und zurueck -- der Fall, den ein Schwellwert falsch meldet."""
    pipeline = pipeline_for(settings, road)
    edge = road.boundary_x(1) - 12          # dicht an der Grenze, aber links davon
    trajectory = path(hold(road.lane_center(0), 5),
                      glide(road.lane_center(0), edge, 6), hold(edge, 5),
                      glide(edge, road.lane_center(0), 6), hold(road.lane_center(0), 5))
    events = drive(pipeline, road, {1: trajectory})
    assert "cut_in" not in kinds(events)
    assert "aborted" in kinds(events)


# --------------------------------------------------------------------------- #
# Zeitpunkt                                                                   #
# --------------------------------------------------------------------------- #
def test_cut_in_fires_close_to_the_actual_crossing(settings, road):
    """Das Ereignis darf nachlaufen, aber nicht beliebig weit.

    Die Entprellung kostet `confirm_frames`; mehr als das plus etwas Reserve
    waere ein Hinweis auf eine traege oder haengende Zustandslogik.
    """
    pipeline = pipeline_for(settings, road)
    approach, cross, settle = 6, 10, 10
    trajectory = path(hold(road.lane_center(0), approach),
                      glide(road.lane_center(0), road.lane_center(1), cross),
                      hold(road.lane_center(1), settle))
    events = drive(pipeline, road, {1: trajectory})

    # Ueberlappung >= 0.5 heisst: die Fahrzeugmitte hat die Spurgrenze passiert.
    crossing = next(i for i, x in enumerate(trajectory) if x >= road.boundary_x(1))
    assert len(events) == 1
    assert crossing <= events[0].frame <= crossing + CONFIRM + 2


# --------------------------------------------------------------------------- #
# Robustheit -- hier zahlt sich die ego-relative Indizierung aus              #
# --------------------------------------------------------------------------- #
def test_cut_in_survives_a_missing_far_field(settings, road):
    """Faellt der Fernbereich aus, rutschen positionsbasierte Indizes.

    Genau dagegen gibt es `lanes.indexing`: die ego-relative Nummer bleibt, und
    das Ereignis darf weder ausfallen noch sich verdoppeln.
    """
    pipeline = pipeline_for(settings, road)
    trajectory = path(hold(road.lane_center(0), 6),
                      glide(road.lane_center(0), road.lane_center(1), 10),
                      hold(road.lane_center(1), 8))
    events = drive(pipeline, road, {1: trajectory}, mask_top=90)
    assert kinds(events) == ["cut_in"]


def test_cut_in_survives_noise_and_stray_markings(settings):
    """Rauschen und falsche Striche -- die Ausreisser aus der Kruemmungsmessung."""
    road = SyntheticRoad(lane=settings.lane, bev=settings.bev, lanes=3,
                         noise_sigma=6.0, stray_marks=3, seed=7)
    pipeline = pipeline_for(settings, road)
    trajectory = path(hold(road.lane_center(0), 6),
                      glide(road.lane_center(0), road.lane_center(1), 10),
                      hold(road.lane_center(1), 8))
    assert kinds(drive(pipeline, road, {1: trajectory})) == ["cut_in"]


def test_two_vehicles_are_reported_separately(settings, road):
    """Ein Einschervorgang und ein Ausschervorgang gleichzeitig."""
    pipeline = pipeline_for(settings, road)
    frames = 24
    einscherer = path(hold(road.lane_center(0), 6),
                      glide(road.lane_center(0), road.lane_center(1), 10),
                      hold(road.lane_center(1), 8))
    ausscherer = path(hold(road.lane_center(1), 6),
                      glide(road.lane_center(1), road.lane_center(2), 10),
                      hold(road.lane_center(2), 8))
    events = drive(pipeline, road, {1: einscherer, 2: ausscherer},
                   y_bev=430.0)
    assert sorted(kinds(events)) == ["cut_in", "cut_out"]
    assert {e.track for e in events} == {"ID1", "ID2"}


# --------------------------------------------------------------------------- #
# Ungueltige Messungen                                                        #
# --------------------------------------------------------------------------- #
def test_far_field_artefact_never_becomes_an_event(settings, road):
    """Ein zu breiter Footprint ist ein Projektionsartefakt, keine Messung.

    Er wird vor der State Machine aussortiert -- sonst glaettet sie ein
    falsches Ereignis glatt.
    """
    pipeline = pipeline_for(settings, road)
    frames, events = 20, []
    for index in range(frames):
        # Footprint deutlich breiter als eine Spur: physikalisch unmoeglich.
        breit = road.vehicle(1, road.lane_center(1), 430.0,
                             width_bev=road.lane_width * 1.4)
        events.extend(pipeline.process(index, f"f{index}", road.frame(),
                                       [road.ego(), breit]).events)
    assert events == []


def test_invalid_samples_are_marked_not_silently_dropped(settings, road):
    """Verworfen heisst nicht unsichtbar: die Ansicht muss es zeigen koennen."""
    pipeline = pipeline_for(settings, road)
    breit = road.vehicle(1, road.lane_center(1), 430.0, width_bev=road.lane_width * 1.4)
    analysis = pipeline.process(0, "f0", road.frame(), [road.ego(), breit])
    assert [o.valid for o in analysis.occupancies] == [False]
    assert analysis.worst_state == "invalid"

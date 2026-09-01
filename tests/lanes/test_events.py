"""Temporale Ereignisableitung.

Was diese Tests festschreiben, ist der Unterschied zwischen einem Schwellwert
und einem Ereignis: abgebrochene Spurwechsel, Flackern und der eigene
Spurwechsel duerfen KEIN cut_in erzeugen.
"""

from __future__ import annotations

from adascope.config import EventConfig
from adascope.lanes.events import CutInTracker


def feed(tracker: CutInTracker, samples, start: int = 0,
         ego_in_lane: float = 1.0) -> list:
    """Mehrere Frames einspeisen und alle Ereignisse einsammeln.

    `ego_in_lane` < 1.0 bedeutet: das Ego beruehrt seine eigene Spurgrenze --
    der Beleg, ohne den kein eigener Spurwechsel anerkannt wird.
    """
    events = []
    for offset, observation in enumerate(samples):
        events.extend(tracker.update(start + offset, observation, ego_in_lane))
    return events


def steady(observation: dict, frames: int) -> list[dict]:
    return [dict(observation) for _ in range(frames)]


CFG = EventConfig(confirm_frames=2, max_missing=2)


# --------------------------------------------------------------------------- #
# Der Normalfall                                                              #
# --------------------------------------------------------------------------- #
def test_cut_in_requires_passing_through_encroaching():
    tracker = CutInTracker(CFG)
    events = feed(tracker, [*steady({"A": (1, 0.0)}, 3),      # aussen
                            *steady({"A": (0, 0.3)}, 3),      # anschmiegend
                            *steady({"A": (0, 0.8)}, 3)])     # drin
    assert [e.kind for e in events] == ["cut_in"]
    assert events[0].track == "A"


def test_cut_out_after_leaving_the_ego_lane():
    tracker = CutInTracker(CFG)
    events = feed(tracker, [*steady({"A": (0, 0.9)}, 3),
                            *steady({"A": (1, 0.0)}, 3)])
    assert [e.kind for e in events] == ["cut_out"]


# --------------------------------------------------------------------------- #
# Die drei Faelle, die ein Schwellwert nicht unterscheidet                    #
# --------------------------------------------------------------------------- #
def test_aborted_lane_change_emits_no_cut_in():
    """Fahrzeug driftet zur Linie und kehrt zurueck -- INSIDE nie erreicht."""
    tracker = CutInTracker(CFG)
    events = feed(tracker, [*steady({"A": (1, 0.0)}, 3),
                            *steady({"A": (0, 0.3)}, 3),
                            *steady({"A": (1, 0.0)}, 3)])
    assert [e.kind for e in events] == ["aborted"]


def test_flicker_below_confirm_frames_emits_nothing():
    """Messrauschen um die Schwelle darf keine Ereignissalve ausloesen."""
    tracker = CutInTracker(EventConfig(confirm_frames=4))
    events = feed(tracker, [{"A": (0, 0.8)}, {"A": (1, 0.0)},
                            {"A": (0, 0.8)}, {"A": (1, 0.0)},
                            {"A": (0, 0.8)}, {"A": (1, 0.0)}])
    assert events == []


def test_own_lane_change_suppresses_the_vehicle_events():
    """Wechselt Ego selbst, verschieben sich ALLE relativen Nummern zugleich.

    Pro Fahrzeug betrachtet sieht das aus wie ein Einscheren -- die Szenenebene
    erkennt die gleichsinnige Verschiebung und unterdrueckt die Fahrzeug-
    ereignisse.
    """
    tracker = CutInTracker(EventConfig(confirm_frames=2, ego_shift_min_tracks=2))
    feed(tracker, steady({"A": (1, 0.0), "B": (2, 0.0)}, 3))
    # Das Ego steht dabei quer auf der Grenze -- ohne diesen Beleg waere die
    # gleichsinnige Verschiebung ununterscheidbar von zwei echten Wechseln.
    events = feed(tracker, steady({"A": (0, 0.9), "B": (1, 0.0)}, 4), start=3,
                  ego_in_lane=0.6)

    kinds = [e.kind for e in events]
    assert "ego_lane_change" in kinds
    assert "cut_in" not in kinds


def test_single_vehicle_shift_is_not_an_own_lane_change():
    # Verschiebt sich nur eines, ist es dessen Bewegung, nicht die eigene.
    tracker = CutInTracker(EventConfig(confirm_frames=2, ego_shift_min_tracks=2))
    feed(tracker, steady({"A": (1, 0.0), "B": (2, 0.0)}, 3))
    events = feed(tracker, steady({"A": (0, 0.9), "B": (2, 0.0)}, 4), start=3)
    assert "ego_lane_change" not in [e.kind for e in events]


# --------------------------------------------------------------------------- #
# Track-Verwaltung                                                            #
# --------------------------------------------------------------------------- #
def test_missing_track_is_dropped_after_max_missing():
    tracker = CutInTracker(EventConfig(confirm_frames=1, max_missing=2))
    feed(tracker, steady({"A": (0, 0.9)}, 2))
    assert tracker.state_of("A") == "inside"
    feed(tracker, [{}, {}, {}], start=2)
    assert tracker.state_of("A") is None


def test_reappearing_vehicle_starts_outside_and_emits_nothing():
    """Nach einer Luecke darf kein Ereignis aus dem Nichts entstehen.

    Ohne durchlaufenes `encroaching` gibt es kein cut_in -- ein wieder
    auftauchendes Fahrzeug mitten in der Ego-Spur ist keine Messung eines
    Einscherens.
    """
    tracker = CutInTracker(EventConfig(confirm_frames=1, max_missing=1))
    feed(tracker, steady({"A": (0, 0.9)}, 2))
    feed(tracker, [{}, {}, {}], start=2)
    events = feed(tracker, steady({"A": (0, 0.9)}, 3), start=5)
    assert [e.kind for e in events] == []


def test_confirm_frames_zero_reacts_immediately():
    tracker = CutInTracker(EventConfig(confirm_frames=1))
    events = feed(tracker, [{"A": (1, 0.0)}, {"A": (0, 0.3)}, {"A": (0, 0.9)}])
    assert [e.kind for e in events] == ["cut_in"]


def test_two_vehicles_changing_alike_are_not_an_own_lane_change():
    """Der Fehlalarm, den der Ego-Beleg abstellt.

    Zwei Fahrzeuge wechseln gleichzeitig in dieselbe Richtung. Fuer die
    Szenenebene sieht das aus wie eine eigene Bewegung -- und die Unterdrueckung
    haette genau ihre beiden echten Ereignisse geloescht. Das Ego liegt dabei
    ruhig in seiner Spur, also gilt der Verdacht als widerlegt.
    """
    tracker = CutInTracker(EventConfig(confirm_frames=2, ego_shift_min_tracks=2))
    feed(tracker, steady({"A": (-1, 0.0), "B": (0, 0.9)}, 3), ego_in_lane=1.0)
    events = feed(tracker, [*steady({"A": (-1, 0.3), "B": (1, 0.3)}, 2),
                            *steady({"A": (0, 0.9), "B": (1, 0.0)}, 3)],
                  start=3, ego_in_lane=1.0)

    kinds = [e.kind for e in events]
    assert "ego_lane_change" not in kinds
    assert "cut_in" in kinds and "cut_out" in kinds


def test_brisk_lane_change_is_not_swallowed():
    """Der Defekt, den die synthetische Szene aufgedeckt hat.

    Ein zuegiger Wechsel durchquert das Band zwischen den Schwellen in einem
    Frame. Frueher verlangte der Automat einen BESTAETIGTEN Zwischenzustand --
    das cut_in fiel stumm aus, und auf echtem Material war das nicht von
    'da war eben nichts' zu unterscheiden.
    """
    tracker = CutInTracker(EventConfig(confirm_frames=3))
    events = feed(tracker, [*steady({"A": (1, 0.0)}, 4),
                            {"A": (1, 0.3)},              # nur EIN Frame dazwischen
                            *steady({"A": (0, 0.9)}, 4)])
    assert [e.kind for e in events] == ["cut_in"]


def test_vehicle_appearing_inside_still_emits_nothing():
    """Die Absicherung, die dabei erhalten bleiben musste.

    Ohne beobachtete Anfahrt gibt es kein cut_in -- ein nach einer Trackluecke
    mitten in der Ego-Spur auftauchendes Fahrzeug ist keine Messung eines
    Einschervorgangs.
    """
    tracker = CutInTracker(EventConfig(confirm_frames=2))
    assert feed(tracker, steady({"A": (0, 0.95)}, 6)) == []


def test_slow_cut_out_is_a_cut_out_not_an_abort():
    """Der dritte Defekt derselben Familie.

    Ein langsames Ausscheren geht inside -> encroaching -> outside. Frueher
    verlangte der cut_out, dass der VORHERIGE Zustand `inside` war -- beim
    letzten Uebergang ist er aber `encroaching`, und das Fahrzeug wurde als
    'Abbruch' gemeldet, obwohl es nachweislich in der Ego-Spur war.
    """
    tracker = CutInTracker(EventConfig(confirm_frames=2))
    events = feed(tracker, [*steady({"A": (0, 0.9)}, 3),     # drin
                            *steady({"A": (0, 0.3)}, 3),     # driftet raus
                            *steady({"A": (1, 0.0)}, 3)])    # draussen
    assert [e.kind for e in events] == ["cut_out"]


def test_abort_still_reported_when_the_vehicle_was_never_inside():
    tracker = CutInTracker(EventConfig(confirm_frames=2))
    events = feed(tracker, [*steady({"A": (1, 0.0)}, 3),
                            *steady({"A": (1, 0.3)}, 3),
                            *steady({"A": (1, 0.0)}, 3)])
    assert [e.kind for e in events] == ["aborted"]


# --------------------------------------------------------------------------- #
# Falschalarme, vom Anwender im Debugvideo bestaetigt                          #
# --------------------------------------------------------------------------- #
def test_a_vehicle_already_inside_at_startup_is_not_a_cut_in():
    """`adjusting_speed_scenario_8`: cut_in fuer ID2 in Frame 1-5.

    Jeder Track startete als `outside`. Ein Fahrzeug, das beim Aufnahmestart
    bereits in der Ego-Spur steht, erzeugte dadurch sofort einen Uebergang
    outside -> inside. Die erste Beobachtung ist aber kein Uebergang -- es gibt
    kein Davor.
    """
    from adascope.config import EventConfig
    from adascope.lanes.events import CutInTracker

    fsm = CutInTracker(EventConfig(confirm_frames=2))
    events = []
    for f in range(10):
        events += fsm.update(f, {"ID2": (0, 1.0)}, lateral={"ID2": 300.0})
    assert [e.kind for e in events] == []


def test_a_vehicle_inside_from_the_start_can_still_cut_out():
    """Die Gegenprobe: wer von Anfang an drin ist, darf ausscheren."""
    from adascope.config import EventConfig
    from adascope.lanes.events import CutInTracker

    fsm = CutInTracker(EventConfig(confirm_frames=2))
    events = []
    for f in range(6):
        events += fsm.update(f, {"ID1": (0, 1.0)}, lateral={"ID1": 300.0})
    for f in range(6, 10):                       # hinausdriften
        events += fsm.update(f, {"ID1": (0, 0.3)}, lateral={"ID1": 340.0})
    for f in range(10, 16):
        events += fsm.update(f, {"ID1": (1, 0.0)}, lateral={"ID1": 380.0})
    assert [e.kind for e in events] == ["cut_out"]
    assert events[0].direction == "rechts"


def test_an_entry_without_an_observed_approach_does_not_arm_a_cut_out():
    """`adjusting_speed_scenario_5`: zweimal cut_out fuer ID4 in 15 Frames.

    Ein Eintritt OHNE beobachtete Anfahrt ist ein Sprung -- schon fuer ein
    cut_in zu unsicher. Dann darf er auch kein cut_out scharfmachen. Die
    Asymmetrie war der Defekt.
    """
    from adascope.config import EventConfig
    from adascope.lanes.events import CutInTracker

    fsm = CutInTracker(EventConfig(confirm_frames=2))
    events = []
    for f in range(4):                           # weit draussen beginnen
        events += fsm.update(f, {"ID4": (1, 0.0)}, lateral={"ID4": 400.0})
    for f in range(4, 8):                        # aus dem Nichts drin
        events += fsm.update(f, {"ID4": (0, 1.0)}, lateral={"ID4": 300.0})
    for f in range(8, 12):                       # und wieder raus
        events += fsm.update(f, {"ID4": (1, 0.0)}, lateral={"ID4": 400.0})
    assert [e.kind for e in events] == []


def test_an_event_without_a_determinable_direction_is_refused():
    """FR-1.2 verlangt LINKS/RECHTS -- ein Spurwechsel ohne Querweg ist keiner."""
    from adascope.config import EventConfig
    from adascope.lanes.events import CutInTracker

    cfg = EventConfig(confirm_frames=2, require_direction=True)
    fsm = CutInTracker(cfg)
    events = []
    for f in range(4):
        events += fsm.update(f, {"ID1": (1, 0.0)}, lateral={"ID1": 400.0})
    for f in range(4, 8):                        # Anfahrt beobachtet ...
        events += fsm.update(f, {"ID1": (0, 0.3)}, lateral={"ID1": 400.0})
    for f in range(8, 12):                       # ... aber ohne Querbewegung
        events += fsm.update(f, {"ID1": (0, 1.0)}, lateral={"ID1": 400.0})
    assert [e.kind for e in events] == []

    ohne_gate = CutInTracker(EventConfig(confirm_frames=2, require_direction=False))
    events = []
    for f in range(4):
        events += ohne_gate.update(f, {"ID1": (1, 0.0)}, lateral={"ID1": 400.0})
    for f in range(4, 8):
        events += ohne_gate.update(f, {"ID1": (0, 0.3)}, lateral={"ID1": 400.0})
    for f in range(8, 12):
        events += ohne_gate.update(f, {"ID1": (0, 1.0)}, lateral={"ID1": 400.0})
    assert [e.kind for e in events] == ["cut_in"]


def test_the_gate_does_not_fire_when_no_lateral_data_exists():
    """"Keine Daten" ist etwas anderes als "Daten zeigen keine Bewegung".

    Die State Machine ist bewusst ohne `lateral` isoliert testbar; ein
    fehlender Eingang darf kein Ereignis verschlucken.
    """
    from adascope.config import EventConfig
    from adascope.lanes.events import CutInTracker

    fsm = CutInTracker(EventConfig(confirm_frames=2, require_direction=True))
    events = []
    for f in range(4):
        events += fsm.update(f, {"ID1": (1, 0.0)})
    for f in range(4, 8):
        events += fsm.update(f, {"ID1": (0, 0.3)})
    for f in range(8, 12):
        events += fsm.update(f, {"ID1": (0, 1.0)})
    assert [e.kind for e in events] == ["cut_in"]

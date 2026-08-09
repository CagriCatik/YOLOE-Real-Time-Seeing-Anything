"""Die zeitliche Stabilisierung der Spurgrenzen.

Der Kern ist eine Wette: eine Grenze, die einen Frame lang fehlt, war meist
trotzdem da. Diese Wette darf nur unter Bedingungen gelten -- die Tests hier
halten beide Richtungen fest, das Halten UND das Nicht-Halten.
"""

from __future__ import annotations

import numpy as np
import pytest

from adascope.config import BoundaryTrackConfig
from adascope.lanes.boundaries import Boundaries
from adascope.lanes.stabilize import BoundaryStabilizer

PEAK_MIN = 100.0


def cfg(**kw) -> BoundaryTrackConfig:
    # `stabilize` steht ausdruecklich drin: der Standard ist AUS (gemessen
    # schaedlich auf dem Projektmaterial), die Tests pruefen den Mechanismus.
    base = dict(stabilize=True, confirm_frames=1, search_radius=22.0,
                max_missing=12)
    return BoundaryTrackConfig(**{**base, **kw})


def hist(*peaks: tuple[float, float], width: int = 400) -> np.ndarray:
    """Ein Histogramm mit Rechteckpeaks: (position, hoehe)."""
    out = np.zeros(width, np.float32)
    for x, height in peaks:
        out[max(0, int(x) - 5):int(x) + 6] = height
    return out


def positions(b: Boundaries) -> list[float]:
    return [round(x, 1) for x in b.at(b.y_reference)]


def feed(stab: BoundaryStabilizer, frames: list[list[float]],
         histogram: np.ndarray | None = None) -> list[list[float]]:
    """Eine Folge von Messungen durchlaufen lassen."""
    return [positions(stab.update(Boundaries.from_positions(f, 0.0), histogram))
            for f in frames]


# --------------------------------------------------------------------- #
# Halten: der Ausfall-Fall                                              #
# --------------------------------------------------------------------- #
def test_a_single_frame_dropout_is_filled_in():
    """Das Kernversprechen: eine Luecke von einem Frame verschwindet."""
    stab = BoundaryStabilizer(cfg(), PEAK_MIN)
    h = hist((100, 500), (200, 500), (300, 500))
    out = feed(stab, [[100, 200, 300], [100, 300], [100, 200, 300]], h)
    assert [len(f) for f in out] == [3, 3, 3]
    assert out[1] == [100.0, 200.0, 300.0]


def test_the_held_boundary_is_marked_as_held_not_measured():
    """Ergaenzt ist nicht gemessen -- der Support muss das zeigen."""
    stab = BoundaryStabilizer(cfg(hold_support=0.5), PEAK_MIN)
    h = hist((100, 500), (200, 500), (300, 500))
    feed(stab, [[100, 200, 300]], h)
    held = stab.update(Boundaries.from_positions([100, 300], 0.0), h)
    assert held.support[1] == pytest.approx(0.5)
    assert held.support[0] == pytest.approx(1.0)
    assert stab.report.held == 1


def test_holding_ends_after_max_missing():
    """Unbegrenztes Halten waere Erfindung, kein Gedaechtnis."""
    stab = BoundaryStabilizer(cfg(max_missing=3), PEAK_MIN)
    h = hist((100, 500), (200, 500), (300, 500))
    out = feed(stab, [[100, 200, 300]] + [[100, 300]] * 6, h)
    assert [len(f) for f in out] == [3, 3, 3, 3, 2, 2, 2]


def test_a_boundary_with_no_signal_left_is_not_held():
    """Der sachliche Test: unter der Grenze ist nichts mehr.

    Ohne ihn ueberlebt die alte Nachbargrenze den eigenen Spurwechsel und
    erzeugt einen Korridor, den es nicht gibt.
    """
    stab = BoundaryStabilizer(cfg(hold_min_ratio=0.45), PEAK_MIN)
    voll = hist((100, 500), (200, 500), (300, 500))
    feed(stab, [[100, 200, 300]], voll)
    # Bei 200 ist das Signal weg -- nicht nur der Peak, das Restsignal auch.
    leer = hist((100, 500), (200, 10), (300, 500))
    out = stab.update(Boundaries.from_positions([100, 300], 0.0), leer)
    assert positions(out) == [100.0, 300.0]
    assert stab.report.dropped == 1


def test_residual_signal_above_the_floor_still_holds():
    """Die Gegenprobe: Restsignal ueber der Schwelle haelt."""
    stab = BoundaryStabilizer(cfg(hold_min_ratio=0.45), PEAK_MIN)
    feed(stab, [[100, 200, 300]], hist((100, 500), (200, 500), (300, 500)))
    schwach = hist((100, 500), (200, 60), (300, 500))     # 60 > 0.45 * 100
    out = stab.update(Boundaries.from_positions([100, 300], 0.0), schwach)
    assert positions(out) == [100.0, 200.0, 300.0]


def test_without_a_histogram_holding_is_purely_temporal():
    stab = BoundaryStabilizer(cfg(), PEAK_MIN)
    out = feed(stab, [[100, 200, 300], [100, 300]], None)
    assert out[1] == [100.0, 200.0, 300.0]


# --------------------------------------------------------------------- #
# Bestaetigen: der Scheingrenzen-Fall                                   #
# --------------------------------------------------------------------- #
def test_a_one_frame_spurious_peak_is_suppressed():
    stab = BoundaryStabilizer(cfg(confirm_frames=2), PEAK_MIN)
    h = hist((100, 500), (200, 500), (300, 500))
    out = feed(stab, [[100, 200], [100, 200], [100, 200, 350], [100, 200]], h)
    assert [len(f) for f in out] == [2, 2, 2, 2]


def test_a_persistent_new_boundary_is_reported_after_confirmation():
    """Unterdruecken darf nicht heissen: nie melden."""
    stab = BoundaryStabilizer(cfg(confirm_frames=2), PEAK_MIN)
    h = hist((100, 500), (200, 500), (350, 500))
    out = feed(stab, [[100, 200]] * 2 + [[100, 200, 350]] * 3, h)
    assert [len(f) for f in out] == [2, 2, 2, 3, 3]


def test_a_cold_start_reports_immediately_despite_confirm_frames():
    """Beim Kaltstart ist jede Grenze neu -- Bestaetigung waere nur Verzoegerung.

    Diese Luecke hatte die Suite aufgedeckt: der erste Frame lieferte gar
    keine Grenzen, weil alle noch auf Bestaetigung warteten. Sichtbar wurde es
    nicht hier, sondern in acht Renderingtests, deren Fixture leer blieb.
    """
    stab = BoundaryStabilizer(cfg(confirm_frames=3), PEAK_MIN)
    out = stab.update(Boundaries.from_positions([100, 200, 300], 0.0), None)
    assert positions(out) == [100.0, 200.0, 300.0]


def test_after_a_total_loss_reacquisition_is_immediate_again():
    """Ein Abriss ist ein Kaltstart -- sonst kostet jede Wiederaufnahme Frames."""
    stab = BoundaryStabilizer(cfg(confirm_frames=3, max_missing=0), PEAK_MIN)
    feed(stab, [[100, 200]], hist((100, 500), (200, 500)))
    assert positions(stab.update(Boundaries(), np.zeros(400, np.float32))) == []
    out = stab.update(Boundaries.from_positions([150, 250], 0.0), None)
    assert positions(out) == [150.0, 250.0]


def test_confirm_frames_of_one_reports_immediately():
    stab = BoundaryStabilizer(cfg(confirm_frames=1), PEAK_MIN)
    out = feed(stab, [[100, 200], [100, 200, 350]], hist((350, 500)))
    assert [len(f) for f in out] == [2, 3]


# --------------------------------------------------------------------- #
# Zuordnung                                                             #
# --------------------------------------------------------------------- #
def test_boundaries_follow_movement_within_the_search_radius():
    """Wandern ist erlaubt -- sonst wuerde jede Kurvenfahrt neu erfinden."""
    stab = BoundaryStabilizer(cfg(search_radius=22.0), PEAK_MIN)
    out = feed(stab, [[100, 200], [110, 210], [120, 220]], hist((120, 500)))
    assert out == [[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]]
    assert stab.report.held == 0 and stab.report.pending == 0


def test_a_jump_beyond_the_radius_is_a_new_boundary_not_a_move():
    """Sonst zoege eine Grenze quer ueber die Fahrbahn statt zu wechseln."""
    stab = BoundaryStabilizer(cfg(search_radius=22.0, max_missing=0), PEAK_MIN)
    out = feed(stab, [[100]], hist((100, 500)))
    assert out == [[100.0]]
    out2 = stab.update(Boundaries.from_positions([180], 0.0), hist((180, 500)))
    assert positions(out2) == [180.0]
    assert stab.report.dropped == 1


def test_matching_is_greedy_by_distance_not_by_order():
    """Ein Ausreisser am Rand darf die uebrige Zuordnung nicht verschieben."""
    stab = BoundaryStabilizer(cfg(search_radius=22.0), PEAK_MIN)
    feed(stab, [[100, 200, 300]], hist((100, 500), (200, 500), (300, 500)))
    out = stab.update(Boundaries.from_positions([205, 302, 500], 0.0),
                      hist((205, 500), (302, 500), (500, 500)))
    # 205 gehoert zu 200, nicht zu 100 -- und 100 wird gehalten, nicht ersetzt.
    assert 205.0 in positions(out) and 302.0 in positions(out)


# --------------------------------------------------------------------- #
# Abschaltbarkeit und Zuruecksetzen                                     #
# --------------------------------------------------------------------- #
def test_disabled_passes_everything_through_unchanged():
    """Der Vergleichsfall muss exakt das alte Verhalten sein."""
    stab = BoundaryStabilizer(BoundaryTrackConfig(stabilize=False), PEAK_MIN)
    out = feed(stab, [[100, 200, 300], [100, 300], [100, 200, 300]])
    assert [len(f) for f in out] == [3, 2, 3]


def test_reset_forgets_the_previous_geometry():
    stab = BoundaryStabilizer(cfg(), PEAK_MIN)
    feed(stab, [[100, 200, 300]], hist((100, 500), (200, 500), (300, 500)))
    stab.reset()
    out = stab.update(Boundaries.from_positions([100, 300], 0.0),
                      hist((100, 500), (300, 500)))
    assert positions(out) == [100.0, 300.0]


def test_curves_survive_the_stabilizer():
    """Grad-2-Grenzen aus der Fenstersuche duerfen nicht flachgedrueckt werden."""
    stab = BoundaryStabilizer(cfg(), PEAK_MIN)
    curved = Boundaries(((0.001, -0.2, 100.0), (0.001, -0.2, 200.0)), 0.0,
                        (1.0, 1.0), "windows")
    out = stab.update(curved, None)
    assert out.is_curved and out.method == "windows"
    assert out.curves[0] == curved.curves[0]


# --------------------------------------------------------------------- #
# Konfiguration                                                         #
# --------------------------------------------------------------------- #
def test_search_radius_above_max_shift_is_rejected():
    """Sonst ordnet die Stabilisierung zu, was die Kennungsvergabe ablehnt."""
    with pytest.raises(ValueError, match="search_radius"):
        BoundaryTrackConfig(search_radius=50.0, max_shift=35.0)


@pytest.mark.parametrize("kw", [{"confirm_frames": 0}, {"hold_min_ratio": 1.5},
                                {"search_radius": -1.0}, {"hold_support": 2.0}])
def test_invalid_configuration_is_rejected(kw):
    with pytest.raises(ValueError):
        BoundaryTrackConfig(**kw)

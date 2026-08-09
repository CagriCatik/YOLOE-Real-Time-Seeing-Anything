"""Ego-relative Spurnummerierung.

Der Kern des Fehlermodus, den dieses Modul aufloest: positionsbasierte Indizes
rutschen, sobald eine Spurgrenze ausfaellt. Die Tests konstruieren Korridore
direkt, ohne Bild und ohne Homographie -- so ist geprueft, was das Modul
verspricht, nicht was ein bestimmter Frame gerade liefert.
"""

from __future__ import annotations

import pytest

from adascope.config import IndexConfig
from adascope.lanes.indexing import (
    build_lane_index, classify_corridor, ego_overlap, estimate_lane_width, locate,
    split_corridors,
)

LANE = 76.0


def corridors(*widths: float, start: float = 0.0):
    """Aneinandergrenzende Korridore aus Breiten bauen."""
    out, x = [], start
    for width in widths:
        out.append((x, x + width))
        x += width
    return out


# --------------------------------------------------------------------------- #
# Spurbreitenschaetzung                                                       #
# --------------------------------------------------------------------------- #
def test_estimates_width_from_the_smallest_recurring_corridor():
    # Standstreifen (110) darf die Schaetzung nicht nach oben ziehen.
    assert estimate_lane_width(corridors(110, 76, 77, 75)) == pytest.approx(76, abs=1)


def test_estimate_is_dragged_down_by_spurious_narrow_corridors():
    # Genau der gemessene Fehlermodus: mit peak_min_distance=25 ueberlebten
    # Kleinstkorridore, und die Schaetzung griff sie ab (30 statt 76 px).
    # Der Test haelt fest, DASS die Heuristik so versagt -- die Abhilfe ist
    # peak_min_distance, nicht eine Sonderbehandlung hier.
    assert estimate_lane_width(corridors(110, 32, 42, 76, 77)) < 45


# --------------------------------------------------------------------------- #
# Korridor-Plausibilitaet                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("width,kind,k", [
    (76, "lane", 1),
    (152, "merged", 2),
    (228, "merged", 3),
    (110, "non_lane", 0),      # Standstreifen: kein ganzzahliges Vielfaches
    (30, "non_lane", 0),       # zu schmal
])
def test_classifies_corridor_width(width, kind, k):
    assert classify_corridor(width, LANE, IndexConfig()) == (kind, k)


def test_merged_corridor_is_split_into_equal_lanes():
    lanes, synthetic, width = split_corridors(corridors(152), IndexConfig(lane_width=LANE))
    assert len(lanes) == 2
    assert synthetic == [True, True]
    assert lanes[0] == (0.0, 76.0) and lanes[1] == (76.0, 152.0)


def test_hard_shoulder_drops_out_automatically():
    # Kein Sonderfall im Code: die Breite des Standstreifens ist schlicht kein
    # ganzzahliges Vielfaches der Spurbreite.
    lanes, _, _ = split_corridors(corridors(110, 76, 76, 76), IndexConfig(lane_width=LANE))
    assert len(lanes) == 3


def test_too_loose_a_tolerance_turns_the_hard_shoulder_into_a_lane():
    """Warum `multiple_tolerance` nicht grosszuegig sein darf.

    Der Standstreifen misst 110 px = 1.45 Spurbreiten. Streng gilt er als
    Nicht-Spur; bei 0.45 Toleranz rundet 1.45 auf k=1 und er wird als volle
    Fahrspur mitgezaehlt -- alle ego-relativen Nummern verschieben sich.
    """
    assert classify_corridor(110, LANE, IndexConfig(multiple_tolerance=0.01)) == ("non_lane", 0)
    assert classify_corridor(110, LANE, IndexConfig(multiple_tolerance=0.45)) == ("lane", 1)


def test_tolerance_governs_when_a_merge_is_recognised():
    # 145 px = 1.9 Spurbreiten: bei enger Toleranz kein sauberes Vielfaches,
    # bei weiter Toleranz zwei verschmolzene Spuren.
    assert classify_corridor(145, LANE, IndexConfig(multiple_tolerance=0.05)) == ("non_lane", 0)
    assert classify_corridor(145, LANE, IndexConfig(multiple_tolerance=0.15)) == ("merged", 2)


# --------------------------------------------------------------------------- #
# Ego-relative Nummerierung                                                   #
# --------------------------------------------------------------------------- #
def test_numbers_lanes_relative_to_the_ego_footprint():
    lanes, _ = build_lane_index(corridors(76, 76, 76), 80, 140, IndexConfig(lane_width=LANE))
    assert [L.rel for L in lanes] == [-1, 0, 1]
    assert [L.label for L in lanes] == ["links_1", "ego", "rechts_1"]


def test_dropout_outside_the_ego_span_cancels_out():
    """Der eigentliche Zweck des Moduls.

    Faellt links aussen eine Grenze weg, verschieben sich alle positionsbasier-
    ten Indizes um eins. Die ego-relative Nummer des Zielfahrzeugs bleibt gleich,
    weil sich Ego und Ziel gemeinsam verschieben.
    """
    cfg = IndexConfig(lane_width=LANE)
    target = (170, 220)                    # Fahrzeug in der Spur rechts vom Ego

    full = build_lane_index(corridors(76, 76, 76, 76), 80, 140, cfg)[0]
    # Die beiden linken Korridore verschmelzen zu einem doppelt breiten.
    merged = build_lane_index(corridors(152, 76, 76), 80, 140, cfg)[0]

    assert locate(full, *target)[0] == locate(merged, *target)[0] == 1


def test_ego_overlap_is_one_when_fully_inside():
    lanes, _ = build_lane_index(corridors(76, 76, 76), 80, 140, IndexConfig(lane_width=LANE))
    assert ego_overlap(lanes, 80, 140) == pytest.approx(1.0)


def test_ego_overlap_drops_when_crossing_the_boundary():
    # Das Lane-Departure-Signal: der Ego-Footprint ragt ueber die Spurgrenze.
    lanes, _ = build_lane_index(corridors(76, 76, 76), 80, 140, IndexConfig(lane_width=LANE))
    assert ego_overlap(lanes, 130, 170) == pytest.approx(0.55, abs=0.02)


def test_locate_returns_none_outside_every_lane():
    lanes, _ = build_lane_index(corridors(76, 76, 76), 80, 140, IndexConfig(lane_width=LANE))
    assert locate(lanes, 900, 950) == (None, 0.0)


def test_ego_outside_every_plausible_lane_is_an_error():
    # Lieber laut scheitern als eine Spur raten: der Aufrufer entscheidet, ob
    # der Frame verworfen oder die Geometrie gehalten wird.
    with pytest.raises(ValueError, match="Ego-Footprint"):
        build_lane_index(corridors(76, 76), 900, 950, IndexConfig(lane_width=LANE))


def test_configured_width_overrides_the_estimate():
    _, width = build_lane_index(corridors(50, 50, 50), 10, 40, IndexConfig(lane_width=50))
    assert width == 50


# --------------------------------------------------------------------------- #
# Spurbreitenschaetzung ueber ganzzahlige Vielfache                           #
# --------------------------------------------------------------------------- #
def estimate(*widths: float) -> float:
    from adascope.lanes.indexing import estimate_lane_width_by_multiples
    return estimate_lane_width_by_multiples(corridors(*widths), 0.18, 4)


def test_a_clipped_edge_corridor_no_longer_ruins_the_estimate():
    """Der Fehler, der die zweitmeisten Frames gekostet hat.

    Gemessen auf `acc_plus_6`: Korridorbreiten [103, 80, 77, 80, 59]. Die
    Minimum-Heuristik nahm 59 -- den am Bildrand angeschnittenen Korridor.
    Damit war 80/59 = 1.36 kein Vielfaches, vier von fuenf Korridoren fielen
    als "keine Spur" heraus, das Ego lag in keiner mehr. 57 von 100 Frames
    gingen so verloren.
    """
    assert estimate(103, 80, 77, 80, 59) == pytest.approx(79, abs=2)


def test_merged_corridors_do_not_win_over_the_true_width():
    """Die Sorge, die gegen den Median sprach -- hier ausgeschlossen.

    Bei [76, 152, 152] erklaert 76 alle drei Korridore (1x, 2x, 2x), 152 nur
    zwei. Der Median waere 152 und damit falsch.
    """
    assert estimate(76, 152, 152) == pytest.approx(76, abs=2)


def test_hard_shoulder_does_not_shift_the_estimate():
    assert estimate(110, 75, 76, 78) == pytest.approx(76, abs=2)


def test_spurious_narrow_peaks_no_longer_dominate():
    """Auch der alte Fehlpeak-Fall wird besser -- 42 statt 32.

    Er bleibt falsch (richtig waeren ~76), aber die Abhilfe dagegen ist
    `bev.peak_min_distance`, nicht die Schaetzung. Der Test haelt fest, dass
    die Schaetzung ihn nicht mehr VERSTAERKT.
    """
    assert estimate(110, 32, 42, 76, 77) > 40


def test_two_lane_road_estimates_its_own_width():
    assert estimate(168, 168) == pytest.approx(168, abs=2)


def test_single_corridor_falls_back_to_its_own_width():
    assert estimate(76) == pytest.approx(76, abs=1)


# --------------------------------------------------------------------------- #
# Der Ego-Korridor ist vor dem Verwerfen geschuetzt                            #
# --------------------------------------------------------------------------- #
def test_the_corridor_the_ego_drives_in_is_never_discarded():
    """Der haeufigste Ausfall des ganzen Systems.

    Die Breitenpruefung kennt nur Geometrie. Verwirft sie den Korridor, in dem
    das Ego steht, entsteht ein Loch, und `build_lane_index` meldet danach
    "Ego-Footprint liegt in keiner plausiblen Spur" -- ueber einen Korridor, in
    dem das Ego nachweislich faehrt.

    Gemessen: auf `acc_plus_7` 17 von 32 Ausfaellen, auf `acc_plus_3` 20 von
    27, auf `adjusting_speed_scenario_5` 17 von 20.
    """
    from adascope.config import IndexConfig
    from adascope.lanes.indexing import build_lane_index

    cfg = IndexConfig(lane_width=70.0)
    # Mittlerer Korridor ist mit 30 px zu schmal fuer eine Spur -- und genau
    # dort steht das Ego.
    corridors = [(0.0, 70.0), (70.0, 100.0), (100.0, 170.0)]
    lanes, _ = build_lane_index(corridors, 80.0, 92.0, cfg)

    assert lanes, "der Frame darf nicht verworfen werden"
    ego = [L for L in lanes if L.rel == 0]
    assert len(ego) == 1
    assert (ego[0].x_lo, ego[0].x_hi) == (70.0, 100.0)
    assert [L.rel for L in lanes] == [-1, 0, 1]


def test_a_non_lane_without_the_ego_is_still_discarded():
    """Der Schutz gilt genau einem Korridor, nicht der Breitenpruefung."""
    from adascope.config import IndexConfig
    from adascope.lanes.indexing import split_corridors

    cfg = IndexConfig(lane_width=70.0)
    corridors = [(0.0, 70.0), (70.0, 100.0), (100.0, 170.0)]

    ohne, _, _ = split_corridors(corridors, cfg)
    assert (70.0, 100.0) not in ohne

    mit, _, _ = split_corridors(corridors, cfg, keep_x=85.0)
    assert (70.0, 100.0) in mit


def test_the_protected_corridor_is_not_marked_synthetic():
    """`synthetic` heisst rekonstruiert -- ein gemessener Korridor ist es nicht.

    FR-4.2 lehnt absolute Spurnummern ab, sobald eine Spur rekonstruiert wurde.
    Den Ego-Korridor faelschlich so zu markieren wuerde die Mapping-Schicht
    ohne Grund verstummen lassen.
    """
    from adascope.config import IndexConfig
    from adascope.lanes.indexing import split_corridors

    cfg = IndexConfig(lane_width=70.0)
    lanes, synthetic, _ = split_corridors(
        [(0.0, 70.0), (70.0, 100.0), (100.0, 170.0)], cfg, keep_x=85.0)
    assert synthetic[lanes.index((70.0, 100.0))] is False


def test_a_single_lane_road_is_a_valid_scene():
    """Zwei Grenzen, ein Korridor -- eine vollstaendige Szene, kein Ausfall.

    Auf `lane_departure_1_lane` (489 Frames) haben 312 Frames genau zwei
    Grenzen. Die frueher geforderten zwei Korridore verwarfen sie alle und
    liessen von der Aufnahme 36 % uebrig.

    Das Bittere daran: die Fahrzeugdetektion arbeitete korrekt. Ohne sie
    erzeugten Fahrzeugpixel Scheingrenzen, die auf drei kamen -- die Aufnahme
    lief also nur, WEIL die Spurmaske verschmutzt war.
    """
    from adascope.config import IndexConfig
    from adascope.lanes.indexing import build_lane_index

    cfg = IndexConfig(lane_width=70.0)
    lanes, width = build_lane_index([(0.0, 70.0)], 20.0, 50.0, cfg)

    assert [L.rel for L in lanes] == [0], "das Ego faehrt in der einzigen Spur"
    assert width == 70.0


def test_the_minimum_corridor_count_is_configurable():
    """Wer die alte, strengere Forderung will, kann sie einstellen."""
    from adascope.config import IndexConfig

    assert IndexConfig().min_corridors == 1
    assert IndexConfig(min_corridors=2).min_corridors == 2

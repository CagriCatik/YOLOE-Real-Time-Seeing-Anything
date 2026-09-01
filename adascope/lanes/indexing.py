"""
Ego-relative Spurnummerierung mit Korridor-Plausibilitaet.

Problem der Vorstufe
--------------------
`bev_lane_occupancy` nummeriert Korridore absolut von links (L0..L3). Faellt
eine Spurgrenze aus, verschmelzen zwei Korridore und alle rechts davon
liegenden Indizes rutschen -> Falschalarm ohne Szenenaenderung.

Zwei Massnahmen
---------------
1. RELATIVE NUMMERIERUNG: Ego = 0, links negativ, rechts positiv. Ein Ausfall
   ausserhalb der Spanne zwischen Ego und Ziel verschiebt beide gleich und
   kuerzt sich weg. Das deckt den Grossteil der Ausfaelle ab.

2. KORRIDOR-PLAUSIBILITAET: Ein Ausfall ZWISCHEN Ego und Ziel kuerzt sich
   nicht weg. Er ist aber an der Breite erkennbar: ein verschmolzener
   Korridor misst ein ganzzahliges Vielfaches der Spurbreite. Solche
   Korridore werden durch virtuelle Grenzen wieder aufgeteilt.

Zusaetzlich faellt damit der Standstreifen automatisch heraus: seine Breite
ist KEIN ganzzahliges Vielfaches der Spurbreite und er wird nicht als Spur
gezaehlt.

Die Spurbreite ist die zentrale Kalibriergroesse. Sie kann von aussen
gesetzt werden (-> calibrator.py) oder wird aus dem Bild geschaetzt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..config import IndexConfig

Kind = Literal["lane", "merged", "non_lane"]

Corridor = tuple[float, float]


@dataclass
class Lane:
    """Eine erkannte Fahrspur mit ego-relativer Nummer."""
    x_lo: float
    x_hi: float
    rel: int                     # 0 = Ego, -1 = eine links, +1 = eine rechts
    synthetic: bool = False      # aus einem verschmolzenen Korridor rekonstruiert

    @property
    def width(self) -> float:
        return self.x_hi - self.x_lo

    @property
    def label(self) -> str:
        if self.rel == 0:
            return "ego"
        side = "links" if self.rel < 0 else "rechts"
        return f"{side}_{abs(self.rel)}"


# --------------------------------------------------------------------------- #
def estimate_lane_width(corridors: list[Corridor], factor: float = 1.15) -> float:
    """Schaetzt die Spurbreite als kleinste wiederkehrende Korridorbreite.

    Auf einer mehrspurigen Fahrbahn tritt die einfache Spurbreite mindestens
    zweimal auf. Der Median wuerde bei verschmolzenen Korridoren wegdriften,
    das Minimum allein waere anfaellig fuer einen einzelnen Fehlpeak - deshalb
    das Mittel aller Breiten, die nahe am Minimum liegen.
    """
    widths = sorted(b - a for a, b in corridors)
    base = widths[0]
    cluster = [w for w in widths if w <= base * factor]
    return float(np.mean(cluster))


def estimate_lane_width_by_multiples(corridors: list[Corridor], tolerance: float,
                                     max_merge: int) -> float:
    """Die Breite, unter der die MEISTEN Korridore ganzzahlige Vielfache sind.

    Warum nicht die kleinste wiederkehrende Breite
    ----------------------------------------------
    Die kleinste Breite ist genau dann falsch, wenn der schmalste Korridor kein
    echter ist: ein am Bildrand angeschnittener Bereich oder ein Fehlpeak.
    Gemessen auf `acc_plus_6`, Korridorbreiten [103, 80, 77, 80, 59]:

        Minimum-Heuristik  ->  59 px. Damit ist 80/59 = 1.36 kein Vielfaches,
                               77/59 = 1.31 auch nicht -- VIER von fuenf
                               Korridoren fallen als "keine Spur" heraus, das
                               Ego liegt in keiner mehr, der Frame ist verloren.
                               57 von 100 Frames endeten so.
        dieses Verfahren   ->  79 px. 80, 77, 80 sind dann einfache Spuren,
                               103 und 59 fallen als Nicht-Spur heraus.

    Wie
    ---
    Jede beobachtete Breite ist ein Kandidat. Bewertet wird, wie viele Korridore
    unter ihr ein ganzzahliges Vielfaches ergeben -- ein Wert, der zu allem
    passt, ist die Spurbreite. Zum Schluss wird der Gewinner auf den Mittelwert
    der einfachen Spuren nachgezogen, damit das Ergebnis nicht an einem
    zufaellig etwas schmalen Kandidaten haengt.

    Verschmolzene Korridore stoeren nicht: bei [76, 152, 152] gewinnt 76 mit
    drei Treffern (1x, 2x, 2x) gegen 152 mit zwei -- das Minimum waere hier
    zufaellig auch richtig, der Median aber nicht.
    """
    widths = sorted(b - a for a, b in corridors)
    if not widths:
        return 0.0

    def multiples_under(candidate: float) -> list[float]:
        """Die Breiten, die unter `candidate` ein Vielfaches sind."""
        matching = []
        for width in widths:
            ratio = width / candidate
            k = round(ratio)
            if 1 <= k <= max_merge and abs(ratio - k) <= tolerance:
                matching.append(width)
        return matching

    # Bester Kandidat: die meisten Treffer, bei Gleichstand der kleinere -- ein
    # verschmolzener Korridor erklaert nie mehr Korridore als die echte Breite.
    best = max(widths, key=lambda w: (len(multiples_under(w)), -w))
    singles = [w for w in multiples_under(best) if abs(w / best - 1) <= tolerance]
    return float(np.mean(singles)) if singles else float(best)


def classify_corridor(width: float, lane_width: float,
                      cfg: IndexConfig) -> tuple[Kind, int]:
    """Ordnet einer Korridorbreite ein ganzzahliges Vielfaches zu."""
    ratio = width / lane_width
    k = int(round(ratio))
    if k < 1 or k > cfg.max_merge:
        return "non_lane", 0
    if abs(ratio - k) > cfg.multiple_tolerance:
        return "non_lane", 0
    return ("lane" if k == 1 else "merged"), k


def split_corridors(corridors: list[Corridor], cfg: IndexConfig,
                    keep_x: float | None = None
                    ) -> tuple[list[Corridor], list[bool], float]:
    """Zerlegt verschmolzene Korridore und behaelt NUR die eigene Fahrbahn.

    `keep_x` ist die Ego-Position. Sie leistet zweierlei: sie schuetzt den
    Korridor, in dem das Ego steht, vor dem Verwerfen -- und sie waehlt aus,
    welche Fahrbahn ueberhaupt gemeint ist.

    Rueckgabe: (Spurkorridore, synthetic-Flags, verwendete Spurbreite)

    Warum eine Nicht-Spur-Flaeche TRENNT statt nur zu fehlen
    -------------------------------------------------------
    Frueher wurden zu breite oder zu schmale Korridore uebersprungen und die
    Spuren links UND rechts davon weiterverwendet. Damit lief die Auswertung
    ueber den Mitteltrennstreifen hinweg bis in die Gegenfahrbahn.

    Im Debugvideo sichtbar als `left_solid s2` und `left_solid s5` weit links
    jenseits der Trennung, bei `Korridore: 5, Spuren: 4` auf einer dreispurigen
    eigenen Fahrbahn. Diese Linien zappeln (sie sind fern und flach), ihr
    Zappeln ging in `estimate_lane_width` und in die Korridorzaehlung ein --
    und weil `outer_solid_pair` die AEUSSERSTE Linie als Stuetzpunkt nimmt,
    wurde die Homographie ueber die Gegenfahrbahn aufgespannt.

    Fuer die Aufgabenstellung ist die Gegenfahrbahn ohne Belang: es geht um
    Ein- und Ausscheren in der eigenen Fahrtrichtung.

    Warum der Ego-Korridor geschuetzt wird
    --------------------------------------
    Die Breitenpruefung kennt nur Geometrie. Faellt der Korridor, in dem das
    Ego steht, durch sie hindurch, entsteht ein LOCH in der Spurliste, und
    `build_lane_index` meldet danach "Ego-Footprint liegt in keiner plausiblen
    Spur" -- ueber einen Korridor, in dem das Ego nachweislich faehrt.

    Gemessen war das der haeufigste Ausfall ueberhaupt: auf `acc_plus_7` 17 von
    32 Ausfaellen, auf `acc_plus_3` 20 von 27, auf `adjusting_speed_scenario_5`
    17 von 20 -- jeweils Ego INNERHALB der Spanne, aber in einer Luecke.

    Dass das Ego dort faehrt, ist die staerkste verfuegbare Evidenz dafuer,
    dass die Flaeche befahrbar ist -- staerker als eine Breitenschwelle. Faehrt
    es tatsaechlich auf dem Standstreifen, ist es ebenfalls richtig, das zu
    zeigen statt den ganzen Frame zu verwerfen.
    """
    lane_width = cfg.lane_width or estimate_lane_width_by_multiples(
        corridors, cfg.multiple_tolerance, cfg.max_merge)

    # Zusammenhaengende Laeufe von Spurkorridoren. Jede Nicht-Spur-Flaeche
    # TRENNT -- sie ist die Grenze zwischen zwei Fahrbahnen.
    runs: list[tuple[list[Corridor], list[bool]]] = []
    aktuell: tuple[list[Corridor], list[bool]] = ([], [])

    def abschliessen() -> None:
        if aktuell[0]:
            runs.append((aktuell[0][:], aktuell[1][:]))
        aktuell[0].clear()
        aktuell[1].clear()

    for lo, hi in corridors:
        kind, k = classify_corridor(hi - lo, lane_width, cfg)
        if kind == "non_lane":
            if keep_x is not None and lo <= keep_x <= hi:
                # Ego faehrt hier -- also Spur, trotz Breitenpruefung.
                aktuell[0].append((lo, hi))
                aktuell[1].append(False)
            else:
                abschliessen()              # Mitteltrennung, Standstreifen, Rand
            continue
        if kind == "lane":
            aktuell[0].append((lo, hi))
            aktuell[1].append(False)
            continue
        step = (hi - lo) / k                # verschmolzen -> virtuell teilen
        for i in range(k):
            aktuell[0].append((lo + i * step, lo + (i + 1) * step))
            aktuell[1].append(True)
    abschliessen()

    if not runs:
        return [], [], lane_width
    if keep_x is None:
        # Ohne Ego-Bezug laesst sich die eigene Fahrbahn nicht bestimmen --
        # dann bleibt es beim alten Verhalten, alle Laeufe zusammen.
        return ([c for r, _ in runs for c in r],
                [s for _, f in runs for s in f], lane_width)

    for lauf, flags in runs:
        if lauf[0][0] <= keep_x <= lauf[-1][1]:
            return lauf, flags, lane_width
    # Ego ausserhalb jedes Laufs: den naechstgelegenen nehmen, statt alles
    # zusammenzuwerfen -- die Fahrbahn, auf der es faehrt, ist die naechste.
    lauf, flags = min(runs, key=lambda rf: min(abs(rf[0][0][0] - keep_x),
                                               abs(rf[0][-1][1] - keep_x)))
    return lauf, flags, lane_width


# --------------------------------------------------------------------------- #
def _containing_index(lanes: list[Corridor], x_lo: float, x_hi: float) -> int:
    """Index der Spur mit der groessten Ueberlappung; -1 wenn keine."""
    best, best_ov = -1, 0.0
    for i, (lo, hi) in enumerate(lanes):
        ov = max(0.0, min(x_hi, hi) - max(x_lo, lo))
        if ov > best_ov:
            best, best_ov = i, ov
    return best


def build_lane_index(corridors: list[Corridor], ego_x_lo: float, ego_x_hi: float,
                     cfg: IndexConfig = IndexConfig()) -> tuple[list[Lane], float]:
    """Baut die ego-relative Spurliste.

    ego_x_lo/hi: laterale Grenzen des Ego-Footprints in BEV.
    """
    lanes, synthetic, lane_width = split_corridors(
        corridors, cfg, keep_x=(ego_x_lo + ego_x_hi) / 2)
    if not lanes:
        return [], lane_width

    ego_i = _containing_index(lanes, ego_x_lo, ego_x_hi)
    if ego_i < 0:
        raise ValueError("Ego-Footprint liegt in keiner plausiblen Spur")

    return ([Lane(lo, hi, rel=i - ego_i, synthetic=syn)
             for i, ((lo, hi), syn) in enumerate(zip(lanes, synthetic))],
            lane_width)


def locate(lanes: list[Lane], x_lo: float, x_hi: float) -> tuple[int | None, float]:
    """Ego-relative Spurnummer und Ueberlappungsanteil eines Footprints."""
    width = max(x_hi - x_lo, 1e-6)
    best, best_ratio = None, 0.0
    for L in lanes:
        ov = max(0.0, min(x_hi, L.x_hi) - max(x_lo, L.x_lo))
        if ov / width > best_ratio:
            best, best_ratio = L.rel, ov / width
    return best, best_ratio


def ego_overlap(lanes: list[Lane], x_lo: float, x_hi: float) -> float:
    """Anteil der Fahrzeugbreite, der in der Ego-Spur liegt."""
    width = max(x_hi - x_lo, 1e-6)
    for L in lanes:
        if L.rel == 0:
            return max(0.0, min(x_hi, L.x_hi) - max(x_lo, L.x_lo)) / width
    return 0.0

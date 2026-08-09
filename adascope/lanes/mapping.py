"""Optionale Zuordnung absoluter Spurnummern (FR-4).

Strikt getrennt vom Kern
------------------------
Dieses Modul liest `FrameAnalysis` und gibt Nummern zurueck. Es schreibt
nichts, nichts importiert es aus dem Kern, und der Kern kennt es nicht
(FR-4.3). Faellt es aus, laeuft die Grenzuebergangs-Erkennung unveraendert
weiter -- genau das ist der Zweck der Trennung.

Warum es ueberhaupt existiert
-----------------------------
Das Acceptance-Kriterium verlangt woertlich, die "aktuelle Spur" auszugeben.
Der Kern erfuellt die ABSICHT (Wechsel + Richtung, ohne Spurnummern), diese
Schicht den WORTLAUT (FR-4.4).

Warum es optional bleiben MUSS
------------------------------
Eine absolute Nummer ist nur vergebbar, wenn die vollstaendige Spurstruktur im
Bild ist. Ist sie es nicht, wird geraten -- und genau dieses Raten war der
Fehlermodus, gegen den `lanes.indexing` gebaut wurde: bei einem Grenzausfall
rutschten alle Nummern und erzeugten einen Falschalarm ohne Szenenaenderung.
Deshalb liefert diese Schicht `None` statt einer plausiblen Zahl (FR-4.2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaneNumbering:
    """Absolute Spurnummern eines Frames, oder die Begruendung ihres Fehlens."""

    # Fahrzeug -> Spurnummer ab 1 von links. Leer, wenn nicht vergebbar.
    numbers: dict[str, int]
    total_lanes: int | None = None
    complete: bool = False
    reason: str = ""

    def of(self, vehicle: str) -> int | None:
        """Spurnummer oder None -- nie eine geratene Zahl."""
        return self.numbers.get(vehicle)

    def __bool__(self) -> bool:
        return self.complete


def structure_is_complete(analysis, expected_lanes: int | None) -> tuple[bool, str]:
    """Ist die Spurstruktur vollstaendig genug fuer absolute Nummern?

    Drei Bedingungen, jede fuer sich hinreichend zum Ablehnen:

    1. Es gibt ueberhaupt eine ego-relative Spurliste.
    2. Keine Spur wurde aus einem verschmolzenen Korridor rekonstruiert --
       eine virtuelle Grenze ist eine Annahme, keine Beobachtung.
    3. Die erwartete Spurzahl stimmt, falls sie vorgegeben ist.
    """
    lanes = analysis.lanes_rel
    if not lanes:
        return False, "keine verwertbare Spurliste"
    if any(lane.synthetic for lane in lanes):
        return False, "enthaelt virtuell rekonstruierte Grenzen"
    if expected_lanes is not None and len(lanes) != expected_lanes:
        return False, f"{len(lanes)} statt {expected_lanes} Spuren sichtbar"
    return True, ""


def assign_lane_numbers(analysis, expected_lanes: int | None = None) -> LaneNumbering:
    """Absolute Spurnummern ab 1 von links (FR-4.1).

    `expected_lanes` ist die bekannte Spurzahl des Fahrzeugprojekts. Ohne
    Angabe wird die im Frame sichtbare Zahl genommen -- dann ist die Nummer
    zwar konsistent innerhalb des Frames, aber nicht ueber Abschnitte hinweg
    vergleichbar. Fuer die Abnahme sollte sie gesetzt sein.
    """
    complete, reason = structure_is_complete(analysis, expected_lanes)
    if not complete:
        return LaneNumbering({}, None, False, reason)

    lanes = analysis.lanes_rel
    # Die ego-relative Nummer 0 liegt an Position `offset` von links; daraus
    # wird jede relative Nummer zu einer absoluten.
    offset = next(index for index, lane in enumerate(lanes) if lane.rel == 0)
    numbers = {"EGO": offset + 1}
    for occupancy in analysis.occupancies:
        if occupancy.valid and occupancy.rel is not None:
            number = offset + occupancy.rel + 1
            if 1 <= number <= len(lanes):
                numbers[occupancy.track] = number
    return LaneNumbering(numbers, len(lanes), True, "")

"""Persistente Kennungen fuer Spurgrenzen ueber Frames hinweg (FR-1.4, FR-5.1).

Warum das hier steht -- und warum es die ego-relative Nummerierung NICHT ersetzt
-------------------------------------------------------------------------------
`lanes.indexing` hat positionsbasierte Spurindizes durch ego-relative ersetzt,
weil absolute Nummern bei einem Grenzausfall rutschen. Diese Entscheidung
bleibt: die Belegungsauswertung laeuft weiter ego-relativ.

Eine Grenzen-ID beantwortet eine andere Frage. Nicht "die wievielte Spur ist
das", sondern "ist das DIESELBE Linie wie im letzten Frame". Damit laesst sich
ein Ereignis benennen (`ueber Grenze B3 nach links`), ohne dass die Zaehlung
von links wieder eingefuehrt wird.

Wie
---
Ungarisch waere sauber, aber fuer 3-6 Grenzen je Frame ist ein gieriger
Nachbarabgleich ausreichend und nachvollziehbar: jede neue Grenze nimmt die
naechstgelegene bekannte innerhalb von `max_shift`. Nicht zugeordnete bekannte
Grenzen ueberleben `max_missing` Frames -- eine Strichluecke soll keine neue
Identitaet erzeugen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import BoundaryTrackConfig
from .boundaries import Boundaries


@dataclass
class _Known:
    identity: int
    x: float                 # zuletzt gesehene Position an der Referenzhoehe
    missing: int = 0
    seen: int = 1            # wie oft insgesamt gesehen -- speist die Konfidenz


@dataclass
class BoundaryTracker:
    """Vergibt stabile Kennungen an Spurgrenzen.

    Der Zustand ist bewusst klein: Position und Fehlzaehler je Kennung. Mehr
    braucht es nicht, und mehr waere ein zweiter Ort, an dem Geometrie lebt.
    """

    cfg: BoundaryTrackConfig = field(default_factory=BoundaryTrackConfig)
    _known: list[_Known] = field(default_factory=list)
    _next_id: int = 0

    def update(self, boundaries: Boundaries) -> list[int]:
        """Kennungen fuer die Grenzen dieses Frames, in deren Reihenfolge."""
        if not boundaries:
            self._age_out(matched=set())
            return []

        positions = boundaries.at(boundaries.y_reference)
        assigned: list[int | None] = [None] * len(positions)
        matched: set[int] = set()

        # Gierig nach Naehe: das beste Paar zuerst, damit ein knapper Treffer
        # keinen besseren verdraengt.
        pairs = sorted(
            ((abs(position - known.x), index, slot)
             for index, position in enumerate(positions)
             for slot, known in enumerate(self._known)
             if abs(position - known.x) <= self.cfg.max_shift),
            key=lambda item: item[0])
        used_slots: set[int] = set()
        for _, index, slot in pairs:
            if assigned[index] is not None or slot in used_slots:
                continue
            known = self._known[slot]
            assigned[index] = known.identity
            known.x, known.missing, known.seen = positions[index], 0, known.seen + 1
            used_slots.add(slot)
            matched.add(known.identity)

        for index, identity in enumerate(assigned):
            if identity is None:
                assigned[index] = self._next_id
                self._known.append(_Known(self._next_id, positions[index]))
                matched.add(self._next_id)
                self._next_id += 1

        self._age_out(matched)
        return [identity for identity in assigned]

    def confidence(self, identity: int) -> float:
        """Wie belastbar diese Grenze ist: 0 unbekannt, 1 durchgehend gesehen.

        Bewusst aus der Beobachtungsgeschichte und nicht aus der Peakhoehe --
        eine Grenze, die seit zwanzig Frames stabil an derselben Stelle liegt,
        ist verlaesslicher als ein starker Peak in einem Einzelframe.
        """
        for known in self._known:
            if known.identity == identity:
                return min(known.seen / self.cfg.confident_after, 1.0)
        return 0.0

    def reset(self) -> None:
        self._known.clear()
        self._next_id = 0

    def _age_out(self, matched: set[int]) -> None:
        for known in self._known:
            if known.identity not in matched:
                known.missing += 1
        self._known = [k for k in self._known if k.missing <= self.cfg.max_missing]


def crossing_direction(before: float, after: float) -> str:
    """Bewegungsrichtung aus zwei lateralen Positionen (FR-1.2, FR-2.3).

    In BEV-Koordinaten waechst x nach rechts. Das Vorzeichen der Differenz ist
    damit die Richtung -- unabhaengig von Spurnummern, wie FR-1.3 verlangt.
    """
    if after < before:
        return "links"
    if after > before:
        return "rechts"
    return "unbestimmt"

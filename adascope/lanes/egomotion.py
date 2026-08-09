"""Eigener Spurwechsel aus der Bewegung der Linienstruktur (FR-3.1 bis FR-3.3).

Das Prinzip
-----------
Wechselt das Ego die Spur, verschiebt sich die **gesamte** Linienstruktur im
BEV seitlich -- alle Grenzen gleich weit, in Summe etwa eine Spurbreite. Das
ist eine Translation.

Faehrt das Ego durch eine Kurve, verschiebt sich die Struktur ebenfalls, aber
**ungleich**: die nahen Grenzen wenig, die fernen viel. Das ist eine Drehung.

    Translation (Wechsel)          Drehung (Kurve)
    ||||  ->  ||||                 ||||  ->  \\\\\\\\
    alle gleich weit               fern mehr als nah

Beide zu trennen ist laut Anforderung die schwaechste Annahme des Systems.
Deshalb misst dieses Modul die **Streuung** der Einzelverschiebungen mit und
meldet ohne bestandene Trennung `unsicher` statt eines Ereignisses (FR-3.3).

Warum nicht ueber die Fremdfahrzeuge
------------------------------------
Der Vorgaenger schloss aus der gleichsinnigen Verschiebung der ego-relativen
Nummern anderer Fahrzeuge. Gemessen auf dem Projektmaterial: in 1090 von 2117
Frames ist ausser dem Ego kein Fahrzeug im Bild -- die Szenenebene konnte dort
nie greifen. Die Linienstruktur ist immer da, wenn die Homographie steht.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..config import EgoMotionConfig
from .boundaries import Boundaries
from .tracking_ids import crossing_direction

Verdict = Literal["kein_wechsel", "wechsel", "unsicher"]


@dataclass(frozen=True)
class EgoMotion:
    """Das Urteil eines Frames ueber die eigene Querbewegung."""

    verdict: Verdict = "kein_wechsel"
    direction: str = "unbestimmt"
    # Kumulierte parallele Verschiebung im Fenster, in Spurbreiten.
    shift_lanes: float = 0.0
    # Streuung der Einzelverschiebungen, relativ zur mittleren. Klein =
    # Translation, gross = Drehung. Der Kurven-Confounder-Test aus FR-3.2.
    spread: float = 0.0
    boundaries_used: int = 0
    reason: str = ""

    @property
    def is_change(self) -> bool:
        return self.verdict == "wechsel"


@dataclass
class EgoMotionDetector:
    """Beobachtet die Linienstruktur und urteilt ueber eigene Spurwechsel."""

    cfg: EgoMotionConfig = field(default_factory=EgoMotionConfig)
    _history: deque = field(default_factory=lambda: deque(maxlen=64))
    _refractory: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._refractory = 0

    def update(self, boundaries: Boundaries, identities: list[int],
               lane_width: float) -> EgoMotion:
        """Einen Frame verarbeiten.

        `identities` verbindet die Grenzen mit denen der Vorframes -- ohne sie
        waere nicht entscheidbar, ob eine Grenze sich bewegt hat oder eine
        andere an ihre Stelle getreten ist.
        """
        if self._refractory > 0:
            self._refractory -= 1

        if not boundaries or not identities or lane_width <= 0:
            self._history.clear()          # Luecke -> Fenster neu beginnen
            return EgoMotion(reason="keine verwertbare Linienstruktur")

        positions = boundaries.at(boundaries.y_reference)
        self._history.append(dict(zip(identities, positions)))
        if len(self._history) < self.cfg.window:
            return EgoMotion(reason=f"Fenster erst {len(self._history)}/{self.cfg.window}")

        first, last = self._history[-self.cfg.window], self._history[-1]
        shared = sorted(set(first) & set(last))
        if len(shared) < self.cfg.min_boundaries:
            return EgoMotion(boundaries_used=len(shared),
                             reason=f"nur {len(shared)} durchgehend verfolgte Grenzen")

        shifts = np.array([last[key] - first[key] for key in shared], float)
        mean_shift = float(shifts.mean())
        shift_lanes = abs(mean_shift) / lane_width
        # Streuung relativ zur mittleren Verschiebung. Bei reiner Translation
        # sind alle Werte gleich -> nahe 0. In einer Kurve laufen sie
        # auseinander -> gross.
        spread = float(shifts.std() / max(abs(mean_shift), 1e-6))

        if shift_lanes < self.cfg.shift_fraction:
            return EgoMotion(shift_lanes=shift_lanes, spread=spread,
                             boundaries_used=len(shared),
                             reason="Verschiebung unter der Schwelle")
        if self._refractory > 0:
            return EgoMotion(shift_lanes=shift_lanes, spread=spread,
                             boundaries_used=len(shared),
                             reason="Sperrzeit nach vorigem Wechsel")

        # Die Struktur hat sich weit genug bewegt. Jetzt die entscheidende
        # Frage: parallel (Wechsel) oder gedreht (Kurve)?
        direction = crossing_direction(0.0, -mean_shift)   # Ego bewegt sich GEGEN die Linien
        if spread > self.cfg.max_spread:
            if not self.cfg.report_uncertain:
                return EgoMotion(shift_lanes=shift_lanes, spread=spread,
                                 boundaries_used=len(shared),
                                 reason="Drehung statt Translation")
            return EgoMotion("unsicher", direction, shift_lanes, spread, len(shared),
                             f"Verschiebung uneinheitlich (Streuung {spread:.2f} > "
                             f"{self.cfg.max_spread}) -- Kurve nicht ausgeschlossen")

        self._refractory = self.cfg.refractory
        self._history.clear()
        return EgoMotion("wechsel", direction, shift_lanes, spread, len(shared),
                         "parallele Verschiebung ueber Schwelle")

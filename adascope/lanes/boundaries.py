"""Spurgrenzen als Kurven statt als feste x-Positionen.

Warum das auch geraden Fahrbahnen nuetzt
----------------------------------------
Bisher war eine Spurgrenze eine Zahl: die x-Position eines Histogramm-Peaks,
gueltig fuer die ganze BEV-Hoehe. Ein Fahrzeug bei y=200 wurde also gegen
Grenzen verglichen, die ueber alle y gemittelt waren.

Hier ist eine Grenze ein Polynom `x = p(y)` und wird **dort ausgewertet, wo das
Fahrzeug steht**. Fuer gerade Spuren ist das Polynom vom Grad 0 und das
Ergebnis identisch zu vorher; in einer Kurve ist es der Unterschied zwischen
richtig und falsch.

Damit kennt der nachgelagerte Code nur eine Darstellung, egal welches Verfahren
die Grenzen gefunden hat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Boundaries:
    """Ein Polynom je Spurgrenze, aufsteigend nach x an der Referenzhoehe."""

    # Koeffizienten je Grenze in np.polyval-Reihenfolge (hoechster Grad zuerst).
    curves: tuple[tuple[float, ...], ...] = ()
    # Hoehe, an der die Grenzen sortiert wurden -- ueblicherweise der Nahbereich.
    y_reference: float = 0.0
    # Anteil getroffener Fenster je Grenze; 1.0 beim Histogramm.
    support: tuple[float, ...] = ()
    method: str = "histogram"

    def __len__(self) -> int:
        return len(self.curves)

    def __bool__(self) -> bool:
        return bool(self.curves)

    @property
    def is_curved(self) -> bool:
        return any(len(c) > 1 for c in self.curves)

    def at(self, y: float) -> list[float]:
        """x-Positionen aller Grenzen bei dieser Hoehe, aufsteigend."""
        return sorted(float(np.polyval(c, y)) for c in self.curves)

    def corridors_at(self, y: float) -> list[tuple[float, float]]:
        """Die Lucken zwischen benachbarten Grenzen bei dieser Hoehe."""
        xs = self.at(y)
        return [(a, b) for a, b in zip(xs, xs[1:])]

    def polyline(self, index: int, y_from: float, y_to: float,
                 steps: int = 24) -> np.ndarray:
        """Stuetzpunkte einer Grenze zum Zeichnen: (steps, 2) als (x, y)."""
        ys = np.linspace(y_from, y_to, steps)
        return np.stack([np.polyval(self.curves[index], ys), ys], axis=1)

    @classmethod
    def from_positions(cls, positions, y_reference: float = 0.0) -> "Boundaries":
        """Feste x-Positionen als Polynome vom Grad 0.

        Der Weg, auf dem das Spaltenhistogramm in dieselbe Darstellung kommt --
        ohne Sonderfall im nachgelagerten Code.
        """
        ordered = sorted(float(x) for x in positions)
        return cls(tuple((x,) for x in ordered), y_reference,
                   tuple(1.0 for _ in ordered), "histogram")


def fit_curve(ys: np.ndarray, xs: np.ndarray, degree: int,
              min_points_per_degree: int) -> tuple[float, ...]:
    """Polynom x = p(y) mit automatisch gesenktem Grad.

    Ein Polynom 2. Grades durch zwei Punkte ist kein Fit, sondern Rauschen mit
    drei Koeffizienten. Deshalb wird der Grad an die Punktzahl angepasst, statt
    einen instabilen Fit zurueckzugeben.
    """
    usable = min(degree, max(1, len(ys) // min_points_per_degree))
    usable = min(usable, len(ys) - 1)
    if usable < 1:
        return (float(np.mean(xs)),)
    return tuple(float(c) for c in np.polyfit(ys, xs, usable))

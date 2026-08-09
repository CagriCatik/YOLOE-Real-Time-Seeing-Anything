"""Konfiguration der Spurgrenzensuche in der Bodenebene.

Zwei Verfahren, dieselbe Ausgabe
--------------------------------
`histogram`  Spaltensumme ueber die volle Hoehe, Peaks = Spurgrenzen.
             Schnell und stabil -- setzt aber SENKRECHTE Saeulen voraus, also
             gerade Spuren. In einer Kurve verschmiert die Summe und Peaks
             fallen aus.

`windows`    Sliding Windows: unten anfangen, fensterweise nach oben laufen und
             die Fenstermitte jeweils dorthin nachfuehren, wo die Pixel liegen.
             Aus den Fenstermitten ein Polynom fitten. Folgt der Kruemmung,
             kostet aber mehr Rechenzeit und braucht genug Pixel je Fenster.

Beide liefern `Boundaries` -- ein Polynom je Grenze. Beim Histogramm ist es vom
Grad 0 (eine Konstante), bei den Fenstern vom konfigurierten Grad. Damit muss
der nachgelagerte Code nur EINE Darstellung kennen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .loader import require_positive, require_range

Method = Literal["histogram", "windows"]


@dataclass(frozen=True)
class WindowConfig:
    """Kalibrierflaeche von `lanes.windows`."""

    # Welches Verfahren die Spurgrenzen sucht.
    method: Method = "histogram"

    # --- Startpunkte ---------------------------------------------------- #
    # Anteil der BEV-Hoehe von UNTEN, aus dem die Startpositionen kommen.
    # Nah an der Kamera ist die Maske am dichtesten und am wenigsten verzerrt;
    # eine Kurve ist dort noch fast gerade.
    start_band: float = 0.30

    # --- Fensterlauf ------------------------------------------------------ #
    n_windows: int = 12          # Fenster von unten nach oben
    margin: int = 40             # halbe Fensterbreite in BEV-Pixeln
    min_pixels: int = 25         # so viele Pixel braucht ein Fenster, um die
                                 #   Mitte nachzufuehren; sonst wird die letzte
                                 #   Richtung beibehalten
    # Wie viele Fenster mindestens getroffen haben muessen, damit die Grenze
    # ueberhaupt gilt. Zu niedrig: ein einzelner Fleck wird zur Spurlinie.
    min_windows_hit: int = 4

    # --- Fit ---------------------------------------------------------------- #
    # Grad 2 ist die uebliche Wahl fuer Fahrbahnkruemmung; Grad 1 macht die
    # Fenster zu einem reinen Ausreisserfilter, Grad 3 neigt zum Schwingen.
    poly_degree: int = 2
    # Wenn zu wenige Fenster getroffen haben, wird der Grad automatisch
    # gesenkt -- ein Polynom 2. Grades durch zwei Punkte ist Rauschen.
    min_points_per_degree: int = 3

    # --- Fortsetzung ueber Frames ------------------------------------------ #
    # Startpositionen des Vorframes wiederverwenden, statt jedes Mal neu zu
    # suchen. Ueberbrueckt eine Strichluecke im Nahbereich.
    reuse_previous_starts: bool = True
    max_start_shift: int = 60    # so weit darf ein Start je Frame wandern

    def __post_init__(self) -> None:
        if self.method not in ("histogram", "windows"):
            raise ValueError(f"unbekanntes Verfahren {self.method!r}; "
                             "erlaubt: histogram, windows")
        require_range(self.start_band, 0.05, 1.0, "windows.start_band")
        require_positive(self.n_windows, "windows.n_windows")
        require_positive(self.margin, "windows.margin")
        if self.min_pixels < 1:
            raise ValueError("windows.min_pixels muss mindestens 1 sein")
        if not 1 <= self.min_windows_hit <= self.n_windows:
            raise ValueError("windows.min_windows_hit muss zwischen 1 und "
                             "n_windows liegen")
        if not 1 <= self.poly_degree <= 3:
            raise ValueError("windows.poly_degree muss 1, 2 oder 3 sein")
        if self.min_points_per_degree < 2:
            raise ValueError("windows.min_points_per_degree muss mindestens 2 sein")
        require_positive(self.max_start_shift, "windows.max_start_shift")

    @property
    def uses_windows(self) -> bool:
        return self.method == "windows"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WindowConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"windows: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)

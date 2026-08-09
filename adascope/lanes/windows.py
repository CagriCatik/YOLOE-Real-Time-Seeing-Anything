"""Spurgrenzen per Sliding Windows -- das Verfahren fuer gekruemmte Fahrbahnen.

Das Problem, das es loest
-------------------------
Das Spaltenhistogramm summiert ueber die volle BEV-Hoehe. Das setzt voraus,
dass eine Spurlinie in einer Spalte bleibt -- also dass die Fahrbahn gerade
ist. In einer Kurve wandert dieselbe Linie ueber viele Spalten, die Summe
verschmiert, und ab einer gewissen Kruemmung faellt der Peak unter die
Schwelle.

Kruemmung ist ein MODELL-Problem, kein Ausreisser-Problem: die Annahme
"senkrecht" ist falsch, nicht die Daten. Ein robuster Fit hilft dagegen nicht,
mehr Modellkapazitaet schon.

Wie es arbeitet
---------------
1. Startpositionen aus dem Histogramm der untersten `start_band` der BEV-Hoehe.
   Nah an der Kamera ist die Maske am dichtesten, und dort ist auch eine Kurve
   noch fast gerade.
2. Von unten nach oben in `n_windows` Fenstern der halben Breite `margin`
   laufen. Enthaelt ein Fenster genug Pixel, wird die naechste Fenstermitte auf
   deren Schwerpunkt gesetzt -- so folgt das Fenster der Linie.
3. Aus den getroffenen Fenstermitten ein Polynom `x = p(y)` fitten.

Die Fenster sind damit zugleich ein Ausreisserfilter: was ausserhalb des
Fensters liegt, geht gar nicht erst in den Fit ein.
"""

from __future__ import annotations

import numpy as np

from ..config import BevConfig, WindowConfig
from .bev import lane_histogram, peaks_from_histogram
from .boundaries import Boundaries, fit_curve


def start_positions(mask_bev: np.ndarray, bev: BevConfig,
                    windows: WindowConfig) -> list[int]:
    """Wo die Spurlinien am unteren Bildrand stehen.

    Bewusst nur das untere Band statt der vollen Hoehe: dort ist die Maske am
    dichtesten, und die Kruemmung hat noch nicht zugeschlagen.
    """
    height = mask_bev.shape[0]
    band = mask_bev[int(height * (1 - windows.start_band)):, :]
    return peaks_from_histogram(lane_histogram(band), bev)


def trace(mask_bev: np.ndarray, start_x: int, windows: WindowConfig
          ) -> tuple[np.ndarray, np.ndarray, int]:
    """Eine Linie von unten nach oben verfolgen.

    Rueckgabe: (y-Mitten, x-Mitten, Anzahl getroffener Fenster). Fenster ohne
    genug Pixel liefern keinen Punkt -- die Mitte laeuft dann mit der zuletzt
    gemessenen Richtung weiter, statt stehenzubleiben. Eine Strichluecke
    entgleist den Lauf so nicht.
    """
    height, width = mask_bev.shape
    window_height = height / windows.n_windows
    ys = np.nonzero(mask_bev)[0]
    xs = np.nonzero(mask_bev)[1]

    centre = float(start_x)
    drift = 0.0
    found_y: list[float] = []
    found_x: list[float] = []

    for index in range(windows.n_windows):
        low = height - (index + 1) * window_height
        high = height - index * window_height
        inside = ((ys >= low) & (ys < high)
                  & (xs >= centre - windows.margin) & (xs < centre + windows.margin))
        count = int(inside.sum())
        if count >= windows.min_pixels:
            new_centre = float(xs[inside].mean())
            drift = new_centre - centre          # Richtung merken
            centre = new_centre
            found_y.append((low + high) / 2)
            found_x.append(new_centre)
        else:
            centre += drift                       # blind weiterlaufen
        if not 0 <= centre < width:
            break
    return np.array(found_y), np.array(found_x), len(found_y)


def find_boundaries(mask_bev: np.ndarray, bev: BevConfig, windows: WindowConfig,
                    previous: list[int] | None = None) -> Boundaries:
    """Alle Spurgrenzen als Polynome.

    `previous` sind die Startpositionen des Vorframes. Sie werden benutzt, wenn
    im Nahbereich gerade eine Strichluecke liegt und das Histogramm dort nichts
    findet -- die Linie ist ja nicht weg, nur unsichtbar.
    """
    starts = start_positions(mask_bev, bev, windows)
    if previous and windows.reuse_previous_starts:
        starts = _merge_starts(starts, previous, bev.peak_min_distance,
                               windows.max_start_shift)
    if not starts:
        return Boundaries(method="windows")

    height = mask_bev.shape[0]
    reference = height * (1 - windows.start_band / 2)

    curves: list[tuple[float, ...]] = []
    support: list[float] = []
    for start in starts:
        ys, xs, hits = trace(mask_bev, start, windows)
        if hits < windows.min_windows_hit:
            continue                              # zu wenig Beleg -> keine Grenze
        curves.append(fit_curve(ys, xs, windows.poly_degree,
                                windows.min_points_per_degree))
        support.append(hits / windows.n_windows)

    if not curves:
        return Boundaries(method="windows")
    order = np.argsort([np.polyval(c, reference) for c in curves])
    return Boundaries(tuple(curves[i] for i in order), float(reference),
                      tuple(support[i] for i in order), "windows")


def _merge_starts(found: list[int], previous: list[int], min_distance: int,
                  max_shift: int) -> list[int]:
    """Gefundene Startpositionen um die des Vorframes ergaenzen.

    Nur solche, die zu keiner gefundenen passen -- sonst wuerde dieselbe Linie
    zweimal verfolgt. `max_shift` verhindert, dass eine laengst verlassene
    Position ewig weitergeschleppt wird.
    """
    merged = list(found)
    for candidate in previous:
        if all(abs(candidate - x) >= min_distance for x in merged):
            if any(abs(candidate - x) <= max_shift for x in found) or not found:
                merged.append(candidate)
    return sorted(merged)

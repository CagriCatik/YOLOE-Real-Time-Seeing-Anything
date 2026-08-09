"""CSV-Ausgabe. Die Spalten ergeben sich aus der ersten Zeile."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def write_rows(rows: Iterable[Mapping], path: str | Path) -> int:
    """Schreibt Mappings als CSV und liefert die Zeilenzahl.

    Eine leere Eingabe schreibt nichts -- ohne Zeile gibt es keine Spalten, und
    eine Datei mit erfundenem Kopf waere schlechter als keine.
    """
    rows = list(rows)
    if not rows:
        return 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_named(rows: Iterable[Mapping], path: str | Path, fields: list[str]) -> int:
    """Wie `write_rows`, aber mit fester Spaltenreihenfolge.

    Fuer Ausgaben, deren Kopf stabil bleiben muss, auch wenn eine Zeile ein
    optionales Feld nicht setzt.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
            count += 1
    return count

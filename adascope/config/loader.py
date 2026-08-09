"""YAML-Laden, Deep-Merge und Szenario-Overlays.

Regeln, die fuer *jede* Domaenen-Config in diesem Paket gelten:

* **Der Default steht im Code, nicht in der YAML.** Jede Config-Dataclass ist
  ohne Datei konstruierbar. Tests und Bibliotheksnutzung brauchen damit kein
  Dateisystem, und eine fehlende Datei ist kein Fehler, sondern „Defaults".
* **Die YAML ueberschreibt nur, was sie nennt.** Ein Szenario-Overlay enthaelt
  ausschliesslich Abweichungen; alles andere kommt aus der Basisdatei bzw. dem
  Code-Default.
* **Validiert wird beim Laden, einmal.** Danach sind die Objekte frozen und es
  gibt keinen Pfad mehr, auf dem ein ungueltiger Wert in die Pipeline kommt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

T = TypeVar("T")

DEFAULT_CONFIG_DIR = Path("config")
SCENARIO_DIR = "scenarios"


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Liest eine YAML-Datei; eine fehlende Datei ergibt ein leeres Mapping."""
    target = Path(path)
    if not target.exists():
        return {}
    with target.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{target}: Wurzel muss ein Mapping sein, ist {type(raw).__name__}")
    return raw


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Rekursives Mischen. Listen werden ersetzt, nicht verkettet.

    Verkettung waere fuer Prompt-Listen bequem, aber dann liesse sich ein
    geerbter Eintrag nie mehr entfernen. Ersetzen ist die vorhersagbare Regel.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def require_range(value: float, low: float, high: float, name: str) -> float:
    if not low <= value <= high:
        raise ValueError(f"{name}={value} liegt ausserhalb von [{low}, {high}]")
    return value


def require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name}={value} muss groesser als 0 sein")
    return value


def as_int_pairs(value: Any, name: str) -> tuple[tuple[int, int], ...]:
    """Punktliste [[x, y], ...] -> Tupel von Integer-Paaren."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{name} braucht mindestens drei Punkte")
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name}: jeder Punkt muss [x, y] sein, gefunden {point!r}")
        points.append((int(point[0]), int(point[1])))
    return tuple(points)


def as_bgr(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} muss ein BGR-Tripel sein")
    channels = tuple(int(v) for v in value)
    if any(not 0 <= c <= 255 for c in channels):
        raise ValueError(f"{name}: BGR-Kanaele muessen in [0, 255] liegen")
    return channels  # type: ignore[return-value]


def load_section(cls: type[T], config_dir: str | Path, filename: str,
                 scenario: str | None = None) -> T:
    """Laedt `<config_dir>/<filename>`, optional ueberlagert vom Szenario.

    Das Szenario liegt unter `<config_dir>/scenarios/<scenario>.yaml` und ist
    nach Domaenen-Sektionen gegliedert; nur die Sektion mit dem Dateinamen-Stamm
    wird angewandt. So beschreibt eine einzelne Szenariodatei Abweichungen ueber
    alle Domaenen hinweg.
    """
    base_dir = Path(config_dir)
    raw = read_yaml(base_dir / filename)
    if scenario:
        overlay = read_yaml(base_dir / SCENARIO_DIR / f"{scenario}.yaml")
        section = overlay.get(Path(filename).stem, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"Szenario {scenario!r}: Sektion "
                             f"{Path(filename).stem!r} muss ein Mapping sein")
        raw = deep_merge(raw, section)
    return cls.from_dict(raw)          # type: ignore[attr-defined]

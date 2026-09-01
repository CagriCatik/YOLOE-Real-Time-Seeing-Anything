"""Szenarien finden, ausfuehren und vergleichen.

Ein *Szenario* ist eine Aufnahmesituation: ein Video (oder ein Frame-Ordner) in
`scenarien/` plus optional eine gleichnamige Kalibrier-Ueberlagerung in
`configs/scenarios/`. Der Name ist der Dateistamm und verbindet beides:

    scenarien/lane_departure_3_lanes.mp4     die Aufnahme
    configs/scenarios/lane_departure_3_lanes.yaml   nur die Abweichungen
    results/lane_departure_3_lanes/          alles, was dabei entsteht

Damit ist eine Regression eine Frage von einem Kommando: `adascope scenarios`
laeuft ueber alle Aufnahmen und schreibt eine Vergleichstabelle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .ground_truth import GroundTruth, Score
from .perception_ground_truth import PerceptionScore
from .io import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS

DEFAULT_SCENARIO_DIR = Path("scenarien")
DEFAULT_RESULT_DIR = Path("results")
CONFIG_SCENARIO_DIR = "scenarios"


@dataclass(frozen=True)
class Scenario:
    """Eine Aufnahme mit optionaler eigener Kalibrierung."""

    name: str
    source: Path                      # Videodatei oder Frame-Ordner
    config_overlay: Path | None       # configs/scenarios/<name>.yaml, falls vorhanden

    @property
    def is_video(self) -> bool:
        return self.source.is_file()

    @property
    def overlay_name(self) -> str | None:
        """Der Name, den `Settings.load(scenario=...)` erwartet."""
        return self.name if self.config_overlay else None

    def result_dir(self, root: Path = DEFAULT_RESULT_DIR) -> Path:
        return Path(root) / self.name

    def settings(self, config_dir: Path) -> Settings:
        return Settings.load(config_dir, self.overlay_name)

    def ground_truth(self, directory: str | Path = "ground_truth") -> GroundTruth | None:
        return GroundTruth.load(self.name, directory)

    def describe(self, ground_truth_dir: str | Path = "ground_truth") -> str:
        kind = "Video" if self.is_video else "Frames"
        overlay = "eigene Kalibrierung" if self.config_overlay else "Basiskalibrierung"
        truth = self.ground_truth(ground_truth_dir)
        annotated = (f"{len(truth.events)} Ereignis(se), "
                     f"{len(truth.perception)} Wahrnehmungsframe(s)"
                     if truth is not None else "keine Annotation")
        return f"{self.name:<32s} {kind:<7s} {overlay:<22s} {annotated}"


def discover(scenario_dir: Path = DEFAULT_SCENARIO_DIR,
             config_dir: Path = Path("configs")) -> list[Scenario]:
    """Alle Szenarien in `scenario_dir`, alphabetisch.

    Erkannt werden Videodateien und Ordner, die mindestens ein Bild enthalten.
    Ein leerer Ordner ist kein Szenario -- er waere sonst ein Lauf, der sofort
    mit einer Fehlermeldung endet.
    """
    base = Path(scenario_dir)
    if not base.is_dir():
        return []
    overlays = Path(config_dir) / CONFIG_SCENARIO_DIR

    found: dict[str, Scenario] = {}
    for entry in sorted(base.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            name = entry.stem
        elif entry.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS
                                    for p in entry.iterdir() if p.is_file()):
            name = entry.name
        else:
            continue
        # Ein Video gewinnt gegen einen gleichnamigen Ordner: es ist die
        # kompaktere Quelle und beide enthalten dieselben Frames.
        if name in found and found[name].is_video:
            continue
        overlay = overlays / f"{name}.yaml"
        found[name] = Scenario(name, entry, overlay if overlay.exists() else None)
    return [found[name] for name in sorted(found)]


def resolve(names: list[str], scenario_dir: Path = DEFAULT_SCENARIO_DIR,
            config_dir: Path = Path("configs")) -> list[Scenario]:
    """Namen zu Szenarien aufloesen; leere Liste bedeutet „alle"."""
    available = discover(scenario_dir, config_dir)
    if not names:
        return available
    by_name = {s.name: s for s in available}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"Unbekannte(s) Szenario(s): {unknown}. "
                         f"Verfuegbar: {sorted(by_name) or '(keine)'}")
    return [by_name[name] for name in names]


# --------------------------------------------------------------------------- #
# Zusammenfassung eines Laufs                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RunSummary:
    """Die Kennzahlen eines Szenariolaufs -- die Grundlage des Vergleichs."""

    scenario: str
    frames: int = 0
    homography: dict[str, int] = field(default_factory=dict)
    usable_lanes: int = 0
    index_jumps: int = 0
    index_transitions: int = 0
    lane_width_median: float = 0.0
    ego_in_lane_min: float = 1.0
    ego_departing_frames: int = 0
    events: dict[str, int] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    error: str = ""
    # Nur gesetzt, wenn es zu diesem Szenario eine Annotation gibt.
    score: Score | None = None
    perception_score: PerceptionScore | None = None

    @property
    def fresh_pct(self) -> float:
        return 100 * self.homography.get("fresh", 0) / max(self.frames, 1)

    @property
    def usable_pct(self) -> float:
        return 100 * self.usable_lanes / max(self.frames, 1)

    @property
    def jump_pct(self) -> float:
        return 100 * self.index_jumps / max(self.index_transitions, 1)

    def as_row(self) -> dict[str, str]:
        if self.error:
            return {"Szenario": self.scenario, "Frames": "--", "H fresh": "--",
                    "Spurliste": "--", "Index-Spruenge": "--", "Ego min": "--",
                    "Ereignisse": f"FEHLER: {self.error}", "Bewertung": "--",
                    "Wahrnehmung": "--"}
        events = ", ".join(f"{k}={v}" for k, v in sorted(self.events.items())) or "keine"
        return {
            "Szenario": self.scenario,
            "Frames": str(self.frames),
            "H fresh": f"{self.fresh_pct:.0f} %",
            "Spurliste": f"{self.usable_pct:.0f} %",
            "Index-Spruenge": f"{self.jump_pct:.1f} %",
            "Ego min": f"{self.ego_in_lane_min:.2f}",
            "Ereignisse": events,
            "Bewertung": self.score.label() if self.score else "keine Annotation",
            "Wahrnehmung": (self.perception_score.label()
                              if self.perception_score else "nicht annotiert"),
        }

    def as_text(self) -> str:
        if self.error:
            return f"Szenario {self.scenario}\n  FEHLER: {self.error}\n"
        lines = [
            f"Szenario {self.scenario}",
            f"  Frames                 {self.frames}",
            "  Homographie            " + "  ".join(
                f"{k}={v} ({100 * v / max(self.frames, 1):.0f} %)"
                for k, v in sorted(self.homography.items())),
            f"  Verwertbare Spurliste  {self.usable_lanes} ({self.usable_pct:.0f} %)",
            f"  Spurbreite (Median)    {self.lane_width_median:.0f} px",
            f"  Ego-Index-Spruenge     {self.index_jumps} ({self.jump_pct:.1f} % der Uebergaenge)",
            f"  Ego in eigener Spur    min {self.ego_in_lane_min:.2f}, "
            f"{self.ego_departing_frames} Frames unter 1.00",
            "  Ereignisse             " + (
                "  ".join(f"{k}={v}" for k, v in sorted(self.events.items())) or "keine"),
        ]
        if self.score is not None:
            lines.append(self.score.as_text())
        if self.perception_score is not None:
            lines.append(self.perception_score.as_text())
        if self.outputs:
            lines.append("  Ausgaben")
            lines += [f"    {p}" for p in self.outputs]
        return "\n".join(lines) + "\n"


def render_table(summaries: list[RunSummary]) -> str:
    """Markdown-Tabelle ueber alle Laeufe -- der Regressionsblick."""
    rows = [s.as_row() for s in summaries]
    if not rows:
        return "_Keine Szenarien ausgefuehrt._\n"
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}
    line = lambda cells: "| " + " | ".join(
        cells[h].ljust(widths[h]) for h in headers) + " |"
    return "\n".join([
        line({h: h for h in headers}),
        "|" + "|".join("-" * (widths[h] + 2) for h in headers) + "|",
        *(line(r) for r in rows),
    ]) + "\n"

"""Sollereignisse je Szenario und ihre Bewertung.

Die synthetischen Szenen (`adascope.synthetic`) beweisen, dass die LOGIK
stimmt. Ob sie auf echtem Material greift, kann nur eine Annotation sagen --
ohne sie ist „keine Ereignisse" nicht von „nichts erkannt" zu unterscheiden.
Genau diese Verwechslung hat drei Defekte in der State Machine monatelang
verdeckt.

Format: `ground_truth/<szenario>.yaml`

    tolerance: 8          # optional, Frames Abweichung je Ereignis
    events:
      - {frame: 142, kind: cut_in,  track: any, direction: rechts}
      - {frame: 210, kind: cut_out, track: any, direction: links}
    # leere Liste = "hier passiert nachweislich nichts" -- eine Aussage,
    # keine fehlende Datei

`track: any` laesst die Track-ID offen: sie haengt am Detektor und aendert sich
mit jedem Modellwechsel, das Ereignis nicht.

Zuordnung: gieriges Matching nach zeitlicher Naehe innerhalb der Toleranz.
Jedes Sollereignis bindet hoechstens ein gemeldetes und umgekehrt -- eine
Ereignissalve zaehlt damit als ein Treffer plus Falschalarme, nicht als
mehrfacher Treffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config.loader import read_yaml

DEFAULT_GROUND_TRUTH_DIR = Path("ground_truth")
DEFAULT_TOLERANCE = 8
ANY_TRACK = "any"


@dataclass(frozen=True)
class ExpectedEvent:
    frame: int
    kind: str
    track: str = ANY_TRACK
    # FR-7.2 verlangt ausdruecklich die Pruefung der RICHTUNG. `any` laesst
    # sie offen -- fuer Ereignisse, bei denen sie nicht relevant ist.
    direction: str = ANY_TRACK

    def matches(self, kind: str, track: str | None,
                direction: str = ANY_TRACK) -> bool:
        if kind != self.kind:
            return False
        if self.track != ANY_TRACK and self.track != track:
            return False
        return self.direction == ANY_TRACK or self.direction == direction


@dataclass(frozen=True)
class GroundTruth:
    events: tuple[ExpectedEvent, ...]
    tolerance: int = DEFAULT_TOLERANCE

    @classmethod
    def load(cls, name: str, directory: str | Path = DEFAULT_GROUND_TRUTH_DIR
             ) -> "GroundTruth | None":
        path = Path(directory) / f"{name}.yaml"
        if not path.exists():
            return None
        raw = read_yaml(path)
        unknown = set(raw) - {"events", "tolerance", "note"}
        if unknown:
            raise ValueError(f"{path}: unbekannte Schluessel {sorted(unknown)}")
        events = []
        for entry in raw.get("events") or []:
            missing = {"frame", "kind"} - set(entry)
            if missing:
                raise ValueError(f"{path}: Ereignis ohne {sorted(missing)}: {entry}")
            unknown_keys = set(entry) - {"frame", "kind", "track", "direction"}
            if unknown_keys:
                raise ValueError(f"{path}: unbekannte Schluessel im Ereignis: "
                                 f"{sorted(unknown_keys)}")
            direction = str(entry.get("direction", ANY_TRACK))
            if direction not in (ANY_TRACK, "links", "rechts"):
                raise ValueError(f"{path}: Richtung muss links, rechts oder "
                                 f"{ANY_TRACK} sein, ist {direction!r}")
            events.append(ExpectedEvent(int(entry["frame"]), str(entry["kind"]),
                                        str(entry.get("track", ANY_TRACK)), direction))
        return cls(tuple(sorted(events, key=lambda e: e.frame)),
                   int(raw.get("tolerance", DEFAULT_TOLERANCE)))


@dataclass
class Score:
    """Das Ergebnis einer Bewertung. `expected == 0` ist ein gueltiger Fall."""

    expected: int = 0
    matched: int = 0
    missed: list[ExpectedEvent] = field(default_factory=list)
    spurious: list = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)

    @property
    def detected(self) -> int:
        return self.matched + len(self.spurious)

    @property
    def recall(self) -> float:
        return self.matched / self.expected if self.expected else 1.0

    @property
    def precision(self) -> float:
        return self.matched / self.detected if self.detected else 1.0

    @property
    def perfect(self) -> bool:
        return not self.missed and not self.spurious

    @property
    def mean_offset(self) -> float:
        return sum(abs(o) for o in self.offsets) / len(self.offsets) if self.offsets else 0.0

    def label(self) -> str:
        if self.perfect:
            return f"{self.matched}/{self.expected} ok" + (
                f" (±{self.mean_offset:.0f}f)" if self.offsets else "")
        parts = [f"{self.matched}/{self.expected}"]
        if self.missed:
            parts.append(f"{len(self.missed)} fehlt")
        if self.spurious:
            parts.append(f"{len(self.spurious)} falsch")
        return " ".join(parts)

    def as_text(self) -> str:
        lines = [f"  Sollereignisse         {self.expected}",
                 f"  Treffer                {self.matched}"
                 + (f"  (Zeitversatz im Mittel {self.mean_offset:.1f} Frames)"
                    if self.offsets else "")]
        for event in self.missed:
            lines.append(f"  NICHT ERKANNT          f{event.frame} {event.kind} "
                         f"{event.track} nach {event.direction}")
        for event in self.spurious:
            lines.append(f"  FALSCHALARM            f{event.frame} {event.kind} "
                         f"{event.track or 'ego'} nach "
                         f"{getattr(event, 'direction', '?')}")
        return "\n".join(lines)


def score_events(expected: GroundTruth, detected: list) -> Score:
    """Gemeldete gegen erwartete Ereignisse zuordnen.

    Gierig nach zeitlicher Naehe: jedes Sollereignis nimmt das naechstgelegene
    passende gemeldete innerhalb der Toleranz. Nicht zugeordnete Sollereignisse
    sind Fehlerkennungen, nicht zugeordnete gemeldete sind Falschalarme.
    """
    score = Score(expected=len(expected.events))
    remaining = list(detected)
    for wanted in expected.events:
        candidates = [(abs(event.frame - wanted.frame), index)
                      for index, event in enumerate(remaining)
                      if wanted.matches(event.kind, event.track,
                                        getattr(event, "direction", ANY_TRACK))
                      and abs(event.frame - wanted.frame) <= expected.tolerance]
        if not candidates:
            score.missed.append(wanted)
            continue
        offset, index = min(candidates)
        score.matched += 1
        score.offsets.append(offset)
        remaining.pop(index)
    score.spurious = remaining
    return score

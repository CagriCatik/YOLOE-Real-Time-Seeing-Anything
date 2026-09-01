"""
Temporale Ereignisableitung fuer Cut-In / Cut-Out.

Warum eine State Machine und nicht nur ein Schwellwert
------------------------------------------------------
Ein Schwellwert auf `ego_overlap` liefert pro Frame eine Aussage, aber kein
EREIGNIS. Drei Faelle unterscheidet er nicht:

  1. ABGEBROCHENER Spurwechsel: Fahrzeug driftet zur Linie und kehrt zurueck.
     Ein Schwellwert feuert, ein Zustandsautomat nicht - weil INSIDE nie
     erreicht wurde.
  2. FLACKERN: Messrauschen um die Schwelle erzeugt eine Ereignissalve.
     Entprellung ueber N aufeinanderfolgende Frames unterdrueckt das.
  3. EIGENER Spurwechsel: wechselt Ego selbst die Spur, verschiebt sich die
     ego-relative Nummer JEDES verfolgten Fahrzeugs gleichzeitig. Pro Fahrzeug
     betrachtet sieht das wie ein Cut-In aus.

Fall 3 ist der Grund fuer ZWEI Ebenen: eine Szenenebene, die den eigenen
Spurwechsel erkennt, und je Track eine Ebene fuer das Einscheren. Die
Szenenebene hat Vorrang - sie unterdrueckt die Fahrzeugereignisse, die durch
die eigene Bewegung nur scheinbar entstehen.

Was die State Machine NICHT leistet
-----------------------------------
Sie repariert keine schlechten Messungen. Ungueltige Samples (Fernfeld-
Artefakte, siehe `footprint_is_plausible`) muessen VOR dem Automaten
ausgefiltert werden, nicht in ihm. Ein Automat auf verrauschten Eingaengen
liefert geglaettete falsche Ereignisse.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from ..config import EventConfig

State = Literal["outside", "encroaching", "inside"]
EventKind = Literal["cut_in", "cut_out", "aborted", "ego_lane_change"]


@dataclass
class Event:
    """Ein verdichteter Grenzuebergang (FR-5.1).

    Traegt bewusst BEIDE Kennungen: `direction` beantwortet FR-1.2 ohne
    jede Spurnummer, `boundary_id` benennt die ueberquerte Linie, ohne die
    Zaehlung von links wieder einzufuehren.
    """

    frame: int                       # Frame, in dem das Ereignis feststeht
    kind: EventKind
    track: str | None = None         # None = Ego
    detail: str = ""
    direction: str = "unbestimmt"    # links | rechts | unbestimmt (FR-1.2)
    boundary_id: int | None = None   # ueberquerte Grenze (FR-5.1)
    frame_start: int | None = None   # Beginn der Bewegung
    frame_end: int | None = None     # Abschluss der Bewegung
    confidence: float = 1.0          # Belastbarkeit der Aussage (FR-1.4)
    certain: bool = True             # False = als unsicher gemeldet (FR-3.3)

    @property
    def frames(self) -> tuple[int, int]:
        """Spanne der Bewegung -- FR-5.1 verlangt sie, nicht nur einen Frame."""
        return (self.frame if self.frame_start is None else self.frame_start,
                self.frame if self.frame_end is None else self.frame_end)

    def __str__(self) -> str:
        who = self.track or "ego"
        start, end = self.frames
        span = f"f{start:03d}" if start == end else f"f{start:03d}-{end:03d}"
        parts = [f"[{span}]", f"{self.kind:15s}", who]
        if self.direction != "unbestimmt":
            parts.append(f"nach {self.direction}")
        if self.boundary_id is not None:
            parts.append(f"ueber B{self.boundary_id}")
        if not self.certain:
            parts.append("[UNSICHER]")
        if self.detail:
            parts.append(self.detail)
        return " ".join(parts)


@dataclass
class _Track:
    state: State = "outside"
    candidate: State = "outside"
    streak: int = 0
    missing: int = 0
    last_rel: int | None = None
    encroach_frames: int = 0
    was_inside: bool = False
    # Fuer Richtung und Frame-Spanne: wo und wann die Bewegung begann.
    approach_from: float | None = None
    approach_frame: int | None = None
    # Letzte gesehene Querposition. Rueckfall fuer die Richtung, wenn ein
    # Manoever so zuegig ist, dass es das Band zwischen den Schwellen in einem
    # Frame durchquert -- dann gibt es keinen `approach_from`, aber sehr wohl
    # eine Bewegung. Ohne diesen Rueckfall verschluckt das Richtungs-Gate
    # genau die schnellen Spurwechsel.
    last_position: float | None = None


# --------------------------------------------------------------------------- #
def _direction(before: float | None, after: float | None) -> str:
    """Richtung aus zwei lateralen Positionen -- ohne jede Spurnummer."""
    if before is None or after is None:
        return "unbestimmt"
    from .tracking_ids import crossing_direction
    return crossing_direction(before, after)


def _state_for(overlap: float, cfg: EventConfig) -> State:
    if overlap >= cfg.thr_inside:
        return "inside"
    if overlap >= cfg.thr_encroaching:
        return "encroaching"
    return "outside"


class CutInTracker:
    """Leitet Ereignisse aus einer Folge von Per-Frame-Belegungen ab.

    Eingang je Frame: track_id -> (rel, ego_overlap). Ungueltige Messungen
    werden vom Aufrufer weggelassen, nicht mit Ersatzwerten gefuellt.
    """

    def __init__(self, cfg: EventConfig = EventConfig()) -> None:
        self.cfg = cfg
        self._tracks: dict[str, _Track] = {}
        self._refractory = 0

    # -- Szenenebene -------------------------------------------------------- #
    def _detect_ego_shift(self, obs: dict[str, tuple[int, float]]) -> int | None:
        """Gleichsinnige Verschiebung ALLER Tracks = eigener Spurwechsel."""
        deltas = [rel - t.last_rel
                  for tid, (rel, _) in obs.items()
                  if (t := self._tracks.get(tid)) and t.last_rel is not None]
        moved = [d for d in deltas if d != 0]
        if len(moved) < self.cfg.ego_shift_min_tracks:
            return None
        if len(moved) != len(deltas):        # nicht alle -> einzelnes Fahrzeug
            return None
        common, count = Counter(moved).most_common(1)[0]
        return common if count == len(moved) else None

    # -- Hauptschritt ------------------------------------------------------- #
    def _direction_ok(self, direction: str, position: float | None) -> bool:
        """Ein Ereignis ohne bestimmbare Richtung erfuellt FR-1.2 nicht.

        Es entsteht, wenn sich die Querposition zwischen Anfahrt und Uebergang
        nicht bewegt hat -- also aus einem Indexsprung, nicht aus einem
        Fahrmanoever.

        Das Gate greift NUR, wenn Lateraldaten ueberhaupt vorlagen. "Keine
        Daten" ist etwas anderes als "Daten zeigen keine Bewegung": die State
        Machine ist bewusst ohne `lateral` isoliert testbar, und ein fehlender
        Eingang darf kein Ereignis verschlucken.
        """
        if not self.cfg.require_direction or position is None:
            return True
        return direction in ("links", "rechts")

    def update(self, frame: int, obs: dict[str, tuple[int, float]],
               ego_in_lane: float = 1.0,
               lateral: dict[str, float] | None = None,
               boundary_at: dict[str, int] | None = None) -> list[Event]:
        """Ein Frame verarbeiten.

        `lateral` ist die laterale BEV-Position je Track; daraus kommt die
        Richtung (FR-1.2, FR-2.3). `boundary_at` nennt die naechstgelegene
        Grenzenkennung je Track (FR-5.1). Beide sind optional, damit die
        State Machine isoliert testbar bleibt.

        `ego_in_lane` ist der Anteil des EIGENEN Footprints in der eigenen Spur.
        Er belegt einen vermuteten eigenen Spurwechsel: wechselt Ego wirklich,
        ueberschreitet sein Footprint die eigene Spurgrenze. Ohne diesen Beleg
        genuegten zwei gleichzeitig gleichsinnig wechselnde Fremdfahrzeuge, um
        die Szenenebene auszuloesen -- und die unterdrueckt dann genau die
        beiden echten Ereignisse.
        """
        events: list[Event] = []

        ego_shift = self._detect_ego_shift(obs)
        if ego_shift is not None and ego_in_lane > self.cfg.ego_departure_max:
            ego_shift = None
        if ego_shift is not None:
            # Verschieben sich die relativen Nummern nach LINKS, ist das Ego
            # nach RECHTS gefahren -- die Richtung gehoert ins Feld, nicht in
            # den Freitext (FR-1.2).
            side = "rechts" if ego_shift < 0 else "links"
            events.append(Event(frame, "ego_lane_change", None,
                                f"aus der Verschiebung der Tracks (rel {ego_shift:+d})",
                                direction=side))
            self._refractory = self.cfg.ego_refractory_frames
        elif self._refractory > 0:
            self._refractory -= 1
        suppressed = ego_shift is not None or self._refractory > 0

        lateral = lateral or {}
        boundary_at = boundary_at or {}
        for tid, (rel, overlap) in obs.items():
            neu = tid not in self._tracks
            t = self._tracks.setdefault(tid, _Track())
            t.missing = 0
            observed = _state_for(overlap, self.cfg)
            position = lateral.get(tid)

            if neu:
                # ERSTE Beobachtung eines Tracks ist kein Uebergang -- es gibt
                # kein Davor. Frueher startete jeder Track als `outside`; ein
                # Fahrzeug, das beim Aufnahmestart bereits in der Ego-Spur
                # steht, erzeugte dadurch sofort ein `cut_in`.
                #
                # Gemessen auf `adjusting_speed_scenario_8`: cut_in fuer ID2 in
                # Frame 1-5, dem allerersten Frame der Aufnahme. Vom Anwender
                # im Debugvideo als Falschalarm bestaetigt.
                t.state = t.candidate = observed
                t.streak = self.cfg.confirm_frames
                # Wer von Anfang an drin ist, darf spaeter ausscheren -- das
                # ist ein echtes Ereignis, auch ohne beobachtete Anfahrt.
                t.was_inside = observed == "inside"
                t.last_rel, t.last_position = rel, position
                continue

            # Anfahrt zaehlen, sobald sie BEOBACHTET wird -- nicht erst, wenn
            # `encroaching` als Zustand bestaetigt ist. Ein zuegiger Spurwechsel
            # durchquert das Band zwischen den Schwellen in ein bis zwei Frames;
            # bei einer Bestaetigungsforderung faellt sein cut_in stumm aus.
            # Der Zaehler unterscheidet trotzdem noch die drei Faelle:
            #   0            aus dem Nichts drin  -> kein Ereignis (Trackluecke)
            #   >= 1         angefahren           -> cut_in zulaessig
            #   >= confirm   wirklich verweilt    -> Abbruch ist ein Abbruch
            #                                        und kein Messrauschen
            if observed == "encroaching":
                t.encroach_frames += 1
                if t.approach_from is None:
                    # Startpunkt der Bewegung merken -- daraus wird spaeter
                    # die Richtung und der Beginn der Frame-Spanne.
                    t.approach_from, t.approach_frame = position, frame
            elif observed == "outside" and t.state == "outside":
                t.approach_from, t.approach_frame = None, None

            if observed == t.candidate:
                t.streak += 1
            else:
                t.candidate, t.streak = observed, 1

            if t.streak >= self.cfg.confirm_frames and observed != t.state:
                prev, t.state = t.state, observed
                if observed == "inside":
                    # Symmetrie zwischen Ein- und Ausfahrt. Ein Eintritt OHNE
                    # beobachtete Anfahrt ist ein Sprung (Trackluecke oder
                    # Indexsprung) -- er ist schon fuer ein `cut_in` zu
                    # unsicher. Dann darf er auch kein `cut_out` scharfmachen.
                    #
                    # Genau daran hing der doppelte Falschalarm auf
                    # `adjusting_speed_scenario_5`: ID4 meldete zweimal
                    # `cut_out` im Abstand von 15 Frames, beide mit
                    # `direction: unbestimmt`. Ein Fahrzeug kann nicht zweimal
                    # ausscheren, ohne dazwischen einzuscheren -- der
                    # Wiedereintritt war ein Indexsprung, kein Fahrmanoever.
                    t.was_inside = t.encroach_frames >= 1
                    richtung = _direction(t.approach_from if t.approach_from
                                          is not None else t.last_position,
                                          position)
                    if (t.encroach_frames >= 1 and not suppressed
                            and self._direction_ok(richtung, position)):
                        events.append(Event(
                            frame, "cut_in", tid, f"({prev} -> inside)",
                            direction=richtung,
                            boundary_id=boundary_at.get(tid),
                            frame_start=t.approach_frame, frame_end=frame))
                    t.encroach_frames = 0
                    t.approach_from, t.approach_frame = None, None
                elif observed == "outside":
                    # `was_inside` statt `prev == "inside"`: ein langsames
                    # Ausscheren geht inside -> encroaching -> outside, dann ist
                    # `prev` beim letzten Uebergang `encroaching`. Das als
                    # Abbruch zu melden waere falsch -- das Fahrzeug WAR drin.
                    richtung = _direction(t.approach_from if t.approach_from
                                          is not None else t.last_position,
                                          position)
                    if t.was_inside:
                        if not suppressed and self._direction_ok(richtung, position):
                            events.append(Event(
                                frame, "cut_out", tid,
                                direction=richtung,
                                boundary_id=boundary_at.get(tid),
                                frame_start=t.approach_frame, frame_end=frame))
                    elif t.encroach_frames >= self.cfg.confirm_frames:
                        events.append(Event(
                            frame, "aborted", tid, "kein Ereignis emittiert",
                            direction=_direction(t.approach_from, position),
                            boundary_id=boundary_at.get(tid),
                            frame_start=t.approach_frame, frame_end=frame))
                    t.encroach_frames, t.was_inside = 0, False
                    t.approach_from, t.approach_frame = None, None

            t.last_rel = rel
            if position is not None:
                t.last_position = position

        for tid in list(self._tracks):
            if tid in obs:
                continue
            self._tracks[tid].missing += 1
            if self._tracks[tid].missing > self.cfg.max_missing:
                del self._tracks[tid]

        return events

    def state_of(self, tid: str) -> State | None:
        t = self._tracks.get(tid)
        return t.state if t else None

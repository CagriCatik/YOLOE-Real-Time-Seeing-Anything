"""Spurgrenzen ueber die Zeit stabilisieren.

Das Problem, gemessen
---------------------
Die Grenzensuche hat kein Gedaechtnis: das Spaltenhistogramm wird in jedem
Frame vollstaendig neu ausgewertet. Auf `adjusting_speed_scenario_8` aendert
sich dadurch in **34 %** der Frameuebergaenge die Korridorzahl -- bei
durchgehend gueltiger Homographie. Eine Strichluecke laesst eine Grenze
verschwinden, der naechste Frame findet sie wieder.

Das Flackern pflanzt sich fort: es erzeugt die Index-Spruenge, die springenden
Grenzen-Kennungen und das sichtbare Zittern im Debugvideo.

Zwei Ursachen, zwei Gegenmittel
-------------------------------
**Ausfall** -- eine echte Grenze fehlt kurz. Gegenmittel: halten. Aber nur,
solange das Histogramm an der Stelle noch *etwas* zeigt (`hold_min_ratio` der
regulaeren Schwelle). Eine Grenze, unter der nichts mehr ist, wird nicht
herbeigehalten -- sonst ueberlebt sie den eigenen Spurwechsel.

**Scheingrenze** -- ein Ausreisserpeak taucht kurz auf. Gegenmittel: erst nach
`confirm_frames` aufeinanderfolgenden Beobachtungen melden.

Beides ist noetig. Nur halten fuellt Luecken und laesst Scheingrenzen durch;
nur bestaetigen unterdrueckt Scheingrenzen und laesst Luecken offen.

Abgrenzung zu `tracking_ids`
----------------------------
Hier wird entschieden, **welche Grenzen es gibt**. Welche *Kennung* jede traegt,
entscheidet weiterhin `BoundaryTracker` -- danach, und auf dem stabilisierten
Ergebnis. Die Trennung ist beabsichtigt: eine Kennung zu vergeben ist etwas
anderes, als eine Beobachtung zu ergaenzen oder zu verwerfen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.boundaries import BoundaryTrackConfig
from .boundaries import Boundaries


@dataclass
class _Tracked:
    """Eine verfolgte Grenze zwischen zwei Frames."""

    x: float                        # Position an der Referenzhoehe
    curve: tuple[float, ...]        # zuletzt gesehene Kurve
    support: float = 1.0
    seen: int = 0                   # Beobachtungen in Folge
    missing: int = 0                # Frames ohne Beobachtung
    emitted: bool = False           # war schon einmal bestaetigt

    @property
    def held(self) -> bool:
        return self.missing > 0


@dataclass
class StabilizeReport:
    """Was in diesem Frame passiert ist -- fuer Debugbild und Kennzahlen."""

    measured: int = 0
    emitted: int = 0
    held: int = 0                   # ergaenzt, weil kurz ausgefallen
    pending: int = 0                # gesehen, aber noch nicht bestaetigt
    dropped: int = 0                # Kennung endgueltig aufgegeben

    @property
    def changed(self) -> bool:
        return self.held > 0 or self.pending > 0


@dataclass
class BoundaryStabilizer:
    """Haelt kurz ausgefallene Grenzen und unterdrueckt Scheingrenzen."""

    cfg: BoundaryTrackConfig = field(default_factory=BoundaryTrackConfig)
    # Schwelle der reglaeren Peaksuche; der Haltetest arbeitet relativ dazu.
    peak_min_pixels: float = 0.0
    _tracked: list[_Tracked] = field(default_factory=list)
    report: StabilizeReport = field(default_factory=StabilizeReport)

    def reset(self) -> None:
        self._tracked.clear()
        self.report = StabilizeReport()

    def update(self, found: Boundaries,
               histogram: np.ndarray | None = None) -> Boundaries:
        """Die stabilisierte Fassung der gefundenen Grenzen.

        `histogram` ist das Spaltenhistogramm desselben Frames. Fehlt es, wird
        rein zeitlich gehalten -- schwaecher, weil dann auch eine wirklich
        verschwundene Grenze `max_missing` Frames ueberlebt.
        """
        if not self.cfg.stabilize:
            return found

        report = StabilizeReport(measured=len(found))
        measured = self._measurements(found)
        matched = self._match(measured)

        survivors: list[_Tracked] = []
        for index, tracked in enumerate(self._tracked):
            hit = matched.get(index)
            if hit is not None:
                x, curve, support = measured[hit]
                tracked.x, tracked.curve, tracked.support = x, curve, support
                tracked.seen += 1
                tracked.missing = 0
                survivors.append(tracked)
            elif self._may_hold(tracked, histogram):
                tracked.missing += 1
                tracked.seen = 0        # Bestaetigung beginnt neu
                report.held += 1
                survivors.append(tracked)
            else:
                report.dropped += 1

        taken = set(matched.values())
        for index, (x, curve, support) in enumerate(measured):
            if index not in taken:
                survivors.append(_Tracked(x, curve, support, seen=1))

        self._tracked = sorted(survivors, key=lambda t: t.x)
        return self._emit(found, report)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _measurements(found: Boundaries) -> list[tuple[float, tuple[float, ...], float]]:
        y = found.y_reference
        support = found.support or tuple(1.0 for _ in found.curves)
        rows = [(float(np.polyval(c, y)), c, float(s))
                for c, s in zip(found.curves, support)]
        return sorted(rows, key=lambda r: r[0])

    def _match(self, measured: list) -> dict[int, int]:
        """Gierige Zuordnung verfolgt -> gemessen, naechste Paare zuerst.

        Gierig nach Abstand statt der Reihe nach: sonst reisst ein einzelner
        Ausreisser am linken Rand die gesamte Zuordnung um eine Position.
        """
        pairs = sorted(
            (abs(t.x - m[0]), ti, mi)
            for ti, t in enumerate(self._tracked)
            for mi, m in enumerate(measured)
            if abs(t.x - m[0]) <= self.cfg.search_radius)
        matched: dict[int, int] = {}
        used: set[int] = set()
        for _, ti, mi in pairs:
            if ti in matched or mi in used:
                continue
            matched[ti] = mi
            used.add(mi)
        return matched

    def _may_hold(self, tracked: _Tracked, histogram: np.ndarray | None) -> bool:
        """Darf diese Grenze ohne Beobachtung weiterleben?

        Zwei Bedingungen. Zeitlich: nicht laenger als `max_missing`. Sachlich:
        das Histogramm zeigt an der Stelle noch Restsignal. Fehlt das
        Histogramm, greift nur die zeitliche Schranke.
        """
        if not tracked.emitted or tracked.missing >= self.cfg.max_missing:
            return False
        if histogram is None or self.peak_min_pixels <= 0:
            return True
        floor = self.cfg.hold_min_ratio * self.peak_min_pixels
        window = max(1, int(round(self.cfg.search_radius / 2)))
        lo = max(0, int(round(tracked.x)) - window)
        hi = min(len(histogram), int(round(tracked.x)) + window + 1)
        return lo < hi and float(histogram[lo:hi].max()) >= floor

    def _emit(self, found: Boundaries, report: StabilizeReport) -> Boundaries:
        # Beim Kaltstart gibt es keine Struktur, gegen die sich etwas als
        # Scheingrenze abheben koennte -- dort waere Bestaetigung nur eine
        # Verzoegerung. Sie gilt fuer Grenzen, die in einer STEHENDEN Struktur
        # neu auftauchen. Das greift auch nach `reset()` und nach einem
        # vollstaendigen Abriss.
        cold = not any(t.emitted for t in self._tracked)
        curves, support = [], []
        for tracked in self._tracked:
            if cold or tracked.emitted or tracked.seen >= self.cfg.confirm_frames:
                tracked.emitted = True
                curves.append(tracked.curve)
                # Eine gehaltene Grenze traegt ihre Unsicherheit im Support --
                # nachgelagerte Auswertung soll sie nicht fuer gemessen halten.
                support.append(tracked.support * self.cfg.hold_support
                               if tracked.held else tracked.support)
            else:
                report.pending += 1
        report.emitted = len(curves)
        self.report = report
        return Boundaries(tuple(curves), found.y_reference, tuple(support),
                          found.method)

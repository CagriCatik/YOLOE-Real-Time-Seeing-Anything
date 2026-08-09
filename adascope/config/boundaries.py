"""Konfiguration der Grenzen-Identitaet und der Ego-Bewegungsanalyse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loader import require_positive, require_range


@dataclass(frozen=True)
class BoundaryTrackConfig:
    """Kalibrierflaeche von `lanes.tracking_ids` (FR-1.4, FR-5.1)."""

    # So weit darf eine Grenze je Frame wandern und gilt noch als dieselbe.
    # Deutlich unter der Spurbreite, sonst uebernimmt eine Grenze die Identitaet
    # ihrer Nachbarin; deutlich ueber der Eigenbewegung je Frame.
    max_shift: float = 35.0
    # Frames, die eine nicht gesehene Grenze ihre Kennung behaelt. Eine
    # Strichluecke soll keine neue Identitaet erzeugen.
    max_missing: int = 12
    # Ab so vielen Beobachtungen gilt eine Grenze als voll belastbar
    # (confidence 1.0). Darunter wird linear hochgezaehlt.
    confident_after: int = 8

    # --- zeitliche Stabilisierung (`lanes.stabilize`) ---------------------
    # AUS als Standard, und das ist ein Messergebnis, keine Vorsicht.
    #
    # Die Annahme dahinter war: die Grenzenzahl schwankt, weil eine stabile
    # Struktur kurz ausfaellt. Nachgemessen stimmt sie nicht -- die Korridore
    # sind legitim (0 % zu schmal, 70-78 % exakt eine Spurbreite). Die Zahl
    # schwankt, weil am BEV-Rand echte Markierungen sichtbar werden und wieder
    # verschwinden. Halten ergaenzt dort keine Luecke, sondern sammelt an:
    # auf `adjusting_speed_scenario_8` steigt die mittlere Korridorzahl von
    # 4.26 auf 6.84 und die verwertbare Spurliste faellt von 93 % auf 70 %.
    #
    # Wo die Struktur wirklich steht, hilft es: auf `lane_departure_1_lane`
    # sinkt das Flackern von 8 % auf 2 %. Deshalb bleibt es einschaltbar --
    # je Aufnahme, nicht global.
    stabilize: bool = False
    # So weit darf eine Messung von der erwarteten Position abweichen und gilt
    # noch als dieselbe Grenze. Enger als `max_shift`: hier wird die Messung
    # zugeordnet, dort die Kennung -- ein Fehlgriff hier erfindet Geometrie.
    search_radius: float = 22.0
    # Restsignal, das eine gehaltene Grenze im Histogramm noch zeigen muss --
    # als Anteil von `bev.peak_min_pixels`. Bei 0 wird rein zeitlich gehalten,
    # dann ueberlebt eine wirklich verschwundene Grenze `max_missing` Frames.
    hold_min_ratio: float = 0.45
    # Beobachtungen in Folge, bevor eine NEUE Grenze gemeldet wird. 1 meldet
    # sofort (und laesst Ausreisser durch), hoehere Werte verzoegern echte
    # neue Grenzen um ebenso viele Frames.
    confirm_frames: int = 2
    # Support-Abschlag einer gehaltenen Grenze. Sie ist ergaenzt, nicht
    # gemessen; das soll im Ergebnis sichtbar bleiben.
    hold_support: float = 0.5

    def __post_init__(self) -> None:
        require_positive(self.max_shift, "boundaries.max_shift")
        if self.max_missing < 0:
            raise ValueError("boundaries.max_missing darf nicht negativ sein")
        require_positive(self.confident_after, "boundaries.confident_after")
        require_positive(self.search_radius, "boundaries.search_radius")
        require_range(self.hold_min_ratio, 0.0, 1.0, "boundaries.hold_min_ratio")
        require_positive(self.confirm_frames, "boundaries.confirm_frames")
        require_range(self.hold_support, 0.0, 1.0, "boundaries.hold_support")
        if self.search_radius > self.max_shift:
            raise ValueError(
                "boundaries.search_radius darf nicht groesser als max_shift sein "
                "-- sonst ordnet die Stabilisierung eine Messung zu, die die "
                "Identitaetsvergabe danach ablehnt")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BoundaryTrackConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"boundaries: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True)
class EgoMotionConfig:
    """Kalibrierflaeche von `lanes.egomotion` (FR-3.1 bis FR-3.3).

    Ein eigener Spurwechsel zeigt sich als **parallele Verschiebung** der
    gesamten Linienstruktur um etwa eine Spurbreite. Eine Kurve zeigt sich als
    **Drehung**: die Grenzen wandern unterschiedlich weit, nah wenig und fern
    viel. Diese beiden zu trennen ist laut Anforderung die schwaechste Annahme
    des ganzen Systems -- deshalb hat jede Schwelle hier ein Abnahmegate.
    """

    # Kumulierte parallele Verschiebung, ab der ein Wechsel angenommen wird --
    # als Anteil der Spurbreite. 0.6 statt 1.0, weil die Erkennung vor dem
    # Abschluss der Bewegung greifen soll.
    shift_fraction: float = 0.60
    # Fenster in Frames, ueber das die Verschiebung aufsummiert wird. Zu kurz:
    # ein langsamer Wechsel faellt durch. Zu lang: eine lange Kurve summiert
    # sich zu einem Scheinwechsel.
    window: int = 20
    # OBERGRENZE fuer die Streuung der Einzelverschiebungen, als Anteil der
    # mittleren Verschiebung. Bei echter Translation bewegen sich alle Grenzen
    # gleich weit (Streuung klein); in einer Kurve nicht (Streuung gross).
    # DAS ist der Kurven-Confounder-Test aus FR-3.2.
    max_spread: float = 0.45
    # Mindestens so viele Grenzen muessen ueber das Fenster durchgehend
    # verfolgt worden sein. Mit zwei Grenzen ist Translation nicht von Drehung
    # zu unterscheiden.
    min_boundaries: int = 3
    # Ohne bestandene Trennung wird nicht geschwiegen, sondern als UNSICHER
    # gemeldet (FR-3.3). Auf False geschaltet unterbleibt auch das.
    report_uncertain: bool = True
    # Frames Sperrzeit nach einem gemeldeten Wechsel.
    refractory: int = 25

    def __post_init__(self) -> None:
        require_range(self.shift_fraction, 0.1, 2.0, "egomotion.shift_fraction")
        require_positive(self.window, "egomotion.window")
        require_range(self.max_spread, 0.0, 5.0, "egomotion.max_spread")
        if self.min_boundaries < 2:
            raise ValueError("egomotion.min_boundaries muss mindestens 2 sein -- "
                             "mit weniger ist Translation nicht von Drehung zu "
                             "unterscheiden")
        if self.refractory < 0:
            raise ValueError("egomotion.refractory darf nicht negativ sein")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EgoMotionConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"egomotion: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)

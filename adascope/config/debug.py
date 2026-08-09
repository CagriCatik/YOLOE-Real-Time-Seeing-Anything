"""Konfiguration der Debug-Ansichten: virtuelle Kameras, Farben, Layout.

Die virtuellen Kameras sind keine neuen Messungen. Die Bodenebene ist durch die
Homographie bereits festgelegt; jede weitere Ansicht davon ist wieder eine
Homographie. Deshalb ist eine Kamera reine Konfiguration -- neue Perspektiven
entstehen durch einen Eintrag in `debug.yaml`, nicht durch Code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .loader import as_bgr, require_positive, require_range

Color = tuple[int, int, int]


@dataclass(frozen=True)
class VirtualCamConfig:
    """Lochkamera auf die BEV-Bodenebene. Alle Laengen in BEV-Pixeln."""

    width: int = 900
    height: int = 560
    x: float = 250.0          # laterale Kameraposition in BEV-x
    behind: float = 240.0     # Abstand hinter der BEV-Unterkante
    elevation: float = 190.0  # Hoehe ueber der Bodenebene
    pitch_deg: float = 24.0   # >0 = nach unten geneigt
    yaw_deg: float = 0.0      # >0 = nach rechts gedreht
    focal: float = 620.0
    # Hauptpunkt-Hoehe. Ueber dem Horizont liegen keine Bilddaten -- den
    # Horizont nach oben zu schieben spart tote schwarze Flaeche.
    cy_ratio: float = 0.5

    def __post_init__(self) -> None:
        require_positive(self.width, "camera.width")
        require_positive(self.height, "camera.height")
        require_positive(self.focal, "camera.focal")
        require_positive(self.elevation, "camera.elevation")
        require_range(self.cy_ratio, 0, 1, "camera.cy_ratio")
        require_range(self.pitch_deg, -89, 89, "camera.pitch_deg")
        require_range(self.yaw_deg, -89, 89, "camera.yaw_deg")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VirtualCamConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"camera: unbekannte Schluessel {sorted(unknown)}")
        return cls(**raw)


DEFAULT_CAMERAS = {
    "oblique": VirtualCamConfig(),
    "shoulder": VirtualCamConfig(width=900, height=460, x=400.0, behind=300.0,
                                 elevation=180.0, pitch_deg=15.0, yaw_deg=-9.0,
                                 focal=850.0, cy_ratio=0.34),
}

# BGR. Namen sind Zustaende und Rollen, keine Farbnamen -- so bleibt die
# Bedeutung stabil, wenn jemand die Palette tauscht.
DEFAULT_COLORS: dict[str, Color] = {
    "state_outside": (0, 210, 0),
    "state_encroaching": (0, 200, 255),
    "state_inside": (0, 0, 255),
    "state_invalid": (150, 150, 150),
    "state_unknown": (200, 200, 200),
    "homography_fresh": (0, 210, 0),
    "homography_held": (0, 190, 255),
    "homography_none": (0, 0, 235),
    "event_cut_in": (0, 0, 255),
    "event_cut_out": (0, 210, 0),
    "event_aborted": (150, 150, 150),
    "event_ego_lane_change": (255, 170, 0),
    "role_solid": (0, 200, 255),
    "role_dashed": (0, 255, 0),
    "role_unknown": (0, 0, 255),
    "ego": (255, 255, 0),
    "ego_departing": (0, 170, 255),
    "boundary_raw": (95, 95, 95),        # Rohpeak aus dem Histogramm
    "boundary_lane": (255, 255, 255),    # uebernommene Spurgrenze
    "boundary_synthetic": (0, 170, 255),  # virtuelle Grenze aus split_corridors
    "untracked": (255, 150, 30),
    "grid": (44, 44, 44),
    "axis_label": (110, 110, 110),
    "caption": (150, 150, 150),
    "roi": (90, 90, 90),
    "smear": (200, 0, 200),
    "footprint": (0, 255, 255),
}


@dataclass(frozen=True)
class DashboardConfig:
    """Rasterung der Komposit-Ansicht. Kantenlaengen muessen gerade sein."""

    width: int = 1220
    front_height: int = 461
    panel_height: int = 480
    timeline_height: int = 179
    # Spaltengrenzen der mittleren Reihe: bev | oblique | hist
    panel_splits: tuple[int, int] = (366, 854)
    # Ausschnitt der Spurhaltungs-Achse im Zeitstreifen (1.0 unten).
    ego_axis_min: float = 0.8

    @property
    def height(self) -> int:
        return self.front_height + self.panel_height + self.timeline_height

    def __post_init__(self) -> None:
        if self.width % 2 or self.height % 2:
            raise ValueError(f"dashboard: {self.width}x{self.height} muss gerade Kanten haben "
                             "(mp4v schneidet sonst stumm eine Zeile ab)")
        low, high = self.panel_splits
        if not 0 < low < high < self.width:
            raise ValueError("dashboard.panel_splits muessen aufsteigend innerhalb der Breite liegen")
        require_range(self.ego_axis_min, 0, 1, "dashboard.ego_axis_min")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DashboardConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"dashboard: unbekannte Schluessel {sorted(unknown)}")
        values = dict(raw)
        if "panel_splits" in values:
            values["panel_splits"] = tuple(int(v) for v in values["panel_splits"])
        return cls(**values)


@dataclass(frozen=True)
class DebugConfig:
    """Kalibrierflaeche von `render.debug_views`."""

    cameras: dict[str, VirtualCamConfig] = field(
        default_factory=lambda: dict(DEFAULT_CAMERAS))
    colors: dict[str, Color] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    # Hoehe des Histogramm-Plots unter der BEV-Maske.
    hist_plot_height: int = 170
    # Fuer die Bodenquader in den Schraegansichten: reale Spur- und
    # Fahrzeughoehe in Metern, umgerechnet ueber die gemessene Spurbreite.
    lane_width_m: float = 3.5
    vehicle_height_m: float = 1.5
    # Ersatz-Spurbreite, wenn keine gemessen werden konnte.
    fallback_lane_width_px: float = 76.0
    # Ereignisse, die im Log der Bildebene stehen bleiben.
    event_log_size: int = 5

    def __post_init__(self) -> None:
        require_positive(self.hist_plot_height, "debug.hist_plot_height")
        require_positive(self.lane_width_m, "debug.lane_width_m")
        require_positive(self.vehicle_height_m, "debug.vehicle_height_m")
        require_positive(self.fallback_lane_width_px, "debug.fallback_lane_width_px")
        missing = set(DEFAULT_COLORS) - set(self.colors)
        if missing:
            raise ValueError(f"debug.colors: fehlende Eintraege {sorted(missing)}")

    def color(self, name: str) -> Color:
        return self.colors[name]

    def camera(self, name: str) -> VirtualCamConfig:
        try:
            return self.cameras[name]
        except KeyError:
            raise ValueError(f"unbekannte virtuelle Kamera {name!r}; "
                             f"verfuegbar: {sorted(self.cameras)}") from None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DebugConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"debug: unbekannte Schluessel {sorted(unknown)}")
        values = dict(raw)
        cameras = dict(DEFAULT_CAMERAS)
        for name, spec in values.pop("cameras", {}).items():
            base = cameras.get(name, VirtualCamConfig())
            merged = {**{f: getattr(base, f) for f in VirtualCamConfig.__dataclass_fields__},
                      **spec}
            cameras[name] = VirtualCamConfig.from_dict(merged)
        colors = dict(DEFAULT_COLORS)
        for name, value in values.pop("colors", {}).items():
            colors[name] = as_bgr(value, f"debug.colors.{name}")
        dashboard = DashboardConfig.from_dict(values.pop("dashboard", {}))
        return cls(cameras=cameras, colors=colors, dashboard=dashboard, **values)

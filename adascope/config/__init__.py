"""Typisierte Konfiguration, eine Datei je Domaene.

    configs/detection.yaml   YOLOE: Modell, ROIs, Carpet, Driving-Area, HUD
    configs/lane.yaml        Spurerkennung in der Bildebene (Hough)
    configs/bev.yaml         Bodenebene und Belegungsschwellen
    configs/tracking.yaml    YOLO11 + ByteTrack
    configs/indexing.yaml    ego-relative Spurnummerierung
    configs/events.yaml      temporale Ereignisableitung
    configs/pipeline.yaml    Zustand ueber Frames hinweg
    configs/windows.yaml     Verfahren der Spurgrenzensuche (Histogramm/Fenster)
    configs/boundaries.yaml  persistente Grenzen-IDs
    configs/egomotion.yaml   eigener Spurwechsel aus der Linienstruktur
    configs/debug.yaml       virtuelle Kameras, Farben, Layout

Eine Datei entspricht genau einer Dataclass. Jede Dataclass ist ohne Datei
konstruierbar; die YAML ueberschreibt nur, was sie nennt. Szenarien liegen unter
`configs/scenarios/<name>.yaml` und enthalten je Domaenen-Sektion ausschliesslich
Abweichungen -- so beschreibt eine Datei eine Aufnahmesituation vollstaendig,
ohne die Basiskalibrierung zu kopieren.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bev import BevConfig
from .boundaries import BoundaryTrackConfig, EgoMotionConfig
from .debug import DashboardConfig, DebugConfig, VirtualCamConfig
from .detection import (
    DEFAULT_AREA_PRIORITY, CarpetConfig, DetectionConfig, DrivingAreaConfig,
    HudConfig, ModelConfig,
    load_detection_config, save_crop_box, save_rois,
)
from .events import EventConfig
from .indexing import IndexConfig
from .lane import LaneConfig
from .loader import DEFAULT_CONFIG_DIR, deep_merge, load_section, read_yaml
from .pipeline import PipelineConfig
from .tracking import TrackingConfig
from .windows import Method, WindowConfig

__all__ = [
    "DEFAULT_AREA_PRIORITY", "BevConfig", "BoundaryTrackConfig", "EgoMotionConfig", "CarpetConfig", "DashboardConfig", "DebugConfig", "DetectionConfig",
    "DrivingAreaConfig", "EventConfig", "HudConfig", "IndexConfig", "LaneConfig",
    "Method", "ModelConfig", "PipelineConfig", "Settings", "TrackingConfig",
    "VirtualCamConfig", "WindowConfig",
    "deep_merge", "load_detection_config", "read_yaml", "save_crop_box", "save_rois",
]


@dataclass(frozen=True)
class Settings:
    """Alle Domaenen-Configs eines Laufs, einmal geladen und validiert."""

    root: Path
    config_dir: Path
    scenario: str | None
    lane: LaneConfig
    bev: BevConfig
    tracking: TrackingConfig
    indexing: IndexConfig
    events: EventConfig
    pipeline: PipelineConfig
    windows: WindowConfig
    boundaries: BoundaryTrackConfig
    egomotion: EgoMotionConfig
    debug: DebugConfig
    # Nur die YOLOE-Pipeline braucht sie; ohne `detection.yaml` bleibt sie None,
    # damit die Spur-Pipeline ohne diese Datei lauffaehig ist.
    detection: DetectionConfig | None = None

    @classmethod
    def load(cls, config_dir: str | Path = DEFAULT_CONFIG_DIR,
             scenario: str | None = None, root: str | Path | None = None) -> "Settings":
        base = Path(config_dir)
        if root is not None:
            project_root = Path(root)
        else:
            # `scripts/configs` ist eine eigenstaendige Kalibrierkopie, aber
            # relative Modellpfade beziehen sich weiterhin auf das Projekt.
            # Den Root an einer stabilen Projektdatei erkennen statt pauschal
            # `config_dir.parent` anzunehmen.
            project_root = base.parent
            for candidate in (base, *base.parents):
                if (candidate / "pyproject.toml").exists():
                    project_root = candidate
                    break
        detection = None
        if (base / "detection.yaml").exists():
            detection = load_section(DetectionConfig, base, "detection.yaml", scenario)
        return cls(
            root=project_root, config_dir=base, scenario=scenario,
            lane=load_section(LaneConfig, base, "lane.yaml", scenario),
            bev=load_section(BevConfig, base, "bev.yaml", scenario),
            tracking=load_section(TrackingConfig, base, "tracking.yaml", scenario),
            indexing=load_section(IndexConfig, base, "indexing.yaml", scenario),
            events=load_section(EventConfig, base, "events.yaml", scenario),
            pipeline=load_section(PipelineConfig, base, "pipeline.yaml", scenario),
            windows=load_section(WindowConfig, base, "windows.yaml", scenario),
            boundaries=load_section(BoundaryTrackConfig, base, "boundaries.yaml", scenario),
            egomotion=load_section(EgoMotionConfig, base, "egomotion.yaml", scenario),
            debug=load_section(DebugConfig, base, "debug.yaml", scenario),
            detection=detection,
        )

    def require_detection(self) -> DetectionConfig:
        if self.detection is None:
            raise ValueError(f"{self.config_dir / 'detection.yaml'} fehlt -- "
                             "die YOLOE-Pipeline braucht diese Datei")
        return self.detection

    def weights_path(self) -> str:
        return self.tracking.resolve_weights(self.root)

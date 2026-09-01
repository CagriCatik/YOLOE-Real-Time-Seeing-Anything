"""YOLOE-Detektionskonfiguration und kommentarerhaltende Schreiber."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str
    classes: list[str]
    conf: float


@dataclass(frozen=True)
class CarpetConfig:
    detect_in: list[str]
    min_frac: float
    white_min_frac: float
    green_hsv: list[list[int]]
    red_hsv: list[list[list[int]]]
    white_hsv: list[list[int]] | None = None


# Reihenfolge = Rangfolge bei Mehrfachtreffern in derselben Spur: der letzte
# Eintrag gewinnt. Rot schlaegt Gruen, weil eine Blockierung die staerkere
# Aussage ist als eine Freigabe.
DEFAULT_AREA_PRIORITY = ("white", "gray", "green", "red")

DEFAULT_AREA_PROMPTS = {
    "green": ["green road"], "red": ["red road"],
    "white": ["white driving path on the road"],
}


@dataclass(frozen=True)
class DrivingAreaConfig:
    enabled: bool = True
    conf: float = 0.03
    max_box_frac: float = 0.5
    side_dead_zone: float = 0.02
    prompts: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_AREA_PROMPTS))
    # Welche Farben es ueberhaupt gibt und in welcher Rangfolge. Frueher fest
    # verdrahtet auf {green, red, white} -- eine weitere Farbe (z.B. `gray` fuer
    # den nicht freigegebenen Bereich) brauchte damit eine Codeaenderung.
    priority: tuple[str, ...] = DEFAULT_AREA_PRIORITY


DEFAULT_HUD_PROMPTS = {
    "takeover_request": ["yellow steering wheel warning icon"],
    "warning_popup": ["dark warning popup"],
    "lane_assist_warning": ["red lane departure warning symbol"],
    "speed_limit_sign": ["round speed limit sign"],
    "acc_indicator": ["green adaptive cruise control icon"],
    "lane_change_arrow": ["green lane change arrow"],
    "sensor_arc": ["green radar sensor arc"],
}


@dataclass(frozen=True)
class HudConfig:
    enabled: bool = False
    conf: float = 0.15
    prompts: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_HUD_PROMPTS))


@dataclass(frozen=True)
class DetectionConfig:
    model: ModelConfig
    rois: dict[str, list[list[float]]]
    roi_colors: dict[str, list[int]]
    crop_box: tuple[float, float, float, float]
    ego_box: tuple[float, float, float, float] | None
    carpet: CarpetConfig
    driving_area: DrivingAreaConfig = field(default_factory=DrivingAreaConfig)
    hud: HudConfig = field(default_factory=HudConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DetectionConfig":
        required = ("model", "rois", "roi_colors", "crop_box", "carpet")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"missing config section(s): {', '.join(missing)}")
        model = raw["model"]
        m = ModelConfig(str(model["checkpoint"]), list(model.get("classes", [])), float(model["conf"]))
        rois = {str(k): [[float(x), float(y)] for x, y in v] for k, v in raw["rois"].items()}
        for name, points in rois.items():
            if len(points) < 3:
                raise ValueError(f"ROI {name!r} needs at least three points")
            if any(not (0 <= n <= 1) for point in points for n in point):
                raise ValueError(f"ROI {name!r} coordinates must be between 0 and 1")
        crop = _box(raw["crop_box"], "crop_box")
        ego = _box(raw["ego_box"], "ego_box") if raw.get("ego_box") is not None else None
        c = raw["carpet"]
        carpet = CarpetConfig(list(c["detect_in"]), float(c["min_frac"]),
                              float(c.get("white_min_frac", c["min_frac"])),
                              c["green_hsv"], c["red_hsv"], c.get("white_hsv"))
        a = raw.get("driving_area", {})
        prompts = a.get("prompts", DEFAULT_AREA_PROMPTS)
        priority = tuple(a.get("priority", DEFAULT_AREA_PRIORITY))
        unknown = set(prompts) - set(priority)
        if unknown:
            raise ValueError(
                f"driving_area: Farben ohne Rangfolge: {', '.join(sorted(unknown))}. "
                f"Entweder in `priority` aufnehmen (bekannt: {', '.join(priority)}) "
                "oder den Prompt entfernen.")
        area = DrivingAreaConfig(bool(a.get("enabled", True)), float(a.get("conf", .03)),
                                 float(a.get("max_box_frac", .5)), float(a.get("side_dead_zone", .02)),
                                 {k: list(v) for k, v in prompts.items()}, priority)
        if not 0 < area.max_box_frac <= 1 or not 0 <= area.side_dead_zone < .5:
            raise ValueError("invalid driving-area geometry settings")
        h = raw.get("hud", {})
        hud = HudConfig(bool(h.get("enabled", False)), float(h.get("conf", .15)),
                        {k: list(v) for k, v in h.get("prompts", DEFAULT_HUD_PROMPTS).items()})
        return cls(m, rois, {k: list(v) for k, v in raw["roi_colors"].items()}, crop, ego, carpet, area, hud)


def _box(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    x0, y0, x1, y1 = map(float, value)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"invalid {name}: expected 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def load_detection_config(path: str | Path = "configs/detection.yaml") -> DetectionConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return DetectionConfig.from_dict(raw)


def save_crop_box(path: str | Path, box: list[float] | tuple[float, ...]) -> None:
    _box(box, "crop_box")
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    value = "crop_box: [" + ", ".join(str(float(v)).rstrip("0").rstrip(".") if float(v) % 1 else str(int(v)) for v in box) + "]"
    updated, count = re.subn(r"(?m)^crop_box\s*:\s*\[[^\n]*\]", value, text, count=1)
    if not count:
        updated = text.rstrip() + "\n" + value + "\n"
    target.write_text(updated, encoding="utf-8")


def save_rois(path: str | Path, rois: dict[str, list[list[float]]]) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    lines = ["rois:"]
    for name, points in rois.items():
        points_text = ", ".join("[" + ", ".join(str(float(n)).rstrip("0").rstrip(".") for n in p) + "]" for p in points)
        lines.append(f"  {name}: [{points_text}]")
    block = "\n".join(lines) + "\n"
    # Keep comments and the next top-level key intact.  ``\s`` is deliberately
    # avoided here because it also consumes newlines.
    updated, count = re.subn(r"(?m)^rois[ \t]*:[ \t]*\n(?:^[ \t]+.*\n)*", block, text, count=1)
    if not count:
        updated = block + text
    target.write_text(updated, encoding="utf-8")

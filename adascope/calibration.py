"""Automatische Kalibrierung der Spurgeometrie je Fahrzeugprojekt (FR-6).

Warum es das braucht
--------------------
`configs/lane.yaml` und `configs/bev.yaml` sind heute von Hand auf genau einen
Zuschnitt getunt. FR-6.2 erklaert die Auto-Kalibrierung zur harten
Voraussetzung fuer das Ausrollen -- und zwar zu Recht: ohne sie kostet jedes
neue Fahrzeugprojekt eine manuelle Kalibriersitzung, und deren Ergebnis ist
nicht nachvollziehbar.

Was hier bestimmt wird -- und was nicht
---------------------------------------
Bestimmt werden **Schwellwerte und Referenzgroessen**, die sich aus dem
Material selbst ablesen lassen:

    y_bottom / y_top     aus der Tiefenverteilung der gefundenen Markierungen
    white_l_min          aus dem Helligkeitshistogramm (Asphalt vs. Markierung)
    peak_min_distance    aus der beobachteten Spurbreite
    lane_width           aus den Korridorbreiten ueber viele Frames

NICHT bestimmt wird die Zuordnung zu Spurnummern -- FR-6.1 verlangt das
ausdruecklich nicht, und der Kern braucht sie auch nicht.

Wie belastbar das ist
---------------------
Jeder Wert kommt mit der Zahl der Frames, aus denen er stammt, und mit der
Streuung. Ein Wert aus fuenf Frames ist keine Kalibrierung, und das Ergebnis
sagt das auch -- statt eine Zahl zu liefern, der man ansieht, dass sie stimmt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import shutil
from statistics import median, pstdev

import cv2
import numpy as np
import yaml

from .config import BevConfig, IndexConfig, LaneConfig, Settings
from .io import iter_source
from .lanes.bev import (
    build_lane_mask, corridors_from, find_lane_boundaries, outer_solid_pair,
    restrict_to_driving_area, source_points, warp_lane_mask,
)
from .lanes.detection import build_masked_edges, cluster_segments, detect_lanes, extract_segments
from .lanes.indexing import estimate_lane_width_by_multiples
from .lanes.pipeline import road_vehicles
from .runner import choose_crop, crop_frame

MIN_FRAMES = 20
MIN_AUTO_FRAMES = 50


@dataclass
class Measurement:
    """Ein gemessener Wert mit seiner Herkunft -- nie eine nackte Zahl."""

    name: str
    value: float
    samples: int
    spread: float = 0.0
    current: float | None = None
    note: str = ""
    stable: bool = True
    auto_apply: bool = True

    @property
    def trustworthy(self) -> bool:
        return self.samples >= MIN_FRAMES

    @property
    def applicable(self) -> bool:
        """Darf dieser Wert ohne manuelle Geometrieentscheidung geschrieben werden?"""
        return self.samples >= MIN_AUTO_FRAMES and self.stable and self.auto_apply

    @property
    def change(self) -> str:
        if self.current is None:
            return ""
        delta = self.value - self.current
        return f"{self.current:.0f} -> {self.value:.0f} ({delta:+.0f})"

    def as_line(self) -> str:
        flag = "" if self.trustworthy else f"  ZU WENIG DATEN ({self.samples} Frames)"
        if self.trustworthy and not self.stable:
            flag = "  INSTABIL -- NICHT AUTOMATISCH UEBERNEHMEN"
        if self.trustworthy and self.stable and not self.auto_apply:
            flag = "  NUR VORSCHLAG -- GEOMETRIE MANUELL PRUEFEN"
        if (self.trustworthy and self.stable and self.auto_apply
                and self.samples < MIN_AUTO_FRAMES):
            flag = f"  NUR VORSCHLAG -- AUTO ERST AB {MIN_AUTO_FRAMES} FRAMES"
        change = f"  {self.change}" if self.change else ""
        note = f"  {self.note}" if self.note else ""
        return f"  {self.name:<20s} {self.value:>8.1f}  aus {self.samples:>4d} Frames" \
               f"  Streuung {self.spread:>6.1f}{change}{note}{flag}"


@dataclass
class Calibration:
    """Das Ergebnis eines Kalibrierlaufs."""

    source: str
    frames: int = 0
    measurements: list[Measurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> Measurement | None:
        return next((m for m in self.measurements if m.name == name), None)

    @property
    def usable(self) -> bool:
        return any(m.applicable for m in self.measurements)

    def as_yaml_fragments(self, *, auto_only: bool = False) -> dict[str, dict[str, float]]:
        """Vorschlaege, nach Zieldatei gruppiert -- zum Uebernehmen von Hand.

        Bewusst kein automatisches Schreiben in die Configs: eine Kalibrierung,
        die sich selbst einspielt, aendert das Verhalten ohne Spur in der
        Historie. Der Vorschlag wird angezeigt, das Uebernehmen bleibt eine
        bewusste Handlung.
        """
        fragments: dict[str, dict[str, float]] = {"lane": {}, "bev": {}, "indexing": {}}
        for measurement in self.measurements:
            if not measurement.trustworthy or (auto_only and not measurement.applicable):
                continue
            if measurement.name in ("y_bottom", "y_top", "white_l_min"):
                fragments["lane"][measurement.name] = round(measurement.value)
            elif measurement.name == "peak_min_distance":
                fragments["bev"][measurement.name] = round(measurement.value)
            elif measurement.name == "lane_width":
                fragments["indexing"][measurement.name] = round(measurement.value)
        return {key: value for key, value in fragments.items() if value}

    def report_data(self) -> dict:
        return {
            "source": self.source,
            "frames": self.frames,
            "usable": self.usable,
            "warnings": self.warnings,
            "measurements": [
                {"name": m.name, "value": round(m.value, 3), "samples": m.samples,
                 "spread": round(m.spread, 3), "current": m.current,
                 "stable": m.stable, "auto_apply": m.auto_apply,
                 "applicable": m.applicable, "note": m.note}
                for m in self.measurements
            ],
            "proposed": self.as_yaml_fragments(),
            "auto_apply": self.as_yaml_fragments(auto_only=True),
        }

    def write_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.report_data(), sort_keys=False,
                                         allow_unicode=True), encoding="utf-8")
        return target

    def as_text(self) -> str:
        lines = [f"Kalibrierung aus {self.source}", f"  Frames ausgewertet  {self.frames}", ""]
        lines += [m.as_line() for m in self.measurements]
        if self.warnings:
            lines += ["", "  Hinweise:"] + [f"    {w}" for w in self.warnings]
        if not self.usable:
            lines += ["", "  KEINE AUTO-ANWENDUNG: kein Wert erfuellt gleichzeitig "
                          f"Stabilitaet, Eignung und mindestens {MIN_AUTO_FRAMES} Frames."]
        return "\n".join(lines) + "\n"


def calibrate(source: Path, settings: Settings, max_frames: int = 300,
              stride: int = 3, on_progress=None, tracker=None) -> Calibration:
    """Referenzgroessen aus dem Material selbst ablesen (FR-6.1)."""
    result = Calibration(source=str(source))
    lane, bev = settings.lane, settings.bev

    marking_tops: list[int] = []
    marking_bottoms: list[int] = []
    lane_widths: list[float] = []
    corridor_gaps: list[float] = []
    bright_thresholds: list[float] = []

    if stride < 1:
        raise ValueError("stride muss mindestens 1 sein")
    frames, _ = iter_source(source, 1, 25.0)
    processed = 0
    homography_frames = 0
    crop = None
    for index, (_, frame) in enumerate(frames):
        if index % max(stride, 1) or processed >= max_frames:
            if processed >= max_frames:
                break
            continue
        if crop is None:
            crop = choose_crop(settings, frame.shape[1], frame.shape[0]) or False
        if crop:
            frame = crop_frame(frame, crop)
        height, width = frame.shape[:2]
        try:
            fitted = lane.scaled_to(width, height)
        except ValueError as exc:
            result.warnings.append(str(exc))
            break
        processed += 1

        bright_thresholds.append(_marking_threshold(frame, fitted))
        edges = build_masked_edges(frame, fitted)
        clusters = cluster_segments(extract_segments(edges, fitted), fitted)
        frame_tops: list[float] = []
        frame_bottoms: list[float] = []
        for cluster in clusters:
            # In Python-Floats wandeln: `statistics` kommt mit numpy-Typen
            # nicht zurecht, und die Hough-Segmente liefern numpy.intc.
            ys = [float(value) for _, _, seg in cluster for value in (seg[1], seg[3])]
            if ys:
                frame_tops.append(min(ys))
                frame_bottoms.append(max(ys))
        # Eine Stimme je Frame, nicht je Cluster. Sonst sahen 28 Frames wie
        # 169 unabhaengige Stichproben aus und bestanden faelschlich MIN_FRAMES.
        if frame_tops:
            marking_tops.append(int(np.percentile(frame_tops, 20)))
            marking_bottoms.append(int(np.percentile(frame_bottoms, 80)))

        vehicles = tracker.update(frame) if tracker is not None else []
        vehicles = road_vehicles(vehicles, fitted, settings.pipeline.road_margin)
        pair = outer_solid_pair(detect_lanes(frame, fitted), fitted, bev)
        if pair is None:
            continue
        homography_frames += 1
        from .lanes.bev import homography_from_pair
        H = homography_from_pair(pair, fitted, bev)
        camera_mask = build_lane_mask(frame, fitted, [v.bbox for v in vehicles])
        camera_mask = restrict_to_driving_area(
            camera_mask, source_points(pair, fitted))
        mask = warp_lane_mask(camera_mask, H, bev)
        corridors = corridors_from(find_lane_boundaries(mask, bev))
        if len(corridors) >= 2:
            widths = [float(b - a) for a, b in corridors]
            corridor_gaps.extend(widths)
            lane_widths.append(float(estimate_lane_width_by_multiples(
                corridors, settings.indexing.multiple_tolerance,
                settings.indexing.max_merge)))
        if on_progress and processed % 25 == 0:
            on_progress(f"    {processed} Frames ausgewertet")

    result.frames = processed
    result.measurements = _summarise(
        marking_tops, marking_bottoms, bright_thresholds, lane_widths, lane, bev)
    if processed and homography_frames / processed < 0.7:
        result.warnings.append(
            f"Homographie nur in {homography_frames}/{processed} Frames "
            "verfuegbar; zuerst Stufe 2-4 stabilisieren.")
    if corridor_gaps:
        spread = pstdev(corridor_gaps) if len(corridor_gaps) > 1 else 0.0
        if spread > 40:
            result.warnings.append(
                f"Korridorbreiten streuen stark (Streuung {spread:.0f} px). Das deutet "
                "auf wechselnde Fahrbahntypen hin -- eine feste Spurbreite waere dann "
                "abschnittsweise falsch.")
    return result


def _marking_threshold(frame: np.ndarray, lane: LaneConfig) -> float:
    """Trennwert zwischen Asphalt und Markierung aus dem L-Histogramm.

    Otsu ueber den ROI-Bereich: die Markierungen sind der helle Modus, der
    Asphalt der dunkle. Robuster als ein fester Wert, weil er sich mit der
    Belichtung mitbewegt.
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    roi = np.zeros(hls.shape[:2], np.uint8)
    cv2.fillPoly(roi, [np.array(lane.roi_polygon, np.int32)], 255)
    values = hls[:, :, 1][roi > 0]
    if values.size < 100:
        return float(lane.white_l_min)
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def _summarise(tops, bottoms, thresholds, lane_widths,
               lane: LaneConfig, bev: BevConfig) -> list[Measurement]:
    out: list[Measurement] = []
    if bottoms:
        out.append(Measurement("y_bottom", median(bottoms), len(bottoms),
                               pstdev(bottoms) if len(bottoms) > 1 else 0.0,
                               lane.y_bottom,
                               "unterster Rand der gefundenen Markierungen",
                               stable=(pstdev(bottoms) if len(bottoms) > 1 else 0.0) <= 25,
                               auto_apply=False))
    if tops:
        out.append(Measurement("y_top", median(tops), len(tops),
                               pstdev(tops) if len(tops) > 1 else 0.0,
                               lane.y_top, "oberster Rand",
                               stable=(pstdev(tops) if len(tops) > 1 else 0.0) <= 25,
                               auto_apply=False))
    if thresholds:
        out.append(Measurement("white_l_min", median(thresholds), len(thresholds),
                               pstdev(thresholds) if len(thresholds) > 1 else 0.0,
                               lane.white_l_min, "Otsu-Trennwert Asphalt/Markierung",
                               stable=(pstdev(thresholds) if len(thresholds) > 1 else 0.0) <= 8))
    if lane_widths:
        width = median(lane_widths)
        spread = pstdev(lane_widths) if len(lane_widths) > 1 else 0.0
        out.append(Measurement("lane_width", width, len(lane_widths), spread,
                               None, "Spurbreite in BEV-Pixeln",
                               stable=spread <= max(5.0, width * 0.15)))
        # Regel aus dem Projekt: deutlich unter der schmalsten echten Spurbreite,
        # deutlich ueber der Breite der Fehlpeaks.
        out.append(Measurement("peak_min_distance", width * 0.72, len(lane_widths),
                               spread * 0.72, bev.peak_min_distance,
                               "72 % der Spurbreite",
                               stable=spread <= max(5.0, width * 0.15)))
    return out


def apply_calibration(result: Calibration, config_dir: str | Path) -> list[Path]:
    """Stabile Vorschlaege anwenden, mit Backup und anschliessender Validierung.

    Kommentare und Dateistruktur bleiben erhalten; nur skalare Top-Level-Werte
    werden ersetzt. Diese Funktion wird nur durch das explizite CLI-Flag
    ``--apply`` aufgerufen.
    """
    fragments = result.as_yaml_fragments(auto_only=True)
    if not fragments:
        raise ValueError("keine stabilen, automatisch anwendbaren Messwerte")
    base = Path(config_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups: list[Path] = []
    changed: list[tuple[Path, Path]] = []
    try:
        for domain, values in fragments.items():
            target = base / f"{domain}.yaml"
            backup = target.with_suffix(target.suffix + f".bak-{stamp}")
            if target.exists():
                shutil.copy2(target, backup)
                backups.append(backup)
            text = target.read_text(encoding="utf-8") if target.exists() else ""
            for key, value in values.items():
                rendered = str(int(value)) if float(value).is_integer() else str(value)
                pattern = re.compile(rf"^(?P<prefix>{re.escape(key)}\s*:\s*)[^#\r\n]*",
                                     re.MULTILINE)
                if pattern.search(text):
                    text = pattern.sub(lambda match: match.group("prefix") + rendered + " ",
                                       text, count=1)
                else:
                    text += f"\n{key}: {rendered}\n"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            changed.append((target, backup))
        Settings.load(base)  # Schema und Werte unmittelbar pruefen.
    except Exception:
        for target, backup in changed:
            if backup.exists():
                shutil.copy2(backup, target)
        raise
    return backups

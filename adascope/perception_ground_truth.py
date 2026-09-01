"""Frame-level ground truth for measurable perception quality.

Event annotations answer *what happened over time*.  These annotations answer
whether the image-processing stages were geometrically correct on selected
frames.  Optional fields stay unavailable instead of being counted as passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PerceptionAcceptance:
    driving_area_iou_min: float = 0.80
    boundary_recall_min: float = 0.90
    boundary_mae_max_px: float = 12.0
    boundary_tolerance_px: float = 15.0
    lane_count_accuracy_min: float = 0.90
    ego_lane_accuracy_min: float = 0.90
    vehicle_lane_accuracy_min: float = 0.90

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PerceptionAcceptance":
        raw = raw or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"acceptance: unbekannte Schluessel {sorted(unknown)}")
        result = cls(**raw)
        for name, value in vars(result).items():
            if name.endswith("_px"):
                if value <= 0:
                    raise ValueError(f"acceptance.{name} muss positiv sein")
            elif not 0 <= value <= 1:
                raise ValueError(f"acceptance.{name} muss zwischen 0 und 1 liegen")
        return result


@dataclass(frozen=True)
class ExpectedVehicleLane:
    point: tuple[float, float]
    rel: int


@dataclass(frozen=True)
class ExpectedPerceptionFrame:
    frame: int
    driving_area: tuple[tuple[float, float], ...] = ()
    boundaries_bev: tuple[float, ...] = ()
    lane_count: int | None = None
    ego_lane_position: int | None = None
    vehicles: tuple[ExpectedVehicleLane, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExpectedPerceptionFrame":
        known = {"frame", "driving_area", "boundaries_bev", "lane_count",
                 "ego_lane_position", "vehicles"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"perception: unbekannte Schluessel {sorted(unknown)}")
        if "frame" not in raw:
            raise ValueError("perception-Eintrag ohne frame")
        polygon = tuple(tuple(map(float, point)) for point in raw.get("driving_area") or ())
        if polygon and (len(polygon) < 3 or any(len(point) != 2 for point in polygon)):
            raise ValueError("driving_area braucht mindestens drei 2D-Punkte")
        boundaries = tuple(map(float, raw.get("boundaries_bev") or ()))
        lane_count = int(raw["lane_count"]) if "lane_count" in raw else None
        ego_lane = int(raw["ego_lane_position"]) if "ego_lane_position" in raw else None
        if lane_count is not None and lane_count < 1:
            raise ValueError("lane_count muss mindestens 1 sein")
        if ego_lane is not None and ego_lane < 0:
            raise ValueError("ego_lane_position darf nicht negativ sein")
        vehicles = []
        for vehicle in raw.get("vehicles") or ():
            if set(vehicle) != {"point", "rel"} or len(vehicle["point"]) != 2:
                raise ValueError("vehicle braucht genau point:[x,y] und rel")
            vehicles.append(ExpectedVehicleLane(tuple(map(float, vehicle["point"])),
                                                 int(vehicle["rel"])))
        if not (polygon or boundaries or lane_count is not None
                or ego_lane is not None or vehicles):
            raise ValueError("perception-Frame enthaelt keine Messgroesse")
        return cls(int(raw["frame"]), polygon, boundaries, lane_count, ego_lane,
                   tuple(vehicles))


def polygon_iou(expected, predicted, shape) -> float | None:
    if not expected:
        return None
    if predicted is None:
        return 0.0
    a = np.zeros(shape[:2], np.uint8)
    b = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(a, [np.rint(np.asarray(expected)).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.rint(np.asarray(predicted)).astype(np.int32)], 1)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def match_boundaries(expected, predicted, tolerance: float
                     ) -> tuple[float | None, float | None]:
    wanted, found = list(map(float, expected)), list(map(float, predicted))
    if not wanted:
        return None, None
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(wanted)
                   for j, b in enumerate(found))
    used_a, used_b, errors = set(), set(), []
    for error, i, j in pairs:
        if error > tolerance or i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); errors.append(error)
    return (len(errors) / len(wanted),
            float(np.mean(errors)) if errors else None)


@dataclass(frozen=True)
class PerceptionMeasurement:
    frame: int
    driving_area_iou: float | None = None
    boundary_recall: float | None = None
    boundary_mae_px: float | None = None
    lane_count_accuracy: float | None = None
    ego_lane_accuracy: float | None = None
    vehicle_lane_accuracy: float | None = None
    expected_lanes: int | None = None
    actual_lanes: int = 0
    expected_ego: int | None = None
    actual_ego: int = -1

    def as_row(self) -> dict:
        return {name: ("" if value is None else round(value, 4)
                       if isinstance(value, float) else value)
                for name, value in vars(self).items()}


def _vehicle_accuracy(expected, analysis) -> float | None:
    if not expected:
        return None
    by_track = {o.track: o for o in analysis.occupancies if o.valid}
    correct = 0
    for item in expected:
        x, y = item.point
        vehicle = next((v for v in analysis.vehicles
                        if v.bbox[0] <= x <= v.bbox[2]
                        and v.bbox[1] <= y <= v.bbox[3]), None)
        occupancy = by_track.get(f"ID{vehicle.track_id}") if vehicle else None
        correct += occupancy is not None and occupancy.rel == item.rel
    return correct / len(expected)


def measure_perception(expected: ExpectedPerceptionFrame, analysis,
                       acceptance: PerceptionAcceptance) -> PerceptionMeasurement:
    recall, mae = match_boundaries(expected.boundaries_bev, analysis.boundaries,
                                   acceptance.boundary_tolerance_px)
    return PerceptionMeasurement(
        frame=expected.frame,
        driving_area_iou=polygon_iou(expected.driving_area,
                                     analysis.driving_area_src, analysis.image.shape),
        boundary_recall=recall, boundary_mae_px=mae,
        lane_count_accuracy=(float(len(analysis.lanes_rel) == expected.lane_count)
                             if expected.lane_count is not None else None),
        ego_lane_accuracy=(float(analysis.ego_lane_pos == expected.ego_lane_position)
                           if expected.ego_lane_position is not None else None),
        vehicle_lane_accuracy=_vehicle_accuracy(expected.vehicles, analysis),
        expected_lanes=expected.lane_count, actual_lanes=len(analysis.lanes_rel),
        expected_ego=expected.ego_lane_position, actual_ego=analysis.ego_lane_pos,
    )


@dataclass
class PerceptionScore:
    expected_frames: int
    measurements: list[PerceptionMeasurement] = field(default_factory=list)
    missing_frames: list[int] = field(default_factory=list)
    acceptance: PerceptionAcceptance = field(default_factory=PerceptionAcceptance)

    def means(self) -> dict[str, float | None]:
        result = {}
        for name in ("driving_area_iou", "boundary_recall", "boundary_mae_px",
                     "lane_count_accuracy", "ego_lane_accuracy",
                     "vehicle_lane_accuracy"):
            values = [getattr(row, name) for row in self.measurements
                      if getattr(row, name) is not None]
            result[name] = float(np.mean(values)) if values else None
        return result

    def checks(self) -> dict[str, bool | None]:
        means = self.means(); a = self.acceptance
        specs = {
            "driving_area_iou": (a.driving_area_iou_min, "min"),
            "boundary_recall": (a.boundary_recall_min, "min"),
            "boundary_mae_px": (a.boundary_mae_max_px, "max"),
            "lane_count_accuracy": (a.lane_count_accuracy_min, "min"),
            "ego_lane_accuracy": (a.ego_lane_accuracy_min, "min"),
            "vehicle_lane_accuracy": (a.vehicle_lane_accuracy_min, "min"),
        }
        return {name: (None if means[name] is None else
                       means[name] >= limit if direction == "min" else
                       means[name] <= limit)
                for name, (limit, direction) in specs.items()}

    @property
    def perfect(self) -> bool:
        checks = [value for value in self.checks().values() if value is not None]
        return not self.missing_frames and bool(checks) and all(checks)

    def label(self) -> str:
        return (f"{len(self.measurements)}/{self.expected_frames} Frames "
                f"{'ok' if self.perfect else 'NICHT OK'}")

    def as_text(self) -> str:
        labels = {
            "driving_area_iou": "Richtungsflaeche IoU",
            "boundary_recall": "Grenzen Recall",
            "boundary_mae_px": "Grenzen MAE",
            "lane_count_accuracy": "Spurzahl korrekt",
            "ego_lane_accuracy": "Ego-Spur korrekt",
            "vehicle_lane_accuracy": "Fahrzeugspur korrekt",
        }
        lines = [f"  Wahrnehmungsframes     {len(self.measurements)}/{self.expected_frames}"]
        means, checks = self.means(), self.checks()
        for name, label in labels.items():
            value = means[name]
            if value is None:
                lines.append(f"  {label:<23s} N/A (nicht annotiert)")
            else:
                unit = " px" if name == "boundary_mae_px" else ""
                lines.append(f"  {label:<23s} {value:.3f}{unit}  "
                             f"{'OK' if checks[name] else 'NICHT OK'}")
        if self.missing_frames:
            lines.append("  Fehlende Frames         " + ", ".join(map(str, self.missing_frames)))
        return "\n".join(lines)


def score_perception(expected: tuple[ExpectedPerceptionFrame, ...], analyses: dict[int, Any],
                     acceptance: PerceptionAcceptance) -> PerceptionScore:
    score = PerceptionScore(len(expected), acceptance=acceptance)
    for frame in expected:
        analysis = analyses.get(frame.frame)
        if analysis is None:
            score.missing_frames.append(frame.frame)
        else:
            score.measurements.append(measure_perception(frame, analysis, acceptance))
    return score


PERCEPTION_FIELDS = list(PerceptionMeasurement.__dataclass_fields__)

"""Stufe 9 - Manuelle Wahrnehmungs-Annotationen gegen die Pipeline pruefen.

Diese POC-Stufe bewertet keine erfundenen Ereignisse. Ein Szenario ohne
Spurwechsel darf ``events: []`` enthalten und trotzdem auf ausgewaehlten
Frames die Bildverarbeitung messen:

* Polygon der eigenen Richtungsfahrbahn (Kamerabild)
* Spurgrenzen als x-Positionen im BEV
* Spurzahl und Ego-Spurposition (von links, nullbasiert)
* optionale Fahrzeugpunkte mit relativer Spur -1/0/+1

Die Annotationen stehen im selben ``ground_truth/<szenario>.yaml`` wie die
Ereignisse; fehlende Messgroessen werden als N/A gemeldet, niemals als Treffer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.config.loader import read_yaml
from adascope.detection import YoloVehicleTracker
from adascope.lanes.pipeline import SequencePipeline, road_vehicles
from adascope.perception_ground_truth import (
    ExpectedPerceptionFrame, PerceptionAcceptance, PerceptionScore,
    measure_perception,
)


def zusatz(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--truth", type=Path, default=None,
                    help="YAML; Standard ground_truth/<Quellname>.yaml")
    ap.add_argument("--detect", action=argparse.BooleanOptionalAction, default=None,
                    help="Fahrzeug-Zuordnungen mit YOLO/Tracking pruefen")


def annotated_view(analysis, annotation, row, bev) -> np.ndarray:
    camera = (analysis.image.astype(np.float32) * 0.45).astype(np.uint8)
    expected = annotation.driving_area
    if expected:
        cv2.polylines(camera, [np.asarray(expected, np.int32)], True, (0, 255, 0), 3)
    if analysis.driving_area_src is not None:
        cv2.polylines(camera, [np.rint(analysis.driving_area_src).astype(np.int32)],
                      True, (0, 255, 255), 2)
    cv2.putText(camera, "GRUEN Soll | GELB Pipeline", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    bird = (cv2.cvtColor(analysis.mask_bev, cv2.COLOR_GRAY2BGR)
            if analysis.mask_bev is not None
            else np.zeros((bev.height, bev.width, 3), np.uint8))
    for x in annotation.boundaries_bev:
        cv2.line(bird, (round(x), 0), (round(x), bev.height - 1), (0, 255, 0), 3)
    for x in analysis.boundaries:
        cv2.line(bird, (round(x), 0), (round(x), bev.height - 1), (0, 255, 255), 1)
    label = (f"IoU {row.get('driving_area_iou', 'N/A')} | "
             f"Grenzen {row.get('boundary_recall', 'N/A')} | "
             f"Spuren {len(analysis.lanes_rel)}")
    cv2.putText(bird, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    bird = cv2.resize(bird, (camera.shape[1], camera.shape[0]))
    return np.vstack([camera, bird])


def main() -> None:
    args = parse(__doc__, zusatz, stufe="stage_09_perception_eval")
    source_name = Path(args.source).stem
    truth_path = args.truth or Path("ground_truth") / f"{source_name}.yaml"
    if not truth_path.exists():
        raise SystemExit(f"Keine Annotation: {truth_path}")
    raw = read_yaml(truth_path) or {}
    parsed = tuple(ExpectedPerceptionFrame.from_dict(item)
                   for item in raw.get("perception") or [])
    annotations = {item.frame: item for item in parsed}
    if not annotations:
        raise SystemExit(
            f"{truth_path} enthaelt keine perception-Frames. "
            "Siehe ground_truth/VORLAGE.yaml")
    acceptance = PerceptionAcceptance.from_dict(raw.get("acceptance"))

    st = Stage(args, "stage_09_perception_eval",
               "ground_truth perception + acceptance")
    pipeline = SequencePipeline(st.settings)
    detector = YoloVehicleTracker(st.settings.tracking) if args.detect else None
    print(f"  Annotation    {truth_path}")
    print(f"  Detektion     {'an' if detector else 'aus'}")
    target_last = max(annotations)
    measurements = []
    for i, name, img, lane in st.frames():
        if i > target_last:
            break
        vehicles = detector.update(img) if detector else []
        vehicles = road_vehicles(vehicles, lane, st.settings.pipeline.road_margin)
        analysis = pipeline.process(i, name, img, vehicles)
        if i not in annotations:
            continue
        ann = annotations[i]
        measurement = measure_perception(ann, analysis, acceptance)
        measurements.append(measurement)
        row = measurement.as_row()
        st.row(**row)
        view = annotated_view(analysis, ann, row, st.settings.bev)
        st.snapshot(i, "01_perception_score", view,
                    "Manuelle Annotation gegen Pipeline")
        st.show(i, view)
    st.finish()
    missing = sorted(set(annotations) - {item.frame for item in measurements})
    score = PerceptionScore(len(annotations), measurements, missing, acceptance)
    print("\n" + score.as_text())
    if not score.perfect:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

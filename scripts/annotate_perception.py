"""Interaktive Hilfe fuer manuelle Wahrnehmungs-Ground-Truth.

Beispiel:

    python scripts/annotate_perception.py lane_departure_3_lanes --frames 0,60,120,180,245

Je ausgewaehltem Frame:
1. Im Kamerabild das Polygon der eigenen Richtungsfahrbahn anklicken, ENTER.
2. Im BEV jede echte Spurgrenze einmal anklicken, ENTER.
3. Spurzahl und Position der Ego-Spur (von links, nullbasiert) eingeben.

ESC ueberspringt einen Frame. Die Pipeline zeichnet nur den Hintergrund; alle
Ground-Truth-Punkte stammen aus den manuellen Klicks. Das Ergebnis wird als
YAML-Entwurf geschrieben und muss vor Verwendung fachlich gesichtet werden.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adascope.config import Settings
from adascope.io import iter_source
from adascope.lanes import SequencePipeline
from adascope.runner import choose_crop, crop_frame


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="Name in scenarien/ (Video oder Ordner)")
    ap.add_argument("--frames", required=True,
                    help="kommaseparierte Quellframe-Nummern")
    ap.add_argument("--config", type=Path, default=Path("configs"))
    ap.add_argument("--scenario-dir", type=Path, default=Path("scenarien"))
    ap.add_argument("--out", type=Path, default=None)
    return ap


def source_for(base: Path, name: str) -> Path:
    directory = base / name
    if directory.is_dir():
        return directory
    matches = sorted(base.glob(f"{name}.*"))
    if matches:
        return matches[0]
    raise SystemExit(f"Szenario nicht gefunden: {name} in {base}")


def click_points(title: str, image: np.ndarray, minimum: int = 1
                 ) -> list[list[int]] | None:
    points: list[list[int]] = []
    base = image.copy()

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)
    while True:
        view = base.copy()
        for index, point in enumerate(points):
            cv2.circle(view, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(view, str(index + 1), (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(points) > 1:
            cv2.polylines(view, [np.asarray(points, np.int32)], False,
                          (0, 255, 0), 2)
        cv2.putText(view, "Links: Punkt | Rechts: zurueck | ENTER: fertig | ESC: skip",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            return None
        if key in (10, 13) and len(points) >= minimum:
            cv2.destroyWindow(title)
            return points


def number(prompt: str, minimum: int = 0) -> int | None:
    value = input(prompt).strip()
    if not value:
        return None
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"Wert muss mindestens {minimum} sein")
    return parsed


def main() -> None:
    args = parser().parse_args()
    wanted = sorted({int(value) for value in args.frames.split(",")})
    source = source_for(args.scenario_dir, args.scenario)
    settings = Settings.load(args.config,
                             args.scenario if (args.config / "scenarios" /
                                               f"{args.scenario}.yaml").exists() else None)
    frames, _ = iter_source(source, 1, 25.0)
    pipeline = SequencePipeline(settings)
    annotations = []
    crop = None
    for index, (name, image) in enumerate(frames):
        if index > wanted[-1]:
            break
        if crop is None:
            crop = choose_crop(settings, image.shape[1], image.shape[0]) or False
        if crop:
            image = crop_frame(image, crop)
        analysis = pipeline.process(index, name, image, [])
        if index not in wanted:
            continue

        camera = image.copy()
        cv2.putText(camera, f"Frame {index}: eigene RICHTUNGSFAHRBAHN",
                    (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        polygon = click_points("1/2 Richtungsfahrbahn", camera, minimum=3)
        if polygon is None:
            print(f"Frame {index} uebersprungen")
            continue

        bird = (cv2.cvtColor(analysis.mask_bev, cv2.COLOR_GRAY2BGR)
                if analysis.mask_bev is not None else
                np.zeros((settings.bev.height, settings.bev.width, 3), np.uint8))
        clicks = click_points("2/2 Echte BEV-Spurgrenzen", bird, minimum=2)
        if clicks is None:
            print(f"Frame {index} uebersprungen")
            continue
        boundaries = sorted({point[0] for point in clicks})
        print(f"\nFrame {index}: {len(boundaries)} Grenzen markiert: {boundaries}")
        lane_count = number("  Anzahl befahrbarer Spuren [leer=N/A]: ", 1)
        ego_position = number("  Ego-Spurposition von links, 0-basiert [leer=N/A]: ", 0)
        entry = {"frame": index, "driving_area": polygon,
                 "boundaries_bev": boundaries}
        if lane_count is not None:
            entry["lane_count"] = lane_count
        if ego_position is not None:
            entry["ego_lane_position"] = ego_position
        annotations.append(entry)

    cv2.destroyAllWindows()
    out = args.out or Path("ground_truth") / f"{args.scenario}.perception_draft.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    canonical = Path("ground_truth") / f"{args.scenario}.yaml"
    if canonical.exists():
        existing = yaml.safe_load(canonical.read_text(encoding="utf-8")) or {}
    document = {
        "note": "DRAFT: manuell sichten, dann in die kanonische YAML uebernehmen",
        "events": existing.get("events", []),
        "perception": annotations,
        "acceptance": existing.get("acceptance", {
            "driving_area_iou_min": 0.80,
            "boundary_recall_min": 0.90,
            "boundary_mae_max_px": 12.0,
            "boundary_tolerance_px": 15.0,
            "lane_count_accuracy_min": 0.90,
            "ego_lane_accuracy_min": 0.90,
            "vehicle_lane_accuracy_min": 0.90,
        }),
    }
    out.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"\nEntwurf: {out} ({len(annotations)} Frames)")
    print("Erst nach manueller Sichtung nach ground_truth/<scenario>.yaml uebernehmen.")


if __name__ == "__main__":
    main()

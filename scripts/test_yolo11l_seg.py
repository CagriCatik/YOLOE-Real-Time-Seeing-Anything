#!/usr/bin/env python3
"""YOLO11l-seg auf einem Bild, Video oder Frame-Ordner testen.

Geschlossenes Vokabular wie yolo11n (dieselben 80 COCO-Klassen), aber grosses
Modell **mit Segmentierungsmasken**. Deutlich langsamer, dafuer pixelgenaue
Umrisse statt Rechtecke.

    python scripts/test_yolo11l_seg.py --source test_images/test_frame.png
    python scripts/test_yolo11l_seg.py --source scenarien/acc_plus_1_vid.mp4 --vehicles
    python scripts/test_yolo11l_seg.py --source ... --masks-only --device 0

Die Gewichte (~50 MB) werden beim ersten Lauf nach `models/` geladen.

WOFUER DIE MASKE TAUGT -- UND WOFUER NICHT
------------------------------------------
Fuer die Spurpipeline ist die Maske **nicht** verwendbar: die Homographie gilt
nur in der Bodenebene, ein Fahrzeugumriss hat Bauhoehe und zerlaeuft beim
Warpen ueber mehrere Spuren. Nur die Bbox-Unterkante darf zwischen den Ebenen
wechseln (siehe docs/10_PIPELINE_UND_ROBUSTHEIT.md).

Nuetzlich ist sie fuer anderes: Fahrzeuge praeziser aus der Weissmaske
ausstanzen, als es ein Rechteck kann, und Verdeckungen beurteilen.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from _common import Detection, add_common_args, resolve_weights, run

DEFAULT_WEIGHTS = "yolo11l-seg.pt"
VEHICLE_CLASSES = (2, 3, 5, 7)


def configure() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--vehicles", action="store_true",
                        help=f"nur Fahrzeugklassen {VEHICLE_CLASSES}")
    parser.add_argument("--masks-only", action="store_true",
                        help="nur Masken zeichnen, keine Rechtecke")
    parser.add_argument("--no-masks", action="store_true",
                        help="Masken ignorieren -- zum Vergleich mit yolo11n")
    return parser


def main(argv=None) -> int:
    args = configure().parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(resolve_weights(args.weights))
    classes = VEHICLE_CLASSES if args.vehicles else args.classes
    options = {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "verbose": False}
    if classes:
        options["classes"] = list(classes)
    if args.device is not None:
        options["device"] = args.device

    def predict(frame):
        result = model.predict(source=frame, **options)[0]
        detections = to_detections(result, frame.shape[:2], not args.no_masks)
        if args.masks_only:
            # Boxen ausserhalb des Bildes schieben statt entfernen: die
            # Zusammenfassung soll die Detektionen weiter zaehlen.
            for det in detections:
                det.box = (0.0, 0.0, 0.0, 0.0)
        return detections

    return run(args, "yolo11l-seg", predict)


def to_detections(result, shape: tuple[int, int], with_masks: bool) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    masks = None
    if with_masks and getattr(result, "masks", None) is not None:
        # Die Maske kommt in Netzaufloesung -- auf das Frame hochskalieren,
        # sonst passt sie nicht zu den Boxkoordinaten.
        height, width = shape
        raw = result.masks.data.cpu().numpy()
        masks = [cv2.resize(m.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST).astype(bool) for m in raw]

    detections = []
    for index, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy.cpu().numpy(),
                                                  boxes.conf.cpu().numpy(),
                                                  boxes.cls.cpu().numpy())):
        detections.append(Detection(
            str(names[int(cls)]), float(conf), tuple(float(v) for v in xyxy),
            mask=masks[index] if masks is not None and index < len(masks) else None))
    return detections


if __name__ == "__main__":
    raise SystemExit(main())

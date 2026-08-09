#!/usr/bin/env python3
"""YOLO11n auf einem Bild, Video oder Frame-Ordner testen.

Geschlossenes Vokabular: die 80 COCO-Klassen, nichts anderes. Schnell (~5 MB),
das Arbeitspferd für Fahrzeugdetektion. Optional mit ByteTrack-IDs über Frames.

    python scripts/test_yolo11n.py --source test_images/test_frame.png
    python scripts/test_yolo11n.py --source scenarien/acc_plus_1_vid.mp4 --track
    python scripts/test_yolo11n.py --source scenarien/acc_plus_1 --vehicles --device 0

Klassenfilter:
    --vehicles          Abkürzung für car, motorcycle, bus, truck (2 3 5 7)
    --classes 0 2 7     beliebige COCO-Indizes
    --list-classes      alle 80 Klassen mit Index anzeigen
"""

from __future__ import annotations

import argparse

from _common import Detection, add_common_args, resolve_weights, run

DEFAULT_WEIGHTS = "yolo11n.pt"
VEHICLE_CLASSES = (2, 3, 5, 7)          # car, motorcycle, bus, truck


def configure() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="COCO-Klassenindizes; ohne Angabe alle 80")
    parser.add_argument("--vehicles", action="store_true",
                        help=f"nur Fahrzeugklassen {VEHICLE_CLASSES}")
    parser.add_argument("--track", action="store_true",
                        help="ByteTrack-IDs über Frames (nur bei Video/Ordner sinnvoll)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        choices=("bytetrack.yaml", "botsort.yaml"))
    parser.add_argument("--list-classes", action="store_true")
    return parser


def main(argv=None) -> int:
    args = configure().parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(resolve_weights(args.weights))
    if args.list_classes:
        for index, name in sorted(model.names.items()):
            print(f"{index:>3d}  {name}")
        return 0

    classes = VEHICLE_CLASSES if args.vehicles else args.classes
    options = {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "verbose": False}
    if classes:
        options["classes"] = list(classes)
    if args.device is not None:
        options["device"] = args.device

    def predict(frame):
        if args.track:
            # persist=True hält Track-IDs über Frames. Beim ERSTEN Aufruf backt
            # Ultralytics den Wert in seinen Callback ein -- deshalb hier immer
            # True und kein Umschalten (siehe adascope/detection/tracking.py).
            result = model.track(source=frame, persist=True, tracker=args.tracker,
                                 **options)[0]
        else:
            result = model.predict(source=frame, **options)[0]
        return to_detections(result)

    label = "yolo11n" + ("_track" if args.track else "")
    return run(args, label, predict)


def to_detections(result) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    ids = (boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None
           else [None] * len(boxes))
    return [Detection(str(names[int(cls)]), float(conf), tuple(float(v) for v in xyxy), track)
            for xyxy, conf, cls, track in zip(boxes.xyxy.cpu().numpy(),
                                              boxes.conf.cpu().numpy(),
                                              boxes.cls.cpu().numpy(), ids)]


if __name__ == "__main__":
    raise SystemExit(main())
